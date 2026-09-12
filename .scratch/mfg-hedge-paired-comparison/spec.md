# Minimal MFG-Hedge vs No-Hedge Paired Comparison

## Goal

Answer exactly one question: facing the same deterministic H→D→F→R failure process, does adding the minimal MFG-Hedge protection controller change P50/P95/P99 completion latency, Replay rate, failure/recovery-window tail latency, drain time, extra executed work, Replica utilization, and Hedge launch burst — compared with pure No-Hedge? First version: single seed, same trace, mechanism-level comparison only. Multi-seed campaigns, significance tests, load sweeps, and Pareto sweeps are not preconditions.

This spec is the single functional authority; nothing is left to implementer improvisation. Decisions still requiring user confirmation are collected in section 17.

## 1. A/B arms

- Arm A: `policy = no_hedge`; every Token Normal; the accepted Common State + No Hedge semantics.
- Arm B: `policy = mfg_hedge`; MFG chooses only among Normal / Delayed Hedge / Immediate Hedge. MFG does not touch the Gate, the Logical Expert choice, the Primary Dispatcher, Replica configuration, training, or any HJB–FPK machinery.
- Both arms consume byte-identical: Token arrivals, Token classes, attempt-0 Primary draws, attempt-1 Replay draws, attempt-2 Hedge draws, the Common State timeline, the base seed, Gate results, and the Primary Dispatcher. The only experimental variable is the Protection Policy.

## 2. Unified evaluation engine

One hedge-capable event engine (`hedge_simulation.py`, ticket 01) evaluates both arms. No-Hedge mode run through the new engine must reproduce `simulate_common_state_no_hedge` Token results exactly (same projection fields as the ticket-02 regression: token_id, class, arrival, primary replica, start, attempt-0 required work, completion, queue delay, latency). A/B differences may only arise from Protection Actions; the timeline, Dispatcher, and service draws must never be adjusted to favor arm B.

Same-time event order in the new engine (fixed):

```text
DOMAIN_STATE_CHANGE
  -> failure/replay substeps
  -> COMPLETE
  -> HEDGE_TIMER
  -> TOKEN_ARRIVAL
  -> dispatch/start
```

A completion at the same instant voids a pending timer; a failure at the same instant creates the Replay and voids the timer before it can fire (spec section 18 scenarios 4-5).

## 3. Copy identity and random keys

- attempt_id 0 = Primary, 1 = Replay (unchanged), 2 = Hedge Backup.
- `WorkloadTrace` gains one more optional field `hedge_service_times: tuple[tuple[float, ...], ...] | None = None`, indexed `[replica][token_id]`, drawn from stream label `service:{replica}:attempt2`.
- Adding the Hedge stream must not perturb arrival, class, attempt-0, or attempt-1 draws (regression-tested). The No-Hedge path never reads attempt-2. No consumption-order randomness; no process-random `hash()`. Existing schema-2 No-Hedge artifacts' scientific content is unchanged.

## 4. Backup Replica

Fixed pairing: Primary 0 ↔ Backup 1, Primary 1 ↔ Backup 0; the Backup is the same Logical Expert's copy in the other failure domain. MFG never chooses the Backup Replica. If the Backup's domain is Failed at decision time, the Hedge action is infeasible: the MFG/quota layer restricts the feasible action set to `{Normal}` in state F, and the executor never silently turns a requested Immediate/Delayed into a no-op — suppressed actions are counted (`hedge_suppressed`), never executed-and-pretended.

## 5. Action semantics

1. **Normal**: Primary only; if the Primary is lost to failure with no other surviving copy, the existing single-Replay rule applies. No-Hedge behavior is preserved.
2. **Delayed Hedge**: Primary enqueues normally; a `HEDGE_TIMER` is scheduled at `arrival_time + tau0`. If the Token is complete when the timer fires, the timer is ignored. If the Token is incomplete and the Backup domain is usable, a Hedge Backup is created. If the Primary was lost before the timer fired and no Hedge Backup exists, the existing Replay path applies and the timer is voided — no attempt 2 follows (section 7).
3. **Immediate Hedge**: Primary and Hedge Backup are both created at the Token's arrival instant and enter their respective Replicas' FCFS queues; the Primary enqueues first, then the Hedge (fixed, testable order); the two copies use different stable random keys.

## 6. Winner and cancellation semantics (conservative, first version)

- The first valid completion wins; the Token's `completion_time` is the winner's.
- A queued loser is cancelled immediately: `cancelled_queued`, executed_work = 0.
- A running loser is not preempted: it runs to completion (`completed_loser`) and its result is discarded; its work counts as consumed resources.
- A loser completion never re-completes the Token; a loser failure never creates a Replay.
- Attempt statuses: `completed_winner`, `completed_loser`, `cancelled_queued`, `failed_running`, `invalidated_queued`. The bare `completed` status must not mix winner and loser meanings.
- Ideal (preemptive) cancellation is a separate later ablation, not this feature.

## 7. Hedge ↔ Replay interaction (case law)

1. Normal Primary lost, no other copy: create Replay attempt 1.
2. Primary lost but a Hedge Backup is queued/running: wait for the Backup; no Replay.
3. Primary lost before the Delayed timer fired: create Replay attempt 1, void the timer, never create attempt 2. A Token never gets both a Replay and a Hedge Backup for the same Primary loss.
4. Hedge Backup sits in the domain that later Fails: the Backup fails/invalidates like any execution on A and never creates a Replay. (In this slice only A fails; a Primary on B never fails, so Primary-lost-with-Backup-on-A cannot occur; the case that must be covered is Primary=B, Backup=A invalidated at F while the Primary continues.)
5. Token already has a winner: no timer, failure, or completion may create a new Replay/Hedge or a second completion.

Per Token invariants: at most one Replay; at most one Hedge Backup; exactly one winner; exactly one Token completion.

## 8. Hedge delay

- `tau0 = P90 of the Healthy service-time distribution`, from config `hedge_delay_quantile = 0.9`.
- The timer starts at Token arrival/enqueue, not at Primary start.
- `tau0` comes from the configured Healthy base distribution: `tau0 = exp(mu + sigma * NormalDist().inv_cdf(q))` using stdlib `statistics.NormalDist` and the `lognormal_parameters` mu/sigma. Never reverse-fit from evaluation results.
- No NumPy/SciPy. `tau0` is recorded in the summary. Timer sweeps are out of scope.

## 9. Protection policy and causal online quota

- The MFG solver outputs per Common State, per TokenClass action probabilities `pi[state][class][action]`.
- Per-Token independent random sampling is forbidden, and offline whole-window rounding is forbidden: the allocator must be causal (it cannot know a window's final Token count when early Tokens arrive).
- **Two-phase online deficit allocator**. For each control window and class:
  1. `requested_action`: `deficit[a] += pi[a]` for all actions; request the action with the largest deficit, tie-break by fixed action order N < D < I; then `deficit[requested] -= 1`. The deficit is updated exactly once for the requested action and is never re-debted by later suppression.
  2. `applied_action`: the feasible-set and class-budget projection of the request. A requested D/I that the projection cannot fund becomes Normal, increments `quota_suppressed`, and leaves the deficit untouched.
  3. Only `applied_action` reaches the hedge event engine.
- Inputs are only: the current Token, prior arrivals in the current window, the solved policy, and used/reserved estimated Hedge budget. Same policy + trace ⇒ identical assignments; no reads of future arrivals, service times, completions, or failure outcomes.
- Error bounds: per window per class, `|requested_count[a] - pi[a] * n| < 1` holds **only** for requested counts; applied counts after budget projection carry no such guarantee.
- Feasible-set projection happens before the deficit step when the state forbids actions: in F the policy is already projected to all-Normal, so nothing is requested and nothing is counted as quota-suppressed. Only a genuinely requested D/I that is then projected to Normal counts.
- **Budget shares per class**: within a window, class k's estimated Hedge budget share is proportional to its planned demand, `share[k] = window_budget * d[k] / max(d_total, epsilon)` with `d[k] = weight[k] * sum_a pi[k,a] * g[k,a]`; if `d_total = 0` no Hedge is assigned at all. The Urgent share is reserved and never taken by Regular Tokens; unused Urgent share expires at window close (no carryover, no borrowing) so early Regular Tokens cannot exhaust future Urgent budget.
- **Control windows and budgets**: `window_budget = hedge_budget_rate * window_duration`. Windows are half-open with maximum width W = 25.0; a Common State boundary forcibly closes the current window and opens a fresh one at the boundary, so phase R starts a new window at 220 whose next regular boundary is 245 (the global 25.0 grid is not resumed). Each new window resets the deficit, class budget shares, and suppression counters. The applied distribution is the solved policy for the window's state; every window lies strictly inside one state by construction.
- **Counters and identities** (reported per window, per class, and globally):
  - `policy_requested_action_counts`, `applied_action_counts`, `quota_suppressed` (D/I→N by projection), `executor_hedge_suppressed` (applied but not launched at event time, e.g. Backup domain Failed — the engine's existing `hedge_suppressed`, semantics unchanged).
  - `requested_D + requested_I = applied_D + applied_I + quota_suppressed`.
  - `engine.hedge_requested = applied_D + applied_I`.
  - `planned_expected_hedge_work` is computed from applied actions only and must satisfy `planned_expected_hedge_work <= window_budget`.

## 10. Hedge budget and the degenerate-defaults check

`hedge_budget(z) = max(0, target_max_utilization * capacity(z) - primary_work_rate(z) - replay_work_rate(z))`: the headroom deducts predicted Primary and Replay work so Replay-occupied capacity is never double-assigned to Hedge. Capacities: H 2.0, D 1.5, F 1.0.

- Current default load 0.7 (λ = 1.4): budgets are approximately H 0.4, D 0, F 0 — the MFG arm would be projected to almost all Normal in D/F and the comparison degenerates toward A ≈ B. This must not be silently ignored.
- Option A: keep 0.7 to observe storm suppression, accepting no D/F protection headroom.
- Option B (confirmed for the first paired run): a new self-contained config `configs/v1_paired_comparison.json` with `healthy_offered_load = 0.5` (λ = 1.0), giving real D-window headroom. Loads 0.6/0.7 stay as stress diagnostics only; the 0.7 configuration stays untouched.
- This is confirmation item 1; `configs/v1_common_state.json` is not modified.

## 11. Fault-aware ActionStats calibration (solver-side only)

Constant-state calibration would make `replay_probability` identically ~0 and kill the Regular/Urgent distinction, so calibration episodes include the real H→D→F→R transitions.

Calibration uses representative-token / tagged-probe semantics. A cell's rho coordinate is the **background total work rate / state capacity** that a representative probe faces on arrival: background work includes background-traffic Primary and Replay work; the probe's own Primary/Hedge/Replay work never enters its cell's background rho. `ActionStats` describe a probe choosing N/D/I against that background. This is a finite-action mean-field approximation in which queueing cost closes only through aggregate rho; it does not claim to capture the background's action composition.

Rules:

1. Each state/grid/class cell uses frozen, independent `calibration:`-namespace background traces; no reads of evaluation traces or A/B artifacts.
2. Within-cell CRN: the N/D/I probes for a cell share the same background trace and the same probe attempt-0/1/2 draws.
3. D-cohort probes genuinely cross the D→F boundary; in-flight work at the boundary fails, Replays, or is Hedge-protected exactly per engine semantics; F-state cells contain Normal only.
4. A deterministic arrival-rate adjustment drives the achieved background rho to each target grid point: bisection on the background arrival rate over bounds [0.05, 3.0], at most 40 iterations, with a fixed per-cell seed set; failure to converge is a fail-fast error. Acceptance tolerance: `abs(achieved_rho - target_rho) <= 0.02` (mechanism-verification value; any change requires a spec amendment with reasons).
5. Every populated cell records `target_rho`, `achieved_rho`, `rho_error`, `background_arrival_rate`, and `sample_count`; each valid cell needs **>= 400 probe cohort samples**, accumulated across independent episodes. Cells are keyed `(state, grid point, class, action)` over the grid {0.5, 0.7, 0.9, 1.1, 1.3}; lookups use linear interpolation between bracketing grid points, clamped at the ends, and a missing grid point is an error.
6. Deterministic acceptance fixture: in at least one D-state cell, Normal and some Hedge action must differ in `q` or `expected_replay_work`; otherwise calibration fails fast.
7. Output per cell: `ActionStats(mean_latency L, replay_probability q, deadline_miss_probability = 0.0, expected_hedge_work g, expected_replay_work)` plus the records of rule 5. This is a mechanism-level Monte Carlo approximation — not an analytic solution, and never reverse-fit from evaluation results.
8. The solver's rho is the same physical quantity — population Primary+Hedge+Replay total work rate over capacity — so the grid and the fixed point share one coordinate.

## 12. Boundary-pinned capacity-feasible MFG solver (ADR-0007/0008, accepted 2026-09-04)

Per Common State (H/D/F; phase R reuses the H policy since `state_at` is H), ADR-0007 supplies the capacity accounting and ADR-0008 supplies the binding-state solve:

```text
incremental_work[k,a] = expected_hedge_work[k,a] + expected_replay_work[k,a]
J[k,a] = L[k,a] + gamma[k]*q[k,a] + price * incremental_work[k,a]
target_work_rate   = target_max_utilization * state_capacity
capacity_gap       = total_work_rate - target_work_rate      # signed
capacity_violation = max(0, capacity_gap); capacity_slack = max(0, -capacity_gap)
```

This is the preserved schema-3 objective used by the completed original
comparison. ADR-0009 and `.scratch/mfg-hedge-runtime-cost/spec.md` define the
schema-4 correction with persistent incremental-work and wasted-work costs;
they do not rewrite the original artifact or its interpretation.

- Price-zero diagnostic (H/D): run the cold-start policy/load fixed point exactly once from all-Normal and `rho = primary/capacity`, using configured policy damping, a 2000-iteration cap, and final undamped policy/load residuals < 1e-6. Failure is `nonconverged/inner_iteration_cap` and never implies that capacity binds. A successful result with `capacity_gap <= 1e-6` is the accepted slack solution at price zero.
- Boundary-pinned branch (H/D): only a successful price-zero result with positive capacity gap enters this branch. Fix `rho = target_max_utilization` (0.9), interpolate ActionStats once, and make `policy(price)` the direct undamped Softmax response at those fixed stats. No nested rho solve, continuation, warm start, or rho damping occurs inside price search.
- Before price search, compute `minimum_total = primary + arrival_rate * sum_k weight[k] * min_a(g+r)[k,a]` at boundary rho. If `minimum_total > target_work_rate + 1e-6`, return `infeasible/boundary_minimum_exceeds_capacity`.
- Otherwise search the one-dimensional monotone signed `capacity_gap(price)`: bracket from `price_step` with at most 60 doublings and bisect at most 80 iterations. A finite cap without a sign crossing is `nonconverged/bracket_failed`. If the boundary price-zero gap is negative after the successful unconstrained result was overloaded, return `nonconverged/binding_branch_inconsistent` rather than use a negative price.
- Binding acceptance requires `|capacity_gap| <= 1e-6`, `|price * capacity_gap| <= 1e-6`, undamped policy/load residuals < 1e-6, and `hedge_work_rate <= hedge_budget_rate + 1e-6`.
- Grid clamp is disqualifying: any clamped lookup stops the state immediately (`nonconverged`, `grid_clamped=true`, offending rho recorded).
- F runs no price search: exact all-Normal probabilities, Normal-only rho fixed point, `status=forced_normal`, `price=0`, `hedge_budget_rate=0`, `structural_capacity_violation = (capacity_gap > 1e-6)` disclosed — never called an equilibrium.
- Final state: one recompute of ActionStats, undamped soft best response, all work rates, and the policy/load/capacity/complementarity residuals from the final `(probabilities, rho, price)`; converged H/D additionally require `hedge_work_rate <= hedge_budget_rate + 1e-6` where `hedge_budget_rate = max(0, target_work_rate - primary_work_rate - replay_work_rate)`.
- Statuses: exactly one of `converged` / `forced_normal` / `infeasible` / `nonconverged`, plus the orthogonal `grid_clamped` flag and a stable reason code. Official paired run gate: H/D `converged` and unclamped; F `forced_normal`, unclamped, converged Normal-only rho; R reuses H. Anything else is diagnostics only.
- `price_step` is only the initial bracket scale; `price_damping` stays in config/provenance but is not consumed by the boundary-pinned solver.
- `gamma[k]`: config keys `replay_penalty_regular = 1.0`, `replay_penalty_urgent = 5.0`. No HJB–FPK, training, gradients, or Gate changes.

## 13. Paired metrics

- Primary grouping: `arrival_phase = phase_at(arrival_time)` (arrival cohorts are untouched by the algorithm). `completion_phase` diagnostics retained, plus the `arrival_phase × completion_phase` migration matrix.
- A/B comparisons: overall P50/P95/P99; per-arrival-cohort (H/D/F/R) P50/P95/P99; Replay rate; drain duration; cumulative queue delay; `policy_requested_action_counts`, `applied_action_counts`, `quota_suppressed`, and `executor_hedge_suppressed` per window, class, and global; Hedge launch counts; cancelled counts; winner-kind counts (Primary/Hedge/Replay); executed hedge work; wasted loser work; total executed work; extra execution ratio; execution amplification; per-Replica utilization; queue lengths at failure/recovery (live entries only); peak Hedge launch rate; peak/mean Hedge launch ratio; sustained overload duration.
- Protection Storm bins: fixed width **1.0** normalized time unit, half-open `[k, k+1)`. Peak launch rate = max per-bin Hedge launches / bin width. Peak/mean ratio = **null when the arm has zero Hedge launches**. Sustained overload duration = the length of the **longest run of consecutive overloaded bins** (not the total count of overloaded bins). The final partial bin (width < 1.0) reports its rate normalized by its actual width and is excluded from peak-rate comparisons.

## 14. Planned versus realized budget reporting

The quota guarantees only `planned_expected_hedge_work <= window_hedge_budget`. Realized service times are random, so this is not a hard realized-capacity guarantee. The summary and comparison must report separately:

- planned expected Hedge work and the budget per window;
- realized Hedge work;
- realized budget excess/violation per window;
- suppressed assignment counts, split per section 9: `quota_suppressed` (projection) and `executor_hedge_suppressed` (runtime) are reported separately and never merged.

## 15. Paired comparison output

One command (ticket 04):

```text
compare-mfg-hedge-vs-no-hedge --config configs/v1_paired_comparison.json --tokens 1000 [--run-id] [--artifacts-root]
```

One config read, one shared WorkloadTrace, arm A run, arm B run, trace-identity verification between arms, then one unique artifact directory:

```text
artifacts/<run-id>/
├── manifest.json           # run identity, config identity, trace identity, versions
├── no_hedge_summary.json   # schema-2 summary, arm A
├── mfg_hedge_summary.json  # schema-3 summary, arm B (adds hedge/mfg blocks)
└── comparison.json         # paired deltas
```

The four files are committed **as one transaction**: staged in a temporary sibling directory and moved into place only when all four are written; any failure removes the partial directory entirely, and an existing run directory is never overwritten.

`comparison.json` contains at least: A/B absolute values; B−A deltas; relative change (null when A = 0); configuration identity; trace identity (seed + stream hashes); MFG policy, price, residuals, convergence per state; tau0; action counts; planned vs realized Hedge work; all invariants. Results must never exist only in terminal output.

## 16. Ticket split

- `01-hedge-capable-event-engine.md`: attempt-2 CRN, HEDGE_TIMER, Normal/Delayed/Immediate, winner/loser/cancellation, failure interactions, No-Hedge regression. No MFG.
- `02-action-stats-calibration.md` (Blocked by: 01): fault-aware calibration episodes, L/q/g table. Not the final evaluator.
- `03-mfg-solver-and-quota.md` (Blocked by: 02): original solver/status/quota implementation. Does not run the final comparison.
- `15-implement-boundary-pinned-solver.md`: accepted ADR-0008 boundary-pinned correction and real-table gate.
- `04-paired-comparison-cli-and-artifacts.md` (Blocked by: 15): same-trace A/B, arrival cohorts, migration matrix, storm metrics, transactional artifacts, single-seed controlled run.

## 17. Confirmed decisions

1. First paired run uses `healthy_offered_load = 0.5` (new `configs/v1_paired_comparison.json`), keeping 0.6/0.7 as stress diagnostics only.
2. Delayed timer starts at Token arrival.
3. Conservative (non-preemptive) cancellation is the primary semantics.
4. attempt_id = 2 for the Hedge Backup.
5. Feasible action set in F is `{Normal}`.
6. Primary lost before the timer fires ⇒ Replay, not a late Hedge.
7. Control window maximum width W = 25.0 with boundary-forced cuts.
8. Protection Storm bin width = 1.0.
9. Solver convergence: price-zero inner cap 2000; binding bracket at most 60 doublings and 80 bisection iterations; final policy/load/capacity/complementarity tolerances are all `1e-6`.

Parameter positioning: `healthy_offered_load = 0.5`, `replay_penalty_regular = 1.0`, `replay_penalty_urgent = 5.0`, the calibration grid {0.5, 0.7, 0.9, 1.1, 1.3}, >= 400 cohort samples per cell, W = 25.0 with boundary cuts, and storm bin 1.0 are **untuned mechanism-verification defaults**. The first round validates the mechanism only; multi-seed campaigns and penalty/load sensitivity analyses are later work, not part of these tickets.

## 18. Mandatory deterministic test scenarios (by ticket)

Ticket 01 (engine):
1. Immediate Hedge enqueues both copies at arrival (Primary then Hedge).
2. Delayed timer suppressed when the Primary completes first.
3. Delayed timer fires and creates the Backup.
4. Timer and completion at the same instant: completion is processed first, timer void.
5a. Timer and failure at the same instant, Primary=B/Backup=A: the failure is processed first, the Backup domain is Failed, no launch, counted as suppressed.
5b. Timer and failure at the same instant, Primary=A: the failure handler creates the Replay and voids the timer; no Hedge is ever created for that Token.
6. Queued loser cancelled with executed_work 0.
7. Running loser completes as `completed_loser`; result discarded; work counted.
8. Primary lost with surviving Backup ⇒ no Replay.
9. Primary lost before timer ⇒ Replay only; timer void; no attempt 2.
10. Backup on the Failed domain cannot launch; counted as suppressed.
11. Exactly one winner and one Token completion per Token.
12. Loser completion never overwrites the winner.
13. Stale timer and stale completion events ignored.
14. attempt-0/1 draws identical with and without the attempt-2 stream.
15. New engine in No-Hedge mode matches `simulate_common_state_no_hedge` Token-for-Token.

Ticket 02 (calibration):
16. Fault-aware episodes produce a D-cohort cell where Normal and a Hedge action differ in `q` or `expected_replay_work`; cells report `sample_count >= 400` via episode accumulation; namespace separation from evaluation streams; table determinism.

Ticket 03 (solver/quota):
17. Online allocator: identical policy + trace ⇒ identical assignments; per-action count error within the stated bound; Urgent share never consumed by Regular; no future knowledge.
18. Fixed point converges on a benign cell with all three residuals below tolerance; a forced-nonconvergent configuration is flagged, not disguised; the Hedge budget deducts primary and replay work rates.

Ticket 15 (boundary-pinned correction):
18a. Price zero is the sole cold-start load fixed point; a binding solve fixes rho at 0.9, uses a monotone direct price response and the policy-aware minimum-work certificate, and passes the real 70-cell gate at load 0.5 without grid clamp.

Ticket 04 (comparison):
19. A/B trace identity verification passes; any perturbation fails loudly.
20. Comparison deltas reproduce a hand-computed manual-trace example; the four artifact files commit transactionally (partial failure leaves no directory).

## 19. Verification rule

Every implementation ticket runs the full unit suite and `check`. Ticket 15 additionally builds the complete 70-cell calibration table and verifies the load-0.5 H/D/F official gate twice for deterministic solver serialization. Ticket 04 owns the first paired experiment artifacts.
