# ADR-0006: Finite-Action MFG Solver and Quota Semantics

Status: Accepted (confirmed by the user on 2026-09-04)

Date: 2026-09-04

Builds on: ADR-0001, ADR-0004

Revised after design review (`.scratch/mfg-hedge-paired-comparison/issues/05-correct-mfg-calibration-and-quota-design.md`): fault-aware calibration, the full load fixed point, the corrected Hedge budget, and a causal online quota allocator. Finalized after `.scratch/mfg-hedge-paired-comparison/issues/07-finalize-rho-and-action-projection-contract.md`: the rho coordinate and the requested/applied action contract. Accepted by the user on 2026-09-04. Note: ADR-0007 (Proposed) would supersede this ADR's solver/price/convergence clauses if the user accepts it; see that ADR.

## Context

The MFG arm needs a policy computation that is coordinated (price-mediated) yet honest about hard capacity. Finite prices with Softmax do not by themselves enforce a hard Hedge budget (CONTEXT invariant 5); solver statistics must stay separate from event-simulator evaluation (ADR-0001); constant-state calibration would zero out `replay_probability` and kill the Regular/Urgent distinction; offline whole-window rounding is acausal (it needs the window's final Token count before early Tokens arrive); the rho coordinate must mean the same physical quantity in calibration and in the solver's fixed point; and the default load makes the D/F Hedge budget zero, which would silently degenerate the comparison to No-Hedge.

## Decision

1. Calibration: representative-token / tagged-probe episodes with real H→D→F→R transitions in an independent `calibration:` seed namespace. A cell's rho is the background total work rate (background Primary + Replay) over state capacity faced by the probe, excluding the probe's own work; queueing cost closes only through aggregate rho (finite-action mean-field approximation, no claim on background action composition). Within a cell, the N/D/I probes share one background trace and one set of probe attempt draws (cell-internal CRN). A deterministic bisection on the background arrival rate (bounds [0.05, 3.0], <= 40 iterations, fixed seeds) drives `abs(achieved_rho - target_rho) <= 0.02`; failure is fail-fast. Every populated cell records `target_rho`, `achieved_rho`, `rho_error`, `background_arrival_rate`, `sample_count` (>= 400 cohort samples) and its ActionStats. A deterministic D-state contrast fixture (Normal vs some Hedge action differing in `q` or `expected_replay_work`) is mandatory. The solver never reads evaluation outcomes.
2. Solver fixed point: per Common State, iterate policy and load together — interpolate ActionStats at the current `rho`, update policy via damped soft best response, compute `hedge_work_rate`, `replay_work_rate`, and `total_work_rate = primary + hedge + replay`, then `rho_new = total_work_rate / state_capacity`, and repeat. `rho` is exactly the calibration grid coordinate (population Primary+Hedge+Replay total work rate over capacity). The price constrains only the Hedge budget. Convergence requires `policy_residual`, `price_residual`, and `load_residual` all `< 1e-6` within `max_iterations = 200`. Official runs never clamp the grid: out-of-grid rho yields diagnostics only, never an equilibrium claim or an official A/B result.
3. Hedge budget: `hedge_budget(z) = max(0, target_max_utilization * capacity(z) - primary_work_rate - replay_work_rate)`, so Replay-occupied capacity is never double-assigned to Hedge.
4. Quota: a causal online two-phase per-class deficit allocator. Phase 1 requests an action from the deficit (`deficit[a] += pi[a]`, largest deficit, tie-break N < D < I, `deficit[requested] -= 1` exactly once, never re-debted). Phase 2 applies feasible-set and class-budget projection; a requested D/I projected to Normal increments `quota_suppressed` and leaves the deficit alone. Only the applied action reaches the engine. `|requested_count[a] - pi[a] * n| < 1` per window per class; applied counts carry no such guarantee. Counters are never merged: `policy_requested_action_counts`, `applied_action_counts`, `quota_suppressed` (projection), and `executor_hedge_suppressed` (runtime, the engine's `hedge_suppressed`). Identities: `requested_D + requested_I = applied_D + applied_I + quota_suppressed`; `engine.hedge_requested = applied_D + applied_I`; `planned_expected_hedge_work` uses applied actions only and never exceeds the window budget. Feasible-set projection before the deficit step (F ⇒ all-Normal policy) does not count as suppression. Window budget = `hedge_budget_rate * window_duration`; windows are half-open, max width 25.0, forcibly cut at state boundaries (a fresh window starts at 220 for R, next boundary 245); deficit, budget shares, and suppression counters reset per window; the Urgent share is reserved, never borrowed by Regular, and expires unused at window close.
5. Quota guarantees only `planned_expected_hedge_work <= window_hedge_budget`; realized Hedge work, realized excess, and suppressed counts are reported separately and never claimed as a hard realized-capacity guarantee.
6. Load choice: the first paired run uses a new self-contained config with `healthy_offered_load = 0.6`; load 0.7 remains for stress tests. The degeneracy at 0.7 is a reported result, not a hidden assumption. All listed parameter values are untuned mechanism-verification defaults.

## Consequences

- Replay pressure is visible to the solver, so the Regular/Urgent penalty design survives calibration.
- Calibration and the solver share one physical rho coordinate; the grid cannot be silently stretched to fit a runaway fixed point.
- The MFG equilibrium claim includes queueing feedback (load fixed point), closing the circularity between policy and utilization.
- Assignments are causal, reproducible, and honestly audited at both suppression layers.
- Rejecting any point requires superseding this ADR before implementation.
