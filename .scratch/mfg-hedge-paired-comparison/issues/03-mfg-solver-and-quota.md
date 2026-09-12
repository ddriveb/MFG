# MFG solver and deterministic quota projection

Type: task
Status: resolved
Blocked by: 02

## Goal

Implement the minimal finite-action price-coordination solver and the deterministic quota projector of `../spec.md` sections 9-12, consuming only the ticket-02 calibration table.

## Scope

- Per-state solver with the full load fixed point: interpolate ActionStats at the current `rho`; `J = L + gamma*q + price*g`; numerically stable soft best response; damped policy update; `hedge_work_rate`, `replay_work_rate`, `total_work_rate = primary + hedge + replay`; `rho_new = total_work_rate / state_capacity`; iterate. The price constrains only the Hedge budget; `rho` includes all three work kinds.
- Hedge budget `B(z) = max(0, target_max_utilization * capacity(z) - primary_work_rate - replay_work_rate)`.
- Convergence requires `policy_residual`, `price_residual`, and `load_residual` all `< 1e-6` within `max_iterations = 200`; a non-converged policy is reported as diagnostics only and never called an equilibrium.
- Feasible-set projection: state F ⇒ `{Normal}` before assignment.
- Causal online two-phase per-class deficit allocator (no offline whole-window rounding): phase 1 requests an action from the deficit (`deficit[a] += pi[a]`, largest deficit, tie-break N < D < I, `deficit[requested] -= 1` exactly once, never re-debted); phase 2 applies feasible-set and class-budget projection, so a requested D/I projected to Normal increments `quota_suppressed` and leaves the deficit alone; only the applied action reaches the engine. `|requested_count[a] - pi[a] * n| < 1` per window per class; applied counts carry no such guarantee. Feasible-set projection before the deficit step (F ⇒ all-Normal policy) does not count as suppression. No future knowledge of arrivals, service times, completions, or failures.
- Counters and identities: `policy_requested_action_counts`, `applied_action_counts`, `quota_suppressed`, `executor_hedge_suppressed` (the engine's unchanged `hedge_suppressed`); `requested_D + requested_I = applied_D + applied_I + quota_suppressed` per window, class, and global; `engine.hedge_requested = applied_D + applied_I`; `planned_expected_hedge_work` uses applied actions only and never exceeds the window budget.
- Window budgets: `window_budget = hedge_budget_rate * window_duration`; windows half-open, max width 25.0, forcibly cut at Common State boundaries with fresh windows starting at the boundary (R starts at 220, next boundary 245 — the global 25.0 grid is not resumed); deficit, class budget shares, and suppression counters reset per window; the Urgent share is reserved, never borrowed by Regular, and expires unused at window close.
- Outputs per state: final probabilities per class, price, rho, the three work rates, load_residual, grid-clamp flag, iteration count, `converged`.

## Acceptance criteria

- Spec section 18 scenarios 17-18 pass: identical policy + trace ⇒ identical assignments; requested-count error within the stated `< 1` bound (applied counts carry no such guarantee); the Urgent share is never consumed by Regular Tokens; a forced-nonconvergent configuration is flagged, not disguised; the Hedge budget deducts primary and replay work rates; the counter identities hold per window, class, and global (`requested_D + requested_I = applied_D + applied_I + quota_suppressed`; `engine.hedge_requested = applied_D + applied_I`); feasible-set projection before the deficit step never counts as `quota_suppressed`.
- Degenerate-defaults test: at `healthy_offered_load = 0.7` the D/F budgets are zero and the projected policy is all-Normal there; at 0.6 the D headroom is positive.
- Determinism: same calibration table + config ⇒ identical policy and assignments.
- `python -m unittest discover -s tests -v` green; no evaluation CLI; no third-party dependency.

## Progress log

### Update: 2026-09-04 — MFG solver and deterministic quota

Status: completed

#### Goal

Implement the finite-action MFG fixed-point solver and the causal online two-phase quota allocator over the ticket-02 calibration table, with immutable result/audit types, unit tests, and one real calibration-table integration run. No paired CLI/config, comparison metrics, artifacts, or A/B experiment.

#### Changed

- New `src/mfg_hedge/mfg_solver.py`: `SolverParameters` (strict validation; `from_config`), `StatePolicySolution`, `MFGSolution` (deterministic `to_json_bytes`, fixed state/class/action order, no timestamps), `softmax_stable` (max-subtracted), `arrival_rate_of` and `state_capacity` using exactly the spec's formulas, and `solve_mfg` running the per-state fixed point: stats lookup at current rho (clamp flagged), J with replay penalties (Regular 1.0 / Urgent 5.0), softmax best response over the feasible set (F is exactly all-Normal), damped policy update, work rates computed from the candidate policy, budget deducting primary+replay, damped price step, three residuals, 200-iteration cap, non-convergence returned as full diagnostics with `converged=false`, and a final recomputation pass so reported probabilities/rho/price/rates/stats share one iteration state.
- New `src/mfg_hedge/quota.py`: `project_actions` online allocator — deficit accumulates the solved (feasible-set-projected) policy, requested = max deficit with fixed N<D<I tie-break, deficit decremented exactly once, class-budget projection converts underfunded D/I to Normal with `quota_suppressed` recorded and the deficit untouched; windows are half-open, max width 25.0, forcibly cut at state boundaries (R restarts at 220 with the H policy); class budget shares are proportional to class-weighted planned demand, Urgent's share is never borrowed, unused share expires at window close; per-Token `ActionAssignment`, per-window/class `ClassWindowAudit`, `QuotaWindowAudit`, and `QuotaProjectionResult` with global counters and `to_action_plan()`.
- `src/mfg_hedge/__init__.py`: exported the new API; version 0.7.0 -> 0.8.0 (`pyproject.toml` synced) — new public solver/quota API; calibration schema and scientific results unchanged.
- Tests: `tests/test_mfg_solver.py` (12) and `tests/test_quota.py` (12), written first.

#### Verification

- Red (before implementation): `Ran 225 tests ... FAILED (errors=2)` — `ModuleNotFoundError: No module named 'mfg_hedge.mfg_solver'` (both new suites).
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 251 tests ... OK` (223 + 28 new). Green-phase corrections: class weights were initially missing from quota demand shares (caught by the Urgent-share test); the benign-convergence fixture needed stats where hedge demand dies exponentially (latency-dominated N), since at load 0.7 the D budget is zero by design; test-side window-budget arithmetic was corrected to `budget_rate x duration`.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Degenerate-load tests: at 0.7 the D/F budgets are 0 and quota applies all-Normal there; at 0.6 the D budget rate is 0.15 and a forced all-Immediate policy applies 6 Immediate hedges in one D window.
- Engine integration: applied plan into the hedge engine gives `engine.hedge_requested == applied_D + applied_I` exactly.
- Real integration (recorded, no artifacts): full 70-cell calibration table built in 147.9 s; `solve_mfg` at `healthy_offered_load=0.6` in 0.02 s. Per state:
  - H: probs R {N 0.0025, D 0.0692, I 0.9283} / U {N 0.004, D 0.0987, I 0.8972}; price 0.2267; rho 0.9586; hedge_rate 0.7171 vs budget 0.6000; iterations 200; residuals 1.96e-04 / 1.17e-03 / 2.36e-05; **converged=false**, no clamping.
  - D: probs ~99.7-99.9% I; price 0.8642; rho 1.1928; hedge_rate 0.5891 vs budget 0.1500; iterations 200; residuals 1.68e-06 / 4.39e-03 / 2.06e-08; **converged=false**, no clamping.
  - F: exactly all-Normal; converged in 1 iteration; no clamping.
- Honest report: with the real calibration table, hedging is so latency-effective that the price controller cannot reach the budget fixed point within 200 iterations at the configured step/damping. Per the ticket's rule, this is reported as-is — no retuning, no grid extension, no disguised equilibrium. **Ticket 04's official A/B comparison is blocked on this**: it requires all used state policies converged. The user must decide (spec/ADR-level) whether to raise `max_iterations`/`price_step`, accept diagnostics-only runs, or narrow the mechanism claim.

#### Artifacts

None generated. Source: `src/mfg_hedge/mfg_solver.py`, `src/mfg_hedge/quota.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`. Tests: `tests/test_mfg_solver.py`, `tests/test_quota.py`.

#### Decisions and risks

- The solver deliberately does not auto-extend iterations or relax tolerances; non-convergence is a first-class diagnostic.
- Quota costs come from the solution's final interpolated stats at the solved rho (never from the Token's real attempt-2 draw).
- The real-table non-convergence is a mechanism finding: hedge demand vastly exceeds budget at load 0.6, and the integral price controller with step 0.05/damping 0.2 is slow. Parameters are confirmed mechanism-verification defaults, so changing them needs user/spec approval, not an implementer shortcut.
- `MFGSolution.class_weights` was added so quota demand shares are class-weighted per spec (its absence caused a real bug caught by the Urgent-share test).

#### Next

User decides how to handle the convergence block (ticket 04 prerequisite). Ticket 04 remains open.

## Answer

All acceptance criteria pass: solver fixed point with three residuals and honest convergence flags; causal online quota with all identities (`requested_D+I = applied_D+I + quota_suppressed`, `engine.hedge_requested = applied_D + applied_I`, planned <= share <= window budget); 251/251 tests green; `check` exit 0; degenerate-load behavior verified at 0.7 and 0.6; real integration recorded with non-convergence reported honestly. No paired CLI/config/artifacts/A-B experiment; no third-party dependency; version 0.8.0 for the new public API.

## Comments

- 2026-09-04 (ticket 12): ADR-0007 was implemented exactly as written and is unit-green, but the real 70-cell gate failed honestly — the inner policy/load map is multi-valued near the capacity boundary (bisection's monotonicity assumption fails), and D at load 0.6 is Replay-inclusive over target under every policy. See `.scratch/mfg-hedge-paired-comparison/issues/12-implement-capacity-feasible-solver.md`; ticket 04 remains blocked.

