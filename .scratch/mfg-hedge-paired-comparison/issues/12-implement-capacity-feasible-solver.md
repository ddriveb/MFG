# Implement the capacity-feasible solver

Type: task
Status: blocked
Blocked by: none

## Goal

Implement user-accepted ADR-0007: total-incremental-work pricing, signed capacity-gap root finding, the four-status taxonomy, the forced-Normal F exception, and the official solver→quota gate. No paired CLI/config/metrics/artifacts; no A/B experiment.

## Scope

- `mfg_solver.py`: new solver contract per ADR-0007 (inner cold-start fixed point, outer bracket+bisection, statuses, final self-consistency, infeasibility certificate).
- `quota.py`: consume the final `hedge_budget_rate` and reject non-official solutions (H/D must be converged/unclamped/feasible; F must be forced_normal).
- Spec section 12 becomes the ADR-0007 contract; ADR-0007 status becomes Accepted (confirmed by the user on 2026-09-04); ticket 04 becomes blocked by this ticket.
- Version bump per project rules with justification.

## Acceptance criteria

- The twelve required test groups pass (price-0 slack, positive root, complementarity rejection, bracket sign, bracket failure, Primary-only infeasible, F forced-Normal, clamp fail-fast, undamped final residuals, byte determinism, budget closure, gate rejection).
- Real 70-cell table at load 0.6: H/D converged + capacity-feasible + unclamped, F forced_normal with structural violation disclosed, byte-identical rerun. If H or D fails the gate, report honestly and stop.
- Full suite and `check` pass; no third-party dependency; no artifacts.

## Progress log

### Update: 2026-09-04 — Implement the capacity-feasible solver

Status: blocked

#### Goal

Implement ADR-0007 (accepted 2026-09-04): total-incremental-work pricing, signed capacity-gap root finding, status taxonomy, forced-Normal F, official quota gate.

#### Changed

- `docs/adr/0007-...`: Status Proposed -> Accepted (confirmed by the user on 2026-09-04); decisions untouched.
- `spec.md` section 12: now the normative ADR-0007 contract.
- Ticket 04: `Blocked by: 03` -> `Blocked by: 12`.
- `src/mfg_hedge/mfg_solver.py` rewritten: `SolverStatus` (converged/forced_normal/infeasible/nonconverged), signed capacity quantities (`target_work_rate`, `capacity_gap`, `capacity_violation`, `capacity_slack`, `complementarity_residual`), `validate_capacity_acceptance` pure predicate, cold-started inner fixed point (cap 2000, no warm starts, undamped policy residual), bracket expansion (cap 60 doublings from `price_step`) and signed bisection (cap 80), Primary-only infeasibility certificate, forced-Normal F with structural violation, final-state recomputation of every derived quantity, deterministic serialization. `price_damping` retained but unconsumed.
- `src/mfg_hedge/quota.py`: official gate — H/D must be converged, unclamped, capacity-feasible; F must be forced_normal/unclamped; quota consumes the final `hedge_budget_rate`.
- `tests/test_mfg_solver.py` rewritten to the new contract (14 tests); `tests/test_quota.py` updated (gate rejection, new solution fields).
- Version 0.8.0 -> 0.9.0 (`__init__.py`, `pyproject.toml`): public API and serialization changed (status taxonomy replaces the converged flag; new capacity fields; SolverParameters fields changed).

#### Verification

- Red: `Ran 225 tests ... FAILED (errors=2)` — `ImportError: cannot import name 'SolverStatus'`.
- Unit Green: `Ran 256 tests ... OK`; `check` exit 0. All twelve required test groups pass on synthetic tables, including bracket sign crossing, bracket failure (not infeasible), Primary-only infeasibility, clamp fail-fast, undamped final residuals, byte determinism, and quota gate rejection.
- Real 70-cell table (built 121.6 s) + load 0.6 — **the official gate FAILS honestly**:
  - H: `nonconverged / inner_iteration_cap`. The inner policy/load map is multi-valued in the transition region: bisection midpoints at price 0.875 vs 0.8875 return totals 1.8301 vs 1.4410 (non-monotone gap), and at price 0.88359375 the inner iteration hits the 2000 cap. The signed-gap bisection precondition (monotone single root) is violated by the calibration feedback.
  - D: `nonconverged / bracket_failed`. Even at price 50 (hedge ~0.0001), total work = primary 1.2 + replay 0.2318 = 1.4318 > target 1.35. Replay-inclusive base load exceeds the D target at load 0.6; the narrow Primary-only certificate does not trigger, so the honest status is nonconverged, not infeasible.
  - F: `forced_normal`, exact all-Normal, price 0, `structural_capacity_violation=true` (gap +0.30), unclamped — as designed.
  - Byte-identical re-solve on the same table: True.
- No scientific parameter was adjusted to force convergence; no result is labeled an equilibrium.

#### Artifacts

None generated. Source: `src/mfg_hedge/mfg_solver.py`, `src/mfg_hedge/quota.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`. Tests: `tests/test_mfg_solver.py`, `tests/test_quota.py`. Docs: ADR-0007 status line, spec section 12.

#### Decisions and risks

- The implementation is complete and unit-verified against ADR-0007's algorithm, but the real-table verification exposes two ADR-level gaps: (1) the inner fixed point is not single-valued/contractive near the capacity boundary, so bisection's monotonicity assumption fails; (2) with Replay work priced in, the D window at load 0.6 appears structurally over target under every policy — the "small real D headroom" premise of the paired comparison may not exist. These need a design-level decision (e.g., fixed-point selection/continuation rules, a replay-aware feasibility boundary, or a lower evaluation load).
- Ticket 04 remains blocked; nothing may call these results an official A/B outcome.

#### Next

User decision required: amend ADR-0007 (multi-valued fixed-point handling and the D feasibility boundary) or re-scope the evaluation load. Do not start ticket 04.

## Answer

Implementation complete and unit-green, but the mandatory real-table gate failed honestly: H nonconverged (multi-valued inner map), D bracket_failed (replay-inclusive overload). Ticket is blocked pending a design decision; ticket 04 remains unstarted.

## Comments

- 2026-09-04 (ticket 13 / ADR-0008): the user's ruling records this ticket's honest gate failure as the evidence for the boundary-pinned redesign (Proposed `docs/adr/0008-boundary-pinned-capacity-solver.md`) and the first-round load change to 0.5. This ticket stays blocked; its evidence and implementation are the baseline for the ADR-0008 solver revision.
- 2026-09-04 (ticket 15): ADR-0008 was accepted and its boundary-pinned implementation moved to `15-implement-boundary-pinned-solver.md`. This ticket's blocked result remains immutable historical evidence and is not reopened or rewritten.
