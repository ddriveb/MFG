# Design the boundary-pinned solver and load 0.5 (ADR-0008)

Type: task
Status: resolved
Blocked by: none

## Goal

Record the user's 2026-09-04 ruling as Proposed ADR-0008: replace the nested rho fixed point with a capacity-boundary-pinned solve (rho pinned at target utilization, one-dimensional monotone price root) and move the first-round load from 0.6 to 0.5. Documentation only; no code, tests, configs, or artifacts.

## Scope

- ADR-0008 (Proposed): the boundary-pinned solver, the policy-aware feasibility certificate, the load change, and the exact supersession boundaries with ADR-0006/0007.
- spec section 12 gains a pending-revision note pointing to ADR-0008; ticket 04 stays blocked by ticket 12.
- Ticket 12 gets a Comments pointer; its blocked status and history stay untouched.

## Acceptance criteria

- ADR-0008 Proposed, short, and consistent with the user's verified numbers (D min incremental work, 0.5636 load bound, 0.5 headroom, ~36.5%/99.98% hedge at the D boundary).
- Full suite and `check` pass with zero code change.

## Progress log

### Update: 2026-09-04 — Design the boundary-pinned solver and load 0.5 (ADR-0008)

Status: completed

#### Goal

Record the user's 2026-09-04 ruling as Proposed ADR-0008 and align the design documents, without touching code, tests, configs, or artifacts.

#### Changed

- `docs/adr/0008-boundary-pinned-capacity-solver.md` (Proposed): split the solve at the capacity-binding diagnostic (price-0 slack path vs rho pinned at target utilization with a one-dimensional monotone price root); policy-aware feasibility certificate (`primary + arrival * Σ weight * min_a (g+r)` at the boundary rho); first-round load 0.5; explicit supersession boundaries with ADR-0006 (load choice) and ADR-0007 (nested iteration mechanics, certificate breadth); ADR-0007's accounting, taxonomy, F exception, quota closure, and gate remain in force.
- `spec.md` section 12: pending-revision note updated to ADR-0008 with the user's verified numbers; ticket 04 remains blocked by ticket 12.
- Ticket 12: Comments pointer to ADR-0008; blocked status and history untouched.

#### Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 256 tests ... OK` (zero code change).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.

#### Artifacts

Documentation only: ADR-0008 (Proposed), spec section 12 note, tickets 12 (Comments) and 13 (this one).

#### Decisions and risks

- The ADR records the user's verified diagnostics: D minimal incremental work 0.19356 (R) / 0.21409 (U), weighted 0.19766, strict feasibility bound ≈ 0.5636, load-0.6 minimum total 1.4372 > 1.35, and the load-0.5 boundary solution (≈ 36.5% Regular / ≈ 99.98% Urgent Hedge).
- The boundary rho (0.9) lies inside the calibration grid, so the binding solve never clamps; non-binding solves that wander off-grid still fail fast.
- Whether the boundary-pinned solve passes the real 70-cell gate is an implementation result verified after acceptance; the ADR does not assume it.

#### Next

User confirms ADR-0008; then one solver revision ticket against the real table, then ticket 04.

## Answer

ADR-0008 (Proposed) records the boundary-pinned solver and load-0.5 ruling with exact supersession boundaries; spec and tickets aligned; 256/256 tests and `check` pass with zero code change; ticket 04 stays blocked.

## Comments

