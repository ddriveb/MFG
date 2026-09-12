# Finalize rho coordinate and action-projection contract

Type: task
Status: resolved
Blocked by: none

## Goal

Remove the last ambiguities flagged for ADR-0006: the calibration rho coordinate (representative-token/tagged-probe semantics) and the two-phase requested/applied quota contract. Documentation only: no Python, no ticket 02, no config/test/artifact changes.

## Scope

1. Define calibration rho as the background total work rate (background Primary + Replay) over state capacity faced by a representative probe, excluding the probe's own work; cell-internal CRN (same background, same probe draws for N/D/I); deterministic arrival-rate adjustment with fixed seeds, bounds, iteration cap, and tolerance (<= 0.02); per-cell records (target/achieved rho, rho_error, background rate, sample_count); >= 400 probe cohort samples; D-cohort fault contrast; official runs never clamp the grid.
2. Split the online allocator into requested (deficit choice, deficit updated exactly once) and applied (feasible-set + budget projection) phases; distinct counters `policy_requested_action_counts`, `applied_action_counts`, `quota_suppressed`, `executor_hedge_suppressed`; the count identities of the review request.
3. Window budget = hedge_budget_rate x window_duration; boundary-forced windows restart width (R starts a fresh window at 220); per-window reset of deficit/budget/suppression.
4. Sync spec, ADR-0006 (stays Proposed), tickets 02/03/04.

## Acceptance criteria

- All four documents consistent; no remaining ambiguity about rho or requested/applied semantics.
- ADR-0006 still Proposed; tickets 01-04 statuses unchanged (01 resolved; 02-04 open).
- Full suite and `check` pass (zero code change).

## Progress log

### Update: 2026-09-04 — Finalize rho coordinate and action-projection contract

Status: completed

#### Goal

Eliminate the last ADR-0006 ambiguities: the calibration rho coordinate (representative-token / tagged-probe) and the two-phase requested/applied quota contract. Documentation only.

#### Changed

- `../spec.md`: section 9 rewritten around the two-phase online deficit allocator (deficit updated exactly once for the requested action; feasible-set/budget projection yields the applied action; only applied actions reach the engine), the counter set and identities, and the window-budget rules (`window_budget = hedge_budget_rate * window_duration`, boundary-forced fresh windows, per-window resets, Urgent share never borrowable). Section 11 rewritten around tagged-probe calibration (background rho excludes the probe's own work; cell-internal CRN; deterministic bisection [0.05, 3.0] with <= 40 iterations and fixed seeds; tolerance 0.02; per-cell records; >= 400 cohort samples; D-state contrast fixture; same physical rho as the solver). Section 12: rho defined as the calibration coordinate; official runs never clamp the grid. Sections 13-14: counter names and the never-merged suppression split.
- `docs/adr/0006-finite-action-mfg-and-quota-semantics.md`: decisions 1, 2, and 4 rewritten to match; Status remains Proposed.
- Tickets 02 (probe scope, rho records, adjustment acceptance), 03 (two-phase allocator, identities, window budgets), 04 (counter fields, identities, suppression split) synchronized.
- No production code, tests, configs, or artifacts touched; ticket 02 not claimed.

#### Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 185 tests ... OK` after the edits (zero code change).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- `grep` confirms ADR-0006 still reads `Status: Proposed`.

#### Artifacts

- `.scratch/mfg-hedge-paired-comparison/spec.md`, `issues/02..04-*.md`, `issues/07-*.md`
- `docs/adr/0006-finite-action-mfg-and-quota-semantics.md` (still Proposed)

#### Decisions and risks

- Final rho definition: background (calibration) / population (solver) total work rate — Primary plus Hedge plus Replay work — divided by state capacity; the probe's own work is excluded from its calibration cell's background rho.
- Final counter relation: `requested` comes from the deficit step; `applied` after feasible-set and budget projection; `quota_suppressed` counts requested D/I projected to Normal (never pre-projected F-state cases); `executor_hedge_suppressed` (the engine's `hedge_suppressed`, semantics unchanged) counts applied actions not launched at event time. Identity: `requested_D + requested_I = applied_D + applied_I + quota_suppressed`, and `engine.hedge_requested = applied_D + applied_I`.
- The 0.02 rho tolerance and bisection bounds are mechanism-verification defaults; changing them requires a spec amendment.

#### Next

User confirms ADR-0006 (still Proposed); then ticket 02 (calibration) can be claimed.

## Answer

All ambiguities resolved consistently across spec, ADR-0006 (still Proposed), and tickets 02-04; 185/185 tests and `check` pass with zero code/config/test/artifact changes; ticket 02 remains open; ticket 07 resolved.

## Comments

