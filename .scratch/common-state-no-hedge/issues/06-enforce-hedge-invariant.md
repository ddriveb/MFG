# Enforce the hedge_launches invariant

Type: task
Status: resolved
Blocked by: none

## Goal

Close the guard-rail hole in `build_common_state_summary`: `invariants["hedge_launches"]` carried the raw counter and the violation scan only caught `value is False`, so a result with `hedge_launches=7` produced a summary instead of raising. Add an explicit zero-check without changing the structure or scientific content of valid summaries.

## Scope

- Red first: prove via `dataclasses.replace(result, hedge_launches=1/7)` that an anomalous result is wrongly accepted today.
- Green: explicitly require `result.hedge_launches == 0` and raise `RuntimeError` naming `hedge_launches` otherwise; no truthiness or `value is False` reliance; keep `faults.hedge_launches = 0` and `invariants.hedge_launches = 0` byte structure for valid runs; no new or renamed fields; other boolean invariants unchanged.
- Append a correction note to ticket 03's Comments citing this ticket.

## Acceptance criteria

- New tests: 0 passes; 1, 7, and -1 raise RuntimeError naming hedge_launches; a freshly built controlled summary equals the existing schema-2 artifact content minus run_id.
- Full suite and `check` pass; an in-memory probe demonstrates hedge_launches=7 is now rejected.
- Version stays 0.3.0; no config, engine, or metric-definition changes; no new artifacts; no Hedge/MFG/multi-seed work.

## Progress log

### Update: 2026-09-03 — Enforce the hedge_launches invariant

Status: completed

#### Goal

Close the guard-rail hole where `build_common_state_summary` accepted a result with nonzero `hedge_launches` because the violation scan only caught `value is False`, without changing valid summary structure or content.

#### Changed

- `src/mfg_hedge/common_state_metrics.py`: explicit `if result.hedge_launches != 0: raise RuntimeError("hedge_launches must be 0 in this slice, got ...")` before the invariants block; no truthiness reliance; `faults.hedge_launches = 0` and `invariants.hedge_launches = 0` unchanged for valid runs; no fields added or renamed.
- `tests/test_common_state_metrics.py`: `HedgeInvariantTests` (0 passes; 1/7/-1 raise RuntimeError naming `hedge_launches`, via `dataclasses.replace` on a legal result) and `ControlledSummaryStabilityTests` (fresh in-memory controlled summary equals the existing schema-2 artifact minus `run_id`; skips cleanly when no artifact exists).
- Ticket 03: one correction note appended to Comments; its Progress log untouched.

#### Verification

- Red (before the fix): `Ran 152 tests ... FAILED (failures=1)` — `test_nonzero_hedge_launches_are_rejected` failed with `AssertionError: RuntimeError not raised`, proving `hedge_launches=1` was wrongly accepted. (A missing `PROJECT_ROOT` import in the new test module caused a transient loader error during test authoring, fixed before the recorded red run.)
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 152 tests ... OK` (149 + 3 new).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- In-memory probe: a 200-Token controlled result with `dataclasses.replace(hedge_launches=7)` now raises `RuntimeError: hedge_launches must be 0 in this slice, got 7`.
- The artifact-consistency test passed against `artifacts/single_expert_mechanism_check-common-state-no-hedge-t1000-20260903T113119835328Z/summary.json`, proving valid output is byte-content stable across the fix; no artifact was created, modified, or overwritten.

#### Artifacts

None generated. Modified: `src/mfg_hedge/common_state_metrics.py`, `tests/test_common_state_metrics.py`, ticket 03 Comments.

#### Decisions and risks

- Version stays 0.3.0: valid runs produce identical structure and scientific content; only the anomalous-input guard was added.
- The engine can never produce nonzero `hedge_launches` today; the guard exists for future construction paths and corrupted results.
- The other boolean invariants keep their existing `is False` scan semantics, which is exact for booleans.

#### Next

Wait for user direction (Hedge slice spec or multi-seed pilot); neither started here.

## Answer

All acceptance criteria pass: nonzero `hedge_launches` (1/7/-1) now raises `RuntimeError` naming the field, zero passes, valid summaries are unchanged (artifact-consistency test), 152/152 tests green, `check` exit 0, in-memory probe confirms rejection. No config, engine, or metric-definition changes; no new artifacts; no Hedge/MFG/multi-seed work.

## Comments

