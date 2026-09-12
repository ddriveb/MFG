Type: implementation

Status: resolved

Blocked by: none

## Goal

Apply only the three authorized T4c execution-contract corrections before the
formal 4,096 x 20 diagnostic: use the frozen supported terminal status,
include the existing per-round physical metrics, and fail closed as
`physical_failed` when an ordinary simulation or invariant exception occurs.

The finite-population model, sampling protocol, physical engine, parameters,
and call budget are out of scope.

## Scope and acceptance criteria

1. Successful convergence is reported as `supported_soft_fixed_point` at both
   the terminal result and the converged iteration row.
2. Every round row reports latency, Replay, Hedge, total/wasted work, and
   Protection-Storm metrics using the existing attribution metric function and
   its existing field definitions.
3. Ordinary `Exception` raised at simulation or invariant boundaries produces a
   `physical_failed` terminal result and preserves the original exception type,
   message, round, and consumed calls. `KeyboardInterrupt` and `SystemExit`
   are not caught.

No ADR, mathematical model, parameter, sample, or unrelated production change
is permitted in this ticket.

## Progress log

### Update: 2026-09-08 — Created narrow T4c execution-contract fix

Status: claimed

#### Goal

Create the sole implementation ticket for the three authorized T4c output and
failure-contract corrections, while keeping formal ticket 16 blocked.

#### Changed

- Created this ticket and claimed it as the only active implementation ticket.
- Ticket 16 remains blocked by this ticket until the focused verification and
  config check pass.

#### Verification

- Read-only inspection confirmed the three frozen gaps in the current T4c
  runner: terminal status spelling, missing per-round physical metrics, and
  missing `physical_failed` normalization.
- No production code or artifact was changed in this update.

#### Artifacts

None.

#### Decisions and risks

- The correction is deliberately limited to the accepted execution output
  contract; no new ADR or scientific semantic change is introduced.

#### Next

Write only the focused red tests for the three acceptance criteria, then make
the minimal implementation changes.

### Update: 2026-09-08 — T4c execution contract corrected

Status: completed

#### Goal

Correct only the accepted T4c execution output contract before formal
diagnostic launch.

#### Changed

- Added `supported_soft_fixed_point` as the successful terminal and iteration
  status.
- Added per-round latency, Replay, Hedge, total/wasted work, and
  Protection-Storm summaries by reusing `build_episode_metrics` fields from
  `attribution_metrics.py`.
- Added fail-closed `physical_failed` handling for ordinary exceptions at
  simulation, pathwise-result, and attribution-invariant boundaries, retaining
  exception type, message, round, reserved calls, and attempted calls.
- Added only the three focused tests covering these changes.
- No version, ADR, mathematical model, parameter, sample, or artifact change.

#### Verification

- Real Red: before implementation, the focused suite reported 12 tests with 1
  failure and 2 errors: missing successful-status constant and metric helper,
  and injected simulation failure incorrectly returned `statistics_insufficient`.
- Focused: `.venv/Scripts/python.exe -m unittest tests.test_token_mfg_t4c_supported_soft_response -v` — `12/12` passed.
- Full: `.venv/Scripts/python.exe -m unittest discover -s tests -v` — `597/597` passed.
- Config: `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json` — `status: ok`.
- The focused exception injection confirmed `RuntimeError`, original message,
  round `0`, and one attempted call are preserved as `physical_failed`.

#### Artifacts

None. The formal run directory remains fresh and absent.

#### Decisions and risks

- `KeyboardInterrupt` and `SystemExit` remain uncaught because handling is
  limited to ordinary `Exception` boundaries.
- The ticket changes only execution reporting/fail-closed behavior; it does
  not alter the physical or statistical model.

#### Next

Unblock and claim ticket 16, then launch the authorized formal T4c diagnostic
into its fresh non-overwriting artifact directory.
