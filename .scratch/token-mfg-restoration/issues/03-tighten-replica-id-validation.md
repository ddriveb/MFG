# Tighten strict Replica ID validation

Type: implementation
Status: resolved
Blocked by: none

## Scope

Apply the existing strict Replica-ID contract to the three T1 online entry
points: `TokenObservation`, `OnlineAdmissionDecision`, and
`ReservationLedger.preview()`. A Replica ID is valid only when
`type(primary_replica) is int and primary_replica in (0, 1)`.

This is a narrow validation correction. It must not change valid T1 results,
the historical static path, reservation arithmetic, version, or any T2/T3
scope.

## Acceptance criteria

- `bool` and `float` values `False`, `True`, `0.0`, and `1.0` are rejected at
  all three entry points.
- integer IDs `0` and `1` remain accepted.
- Focused and full regression tests pass without a version bump.
- Shared-Backup ticket 15 has `Blocked by: none`, while retaining its prior
  deferral history.

## Progress log

### Update: 2026-09-06 — Claim strict Replica ID correction

Status: partial

#### Goal

Close the T1 boundary gap that allowed bool/float values to pass as Replica
IDs, before starting the separate T2 Token-cost slice.

#### Changed

- Created and claimed this narrow correction ticket after T1 review.
- No production code or artifact has been changed yet.

#### Verification

- Confirmed tickets 01 and 02 are resolved and no other ticket is claimed.
- Confirmed the three permissive checks in `token_online.py` and prepared
  focused regression coverage before implementation.

#### Artifacts

None.

#### Decisions and risks

This correction preserves all valid T1 behavior and does not authorize Token
payoff, deviation, predictor, price, solver, forward, scaling, Nash, MFG, or
formal experiment work. The package version remains 0.24.0.

#### Next

Run the new failing tests, replace all three checks with strict type identity,
then rerun focused and full validation.

### Update: 2026-09-06 — Complete strict Replica ID correction

Status: completed

#### Goal

Close the T1 validation gap that accepted `False`, `True`, `0.0`, and `1.0`
as Replica IDs, without changing valid T1 results or starting T2 work.

#### Changed

- Added one shared `_replica_id` validator using
  `type(value) is int and value in (0, 1)`.
- Applied the strict validator at `TokenObservation`,
  `OnlineAdmissionDecision`, and `ReservationLedger.preview()`.
- Added focused coverage for all four invalid bool/float values at all three
  entry points; valid integer Replica IDs remain covered by the existing
  online and historical tests.
- Updated Shared-Backup ticket 15's current header to `Blocked by: none` and
  appended a separate dependency-cleanup note, preserving its earlier
  deferral history and `open` status.

#### Verification

- Real Red before the production change:
  `.venv\\Scripts\\python.exe -m unittest
  tests.test_token_online.TokenOnlineContractTests.test_replica_id_rejects_bool_and_float_at_all_online_entry_points -v`
  failed with 12 missing `ValueError` assertions (four invalid values across
  three entry points).
- Focused suite:
  `.venv\\Scripts\\python.exe -m unittest tests.test_token_online -v` —
  `Ran 22 tests`, `OK`.
- Full suite:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `Ran 517 tests in 163.562s`, `OK`.
- Configuration:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json` — `status: ok`; the existing diagnostic
  `single-domain failure ... 1.400 > 0.900` remains unchanged.
- No version bump was made; package version remains 0.24.0. No Python worker
  or formal experiment was started.

#### Artifacts

`src/mfg_hedge/token_online.py`, `tests/test_token_online.py`, this ticket,
and the Shared-Backup ticket-log update. No experiment, candidate, Nash, MFG,
or other generated artifact.

#### Decisions and risks

This was a validation-only correction. It preserves the historical static
path, all legal integer Replica IDs, Reservation arithmetic, and T1 event
semantics. T2 remains limited to Token private cost and the complete single-
Token deviation path; predictor, price, best response, forward, and MFG remain
out of scope.

#### Next

等待用户审查该窄修正；下一正式切片才是 T2 的单 Token 私人成本与完整单边偏离。
