# Harden ticket-01 boundaries

Type: task
Status: resolved
Blocked by: none

## Goal

Three narrow post-ticket-01 corrections: strict int-only `replica_id`/`replica_count` validation, the ADR-0004 status sentence, and the spec's over-strong Healthy byte-stability wording. Ticket 01 stays resolved; its history is not rewritten.

## Scope

1. `src/mfg_hedge/common_state.py`: `_require_replica_id` accepts only true `int` 0 or 1 — reject bool, float (including 0.0/1.0), str, None, and out-of-range ints — shared by `speed_at`, `work_executed_between`, and `completion_after`; legitimate 0/1 results unchanged. `src/mfg_hedge/workload.py`: `validate_replay_stream_shape` requires a true positive-int `replica_count` (reject bool, float, str, None) without duplicating Token/Primary shape checks.
2. `docs/adr/0004-common-state-failure-semantics.md`: remove the stale "Status remains Proposed" sentence so the whole text reads Accepted.
3. `.scratch/common-state-no-hedge/spec.md`: replace the over-strong "Healthy byte-level outputs stay as they are" with the same-version stability contract; append a correction note to ticket 01's Comments citing this ticket.

## Acceptance criteria

- Failing tests demonstrating 0.0/1.0 replica_id acceptance are added first, then pass after the fix; `validate_replay_stream_shape` rejects non-int `replica_count`.
- ADR-0004 contains no Proposed wording; spec states the same-version byte-stability contract.
- Full test suite and `check` pass; no ticket 02/03 content; no third-party dependency.

## Progress log

### Update: 2026-09-03 — Harden ticket-01 boundaries

Status: completed

#### Goal

Apply the three narrow corrections: strict int-only `replica_id`/`replica_count` validation, the ADR-0004 stale status sentence, and the spec's over-strong Healthy byte-stability wording — without reopening ticket 01 or starting ticket 02.

#### Changed

- `src/mfg_hedge/common_state.py`: `_require_replica_id` now requires a true `int` in {0, 1}; floats (including 0.0/1.0), bools, str, None, and out-of-range ints are rejected. All three public functions (`speed_at`, `work_executed_between`, `completion_after`) share this single entry point; legitimate 0/1 results unchanged.
- `src/mfg_hedge/workload.py`: `validate_replay_stream_shape` now requires `replica_count` to be a true positive int (rejecting bool, float, str, None, non-positive); no Token/Primary shape logic duplicated.
- `docs/adr/0004-common-state-failure-semantics.md`: removed the stale "Status remains Proposed pending user confirmation" sentence; the text now consistently reads Accepted (confirmed 2026-09-03).
- `.scratch/common-state-no-hedge/spec.md`: the over-strong boundary bullet replaced by an explicit "Healthy stability contract" (scientific content stable; byte-identity under same simulator_version+config+seed except run_id; provenance metadata may move across versions; 0.3.0 changes only `simulator_version`; existing artifacts never rewritten).
- Ticket 01: history untouched; one correction note appended to its Comments citing this ticket.
- Tests: `ReplicaIdValidationTests` in `tests/test_common_state.py` (3 tests) and a `replica_count` type test in `tests/test_workload.py`.

#### Verification

- Red (before the fix): `Ran 97 tests ... FAILED (failures=2)` — `test_float_zero_and_one_are_rejected_everywhere` failed with `AssertionError: ValueError not raised : speed_at replica=0.0`, and `test_replica_count_must_be_a_true_positive_int` failed because `replica_count=2.0` was silently accepted.
- Green/Refactor: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 97 tests ... OK`; a transient dead line introduced during refactoring was removed and the final source of `_require_replica_id` was verified by inspection.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- `grep` confirms no "Proposed" wording remains in ADR-0004.

#### Artifacts

- Source: `src/mfg_hedge/common_state.py`, `src/mfg_hedge/workload.py`.
- Docs: `docs/adr/0004-common-state-failure-semantics.md`, `.scratch/common-state-no-hedge/spec.md`, `.scratch/common-state-no-hedge/issues/01-common-state-and-work-requirements.md` (Comments only).
- Tests: `tests/test_common_state.py`, `tests/test_workload.py`.
- No experiment artifacts generated; no configuration file changed.

#### Decisions and risks

- Version stays 0.3.0: the fix narrows validation of inputs that were never valid per the contract; no released scientific behavior changes, so no further bump per the user's instruction.
- The validation tightening could reject callers that previously passed floats by accident; all in-repo callers use ints and the full suite passes.
- Ticket 02 has not been started; tickets 02/03 remain open; ticket 01 remains resolved.

#### Next

Wait for the user's ticket 02 instruction.

## Answer

All three corrections are complete: int-only `replica_id`/`replica_count` validation with red-first tests (0.0/1.0 demonstrably rejected now), ADR-0004 free of Proposed wording, and the spec's Healthy stability contract restated in same-version terms with a pointer from ticket 01's Comments. 97/97 tests green; `check` exit 0; no ticket 02/03 content; no third-party dependency.

## Comments

