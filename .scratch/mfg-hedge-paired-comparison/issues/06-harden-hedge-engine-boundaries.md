# Harden hedge-engine boundaries

Type: task
Status: resolved
Blocked by: none

## Goal

Two narrow boundary fixes on the ticket-01 hedge engine: (1) failure/recovery queue snapshots must count only live (non-cancelled) queued attempts, not cancelled tombstones; (2) service-stream values that the run may consume must be validated (finite, > 0, non-bool) before the event loop, and the hedge engine must fail fast on illegal arrival times and Token classes. Ticket 01 stays resolved; its history is untouched.

## Scope

- Live-queue-length accounting for both snapshots, without changing the tombstone deferred-cleanup strategy, failure/replay order, snapshot timing, or dispatch skip behavior.
- Pre-loop validation: attempt-0 and attempt-1 stream values always; attempt-2 only when the action plan contains Delayed/Immediate Hedge; errors locate (attempt_id, replica_id, token_id). All-Normal runs must not read or validate attempt-2 values (the NaN sentinel test keeps passing).
- Hedge-engine trace boundary validation: arrival times finite, >= 0, non-decreasing; TokenClass must be the real enum; all rejected before the event loop.
- Ticket 01 gets a Comments pointer to this ticket.

## Acceptance criteria

- Red evidence: tombstone snapshots return 1 where 0 is correct; NaN attempt-2 with Immediate Hedge returns a result instead of raising.
- After the fix: snapshots count live entries only; illegal stream values and illegal trace fields raise ValueError before the engine runs.
- Legal controlled-trace scientific results unchanged; full suite and `check` pass; version stays 0.4.0; no ticket 02 or ADR-0006 work.

## Progress log

### Update: 2026-09-04 — Harden hedge-engine boundaries

Status: completed

#### Goal

Fix the two reproduced boundary defects: cancelled tombstones miscounted in queue snapshots, and unvalidated service-stream values flowing into the engine (NaN attempt-2 produced NaN terminal times). Plus hedge-engine trace boundary validation.

#### Changed

- `src/mfg_hedge/hedge_simulation.py`: `_live_queue_length` counts only non-cancelled queued attempts and backs both `queue_length_at_failed_start` and `queue_length_at_recovered_start`; tombstone deferred cleanup, failure/replay order, snapshot timing, and dispatch skip behavior unchanged. Pre-loop validation added: `validate_token_specs`, `validate_stream_values` for attempt-0/1 always and attempt-2 only when the plan hedges.
- `src/mfg_hedge/workload.py`: new `validate_stream_values(streams, attempt_id, kind)` (real, finite, > 0, non-bool; error locates `(token_id, replica_id, attempt_id)`) and `validate_token_specs` (finite, >= 0, non-decreasing arrivals; real TokenClass). Existing validators untouched; Healthy/No-Hedge paths unchanged.
- `tests/test_hedge_simulation.py`: 10 new tests — 4 tombstone snapshot tests (both Replicas, both snapshots), 3 stream-value tests (attempt-2 rejected when hedging; NaN attempt-2 still ignored in all-Normal mode; attempt-0/1 rejected), 3 trace-boundary tests (negative/NaN/decreasing/bool arrivals, same-time arrivals allowed, illegal TokenClass).
- Ticket 01: Comments pointer added; history untouched.

#### Verification

- Red (before fixes): `Ran 185 tests ... FAILED (failures=7)` — including `test_failed_start_snapshot_excludes_replica_one_tombstone` ((0,1) returned where (0,0) is correct), `test_failed_start_snapshot_excludes_replica_zero_tombstone` ((1,0) vs (0,...)), and `test_illegal_attempt2_values_rejected_when_hedging` (NaN accepted, engine returned a result).
- Green: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 185 tests ... OK` (175 + 10 new). One test expectation was corrected during green: trace B's failed_start snapshot is (0,1) because tok0's freshly enqueued Replay legitimately occupies R1 (snapshot is taken after Replay enqueue per spec).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0.
- Manual probe 1: cancelled Hedge tombstone on R1 — snapshots (0, 0) at both F and R.
- Manual probe 2: Immediate Hedge with NaN attempt-2 — `ValueError: hedge draw for (token_id=0, replica_id=1, attempt_id=2) must be finite and > 0, got nan`; no result produced.
- Legal-trace invariance: the full suite — including the 300-Token mixed-action run, the No-Hedge Token-for-Token regression, and all hand-computed scenarios — passes unchanged; only snapshot accounting and pre-loop rejection changed, and no valid trace is affected.

#### Artifacts

None generated. Modified: `src/mfg_hedge/hedge_simulation.py`, `src/mfg_hedge/workload.py`, `tests/test_hedge_simulation.py`, ticket 01 Comments, this ticket.

#### Decisions and risks

- Snapshots keep their spec-defined instants; only the counting changed from raw deque length to live entries.
- attempt-2 values are validated only when a Delayed/Immediate action exists, preserving the No-Hedge never-reads-attempt-2 contract (NaN sentinel test passes).
- Version stays 0.4.0: no published legal scientific behavior changed.

#### Next

Wait for the user's ticket 02 instruction (calibration). ADR-0006 remains Proposed and untouched.

## Answer

Both defects fixed with red-first evidence: snapshots count live queued attempts only (probes: (0,0)/(0,0)); illegal stream values and trace fields fail fast with located errors; 185/185 tests green; `check` exit 0; legal controlled-trace results unchanged; no ticket 02 or ADR-0006 work.

## Comments

