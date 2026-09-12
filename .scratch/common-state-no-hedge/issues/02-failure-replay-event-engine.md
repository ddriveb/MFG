# Failure/Replay event engine

Type: task
Status: resolved
Blocked by: 01

## Goal

Implement the explicit event-queue engine of `../spec.md` sections 3, 4, and 6 on top of ticket 01's pure functions: piecewise execution, failure, single Replay, recovery, and stale-event handling.

## Scope

- Min-heap event queue (`heapq`) with event records carrying `event_time`, `event_type`, `deterministic_sequence`, `token_id`, `replica_id`, `attempt_id`, `generation` (spec section 6).
- Same-time order exactly as spec section 3, with invalidation/Replay-enqueue as ordered sub-steps of a state change into F and a dispatch pass as the final step at each instant.
- Piecewise remaining-work processing at state boundaries; running and queued domain-A invalidation at F; single Replay on Replica 1 with key `(token_id, 1, attempt_id = 1)`; recovery with empty Replica 0; pure-function dispatch `1 if F else token_id % 2` (no cursor).
- Stale completion handling via per-Replica generation counters plus a Token terminal-state guard.
- Engine runs until every Token has exactly one terminal outcome.
- Result type exposing per-Token outcomes (completion time, `completion_phase` via `phase_at`, `replay_count`, per-attempt executed work) sufficient for ticket 03 metrics; the field name `completion_state` must not appear.
- Manual-trace tests only (no metrics/CLI work).

## Acceptance criteria

- Spec section 8 scenarios 1-10 and 12-16 pass as deterministic unit tests with exact expected values, including: boundary-exact failure at F; the fixed Replay order at `failed_start` (running Token first, then queued Tokens in original FCFS order, all ahead of same-time new arrivals on Replica 1); F-period arrivals not counted as Replays; post-recovery parity dispatch; stale-completion suppression; and the H-only regression where every state-change boundary lies beyond the same trace's last completion time under the Healthy baseline (or a never-trigger fixture), yielding `TokenResult`s identical to `simulate_healthy_no_hedge`.
- Invariants asserted: every Token completes at most once; `replay_count <= 1`; `hedge_launches == 0`.
- Same seed repeated engine runs give identical results (scenario 11 at engine level).
- `python -m unittest discover -s tests -v` green; no third-party dependency; Healthy + No Hedge behavior untouched.

## Progress log

### Update: 2026-09-03 — Failure/Replay event engine

Status: completed

#### Goal

Implement the deterministic Failure/Replay event engine of `../spec.md` sections 2-4 and 6 on ticket 01's pure functions: explicit event queue, piecewise execution, failure, single Replay, recovery, stale-event handling, and raw result records. No metrics, CLI, config file, schema, or artifact work (ticket 03 untouched).

#### Changed

- New `src/mfg_hedge/common_state_simulation.py`: `simulate_common_state_no_hedge(trace, timeline, degraded_slowdown, replica_count=2)`; `_EventType` IntEnum whose values are the same-time ranks (DOMAIN_STATE_CHANGE=1, COMPLETE=4, TOKEN_ARRIVAL=5); `_Event` records carrying event_time/type/sequence/token_id/replica_id/attempt_id/generation; per-Replica deque FCFS queues with no cursor; pure dispatch `1 if F else token_id % 2`; same-time batches processed by rank with invalidation and Replay-enqueue as ordered sub-steps of the F transition and a dispatch pass as the final step per instant; piecewise settle-and-reschedule at the D boundary; fault-first failure at F (running fails even with remaining_work == 0); fixed Replay order (running first, then queued in FCFS order, behind Replica 1's existing queue, ahead of same-time arrivals); generation-based stale completion suppression plus a Token terminal guard; frozen `AttemptRecord` (with `status`, nullable `start_time`/`queue_delay`) and `CommonStateTokenResult` (with `completion_phase`); `CommonStateSimulationResult` with raw counters, drain fields, queue snapshots at `failed_start`/`recovered_start`, and the diagnostic `stale_completion_events_ignored`.
- Minimal post-Red refactor of shared read-only validation: `_validate_trace_shape` moved from `simulation.py` to `workload.py` as public `validate_trace_shape` (Healthy simulator now imports it; identical messages); `_require_slowdown` renamed public `validate_slowdown` in `common_state.py`. Healthy behavior proven unchanged by the unmodified pre-existing suite and scenario 15.
- `src/mfg_hedge/__init__.py`: exported the engine entry point and result types. Version stays 0.3.0.
- New `tests/test_common_state_simulation.py` (26 tests) written before the implementation.

#### Verification

- Red (before implementation): `Ran 98 tests ... FAILED (errors=1)` — `ModuleNotFoundError: No module named 'mfg_hedge.common_state_simulation'`; all other suites green.
- Green/Refactor: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 123 tests ... OK` (97 + 26 new).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- Deterministic engine self-check (1000 Tokens, timeline 100/200/220, seed 20260901): `completed == generated` True; `hedge_launches == 0` True; every `replay_count <= 1` True; `drain_end_time >= last_arrival_time` True; replay_executions = 38 (1 running + 37 queued); queue snapshots (0, 38) at F and (0, 46) at R; drain 1.166.
- Spec section 8 scenario coverage: 1 `Scenario01...`, 2 `Scenario02...`, 3 `Scenario03...`, 4 `Scenario04...` (fault-first at exactly `failed_start`, stale >= 1), 5 `Scenario05...` (executed 1.0 / remaining 1.0 / replay completes 20.75), 6 `Scenario06...` (invalidated queued with executed 0), 7 `Scenario07...`, 8 `Scenario08...`, 9 `Scenario09...` (sentinel 7.25 draw binding), 10 `Scenario10...` (stale == 1, single completion), 11 `Scenario11...` (full-result equality across runs), 12-14 `Scenario121314...` on a 300-Token controlled trace, 15 `Scenario15...` (projection over the nine shared scientific fields, boundaries beyond the Healthy last completion), 16 `Scenario16...` (R1 start order tok1 -> tok3 -> replay(tok0) -> replay(tok2) -> tok4; queue snapshot (0, 3)).
- Engine-boundary tests: replica_count rejects 3/0/2.0/True/"2"/None; slowdown rejects 0.5/NaN/inf/bool/str; timeline type checked; missing/undersized replay streams rejected; non-dense/duplicate/out-of-order Token ids rejected; input trace unchanged after simulation; tokens returned sorted by token_id; `drain_duration == drain_end_time - last_arrival_time`.
- No CLI, config, summary, or artifact changes; no experiment artifacts generated; no third-party dependency.

#### Artifacts

- Source: `src/mfg_hedge/common_state_simulation.py` (new), `src/mfg_hedge/workload.py`, `src/mfg_hedge/simulation.py`, `src/mfg_hedge/common_state.py`, `src/mfg_hedge/__init__.py`.
- Tests: `tests/test_common_state_simulation.py` (new).

#### Decisions and risks

- Completions are scheduled from current-state speed only and rescheduled at boundaries, so stale completions occur naturally and are suppressed by per-Replica generations; the D handler deliberately leaves a physically-exact completion valid (spec section 4 edge), while the F handler fails it (fault-first).
- Queue snapshots are taken inside the state-change handler (failed_start: after Replay enqueue; recovered_start: as queues stand), before same-time completions/arrivals.
- `replay_executions` counts started attempt-1 executions; equality with replayed Tokens is asserted by test and holds because Replica 1 never fails.
- The engine raises AssertionError if dispatch ever starts on a Failed Replica — unreachable by construction, kept as an internal guard.
- No epsilon is used anywhere in time comparisons; manual tests use exactly representable binary fractions.
- Ticket 03 remains open and unstarted: no replay_rate, percentiles, utilization, work ratios, or capacity metrics were computed here.

#### Next

Claim ticket 03 (`03-fault-metrics-cli-and-artifacts.md`), now unblocked, for metrics, CLI, `configs/v1_common_state.json`, summary schema 2, and determinism verification.

## Answer

All acceptance criteria pass: spec scenarios 1-10 and 12-16 are covered by 26 new deterministic tests with exact expected values; 123/123 tests green; `check` exits 0; Healthy + No Hedge behavior unchanged; no ticket 03 content implemented; no artifacts generated.

## Comments

