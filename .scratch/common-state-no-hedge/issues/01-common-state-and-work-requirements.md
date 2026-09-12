# Common-state timeline, work requirements, and attempt-1 keys

Type: task
Status: resolved
Blocked by: none

## Goal

Implement the pure-function foundation of `../spec.md`: the Common State timeline, the speed model, piecewise work integration, and the attempt-1 stable random key extension of `WorkloadTrace`. No event loop in this ticket.

## Scope

- A frozen `CommonStateTimeline` type (`degraded_start`, `failed_start`, `recovered_start`) with validation `0 <= degraded_start < failed_start < recovered_start` (spec section 1).
- `state_at(timeline, t)` implementing the half-open intervals exactly; Recovered is a return to `H`, no new `CommonState` member.
- `phase_at(timeline, t)` returning the metrics Phase `H`/`D`/`F`/`R`, separating the initial Healthy window from the post-recovery window (spec section 1); metrics and result fields must use the name `completion_phase`, never `completion_state`.
- `speed_at(replica_id, state)` per spec section 2 (Replica 1 always 1.0; Replica 0 is 1.0 in H, `1 / degraded_slowdown` in D; F handled as work loss, not a speed).
- Piecewise work-integration pure functions: `work_executed_between(replica_id, t0, t1, timeline, degraded_slowdown)` and `completion_after(replica_id, start, work, timeline, degraded_slowdown)` returning completion instant or a failure outcome (spec section 2).
- `WorkloadTrace` gains optional `replay_service_times` (default `None`), indexed `[replica][token_id]` for key `(token_id, replica_id, attempt_id = 1)`; a scenario generator draws attempt-1 streams from a new stream label without perturbing arrival, Token-class, or attempt-0 streams (spec section 5).
- Red-green-refactor with pure-function unit tests only.

## Acceptance criteria

- Boundary tests: `state_at` at exactly `degraded_start`, `failed_start`, `recovered_start`, and inside each interval; `phase_at` tests proving phase `R` differs from phase `H` while both report state `H`.
- Speed tests: H/D values on both Replicas with `degraded_slowdown = 2`.
- Work-integration tests with exact hand-computed values, including the spec's example (start t=9, W=2, D at 10 -> completes at 12) and an H->F crossing that reports failure with correct executed/wasted work.
- Trace tests: attempt-0 arrivals/classes/service values identical between plain and replay-augmented traces at the same seed; attempt-1 draws differ from attempt-0; existing Healthy tests stay green unmodified.
- `python -m unittest discover -s tests -v` green; `python -m mfg_hedge check --config configs/v1_minimal.json` exit 0.
- No changes to existing production behavior; no third-party dependency.

## Progress log

### Update: 2026-09-03 — Common-state timeline, work requirements, and attempt-1 keys

Status: completed

#### Goal

Deliver the pure-function foundation of `../spec.md` sections 1-2 and 5: `CommonStateTimeline`, `state_at`/`phase_at`, the speed model, piecewise work integration, the explicit `ExecutionOutcome` contract, and the attempt-1 CRN extension — without any event engine, Replay flow, metrics, CLI, config file, or artifact work.

#### Changed

- Prerequisite record (per user's 2026-09-03 confirmation): `docs/adr/0004-common-state-failure-semantics.md` Status Proposed -> Accepted; `../spec.md` section 11 rewritten from open questions to the three confirmed conclusions (schema_version 2; self-contained `configs/v1_common_state.json`; longer windows + multi-seed for formal experiments).
- New `src/mfg_hedge/common_state.py`: `Phase` enum (H/D/F/R); `CommonStateTimeline` frozen dataclass validating `0 <= degraded_start < failed_start < recovered_start` and rejecting NaN/Infinity/bool/non-numeric boundaries; `state_at` (CommonState H/D/F, Recovered reads H) and `phase_at` (Phase H/D/F/R) over half-open intervals with negative/non-finite/bool time rejected; `speed_at` (Replica 0: 1.0 / `1/degraded_slowdown` / 0.0; Replica 1 always 1.0; slowdown must be finite and >= 1); `work_executed_between` integrating speed piecewise over half-open segments; `completion_after` returning frozen `ExecutionOutcome(completed, terminal_time, executed_work, remaining_work)` — failure at `failed_start` (or at `start` inside F), never resuming after recovery, and a completion exactly at `failed_start` reported as completed with the engine's fault-first ordering left to decide.
- `src/mfg_hedge/workload.py`: `WorkloadTrace` gains optional `replay_service_times` (default `None`); `generate_workload_with_replay` reuses `generate_workload` verbatim and adds attempt-1 streams labeled `service:{replica}:attempt1`; `validate_replay_stream_shape` placed at the trace/engine consumption boundary.
- `src/mfg_hedge/__init__.py`: exported the new API; version bumped to 0.3.0 with `pyproject.toml` (additive capability; Healthy behavior unchanged).
- Tests: new `tests/test_common_state.py` (25 tests) and `ReplayStreamTests` in `tests/test_workload.py` (11 tests), written before implementation.

#### Verification

- Red (before implementation): `.\.venv\Scripts\python.exe -m unittest discover -s .\tests` — `FAILED (errors=2)`: `ModuleNotFoundError: No module named 'mfg_hedge.common_state'` and `ImportError: cannot import name 'generate_workload_with_replay' from 'mfg_hedge.workload'`; all loadable pre-existing tests stayed green.
- Green/Refactor: `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v` — `Ran 93 tests ... OK` (57 pre-existing including all Healthy tests unmodified, + 36 new covering every acceptance bullet: boundaries, H/R state-vs-phase distinction, speeds, the spec's start=9/work=2/D=10 -> 12 example, H/D->F failure splits, Replica 1 immunity, invalid inputs, old trace constructor validity, attempt-0 value identity between plain and augmented traces, attempt-1 independence/determinism/seed sensitivity, shape validation, and the Healthy simulator ignoring replay streams).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.
- No experiment artifacts were generated; `configs/v1_minimal.json` untouched; no third-party dependency added.

#### Artifacts

- Source: `src/mfg_hedge/common_state.py` (new), `src/mfg_hedge/workload.py`, `src/mfg_hedge/__init__.py`, `pyproject.toml`.
- Tests: `tests/test_common_state.py` (new), `tests/test_workload.py`.
- Docs: `docs/adr/0004-common-state-failure-semantics.md` (Accepted), `.scratch/common-state-no-hedge/spec.md` (section 11).

#### Decisions and risks

- `Phase` lives in `common_state.py` next to `phase_at`, keeping `domain.py`'s vocabulary untouched; `completion_phase` naming is reserved for ticket 02/03 result types.
- `completion_after` reports a completion landing exactly on `failed_start` as completed; the spec's fault-first rule is enforced by ticket 02's event ordering, keeping the pure function physics-only.
- `speed_at` returns 0.0 for Replica 0 in F as a capacity representation only; termination is always via `ExecutionOutcome`, never suspension until recovery.
- replica_id is validated as exactly 0 or 1, encoding this slice's two-Replica boundary; a future multi-Expert slice must widen it deliberately.
- The 0.3.0 bump changes `simulator_version` in future summaries (including Healthy runs); schema and metric definitions are unchanged, and existing artifacts remain untouched.
- Tickets 02 and 03 are explicitly not implemented: no event heap, Replay enqueue, fault counters, metrics, CLI, config file, or artifacts were added.

#### Next

Claim ticket 02 (`02-failure-replay-event-engine.md`), now unblocked, to build the event engine on these pure functions.

## Answer

All acceptance criteria pass: the pure timeline/state/phase/speed/integration/outcome layer and the attempt-1 CRN extension are implemented with 36 new deterministic tests; 93/93 tests green; `check` exits 0; Healthy behavior byte-identical; no ticket 02/03 content implemented.

## Comments

- 2026-09-03 (ticket 05): the "byte-identical" wording in this ticket's history and in earlier spec wording means byte-identity **under the same `simulator_version`, configuration, and seed** (excluding `run_id`). The 0.3.0 bump legitimately changes `simulator_version` in future Healthy summaries without changing their scientific content. See `.scratch/common-state-no-hedge/issues/05-harden-ticket-01-boundaries.md` and the spec's "Healthy stability contract".

