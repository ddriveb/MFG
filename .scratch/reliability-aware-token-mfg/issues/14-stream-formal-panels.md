# Stream formal qualification panels

Type: performance-correction
Status: resolved
Blocked by: none

## Goal

Prevent K-scale qualification from materializing, copying, or embedding a full
32-episode K=64 trace panel in one process or one JSON checkpoint while keeping
the accepted samples, identities, physics, call counts, and statistical
aggregation exactly unchanged.

## Scope

- Generate each formal trace from its frozen identity only inside its assigned
  work unit.
- Bound each worker payload and live trace set to one episode.
- Persist large immutable scientific payloads as content-addressed atomic blobs;
  keep only hashes and compact aggregates in `checkpoint.json`.
- Aggregate episode outputs in canonical episode order and reproduce the current
  bounded full-panel result field by field.
- Preserve r2 namespaces, seeds, 32-episode scenario schedule, model equations,
  CRN, no-retry semantics, call ceiling, and ADR-0039 gates.

## Acceptance

- A failing test demonstrates that the previous K=64 library payload is larger
  than the frozen bounded-payload threshold.
- Bounded fixtures prove streamed/full-panel field equality, 1/8-worker equality,
  interruption recovery, content-address validation, and exact call accounting.
- An isolated K=64 preflight proves that no serialized work unit or checkpoint
  record contains the 32-trace panel and that peak live traces per worker is one.
- Focused, affected, full-suite, config, compile, and diff checks pass.
- No r2 qualification or holdout identity is dispatched.

## Progress log

### Update: 2026-09-11 — measured unsafe full-panel payload before launch

Status: partial

#### Goal

Verify that ADR-0039 completion was sufficient to start r2 without a mid-run
memory or checkpoint failure.

#### Changed

- Created this execution-only correction ticket and blocked ticket 06 on it.
- No production code has yet been changed under this ticket.

#### Verification

- Built one isolated K=64 S0 trace using the preflight namespace.
- The trace contained 1,904 Tokens and serialized to 36,910,699 bytes.
- The current 32-trace library therefore projects to 1,181,142,368 bytes before
  forward episodes, JSON/base85 expansion, parent copies, or spawned-worker
  copies.
- No formal r2 namespace, checkpoint, scheduler call, or artifact was used.

#### Artifacts

None.

#### Decisions and risks

Starting the current runner would risk multi-gigabyte memory and checkpoint
growth.  Streaming is execution plumbing only; it may not change any frozen
scientific protocol.

#### Next

Add Red tests for bounded episode work units and content-addressed checkpoint
payloads before implementing the streaming path.

### Update: 2026-09-11 — streaming episode execution accepted

Status: completed

#### Goal

Complete Ticket 14's execution-only streaming layer without dispatching the
formal r2 qualification or holdout.  Keep one episode identity per work unit,
bound worker live trace state to one, and make checkpoint recovery atomic and
deterministic.

#### Changed

- Added `StreamingEpisodeSpec` and `StreamingPanel`; panels carry only frozen
  namespace/seed/scenario/episode/K identities and a deterministic panel
  fingerprint, never materialized traces or a `FixedEpisodeLibrary`.
- Added the top-level Windows-spawn-safe
  `streaming_formal_episode_worker`.  It reconstructs exactly one real trace
  inside the worker, delegates to the existing qualification backend, returns
  only packed scientific output plus compact identity/fingerprint and
  `live_trace_count=1` audit fields, and never returns its worker-local
  library.
- Added `execute_streaming_episode_panel` with canonical episode work keys,
  parent-owned reserve/commit accounting, deterministic 1/8-worker merge
  order, and content-addressed atomic JSON blobs.  `checkpoint.json` retains
  only one blob reference per completed unit; committed blobs are hash
  validated on resume.
- Preserved the historical inline checkpoint path for existing callers.  The
  real backend now permits a one-episode worker-local library only in the
  unbounded streaming path; bounded multi-episode behavior is unchanged.
- Added six focused tests covering identity-only payloads, real single-episode
  spawn-safe execution, compact blob checkpoints, 1/8-worker byte identity,
  resume with zero dispatch, and fingerprint rejection.

#### Verification

- Real Red: before implementation,
  `tests.test_token_mfg_qualification_streaming` failed during collection with
  `ModuleNotFoundError: No module named 'mfg_hedge.token_mfg_streaming'`.
- Streaming focused suite: `6/6` passed, including streamed/full-panel
  `ForwardEpisode` byte equality for the bounded real fixture.
- Affected qualification/MFG suites: `51/51` passed.
- Full suite: `820/820` passed.
- `\.venv\\Scripts\\python.exe -m compileall -q src tests`: passed.
- `\.venv\\Scripts\\python.exe -m mfg_hedge check --config
  configs\\v1_minimal.json`: exit `0`, `status: ok`.
- `git diff --check`: passed.
- 1-worker and 8-worker streamed science bytes matched; resumed committed
  units dispatched `0` additional scheduler calls.
- Checkpoint records contained only content-addressed blob references; blob
  hashes were validated on read.  Work-unit payloads contained identities,
  not a 32-trace K=64 panel, and worker audit reported one live trace.
- No formal r2 identity, qualification checkpoint, holdout identity, or
  experiment artifact was generated; no formal scheduler call was dispatched.

#### Artifacts

None.  Source/test changes only; the historical r1 contamination directory
was not modified.

#### Decisions and risks

- Ticket 14 is resolved as an execution-layer correction only.  It does not
  change K, panel counts, namespaces, seeds, CRN, physics, statistical gates,
  or the 59,168-call ceiling.
- The streaming API is ready for the dynamic qualification orchestration;
  this update intentionally does not start that orchestration.
- Ticket 06 remains unclaimed and must still perform its own timing-gated,
  fail-closed qualification sequence.

#### Next

Review and claim Ticket 06 before any formal qualification or holdout start.

## Answer

Pending.
