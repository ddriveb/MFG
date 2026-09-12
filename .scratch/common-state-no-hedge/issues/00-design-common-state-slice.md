# Design the Common State + No Hedge slice

Type: task
Status: resolved
Blocked by: none

## Goal

Produce the complete, acceptance-ready design for the deterministic Common State + No Hedge failure simulation: `../spec.md`, implementation tickets 01-03, and any needed ADR drafts. No Python production code is written or modified in this ticket.

## Scope

- Resolve every design question listed in the feature request: common-state timeline, work-requirement semantics, same-time event ordering, failure/Replay semantics, stable random keys for attempt 1, event engine shape, metrics, verification scenarios.
- Split implementation into tickets 01, 02, 03 with an explicit blocking chain.
- Draft new ADRs as Proposed only.

## Acceptance criteria

- `../spec.md` exists and fixes all design decisions so that no item is left for implementers to improvise.
- `01-common-state-and-work-requirements.md`, `02-failure-replay-event-engine.md`, `03-fault-metrics-cli-and-artifacts.md` exist with bounded scopes and executable acceptance criteria; 02 is blocked by 01, 03 is blocked by 02.
- No Python production code or existing artifacts are changed.

## Progress log

### Update: 2026-09-03 — Design the Common State + No Hedge slice

Status: completed

#### Goal

Produce the complete design for deterministic Common State + No Hedge failure simulation: spec, three implementation tickets with a blocking chain, and ADR draft(s), without touching production code.

#### Changed

- Created `../spec.md` fixing all ten design questions: half-open common-state timeline with recommended defaults 100/200/220 and per-window Token budgets; work-requirement speed model with piecewise integration; the six-level same-time event order with rationale; failure/Replay/recovery semantics including the pure-function dispatch rule `1 if F else token_id % 2`; attempt-1 stable keys via optional `WorkloadTrace.replay_service_times`; explicit event-queue engine with generation-based stale completion handling; the full metric definitions; and 15 mandatory verification scenarios.
- Created `01-common-state-and-work-requirements.md` (open), `02-failure-replay-event-engine.md` (open, Blocked by: 01), `03-fault-metrics-cli-and-artifacts.md` (open, Blocked by: 02).
- Created `docs/adr/0004-common-state-failure-semantics.md` with Status: Proposed.
- No Python production code, configuration, tests, or existing artifacts were modified.

#### Verification

- Read before designing: `AGENTS.md`, `CONTEXT.md`, `docs/agents/issue-tracker.md`, `docs/agents/update-format.md`, ADR-0001/0002/0003, `.scratch/healthy-no-hedge/spec.md`, current `src/mfg_hedge/workload.py`, `src/mfg_hedge/simulation.py`, `src/mfg_hedge/metrics.py`, and all six current test files.
- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests` — `Ran 57 tests ... OK` (run after the design work; confirms the design round changed no behavior).
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` — exit 0, `status: ok`.

#### Artifacts

- `.scratch/common-state-no-hedge/spec.md`
- `.scratch/common-state-no-hedge/issues/00-design-common-state-slice.md` (this ticket)
- `.scratch/common-state-no-hedge/issues/01-common-state-and-work-requirements.md`
- `.scratch/common-state-no-hedge/issues/02-failure-replay-event-engine.md`
- `.scratch/common-state-no-hedge/issues/03-fault-metrics-cli-and-artifacts.md`
- `docs/adr/0004-common-state-failure-semantics.md` (Proposed)

#### Decisions and risks

- Recommended timeline: H `[0,100)`, D `[100,200)`, F `[200,220)`, R `[220,+inf)`; expected arrivals per window at lambda=1.4 are ~140 / ~140 / ~28 / remainder; controlled runs require the last arrival beyond `recovered_start` (fail fast otherwise).
- Same-time event order adopted as recommended (state change, invalidation, Replay enqueue, completion, arrival, dispatch) with recorded rationale; changes require spec amendment.
- Dispatch is a pure function with no cursor, so the post-recovery round-robin rule is defined by construction.
- Replay uses `(token_id, replica_id, attempt_id = 1)` stored in an optional trace field; attempt-0 sampling is provably untouched.
- Summary schema for the new scenario is proposed as `schema_version: 2`; Healthy stays 1.
- ADR-0004 is Proposed and awaits user confirmation; the spec's section 11 lists the remaining research questions.

#### Next

User confirms ADR-0004 and the spec's open research questions, then claims ticket 01.

## Answer

All acceptance criteria pass: `../spec.md` fixes every listed design decision; tickets 01-03 exist with bounded scopes, executable acceptance criteria, and the required blocking chain (02 by 01, 03 by 02); no production code or existing artifacts were changed (57/57 tests green after the design round).

## Comments

