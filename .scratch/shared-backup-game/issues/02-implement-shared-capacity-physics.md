# Implement shared-capacity physics

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement only the first Shared-Backup Expert game slice in a new isolated
module, `src/mfg_hedge/shared_backup.py`:

- immutable, strictly validated Expert/Token/attempt/work trace and state
  contracts;
- a deterministic finite B/C equal-sharing scheduler with three local FCFS
  queues per Expert;
- online per-arrival action requests and per-Expert non-refundable 25-time
  window budget projection for N/D/S/X;
- Primary, Replay, Hedge 0/1/2/3 lifecycle, failure-first batches, timers,
  conservative cancellation, stale completion generations, and full drain;
- pool interval, per-queue conservation, action/budget, token and invariant
  audits;
- deterministic fixtures 1-7, 9, and 12 from the feature specification.

Historical engines, configurations, traces, artifacts, CLI behavior, random
fault sampling, conditional-law/particle models, deviation harnesses, prices,
campaigns, qualification and solver/MFG work are out of scope.

## Acceptance criteria

1. Exact shared capacity satisfies `K=sum(B/C running heads)` and
   `v=min(1, N*c_B/K)` for `K>0`, with `K=0` handled without division.
2. Every event interval records active heads, common per-head speed,
   executed work, capacity bound and contributing attempt keys; all intervals
   satisfy the capacity integral and every busy constrained interval reaches
   the bound.
3. Every one of the 3N queues exposes enqueue/executed/cancelled/discarded/
   remaining audit data satisfying the work conservation identity.
4. Same-time processing is fault/state first, A failure/replay, completion
   batch, timer, arrival, dispatch, then speed/completion rescheduling. A
   completion generation that is no longer current is ignored.
5. Running losers remain in the shared pool and drain to completed losers;
   queued losers cancel with zero executed work; failed work remains in total
   executed and wasted work. Each Token has exactly one winner and at most one
   Replay.
6. Only eligible D Primary-A arrivals can apply protection. Online policies
   receive no hidden requirements, future events, or other Expert state. N/D/S/X
   charges are 0/1/1/2; unaffordable requests project as a whole to N.
7. Focused tests cover the hand-computable fixtures in the scope, cross-Expert
   latency interference, determinism and strict pre-loop validation. Full
   historical tests and the config check remain green.

## Progress log

### Update: 2026-09-05 — Claimed first physics slice

Status: partial

#### Goal

Claim the sole implementation ticket and prepare the red-green-refactor
boundary for the isolated Shared-Backup scheduler.

#### Changed

- Claimed this ticket as the only active implementation ticket for the feature.
- Accepted ADR-0015 and corrected spec section 11 so fixtures 1-7, 9, and 12
  belong to this slice; fixtures 8, 10, and 11 remain assigned to later slices.

#### Verification

- Read `AGENTS.md`, `CONTEXT.md`, issue tracker/update format, ADR-0005,
  ADR-0012, ADR-0014, ADR-0015, the feature spec, ticket 01, and the existing
  scaled/legacy engine and test sources before code changes.
- No implementation or test command claimed in this preparatory update.

#### Artifacts

- `docs/adr/0015-shared-backup-expert-game.md`
- `.scratch/shared-backup-game/spec.md`
- This ticket.

#### Decisions and risks

The scheduler remains isolated from `scaled_experiment.py` and the historical
two-Replica engines. The new module will expose a deliberate isolated API, so
the package version will be reviewed when exports are finalized. No random
common-fault, solver, experiment or artifact workflow is activated.

#### Next

Add failing focused tests for the exact pool, lifecycle, budget, batch and
conservation contracts before implementing the scheduler.

### Update: 2026-09-05 — Deterministic shared physics implemented

Status: completed

#### Goal

Implement the first ADR-0015 slice as an isolated deterministic shared B/C
scheduler with online action admission, conservative copy lifecycle, complete
drain, and auditable capacity and queue conservation, while preserving every
historical engine and experiment entry point.

#### Changed

- Added `src/mfg_hedge/shared_backup.py` with frozen, strictly validated
  `SharedBackupTrace`, global/local Token identities, `AttemptKey`, complete
  attempt-0/1/2/3 work streams, N/D/S/X actions, online `ActionSource`, clean
  `ActionObservation`, per-Expert Decimal budget windows, and immutable result
  records.
- Implemented 3N local FCFS queues, A health speeds, global B/C equal sharing
  with `K` active heads and `v=min(1,N*c_B/K)`, generation-tagged completion
  rescheduling, full same-time fault/completion/timer/arrival ordering, batch
  completion handling, Replay, timers, queued cancellation, running losers,
  failure discard, and full loser drain.
- Added per-interval Pool audits and per-event/cumulative Queue audits. The
  result validates winner, Replay, request/application, timer, loser,
  cancellation, enqueue, and capacity counter identities.
- Added 14 focused tests in `tests/test_shared_backup.py`, including fixtures
  1-7, 9, and 12 plus online observation, deterministic validation, and
  cross-Expert action interference. No legacy engine source was modified.
- Updated ADR-0015 to `Accepted (confirmed by the user on 2026-09-05)` and
  corrected only spec section 11's implementation ownership: fixtures 1-7, 9,
  and 12 are this slice; fixture 8 remains random-fault/information boundary,
  fixture 10 finite deviation evaluation, and fixture 11 nested workload/scale
  validation.
- Exported the isolated API and bumped the package from 0.18.0 to 0.19.0 as a
  new public namespace; historical scientific results and behavior are not
  changed.

Hand-computed fixture evidence from the focused tests:

- Fixture 1: with two F-arrival B/C heads at time 3, both speed `.5` and
  complete at time 5 (latency 2); alone the same unit head completes at time 4
  (latency 1).
- Fixture 2: requirements 1 and 2 complete at times 5 and 6 (latencies 2 and
  3); integrated pool work is exactly 3.
- Fixture 3: three heads run at `1/3` and all complete at time 6 (latency 3).
- Fixture 4: the queued loser has executed work 0; the two running B/C copies
  include a winner at 11.1 and a loser at 14.6, with an active-head-2
  interval; full drain including the other Token's Replay ends at 22.0.
- Fixture 5: an N=1, `c_B=.5` unit head completes at time 5; at `c_B=2`, two
  heads complete at time 4; the idle `K=0` path is finite and does not divide
  by zero.
- Fixture 6: equal B/C work 4 gives spread latencies `[3.0, 3.0]` and a
  failure-time Replay burst latencies `[5.0, 5.0]`; both pool integrals are
  4.0.
- Fixture 7: two S requests apply with charges `[1,1]`, balances
  `2.8125 -> 1.8125 -> .8125`; subsequent X requests project wholly to N
  with zero charge and `.8125` retained; an empty-history X also leaves `.8125`.
- Fixture 9: fault-first produces attempt 0 failed at 20 with executed work 5,
  Replay winner at 22, one voided timer, and one stale completion ignored. A
  simultaneous dual B/C finish at 11 selects Replica 1/attempt 2 as winner and
  Replica 2/attempt 3 as completed loser.
- Fixture 12: the mixed trace has 6 cumulative queue audits and 96 eventwise
  audit rows; maximum queue-conservation residual is `0.0`. Busy B/C executed
  work is `7.0`, exactly the `7.0` capacity integral; total executed work is
  `13.5` and non-winner/wasted work is `8.5`.

Cross-Expert evidence: changing only Expert 0 from N to S changes Expert 1's
completion from 22.0 (latency 1.0) to 23.0 (latency 2.0), while Expert 1's
requested/applied actions and budget balances remain identical. The peer's
immutable trace is reused; the shared scheduler is rerun.

#### Verification

- Red: before implementation,
  `.venv/Scripts/python.exe -m unittest tests.test_shared_backup -v` failed
  during collection with the real `ModuleNotFoundError:
  No module named 'mfg_hedge.shared_backup'`.
- Green focused:
  `.venv/Scripts/python.exe -m unittest tests.test_shared_backup -v` — 14
  tests passed.
- Full regression:
  `.venv/Scripts/python.exe -m unittest discover -s tests -v` — 399 tests
  passed.
- Config gate:
  `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  — `status: ok`; it retained the pre-existing diagnostic that single-domain
  failure load is `1.4` above the target `0.9`.
- Syntax check:
  `.venv/Scripts/python.exe -m py_compile src/mfg_hedge/shared_backup.py
  tests/test_shared_backup.py` — passed.

#### Artifacts

- Source: `src/mfg_hedge/shared_backup.py`
- Focused tests: `tests/test_shared_backup.py`
- Documentation: `docs/adr/0015-shared-backup-expert-game.md`,
  `.scratch/shared-backup-game/spec.md`
- Ticket: this file
- No experiment artifact, report, dataset, qualification/holdout output, or
  new run directory was generated.

#### Decisions and risks

The new API is intentionally isolated and uses deterministic caller-supplied
traces; it does not sample random common faults or expose future state to the
ActionSource. `c_B` is a finite shared physical pool, not a tariff or solver
input. The package version was bumped to 0.19.0 solely for the new exported
namespace. No CLI, formal config, best response, 256-rule search, unilateral
deviation harness, conditional law, particle mean field, solver/MFG,
qualification, holdout, or experiment conclusion was started.

#### Next

Proceed to the separately scoped random public-fault and information-boundary
slice only when its ticket is created and claimed.

## Answer

The deterministic Shared-Backup capacity and physics slice is complete and
resolved. Its exact focused fixtures, capacity integrals, local queue
conservation and cross-Expert interference evidence pass; historical tests and
the minimal config gate remain green. Random faults, solver/MFG, deviation
evaluation, campaigns and artifacts remain explicitly unimplemented.
