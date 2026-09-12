# Fix LÆDGE online boundary audit

Type: correction
Status: resolved
Blocked by: none

## Scope

Correct only the idle-release LÆDGE boundary audit used by the frozen
cancellation-semantics ablation. Capture the failed-start and recovered-start
states online, before same-time completion, arrival, and idle-release handling,
without changing scheduling, cancellation, Replay, CRN, workload, sample, or
statistical semantics. Preserve the fixed-dispatcher reconstruction and all
historical artifacts.

The audit must distinguish running/live attempts, pinned queued attempts, and
unassigned waiting work. Unassigned waiting work must not be assigned to a
future Replica by terminal-record reconstruction.

## Acceptance

- A regression test is red against the existing terminal-record snapshot.
- Online boundary capture makes `failed_replica_a_is_empty` reflect the actual
  failed-start state.
- The 47 failed r1 episodes replay successfully with matching CRN fingerprints,
  no new physical invariant failures, and zero Replica-A live/running work at
  failed_start.
- Focused, affected, full suite, and config check pass.
- Ticket 06 is restored to `claimed` only after this ticket is resolved.

## Progress log

### 2026-09-09 — claimed

#### Goal

Repair the narrow online LÆDGE boundary-audit defect before resuming the
frozen cancellation-semantics experiment.

#### Changed

None yet. Ticket 06 is blocked while this correction is in progress.

#### Verification

Pending red test, implementation, 47-episode replay, and repository checks.

#### Artifacts

No artifacts are created or modified by this correction.

#### Decisions and risks

The correction must not alter the physical engine or the frozen experiment
protocol. The r1 physical-failed artifact remains immutable.

#### Next

Add the failing boundary regression test, then implement online capture and
run the required verification.

### 2026-09-09 — resolved

#### Goal

Repair the narrow online LÆDGE boundary-audit defect before resuming the
frozen cancellation-semantics experiment.

#### Changed

- Added online failed-start and recovered-start capture in
  `src/mfg_hedge/laedge_episode.py`, after fault/Replay handling and before
  same-time completion, arrival, and idle-release processing.
- Added private idle-release audit data distinguishing running attempts,
  pinned queued attempts, and unassigned waiting attempts.
- Updated idle-release attribution validation to use the captured online
  boundary state; `failed_replica_a_is_empty` remains a live invariant and was
  not excluded.
- Preserved terminal-record reconstruction for the fixed dispatcher and kept
  the public historical result schema unchanged.
- Added the regression test for future Replica backfill in
  `tests/test_protection_cancellation_ablation.py`.

#### Verification

- Real Red: the new regression test failed before the fix with
  `failed.live_attempts[0] == 1` instead of `0`.
- Focused: `12/12` passed.
- Affected: `92/92` passed.
- Full suite: `667/667` passed.
- Config check: `status: ok`.
- r1 replay: all `47` failed episodes, `188/188` arm calls, `47/47` trace
  fingerprints matched, `94/94` LÆDGE failed-start A-zero checks passed, and
  `188/188` attribution/invariant validations passed with no failures.

#### Artifacts

No artifact was created or overwritten. The r1 physical-failed artifact
remains unchanged.

#### Decisions and risks

This is an audit-boundary correction only. It does not change scheduling,
cancel semantics, Replay, CRN streams, workloads, samples, or statistics.
Unassigned waiting work is not retroactively assigned to a Replica.

#### Next

Ticket 06 is restored to `claimed` and will rerun the original frozen r2
panel with a new unique artifact directory.
