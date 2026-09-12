# Fix Lazarus batch remainder fairness

Type: bugfix
Status: resolved
Blocked by: none

## Scope

Correct only the finite Lazarus comparator's small-batch integerization. The
fix must preserve locality-first assignment, all non-Lazarus policy behavior,
the physical event engine, CRN streams, smoke sample identities, and the
formal development/holdout gate.

## Frozen correction

- The canonical `batch_identity` is the first Token ID in the current
  simultaneously released batch. It is exogenous and is not derived from
  policy output or realized service.
- Integral residual capacity is assigned in a circular order over the sorted
  currently eligible Replica IDs, starting at
  `batch_identity % eligible_replica_count`. The order is stable within the
  batch and ties remain deterministic.
- Locality-first remains strict: an ingress-local Token uses its local slot
  whenever that Replica still has assigned capacity; only residual Tokens use
  the circular residual order.
- The homogeneous audit uses 512 synthetic two-Token batches, all eight
  Replicas eligible, balanced ingress identities, and checks
  `max(assignments)-min(assignments) <= 1`. This audit is a pure comparator
  audit and consumes no formal scheduler call.
- The original smoke remains exactly 192 scheduler calls. The correction is
  validated by a fresh unique artifact; no old artifact is overwritten.

## Acceptance criteria

- A red test reproduces fixed low-ID residual allocation and passes only with
  batch-identity rotation.
- Locality-first and deterministic tie behavior remain covered.
- The 512-batch homogeneous audit is emitted and satisfies the explicit gap
  bound.
- A fresh 192-call smoke has identical trace/token fingerprints and identical
  rows for the other five policies compared with the prior smoke; Lazarus is
  compared separately under its corrected semantics.
- Focused, affected, full suite, and config check pass.
- No development/holdout experiment starts until this ticket is resolved.

## Progress log

### Update: 2026-09-10 — Narrow fairness correction opened

Status: partial

#### Goal

Remove fixed Replica-ID bias from Lazarus small-batch residual allocation
before freezing any formal development or holdout protocol.

#### Changed

- Created and claimed this isolated comparator bugfix ticket.
- No production code or artifact has been changed in this session.

#### Verification

- Confirmed ticket 02 is resolved and no other ticket is claimed in this
  feature.
- No tests or scheduler calls were run before the red test is added.

#### Artifacts

None.

#### Decisions and risks

The correction is limited to Lazarus batch integerization and its fairness
audit. It does not authorize formal evaluation or reinterpret the existing
mechanism smoke.

#### Next

None.

### Update: 2026-09-10 — Lazarus remainder fairness corrected

Status: completed

#### Goal

Remove fixed low-Replica-ID bias from Lazarus small-batch integerization while
preserving locality-first behavior, CRN identity, and all non-Lazarus physics.

#### Changed

- Updated the accepted routing specification and ADR-0035 Lazarus wording to
  freeze the first Token ID as `batch_identity` and assign residual capacity
  circularly over sorted eligible Replica IDs.
- Updated `src/mfg_hedge/reliability_aware_routing.py` with the rotating
  residual allocator, a pure `lazarus_batch_assignment` helper, and the
  512-batch homogeneous fairness audit. Locality-first remains unchanged.
- Added focused tests for residual rotation, locality, and the explicit
  homogeneous count-gap bound.

#### Verification

- Real Red: the new focused test initially failed at import because the
  fairness audit/assignment API did not yet exist.
- Lazarus focused suite: `.venv/Scripts/python.exe -m unittest
  tests.test_reliability_aware_routing_lazarus -v` — `3/3` passed.
- Affected suite: `.venv/Scripts/python.exe -m unittest
  tests.test_reliability_aware_routing tests.test_reliability_aware_routing_lazarus
  tests.test_replica_routing_baselines tests.test_protection_baseline_experiment
  -v` — `34/34` passed.
- Full suite: `.venv/Scripts/python.exe -m unittest discover -s tests -v` —
  `726/726` passed.
- Config check: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` — exit `0`, `status: ok`.
- `git diff --check` passed.
- Fresh smoke: exactly `192/192` calls completed. The first commit attempt
  under the existing `artifacts` root failed with Windows `PermissionError`
  before publishing and left no r2 directory or staging residue; the retry
  used a fresh artifact root without changing protocol identities or calls.
- r1/r2 comparison: all `160/160` rows for the other five policies are
  byte-value equal across metrics, invariants, trace fingerprint, and token
  fingerprint. All `32` trace fingerprints remain present and matched. The
  `32/32` Lazarus rows changed as expected under the correction.
- Homogeneous comparator audit: `512` synthetic two-Token batches, counts
  `{0:128,1:128,2:128,3:128,4:128,5:128,6:128,7:128}`, maximum gap `0`,
  explicit allowed bound `1`. This pure audit consumes no scheduler call.

#### Artifacts

- `artifacts/lazarus-fairness-r2/reliability-aware-multi-replica-routing-smoke-20260909-r1/`
- Existing r1 artifact was preserved and not overwritten:
  `artifacts/reliability-aware-multi-replica-routing-smoke-20260909-r1/`

#### Decisions and risks

- The correction is comparator-only. Other five policies, physical engine,
  CRN streams, scenario identities, and the 192-call smoke size are unchanged.
- The smoke remains non-inferential and does not establish a routing winner or
  justify formal development/holdout claims.
- The Windows artifact-root commit issue is recorded as an operational
  artifact-path limitation; the successful r2 artifact is complete and
  transactionally committed under a new root.

#### Next

Only after review of this corrected baseline may a separate ticket freeze and
run formal development/holdout routing evaluation.

## Answer

Lazarus comparator fairness is corrected and audited. Ticket 03 is resolved;
formal development/holdout remains intentionally unopened.
