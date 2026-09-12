# Requalify after qualification namespace contamination and panel-cardinality correction

Type: correction
Status: resolved
Blocked by: none

## Goal

Retire the contaminated r1 qualification identities, correct the target-panel
cardinality without changing the frozen MFG protocol, validate the correction
on a fresh non-formal dry run, and only then permit a fresh r2 qualification
run.

## Scope

- Permanently abandon r1; retain its empty checkpoint and audit log without
  resuming or reusing them.
- Freeze ADR-0038 r2 namespaces, seeds, run ID, fingerprints, and the existing
  qualification/holdout call ceiling.
- Select no more than one future-blind target per episode for continuation and
  finite-K panels.
- Count exactly one baseline plus all supported actions for that target.
- Enforce `forward=32`, `continuation<=288`, and `finite-K<=288` per
  model/iteration panel. No truncation, padding, or duplicate episode.
- Run only the independent 32-episode dry run before r2.

## Frozen r2 identities

- Run ID: `reliability-aware-token-mfg-qualification-20260911-r2`
- Qualification namespaces/seeds: ADR-0038
- Dry-run namespace: `reliability-aware-token-mfg:v1:r2:dry-run`
- Dry-run macro seed: `20260926`
- The frozen holdout identity remains unused.

## Acceptance gates

1. Red tests reproduce the old `continuation=416` multi-target count and the
   one-target-per-episode requirement.
2. Green tests show actual continuation and finite-K calls are bounded by
   `32*9`, with no artificial truncation or fill.
3. Dry run uses only the dry-run namespace and fresh seed; no r1 identity is
   present in its trace or branch keys.
4. Dry run proves sample-floor, CRN, checkpoint-resume, and 1/8-worker byte
   identity.
5. Any missing target, unsupported action, physical failure, unresolved
   checkpoint reservation, or fingerprint mismatch is fail-closed.

## Non-goals

No r2 qualification, K-scale campaign, holdout, candidate selection, MFG
claim, or mathematical/protocol change is allowed under this ticket.

## Next

After all gates pass, explicitly close this ticket and create/use the fresh r2
checkpoint. Do not resume r1 and do not launch holdout in the same action.

### Update: 2026-09-11 — r2 identity and panel correction verified

Status: resolved

#### Goal

Retire the contaminated r1 namespace and verify the one-target panel
construction on fresh non-formal identities before any r2 qualification call.

#### Changed

- Added Proposed [ADR-0038](../../../docs/adr/0038-qualification-r2-after-namespace-contamination.md).
- Added explicit `QualificationPlan.r2()` with fresh qualification
  namespaces/seeds and run ID; the frozen holdout namespace/seed remains
  unchanged and unused.
- Changed continuation and finite-K panel construction to select one target
  per episode and count only that target's baseline plus supported actions.
- Added strict per-episode nine-call accounting; no truncation, padding,
  duplicate calls, or cross-episode target borrowing.
- Expanded the qualification source bundle to include the concrete backend and
  dynamic state-machine implementations, so the abandoned r1 checkpoint
  cannot be resumed under the new source fingerprint.

#### Verification

- Real Red: the new correction tests initially errored because the backend
  had no r2 branch-namespace parameters; the old panel logic also exceeded
  the per-episode bound.
- Focused correction/backend/state-machine tests: `7/7`.
- Affected suites: `72/72`.
- Full suite: `808/808`.
- Config check: exit `0`, `status: ok`.
- `git diff --check`: exit `0`, with only pre-existing unrelated parent
  worktree line-ending warnings.
- Fresh 32-episode dry run used
  `reliability-aware-token-mfg:v1:r2:dry-run`, macro seed `20260926`, and
  produced `forward=32`, `continuation=160`, `finite-K=160`; all values are
  within the frozen `32/288/288` ceilings.
- Dry-run one-worker/eight-worker scientific output was byte-identical.
  Interrupt-after-forward resume reused one committed boundary and matched a
  fresh run byte-for-byte; resumed dispatch accounted for only the remaining
  `672` calls.

#### Artifacts

No r2 qualification checkpoint, qualification result, holdout result, or MFG
artifact was created. The contaminated r1 checkpoint remains retained only
for audit and was not modified.

#### Decisions and risks

- r1 is permanently abandoned; its 32 uncommitted qualification-namespace
  probe calls are not reported as qualification results and cannot be retried
  or reused.
- The correction ticket is complete, but ticket 06 remains blocked because
  the concrete backend is still a bounded K=8 adapter and does not yet
  implement the full two-model, six-iteration formal campaign.
- No holdout identity was consumed and no r2 qualification was started.

#### Next

Provide the formal r2 backend and reconcile the abandoned r1 probe in the
campaign accounting before creating the fresh r2 checkpoint. Only then may
ticket 06 be claimed and K=8 qualification begin.
