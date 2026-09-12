# Fix MFG-readiness defects before further experiment execution

Type: implementation
Status: resolved
Blocked by: none

## Scope

Fix the seven defects documented by the 2026-09-06 MFG readiness review while
preserving the finite Shared-Backup physics, CRN inputs, causal observations,
complete counterfactual reruns, frozen Stage 4 samples, and historical
artifacts.  The scope is limited to Dual attempt-identity binding, validation
inference direction and fail-closed handling, episode-private stateful policy
lifecycle, timer observation, partial-call accounting, and formal campaign
provenance fields.  No new Fit, Validation, MFG, or other formal campaign is
authorized by this ticket.

## Acceptance criteria

- Dual preserves the default Backup's attempt-2 work/identity and assigns the
  second replica attempt 3, including when the default destination is C.
- The Validation one-sided max-t bound uses the declared inversion direction;
  incomplete jackknife rows return `statistics_insufficient` without raising.
- A stateful online policy receives fresh episode-private state for each
  Expert and episode/counterfactual rerun.
- Voided timers are observed as `voided`, not `pending`.
- Scheduler calls are counted before each attempted baseline call, including a
  partial failure.
- New formal manifests carry deterministic source, full configuration, and
  inference-protocol fingerprints; existing artifacts are untouched.
- Focused regression tests are red before implementation and green afterward;
  the full test suite and config check pass.  No formal campaign is run.

## Progress log

### Update: 2026-09-06 — claim and red-test preparation

Status: partial

#### Goal

Establish the single claimed implementation ticket and add regression tests
for every readiness-review defect before changing production code.

#### Changed

Claimed this ticket.  Production code is unchanged at the start of the
implementation slice.

#### Verification

Pending the focused red test run after the regression tests are added.

#### Artifacts

None.

#### Decisions and risks

The existing negative `fit_no_candidate` result remains historical and will
not be reinterpreted.  This ticket does not authorize a campaign or an MFG
forward model.

#### Next

Add and run the focused regression tests, then implement only the failing
defects.

### Update: 2026-09-06 — readiness defects fixed

Status: completed

#### Goal

Fix the seven MFG-readiness review defects without changing finite-game
physics, CRN, causal observations, scoring semantics, frozen samples, or
campaign authorization.  This update deliberately did not run Fit,
Validation, MFG, or any formal campaign.

#### Changed

- Fixed Dual hedge attempt binding in
  `src/mfg_hedge/shared_backup.py`: the default Backup always remains
  attempt 2 and the other Backup is attempt 3, including default destination
  C.  This restores Single/Dual CRN identity and work preservation.
- Fixed observation timer history to expose the mutually exclusive
  `pending`, `fired`, and `voided` states.
- Added the episode-private `new_episode()` policy factory contract in
  `src/mfg_hedge/game_deviations.py`; stateless immutable ActionRule policies
  remain backwards compatible, while stateful policies receive a fresh
  instance for every Expert and episode/counterfactual rerun.
- Fixed jackknife connector behavior in
  `src/mfg_hedge/stage4_campaign.py`: incomplete rows retain their cluster
  shape and fail closed as `statistics_insufficient`, including legacy empty
  rows; the one-sided bootstrap pivot is now
  `(pseudo_mean-bootstrap_mean)/standard_error`.
- Added per-attempt baseline call observers so calls are counted before the
  scheduler invocation and partial failures report the calls actually tried.
- Added deterministic Stage 4 provenance containing source file hashes,
  full physical/configuration values, config hash, ADR/spec protocol hashes,
  inference formula/seed/replicate/budget fields, attempt binding, and event
  order.  It is frozen before scheduler dispatch and included in both formal
  manifest branches.
- Added `tests/test_mfg_readiness_fixes.py` with regressions for all review
  findings and updated the existing missing-jackknife expectation to the new
  fixed-shape fail-closed contract.  No public API was added; package and
  project versions remain `0.24.0`.

#### Verification

- Red first: `.venv\Scripts\python.exe -m unittest tests.test_mfg_readiness_fixes -v`
  ran 7 tests before implementation and failed with 5 failures and 2
  collection/runtime errors, reproducing Dual/C attempt mismatch, wrong
  max-t direction, S/N state leakage, pending voided timer, empty pseudo-row,
  missing call observer, and missing provenance.
- Readiness regression suite after implementation: 8/8 passed.
- `tests.test_campaign_execution`: 7/7 passed.
- `tests.test_shared_backup`: 14/14 passed.
- `tests.test_game_workload`: 15/15 passed.
- `tests.test_expert_game`: 9/9 passed.
- `tests.test_game_deviations`: 10/10 passed.
- `tests.test_game_solver`: 10/10 passed.
- `tests.test_stage4_campaign`: 9/9 passed.
- `tests.test_shared_backup_performance`: 8/8 passed.
- Full suite: `.venv\Scripts\python.exe -m unittest discover -s tests -v` —
  475/475 passed.
- Config check: `.venv\Scripts\python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  — `status: ok`.
- No Python worker processes remained after verification.  No Fit,
  Validation, candidate, Nash, MFG, or timing campaign was run.

#### Artifacts

No new artifact was generated.  The existing readiness report/probe and the
historical `stage4-pi256-fit-validation-20260906-r2` negative Fit result were
not overwritten or reinterpreted.

#### Decisions and risks

The finite Shared-Backup game, its action projection, CRN streams, event
ordering, scoring and inference samples are unchanged.  The new policy
lifecycle hook is an execution contract: custom stateful policies must
implement `new_episode()`; immutable/stateless ActionRule behavior remains
compatible.  The corrected validation direction is a bug fix to an
unexecuted validation path and does not alter the old `fit_no_candidate`
artifact.  Provenance is internal to Stage 4 manifests, so no version bump is
needed.  None of these fixes establishes a Nash or MFG result.

#### Next

Keep ticket 06 open for a separately authorized campaign launch disposition;
do not reinterpret the historical `fit_no_candidate` result or start MFG.

## Comments

Correction: ticket 06 is already `resolved`, not open.  The next design
frontier is recorded in [ticket 14](.scratch/shared-backup-game/issues/14-design-stage4b-precision-pilot.md)
and Proposed ADR-0019; this pointer does not alter the historical Progress
log above or authorize a pilot/campaign.
