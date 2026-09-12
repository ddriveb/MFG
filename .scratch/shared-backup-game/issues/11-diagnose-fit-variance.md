# Diagnose frozen Fit uncertainty

Type: diagnostic analysis
Status: resolved
Blocked by: none

## Scope

Explain the large paired jackknife standard errors in the completed r2 Fit.
Inspect all retained rule rows and inference code. If component-level evidence
is absent from the artifact, replay at most eight profiles on the existing32
Fit scenarios (256 scheduler calls maximum), solely as a separately accounted
diagnostic. Never generate new Fit samples or Validation, change thresholds,
select a candidate, edit historical artifacts, or restart the formal campaign.

## Acceptance criteria

1. Independently reproduce all1536 stored paired jackknife SE values and
   distinguish the worst-bank precision gate from the best/runner-up gate.
2. Identify influential common paths and near-best rules using leave-one
   results, and quantify objective-component/tail contributions where feasible.
3. Check pairing/key alignment and scorer semantics; distinguish demonstrated
   causes from hypotheses and numerical sample-size extrapolations.
4. Write a reproducible diagnostic script and report under this feature;
   generated diagnostics go into a fresh artifact directory. Preserve original
   artifact hashes and record any diagnostic scheduler calls separately.
5. Run the diagnostic checks, required regression and config check; resolve
   this ticket without a Nash/MFG or changed campaign outcome claim.

## Progress log

- Claimed after reading current project instructions, accepted ADR-0016/0017/
  0018, the feature spec and completed campaign/inference tickets. The user
  requested variance diagnosis; the current r2 outcome remains terminal.

### Update: 2026-09-06 — Fit uncertainty traced to sparse fault tails and quota effects

Status: completed

#### Goal

Explain the observed large paired jackknife SE without altering the frozen
campaign or inferring a candidate from insufficient fitting precision.

#### Changed

- Added a reproducible diagnostic script and a detailed evidence report.
- Independently checked all1536 retained SE values and reconstructed the32
  original fitting traces with identical fingerprints.
- Ran6 existing profiles over the32 fitting scenarios,192 diagnostic scheduler
  calls, against the planned256-call diagnostic cap. Saved tagged raw rows and
  exact score/component/leave-one summaries in a fresh diagnostic artifact.
- Found common2/common13 contribute92.29% of the NSNS-vs-NSSS contrast variance;
  F CVaR contributes77.67% and D/F tail components together89.76% using exact
  covariance attribution of delete-one contrasts.
- Identified the pathwise budget mechanism: admitting early Regular copies
  sometimes reduces later Replay, but sometimes suppresses late Urgent
  protection and increases Replay. The two key paths show opposite outcomes.

#### Verification

- `.venv/Scripts/python.exe .scratch/shared-backup-game/diagnose_fit_variance.py`
  completed successfully. All1536 SE values,32 trace fingerprints,6 pooled
  scores and96 paired delete-one values reproduced; source hashes unchanged.
- An independent stdin Python check confirmed all three terminal NSNS banks
  are identical, covariance contributions sum to100%, and original artifact
  SHA-256 hashes still match the pre-diagnosis manifest.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`:
  467 tests passed in51.691 seconds.
- `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`:
  status ok, with the existing single-domain failure headroom finding.

#### Artifacts

- `.scratch/shared-backup-game/diagnose_fit_variance.py`
- `.scratch/shared-backup-game/fit-variance-diagnosis-20260906.md`
- `artifacts/stage4-fit-variance-diagnosis-20260906/diagnosis.json`
- `artifacts/stage4-fit-variance-diagnosis-20260906/tagged-replay-rows.json`

#### Decisions and risks

No production source/config/version/protocol edits, new fitting samples,
Validation workload/evaluation or candidate selection. Formal campaign counts
remain unchanged;192 replay calls are additional separately accounted
diagnostics, not additional independent evidence. The 16-path jackknife itself
is a small-sample approximation. We cannot reliably separate common-fault
variance from nested local randomness with only two populations per path.
Sample-size extrapolations in the report are illustrative, not a new run plan.

#### Next

Design a separate precision/variance-reduction protocol before authorizing
more independent paths or population replicates; retain fit_no_candidate.

## Answer

Resolved. Large uncertainty is quantitatively localized to sparse F tail
observations and opposing policy effects under FIFO reservations. No pairing
or jackknife arithmetic error was found in the audited paths. The completed
formal Fit result remains unchanged and provides no Nash/MFG certificate.
