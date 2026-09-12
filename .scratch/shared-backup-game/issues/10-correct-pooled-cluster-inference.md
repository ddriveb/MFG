# Correct pooled nonlinear objective inference

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement ADR-0018 only: replace standalone common-path objective scoring with
exact delete-one-common-path pooled recomputation, paired cluster-jackknife
standard errors, and simultaneous max-t resampling of paired jackknife
pseudo-values. Preserve all frozen Stage 4 samples, physics, CRN, rule bank,
seeds, budgets and claim boundaries. Do not restart Fit or Validation.

## Acceptance criteria

1. A fixture with one path missing F/U but a complete pooled sample produces a
   finite point objective and finite delete-one values when the remaining pool
   is complete.
2. Full-sample or delete-one missing cohorts fail closed as
   `statistics_insufficient`, never zero fill or path deletion.
3. Hand-computed paired jackknife SE and pseudo-values match the formulas in
   ADR-0018.
4. Fit selection uses paired delete-one recomputations; Validation max-t uses
   jointly resampled pseudo-values and is deterministic.
5. Worker results remain compact, deterministic and contain no full non-tagged
   Token result objects.
6. Existing equivalence, full regression and config checks pass. No formal
   campaign or scientific artifact is produced.

## Progress log

### Update: 2026-09-06 — Claim pooled-inference correction

Status: partial

#### Goal

Correct the scorer/cluster conflict exposed by the first formal Fit attempt
without changing the frozen sample or physical experiment.

#### Changed

- Added and accepted ADR-0018 from the user's explicit instruction to begin
  the proposed correction.
- Created and claimed this ticket as the sole active ticket.
- Kept tickets 06 and 09 blocked until the correction passes its tests.

#### Verification

- Confirmed the full scorer pools raw Tokens across episodes, while the failed
  adapter separately called it on each common path and therefore required a
  cohort that the estimand does not require at path level.
- No scheduler call or formal campaign was started in this claim step.

#### Artifacts

- `docs/adr/0018-pooled-objective-cluster-inference.md`
- This ticket only; no experiment artifact.

#### Decisions and risks

- Common fault paths remain the top-level clusters and both nested populations
  remain inside their path cluster.
- Missing cohorts remain fail-closed for the full or any delete-one pooled
  sample.

#### Next

Write failing inference tests, then implement the smallest exact correction.

### Update: 2026-09-06 — Complete pooled jackknife inference correction

Status: completed

#### Goal

Replace invalid standalone-cluster scoring with exact pooled nonlinear
objective inference while retaining the frozen Stage 4 experiment.

#### Changed

- Replaced worker `cluster_objectives` with exact `delete_one_objectives`;
  deleting a common path removes both nested population rows and scores every
  retained path together.
- Replaced solver `paired_cluster_deltas` with
  `paired_delete_one_deltas` and the ADR-0018 cluster-jackknife SE formula.
- Renamed serialized iteration fields to identify delete-one and jackknife
  semantics explicitly.
- Validation now retains full-sample improvements, paired delete-one
  improvements and paired jackknife pseudo-values. Its deterministic max-t
  resampling uses the same common-path draw jointly across all Expert/rule
  rows.
- Kept missing full or delete-one cohorts fail-closed; no zero fill, cluster
  removal or sample extension was introduced.
- Updated the spec and ADR-0016 with ADR-0018's narrow supersession and bumped
  the public serialization/API version from 0.23.0 to 0.24.0.
- Added three new focused tests and extended serialization assertions.

#### Verification

- Real Red: `.venv/Scripts/python.exe -m unittest tests.test_stage4_campaign -v`
  failed at import because `_delete_one_objectives` did not exist.
- Focused Green:
  `.venv/Scripts/python.exe -m unittest tests.test_stage4_campaign tests.test_game_solver tests.test_campaign_execution -v`
  ran 26 tests, all passed.
- Full Green: `.venv/Scripts/python.exe -m unittest discover -s .\tests -v`
  ran 467 tests, all passed in 50.493 seconds.
- Config gate: `.venv/Scripts/python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  returned exit 0 and `status: ok`, retaining the known capacity warning.
- Frozen NNNN baseline diagnostic used all 32 Fit scenarios and found all 16
  delete-one values finite; omitted cluster 13 produced
  `36.40114007370042` instead of an incomplete standalone F/U score.
- A separate one-task exact tagged worker diagnostic completed 32 calls and
  returned the same 16 finite delete-one values, including the same cluster-13
  value. Neither diagnostic was entered into formal Fit accounting.
- Hand check: delete-one values `(1,2,4)` produce ADR-0018's exact jackknife
  SE; full estimate 2.5 produces pseudo-values `(5.5,3.5,-0.5)`.

#### Artifacts

- `docs/adr/0018-pooled-objective-cluster-inference.md`
- `src/mfg_hedge/campaign_execution.py`
- `src/mfg_hedge/game_solver.py`
- `src/mfg_hedge/stage4_campaign.py`
- `tests/test_stage4_campaign.py`
- Updated spec, ADR-0016, package version and ticket records.
- No Fit, Validation, candidate, Nash, MFG or campaign artifact.

#### Decisions and risks

- Both nested populations stay inside their common-path cluster; Tokens,
  populations and Experts are not promoted to independent top-level samples.
- The 4096-replicate max-t method remains numerical inference, not a
  finite-sample theorem.
- The failed launch's 8,224 formal calls remain disclosed. A future complete
  run's cumulative maximum is 929,312 calls across attempts, below the hard
  one-million boundary.

#### Next

Re-claim ticket 06 in a separate turn and rerun the unchanged frozen campaign.

## Answer

Resolved. The original cluster-13 F/U absence no longer invalidates the pooled
estimand, every inferential replicate is explicit and fail-closed, and no
formal campaign was restarted.
