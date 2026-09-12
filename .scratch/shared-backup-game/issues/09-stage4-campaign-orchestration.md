# Complete Stage 4 frozen campaign orchestration

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement only the missing execution and statistical orchestration required to
run the already frozen Stage 4 finite `Pi_256` campaign. This ticket does not
change the Shared-Backup physics, action rules, scorer mathematics, CRN,
sample counts, population counts, iteration limit, call budget, or claim
boundary.

The implementation must connect the existing exact tagged batch runner to the
three-start Fit state machine, freeze a Fit candidate by value before
Validation, run all eight Expert/all-256 Validation deviations on independent
data, compute the predeclared simultaneous one-sided 95% restricted-regret
bound, and commit complete results transactionally under a fresh run ID. It
must not start a formal campaign until its own tests and preflight pass.

## Frozen protocol

- `N=8`, `c_B=0.5`, slowdown `2.0`, hedge delay `1.5`, zero tariff.
- Canonical 256 `N/D/S/X` rules; starts `NNNN`, `NSSN`, `XXXX`; at most 32
  updates per start.
- Fit: namespace `shared-backup-game:v1:stage4-fit`, macro seed `20260905`,
  common paths `0..15`, populations `0` and `1`, 32 scenarios.
- Validation: namespace `shared-backup-game:v1:stage4-validation`, macro seed
  `20260906`, common paths `0..31`, populations `0` and `1`, 64 scenarios;
  trace and fault fingerprints must be disjoint from Fit.
- Call upper bounds are exactly `789504` Fit calls and `131584` Validation
  calls, `921088` total, below the hard one-million limit. No retry, adaptive
  sample extension, or result-driven budget change.
- Windows spawn, at most 8 workers and 8 in-flight tasks, task grain
  `(deviating_expert, candidate_rule)`, fixed scenario order, compact worker
  rows, parent-owned call accounting, deterministic merge, and no worker file
  writes.

## Required implementation

1. Add deterministic frozen scenario construction and identity validation for
   both namespaces, preserving common-path clustering and nested population
   streams.
2. Add the missing parallel Fit adapter. Each iteration must execute one
   current-rule forward baseline plus all 256 candidate deviations, retain all
   256 `SolverRuleRow` values, paired common-path cluster deltas, environment
   fingerprint, call interval, selected/runner-up and precision disposition.
   Preserve fixed point, cycle, `not_converged_32`, statistics insufficiency,
   budget exhaustion, and execution failure as distinct outcomes.
3. Freeze only a protocol-admissible Fit candidate by immutable rule name and
   fingerprint. If the three starts do not yield a protocol-admissible
   candidate, retain the negative Fit result and do not run Validation.
4. Add independent Validation orchestration using the same complete rerun
   semantics. For each of eight Experts execute one baseline and all 256
   deviations, then retain deterministic deviation rows and sufficient cluster
   data for regret inference.
5. Implement the frozen simultaneous one-sided 95% bound over common-fault
   clusters. Recompute the maximum-selected statistic under the declared
   cluster resampling method; report explicit `pass`, `fail`,
   `statistics_insufficient`, or `execution_failed`. Do not relabel a
   restricted result as unrestricted Nash or MFG.
6. Add complete schema/completeness validation and a fresh-directory,
   transactional artifact commit. A failed or partial run must not leave a
   directory that looks complete; an existing run must never be overwritten.

## Forbidden changes

No new physical mechanism, altered event ordering, approximation, frozen
pool/queue trajectory, replayed action list, future observation, statistical
sample extension, new strategy search, tariff/market model, N>8 campaign,
MFG/HJB-FPK run, training, qualification, holdout, or third-party dependency.

## Acceptance criteria

- Failure-first tests cover frozen scenario identity, call arithmetic, all
  three starts, complete iteration rows, candidate freeze, independent
  fingerprints, all-Expert/all-256 validation, simultaneous bound dispositions,
  deterministic ordering, no retry, and transactional no-overwrite writes.
- A small deterministic fixture proves parallel Fit/Validation rows are
  identical to the reference evaluator and that parent merge order is
  independent of worker completion order.
- The formal runner has a dry preflight that proves the `921088` call ceiling
  before dispatch and rejects incomplete orchestration.
- Only after tests pass may the frozen real campaign be launched. Its output
  must record actual calls, elapsed time, fingerprints, every declared row,
  stop reasons, bound values, and the restricted claim wording.

## Progress log

### Update: 2026-09-06 — Claim missing campaign orchestration

Status: partial

#### Goal

Complete the missing Stage 4 Fit/Validation orchestration and simultaneous
restricted-regret inference without changing the frozen finite game, then make
the campaign launchable under the accepted timing gate.

#### Changed

- Confirmed ticket 08 is resolved and claimed this ticket as the sole active
  ticket.
- Changed ticket 06 to `Status: blocked` with `Blocked by: 09`; no Fit or
  Validation call was started.
- Accepted ADR-0016 by changing only its status to the user-confirmed value;
  its decision text remains unchanged.
- Identified the missing pieces: the existing `run_deviation_batch` is only a
  low-level exact tagged executor; `game_solver.py` has no parallel campaign
  adapter, no complete frozen scenario campaign entry point, no implemented
  simultaneous bound, and no Stage 4 transactional schema/completeness
  pipeline.

#### Verification

- Focused baseline:
  `.venv\Scripts\python.exe -m unittest tests.test_game_solver tests.test_campaign_execution tests.test_campaign_cli -v`
  — 19 tests passed.
- Full baseline:
  `.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  — 458 tests passed.
- Configuration check:
  `.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  — exit 0, `status: ok`, with the pre-existing single-domain capacity
  warning.
- No formal Stage 4 scheduler call, candidate selection, Validation, or
  campaign artifact was produced.

#### Artifacts

None. This ticket update only records the missing orchestration boundary; no
Fit/Validation/candidate artifact was created or overwritten.

#### Decisions and risks

- ADR-0016 is accepted, but acceptance does not waive its frozen protocol or
  authorize an incomplete runner.
- The 5.89-hour timing gate is an accepted execution gate, not a statistical
  result. No campaign claim exists yet.
- If the simultaneous-bound or transaction contract cannot be implemented
  without inventing a statistical meaning, retain ticket 06 blocked and
  report the exact conflict rather than changing the protocol.

#### Next

Write failing orchestration tests, implement the smallest exact adapters and
schema checks, run all required verification, and only then restore ticket 06
to `open` before claiming it for the formal campaign.

### Update: 2026-09-06 — Orchestration and statistical gate completed

Status: completed

#### Goal

Implement the missing exact Stage 4 campaign orchestration and simultaneous
restricted-regret inference so the already accepted timing gate can be used
without changing the finite game or its frozen sample protocol.

#### Changed

- Added `src/mfg_hedge/stage4_campaign.py` with frozen nested-scenario
  construction, exact call preflight, parallel Fit adapter, immutable
  candidate selection, independent all-Expert/all-256 Validation, cluster
  bootstrap max-t simultaneous one-sided 95% bound, completeness checks, and
  transactional fresh-run artifact assembly.
- Extended batch rows with per-common-path cluster objectives so paired
  candidate-minus-incumbent deltas are computed after worker aggregation.
- Allowed finite signed paired cluster deltas in `game_solver.py`; standard
  errors remain computed from finite differences without changing ranking.
- Kept the new orchestration internal; package version remains `0.23.0`.
- Resolved ticket 09 and restored ticket 06 to the sole claimed campaign
  ticket with no blocker.

#### Verification

- Real Red:
  `.venv\Scripts\python.exe -m unittest tests.test_stage4_campaign -v`
  initially failed during collection with
  `ModuleNotFoundError: No module named 'mfg_hedge.stage4_campaign'`.
- Focused green:
  `.venv\Scripts\python.exe -m unittest tests.test_stage4_campaign tests.test_game_solver tests.test_campaign_execution tests.test_shared_backup_performance -v`
  — 30 tests passed.
- Full green:
  `.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  — 463 tests passed in 32.735 seconds.
- Config check:
  `.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  — exit 0, `status: ok`, with the existing single-domain capacity warning.
- Deterministic fixture checks passed for 256 retained Fit rows, exact
  514-call two-Expert Validation, disjoint trace/fault fingerprints, and
  deterministic 128-replicate simultaneous-bound output.
- No formal Fit/Validation call or campaign artifact was created in this
  implementation ticket.

#### Artifacts

- `src/mfg_hedge/stage4_campaign.py`
- `tests/test_stage4_campaign.py`
- No Fit/Validation/candidate artifact.

#### Decisions and risks

- The exact tagged worker path still evolves all Experts and shared physics;
  the parent merges compact rows deterministically and owns call accounting.
- The simultaneous inference method is the declared deterministic common-path
  cluster bootstrap max-t one-sided 95% numerical bound with seed `20260906`
  and `4096` resamples; it is not a finite-sample theorem.
- No unrestricted Nash or MFG claim is produced by this ticket.

#### Next

Use the launch disposition on ticket 06 to run the already frozen Fit and, only
if Fit yields a protocol-admissible candidate, independent Validation.

### Update: 2026-09-06 — Frozen Fit launch found incomplete cluster cohort

Status: blocked

#### Goal

Execute the authorized frozen Stage 4 campaign and retain a valid result only
if every fixed common-path cluster can support the declared scorer and paired
precision calculation.

#### Changed

- Started one fresh formal launch with the frozen Fit inputs, Windows
  spawn-safe 8-worker batcher, and no retry.
- The first Fit baseline and all 256 deviation tasks for the first iteration
  were dispatched. The worker batch failed closed because common fault cluster
  index 13 has no empirical `F/U` cohort for representative Expert 0 after
  pooling its two frozen population traces.
- Preserved this missing cluster objective as an explicit null/infinite-SE
  precision condition in the in-memory boundary; no zero fill, path omission,
  sample extension, or changed scorer was introduced.
- Corrected the campaign writer so an incomplete/execution-failed Fit cannot
  create a half-complete artifact directory. The failed launch directory
  `artifacts/stage4-pi256-fit-validation-20260906-r1` was removed after
  verifying its exact path; it was not a valid scientific artifact.
- Changed ticket 06 to `Status: blocked` with `Blocked by: 09`; ticket 09 is
  the sole claimed implementation ticket while this boundary is unresolved.

#### Verification

- Formal launch command:
  `.venv\\Scripts\\python.exe .\\.scratch\\shared-backup-game\\run_stage4_campaign_20260906.py`
  — completed with `status=execution_failed`, `fit_calls=8224`,
  `validation_calls=0`, and no retry.
- Direct frozen-trace diagnostic over all 16 Fit common paths and both
  populations found exactly one incomplete cluster: sorted cluster index 13,
  missing `('F', 'U')`; the other 15 clusters were complete.
- Focused regression after the fail-closed null handling and artifact guard:
  `.venv\\Scripts\\python.exe -m unittest tests.test_stage4_campaign tests.test_game_solver tests.test_campaign_execution -v`
  — 23 tests passed.
- No complete Fit, candidate freeze, Validation, simultaneous bound, or
  scientific Stage 4 artifact exists.

#### Artifacts

None. The exact failed launch directory was removed and no replacement
artifact was written.

#### Decisions and risks

- The frozen spec says missing empirical pooled cohorts are incomplete, while
  ADR-0016 requires a paired scalar objective for every fixed common-path
  cluster. The observed frozen Fit sample therefore cannot supply the
  declared cluster standard error for all 16 paths without either imputing the
  missing cohort, dropping a fixed cluster, or changing the scorer/statistical
  meaning. None of those changes is authorized.
- Fit/Validation must not be rerun or continued until this conflict is
  resolved by an explicit protocol decision. There is no Nash, restricted
  candidate, or MFG result.

#### Next

Resolve the frozen scorer-versus-cluster-completeness conflict in a new
explicit ADR/ticket decision before any further campaign execution.

### Update: 2026-09-06 — Final fail-closed verification

Status: blocked

#### Goal

Close the launch audit without converting a failed partial Fit into a formal
scientific artifact or changing the frozen statistical meaning.

#### Changed

- Kept ticket 09 blocked and ticket 06 blocked by 09; no ticket is claimed
  while the protocol conflict awaits an explicit decision.
- Kept the null cluster objective/infinite paired-SE representation and the
  no-artifact-on-execution-failure guard.
- Removed the temporary cluster diagnostic script; the guarded launch helper
  remains only under `.scratch/shared-backup-game/`.

#### Verification

- Focused suites:
  `.venv\\Scripts\\python.exe -m unittest tests.test_shared_backup_performance tests.test_campaign_execution tests.test_shared_backup tests.test_game_workload tests.test_expert_game tests.test_game_deviations tests.test_game_solver tests.test_stage4_campaign -v`
  — 79 tests passed.
- Full suite:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`
  — 464 tests passed in 50.781 seconds.
- Configuration check:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`
  — exit 0, `status: ok`, with the existing single-domain capacity warning.
- Process cleanup check found no remaining Python worker; the exact failed
  run directory does not exist.

#### Artifacts

None. No Fit/Validation/candidate artifact was retained or overwritten.

#### Decisions and risks

- The first formal launch is a real Red at the orchestration/statistical
  boundary, not a negative equilibrium result: it stopped after `8224`
  reserved Fit calls (`32` baseline plus `8192` deviations), with `0`
  Validation calls and no retry.
- The observed missing `F/U` cohort in fixed Fit cluster 13 cannot be
  imputed, dropped, or adaptively repaired under the accepted protocol.
- The complete historical and Stage 4 regression suites remain green, but this
  does not authorize a campaign rerun or any Nash/MFG claim.

#### Next

Obtain an explicit ADR/ticket decision for incomplete per-cluster cohorts,
then rerun the frozen campaign only if that decision preserves the declared
sample and scorer semantics.

### Update: 2026-09-06 — Statistical conflict resolved by ticket 10

Status: completed

#### Goal

Record the explicit ADR-0018 resolution of the campaign orchestration's
standalone-cluster scoring conflict.

#### Changed

- Replaced the invalid cluster-scalar boundary through ticket 10 and retained
  the already completed orchestration, max-t and transactional write layers.
- No formal Fit or Validation execution was started.

#### Verification

- Ticket 10 records 467 passing tests, the passing config gate, and two frozen
  32-scenario diagnostics with 16/16 finite delete-one pooled objectives.

#### Artifacts

- See ticket 10 and ADR-0018; no experiment artifact.

#### Decisions and risks

- The orchestration ticket is resolved; experiment execution remains a
  separate explicit claim of ticket 06.

#### Next

Re-claim ticket 06 when ready to run the formal campaign.

## Answer

Resolved. The orchestration exists and its pooled nonlinear inference boundary
is now corrected by ADR-0018; the formal campaign has not been rerun.
