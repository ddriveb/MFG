# Run frozen finite Pi256 rule-game campaign

Type: experiment
Status: resolved
Blocked by: none

## Scope

Run the already frozen Stage 4 finite Shared-Backup campaign only. This
ticket may generate fresh immutable fit/validation campaign artifacts under a
new artifact run directory, but it may not change the physics engine,
historical configurations, the frozen sample counts, the candidate rule after
fit, or the claim boundary.

## Frozen execution contract

- Fit: namespace `shared-backup-game:v1:stage4-fit`, macro seed `20260905`,
  16 common fault paths, population IDs `0` and `1`, 32 scenario traces,
  starts `NNNN`, `NSSN`, `XXXX`, at most 32 rounds.
- Validation: namespace `shared-backup-game:v1:stage4-validation`, macro seed
  `20260906`, 32 new common fault paths, population IDs `0` and `1`, 64
  scenario traces, disjoint trace and fault fingerprints.
- System: `N=8`, `c_B=0.5`, degraded slowdown `2.0`, hedge delay `1.5`, zero
  tariff, the canonical 256-rule bank.
- Fit candidates are frozen before validation. Validation evaluates all eight
  Experts against all 256 deviations and cannot tune the candidate.
- Preserve fixed points, rule cycles, `not_converged_32`, budget exhaustion,
  and `statistics_insufficient`; until the simultaneous one-sided 95% bound
  is complete, no Nash or MFG claim is allowed.
- Call budget: the predeclared upper bound is 789,504 fit calls plus 131,584
  validation calls, 921,088 total, below the one-million hard limit. No
  adaptive sample extension is allowed.

## Acceptance criteria

1. A real-load timing preflight is completed before the campaign and reports
   runtime and call-count estimates without changing the frozen protocol.
2. Fit retains every start, every complete iteration row, environment
   fingerprint, selected/runner-up pair, paired cluster precision result, and
   stop reason.
3. Any fit candidate is frozen by value and validation uses independent data
   with all Expert/all-256 complete deviations.
4. Simultaneous restricted-regret uncertainty is computed from validation
   cluster data with an explicit pass/fail/insufficient disposition; results
   remain restricted-game candidates unless the declared finite-system gate
   is met.
5. Existing tests, historical behavior, and no-overwrite artifact guards
   remain green. No MFG forward model is started.

## Progress log

### Update: 2026-09-05 — Claim frozen campaign execution

Status: partial

#### Goal

Run the frozen Stage 4 fit and independent validation campaign after a real
load timing preflight, preserving all negative and statistically insufficient
outcomes.

#### Changed

- Created and claimed the sole campaign ticket after resolved solver ticket
  05. No source, configuration, or artifact was changed in this claim step.

#### Verification

- Not run yet; trace generation, timing preflight, campaign execution, and
  simultaneous bound remain outstanding.

#### Artifacts

- This ticket only; no generated campaign artifact.

#### Decisions and risks

- The Stage 4 protocol is inherited unchanged from ADR-0016 and ticket 05.
- No fit or validation execution may begin until real-load timing preflight
  is recorded. A timing result cannot authorize changing frozen samples or
  call budget.

#### Next

Inspect the frozen nested-trace generation path, run real-load timing
preflight, then execute fit and validation within the declared budget.

### Update: 2026-09-05 — Real-load timing gate blocks campaign launch

Status: partial

#### Goal

Complete the required real-load timing preflight before launching the frozen
Stage 4 fit and validation campaign.

#### Changed

- Built one in-memory frozen-protocol probe for fit namespace
  `shared-backup-game:v1:stage4-fit`, macro seed `20260905`, common path `0`,
  population `0`, and `N=8`; no source, configuration, or historical
  artifact was modified.
- Measured the three frozen start profiles against that same real nested
  workload: `NNNN`, `NSSN`, and `XXXX`.

#### Verification

- Real-load timing command used `build_common_fault_path`,
  `build_nested_population`, `DeviationScenario`, and
  `measure_scheduler_timing` with `c_b=0.5`, slowdown `2.0`, and hedge delay
  `1.5`.
- The probe contained `1,276` Tokens. Timings were `NNNN=0.48979639998287894`,
  `NSSN=0.4971741000190377`, and `XXXX=0.5149612000095658` seconds per
  scheduler call.
- Using the frozen `921,088` call upper bound, the NNNN timing estimate was
  `458,808.5781007856` seconds, approximately `127.45` hours.
- Host diagnostic reported `12` logical processors. Even ideal linear use of
  all twelve would still be roughly `10.6` hours before orchestration and
  serialization overhead.
- No fit scheduler call, candidate freeze, validation call, or simultaneous
  restricted-regret calculation has been run.

#### Artifacts

None. The probe was in memory only; no campaign directory or output file was
created.

#### Decisions and risks

- The campaign is intentionally not launched because the real-load timing
  gate shows a multi-day single-process run, contradicting the earlier smoke
  extrapolation. The frozen sample counts, common paths, precision rule, and
  one-million call boundary remain unchanged.
- Running a partial campaign would not produce a valid frozen fit/validation
  result. A parallel execution layer or semantics-preserving scheduler
  performance optimization would be a new implementation decision and must
  preserve deterministic ordering, call accounting, and no-overwrite output.
- No Nash candidate, finite-system validation result, MFG forward model,
  experiment conclusion, qualification, or holdout result exists.

#### Next

Choose and freeze a performance execution plan (parallel campaign runner or
semantics-preserving scheduler optimization) before any fit or validation
campaign call is authorized.

### Update: 2026-09-05 — Profile real scheduler hotspots

Status: partial

#### Goal

Identify the dominant cost in one real N=8 scheduler call before implementing
the frozen performance plan.

#### Changed

- Added proposed [ADR-0017](../../../docs/adr/0017-stage4-campaign-performance.md)
  freezing the semantics-preserving optimization boundary, exact tagged mode,
  `(deviating_expert, candidate_rule)` worker granularity, parent-owned call
  settlement, deterministic merge, 8-worker/512 MiB worker/4 GiB parent
  execution limits, and the pre-campaign worst-case timing gate.
- Ran an in-memory cProfile probe only; no source, configuration, or campaign
  artifact was changed.

#### Verification

- cProfile `Profile.runcall` on one frozen-protocol N=8 fit trace with 1,276
  Tokens showed the scheduler event loop dominates: `_handle_arrival` and
  `_observation` accounted for about `0.45s`, and `_audit_queues` about
  `0.24s` under profiling.
- The same profile showed `score_population` at about `0.016s` and
  `shared_trace_fingerprint` at about `0.006s`; scorer and hashing are not
  the primary 0.5-second hotspot.
- The profile captured `1,306` attempt rows for the probe. It confirmed that
  repeated per-arrival observation scans and queue-audit construction are the
  first optimization targets; no policy action, speed, queue, Replay, or
  completion state may be cached across calls.

#### Artifacts

None. The cProfile report and probe trace were held in memory and the
temporary profiling file was removed.

#### Decisions and risks

- Implement optimization in this order: first reduce repeated observation
  and audit/input overhead with reference-vs-optimized field comparisons;
  then add `PreparedTrace` and exact tagged output; only after those pass add
  the frozen 8-worker process batcher.
- The frozen sample counts, rules, common paths, populations, starts, maximum
  rounds, call budget, and claim boundary are unchanged. The campaign remains
  paused because the current worst-case single-process estimate is about
  `127.45` hours.

#### Next

Write failing reference-equivalence tests for prepared inputs and tagged
scoring output, then implement the first single-scheduler optimization.

### Update: 2026-09-06 — Block campaign pending performance implementation

Status: blocked

#### Goal

Record that the real-load timing gate blocks campaign launch and transfer the
performance-layer implementation to the new bounded ticket 07 without
starting Fit, Validation, or a formal campaign.

#### Changed

- Changed this campaign ticket to `Status: blocked` with `Blocked by: 07`.
- Preserved all earlier timing and no-campaign history.
- Kept the frozen sample split, rule bank, common paths, populations,
  precision rule, call budget, and claim boundary unchanged.

#### Verification

- The recorded real-load probe remains: 1,276 Tokens; NNNN `0.4897964s`, NSSN
  `0.4971741s`, XXXX `0.5149612s` per reference scheduler call.
- The frozen 921,088-call upper bound remains approximately 127.45 hours in
  the single-process implementation; no campaign scheduler call was run.

#### Artifacts

None. No Fit/Validation/campaign output was created or overwritten.

#### Decisions and risks

- ADR-0017 is now accepted by the user, but its implementation authorization
  is limited to performance-layer equivalence work in ticket 07.
- This ticket is not resolved until ticket 07 passes its performance and
  equivalence gate and is deliberately not claimed while blocked.

#### Next

Complete and verify ticket 07, then return this ticket to `open` without
automatically claiming it or launching the campaign.

### Update: 2026-09-06 — Parallel scaling diagnostic blocks campaign

#### Goal

Record that the Stage 4 campaign timing gate remains blocked and hand the
parallel-scaling diagnosis to ticket 08 without changing the frozen campaign
protocol.

#### Changed

- Changed the ticket header to `Status: blocked` with `Blocked by: 08`.
- Appended the new dependency without rewriting any prior Progress log entry.
- Preserved the frozen 921,088-call budget, fit/validation samples, rules,
  populations, starts, iteration cap, and statistical precision rule.

#### Verification

- The latest valid 8-worker real-load batch measured 4.3153348986 calls/s,
  implying approximately 59.29 hours for the declared upper bound.
- The observed parallel expansion is insufficient for launch; no Fit,
  Validation, candidate, Nash, MFG, or formal campaign result exists.
- No campaign scheduler call was started as part of this update.

#### Artifacts

None. No campaign artifact was created or overwritten.

#### Decisions and risks

- The blocker is execution-layer parallel scaling only; no physics, scoring,
  CRN, observation boundary, sample protocol, or call-budget change is
  authorized.
- Ticket 08 is the sole claimed ticket. Ticket 06 must remain unclaimed until
  a later launch disposition after ticket 08.

#### Next

Diagnose 1/2/4/8-worker stable throughput, startup versus steady state,
PreparedTrace reuse, task/result serialization, worker aggregation, CPU time,
parent wait, and RSS in ticket 08. Do not launch the campaign before that
diagnosis and a separate disposition.

### Update: 2026-09-06 — Launch disposition after accepted timing gate

Status: partial

#### Goal

Authorize execution of the already frozen Stage 4 Fit/Validation campaign
after ticket 08 and the missing orchestration ticket 09 are complete.

#### Changed

- Changed the ticket header to `Status: claimed` with `Blocked by: none`.
- Recorded the user's accepted timing gate: 8-worker measured throughput
  `43.414 calls/s`; the mechanical `921088`-call upper bound is approximately
  `5.89 hours`.
- Preserved every frozen physics, sample, namespace, population, start,
  iteration, precision, and call-budget value.

#### Verification

- Ticket 08 is resolved with the accepted real-load scaling diagnosis.
- Ticket 09 is resolved with the exact Fit/Validation orchestration,
  simultaneous-bound implementation, deterministic rows, and transactional
  no-overwrite writer.
- Focused and full regression suites plus config check passed before launch.
- No campaign call has been started in this status update; the next command is
  the single formal campaign launch.

#### Artifacts

None created by this launch disposition. Any campaign output must use a fresh
unique directory under `artifacts/` and must never overwrite an older run.

#### Decisions and risks

- Only ticket 06 is claimed. The accepted 5.89-hour figure is a timing gate,
  not a scientific result or a guarantee of candidate existence.
- A failed, partial, statistically insufficient, or no-candidate Fit remains
  a valid negative result; no adaptive samples, retries, rule changes, or
  forced Validation are allowed.
- Any successful result may use only the restricted `Pi256` claim wording;
  it is not an unrestricted Nash or formal MFG proof.

#### Next

Run the formal frozen Stage 4 Fit, freeze any admissible candidate by value,
then run independent Validation and its simultaneous restricted-regret bound.

### Update: 2026-09-06 — Formal launch blocked by frozen cluster completeness

Status: blocked

#### Goal

Run the user-authorized frozen Fit/Validation campaign under the accepted
timing gate, preserving the fixed samples, complete reruns, and statistical
claim boundary.

#### Changed

- Handed the campaign back to ticket 09 after the first formal Fit launch
  failed closed on an incomplete frozen common-path cluster scorer input.
- No campaign protocol, sample count, call budget, action rule, scorer, or
  physical semantics was changed to make the result run.

#### Verification

- First launch completed one Fit iteration's baseline plus 256 candidate tasks
  across 32 scenarios: `8224` calls reserved, `0` Validation calls, no retry.
- The failure is localized to Fit common-path cluster index 13, where pooled
  population traces lack representative Expert 0's `F/U` cohort. This is
  governed by the existing missing-cohort-incomplete rule.
- No candidate was selected or frozen; Validation and simultaneous regret
  bound were not started.
- The failed half-complete directory was removed; no formal campaign
  artifact remains.

#### Artifacts

None. No Fit/Validation/candidate artifact was retained.

#### Decisions and risks

- Ticket 06 is blocked by ticket 09 pending an explicit decision on how the
  frozen scorer handles an incomplete per-cluster empirical cohort. Filling,
  dropping, or adaptively extending the cluster would violate the frozen
  protocol.
- Current status remains: no restricted Pi256 candidate, no Nash claim, and
  no MFG result.

#### Next

Resolve ticket 09's explicit protocol conflict, then re-claim ticket 06 before
any new campaign run.

### Update: 2026-09-06 — Final campaign disposition

Status: blocked

#### Goal

Record the final disposition of the authorized Stage 4 launch while keeping
the frozen Fit/Validation protocol and claim boundary intact.

#### Changed

- Kept ticket 06 blocked by ticket 09; no further Fit or Validation phase was
  launched after the fail-closed first iteration.
- Retained the accepted frozen parameters, `921088` planned calls, and the
  no-retry/no-adaptive-sample rules.

#### Verification

- Real launch result: `execution_failed`, `8224` reserved Fit calls, `0`
  Validation calls, no candidate freeze, and no simultaneous bound.
- Focused Stage 4/physics/scorer/deviation/solver/performance suites: `79`
  tests passed.
- Full suite: `464` tests passed in `50.781s`.
- Config check exited `0` with `status: ok` and the pre-existing single-domain
  capacity warning.
- No Python workers remain and no `stage4-pi256-*` artifact directory exists.

#### Artifacts

None. The failed partial directory was removed after exact-path verification;
there is no Fit, candidate, Validation, or experiment artifact.

#### Decisions and risks

- This is neither a no-candidate Fit nor a restricted-game candidate: the
  launch failed before a complete first iteration could be retained.
- The blocker is the concrete frozen-data conflict between missing-cohort
  incompleteness and the requirement for a paired scalar on every fixed
  common-path cluster. Resolving it requires an explicit protocol decision.
- There is no Nash claim, no MFG result, and no campaign conclusion.

#### Next

Resolve ticket 09's protocol conflict and explicitly re-claim ticket 06 before
any new formal campaign execution.

### Update: 2026-09-06 — Re-claim after ADR-0018 correction

Status: claimed

#### Goal

Restart the frozen Stage 4 Fit only after the pooled nonlinear objective and
common-path delete-one jackknife inference were implemented and verified.

#### Changed

- Re-claimed ticket 06 as the only active ticket; tickets 09 and 10 are
  resolved and no other ticket is claimed.
- Accepted the ADR-0018 execution boundary already recorded in the repository:
  pooled full objective, exact delete-one common-path recomputations,
  paired jackknife standard errors, and validation pseudo-values.
- Selected the fresh launch run ID
  `stage4-pi256-fit-validation-20260906-r2`; the prior failed `r1` directory
  will not be reused or overwritten.
- Preserved the failed `r1` accounting of `8224` calls. The new run retains
  the unchanged `921088` campaign ceiling; cumulative declared accounting is
  `929312`, below the one-million hard limit.

#### Verification

- Focused preflight:
  `.venv\\Scripts\\python.exe -m unittest tests.test_stage4_campaign tests.test_campaign_execution tests.test_game_solver -v`
  — `26` tests passed in `27.102s`.
- Configuration check:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`
  — exit 0, `status: ok`, with the pre-existing single-domain capacity
  warning.
- ADR-0018 fixture verification, including the former cluster-13 missing
  `F/U` case, produces finite delete-one objective `36.40114007370042` and
  valid values for all 16 Fit common paths.
- No new formal Fit or Validation call has started in this update.

#### Artifacts

None. No campaign artifact was created or overwritten during the re-claim.

#### Decisions and risks

- The new inference is an accepted protocol correction, not a sample,
  physics, CRN, rule-bank, or budget change. Jackknife/max-t remains a
  statistical approximation and cannot establish unrestricted Nash or MFG.
- The formal run may produce fixed point, cycle, non-convergence, statistical
  insufficiency, or execution failure; all rows and stop reasons must remain
  auditable.

#### Next

Start the fresh r2 formal campaign with the frozen 8-worker runner; enter
Validation only after Fit produces a complete protocol-admissible candidate.

### Update: 2026-09-06 — Frozen Fit completed with insufficient precision

Status: resolved

#### Goal

Complete the frozen Stage 4 Fit after ADR-0018, preserve every finite-game
and statistical audit row, and enter Validation only if a common
precision-qualified candidate is produced.

#### Changed

- Completed fresh run
  `stage4-pi256-fit-validation-20260906-r2` without reusing or overwriting
  the failed r1 directory.
- Ran the three frozen starts `NNNN`, `NSSN`, and `XXXX`; each reached the
  same `NSNS` pure fixed point after two iterations.
- Retained all six iteration records and all `6 * 256 = 1536` canonical rule
  rows. Point objectives use the pooled scorer and precision rows use the
  accepted delete-one-common-path jackknife values.
- Because every start has `statistics_insufficient` precision, no candidate
  was frozen and Validation was correctly not started.
- Resolved ticket 06 after the complete negative Fit result; no ticket is
  currently claimed.

#### Verification

- Fit artifact status: `fit_no_candidate`; candidate: `None`; claim boundary:
  `no restricted Pi256 approximate equilibrium candidate`.
- Exact Fit calls: `49344` (`6` iterations × `8224` calls); Validation calls:
  `0`; total in r2: `49344`; elapsed time: `1311.317076700012` seconds.
- Previous failed r1 calls remain disclosed: `8224`. Actual cumulative calls
  across both attempts: `57568`; worst-case cumulative ceiling remains
  `929312` (`8224 + 921088`), below the one-million hard limit.
- All six iterations contain exactly 256 canonical rows and all rows are
  complete. The artifact manifest contains 32 Fit trace fingerprints and 64
  independent Validation fingerprints; no Validation result was written.
- Precision target was `0.18049367342477127`. The per-start records were:
  `NNNN` round 0 selected `NSNS` over `NSND`, gap
  `0.11384705192135769`, selected jackknife SE
  `0.08027056403962067`, max candidate SE
  `1.9146924556915614`; round 1 selected `NSNS` over `NSSS`, gap
  `0.10598514223075028`, selected jackknife SE
  `1.6457941083202814`, max candidate SE
  `1.8983582165586368`.
  `NSSN` round 0 selected `NSNS` over `NSND`, gap
  `0.10099371541862112`, selected jackknife SE
  `0.04934394284079787`, max candidate SE
  `1.8822819020144004`; round 1 matched the NNNN round-1 values.
  `XXXX` round 0 selected `NSNS` over `NSND`, gap
  `0.1156101345125009`, selected jackknife SE
  `0.06678338654865021`, max candidate SE
  `1.7055730854631128`; round 1 matched the NNNN round-1 values.
  All six precision statuses were `statistics_insufficient`.
- Focused regression:
  `.venv\\Scripts\\python.exe -m unittest tests.test_stage4_campaign tests.test_campaign_execution tests.test_game_solver -v`
  — `26/26` passed.
- Full regression:
  `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`
  — `467/467` passed in `51.303s`.
- Configuration check:
  `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`
  — exit 0, `status: ok`, with the existing single-domain capacity warning.
- No Python workers remain; artifact files are exactly `fit.json`,
  `manifest.json`, and `result.json`.

#### Artifacts

- [stage4-pi256-fit-validation-20260906-r2](D:/project/mfg_hedge_v1/artifacts/stage4-pi256-fit-validation-20260906-r2)
  — complete negative Fit artifact only.
- No candidate, Validation, simultaneous-bound, Nash, or MFG artifact.

#### Decisions and risks

- `NSNS` is a finite best-response fixed point of the retained Fit point
  estimates, but it is not a restricted-game candidate because the frozen
  jackknife precision gate failed. It must not be called Nash or MFG.
- No samples, common paths, populations, rules, physics, CRN, iteration cap,
  action policy, or call budget were changed. No Validation tuning or forced
  evaluation was performed.
- The jackknife/max-t inference remains a statistical approximation and does
  not establish an equilibrium theorem. Package version remains `0.24.0` as
  required by ADR-0018.

#### Next

Keep the complete negative Fit result as the terminal Stage 4 campaign
disposition; any new candidate search or MFG work requires a separately
authorized protocol decision.

## Answer

Not resolved. The failed first launch remains part of the audit history;
ADR-0018 and ticket 10 have corrected its statistical blocker, but no second
formal Fit or Validation run has started. Ticket 06 is open and unclaimed.
