# Diagnose Stage 4 parallel scaling

Type: implementation
Status: resolved
Blocked by: none

## Scope

Diagnose and, only where reference-equivalence is preserved, correct the
parallel execution bottleneck behind the Stage 4 campaign timing gate. This
ticket is a performance-execution ticket only. It must not start Fit,
Validation, or any formal campaign, and it must not produce candidate, Nash,
MFG, qualification, holdout, or experiment artifacts.

The unit of work remains `(deviating expert, canonical rule)`; one worker must
process that pair's complete frozen scenario list in canonical order. The
worker must aggregate tagged scorer sufficient data and return only a compact
rule row plus call and resource accounting. The parent alone owns call
reservation/settlement, deterministic merge, and failure status.

## Frozen constraints

- Preserve the Shared-Backup physics, event ordering, processor-sharing
  semantics, Replay, timer, failure, winner, cancellation, drain, action
  projection, causal observation, CRN, scorer, deviation rerun, and solver
  semantics.
- Preserve N=8, `c_B=0.5`, the canonical 256-rule bank, all frozen fit and
  validation common paths/populations/namespaces/seeds, three starts, 32-round
  limit, paired-cluster precision rule, and the declared 921,088-call upper
  bound.
- Preserve exact tagged evaluation: all Experts and all shared physical work
  still participate; only non-tagged materialization is reduced.
- Do not reduce rules, Experts, scenarios, populations, iterations, or calls;
  do not replay actions, freeze `K`/speed/completion, discretize time, remove
  participants, disable invariants, or add dependencies.
- Windows process execution must use spawn-safe top-level workers, at most 8
  workers and 8 in-flight tasks, no automatic retry, deterministic parent
  merge, and no file writes from workers.

## Required diagnosis and implementation checks

1. Measure stable 1/2/4/8-worker throughput on the same frozen real-load
   probe, and separate process startup, steady-state scheduler execution,
   serialization, parent wait, and merge time.
2. Record wall time and CPU process time, serialization byte counts, worker
   and parent RSS, worker count, start method, task count, and failure reason.
3. Prove `PreparedTrace` is constructed/validated/fingerprinted once per worker
   initialization rather than once per task or scheduler call. A task must not
   transfer the trace repeatedly, and a result must not return per-Token full
   objects.
4. Prove tagged score aggregation happens in the worker and that the parent
   receives only compact sufficient data/rule rows.
5. Add deterministic tests for scaling instrumentation, task grain, prepared
   trace reuse, no retry, exact reservation/settlement, stable merge order,
   duplicate/missing results, and worker shutdown. Tests must not depend on
   wall-clock completion order.
6. Compare reference and optimized paths field-by-field for N/D/S/X, Replay
   burst, failure/completion/timer ties, simultaneous completion tie-break,
   queued and running losers, failure drain, survivor acceleration, and budget
   suppression. Any scientific mismatch keeps the reference path and records
   the failed optimization instead of widening tolerances or changing physics.

## Acceptance criteria

- A reproducible 1/2/4/8-worker scaling table exists for the frozen probe,
  including startup versus steady-state and all required accounting fields.
- The coarse task contract, worker-local PreparedTrace reuse, compact result
  path, and parent deterministic merge are demonstrated by tests.
- Reference, optimized, prepared, and tagged scientific rows remain exactly
  equivalent, except only explicitly justified `abs_tol <= 1e-12` arithmetic
  differences with unchanged event ordering, actions, winners, scores, and
  stop reasons.
- Any failure leaves the batch `partial`/`failed`, does not retry, and cannot
  enter candidate selection. Call accounting never exceeds the frozen budget.
- The final report is diagnostic only. It must not claim that the campaign
  timing gate passed merely because throughput improved.

## Progress log

### Update: 2026-09-06 — Claim parallel-scaling diagnosis

#### Goal

Diagnose the failed Stage 4 campaign timing gate and apply only
reference-equivalent execution-layer corrections. No Fit, Validation, or
formal campaign is authorized in this ticket.

#### Changed

- Created and claimed this ticket as the sole active implementation ticket.
- Fixed the campaign dependency at ticket 06: it is blocked by ticket 08 and
  is not claimed.
- Froze the diagnosis around process startup, steady-state execution, trace
  transfer, worker aggregation, parent wait, serialization, and CPU/RSS
  accounting; no statistical or sample-protocol change is in scope.

#### Verification

- Existing real-load evidence records 4.3153348986 calls/s for the valid
  8-worker batch and an approximately 59.29-hour worst-case estimate for
  921,088 calls. This is a timing-gate observation, not a campaign result.
- Existing performance/equivalence suites remain the baseline for this ticket;
  no campaign scheduler call was started in this claim.

#### Artifacts

None. No Fit/Validation/candidate artifact was created or overwritten.

#### Decisions and risks

- The finite game, scoring protocol, CRN, causal observation boundary, and
  frozen call budget remain unchanged.
- The current parallel efficiency is insufficient to authorize campaign launch;
  the target is diagnostic, not a new acceptance threshold.
- If an optimization cannot pass the reference harness, the reference
  implementation remains authoritative and the optimization is recorded as
  failed.

#### Next

Implement the instrumentation and failure-first tests, then run the stable
1/2/4/8-worker real-load probe. Do not claim ticket 06, launch campaign, or
generate campaign artifacts until a separate disposition is made.

### Update: 2026-09-06 — Locate scaling loss with the frozen task shape

Status: partial

#### Goal

Separate Windows process startup, steady scheduler work, serialization and
parent overhead on the actual 32-scenario Fit task shape, and identify the
remaining CPU hotspot without launching the campaign.

#### Changed

- Added reproducible diagnostic scripts
  `.scratch/shared-backup-game/parallel_scaling_probe.py` and
  `.scratch/shared-backup-game/optimized_worker_profile.py`.
- Added the read-only diagnosis report
  `.scratch/shared-backup-game/stage4-parallel-scaling-diagnosis-20260906.md`.
- No production source, configuration, frozen protocol, sample, call budget or
  artifact was changed.

#### Verification

- The 1/2/4/8-worker probe used 32 frozen Fit scenarios, eight coarse tasks and
  256 scheduler calls per measurement. Steady throughputs were respectively
  3.0620, 5.8758, 10.3707 and 13.8816 calls/second.
- Eight-worker startup was 2.766 seconds and steady execution 18.442 seconds.
  The PreparedScenario initializer payload was 9,779,555 bytes once per worker,
  each task payload 212 bytes, and each compact result about 26.9 KiB. Parent
  CPU time was .75 seconds; transport and merge are not the steady bottleneck.
- The corrected mechanical upper bound is 18.43 hours at eight-worker steady
  throughput. The earlier 59.29-hour estimate used one scenario per task and
  incorrectly amortized spawn over only eight scheduler calls.
- Worker occupancy was about 99%, but true speedup was 4.533x and scaling
  efficiency 56.7%. Median task time increased from 10.357 seconds at one
  worker to 18.254 at eight, locating the remaining scaling loss in compute
  contention rather than idle parent or IPC.
- An in-process 32-scenario cProfile recorded `_observation` at 10.758 seconds
  and `_audit_queues` at 2.698 seconds of profiled cumulative time;
  `score_tagged_result` was .209 seconds. Observation still reconstructs a
  complete per-Expert history at every arrival.
- A read-only physical-core query was attempted but the Windows CIM provider
  returned access denied. The report therefore makes no unsupported claim
  about physical core count.

#### Artifacts

- `.scratch/shared-backup-game/parallel_scaling_probe.py`
- `.scratch/shared-backup-game/optimized_worker_profile.py`
- `.scratch/shared-backup-game/stage4-parallel-scaling-diagnosis-20260906.md`
- No Fit, Validation, candidate or campaign artifact.

#### Decisions and risks

- The production runner already keeps PreparedScenario state in the worker
  initializer, uses the frozen coarse task and returns compact rows. No
  production parallel-runner correctness bug was found.
- The 18.43-hour number is a corrected worst-case estimate, not timing-gate
  acceptance. Ticket 06 remains blocked and no Nash/MFG claim exists.
- Further reduction requires reference-equivalent single-call optimization of
  Pi256 observation construction and tagged non-output audit work. This has not
  been implemented in this diagnostic update.

#### Next

Add red tests for an exact compact Pi256 online observation path and tagged
terminal global-conservation verification, then repeat the scaling table. Do
not launch the campaign.

### Update: 2026-09-06 — Implement exact compact policy and tagged audit paths

Status: completed

#### Goal

Remove the measured Pi256 observation and tagged queue-audit CPU hotspots
without changing shared physics, causal decisions, CRN, scoring, frozen
samples, or the campaign protocol.

#### Changed

- Added a private compact request path in `shared_backup.py` and
  `campaign_execution.py`. It supplies exactly current state, phase, phase
  age, Token class, and Primary Replica to the exact base Pi256 rule.
- Kept full immutable `ActionObservation` construction for every generic
  source and every custom `ActionRule` subclass; `NotImplemented` explicitly
  falls back to that reference contract.
- Restricted tagged per-event queue auditing to the tagged Expert, while
  retaining O(1) ledgers for all Experts and adding a terminal comparison of
  every ledger against both the physical queues and the work equation.
- Added two focused regression tests for compact/full result identity and
  detection of a corrupted non-tagged ledger.
- Updated the diagnosis report with the post-change profile and scaling table.
- Did not start Fit, Validation, candidate selection, or a campaign.

#### Verification

- Red before implementation:
  `.venv/Scripts/python.exe -m unittest tests.test_shared_backup_performance -v`
  ran 8 tests with 2 failures: the compact source received a full observation,
  and corrupted non-tagged work was not rejected.
- Focused green:
  `.venv/Scripts/python.exe -m unittest tests.test_shared_backup_performance tests.test_campaign_execution tests.test_game_workload -v`
  ran 30 tests, all passed. This includes the intentional worker-failure/no-
  retry contract and generic online-observation causality.
- Full green:
  `.venv/Scripts/python.exe -m unittest discover -s .\tests -v` ran 458 tests,
  all passed in 28.574 seconds.
- Config gate:
  `.venv/Scripts/python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  returned exit 0 and `status: ok`, retaining the known single-domain
  capacity warning.
- Post-change 32-scenario profile: about 12.4 million calls and 5.6 profiled
  seconds versus the prior 46.6 million / 18.3 seconds; `_observation` left
  the leading hotspot list and `_audit_queues` fell from 2.698 to .892 seconds.
- Frozen 256-call scaling probe: 1/2/4/8 workers sustained respectively
  9.107 / 16.568 / 29.364 / 43.414 calls per second. Eight-worker startup was
  3.019 seconds and steady work 5.897 seconds.

#### Artifacts

- `src/mfg_hedge/shared_backup.py`
- `src/mfg_hedge/campaign_execution.py`
- `tests/test_shared_backup_performance.py`
- `.scratch/shared-backup-game/stage4-parallel-scaling-diagnosis-20260906.md`
- No Fit, Validation, candidate, Nash, MFG, or campaign artifact.

#### Decisions and risks

- Package version remains 0.23.0 because no exported API or scientific result
  contract changed; the compact method is private to the campaign wrapper.
- The new eight-worker rate is 3.127x the prior corrected rate. The mechanical
  921,088-call projection is now about 5.89 hours (Fit 5.05 + Validation .84),
  but this ticket does not decide or authorize campaign launch.
- CPU contention remains: eight-worker median task time is 5.857 seconds versus
  3.504 at one worker. Parent/IPC is still not the limiting path.

#### Next

Make a separate timing-gate disposition on ticket 06; if accepted, claim that
experiment ticket and run the already frozen campaign without further physics
or sample changes.

## Answer

Resolved. Both measured single-call hotspots were reduced with exact reference
equivalence, all required tests pass, and the corrected eight-worker mechanical
upper bound is 5.89 hours. No formal campaign was started.
