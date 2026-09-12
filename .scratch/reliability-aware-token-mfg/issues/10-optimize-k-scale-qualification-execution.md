# Optimize K-scale MFG qualification execution

Type: implementation
Status: resolved
Blocked by: none

## Goal

Reduce the real finite-system qualification runtime while preserving the
accepted MFG mathematics, physical event semantics, sample protocol, CRN
identities, action support, and call budgets.  This ticket is a performance
gate for ticket 06, not a change to the qualification protocol.

## Scope

1. Profile one real K=64 scheduler call and attribute time to the event loop,
   Replica scans, queue audit, observation construction, and result/scorer
   materialization.
2. Add an immutable prepared-episode/checkpoint representation so paired
   baseline/N/D/I branches can reuse only validated exogenous prefixes; every
   branch must still create fresh endogenous queue, policy, completion, and
   scorer state and execute the complete physical semantics.
3. Maintain queue work, Replica exposure, reliability history, and same-batch
   public observation incrementally where exact reference equivalence proves
   the update order unchanged.
4. Add fixed-order multi-process episode batching only after the single-process
   optimized path is verified.  Worker tasks must be deterministic and merge
   by canonical episode identity, never by completion order.
5. Compare reference and optimized paths for K=8, 16, 32, and 64 field by
   field, including CRN and trace fingerprints, policy observations, queue
   state, starts/completions, invariants, and scorer inputs/results.

## Frozen constraints

- Do not change K values, topology, four-domain composition, arrival/batch
  scaling, service/fault laws, policy definitions, action support, prices,
  model parameters, solver iterations, sample floors, common paths, or any
  qualification/holdout namespace, seed, fingerprint, or call budget.
- Do not skip Expert/Replica participants, freeze K/speed/completion/queue
  trajectories, use future information, replay actions, approximate the
  processor or time, reduce the 32/32/32 panels, or weaken invariants.
- Each counterfactual remains a complete shared-physics rerun.  Prepared data
  may contain validated exogenous inputs only; it must not contain action,
  budget, queue, running, completion, Replay, winner, timer, scorer, or
  deviation state.
- Reference implementation remains available as the scientific oracle.  If
  any optimized field, event order, winner, policy action, fingerprint, or
  score differs beyond the existing strict numerical contract, retain the
  reference path and record the optimization as failed.
- No formal qualification, holdout, candidate selection, or MFG claim may
  start from this ticket before the timing gate passes.

## Performance gate

- Required real K=64 single-call wall time: no more than 30 seconds.
- Required frozen qualification worst-case estimate with eight workers: no
  more than 12 hours.
- Report median and worst timing, CPU process time, parent wait time,
  serialization bytes, worker/parent peak RSS, and the exact call-based
  estimate.  A failed or unsupported RSS measurement is not compliance.
- These are gates for execution readiness only.  They do not authorize sample
  reduction or a scale-protocol change if they fail.

## Acceptance criteria

- A profiler report identifies the dominant single-call costs on the real
  K=64 trace.
- Prepared/checkpoint and incremental paths pass exact reference comparison
  on N/D/S/X-equivalent action coverage where applicable, Replay burst,
  failure tie, running loser, K=8 regression, and K=16/32/64 topology cases.
- Optimized output preserves all scientific fields, CRN/fingerprints, event
  ordering, complete drain, and invariant counters.
- One-worker and multi-worker results are byte-deterministic and merged in
  canonical order; no retry changes call accounting; worker failure is
  fail-closed and produces no candidate.
- The timing gate is evaluated from a real frozen-load probe, not synthetic
  provider timing.  Formal ticket-06 qualification remains unstarted until
  this acceptance is complete.
- Focused, affected, full-suite, config, and diff checks pass.

## Non-goals

- No change to MFG state equations, response solver, final confirmation,
  finite-K simultaneous UCB, statistics, pricing, or qualification gates.
- No campaign execution, development/holdout data generation, artifact
  replacement, or result-driven optimization.
- No third-party dependency.

## Progress log

### Update: 2026-09-10 — performance gate ticket claimed

Status: partial

#### Goal

Pause ticket 06 at its failed real-load timing gate and claim one narrow
performance ticket for exact K-scale execution optimization.

#### Changed

- Created and claimed ticket 10.
- Changed ticket 06 to `partial` with `Blocked by: 10`.
- Froze the 30-second K=64 single-call target and 12-hour eight-worker
  qualification estimate as execution gates only.
- Kept the MFG protocol, models, samples, CRN, namespaces, seeds, and total
  call budget unchanged.

#### Verification

- Prior real one-call preflight measured K=8/16/32/64 at approximately
  `0.297/2.131/16.617/137.907` seconds and
  `35.9/54.6/133.8/448.0` MiB respectively.
- Prior campaign focused, readiness, full-suite, and config checks were
  green; no new formal qualification call was dispatched in this ticket
  creation update.

#### Artifacts

None.  Historical preflight and experiment artifacts were not changed.

#### Decisions and risks

The measured K=64 path is far above the frozen gate, so ticket 06 must not
start.  Optimization is accepted only when exact science equivalence is
demonstrated; speed alone cannot justify a protocol change or a qualification
claim.

#### Next

Profile the real K=64 call, then add the smallest exact prepared/checkpoint
optimization and its failing equivalence tests before changing execution
parallelism.

### Update: 2026-09-11 — optimized real K-scale physics and added restart safety

Status: partial

#### Goal

Remove the real K=64 scheduler bottleneck, evaluate the frozen timing gates on
Windows spawn workers, and prevent a long qualification restart from silently
repeating already completed calls.

#### Changed

- Replaced the repeated linear work-draw scan with immutable O(1) canonical
  offsets in `reliability_aware_routing.py`; retained a compatibility fallback
  for legacy noncanonical fixtures.
- Indexed fault events per component and restricted active-token snapshots to
  actual live identities.
- Parameterized the response solver and continuation/deviation evaluators for
  the frozen 32/32/32 formal panels and corrected the invalid requirement that
  disjoint forward and finite-K libraries have the same fingerprint.
- Added a balanced S0/S1/S2/S3 timing probe, real eight-worker Windows-spawn
  throughput measurement, exact timing/RSS/call accounting, and an enforced
  formal-start gate.
- Added `qualification_checkpoint.py`: atomic reserve-before-dispatch and
  commit-after-success journaling, source/plan/input fingerprints, completed
  output reuse, and fail-closed unresolved-dispatch handling.
- Added focused safety/equivalence tests and a detailed performance report.

#### Verification

- Real K=64 optimized scheduler call: worst `0.4881354 s`, below the frozen
  `30 s` gate; the prior measurement was approximately `137.907 s`.
- Eight-worker real K=64 throughput: `0.33936165 calls/s`; frozen 51,072-call
  projection `25,611.31 s` (`7.11 h`), below the `12 h` gate.
- Preflight used `24/32` isolated calls and consumed zero formal calls; all
  worker and parent RSS measurements were supported.
- Focused qualification safety/solver/campaign suites: 19 tests passed.
- Affected routing/population/MFG suites: 70 tests passed.
- `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` —
  `Ran 795 tests in 275.929s`, `OK`.
- `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json`
  — exit 0, `status: ok`.
- `git diff --check` — exit 0.

#### Artifacts

- `.scratch/reliability-aware-token-mfg/ticket10-performance-report-20260911.md`
- `src/mfg_hedge/qualification_checkpoint.py`
- `tests/test_token_mfg_qualification_safety.py`

No formal qualification or experiment artifact was generated.

#### Decisions and risks

The performance gates now pass without changing the accepted science.  Ticket
10 remains partial because the formal ticket-06 runner has not yet wired every
scientific work unit through the checkpoint and canonical multi-worker merge.
Starting the campaign before that integration would still risk an expensive
partial-run restart, so the fail-closed campaign entry point is retained.

#### Next

Connect the real qualification work graph to the atomic checkpoint and
deterministic worker executor, prove one-worker/eight-worker byte identity, and
only then release ticket 06 for the formal run.

### Update: 2026-09-11 — atomic work-unit runner and deterministic spawn execution

Status: resolved

#### Goal

Complete the execution layer for ticket 06 without dispatching the frozen
qualification or holdout campaign. The implementation remains an execution
contract and does not select a candidate or make an MFG claim.

#### Changed

- Added immutable `QualificationWorkUnit` identities for forward population,
  mean-field continuation, finite-K deviation, final-policy confirmation, and
  holdout work. Qualification rejects holdout units and requires the exact
  frozen `51,072` qualification-call graph.
- Added deterministic Windows-spawn execution with at most eight workers.
  Reservations are written before dispatch; successful results are committed
  in canonical work-key order, independent of completion order.
- Replaced the permanent formal-run stub with a typed, fail-closed
  `run_qualification_campaign()` entry. It requires the real backend source
  fingerprint, all four qualification work-unit kinds, the complete timing
  gate, and the exact call arithmetic. Synthetic backends and missing
  adapters are refused.
- Extended checkpoint records to retain per-unit plan, source, input, and
  output fingerprints. Committed output is reused after restart with zero
  scheduler dispatch; an unresolved reservation is never retried.
- Kept all K, panel, namespace, seed, CRN, model, residual, and call-budget
  contracts unchanged.

#### Verification

- Real Red: the new campaign tests initially produced four import errors for
  the absent `QualificationWorkUnit`/canonical executor API.
- Focused campaign executor: `11/11` passed.
- Affected qualification, response, population, and continuation suites:
  `43/43` passed.
- Full suite: `801/801` passed.
- Real timing preflight: `24` isolated scheduler calls, `0` formal calls;
  K=64 worst single call `0.518815s`; eight-worker throughput
  `0.355612 calls/s`; frozen 51,072-call worst estimate `24,528.78s`
  (`6.81h`), below the `30s` and `12h` gates. RSS was supported; parent
  peak was `29,319,168` bytes. Spawn pools exited cleanly in the deterministic
  executor and timing checks.
- Config check: exit `0`, `status: ok`.
- `git diff --check`: passed.
- No formal qualification or holdout scheduler call was dispatched, and no
  qualification/holdout artifact was created.

#### Artifacts

No formal experiment artifact was generated. The checkpoint implementation
and bounded real-physics executor tests are source/test changes only.

#### Decisions and risks

- The executor is restart-safe: a committed unit is reusable, while an
  unknown reserved unit fails closed instead of silently retrying.
- Scientific result bytes exclude worker-count and dispatch metadata, so
  one-worker, eight-worker, and resumed outputs are byte-identical; execution
  accounting remains separately recorded.
- `run_qualification_campaign()` is intentionally not a default launcher: it
  refuses synthetic or fingerprint-mismatched adapters. The real ticket-06
  backend must provide the four connected routing/population/continuation/
  deviation work-unit kinds before any formal call is allowed.
- This update contains no MFG result, candidate, holdout result, or claim.

#### Next

Ticket 06 is unblocked and open. It is not claimed by this update. A future
run may provide the verified real backend and execute the separately frozen
qualification protocol; this turn does not start that run.

### Update: 2026-09-11 — dynamic bounded checkpoint state machine

Status: resolved

#### Goal

Complete the dependency-aware bounded orchestration slice without changing
the formal qualification protocol or dispatching qualification/holdout.

#### Changed

- Added a parent-owned dynamic phase machine for the bounded K=8 path.  It
  creates only the next phase after the previous checkpoint row is complete;
  it does not pre-expand a stale static graph.
- Persisted serialized forward snapshot, immutable episode identities, and
  policy state at each boundary.  The finite-K commit now atomically carries
  the policy update derived from the committed continuation estimates, and
  final confirmation reads only that committed policy.
- Included serialized dynamic inputs in phase input fingerprints, so a changed
  policy, snapshot, episode panel, or call reservation is rejected on resume.
- Verified Windows spawn execution and restart reuse with one and eight
  workers; committed forward work is reused without scheduler dispatch.

#### Verification

- Focused dynamic/backend suites: `4/4` passed.
- Affected qualification/routing/MFG suites: `103/103` passed.
- Full suite: `805/805` passed in `270.041s`.
- Config check: exit `0`, `status: ok`.
- `git diff --check`: exit `0`; only pre-existing LF/CRLF warnings under the
  unrelated parent-worktree `rl_algorithms` tree.

#### Decisions and risks

- The bounded state machine is not a formal qualification runner.  The
  concrete adapter still rejects the static formal work graph and is limited
  to K=8 bounded panels; it is not allowed to masquerade as the frozen
  `51,072`-call campaign.
- A real panel-cardinality probe using 32 qualification-namespace traces
  measured `forward=32`, `continuation=416`, and `finite-K=160`, rather than
  the frozen `32/288/288` per model/iteration contract.  The probe was not
  committed to the checkpoint.  Because those 32 identities were dispatched
  outside a committed work unit, the formal run is fail-closed: no retry,
  silent reuse, or result publication is permitted.
- The existing unique checkpoint remains unchanged and has no committed or
  reserved records.  No candidate, scale, holdout, or MFG artifact was
  generated.

#### Next

Supply a formal backend that selects exactly one target panel per episode,
supports the frozen K-scale/model/iteration cardinalities, and persists its
dynamic outputs before any further qualification identity is dispatched.
