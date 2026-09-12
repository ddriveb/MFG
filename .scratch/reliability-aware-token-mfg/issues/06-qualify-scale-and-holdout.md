# Qualify K-scale reliability-aware Token MFG and run independent holdout

Type: experiment
Status: claimed
Blocked by: none

## Goal

Only after the finite engine, share estimator, tagged deviations, and solver
are accepted, qualify the `K={8,16,32,64}` scaling convention and evaluate an
independent holdout under a separately accepted experiment protocol.

## Scope

- Freeze development and holdout episode counts, namespaces, macro seeds,
  source/physics/protocol fingerprints, common-noise libraries, bucket grid,
  confidence procedure, and maximum scheduler calls before any trace is
  generated.
- Include all named simultaneous baselines and the explicitly labelled
  centralized references.
- Verify `K=8` regression and report any larger-K approximation.
- Report physical, population, finite-deviation, social-cost, transfer, and
  predicted/realized-share metrics with episode-level statistical units.
- Use transactional unique artifact directories; a partial panel cannot
  publish candidate or equilibrium evidence.

## Frozen qualification protocol

This protocol is frozen before any qualification trace is generated.  It is
the only execution contract for this ticket; no result-driven sample,
parameter, or stopping-rule change is permitted.

### Solver panel and starts

- Each solver iteration uses 32 forward episodes, 32 mean-field continuation
  episodes with at most 9 calls per episode, and 32 finite-`K` deviation
  episodes with at most 9 calls per episode.
- The two separately reported models are `unpriced_mfg` and `priced_mfg`.
- Each run has at most 6 response iterations plus one final confirmation.
- Every supported cell/action requires at least 8 complete episodes.  A cell
  is active only when occupancy is at least 0.5% and it appears in at least 8
  episodes.  An occupied but under-supported cell is
  `statistics_insufficient`; there is no top-up or fallback.
- The K=8 starts are `uniform`, `simultaneous_loew_projection`, and
  `corrected_risk_aware_projection`.  All three must produce the same
  confirmed regularized candidate.  If confirmed candidates differ, the
  terminal status is `multiple_candidates`; no performance-based choice is
  allowed.
- If K=8 has no confirmed candidate, the terminal status is `no_candidate` and
  no larger-K or holdout run is started.
- If K=8 passes, initialize K=16 from the K=8 candidate, K=32 from K=16, and
  K=64 from K=32.  This is frozen scale continuation, not a uniqueness claim.

### Calls, identities, and CRN

- A solver run reserves at most
  `2*7*(32 + 32*9 + 32*9) = 8,512` calls, including final confirmation.
- Six runs are allowed: three K=8 starts and one run at each of K=16, K=32,
  and K=64, for at most 51,072 solver/confirmation calls.
- Preflight is isolated, at most 32 calls, and never consumes formal episode
  identities.
- Qualification forward, continuation, finite-deviation, and holdout inputs
  use distinct namespaces and macro seeds recorded in the manifest.  The
  qualification namespaces are
  `reliability-aware-token-mfg:v1:qualification:forward`,
  `reliability-aware-token-mfg:v1:qualification:continuation`,
  `reliability-aware-token-mfg:v1:qualification:finite-k`, and
  `reliability-aware-token-mfg:v1:qualification:holdout`; their macro seeds
  are respectively `20260916`, `20260917`, `20260918`, and `20260919`.
  Preflight uses the isolated namespace
  `reliability-aware-token-mfg:v1:qualification:preflight` and seed `20260920`.
- Source, physics, protocol, action-support, and per-library fingerprints are
  computed and recorded before formal execution.  Action-uniform CRN keys
  include namespace, macro seed, scenario, episode, batch, Token, and action
  or bucket identity, and exclude model, iteration, price, queue outcome, and
  realized service information.
- The same exogenous episode library is reused across response iterations for
  a given split; endogenous queues, actions, completions, and scorer state are
  fresh on every run.  Qualification and holdout libraries are disjoint.

### Qualification gates and terminal states

- Every iteration records complete policy rows, environment fingerprints,
  population/share residuals, Q rows, finite-`K` diagnostics, call counts, and
  stop reason.
- A confirmed run requires the final policy/environment/Q tuple from the same
  policy, policy/population/share residuals no greater than `0.01`, share
  calibration, complete physics/CRN/drain invariants, and finite-`K` gain
  simultaneous UCB diagnostics.
- Across K, normalized finite-`K` UCB must trend downward overall, K=64 must
  have normalized UCB no greater than 2%, and the K=32/K=64 policy L1 distance
  must be no greater than 0.05.
- Formal terminal states are `no_candidate`, `multiple_candidates`,
  `statistics_insufficient`, `physical_failed`, `not_converged_6`,
  `qualification_pass`, and `holdout_failed`.  No failed or partial panel
  can publish a candidate or Pareto/holdout conclusion.

## Frozen independent holdout

- Holdout starts only after K=64 qualification passes all gates.
- Each K in `{8,16,32,64}` uses 224 fresh episodes and exactly nine unique
  arms: simultaneous RR, simultaneous JSQ, simultaneous LOEW,
  Reliability-only, corrected risk-aware, Lazarus, `unpriced_mfg`,
  `priced_mfg`, and sequential LOEW centralized reference.
- Holdout calls are `224*4*9 = 8,064`; with 32 preflight calls and 51,072
  qualification calls the hard total is `59,168`, below 60,000.
- Holdout is paired within episode on identical workload, fault, service, and
  action-uniform CRN streams.  It cannot tune policy, price, state grid,
  temperature, damping, or stopping rules.
- Qualification and holdout outputs are finite-system diagnostics.  A passing
  consistency gate may be labelled `regularized_mfg_approximation`; it is not
  exact Nash, unique MFG, optimality, or universal evidence.  The priced arm
  represents only the frozen `0.25` observable charge.

## Acceptance criteria

- Development and holdout identities are disjoint and holdout cannot tune any
  parameter, price, state grid, policy, or stopping rule.
- All CRN and complete-drain/invariant checks pass or the run is fail-closed.
- The report distinguishes a finite routing result, regularized candidate,
  finite-N epsilon evidence, and any MFG consistency residual.
- No result is called exact Nash, unique MFG, optimal, or universal without a
  separately accepted proof/protocol supporting that claim.

## Next dependency

This is the final ticket in the design chain.  A separate review must accept
the qualification protocol before any formal campaign starts.

### Update: 2026-09-11 — paused full regression and config check completed green

Status: completed

#### Goal

Execute the full regression and config verification that was paused by the user
when the r2 panel finished, before any further ticket-06 work.

#### Changed

None. Read-only verification session.

#### Verification

- `.venv\Scripts\python.exe -m unittest discover -s .\tests -v` —
  **824/824 passed in 294.139s, OK** (full suite, executed in this update).
- `.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json` —
  exit 0, `status: ok` (audit finding unchanged: single-domain failure load
  1.4 > 0.9 capacity, reported as a transient-scenario caveat).
- No regression failure; task B was therefore not stopped.

#### Artifacts

None.

#### Decisions and risks

The streaming formal backend (ticket 14) is fully regression-clean against the
frozen suite. This verifies code health only; it does not change the r2
`statistics_insufficient` outcome or create any qualification evidence.

#### Next

Proceed to the analytic protocol-redesign derivation (recorded in the
following update) awaiting user adjudication; no new experiment.

### Update: 2026-09-11 — protocol redesign derivation from r2 occupancy data

Status: completed

#### Goal

Turn "43 states + 32 episodes is infeasible" into a quantified, adjudicable
proposal using only r2 artifacts and frozen source definitions, with no new
experiment.

#### Changed

- Added read-only analysis scripts (decode committed r2 blobs; zero scheduler
  calls, no trace generation): `.scratch/reliability-aware-token-mfg/analyze_r2_cell_occupancy.py`,
  `analyze_r2_coarsening.py`, with outputs `r2_cell_occupancy_report.json`
  and `r2_cell_bucket_table.json`.
- Added the derivation document `.scratch/reliability-aware-token-mfg/protocol-redesign-derivation-20260911.md`.
- No production code, protocol, configuration, or artifact changed; no new
  `artifacts/` directory.

#### Verification

- Blob-decode cross-check: recomputed `max((gain_mean + 2*SE)/baseline)` over
  the 32 committed r2 episode blobs = 1.190440, identical to the recorded
  normalized finite-K UCB; per-cell counts {1:48, 2:7, 3:4, 11:4}, 59/63 below
  the frozen floor 8, matching the recorded fail-closed outcome.
- Exact commands: `./.venv/Scripts/python.exe .scratch/reliability-aware-token-mfg/analyze_r2_cell_occupancy.py`
  and `./.venv/Scripts/python.exe .scratch/reliability-aware-token-mfg/analyze_r2_coarsening.py`
  (read-only artifact decoding, no scheduler dispatch).

#### Artifacts

- `.scratch/reliability-aware-token-mfg/protocol-redesign-derivation-20260911.md`
- `.scratch/reliability-aware-token-mfg/analyze_r2_cell_occupancy.py`,
  `r2_cell_occupancy_report.json`, `analyze_r2_coarsening.py`,
  `r2_cell_bucket_table.json`

#### Decisions and risks

Derivation summary (full numbers in the document; all protocol changes are
proposals awaiting user adjudication, nothing frozen modified):
- Occupancy profile: 63 cells over 18 target states (43 forward states; 25
  occupancy >= 1.27% states never targeted in 32 episodes). Floor-8 needs
  N ~ 632 episodes (point) / ~1,100 (95%) at the observed occupancy floor, and
  up to ~2,800 for a state at the 0.5% active boundary — open-ended under the
  frozen definition.
- Fork 1 (coarsening): all 63 cells share one replica-side action bucket, so
  only token-side coarsening acts. Token-bucket-only state identity (C1) caps
  cells at 120, makes the state space K-invariant, and needs N ~ 280-460;
  per-state token bucket labels are not persisted in r2 blobs, so merged
  occupancies are bounds marked 【待确认】, not measurements.
- Fork 2 (threshold): UCB ~ 1/sqrt(n) gives multipliers 3,543x / 567x / 142x /
  23x for thresholds 0.02 / 0.05 / 0.10 / 0.25. The gate failed on the floor,
  not the threshold, so threshold relaxation alone cannot unblock.
- Fork 3 (share residual primary): 1.7143 = 12/7 is the n=1 one-hot artifact;
  the frozen 0.01 gate needs ~1e4-1e5 episodes per state — strictly harder than
  the UCB gate.
- Cost wall: any floor-satisfying N (>= ~300) pushes every K=64 iteration past
  the 12h gate (37h+ at N=280); K=8 stays feasible to N ~ 2,800 (<= 4.3h).
- Recommendation: token-bucket-only state identity + K=8-only formal
  qualification at N ~ 600 + K=16/32/64 demoted to out-of-sample diagnostics;
  if the protocol is frozen verbatim, substitute a support-sufficient-cell-
  subset qualification question.

#### Next

User adjudication of the recommended package or the substitute research
question. Do not start any new experiment, K-scale panel, or holdout until the
protocol question is decided.

### Update: 2026-09-11 — r2 streamed qualification stopped at the first gate

Status: partial

#### Goal

Connect the memory-bounded streamed backend to the accepted r2 protocol and
start the formal qualification without materializing K-scale trace libraries.

#### Changed

- Added a per-episode formal worker that performs forward, continuation, and
  finite-K physics while returning compact sufficient statistics only.
- Added parent-side panel aggregation and a formal r2 runner with canonical,
  content-addressed checkpoint units.
- Corrected the r2 timing worker to carry the active plan and corrected the
  streaming scenario schedule/work-unit identity.

#### Verification

- Focused streaming/response tests passed before formal dispatch.
- The fresh r2 timing preflight passed with 24 isolated calls, zero formal
  calls, K=64 worst call 0.455867 seconds, and a projected 8-worker ceiling of
  23668.81 seconds.
- The first K=8/uniform/unpriced panel completed all 32 episode identities:
  608 calls reserved, 332 actually completed, and no unresolved work units.
- The formal result is `statistics_insufficient`; finite-K support is below
  the frozen per-cell floor, normalized simultaneous UCB is 1.19044, and the
  share residual is 1.71429.  Confirmation and holdout were not started.

#### Artifacts

- `artifacts/reliability-aware-token-mfg-qualification-20260911-r2/`
- `.scratch/reliability-aware-token-mfg/qualification-runs/reliability-aware-token-mfg-qualification-20260911-r2/`

#### Decisions and risks

This is a protocol-defined fail-closed qualification outcome, not an MFG or
Nash result.  The user requested a pause immediately as the first panel was
finishing; no later start, model, K-scale panel, or holdout was dispatched.
The full regression/config verification for the new execution bridge remains
pending because work was paused.

#### Next

Pause.  Do not resume or reinterpret the r2 identities without a new user
instruction.

### Update: 2026-09-10 — readiness dependency cleared

Status: partial

#### Goal

Record that the K-scale and final-policy confirmation readiness dependency is
complete without claiming or starting qualification.

#### Changed

- Changed the dependency marker from `Blocked by: 08` to `Blocked by: none`.
- Kept ticket 06 unclaimed and retained its independent review, claim, sample,
  and campaign-start gates.

#### Verification

- Ticket 08 is resolved after its focused, affected, full-suite, syntax, and
  config checks passed.
- No ticket-06 qualification, holdout, or formal campaign calls were run.

#### Artifacts

None.

#### Decisions and risks

Readiness completion does not authorize a campaign.  Ticket 06 must still be
reviewed and claimed under its frozen protocol; no K-scale qualification
result or MFG conclusion exists.

#### Next

Perform the normal ticket-06 review/claim decision before any qualification or
holdout execution.

### Update: 2026-09-10 — correction dependency cleared

Status: partial

#### Goal

Record completion of ticket 09 without claiming or starting the qualification
campaign.

#### Changed

- Changed the dependency marker from `Blocked by: 09` to `Blocked by: none`.
- Kept ticket 06 open and unclaimed with its frozen K-scale qualification
  scope, sample protocol, and call budget unchanged.

#### Verification

- Ticket 09 is resolved after its focused, affected, full-suite, diff, and
  config checks passed.
- No qualification, development, holdout, or formal experiment calls were
  run in this update.

#### Artifacts

None.

#### Decisions and risks

The corrected modulo-8 risk composition and action-CRN identity are now
verified. This only clears the dependency; it is not a qualification result
and does not create any MFG or scale claim.

#### Next

Review and explicitly claim ticket 06 before any qualification execution.

### Update: 2026-09-10 — qualification protocol frozen and ticket claimed

Status: partial

#### Goal

Freeze the formal K-scale qualification and independent holdout contract and
claim ticket 06 for execution, without consuming calls before the runner is
verified.

#### Changed

- Set ticket 06 to `claimed`.
- Froze 32/32/32 episode panels, at most 9 calls per action panel, minimum
  8 complete episodes per active cell/action, and six solver runs with one
  final confirmation each.
- Froze the three K=8 starts, scale-continuation order, terminal statuses,
  residual/UCB gates, independent 224-episode-per-K holdout, nine arms, and
  the exact 59,168 total call ceiling.
- Froze qualification and holdout namespaces, macro seeds, and the action-CRN
  exclusions; fingerprints remain manifest values computed before execution.

#### Verification

- Read-only inspection confirmed the existing solver is still the bounded
  callback kernel and that no formal qualification call has run.
- No preflight, qualification, holdout, or experiment call was consumed in
  this protocol-freezing update.

#### Artifacts

None.

#### Decisions and risks

The protocol is now numerically closed, but the production tree still needs a
real runner that binds the actual finite routing/population/response layers;
synthetic providers remain test-only.  No result can be called a candidate
until final confirmation and all gates pass.

#### Next

Add Red tests for the fail-fast runner, exact reservations, start ordering,
final confirmation, and holdout gating before implementing execution code.

### Update: 2026-09-10 — K=8 readiness smoke

Status: partial

#### Goal

Run the existing real K-scale readiness checks before implementing or starting
the formal qualification runner.

#### Changed

None.  No production code, protocol, configuration, or artifact changed.

#### Verification

- `.venv\\Scripts\\python.exe -m unittest tests.test_token_mfg_qualification_readiness -v`
  passed `8/8` in `1.282s`.
- The suite includes real K=8 physics regression, simultaneous decision
  semantics, population/action support, fixed-library identity, final-policy
  confirmation, and finite-K UCB contract checks; its readiness cases also
  exercise K=16/32/64 topology lengths.
- No formal qualification solver, holdout, or synthetic-provider result was
  used, and no qualification call budget was consumed.

#### Artifacts

None.

#### Decisions and risks

K=8 readiness remains green, but this is not a MFG qualification result.  The
formal runner and exact 32-episode/9-call qualification panel are still not
implemented, so no candidate or scale conclusion is available.

#### Next

Implement the real qualification runner under the frozen ticket-06 protocol,
then run its <=32-call preflight before any formal panel.

### Update: 2026-09-10 — CRN and K-scale correction dependency

Status: blocked

#### Goal

Pause qualification until the K-scale risk composition and action-CRN identity
contract are corrected and independently verified.

#### Changed

- Set the dependency marker to `Blocked by: 09`.
- Kept the ticket-06 qualification protocol, samples, namespaces, seeds, and
  call budget unchanged.

#### Verification

- Read-only review confirmed that ticket 08 is resolved and no qualification
  calls have been started in this update.

#### Artifacts

None.

#### Decisions and risks

The correction is required before scale comparisons or residual confidence
intervals can be interpreted. Ticket 06 remains unclaimed; this update does
not authorize qualification or alter the MFG model.

#### Next

Complete ticket 09, then restore this ticket to `Blocked by: none` without
claiming it or starting qualification.

### Update: 2026-09-10 — real qualification preflight failed timing gate

Status: partial

#### Goal

Connect ticket 06 to the real finite routing engine and measure the frozen
K-scale workload before any qualification call is dispatched.  Formal
qualification and holdout were not started.

#### Changed

- Added the isolated
  `src/mfg_hedge/token_mfg_qualification_campaign.py` execution-contract
  module.
- Added immutable plan validation for the frozen K values, three K=8 starts,
  two models, 32/32/32 panels, six-run 51,072-call qualification ceiling,
  8,064-call holdout ceiling, and 59,168 total ceiling.
- Added real trace-library construction and a real routing-engine preflight;
  preflight identities use the frozen preflight namespace/seed and cannot
  consume formal episode identities.
- Added Windows `PROCESS_MEMORY_COUNTERS` working-set measurement.  The
  formal campaign entry remains fail-closed until a concrete adapter connects
  the real population/continuation/deviation layers; no synthetic provider is
  accepted.

#### Verification

- Red: the new campaign focused suite initially failed at import with
  `ModuleNotFoundError` because no campaign module existed.
- Campaign focused suite: `5/5` passed.
- Existing qualification-readiness suite: `8/8` passed.
- Full suite: `788/788` passed in `419.791s`.
- Config check: `status: ok`, exit 0.
- `git diff --check`: passed.
- Real isolated preflight, one complete episode per K, four scheduler calls in
  total, all complete-drain and physical invariant counters passed:
  `K=8 0.2971079s / 37,654,528 bytes`,
  `K=16 2.1314786s / 57,270,272 bytes`,
  `K=32 16.6170904s / 140,308,480 bytes`,
  `K=64 137.9065687s / 469,762,048 bytes`.
- All four preflight rows report `formal_calls_consumed=0` and
  `rss_supported=true`.  The preflight plan fingerprint was
  `ff67595d8387d0a2469fdb9d971b1cf7c83ff8c815cd6ee3b8f5c45032c27d02`.
- Based on the frozen maximum 8,512 calls per solver run, the raw worst-case
  estimate is approximately 15.5 days on this host; this is a timing-gate
  failure, not a qualification result.

#### Artifacts

No qualification, holdout, candidate, Nash, MFG, or other formal experiment
artifact was generated.  No historical artifact was modified.

#### Decisions and risks

- Formal qualification was not started because the real K=64 preflight is
  approximately 138 seconds per physical call and the resulting frozen
  maximum is not an acceptable wall-clock execution.
- K=64 preflight peak working set is approximately 448 MiB, below but close
  to the 512 MiB worker limit; this is recorded as a real measurement, not a
  Python-heap estimate.
- The preflight only verifies real routing physics and identity isolation.  It
  does not establish convergence, a population candidate, finite-K UCB
  qualification, or an MFG result.
- Ticket 06 remains `claimed`; no holdout gate was evaluated and no start was
  selected based on performance.

#### Next

Implement and test the concrete real adapter for population forward,
mean-field continuation, final confirmation, and finite-K simultaneous
deviation before any formal calls.  Re-run the timing gate after that adapter
is complete; do not launch the frozen campaign while this gate remains
failed.

### Update: 2026-09-10 — blocked by K-scale performance gate

Status: blocked

#### Goal

Record that the real K-scale qualification is paused for a dedicated exact
execution optimization, without changing or starting the frozen campaign.

#### Changed

- Set ticket 06 to `partial` with `Blocked by: 10` at the ticket header.
- Preserved the qualification/holdout sample counts, namespaces, seeds,
  fingerprints, gates, and call ceilings unchanged.

#### Verification

- The real preflight measured approximately `0.297s`, `2.131s`, `16.617s`,
  and `137.907s` per physical call for K=8, 16, 32, and 64 respectively.
- No qualification, holdout, or campaign call was dispatched in this update.

#### Artifacts

None.

#### Decisions and risks

The K=64 path fails the frozen execution gate by a wide margin.  This is a
performance blocker, not evidence about MFG convergence or model validity.
Ticket 06 remains unclaimed for execution while ticket 10 is active.

#### Next

Complete ticket 10's exact profiling and optimization gate, then restore
ticket 06 to `open` without automatically launching qualification.

### Update: 2026-09-11 — r2 pre-launch audit blocked unsafe scale handoff

Status: blocked

#### Goal

Start r2 only after verifying that every formal start, scale continuation, and
qualification gate is derived inside the checkpointed execution path.

#### Changed

- Set `Blocked by: 13` before creating an r2 checkpoint.
- Proposed ADR-0039 for the missing start-policy, cross-K projection,
  normalized-UCB, policy-distance, timing, and call-ceiling contracts.
- Did not modify the accepted r2 identities or dispatch any formal work.

#### Verification

- The current formal runner accepts caller-supplied policies for all K values
  and does not consume its stored preceding-K policy during scale continuation.
- The current runner does not require its timing report and can reach
  `qualification_pass` without the frozen cross-K UCB/distance gates.
- Formal r2 scheduler calls remain zero; no r2 checkpoint or artifact exists.

#### Artifacts

- `docs/adr/0039-formal-token-mfg-scale-handoff-and-gates.md`
- `.scratch/reliability-aware-token-mfg/issues/13-correct-formal-start-and-scale-handoff.md`

#### Decisions and risks

This is a pre-launch correctness stop, not an MFG or performance result.  The
missing definitions affect the candidate itself and cannot be chosen after
observing formal data.

#### Next

Resolve ticket 13 after explicit ADR-0039 acceptance, then create the unique r2
checkpoint and begin K=8.

### Update: 2026-09-11 — execution gate cleared

Status: open
Blocked by: none

Ticket 10 completed the atomic checkpoint and deterministic Windows-spawn
execution layer. Its real timing preflight passed the K=64 `30s` and frozen
eight-worker `12h` gates, with full tests and config check green. No formal
qualification or holdout call was dispatched. Ticket 06 remains unclaimed;
the real backend must be supplied at campaign launch and all qualification
fail-closed gates remain unchanged.

### Update: 2026-09-11 — formal run claimed; adapter gate before K=8

Status: claimed
Blocked by: none

#### Goal

Start the unique frozen qualification run with checkpointed, fail-fast
execution. The run is ordered K=8 three starts, then scale continuation, then
holdout only after all qualification gates pass.

#### Changed

- Claimed ticket 06 and reserved the unique run identity
  `reliability-aware-token-mfg-qualification-20260911-r1`.
- Created its parent-owned checkpoint before any scheduler dispatch.
- The checkpoint records plan fingerprint
  `e5e42f9d50586ce3767bd5cf4598e517ecd81832386b08759e9d7213b2342ee5`
  and source fingerprint
  `99c674cfcf61ecefbd05e60043248c8df0fa3eda281047d7de63bf3586f0be8f`.
- Verified that the repository currently contains no concrete real
  qualification backend implementing the required forward, continuation,
  finite-K deviation, and final-confirmation work graph.

#### Verification

- Formal scheduler calls dispatched: `0`.
- K=8 qualification was not started because the required real backend is
  absent; synthetic providers are forbidden by the frozen protocol.

#### Artifacts

Only the unique empty checkpoint/run journal was created. No qualification,
candidate, scale, holdout, or MFG result artifact was generated.

#### Decisions and risks

This is a fail-closed pre-dispatch stop, not `no_candidate` or a qualification
result. Starting without the concrete adapter would invalidate the work-unit
identity and scientific execution contract.

#### Next

Implement or supply the fingerprint-checked real qualification backend, then
resume this same checkpointed run at K=8. Do not create a second run, retry a
reserved unit, start larger K, or start holdout before K=8 confirmation.

### Update: 2026-09-11 — bounded concrete backend integration

Status: partial

#### Goal

Connect a bounded K=8 adapter to the real forward population, conditional
continuation, finite-K deviation, response solver, and final-policy
confirmation layers without dispatching the frozen qualification or holdout
campaign.

#### Changed

- Added
  `src/mfg_hedge/token_mfg_qualification_backend.py` with a real bounded
  `ConcreteQualificationBackend` over immutable K=8 episode traces.
- The adapter runs the complete simultaneous routing engine for forward
  episodes, materializes the first causal decision cohort, calls
  `evaluate_mean_field_continuation`, re-runs every finite-K action branch
  through `evaluate_finite_k_deviation`, invokes the two-model response solver,
  and invokes `confirm_final_policy` on the returned policy.
- Added checkpoint-safe bounded result serialization and resume projection;
  resumed output is read from the existing committed row and does not dispatch
  scheduler calls.
- Added
  `tests/test_token_mfg_qualification_backend.py` with real K=8 two-episode
  fixtures, layer/call-count assertions, source/trace fingerprint assertions,
  and checkpoint reuse coverage.

#### Verification

- Red: focused backend suite initially failed at import with
  `ModuleNotFoundError: No module named 'mfg_hedge.token_mfg_qualification_backend'`.
- Green focused backend suite: `2/2` passed.
- Affected suites (backend, campaign executor, population estimator,
  continuation, and response solver): `45/45` passed.
- Full suite: `803/803` passed in `264.507s`.
- Config check: exit 0, `status: ok`.
- `git diff --check`: exit 0; only pre-existing LF/CRLF warnings under the
  unrelated `rl_algorithms` tree were reported.
- Bounded real K=8 result: `forward=10`,
  `mean_field_continuation=80`, `finite_k_deviation=40`,
  `final_confirmation=44`, total scheduler calls `174`, response invocations
  `2`; all five requested layer identities were observed.
- The bounded run used the real trace and physical branch fingerprints; no
  formal calls were consumed.  The existing run checkpoint remains bytewise
  unchanged with an empty record set, `reserved=0`, and `completed=0`.
- Checkpoint resume returned byte-identical scientific output with
  `dispatched_calls=0` and `reused_calls=1`.

#### Artifacts

- Source: `src/mfg_hedge/token_mfg_qualification_backend.py`
- Focused tests: `tests/test_token_mfg_qualification_backend.py`
- Existing empty run checkpoint only; no qualification, holdout, candidate,
  scale, or MFG artifact was generated.

#### Decisions and risks

- The adapter is deliberately bounded and does not claim a qualification
  result.  It refuses to masquerade as the frozen formal work graph.
- The current static executor cannot safely pass each iteration's newly
  produced `PolicyState` and `ForwardEpisode` to later checkpointed units
  without replaying physical scheduler calls or persisting endogenous state.
  Therefore `build_qualification_work_units()` remains fail-closed for the
  formal 51,072-call graph; no formal run was started.
- The same run/checkpoint is retained.  No retry, second run, larger K, or
  holdout was started.

#### Next

Add the dependency-aware formal work-graph adapter (with persisted policy and
forward outputs) before resuming the same checkpoint at K=8; do not launch
qualification or holdout while that dependency remains unresolved.

### Update: 2026-09-11 — dynamic checkpoint orchestration is bounded only

Status: claimed (fail-closed)
Blocked by: formal panel-cardinality backend

#### Goal

Connect the real bounded routing/population/continuation/deviation layers to a
dynamic, restart-safe checkpoint state machine before consuming the frozen
qualification campaign.

#### Changed

- Added the bounded dynamic order
  `forward_population -> mean_field_continuation -> finite_k_deviation ->
  policy update -> final_policy_confirmation`.
- Each committed boundary carries serialized policy state and canonical
  episode identities; subsequent phases are derived only from committed
  output.  Resume reuses committed work and never dispatches it again.
- The finite-K boundary atomically commits the policy update used by final
  confirmation; no policy update is accepted from an uncommitted continuation
  result.
- Retained the existing unique run and checkpoint; no static formal graph or
  synthetic provider was used.

#### Verification

- Real Red already recorded for the absent concrete backend import; the
  bounded implementation now passes its focused backend/state-machine tests.
- Focused backend/state-machine tests: `4/4`.
- Affected suites: `103/103`.
- Full suite: `805/805` in `270.041s`.
- Config check: exit `0`, `status: ok`.
- `git diff --check`: exit `0` apart from pre-existing unrelated line-ending
  warnings.
- One/eight-worker bounded outputs are byte-identical; interruption after the
  forward commit resumes without repeating committed scheduler calls.
- A real 32-episode panel-cardinality probe measured `forward=32`,
  `mean_field_continuation=416`, and `finite_k_deviation=160`; this does not
  satisfy the frozen per-model/iteration `32/288/288` contract.

#### Artifacts

No qualification, candidate, scale, holdout, or MFG result artifact was
generated.  The unique checkpoint remains unchanged with no committed or
reserved work records.

#### Decisions and risks

- The formal run is stopped before K=8 qualification because the current
  adapter does not yet provide the exact frozen panel cardinality and still
  explicitly refuses the formal work graph.  This is a protocol/backend
  blocker, not a qualification result.
- The 32-episode cardinality probe used qualification namespace identities but
  was not checkpoint-committed.  Under the frozen no-retry/no-unknown-state
  rule, those identities cannot be silently replayed or counted as a formal
  result; the run remains fail-closed.
- Formal scheduler calls must not be reported as a candidate or MFG result;
  the existing checkpoint metadata itself still has `reserved=0` and
  `completed=0`.

#### Next

Implement or provide the formal panel backend with one future-blind target per
episode and exact `32/288/288` call accounting, then reconcile the
uncommitted probe identities under the frozen checkpoint policy before any
further campaign dispatch.  Do not start holdout.

### Update: 2026-09-11 — formal backend dependency cleared

Status: open
Blocked by: none

#### Goal

Record that the accepted ADR-0038 backend dependency is complete without
starting fresh r2 qualification or consuming formal calls.

#### Changed

- Cleared `Blocked by: 12` after ticket 12 completed the dynamic formal
  backend, simultaneous-UCB payload, final confirmation path, and bounded
  deterministic verification.
- Kept this ticket unclaimed and preserved the frozen r2 namespace, seed,
  panel, scale, gate, and holdout contracts.

#### Verification

- Ticket 12 focused/affected/full verification is green (`6/6`, `34/34`,
  `811/811`); config check is `status: ok`.
- No r2 checkpoint, qualification call, holdout call, or experiment artifact
  was created.

#### Decisions and risks

This is dependency clearance only.  The retired r1 namespace remains
unusable, and no qualification or MFG conclusion exists.

#### Next

Claim this ticket separately, create the unique fresh r2 checkpoint, run the
<=32-call timing/coverage preflight, and stop immediately on any gate failure.
Start holdout only after the complete K=64 qualification pass.

### Update: 2026-09-11 — ADR-0039 correction passed; launch held for streaming safety

Status: claimed
Blocked by: none

#### Goal

Resume r2 only if the corrected formal runner can finish every K without a
foreseeable payload or checkpoint failure.

#### Changed

- Ticket 13 completed Accepted ADR-0039 and passed its full acceptance suite.
- This ticket is now blocked by execution correction ticket 14 rather than 13.
- No qualification or holdout code path was launched.

#### Verification

- Ticket 13 full suite: 815/815 passed; config check returned `status: ok`.
- One isolated preflight-namespace K=64 trace serialized to 36,910,699 bytes;
  the current 32-trace formal argument projects to 1,181,142,368 bytes before
  forward episodes, JSON expansion, and multiprocessing copies.
- Formal r2 scheduler calls, checkpoint records, and artifacts remain zero.

#### Artifacts

None.

#### Decisions and risks

Launching with the current monolithic panel would create a predictable mid-run
memory/checkpoint risk and invalidate the purpose of resumability.  Ticket 14
may change execution storage and work-unit granularity only; scientific inputs
remain frozen.

#### Next

Complete ticket 14, then resume this same r2 protocol with fresh, unconsumed
formal identities.
