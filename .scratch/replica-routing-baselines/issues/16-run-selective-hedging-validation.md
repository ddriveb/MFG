# Run Selective Hedge development and holdout validation

Type: experiment
Status: resolved
Blocked by: none

## Scope

Run the independent ADR-0034 Selective Hedge development/holdout validation
after every required execution parameter is frozen. Development may select
only one `(theta_Regular, theta_Urgent)` pair from the frozen 3×3 normalized-
slack grid. After selection, the pair and all other policy parameters are
immutable for holdout. Holdout must compare the selected policy primarily
against `budgeted-laedge:delta=0%` dynamic No-Hedge, while retaining fixed
No-Hedge and NIIN conservative as routing/protection references.

The fixed `planned_budget_rate` is
`0.09304612180547656`, reused from the Budgeted LÆDGE v2 nominal 5% mapping.
It is a fixed work-capacity envelope, not a guarantee of 5% realized work
increase. Underfill is valid and must be reported; no arm may be labeled
`achieved_5_percent` from this planned rate alone.

## Frozen protocol already supplied by ADR-0034

- Slack is `s_i(t) = deadline_class - (t - arrival_time)`.
- Normalized slack is `bar_s_i(t) = s_i(t)/deadline_class`.
- Development grid:
  `theta_Regular, theta_Urgent ∈ {0.15, 0.30, 0.50}`.
- Eligibility requires `0 < bar_s_i(t) <= theta_class`, D state, Primary on
  degraded A, healthy/startable B, no existing Hedge, and no startable
  Primary/Replay waiting work.
- H/R are disabled and F is forbidden; cancellation is conservative.
- Development namespace:
  `replica-routing-baselines:selective-hedging:v1:development`.
- Development macro seed: `20260916`; episodes: `256`.
- Development arms: fixed No-Hedge, NIIN conservative, Budgeted LÆDGE
  `delta=0%`, and nine threshold candidates; call budget `3,072`.
- Holdout namespace:
  `replica-routing-baselines:selective-hedging:v1:holdout`.
- Holdout macro seed: `20260917`; episodes: `1,024`.
- Holdout arms: the same three references plus the one selected candidate;
  call budget `4,096`.
- Existing episode CRN fingerprint, metrics, paired episode-cluster unit, and
  4,096-replicate bootstrap remain unchanged.
- All nine development candidates and the selected holdout candidate use this
  exact fixed rate; neither panel may recalibrate it.
- No retry, supplement, deletion, result-driven retuning, or artifact
  overwrite is allowed.

## Fixed budget envelope

Record the fixed rate in the protocol, manifest, source/provenance
fingerprint, and candidate-selection rows. It is not a counterfactual NIIN
work target and must not be changed after development selection. Actual
realized work may underfill the envelope and must be reported separately.

## Development selection and holdout gate

The selection and acceptance gate is against the paired dynamic No-Hedge arm:

- at least one of D/F CVaR95 or D/F deadline-miss rate has a paired 95% CI
  strictly below zero for candidate minus delta=0;
- the other metric has a paired 95% CI with upper endpoint no greater than
  zero;
- mean realized total-work increase is at most 5% relative to delta=0;
- Protection-Storm peak is at most 2.

If no development candidate passes, status is `non_improving` and holdout is
not run. If development or holdout has a missing/failing arm, CRN mismatch,
invariant failure, or incomplete counterfactual, status is `physical_failed`
or `statistics_insufficient` as appropriate and no partial Pareto result is
published.

Both panels must report overall and D/F/H/R latency tails, D/F CVaR95,
deadline miss, Replay, total/Hedge/incremental/wasted work, Hedge launches,
winner rate, suppression reasons, D-phase Hedge timing and Token-class
distribution, Protection-Storm peak/duration, drain, and complete invariant
counters.

## Acceptance criteria

1. The planned budget rate is frozen before any scheduler call.
2. Red tests cover rate guard, nine-candidate selection, delta=0 primary
   comparison, fixed CRN identity, no holdout retuning, exact call counts,
   failure fail-closed, suppression audit, and transactional non-overwriting
   artifacts.
3. Development completes exactly 3,072 attempted calls or fails closed.
4. A successful development selection freezes exactly one threshold pair and
   its rate before holdout.
5. Holdout completes exactly 4,096 attempted calls or fails closed.
6. No result is labeled optimal, BR, Nash, or MFG.

## Progress log

### Update: 2026-09-09 — Created validation ticket and found rate gap

Status: blocked

#### Goal

Create the independent Selective Hedge development/holdout experiment ticket
without starting execution until the causal budget rate is fully specified.

#### Changed

- Created ticket 16 with the ADR-0034 development/holdout sample, namespace,
  seed, CRN, comparison, and call contracts.
- Recorded the missing `planned_budget_rate` as an execution blocker because
  it changes Hedge admission, realized work, Protection-Storm, and the
  delta=0 Pareto comparison.

#### Verification

- Confirmed ADR-0034 freezes the threshold grid and panel arithmetic but does
  not specify a Selective Hedge planned budget rate.
- Confirmed no scheduler call, calibration, holdout run, or artifact was
  created.
- Confirmed ticket 15 is resolved and no implementation ticket is claimed.

#### Artifacts

- `.scratch/replica-routing-baselines/issues/16-run-selective-hedging-validation.md`

#### Decisions and risks

Choosing a rate from holdout outcomes or silently borrowing a Budgeted LÆDGE
rate would change the experimental estimand. The ticket remains blocked until
the single missing parameter is explicitly frozen.

#### Next

Freeze one `planned_budget_rate` for all nine development candidates and the
selected holdout candidate, then claim this ticket and write red tests before
implementing the runner.

### Update: 2026-09-09 — Fixed nominal 5% budget envelope confirmed

Status: partial

#### Goal

Resolve the only execution blocker without changing the threshold, sample,
CRN, or call protocol.

#### Changed

- Frozen `planned_budget_rate = 0.09304612180547656` for all nine development
  candidates and the selected holdout candidate.
- Recorded provenance as the Budgeted LÆDGE v2 nominal 5% mapping.
- Recorded that the rate is only a planned capacity envelope: realized
  underfill is legal and must not be labeled achieved 5%.
- Changed ticket 16 to `claimed` with no blocker.

#### Verification

- ADR-0034 contains the fixed rate and underfill declaration.
- No scheduler call or artifact was generated before the runner tests.
- `git diff --check` passed.

#### Artifacts

- `docs/adr/0034-fault-aware-selective-hedging.md`
- `.scratch/replica-routing-baselines/issues/16-run-selective-hedging-validation.md`

#### Decisions and risks

The fixed rate is not recalibrated from development or holdout. Any realized
work delta is measured and reported independently.

#### Next

Write failing runner tests, implement the isolated development/holdout
execution path, and run development only after focused verification.

### Update: 2026-09-09 — Development completed; no candidate passed

Status: blocked

#### Goal

Run the frozen Selective Hedge development panel with the fixed nominal-5%
Budgeted LÆDGE capacity envelope, select at most one threshold pair, and
allow holdout only if the paired delta=0 gate passes.

#### Changed

- Added the isolated development runner with nine normalized-slack threshold
  candidates, three fixed reference arms, the fixed rate
  `0.09304612180547656`, and no holdout auto-start.
- Added an isolated two-episode preflight using its own namespace and macro
  seed; formal episode identities are not used by preflight.
- Corrected Selective Hedge audit accounting so eligibility diagnostics remain
  policy suppression reasons, while only an already-requested budget rejection
  enters the legacy executor suppression counters/events. This preserves the
  existing attribution invariants without changing dispatch, budget, CRN, or
  policy semantics.

#### Verification

- Real Red: the generated selective preflight failed with
  `hedge_request_lifecycle_counters_match` and
  `executor_suppression_counter_matches_events`; after the narrow accounting
  fix the same preflight passed.
- Focused runner suite: `9/9` passed.
- Affected suites (`test_selective_hedging_experiment`,
  `test_selective_hedging`, `test_budgeted_laedge`,
  `test_budgeted_laedge_campaign`, `test_budgeted_laedge_holdout`): `40/40`
  passed.
- Full suite: `713/713` passed in `206.866s`.
- Config check: `status: ok`; the existing informational single-domain
  failure-load finding remains unchanged.
- Formal development: `256/256` episodes, `3072/3072` scheduler calls,
  `0` failures, `114.08984390005935s`, `26.926147805855678` calls/s.
  All 256 episode rows completed and all 12 arms retained the same trace
  identity per episode.
- Development paired bootstrap: `4096` fixed replicates, `9` candidate
  comparisons, episode-cluster unit; no candidate satisfied the gate. The
  best point estimate for D/F CVaR95 was `-0.009598602243537` at
  `(theta_Regular, theta_Urgent)=(0.30,0.50)`, but its paired CI was
  `[-0.10049261328928293, 0.0716691583441405]` and its D/F miss-rate CI was
  `[0.00028203739931329696, 0.0074972640246124215]`.
- Across candidates, D/F CVaR95 CI upper endpoints were positive
  (`0.052841812380176226` through `0.14268445647185135`) and D/F miss-rate
  CI upper endpoints were positive (`0.003422845826322537` through
  `0.011278483462417482`), so the fail-closed `non_improving` result is
  deterministic and not a missing-statistics result.
- The fixed rate is recorded only as a work-capacity envelope. Candidate
  realized work increases ranged from approximately `0.1028%` to `0.4316%`
  relative to dynamic No-Hedge, which is legal underfill; no arm is labeled
  `achieved_5_percent`.

#### Artifacts

- `artifacts/selective-hedging-development-20260909-r1/summary.json`
- `artifacts/selective-hedging-development-20260909-r1/paired_statistics.json`
- `artifacts/selective-hedging-development-20260909-r1/episode_rows.jsonl`
- `artifacts/selective-hedging-development-20260909-r1/manifest.json`
- `artifacts/selective-hedging-development-20260909-r1/protocol.json`
- Artifact checks: `256` episode rows, `9` paired comparisons, `60` source
  bundle entries, no staging directory, and all claim flags false.

#### Decisions and risks

- Development status is `non_improving`; holdout is not started and its
  `4096` calls were not consumed. No candidate threshold pair is frozen for
  holdout.
- The fixed nominal-5% rate was not recalibrated from development results;
  realized underfill remains a valid observation.
- This is a finite-system development diagnosis only. It makes no optimal,
  best-response, Nash, or MFG claim.

#### Next

Do not start holdout under the current ADR-0034 gate; retain this
`non_improving` development artifact and decide separately whether to close
the validation ticket or revise the algorithm/protocol.

### Update: 2026-09-09 — Closed as development_no_candidate

Status: completed

#### Goal

Close the frozen validation ticket with the protocol-defined terminal outcome
after development produced no eligible Selective Hedge candidate.

#### Changed

- Classified the completed development result as
  `development_no_candidate`.
- Confirmed that the holdout panel is not pending work: its non-execution is
  the required fail-closed disposition when no development candidate passes.
- Changed ticket 16 from `blocked` to `resolved`.

#### Verification

- Confirmed `summary.json` reports `status: non_improving`, `3072` formal
  scheduler calls, `0` failures, and no selected arm.
- Confirmed the artifact contains `256` completed episode rows and `9`
  completed candidate comparisons.
- Confirmed no holdout artifact or holdout scheduler call was created.

#### Artifacts

- `artifacts/selective-hedging-development-20260909-r1/`
- `docs/adr/0034-fault-aware-selective-hedging.md`

#### Decisions and risks

The result is evidence that this Selective Hedge mechanism is non-improving on
the frozen development distribution. It is not a claim that Hedge is
universally ineffective, and it does not authorize threshold, budget, or
holdout changes within this ticket.

#### Next

Begin the separate pure-design feature
`reliability-aware-multi-replica-routing`; do not reuse this ticket for its
design or implementation.

## Answer

`development_no_candidate`: resolved. Holdout was correctly not started under
the frozen ADR-0034 selection gate.
