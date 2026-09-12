# ADR-0022: Token observable-bin continuation cost oracle (T3A)

Status: Accepted (confirmed on 2026-09-07)

Date: 2026-09-07

## Context and scope

ADR-0021 completed the finite Token pathwise layer: one requested-action
intervention can be rerun on the same immutable physical trace, while every
other Token continues to use its policy function online.  A pathwise cost
difference is not an estimate of which action is better under the information
available at arrival.  Exact conditioning on the full observation is not
claimed here either.  T3A adds only a pre-registered observable-bin
continuation estimator:

```text
Q_m(u | B(O)=b, S)
  = E[C_m | B(O)=b, S, requested_action=u],  u in {N, D, I}
```

where `m` is exactly one of the three independent ADR-0021 cost models.  This
ADR is a design contract for a pure Monte Carlo rollout oracle.  It does not
authorize production implementation, a training pipeline, a price loop, a
best-response calculation, a regret calculation, a forward mean-field model,
or an experiment campaign.  The accepted T1/T2 contracts and the historical
static path remain unchanged.

## Decision status and activation gate

This ADR was accepted by the user on 2026-09-07.  Acceptance authorized the
separate T3A implementation ticket only; it did not authorize a statistical
pilot or cost campaign.  The implementation-validation smoke remains distinct
from the occupancy-calibration and validation protocol proposed in ADR-0023.

## 1. Pre-registered selection and observable binning

Before any episode is generated, freeze both a target-selection rule `S` and a
deterministic binning function `B`.  `S` includes the observation-time rule,
eligibility predicate, tie/order convention, and the at-most-one-target rule.
For example, `S` may scan arrivals causally from a pre-registered time `t*`
and select the first eligible Token after `t*`.  It must decide at each
arrival using only the prefix observed at that arrival; it cannot inspect later
arrivals, faults, work, completions, actions, or outcomes.

`B` is frozen with its complete schema: field selection, categorical levels,
numeric boundaries, endpoint conventions, missing-value handling, bin ordering,
and schema version/fingerprint.  Its boundaries are not learned, widened,
merged, or selected from sampled costs.  Every selected target maps to one
canonical bin ID `b`; an out-of-domain or ambiguous observation is
`missing_bin`, not silently assigned to a neighboring bin.

The estimand is explicitly tied to the selection rule:

```text
Q_m(u | B(O)=b, S)
  = E[C_m^u | S selects a target with B(O_e)=b,
       requested_action=u]
```

This is not exact conditioning on `O`.  The target-selection rule, bin schema,
and their fingerprints are part of the estimand provenance.

## 2. Independent episode generation and target observation

`ContinuationFactory` receives only the frozen episode index, split/namespace,
protocol, and declared exogenous-generation parameters.  It returns one
independent complete *exogenous* episode input: arrival/work/fault streams,
identity projections, and trace fingerprints.  It does not receive a raw
observation, a target, a bin, or a prefix to match, and it does not filter or
resample episodes to obtain a desired bin.

The pre-registered episode count and namespace are fixed before generation.
Each episode key is derived from the T3A namespace, macro seed, split, episode
index, and protocol version; it is not derived from a global Token ID alone.
Different episodes have independent draw streams and may have different
arrival times, queue states, raw observations, and prefix fingerprints.

Run `S` causally on each generated episode.  If `S` selects a target, construct
that episode's immutable `ContinuationObservation O_e` at the target arrival,
immediately before the policy request, from the T1 `TokenObservation`.  Its
canonical payload contains exactly the fields visible to that policy: Token
identity/class, arrival time, current phase and age, primary replica, local
queue/running snapshot, observed history, reservation window/cap/balance,
public price, and protocol/observation-schema versions.  Its observation
fingerprint is computed before any candidate branch or future draw is
inspected.  Then compute `b_e=B(O_e)`.

The observation must not contain true service requirement or remaining work,
future arrivals, future faults, future policy actions, future completions, or
any hidden queue/work timeline.  Observation tuple order is canonical and
stable; no future policy result is cached across a branch.  A targetless
episode records `missing_target`, consumes its pre-registered episode slot,
and is never automatically replaced.  An episode with an invalid or
out-of-domain observation records `missing_bin` and is likewise not replaced.

Only episodes with the same frozen `bin_schema_fingerprint` and `bin_id` are
aggregated.  Cross-episode raw observations, arrival times, queue snapshots,
observation fingerprints, and prefixes need not match.

Calibration and validation are disjoint libraries, namespaces, fingerprints,
and seed ranges.  An episode fingerprint may occur in only one split; overlap,
missing identity fields, duplicate indices, or an out-of-order library is a
fail-closed input error.  Sample counts and split membership are frozen before
execution; neither is changed because an estimate is noisy.

There is no exact-prefix matching requirement across episodes, no conditional
fallback, and no automatic supplemental draw.  A failed, targetless, or
missing-bin episode consumes its declared attempt and remains in the audit.

## 3. Paired N/D/I physical reruns

For each valid independent episode, run three fresh complete branches in canonical
action order `N, D, I`.  The three branches use the identical continuation
draws, fault timeline, service/work streams, arrival order, immutable protocol
parameters, and exogenous quote.  This is paired CRN within one episode;
it is not a reuse of a realized action or queue trajectory.  Different
episode indices use independent namespaces and are not treated as paired
observations with one another.

Within one episode, the target identity, `O_e`, observation fingerprint,
`b_e`, and complete physical prefix through the target request must be exactly
the same in all three branches.  Across different episodes none of those raw
prefix fields or fingerprints need to match; only the declared bin key is
shared for aggregation.

The N/D/I panel is all-or-nothing.  If any one action branch fails, has a
non-finite or identity-mismatched result, or fails complete physical drain,
the entire episode panel is `failed` and none of its three action costs enters
the numeric estimate.  The failed episode is retained with its reason and
attempted calls.  No branch is retried and no replacement episode is
automatically sampled.

At the target arrival, the T2 runner calls the base policy once, then replaces
only the requested action with the candidate.  Every other Token is created
from the policy factory for that branch and chooses online from the changed
observations.  All branches therefore rerun the complete queue, Reservation,
fault, timer, Replay, winner/cancellation, running-loser and drain semantics.
No opponent action list, completion time, queue path, or realized baseline
outcome is replayed.

The target policy call and all non-target policy calls remain part of the
causal branch.  A same-action N/D/I branch is an identity check, not a
deduplicated baseline.  Candidate action labels are not random-stream keys.

## 4. Reservation and action outcome

The actual T1 ReservationLedger is applied in every branch.  A requested D or
I that is ineligible or unaffordable is recorded as requested but applied as
N, with the actual suppression, charge, balance, and physical outcome carried
into the scorer.  It is never scored as a hypothetical admitted Hedge and is
never allowed to borrow budget.  An admitted charge is non-refundable,
including when a Delayed timer is later voided.  The prediction therefore
includes the possibility that D/I has the same physical and payoff outcome as
N while retaining its correctly audited requested/applied fields.

## 5. Separate cost-model estimands

Run and aggregate the three ADR-0021 models independently:

- `token_initial_price_v0`;
- `token_runtime_adr0009_v1`;
- `token_extended_reservation_v1`.

For each model `m`, action `u`, frozen observable bin `b`, and frozen selection
rule `S`, the primary sample is the completed Token payoff `C_m` from each
complete episode panel selected by `S` with `b_e=b`.  Do
not pool model IDs, parameter schemas, price bases, or raw costs across
models, even when two parameter settings happen to produce equal numbers.
The policy receives the same declared quote and model-specific provenance that
the scorer uses; retrospective rescoring is not a substitute for a rerun.

## 6. Oracle output contract

The proposed immutable records are:

- `EpisodeRecord`: split, episode index/namespace, exogenous trace and fault
  fingerprints, target-selection rule fingerprint, selection status, and
  `missing_target`/`missing_bin` reason when applicable;
- `ContinuationObservation`: this episode's canonical observable payload,
  observation fingerprint, target identity, bin ID/schema fingerprint, and
  protocol provenance;
- `ContinuationRequest`: split, episode index, target-selection and bin
  fingerprints, requested action, model ID, and input fingerprints;
- `ContinuationRollout`: branch status, applied action/charge, payoff payload,
  physical and trace fingerprints, and failure reason if incomplete;
- `ConditionalCostEstimate`: model ID, bin ID, requested action, completed
  episode count, effective sample count, mean, sample standard error,
  missing/status code, and calibration/validation provenance;
- `ObservableBinContinuationResult`: one row for each model/action pair within
  a bin and selection rule, bin/selection/split fingerprints,
  required/completed/failed/missing-target episode counts, and global audit
  counters.

The only numeric estimate fields are action-conditioned means and their
standard errors, with effective sample counts and explicit missing status.
`effective_n` counts independent complete episode panels, not the three paired
action branches.  A paired delta may be retained as an audit diagnostic, but it
is not an action selector or an expected-regret field.  Missing, incomplete,
failed, non-finite, identity-mismatched, or insufficient cohorts remain
non-numeric and fail closed.  A failed episode cannot contribute one or two
surviving action rows.

The evidence label is:

```text
observable_bin_conditional_cost_oracle
```

The result must carry `claims_best_response=False`, `claims_regret=False`,
`claims_nash=False`, and `claims_mfg=False`.  It must not contain `best_action`,
`best_response`, `regret`, `epsilon_nash`, `equilibrium`, or an implicit
`argmin` decision.  Equal or lower estimated `Q` values may be displayed as
diagnostic comparisons only; choosing an action is a later slice.

## 7. Statistical and execution boundary

The first implementation uses ordinary Monte Carlo sample means and sample
standard errors over independent complete episode panels within each
pre-registered `(S, bin)` cell.  It does not claim exact-observation
conditioning,
simultaneous confidence bounds, conditional-law correctness, or coverage
guarantees.  A later implementation ticket must freeze the requested episode
count per selection rule/bin cell, target-selection rule, minimum complete
count, handling of failed and targetless episodes, and any precision diagnostic
before execution.  Adaptive
sampling, result-driven bin changes, silent failure filtering, automatic
supplemental draws, and validation reuse are prohibited.

Every failed branch increments an auditable attempted-call counter and makes
its whole episode panel failed.  No retry is automatic.  A required `(S, bin)`
cell with too few complete independent episodes is `insufficient`, not a
favorable estimate; missing-target episodes count against the frozen episode
allocation.  The N/D/I pairing reduces variance of comparisons but does not
turn the branches into independent actions or permit an action-selection
claim.  Bins are never merged to repair an insufficient cell.

## 8. Required design tests before implementation

The future T3A implementation ticket must write failing tests before code for
at least these cases:

1. The factory receives only a frozen episode index/namespace/protocol and
   generates an independent exogenous episode; it never receives a raw
   observation or bin to match.
2. The same causal target prefix produces the same episode-local observation
   and pre-registered bin, with no future work, arrival, fault, or outcome
   fields.
3. Numeric bin boundaries, endpoint rules, and bin ordering are frozen before
   sample generation and cannot be changed by observed costs.
4. Different episodes may have different raw observations, arrivals, queues,
   and prefix fingerprints while aggregating into the same bin.
5. N/D/I in one episode share exactly one target prefix fingerprint and differ
   only by the target requested-action override; separate episode indices have
   independent namespace keys.
6. One episode can select at most one target, and target selection cannot read
   future arrivals, faults, work, completions, or outcomes.
7. A targetless episode records `missing_target`, consumes its slot, and is not
   automatically replaced.
8. Non-target policy functions are called online again and can react to the
   changed queue/budget; no action trajectory is cached.
9. Ineligible or unaffordable D/I projects to applied N, charges zero, and is
   scored as the actual N lifecycle.
10. The three cost models produce separate estimates and provenance, including
   equal numeric costs with distinct model IDs.
11. Complete loser drain, Replay, timer voiding, running work, and failure work
   reach the payoff scorer in every branch.
12. Missing bins, incomplete rollouts, duplicate episode keys, split
   overlap, invalid fingerprints, and non-finite costs fail closed.
13. If one N/D/I branch fails, the complete episode panel fails; no branch is
   retained and no automatic replacement episode is sampled.
14. Reusing one observation across multiple fresh rollouts does not reuse
   endogenous engine, ReservationLedger, policy state, or queue state.
15. Calibration and validation outputs have disjoint sample fingerprints and
   deterministic ordering; no validation statistic affects calibration.
16. Repeated execution with the same frozen inputs is byte-for-byte stable and
   produces only the oracle label and non-claim flags.

## 9. Explicitly deferred work

This ADR does not define or authorize posted-price pacing, endogenous price
feedback, Soft BR, best-response selection, regret or epsilon-Nash tests,
population forward evolution, conditional mean-field closure, HJB/FPK,
many-server scaling, Nash/MFG claims, or any formal calibration/validation
campaign.  Ticket 06's resolved finite evolution/closure derivation remains a
theory reference; it is not activated by this oracle design.

## References

- [ADR-0020](0020-restore-token-player-model.md): accepted T1 online boundary.
- [ADR-0021](0021-token-private-cost-and-deviation-semantics.md): accepted T2
  payoff, Reservation and complete pathwise rerun semantics.
- [T2 implementation ticket](../../.scratch/token-mfg-restoration/issues/05-implement-token-payoff-and-deviations.md).
- [finite evolution/closure derivation](../../.scratch/token-mfg-restoration/issues/06-derive-token-evolution-and-closure.md).
