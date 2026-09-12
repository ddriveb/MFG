# ADR-0034: Fault-aware, deadline-aware selective hedging

Status: Accepted (confirmed by the user on 2026-09-09)

## Context

The v2 Budgeted LÆDGE holdout shows that the main benefit comes from
work-conserving dynamic Primary routing. Positive Hedge budgets increase work,
wasted work, Protection-Storm intensity, and D/F deadline misses in the tested
finite environment. The existing generic Hedge rule therefore cannot be
treated as a useful default protection policy.

This ADR proposes a narrow next policy slice for causal finite-system
evaluation. It does not claim optimality, best response, Nash equilibrium,
mean-field consistency, or deployment readiness.

## Decision proposed

### 1. Policy boundary

The policy keeps the accepted Budgeted LÆDGE conservative-cancellation
executor and its work/accounting semantics. It changes only Hedge eligibility
and candidate ordering; it does not modify Primary/Replay execution, CRN
streams, fault ordering, queue physics, Reservation windows, or historical
engines.

At arrival, a Token may enter the same unassigned waiting queue as before.
When capacity is released, the dispatcher applies this order:

1. start an unserved Primary or Replay with no live copy;
2. only if none is startable, consider an eligible Hedge;
3. otherwise leave the capacity idle.

Each Token has at most one Hedge and one Replay. A Hedge is conservative:
running losers continue to completion; queued losers may be cancelled with
zero executed work.

### 2. Causal fault-aware eligibility

The first implementation candidate may start a Hedge only when all conditions
are true at the decision event:

- common state is `D`;
- the Token's live Primary is on Replica A;
- Replica A is the degraded/failing domain and Replica B is currently healthy
  and startable;
- the Token has no existing Hedge;
- no unserved Primary or Replay is startable;
- the Token is not already terminal or cancelled.

Hedges are forbidden in `F`. Hedges are disabled by default in `H` and `R`.
Any later relaxation of the H/R rule requires a separate protocol decision.
The policy never selects a Hedge whose destination is the unavailable or
failed Replica.

### 3. Deadline-aware candidate ordering

Among simultaneously eligible candidates, order by the observable tuple

`(deadline_slack, token_class_priority, arrival_time, token_id)`

where smaller `deadline_slack` is more urgent, `Urgent` precedes `Regular`,
and the final two fields provide stable deterministic tie-breaking.

Deadline slack is computed only from the Token's declared deadline and the
current event time. The policy may not inspect service requirements, remaining
work, future arrivals, future faults, future completions, or another Token's
private state.

The frozen absolute slack is

`s_i(t) = deadline_{class(i)} - (t - arrival_time_i)`.

The threshold uses the explicitly normalized, unitless slack

`bar_s_i(t) = s_i(t) / deadline_{class(i)}`.

A candidate is eligible only when `0 < bar_s_i(t) <= theta_class`; an already
expired Token is suppressed with the explicit reason `deadline_expired`.  The
development grid is the Cartesian product
`theta_Regular, theta_Urgent ∈ {0.15, 0.30, 0.50}`.  Exactly one pair is
selected on development episodes and then frozen for holdout; holdout results
may not change it.  The thresholds are therefore dimensionless fractions of
the class deadline, not absolute time values.

The class priority is `Urgent` before `Regular`.  Within equal class and slack,
the stable tie-break is `(arrival_time, token_id)`.  No minimum-slack or other
threshold may be inferred from a holdout outcome.

### 4. Work/value gate

The first implementation must expose a deterministic, auditable eligibility
gate rather than claim an unimplemented expected-value oracle. A Hedge can be
admitted only if its pre-registered causal gate passes and the existing
non-refundable work budget admits the fixed expected work charge.

The gate must report, per candidate, the observed reason for admission or
suppression. It must not read a future service draw or use a counterfactual
NIIN trajectory. A later predictor or price-guided score is a separate ADR
and may not be smuggled into this slice.

The development and holdout validation use the fixed causal budget envelope

`planned_budget_rate = 0.09304612180547656`.

This value is reused from the Budgeted LÆDGE v2 nominal 5% mapping. It is a
fixed work-capacity envelope, not a claim that Selective Hedge will achieve a
5% realized work increase. Underfill is a valid result and must be reported;
the Selective Hedge arm must not be labeled `achieved_5_percent` merely
because this planned rate is used.

### 5. Evaluation contract

Every comparison uses the same immutable episode and CRN streams. The new
policy is compared first against Budgeted LÆDGE `delta=0` and then against
NIIN conservative. Required metrics retain the existing definitions:

- D/F CVaR95 and deadline-miss rate;
- overall and phase latency tails;
- total, Hedge, Replay, and wasted work;
- Hedge launches and winner rate;
- Protection-Storm peak and duration;
- budget-window planned, committed, realized, overshoot, and suppression
  audits;
- complete invariant and drain counters.

No result may be called a Pareto improvement unless the comparison protocol
is frozen in a separate experiment ticket. A proposed success gate is:

- statistically supported D/F CVaR95 improvement over `delta=0`, or no
  deterioration with a supported D/F miss improvement;
- realized total-work increase no greater than the pre-registered 3% or 5%
  budget point;
- Protection-Storm peak no greater than 2.

For the selected candidate, qualification against the paired dynamic No-Hedge
baseline (`Budgeted LÆDGE delta=0`) is defined before execution as follows:

- at least one of D/F CVaR95 or D/F deadline-miss rate has a paired 95% CI
  strictly below zero for candidate minus baseline;
- the other metric has a paired 95% CI whose upper endpoint is no greater than
  zero;
- mean realized total-work increase is at most 5% relative to delta=0;
- Protection-Storm peak is at most 2.

Failure of any condition is reported as `non_improving`; it does not trigger
retuning or a second selection pass.  These are design gates for a future
evaluation, not results or guarantees.

### 6. Frozen development and holdout protocol

The later execution uses the existing episode/CRN contract
`episode_trace_fingerprint_v1`, with a complete transitive source/physics
fingerprint recorded before the first call.

- Development: 256 episodes, namespace
  `replica-routing-baselines:selective-hedging:v1:development`, macro seed
  `20260916`.
- Holdout: 1,024 fresh episodes, namespace
  `replica-routing-baselines:selective-hedging:v1:holdout`, macro seed
  `20260917`.
- The namespaces and episode identities are disjoint from each other and all
  existing Budgeted LÆDGE, cancellation, Token, and MFG runs.
- Every arm in one episode shares the exact workload, fault timeline, and
  attempt-key CRN streams.  The episode index is the statistical cluster.
- Development runs three fixed references (`fixed-dispatcher:no-hedge`,
  `niin:conservative`, and `budgeted-laedge:delta=0%`) plus all nine
  `(q_Regular, q_Urgent)` grid candidates: `256 × 12 = 3,072` calls.
- Holdout runs the same three references plus the one frozen candidate:
  `1,024 × 4 = 4,096` calls.
- Calls are hard-counted; failures are fail-closed, with no retry,
  supplementation, deletion, or result-driven sample change.
- All nine development threshold candidates and the one selected holdout
  candidate use exactly the fixed planned budget rate above. Neither
  development nor holdout may recalibrate it.
- Paired episode-cluster bootstrap uses the existing frozen 4,096-replicate
  statistics protocol and consumes no scheduler calls.

The development and holdout metrics retain the Budgeted LÆDGE definitions:
overall and D/F/H/R latency tails, D/F CVaR95, D/F deadline miss, Replay,
total/Hedge/wasted work, Hedge launches and winner rate, suppression,
Protection-Storm, drain duration, and planned/committed/realized/overshoot
budget audits.  Development selection is the sole grid-selection step.

### 7. Explicit non-goals

This ADR does not add prices, online learning, continuation prediction,
population feedback, SoftBR, best response, regret, MFG, new common paths,
or a claim that selective hedging is superior. The existing Budgeted LÆDGE
holdout artifact remains immutable and is not reinterpreted.

## Consequences and risks

The proposed rule directly tests whether aligning Hedge direction with the
degraded domain and deadline slack removes the harmful generic Hedge
externality. It may produce no useful Hedge activity; that is a valid
finite-system outcome. Dynamic Primary routing remains the baseline benefit
and must be reported separately from any Hedge effect.

The policy is intentionally conservative and causal. The slack definition,
threshold grid, sample identities, and call budgets are now frozen; no
experiment may start before the ADR is explicitly accepted and the
implementation passes its focused contract tests.
