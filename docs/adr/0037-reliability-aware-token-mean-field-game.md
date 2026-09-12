# ADR-0037: Reliability-aware simultaneous Token mean-field game

Status: Accepted (confirmed by the user on 2026-09-10)

Date: 2026-09-10

## Numbering note

The requested ADR number `0036` is already occupied by the accepted
`0036-three-round-reliability-aware-routing-evaluation.md`.  This new ADR is
therefore recorded as the next available number, `0037`, without modifying or
overwriting the historical ADR-0036.

## Context

ADR-0035 established an isolated finite routing object: one Logical Expert,
eight fixed equivalent Replicas, causal reliability history, congestion-aware
single-copy routing, and complete failure/replay drain.  The corrected
three-round result remains finite-system baseline evidence only.  It does not
provide a population consistency condition or a representative Token's
response to a distribution of other Tokens.

The next question is whether routing choices made simultaneously by many
Tokens create a reliability--congestion feedback loop that can be represented
by a conditional population law and solved as a Token MFG.

## Decision (proposed)

### 1. Player and information structure

The player is an individual Token.  The Gate and placement are exogenous; an
action selects one same-Expert physical Replica and never changes the Logical
Expert.  At each micro-batch boundary all Tokens in the new/Replay decision
cohort use the same immutable pre-batch public state and the same predicted
population trend.  Tokens already waiting or running are not re-counted as new
routing decisions.  No Token sees another Token's concrete action in that
batch.  All actions are committed atomically before queue state changes.

This is different from a central router that scans Token IDs, observes the
first assignment, and lets later assignments react to it.  Such a sequential
dispatcher is retained only as a separately labelled centralized reference.

### 2. Common noise, public observation, and transient population state

Common health/failure history through time `t` is the common-noise filtration
`F_t^0`.  It contains only exogenous public fault and recovery history.  The
public observation filtration is distinct:

`G_t = F_t^0 + placement + current public health/queue state`
`      + current batch composition + published population trend`.

The open-system population path is represented by four separate objects:

- `mu_t`: distribution of currently active waiting/running Token states;
- `nu_t`: distribution of public Replica health, queue, service-age, hazard,
  domain, and locality states;
- `eta_t`: distribution of new-arrival or Replay Tokens requiring a routing
  decision at `t`;
- `x_t`: predicted action shares for `eta_t` only.

The local consistency and transition equations are

`x_t = Aggregate(pi_t, eta_t, nu_t)`

`(mu_{t+}, nu_{t+}) = Phi_t(mu_t, nu_t, eta_t, x_t, Z^0_{t+})`.

Completed Tokens leave `mu_t`; failures may create a later Replay cohort in
`eta_t`.  Already queued/running Tokens are not counted again in `eta_t`.
Common noise updates the conditioning history but does not reveal future
faults or future realized actions.  The tracked object is the transient path
`(mu_t,nu_t,eta_t,x_t,pi_t)_{t in [0,T]}`, not a static population fixed
point.  Its core transition is

`(mu_t, nu_t, eta_t) -> pi_t -> x_t -> simultaneous commit`
` -> (mu_{t+}, nu_{t+})`.

### 3. Finite action/state approximation

The initial solver uses a pre-registered finite state grid and observable
Replica-state buckets.  Buckets distinguish health, queue/work, service age,
individual and domain hazard history, domain/locality relation, and the
required Token-local fields.  A bucket is actionable only when it contains an
eligible Replica; within-bucket selection is stable and uniform.  Boundaries,
endpoint conventions, missing values, ordering, and schema fingerprints are
frozen before trace generation.

This is an approximation to the continuous state/action problem.  Empty or
unsupported cells are reported and cannot be silently merged or filled.

### 4. Physical scaling convention

Eight fixed Replicas with more and more Tokens is not a mean-field limit.  The
proposed family uses `K in {8,16,32,64}` equivalent Replicas, where `K` is a
multiple of four.  The batch release rate is fixed and the expected batch
size is `E[B_K]=b*K`; hence total Token arrival rate is
`lambda_K=lambda_0*K`, not `O(K^2)`.  `N_K` names the actual finite Token
count in one batch and is distinct from Replica count `K` and finite-`N`
deviation notation.

There are four fixed macro failure domains with `K/4` Replicas each, so a
domain shock remains common noise.  Total service capacity scales with `K`,
while per-Replica service ability and fault laws remain unchanged.  Offered
load is fixed and one Token's influence on the empirical state is intended to
be `O(1/K)`.  `K=8` must regress to the accepted finite engine; unavoidable
non-equivalences are explicit approximations, not hidden claims.

### 5. Private and social objectives

The No-Hedge private objective is

`C_i = latency_i + alpha_k deadline_miss_i + gamma_k replay_count_i`
`    + c_lost lost_work_i + congestion_charge_i`.

The charge is a frozen causal function of observable population load and is
zero in `unpriced_mfg`.  `priced_mfg` adds a named congestion-externality
transfer.  It is not automatically a market-clearing price or a requirement
for MFG existence.

Social cost excludes transfers unless a later protocol explicitly defines
them as resource costs:

`S = mean_i[latency_i + alpha_k deadline_miss_i + gamma_k replay_count_i`
`    + c_lost lost_work_i]`.

All terms are settled after complete physical drain and never use future
service draws, future faults, or hidden remaining work.

### 6. Separate response and consistency conditions

For the transient mean-field object, the response and consistency conditions
are expressed over time:

`pi_t` is a best response to the declared transient `(mu_t,nu_t,x_t)`;
`x_t = Aggregate(pi_t, eta_t, nu_t)`; and
`(mu_{t+},nu_{t+}) = Phi_t(mu_t,nu_t,eta_t,x_t,Z^0_{t+})`.

For a mean-field BR, the tagged Token treats the entire environment path
`(mu,nu,x)` as exogenous.  Its action changes neither that path nor the
aggregate trend.  This is a different estimand from a finite-`K` unilateral
deviation: there only the tagged action changes, other Tokens retain online
policy functions, and the complete finite queue/failure/Replay system is
rerun on paired CRN.  The finite deviation may change later queues, public
states, cohorts, other actions, and delays.  A realized baseline trajectory
must not be frozen.

Mean-field BR and finite-`K` deviation require separate record types,
fingerprints, confidence calculations, and labels.  One cannot substitute
the latter for the former.

Pure BR, logit/entropy-regularized response, finite-`N` epsilon-regret, exact
Nash, and MFG consistency residual are different quantities.  Softmax
convergence is reported only as `regularized_mfg_candidate` until all
population, finite-deviation, coverage, and uncertainty gates are met.

No existence or uniqueness theorem is assumed or implied.  Cycles,
multiple fixed points, insufficient support, and physical failures remain
formal outcomes.

## Why this is an MFG rather than mean-field control

An MFG describes decentralized Tokens optimizing private costs against an
aggregate state/trend they do not individually control.  A mean-field control
problem instead has one central controller choosing the population action to
minimize social cost.  This ADR's simultaneous Token policies, private
congestion charge, tagged deviation, and aggregate consistency are the former
object.  A social controller and sequential centralized LOEW are comparators,
not substitutes for the Token MFG.

## Consequences

- Simultaneous action semantics prevent a hidden central queue policy from
  being mistaken for a decentralized population game.
- The existing reliability-aware finite engine is reused as physics, while
  forward population estimation and response layers remain separate from it.
- Unpriced and priced models have distinct provenance and claims; fees are
  transfers in the social objective.
- The K-scaling and K=8 regression make finite evidence and mean-field
  approximation distinguishable.
- Unsupported bins and statistical failure stop the solver rather than
  creating artificial population mass or a favorable action.
- A successful finite diagnostic can still fail to support an MFG claim if
  predicted shares, finite deviations, or scaling do not close.

## Explicitly not accepted by this ADR

This proposal does not authorize production implementation, formal
qualification, or an experiment campaign until the implementation and
experiment tickets are separately executed.  It does not authorize Hedge,
placement optimization, price feedback beyond the frozen priced objective,
HJB/FPK, a Nash/MFG claim, or an optimality certificate.  The corrected
ADR-0035 routing result remains a finite baseline.

## References

- [ADR-0020: Token player and T1 online boundary](0020-restore-token-player-model.md)
- [ADR-0021: Token payoff and finite deviation semantics](0021-token-private-cost-and-deviation-semantics.md)
- [ADR-0022: Observable-bin continuation oracle](0022-token-conditional-continuation-oracle.md)
- [ADR-0035: finite reliability-aware routing](0035-reliability-aware-multi-replica-routing.md)
- [ADR-0036: accepted finite routing evaluation](0036-three-round-reliability-aware-routing-evaluation.md)
- [Feature specification](../../.scratch/reliability-aware-token-mfg/spec.md)
