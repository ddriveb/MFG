# ADR-0035: Reliability-aware multi-Replica routing without Hedge

Status: Accepted (confirmed by the user on 2026-09-09)

## Context

The ADR-0034 Selective Hedge development panel ended with
`development_no_candidate`. The strongest positive mechanism observed so far
is work-conserving dynamic Primary routing, while adding duplicate work did
not improve the frozen development objective. That is a reason to isolate
routing before adding another Hedge or price mechanism; it is not a claim
that Hedge is universally ineffective.

Lazarus separates three controls: Expert replica allocation, fault-tolerant
placement, and a batch Token dispatcher. Its dispatcher balances the current
batch across available same-Expert replicas and preserves locality, but it
does not use per-Replica failure history, queueing delay, Token deadline, or
active duplicate execution. Therefore it is a relevant named routing
comparator, not the mechanism implemented by this ADR.

The research question is conditional on an upstream placement decision:
given several already placed copies of one Logical Expert, can a causal router
trade observed reliability against current congestion without changing the
Gate decision or duplicating a Token?

## Decision

### 1. Placement is exogenous

The first slice has one Logical Expert, eight physical Replicas, and four
failure domains. Replica IDs are the true integers `0..7`; the canonical map
is `domain(replica_id) = replica_id % 4`. The placement snapshot is immutable
within an episode. The router cannot create, remove, migrate, disable, or
reassign an Expert replica.

The Gate selects the Logical Expert before this controller runs. The routing
action selects only one physical Replica that holds the same Expert.

### 2. Execution is single-copy No-Hedge

Every Token has at most one live attempt and exactly one assigned Replica at a
time. No backup copy, cancellation policy, protection budget, or price is
part of this slice. `Primary` denotes the sole active execution, not a
different Replica type.

### 3. Reliability information is causal and heterogeneous

Replica availability is the conjunction of an idiosyncratic health path and
its failure-domain path. Policies observe only health transitions through the
current event time. They never observe future faults, future arrivals, latent
hazard regimes, service draws, or true remaining work.

The frozen history feature is a trailing-window Gamma-Poisson posterior mean:

`lambda_hat_c(t) = (alpha_h + N_c(t)) / (beta_h + E_c(t))`,

computed separately for a Replica and its domain. Their posterior-mean
hazards are added for the effective Replica hazard. A history-only burn-in
precedes the workload. The implementation ticket must freeze the numerical
window, prior, and burn-in values before code is written.

### 4. Failures create real routing feedback

A running attempt invalidated by failure loses its executed work and returns
to the global unassigned queue as a Replay. Queued but unstarted work returns
without a Replay charge. Displaced work is routed before same-time new
arrivals. At one timestamp, recovery and failure transitions are processed
before completion and arrival.

This feature may experience more than one reactive Replay when successive
selected Replicas fail, but it never has more than one live attempt. Attempt
IDs increase monotonically, fault paths end `UP`, and every episode runs to
complete drain. This is an isolated feature contract; historical engines with
an at-most-one-Replay invariant remain unchanged.

### 5. The finite comparison uses six named policies

The implementation must include:

1. deterministic uniform round-robin;
2. join-shortest-queue;
3. causal least observable expected work;
4. reliability-only routing;
5. risk-aware JSQ using queue exposure plus estimated failure risk;
6. a faithful deterministic adaptation of Lazarus Algorithm 1.

All policies remove `DOWN` Replicas before scoring and end ties with the
smallest Replica ID. If no Replica is eligible, work waits globally.

### 6. Lazarus retains its batch semantics

Lazarus Algorithm 1 operates on the entire current micro-batch, prioritizes
local Tokens, and distributes residual Tokens by available Replica-slot
capacity. Integral residual allocation uses the sorted eligible-Replica order
with a circular start at `batch_identity % r`, where `batch_identity` is the
first Token ID in the current batch. `DOWN` Replicas contribute zero slots.
Existing queued work is not migrated. This avoids fixed low-Replica-ID bias
while preserving deterministic locality-first semantics.

All six policies receive the same immutable micro-batches. The implementation
does not serialize only Lazarus Tokens into singleton arrivals, nor claim to
reproduce CUDA or all-to-all wall-clock performance.

### 7. Four mechanisms are tested separately

The scenario family contains:

- homogeneous reliability;
- two static high-risk Replicas in different domains;
- an unannounced switch in the high-risk Replica identities;
- correlated failure-domain shocks.

Exact rates, repair laws, deadlines, load, batch distribution, namespaces,
seeds, and policy coefficients are frozen in the implementation ticket before
any trace is generated. A nominally capacity-feasible condition and a
single-domain-loss diagnostic are reported separately.

### 8. Queue estimates cannot leak hidden work

JSQ uses observable running-plus-queued counts. Least observable expected
work uses the conditional mean residual of the frozen service distribution
given observable elapsed service age. It cannot inspect a sampled service
requirement. A true-remaining-work comparator, if later useful, must be
labelled as an oracle and is not one of the six formal policies.

### 9. The first execution is only a mechanism smoke

The implementation slice is limited to deterministic boundary fixtures and a
non-inferential smoke of at most 192 complete scheduler calls:

`4 scenarios * 8 episodes * 6 policies`.

It may validate physics, determinism, CRN identity, and non-degenerate routing
behavior. It may not select a winner or report formal confidence intervals,
optimality, regret, Nash, MFG, placement, or general Hedge conclusions. A
later experiment protocol must freeze independent development and holdout
evidence.

### 10. The engine exposes, but does not solve, the population problem

The finite engine records Token state, Replica health/queue/history state,
population occupancy, selected Replica, latency, misses, Replays, and work.
This is the interface for a later population model.

If decentralized Tokens minimize private cost under congestion prices, the
later object is an MFG. If one central router minimizes social cost, it is a
mean-field-control problem. Those solvers and claims are outside this ADR.

## Consequences

- Placement quality is conditioned on rather than credited to the router, so
  the experiment measures post-placement routing value.
- Reliability-only and congestion-only controls make the reliability versus
  queueing trade-off identifiable. Beating round-robin alone is insufficient.
- Lazarus supplies a relevant published system comparator without pretending
  that its batch dispatcher already solves history-aware online routing.
- Multiple reactive Replays make the new engine more general than the old
  two-Replica slice, but require an explicit attempt ledger and complete-drain
  audit.
- Eight Replicas are a finite system. Scaling Token count on this topology is
  not, by itself, an MFG limit.

## Rejected alternatives

- Optimizing placement and routing together: rejected because it prevents
  attribution of post-placement routing value.
- Adding Hedge immediately: rejected because it mixes duplicate-work effects
  with the reliability/congestion routing question.
- Giving policies true hazard rates or future fault paths: rejected as a
  non-causal oracle.
- Compressing every queue into one utilization scalar: rejected because FCFS
  order, running age, domain failures, and Replay pulses affect outcomes.
- Calling a centralized risk-aware heuristic an MFG: rejected because no
  population consistency or decentralized best-response condition is solved.

## Specification and source

The complete executable design contract is in
`.scratch/reliability-aware-multi-replica-routing/spec.md`.

Lazarus allocation, placement, and Algorithm 1 dispatcher semantics:
<https://arxiv.org/abs/2407.04656>.

## Next step

The bounded implementation ticket freezes all numerical scenario, estimator,
CRN, and smoke parameters before trace generation. It implements only the
isolated finite engine, six policies, and the non-inferential mechanism smoke;
it does not start a formal evaluation or an MFG solver.
