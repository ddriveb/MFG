# Reliability-aware multi-Replica routing

Status: Accepted design, 2026-09-09.
Design authority: Accepted ADR-0035. Only the bounded implementation and
non-inferential smoke described in ticket 02 are activated.

## 1. Research question

Given an externally supplied placement of multiple physical replicas of one
Logical Expert, can a causal Token router use observed Replica failure history
and current congestion to improve latency, deadline attainment, and recovery
cost without duplicating Token execution?

The first slice is a finite-system routing reference:

`external placement -> one Primary choice -> queue/failure/retry dynamics`.

It does not optimize placement, launch Hedge copies, post prices, or claim a
mean-field equilibrium. The intended research sequence is:

`finite routing physics -> strong routing baselines -> history value`
`-> MFG/MFC coordination -> optional selective Hedge`.

## 2. Separation of control planes

The upstream placement controller supplies a placement snapshot `P` containing
Replica identities, node identities, failure-domain membership, capacities,
and the set of currently provisioned replicas. This feature neither explains
nor optimizes why those replicas were provisioned.

The Gate has already selected the Logical Expert. A routing action selects one
physical Replica holding the same Expert parameters. It never changes the
Logical Expert, model output, or Gate probability.

Placement may change between routing epochs in a future integration. Within
one finite episode in this slice, the placement map is immutable. A policy is
conditioned on `P`; it cannot migrate, add, remove, or disable a Replica.

## 3. Frozen topology contract

1. There is one Logical Expert with eight physical Replicas, identified by the
   true integers `0..7`.
2. There are four failure domains `0..3`. The canonical placement is
   `domain(replica_id) = replica_id % 4`, so each domain contains two Replicas.
3. Replica `0` and Replica `1` are in different domains. Scenarios that mark
   exactly two Replicas high-risk use these identities, separating individual
   reliability heterogeneity from a single correlated-domain effect.
4. Every Replica has normalized nominal service speed `1.0`. Scenario-specific
   health may reduce the currently available speed to zero, but capacity is not
   adjusted by the routing policy.
5. One FCFS queue belongs to each Replica. There is also one global unassigned
   waiting queue for new Tokens and work displaced by failures.
6. Every Token has a stable `token_id`, Token class, deadline, arrival time,
   ingress rank, and potential service stream. The ingress rank is required to
   reproduce the locality preference in Lazarus Algorithm 1; it is not a Gate
   choice or a preferred Expert.

The implementation ticket must not replace this topology with a Primary plus
seven semantically different Backups. All eight are equivalent parameter
replicas; `Primary` means the single Replica selected for one execution.

## 4. Workload and decision epochs

The finite reference uses immutable micro-batches. A batch release time and
its member Token IDs are generated before any policy runs. All policies see the
same batches in the same order.

At one release time:

1. all state transitions and failure invalidations at that time are processed;
2. valid completions at that time are processed;
3. displaced work already waiting is considered before the new batch;
4. new Tokens become observable together;
5. the policy computes assignments using no later batch or future event;
6. assigned work joins per-Replica FCFS queues, then idle Replicas start work.

Online policies process simultaneous new Tokens in increasing `token_id` order
and observe assignments made earlier in that same order. Lazarus Algorithm 1
processes the entire current batch, as the published algorithm requires. This
difference is part of the named policy semantics and must be reported; it must
not be hidden by holding only one comparator's Tokens for an artificial batch.

## 5. Causal observation

At decision time `t`, a router may observe, for each Replica `j`:

- `replica_id`, node ID, and failure-domain ID from placement `P`;
- current `UP` or `DOWN` state and current observable service speed;
- number of running attempts (`0` or `1`) and queued attempts;
- elapsed wall time of the running attempt, if any;
- stable arrival and Token IDs of queued work when required for FCFS audit;
- observed individual and domain failure/recovery transitions no later than
  `t`;
- the causal hazard estimates defined in Section 7.

The policy may observe its Token class, arrival time, current age, remaining
deadline slack, and ingress rank.

It may not observe future arrivals, future failure or recovery events, latent
hazard regime changes, sampled service requirements, true remaining work,
counterfactual assignments, or another policy's realized trajectory.

## 6. Failure, retry, and event semantics

### 6.1 Health processes

Replica availability is the conjunction of an idiosyncratic health process and
its failure-domain health process. Both are exogenous, immutable alternating
`UP`/`DOWN` paths generated before policy evaluation. A policy sees only the
path prefix through the current time.

All intervals are half-open: a component is unavailable on
`[failure_time, recovery_time)` and available at `recovery_time`.

The event order at one timestamp is:

1. recovery transitions, ordered by domain then Replica ID;
2. failure transitions, ordered by domain then Replica ID;
3. invalidation and requeue caused by those failures;
4. valid completions;
5. batch arrival;
6. routing and FCFS start.

Because failure transitions precede completion, an attempt whose completion is
exactly at its failure time fails. Applying recovery before failure makes a
same-time new failure the final state. Generated formal scenarios should avoid
such coincidences except for deterministic boundary tests.

### 6.2 Failure of assigned work

- A `DOWN` Replica is ineligible for new assignments and cannot start work.
- A running attempt on a newly failed Replica is invalidated. Its executed work
  is retained as lost work, the Token's `replay_count` increments, and the Token
  enters the global unassigned queue at the same timestamp.
- A queued but unstarted attempt on a newly failed Replica has executed no
  work. It returns to the global unassigned queue without incrementing
  `replay_count`.
- Displaced running work is ordered before displaced queued work; within each
  group use original FCFS order. All displaced work precedes same-time new
  arrivals.
- A recovered Replica returns with an empty local queue and becomes eligible
  before same-time routing.

Unlike the historical two-Replica v1 engines, this isolated feature permits
more than one reactive Replay when different selected Replicas fail. It still
permits exactly one live attempt per Token and never creates active redundancy.
Attempt IDs increase monotonically and are keyed independently. Failure paths
have a finite last transition and finish `UP`, so complete drain is defined.
The historical at-most-one-Replay engines and artifacts are not modified.

## 7. Observable failure-history estimator

The first slice uses a frozen, non-learned Gamma-Poisson posterior-mean hazard
estimator. It exists to make history information explicit and reproducible,
not to claim optimal failure prediction.

For component `c` (an individual Replica or failure domain), define over the
trailing window `(t-W_h, t]`:

- `N_c(t)`: observed `UP -> DOWN` transitions;
- `E_c(t)`: wall time for which the component was observed `UP`.

With frozen positive prior parameters `alpha_h` and `beta_h`,

`lambda_hat_c(t) = (alpha_h + N_c(t)) / (beta_h + E_c(t))`.

The effective hazard estimate of Replica `j` is additive:

`h_hat_j(t) = lambda_hat_replica_j(t) + lambda_hat_domain(j)(t)`.

The episode includes a history-only burn-in interval before time zero. It
contains no Token workload but its observed health transitions are available
to all policies. The implementation ticket must freeze `W_h`, burn-in length,
`alpha_h`, and `beta_h` before generating a trace.

For deployable queue-cost estimates, define:

- `n_j(t)`: running plus queued attempts currently assigned to Replica `j`;
- `T_hat_j(t) = (n_j(t) + 1) / current_speed_j` for an arriving unit-mean Token;
- `r_hat_j(t) = 1 - exp(-h_hat_j(t) * T_hat_j(t))`.

`T_hat` is an observable mean-work proxy, not a prediction using the selected
Token's future service draw. A `DOWN` Replica has no finite routing score and
is excluded before scoring.

## 8. Exogenous scenario family

Every scenario uses the topology and observation contract above. Exact hazard
levels, repair-duration distributions, arrival load, batch size, time horizon,
deadline values, and stream namespace are frozen in the implementation ticket.
The semantics below may not be reopened there.

### S0 - homogeneous reliability

All eight Replicas have the same idiosyncratic failure law and all four domains
have the same common-shock law. This is the negative control: a history-aware
router should not manufacture persistent Replica preference from identity.

### S1 - two static high-risk Replicas

Replicas `0` and `1` have a higher idiosyncratic hazard than Replicas `2..7`.
Capacities remain identical. Policies do not receive the latent rates; they
must infer differences from the observed burn-in and online history.

### S2 - non-stationary individual risk

The latent high-risk identities change once at an unannounced time. The old
pair and new pair are fixed in the trace protocol, but neither the switch time
nor new rates are exposed to policies. This measures estimator lag and prevents
a static Replica blacklist from being mislabeled history-aware routing.

### S3 - correlated domain failure

All Replicas retain individual failure processes, and domain-level common
shocks simultaneously make both Replicas in a domain unavailable. Individual
and domain histories are separately observable and separately estimated.

Each scenario must include a capacity-feasible nominal load and a declared
single-domain-loss diagnostic. Base infeasibility, if intentionally tested, is
reported separately and cannot be counted as a routing failure.

## 9. Deterministic routing policies

All tie-breaks end with the smallest `replica_id`. Every policy first removes
currently `DOWN` Replicas. If none is eligible while work is waiting, the work
remains in the global queue until a recovery; no policy may invent capacity.

### 9.1 Uniform round-robin (`uniform_rr`)

Sort eligible Replica IDs. Token `k` chooses index `k mod eligible_count`.
There is no mutable cursor and no reliability input.

### 9.2 Join shortest queue (`jsq`)

Choose the minimum tuple `(n_j, replica_id)`, where `n_j` includes one running
attempt plus queued attempts. It uses current congestion but no history.

### 9.3 Least observable expected work (`loew`)

For queued attempts use unit mean work. For a running attempt with observed
elapsed service age `a`, use the conditional mean residual of the frozen
service distribution, `E[S-a | S>a]`. Divide the sum by current speed and
choose `(estimated_clear_time, n_j, replica_id)`.

This is the causal comparator corresponding to least unfinished work. An
optional `true_remaining_work_oracle` may be reported only as an oracle and is
not one of the six formal finite policies.

### 9.4 Reliability-only (`reliability_only`)

Choose `(r_hat_j, T_hat_j, replica_id)`. Queue enters only through the failure
exposure horizon `T_hat`; the primary ordering is estimated failure risk.

### 9.5 Risk-aware JSQ (`risk_aware_jsq`)

For Token class `k`, choose the minimum:

`T_hat_j + gamma_k * r_hat_j`, then `(n_j, replica_id)`.

The class-specific replay-risk penalties are frozen before implementation and
shared with all future extensions. This is a causal finite heuristic, not an
MFG equilibrium or an optimal policy.

### 9.6 Lazarus Algorithm 1 (`lazarus_algorithm1`)

This comparator reproduces the published batch semantics rather than applying
an unrelated online heuristic:

1. gather the current batch's per-ingress Token count for the Logical Expert;
2. let `t` be the total current-batch Tokens and `r` the count of eligible
   Replica slots;
3. set the target per slot to `p=t/r`;
4. derive each rank's target capacity from its eligible Replica-slot count;
5. prioritize Tokens local to their ingress rank up to that capacity;
6. distribute the remaining current-batch Tokens across residual capacities;
7. use integral remainder allocation in the sorted eligible-Replica order,
   circularly starting at `batch_identity % r`, where `batch_identity` is the
   first Token ID in the current simultaneous batch. Locality-first is applied
   before residual Tokens use that circular order; the adaptation is integral,
   deterministic, and does not privilege low Replica IDs across batches.

`DOWN` replicas contribute zero slots. Work already queued before the batch is
not migrated by this dispatcher. The paper's CUDA/all-to-all performance is
out of scope; the semantic adaptation and source revision are provenance.

## 10. Workload and CRN identity

The workload generator owns independent SHA-256-derived streams for:

- batch release times and batch sizes;
- Token class and ingress rank;
- potential service work keyed by
  `(episode, token_id, replica_id, attempt_id)`;
- individual failure and repair paths keyed by Replica ID;
- domain-shock and recovery paths keyed by domain ID.

Policy names, assignments, queue states, and outcomes never enter exogenous
stream keys. All policies receive the same immutable workload and fault paths.
Potential service work is read only for the selected Replica and attempt; the
existence of other draws does not make them observable.

Formal comparison uses paired episodes. A trace fingerprint covers the full
exogenous stream identities and resolved scenario parameters, while a policy
fingerprint covers code/source provenance and policy parameters. Duplicate,
missing, or mismatched fingerprints fail closed.

## 11. Required physical and scientific outputs

### 11.1 System and Token metrics

- generated, completed, and terminally failed Tokens;
- overall and per-class latency mean, P50, P95, P99, and CVaR95;
- per-class deadline-miss rate;
- Replay count/rate and distribution of Replays per Token;
- nominal, executed, lost-on-failure, and total work;
- last arrival, drain end, and drain duration.

### 11.2 Per-Replica audit

- assignments, starts, completions, running failures, and queued displacements;
- time-average and peak queue depth;
- busy time, available time, and utilization conditional on availability;
- individual/domain failures and recoveries;
- estimated hazard at decisions and latent true hazard for evaluation only;
- fraction of Regular/Urgent traffic and realized failure exposure.

### 11.3 Routing-distribution metrics

- assignment share per Replica and per failure domain;
- maximum share, coefficient of variation, and normalized entropy;
- time spent with no eligible Replica;
- S2 reaction lag after the latent risk switch;
- decision-weighted hazard-estimation MAE and Brier score for failure before
  the causal `T_hat` horizon.

### 11.4 Invariants

- every completed Token has exactly one winning execution;
- at most one live attempt exists per Token at any time;
- no assignment or start occurs on a `DOWN` Replica;
- executed work equals completed plus lost work under the attempt ledger;
- FCFS and global displacement ordering are preserved;
- all exogenous fingerprints agree across policies;
- all Tokens complete after the finite fault horizon and complete drain.

Any false invariant makes that policy/episode physically unavailable. Failed
episodes are not retried, removed, or supplemented.

## 12. Statistical and claim boundary

The first implementation ticket is authorized only for deterministic unit
fixtures plus a non-inferential mechanism smoke of at most:

`4 scenarios * 8 episodes * 6 policies = 192 complete scheduler calls`.

The smoke uses a dedicated namespace and cannot be reused as development or
holdout evidence. It may establish determinism, completion, invariants, and
non-degenerate history/congestion effects only.

A later experiment ADR must independently freeze development/holdout sizes,
seeds, namespaces, timing budget, paired estimands, simultaneous comparisons,
and success criteria. Episode is the statistical cluster. Token-level samples
must not be treated as independent replications.

The finite slice may report that a history-aware heuristic improves selected
metrics under a named scenario. It may not claim optimal routing, Nash,
mean-field equilibrium, universal fault prediction, or placement superiority.

## 13. Future MFG/MFC interface, not implementation scope

The finite engine must expose enough causal state for a later model without
embedding a solver:

- Token state: class, age/slack, ingress, retry count;
- Replica state: health, queue proxy, service age, `h_hat`, domain;
- population state: assignment/occupancy mass by Replica and health/history
  bin;
- action: one eligible Replica ID;
- transition output: latency, miss, failure/Replay, and executed/lost work.

If decentralized Tokens minimize private cost under congestion prices, the
later object is an MFG. If one central controller minimizes social cost, the
later object is mean-field control. The finite physical engine is shared, but
the two optimization problems and their claims must remain distinct.

## 14. Explicit non-goals

- Expert placement, allocation, migration, or failure-domain optimization;
- Hedge, duplicate execution, cancellation, protection budget, or Storm;
- learned failure prediction;
- price updates, best response, equilibrium, HJB/FPK, or MFG/MFC solver;
- cross-Expert shared capacity, training correctness, GPU kernels, or
  all-to-all wall-clock modeling;
- changing historical two-Replica engines, configurations, or artifacts.

## 15. Implementation sequence

1. Implement immutable eight-Replica workload/fault identities and the causal
   Gamma-Poisson history projection.
2. Implement the isolated single-copy event engine and deterministic boundary
   fixtures.
3. Implement the six routers, including the batched Lazarus adaptation.
4. Implement metrics, invariant audit, provenance, and the bounded 192-call
   mechanism smoke.
5. Stop for review. Do not start formal evaluation or MFG work.

## 16. Primary reference

Lazarus allocation, placement, and Algorithm 1 dispatcher semantics:
<https://arxiv.org/abs/2407.04656>.
