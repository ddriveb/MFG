# Reliability-aware simultaneous Token MFG

Status: Proposed design, 2026-09-10.
Design authority: Proposed ADR-0037.  This feature is design-only until the
ADR is accepted and the implementation tickets are activated.

## 1. Research question and boundary

The fixed upstream placement and Gate expose one Logical Expert with several
equivalent physical Replicas.  At a common micro-batch boundary, Tokens must
choose which Replica of that same Expert will execute their single Primary
copy.  The research question is whether a decentralized Token policy can use
causal reliability and congestion information, together with an aggregate
population routing trend, without observing the concrete choices of other
Tokens.

The causal loop is:

`placement and Gate -> simultaneous Token routing -> Replica congestion and`
`reliability outcomes -> updated population state and routing trend`.

This is a Token-player problem.  It is not a central dispatcher that sees the
batch's realized actions and assigns the batch sequentially.  It does not
change the Logical Expert selected by the Gate, optimize placement, create
Hedge copies, or introduce a price before the priced model is explicitly
selected.

The first implementation is a finite, discrete-state approximation that
exposes the interfaces needed for a later MFG computation.  It does not by
itself establish an MFG solution, a Nash equilibrium, optimality, or a valid
many-server limit.

## 2. Fixed control planes and physical object

Placement is an immutable exogenous input within an episode.  The initial
scale is one Logical Expert and eight equivalent Replicas in four failure
domains.  Replica identity and domain membership come from placement; a
routing action selects one same-Expert Replica and never another Expert.

The finite engine retains the corrected ADR-0035 physical semantics:

- at most one live attempt exists for a Token;
- a failed running attempt loses its already executed work and creates a
  causal Replay in the global waiting queue;
- queued work displaced by failure has zero executed work;
- a Replica that is DOWN cannot receive or start work;
- failure-first ordering, complete drain, FCFS ordering, and work conservation
  are preserved;
- repeated passive Replays are permitted in this isolated multi-Replica
  feature, while the historical two-Replica engines remain unchanged.

Hedge, duplicate execution, protection budgets, and cancellation are outside
this design.  The current corrected risk-aware finite routing result is a
named finite baseline only; it is not evidence of MFG consistency.

## 3. Decision timing and information structure

Every batch release at time `t` uses this exact order:

1. Process all health transitions, invalidations, and valid completions due at
   `t` under the existing fault-first contract.
2. Construct one immutable pre-batch public state from the resulting prefix.
3. Freeze the current active-Token state `mu_t`, Replica state `nu_t`,
   decision cohort `eta_t`, and predicted routing shares `x_t` for the whole
   batch.  `eta_t` contains only new or Replay Tokens that require a routing
   decision at `t`.
4. Let every Token in `eta_t` independently sample or choose from its own
   causal observation and the same `(mu_t, nu_t, eta_t, x_t)` public context.
5. Prevent every Token from reading any other Token's realized action in that
   batch.  No in-batch assignment mutates an observation.
6. Submit all chosen assignments atomically to the Replica queues.
7. Continue queue service, failures, and Replays.
8. Construct the next transient state path only after the resulting
   transitions have been observed.

The batch's policy evaluation is simultaneous.  Token ID order may be used
  only for canonical serialization and deterministic tie-breaking, never as
  an information channel or a sequential update order.

For finite mixed policies, the action draw is derived from a stable causal
key:

`action_u = SHA256(namespace, episode, batch_id, token_id, policy_iteration)`.

The digest is mapped to a uniform variate and then through the frozen policy's
inverse CDF.  This key is independent of realized queue outcomes and future
events.  The same workload, class, service, and fault streams are reused
across paired policy/deviation branches; a branch always constructs fresh
endogenous state.

## 4. Token, Replica, and common public state

### 4.1 Token-local state

The policy may receive only the Token's current causal information:

- Token class;
- arrival time and current observable age;
- deadline and normalized/absolute slack derived from current time;
- retry count;
- ingress/locality identity;
- the current local observation needed by the routing contract.

It may not receive a sampled service requirement, true remaining work, a
future fault or arrival, an opponent action, or a counterfactual outcome.

### 4.2 Replica public state

The pre-batch public state exposes, for each eligible or ineligible Replica:

- health (`UP`/`DOWN`) and current observable capacity/speed;
- queue/work bin and running service-age bin;
- queue depth and other explicitly observable FCFS fields;
- causal individual-Replica hazard estimate;
- causal failure-domain hazard estimate and their effective estimate;
- failure-domain and locality identity;
- placement-fixed capacity metadata.

The hazard estimator is the accepted finite Gamma-Poisson history projection:

`lambda_hat_c(t) = (alpha_h + N_c(t)) / (beta_h + E_c(t))`.

`N_c` and `E_c` contain only the observed prefix through `t`, over a frozen
trailing history window and prior.  Exact numerical window, burn-in, rates,
service law, and fault law are implementation-protocol fields, to be frozen
in ticket 02 before trace generation.

### 4.3 Separate filtrations and transient population objects

The common-noise filtration contains only exogenous public failure history:

`F_t^0 = common fault and recovery history through t`.

It must not be used as a name for every fact visible to a policy.  The public
observation filtration is separate:

`G_t = F_t^0 + placement + current public health/queue state`
`      + current batch composition + published population trend`.

The policy may use the declared portion of `G_t`, subject to the causal
observation contract.  It cannot use future events or private realized
choices.

This is an open, transient system.  Define four distinct population objects:

- `mu_t`: the distribution of currently active Token states (waiting or
  running, including the fields needed by the transient physical model);
- `nu_t`: the distribution of public Replica states, including health, queue,
  service-age, hazard, domain, and locality buckets;
- `eta_t`: the cohort of new-arrival or Replay Tokens that actually require a
  routing decision at `t`;
- `x_t`: the action-share prediction for that decision cohort.

Completed Tokens leave `mu_t`; failed Tokens may re-enter through `eta_t` as
Replay cohorts.  Tokens already waiting or running are not repeatedly counted
in `eta_t` and therefore do not re-enter the routing-share integral.

## 5. Population routing trend and realized shares

Let `pi_j(s, nu_t, x_t)` be the probability that a Token in decision-cohort
state `s` selects Replica or Replica-state bucket `j`.  It is evaluated only
for `eta_t`, with the current Replica distribution `nu_t` and the previously
published trend available as public context.  The predicted trend is therefore

`x_t = Aggregate(pi_t, eta_t, nu_t)`.

Equivalently, when the cohort has a density, each component is the integral of
`pi_j` over `eta_t` (and the declared Replica-state conditioning in `nu_t`).
The normalization is by the current decision cohort, not by all active
Tokens.  This prevents already queued/running Tokens from being counted as
new routing choices.

The following quantities are distinct and must have distinct field names in
all outputs:

- `mu_t`: active Token-state distribution;
- `nu_t`: Replica-state distribution;
- `eta_t`: new/Replay decision-cohort distribution;
- `x_t`: policy-implied predicted action-share distribution for `eta_t`;
- `realized_share_t`: empirical share after the complete action cohort commits.

No realized action list is passed back into same-batch policy decisions.
Realized shares are an audit and calibration target, not a substitute for
`x_t`.  Every share record includes the cohort definition, public-filtration
prefix, bucket schema, and their fingerprints.

## 6. Scalable finite-to-mean-field convention

Increasing Token count on a fixed eight-Replica topology is not a valid MFG
limit.  The scaling convention is a family indexed by

`K in {8, 16, 32, 64}`

where `K` is the number of equivalent Replicas of one Logical Expert and `K`
must be a multiple of four.  The batch release rate is held fixed.  If
`E[B_K] = b*K` is the expected number of Tokens in a released batch, the total
Token arrival rate is explicitly `lambda_K = lambda_0*K`; there is no second
factor of `K` from also scaling batch release frequency.  Let `N_K` denote the
actual Token count in one finite batch; `N_K` is not the Replica count `K` and
must not be confused with finite-`N` deviation notation.

There are four fixed macro failure domains, each containing `K/4` Replicas,
so a domain shock remains common noise in the scaling family.  Total service
capacity is proportional to `K`, while per-Replica service ability and fault
laws remain unchanged.  Offered load is held fixed, and one Token's influence
on empirical state and share is `O(1/K)`.

The `K=8` instance must have an explicit regression mapping to the accepted
finite routing engine.  Any feature of the larger-K construction that cannot
be reproduced exactly at `K=8` must be named as an approximation in the
protocol and report; it may not be hidden under the word “limit”.

## 7. Private cost and social cost

The first Token cost is No-Hedge and pathwise:

`C_i = latency_i`
`    + alpha_k * deadline_miss_i`
`    + gamma_k * replay_count_i`
`    + c_lost * lost_work_i`
`    + congestion_charge_i`.

Latency, deadline, Replay, and lost work are settled from the complete
physical run.  The cost does not include Hedge or cancellation because this
slice has no duplicate execution.  `congestion_charge` is computed from a
pre-registered observable population-load function and current public state;
it must not read future service draws, future faults, or hidden remaining
work.  Exact coefficients and charge units are protocol parameters, not
derived from a posterior experiment result.

Two separately named models are required:

1. `unpriced_mfg`: `congestion_charge = 0`; this studies natural dispersion
   under reliability and queue congestion.
2. `priced_mfg`: a frozen observable congestion-externality charge is added;
   this studies whether a transfer can coordinate routing.

Price is not required for an MFG to exist and is not automatically a market
clearing price.  In the social objective, transfers are excluded:

`S = mean_i[latency_i + alpha_k*deadline_miss_i`
`    + gamma_k*replay_count_i + c_lost*lost_work_i]`.

If a future model treats a fee as a real resource cost, that term must be
added under a separately named social-cost definition.

## 8. Transient MFG consistency and separate response semantics

This is an open transient system.  The object being followed is the path

`(mu_t, nu_t, eta_t, x_t, pi_t) for t in [0,T]`.

Its local consistency and transition conditions are:

`x_t = Aggregate(pi_t, eta_t, nu_t)`

`(mu_{t+}, nu_{t+}) = Phi_t(mu_t, nu_t, eta_t, x_t, Z^0_{t+})`.

`Z^0` is the new common-noise increment (for example, an exogenous domain
failure/recovery transition).  The active population is not conserved:
arrivals increase `mu`, completions remove Tokens, and failures can create a
new Replay decision cohort.  A stationary population fixed-point equation is
not used unless a future protocol explicitly defines and proves a stationary
regime.

The core transition is:

`(mu_t, nu_t, eta_t) -> pi_t -> x_t -> simultaneous commit`
` -> (mu_{t+}, nu_{t+})`.

### 8.1 Mean-field best response

For a mean-field BR, the tagged Token receives the declared transient
environment path `(mu, nu, x)` as exogenous.  Its action changes neither that
path nor the aggregate trend.  The response evaluates the tagged action
against the specified conditional environment law; it is not a finite-system
rerun with a tagged Token's congestion fed back into `mu`, `nu`, or `x`.

### 8.2 Finite-K unilateral deviation

For a finite-`K` deviation, only the tagged Token's action is changed.  Other
Tokens keep their policy functions and are called online.  The complete
finite system is rerun with the same workload, fault, and service CRN.  The
deviation may change subsequent queues, public observations, `eta` cohorts,
other Tokens' actions, and delays.  Freezing the realized baseline trajectory
is invalid.

These are two different estimands and must have separate record types,
fingerprints, confidence calculations, and labels.  The solver may compare
them diagnostically, but it may not substitute one for the other.

The implementation and reports distinguish:

- pure best response;
- entropy-regularized/logit response;
- finite-`N` epsilon-regret;
- exact Nash;
- MFG consistency residual.

Softmax convergence is only a `regularized_mfg_candidate` until all declared
population, response, and finite-system validation gates are passed.  No
existence or uniqueness theorem is assumed.  Grid and bucket approximation,
sampling error, possible cycles, multiple fixed points, and unsupported
states are explicit failure/diagnostic outcomes.

## 9. Finite state and action approximation

The first action is not an arbitrary concrete Replica ID.  It is a choice of
an observable Replica-state bucket.  If a bucket has multiple eligible
Replicas, the selected Replica is sampled uniformly using the stable action
key, with a canonical ID tie/order rule for serialization.  A DOWN or empty
bucket is unavailable and cannot be selected.

The frozen bucket schema must distinguish at least:

- health;
- queue/work level;
- running service-age level;
- individual hazard level;
- domain hazard/fault-history level;
- failure-domain and locality relation.

All categorical levels, numeric boundaries, half-open endpoint rules, missing
value handling, bucket ordering, and schema fingerprint are fixed before any
sample is generated.  Empty buckets are rejected from the action support, not
filled from another state or silently merged.

## 10. Solver and estimator structure

The first solver shape is an explicit diagnostic loop:

1. Freeze a policy `pi_n` and its policy/source fingerprint.
2. Run independent forward episodes to estimate the transient path
   `(mu_n, nu_n, eta_n)` and realized shares.
3. Run independent mean-field continuation episodes to estimate
   `Q_MF(s,a | mu_n, nu_n, eta_n, x_n)` for supported state/action cells.
4. Construct either a pure BR or a frozen-temperature regularized response.
5. Apply a declared damping rule to obtain `pi_{n+1}`.
6. Run the next independent forward panel.
7. Separately run finite-`K` tagged deviations and check policy, transient
   population-state, predicted/realized-share, finite-deviation, and
   bucket-coverage residuals.

Forward and continuation libraries are independent and have disjoint
namespace, seed, and source/fingerprint identities.  One continuation episode
may contribute at most one tagged target; action branches within that episode
use paired CRN, but the episode remains the independent statistical unit.
No insufficient cell is filled with zero, merged, or adaptively resampled.

The exact panel sizes, physical parameters, action bucket boundaries, macro
seeds, namespace strings, and call budgets are frozen in implementation and
experiment tickets before input traces are generated.  A later holdout may
not tune a policy, price, grid, temperature, or damping value.

## 11. Baselines

The design retains these named baselines:

- uniform round-robin;
- simultaneous JSQ;
- simultaneous LOEW;
- reliability-only;
- the corrected finite risk-aware heuristic;
- Lazarus Algorithm 1;
- sequential centralized LOEW, explicitly `centralized_reference`, not an
  optimality upper bound;
- `unpriced_mfg`;
- `priced_mfg`;
- an optional centralized social controller only when a separate social
  optimization is actually solved.

Every simultaneous baseline receives one identical pre-batch observation and
cannot read an earlier Token's assignment from the same batch.  Sequential
centralized LOEW intentionally has different information and is reported as a
reference, not as a fair simultaneous player policy.

## 12. Verification layers

### A. Physics and information

- same-batch observations have one common pre-batch fingerprint;
- no Token sees another same-batch realized action;
- queues change only after the full action batch commits;
- CRN, Replay, complete drain, and attempt/work invariants hold.

### B. Population trend

- predicted shares agree with large independent realized-share samples within
  a predeclared calibration rule;
- Replica/type occupancy calibration is reported by common-noise history;
- only public fault-history prefixes affect common-state conditioning.

### C. Finite deviations

For each `K` in 8, 16, 32, and 64, freeze all other Token policy functions,
enumerate all supported tagged actions, rerun the full finite physical system,
and report the maximum single-deviation gain with simultaneous confidence
control.  A mean-field trajectory is not a replacement for this finite rerun.

### D. Decision value

Report latency mean/P95/P99/CVaR95, deadline miss, Replay and lost work,
load concentration, predicted/realized share error, common-fault recovery,
social cost, price transfer, price of anarchy where formally defined, and the
gap to the centralized reference.  These are metrics, not automatic claims.

## 13. Deterministic hand fixtures

Before stochastic validation, the implementation must cover:

1. Two identical Tokens see one reliable/congested and one riskier/idle
   Replica.
2. Without a population trend both Tokens select the reliable Replica,
   demonstrating herding.
3. A frozen predicted share or congestion charge splits the mixed flow.
4. One Token's finite action changes another Token's delay but not the declared
   mean-field environment.
5. An excessive price over-avoids a reliable Replica.
6. A domain common failure updates the conditional population distribution.
7. Sequential and simultaneous LOEW differ because their information differs.
8. As `K` grows, one Token's empirical-state effect decreases.

These are deterministic mechanism/contract tests, not experimental evidence.

## 14. Claim boundary and non-goals

This design does not authorize:

- placement optimization or Gate changes;
- Hedge, duplicate execution, cancellation, or protection budgets;
- posted-price pacing outside the frozen priced model;
- an HJB/FPK implementation;
- a claim of existence, uniqueness, Nash, MFG, optimality, or universal
  reliability prediction;
- a formal qualification/holdout campaign before its own protocol is
  accepted;
- changes to historical engines, configs, artifacts, or scientific results.

The current corrected ADR-0035 result is only a finite eight-Replica baseline.
It motivates, but does not validate, this MFG extension.

## 15. Implementation sequence

The only tickets created by this design are:

`02 -> 03 -> 04 -> 05 -> 06`

- 02: simultaneous finite engine and information tests;
- 03: population state/bucket and forward/share calibration;
- 04: tagged continuation and finite-`N` deviations;
- 05: unpriced/priced response solver;
- 06: `K=8,16,32,64` qualification and independent holdout.

Each ticket must freeze its own numerical parameters, source/protocol
fingerprints, sample identities, and call budget before consuming traces.
Design ticket 01 is the only completed ticket in this feature.  No production
code, experiment, solver, or artifact is authorized by this document alone.
