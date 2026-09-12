# MFG-Hedge attribution study

> Pending mathematical redesign (2026-09-05): implementation ticket 03 is
> paused by the user's design-first request. See `transient-model-design.md`,
> ticket 08 and Proposed ADR-0011. The accepted text below is retained as
> history; a resolved design ticket alone does not activate its proposed changes.

> Bounded activation (2026-09-05): Accepted ADR-0012 and ticket 10 authorize
> only transient development admission and realized objective scoring. The old
> controller/campaign tickets stay paused; full ADR-0011 remains Proposed.

> Development experiment activation (2026-09-05): Accepted ADR-0013 freezes
> and authorizes the 64-episode, 81-rule fit search only. These data are burned
> development evidence; qualification, holdout and MFG claims remain inactive.

## 1. Scientific question and claim boundary

The confirmatory question is:

> Under the same predeclared expected protection-work budget, does the
> conditional finite-action mean-field controller reduce fault-window tail
> latency more efficiently than frozen non-mean-field Hedge controllers, while
> preserving Healthy/Recovered service and passing the declared Replica-level
> resource audits?

The study separates six possible claims:

1. Hedge package value: a Hedge arm beats No-Hedge.
2. Rule-package value: a frozen state-aware rule beats No-Hedge; this does not
   separately identify the value of state observation versus Hedge itself.
3. Allocation value: the full controller beats the frozen rule.
4. Mean-field value: the full controller also beats a myopic controller that
   has the same observations, ActionStats, costs, and quota but no population
   load closure.
5. Quota value: quota reduces a Protection Storm without destroying fault-tail
   benefit.
6. Runtime-cost value: the 11 model passes the predeclared comparisons against
   the 00/10/01 cost ablations.

Beating No-Hedge alone is never evidence that mean-field allocation helped.
Even the strongest permitted claim is about this conditional finite-action
mean-field approximation, not a full dynamic/common-noise HJB-FPK MFG.

## 2. Historical evidence is development-only

All schema-3/schema-4 runs, including evaluation seeds 20260901..20260920 and
the 1,000/100,000-Token runtime-cost artifacts, are burned pilot/development
evidence. They may verify plumbing and inform this design, but may not enter a
confirmatory estimate, sample selection, threshold change, or holdout decision.

Existing configurations, calibration tables, solver behavior, ADRs, and
artifacts are immutable. This study uses a new schema and run family; it does
not reinterpret or overwrite old evidence.

## 3. Causal policy context and allowed actions

`CommonState` remains exactly H/D/F. R remains a metrics `Phase`; it is not a
fourth physical state. The new arrival-time control context is:

```text
(phase in H/D/F/R, token_class in Regular/Urgent, primary_replica in 0/1)
```

It is frozen when a Token arrives. `primary_replica` is the already-completed
fixed Dispatcher choice and is causally observable; it does not change the
Gate, Logical Expert, or Dispatcher. Completion phase and future event outcomes
are forbidden policy inputs.

For every attribution arm, including diagnostics:

- H: exact Normal;
- F: exact Normal by feasibility;
- R: exact Normal, separately audited from H;
- D: Normal / Delayed Hedge / Immediate Hedge are eligible only when their
  destination Replica passes the planned resource envelope. A future entropy
  diagnostic may differ only under a separately approved protocol.

At the confirmatory load, Replica 0 is the degraded/future-failed domain A and
Replica 1 is the healthy domain B. A D Token with Primary 0 may protect A -> B.
A D Token with Primary 1 would protect B -> A and is forced Normal because A
has no planned headroom. F-arrival Normal does not rewrite a Hedge/timer created
by a D-arrival decision; ADR-0005 copy lifecycle and event ordering remain exact.

The experiment assumes oracle observation of the current domain state. Detection
delay, noisy state belief, and stochastic transition duration are explicitly
outside the confirmatory claim and require later robustness work.

## 4. Primary and diagnostic arms

All arms use the same workload/failure trace, event engine, Replica placement,
Dispatcher, attempt draws, delay `tau0`, control-window boundaries, and quota
implementation.

### 4.1 Confirmatory primary arms

- `A0_no_hedge`: exact Normal in every context.
- `A1_rule_quota`: H/F/R Normal; in D, Primary 1 is Normal, while Primary 0
  uses Regular=Delayed and Urgent=Immediate before the common quota projection.
- `A2_myopic_11_quota`: H/F/R Normal; D makes a hard best response using the
  all-Normal baseline destination load and the same 1/1 cost model, but never
  feeds its induced destination-B load back into ActionStats. It may use the
  common-budget dual `price_Q`, but has no destination externality price
  `price_B`.
- `A3_mean_field_11_quota`: H/F/R Normal; D closes the population policy to the
  induced destination-B load and satisfies the hard-response/dual feasibility
  contract in section 7.

### 4.2 Diagnostic arms

- `D0_mean_field_11_no_quota`: uses exactly A3's causal phase-1 deficit requests,
  including the same ordering and tie-breaking, but skips phase-2 quota
  projection; diagnostic only because resource use is not matched.
- `D1_mean_field_00_quota`, `D2_mean_field_10_quota`, and
  `D3_mean_field_01_quota`: neither, incremental-only, and waste-only persistent
  costs. The two digits are `(incremental_work_cost, wasted_work_cost)`.
- Fixed all-Delayed and all-Immediate D rules may be reported as additional
  Pareto boundary diagnostics, always under the same predeclared quota scales.

No arm may be parameterized from a realized holdout work total. “Same budget”
means the same ex-ante cap, not post-hoc equality of realized work.
Ticket 06 freezes each cost diagnostic as either qualified/scheduled or
`NOT_EVALUABLE` with its deterministic solver reason; this status cannot be
changed after holdout labels are touched.

## 5. SLO and ActionStats contract

Normalized class deadlines and weights are frozen before implementation:

```text
deadline_regular = 3.0
deadline_urgent  = 2.0
class_weight_regular = 0.8
class_weight_urgent  = 0.2
excess_penalty_regular = 1.0
excess_penalty_urgent  = 5.0
replay_penalty_regular = 1.0
replay_penalty_urgent  = 5.0
incremental_work_cost  = 1.0
wasted_work_cost       = 1.0
```

They are untuned mechanism values, not measured cloud prices. Cost ablations
are named diagnostic arms rather than alternative schema-4 interpretations.
Below, `w_k` denotes the corresponding frozen class weight.

For class `k`, deadline `d_k`, action `a`, and mean field `m`:

```text
expected_excess[k,a,m] = E[max(0, latency - d_k)]
delta_total_work[k,a,m] = E[total lifecycle work | a,m]
                            - E[total lifecycle work | Normal,m]

base_cost[k,a,m] = E[latency]
                 + alpha[k] * expected_excess[k,a,m]
                 + gamma[k] * P(Replay | a,m)
                 + incremental_work_cost * delta_total_work[k,a,m]
                 + wasted_work_cost * E[non-winner executed work | a,m]
```

The destination congestion price adds
`price_B * E[lifecycle work sent to B | a,m]`. Lifecycle work is used by the
decision model; actual phase-and-Replica work is an evaluation audit. Work that
executes after a state boundary is reported in its actual execution phase and
must not be relabeled as earlier physical utilization merely because its Token
arrived earlier.

The new ActionStats schema must contain and validate:

- mean latency;
- actual deadline-miss probability;
- expected excess latency;
- Replay probability;
- expected Primary, Hedge, Replay, total, and wasted lifecycle work;
- the same work decomposition by execution phase and Replica.

It must prove `total = primary + hedge + replay`, scalar total equals the sum
over Replicas/phases, and `0 <= wasted <= total`. Missing new fields in a legacy
table are an error for an attribution controller, never implicit zero.

Calibration remains in a separate stable namespace and never reads evaluation
outcomes. The official decision table is deliberately a one-dimensional
destination-B-load surrogate. Independently keyed calibration-only load-carrier
streams make a tagged probe face nominal D `rho_A=0.90` and total destination
load `rho_B in {0.45, 0.55, 0.65, 0.75, 0.85, 0.90}`. These streams use the
declared service distributions, carry no measured logical outcome, and never
enter evaluation traces. Thus `rho_B` is total queue load faced by the probe,
not “extra Hedge load,” and even the 0.45 cell is a valid exogenous calibration
point.

For Primary 0 the table contains both classes and N/D/I (36 action cells);
Primary 1 is protocol-forced Normal and remains part of population base-rate
accounting rather than an optimizable action cell. At least 2,000 independent
tagged probes populate every official action cell; N/D/I variants of a probe
share CRN. Define, from those cells,

```text
g_B[k,a,rho] = E[Hedge_B work + Replay_B work | Primary=0,k,a,rho]
L_B(pi,rho) = (lambda/2) * E[Primary_B work | Primary=1,Normal]
              + (lambda/2) * sum_k w_k * sum_a pi[k,a] * g_B[k,a,rho]
rho_B = L_B(pi,rho_B) / capacity_B
```

Normal's Replay term in `g_B` is never treated as zero. Before any arm is built,
the table must yield a unique deterministic all-Normal fixed point `rho_B^N`
inside the grid; absence, multiplicity, or clamp makes the study invalid.
MYOPIC queries `rho_B^N`; A3 closes only its
induced `rho_B`. Each quota scale solves A2 and A3 separately because the common
budget dual changes their hard-response support.

This is a conditional destination-load mean-field approximation. It neither
claims that an all-Normal and mixed-Hedge background with equal `rho_B` have the
same queue distribution nor closes a joint two-Replica state distribution.
Action-dependent cancellation may reduce realized work on A; the fixed
`rho_A=0.90` is a conservative nominal reservation and is audited against
realized execution rather than presented as an endogenous A fixed point.

## 6. Per-Replica feasibility and confirmatory load

With mean service 1, two Replicas, slowdown 2, and round-robin Primary dispatch:

```text
arrival_rate = 2 * healthy_offered_load
D primary work rate to A = healthy_offered_load
D target work rate on A  = 0.9 * (1 / 2) = 0.45
```

Therefore the historical load 0.5 has base D utilization `0.5/0.5 = 1.0` on A
before any Hedge and is not per-Replica capacity-feasible. The confirmatory load
is frozen at `healthy_offered_load = 0.45`: D has nominal `rho_A=0.90` and
B Primary-only load 0.45. B's total all-Normal surrogate load is `rho_B^N`,
which also includes any Normal Replay and is not identified with 0.45.
Protection into A is forbidden; protection from A into B uses B's remaining
planned envelope. Load 0.5 and higher are structural stress diagnostics, not
official feasible results under this contract.

Resource accounting has three explicit views. First, the D-arrival cohort
solver reserves nominal Primary work on A and models expected lifecycle work
sent to destination B. Second, quota charges only expected lifecycle Hedge work
sent to B. Third, evaluation reports actual executed work by execution phase and
Replica, including carry-in and spillover. The planned envelope applies to the
nominal B Primary base and controllable D decisions; F and early-R backlog are
structural transients and are reported rather than used to invalidate an
otherwise valid controller. Expected feasibility is never described as a hard
realized-work guarantee.

For resource auditing, `pi_request` is the frozen pre-quota requested policy.
A0 uses `rho_B^N`. After A1 and A2 freeze their decisions, each receives a
resource-audit-only fixed point `rho = L_B(pi_request,rho)/capacity_B`; that rho
never feeds back into their action choice. A3's controller fixed point is the
same equation and does feed back. The planned B-envelope gate evaluates
`pi_request` at those audit coordinates. Separately, quota certifies the charge
of actually applied actions, and evaluation reports their realized work. These
three objects must never be substituted for one another.
Each A1/A2 audit-only fixed point must be unique under piecewise-linear
interpolation and inside the frozen grid; otherwise that arm fails qualification.

## 7. Decision rule and mean-field claim

Softmax is not used by any confirmatory primary arm. A slack context uses an
exact hard best response with fixed tie order `Normal < Delayed < Immediate`.
Positive mass on a strictly more expensive action is forbidden.

When a resource constraint binds, an A3 mixture is valid only if:

```text
every supported action minimizes
    base_cost + price_B * g_B[k,a,rho_B] + price_Q * g_budget[k,a]
planned_B_work_rate <= 0.9 * capacity_B
price_B >= 0
price_B * (planned_B_work_rate - 0.9 * capacity_B) = 0 within tolerance
planned_hedge_budget_rate <= s * B_nominal
price_Q >= 0
price_Q * (planned_hedge_budget_rate - s * B_nominal) = 0 within tolerance
policy -> planned rho_B -> queried ActionStats is self-consistent
```

Replica A is not solved as a second endogenous fixed point: its nominal Primary
reservation is fixed at the D target and realized A work is audited as described
in section 6. `price_Q` is the common protection-budget dual, distinct from B's
capacity price. Mixture is allowed only across actions tied under the final
prices. A deterministic multi-solution rule first minimizes unpriced fault/SLO
loss, then total work, then uses the fixed lexical action order. Solver output
includes support, both prices, destination load, both resource residuals,
selection witness, and a stable status/reason. No Softmax fallback is allowed
after non-convergence.

A controller produced directly by minimizing a joint population/social
objective is Mean-Field Control (MFC), regardless of how many policy effects its
implementation happens to internalize. The A3 name is permitted only for a
price-taking hard-best-response population/load/dual certificate; an LP or
central planner without that witness is labeled MFC. Quota is a causal
realization layer and does not itself prove an equilibrium.

An entropy diagnostic, if ever added, must be named `entropy_regularized` and
write its full objective:

```text
sum_a pi[a] * J[a] + (1 / eta) * sum_a pi[a] * log(pi[a])
```

It is never presented as numerical smoothing or used for a primary conclusion.

## 8. Common ex-ante protection budget

At load 0.45, the nominal protection-budget base is B's target headroom after
base Primary work but before Replay or Hedge:

```text
B_nominal = 0.9 * 1.0 - 0.45 = 0.45 work / time
```

Every quota-controlled Hedge arm is evaluated at fixed scales
`s in {0.25, 0.50, 0.75, 1.00}` of this envelope. A 25-time-unit D window has
caps `{2.8125, 5.625, 8.4375, 11.25}`; a whole 100-unit D episode has caps
`{11.25, 22.5, 33.75, 45.0}`. The confirmatory operating point is `s*=0.25`;
other scales form a secondary Pareto curve and may not replace the main point
after results are observed.

All arms use one extended causal deficit/quota contract with the same window
boundaries and one arm-independent charge table frozen from the all-Normal
destination fixed point:

```text
g_budget[k,a] = E[Hedge_B lifecycle work | Primary=0,k,a,rho_B^N]
g_budget[k,Normal] = 0
sum_applied g_budget[k,a] <= s * B_nominal * window_duration
planned_hedge_budget_rate = 0.45 * sum_k w_k * sum_a pi_request[k,a]
                                    * g_budget[k,a]
planned_B_rate = 0.45
                 + 0.45 * sum_k w_k * sum_a pi_request[k,a]
                          * g_B[k,a,rho_B]
planned_B_rate <= 0.9
```

Decision ActionStats may vary with an arm's `rho_B`, but quota charging never
does; therefore the same applied action consumes the same declared budget in A1,
A2, A3, and cost diagnostics. Per class, expected demand is
`d_k = w_k * sum_a pi[k,a] * g_budget[k,a]`; if total demand is positive its
isolated share is `window_budget * d_k / sum_j d_j`. Neither class borrows the
other's share, unused share expires at window close, and every window resets
deficits and shares. Requested/applied/suppressed identities and planned/realized
separation from ADR-0006 remain mandatory.

The solver separately reports the total B envelope above, which includes base
Primary and expected Replay/Hedge work. A0-A3 at `s*=0.25` must pass it; other
Pareto-scale points that fail are labeled structural diagnostics rather than
capacity-feasible results. A realized-work comparison is called matched only if
its predeclared confidence gate passes; otherwise the Pareto coordinates are
reported without that label.

## 9. Repeated episode and sampling unit

One independent episode starts with empty queues and uses:

```text
H [0,100), D [100,200), F [200,220), R [220,320)
stop new arrivals at 320, then drain every accepted attempt
```

There is no queue carry between episodes. Boundary/event semantics remain those
of ADR-0004/0005. A macro-seed contains 50 independently keyed episodes. The
confirmatory campaign contains exactly 40 macro-seeds. All arms and quota scales
within an episode share byte-identical arrivals, classes, attempt-0/1/2 draws,
and failure trace.

At load 0.45 (`arrival_rate=0.9`), expected counts—not guarantees—are:

```text
per episode:    H=90, D=90, F=18, R=90, total=288
per macro-seed: H=4500, D=4500, F=900, R=4500
full campaign:  H=180000, D=180000, F=36000, R=180000, total=576000
```

The inferential sample size is 40 macro-seeds (`df=39`), never 576,000 Tokens
or 2,000 episodes. Episodes stabilize each macro-seed estimate. Metrics first
aggregate inside a macro-seed and only then form same-macro paired arm
differences. Token-distribution metrics (including CVaR and percentiles) pool
the relevant Tokens across its 50 episodes; rates and work ratios sum their raw
numerators and denominators before division. Episode-time objects—peak launch
rate, peak/mean ratio, longest overload duration, boundary live/work backlog,
recovery-clear time, and drain duration—are computed within each episode first,
then averaged arithmetically across the 50 episodes. Independent episode clocks
are never stacked, concatenated into a fake run, or allowed to form a cross-
episode overload interval.

A small Healthy-only family is a development no-op regression proving that H
requested/applied actions remain exact Normal. It is neither a confirmatory
endpoint nor pooled with the fault-episode estimates. Confirmatory H-arrival
non-inferiority is a safety/spillover check for the common fault episodes; it is
not called protection tax or controller wall-clock overhead.

## 10. Metrics

### 10.1 Co-primary efficacy metrics

- arrival-D empirical CVaR95 latency;
- arrival-F empirical CVaR95 latency.

For `n` observations, empirical CVaR95 is the arithmetic mean of the largest
`ceil(0.05*n)` finite values after stable sorting. Arrival phase fixes cohort
membership; completion phase remains a separate migration diagnostic.

### 10.2 Required safety/resource metrics

- Replay rate;
- H and R mean/P50/P95/P99;
- execution amplification and total executed work;
- planned and realized utilization/work by execution phase and Replica;
- every trace, quota, completion, attempt-count, and solver invariant.

### 10.3 Secondary explanatory metrics

- D/F P99 and every phase x class mean/P50/P95/P99;
- deadline-miss probability and mean excess latency by phase/class;
- Primary/Hedge/Replay/total/wasted work and waste ratio;
- Hedge requested/launched/winner/cancelled/loser counts and win/launch ratio;
- extra work per Replay avoided and extra work per unit CVaR reduction;
- live count and remaining normalized workload by Replica at F and R boundaries;
- R backlog area for pre-R arrivals, recovery-clear time for pre-R arrivals,
  and post-320 drain duration;
- peak Hedge launch rate, peak/mean ratio, and longest overload duration.

F/R improvements are described as spillover/recovery effects of D protection,
not active F/R policy improvements.

## 11. Statistical analysis

Each metric produces 40 paired macro-seed values. Positive-valued efficacy
metrics use `log(arm/comparator)`; reports include the geometric ratio, raw
values, sign counts, and two-sided 95% Student-t confidence interval. Rates use
paired percentage-point differences. Non-inferiority uses the predeclared
one-sided 95% t upper bound.

Paired-t degeneracy is deterministic. If all 40 paired differences are zero,
the confidence interval is `[0,0]`, every superiority p-value is 1, and there is
no superiority evidence. If sample variance is zero at a common nonzero
difference, the confidence interval collapses to that value; a lower-is-better
one-sided superiority p-value is 0 for a negative difference and 1 for a
positive difference (two-sided p=0). Holm adjustment is then applied to these
finite p-values exactly as in the nondegenerate case.

The D/F co-primary gate is an intersection requirement: both must pass. For an
“at least one of D/F is superior” attribution claim, the two superiority tests
use Holm family alpha 0.05 and the other endpoint must pass its non-inferiority
margin. Gate order is A3 vs A0, then A3 vs A1, then A3 vs A2; a later claim is
not tested or made when an earlier gate fails. Secondary metrics are descriptive
unless this spec gives them an explicit gate.

## 12. Predeclared success gates at s*=0.25

### 12.1 Validity gate

- all 40 macro-seeds and all 50 episodes per macro-seed are present for every
  arm/scale frozen as scheduled;
- all scheduled arms share the required trace hashes;
- every Token completes with exactly one winner, at most one Replay, and at most
  one Hedge;
- H/F/R requested and applied actions are exact Normal in every scheduled arm;
- solver, work algebra, the nominal A reservation, the planned D-cohort B
  envelope, and quota identities pass;
- A2/A3 converge without grid clamp; a D1-D3 arm that failed development
  qualification is frozen as `NOT_EVALUABLE` before holdout and does not make
  the primary study invalid, while an unexpected scheduled-arm failure does;
- no seed or episode is removed because of its result.

Any failure makes the confirmatory run invalid rather than partially reportable.
F and early-R carry-in/spillover work are reported as structural transient load,
not folded into the D decision envelope and never called a capacity-feasible
equilibrium.

### 12.2 A3 versus A0: protection package value

- D and F CVaR95 point estimates each improve by at least 10%, and each paired
  two-sided 95% CI has an upper bound below zero log-ratio;
- Replay-rate paired difference has a one-sided 95% upper bound <= 0;
- H and R mean-latency relative deltas have one-sided upper bounds <= +1%;
- H and R P95 relative deltas have one-sided upper bounds <= +2%;
- A3 execution-amplification one-sided 95% upper bound is <= 1.05.

H outcome equality is not required: an H-arrival long-tail Token can cross a
later boundary and interact with D work. Strictly pre-D completions are reported
as a mechanism regression, not used to select the main cohort.

### 12.3 A3 versus A1/A2: attribution value

Against A1 and then A2, under the same ex-ante cap:

- at least one D/F CVaR95 endpoint is superior under Holm alpha 0.05;
- the other endpoint passes a +2% non-inferiority margin;
- the total-work ratio one-sided 95% upper bound is <= 1.005;
- the H/R and global amplification gates above remain satisfied.

Passing A1 supports an allocation-value claim. Passing A2 in the fixed gate
order additionally supports a mean-field-closure claim. Failure against either
is retained as a valid negative result.

### 12.4 A1 versus A0: frozen-rule value

A1 must independently pass the same D/F co-primary efficacy, Replay, H/R
non-inferiority, and amplification gates specified for A3 versus A0. Passing
this gate supports a frozen-rule/protection claim only; it is not evidence for
allocation or mean-field closure.

### 12.5 Diagnostic attribution gates

- Zero-safe diagnostic resource effects use paired macro-seed arithmetic
  differences, never log ratios. Quota evidence requires A3-D0 to have a
  negative point estimate and Holm-adjusted one-sided paired-t `p < 0.05` for at
  least one of peak Hedge launch rate or longest sustained-overload duration;
  those two resource tests are one Holm family. Both D/F CVaR95 log-ratio one-
  sided upper bounds must also be `<= log(1.02)`. It does not establish equal-
  resource efficiency.
- Incremental-cost evidence uses the arithmetic A3(11)-D3(01) execution-
  amplification difference; waste-cost evidence uses the arithmetic A3(11)-
  D2(10) waste-ratio difference. Each must have a negative point estimate and
  Holm-adjusted one-sided paired-t `p < 0.05`; those two resource tests form one
  Holm family. Both corresponding D/F CVaR95 log-ratio upper bounds must be
  `<= log(1.02)`.
- 11 versus 00 is a joint/interaction diagnostic only.

A cost flag whose required development arm was frozen `NOT_EVALUABLE` remains
`NOT_EVALUABLE`, not FAIL or PASS. The main VALID/C0-C3 result is still computed.

## 13. Seed isolation and one-shot holdout

Stable namespaces are disjoint:

```text
attribution:v1:calibration:...
attribution:v1:development:...
attribution:v1:holdout:<macro>:episode:<episode>:...
```

Before the first holdout event is generated, ticket 06 must freeze and hash the
spec, ADR, code, configuration, analysis implementation, calibration table,
arm list, quota scales/main point, 40 x 50 seed registry, and every gate. Holdout
labels are not touched during development. There is no peeking, early stopping,
seed deletion, threshold change, or result-dependent rerun.

Infrastructure failure may retry the identical stable key with an audit record.
A scientific/code defect discovered after unblinding invalidates the entire v1
holdout; the defect is published and a repaired study uses a fresh
`attribution:v2:holdout:` namespace. Unaffected v1 seeds are not recycled.

## 14. Scenarios and interpretation

The confirmation scenario is nominal load 0.45, slowdown 2, failure duration
20, and the fixed 100/200/220/320 episode.
After the primary gate is reported, predeclared descriptive robustness may use
load 0.40, failure durations 5/50, and slowdown 1.5. Load >=0.50 or slowdown 3
is capacity/infeasibility stress unless a separate per-Replica certificate says
otherwise; it is not silently pooled with confirmation.

The main explanatory plot is fault-tail latency versus realized execution
amplification, with SLO miss/excess shown as secondary diagnostics, for every
arm and quota scale. All points are shown; the best post-hoc point never
replaces `s*=0.25`.

Completeness/validity is reported separately as `INVALID` or `VALID`. Only a
`VALID` study receives the ordered primary conclusion code:

- C0: valid complete study, but no primary protection-package evidence;
- C1: protection-package evidence (A3 passes A0 and all safety gates);
- C2: allocation evidence (C1 plus A3 passes A1);
- C3: conditional destination-load mean-field evidence (C2 plus A3 passes A2).

`RULE`, `QUOTA`, `COST_WORK`, and `COST_WASTE` are independent diagnostic flags
with values PASS/FAIL/NOT_EVALUABLE, set only for a `VALID` study and only by
their own gates; they are not extra rungs that can inflate C0-C3.
`RULE` means evidence for the frozen rule package, not isolated state-sensing
value.

## 15. Artifact and failure contract

A complete, protocol-valid campaign is transactionally committed to one fresh
`artifacts/<run-id>/` directory whether its scientific gates pass or fail. It
contains immutable manifest/config/code/spec/seed/calibration hashes, controller
semantics, policy/dual witnesses, one complete record per macro-seed, the
aggregate/gate report, and plot-ready Pareto rows. A complete negative result is
scientific evidence and must not disappear. Infrastructure, completeness, hash,
or invariant failure commits no partial scientific directory; its human-readable
audit remains in the responsible ticket. No timestamp is scientific content and
no old directory is overwritten. Ticket 06 writes the immutable pre-run freeze
bundle to `artifacts/attribution-v1-freeze-<manifest-sha256>/` before ticket 07
may generate any holdout trace.

## 16. Scope boundary and ticket order

This feature does not add HJB-FPK, training, Gate changes, cross-Expert routing,
dynamic Replicas, multiple Experts, detector learning, ideal cancellation, or
policy-composition-consistent background calibration.

Implementation order is fixed:

1. ticket 01: freeze this design and ADR-0010;
2. ticket 02: fixed-horizon episode protocol and metrics;
3. ticket 03: ActionStats, tail/work calibration, and decision objective;
4. ticket 04: RULE/MYOPIC/mean-field controllers and quota integration;
5. ticket 05: multi-arm budget-matched campaign and transactional artifacts;
6. ticket 06: development qualification and holdout freeze, without execution;
7. ticket 07: one-shot holdout execution and conclusion grading.

A policy-composition-consistent mean-field calibration is a separate future
feature only after the attribution result justifies the additional complexity.
