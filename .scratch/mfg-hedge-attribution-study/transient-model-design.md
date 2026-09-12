# Proposed mathematical model: transient queue control with explicit execution

Status: Proposed design; governed prospectively by Proposed ADR-0011.

This document responds to the user's request to design the mathematical model
before code changes. It proposes a physical reference and a bounded first
solution method. It does not amend the accepted old experiment in place.

## 1. Research question and solution concept

Engineering question: can a causal protection policy reduce D/F tail latency
at a controlled computational cost without damaging H/R service?

Attribution question: how much benefit is due to a fixed rule, optimized
allocation, and eventually a validated population-interaction approximation?

The physical system is a finite, stochastic, partially observed queue-control
problem. One controller operates both Replicas. A global episode objective
naturally defines stochastic control. Calling its solution MFG would require
additional game/population assumptions; these are stated in section 10.

The first solver is an exhaustive reference over 81 predeclared causal rules.
Its claim is restricted-policy control value. It does not replace the user's
longer-term MFG question with a purported MFG result from a central planner.

## 2. Exogenous experiment and information

Use normalized work/time units. A healthy server executes one work unit per
time unit. Service requirements have mean 1 and CV 0.5. Let

    lambda = 0.9 Tokens / time,  P(Regular)=0.8, P(Urgent)=0.2.

Arrivals are Poisson on [0,320). Class draws and every service requirement
S[i,r,a] use independent stable streams; a=0/1/2 means Primary/Replay/Hedge.
The same complete exogenous trace is reused across policies. A policy cannot
see its Token's sampled service requirements or any future arrival/outcome.

The deterministic scenario is H [0,100), D [100,200), F [200,220), R [220,infinity).
All accepted copies drain after 320. Replica capacities in work/time are:

    mu_A(t) = 1 in H/R, 0.5 in D, 0 in F;  mu_B(t) = 1.

The first development scenario assumes the clock and the scheduled scenario
are known. A time-aware policy can therefore exploit proximity to failure;
this is scheduled/oracle-fault evidence, not unexpected-failure robustness.
An unknown failure time would require a separately declared hazard/belief model.

Physical state X(t) includes ordered queues on both Replicas; running copies
and their progress; Token-copy links; pending timer deadlines; winner flags;
Replay flags; state/clock; and admission balances. Residual sampled work is
internal state, not an observation available to the policy.

The allowed causal observation O_i at arrival includes time, phase, class,
fixed Primary, visible queue counts, elapsed service ages, pending-copy counts,
and admission balance. The reference policy deliberately uses only time block,
class and Primary. A later feedback policy may use the richer observation;
it may not query future completion or internal residual work.

For lognormal service, queue length alone is not an exact Markov state. A
partially observed controller would need the conditional residual distribution
given observed service age, rather than access to the true sampled residual.

## 3. Exact queue dynamics and work identities

Each Replica uses FCFS and serves at most one copy. Let V_r(t) be total
remaining work of all live copies on Replica r. Define on (s,t]:

- A_r(s,t]: required work of copies actually enqueued;
- E_r(s,t]: work actually executed (integral of speed times busy indicator);
- C_r(s,t]: remaining work removed by cancellation of queued copies;
- D_r(s,t]: remaining work discarded by failure.

The exact sample-path conservation law is

    V_r(t) = V_r(s) + A_r(s,t] - E_r(s,t] - C_r(s,t] - D_r(s,t].

Use consistent right-continuous endpoint conventions and the event ordering
below. This is bookkeeping, not a fluid approximation. V alone does not
determine future behavior: queue order, timers and duplicate links remain in X.

At t=200, A's running and queued copies fail. Their unexecuted work leaves
V_A; previously executed work stays in the execution-cost ledger. Eligible
Tokens create one Replay on B in running-then-FCFS order. This creates an
explicit jump in A_B and V_B. Do not spread that jump across the D interval.

Keep ADR-0005: state/failure/replay -> completion -> timer -> arrival -> dispatch.
First valid completion wins; queued losers cancel; running losers continue
until completion or domain failure. A Primary failure with a live Backup has
no Replay. A Primary failure before its timer creates Replay and voids timer.

For Token i:

    L_i = winner_completion_i - arrival_i
    W_i = W_i^primary + W_i^hedge + W_i^replay
    W_i^waste = sum executed work on every non-winning copy.

For every execution phase z and Replica r, split actual service integrals at
100/200/220/320. Sum of these pieces equals lifecycle W_i. Drain remains
separate for fixed-window metrics. Cancelled, never-started work contributes
zero execution cost; failed executed work contributes positive waste.

## 4. One causal action-to-execution mechanism

Primary is i mod 2 except F arrivals go to B. No Gate or Dispatcher changes.
H/F/R and D/Primary-B arrivals request exact Normal. Eligible D/Primary-A
arrivals request Normal, Delayed or Immediate. Delayed means arrival+tau0,
with tau0 the existing 0.9 service quantile.

Let u_i be the requested action, b_i the ledger before arrival, and

    a_i = P(u_i,b_i);    b_(i+1) = Update(b_i,u_i,a_i).

The physical transition is X_next = F(X,a_i,exogenous events), never F(X,u_i,...).
Both a development prediction and evaluation must run this same composite
policy. Suppression changes later queueing, cancellation, Replay and budget
availability and therefore affects all later outcomes.

Proposed simple ledger, shared by every controlled arm:

- D windows are [100,125), [125,150), [150,175), [175,200).
- Window reservation cap B_w = s * 0.45 * 25, s in {0.25,0.5,0.75,1}; main s=0.25.
- Normal costs zero. An admitted D/I request reserves h=E[S_backup]=1.
- Admit if balance >= h; otherwise apply Normal and record suppression.
- Debit at request time, including Delayed. No refund when a timer is voided,
  the copy cancels, or a failure prevents launch. Reset at each window.
- One pooled ledger; FIFO arrival admission; no class-specific reservation.
  Class preference is encoded by the request policy and objective.

Thus sum of reservation charges <= B_w holds on every path. At main s=0.25,
B_w=2.8125 admits at most two requests per window; unused fractional balance
expires. This granularity is reported, not hidden or replaced by a more
favorable retrospective budget. One can later study larger windows, but not
change them after observing the main results.

Because admission does not see S_backup and those requirements are independent
of the admission history, expected executed Hedge work is bounded by expected
reserved work, pooled over the admitted cohort. This statement needs the stated
independence; it is not a realized-work guarantee. Arbitrarily large lognormal
draws preclude a deterministic executed-work cap with conservative cancellation.
Replay is outside this controllable ledger and always enters W and V_B.

This prospective rule changes ADR-0010's class-share and expected-work-table
semantics. It eliminates the circular quota-charge calibration. Every arm gets
identical admission mechanics. Pooled admission can disadvantage later Urgent
arrivals; report per-class suppression and SLOs and retain this limitation.

## 5. Population losses and persistent operating cost

For policy pi, define a class/arrival-phase Token distribution using pooled
expected counts across independent episodes. For any f:

    E_(z,k)^pi[f] = E[sum_(i:arrival_phase=z,class=k) f_i] / E[N_(z,k)].

This is not an average of per-episode ratios with empty cohorts omitted.
Arrival processes and counts are policy independent. Let w_R=.8, w_U=.2,
d_R=3, d_U=2, alpha_R=gamma_R=1, alpha_U=gamma_U=5. Define

    Loss_z(pi) = sum_k w_k * (E_(z,k)[L]
                    + alpha_k E_(z,k)[(L-d_k)_+]
                    + gamma_k P_(z,k)(Replay)).

Use standard fractional-tail population CVaR for pooled class-mixed D/F latency:

    C_z(pi) = inf_eta { eta + 20 E_z^pi[(L-eta)_+] }, z in {D,F}.

The proposed system objective is

    J(pi) = (1/4) sum_(z=H,D,F,R) Loss_z(pi)
              + (1/2)(C_D(pi)+C_F(pi))
              + c_work E[sum_i W_i] / E[N]
              + c_waste E[sum_i W_i^waste] / E[N],
    c_work = c_waste = 1.

All terms are in normalized time after assigning work/time conversion to the
coefficients; gamma has time units, alpha and the tail weight are dimensionless.
Equal phase weighting prevents long phases dominating by their Token counts.
These are proposed untuned engineering preferences, to freeze before fitting.

Charging total W versus W-W_NoHedge changes J by a policy-independent constant
on the same exogenous distribution. Optimize total W; report the paired delta
for interpretation. Waste cost deliberately adds a preference against useless
execution; it is not a second physical resource ledger.

Including F and R outcomes makes D protection's spillover part of the system
objective. F/R still perform no protection at arrival. A local one-Token
best response cannot be claimed to optimize this system objective.

Sample CVaR uses exactly 5% probability mass, including a fractional boundary
observation when 0.05*n is noninteger. Old empirical mean-of-largest-ceil(0.05*n)
is a different finite-sample statistic: retain it as legacy diagnostic under
its old name/schema; use a new explicit name such as cvar95_fractional_tail.
For n=21 with largest values 100 and 10 the new value is
(100+0.05*10)/1.05, whereas the old rounded-tail mean is 55. Do not conflate them.

## 6. Resource and safety constraints

Admission reserves a controllable work envelope; server service rates enforce
physical capacity; these are different constraints. Define fixed-window
executed utilization U_r(w)=E_r(w)/integral_w mu_r(t)dt when the denominator
is positive. By construction U<=1; an audit reporting >1 indicates an error.
Admitted-work pressure can exceed 1 and is not executed utilization.

Nominal D Primary load is 0.45 on each server: A is at nominal rho=.9, B=.45.
This is a reference reservation, not a prediction of either queue's backlog.
Expected or realized short-window traffic can exceed nominal capacity.
Allow physical transient overload and audit V_A/V_B, admission pressure,
actual service, Replay pulse, recovery and drain.

For reference policy selection propose the existing safety preferences as
population constraints against the all-Normal policy on the same scenario:

- H and R mean ratios <=1.01; H and R P95 ratios <=1.02;
- expected total executed-work ratio versus No-Hedge <=1.05.

The last ratio differs from historical execution amplification W/nominal
Primary work; report both and do not rename one as the other. No-Hedge is in
the candidate bank and is feasible against itself. Feasibility estimates on
development data do not certify the population constraints; validation reports
uncertainty and can fail qualification. Do not tighten, relax, or retune based
on holdout. Co-primary D/F improvement remains a separate result gate, not
something a nonconverged solver may assert.

## 7. Bounded first solution: 81 rules

Split controllable D into early [100,150) and late [150,200). For each of
(early,Regular), (early,Urgent), (late,Regular), (late,Urgent), select N/D/I.
This gives 3^4=81 request rules. H/F/R and Primary-B are always N. Every rule
passes through the ledger in section 4. The No-Hedge and frozen
Regular-Delayed/Urgent-Immediate rule are included.

For each predeclared scale, evaluate every rule on the same development traces
through the full event process; compute its objective and safety constraints.
Choose the minimum feasible sample J, then sample total work, then lexical
N<D<I order in the four contexts. Enumerate the full bank; no local-search or
fixed-point convergence claim is needed. Exact numerical ties use the stated
lexical rule; statistically indistinguishable policies are reported as such,
not silently assigned an arbitrary numerical tolerance for optimality.

The output is the sample-best policy in this 81-rule class. It is not optimal
over arbitrary feedback policies, does not establish equilibrium, and is not
called MFC without a mean-field limiting control model. Use the descriptive
arm name finite_policy_control_81.

All selection is performed on fitting data. Evaluate the already selected
policy on a disjoint qualification set. Qualification failure is reported;
do not choose a different bank member on the qualification set. Any redesigned
bank/parameters consume that qualification set as development and require a
new untouched validation namespace. Full 81-rule validation, if performed,
is descriptive and cannot alter the selected rule.

Initial compute sizing proposal: 64 fitting episodes and 128 qualification
episodes, grouped into 8 and 16 groups of 8 for development uncertainty only.
These are pipeline sizing defaults, not an assertion of statistical power.
Run a small complete-bank timing smoke test before large development work.
The old 40x50 confirmatory campaign stays frozen/unexecuted until a revised
arm/gate protocol is designed. Do not infer new confirmatory sample sizes here.

## 8. Separation of model, solution and evaluation

The solver consumes development episode outcomes and emits a frozen causal
rule, parameters, and provenance. Evaluation receives only that rule and an
independent exogenous trace. It returns no outcomes to the frozen solver.
Reusing verified lifecycle primitives or the exact engine in development is
permitted; this is simulation-based control with statistical sample separation,
not independent software validation. Hand-computed fixtures test shared physics.

Use new isolated namespaces:

    transient-control:v1:fit:...
    transient-control:v1:qualification:...

No holdout namespace is generated by this design or its model-validation stage.
Every arm within an episode shares stable arrival/class/attempt keys, and its
action cannot alter any exogenous draw. Ledger reset and empty initial queues
are part of the episode identity. Runtime timing is diagnostic, not scientific
content. No neural fitting or third-party dependency is required.

## 9. Validation of any later reduced model

A reduced model must predict under complete policy composition and the same
admission mechanism, with distinct fit/validation episode pools. It must at
least model time, both queue environments, residual-service uncertainty,
pending Hedging/Replay, and remaining budget or explicitly bound the error
from omitting each state. Means of V_A/V_B alone are not claimed sufficient.

The exact forward object is the law

    m_t^pi = Law(X_t^pi | known common scenario),

and its evolution is generated by applied actions. A compressed approximation
may propagate particles or bins without solving a full HJB-FPK PDE. It must
carry failure jumps and cross-copy cancellation links, or demonstrate their
approximation error. No physical rho is constructed by distributing future
Replay lifecycle work uniformly over earlier D time.

Proposed development model-quality gates (freeze before seeing their results):

- Exact hand fixtures below pass without statistical tolerance.
- For each predeclared validation rule, mean total/replay/hedge work and F/R
  mean remaining-work snapshots: abs prediction error <=max(0.05 work,10% reference).
- D/F mean and fractional CVaR95: abs error <=max(0.1 time,10% reference).
- For a sampled proposed MFG witness, requested/applied counts, suppression and
  phase/Replica work must all be reproduced; checking total rho alone is insufficient.
- Qualification sampling uncertainty must also be reported. If the reference
  interval is too wide to assess the error threshold, report inconclusive and
  follow a predeclared larger validation batch; no automatic pass.

Ten-percent predictive tolerances qualify only a coarse development surrogate;
they cannot substantiate a two-percent superiority claim. A final small-effect
claim is made by independent exact evaluation with its own precision analysis.

## 10. What an MFG extension would need

An exact finite-system control model does not become MFG by defining m_t.
An explicit game must specify each player's private objective, allowed
observations/deviations, admission mechanism, and population scaling. With two
fixed servers, adding more Tokens increases load and is not by itself a valid
negligible-player limit. Scaling service/arrival/population must preserve the
queue interaction that the approximation claims to model.

For a well-defined future nonatomic model, its proposed equilibrium would have:

    pi* in BestResponse(m*, admission mechanism, prices),
    m* = ForwardLaw(applied(pi*, admission mechanism), common scenario).

The best response must include admission probability and future state costs.
Using cost of a granted Hedge for a request likely to be denied is inconsistent.
If prices are introduced, the constraint they price must have the same units,
time support and population accounting in both player and forward equations.
Capacity shadow prices are not automatically marginal social-delay tolls.

In the finite shared-ledger system, one Token can consume a significant share
of a window budget. Its effect must be retained in a finite-game deviation
calculation; it cannot be declared negligible. A reduced empirical best-response
procedure may be evaluated as a heuristic, with no MFG limit claim.

To report MFG evidence later require: an explicit scaling/approximation argument;
applied-process prediction validation; a best-response/deviation residual with
Monte Carlo confidence; and an otherwise matched non-closure comparator.
Central control, finite-game equilibrium, and nonatomic MFG results retain
distinct labels. No score improvement alone supplies these missing conditions.

## 11. Hand-checkable acceptance fixtures

1. Same 20 units uniformly enqueued versus all enqueued at failure: total work
   matches while V_B path and delay differ; a reduced model must distinguish them.
2. Backup wins before Primary starts: remove its queued A work, no negative V_A,
   and let later A work start earlier. If Primary already runs, retain its work.
3. Admission suppresses a Hedge: simulate applied Normal, including any extra
   later Replay. Requested-policy outcomes cannot be reused.
4. Failure exactly at a completion/timer: reproduce ADR-0005 ordering.
5. Pending D timer after F: void/suppress per lifecycle; reservation stays charged.
6. A reservation cap of 2.8125 admits two mean-one requests and denies the third;
   realized sampled Hedge work may exceed 2.8125 without a reservation violation.
7. Conservation at every event and both boundaries, plus full lifecycle/phase
   execution algebra and zero service on failed A.
8. Fractional CVaR n=20/21 agrees with its variational definition and is labeled
   distinctly from the legacy rounded statistic.
9. H/F/R and D/Primary-B are Normal for all 81 rules; all-Normal regresses exactly.
10. Identical fit data chooses identical rule; modifying qualification outcomes
    cannot alter the selected rule; no trace-key/queue state crosses episodes.
11. A simulation-spy test proves no policy reads future draws, sampled residuals,
    completion outcomes, or evaluation results to choose an earlier request.
12. No-Hedge remains an eligible candidate; a negative control result is valid
    evidence and cannot trigger automatic cost/budget/seed changes.

## 12. Implementation sequence after design disposition

1. Reconcile ADR-0011 and the existing spec/tickets; freeze the new objective,
   ledger and solution label. Do not implement the old 36-cell solver scope.
2. Add the proposed ledger, causal observation contract and conservation tests,
   reusing the accepted engine and episode infrastructure.
3. Add prospective fractional-tail objective and the 81-rule enumerator;
   validate hand fixtures and historical API compatibility.
4. Run development timing/selection/qualification with exact queue dynamics.
5. Decide from these results whether a reduced forward model is justified;
   validate it before any mean-field policy solver.
6. Design the revised attribution arms and confirmatory statistical protocol,
   then freeze independent holdout. The present design runs none of these steps.

## 13. Sources and scope of support

The queue conservation, ledger and reference bank above are proposed design
choices. Literature supports the distinctions, not validation of this implementation.

- Dependence and burstiness matter beyond mean traffic intensity: Whitt and You,
  https://www.columbia.edu/~ww2040/robust_WY_101016.pdf
- CVaR variational representation: Rockafellar and Uryasev,
  https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf
- Optimization and equilibrium have different limiting/control interpretations:
  Carmona, Delarue and Lachapelle, https://arxiv.org/abs/1210.5771
- Finite-player approximation is a separate result, not merely fixed-point
  residual: Carmona and Delarue, https://arxiv.org/abs/1210.5780
