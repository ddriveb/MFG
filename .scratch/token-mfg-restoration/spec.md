# Token-player protection game: completed design and explicit MFG extension

Date: 2026-09-06

Status: ADR-0020 accepts Token identity and T1 online execution/Reservation only;
T1 tickets02/03 are resolved. Later preferences, price mechanisms, deviations
and mean-field/scaling sections below remain proposals. The detailed T2 choice
of three named private costs and finite pathwise deviations is now submitted in
[Proposed ADR-0021](../../docs/adr/0021-token-private-cost-and-deviation-semantics.md)
and [design ticket04](issues/04-design-token-payoff-and-deviation.md).
No later implementation or campaign is activated by this file.

## 1. What is preserved, and what is being completed

The question is whether price-mediated Token protection, responding to the
population it creates, adds value beyond NIIN, quota and strong causal rules.
Each logical Token request is a player. Its Primary/Hedge/Replay attempts are
parts of the SAME player. An Expert controller implements policies on behalf
of Tokens; it does not thereby become the utility-maximizing player.

Expert selection is fixed by Gate. Replicas are same-Expert execution resources.
Neither Expert rerouting nor a cross-Expert shared processor pool is necessary.
An Expert-level aggregate objective and a single-Token objective are different
games even when the same function is called once per arrival.

The original text specified a mechanism outline, not a closed queueing MFG.
This document supplies a complete finite information/action/payoff contract,
an operational feedback-price contract, and a specific candidate scaling/forward
closure. It does NOT assert an existence/uniqueness or epsilon-Nash theorem.

## 2. Finite reference: restore the historical physical system

Use the latest original two-Replica transient reference, rather than importing
the later fixed-Primary three-Replica Shared-Backup topology:

| Quantity | Finite reference |
| --- | --- |
| Experts | one initially; multiple Experts are resource populations, not players |
| Replicas | A/B, one FCFS server each |
| Arrivals | Poisson rate 0.9, on [0,320) |
| Class | Regular .8 / Urgent .2 |
| Required work per attempt | independent lognormal, mean 1, CV .5 |
| Dispatcher | Token ID parity outside F; F arrivals use B |
| A speed | H/R: 1; D: .5; F: 0 |
| B speed | 1 |
| Scheduled reference | D at100, F at200, R at220, known scenario |
| Actions | N / D / I; only D arrivals with Primary A are eligible |
| Delayed time | healthy service .9 quantile, not inherited Stage4 delay1.5 |
| Budget | shared by Tokens at this Expert, D-relative windows25, cap2.8125 |
| Charge | N:0; admitted D/I:1 reserved mean-work unit, no refund |
| Drain | all copies, including running losers, until physically complete |

This choice follows the transient reference. Earlier paired comparisons allowed
other eligibility/class-share semantics and must remain separately named.
At additional Experts, keep Gate/routing exogenous and publish each resource
population's arrival rate, budget and capacities. No cross-Expert capacity
coupling is assumed without an actual declared shared resource.

### Stochastic common-fault extension

A separate namespace may use D start100, D duration U[75,125], F duration
U[10,30], arrivals still stopping at320. These are an explicit prospective
extension, not the interpretation of historical scheduled traces. Observe only
current/past health and phase age s. The conditional hazards are

    q_D(s)=0 (s<75), 1/(125-s) (75<=s<125);
    q_F(s)=0 (s<10), 1/(30-s) (10<=s<30).

The phase changes before the upper endpoint almost surely. H->D is a known
time boundary. R is a scoring phase with healthy physical speed.
Random-fault windows receive their FULL nominal cap at opening; stop admission
when failure is observed. Never prorate by an unrevealed future failure time.
Late D means elapsed age>=50, never half of the unknown realized D duration.

## 3. Players, information and one-shot decision timing

Player i is born at t_i with fixed class k_i, Expert e_i, Primary p_i. It chooses
one N/D/I action at arrival; D commits to a timer, not to a second decision.
N cannot later be changed to I. Winner/cancellation/Replay are automatic.
Online feedback means DIFFERENT arriving Tokens react to the current state.
Adding waiting-time redecisions would be a separate optimal-stopping game.

Its marked state is

    x_i=(e_i,k_i,t_i,p_i,u_i_requested,u_i_applied,
         {(attempt role, replica, status, enqueue priority, executed work)},
         delayed deadline/status, replay flag, winner flag,
         work/waste accounting still needed for settlement).

Status distinguishes not-created, queued, running, failed, cancelled, finished.
At most one Replay; no Dual in this action bank. Enqueue priority preserves
ordered FCFS and simultaneous Replay order. Keep the Token after its winner
if a running loser remains: it still occupies capacity and incurs work.

Finite physical state Y additionally contains both ordered queues, running
copies and true work draws, all links/timers, budget, price-controller memory,
absolute clock and public phase history. True work draws belong only to Y.

The arrival observation O_i may include its class/Primary, public time/phase
age, observable queue counts/order and already-executed head progress, budget,
posted price and the published population estimate. It contains no future
arrivals/faults, sampled total or remaining work, future winner, or other
Tokens' future outcomes. Future scenarios in a BR branch from observed history.
Policies/class labels are prescribed; bidding, lying about urgency and choosing
Gate output are outside the game.

For service S already executed by e without completion:

    P(S-e>v | S>e)=survival(e+v)/survival(e),
    beta(e)=f_S(e)/survival(e), completion intensity=speed*beta(e).

Executed work, not wall-clock age, is the correct service-age coordinate when
speed changes. Independent service assumptions make this residual law useful;
general hidden dependencies require a full belief on Y conditional on O.

## 4. Exact event physics and admission

For each replica r, pathwise remaining work obeys

    V_r(t)=V_r(0)+A_r(t)-E_r(t)-C_r(t)-D_r(t).

E includes Primary/Hedge/Replay and running losers. Queued cancellations remove
remaining work but cost zero executed work. Failed executed work remains waste.
First completion wins; queued losers cancel, running losers are nonpreemptive.
A failure removes A copies; if no winner and no live Backup, enqueue Replay on
B and void a pending Delayed timer. Existing live Backup prevents Replay.

Process same-time events in the existing order: fault/Replay -> completions ->
timers -> arrivals -> dispatch. Include full loser drain beyond cutoff.

Let q(N)=0, q(D)=q(I)=1. With current balance B:

    applied Gamma(u,O)=u if eligible and B>=q(u), otherwise N;
    B_after=B_before-q(applied), B reset to declared cap at window open.

Record requested/applied/executor-suppressed separately. A rejected D/I incurs
no reservation charge and follows actual N, including possible Replay. A
Delayed admission is charged now even if later voided. Use the same projection
inside BR evaluation and the execution engine, never only after solving.

Independent mixed actions use a private per-Token uniform CRN key (inverse CDF
on N<D<I). Reusing the old deficit allocator would change independent mixing
into a correlated prescription and add allocator memory to the game; keep it
as a named comparator, not silently as the implementation of mixed Nash.

## 5. Individual utility and separate system evaluation

Proposed base private objective at information O_i:

    Q_i(u | O_i, environment)
      = E[L_i + alpha_k (L_i-d_k)+ + gamma_k 1{Replay_i}
          + c_W W_i + c_waste W_i_waste
          + posted_price(t_i)*q(applied_i) | O_i, request u].

Set d_R=3,d_U=2, alpha_R=gamma_R=1, alpha_U=gamma_U=5,
c_W=c_waste=1 as explicit untuned starting preferences. These combine the
latest latency/SLO/work preferences at TOKEN scope. They are not the historical
solver's exact cost, which used mean latency, Replay and incremental work.
No extra class-mix weight multiplies a Token's entire private cost: class
frequencies belong to population aggregation. Price is in cost/reserved-work
units and is locked at admission, even if Delayed starts later.

Each player accounts for its OWN requests' copies only. It does not include
other Tokens' loss in its BR. Prices are mechanism transfers/virtual penalties
and are excluded from physical social performance.

Retain the historical pooled system score for attribution:

    J_sys = (1/4) sum_z sum_k w_k
              { E_zk[L]+alpha_k E_zk[(L-d_k)+]+gamma_k P_zk(Replay) }
            +(CVaR95_D+CVaR95_F)/2
            + E[sum_i W_i]/E[N] + E[sum_i W_i_waste]/E[N].

Here E_zk[f]=E[sum_(i in z,k) f_i]/E[N_zk], NOT mean of per-episode ratios.
Use exact fractional tail mass. Missing pooled cohorts make J incomplete.
The different phase weights and system CVaR mean J_sys is NOT automatically
the mean private utility. This is deliberate and must be reported in welfare
comparisons. Do not optimize an individual Token by assigning it the full
population CVaR, nor claim Nash minimizes this system score.

If private CVaR is later desired, define it over this Token's conditional future
loss at arrival, choose an ex-ante threshold, and solve that separate risk game.
No such private risk extension is required by this first restoration design.

## 6. Price mechanism: complete feedback law without a false KKT claim

Keep hard FIFO reservation as the safety actuator. It already caps admitted
work, so updating price from `admitted_work-cap` alone cannot detect rejected
demand and can falsely look cleared when almost everyone is suppressed.

Proposed first price mechanism is a causal posted-price pacing controller,
NOT a declared Walrasian market or capacity dual. At window open: price=0;
at predeclared subinterval times t_j, let h_j be time to the next update or
nominal window end. Let B_j be balance at the START of this interval. Compute

    target_j = B_j * h_j/(nominal_window_end-t_j),
    R_req,j  = sum q(requested_i) over eligible arrivals in the interval,
    price_(j+1) = max(0, price_j + alpha*(R_req,j-target_j)/K).

For finite reference K=1. Price_j is held until the update; charge admitted
Tokens only. Alpha>0 and update interval Delta>0 are explicit controller
parameters to be frozen on development data before any confirmatory run.
After budget is exhausted requests may still be recorded and projected to N;
at phase F admission is disabled; at the next window price resets to0.
Never divide by a zero remaining duration. At an unexpected failure stop the
D controller using observed time, do not retrospectively change target/cap.

This is an endogenous causal price depending on population requests. Given
this fixed mechanism, a Token equilibrium is well defined. The controller's
price trajectory need not converge to a constant in a transient episode.
Its update residual and admitted-budget feasibility are auditable identities,
not equilibrium/market-clearing certificates.

An alternative shadow-price formulation would need a specified ex-ante
reservation demand, supply conditional on available information, a clearing
mechanism and 0<=p perpendicular supply-demand>=0. Hard clipping of actual
admissions is not such a demand equation. Do not mix the two price meanings.
Also do not price total execution with this reservation price; Replay and
Primary consume service capacity but do not debit Hedge reservation.

## 7. Best response and solution labels

At arrival, for given causal population/price continuation law:

    V_entry(O)=min_(u in N,D,I) Q(u|O),
    support(pi*(.|O)) subset argmin Q(.|O).

Actions outside eligibility are canonicalized to N. Budget rejection is inside
Q. Equal-cost admitted/rejected aliases can tie; choose N before D before I
for a deterministic pure reference. Pure fixed-point convergence is not promised.

With explicit entropy temperature theta>0:

    min_pi sum_u pi_u Q_u + theta sum_u pi_u log(pi_u),
    pi_u=exp(-Q_u/theta)/sum_v exp(-Q_v/theta).

This is a REGULARIZED equilibrium. At one decision, its raw expected-cost regret
is at most theta*log(3) with exact Q; numerical, field and sampling errors remain
additional. Report the measured unregularized regret; no `softmax = exact Nash`.
The one-shot action set explains why a finite-action BR is sufficient initially.
There is no continuous min over new actions after every timer/completion event.

## 8. Exact finite population versus an actual mean field

The exact conditional joint law is

    Gamma_t = Law(Y_t | public fault history through t).

Its forward equation uses the full queue generator and the prescribed Token
policy. Each private cost is evaluated by conditional tagged-Token deviations.
This completely defines a finite stochastic queueing game (with arrival-time
information) even though computational solution can be difficult.

A low-dimensional Token marginal m=(Hedge fraction,rho,queue count) is NOT
an exact closure: FCFS order, work ages, cross-copy cancellation and Replay
batch priority affect future transitions. Empirical measures with unique
queue/link marks can encode far more state, but ignoring their correlations
does not magically produce a negligible-player limit.

### Candidate MFG scaling: many executors inside fixed resources

For a real Token mean-field study, propose a separate family indexed by K:

- Keep one Logical Expert and A/B failure domains; each domain has K identical
  executors and ONE FCFS queue. At K=1 this is the original reference.
- Arrivals have rate .9K; class/service distributions, each executor's speeds,
  deadlines, delay, event clock and Token payoffs stay fixed.
- Budget per window is K*2.8125, each admitted D/I still reserves1.
- Dispatcher remains the fixed alternating rule outside F and B in F; no
  policy-dependent routing. Its limiting primary proportions are .5/.5.
- All A executors share the original fault; all B executors remain healthy.
  Replay batching uses running first, then FCFS, with declared stable ties.
- Use K=1,2,4,8,... as resource scaling, NOT number of Experts or more Monte
  Carlo episodes. Merely adding Tokens to two fixed servers is overload.

This is a NEW explicit scale extension, not a claim that current hardware
already contains K executors. It preserves individual O(1) service times and
allows individual resource influence of order1/K away from singular events.
That alone does not prove convergence through quota boundaries/common jumps.
K=1 reservations consume 1/2.8125=35.56% of a window; finite-player effects may
be large. Many-server pooling can remove ordinary single-server waiting in
underloaded periods; validate the approximation rather than transferring
historical P99 improvements to the limit.

## 9. Candidate forward system: marked Token mass, birth and settlement

In the scale family use an unnormalized live Token measure:

    m_t^K = (1/K) sum_(i not fully settled at t) delta_(x_i(t)),
    m_t = conditional large-K limit, if it exists.

Its total mass varies with arrivals/settlement; do not normalize it to1 at
every time step. Keep winner-known running-loser states in m. Retain separate
cumulative cohort outcome measures for tail metrics after Tokens leave m.

For each domain r define from m:

    n_r(m)=integral number_of_running_r_copies(x) m(dx),
    q_r(m)=integral number_of_queued_r_copies(x) m(dx),
    d_r(m,z)=integral sum_(running r) mu_r(z)*beta(e_l) m(dx).

Capacity and non-idling: 0<=n_r<=1 and q_r*(1-n_r)=0 on healthy/degraded
domains after dispatch. A has no service/dispatch in F. Let K_r(dt,dx) be the
entry-into-service measure supported on queued r copies, selecting oldest
enqueue priorities first. In smooth intervals dn_r=dK_r-d_r dt (plus any
specified other running exits); common failures/reset maps are separate.
Available idle slots are filled immediately, possibly via atoms of K_r.
Cancellation removes queued peers through the SAME Token's state map.
At a common Replay pulse, enqueue timestamp alone is insufficient; preserve
the running-first/old-queue-order batch rank. Fractional dispatch at an atom
uses that priority order, not random overtaking.

For a test function phi zero at the fully-settled cemetery state:

    d<m,phi> = <m,F_x dot grad(phi)> dt
       + <m,sum_l mu_r(z)*beta(e_l)[phi(C_l x)-phi(x)]> dt
       + .9 * integral phi(A_(z,k,p,u,Gamma) newborn) w_k p_z(dp)
                    pi(du|O) dt                         [t<320]
       + sum_r integral [phi(Start_r x)-phi(x)] K_r(dt,dx)
       + deterministic timer-boundary transfers.

F_x increments request ages, executing work and timers. C_l determines winner,
cancels queued peers, or settles a running loser; the state goes to cemetery
only when no physical liability remains. A includes admission/creation;
newborn service enters through dispatch, not an invented exponential wait.
The FCFS entry measure and its boundary priorities are part of the closure.
Replacing it with a scalar wait-rate is a DIFFERENT approximation to calibrate.

With phi=1 on live states, births minus fully-settled exits give the mass law.
At common failure, m+=(T_F)#m- with settled states removed; drop A work,
insert B Replay, void relevant timers and then dispatch. R changes speed but
does not restore discarded attempts. At cutoff the birth source becomes0;
continue work/mass evolution through full drain.

Normalized budget b=B/K obeys db=-g_admitted dt in the fluid intervals,
b resets to2.8125, and admission stops when b reaches0. Delayed consumes
reservation at arrival, not timer launch. Price feedback is section6 in
normalized units. At finite K always use the exact indivisible ledger.

This is a specified measure/boundary problem, not a claimed finite-dimensional
PDE implementation. Paired attempt marks and common shocks make its analysis
more involved than standard single-queue many-server fluid limits.

## 10. Tagged HJB and self-consistency

For fixed candidate population feedback, let y include public phase/age,
predicted marked field, normalized budget, price, update clock and current
request-demand accumulator. In a Markov realization its public generator
G_env is fixed during tagged BR. For the tagged Token after action commitment,
the continuation value satisfies formally, away from deterministic boundaries:

    0 = partial_t V + G_env V + F_x dot D_x V + c_W sum_l v_l
        + sum_(running l) v_l beta(e_l)
            [c_l(x)+V(C_l x,y)-V(x,y)]
        + q_z(s)[c_F(x)+V(T_z x,T_z^env y)-V(x,y)].

Here G_env excludes the explicitly written common-fault jump to avoid double
counting. Its drift includes the fixed field, ledger and price-controller
memory evolution; derivative in m is formal on measure space. At FCFS entry,
timer/price/budget boundaries, apply the corresponding value matching map.
At winner completion, c_l includes L+alpha(L-d)+ and the Replay penalty;
at nonwinner retirement c_l includes c_waste*executed work. Work accrues
continuously for all live copies. Fully settled value=0, not merely winner
value=0. There is no artificial V(320)=0 for backlogged Tokens.

Entry action value adds posted_price*q(applied) plus continuation from the
admitted state, then takes section7's minimum or declared entropy response.
Common-fault branches use history-adapted laws, never a revealed future trace.
Scheduled-reference branches use their explicitly known deterministic clock.

Close the candidate system with

    pi* = BR(m*, b*, price*; fixed causal mechanism),
    m*  = Forward(pi*, admission*, FCFS*, common-fault history),
    b*  = ReservationLedger(pi*,m*),
    price* = PostedPriceFeedback(pi*,m*,b*).

Consistency concerns the entire time/history-indexed trajectory, not the false
requirement m_t=m_(t+1) in a transient queue. Numerical iteration index must
be distinct from physical time. Existence, uniqueness, regularity, convergence
and finite-K error are separate obligations, not supplied by these equations.

## 11. Implementation-facing validation contract

1. Engine: original N/D/I physics, event ties, work conservation, failed/queued/
   running loser costs, complete drain, request/admit/launch identities.
2. Information: equal observed prefixes yield equal decisions with identical
   private policy keys despite different hidden future service/faults.
3. Admission: BR and execution agree with budget exhaustion, Delayed charge,
   denied Hedge->N and Replay; expose the class that consumes scarce slots.
4. Token deviation: hold others' POLICY FUNCTIONS fixed, change only one
   Token's request, and rerun the full finite system, including others' online
   reactions, budget, price and queues. Do not swap an Expert's whole rule.
5. Conditional regret: estimate Q_u(O) by independent continuation samples or
   predeclared observable bins; then compare expected costs. Never average
   per-trace min_u realized_cost(u), which grants hindsight. Restricted bins/
   policy classes yield restricted regret, not unrestricted Nash.
6. Mean field: independently test marked distributions, service-start flow,
   Replay pulse, budget and action-conditioned costs; refine state/time/particle
   resolution. If only a fitted (h,rho,q) surrogate is built, label it as such.
7. Scale: N stays a Token count; K is executor scale. Compare finite K and the
   candidate field, regret and queue tails. Independent episodes reduce MC
   error but do not reduce the effect of one Token in a K=1 queue.
8. Attribution: full controller vs frozen NIIN, same-budget urgency/risk/load
   baselines, no-price and no-field variants, and a centralized comparator
   with stated information/optimization limits. Same physical traces/action
   bank/reservation cap; actual work matching uses development calibration or
   a predeclared resource frontier, never holdout tuning to a realized count.
9. Inference: separate fit/validation common-fault paths; CRN pair across arms;
   retain path clusters and pooled nonlinear tail scoring. Adequate Token count
   does not replace independent common-fault paths. Report native units and SE.

## 12. Sources and boundaries of attribution

Local basis: CONTEXT; ADR-0005/0006/0009/0012; transient-model-design; the user's
pasted Token-MFG attribution text. No historical numerical result is updated.

- [Kang and Ramanan, fluid limits of many-server queues](https://arxiv.org/abs/1011.2921):
  service/waiting-age measures and growing arrivals/server counts motivate the
  scale construction. Their model does not prove our linked-copy fault model.
- [Burzoni and Campi, absorption and common noise](https://arxiv.org/abs/2107.00603):
  conditional mass loss is a relevant formulation precedent. Our immigration
  and physical settlement definitions are derived here, not their theorem.
- [Guan et al., entropy-regularized MFG](https://arxiv.org/abs/2110.07469):
  entropy regularization specifies a different game; do not borrow their
  finite-state convergence guarantees for these unbounded queues.

The essential new assumption is section8's scale family. Until its forward
closure and finite-system error are validated, the deliverable is a Token
queueing-game / regularized mean-field controller candidate, not proven MFG.
