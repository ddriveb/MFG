# Shared-Backup Expert game: finite physics before a mean-field equilibrium

Status: Proposed mathematical design, 2026-09-05.
Design disposition: Proposed ADR-0015. Implementation and experiments are not
activated by this document. Ticket 01 is the design-only work record.

## 1. Scope and changes from the implemented baseline

ADR-0014 implements eight conditionally independent Expert queue triplets.
Each Expert has fixed Primary A and Backups B/C, not three Active-Active
Primaries: H/D/R arrivals use A; F arrivals and Replay use B/C. Preserve this
topology, fixed Gate, work distributions and conservative cancellation.

New coupling: all Expert B/C executions draw compute from one shared pool.
B/C remain separate failure domains and separate same-Expert FCFS queues;
the pool models a common throughput bottleneck, such as a shared compute
allocation. This is a deliberate physical resource assumption, not a claim
that present deployed replicas necessarily share such hardware. No extra
failures of this pool or of B/C are introduced in this version.

Players are Logical Expert controllers i=1,...,N, not individual Tokens.
Each controller chooses protection for its own arriving Tokens and minimizes
its own multi-Token episode-distribution loss. Arrivals/Gate are exogenous.
Retain independent per-Expert budgets; the shared server allocation alone
creates a strategic interaction. Public faults alone do not create a game.

Initial physical parameters (proposed development defaults):

| Quantity | Value |
| --- | --- |
| N | 8; size diagnostics 16,32,64 |
| Per-Expert arrival rate | 0.45 / time |
| Work | lognormal, mean 1, CV 0.5, independent copy keys |
| Classes | Regular .8, Urgent .2 |
| A speed | H/R:1, D:.5, F:0 |
| B/C maximum speed per running copy | 1 work/time |
| Shared pool capacity | K_N=N*c_B; main c_B=.5 work/time per Expert |
| Capacity diagnostics | c_B=.25 (F base overload), 1, 2 (uncoupled) |
| Arrival cutoff | 360, followed by complete drain |

The new cutoff, shared resource and later random fault law define a new run
family. Historical NIIN numbers cannot be carried over as new-system results.
At c_B=.5, F-arrival base work is .45N against capacity .5N; Replay can cause
transient overload. A rate comparison is not a queue prediction. c_B=.25 is
structural F overload, not a Hedge-budget failure. These values are not fitted.

## 2. Exact shared capacity and queue conservation

For r in {1,2}, let b_ir(t)=1 when that local Replica has a running head copy,
including a running loser, and 0 otherwise. Dispatch nonempty idle queues
before assigning speeds. Define

    K(t) = sum_i sum_(r=1,2) b_ir(t),    k_N(t)=K(t)/N in [0,2]
    h(k) = 1                          if 0<=k<=c_B
           c_B/k                     if k>c_B
    v_ir(t) = b_ir(t) h(k_N(t)).

All numeric speeds use normalized units with maximum speed 1. This is equal
processor sharing across active Replica heads, with an individual speed cap;
FCFS is retained inside each Replica. It is not a global FCFS queue and does
not prioritize Hedge over Replay or F Primary. Sharing can change speeds
without changing the identity of a running copy.

    sum_i,r v_ir(t) = min(K(t),N*c_B) <= N*c_B.

When only one head is active it can run at speed 1 if pool capacity permits;
idle Expert shares are borrowable. A dual launch can obtain two active head
shares; this allocation rule intentionally allows that competitive incentive.
Per-Expert fair sharing is a different future scheduler ablation.

Count ALL work on B/C: F-arrival Primary, Replay, Hedge and running losers.
Never charge only Hedge against physical pool capacity. Do not add a separate
analytic waiting-time penalty on top of simulated queue delay.

For each of the 3N queues, retain the exact eventwise identity

    V_ir(t)=V_ir(s)+A_ir(s,t]-E_ir(s,t]-C_ir(s,t]-D_ir(s,t].

Here A is actually enqueued required work, E is executed work, C removes
queued losers' remaining work, and D discards failed remaining work. For B/C,
E_ir is the integral of v_ir. For A, integrate the health-dependent speed.
Previously executed failed work stays in total work and waste. Sum individual
B/C execution integrals to audit the shared capacity on every event interval.

Use distinct observables with correct units:

- k_N: active heads per Expert, dimensionless; sufficient to set current speed.
- V_B^N/N: queued/running remaining work per Expert; internal audit only.
- rho_offer(w): actually enqueued B/C work in w divided by N*c_B*length(w);
  a work-pressure diagnostic that may exceed 1 and includes Replay jumps.
- U_B(w): executed B/C work divided by N*c_B*length(w); always <=1.

Backlog divided by capacity is time, not a dimensionless utilization. A
cumulative lifecycle W cannot be substituted for instantaneous offered rate.
k alone is not a closed dynamical state; the full local-state distribution
is needed to predict its future values.

## 3. State, observations and event integration

Private physical state Y_i contains all three ordered marked queues; running
copy requirements and executed progress; Token-copy links; arrival/class marks;
winner and Replay flags; pending timers; local budget and window anchor;
local Token counter; and accrued outcome/work records needed for scoring.
Queue counts alone do not specify the transition law. The current Token class
is an arrival-event mark, not a sufficient summary of all queued classes.

Global physical state is (Y_1,...,Y_N,Z,phase_age,clock). R remains a metrics
phase and physical health H, not a new CommonState enum value.

The controller observes its own queue identities/counts, service progress
already executed, elapsed ages, copy/timer/winner status, class and budget;
public health/phase history, clock and k_N. It cannot see true remaining work,
future arrivals, future faults, or other Experts' private state. With variable
speed, residual-service inference conditions on cumulative executed work e:
Law(S-e | S>e), not just wall-clock age. The physical simulator may retain S;
a policy interface must not expose it. An exact partially observed best
response uses a belief state or full admissible observation history.

Event loop contract:

1. Advance all running service integrals to the earliest event using OLD speeds.
2. Process health change, A failures and Replay; then the batch of valid
   completions; then timers; then arrivals; finally dispatch, as ADR-0005.
3. Batch equal-time valid completions before redispatch. For simultaneous
   winners of one Token, use (Replica ID, attempt ID) as deterministic tie order.
4. Recompute K and every B/C speed after the whole event batch. Reschedule
   prospective completion events with generation IDs; discard stale events.
5. Split ledgers at phase, budget and cutoff boundaries. Drain until all
   accepted work, including running losers, finishes, not merely until winners.

Failure Replay order within an Expert is running then FCFS; cross-Expert
processing uses Expert ID, without interleaved service or dispatch. Therefore
the cross-Expert event-loop order cannot allocate extra compute to a low ID.
If an event time equals a budget reset, reset before that time's arrival
admission, but after any same-time failure has determined action feasibility.
Policy feedback must execute online, not through the old metadata prepass.

## 4. Actions, destinations and local admission

Only a D arrival with Primary A can request protection. Four canonical actions:

| Code | Action | Execution | Reservation charge |
| --- | --- | --- | --- |
| N | Normal | Primary only | 0 |
| D | Delayed | one Backup at arrival+tau0 if still eligible | 1 |
| S | Single | one immediate Backup | 1 |
| X | Dual | two immediate Backups, one on each B/C | 2 |

tau0 is the existing healthy service .9 quantile in time units at speed 1.
Delayed dual is outside this first bank. H/F/R request N. Single/Delayed,
F-arrival Primary and Replay use the same fixed destination mark in {1,2}.
Primary=attempt0, Replay=1, first Hedge=2, second Hedge=3. For a dual action,
first Hedge uses that mark and second Hedge the other Replica, ensuring its
first-Hedge draw is exactly shared with Single on the same Token.

Retain first winner, queued-loser cancellation, nonpreemptive running losers,
no new copies after a winner, and at most one Replay. A failure with any live
Backup creates no Replay. A failure before a Delayed launch voids its timer
and creates Replay. Fault-first boundary semantics remain unchanged.

Per Expert, D-relative windows have width 25 and cap 2.8125. Debit at request,
without refund. Insufficient balance converts the ENTIRE request to Normal;
Dual is not silently downgraded to Single. A pending timer retains the window
charge in which it was requested. Budget does not scale with N per Expert.

In the random-fault case a window that ends early at failure still received
its full cap at its start: prorating by the realized future failure time would
leak information. Fault disables future D admissions; reservations remain in
the audit. All mechanisms and arms use the same rule. An endogenous global
quota would create a second coupling and is deferred.

## 5. Common noise and causal scenarios

Two explicitly separate scenarios use arrivals on [0,360):

- scheduled diagnostic: H->D at100, D->F at200, F->R at220, all known.
- stochastic common-noise development: D starts at100; D duration U[75,125];
  F duration U[10,30], independently sampled on a shared fault stream.
  Every Expert sees transitions when they occur; neither duration is revealed
  in advance. B/C stay healthy. Maximum recovery time is255.

Distributions are public; elapsed duration conveys survival information.
An observation model must include phase age, because health alone is not
Markov for these duration distributions. Early/late D means elapsed D age
less than50 or at least50, never the first/second half of a future duration.

Let F_t^0=sigma(public fault/phase history through t). This includes observed
phase age, not the pre-generated entire fault trace. The candidate mean field
is adapted and satisfies

    m_t = Law(Y_t | F_t^0).

Conditional simulation may reuse the same whole common trace across particles
to evaluate a frozen causal policy. A best response may not optimize with
foreknowledge of that trace; its continuation scenarios must branch over
future faults consistent with the currently observed history. Conditioning on
Z_t alone omits history; conditioning decisions on future Z leaks information.
For the first rule bank decisions do not inspect future scenarios at runtime.

Scheduled faults produce a deterministic environment law in the limit and
are not evidence of a nontrivial common-noise solution. A strong adapted
equilibrium is a target, not an existence assumption verified by simulation.

## 6. Individual risk objective, social welfare and prices

For player i, let N_i,z,k count its arrival-phase/class cohort. Define pooled
population expectations over independent episodes as

    E_i,z,k[f] = E[sum_(j in i,z,k) f_j] / E[N_i,z,k].

All denominators are exogenous and policy-independent, including under random
phase duration. Do not omit empty individual episodes then average ratios.
Missing empirical pooled cohorts make the score incomplete, not zero.

Use the ADR-0012 preferences applied to EACH player's own Token population:

    Loss_i,z = sum_k w_k (E_i,z,k[L] + alpha_k E_i,z,k[(L-d_k)+]
                         + gamma_k P_i,z,k(Replay))
    C_i,z = inf_eta {eta + 20 E_i,z[(L-eta)+]}, z=D,F
    J_i^0 = (1/4)sum_z Loss_i,z + (C_i,D+C_i,F)/2
            + E[sum_j W_ij]/E[N_i] + E[sum_j W_ij^waste]/E[N_i].

w=(.8,.2), d=(3,2), alpha=gamma=(1,5). Excess is overrun magnitude, not
SLO violation probability; report both. Tail statistics include exact
fractional boundary mass. Waste adds a deliberate preference, not extra
physical work. Keep per-class tail diagnostics and fixed H/R safety metrics.

This is an ex-ante commitment game over causal policies. CVaR is a functional
of the latency distribution, not a per-event reward inside an expectation.
For a more general optimizer one can jointly optimize policy and eta_D,eta_F
using the variational expression; do not insert ordinary additive Bellman
recursion without handling risk state/thresholds and information. Re-solving
conditional CVaR at each arrival defines a different time-consistency problem.

The primary game is unpriced: J_i=J_i^0. Physical sharing already creates
congestion and strategic incentives. Avoid assuming a market is needed to
make the game nondegenerate.

Optional, separately named tariff ablation:

    p_t = kappa * max(k_N(t)-c_B,0)/c_B
    J_i^p = J_i^0 + E[integral_0^drain p_t dE_i,B(t)]/E[N_i].

Price has cost/work units; kappa=1 is an untuned proposed diagnostic. Charge
all actual B/C work in its execution phase, including Replay/F Primary and
losers. The same published tariff rule applies to every arm and recomputes
under finite deviations. This is an occupancy tariff, NOT a capacity dual,
market-clearing price, or demonstrated marginal social-delay toll. Its
definition residual is checked; price complementarity is not applicable.
An actual market requires specifying allocated demand, seller clearing,
constraints and transfers in a further ADR; scheduler feasibility alone does
not identify a unique shadow price or justify complementarity equations.

Social cost for comparison is S_N=(1/N)sum_i J_i^0. Tariff payments are transfers
and excluded from this welfare score, reported separately. Also report the
historical pooled-across-Experts J as a distinct diagnostic: in general CVaR
of a mixture differs from the average of Expert CVaRs. Under symmetric
population laws they agree at population level, not necessarily in samples.
Any centralized comparator minimizes this same S_N and uses the same resource,
budget, action and fault-information contract. It can observe all allowable
local observations but never true future service or faults; disclose its
information advantage over decentralized controllers.

## 7. Population scaling and candidate mean-field dynamics

For each N, per-Expert arrivals, work distribution, budget, local speed caps,
action set and fault law remain fixed; shared capacity grows as N*c_B and
total arrival rate grows as .45N. No change to policy with Expert index.

Generate independent Poisson(.45) streams for each Expert and merge them;
this has the same uniform exogenous Gate law as marking a Poisson(.45N)
stream, with easier nested size comparisons. Draw destinations with an
independent stable per-Token Bernoulli mark instead of old global-ID parity.
The new family explicitly changes that deterministic routing coupling; old
arrays/results remain immutable. Merge assigns dense global IDs for output
only, never for randomness or routing. Across N, first eight local streams
are identical, though their queues change because competition changes.

Keys: (namespace, macro, common_episode, expert_id, local_token_id, replica,
attempt); plus distinct arrival/class/destination/private-policy streams.
Common faults exclude expert_id. Primary/Replay/Hedge draws are always keyed
by this tuple; decisions never consume or regenerate exogenous streams.

Define m_t^N=N^-1 sum_i delta_(Y_i(t)), and

    k(m)=integral [b_1(y)+b_2(y)] m(dy).

The representative Expert uses B/C head speed h(k(m_t)); its A dynamics,
local arrival process, action projection and copy links remain exact. Between
common jumps, describe the forward evolution weakly by the marked-queue
generator L_(m_t,pi,Z_t), or simulate particles with exactly those transitions.
For stochastic observations/beliefs, augment the state with the policy memory.
At an observed failure, apply the local failure/Replay map T_F simultaneously:

    m_t = (T_F)# m_(t-).

The jump removes A work and inserts B/C Replay; recompute active heads after
dispatch. Do not diffuse this pulse into previous D service. A recovery changes
A speed and dispatch feasibility; it does not restore discarded attempts.

Candidate equilibrium (with tariff rule fixed at zero or the named ablation):

    pi* in argmin_(causal pi) J(pi;m*,p*)
    m*_t = Law(Y_t^(pi*,m*,p*) | F_t^0)
    p*_t = P(k(m*_t)).

In a nonatomic best response the environment is held fixed; in the finite
game own actions alter k_N and others' outcomes. At a fixed configuration an
Expert changes the head statistic by at most2/N; h is Lipschitz for c_B>0.
These facts motivate the scaling but do NOT prove path-law convergence,
equilibrium existence/uniqueness or an epsilon-Nash theorem for this model.
Unbounded lognormal work, FCFS identity links, partial observation, risk
objectives and common jumps require their own argument/error control.

## 8. Bounded solver before an unrestricted MFG claim

First freeze Pi_256: choose N/D/S/X separately for (early-Regular,
early-Urgent, late-Regular, late-Urgent), yielding4^4=256 causal rules.
They operate through each player's own ledger. The old single-Backup NIIN
maps to NSSN in this alphabet, not to a new fitted rule. NNNN is included.
This bank can vary immediate Backup multiplicity by context, unlike the old
fixed-backup-count 81-rule search. It remains a restricted rule game with
no local congestion feedback; its limitations must appear in every claim.

Initial solver: restricted symmetric pure-strategy best-response iteration.

1. Start independently from NNNN, NSSN and XXXX; retain all outcomes.
2. Under each population rule simulate its conditional particle environment
   across common scenarios. Use exact local queues and the shared rate rule.
3. Freeze that environment for tagged-Expert evaluation; evaluate ALL256
   deviations on independent tagged streams with paired common scenarios.
   Score the tagged Expert's own distributional objective, not population J.
4. Select minimum estimated J, then total work, then lexical N<D<S<X; replay
   the selected population rule to obtain the next forward law.
5. Stop at a candidate self-best response, a repeated rule/cycle or32 updates
   per start. Cross-check candidate environments and deviations on separate
   validation data; exact numerical ties do not prove statistical equivalence.

No convergence is promised. Cycles/no pure equilibrium within the visited
rules are valid outcomes and do not imply nonexistence over all strategies.
Finite-grid interpolation, particle time stepping or environment compression
must have separate refinement errors. Exact interacting particles have finite
population error even if their event integration is exact.

Do not call softmax a Nash solver: fixed-temperature entropy changes the
objective. A mixture of whole rules requires fresh definition/evaluation of
its pooled CVaR payoff; support on pure-rule minima does not by itself ensure
a risk-sensitive mixed equilibrium. No mixed-equilibrium existence theorem
is assumed. A later feedback policy class must be registered before fitting,
include current budget and local/public congestion, and receive a new
deviation certificate. Pi_256 results are not unrestricted MFG results.

## 9. Validation and claims

### 9.1 Physical interaction and numerical consistency

Require exact fixtures in section10; eventwise capacity, local conservation,
copy counts, work/waste and requested/applied ledgers. Check forward-law and
policy consistency on separate data. Price checks are tariff-definition checks
only unless a future explicit market model activates complementarity.

Report k(t), busy fraction, conditional queue-count distribution with overflow
bins, remaining-work audit moments, Replay pulse, suppression, and phase work.
Freeze feature bins and time/phase-age grids on development data. These
distances measure named features, not an unspecified metric on full queue law.
Include completed latency mean/CVaR and all work components: matching k alone
does not certify the forward approximation. Refine particle size and time/grid
resolution separately from changing finite-game N.

### 9.2 Finite-game unilateral deviations

For each i, keep opponents' POLICY FUNCTIONS fixed and rerun the whole N-player
system with every beta in Pi_256. Recompute service, queues, observations,
opponents' feedback decisions (if any), tariffs and local ledgers. Never freeze
opponents' realized action lists, the pool trajectory or a player's quota
outcomes when claiming a finite-system deviation test.

    epsilon_(N,Pi) = max_i [J_i(pi_i*,pi_-i*)
                           - min_(beta in Pi union {pi_i*}) J_i(beta,pi_-i*)].

Paired evaluation preserves arrivals, work, common faults and opponents'
private random seeds. It does not preserve endogenous completion times.
Evaluate all8 players for the N=8 primary certificate. At larger N either
evaluate all players or use symmetry to report a representative EXPECTED
regret, explicitly not a maximum-over-players certificate.

Predeclared development epsilon target: .01*J_i^0(No-Hedge population), a
player-specific positive exogenous-scenario baseline; report absolute and
normalized values. Validation requires simultaneous one-sided95% upper
confidence bounds for all tested improvement differences to fall below
their targets. A restricted bank gap lower-bounds unrestricted exploitability;
a small gap certifies only the tested class. A separate fit-trained feedback
challenger can falsify this restricted result's practical adequacy.

Freeze deviations/candidates before validation. For Pi_256 all deviations
are enumerated; simultaneous inference addresses the maximum selection.
If a best-response optimizer searches a larger class on fitting data, freeze
its challenger for independent evaluation; its observed improvement alone
cannot upper-bound the unknown best possible deviation.

### 9.3 Size and common-noise validation

Apply the SAME frozen limit policy at N=8,16,32,64 with K_N=N*c_B. Estimate
conditional feature errors and finite-game regrets with uncertainty. Do not
re-fit a new policy at every N and call that convergence to one equilibrium.

For each of multiple common-fault paths, simulate multiple independent
idiosyncratic populations at each N; compare conditional empirical laws with
an independently simulated large-particle reference under that same common
path. A single common path supplies no estimate of across-fault uncertainty.
The reference particle count is increased until its reported numerical error
is below the intended measurement precision, or results are inconclusive.

Smaller error/regret with increasing N is empirical support, not a theorem;
strict monotonicity at four noisy sizes is not required. A floor from the
restricted policy class or numerical error may remain. N=8 may fail even if
large-N approximation improves, and must retain its own qualification result.

### 9.4 Decision value and welfare

On identical new-system CRN evaluate NNNN, frozen NSSN, a newly fit homogeneous
social rule from Pi_256, any bounded heterogeneous centralized search with
the same actions/budgets, and the validated restricted equilibrium candidate.
Separate zero-tariff and tariff experiments; changing payment rules changes
the game. Include c_B=2 as a no-sharing-coupling control.

Primary decision reports: each Expert's D/F tail, Replay, per-class miss/excess,
work/waste, S_N, per-Expert cost dispersion/worst Expert, budget exhaustion,
pool busy/congested duration and per-Expert executed share. Equal sharing per
copy does not guarantee equal service quality per Expert.

Retain H/R mean<=1.01, P95<=1.02 and total work<=1.05 versus same-system NNNN
as deployment qualification preferences, with uncertainty. They are not
deviation restrictions: excluding a selfish but unsafe deviation would hide
exploitability. The unconstrained game may produce an unsafe equilibrium.
Label a safety-constrained centralized comparator separately from unrestricted
welfare minimization when their feasible sets differ.

A heuristic centralized policy supplies a feasible social-cost value, an
upper bound on the minimum cost in its declared feasible set. It is not the
true optimum or a certified maximum improvement. Report its observed gap
without claiming exact price of anarchy. MFG need not beat the centralized
planner or even NSSN; coordination and welfare are separate measurements.

### 9.5 Data separation and inference

New namespaces: shared-backup-game:v1:{test,fit,model-validation,deviation,
qualification}:...; no historical data becomes untouched evidence. This design
generates no data and freezes no confirmatory holdout. Record scenario, bank,
solver start, capacity, fault law, tariff, observation contract, key scheme,
engine/code/config hashes, all rule rows and failure reasons.

Use common-scenario clusters and never treat Tokens or Experts sharing a
fault/pool as independent.  For the frozen Stage 4 campaign, ADR-0018 requires
exact delete-one-common-path recomputation of every pooled ratio/quantile/CVaR;
both idiosyncratic population replicates stay inside the omitted or retained
path cluster.  Validation resamples the resulting paired jackknife
pseudo-values jointly across every Expert/rule for its max-t bound.  A cluster
need not define a standalone objective. Tail counts and independent cluster
counts are both reported. This numerical inference is not a finite-sample
mathematical guarantee. A future design using more population replicates may
predeclare a hierarchical population-within-path resampling protocol.

Development timing starts with4 episodes and the full256 bank; it establishes
cost only. Sample sizes for the multi-start, conditional-law and simultaneous
regret campaign require a separate precision/timing protocol before execution.
Do not reuse old64/128 sizing as a power guarantee for256 deviations. If
intervals cannot resolve the .01 normalized-regret target, report inconclusive;
predeclare any additional batches before looking at qualification outcomes.

## 10. Hand-computable acceptance fixtures

1. Cross-Expert interference: N=2,c_B=.5, all A unavailable and two unit-work
   B heads belong to different Experts. Both run at .5 and finish at2.
   Without Expert2's head, Expert1 finishes at1. Its ledger/actions unchanged.
2. Same fixture with requirements1 and2: first completion at2, survivor then
   accelerates to1 and completes at3. Executed pool work=3, no stale completion.
3. N=2,c_B=.5; three heads, one at Expert1 and two at Expert2, requirements1
   each: speed1/3 each, completion at3. Dual obtains two shares and delays peer.
4. N=2,c_B=.5; Expert2's running loser still contributes to K. A queued loser
   has zero executed cost and is removed before dispatch. Canceling queued
   work may prevent future sharing but does not erase executed work.
5. Idle pool: K=0 yields no divide-by-zero; N=1,c_B=.5,K=1 gives speed.5.
   At c_B=2 all B/C heads run at1; changing another Expert cannot alter service.
6. Replay burst: same total enqueued work spread over D versus inserted at F
   gives different speed/backlog paths. F Primary and Replay consume pool work.
7. Reserve2.8125: two Single charges fit; a Dual after one Single is suppressed
   to N; a Dual on empty balance history fits but leaves.8125; no refund.
8. Random fault ends a budget window early: no retrospective prorating; a
   future-duration edit cannot change earlier observations/admission. Identical
   fault prefixes produce identical prior actions with identical local history.
9. Failure/completion/timer ties preserve fault-first semantics; simultaneous
   B/C wins have one winner and deterministic work/waste; no dispatch mid-batch.
10. Change Expert1 policy with CRN: all immutable traces and others' policies
    stay fixed; others' latency can change. The deviation evaluator must rerun
    the shared scheduler and not replay cached population service rates.
11. N=8 vs16 preserves the first8 local trace keys/draws, not completions;
    c_B and arrival rate per Expert and local budgets remain unchanged.
12. Full per-event queue conservation and shared execution bound, including
    drain losers; zero A execution while F; exactly one winner per Token.
13. Conditional-law estimator distinguishes different realized fault paths;
    control never receives future path. CVaR is computed over pooled own Tokens,
    not averaged per-episode CVaR or silently pooled across unequal Experts.
14. All256 rules are distinct, NNNN/NSSN included, D/S charge1 and X charge2;
    metadata prepass rejected for feedback policies; hidden remaining work
    unavailable through observation objects.
15. Best-response search cycles or uncertainty fail the equilibrium gate;
    all rows retained. Pure fixed-point convergence alone cannot bypass the
    full finite-system deviation and safety checks.

## 11. Implementation slices and disposition

This document authorizes the following implementation slices; it does not
authorize experiments or solver work:

1. Shared finite scheduler and state/ledger/trace contract. Implement fixtures
   1-7, 9, and 12 using deterministic fault fixtures, retaining old engines
   unchanged. Fixture 8 is reserved for random public-fault and information
   boundaries; fixture 10 is reserved for the finite-system deviation
   evaluator; fixture 11 is reserved for nested N-workloads and scale
   validation.
   First deliverable is peer latency changing under a unilateral action with
   correct capacity and work integrals; do not begin a price/fixed-point solver.
2. Random common-fault observation boundary and online four-action policy API;
   verify no future leakage and conditional sampling (fixtures8,13,14).
3. Individual/social scorer and finite unilateral-deviation harness; validate
   exact fixtures and run only separately scoped development timing smoke.
4. Freeze campaign precision protocol, then conditional particle forward model
   and bounded restricted-game solver; preserve cycles and negative results.
5. Independent conditional-law, finite-N regret and decision/safety validation.
   Only after this assess whether a richer feedback MFG is justified.

Potential modules: shared_backup.py (global scheduler), expert_game.py
(observations/actions/payoffs), game_workload.py (nested traces/common noise),
game_deviations.py (full finite counterfactuals), conditional_field.py (separate
model/solver). These are proposed boundaries, not files created this turn.
Reusing lifecycle primitives requires independent hand fixtures; shared code
does not supply independent software validation. Standard library first.

ADR-0014 old independence, capacity, global-ID parity, fixed-backup-count arms
and cutoff remain historical. ADR-0015 prospectively replaces those clauses
only in a new run family if activated. Retain Gate identity, normalized work,
copy lifecycle, per-copy charge, old schemas, no holdout tuning, and all old
artifacts. Existing paused solver tickets remain paused. No Context or old
Accepted ADR is silently rewritten by this proposal.

## 12. Sources and limits of support

The processor-sharing scheduler, durations, capacity, policy bank and numerical
gates are proposed engineering choices. The cited literature distinguishes
mathematical objects; it does not establish this marked-queue model's theorem.

- Carmona, Delarue, Lachapelle, Control of McKean-Vlasov Dynamics versus Mean
  Field Games: https://arxiv.org/abs/1210.5771
- Carmona, Delarue, Lacker, Mean field games with common noise:
  https://arxiv.org/abs/1407.6181
- Carmona, Delarue, Probabilistic Analysis of Mean-Field Games:
  https://arxiv.org/abs/1210.5780
