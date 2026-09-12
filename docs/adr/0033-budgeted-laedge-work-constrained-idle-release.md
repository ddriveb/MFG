# ADR-0033: Budgeted LÆDGE work-constrained idle-release baseline

Status: Accepted (confirmed by the user on 2026-09-09)

## Context

The ADR-0032 cancellation ablation separates the major LÆDGE effect from
running-loser cancellation.  Conservative LÆDGE retains most of the latency
benefit of idle-release scheduling, but it can consume substantially more
Hedge and wasted work than NIIN.  A next baseline must therefore constrain
work through causal admission while preserving the work-conserving idle-
release scheduler.

This ADR is now accepted and authorizes the isolated Budgeted LÆDGE core
implementation, but not a development/holdout experiment.  Concrete episode counts,
namespace strings, macro seeds, scheduler-call limits, and artifact run IDs
must be frozen in a later implementation or experiment ticket after this ADR
is accepted.  Missing values are a fail-closed execution block, not free
parameters.

## Decision

### 1. Physical and causal boundary

1. The baseline is a finite one-Expert/two-Replica idle-release executor on
   the existing common-state, workload, CRN, and complete-drain contracts.
2. The fixed Gate and same-Expert replica identity remain unchanged.
3. Every arm uses the same immutable arrival, Token-class, attempt-0/1/2,
   Replay, Hedge, and common-fault streams for one episode.  No arm may replay
   another arm's realized actions, queue trajectory, completion times, or
   budget outcomes.
4. The formal primary arm uses conservative cancellation.  Preemptive
   cancellation is retained only as an explicitly labeled oracle/upper-bound
   comparator.

### 2. Idle-release scheduling semantics

1. A Token arrival enters an unassigned waiting queue.  Arrival does not bind
   it to a future Replica.
2. Whenever a Replica becomes available, the dispatcher first considers
   waiting Primary or Replay requests with no live copy.
3. Only when no startable unserved Primary or Replay request exists may it
   consider a Hedge for a Token that already has exactly one live copy.
4. Each Token may have at most one Hedge and at most one Replay.
5. Unserved requests are selected in stable `(arrival_time, token_id)` order.
   Hedge candidates use the same stable order after the unserved queue is
   empty.
6. Work never starts on Replica A while the common state makes A unavailable
   in F.  Fault-first ordering, complete drain, stale-event handling, and
   attempt-key CRN semantics retain their existing definitions.
7. A winner does not change the conservative primary semantics: a queued loser
   may be cancelled with zero executed work, while a running loser continues
   to completion and remains in realized and wasted work.  The preemptive
   comparator may cancel a running loser only under its explicitly labeled
   oracle semantics.

### 3. Work budget and online ledger

1. The budget is measured in work, never in Hedge launch count.  The final
   resource metric is realized incremental executed Hedge work relative to the
   same episode's No-Hedge reference.
2. Version 1 uses the fixed causal estimate
   `expected_work_i = 1.0` normalized work for every admitted Hedge.  This is
   the known mean of the current attempt-2 service-work distribution; the
   controller must not read the true future service draw or remaining service
   work.
3. Admission is frozen as:

   `accept_i iff nonrefundable_charged_work_w + expected_work_i <= planned_budget_w`.

   `nonrefundable_charged_work_w` is the sum of the expected work charges for
   every Hedge accepted in window `w`; it is monotone nondecreasing within the
   window and is never reduced by execution, early completion, or realized
   work below expectation.
4. `outstanding_expected_work` is an audit field for still-running admitted
   Hedges.  It does not release, refund, or otherwise restore admission
   allowance.  The audit may additionally expose the signed reconciliation
   quantity `observed_realized_hedge_work - committed_expected_remaining_hedge_work`,
   but that quantity is not an admission refund mechanism.
5. If the nonrefundable charge plus a new expected work charge would exceed
   the current window's planned budget, the Hedge is suppressed.  A Hedge that
   has started is not forcibly terminated when later realized work exhausts
   the plan.
6. Because service draws are unknown at admission and conservative running
   losers continue, the protocol must not claim a pathwise hard cap on
   realized work.
7. Each window reports planned budget, nonrefundable charged work,
   outstanding expected work, observed realized Hedge work, realized
   incremental total work, overshoot, unused budget, launches, and
   suppressions.  Planned, charged, outstanding, and realized quantities are
   distinct fields.
8. Budget is not refunded and cannot carry across windows.

### 4. Windows

1. The maximum window width is `W=25` time units.
2. H/D/F/R state boundaries force a window cut even when the width limit has
   not been reached.
3. Every new window resets only the new window's planned budget and
   `nonrefundable_charged_work_w` allowance.  A running attempt admitted in a
   previous window continues to occupy its Replica and remains in physical
   execution; it neither disappears nor receives a refund.
4. Work is reported under both of these non-additive, complementary views:
   - `launch_cohort_realized_work`: all realized work of a Hedge attributed to
     its launch window, used for that window's planned-budget error, overshoot,
     and unused-budget audit;
   - `execution_time_realized_work`: work integrated in the window where it
     physically executes, used for capacity, congestion, and Protection-Storm
     analysis.
5. The same work unit may appear in both views for different purposes, but
   the views must never be summed together or counted as two physical units.
6. The audit records window start/end, duration, state/phase, planned budget,
   nonrefundable charged work, outstanding expected work, both realized-work
   views, overshoot, unused budget, launches, and suppression counts.

### 5. Development calibration of the work targets

1. The frozen evaluation targets are relative to the mean total executed work
   of NIIN conservative:
   `delta = {0%, 1%, 3%, 5%, 8%, 12%, 18%}`.
2. `delta` is an evaluation work target, not a counterfactual NIIN workload
   observable available to the online controller.
3. Independent development episodes map each target to a fixed causal window
   budget rate before holdout execution.  The mapping uses the fixed
   `expected_work_i = 1.0` admission charge; its priority parameters,
   estimator version, and source fingerprint are frozen after development.
4. Holdout execution cannot retune the budget rate, priority rule, action
   threshold, or suppression behavior.
5. Calibration that fails, is non-monotone, or cannot reach a target is
   fail-closed.  The latest feasible diagnostic is retained, but no target is
   silently substituted or relabeled.

### 6. Development/holdout isolation

1. Development and holdout use disjoint namespaces, macro seeds, episode
   identities, and trace fingerprints.  The exact values must be listed in the
   later execution ticket before any run.
2. Development is for mapping the seven target deltas to fixed budget rates
   only.  It cannot produce a holdout policy choice.
3. Holdout is for the frozen Pareto evaluation only.  It cannot modify the
   mapping, priority, budget, or action rules.
4. Any duplicate fingerprint, missing episode, failed arm, CRN mismatch, or
   invariant failure makes the relevant result unavailable; there is no retry,
   supplement, deletion, or result-driven sample change.

### 7. Required comparison arms

The later evaluation must include exactly 11 unique semantic arms:

1. fixed-dispatcher No-Hedge;
2. NIIN conservative;
3. Budgeted LÆDGE conservative at the seven frozen delta points
   `{0%, 1%, 3%, 5%, 8%, 12%, 18%}`; the `delta=0` row is the unique
   work-conserving No-Hedge arm;
4. unconstrained LÆDGE conservative;
5. unconstrained LÆDGE preemptive oracle.

The runner must not add a second work-conserving No-Hedge arm outside the
`delta=0` Budgeted LÆDGE row.  Budgeted LÆDGE at `delta=0` is not the same as
fixed-dispatcher No-Hedge: dynamic Primary placement is itself part of the
routing mechanism.

### 8. Metrics

Each complete arm must report, using the existing metric definitions where
available:

- overall latency mean, P95, and P99;
- D/F latency mean, P95, P99, and CVaR95;
- D/F deadline-miss rate;
- H/R P99;
- Replay rate;
- total executed work and incremental Hedge work;
- wasted work;
- Hedge launches and winner rate;
- suppression count/rate;
- Protection-Storm peak;
- drain duration;
- planned budget, nonrefundable charged work, outstanding expected work,
  `launch_cohort_realized_work`, `execution_time_realized_work`, budget error,
  overshoot, and unused budget.

Episode completion, invariant, failure, and per-window counter fields remain
mandatory.  Hedge launch count is not a substitute for work accounting.

### 9. Pareto analysis

1. The horizontal coordinate is relative realized total-work increase against
   NIIN conservative:
   `(realized_total_work - niin_conservative_work) /
   niin_conservative_work`.
2. The primary vertical coordinate is D/F CVaR95.  D/F deadline miss rate and
   H/R P99 are reported alongside it and are not hidden by a single scalar.
3. Tail gain per extra work is reported as
   `(CVaR95_NIIN - CVaR95_policy) /
   (realized_work_policy - realized_work_NIIN)` when the denominator is
   nonzero; zero or negative denominators are reported as a separate case,
   not silently divided.
4. A point is marked `dominated` only when another strategy is no worse in
   tail, deadline miss, realized work, and Protection-Storm, and is strictly
   better in at least one of those dimensions.
5. No delta point is predeclared successful or selected during this design
   stage.

## Consequences

This design isolates the causal value of work-conserving idle-release
scheduling from launch-count heuristics and from running-loser preemption. It
also makes conservative overshoot observable instead of incorrectly promising
a pathwise hard work cap. The resulting output is a finite-system Budgeted
LÆDGE Pareto baseline.

## Non-goals and claim boundary

This ADR does not itself run calibration, run holdout, choose a budget point,
or create an artifact. It does not define prices, posted-price
feedback, best response, regret, Nash equilibrium, or an MFG fixed point.
Preemptive LÆDGE is not the deployable primary arm; it is an oracle/upper
bound when running cancellation is unavailable or not causally deployable.
Only after a low-budget idle-release benefit is established may a separate
accepted protocol define price-guided protection and a Token-MFG loop.

## Rejected alternatives

- Budgeting by Hedge launch count, which confounds short and long service work.
- Reading future service requirements to enforce a false pathwise hard cap.
- Retuning the budget or priority on holdout episodes.
- Treating dynamic work-conserving No-Hedge as identical to fixed-dispatcher
  No-Hedge.
- Using preemptive cancellation as the main deployable result.
- Introducing price feedback or MFG claims before the finite-system Pareto
  baseline is established.
