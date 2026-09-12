# ADR-0009: Persistent runtime work and waste costs

Status: Accepted (user-directed correction on 2026-09-05)

## Context

ADR-0008 correctly treats incremental Hedge+Replay work as capacity load, but
its shadow price is zero whenever capacity has slack. The formal load-0.5
campaign then reused a highly protective Healthy policy indefinitely in phase R:
fault cohorts improved while overall median/tail latency and executed work became
worse. Capacity feasibility alone is not a complete operating-cost objective.

## Decision

1. Add a nonnegative persistent incremental-work price and a nonnegative waste
   price to the objective defined in the feature spec. The capacity shadow price
   remains a separate endogenous congestion signal.
2. Define wasted work as executed work on all non-winning attempts. Resource and
   waste prices deliberately overlap for a losing Hedge: one accounts for consumed
   capacity and one for producing no winning result.
3. Preserve schema-3 behavior with implicit zero coefficients. Schema 4 requires
   both coefficients explicitly and strictly positive. The first corrected config
   uses 1.0/1.0 in normalized service-time units and labels them untuned mechanism
   defaults.
4. Add `expected_wasted_work` to ActionStats and calibration schema 2. Random
   streams, samples, rho coordinates, and calibration episodes do not change.
5. ADR-0008 capacity feasibility, boundary-pinned price, forced-Normal F,
   official gate, and quota budgets are unchanged.

## Consequences

Hedge is no longer free at price zero. A lower Hedge rate is not itself the
success criterion: the paired result must report latency, Replay, work, and fault
cohorts. Coefficient sensitivity is required before treating 1.0/1.0 as an
economic optimum, but is deliberately outside this correction ticket.
