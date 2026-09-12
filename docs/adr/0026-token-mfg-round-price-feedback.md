# ADR-0026: Bounded Token-MFG round-level price feedback

Status: Accepted for a bounded diagnostic (confirmed by the user on 2026-09-08)

Date: 2026-09-08

## Context

The T4c exploratory run reduced its policy residual but slightly increased
latency, Replay, and wasted work.  The runtime private-cost model already has
a fixed incremental-work coefficient, but its external execution quote stayed
at zero.  The user authorized adding price feedback before any longer run.

The full within-episode pacing controller proposed in the restoration spec
would require a time-varying quote and per-Token price settlement.  This first
bounded diagnostic instead tests the smaller question: does a public price
updated between population-response rounds reduce excess protection demand and
improve the direction of the finite-population iteration?

## Decision

Add an isolated T4d exploratory runner.  It keeps the physical engine,
Reservation, runtime cost model, and exact NIIN fallback repaired in ticket 20,
and uses the refined observation/target protocol validated in ADR-0027.
Within response round `r`, every baseline and N/D/I branch uses one immutable
execution-work quote `p_r`, visible to the policy and scorer.  After the round:

    demand_pressure_r = (requested_reserved_work_r - reservation_capacity_r)
                        / reservation_capacity_r
    p_(r+1) = max(0, p_r + alpha * demand_pressure_r)

Use `alpha=0.25`, initial price zero, `beta=4.0`, `eta=0.2`, and the
`incremental_executed_work` price basis required by
`token_runtime_adr0009_v1`.  Requested reservation work counts every eligible
non-N request before Reservation projection; capacity is the available D-window
Reservation budget over the evaluated baseline episodes.  Admitted work,
suppression, and the price path are reported separately.

This is an epoch-level feedback controller, not the within-window controller
from the restoration spec, a Walrasian market, or a capacity shadow price.
It is accepted only for a 256-identity, four-round exploratory comparison.
This is at most 5,120 full scheduler calls; the user was informed of this bound
before execution.

Convergence additionally requires `abs(p_(r+1)-p_r) <= 0.01`, alongside the
existing T4c policy and occupancy residual gates.  The terminal success label
is `price_supported_soft_fixed_point`; all BR, regret, Nash, and MFG claim flags
remain false.

## Consequences

The zero-price T4c path remains unchanged.  The price runner can show whether
feedback suppresses protection demand and changes latency/work trends on the
same CRN library, but it cannot establish a market equilibrium or a formal MFG
solution.  A positive result is a reason to design the causal within-window
controller; a negative result is evidence against spending hours on the same
price rule.

The bounded run produced the negative case: the controller raises price from
requested pre-projection demand, while `incremental_executed_work` charges only
work that survives Reservation projection.  A denied request therefore creates
demand pressure without paying the same price.  This price path is retained as
a diagnostic, but the mismatch must not be promoted to a causal within-window
controller.

The first attempted execution was stopped before artifact commit after the
baseline audit found that T4c's claimed NIIN fallback was actually NIII.  The
repair and refined observation pilot are now complete.  ADR-0027 showed that a
fixed price from 0 to 1 monotonically reduced expected applied protection while
changing conditional unpriced cost only slightly; that is the activation
evidence for this bounded feedback run.

## References

- `.scratch/token-mfg-restoration/spec.md`, section 6
- `docs/adr/0021-token-private-cost-and-deviation-semantics.md`
- `docs/adr/0025-token-mfg-t4c-supported-soft-response.md`
- `.scratch/token-mfg-restoration/issues/19-implement-and-run-price-feedback-pilot.md`
