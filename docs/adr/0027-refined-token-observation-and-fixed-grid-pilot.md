# ADR-0027: Refined Token observation and fixed price-temperature pilot

Status: Accepted for a bounded diagnostic (confirmed by the user on 2026-09-08)

Date: 2026-09-08

## Context

The first T4c exploratory artifact was invalidated by the NIII-versus-NIIN
fallback bug. Independent review also showed two modeling risks: aggregate
queue load erased Primary/Backup direction, and selecting the first eligible
Token in an age interval concentrated targets near the interval boundary.
Dynamic pricing would confound these issues with a new feedback law.

## Decision

Run one bounded, non-iterative conditional-cost pilot with the corrected exact
NIIN population policy.

The diagnostic observation schema uses:

- Token class and Primary replica;
- D phase age buckets `[0,25)`, `[25,50)`, `[50,75)`, `[75,+inf)`;
- separate Primary and Backup live-load buckets `[0,3)`, `[3,+inf)` where live
  load is queued plus running attempts on that Replica;
- Reservation balance buckets `[0,1)`, `[1,2.8125)`, `[2.8125,+inf)`.

For episode `e`, choose one of the four age strata by `e mod 4`. Before the
episode begins, derive a uniform threshold inside that stratum from a stable
SHA-256 key. Select the first eligible D/Primary-A Token at or after that
threshold. This remains future-blind while removing the first-at-stratum-start
bias. A missing target consumes its episode and is not replaced.

Use 1,024 deterministic episode identities and at most 5,120 physical calls.
Each complete panel runs selection, baseline, N, D, and I once under exact NIIN
and shared CRN. Store the runtime-v1 zero-price payoff and raw incremental
Hedge+Replay work for each action.

Without rerunning physics, evaluate the descriptive grid:

    beta in {1, 2, 4}
    fixed execution price p in {0, 0.5, 1}

For each cell, add `p * (W_H + W_R)` to the zero-price runtime payoff and form
the Softmax response. Report supported-bin coverage, Hedge probability,
priced and unpriced expected private cost, incremental work, and the Softmax
response gap over the priced argmin. Require at least eight complete panels for
a bin to enter the grid aggregate; unsupported bins remain explicit.

No population update is executed here.  For interpretation, the diagnostic
reports both the undamped `response_l1` and the hypothetical
`update_l1 = 0.2 * response_l1`; future iterative runners must keep the same
two quantities separate.

## Consequences

This pilot separates observation, sampling, temperature, and a fixed price
before any endogenous feedback. It does not choose a production policy or
claim BR, regret, Nash, MFG, or population improvement. Grid reuse is valid
because exact NIIN does not read the fixed quote and the runtime price is a
linear retrospective term over already recorded incremental work; a later
price-conditioned policy must receive the price online and rerun physics.

## References

- `docs/adr/0021-token-private-cost-and-deviation-semantics.md`
- `docs/adr/0025-token-mfg-t4c-supported-soft-response.md`
- `.scratch/token-mfg-restoration/issues/21-run-refined-fixed-grid-pilot.md`
