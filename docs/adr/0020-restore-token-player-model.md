# ADR-0020: Restore Token players and separate the finite reference from MFG scaling

Status: Accepted (Token-player restoration boundary and T1 only; confirmed on 2026-09-06)

Date: 2026-09-06

## Context

The original attribution design studies Tokens coordinating protection through
congestion and a resource price. ADR-0015 instead defines Logical Expert
controllers as players, adds a cross-Expert pool, and optimizes each Expert's
aggregate loss. These are different games. A per-arrival callback does not
make an Expert-payoff/profile-deviation solver a Token game.

The Token restoration therefore needs a narrow accepted boundary for the first
physical slice. The full design and all later pricing, preference, deviation,
and mean-field choices remain reviewable proposals in
`.scratch/token-mfg-restoration/spec.md`.

## Accepted boundary: Token restoration and T1 physical contract

1. A Token is the player, including all of its linked Primary, Hedge and
   Replay attempts. Experts and Replicas are execution resources.
2. Gate and Dispatcher choices are exogenous and fixed. The first finite
   reference is one Expert with two FCFS Replicas A/B. Do not inherit the
   Expert Shared-Backup topology or cross-Expert shared pool.
3. At arrival, each Token makes exactly one causal N/D/I request. A Delayed
   request is completed by its timer and is not a second policy decision;
   Replay, winner selection, and cancellation remain engine mechanisms.
4. `TokenObservation` contains only public/current state and the Token's
   observable local queue and reservation facts. It excludes future arrivals,
   future faults, service draws, true remaining work, and future outcomes.
5. Keep `requested`, reservation admission, `applied`, reservation
   suppression, and executor suppression as distinct auditable facts.
6. Reservation is immediate, Decimal-audited, pooled, half-open-window, and
   non-refundable. An unaffordable or otherwise denied D/I request is recorded
   as requested but applied as N; it then follows normal Replay behavior.
7. Preserve running losers, Replay, timer ordering, and complete drain. Retain
   the legacy static `actions: Mapping[token_id, action]` entry point and make
   the online entry a compatible addition.
8. The Expert Shared-Backup implementation is an independent V2 branch and is
   not modified or resumed by this ADR.

T1 restores only this online entry and Reservation contract. It does not
authorize a campaign or any equilibrium/MFG claim.

## Deferred proposals — not accepted by this ADR

The following remain Proposed in `spec.md` and require separate ADRs and
tickets:

- `c_W`, `c_waste`, deadlines, Replay preferences, Token payoff, and
  continuation predictors;
- posted-price pacing, price updates, entropy/soft best response, and any
  price or market interpretation;
- single-Token deviations, best response, solver, forward equation, HJB/FPK,
  K-executor scaling, and any Nash/MFG claim;
- formal baseline, Fit/Validation, qualification, or experiment artifact.

## Consequences and limits

The restored finite reference can reuse the original A/B event engine while
making the decision causal at the arrival boundary. Static action mappings and
historical modules remain valid. The implementation must not infer a future
fault, sampled work, or future queue outcome from the observation, and must
not change the existing Common Random Number streams.

This ADR does not accept a private payoff, a price equilibrium, a Token
deviation evaluator, a forward law, or a many-server limit. T1 is online
execution and admission infrastructure only.

## References

- Specification: `.scratch/token-mfg-restoration/spec.md`
- Engineering audit: `.scratch/token-mfg-restoration/engineering-audit.md`
- T1 ticket: `.scratch/token-mfg-restoration/issues/02-implement-online-token-entry-and-reservation.md`
- Prior audit ticket: `.scratch/token-mfg-restoration/issues/01-complete-token-model-and-engineering-audit.md`
- Original purpose: `CONTEXT.md`; ADR-0005/0006/0009/0012.
