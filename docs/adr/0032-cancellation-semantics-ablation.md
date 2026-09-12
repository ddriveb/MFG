# ADR-0032: Cancellation-semantics ablation

Status: Accepted (confirmed by the user on 2026-09-09)

## Context

The protection-baseline result compares a conservative N/D/I engine with an
isolated LÆDGE executor whose native behavior immediately cancels a running
loser when another copy wins. The resulting latency and work differences mix
admission/scheduling with loser-cancellation semantics. A narrow 2×2 ablation
is needed before changing prices or designing a new protection policy.

## Decision

1. Define four isolated arms as the Cartesian product of:
   - policy family: the existing one-shot NIIN arrival policy, or the existing
     idle-release LÆDGE scheduler;
   - running-loser handling: conservative completion, or immediate running-
     loser cancellation.
2. Use the same immutable episode, arrival, service, Replay, Hedge, and fault
   streams for all four arms. The default historical paths remain unchanged.
3. Conservative handling leaves a running loser in service until its own
   completion and records `completed_loser`; immediate cancellation settles
   work through the winner time, retains executed work, records
   `cancelled_running`, invalidates its scheduled completion by generation, and
   releases the replica only after the complete same-time event batch.
4. Queued losers remain `cancelled_queued` with zero executed work in both
   modes. Failed running work and Replay behavior remain fault-first and are
   not redefined by this ablation.
5. The implementation is an isolated comparison hook. It does not change
   prices, Reservation parameters, workload samples, CRN keys, historical
   artifacts, or the scientific claim boundary. This slice does not run the
   formal panel.

## Consequences

The four arms can attribute a bounded part of the LÆDGE advantage to running-
loser cancellation without claiming hardware equivalence, optimality, Nash,
or MFG results. Any later campaign must freeze its sample count, budget,
metrics, and artifact namespace in a separate accepted protocol.

## Rejected alternatives

- Lowering the current protection price or increasing Hedge requests without
  separating cancellation semantics.
- Replaying realized actions or deleting other Tokens from the event loop.
- Modifying the historical default engine or the existing r2 artifact.
