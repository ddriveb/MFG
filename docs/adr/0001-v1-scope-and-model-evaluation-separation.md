# ADR-0001: Keep V1 Narrow and Separate Policy Models from Evaluation

Status: Accepted

Date: 2026-09-03

## Context

The research question is whether population coordination can reduce protection bursts under a shared degradation or failure. Adding Gate changes, dynamic placement, cross-layer behavior, training, or a full HJB-FPK model would make causal interpretation difficult. Using the same aggregate formulas both to choose and evaluate a policy would also create circular evidence.

## Decision

V1 is limited to the boundary in `CONTEXT.md`. Policy computation may use coarse Action Stats, but final performance comparisons must run on an independent Token-level queue/event simulator. Both sides may share versioned configuration and domain types, but the solver must not read future event outcomes and the evaluator must not replace realized events with lookup-table expectations.

All policy comparisons use common random numbers: the same arrival, Token class, service-time, and failure traces.

## Consequences

- The first vertical slice is Healthy + No Hedge for one Expert and two Replicas.
- MFG, faults, Delayed/Immediate Hedge, and multi-Expert behavior are out of scope for that slice.
- More sophisticated theory is added only after the mechanism survives strong heuristic baselines and model mismatch.
- Base-load infeasibility and policy-caused capacity violations must be reported separately.

