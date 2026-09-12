# ADR-0030: Baseline taxonomy for replicated-Expert routing

Status: Accepted (user requested repository integration on 2026-09-08)

## Context

No-Hedge and the repository's fitted NIIN rule are insufficient as the only
comparators for a replicated-MoE routing contribution. Public systems and
request-hedging papers act at different layers, so placing every name behind a
single N/D/I policy interface would misrepresent their semantics.

## Decision

1. Maintain separate placement, routing, and hedging baseline interfaces.
2. Treat DeepSeek EPLB as placement, LPLB as batch routing, and LÆDGE/P95 as
   hedging. Round-robin, JSQ, least unfinished work, and failover-only are
   transparent control baselines.
3. Implement deterministic standard-library adaptations in an isolated module.
   Record upstream provenance and license; do not add Torch/CUDA/P4 dependencies.
4. Use a common workload, fault trace, placement input, and metrics contract
   before publishing comparative numbers. Distinguish algorithm quality from
   hardware/runtime overhead.
5. Preserve the fixed Gate: routing chooses a physical replica of the already
   selected Logical Expert and never changes the Expert identity.

## Consequences

The repository gains credible executable kernels without claiming bitwise or
performance equivalence to GPU and programmable-switch artifacts. A separate
adapter and experiment protocol remain necessary before drawing performance
conclusions.

## References

See `.scratch/replica-routing-baselines/spec.md`.
