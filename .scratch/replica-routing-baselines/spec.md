# Replica-routing baseline integration

## Goal

Add credible, named comparators for a fixed replicated-MoE topology without
changing the Gate, existing simulators, historical artifacts, or MFG claims.

## Taxonomy

The baseline package keeps three layers separate:

1. `placement`: EPLB-style replication and physical placement from historical
   logical-Expert loads.
2. `routing`: Uniform round-robin, JSQ, least unfinished work, and LPLB-style
   batch min-max assignment over the replicas made available by placement.
3. `hedging`: failover-only, percentile-delayed hedging, and LÆDGE-style
   work-conserving hedging.

EPLB is an upstream placement baseline, not a Token-routing result. LPLB is a
batch token-count balancing baseline, not a failure-protection policy. P95 and
LÆDGE are request-hedging baselines, not replica-placement algorithms.

## First implementation slice

Implement a pure-standard-library, deterministic baseline kernel with strict
inputs and stable tie-breaking:

- global and hierarchical EPLB-style replication/packing;
- round-robin, JSQ, and least-unfinished-work online routing;
- exact integral min-max batch assignment under per-Expert eligibility, named
  `lplb_token_count_minmax` and explicitly distinguished from DeepSeek's CUDA
  production implementation;
- failover-only, P95 delayed-hedge, and generalized LÆDGE scheduling kernels;
- upstream provenance, license notices, semantic-class labels, and focused
  hand-computed tests.

This slice does not run a new experiment and does not retrofit the policies
into an existing engine. A later adapter ticket must make every comparator use
one immutable workload/fault trace and one common metrics contract.

## Second implementation slice: comparison adapters

Create a fail-closed two-panel adapter registry:

- `protection_same_topology`: Failover-only, P95, LÆDGE, exact NIIN, and the
  sealed requested-price candidate on the one-Expert/two-Replica topology.
- `routing_after_placement`: RR, JSQ, least unfinished work, and LPLB under one
  frozen EPLB placement on a batched multi-Expert/GPU topology.

The registry must expose the physical capabilities each arm requires and must
reject an execution backend that lacks them. In particular, the current
continuous-arrival engine cannot silently run LPLB without batch boundaries,
and the current per-Replica FCFS engine cannot silently approximate LÆDGE
without an idle-release work-conserving callback. EPLB is panel provenance,
not a measured arm.

## Upstream references

- DeepSeek EPLB: <https://github.com/deepseek-ai/EPLB> (MIT)
- DeepSeek LPLB: <https://github.com/deepseek-ai/LPLB> (MIT)
- LÆDGE paper: <https://www.usenix.org/conference/nsdi21/presentation/primorac>
- NetClone artifact containing a LÆDGE comparator:
  <https://github.com/GyuyeongKim/NetClone-public> (MIT)
- The Tail at Scale: <https://research.google/pubs/the-tail-at-scale/>

## Protection-panel experiment

ADR-0031 freezes a 1,024-episode paired evaluation on the existing Token
topology. P95 uses a disjoint 256-episode No-Hedge calibration library. LÆDGE
uses an isolated work-conserving executor because its idle-release decisions
cannot be faithfully encoded as one-shot N/D/I actions. The routing panel is
not part of this experiment.

## Claim boundary

Names ending in `-style` are independent standard-library adaptations of the
published algorithmic semantics. They are not claimed to reproduce CUDA,
network, communication, or wall-clock overheads of the upstream systems.
No benchmark result may compare different placement/topology inputs without
labeling that difference.
