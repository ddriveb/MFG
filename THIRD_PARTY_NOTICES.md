# Third-party algorithm references

The project contains independent, standard-library adaptations of published
algorithmic semantics. It does not vendor the upstream CUDA, PyTorch, P4, C,
or network-runtime source trees.

## DeepSeek EPLB

- Source: <https://github.com/deepseek-ai/EPLB>
- Inspected: `main`, 2026-09-08 web snapshot
- License: MIT
- Copyright: Copyright (c) 2024 DeepSeek
- Adapted behavior: greedy Expert replication by current load per replica and
  equal-cardinality hierarchical packing.

## DeepSeek LPLB

- Source: <https://github.com/deepseek-ai/LPLB>
- Inspected revision: `0490f79452f7ef277e814449600b1b1dd4c663b3`
- License: MIT
- Copyright: Copyright (c) 2025 DeepSeek
- Adapted behavior: batch token-count min-max assignment over a fixed redundant
  Expert topology. The local implementation uses deterministic integral flow,
  not the upstream CUDA single-SM interior-point solver.

## LÆDGE and NetClone artifact

- Paper: <https://www.usenix.org/conference/nsdi21/presentation/primorac>
- Related comparator artifact:
  <https://github.com/GyuyeongKim/NetClone-public>
- Artifact inspected: `main`, 2026-09-08 web snapshot
- Artifact license: MIT; Copyright (c) 2023 Gyuyeong Kim
- Adapted behavior: generalized LÆDGE's work-conserving priority: an idle
  replica serves oldest unserved work before it hedges oldest singly running
  work. Hardware/network behavior is not copied or claimed.

## The Tail at Scale

- Publication: <https://research.google/pubs/the-tail-at-scale/>
- Adapted behavior: a frozen empirical percentile delay before launching an
  otherwise unnecessary copy. No upstream implementation is vendored.

The referenced MIT licenses permit use, modification, and redistribution
provided the copyright and permission notice are retained. Consult each linked
repository for the full license text.
