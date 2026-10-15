"""Opt-in constructive legal Top8 demand variation; old generators unchanged.

Groups of identical tokens keep offline sampling inexpensive. This synthetic
family has fixed positive support and grouped count granularity; it is not a
real workload trace and does not cover changing active-row identities.
"""
from dataclasses import dataclass
import math
import random

from .moe_communication_social_v2 import RouteRow

FAMILIES = ('balanced', 'global_hotspot', 'node_local', 'source_skew')


@dataclass(frozen=True)
class GateCountSnapshot:
    support: tuple[tuple[int, ...], ...]
    counts: tuple[tuple[int, ...], ...]
    seed: int
    family: str
    tokens_per_source: int
    top_k: int
    groups: int
    patterns: tuple[tuple[tuple[int, ...], ...], ...] = ()


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')


def _seed(value):
    if type(value) is not int or value < 0:
        raise ValueError('seed must be a nonnegative integer')


def fixed_candidate_support(*, gpu_count=256, expert_count=256,
                            candidates=16, seed=940001):
    for value, name in ((gpu_count, 'gpu_count'), (expert_count, 'expert_count'),
                        (candidates, 'candidates')):
        _positive_int(value, name)
    _seed(seed)
    if candidates > expert_count:
        raise ValueError('candidate count exceeds Expert population')
    rng = random.Random(seed)
    return tuple(tuple(sorted(rng.sample(range(expert_count), candidates)))
                 for _ in range(gpu_count))


def sample_gate_counts(support, *, seed, family='balanced',
                       tokens_per_source=1024, groups=64, top_k=8,
                       gpus_per_node=8, shift=False, return_patterns=False):
    """Sample distinct-Expert token patterns and aggregate their exact counts.

    Two complementary initial patterns ensure every candidate occurs. Thus the
    stable positive support is constructed, not obtained by deleting zeros.
    Source-specific counts are retained in their original support order.
    """
    _seed(seed)
    for value, name in ((tokens_per_source, 'tokens_per_source'),
                        (groups, 'groups'), (top_k, 'top_k'),
                        (gpus_per_node, 'gpus_per_node')):
        _positive_int(value, name)
    if family not in FAMILIES or top_k != 8:
        raise ValueError('supported family and Top8 are required')
    if groups < 2 or tokens_per_source % groups:
        raise ValueError('at least two equal integer token groups are required')
    support = tuple(tuple(row) for row in support)
    if not support or any(len(row) != 2*top_k or len(set(row)) != len(row)
                          or any(type(e) is not int or e < 0 for e in row)
                          for row in support):
        raise ValueError('each source requires sixteen distinct nonnegative Experts')
    if type(shift) is not bool:
        raise ValueError('shift must be boolean')
    rng = random.Random(seed)
    all_experts = sorted(set(e for row in support for e in row))
    # Shared Expert variation is independent of source ordering and support seed.
    global_noise = {e: rng.gauss(0., rng.uniform(.05, .35)) for e in all_experts}
    hot = set(rng.sample(all_experts, min(16, len(all_experts))))
    strength = rng.uniform(3., 4.) if shift else rng.uniform(1., 2.5)
    group_tokens = tokens_per_source // groups
    counts_by_source = []
    patterns_by_source = []
    gpu_count = len(support)
    for source, candidates in enumerate(support):
        source_hot = set(rng.sample(candidates, 4))
        weights = []
        for e in candidates:
            log_weight = global_noise[e] + rng.gauss(0., .15)
            if family == 'global_hotspot' and e in hot:
                log_weight += strength
            elif family == 'node_local' and ((e % gpu_count) // gpus_per_node
                                            == source // gpus_per_node):
                log_weight += strength
            elif family == 'source_skew' and e in source_hot:
                log_weight += strength
            weights.append(math.exp(log_weight))
        indices = list(range(len(candidates)))
        rng.shuffle(indices)
        patterns = [tuple(indices[:top_k]), tuple(indices[top_k:])]
        counts = [group_tokens] * len(candidates)
        for _ in range(groups-2):
            # Exponential-race weighted sampling without replacement.
            keys = [math.log(max(rng.random(), 1e-300))/w for w in weights]
            chosen = tuple(sorted(range(len(keys)), key=keys.__getitem__,
                                  reverse=True)[:top_k])
            for position in chosen:
                counts[position] += group_tokens
            if return_patterns:
                patterns.append(chosen)
        counts_by_source.append(tuple(counts))
        if return_patterns:
            patterns_by_source.append(tuple(tuple(candidates[i] for i in pattern)
                                            for pattern in patterns))
    return GateCountSnapshot(support, tuple(counts_by_source), seed, family,
                             tokens_per_source, top_k, groups,
                             tuple(patterns_by_source))


def snapshot_rows(snapshot, *, gpu_count=256):
    """Keep source identities and the existing adjacent two-replica layout."""
    _positive_int(gpu_count, 'gpu_count')
    if len(snapshot.support) != gpu_count:
        raise ValueError('snapshot source count differs from GPU count')
    return tuple(RouteRow(source, expert, demand,
                          (expert % gpu_count, (expert+1) % gpu_count))
                 for source, (candidates, counts) in enumerate(
                     zip(snapshot.support, snapshot.counts))
                 for expert, demand in zip(candidates, counts))

