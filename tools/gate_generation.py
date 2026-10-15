"""Side-effect-free reconstructions of the three frozen Gate generators."""

from __future__ import annotations

import math
import random
from pathlib import Path
import sys

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))

from mfg_hedge.moe_legal_variable_gate_v1 import (  # noqa: E402
    FAMILIES,
    sample_gate_counts,
)


HOTSPOT_RATIOS = (0.4, 0.6, 0.8)


def multiscale_gate(support, seed: int, family: str, ranks: int):
    """Reproduce local132/scale_plan.py's per-campaign random stream."""
    strength = random.Random(seed).uniform(1, 2.5)
    hot = set(sorted(range(ranks), key=lambda e: random.Random(seed * 1000 + e).random())[:ranks // 4])
    counts = []
    patterns = []
    for source, candidates in enumerate(support):
        rng = random.Random(seed * 10000 + source)
        source_hot = set(rng.sample(candidates, 4))
        weights = []
        for expert in candidates:
            value = random.Random(seed * 1000 + expert + 500).gauss(0, 0.2) + rng.gauss(0, 0.15)
            if family == "global_hotspot" and expert in hot:
                value += strength
            if family == "node_local" and expert // 8 == source // 8:
                value += strength
            if family == "source_skew" and expert in source_hot:
                value += strength
            weights.append(math.exp(value))
        indices = list(range(16))
        rng.shuffle(indices)
        pattern_rows = [indices[:8], indices[8:]]
        row_counts = [16] * 16
        for _ in range(62):
            keys = [math.log(max(rng.random(), 1e-300)) / weight for weight in weights]
            selected = sorted(range(16), key=keys.__getitem__, reverse=True)[:8]
            for j in selected:
                row_counts[j] += 16
            pattern_rows.append(selected)
        assert len(pattern_rows) == 64 and all(len(set(row)) == 8 for row in pattern_rows)
        assert [16 * sum(j in row for row in pattern_rows) for j in range(16)] == row_counts
        assert sum(row_counts) == 8192 and min(row_counts) > 0
        counts.extend(row_counts)
        patterns.append([[candidates[j] for j in row] for row in pattern_rows])
    return counts, patterns, strength


def hotspot_support(ranks: int, seed: int):
    if ranks not in (32, 64):
        raise ValueError("global-hotspot campaign uses 32 or 64 Experts")
    hot = tuple(sorted(random.Random(seed).sample(range(ranks), 8)))
    cold = [expert for expert in range(ranks) if expert not in hot]
    support = tuple(
        tuple(sorted(hot + tuple(random.Random(seed * 10000 + source).sample(cold, 8))))
        for source in range(ranks)
    )
    return support, hot


def hotspot_gate(support, hot, seed: int, ratio: float, ranks: int):
    if ratio not in HOTSPOT_RATIOS:
        raise ValueError("global-hotspot levels are the frozen 40/60/80 percent values")
    if len(support) != ranks or len(hot) != 8 or len(set(hot)) != 8:
        raise ValueError("invalid global-hotspot layout")
    hot = set(hot)
    target = round(ratio * 512)
    counts = []
    patterns = []
    for source, candidates in enumerate(support):
        if len(candidates) != 16 or len(set(candidates)) != 16 or not hot <= set(candidates):
            raise ValueError("each source requires the same eight hot and eight cold candidates")
        rng = random.Random(seed * 10000 + source)
        weights = [
            math.exp(random.Random(seed * 1000 + e + 500).gauss(0, 0.2) + rng.gauss(0, 0.15))
            for e in candidates
        ]
        indices = list(range(16))
        rng.shuffle(indices)
        pattern_rows = [indices[:8], indices[8:]]
        q, remainder = divmod(target - 8, 62)
        levels = [q + int(i < remainder) for i in range(62)]
        rng.shuffle(levels)
        hot_indices = [j for j, expert in enumerate(candidates) if expert in hot]
        cold_indices = [j for j, expert in enumerate(candidates) if expert not in hot]
        for number in levels:
            def choose(pool, k):
                return sorted(
                    pool,
                    key=lambda j: math.log(max(rng.random(), 1e-300)) / weights[j],
                    reverse=True,
                )[:k]

            pattern_rows.append(choose(hot_indices, number) + choose(cold_indices, 8 - number))
        row_counts = [16 * sum(j in row for row in pattern_rows) for j in range(16)]
        assert len(pattern_rows) == 64 and all(len(set(row)) == 8 for row in pattern_rows)
        assert sum(row_counts) == 8192 and min(row_counts) > 0 and max(row_counts) <= 1024
        assert sum(n for e, n in zip(candidates, row_counts) if e in hot) == target * 16
        counts.extend(row_counts)
        patterns.append([[candidates[j] for j in row] for row in pattern_rows])
    return counts, patterns, target / 512


def g64_gate(support, seed: int, family: str):
    return sample_gate_counts(support, seed=seed, family=family, return_patterns=True)
