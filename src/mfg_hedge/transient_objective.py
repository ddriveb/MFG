"""Prospective ADR-0012 realized objective, never a calibration/solver estimate."""

from __future__ import annotations

import math
from typing import Iterable

from .attribution_episode import EpisodeSimulationResult
from .attribution_metrics import build_episode_metrics
from .common_state import Phase, phase_at
from .domain import TokenClass
from .hedge_simulation import HedgeAttemptStatus


def cvar95_fractional_tail(values: Iterable[float]) -> float:
    """Mean of exactly the largest 5% empirical probability mass.

    The fractional observation is weighted before division. For n < 20 the
    entire tail mass lies in the maximum. Historical empirical_cvar95 is not
    changed. Input is nonnegative finite latency, not a generic signed loss.
    """
    samples = []
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("fractional CVaR requires finite nonnegative latency values")
        samples.append(float(value))
    if not samples:
        raise ValueError("fractional CVaR requires at least one latency")
    samples.sort(reverse=True)
    whole, remainder = divmod(len(samples), 20)
    # Average weighted observations directly to avoid overflowing the tail sum.
    weights = [20 / len(samples)] * whole
    if remainder:
        weights.append(remainder / len(samples))
    return math.fsum(x * w for x, w in zip(samples, weights))


def build_transient_objective(runs: Iterable[EpisodeSimulationResult]) -> dict:
    """Pool raw Tokens/work, retain every episode, and expose missing cohorts.

    A full scalar requires all eight phase/class cohorts. No per-episode
    averaging, missing-cohort zero fill, legacy rounded CVaR, or outcome-based
    episode exclusion. Sorting by identity makes input iteration order irrelevant.
    This score alone does not test safety constraints or select a policy.
    """
    episodes = list(runs)
    if not episodes or any(not isinstance(r, EpisodeSimulationResult) for r in episodes):
        raise ValueError("objective requires nonempty EpisodeSimulationResult collection")
    keys = [r.episode.identity.key for r in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode identity in objective")
    episodes.sort(key=lambda r: r.episode.identity.key)
    first = episodes[0]
    contract = (first.episode.protocol, first.degraded_slowdown, first.hedge_delay,
                first.episode.identity.namespace, first.episode.workload.arrival_rate)
    for run in episodes:
        other = (run.episode.protocol, run.degraded_slowdown, run.hedge_delay,
                 run.episode.identity.namespace, run.episode.workload.arrival_rate)
        if other != contract:
            raise ValueError("objective episode contract mismatch")
        # Reuse the complete existing physical/work/counter validation, not just
        # a permissive projection of arbitrary dataclasses onto scalar costs.
        build_episode_metrics(run, run.degraded_slowdown)
    timeline = first.episode.protocol.timeline
    tokens = tuple(t for run in episodes for t in run.simulation.tokens)
    attempts = tuple(a for run in episodes for a in run.simulation.attempts)
    cohorts, phase_losses, tails, missing = {}, {}, {}, []
    for phase in Phase:
        phase_tokens = [t for t in tokens if phase_at(timeline, t.arrival_time) is phase]
        cells = {}
        for cls, weight, deadline, penalty in (
                (TokenClass.REGULAR, 0.8, 3.0, 1.0),
                (TokenClass.URGENT, 0.2, 2.0, 5.0)):
            rows = [t for t in phase_tokens if t.token_class is cls]
            label = cls.name.lower()
            n = len(rows)
            mean = math.fsum(t.latency for t in rows) / n if n else None
            excess = math.fsum(max(0.0, t.latency - deadline) for t in rows) / n if n else None
            replay = sum(t.replay_count > 0 for t in rows) / n if n else None
            loss = mean + penalty * excess + penalty * replay if n else None
            cells[label] = {"sample_count": n, "weight": weight, "deadline": deadline,
                            "excess_penalty": penalty, "replay_penalty": penalty,
                            "mean_latency": mean, "mean_slo_excess": excess,
                            "replay_rate": replay, "loss": loss}
            if not n:
                missing.append(f"{phase.value}/{label}")
        cohorts[phase.value] = cells
        phase_losses[phase.value] = (
            math.fsum(c["weight"] * c["loss"] for c in cells.values())
            if all(c["sample_count"] for c in cells.values()) else None
        )
        if phase in (Phase.DEGRADED, Phase.FAILED):
            tails[phase.value] = {
                "sample_count": len(phase_tokens),
                "cvar95_fractional_tail": cvar95_fractional_tail(t.latency for t in phase_tokens)
                if phase_tokens else None,
            }
    work = math.fsum(a.executed_work for a in attempts)
    waste = math.fsum(a.executed_work for a in attempts
                      if a.status is not HedgeAttemptStatus.COMPLETED_WINNER)
    phase_term = math.fsum(phase_losses.values()) / 4 if not missing else None
    tail_term = math.fsum(t["cvar95_fractional_tail"] for t in tails.values()) / 2 \
        if all(t["sample_count"] for t in tails.values()) else None
    components = {"phase_loss": phase_term, "fault_tail": tail_term,
                  "executed_work": work / len(tokens), "wasted_work": waste / len(tokens)}
    total = math.fsum(components.values()) if not missing else None
    if any(v is not None and not math.isfinite(v) for v in (*components.values(), total)):
        raise ValueError("objective overflow")
    return {"objective_contract": "transient_realized_pooled_v1",
            "status": "incomplete" if missing else "complete", "total": total,
            "episode_count": len(episodes), "episode_keys": sorted(keys),
            "generated_tokens": len(tokens), "missing_cohorts": missing,
            "coefficients": {"phase_weight": 0.25, "fault_tail_weight": 0.5,
                             "executed_work_cost": 1.0, "wasted_work_cost": 1.0},
            "cohorts": cohorts, "phase_losses": phase_losses, "fault_tails": tails,
            "raw_work": {"total_executed_work": work, "non_winner_executed_work": waste},
            "work_observation": "all_attempts_through_drain",
            "components": components}
