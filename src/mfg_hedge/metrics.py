"""Summary metrics with an explicit percentile definition.

Percentiles use linear interpolation over the sorted sample: for ``n`` values
and percent ``p``, the fractional rank is ``(n - 1) * p / 100`` and the result
is interpolated between the two neighboring order statistics. No third-party
statistics package is involved.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from statistics import fmean
from typing import Any, Sequence

from . import __version__
from .config import ExperimentConfig
from .simulation import SimulationResult
from .workload import WorkloadTrace


def percentile(values: Sequence[float], percent: float) -> float:
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be within [0, 100]")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def build_summary(
    config: ExperimentConfig,
    trace: WorkloadTrace,
    result: SimulationResult,
    run_id: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build the summary.json payload for one Healthy + No Hedge run.

    The observation interval is ``[0, horizon]`` where ``horizon`` is the last
    completion time across Replicas. Throughput and utilization are measured
    over that interval. Configuration identity is pinned by
    ``config_schema_version``, the full ``resolved_config``, and
    ``config_sha256`` over the configuration file bytes. ``run_id`` is the
    only field allowed to differ between runs that share configuration and
    seed.
    """
    latencies = [token.latency for token in result.tokens]
    queue_delays = [token.queue_delay for token in result.tokens]
    token_count = len(result.tokens)
    horizon = result.horizon

    replicas = [
        {
            "replica_id": replica_id,
            "assigned": result.assigned_counts[replica_id],
            "busy_time": result.busy_times[replica_id],
            "utilization": (
                result.busy_times[replica_id] / horizon if horizon > 0.0 else 0.0
            ),
        }
        for replica_id in range(len(result.assigned_counts))
    ]

    return {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": "healthy_no_hedge",
        "dispatcher": "round_robin",
        "policy": "no_hedge",
        "simulator_version": __version__,
        "experiment_name": config.experiment_name,
        "config_schema_version": config.schema_version,
        "resolved_config": asdict(config),
        "config_sha256": config_sha256,
        "base_seed": trace.base_seed,
        "token_count": token_count,
        "arrival_rate": trace.arrival_rate,
        "healthy_offered_load": config.healthy_offered_load,
        "replica_count": len(result.assigned_counts),
        "mean_latency": fmean(latencies),
        "latency_p50": percentile(latencies, 50.0),
        "latency_p95": percentile(latencies, 95.0),
        "latency_p99": percentile(latencies, 99.0),
        "mean_queue_delay": fmean(queue_delays),
        "throughput": token_count / horizon if horizon > 0.0 else 0.0,
        "observation_interval": {"start": 0.0, "end": horizon},
        "replicas": replicas,
        "invariants": {
            "primary_executions": result.counters.primary_executions,
            "hedge_launches": result.counters.hedge_launches,
            "replay_executions": result.counters.replay_executions,
        },
    }
