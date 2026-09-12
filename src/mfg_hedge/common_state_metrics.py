"""Fault-aware summary (schema 2) for Common State + No Hedge runs.

Definitions follow `.scratch/common-state-no-hedge/spec.md` section 7,
including observed/unobserved phase rules and cumulative Token queue delay.
The Healthy `build_summary` in `metrics.py` is untouched.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from statistics import fmean
from typing import Any

from . import __version__
from .common_state import Phase
from .common_state_simulation import (
    AttemptStatus,
    CommonStateSimulationResult,
    CommonStateTokenResult,
)
from .config import CommonStateExperimentConfig
from .metrics import percentile
from .workload import WorkloadTrace


def _cumulative_queue_delay(token: CommonStateTokenResult) -> float:
    total = 0.0
    for attempt in token.attempts:
        if attempt.start_time is not None:
            total += attempt.start_time - attempt.enqueue_time
        elif attempt.status is AttemptStatus.INVALIDATED_QUEUED:
            total += attempt.terminal_time - attempt.enqueue_time
    return total


def _phase_block(
    phase: Phase,
    config: CommonStateExperimentConfig,
    result: CommonStateSimulationResult,
    arrival_rate: float,
) -> dict[str, Any]:
    timeline = config.timeline()
    window_start, window_end = {
        Phase.HEALTHY: (0.0, timeline.degraded_start),
        Phase.DEGRADED: (timeline.degraded_start, timeline.failed_start),
        Phase.FAILED: (timeline.failed_start, timeline.recovered_start),
        Phase.RECOVERED: (timeline.recovered_start, math.inf),
    }[phase]
    drain_end = result.drain_end_time
    observation_start = max(window_start, 0.0)
    observation_end = min(window_end, drain_end)
    observed = observation_end > observation_start

    samples = [
        token.latency for token in result.tokens if token.completion_phase is phase
    ]

    degraded_speed = 1.0 / config.degraded_slowdown
    per_replica: list[tuple[float | None, float | None]]  # (arrival, speed)
    if phase is Phase.FAILED:
        per_replica = [None, (arrival_rate, 1.0)]
    elif phase is Phase.DEGRADED:
        per_replica = [(arrival_rate / 2.0, degraded_speed), (arrival_rate / 2.0, 1.0)]
    else:
        per_replica = [(arrival_rate / 2.0, 1.0), (arrival_rate / 2.0, 1.0)]

    replica_ratios = []
    ratios: list[float] = []
    for replica_id, entry in enumerate(per_replica):
        if entry is None:
            replica_ratios.append(
                {"replica_id": replica_id, "assigned_arrival_rate": None,
                 "speed": None, "ratio": None}
            )
            continue
        arrival, speed = entry
        ratio = arrival / speed
        ratios.append(ratio)
        replica_ratios.append(
            {"replica_id": replica_id, "assigned_arrival_rate": arrival,
             "speed": speed, "ratio": ratio}
        )

    alive_speed = {
        Phase.HEALTHY: 2.0,
        Phase.DEGRADED: 1.0 + degraded_speed,
        Phase.FAILED: 1.0,
        Phase.RECOVERED: 2.0,
    }[phase]

    return {
        "observed": observed,
        "observation_start": observation_start if observed else None,
        "observation_end": observation_end if observed else None,
        "sample_count": len(samples),
        "latency_p50": percentile(samples, 50.0) if samples else None,
        "latency_p95": percentile(samples, 95.0) if samples else None,
        "latency_p99": percentile(samples, 99.0) if samples else None,
        "replica_ratios": replica_ratios,
        "aggregate_ratio": arrival_rate / alive_speed,
        "capacity_violation": (
            any(ratio > 1.0 for ratio in ratios) if observed else None
        ),
    }


def build_common_state_summary(
    config: CommonStateExperimentConfig,
    trace: WorkloadTrace,
    result: CommonStateSimulationResult,
    run_id: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Build the schema-2 summary.json payload for a Common State + No Hedge run.

    `throughput` and per-Replica `utilization` are measured over
    `[0, drain_end_time]` and include the drain period; the throughput is not
    a fixed-window capacity figure. `run_id` is the only field allowed to
    differ between runs that share configuration and seed.
    """
    generated = len(trace.tokens)
    drain_end = result.drain_end_time

    primary_attempts = []
    replay_attempts = []
    for attempt in result.attempts:
        if attempt.attempt_id == 0:
            primary_attempts.append(attempt)
        elif attempt.attempt_id == 1:
            replay_attempts.append(attempt)
        else:
            raise RuntimeError(f"unexpected attempt_id {attempt.attempt_id}")
    if len(primary_attempts) + len(replay_attempts) != len(result.attempts):
        raise RuntimeError("attempt partition does not cover all attempts")

    nominal_primary_work = math.fsum(a.required_work for a in primary_attempts)
    executed_primary_work = math.fsum(a.executed_work for a in primary_attempts)
    wasted_work = math.fsum(
        a.executed_work
        for a in primary_attempts
        if a.status is AttemptStatus.FAILED_RUNNING
    )
    executed_replay_work = math.fsum(a.executed_work for a in replay_attempts)
    total_executed_work = executed_primary_work + executed_replay_work
    accounted = math.fsum(a.executed_work for a in result.attempts)
    if not math.isclose(
        total_executed_work, accounted, rel_tol=0.0, abs_tol=1e-9 * max(1.0, accounted)
    ):
        raise RuntimeError("work accounting does not cover all attempts")
    if nominal_primary_work <= 0.0:
        raise RuntimeError("nominal_primary_work must be positive by construction")
    extra_execution_ratio = executed_replay_work / nominal_primary_work
    execution_amplification = total_executed_work / nominal_primary_work

    replayed_tokens = sum(1 for token in result.tokens if token.replay_count == 1)
    if result.hedge_launches != 0:
        raise RuntimeError(
            f"hedge_launches must be 0 in this slice, got {result.hedge_launches}"
        )
    invariants = {
        "completed_tokens_equals_generated_tokens": result.completed_tokens == generated,
        "primary_executions_equals_generated_tokens": result.primary_executions == generated,
        "failed_primary_equals_running_plus_queued": (
            result.failed_primary_executions
            == result.failed_running_primary_executions
            + result.invalidated_queued_primary_executions
        ),
        "replay_executions_equals_replayed_tokens": (
            result.replay_executions == replayed_tokens
        ),
        "max_replay_count_at_most_one": all(
            token.replay_count <= 1 for token in result.tokens
        ),
        "hedge_launches": result.hedge_launches,
        "drain_end_time_ge_last_arrival_time": (
            result.drain_end_time >= result.last_arrival_time
        ),
    }
    violated = [name for name, value in invariants.items() if value is False]
    if violated:
        raise RuntimeError(f"engine invariants violated: {violated}")

    latencies = [token.latency for token in result.tokens]
    queue_delays = [_cumulative_queue_delay(token) for token in result.tokens]

    replicas = []
    for replica_id in range(2):
        replica_attempts = [
            a for a in result.attempts if a.replica_id == replica_id
        ]
        busy_wall_time = math.fsum(
            a.terminal_time - a.start_time
            for a in replica_attempts
            if a.start_time is not None
        )
        replicas.append(
            {
                "replica_id": replica_id,
                "assigned": sum(
                    1 for t in result.tokens if t.primary_replica == replica_id
                ),
                "replay_executions": sum(
                    1 for a in replica_attempts if a.attempt_id == 1
                ),
                "busy_wall_time": busy_wall_time,
                "normalized_work": math.fsum(
                    a.executed_work for a in replica_attempts
                ),
                "utilization": (
                    busy_wall_time / drain_end if drain_end > 0.0 else 0.0
                ),
            }
        )

    timeline = config.timeline()
    return {
        "schema_version": 2,
        "run_id": run_id,
        "scenario": "common_state_no_hedge",
        "dispatcher": "round_robin_failure_aware",
        "policy": "no_hedge",
        "simulator_version": __version__,
        "experiment_name": config.experiment_name,
        "config_schema_version": config.schema_version,
        "resolved_config": asdict(config),
        "config_sha256": config_sha256,
        "base_seed": trace.base_seed,
        "generated_tokens": generated,
        "token_count": generated,
        "arrival_rate": trace.arrival_rate,
        "healthy_offered_load": config.healthy_offered_load,
        "replica_count": 2,
        "timeline": {
            "degraded_start": timeline.degraded_start,
            "failed_start": timeline.failed_start,
            "recovered_start": timeline.recovered_start,
        },
        "mean_latency": fmean(latencies),
        "latency_p50": percentile(latencies, 50.0),
        "latency_p95": percentile(latencies, 95.0),
        "latency_p99": percentile(latencies, 99.0),
        "mean_queue_delay": fmean(queue_delays),
        "last_arrival_time": result.last_arrival_time,
        "drain_end_time": result.drain_end_time,
        "drain_duration": result.drain_duration,
        "throughput": generated / drain_end if drain_end > 0.0 else 0.0,
        "throughput_basis": "end_to_end_including_drain",
        "observation_interval": {
            "start": 0.0,
            "end": result.drain_end_time,
            "includes_drain": True,
        },
        "phases": {
            phase.value: _phase_block(phase, config, result, trace.arrival_rate)
            for phase in Phase
        },
        "faults": {
            "completed_tokens": result.completed_tokens,
            "failed_primary_executions": result.failed_primary_executions,
            "failed_running_primary_executions": result.failed_running_primary_executions,
            "invalidated_queued_primary_executions": result.invalidated_queued_primary_executions,
            "replayed_tokens": replayed_tokens,
            "replay_executions": result.replay_executions,
            "replay_rate": replayed_tokens / generated,
            "hedge_launches": result.hedge_launches,
            "queue_length_at_failed_start": list(result.queue_length_at_failed_start),
            "queue_length_at_recovered_start": list(
                result.queue_length_at_recovered_start
            ),
        },
        "work": {
            "nominal_primary_work": nominal_primary_work,
            "wasted_work": wasted_work,
            "executed_primary_work": executed_primary_work,
            "executed_replay_work": executed_replay_work,
            "total_executed_work": total_executed_work,
            "extra_execution_ratio": extra_execution_ratio,
            "execution_amplification": execution_amplification,
        },
        "replicas": replicas,
        "invariants": invariants,
    }
