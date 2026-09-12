"""Paired A/B metrics for MFG-Hedge vs No-Hedge (spec sections 13-14).

Arm summaries share one schema; No-Hedge arms report zero for every Hedge
counter and null storm ratios. Planned Hedge work (quota expectation) and
realized Hedge work (engine outcome) are always reported separately.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from .common_state import CommonStateTimeline, Phase, phase_at
from .common_state_simulation import CommonStateSimulationResult
from .domain import ProtectionAction
from .hedge_simulation import HedgeAttemptStatus, HedgeSimulationResult
from .metrics import percentile
from .quota import QuotaProjectionResult
from .workload import WorkloadTrace


_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)
_HEDGE_STATUSES_LOSER = (
    HedgeAttemptStatus.COMPLETED_LOSER,
    HedgeAttemptStatus.CANCELLED_QUEUED,
)


def _cumulative_queue_delay(token) -> float:
    total = 0.0
    for attempt in token.attempts:
        if attempt.start_time is not None:
            total += attempt.start_time - attempt.enqueue_time
        elif attempt.status in (
            HedgeAttemptStatus.INVALIDATED_QUEUED,
            HedgeAttemptStatus.CANCELLED_QUEUED,
        ):
            total += attempt.terminal_time - attempt.enqueue_time
    return total


def _phase_latency_block(latencies: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(latencies),
        "latency_p50": percentile(latencies, 50.0) if latencies else None,
        "latency_p95": percentile(latencies, 95.0) if latencies else None,
        "latency_p99": percentile(latencies, 99.0) if latencies else None,
    }


def _storm_metrics(
    result: HedgeSimulationResult,
    timeline: CommonStateTimeline,
    bin_width: float,
    capacities: dict,
) -> dict[str, Any]:
    horizon = result.drain_end_time
    launches = [
        attempt.enqueue_time
        for attempt in result.attempts
        if attempt.attempt_id == 2
    ]
    full_bins = int(math.floor(horizon / bin_width))
    launch_bins = [0] * full_bins
    partial_launches = 0
    for moment in launches:
        index = int(moment // bin_width)
        if index < full_bins:
            launch_bins[index] += 1
        else:
            partial_launches += 1
    if not launches:
        peak_rate = None
        mean_rate = None
        ratio = None
    else:
        peak_rate = max(launch_bins, default=0) / bin_width
        mean_rate = len(launches) / horizon if horizon > 0.0 else 0.0
        ratio = peak_rate / mean_rate if mean_rate > 0.0 else None

    # Demand-side overload per bin: (primary work arriving + Hedge work
    # launched) / width vs alive capacity at the bin's start. The partial bin
    # is normalized by its actual width and excluded from peak-rate inputs.
    total_bins = full_bins + (1 if horizon - full_bins * bin_width > 1e-12 else 0)
    arrivals_work_bins = [0.0] * total_bins
    hedge_work_bins = [0.0] * total_bins
    for token in result.tokens:
        index = int(token.arrival_time // bin_width)
        if index < total_bins:
            start = index * bin_width
            end = min(start + bin_width, horizon)
            if start <= token.arrival_time < end:
                arrivals_work_bins[index] += token.attempts[0].required_work
    for attempt in result.attempts:
        if attempt.attempt_id != 2:
            continue
        index = int(attempt.enqueue_time // bin_width)
        if index < total_bins:
            start = index * bin_width
            end = min(start + bin_width, horizon)
            if start <= attempt.enqueue_time < end:
                hedge_work_bins[index] += attempt.required_work

    overloaded = []
    for index in range(total_bins):
        start = index * bin_width
        end = min(start + bin_width, horizon)
        width = end - start
        capacity = capacities[phase_at(timeline, start).value]
        overloaded.append(
            (arrivals_work_bins[index] + hedge_work_bins[index]) / width > capacity
        )
    longest = 0
    current = 0
    for index, flag in enumerate(overloaded):
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    sustained = 0.0
    run = 0.0
    longest = 0.0
    for index, flag in enumerate(overloaded):
        start = index * bin_width
        width = min(start + bin_width, horizon) - start
        if flag:
            run += width
            longest = max(longest, run)
        else:
            run = 0.0
    return {
        "bin_width": bin_width,
        "hedge_launch_bins": launch_bins,
        "partial_bin_hedge_launches": partial_launches,
        "peak_hedge_launch_rate": peak_rate,
        "mean_hedge_launch_rate": mean_rate,
        "peak_to_mean_hedge_launch_ratio": ratio,
        "overloaded_bin_count": sum(overloaded),
        "sustained_overload_duration": longest,
    }


def build_arm_summary(
    result: HedgeSimulationResult,
    timeline: CommonStateTimeline,
    *,
    arm: str,
    projection: QuotaProjectionResult | None,
    bin_width: float,
    capacities: dict,
) -> dict[str, Any]:
    """Build one arm's summary; arm B passes its quota projection."""
    tokens = result.tokens
    generated = len(tokens)
    latencies = [token.latency for token in tokens]
    queue_delays = [_cumulative_queue_delay(token) for token in tokens]

    arrival_phase_latency = {}
    completion_phase_latency = {}
    migration = {
        arrival.value: {completion.value: 0 for completion in _PHASES}
        for arrival in _PHASES
    }
    for phase in _PHASES:
        arrival_phase_latency[phase.value] = _phase_latency_block(
            [
                token.latency
                for token in tokens
                if phase_at(timeline, token.arrival_time) is phase
            ]
        )
        completion_phase_latency[phase.value] = _phase_latency_block(
            [
                token.latency
                for token in tokens
                if token.completion_phase is phase
            ]
        )
    for token in tokens:
        migration[phase_at(timeline, token.arrival_time).value][
            token.completion_phase.value
        ] += 1

    attempts = result.attempts
    primary_attempts = [a for a in attempts if a.attempt_id == 0]
    hedge_attempts = [a for a in attempts if a.attempt_id == 2]
    replay_attempts = [a for a in attempts if a.attempt_id == 1]
    nominal = math.fsum(a.required_work for a in primary_attempts)
    executed_primary = math.fsum(a.executed_work for a in primary_attempts)
    executed_hedge = math.fsum(a.executed_work for a in hedge_attempts)
    executed_replay = math.fsum(a.executed_work for a in replay_attempts)
    wasted = math.fsum(
        a.executed_work
        for a in attempts
        if a.status
        in (
            HedgeAttemptStatus.FAILED_RUNNING,
            HedgeAttemptStatus.COMPLETED_LOSER,
            HedgeAttemptStatus.CANCELLED_QUEUED,
        )
    )
    total = executed_primary + executed_hedge + executed_replay

    winners = {"primary": 0, "hedge": 0, "replay": 0}
    for token in tokens:
        winners[{0: "primary", 1: "replay", 2: "hedge"}[token.winner_attempt_id]] += 1

    replicas = []
    for replica_id in range(2):
        replica_attempts = [a for a in attempts if a.replica_id == replica_id]
        busy = math.fsum(
            a.terminal_time - a.start_time
            for a in replica_attempts
            if a.start_time is not None
        )
        replicas.append(
            {
                "replica_id": replica_id,
                "assigned": sum(
                    1 for token in tokens if token.primary_replica == replica_id
                ),
                "busy_wall_time": busy,
                "normalized_work": math.fsum(
                    a.executed_work for a in replica_attempts
                ),
                "utilization": (
                    busy / result.drain_end_time if result.drain_end_time > 0 else 0.0
                ),
            }
        )

    replayed = sum(1 for token in tokens if token.replay_count > 0)
    hedge = {
        "requested": result.hedge_requested,
        "launched": result.hedge_launches,
        "executor_suppressed": result.hedge_suppressed,
        "cancelled_queued": result.cancelled_queued_total,
        "completed_loser": result.completed_loser_total,
    }

    if projection is not None:
        actions = {
            "global": {
                "requested_counts": {
                    action.value: count
                    for action, count in projection.global_requested_counts.items()
                },
                "applied_counts": {
                    action.value: count
                    for action, count in projection.global_applied_counts.items()
                },
                "quota_suppressed": projection.global_quota_suppressed,
            },
            "per_class": {},
            "per_window": [],
        }
        for window in projection.windows:
            requested = {"N": 0, "D": 0, "I": 0}
            applied = {"N": 0, "D": 0, "I": 0}
            suppressed = 0
            for class_audit in window.classes:
                for action in ("N", "D", "I"):
                    key = action
                    requested[key] += class_audit.requested_counts[
                        ProtectionAction(key)
                    ]
                    applied[key] += class_audit.applied_counts[ProtectionAction(key)]
                suppressed += class_audit.quota_suppressed
            actions["per_window"].append(
                {
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "phase": window.phase.value,
                    "state": window.state.value,
                    "requested_counts": requested,
                    "applied_counts": applied,
                    "quota_suppressed": suppressed,
                }
            )
        for cls in ("R", "U"):
            requested = {"N": 0, "D": 0, "I": 0}
            applied = {"N": 0, "D": 0, "I": 0}
            suppressed = 0
            for assignment in projection.assignments:
                if assignment.token_class.value == cls:
                    requested[assignment.requested_action.value] += 1
                    applied[assignment.applied_action.value] += 1
                    suppressed += int(assignment.quota_suppressed)
            actions["per_class"][cls] = {
                "requested_counts": requested,
                "applied_counts": applied,
                "quota_suppressed": suppressed,
            }
        planned_vs_realized = []
        for window in projection.windows:
            realized = math.fsum(
                a.executed_work
                for a in hedge_attempts
                if window.window_start <= a.enqueue_time < window.window_end
            )
            planned = math.fsum(c.planned_expected_hedge_work for c in window.classes)
            suppressed = sum(c.quota_suppressed for c in window.classes)
            planned_vs_realized.append(
                {
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "state": window.state.value,
                    "phase": window.phase.value,
                    "planned_expected_hedge_work": planned,
                    "hedge_budget": window.window_budget,
                    "realized_hedge_work": realized,
                    "realized_excess": max(0.0, realized - window.window_budget),
                    "realized_violation": realized > window.window_budget,
                    "quota_suppressed": suppressed,
                    "executor_hedge_suppressed": sum(
                        1
                        for moment, _ in result.hedge_suppressed_events
                        if window.window_start <= moment < window.window_end
                    ),
                }
            )
    else:
        actions = {
            "global": {
                "requested_counts": {"N": generated, "D": 0, "I": 0},
                "applied_counts": {"N": generated, "D": 0, "I": 0},
                "quota_suppressed": 0,
            },
            "per_class": {},
            "per_window": [],
        }
        planned_vs_realized = []

    invariants = {
        "completed_tokens_equals_generated_tokens": result.completed_tokens == generated,
        "at_most_one_replay_per_token": all(t.replay_count <= 1 for t in tokens),
        "at_most_one_hedge_per_token": all(
            sum(1 for a in t.attempts if a.attempt_id == 2) <= 1 for t in tokens
        ),
        "exactly_one_winner_per_token": all(
            sum(
                1
                for a in t.attempts
                if a.status is HedgeAttemptStatus.COMPLETED_WINNER
            )
            == 1
            for t in tokens
        ),
        "drain_end_ge_last_arrival": result.drain_end_time
        >= result.last_arrival_time,
    }

    return {
        "arm": arm,
        "generated_tokens": generated,
        "completed_tokens": result.completed_tokens,
        "mean_latency": fmean(latencies),
        "latency_p50": percentile(latencies, 50.0),
        "latency_p95": percentile(latencies, 95.0),
        "latency_p99": percentile(latencies, 99.0),
        "mean_queue_delay": fmean(queue_delays),
        "last_arrival_time": result.last_arrival_time,
        "drain_end_time": result.drain_end_time,
        "drain_duration": result.drain_duration,
        "replay_rate": replayed / generated,
        "arrival_phase_latency": arrival_phase_latency,
        "completion_phase_latency": completion_phase_latency,
        "migration_matrix": migration,
        "actions": actions,
        "hedge": hedge,
        "winners": winners,
        "work": {
            "nominal_primary_work": nominal,
            "executed_primary_work": executed_primary,
            "executed_hedge_work": executed_hedge,
            "executed_replay_work": executed_replay,
            "wasted_work": wasted,
            "total_executed_work": total,
            "extra_execution_ratio": executed_hedge / nominal if nominal else 0.0,
            "execution_amplification": total / nominal if nominal else 0.0,
        },
        "replicas": replicas,
        "queue_snapshots": {
            "failed_start": list(result.queue_length_at_failed_start),
            "recovered_start": list(result.queue_length_at_recovered_start),
        },
        "storm": _storm_metrics(result, timeline, bin_width, capacities),
        "planned_vs_realized": planned_vs_realized,
        "invariants": invariants,
    }


_COMPARED_METRICS = (
    "mean_latency",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "mean_queue_delay",
    "replay_rate",
    "drain_duration",
)


def _metric(summary: dict, name: str) -> float | None:
    if name in summary:
        return summary[name]
    if name in summary["work"]:
        return summary["work"][name]
    if name in summary["storm"]:
        return summary["storm"][name]
    if name == "hedge_launches":
        return summary["hedge"]["launched"]
    raise KeyError(name)


def build_comparison(summary_a: dict, summary_b: dict) -> dict:
    """A/B absolutes, B−A deltas, and relative change (null when A == 0)."""
    names = list(_COMPARED_METRICS)
    for phase in ("H", "D", "F", "R"):
        names.append(f"latency_p99_arrival_{phase}")
    names += [
        "nominal_primary_work",
        "executed_primary_work",
        "executed_hedge_work",
        "executed_replay_work",
        "wasted_work",
        "total_executed_work",
        "extra_execution_ratio",
        "execution_amplification",
        "peak_hedge_launch_rate",
        "sustained_overload_duration",
        "hedge_launches",
    ]
    metrics = {}
    for name in names:
        if name.startswith("latency_p99_arrival_"):
            phase = name.rsplit("_", 1)[1]
            value_a = summary_a["arrival_phase_latency"][phase]["latency_p99"]
            value_b = summary_b["arrival_phase_latency"][phase]["latency_p99"]
        else:
            value_a = _metric(summary_a, name)
            value_b = _metric(summary_b, name)
        if value_a is None or value_b is None:
            delta = None if value_a is None and value_b is None else "incomparable"
            metrics[name] = {
                "arm_a": value_a,
                "arm_b": value_b,
                "delta": delta,
                "relative": None,
            }
            continue
        delta = value_b - value_a
        relative = (value_b - value_a) / value_a if value_a != 0 else None
        metrics[name] = {
            "arm_a": value_a,
            "arm_b": value_b,
            "delta": delta,
            "relative": relative,
        }
    return {"metrics": metrics}
