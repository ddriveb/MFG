"""Episode and macro-seed metrics for the attribution study.

This module is intentionally separate from the historical schema-1 through
schema-4 summary builders.  Token-distribution quantities pool Tokens inside a
macro-seed, work/rate quantities add raw components before division, and
episode-clock quantities are measured per episode before arithmetic averaging.
Independent episode clocks are never concatenated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any, Iterable, Sequence

from .attribution_episode import EpisodeSimulationResult, episode_trace_fingerprint
from .common_state import (
    Phase,
    phase_at,
    state_at,
    validate_slowdown,
    work_executed_between,
)
from .domain import CommonState, ProtectionAction, TokenClass
from .hedge_simulation import HedgeAttemptStatus
from .metrics import percentile


_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)
_CLASSES = (TokenClass.REGULAR, TokenClass.URGENT)
_KINDS = {0: "primary", 1: "replay", 2: "hedge"}


def _positive_real(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class SLODeadlines:
    """Frozen normalized deadlines; a latency equal to its deadline passes."""

    regular: float = 3.0
    urgent: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "regular", _positive_real(self.regular, "regular deadline")
        )
        object.__setattr__(
            self, "urgent", _positive_real(self.urgent, "urgent deadline")
        )

    def for_class(self, token_class: TokenClass) -> float:
        if token_class is TokenClass.REGULAR:
            return self.regular
        if token_class is TokenClass.URGENT:
            return self.urgent
        raise ValueError(f"token_class must be a TokenClass, got {token_class!r}")


def empirical_cvar95(values: Iterable[float]) -> float:
    """Mean of the largest ``ceil(0.05*n)`` finite observations."""
    samples = list(values)
    if not samples:
        raise ValueError("empirical_cvar95 requires at least one value")
    converted: list[float] = []
    for value in samples:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                "empirical_cvar95 values must be finite real numbers, "
                f"got {value!r}"
            )
        converted.append(float(value))
    ordered = sorted(converted)
    # Exact integer form of ceil(n / 20); avoids floating-point 0.05 drift.
    count = (len(ordered) + 19) // 20
    return math.fsum(ordered[-count:]) / count


def _latency_block(tokens: Sequence, deadlines: SLODeadlines) -> dict[str, Any]:
    latencies = [token.latency for token in tokens]
    misses = 0
    excesses: list[float] = []
    for token in tokens:
        excess = max(0.0, token.latency - deadlines.for_class(token.token_class))
        excesses.append(excess)
        misses += int(excess > 0.0)
    sample_count = len(tokens)
    if not sample_count:
        return {
            "sample_count": 0,
            "latency_sum": 0.0,
            "mean_latency": None,
            "latency_p50": None,
            "latency_p95": None,
            "latency_p99": None,
            "cvar95_latency": None,
            "deadline_miss_count": 0,
            "deadline_miss_probability": None,
            "excess_latency_sum": 0.0,
            "mean_excess_latency": None,
        }
    return {
        "sample_count": sample_count,
        "latency_sum": math.fsum(latencies),
        "mean_latency": math.fsum(latencies) / sample_count,
        "latency_p50": percentile(latencies, 50.0),
        "latency_p95": percentile(latencies, 95.0),
        "latency_p99": percentile(latencies, 99.0),
        "cvar95_latency": empirical_cvar95(latencies),
        "deadline_miss_count": misses,
        "deadline_miss_probability": misses / sample_count,
        "excess_latency_sum": math.fsum(excesses),
        "mean_excess_latency": math.fsum(excesses) / sample_count,
    }


def _latency_views(tokens: Sequence, timeline, deadlines: SLODeadlines) -> dict:
    arrival: dict[str, dict[str, dict]] = {}
    completion: dict[str, dict[str, dict]] = {}
    for phase in _PHASES:
        arrival[phase.value] = {}
        completion[phase.value] = {}
        for token_class in _CLASSES:
            arrival[phase.value][token_class.value] = _latency_block(
                [
                    token
                    for token in tokens
                    if phase_at(timeline, token.arrival_time) is phase
                    and token.token_class is token_class
                ],
                deadlines,
            )
            completion[phase.value][token_class.value] = _latency_block(
                [
                    token
                    for token in tokens
                    if token.completion_phase is phase
                    and token.token_class is token_class
                ],
                deadlines,
            )
    arrival_all = {
        phase.value: _latency_block(
            [
                token
                for token in tokens
                if phase_at(timeline, token.arrival_time) is phase
            ],
            deadlines,
        )
        for phase in _PHASES
    }
    completion_all = {
        phase.value: _latency_block(
            [token for token in tokens if token.completion_phase is phase],
            deadlines,
        )
        for phase in _PHASES
    }
    return {
        "overall": _latency_block(tokens, deadlines),
        "arrival_phase": arrival_all,
        "completion_phase": completion_all,
        "arrival_phase_class": arrival,
        "completion_phase_class": completion,
    }


def _migration_matrix(tokens: Sequence, timeline) -> dict[str, dict[str, int]]:
    matrix = {
        arrival.value: {completion.value: 0 for completion in _PHASES}
        for arrival in _PHASES
    }
    for token in tokens:
        arrival = phase_at(timeline, token.arrival_time).value
        matrix[arrival][token.completion_phase.value] += 1
    return matrix


def _phase_bounds(timeline, drain_end: float) -> dict[Phase, tuple[float, float]]:
    return {
        Phase.HEALTHY: (0.0, timeline.degraded_start),
        Phase.DEGRADED: (timeline.degraded_start, timeline.failed_start),
        Phase.FAILED: (timeline.failed_start, timeline.recovered_start),
        Phase.RECOVERED: (timeline.recovered_start, drain_end),
    }


def _execution_work(run: EpisodeSimulationResult, slowdown: float) -> tuple[dict, dict]:
    result = run.simulation
    timeline = run.episode.protocol.timeline
    cells = {
        phase.value: {
            str(replica): {
                "primary": 0.0,
                "hedge": 0.0,
                "replay": 0.0,
                "total": 0.0,
                "wasted": 0.0,
                "post_cutoff_primary": 0.0,
                "post_cutoff_hedge": 0.0,
                "post_cutoff_replay": 0.0,
                "post_cutoff_total": 0.0,
                "post_cutoff_wasted": 0.0,
            }
            for replica in range(2)
        }
        for phase in _PHASES
    }
    split_matches = True
    observation_end = max(
        run.episode.protocol.arrival_cutoff, result.drain_end_time
    )
    bounds = _phase_bounds(timeline, observation_end)
    for attempt in result.attempts:
        if attempt.start_time is None or attempt.executed_work == 0.0:
            continue
        kind = _KINDS[attempt.attempt_id]
        split_total = 0.0
        for phase in _PHASES:
            phase_start, phase_end = bounds[phase]
            start = max(attempt.start_time, phase_start)
            end = min(attempt.terminal_time, phase_end)
            if end <= start:
                continue
            work = work_executed_between(
                attempt.replica_id, start, end, timeline, slowdown
            )
            cell = cells[phase.value][str(attempt.replica_id)]
            cell[kind] += work
            cell["total"] += work
            if attempt.status is not HedgeAttemptStatus.COMPLETED_WINNER:
                cell["wasted"] += work
            post_start = max(start, run.episode.protocol.arrival_cutoff)
            if end > post_start:
                post_work = work_executed_between(
                    attempt.replica_id,
                    post_start,
                    end,
                    timeline,
                    slowdown,
                )
                cell[f"post_cutoff_{kind}"] += post_work
                cell["post_cutoff_total"] += post_work
                if attempt.status is not HedgeAttemptStatus.COMPLETED_WINNER:
                    cell["post_cutoff_wasted"] += post_work
            split_total += work
        if not math.isclose(
            split_total, attempt.executed_work, rel_tol=1e-9, abs_tol=1e-9
        ):
            split_matches = False

    for phase in _PHASES:
        phase_start, phase_end = bounds[phase]
        duration = max(0.0, phase_end - phase_start)
        for replica in range(2):
            speed = 1.0
            if replica == 0 and phase is Phase.DEGRADED:
                speed = 1.0 / slowdown
            elif replica == 0 and phase is Phase.FAILED:
                speed = 0.0
            capacity = duration * speed
            cell = cells[phase.value][str(replica)]
            cell["duration"] = duration
            cell["available_normalized_capacity"] = capacity
            cell["realized_utilization"] = (
                cell["total"] / capacity if capacity > 0.0 else None
            )
            fixed_end = min(phase_end, run.episode.protocol.arrival_cutoff)
            fixed_duration = max(0.0, fixed_end - phase_start)
            fixed_capacity = fixed_duration * speed
            for kind in ("primary", "hedge", "replay"):
                cell[f"fixed_window_{kind}"] = (
                    cell[kind] - cell[f"post_cutoff_{kind}"]
                )
            cell["fixed_window_total"] = (
                cell["total"] - cell["post_cutoff_total"]
            )
            cell["fixed_window_wasted"] = (
                cell["wasted"] - cell["post_cutoff_wasted"]
            )
            cell["fixed_window_duration"] = fixed_duration
            cell["fixed_window_available_normalized_capacity"] = fixed_capacity
            cell["fixed_window_realized_utilization"] = (
                cell["fixed_window_total"] / fixed_capacity
                if fixed_capacity > 0.0
                else None
            )

    primary = math.fsum(
        attempt.executed_work
        for attempt in result.attempts
        if attempt.attempt_id == 0
    )
    replay = math.fsum(
        attempt.executed_work
        for attempt in result.attempts
        if attempt.attempt_id == 1
    )
    hedge = math.fsum(
        attempt.executed_work
        for attempt in result.attempts
        if attempt.attempt_id == 2
    )
    nominal = math.fsum(
        attempt.required_work
        for attempt in result.attempts
        if attempt.attempt_id == 0
    )
    wasted = math.fsum(
        attempt.executed_work
        for attempt in result.attempts
        if attempt.status is not HedgeAttemptStatus.COMPLETED_WINNER
    )
    total = primary + replay + hedge
    cell_total = math.fsum(
        cells[phase.value][str(replica)]["total"]
        for phase in _PHASES
        for replica in range(2)
    )
    scalar = {
        "nominal_primary_work": nominal,
        "executed_primary_work": primary,
        "executed_hedge_work": hedge,
        "executed_replay_work": replay,
        "total_executed_work": total,
        "wasted_work": wasted,
        "waste_ratio": wasted / total if total > 0.0 else None,
        "execution_amplification": total / nominal if nominal > 0.0 else None,
        "by_execution_phase_and_replica": cells,
        "post_cutoff_by_replica": {
            str(replica): {
                name: math.fsum(
                    cells[phase.value][str(replica)][f"post_cutoff_{name}"]
                    for phase in _PHASES
                )
                for name in ("primary", "hedge", "replay", "total", "wasted")
            }
            for replica in range(2)
        },
    }
    checks = {
        "attempt_phase_work_matches_executed_work": split_matches,
        "work_kind_sum_matches_total": math.isclose(
            primary + replay + hedge, total, rel_tol=1e-12, abs_tol=1e-12
        ),
        "phase_replica_work_matches_total": math.isclose(
            cell_total, total, rel_tol=1e-9, abs_tol=1e-9
        ),
        "wasted_work_within_total": -1e-12 <= wasted <= total + 1e-9,
        "fixed_window_plus_post_cutoff_matches_total": all(
            math.isclose(
                cells[phase.value][str(replica)]["fixed_window_total"]
                + cells[phase.value][str(replica)]["post_cutoff_total"],
                cells[phase.value][str(replica)]["total"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for phase in _PHASES
            for replica in range(2)
        ),
    }
    return scalar, checks


def _snapshot_dict(snapshot) -> dict:
    return {
        "queued_attempts": list(snapshot.queued_attempts),
        "live_attempts": list(snapshot.live_attempts),
        "remaining_work": list(snapshot.remaining_work),
    }


def _reconstruct_boundary_snapshot(
    run: EpisodeSimulationResult,
    slowdown: float,
    boundary: float,
    *,
    failed_transition: bool,
) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
    """Reconstruct the post-state/pre-same-time-event snapshot from records."""
    queued = [0, 0]
    live = [0, 0]
    remaining = [0.0, 0.0]
    timeline = run.episode.protocol.timeline
    for attempt in run.simulation.attempts:
        created_before_snapshot = attempt.enqueue_time < boundary or (
            failed_transition
            and attempt.attempt_id == 1
            and attempt.enqueue_time == boundary
        )
        if not created_before_snapshot or attempt.terminal_time < boundary:
            continue
        if (
            failed_transition
            and attempt.terminal_time == boundary
            and attempt.status
            in (
                HedgeAttemptStatus.FAILED_RUNNING,
                HedgeAttemptStatus.INVALIDATED_QUEUED,
            )
        ):
            continue
        replica = attempt.replica_id
        live[replica] += 1
        if attempt.start_time is None or attempt.start_time >= boundary:
            queued[replica] += 1
            remaining[replica] += attempt.required_work
        else:
            executed = work_executed_between(
                replica,
                attempt.start_time,
                boundary,
                timeline,
                slowdown,
            )
            remaining[replica] += max(0.0, attempt.required_work - executed)
    return (
        (queued[0], queued[1]),
        (live[0], live[1]),
        (remaining[0], remaining[1]),
    )


def _snapshot_matches(
    snapshot,
    expected: tuple[tuple[int, int], tuple[int, int], tuple[float, float]],
) -> bool:
    queued, live, remaining = expected
    return (
        snapshot.queued_attempts == queued
        and snapshot.live_attempts == live
        and all(
            math.isclose(actual, wanted, rel_tol=1e-9, abs_tol=1e-9)
            for actual, wanted in zip(snapshot.remaining_work, remaining)
        )
    )


def _recovery(run: EpisodeSimulationResult) -> dict[str, float | int]:
    recovered_start = run.episode.protocol.timeline.recovered_start
    cohort = [
        token
        for token in run.simulation.tokens
        if token.arrival_time < recovered_start
    ]
    if not cohort:
        return {
            "pre_recovery_arrival_count": 0,
            "logical_backlog_area": 0.0,
            "logical_clear_time": recovered_start,
            "logical_clear_delay": 0.0,
            "resource_clear_time": recovered_start,
            "resource_clear_delay": 0.0,
        }
    logical_clear = max(
        recovered_start, max(token.completion_time for token in cohort)
    )
    resource_attempts = [
        attempt
        for token in cohort
        for attempt in token.attempts
    ]
    resource_clear = max(
        recovered_start,
        max(attempt.terminal_time for attempt in resource_attempts),
    )
    return {
        "pre_recovery_arrival_count": len(cohort),
        "logical_backlog_area": math.fsum(
            max(0.0, token.completion_time - recovered_start) for token in cohort
        ),
        "logical_clear_time": logical_clear,
        "logical_clear_delay": max(0.0, logical_clear - recovered_start),
        "resource_clear_time": resource_clear,
        "resource_clear_delay": max(0.0, resource_clear - recovered_start),
    }


def _normalized_storm(
    run: EpisodeSimulationResult, slowdown: float, bin_width: float
) -> dict:
    """Measure storm behavior on the fixed half-open arrival clock.

    Drain is reported separately. A delayed timer accepted before cutoff may
    launch afterward; those launches remain lifecycle facts but are listed as
    post-cutoff counts and cannot dilute the fixed-window mean rate.
    """
    timeline = run.episode.protocol.timeline
    result = run.simulation
    horizon = run.episode.protocol.arrival_cutoff
    all_launches = [
        attempt.enqueue_time
        for attempt in result.attempts
        if attempt.attempt_id == 2
    ]
    launches = [moment for moment in all_launches if moment < horizon]
    post_cutoff_launches = [moment for moment in all_launches if moment >= horizon]
    full_bin_count = int(math.floor(horizon / bin_width))
    launch_bins = [0] * full_bin_count
    partial_launches = 0
    for moment in launches:
        index = int(moment // bin_width)
        if index < full_bin_count:
            launch_bins[index] += 1
        else:
            partial_launches += 1
    peak_rate = max(launch_bins, default=0) / bin_width
    mean_rate = len(launches) / horizon
    ratio = peak_rate / mean_rate if mean_rate > 0.0 else None

    has_partial = horizon - full_bin_count * bin_width > 1e-12
    total_bin_count = full_bin_count + int(has_partial)
    admitted_work = [[0.0] * total_bin_count for _ in range(2)]
    for attempt in result.attempts:
        index = int(attempt.enqueue_time // bin_width)
        if attempt.enqueue_time < horizon and index < total_bin_count:
            admitted_work[attempt.replica_id][index] += attempt.required_work

    overloaded_by_replica = [[False] * total_bin_count for _ in range(2)]
    load_ratio_by_replica: list[list[float | None]] = [[], []]
    overloaded: list[bool] = []
    widths: list[float] = []
    for index in range(total_bin_count):
        start = index * bin_width
        end = min(start + bin_width, horizon)
        width = end - start
        widths.append(width)
        phase = phase_at(timeline, start).value
        any_overloaded = False
        for replica in range(2):
            if replica == 1:
                capacity = 1.0
            elif phase == "D":
                capacity = 1.0 / slowdown
            elif phase == "F":
                capacity = 0.0
            else:
                capacity = 1.0
            demand_rate = admitted_work[replica][index] / width
            flag = demand_rate > capacity
            overloaded_by_replica[replica][index] = flag
            any_overloaded = any_overloaded or flag
            load_ratio_by_replica[replica].append(
                demand_rate / capacity if capacity > 0.0 else None
            )
        overloaded.append(any_overloaded)
    longest = 0.0
    current = 0.0
    for flag, width in zip(overloaded, widths):
        if flag:
            current += width
            longest = max(longest, current)
        else:
            current = 0.0

    return {
        "overload_basis": (
            "sampled_required_work_by_attempt_enqueue_bin_and_replica"
        ),
        "overload_rule": "any_replica_admitted_work_rate_gt_physical_capacity",
        "observation_interval": [0.0, horizon],
        "observation_end": horizon,
        "observation_duration": horizon,
        "bin_width": bin_width,
        "hedge_launch_count": len(launches),
        "post_cutoff_hedge_launch_count": len(post_cutoff_launches),
        "post_cutoff_hedge_launch_work": math.fsum(
            attempt.required_work
            for attempt in result.attempts
            if attempt.attempt_id == 2 and attempt.enqueue_time >= horizon
        ),
        "hedge_launch_bins": launch_bins,
        "partial_bin_hedge_launches": partial_launches,
        "peak_hedge_launch_rate": peak_rate,
        "mean_hedge_launch_rate": mean_rate,
        "peak_to_mean_hedge_launch_ratio": ratio,
        "overloaded_bin_count": sum(overloaded),
        "admitted_work_bins_by_replica": admitted_work,
        "admitted_load_ratio_bins_by_replica": load_ratio_by_replica,
        "overloaded_bins_by_replica": overloaded_by_replica,
        "sustained_overload_duration": longest,
    }


def build_episode_metrics(
    run: EpisodeSimulationResult,
    degraded_slowdown: float,
    *,
    deadlines: SLODeadlines | None = None,
    storm_bin_width: float = 1.0,
    placement_mode: str = "fixed_dispatcher",
) -> dict[str, Any]:
    """Build all attribution metrics for one independent episode."""
    if not isinstance(run, EpisodeSimulationResult):
        raise ValueError(
            f"run must be an EpisodeSimulationResult, got {run!r}"
        )
    slowdown = validate_slowdown(degraded_slowdown)
    if run.degraded_slowdown is not None:
        recorded_slowdown = validate_slowdown(run.degraded_slowdown)
        if slowdown != recorded_slowdown:
            raise ValueError(
                "degraded_slowdown does not match the value used to simulate "
                f"the episode: {slowdown!r} != {recorded_slowdown!r}"
            )
    width = _positive_real(storm_bin_width, "storm_bin_width")
    if placement_mode not in {
        "fixed_dispatcher",
        "idle_release",
        "budgeted_idle_release",
    }:
        raise ValueError(
            "placement_mode must be 'fixed_dispatcher', 'idle_release', "
            "or 'budgeted_idle_release'"
        )
    deadlines = SLODeadlines() if deadlines is None else deadlines
    if not isinstance(deadlines, SLODeadlines):
        raise ValueError(f"deadlines must be SLODeadlines, got {deadlines!r}")

    result = run.simulation
    tokens = result.tokens
    timeline = run.episode.protocol.timeline
    trace = run.episode.workload
    work, work_checks = _execution_work(run, slowdown)
    storm = _normalized_storm(run, slowdown, width)
    replayed = sum(token.replay_count > 0 for token in tokens)
    hedge_attempt_records = [
        attempt for attempt in result.attempts if attempt.attempt_id == 2
    ]
    hedge_winners = sum(
        attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
        for attempt in hedge_attempt_records
    )
    flattened_attempts = tuple(
        attempt for token in tokens for attempt in token.attempts
    )

    def expected_required_work(attempt) -> float:
        if attempt.attempt_id == 0:
            streams = trace.service_times
        elif attempt.attempt_id == 1:
            streams = trace.replay_service_times
        elif attempt.attempt_id == 2:
            streams = trace.hedge_service_times
        else:
            return math.nan
        if streams is None:
            return math.nan
        return streams[attempt.replica_id][attempt.token_id]

    token_facts_match_trace = len(tokens) == len(trace.tokens) and all(
        token.token_id == spec.token_id
        and token.arrival_time == spec.arrival_time
        and token.token_class is spec.token_class
        for token, spec in zip(tokens, trace.tokens)
    )
    dispatcher_assignment_matches_protocol = all(
        token.primary_replica
        == (
            1
            if state_at(timeline, token.arrival_time) is CommonState.FAILED
            else token.token_id % 2
        )
        for token in tokens
    )
    attempt_draws_match_trace = all(
        attempt.required_work == expected_required_work(attempt)
        for attempt in result.attempts
    )
    winner_facts_are_consistent = True
    token_algebra_is_consistent = True
    for token in tokens:
        winners = [
            attempt
            for attempt in token.attempts
            if attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
        ]
        if (
            len(winners) != 1
            or winners[0].attempt_id != token.winner_attempt_id
            or winners[0].terminal_time != token.completion_time
            or token.completion_phase
            is not phase_at(timeline, token.completion_time)
            or not math.isclose(
                token.latency,
                token.completion_time - token.arrival_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            winner_facts_are_consistent = False
        primary = [attempt for attempt in token.attempts if attempt.attempt_id == 0]
        replay_count = sum(attempt.attempt_id == 1 for attempt in token.attempts)
        if (
            len(primary) != 1
            or primary[0].replica_id != token.primary_replica
            or primary[0].enqueue_time != token.arrival_time
            or replay_count != token.replay_count
            or any(
                attempt.token_id != token.token_id for attempt in token.attempts
            )
        ):
            token_algebra_is_consistent = False

    copy_placement_matches_protocol = True
    for token in tokens:
        primary = next(
            (attempt for attempt in token.attempts if attempt.attempt_id == 0),
            None,
        )
        for attempt in token.attempts:
            if attempt.attempt_id == 2 and (
                token.primary_replica not in (0, 1)
                or attempt.replica_id != 1 - token.primary_replica
                or token.action is ProtectionAction.NORMAL
                or (
                    token.action is ProtectionAction.IMMEDIATE_HEDGE
                    and attempt.enqueue_time != token.arrival_time
                )
                or (
                    token.action is ProtectionAction.DELAYED_HEDGE
                    and (
                        run.hedge_delay is None
                        or attempt.enqueue_time
                        != token.arrival_time + run.hedge_delay
                    )
                )
            ):
                copy_placement_matches_protocol = False
            if attempt.attempt_id == 1 and (
                attempt.replica_id != 1
                or token.primary_replica != 0
                or primary is None
                or primary.status
                not in (
                    HedgeAttemptStatus.FAILED_RUNNING,
                    HedgeAttemptStatus.INVALIDATED_QUEUED,
                )
                or attempt.enqueue_time != timeline.failed_start
            ):
                copy_placement_matches_protocol = False

    primary_attempts = sum(attempt.attempt_id == 0 for attempt in result.attempts)
    replay_executions = sum(
        attempt.attempt_id == 1 and attempt.start_time is not None
        for attempt in result.attempts
    )
    hedge_attempts = sum(attempt.attempt_id == 2 for attempt in result.attempts)
    requested_hedges = sum(token.action.value != "N" for token in tokens)
    cancelled = sum(
        attempt.status is HedgeAttemptStatus.CANCELLED_QUEUED
        for attempt in result.attempts
    )
    completed_losers = sum(
        attempt.status is HedgeAttemptStatus.COMPLETED_LOSER
        for attempt in result.attempts
    )
    failed_running_primary = sum(
        attempt.attempt_id == 0
        and attempt.status is HedgeAttemptStatus.FAILED_RUNNING
        for attempt in result.attempts
    )
    invalidated_queued_primary = sum(
        attempt.attempt_id == 0
        and attempt.status is HedgeAttemptStatus.INVALIDATED_QUEUED
        for attempt in result.attempts
    )
    attempt_record_algebra = all(
        attempt.attempt_id in _KINDS
        and attempt.replica_id in (0, 1)
        and attempt.token_id in range(len(trace.tokens))
        and math.isfinite(attempt.enqueue_time)
        and math.isfinite(attempt.terminal_time)
        and math.isfinite(attempt.required_work)
        and math.isfinite(attempt.executed_work)
        and math.isfinite(attempt.remaining_work)
        and attempt.required_work > 0.0
        and -1e-12 <= attempt.executed_work <= attempt.required_work + 1e-9
        and attempt.remaining_work >= -1e-9
        and math.isclose(
            attempt.remaining_work,
            attempt.required_work - attempt.executed_work,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and attempt.terminal_time >= attempt.enqueue_time
        and (
            (
                attempt.start_time is None
                and attempt.queue_delay is None
                and math.isclose(attempt.executed_work, 0.0, abs_tol=1e-12)
                and attempt.status
                in (
                    HedgeAttemptStatus.CANCELLED_QUEUED,
                    HedgeAttemptStatus.INVALIDATED_QUEUED,
                )
            )
            or (
                attempt.start_time is not None
                and math.isfinite(attempt.start_time)
                and attempt.enqueue_time <= attempt.start_time
                <= attempt.terminal_time
                and attempt.queue_delay is not None
                and math.isclose(
                    attempt.queue_delay,
                    attempt.start_time - attempt.enqueue_time,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and attempt.status
                not in (
                    HedgeAttemptStatus.CANCELLED_QUEUED,
                    HedgeAttemptStatus.INVALIDATED_QUEUED,
                )
            )
        )
        and (
            attempt.status
            not in (
                HedgeAttemptStatus.COMPLETED_WINNER,
                HedgeAttemptStatus.COMPLETED_LOSER,
            )
            or (
                math.isclose(
                    attempt.executed_work,
                    attempt.required_work,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and math.isclose(attempt.remaining_work, 0.0, abs_tol=1e-9)
            )
        )
        for attempt in result.attempts
    )
    counters_are_non_negative_ints = all(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
        for value in (
            result.primary_executions,
            result.replay_executions,
            result.hedge_requested,
            result.hedge_launches,
            result.hedge_suppressed,
            result.hedge_timers_voided,
            result.cancelled_queued_total,
            result.completed_loser_total,
            result.failed_running_primary_executions,
            result.invalidated_queued_primary_executions,
            result.stale_completion_events_ignored,
        )
    )
    expected_failed_snapshot = _reconstruct_boundary_snapshot(
        run,
        slowdown,
        timeline.failed_start,
        failed_transition=True,
    )
    expected_recovered_snapshot = _reconstruct_boundary_snapshot(
        run,
        slowdown,
        timeline.recovered_start,
        failed_transition=False,
    )
    if placement_mode in {"idle_release", "budgeted_idle_release"}:
        # Idle-release has unassigned waiting work that has no Replica
        # identity until a later dispatch.  Its boundary state is therefore
        # the online capture made by the executor, not a terminal-record
        # reconstruction that can backfill a future Replica assignment.
        audit = getattr(run, "_idle_release_boundary_audit", None)
        if not isinstance(audit, dict) or set(audit) != {"failed", "recovered"}:
            raise RuntimeError("idle-release boundary audit is missing or incomplete")
        expected_failed_snapshot = (
            audit["failed"].snapshot.queued_attempts,
            audit["failed"].snapshot.live_attempts,
            audit["failed"].snapshot.remaining_work,
        )
        expected_recovered_snapshot = (
            audit["recovered"].snapshot.queued_attempts,
            audit["recovered"].snapshot.live_attempts,
            audit["recovered"].snapshot.remaining_work,
        )
    invariants = {
        "completed_tokens_equal_generated_tokens": result.completed_tokens
        == len(run.episode.workload.tokens),
        "all_arrivals_before_cutoff": all(
            token.arrival_time < run.episode.protocol.arrival_cutoff
            for token in tokens
        ),
        "exactly_one_winner_per_token": all(
            sum(
                attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
                for attempt in token.attempts
            )
            == 1
            for token in tokens
        ),
        "at_most_one_replay_per_token": all(token.replay_count <= 1 for token in tokens),
        "at_most_one_hedge_per_token": all(
            sum(attempt.attempt_id == 2 for attempt in token.attempts) <= 1
            for token in tokens
        ),
        "attempt_ids_unique_per_token": all(
            len({attempt.attempt_id for attempt in token.attempts})
            == len(token.attempts)
            for token in tokens
        ),
        "result_tokens_match_episode_trace": token_facts_match_trace,
        "dispatcher_assignment_matches_protocol": dispatcher_assignment_matches_protocol,
        "attempt_draws_match_episode_trace": attempt_draws_match_trace,
        "flat_attempt_view_matches_tokens": result.attempts == flattened_attempts,
        "winner_completion_facts_are_consistent": winner_facts_are_consistent,
        "per_token_attempt_algebra_is_consistent": token_algebra_is_consistent,
        "copy_placement_matches_protocol": copy_placement_matches_protocol,
        "attempt_record_algebra_is_consistent": attempt_record_algebra,
        "engine_counters_are_non_negative_ints": counters_are_non_negative_ints,
        "primary_execution_counter_matches_attempts": result.primary_executions
        == primary_attempts,
        "replay_execution_counter_matches_attempts": result.replay_executions
        == replay_executions,
        "hedge_requested_counter_matches_actions": result.hedge_requested
        == requested_hedges,
        "delayed_actions_have_hedge_delay": not any(
            token.action is ProtectionAction.DELAYED_HEDGE for token in tokens
        )
        or run.hedge_delay is not None,
        "hedge_launch_counter_matches_attempts": result.hedge_launches
        == hedge_attempts,
        "hedge_request_lifecycle_counters_match": result.hedge_requested
        == result.hedge_launches
        + result.hedge_suppressed
        + result.hedge_timers_voided,
        "hedge_launches_partition_at_cutoff": result.hedge_launches
        == storm["hedge_launch_count"]
        + storm["post_cutoff_hedge_launch_count"],
        "cancelled_counter_matches_attempts": result.cancelled_queued_total
        == cancelled,
        "completed_loser_counter_matches_attempts": result.completed_loser_total
        == completed_losers,
        "failed_running_counter_matches_attempts": result.failed_running_primary_executions
        == failed_running_primary,
        "invalidated_queued_counter_matches_attempts": result.invalidated_queued_primary_executions
        == invalidated_queued_primary,
        "executor_suppression_counter_matches_events": result.hedge_suppressed
        == len(result.hedge_suppressed_events),
        "last_arrival_matches_episode_trace": result.last_arrival_time
        == max(spec.arrival_time for spec in trace.tokens),
        "legacy_drain_duration_is_consistent": math.isclose(
            result.drain_duration,
            result.drain_end_time - result.last_arrival_time,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "boundary_queue_counts_match_legacy_result": run.failed_start_snapshot.queued_attempts
        == result.queue_length_at_failed_start
        and run.recovered_start_snapshot.queued_attempts
        == result.queue_length_at_recovered_start,
        "failed_boundary_snapshot_matches_attempt_lifecycle": _snapshot_matches(
            run.failed_start_snapshot, expected_failed_snapshot
        ),
        "recovered_boundary_snapshot_matches_attempt_lifecycle": _snapshot_matches(
            run.recovered_start_snapshot, expected_recovered_snapshot
        ),
        "drain_end_matches_last_attempt_terminal": math.isclose(
            result.drain_end_time,
            max(attempt.terminal_time for attempt in result.attempts),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "post_cutoff_drain_matches_protocol": math.isclose(
            run.post_cutoff_drain_duration,
            max(
                0.0,
                result.drain_end_time - run.episode.protocol.arrival_cutoff,
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "boundary_values_non_negative": all(
            value >= 0
            for snapshot in (
                run.failed_start_snapshot,
                run.recovered_start_snapshot,
            )
            for values in (snapshot.live_attempts, snapshot.remaining_work)
            for value in values
        ),
        "failed_replica_a_is_empty": run.failed_start_snapshot.live_attempts[0]
        == 0
        and math.isclose(
            run.failed_start_snapshot.remaining_work[0], 0.0, abs_tol=1e-12
        ),
        **work_checks,
    }
    placement_exceptions = {
        "dispatcher_assignment_matches_protocol",
        "copy_placement_matches_protocol",
        "boundary_queue_counts_match_legacy_result",
    }
    if placement_mode == "budgeted_idle_release":
        # Budgeted idle-release requests are generated when a Backup becomes
        # idle, not as one-shot arrival actions.  The existing counter
        # identity is retained in the audit, but its action-projection check
        # is not a physical invariant for this isolated scheduler.
        placement_exceptions.add("hedge_requested_counter_matches_actions")
    failed = [
        name
        for name, value in invariants.items()
        if value is False
        and not (
            placement_mode in {"idle_release", "budgeted_idle_release"}
            and name in placement_exceptions
        )
    ]
    if failed:
        raise RuntimeError(
            "attribution episode invariant failure: " + ", ".join(failed)
        )

    return {
        "episode_identity": {
            "namespace": run.episode.identity.namespace,
            "macro_seed": run.episode.identity.macro_seed,
            "episode_index": run.episode.identity.episode_index,
            "episode_seed": run.episode.episode_seed,
        },
        "trace_fingerprint": episode_trace_fingerprint(run.episode),
        "metric_contract": {
            "degraded_slowdown": slowdown,
            "deadlines": {
                "regular": deadlines.regular,
                "urgent": deadlines.urgent,
            },
            "storm_bin_width": width,
            "hedge_delay": run.hedge_delay,
        },
        "episode_protocol": {
            "timeline": {
                "degraded_start": timeline.degraded_start,
                "failed_start": timeline.failed_start,
                "recovered_start": timeline.recovered_start,
            },
            "arrival_cutoff": run.episode.protocol.arrival_cutoff,
            "arrival_interval": "[0, arrival_cutoff)",
            "drain_accepted_work": True,
        },
        "generated_tokens": len(tokens),
        "completed_tokens": result.completed_tokens,
        "latency": _latency_views(tokens, timeline, deadlines),
        "migration_matrix": _migration_matrix(tokens, timeline),
        "replay": {
            "replayed_tokens": replayed,
            "rate": replayed / len(tokens),
        },
        "hedge": {
            "requested": result.hedge_requested,
            "launched": len(hedge_attempt_records),
            "winners": hedge_winners,
            "cancelled_queued": sum(
                attempt.status is HedgeAttemptStatus.CANCELLED_QUEUED
                for attempt in hedge_attempt_records
            ),
            "completed_loser": sum(
                attempt.status is HedgeAttemptStatus.COMPLETED_LOSER
                for attempt in hedge_attempt_records
            ),
            "failed_running": sum(
                attempt.status is HedgeAttemptStatus.FAILED_RUNNING
                for attempt in hedge_attempt_records
            ),
            "invalidated_queued": sum(
                attempt.status is HedgeAttemptStatus.INVALIDATED_QUEUED
                for attempt in hedge_attempt_records
            ),
            "executor_suppressed": result.hedge_suppressed,
            "win_per_launch": (
                hedge_winners / len(hedge_attempt_records)
                if hedge_attempt_records
                else None
            ),
        },
        "work": work,
        "boundaries": {
            "failed_start": _snapshot_dict(run.failed_start_snapshot),
            "recovered_start": _snapshot_dict(run.recovered_start_snapshot),
        },
        "recovery": _recovery(run),
        "post_cutoff_drain_duration": run.post_cutoff_drain_duration,
        "storm": storm,
        "invariants": invariants,
    }


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sum_work(episode_metrics: Sequence[dict]) -> dict:
    scalar_names = (
        "nominal_primary_work",
        "executed_primary_work",
        "executed_hedge_work",
        "executed_replay_work",
        "total_executed_work",
        "wasted_work",
    )
    result = {
        name: math.fsum(item["work"][name] for item in episode_metrics)
        for name in scalar_names
    }
    result["waste_ratio"] = (
        result["wasted_work"] / result["total_executed_work"]
        if result["total_executed_work"] > 0.0
        else None
    )
    result["execution_amplification"] = (
        result["total_executed_work"] / result["nominal_primary_work"]
        if result["nominal_primary_work"] > 0.0
        else None
    )
    cells = {}
    for phase in _PHASES:
        cells[phase.value] = {}
        for replica in range(2):
            sources = [
                item["work"]["by_execution_phase_and_replica"][phase.value][
                    str(replica)
                ]
                for item in episode_metrics
            ]
            work_names = (
                "primary",
                "hedge",
                "replay",
                "total",
                "wasted",
                "fixed_window_primary",
                "fixed_window_hedge",
                "fixed_window_replay",
                "fixed_window_total",
                "fixed_window_wasted",
                "post_cutoff_primary",
                "post_cutoff_hedge",
                "post_cutoff_replay",
                "post_cutoff_total",
                "post_cutoff_wasted",
            )
            cell = {
                name: math.fsum(source[name] for source in sources)
                for name in work_names
            }
            cell["duration"] = math.fsum(source["duration"] for source in sources)
            cell["available_normalized_capacity"] = math.fsum(
                source["available_normalized_capacity"] for source in sources
            )
            capacity = cell["available_normalized_capacity"]
            cell["realized_utilization"] = (
                cell["total"] / capacity if capacity > 0.0 else None
            )
            cell["fixed_window_duration"] = math.fsum(
                source["fixed_window_duration"] for source in sources
            )
            cell["fixed_window_available_normalized_capacity"] = math.fsum(
                source["fixed_window_available_normalized_capacity"]
                for source in sources
            )
            fixed_capacity = cell["fixed_window_available_normalized_capacity"]
            cell["fixed_window_realized_utilization"] = (
                cell["fixed_window_total"] / fixed_capacity
                if fixed_capacity > 0.0
                else None
            )
            cells[phase.value][str(replica)] = cell
    result["by_execution_phase_and_replica"] = cells
    result["post_cutoff_by_replica"] = {
        str(replica): {
            name: math.fsum(
                cells[phase.value][str(replica)][f"post_cutoff_{name}"]
                for phase in _PHASES
            )
            for name in ("primary", "hedge", "replay", "total", "wasted")
        }
        for replica in range(2)
    }
    return result


def build_macro_seed_metrics(
    runs: Sequence[EpisodeSimulationResult],
    degraded_slowdown: float,
    *,
    expected_episode_count: int = 50,
    deadlines: SLODeadlines | None = None,
    storm_bin_width: float = 1.0,
) -> dict[str, Any]:
    """Aggregate episodes into exactly one inferential macro-seed value."""
    if (
        isinstance(expected_episode_count, bool)
        or not isinstance(expected_episode_count, int)
        or expected_episode_count <= 0
    ):
        raise ValueError(
            "expected_episode_count must be a positive int, got "
            f"{expected_episode_count!r}"
        )
    if expected_episode_count != 50:
        raise ValueError(
            "expected_episode_count is fixed at 50 for an inferential "
            f"macro-seed, got {expected_episode_count}"
        )
    episodes = tuple(runs)
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"expected {expected_episode_count} episodes, got {len(episodes)}"
        )
    if not all(isinstance(run, EpisodeSimulationResult) for run in episodes):
        raise ValueError("every run must be an EpisodeSimulationResult")

    first = episodes[0].episode
    namespace = first.identity.namespace
    macro_seed = first.identity.macro_seed
    protocol = first.protocol
    hedge_delay = episodes[0].hedge_delay
    for run in episodes:
        identity = run.episode.identity
        if identity.namespace != namespace:
            raise ValueError("all episodes must share one namespace")
        if identity.macro_seed != macro_seed:
            raise ValueError("all episodes must share one macro_seed")
        if run.episode.protocol != protocol:
            raise ValueError("all episodes must share one episode protocol")
        if run.hedge_delay != hedge_delay:
            raise ValueError("all episodes must share one hedge_delay contract")
    episodes = tuple(
        sorted(episodes, key=lambda run: run.episode.identity.episode_index)
    )
    indices = [run.episode.identity.episode_index for run in episodes]
    if indices != list(range(expected_episode_count)):
        raise ValueError(
            "episode indices must be unique, dense, and equal to "
            f"0..{expected_episode_count - 1}; got {indices!r}"
        )
    if len({run.episode.episode_seed for run in episodes}) != len(episodes):
        raise ValueError("episode seeds must be unique within a macro_seed")

    deadlines = SLODeadlines() if deadlines is None else deadlines
    episode_metrics = tuple(
        build_episode_metrics(
            run,
            degraded_slowdown,
            deadlines=deadlines,
            storm_bin_width=storm_bin_width,
        )
        for run in episodes
    )
    tokens = tuple(token for run in episodes for token in run.simulation.tokens)
    latency = _latency_views(tokens, protocol.timeline, deadlines)
    migration = {
        arrival.value: {
            completion.value: sum(
                item["migration_matrix"][arrival.value][completion.value]
                for item in episode_metrics
            )
            for completion in _PHASES
        }
        for arrival in _PHASES
    }
    replayed = sum(token.replay_count > 0 for token in tokens)
    work = _sum_work(episode_metrics)
    hedge_counts = {
        name: sum(item["hedge"][name] for item in episode_metrics)
        for name in (
            "requested",
            "launched",
            "winners",
            "cancelled_queued",
            "completed_loser",
            "failed_running",
            "invalidated_queued",
            "executor_suppressed",
        )
    }
    hedge_counts["win_per_launch"] = (
        hedge_counts["winners"] / hedge_counts["launched"]
        if hedge_counts["launched"]
        else None
    )

    peak_rates = [
        item["storm"]["peak_hedge_launch_rate"] for item in episode_metrics
    ]
    mean_rates = [
        item["storm"]["mean_hedge_launch_rate"] for item in episode_metrics
    ]
    total_launches = sum(
        item["storm"]["hedge_launch_count"] for item in episode_metrics
    )
    total_observation_duration = math.fsum(
        item["storm"]["observation_duration"] for item in episode_metrics
    )
    post_cutoff_launches = sum(
        item["storm"]["post_cutoff_hedge_launch_count"]
        for item in episode_metrics
    )
    post_cutoff_launch_work = math.fsum(
        item["storm"]["post_cutoff_hedge_launch_work"]
        for item in episode_metrics
    )
    ratios = [
        item["storm"]["peak_to_mean_hedge_launch_ratio"]
        for item in episode_metrics
    ]
    defined_ratios = [value for value in ratios if value is not None]
    all_ratios_defined = len(defined_ratios) == len(ratios)

    boundary_means = {}
    for boundary in ("failed_start", "recovered_start"):
        boundary_means[boundary] = {
            name: [
                _mean(
                    [item["boundaries"][boundary][name][replica] for item in episode_metrics]
                )
                for replica in range(2)
            ]
            for name in ("queued_attempts", "live_attempts", "remaining_work")
        }
    recovery_names = (
        "pre_recovery_arrival_count",
        "logical_backlog_area",
        "logical_clear_time",
        "logical_clear_delay",
        "resource_clear_time",
        "resource_clear_delay",
    )
    recovery_means = {
        name: _mean([item["recovery"][name] for item in episode_metrics])
        for name in recovery_names
    }
    invariants = {
        "all_episode_invariants_pass": all(
            all(item["invariants"].values()) for item in episode_metrics
        ),
        "pooled_token_count_matches_episode_sum": len(tokens)
        == sum(item["generated_tokens"] for item in episode_metrics),
        "inferential_unit_is_one_macro_seed": True,
        "hedge_launches_partition_at_cutoff": hedge_counts["launched"]
        == total_launches + post_cutoff_launches,
    }

    return {
        "namespace": namespace,
        "macro_seed": macro_seed,
        "inferential_unit": "macro_seed",
        "inferential_n": 1,
        "episode_count": len(episodes),
        "token_count": len(tokens),
        "generated_tokens": len(tokens),
        "completed_tokens": sum(
            item["completed_tokens"] for item in episode_metrics
        ),
        "aggregation_contract": "token_pool_rate_components_episode_time_mean",
        "metric_contract": episode_metrics[0]["metric_contract"],
        "episode_protocol": episode_metrics[0]["episode_protocol"],
        "latency": latency,
        "migration_matrix": migration,
        "replay": {
            "replayed_tokens": replayed,
            "generated_tokens": len(tokens),
            "rate": replayed / len(tokens),
        },
        "hedge": hedge_counts,
        "storm": {
            "in_window_hedge_launch_count": total_launches,
            "post_cutoff_hedge_launch_count": post_cutoff_launches,
            "post_cutoff_hedge_launch_work": post_cutoff_launch_work,
            "total_hedge_launch_count": total_launches
            + post_cutoff_launches,
            "fixed_window_total_duration": total_observation_duration,
            "mean_hedge_launch_rate": total_launches
            / total_observation_duration,
        },
        "work": work,
        "episode_time_means": {
            "peak_hedge_launch_rate": _mean(peak_rates),
            "mean_hedge_launch_rate": total_launches
            / total_observation_duration,
            "arithmetic_mean_episode_launch_rate": _mean(mean_rates),
            "peak_to_mean_hedge_launch_ratio": (
                _mean(defined_ratios) if all_ratios_defined else None
            ),
            "peak_to_mean_ratio_defined_episode_count": len(defined_ratios),
            "peak_to_mean_ratio_undefined_episode_count": len(ratios)
            - len(defined_ratios),
            "defined_only_peak_to_mean_ratio_mean": (
                _mean(defined_ratios) if defined_ratios else None
            ),
            "sustained_overload_duration": _mean(
                [
                    item["storm"]["sustained_overload_duration"]
                    for item in episode_metrics
                ]
            ),
            "post_cutoff_drain_duration": _mean(
                [item["post_cutoff_drain_duration"] for item in episode_metrics]
            ),
            "boundaries": boundary_means,
            "recovery": recovery_means,
        },
        "episode_audit": {
            "episode_indices": indices,
            "episode_seeds": [run.episode.episode_seed for run in episodes],
            "trace_fingerprints": [
                episode_trace_fingerprint(run.episode) for run in episodes
            ],
        },
        "invariants": invariants,
    }
