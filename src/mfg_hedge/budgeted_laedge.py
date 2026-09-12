"""Causal work-budgeted LÆDGE idle-release execution.

This module is deliberately isolated from the historical LÆDGE entry point.
It adds non-refundable Hedge admission and window accounting while reusing the
existing two-Replica event lifecycle.  It does not run calibration or an
experiment and is not exported from :mod:`mfg_hedge`'s public API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .attribution_episode import EpisodeSimulationResult, EpisodeTrace
from .common_state import Phase, phase_at, validate_slowdown
from .domain import ProtectionAction
from .hedge_simulation import (
    HedgeAttemptStatus,
    HedgeSimulationResult,
    HedgeTokenResult,
)
from .laedge_episode import (
    _LAEdgeEngine,
    _Pending,
    _Running,
)
from .workload import (
    validate_hedge_stream_shape,
    validate_replay_stream_shape,
    validate_stream_values,
    validate_token_specs,
    validate_trace_shape,
)


BUDGETED_LAEDGE_WINDOW_WIDTH = 25.0
BUDGETED_LAEDGE_EXPECTED_WORK = 1.0
BUDGETED_LAEDGE_DELTA_POINTS = (
    0.0,
    0.01,
    0.03,
    0.05,
    0.08,
    0.12,
    0.18,
)
BUDGETED_LAEDGE_ARM_KEYS = (
    "fixed-dispatcher:no-hedge",
    "niin:conservative",
    "budgeted-laedge:delta=0%",
    "budgeted-laedge:delta=1%",
    "budgeted-laedge:delta=3%",
    "budgeted-laedge:delta=5%",
    "budgeted-laedge:delta=8%",
    "budgeted-laedge:delta=12%",
    "budgeted-laedge:delta=18%",
    "unconstrained-laedge:conservative",
    "unconstrained-laedge:preemptive-oracle",
)
BUDGETED_LAEDGE_DEVELOPMENT_EPISODES = 256
BUDGETED_LAEDGE_HOLDOUT_EPISODES = 1024
BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v1:development"
)
BUDGETED_LAEDGE_HOLDOUT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v1:holdout"
)
BUDGETED_LAEDGE_DEVELOPMENT_MACRO_SEED = 20260910
BUDGETED_LAEDGE_HOLDOUT_MACRO_SEED = 20260911
BUDGETED_LAEDGE_ARM_COUNT = 11
BUDGETED_LAEDGE_CALL_BUDGET = (
    (BUDGETED_LAEDGE_DEVELOPMENT_EPISODES + BUDGETED_LAEDGE_HOLDOUT_EPISODES)
    * BUDGETED_LAEDGE_ARM_COUNT
)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative real")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real")
    return result


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


@dataclass(frozen=True)
class BudgetWindowAudit:
    """Immutable audit for one forced/width-limited budget window."""

    start: float
    end: float
    phase: Phase
    planned_budget: float
    nonrefundable_charged_work: float
    outstanding_expected_work: float
    observed_realized_hedge_work: float
    launch_cohort_realized_work: float
    execution_time_realized_work: float
    budget_error: float
    overshoot: float
    unused_budget: float
    hedge_launches: int
    hedge_suppressed: int

    def __post_init__(self) -> None:
        start = _finite_nonnegative(self.start, "start")
        end = _finite_nonnegative(self.end, "end")
        if end <= start:
            raise ValueError("budget window end must be strictly after start")
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be a Phase")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        for name in (
            "planned_budget",
            "nonrefundable_charged_work",
            "outstanding_expected_work",
            "observed_realized_hedge_work",
            "launch_cohort_realized_work",
            "execution_time_realized_work",
            "overshoot",
            "unused_budget",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        object.__setattr__(self, "budget_error", _finite_real(self.budget_error, "budget_error"))
        if not math.isclose(
            self.budget_error,
            self.launch_cohort_realized_work - self.planned_budget,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("budget_error must equal launch-cohort work minus plan")
        if not math.isclose(
            self.observed_realized_hedge_work,
            self.execution_time_realized_work,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "observed realized Hedge work must equal execution-time work"
            )
        if not math.isclose(
            self.overshoot,
            max(0.0, self.budget_error),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.unused_budget,
            max(0.0, -self.budget_error),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("overshoot and unused_budget must match budget_error")
        for name in ("hedge_launches", "hedge_suppressed"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def committed_work(self) -> float:
        """Compatibility name for the non-refundable charged quantity."""

        return self.nonrefundable_charged_work

    @property
    def signed_realized_minus_outstanding(self) -> float:
        """Audit-only reconciliation; it never releases admission allowance."""

        return self.observed_realized_hedge_work - self.outstanding_expected_work


@dataclass(frozen=True)
class BudgetedLaedgeResult:
    """One fresh physical run plus its immutable budget-window audit."""

    simulation: EpisodeSimulationResult
    windows: tuple[BudgetWindowAudit, ...]
    arm_key: str
    planned_budget_rate: float
    expected_work_per_hedge: float = BUDGETED_LAEDGE_EXPECTED_WORK

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, EpisodeSimulationResult):
            raise ValueError("simulation must be an EpisodeSimulationResult")
        if not isinstance(self.windows, tuple) or not self.windows:
            raise ValueError("windows must be a non-empty tuple")
        if any(not isinstance(window, BudgetWindowAudit) for window in self.windows):
            raise ValueError("windows must contain BudgetWindowAudit values")
        if self.windows[0].start != 0.0:
            raise ValueError("budget windows must start at time zero")
        if any(
            left.end != right.start
            for left, right in zip(self.windows, self.windows[1:])
        ):
            raise ValueError("budget windows must be contiguous")
        if not isinstance(self.arm_key, str) or not self.arm_key:
            raise ValueError("arm_key must be a non-empty string")
        rate = _finite_nonnegative(self.planned_budget_rate, "planned_budget_rate")
        expected = _finite_nonnegative(
            self.expected_work_per_hedge, "expected_work_per_hedge"
        )
        if expected != BUDGETED_LAEDGE_EXPECTED_WORK:
            raise ValueError("v1 expected_work_per_hedge is fixed at 1.0")
        object.__setattr__(self, "planned_budget_rate", rate)
        object.__setattr__(self, "expected_work_per_hedge", expected)


@dataclass
class _MutableBudgetWindow:
    start: float
    end: float
    phase: Phase
    planned_budget: float
    nonrefundable_charged_work: float = 0.0
    observed_realized_hedge_work: float = 0.0
    launch_cohort_realized_work: float = 0.0
    execution_time_realized_work: float = 0.0
    hedge_launches: int = 0
    hedge_suppressed: int = 0


class _BudgetedLAEdgeEngine(_LAEdgeEngine):
    def __init__(
        self,
        episode: EpisodeTrace,
        slowdown: float,
        *,
        planned_budget_rate: float,
        cancel_running_losers: bool,
    ) -> None:
        super().__init__(
            episode,
            slowdown,
            cancel_running_losers=cancel_running_losers,
        )
        self.planned_budget_rate = planned_budget_rate
        self.hedge_expected_work = BUDGETED_LAEDGE_EXPECTED_WORK
        self.hedge_requested = 0
        self.hedge_suppressed = 0
        self.suppressed_events: list[tuple[float, int]] = []
        self._windows: dict[float, _MutableBudgetWindow] = {}
        self._launch_window_by_key: dict[tuple[int, int, int], float] = {}
        self._hedge_lifecycle: dict[tuple[int, int, int], tuple[float, float | None]] = {}

    def _window_end(self, start: float) -> float:
        boundaries = (
            self.timeline.degraded_start,
            self.timeline.failed_start,
            self.timeline.recovered_start,
        )
        candidates = [start + BUDGETED_LAEDGE_WINDOW_WIDTH]
        candidates.extend(value for value in boundaries if value > start)
        return min(candidates)

    def _window_at(self, time: float) -> _MutableBudgetWindow:
        if isinstance(time, bool) or not isinstance(time, (int, float)):
            raise ValueError("window time must be numeric")
        time = float(time)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("window time must be finite and non-negative")
        start = 0.0
        while True:
            end = self._window_end(start)
            if time < end:
                window = self._windows.get(start)
                if window is None:
                    midpoint = (start + end) / 2.0
                    window = _MutableBudgetWindow(
                        start=start,
                        end=end,
                        phase=phase_at(self.timeline, midpoint),
                        planned_budget=self.planned_budget_rate * (end - start),
                    )
                    self._windows[start] = window
                return window
            start = end

    def _allocate_execution(
        self,
        start: float,
        end: float,
        speed: float,
        executed_delta: float,
        launch_window_start: float,
    ) -> None:
        if end <= start or executed_delta <= 0.0:
            return
        launch_window = self._window_at(launch_window_start)
        launch_window.launch_cohort_realized_work += executed_delta
        remaining = executed_delta
        cursor = start
        while cursor < end:
            window = self._window_at(cursor)
            segment_end = min(end, window.end)
            segment = (
                remaining
                if segment_end == end
                else min(remaining, max(0.0, (segment_end - cursor) * speed))
            )
            window.execution_time_realized_work += segment
            window.observed_realized_hedge_work += segment
            remaining -= segment
            cursor = segment_end
        if abs(remaining) > 1e-9:
            raise RuntimeError("budget window execution allocation did not conserve work")

    def _settle(self, running: _Running, now: float) -> None:
        previous = running.executed_work
        start = running.last_update
        speed = running.speed
        launch_key = (running.token_id, running.attempt_id, running.replica_id)
        super()._settle(running, now)
        if running.attempt_id == 2:
            launch_window_start = self._launch_window_by_key.get(launch_key)
            if launch_window_start is None:
                raise RuntimeError("Hedge settlement has no launch-window audit")
            self._allocate_execution(
                start,
                now,
                speed,
                running.executed_work - previous,
                launch_window_start,
            )

    def _start(self, pending: _Pending, replica: int, now: float) -> None:
        super()._start(pending, replica, now)
        if pending.attempt_id == 2:
            key = (pending.token_id, pending.attempt_id, replica)
            window = self._window_at(now)
            window.nonrefundable_charged_work += self.hedge_expected_work
            window.hedge_launches += 1
            self._launch_window_by_key[key] = window.start
            self._hedge_lifecycle[key] = (now, None)

    def _record(
        self,
        running: _Running,
        status: HedgeAttemptStatus,
        terminal_time: float,
    ) -> None:
        super()._record(running, status, terminal_time)
        if running.attempt_id == 2:
            key = (running.token_id, running.attempt_id, running.replica_id)
            launched = self._launch_window_by_key.get(key)
            if launched is None:
                raise RuntimeError("Hedge terminal record has no launch-window audit")
            launch_time = self._hedge_lifecycle.get(key, (None, None))[0]
            if launch_time is None:
                raise RuntimeError("Hedge terminal record has no launch time")
            self._hedge_lifecycle[key] = (launch_time, terminal_time)

    def _release(self, now: float) -> None:
        # Keep the historical LÆDGE Primary/Replay release order exactly.
        while self.waiting and self._idle(now):
            idle = self._idle(now)
            selected = next(
                (
                    index
                    for index, item in enumerate(self.waiting)
                    if item.pinned_replica is None or item.pinned_replica in idle
                ),
                None,
            )
            if selected is None:
                break
            pending = self.waiting[selected]
            del self.waiting[selected]
            if self.tokens[pending.token_id].winner is not None:
                continue
            replica = (
                pending.pinned_replica
                if pending.pinned_replica is not None
                else idle[0]
            )
            self._start(pending, replica, now)

        # Only after every startable unserved item is handled may a Hedge be
        # considered.  A suppressed oldest candidate blocks this opportunity;
        # a later candidate cannot leapfrog it.
        while self._idle(now):
            candidates = [
                token_id
                for token_id, token in enumerate(self.tokens)
                if token.winner is None
                and not token.hedge_launched
                and len(self._live_replicas(token_id)) == 1
            ]
            if not candidates:
                break
            token_id = min(
                candidates,
                key=lambda value: (self.tokens[value].arrival_time, value),
            )
            self.hedge_requested += 1
            window = self._window_at(now)
            if (
                window.nonrefundable_charged_work + self.hedge_expected_work
                > window.planned_budget + 1e-12
            ):
                self.hedge_suppressed += 1
                window.hedge_suppressed += 1
                self.suppressed_events.append((now, token_id))
                break
            replica = self._idle(now)[0]
            if replica in self._live_replicas(token_id):
                break
            self.tokens[token_id].hedge_launched = True
            self.hedge_launches += 1
            self._start(_Pending(token_id, 2, now), replica, now)

    def finalize_windows(self, drain_end: float) -> tuple[BudgetWindowAudit, ...]:
        if not math.isfinite(drain_end) or drain_end < 0.0:
            raise ValueError("drain_end must be finite and non-negative")
        if drain_end > 0.0:
            cursor = 0.0
            while cursor < drain_end:
                window = self._window_at(cursor)
                cursor = window.end
        audits = []
        for start in sorted(self._windows):
            window = self._windows[start]
            outstanding = sum(
                self.hedge_expected_work
                for launch_time, terminal_time in self._hedge_lifecycle.values()
                if launch_time < window.end
                and (terminal_time is None or terminal_time > window.end)
            )
            budget_error = (
                window.launch_cohort_realized_work - window.planned_budget
            )
            audits.append(
                BudgetWindowAudit(
                    start=window.start,
                    end=window.end,
                    phase=window.phase,
                    planned_budget=window.planned_budget,
                    nonrefundable_charged_work=window.nonrefundable_charged_work,
                    outstanding_expected_work=outstanding,
                    observed_realized_hedge_work=window.observed_realized_hedge_work,
                    launch_cohort_realized_work=window.launch_cohort_realized_work,
                    execution_time_realized_work=window.execution_time_realized_work,
                    budget_error=budget_error,
                    overshoot=max(0.0, budget_error),
                    unused_budget=max(0.0, -budget_error),
                    hedge_launches=window.hedge_launches,
                    hedge_suppressed=window.hedge_suppressed,
                )
            )
        if not audits:
            raise RuntimeError("Budgeted LÆDGE produced no budget window audit")
        return tuple(audits)


def _validate_episode(episode: EpisodeTrace, slowdown: float) -> None:
    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    trace = episode.workload
    validate_trace_shape(trace, 2)
    validate_replay_stream_shape(trace, 2)
    validate_hedge_stream_shape(trace, 2)
    validate_token_specs(trace)
    validate_stream_values(trace.service_times, 0, "service")
    validate_stream_values(trace.replay_service_times, 1, "replay")
    validate_stream_values(trace.hedge_service_times, 2, "hedge")
    validate_slowdown(slowdown)


def _result_from_engine(
    episode: EpisodeTrace,
    engine: _BudgetedLAEdgeEngine,
    slowdown: float,
) -> EpisodeSimulationResult:
    trace = episode.workload
    token_rows = []
    for spec, state in zip(trace.tokens, engine.tokens):
        if state.completion_time is None or state.winner is None or state.primary_replica is None:
            raise RuntimeError(f"Budgeted LÆDGE terminated with Token {spec.token_id} incomplete")
        token_rows.append(
            HedgeTokenResult(
                token_id=spec.token_id,
                token_class=spec.token_class,
                arrival_time=spec.arrival_time,
                primary_replica=state.primary_replica,
                action=(
                    ProtectionAction.IMMEDIATE_HEDGE
                    if state.hedge_launched
                    else ProtectionAction.NORMAL
                ),
                attempts=tuple(sorted(state.records, key=lambda row: row.attempt_id)),
                winner_attempt_id=state.winner,
                completion_time=state.completion_time,
                completion_phase=phase_at(episode.protocol.timeline, state.completion_time),
                replay_count=state.replay_count,
                latency=state.completion_time - spec.arrival_time,
            )
        )
    tokens = tuple(token_rows)
    attempts = tuple(row for token in tokens for row in token.attempts)
    drain_end = max(row.terminal_time for row in attempts)
    last_arrival = max(spec.arrival_time for spec in trace.tokens)
    simulation = HedgeSimulationResult(
        tokens=tokens,
        attempts=attempts,
        primary_executions=len(tokens),
        replay_executions=engine.replay_executions,
        hedge_requested=engine.hedge_requested,
        hedge_launches=engine.hedge_launches,
        hedge_suppressed=engine.hedge_suppressed,
        hedge_timers_voided=0,
        cancelled_queued_total=0,
        completed_loser_total=engine.completed_loser_total,
        failed_running_primary_executions=engine.failed_running_primaries,
        invalidated_queued_primary_executions=0,
        last_arrival_time=last_arrival,
        drain_end_time=drain_end,
        drain_duration=drain_end - last_arrival,
        stale_completion_events_ignored=engine.stale,
        queue_length_at_failed_start=(0, 0),
        queue_length_at_recovered_start=(0, 0),
        hedge_suppressed_events=tuple(engine.suppressed_events),
    )
    try:
        failed = engine.boundary_audits["failed"].snapshot
        recovered = engine.boundary_audits["recovered"].snapshot
    except KeyError as exc:
        raise RuntimeError("Budgeted LÆDGE did not capture both boundary states") from exc
    result = EpisodeSimulationResult(
        episode=episode,
        simulation=simulation,
        failed_start_snapshot=failed,
        recovered_start_snapshot=recovered,
        post_cutoff_drain_duration=max(
            0.0,
            drain_end - episode.protocol.arrival_cutoff,
        ),
        degraded_slowdown=slowdown,
        hedge_delay=None,
    )
    object.__setattr__(result, "_idle_release_boundary_audit", dict(engine.boundary_audits))
    return result


def simulate_budgeted_laedge_episode(
    episode: EpisodeTrace,
    *,
    planned_budget_rate: float,
    degraded_slowdown: float = 2.0,
    cancel_running_losers: bool = False,
) -> BudgetedLaedgeResult:
    """Run one isolated Budgeted LÆDGE episode with a causal work budget."""

    rate = _finite_nonnegative(planned_budget_rate, "planned_budget_rate")
    if type(cancel_running_losers) is not bool:
        raise ValueError("cancel_running_losers must be bool")
    slowdown = validate_slowdown(degraded_slowdown)
    _validate_episode(episode, slowdown)
    engine = _BudgetedLAEdgeEngine(
        episode,
        slowdown,
        planned_budget_rate=rate,
        cancel_running_losers=cancel_running_losers,
    )
    engine.run()
    simulation = _result_from_engine(episode, engine, slowdown)
    drain_end = simulation.simulation.drain_end_time
    arm_key = (
        "budgeted-laedge:delta=0%"
        if rate == 0.0
        else "budgeted-laedge:custom"
    )
    return BudgetedLaedgeResult(
        simulation=simulation,
        windows=engine.finalize_windows(drain_end),
        arm_key=arm_key,
        planned_budget_rate=rate,
    )


__all__ = [
    "BUDGETED_LAEDGE_ARM_KEYS",
    "BUDGETED_LAEDGE_ARM_COUNT",
    "BUDGETED_LAEDGE_CALL_BUDGET",
    "BUDGETED_LAEDGE_DELTA_POINTS",
    "BUDGETED_LAEDGE_DEVELOPMENT_EPISODES",
    "BUDGETED_LAEDGE_DEVELOPMENT_MACRO_SEED",
    "BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE",
    "BUDGETED_LAEDGE_EXPECTED_WORK",
    "BUDGETED_LAEDGE_HOLDOUT_EPISODES",
    "BUDGETED_LAEDGE_HOLDOUT_MACRO_SEED",
    "BUDGETED_LAEDGE_HOLDOUT_NAMESPACE",
    "BUDGETED_LAEDGE_WINDOW_WIDTH",
    "BudgetWindowAudit",
    "BudgetedLaedgeResult",
    "simulate_budgeted_laedge_episode",
]
