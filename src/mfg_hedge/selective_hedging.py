"""Causal fault-aware and deadline-aware selective Hedge execution.

This module is an isolated implementation slice on top of the accepted
Budgeted LÆDGE executor.  It changes only the causal Hedge eligibility gate
and candidate ordering; the event engine, CRN streams, conservative loser
semantics, queue priority, and budget accounting remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .attribution_episode import EpisodeSimulationResult, EpisodeTrace
from .attribution_metrics import SLODeadlines
from .budgeted_laedge import (
    BUDGETED_LAEDGE_EXPECTED_WORK,
    BudgetWindowAudit,
    _BudgetedLAEdgeEngine,
    _finite_nonnegative,
    _result_from_engine,
    _validate_episode,
)
from .common_state import Phase, phase_at, state_at, validate_slowdown
from .domain import CommonState, TokenClass
from .laedge_episode import _Pending
from .workload import TokenSpec


_SELECTIVE_THRESHOLD_VALUES = (0.15, 0.30, 0.50)
SELECTIVE_HEDGE_THRESHOLD_GRID = _SELECTIVE_THRESHOLD_VALUES


def _finite_unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite real in [0, 1]")
    return result


@dataclass(frozen=True)
class SelectiveHedgeThresholds:
    """Unitless normalized-slack thresholds for the two Token classes."""

    regular: float = 0.30
    urgent: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "regular", _finite_unit_interval(self.regular, "regular"))
        object.__setattr__(self, "urgent", _finite_unit_interval(self.urgent, "urgent"))

    def for_class(self, token_class: TokenClass) -> float:
        if token_class is TokenClass.REGULAR:
            return self.regular
        if token_class is TokenClass.URGENT:
            return self.urgent
        raise ValueError(f"token_class must be a TokenClass, got {token_class!r}")


@dataclass(frozen=True)
class SelectiveSuppression:
    """One causal reason why a potential Hedge was not admitted."""

    time: float
    token_id: int
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or not isinstance(self.time, (int, float)):
            raise ValueError("suppression time must be numeric")
        time = float(self.time)
        if not math.isfinite(time) or time < 0.0:
            raise ValueError("suppression time must be finite and non-negative")
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, int) or self.token_id < 0:
            raise ValueError("suppressed token_id must be a non-negative int")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("suppression reason must be a non-empty string")
        object.__setattr__(self, "time", time)


@dataclass(frozen=True)
class SelectiveHedgeResult:
    """One selective-Hedge simulation and its causal suppression audit."""

    simulation: EpisodeSimulationResult
    windows: tuple[BudgetWindowAudit, ...]
    thresholds: SelectiveHedgeThresholds
    deadlines: SLODeadlines
    suppression_reasons: tuple[SelectiveSuppression, ...]
    planned_budget_rate: float
    arm_key: str
    expected_work_per_hedge: float = BUDGETED_LAEDGE_EXPECTED_WORK

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, EpisodeSimulationResult):
            raise ValueError("simulation must be an EpisodeSimulationResult")
        if not isinstance(self.windows, tuple) or not self.windows:
            raise ValueError("windows must be a non-empty tuple")
        if any(not isinstance(item, BudgetWindowAudit) for item in self.windows):
            raise ValueError("windows must contain BudgetWindowAudit values")
        if not isinstance(self.thresholds, SelectiveHedgeThresholds):
            raise ValueError("thresholds must be SelectiveHedgeThresholds")
        if not isinstance(self.deadlines, SLODeadlines):
            raise ValueError("deadlines must be SLODeadlines")
        if not isinstance(self.arm_key, str) or not self.arm_key:
            raise ValueError("arm_key must be a non-empty string")
        if any(not isinstance(item, SelectiveSuppression) for item in self.suppression_reasons):
            raise ValueError("suppression_reasons must contain SelectiveSuppression values")
        object.__setattr__(
            self,
            "planned_budget_rate",
            _finite_nonnegative(self.planned_budget_rate, "planned_budget_rate"),
        )
        if self.expected_work_per_hedge != BUDGETED_LAEDGE_EXPECTED_WORK:
            raise ValueError("expected_work_per_hedge is fixed at 1.0")


def normalized_slack(
    token: TokenSpec,
    now: float,
    deadlines: SLODeadlines,
) -> float:
    """Return the observable, unitless deadline slack ``s/deadline``."""

    if not isinstance(token, TokenSpec):
        raise ValueError("token must be a TokenSpec")
    if not isinstance(deadlines, SLODeadlines):
        raise ValueError("deadlines must be SLODeadlines")
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValueError("now must be numeric")
    now = float(now)
    if not math.isfinite(now) or now < 0.0:
        raise ValueError("now must be finite and non-negative")
    deadline = deadlines.for_class(token.token_class)
    return (deadline - (now - token.arrival_time)) / deadline


def selective_candidate_key(
    token: TokenSpec,
    now: float,
    deadlines: SLODeadlines,
    *,
    class_priority: int,
) -> tuple[float, int, float, int]:
    """Stable causal ordering key for simultaneously eligible candidates."""

    if isinstance(class_priority, bool) or not isinstance(class_priority, int) or class_priority < 0:
        raise ValueError("class_priority must be a non-negative int")
    return (
        normalized_slack(token, now, deadlines),
        class_priority,
        float(token.arrival_time),
        token.token_id,
    )


def _class_priority(token_class: TokenClass) -> int:
    if token_class is TokenClass.URGENT:
        return 0
    if token_class is TokenClass.REGULAR:
        return 1
    raise ValueError(f"token_class must be a TokenClass, got {token_class!r}")


def selective_arm_key(thresholds: SelectiveHedgeThresholds) -> str:
    """Return the canonical deterministic identity for one threshold pair."""

    if not isinstance(thresholds, SelectiveHedgeThresholds):
        raise ValueError("thresholds must be SelectiveHedgeThresholds")
    return (
        "selective-laedge:"
        f"theta_regular={thresholds.regular:g}:"
        f"theta_urgent={thresholds.urgent:g}"
    )


class _SelectiveLAEdgeEngine(_BudgetedLAEdgeEngine):
    def __init__(
        self,
        episode: EpisodeTrace,
        slowdown: float,
        *,
        planned_budget_rate: float,
        cancel_running_losers: bool,
        thresholds: SelectiveHedgeThresholds,
        deadlines: SLODeadlines,
    ) -> None:
        super().__init__(
            episode,
            slowdown,
            planned_budget_rate=planned_budget_rate,
            cancel_running_losers=cancel_running_losers,
        )
        self.thresholds = thresholds
        self.deadlines = deadlines
        self.suppression_reasons: list[SelectiveSuppression] = []

    def _record_suppression(
        self,
        now: float,
        token_id: int,
        reason: str,
        *,
        executor_suppression: bool,
    ) -> None:
        # Eligibility diagnostics are observable policy outcomes, but they
        # are not executor admission suppressions.  Only an already-requested
        # Hedge rejected by the budget belongs in the legacy lifecycle
        # counters and suppression-event stream.
        if executor_suppression:
            self.hedge_suppressed += 1
            self.suppressed_events.append((now, token_id))
            window = self._window_at(now)
            window.hedge_suppressed += 1
        self.suppression_reasons.append(SelectiveSuppression(now, token_id, reason))

    def _eligibility_reason(self, token_id: int, now: float) -> str | None:
        token = self.tokens[token_id]
        state = state_at(self.timeline, now)
        phase = phase_at(self.timeline, now)
        if phase is Phase.HEALTHY:
            return "state_healthy"
        if state is CommonState.FAILED:
            return "state_failed"
        if phase is Phase.RECOVERED:
            return "state_recovered"
        if token.primary_replica != 0:
            return "primary_not_on_degraded_replica"
        if not self._available(1, now):
            return "backup_not_healthy_or_startable"
        if token.hedge_launched:
            return "hedge_already_launched"
        spec = self.trace.tokens[token_id]
        slack = normalized_slack(spec, now, self.deadlines)
        if slack <= 0.0:
            return "deadline_expired"
        if slack > self.thresholds.for_class(spec.token_class):
            return "slack_above_threshold"
        return None

    def _release(self, now: float) -> None:
        # Preserve Budgeted LÆDGE's exact Primary/Replay priority and queue
        # handling.  Only the Hedge candidate gate below is specialized.
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

        while self._idle(now):
            if 1 not in self._idle(now):
                break
            candidates: list[int] = []
            for token_id, token in enumerate(self.tokens):
                if token.winner is not None or token.hedge_launched:
                    continue
                if len(self._live_replicas(token_id)) != 1:
                    continue
                reason = self._eligibility_reason(token_id, now)
                if reason is None:
                    candidates.append(token_id)
                elif reason not in {"state_healthy", "state_recovered", "backup_not_healthy_or_startable"}:
                    self._record_suppression(
                        now,
                        token_id,
                        reason,
                        executor_suppression=False,
                    )
            if not candidates:
                break
            token_id = min(
                candidates,
                key=lambda value: selective_candidate_key(
                    self.trace.tokens[value],
                    now,
                    self.deadlines,
                    class_priority=_class_priority(self.trace.tokens[value].token_class),
                ),
            )
            self.hedge_requested += 1
            window = self._window_at(now)
            if (
                window.nonrefundable_charged_work + self.hedge_expected_work
                > window.planned_budget + 1e-12
            ):
                self._record_suppression(
                    now,
                    token_id,
                    "budget",
                    executor_suppression=True,
                )
                break
            replica = self._idle(now)[0]
            if replica in self._live_replicas(token_id):
                break
            self.tokens[token_id].hedge_launched = True
            self.hedge_launches += 1
            self._start(_Pending(token_id, 2, now), replica, now)


def simulate_selective_laedge_episode(
    episode: EpisodeTrace,
    *,
    planned_budget_rate: float,
    deadlines: SLODeadlines | None = None,
    thresholds: SelectiveHedgeThresholds | None = None,
    degraded_slowdown: float = 2.0,
    cancel_running_losers: bool = False,
) -> SelectiveHedgeResult:
    """Run one causal selective-Hedge episode without changing old engines."""

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    if type(cancel_running_losers) is not bool:
        raise ValueError("cancel_running_losers must be bool")
    deadlines = SLODeadlines() if deadlines is None else deadlines
    thresholds = SelectiveHedgeThresholds() if thresholds is None else thresholds
    if not isinstance(deadlines, SLODeadlines):
        raise ValueError("deadlines must be SLODeadlines")
    if not isinstance(thresholds, SelectiveHedgeThresholds):
        raise ValueError("thresholds must be SelectiveHedgeThresholds")
    rate = _finite_nonnegative(planned_budget_rate, "planned_budget_rate")
    slowdown = validate_slowdown(degraded_slowdown)
    _validate_episode(episode, slowdown)
    engine = _SelectiveLAEdgeEngine(
        episode,
        slowdown,
        planned_budget_rate=rate,
        cancel_running_losers=cancel_running_losers,
        thresholds=thresholds,
        deadlines=deadlines,
    )
    engine.run()
    simulation = _result_from_engine(episode, engine, slowdown)
    return SelectiveHedgeResult(
        simulation=simulation,
        windows=engine.finalize_windows(simulation.simulation.drain_end_time),
        thresholds=thresholds,
        deadlines=deadlines,
        suppression_reasons=tuple(engine.suppression_reasons),
        planned_budget_rate=rate,
        arm_key=selective_arm_key(thresholds),
    )


__all__ = [
    "SelectiveHedgeResult",
    "SelectiveHedgeThresholds",
    "SelectiveSuppression",
    "SELECTIVE_HEDGE_THRESHOLD_GRID",
    "normalized_slack",
    "selective_arm_key",
    "selective_candidate_key",
    "simulate_selective_laedge_episode",
]
