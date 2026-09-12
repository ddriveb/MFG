"""Causal online two-phase deficit allocator (requested -> applied).

Spec: `.scratch/mfg-hedge-paired-comparison/spec.md` sections 9 and 14;
ADR-0006/ADR-0008. The allocator is online: each Token's assignment depends only on
the current Token, prior in-window Tokens, the solved policy, and the
remaining estimated Hedge budget. The deficit is updated exactly once for the
requested action and never re-debted after a budget suppression. Only the
applied action reaches the event engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .common_state import CommonState, CommonStateTimeline, Phase, phase_at, state_at
from .domain import ProtectionAction, TokenClass
from .mfg_solver import MFGSolution, SolverStatus, validate_capacity_acceptance
from .workload import WorkloadTrace


DEFAULT_CONTROL_WINDOW = 25.0

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_HEDGE_ACTIONS = (ProtectionAction.DELAYED_HEDGE, ProtectionAction.IMMEDIATE_HEDGE)
_CLASSES = (TokenClass.REGULAR, TokenClass.URGENT)


@dataclass(frozen=True)
class ActionAssignment:
    token_id: int
    token_class: TokenClass
    arrival_time: float
    phase: Phase
    state: CommonState
    window_start: float
    window_end: float
    requested_action: ProtectionAction
    applied_action: ProtectionAction
    expected_hedge_work: float
    quota_suppressed: bool
    suppression_reason: str | None


@dataclass(frozen=True)
class ClassWindowAudit:
    window_start: float
    window_end: float
    state: CommonState
    phase: Phase
    token_class: TokenClass
    requested_counts: Mapping[ProtectionAction, int]
    applied_counts: Mapping[ProtectionAction, int]
    quota_suppressed: int
    planned_expected_hedge_work: float
    class_budget_share: float
    remaining_budget: float


@dataclass(frozen=True)
class QuotaWindowAudit:
    window_start: float
    window_end: float
    state: CommonState
    phase: Phase
    window_budget: float
    classes: tuple[ClassWindowAudit, ...]


@dataclass(frozen=True)
class QuotaProjectionResult:
    assignments: tuple[ActionAssignment, ...]
    windows: tuple[QuotaWindowAudit, ...]
    global_requested_counts: Mapping[ProtectionAction, int]
    global_applied_counts: Mapping[ProtectionAction, int]
    global_quota_suppressed: int
    global_planned_expected_hedge_work: float

    def to_action_plan(self) -> dict[int, ProtectionAction]:
        """The only artifact passed to the event engine: applied actions."""
        return {
            assignment.token_id: assignment.applied_action
            for assignment in self.assignments
        }


def _require_official_solution(solution: MFGSolution) -> None:
    """The quota projector only accepts official ADR-0007/0008 solutions."""
    tolerance = solution.parameters.capacity_tolerance
    for state in (CommonState.HEALTHY, CommonState.DEGRADED):
        state_solution = solution.solutions[state]
        violations = validate_capacity_acceptance(
            state_solution.price,
            state_solution.capacity_gap,
            state_solution.policy_residual,
            state_solution.load_residual,
            capacity_tolerance=solution.parameters.capacity_tolerance,
            complementarity_tolerance=(
                solution.parameters.complementarity_tolerance
            ),
            residual_tolerance=solution.parameters.tolerance,
        )
        if (
            state_solution.status is not SolverStatus.CONVERGED
            or state_solution.grid_clamped
            or violations
            or state_solution.hedge_work_rate
            > state_solution.hedge_budget_rate + tolerance
        ):
            raise ValueError(
                "official quota requires converged, unclamped, "
                f"capacity-feasible H/D solutions; state {state.value} has "
                f"status {state_solution.status.value}, violations={violations!r}"
            )
    forced = solution.solutions[CommonState.FAILED]
    forced_probabilities_are_exact = all(
        forced.probabilities[cls][ProtectionAction.NORMAL] == 1.0
        and forced.probabilities[cls][ProtectionAction.DELAYED_HEDGE] == 0.0
        and forced.probabilities[cls][ProtectionAction.IMMEDIATE_HEDGE] == 0.0
        for cls in _CLASSES
    )
    if (
        forced.status is not SolverStatus.FORCED_NORMAL
        or forced.grid_clamped
        or not forced_probabilities_are_exact
        or forced.price != 0.0
        or forced.hedge_work_rate != 0.0
        or forced.hedge_budget_rate != 0.0
        or forced.policy_residual >= solution.parameters.tolerance
        or forced.load_residual >= solution.parameters.tolerance
    ):
        raise ValueError(
            "official quota requires a forced_normal F solution, got "
            f"{forced.status.value}"
        )


class _WindowState:
    """Per-(window, class) allocator state; recreated at every window start."""

    def __init__(self, window_budget: float, shares: Mapping[TokenClass, float]):
        self.deficit = {
            cls: {action: 0.0 for action in _ACTIONS} for cls in _CLASSES
        }
        self.remaining = dict(shares)
        self.window_budget = window_budget
        self.requested = {
            cls: {action: 0 for action in _ACTIONS} for cls in _CLASSES
        }
        self.applied = {cls: {action: 0 for action in _ACTIONS} for cls in _CLASSES}
        self.suppressed = {cls: 0 for cls in _CLASSES}
        self.planned = {cls: 0.0 for cls in _CLASSES}
        self.shares = dict(shares)


def _phase_bounds(timeline: CommonStateTimeline) -> list[tuple[Phase, float, float]]:
    return [
        (Phase.HEALTHY, 0.0, timeline.degraded_start),
        (Phase.DEGRADED, timeline.degraded_start, timeline.failed_start),
        (Phase.FAILED, timeline.failed_start, timeline.recovered_start),
        (Phase.RECOVERED, timeline.recovered_start, math.inf),
    ]


def _window_of(timeline: CommonStateTimeline, t: float, width: float):
    phase = phase_at(timeline, t)
    for name, start, end in _phase_bounds(timeline):
        if name is phase:
            index = math.floor((t - start) / width)
            window_start = start + index * width
            window_end = start + (index + 1) * width
            if math.isfinite(end):
                window_end = min(window_end, end)
            return name, window_start, window_end
    raise AssertionError("phase lookup must succeed")


def project_actions(
    trace: WorkloadTrace,
    solution: MFGSolution,
    timeline: CommonStateTimeline,
    *,
    window_width: float = DEFAULT_CONTROL_WINDOW,
) -> QuotaProjectionResult:
    """Assign applied actions for every Token, online and deterministically."""
    _require_official_solution(solution)
    if isinstance(window_width, bool) or not isinstance(window_width, (int, float)):
        raise ValueError(f"window_width must be a real number, got {window_width!r}")
    width = float(window_width)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"window_width must be finite and positive, got {width!r}")

    assignments: list[ActionAssignment] = []
    window_audits: list[QuotaWindowAudit] = []

    current_window: tuple[float, float] | None = None
    window_phase = None
    window_state = None
    window_budget = 0.0
    window_state_solution = None
    state_allocator: _WindowState | None = None

    def open_window(start: float, end: float, phase: Phase, state: CommonState) -> None:
        nonlocal window_budget, window_state_solution, state_allocator
        window_state_solution = solution.for_state(state)
        window_budget = window_state_solution.hedge_budget_rate * (end - start)
        demands = {}
        for cls in _CLASSES:
            demand = 0.0
            for action in _ACTIONS:
                probability = window_state_solution.probabilities[cls][action]
                cost = window_state_solution.final_stats[cls][action].expected_hedge_work
                demand += probability * cost
            demands[cls] = demand * solution.class_weights[cls]
        total_demand = math.fsum(demands.values())
        if total_demand == 0.0:
            shares = {cls: 0.0 for cls in _CLASSES}
        else:
            shares = {
                cls: window_budget * demands[cls] / total_demand for cls in _CLASSES
            }
        state_allocator = _WindowState(window_budget, shares)

    global_requested = {action: 0 for action in _ACTIONS}
    global_applied = {action: 0 for action in _ACTIONS}
    global_suppressed = 0
    global_planned = 0.0

    for spec in trace.tokens:
        phase, start, end = _window_of(timeline, spec.arrival_time, width)
        state = state_at(timeline, spec.arrival_time)
        if current_window != (start, end):
            if state_allocator is not None:
                window_audits.append(
                    _close_window(
                        current_window, window_phase, window_state,
                        window_budget, state_allocator,
                    )
                )
            current_window = (start, end)
            window_phase = phase
            window_state = state
            open_window(start, end, phase, state)

        cls = spec.token_class
        if state is CommonState.FAILED:
            # Feasible-set projection happens before the deficit step: F is
            # all-Normal and never counts quota suppression.
            projected = {ProtectionAction.NORMAL: 1.0}
        else:
            projected = window_state_solution.probabilities[cls]

        deficit = state_allocator.deficit[cls]
        for action in projected:
            deficit[action] += projected[action]
        requested = ProtectionAction.NORMAL
        for action in _ACTIONS:
            if deficit[action] > deficit[requested]:
                requested = action
        deficit[requested] -= 1.0

        applied = requested
        suppressed = False
        reason = None
        cost = 0.0
        if requested in _HEDGE_ACTIONS:
            cost = window_state_solution.final_stats[cls][requested].expected_hedge_work
            remaining = state_allocator.remaining[cls]
            if cost <= remaining:
                state_allocator.remaining[cls] = remaining - cost
                state_allocator.planned[cls] = math.fsum(
                    (state_allocator.planned[cls], cost)
                )
            else:
                applied = ProtectionAction.NORMAL
                suppressed = True
                reason = "class budget share exhausted"

        state_allocator.requested[cls][requested] += 1
        state_allocator.applied[cls][applied] += 1
        state_allocator.suppressed[cls] += int(suppressed)
        global_requested[requested] += 1
        global_applied[applied] += 1
        global_suppressed += int(suppressed)
        global_planned = math.fsum((global_planned, cost if not suppressed else 0.0))

        assignments.append(
            ActionAssignment(
                token_id=spec.token_id,
                token_class=cls,
                arrival_time=spec.arrival_time,
                phase=phase,
                state=state,
                window_start=start,
                window_end=end,
                requested_action=requested,
                applied_action=applied,
                expected_hedge_work=cost if not suppressed else 0.0,
                quota_suppressed=suppressed,
                suppression_reason=reason,
            )
        )

    if state_allocator is not None and current_window is not None:
        window_audits.append(
            _close_window(
                current_window, window_phase, window_state,
                window_budget, state_allocator,
            )
        )

    return QuotaProjectionResult(
        assignments=tuple(assignments),
        windows=tuple(window_audits),
        global_requested_counts=global_requested,
        global_applied_counts=global_applied,
        global_quota_suppressed=global_suppressed,
        global_planned_expected_hedge_work=global_planned,
    )


def _close_window(window, phase, state, window_budget, allocator: _WindowState) -> QuotaWindowAudit:
    classes = []
    for cls in _CLASSES:
        classes.append(
            ClassWindowAudit(
                window_start=window[0],
                window_end=window[1],
                state=state,
                phase=phase,
                token_class=cls,
                requested_counts=dict(allocator.requested[cls]),
                applied_counts=dict(allocator.applied[cls]),
                quota_suppressed=allocator.suppressed[cls],
                planned_expected_hedge_work=allocator.planned[cls],
                class_budget_share=allocator.shares[cls],
                remaining_budget=allocator.remaining[cls],
            )
        )
    return QuotaWindowAudit(
        window_start=window[0],
        window_end=window[1],
        state=state,
        phase=phase,
        window_budget=window_budget,
        classes=tuple(classes),
    )
