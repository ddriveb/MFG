"""Common State timeline and work-requirement semantics (pure functions).

Spec: `.scratch/common-state-no-hedge/spec.md` sections 1-2. Everything here is
deterministic, draws no random numbers, and holds no global state. The event
engine (ticket 02) builds on these primitives; nothing here schedules events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .domain import CommonState


class Phase(str, Enum):
    """Metrics phase: separates the initial Healthy window from Recovered."""

    HEALTHY = "H"
    DEGRADED = "D"
    FAILED = "F"
    RECOVERED = "R"


def _require_real_non_negative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _require_replica_id(replica_id: int) -> int:
    if (
        isinstance(replica_id, bool)
        or not isinstance(replica_id, int)
        or replica_id not in (0, 1)
    ):
        raise ValueError(
            f"replica_id must be the int 0 (domain A) or int 1 (domain B), "
            f"got {replica_id!r}"
        )
    return replica_id


def validate_slowdown(degraded_slowdown: float) -> float:
    if isinstance(degraded_slowdown, bool) or not isinstance(
        degraded_slowdown, (int, float)
    ):
        raise ValueError(
            f"degraded_slowdown must be a real number, got {degraded_slowdown!r}"
        )
    slowdown = float(degraded_slowdown)
    if not math.isfinite(slowdown) or slowdown < 1.0:
        raise ValueError(
            f"degraded_slowdown must be finite and >= 1, got {degraded_slowdown!r}"
        )
    return slowdown


@dataclass(frozen=True)
class CommonStateTimeline:
    """Ordered state boundaries with half-open interval semantics.

    `[0, degraded_start)` is H, `[degraded_start, failed_start)` is D,
    `[failed_start, recovered_start)` is F, and `[recovered_start, +inf)` is
    engineering-Healthy again (metrics Phase R).
    """

    degraded_start: float
    failed_start: float
    recovered_start: float

    def __post_init__(self) -> None:
        for name in ("degraded_start", "failed_start", "recovered_start"):
            object.__setattr__(
                self, name, _require_real_non_negative(getattr(self, name), name)
            )
        if not self.degraded_start < self.failed_start < self.recovered_start:
            raise ValueError(
                "require degraded_start < failed_start < recovered_start, got "
                f"{self.degraded_start} !< {self.failed_start} !< "
                f"{self.recovered_start}"
            )


def state_at(timeline: CommonStateTimeline, t: float) -> CommonState:
    """Engineering CommonState at time ``t``; Recovered reads as Healthy."""
    moment = _require_real_non_negative(t, "t")
    if moment < timeline.degraded_start:
        return CommonState.HEALTHY
    if moment < timeline.failed_start:
        return CommonState.DEGRADED
    if moment < timeline.recovered_start:
        return CommonState.FAILED
    return CommonState.HEALTHY


def phase_at(timeline: CommonStateTimeline, t: float) -> Phase:
    """Metrics phase at time ``t``; the post-recovery window is R, not H."""
    moment = _require_real_non_negative(t, "t")
    if moment < timeline.degraded_start:
        return Phase.HEALTHY
    if moment < timeline.failed_start:
        return Phase.DEGRADED
    if moment < timeline.recovered_start:
        return Phase.FAILED
    return Phase.RECOVERED


def speed_at(
    replica_id: int, state: CommonState, degraded_slowdown: float
) -> float:
    """Work units processed per normalized time unit.

    Replica 0 (domain A): 1.0 when Healthy, ``1 / degraded_slowdown`` when
    Degraded, 0.0 when Failed. Replica 1 (domain B) is always 1.0. A speed of
    0.0 only denotes capacity; an execution crossing into F is terminated by
    `completion_after`, not suspended until recovery.
    """
    replica = _require_replica_id(replica_id)
    if not isinstance(state, CommonState):
        raise ValueError(f"state must be a CommonState, got {state!r}")
    slowdown = validate_slowdown(degraded_slowdown)
    if replica == 1:
        return 1.0
    if state is CommonState.DEGRADED:
        return 1.0 / slowdown
    if state is CommonState.FAILED:
        return 0.0
    return 1.0


def work_executed_between(
    replica_id: int,
    t0: float,
    t1: float,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
) -> float:
    """Integrate speed over the half-open interval ``[t0, t1)`` piecewise."""
    replica = _require_replica_id(replica_id)
    start = _require_real_non_negative(t0, "t0")
    end = _require_real_non_negative(t1, "t1")
    if start > end:
        raise ValueError(f"t0 must not exceed t1, got [{t0!r}, {t1!r})")
    slowdown = validate_slowdown(degraded_slowdown)

    total = 0.0
    position = start
    boundaries = (
        timeline.degraded_start,
        timeline.failed_start,
        timeline.recovered_start,
    )
    for boundary in boundaries:
        if position >= end:
            break
        if boundary <= position:
            continue
        segment_end = min(boundary, end)
        total += (segment_end - position) * speed_at(
            replica, state_at(timeline, position), slowdown
        )
        position = segment_end
    if position < end:
        total += (end - position) * speed_at(
            replica, state_at(timeline, position), slowdown
        )
    return total


@dataclass(frozen=True)
class ExecutionOutcome:
    """Explicit terminal result of one execution attempt under the timeline."""

    completed: bool
    terminal_time: float
    executed_work: float
    remaining_work: float


def completion_after(
    replica_id: int,
    start: float,
    work: float,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
) -> ExecutionOutcome:
    """Compute the terminal outcome of executing ``work`` from ``start``.

    On success: ``completed=True``, ``terminal_time`` is the completion
    instant, ``remaining_work`` is 0. On Replica 0, if F begins first:
    ``completed=False``, ``terminal_time`` is the instant F was entered
    (``failed_start``, or ``start`` itself when starting inside F), and
    ``executed_work``/``remaining_work`` split ``work`` at that point. A failed
    execution never resumes after recovery. A completion landing exactly on
    ``failed_start`` is reported as completed; the event engine's fault-first
    ordering (spec section 3) decides that case.
    """
    replica = _require_replica_id(replica_id)
    position = _require_real_non_negative(start, "start")
    if isinstance(work, bool) or not isinstance(work, (int, float)):
        raise ValueError(f"work must be a real number, got {work!r}")
    required = float(work)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError(f"work must be finite and > 0, got {work!r}")
    slowdown = validate_slowdown(degraded_slowdown)

    executed = 0.0
    remaining = required
    boundaries = (
        timeline.degraded_start,
        timeline.failed_start,
        timeline.recovered_start,
        math.inf,
    )
    for boundary in boundaries:
        if boundary <= position:
            continue
        speed = speed_at(replica, state_at(timeline, position), slowdown)
        if speed == 0.0:
            return ExecutionOutcome(
                completed=False,
                terminal_time=position,
                executed_work=executed,
                remaining_work=remaining,
            )
        available = (boundary - position) * speed
        if remaining <= available:
            return ExecutionOutcome(
                completed=True,
                terminal_time=position + remaining / speed,
                executed_work=executed + remaining,
                remaining_work=0.0,
            )
        executed += available
        remaining -= available
        position = boundary
    raise AssertionError("unreachable: the final boundary is infinite")
