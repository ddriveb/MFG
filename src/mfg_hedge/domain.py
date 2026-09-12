"""Stable domain vocabulary shared by solver and event simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class CommonState(str, Enum):
    HEALTHY = "H"
    DEGRADED = "D"
    FAILED = "F"


class TokenClass(str, Enum):
    REGULAR = "R"
    URGENT = "U"


class ProtectionAction(str, Enum):
    NORMAL = "N"
    DELAYED_HEDGE = "D"
    IMMEDIATE_HEDGE = "I"


@dataclass(frozen=True)
class ActionStats:
    """Aggregated action outcomes used by a policy solver.

    Hedge and replay work are intentionally separate for capacity accounting.
    Wasted work is executed work on non-winning attempts and can overlap those
    resource terms because it represents an additional operating penalty.
    """

    mean_latency: float
    replay_probability: float
    deadline_miss_probability: float
    expected_hedge_work: float
    expected_replay_work: float
    expected_wasted_work: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "mean_latency",
            "replay_probability",
            "deadline_miss_probability",
            "expected_hedge_work",
            "expected_replay_work",
            "expected_wasted_work",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("replay_probability", "deadline_miss_probability"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} cannot exceed 1")

    @property
    def expected_extra_work(self) -> float:
        return self.expected_hedge_work + self.expected_replay_work
