"""Isolated 2x2 running-loser cancellation comparison hooks.

This module only selects existing engines and their explicit cancellation
semantics.  It does not run a campaign or define a new metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .attribution_episode import EpisodeSimulationResult, EpisodeTrace
from .domain import ProtectionAction
from .laedge_episode import simulate_laedge_episode
from .token_online import (
    OnlineEpisodeResult,
    ReservationParameters,
    simulate_episode_online,
    StaticActionSource,
)


class PolicyFamily(str, Enum):
    NIIN = "niin"
    LAEDGE = "laedge"


@dataclass(frozen=True)
class CancellationAblationArm:
    policy_family: PolicyFamily
    cancel_running_losers: bool

    def __post_init__(self) -> None:
        if not isinstance(self.policy_family, PolicyFamily):
            raise ValueError("policy_family must be PolicyFamily")
        if type(self.cancel_running_losers) is not bool:
            raise ValueError("cancel_running_losers must be bool")

    @property
    def key(self) -> str:
        mode = "preemptive" if self.cancel_running_losers else "conservative"
        return f"{self.policy_family.value}:{mode}"


CANCELLATION_ARMS = (
    CancellationAblationArm(PolicyFamily.NIIN, False),
    CancellationAblationArm(PolicyFamily.NIIN, True),
    CancellationAblationArm(PolicyFamily.LAEDGE, False),
    CancellationAblationArm(PolicyFamily.LAEDGE, True),
)


def simulate_cancellation_arm(
    episode: EpisodeTrace,
    arm: CancellationAblationArm,
    *,
    action_source: object | Mapping[int, ProtectionAction] | None = None,
    reservation_parameters: ReservationParameters | None = None,
    degraded_slowdown: float = 2.0,
    hedge_delay: float | None = None,
    public_price: float = 0.0,
) -> OnlineEpisodeResult | EpisodeSimulationResult:
    """Run one explicit ablation arm on one immutable episode.

    ``action_source`` is required only for the one-shot NIIN family.  A
    mapping is converted to ``StaticActionSource`` at the call boundary so
    action decisions still pass through the online engine.
    """

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    if not isinstance(arm, CancellationAblationArm):
        raise ValueError("arm must be CancellationAblationArm")
    if arm.policy_family is PolicyFamily.NIIN:
        if isinstance(action_source, Mapping):
            action_source = StaticActionSource(action_source)
        if action_source is None:
            raise ValueError("NIIN arm requires action_source")
        return simulate_episode_online(
            episode,
            action_source,
            reservation_parameters,
            degraded_slowdown=degraded_slowdown,
            hedge_delay=hedge_delay,
            public_price=public_price,
            cancel_running_losers=arm.cancel_running_losers,
        )
    if action_source is not None:
        raise ValueError("LÆDGE arm does not accept action_source")
    return simulate_laedge_episode(
        episode,
        degraded_slowdown=degraded_slowdown,
        cancel_running_losers=arm.cancel_running_losers,
    )


__all__ = [
    "CANCELLATION_ARMS",
    "CancellationAblationArm",
    "PolicyFamily",
    "simulate_cancellation_arm",
]
