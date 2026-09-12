"""Narrow contracts that keep calibration, solving, and evaluation separate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .domain import ActionStats, CommonState, ProtectionAction, TokenClass


@dataclass(frozen=True)
class PolicyContext:
    expert_id: int
    common_state: CommonState
    utilization: float
    hedge_price: float


PolicyDistribution = Mapping[TokenClass, Mapping[ProtectionAction, float]]


class ActionStatsEstimator(Protocol):
    """Coarse model used for policy computation, never final evaluation."""

    def estimate(
        self,
        *,
        context: PolicyContext,
        token_class: TokenClass,
        action: ProtectionAction,
    ) -> ActionStats:
        ...


class ProtectionPolicy(Protocol):
    def distribution(self, context: PolicyContext) -> PolicyDistribution:
        ...


class QuotaProjector(Protocol):
    """Turns a mixed policy into capacity-safe per-window assignments."""

    def assign(
        self,
        *,
        token_ids_by_class: Mapping[TokenClass, Sequence[int]],
        distribution: PolicyDistribution,
        hedge_work_budget: float,
    ) -> Mapping[int, ProtectionAction]:
        ...

