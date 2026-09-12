"""Fail-closed adapters for the two replicated-Expert comparison panels.

This module binds named algorithm kernels to execution capabilities and one
common headline metric projection.  It owns no simulator and performs no
artifact I/O.  Missing batch, GPU-capacity, or idle-release physics is an
error, never an invitation to silently substitute a different algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .replica_routing_baselines import (
    BASELINE_CATALOG,
    FailoverOnly,
    JoinShortestQueueRouter,
    LAEdgeScheduler,
    LeastUnfinishedWorkRouter,
    P95DelayedHedge,
    ReplicaLoad,
    RoundRobinRouter,
    lplb_token_count_minmax,
)


class BaselineAdapterError(RuntimeError):
    pass


class ComparisonPanel(str, Enum):
    PROTECTION = "protection_same_topology"
    ROUTING = "routing_after_placement"


@dataclass(frozen=True)
class BackendCapabilities:
    online_arrivals: bool = False
    per_replica_fcfs: bool = False
    failure_replay: bool = False
    delayed_timers: bool = False
    idle_release: bool = False
    primary_router: bool = False
    batch_release: bool = False
    gpu_capacity: bool = False
    reservation_projection: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"backend capability {name} must be bool")

    def missing(self, required: Sequence[str]) -> tuple[str, ...]:
        unknown = tuple(name for name in required if name not in self.__dataclass_fields__)
        if unknown:
            raise BaselineAdapterError(f"unknown backend capabilities: {unknown}")
        return tuple(name for name in required if not getattr(self, name))


@dataclass(frozen=True)
class BaselineArmAdapter:
    baseline_id: str
    panel: ComparisonPanel
    constructor_kind: str
    source_reference: str
    required_capabilities: tuple[str, ...]

    def choose_primary(self, token_id: int, replicas: Sequence[ReplicaLoad]) -> int:
        routers = {
            "uniform_round_robin": RoundRobinRouter,
            "join_shortest_queue": JoinShortestQueueRouter,
            "least_unfinished_work": LeastUnfinishedWorkRouter,
        }
        constructor = routers.get(self.baseline_id)
        if constructor is None:
            raise BaselineAdapterError(
                f"{self.baseline_id} is not a per-arrival primary router"
            )
        return constructor().choose(token_id, replicas)

    def assign_batch(
        self,
        token_experts: Sequence[int],
        eligible_replicas: Mapping[int, Sequence[int]],
        initial_loads: Mapping[int, int] | None = None,
    ):
        if self.baseline_id != "lplb_token_count_minmax":
            raise BaselineAdapterError(f"{self.baseline_id} is not a batch router")
        return lplb_token_count_minmax(token_experts, eligible_replicas, initial_loads)

    def build_kernel(self, **parameters):
        if self.baseline_id == "failover_only":
            if parameters:
                raise BaselineAdapterError("failover_only accepts no kernel parameters")
            return FailoverOnly()
        if self.baseline_id == "p95_delayed_hedge":
            if set(parameters) != {"latency_samples"}:
                raise BaselineAdapterError("P95 requires only latency_samples")
            return P95DelayedHedge.from_latency_samples(parameters["latency_samples"])
        if self.baseline_id == "laedge_work_conserving":
            if set(parameters) != {"replica_ids"}:
                raise BaselineAdapterError("LÆDGE requires only replica_ids")
            return LAEdgeScheduler(parameters["replica_ids"])
        raise BaselineAdapterError(
            f"{self.baseline_id} is constructed by its existing policy runner"
        )


@dataclass(frozen=True)
class ComparisonPanelAdapter:
    panel_id: str
    semantic_class: str
    topology_contract: str
    trace_contract: str
    metrics_contract: str
    arms: Mapping[str, BaselineArmAdapter]
    placement_id: str | None = None
    panel_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.panel_id not in {panel.value for panel in ComparisonPanel}:
            raise ValueError("unknown comparison panel")
        if self.semantic_class not in {"hedging", "routing"}:
            raise ValueError("invalid panel semantic class")
        if not self.arms:
            raise ValueError("comparison panel must contain arms")
        expected_panel = ComparisonPanel(self.panel_id)
        for key, arm in self.arms.items():
            if key != arm.baseline_id or arm.panel is not expected_panel:
                raise ValueError("arm key/panel mismatch")
            if key == "eplb_style":
                raise ValueError("EPLB placement cannot be a measured arm")
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))

    def validate_backend(self, capabilities: BackendCapabilities) -> None:
        if not isinstance(capabilities, BackendCapabilities):
            raise BaselineAdapterError("capabilities must be BackendCapabilities")
        required = list(self.panel_capabilities)
        for arm in self.arms.values():
            required.extend(arm.required_capabilities)
        missing = capabilities.missing(tuple(dict.fromkeys(required)))
        if missing:
            raise BaselineAdapterError(
                f"backend for {self.panel_id} lacks capabilities: {', '.join(missing)}"
            )


def _arm(
    baseline_id: str,
    panel: ComparisonPanel,
    constructor_kind: str,
    required: Sequence[str],
    source_reference: str | None = None,
) -> BaselineArmAdapter:
    if baseline_id in BASELINE_CATALOG:
        source = BASELINE_CATALOG[baseline_id].source_url
    elif source_reference is not None:
        source = source_reference
    else:
        raise AssertionError(f"missing source for {baseline_id}")
    return BaselineArmAdapter(
        baseline_id=baseline_id,
        panel=panel,
        constructor_kind=constructor_kind,
        source_reference=source,
        required_capabilities=tuple(required),
    )


def build_default_comparison_plan() -> Mapping[str, ComparisonPanelAdapter]:
    """Build the frozen two-panel plan without opening any artifact."""

    protection_arms = [
        _arm("failover_only", ComparisonPanel.PROTECTION, "kernel", (
            "online_arrivals", "per_replica_fcfs", "failure_replay",
        )),
        _arm("p95_delayed_hedge", ComparisonPanel.PROTECTION, "calibrated_kernel", (
            "online_arrivals", "per_replica_fcfs", "failure_replay", "delayed_timers",
        )),
        _arm("laedge_work_conserving", ComparisonPanel.PROTECTION, "event_kernel", (
            "online_arrivals", "per_replica_fcfs", "failure_replay", "idle_release",
        )),
        _arm(
            "pi_NIIN_v1", ComparisonPanel.PROTECTION, "existing_policy",
            ("online_arrivals", "per_replica_fcfs", "failure_replay",
             "delayed_timers", "reservation_projection"),
            "src/mfg_hedge/token_three_arm_evaluation.py#FrozenNIINPolicy",
        ),
        _arm(
            "pi_requested_reservation_price_v1", ComparisonPanel.PROTECTION,
            "sealed_existing_policy",
            ("online_arrivals", "per_replica_fcfs", "failure_replay",
             "delayed_timers", "reservation_projection"),
            str(Path("artifacts/token-refined-fixed-grid-20260908-r1/summary.json")),
        ),
    ]
    routing_arms = [
        _arm("uniform_round_robin", ComparisonPanel.ROUTING, "online_router", (
            "online_arrivals", "per_replica_fcfs", "failure_replay", "primary_router",
        )),
        _arm("join_shortest_queue", ComparisonPanel.ROUTING, "online_router", (
            "online_arrivals", "per_replica_fcfs", "failure_replay", "primary_router",
        )),
        _arm("least_unfinished_work", ComparisonPanel.ROUTING, "online_router", (
            "online_arrivals", "per_replica_fcfs", "failure_replay", "primary_router",
        )),
        _arm("lplb_token_count_minmax", ComparisonPanel.ROUTING, "batch_router", (
            "per_replica_fcfs", "failure_replay", "primary_router", "batch_release",
        )),
    ]
    panels = (
        ComparisonPanelAdapter(
            panel_id=ComparisonPanel.PROTECTION.value,
            semantic_class="hedging",
            topology_contract="theta_token_load0p7_v1_one_expert_two_replicas",
            trace_contract="shared_episode_trace_and_fault_path",
            metrics_contract="replica_baseline_headline_metrics_v1",
            arms={arm.baseline_id: arm for arm in protection_arms},
        ),
        ComparisonPanelAdapter(
            panel_id=ComparisonPanel.ROUTING.value,
            semantic_class="routing",
            topology_contract="batched_multi_expert_gpu_replica_topology_v1",
            trace_contract="shared_gate_batch_service_and_fault_trace",
            metrics_contract="replica_baseline_headline_metrics_v1",
            arms={arm.baseline_id: arm for arm in routing_arms},
            placement_id="eplb_style",
            panel_capabilities=("gpu_capacity",),
        ),
    )
    return MappingProxyType({panel.panel_id: panel for panel in panels})


@dataclass(frozen=True)
class CommonBaselineMetrics:
    overall_mean: float
    overall_p95: float
    overall_p99: float
    df_cvar95: float
    df_miss_rate: float
    hr_p99: float
    replay_rate: float
    total_work: float
    wasted_work: float
    storm_peak: float


def _finite_metric(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineAdapterError(f"metric {path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineAdapterError(f"metric {path} must be finite")
    return result


def adapt_common_metrics(record: object) -> CommonBaselineMetrics:
    """Project an arm result onto the frozen headline comparison schema."""

    if is_dataclass(record) and not isinstance(record, type):
        payload = asdict(record)
    elif isinstance(record, Mapping):
        payload = dict(record)
    else:
        raise BaselineAdapterError("metric record must be a mapping or dataclass")
    try:
        latency = payload["latency"]
        overall = latency["overall"]
        df = latency["df"]
        hr = latency["hr"]
        values = {
            "overall_mean": overall["mean"],
            "overall_p95": overall["p95"],
            "overall_p99": overall["p99"],
            "df_cvar95": df["cvar95"],
            "df_miss_rate": df["miss_rate"],
            "hr_p99": hr["p99"],
            "replay_rate": payload["replay_rate"],
            "total_work": payload["total_work"],
            "wasted_work": payload["wasted_work"],
            "storm_peak": payload["storm_peak"],
        }
    except (KeyError, TypeError) as error:
        raise BaselineAdapterError(f"incomplete common metric record: {error}") from error
    return CommonBaselineMetrics(**{
        name: _finite_metric(value, name) for name, value in values.items()
    })


__all__ = [
    "BackendCapabilities",
    "BaselineAdapterError",
    "BaselineArmAdapter",
    "CommonBaselineMetrics",
    "ComparisonPanel",
    "ComparisonPanelAdapter",
    "adapt_common_metrics",
    "build_default_comparison_plan",
]
