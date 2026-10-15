"""LPLB-inspired finite-iteration approximate load balancing for Ticket13."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .granite_moe_replica_plan_v1 import ReplicaPlacement
from .granite_moe_ticket11_adapters_v1 import largest_remainder

EXPERT_COUNT = 32
ACTIVE_DEVICE_COUNTS = (4, 6, 8)
ROUTES_PER_MS_PER_DEVICE = 10.0
APPROX_ITERATIONS = 8192
APPROX_STEP_NUMERATOR = 0.25
ACTIVE_DEVICE_TIE_TOLERANCE_NORMALIZED_LOAD = 1e-15
ROW_SUM_TOLERANCE = 1e-7

class Ticket13PlannerError(ValueError):
    """Invalid input for the Ticket13 approximate fixed-placement adapter."""

@dataclass(frozen=True)
class FixedPlacementPlan:
    strategy: str
    shares_by_expert: Mapping[int, tuple[float, ...]]
    quotas_by_expert: Mapping[int, tuple[int, ...]]
    diagnostics: Mapping[str, object]

def _validate_fixed_inputs(
    counts_in: Mapping[int, int],
    placements_in: Mapping[int, Sequence[ReplicaPlacement]],
    rates_in: Mapping[str, float],
):
    if not isinstance(counts_in, Mapping) or set(counts_in) != set(range(EXPERT_COUNT)):
        raise Ticket13PlannerError("counts must contain exactly Experts 0..31")
    counts = {}
    for expert in range(EXPERT_COUNT):
        value = counts_in[expert]
        if type(value) is not int or value < 0:
            raise Ticket13PlannerError("route counts must be nonnegative native integers")
        counts[expert] = value
    if set(placements_in) != set(counts):
        raise Ticket13PlannerError("placement Expert keys must exactly match counts")
    placements = {}
    for expert in range(EXPERT_COUNT):
        row = tuple(placements_in[expert])
        if len(row) != 2:
            raise Ticket13PlannerError("Ticket13 requires exactly two existing replicas per Expert")
        devices = tuple(getattr(place, "device", None) for place in row)
        replica_ids = tuple(getattr(place, "replica_id", None) for place in row)
        if any(not isinstance(device, str) or not device for device in devices):
            raise Ticket13PlannerError("replica devices must be nonempty strings")
        if any(not isinstance(replica_id, str) or not replica_id for replica_id in replica_ids):
            raise Ticket13PlannerError("replica IDs must be nonempty strings")
        if len(set(devices)) != 2 or len(set(replica_ids)) != 2:
            raise Ticket13PlannerError("each Expert needs two distinct legal replica devices")
        placements[expert] = row
    devices = tuple(sorted({place.device for row in placements.values() for place in row}))
    if len(devices) not in ACTIVE_DEVICE_COUNTS:
        raise Ticket13PlannerError("only the frozen four/six/eight-device layouts are supported")
    if devices != tuple(f"cuda:{index}" for index in range(len(devices))):
        raise Ticket13PlannerError("active devices must be the frozen contiguous cuda:0..G-1 set")
    if set(rates_in) != set(devices):
        raise Ticket13PlannerError("service-rate keys must match active devices")
    rates = {}
    for device in devices:
        raw = rates_in[device]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise Ticket13PlannerError("service rates must be finite positive numbers")
        value = float(raw)
        if not math.isfinite(value) or value != ROUTES_PER_MS_PER_DEVICE:
            raise Ticket13PlannerError("Ticket13 freezes every device rate at 10 routes/ms")
        rates[device] = value
    total = sum(counts.values())
    if total <= 0:
        raise Ticket13PlannerError("at least one route is required")
    return dict(sorted(counts.items())), placements, devices, rates


def _integerize(counts, shares):
    return {
        expert: largest_remainder(counts[expert], shares[expert])
        for expert in sorted(counts)
    }


def _continuous_metrics(counts, placements, devices, rates, shares):
    loads = {device: 0.0 for device in devices}
    max_row_error = 0.0
    for expert in sorted(counts):
        row_shares = tuple(float(value) for value in shares[expert])
        if len(row_shares) != len(placements[expert]):
            raise Ticket13PlannerError("share width differs from existing replica count")
        if any(not math.isfinite(value) or value < -1e-12 for value in row_shares):
            raise Ticket13PlannerError("continuous shares must be finite and nonnegative")
        row_sum = math.fsum(row_shares)
        if counts[expert] > 0 and abs(row_sum - 1.0) > ROW_SUM_TOLERANCE:
            raise Ticket13PlannerError("positive-demand continuous share row must sum to one")
        if counts[expert] == 0 and abs(row_sum - 1.0) > 1e-8:
            raise Ticket13PlannerError("zero-demand share row must use the uniform convention")
        max_row_error = max(max_row_error, abs(row_sum - 1.0))
        for share, place in zip(row_shares, placements[expert]):
            if place.device not in loads:
                raise Ticket13PlannerError("share uses a device outside frozen placement")
            loads[place.device] += counts[expert] * max(0.0, share)
    achieved_peak = max(loads.values())
    return {
        "device_load_routes": loads,
        "Fhard_continuous_ms": achieved_peak / ROUTES_PER_MS_PER_DEVICE,
        "max_device_load_routes": achieved_peak,
        "max_share_row_sum_error": max_row_error,
    }


def eplb_even_dispatch(
    counts_in: Mapping[int, int],
    placements_in: Mapping[int, Sequence[ReplicaPlacement]],
    rates_in: Mapping[str, float],
) -> FixedPlacementPlan:
    """Equal continuous share on frozen replicas followed by Ticket11 LR."""
    counts, placements, devices, rates = _validate_fixed_inputs(
        counts_in, placements_in, rates_in
    )
    shares = {expert: (0.5, 0.5) for expert in counts}
    quotas = _integerize(counts, shares)
    return FixedPlacementPlan(
        "eplb_even_dispatch",
        shares,
        quotas,
        {
            "method": "equal continuous share across frozen legal replicas plus Ticket11 largest remainder",
            "policy_scope": "EPLB-even dispatch adaptation; placement remains frozen and shared",
            "integerizer": "Ticket11 stable_largest_remainder_lower_slot_tie",
        },
    )



def lplb_inspired_approximate(
    counts_in: Mapping[int, int],
    placements_in: Mapping[int, Sequence[ReplicaPlacement]],
    rates_in: Mapping[str, float],
) -> FixedPlacementPlan:
    """Run a frozen finite projected-subgradient load-balancing heuristic.

    This is an explicitly approximate LPLB-inspired adaptation. It contains no
    LP solver, max-flow, Hall bound, adaptive stop, or optimality certificate.
    """
    counts, placements, devices, rates = _validate_fixed_inputs(
        counts_in, placements_in, rates_in
    )
    total = sum(counts.values())
    device_position = {device: index for index, device in enumerate(devices)}
    normalized_counts = tuple(counts[e] / total for e in range(EXPERT_COUNT))
    replica_positions = {
        expert: (device_position[placements[expert][0].device], device_position[placements[expert][1].device])
        for expert in range(EXPERT_COUNT)
    }
    x = [0.5] * EXPERT_COUNT
    accumulated = [0.0] * EXPERT_COUNT
    for iteration in range(1, APPROX_ITERATIONS + 1):
        loads = [0.0] * len(devices)
        for expert in range(EXPERT_COUNT):
            weight = normalized_counts[expert]
            first_device, second_device = replica_positions[expert]
            loads[first_device] += weight * x[expert]
            loads[second_device] += weight * (1.0 - x[expert])
        peak = max(loads)
        active = tuple(
            index for index, value in enumerate(loads)
            if peak - value <= ACTIVE_DEVICE_TIE_TOLERANCE_NORMALIZED_LOAD
        )
        active_set = set(active)
        active_weight = 1.0 / len(active)
        step = APPROX_STEP_NUMERATOR / math.sqrt(iteration)
        for expert in range(EXPERT_COUNT):
            first_device, second_device = replica_positions[expert]
            gradient = normalized_counts[expert] * (
                (active_weight if first_device in active_set else 0.0)
                - (active_weight if second_device in active_set else 0.0)
            )
            x[expert] = min(1.0, max(0.0, x[expert] - step * gradient))
            accumulated[expert] += x[expert]
    shares = {}
    for expert in range(EXPERT_COUNT):
        slot0 = accumulated[expert] / APPROX_ITERATIONS
        shares[expert] = (slot0, 1.0 - slot0)
    quotas = _integerize(counts, shares)
    continuous = _continuous_metrics(counts, placements, devices, rates, shares)
    achieved_peak = float(continuous["max_device_load_routes"])
    average_lower_bound = total / len(devices)
    return FixedPlacementPlan(
        "lplb_inspired_approx_lr",
        shares,
        quotas,
        {
            "method": "deterministic fixed-iteration projected subgradient waterfill; LPLB-inspired approximate load-balancing",
            "policy_scope": "approximate fixed-placement dispatch heuristic; not an LP solution or official DeepSeek LPLB",
            "objective": "reduce the maximum normalized device route load by projected subgradient updates",
            "iterations": APPROX_ITERATIONS,
            "initial_share_replica_slot_0": 0.5,
            "step_schedule": "0.25/sqrt(k), k=1..8192",
            "active_device_tie_tolerance_normalized_load": ACTIVE_DEVICE_TIE_TOLERANCE_NORMALIZED_LOAD,
            "projection": "clip slot-0 share to [0,1]; slot-1 share is the complement",
            "returned_iterate": "arithmetic mean of all post-update iterates",
            "adaptive_stopping": False,
            "exact_lp_or_flow_solver_called": False,
            "optimality_certificate": None,
            "routes_per_ms_per_device": ROUTES_PER_MS_PER_DEVICE,
            "average_load_lower_bound_routes_per_device": average_lower_bound,
            "achieved_minus_average_lower_bound_routes_per_device": achieved_peak - average_lower_bound,
            "achieved_max_load_routes": achieved_peak,
            **continuous,
            "integerizer": "Ticket11 stable_largest_remainder_lower_slot_tie",
            "integer_optimization_after_LR": False,
        },
    )

def continuous_fhard(
    counts_in: Mapping[int, int],
    placements_in: Mapping[int, Sequence[ReplicaPlacement]],
    rates_in: Mapping[str, float],
    shares_by_expert: Mapping[int, Sequence[float]],
):
    counts, placements, devices, rates = _validate_fixed_inputs(
        counts_in, placements_in, rates_in
    )
    if set(shares_by_expert) != set(counts):
        raise Ticket13PlannerError("continuous share Expert keys must match frozen counts")
    return _continuous_metrics(counts, placements, devices, rates, shares_by_expert)


__all__ = [
    "ACTIVE_DEVICE_TIE_TOLERANCE_NORMALIZED_LOAD",
    "APPROX_ITERATIONS",
    "APPROX_STEP_NUMERATOR",
    "ACTIVE_DEVICE_COUNTS",
    "EXPERT_COUNT",
    "FixedPlacementPlan",
    "ROUTES_PER_MS_PER_DEVICE",
    "ROW_SUM_TOLERANCE",
    "Ticket13PlannerError",
    "continuous_fhard",
    "eplb_even_dispatch",
    "lplb_inspired_approximate",
]
