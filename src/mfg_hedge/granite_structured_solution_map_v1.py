"""Bounded analytical solution map for the frozen Granite Top8 cyclic ring.

This module is deliberately specialized to 32 Experts, two adjacent cyclic
replicas, 4, 6 or 8 homogeneous GPUs and zero communication. In the six-GPU
case the 32 Experts form uneven edge groups of 6, 6, 5, 5, 5 and 5; demands
are aggregated by edge without assuming equal group sizes. It uses a bounded
interval-density scan, not a generic assignment/flow/LP planner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .granite_moe_replica_plan_v1 import ReplicaPlacement
from .granite_moe_ticket11_adapters_v1 import largest_remainder


EXPERT_COUNT = 32
ALLOWED_DEVICE_COUNTS = (4, 6, 8)
SERVICE_RATE_ROUTES_PER_MS = 10.0
TAU_MS = 0.02
FW_TOLERANCE_MS = 0.001
_NUMERIC_TOL = 1e-10


class StructuredMapError(ValueError):
    """Malformed input or input outside the frozen Ticket18 fixture."""


@dataclass(frozen=True)
class RingLayout:
    device_count: int
    devices: tuple[str, ...]
    placements_by_expert: Mapping[int, tuple[ReplicaPlacement, ReplicaPlacement]]
    edge_experts: tuple[tuple[int, ...], ...]
    left_slot_by_expert: tuple[int, ...]
    right_slot_by_expert: tuple[int, ...]
    interval_masks: tuple[int, ...]


@dataclass(frozen=True)
class StructuredSolution:
    edge_demands: tuple[int, ...]
    loads_routes: tuple[float, ...]
    edge_left_routes: tuple[float, ...]
    edge_left_units2: tuple[int, ...]
    shares_by_expert: Mapping[int, tuple[float, float]]
    blocks: tuple[tuple[tuple[int, ...], int, int], ...]
    feasible_offset_routes: tuple[float, float]
    canonical_offset_routes: float
    kkt_violation_routes: float
    Fhard_ms: float
    F_tau_ms: float
    fw_gap_ms: float
    fw_gate_pass: bool
    load_scale: int
    load_units: tuple[int, ...]


def _precompute_interval_masks(device_count: int) -> tuple[int, ...]:
    masks: set[int] = set()
    for length in range(1, device_count):
        for start in range(device_count):
            mask = 0
            for offset in range(length):
                mask |= 1 << ((start + offset) % device_count)
            masks.add(mask)
    masks.add((1 << device_count) - 1)
    result = tuple(sorted(masks, key=lambda mask: (mask.bit_count(), mask)))
    if len(result) != device_count * (device_count - 1) + 1:
        raise AssertionError("cyclic interval descriptor count is not bounded as specified")
    return result


def prepare_ring_layout(
    placements_by_expert: Mapping[int, Sequence[ReplicaPlacement]],
    service_rate_by_device: Mapping[str, float],
    *,
    communication_signature: Mapping[str, object] | None = None,
) -> RingLayout:
    """Validate and prepare the frozen ring outside per-state online cost."""
    if not isinstance(service_rate_by_device, Mapping):
        raise StructuredMapError("service rates must be a device mapping")
    devices = tuple(sorted(service_rate_by_device))
    if len(devices) not in ALLOWED_DEVICE_COUNTS or devices != tuple(
        f"cuda:{index}" for index in range(len(devices))
    ):
        raise StructuredMapError("only contiguous cuda:0..G-1 with G in {4,6,8} are supported")
    if any(
        isinstance(service_rate_by_device[device], bool)
        or not isinstance(service_rate_by_device[device], (int, float))
        or not math.isfinite(float(service_rate_by_device[device]))
        or float(service_rate_by_device[device]) != SERVICE_RATE_ROUTES_PER_MS
        for device in devices
    ):
        raise StructuredMapError("only equal 10 routes/ms service rates are supported")
    if communication_signature is not None:
        if not isinstance(communication_signature, Mapping) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
            for value in communication_signature.values()
        ):
            raise StructuredMapError("communication must be absent or explicitly zero")
    if not isinstance(placements_by_expert, Mapping) or set(placements_by_expert) != set(
        range(EXPERT_COUNT)
    ):
        raise StructuredMapError("placement must cover Experts 0..31 exactly")

    device_count = len(devices)
    rows: dict[int, tuple[ReplicaPlacement, ReplicaPlacement]] = {}
    left_slots = [0] * EXPERT_COUNT
    right_slots = [0] * EXPERT_COUNT
    edge_experts: list[list[int]] = [[] for _ in range(device_count)]
    for expert in range(EXPERT_COUNT):
        row = tuple(placements_by_expert[expert])
        if len(row) != 2:
            raise StructuredMapError(f"Expert {expert} must have exactly two replicas")
        if any(not isinstance(place, ReplicaPlacement) for place in row):
            raise StructuredMapError("placement rows must contain ReplicaPlacement records")
        replica_ids = tuple(place.replica_id for place in row)
        gpu_ids = tuple(place.device for place in row)
        if any(not isinstance(value, str) or not value for value in replica_ids + gpu_ids):
            raise StructuredMapError(f"Expert {expert} has an empty placement field")
        if len(set(replica_ids)) != 2 or len(set(gpu_ids)) != 2:
            raise StructuredMapError(f"Expert {expert} replicas must have unique IDs and GPUs")
        expected_left = expert % device_count
        expected_right = (expert + 1) % device_count
        if set(gpu_ids) != {f"cuda:{expected_left}", f"cuda:{expected_right}"}:
            raise StructuredMapError(f"Expert {expert} is outside the frozen oriented ring")
        left_slot = gpu_ids.index(f"cuda:{expected_left}")
        right_slot = gpu_ids.index(f"cuda:{expected_right}")
        left_slots[expert] = left_slot
        right_slots[expert] = right_slot
        edge_experts[expected_left].append(expert)
        rows[expert] = (row[0], row[1])

    return RingLayout(
        device_count=device_count,
        devices=devices,
        placements_by_expert=rows,
        edge_experts=tuple(tuple(values) for values in edge_experts),
        left_slot_by_expert=tuple(left_slots),
        right_slot_by_expert=tuple(right_slots),
        interval_masks=_precompute_interval_masks(device_count),
    )


def _validate_counts(counts: Mapping[int, int]) -> tuple[int, ...]:
    if not isinstance(counts, Mapping) or set(counts) != set(range(EXPERT_COUNT)):
        raise StructuredMapError("counts must cover Experts 0..31 exactly")
    result = []
    for expert in range(EXPERT_COUNT):
        value = counts[expert]
        if type(value) is not int or value < 0:
            raise StructuredMapError("route counts must be nonnegative native integers")
        result.append(value)
    return tuple(result)


def _round_half_up_nonnegative(numerator: int, denominator: int) -> int:
    if type(numerator) is not int or numerator < 0 or type(denominator) is not int or denominator <= 0:
        raise StructuredMapError("half-up rounding needs a nonnegative rational number")
    return (2 * numerator + denominator) // (2 * denominator)


def _pressure(mask: int, active_mask: int, edge_demands: Sequence[int], device_count: int) -> int:
    """Integer unavoidable supply inside mask in the current residual ring."""
    pressure = 0
    for edge, amount in enumerate(edge_demands):
        left = edge
        right = (edge + 1) % device_count
        left_active = bool(active_mask & (1 << left))
        right_active = bool(active_mask & (1 << right))
        if left_active and right_active:
            if mask & (1 << left) and mask & (1 << right):
                pressure += amount
        elif left_active and mask & (1 << left):
            pressure += amount
        elif right_active and mask & (1 << right):
            pressure += amount
    return pressure


def _peel_load_units(
    edge_demands: Sequence[int], device_count: int, interval_masks: Sequence[int]
) -> tuple[tuple[int, ...], int, tuple[tuple[tuple[int, ...], int, int], ...]]:
    """Return exact rational block loads as common-denominator integer units.

    Candidate density comparisons use integer cross-products; exact ties are
    unioned. Since each block size is at most G, the common scale is at most
    lcm(1..G), which is 840 for G=8.
    """
    active_mask = (1 << device_count) - 1
    blocks_raw: list[tuple[int, int, int]] = []
    for _round in range(device_count):
        if not active_mask:
            break
        best_numerator = -1
        best_denominator = 1
        winners: list[int] = []
        for candidate in interval_masks:
            if candidate & active_mask != candidate:
                continue
            denominator = candidate.bit_count()
            numerator = _pressure(candidate, active_mask, edge_demands, device_count)
            lhs = numerator * best_denominator
            rhs = best_numerator * denominator
            if numerator >= 0 and (best_numerator < 0 or lhs > rhs):
                best_numerator, best_denominator = numerator, denominator
                winners = [candidate]
            elif lhs == rhs:
                winners.append(candidate)
        if not winners:
            raise StructuredMapError("density peeling found no residual cyclic interval")
        block_mask = 0
        for candidate in winners:
            block_mask |= candidate
        if block_mask & active_mask != block_mask:
            raise AssertionError("a maximizing block contains an inactive GPU")
        block_numerator = _pressure(block_mask, active_mask, edge_demands, device_count)
        block_denominator = block_mask.bit_count()
        if block_numerator * best_denominator != best_numerator * block_denominator:
            raise StructuredMapError("union of exactly tied maximum-density blocks changed density")
        vertices = tuple(index for index in range(device_count) if block_mask & (1 << index))
        blocks_raw.append((block_mask, block_numerator, block_denominator))
        active_mask &= ~block_mask
    if active_mask:
        raise StructuredMapError("density peeling exceeded the GPU-count round bound")

    scale = math.lcm(*(denominator for _mask, _num, denominator in blocks_raw))
    units = [0] * device_count
    public_blocks = []
    for mask, numerator, denominator in blocks_raw:
        load_unit = numerator * (scale // denominator)
        for gpu in range(device_count):
            if mask & (1 << gpu):
                units[gpu] = load_unit
        public_blocks.append(
            (
                tuple(gpu for gpu in range(device_count) if mask & (1 << gpu)),
                numerator,
                denominator,
            )
        )
    return tuple(units), scale, tuple(public_blocks)


def _reconstruct_left_mass(
    edge_demands: Sequence[int], load_units: Sequence[int], scale: int
) -> tuple[tuple[int, ...], int, int]:
    """Solve the ring prefix equations exactly in half-scale integer units."""
    device_count = len(edge_demands)
    b_units = [0] * device_count
    for gpu in range(1, device_count):
        b_units[gpu] = b_units[gpu - 1] + load_units[gpu] - edge_demands[gpu - 1] * scale
    if load_units[0] != edge_demands[-1] * scale - b_units[-1]:
        raise StructuredMapError("ring wraparound load equation failed")
    lower = max(-b_units[gpu] for gpu in range(device_count))
    upper = min(edge_demands[gpu] * scale - b_units[gpu] for gpu in range(device_count))
    if lower > upper:
        raise StructuredMapError("peeled GPU loads do not admit feasible ring edge shares")
    # Represent the canonical midpoint exactly with denominator 2*scale.
    offset2 = lower + upper
    left2 = tuple(offset2 + 2 * b_units[gpu] for gpu in range(device_count))
    capacity2 = tuple(2 * amount * scale for amount in edge_demands)
    if any(value < 0 or value > capacity for value, capacity in zip(left2, capacity2)):
        raise StructuredMapError("canonical offset produced an infeasible edge split")
    return left2, lower, upper


def _quality_metrics(
    counts: Sequence[int],
    layout: RingLayout,
    shares_by_expert: Mapping[int, Sequence[float]],
    tau_ms: float,
) -> tuple[tuple[float, ...], float, float, float, float]:
    loads = [0.0] * layout.device_count
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        row = tuple(shares_by_expert[expert])
        for slot, share in enumerate(row):
            place = layout.placements_by_expert[expert][slot]
            gpu = int(place.device.split(":")[1])
            loads[gpu] += count * float(share)
    times = [value / SERVICE_RATE_ROUTES_PER_MS for value in loads]
    maximum = max(times, default=0.0)
    exponents = [math.exp((value - maximum) / tau_ms) for value in times]
    f_tau = maximum + tau_ms * math.log(math.fsum(exponents))
    weights = [value / math.fsum(exponents) for value in exponents]
    fw_terms = []
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        rows = layout.placements_by_expert[expert]
        weighted_share_cost = 0.0
        min_cost = math.inf
        for slot, place in enumerate(rows):
            gpu = int(place.device.split(":")[1])
            share = float(shares_by_expert[expert][slot])
            cost = weights[gpu] / SERVICE_RATE_ROUTES_PER_MS
            weighted_share_cost += share * cost
            min_cost = min(min_cost, cost)
        fw_terms.append(count * (weighted_share_cost - min_cost))
    fw_gap = max(0.0, math.fsum(fw_terms))
    return tuple(loads), maximum, f_tau, fw_gap, math.fsum(times) / layout.device_count


def solve_structured_solution_map(
    counts: Mapping[int, int],
    layout: RingLayout,
    *,
    tau_ms: float = TAU_MS,
) -> StructuredSolution:
    """Return the unique optimal GPU loads and one canonical feasible share map."""
    if not isinstance(layout, RingLayout):
        raise StructuredMapError("a prepared Ticket18 RingLayout is required")
    if isinstance(tau_ms, bool) or not isinstance(tau_ms, (int, float)) or float(tau_ms) != TAU_MS:
        raise StructuredMapError("only the frozen tau=0.02 ms objective is supported")
    checked_counts = _validate_counts(counts)
    edge_demands = tuple(
        sum(checked_counts[expert] for expert in experts) for experts in layout.edge_experts
    )
    load_units, scale, blocks = _peel_load_units(
        edge_demands, layout.device_count, layout.interval_masks
    )
    left2, lower, upper = _reconstruct_left_mass(edge_demands, load_units, scale)
    left_routes = tuple(value / (2 * scale) for value in left2)
    loads_routes = tuple(value / scale for value in load_units)

    shares: dict[int, tuple[float, float]] = {}
    for expert, count in enumerate(checked_counts):
        edge = expert % layout.device_count
        if count == 0 or edge_demands[edge] == 0:
            left_share = 0.5
        else:
            left_share = left_routes[edge] / edge_demands[edge]
        left_share = min(1.0, max(0.0, left_share))
        row = [0.0, 0.0]
        row[layout.left_slot_by_expert[expert]] = left_share
        row[layout.right_slot_by_expert[expert]] = 1.0 - left_share
        shares[expert] = (row[0], row[1])

    loads_check, fhard, f_tau, fw_gap, _average = _quality_metrics(
        checked_counts, layout, shares, TAU_MS
    )
    if any(abs(a - b) > _NUMERIC_TOL * max(1, sum(checked_counts)) for a, b in zip(loads_check, loads_routes)):
        raise StructuredMapError("reconstructed shares do not reproduce the peeled GPU loads")
    kkt_violation = 0.0
    tolerance = _NUMERIC_TOL * max(1, sum(checked_counts))
    for edge, amount in enumerate(edge_demands):
        next_gpu = (edge + 1) % layout.device_count
        left_amount = left_routes[edge]
        right_amount = amount - left_amount
        kkt_violation = max(
            kkt_violation,
            max(0.0, loads_routes[edge] - loads_routes[next_gpu]) if left_amount > tolerance else 0.0,
            max(0.0, loads_routes[next_gpu] - loads_routes[edge]) if right_amount > tolerance else 0.0,
        )
    if kkt_violation > tolerance:
        raise StructuredMapError(f"density peel failed edge KKT conditions by {kkt_violation}")
    return StructuredSolution(
        edge_demands=edge_demands,
        loads_routes=loads_routes,
        edge_left_routes=left_routes,
        edge_left_units2=left2,
        shares_by_expert=shares,
        blocks=blocks,
        feasible_offset_routes=(lower / scale, upper / scale),
        canonical_offset_routes=(lower + upper) / (2 * scale),
        kkt_violation_routes=kkt_violation,
        Fhard_ms=fhard,
        F_tau_ms=f_tau,
        fw_gap_ms=fw_gap,
        fw_gate_pass=fw_gap <= FW_TOLERANCE_MS,
        load_scale=scale,
        load_units=load_units,
    )


def integerize_per_expert_largest_remainder(
    counts: Mapping[int, int], solution: StructuredSolution
) -> dict[int, tuple[int, int]]:
    """Apply the historical independent per-Expert largest-remainder rule."""
    checked = _validate_counts(counts)
    _validate_solution_counts(checked, solution)
    return {
        expert: largest_remainder(checked[expert], solution.shares_by_expert[expert])
        for expert in range(EXPERT_COUNT)
    }


def integerize_group_balanced(
    counts: Mapping[int, int], layout: RingLayout, solution: StructuredSolution
) -> dict[int, tuple[int, int]]:
    """Round ring group mass first, then apportion it within each Expert group.

    All group masses use the same half-up rule, which is translation invariant.
    Per-Expert assignments are the floor or ceiling of the original continuous
    left-edge route count, and each group sum equals its rounded continuous mass.
    """
    checked = _validate_counts(counts)
    _validate_solution_counts(checked, solution, layout)
    if len(solution.edge_demands) != layout.device_count:
        raise StructuredMapError("solution and prepared ring layout differ")
    left_by_expert = [0] * EXPERT_COUNT
    for edge, experts in enumerate(layout.edge_experts):
        amount = solution.edge_demands[edge]
        raw_numerators = {
            expert: checked[expert] * solution.edge_left_units2[edge]
            for expert in experts
        }
        raw_denominator = 2 * solution.load_scale * amount if amount else 1
        group_numerator = solution.edge_left_units2[edge]
        group_denominator = 2 * solution.load_scale
        group_target = _round_half_up_nonnegative(group_numerator, group_denominator)
        floors = {
            expert: raw_numerators[expert] // raw_denominator for expert in experts
        }
        remainder = group_target - sum(floors.values())
        ranked = sorted(
            experts,
            key=lambda expert: (
                -(raw_numerators[expert] % raw_denominator),
                expert,
                layout.left_slot_by_expert[expert],
            ),
        )
        if remainder < 0 or remainder > len(ranked):
            raise StructuredMapError("rounded group mass is outside per-Expert floor/ceil bounds")
        for expert in experts:
            left_by_expert[expert] = floors[expert]
        for expert in ranked[:remainder]:
            left_by_expert[expert] += 1
        if sum(left_by_expert[expert] for expert in experts) != group_target:
            raise AssertionError("group-balanced apportionment lost its fixed group total")

    return {
        expert: (
            left_by_expert[expert],
            checked[expert] - left_by_expert[expert],
        )
        if layout.left_slot_by_expert[expert] == 0
        else (
            checked[expert] - left_by_expert[expert],
            left_by_expert[expert],
        )
        for expert in range(EXPERT_COUNT)
    }


def integerize_equal_integral_loads(
    counts: Mapping[int, int], layout: RingLayout, solution: StructuredSolution
) -> dict[int, tuple[int, int]] | None:
    """Preserve equal integral loads using an integer feasible prefix offset.

    Returns ``None`` when the continuous target is not an equal integer load.
    """
    checked = _validate_counts(counts)
    _validate_solution_counts(checked, solution, layout)
    scale = solution.load_scale
    if not solution.load_units or any(value % scale for value in solution.load_units):
        return None
    if len(set(value // scale for value in solution.load_units)) != 1:
        return None
    edge_demands = solution.edge_demands
    b_units = [0] * layout.device_count
    for gpu in range(1, layout.device_count):
        b_units[gpu] = b_units[gpu - 1] + solution.load_units[gpu] - edge_demands[gpu - 1] * scale
    lower_units = max(-value for value in b_units)
    upper_units = min(edge_demands[gpu] * scale - b_units[gpu] for gpu in range(layout.device_count))
    if lower_units > upper_units:
        raise StructuredMapError("equal integral load has no integer feasible prefix")
    lower_integer = -((-lower_units) // scale)
    upper_integer = upper_units // scale
    if lower_integer > upper_integer:
        raise StructuredMapError("equal integral target lacks an integer prefix offset")
    # Midpoint half-up, clamped to the exact feasible integer interval.
    offset = min(upper_integer, max(lower_integer, (lower_integer + upper_integer + 1) // 2))
    left_mass = tuple(offset + b_units[gpu] // scale for gpu in range(layout.device_count))
    if any(value % scale for value in b_units):
        raise StructuredMapError("equal integral target has a nonintegral prefix offset")
    if any(not 0 <= left_mass[gpu] <= edge_demands[gpu] for gpu in range(layout.device_count)):
        raise StructuredMapError("integer prefix offset violated an edge capacity")
    # Use the same stable group apportioner with these integral edge masses by
    # constructing bounded per-Expert floors directly from the original shares.
    left_by_expert = [0] * EXPERT_COUNT
    for edge, experts in enumerate(layout.edge_experts):
        amount = edge_demands[edge]
        raw_numerators = {
            expert: checked[expert] * solution.edge_left_units2[edge]
            for expert in experts
        }
        raw_denominator = 2 * scale * amount if amount else 1
        floors = {
            expert: raw_numerators[expert] // raw_denominator for expert in experts
        }
        remainder = left_mass[edge] - sum(floors.values())
        ranked = sorted(
            experts,
            key=lambda expert: (
                -(raw_numerators[expert] % raw_denominator),
                expert,
                layout.left_slot_by_expert[expert],
            ),
        )
        if remainder < 0 or remainder > len(ranked):
            raise StructuredMapError("equal-load integer group violates per-Expert floor/ceil bounds")
        for expert in experts:
            left_by_expert[expert] = floors[expert]
        for expert in ranked[:remainder]:
            left_by_expert[expert] += 1
    quotas = {
        expert: (
            left_by_expert[expert],
            checked[expert] - left_by_expert[expert],
        )
        if layout.left_slot_by_expert[expert] == 0
        else (
            checked[expert] - left_by_expert[expert],
            left_by_expert[expert],
        )
        for expert in range(EXPERT_COUNT)
    }
    return quotas


def _validate_solution_counts(
    counts: Sequence[int], solution: StructuredSolution, layout: RingLayout | None = None
) -> None:
    if not isinstance(solution, StructuredSolution):
        raise StructuredMapError("integerization needs a StructuredSolution")
    expected_device_count = layout.device_count if layout is not None else len(solution.edge_demands)
    if expected_device_count != len(solution.edge_demands):
        raise StructuredMapError("solution and prepared ring layout differ")
    demands = tuple(
        sum(counts[expert] for expert in range(EXPERT_COUNT) if expert % expected_device_count == edge)
        for edge in range(expected_device_count)
    )
    if demands != solution.edge_demands:
        raise StructuredMapError("solution demand differs from the current Expert counts")


def materialize_top8_route_columns(
    token_experts: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate a Top8 trace and build the original Token/slot route columns."""
    expert_ids, token_indices, route_slots = materialize_top8_routes(token_experts)
    counts = [0] * EXPERT_COUNT
    for expert in expert_ids:
        counts[expert] += 1
    return expert_ids, token_indices, route_slots, tuple(counts)


def materialize_top8_routes(
    token_experts: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate a Top8 trace and materialize only its Expert/Token/slot columns."""
    if not isinstance(token_experts, Sequence) or isinstance(token_experts, (str, bytes)):
        raise StructuredMapError("Top8 token trace must be a sequence of route rows")
    expert_ids: list[int] = []
    token_indices: list[int] = []
    route_slots: list[int] = []
    for token_index, raw_row in enumerate(token_experts):
        row = tuple(raw_row)
        if len(row) != 8 or any(type(expert) is not int or not 0 <= expert < EXPERT_COUNT for expert in row):
            raise StructuredMapError(f"token {token_index} must contain eight native Expert IDs")
        if len(set(row)) != 8:
            raise StructuredMapError(f"token {token_index} repeats an Expert in Top8")
        for route_slot, expert in enumerate(row):
            expert_ids.append(expert)
            token_indices.append(token_index)
            route_slots.append(route_slot)
    return tuple(expert_ids), tuple(token_indices), tuple(route_slots)


def count_top8_routes(token_experts: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Validate a Top8 trace and aggregate its dynamic per-Expert demands."""
    if not isinstance(token_experts, Sequence) or isinstance(token_experts, (str, bytes)):
        raise StructuredMapError("Top8 token trace must be a sequence of route rows")
    counts = [0] * EXPERT_COUNT
    for token_index, raw_row in enumerate(token_experts):
        row = tuple(raw_row)
        if len(row) != 8 or any(
            type(expert) is not int or not 0 <= expert < EXPERT_COUNT for expert in row
        ):
            raise StructuredMapError(f"token {token_index} must contain eight native Expert IDs")
        if len(set(row)) != 8:
            raise StructuredMapError(f"token {token_index} repeats an Expert in Top8")
        for expert in row:
            counts[expert] += 1
    return tuple(counts)


__all__ = [
    "ALLOWED_DEVICE_COUNTS",
    "EXPERT_COUNT",
    "FW_TOLERANCE_MS",
    "RingLayout",
    "SERVICE_RATE_ROUTES_PER_MS",
    "StructuredMapError",
    "StructuredSolution",
    "TAU_MS",
    "count_top8_routes",
    "integerize_equal_integral_loads",
    "integerize_group_balanced",
    "integerize_per_expert_largest_remainder",
    "materialize_top8_route_columns",
    "materialize_top8_routes",
    "prepare_ring_layout",
    "solve_structured_solution_map",
]
