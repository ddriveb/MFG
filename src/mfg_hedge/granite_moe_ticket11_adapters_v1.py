"""Non-exact quota adapters and bounded path repair for Granite Ticket11.

This module intentionally contains no imports from exact flow, Hall, LP, or
minimax planners. Its outputs are fixed-placement virtual quota plans.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Mapping, Sequence


class Ticket11AdapterError(ValueError):
    """Invalid fixed-Gate planner input."""


@dataclass(frozen=True)
class AdapterResult:
    strategy: str
    shares_by_expert: Mapping[int, tuple[float, ...]]
    quotas_by_expert: Mapping[int, tuple[int, ...]]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class RepairResult:
    quotas_by_expert: Mapping[int, tuple[int, ...]]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, order=True)
class _Action:
    expert_id: int
    source_slot: int
    destination_slot: int


def _validate(
    counts_in: Mapping[int, int],
    placements_in: Mapping[int, Sequence[object]],
    rates_in: Mapping[str, float],
):
    if not isinstance(counts_in, Mapping) or not counts_in:
        raise Ticket11AdapterError("expert counts must be a nonempty mapping")
    counts = {}
    for expert, count in counts_in.items():
        if type(expert) is not int or expert < 0:
            raise Ticket11AdapterError("Expert IDs must be nonnegative native integers")
        if type(count) is not int or count < 0:
            raise Ticket11AdapterError("route counts must be nonnegative native integers")
        counts[expert] = count
    if set(placements_in) != set(counts):
        raise Ticket11AdapterError("placement Expert keys must exactly match counts")
    placements = {}
    for expert in sorted(counts):
        row = tuple(placements_in[expert])
        ids = tuple(getattr(place, "replica_id", None) for place in row)
        row_devices = tuple(getattr(place, "device", None) for place in row)
        if len(row) != 2:
            raise Ticket11AdapterError("Ticket11 requires exactly two replicas per Expert")
        if any(not isinstance(value, str) or not value for value in ids + row_devices):
            raise Ticket11AdapterError("replica IDs and devices must be nonempty strings")
        if len(set(ids)) != 2 or len(set(row_devices)) != 2:
            raise Ticket11AdapterError("replica IDs/devices must be unique per Expert")
        placements[expert] = row
    devices = tuple(sorted({place.device for row in placements.values() for place in row}))
    if len(devices) not in (4, 6, 8):
        raise Ticket11AdapterError("only the frozen four/six/eight-device placements are supported")
    if set(rates_in) != set(devices):
        raise Ticket11AdapterError("service-rate keys must match active placement devices")
    rates = {}
    for device in devices:
        value = rates_in[device]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Ticket11AdapterError("service rates must be finite positive numbers")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise Ticket11AdapterError("service rates must be finite positive numbers")
        rates[device] = value
    return dict(sorted(counts.items())), placements, devices, rates


def largest_remainder(total: int, shares: Sequence[float]) -> tuple[int, ...]:
    if type(total) is not int or total < 0:
        raise Ticket11AdapterError("total must be a nonnegative native integer")
    if not shares:
        if total:
            raise Ticket11AdapterError("positive total cannot use an empty share row")
        return ()
    checked = []
    for value in shares:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Ticket11AdapterError("shares must be finite nonnegative numbers")
        value = float(value)
        if not math.isfinite(value) or value < -1e-12:
            raise Ticket11AdapterError("shares must be finite nonnegative numbers")
        checked.append(max(0.0, value))
    if total == 0:
        return tuple(0 for _ in checked)
    norm = math.fsum(checked)
    if norm <= 0:
        raise Ticket11AdapterError("positive demand has zero share")
    exact = [total * value / norm for value in checked]
    result = [math.floor(value) for value in exact]
    remainder = total - sum(result)
    order = sorted(range(len(result)), key=lambda index: (-(exact[index] - result[index]), index))
    for index in order[:remainder]:
        result[index] += 1
    if sum(result) != total:
        raise AssertionError("largest remainder lost route conservation")
    return tuple(result)


def _integerize(counts, shares):
    return {
        expert: largest_remainder(counts[expert], shares[expert])
        for expert in sorted(counts)
    }


def _loads(quotas, placements, devices):
    loads = {device: 0 for device in devices}
    for expert in sorted(quotas):
        for amount, place in zip(quotas[expert], placements[expert]):
            loads[place.device] += amount
    return loads


def _fhard(loads, rates):
    return max(loads[device] / rates[device] for device in sorted(loads))


def equal_quota_like(counts_in, placements_in, rates_in) -> AdapterResult:
    counts, placements, devices, rates = _validate(counts_in, placements_in, rates_in)
    shares = {
        expert: (0.5, 0.5)
        for expert in counts
    }
    quotas = _integerize(counts, shares)
    loads = _loads(quotas, placements, devices)
    return AdapterResult(
        "lazarus_equal_quota_like",
        shares,
        quotas,
        {
            "method": "uniform share over legal copies plus stable largest remainder",
            "device_load_routes": loads,
            "Fhard_virtual_ms": _fhard(loads, rates),
            "exact_solver_called": False,
        },
    )


def greedy_min_load_like(counts_in, placements_in, rates_in) -> AdapterResult:
    counts, placements, devices, rates = _validate(counts_in, placements_in, rates_in)
    device_index = {device: index for index, device in enumerate(devices)}
    loads = {device: 0 for device in devices}
    quotas = {expert: [0, 0] for expert in counts}
    route_decisions = 0
    for expert in sorted(counts):
        for _route in range(counts[expert]):
            slot = min(
                range(2),
                key=lambda index: (
                    loads[placements[expert][index].device]
                    / rates[placements[expert][index].device],
                    device_index[placements[expert][index].device],
                    index,
                ),
            )
            device = placements[expert][slot].device
            quotas[expert][slot] += 1
            loads[device] += 1
            route_decisions += 1
    normalized_quotas = {expert: tuple(row) for expert, row in quotas.items()}
    shares = {}
    for expert, count in counts.items():
        shares[expert] = (
            tuple(value / count for value in normalized_quotas[expert])
            if count
            else (0.5, 0.5)
        )
    return AdapterResult(
        "ultraep_greedy_min_load_like_nonexact",
        shares,
        normalized_quotas,
        {
            "method": "Expert-ordered one-route greedy to least current normalized device load",
            "device_load_routes": loads,
            "Fhard_virtual_ms": _fhard(loads, rates),
            "route_decisions": route_decisions,
            "exact_solver_called": False,
        },
    )


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def topology_masked_sinkhorn_like(
    counts_in,
    placements_in,
    rates_in,
    *,
    epsilon: float = 1.0,
    max_iterations: int = 2000,
    residual_tolerance_routes: float = 1e-6,
) -> AdapterResult:
    counts, placements, devices, rates = _validate(counts_in, placements_in, rates_in)
    if epsilon != 1.0:
        raise Ticket11AdapterError("the frozen zero-cost Sinkhorn epsilon is exactly 1.0")
    if type(max_iterations) is not int or max_iterations != 2000:
        raise Ticket11AdapterError("the frozen Sinkhorn iteration cap is exactly 2000")
    if residual_tolerance_routes != 1e-6:
        raise Ticket11AdapterError("the frozen Sinkhorn residual gate is exactly 1e-6 routes")
    total = sum(counts.values())
    if total <= 0 or (len(devices) in (4, 8) and total % len(devices)):
        raise Ticket11AdapterError("route total must be positive and divisible by device count")
    target = total / len(devices)
    active_experts = tuple(expert for expert in sorted(counts) if counts[expert] > 0)
    device_rows = {
        device: tuple(
            expert for expert in active_experts
            if any(place.device == device for place in placements[expert])
        )
        for device in devices
    }
    if any(not device_rows[device] for device in devices):
        raise Ticket11AdapterError("masked transport has an active column with no positive-demand support")
    if any(not active_experts for _ in (0,)):
        raise Ticket11AdapterError("masked transport has no positive-demand rows")

    # The zero-cost kernel is 1 for every legal edge. Log scaling avoids
    # exponentiating the dual variables until a marginal diagnostic is needed.
    log_v = {device: 0.0 for device in devices}
    log_u = {}
    matrix = {}
    row_residual = math.inf
    column_residual = math.inf
    status = "max_iterations"
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        for expert in active_experts:
            log_u[expert] = math.log(counts[expert]) - _logsumexp(
                [log_v[place.device] for place in placements[expert]]
            )
        for device in devices:
            log_v[device] = math.log(target) - _logsumexp(
                [log_u[expert] for expert in device_rows[device]]
            )
        if any(not math.isfinite(value) for value in (*log_u.values(), *log_v.values())):
            status = "numeric_failure"
            iterations = iteration
            break
        matrix = {}
        for expert in active_experts:
            matrix[expert] = []
            for place in placements[expert]:
                log_value = log_u[expert] + log_v[place.device]
                if log_value > 709.0:
                    status = "numeric_failure"
                    break
                matrix[expert].append(math.exp(log_value))
            if status == "numeric_failure":
                break
        if status == "numeric_failure":
            iterations = iteration
            break
        row_residual = max(
            abs(math.fsum(matrix[expert]) - counts[expert])
            for expert in active_experts
        )
        current_loads = {device: 0.0 for device in devices}
        for expert in active_experts:
            for amount, place in zip(matrix[expert], placements[expert]):
                current_loads[place.device] += amount
        column_residual = max(abs(current_loads[device] - target) for device in devices)
        iterations = iteration
        if max(row_residual, column_residual) <= residual_tolerance_routes:
            status = "converged"
            break

    if status == "numeric_failure":
        raise Ticket11AdapterError("masked Sinkhorn produced nonfinite scaling; arm failed")
    shares = {}
    for expert in sorted(counts):
        if counts[expert] == 0:
            shares[expert] = (0.5, 0.5)
        else:
            row_logs = [log_v[place.device] for place in placements[expert]]
            normalizer = _logsumexp(row_logs)
            shares[expert] = tuple(math.exp(value - normalizer) for value in row_logs)
    quotas = _integerize(counts, shares)
    loads = _loads(quotas, placements, devices)
    return AdapterResult(
        "topology_masked_sinkhorn_like_nonexact",
        shares,
        quotas,
        {
            "method": "one-source-row zero-cost legal-mask Sinkhorn with uniform device margins",
            "epsilon": epsilon,
            "iterations": iterations,
            "status": status,
            "converged": status == "converged",
            "row_marginal_residual_routes": row_residual,
            "column_marginal_residual_routes": column_residual,
            "residual_tolerance_routes": residual_tolerance_routes,
            "column_target_routes": target,
            "device_load_routes_after_integerization": loads,
            "Fhard_virtual_ms_after_integerization": _fhard(loads, rates),
            "exact_solver_called": False,
            "source_rank_dimension_present": False,
            "communication_cost_ms_per_route": 0.0,
        },
    )


def repair_integer_quotas_by_residual_path_v2(
    counts_in,
    placements_in,
    starting_quotas_in,
    rates_in,
    *,
    max_path_batches: int = 8192,
    max_candidate_evaluations: int = 100000,
) -> RepairResult:
    counts, placements, devices, rates = _validate(counts_in, placements_in, rates_in)
    if set(starting_quotas_in) != set(counts):
        raise Ticket11AdapterError("starting quota Expert keys must match route counts")
    if type(max_path_batches) is not int or max_path_batches != 8192:
        raise Ticket11AdapterError("the frozen accepted-path cap is exactly 8192")
    if type(max_candidate_evaluations) is not int or max_candidate_evaluations != 100000:
        raise Ticket11AdapterError("the frozen candidate cap is exactly 100000")
    if len({rates[device] for device in devices}) != 1:
        raise Ticket11AdapterError("bounded path repair only supports the frozen equal-rate model")
    total = sum(counts.values())
    if total <= 0 or total % len(devices):
        raise Ticket11AdapterError("route total must divide evenly across frozen devices")
    target = total // len(devices)
    quota = {}
    for expert in sorted(counts):
        row = tuple(starting_quotas_in[expert])
        if len(row) != 2 or any(type(value) is not int or value < 0 for value in row):
            raise Ticket11AdapterError("starting quotas must be two nonnegative native integers")
        if sum(row) != counts[expert]:
            raise Ticket11AdapterError(f"starting quota for Expert {expert} does not conserve")
        quota[expert] = row
    max_path_edges = len(devices) // 2
    current_loads = _loads(quota, placements, devices)

    def potential(loads):
        f = _fhard(loads, rates)
        overload = sum(max(0, loads[device] - target) for device in devices)
        return f, overload

    current_potential = potential(current_loads)
    initial_loads = dict(current_loads)
    initial_potential = current_potential
    accepted = []
    candidate_evaluations = 0
    status = None

    def action_options(source, destination):
        options = []
        for expert in sorted(counts):
            src_slots = [i for i, place in enumerate(placements[expert]) if place.device == source]
            dst_slots = [i for i, place in enumerate(placements[expert]) if place.device == destination]
            if src_slots and dst_slots:
                options.append(_Action(expert, src_slots[0], dst_slots[0]))
        return options

    def enumerate_paths(source, destination):
        paths = []
        def walk(path):
            current = path[-1]
            edges = len(path) - 1
            if current == destination:
                if edges:
                    paths.append(tuple(path))
                return
            if edges >= max_path_edges:
                return
            for next_device in devices:
                if next_device in path:
                    continue
                if action_options(current, next_device):
                    walk(path + [next_device])
        walk([source])
        return paths

    while status is None:
        current_f, current_overload = current_potential
        if current_overload == 0:
            status = "target_reached"
            break
        if len(accepted) >= max_path_batches:
            status = "budget_exhausted"
            break
        sources = [device for device in devices if current_loads[device] > target]
        destinations = [device for device in devices if current_loads[device] < target]
        all_paths = []
        for source in sources:
            for destination in destinations:
                all_paths.extend(enumerate_paths(source, destination))
        all_paths = sorted(set(all_paths), key=lambda path: (len(path) - 1, path))
        moved = False
        budget_hit = False
        for path in all_paths:
            edge_options = [
                action_options(path[index], path[index + 1])
                for index in range(len(path) - 1)
            ]
            if any(not options for options in edge_options):
                continue
            for actions in itertools.product(*edge_options):
                if candidate_evaluations >= max_candidate_evaluations:
                    budget_hit = True
                    break
                candidate_evaluations += 1
                candidate = {expert: list(row) for expert, row in quota.items()}
                valid = True
                for action in actions:
                    row = candidate[action.expert_id]
                    if row[action.source_slot] <= 0:
                        valid = False
                        break
                    row[action.source_slot] -= 1
                    row[action.destination_slot] += 1
                if not valid:
                    continue
                candidate = {expert: tuple(row) for expert, row in candidate.items()}
                if any(sum(candidate[expert]) != counts[expert] for expert in counts):
                    continue
                next_loads = _loads(candidate, placements, devices)
                next_potential = potential(next_loads)
                if next_potential[0] > current_f:
                    continue
                if not (next_potential < current_potential):
                    continue
                accepted.append({
                    "path_devices": list(path),
                    "actions": [
                        {
                            "expert_id": action.expert_id,
                            "source_slot": action.source_slot,
                            "destination_slot": action.destination_slot,
                        }
                        for action in actions
                    ],
                    "device_load_routes_before": dict(current_loads),
                    "device_load_routes_after": dict(next_loads),
                    "potential_before": list(current_potential),
                    "potential_after": list(next_potential),
                })
                quota = candidate
                current_loads = next_loads
                current_potential = next_potential
                moved = True
                break
            if moved or budget_hit:
                break
        if budget_hit:
            status = "budget_exhausted"
        elif moved and current_potential[1] == 0:
            status = "target_reached"
        elif moved:
            continue
        else:
            status = "local_no_path"

    for expert in sorted(counts):
        if len(quota[expert]) != len(placements[expert]):
            raise AssertionError("repair changed the legal row width")
        if any(value < 0 for value in quota[expert]) or sum(quota[expert]) != counts[expert]:
            raise AssertionError("repair violated quota conservation")
    final_potential = potential(current_loads)
    return RepairResult(
        quota,
        {
            "status": status,
            "route_count": total,
            "device_count": len(devices),
            "target_load_routes_per_device": target,
            "max_path_edges": max_path_edges,
            "max_path_batches": max_path_batches,
            "max_candidate_evaluations": max_candidate_evaluations,
            "candidate_evaluations": candidate_evaluations,
            "accepted_path_batch_count": len(accepted),
            "token_migrations": sum(len(batch["actions"]) for batch in accepted),
            "accepted_path_batches": accepted,
            "initial_device_load_routes": initial_loads,
            "final_device_load_routes": current_loads,
            "initial_Fhard_ms": initial_potential[0],
            "final_Fhard_ms": final_potential[0],
            "initial_overload_routes": initial_potential[1],
            "final_overload_routes": final_potential[1],
            "Fhard_nonincreased": final_potential[0] <= initial_potential[0],
            "exact_solver_called": False,
        },
    )


def quota_metrics(counts, placements, rates, quotas):
    normalized_counts, normalized_placements, devices, normalized_rates = _validate(
        counts, placements, rates
    )
    if set(quotas) != set(normalized_counts):
        raise Ticket11AdapterError("quota Expert keys must match counts")
    checked = {}
    for expert in sorted(normalized_counts):
        row = tuple(quotas[expert])
        if len(row) != len(normalized_placements[expert]):
            raise Ticket11AdapterError("quota row width differs from legal placement")
        if any(type(value) is not int or value < 0 for value in row):
            raise Ticket11AdapterError("quota entries must be nonnegative native integers")
        if sum(row) != normalized_counts[expert]:
            raise Ticket11AdapterError(f"quota row for Expert {expert} does not conserve")
        checked[expert] = row
    loads = _loads(checked, normalized_placements, devices)
    return {
        "device_load_routes": loads,
        "device_time_ms": {
            device: loads[device] / normalized_rates[device] for device in devices
        },
        "Fhard_virtual_ms": _fhard(loads, normalized_rates),
        "route_conservation_valid": True,
        "legal_placement_valid": True,
    }


__all__ = [
    "AdapterResult",
    "RepairResult",
    "Ticket11AdapterError",
    "equal_quota_like",
    "greedy_min_load_like",
    "largest_remainder",
    "quota_metrics",
    "repair_integer_quotas_by_residual_path_v2",
    "topology_masked_sinkhorn_like",
]
