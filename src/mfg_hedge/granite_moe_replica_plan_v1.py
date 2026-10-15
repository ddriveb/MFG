"""Dependency-free deterministic plans for fixed-Gate same-Expert replicas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


class ReplicaPlanError(ValueError):
    """Raised when a fixed-Gate route plan cannot be placed without changing Experts."""


@dataclass(frozen=True)
class ReplicaPlacement:
    replica_id: str
    device: str


@dataclass(frozen=True)
class RouteAssignment:
    route_index: int
    token_index: int
    route_slot: int
    expert_id: int
    replica_id: str
    device: str


@dataclass(frozen=True)
class ReplicaDispatchPlan:
    assignments: tuple[RouteAssignment, ...]
    quotas_by_expert: Mapping[int, tuple[int, ...]]
    placements_by_expert: Mapping[int, tuple[ReplicaPlacement, ...]]

    @property
    def route_count(self) -> int:
        return len(self.assignments)

    def groups(self) -> dict[tuple[int, str, str], tuple[RouteAssignment, ...]]:
        """Return Expert/replica groups in deterministic route order."""
        grouped: dict[tuple[int, str, str], list[RouteAssignment]] = {}
        for route in self.assignments:
            key = (route.expert_id, route.replica_id, route.device)
            grouped.setdefault(key, []).append(route)
        return {key: tuple(value) for key, value in grouped.items()}


def build_replica_dispatch_plan(
    expert_ids: Sequence[int],
    token_indices: Sequence[int],
    route_slots: Sequence[int],
    replicas_by_expert: Mapping[int, Sequence[ReplicaPlacement]],
    quotas_by_expert: Mapping[int, Sequence[int]] | None = None,
) -> ReplicaDispatchPlan:
    """Assign each existing Gate route to a replica of its same logical Expert.

    Route inputs are the *original* route records. This function only produces an
    index map: it never changes Expert IDs, token IDs, route slots, or Gate weights.
    For each Expert, quota membership is stable in ``(token_index, route_slot,
    route_index)`` order. If quotas are omitted, routes are split evenly with the
    first replicas receiving any remainder.
    """
    lengths = (len(expert_ids), len(token_indices), len(route_slots))
    if len(set(lengths)) != 1:
        raise ReplicaPlanError(f"route columns have different lengths: {lengths}")

    routes_by_expert: dict[int, list[tuple[int, int, int]]] = {}
    seen_token_slots: set[tuple[int, int]] = set()
    for route_index, (expert_id, token_index, route_slot) in enumerate(
        zip(expert_ids, token_indices, route_slots)
    ):
        if not isinstance(expert_id, int) or isinstance(expert_id, bool) or expert_id < 0:
            raise ReplicaPlanError(f"invalid Expert id at route {route_index}: {expert_id!r}")
        if not isinstance(token_index, int) or isinstance(token_index, bool) or token_index < 0:
            raise ReplicaPlanError(f"invalid token id at route {route_index}: {token_index!r}")
        if not isinstance(route_slot, int) or isinstance(route_slot, bool) or route_slot < 0:
            raise ReplicaPlanError(f"invalid route slot at route {route_index}: {route_slot!r}")
        identity = (token_index, route_slot)
        if identity in seen_token_slots:
            raise ReplicaPlanError(f"duplicate (token, route slot): {identity}")
        seen_token_slots.add(identity)
        routes_by_expert.setdefault(expert_id, []).append((route_index, token_index, route_slot))

    normalized_placements: dict[int, tuple[ReplicaPlacement, ...]] = {}
    normalized_quotas: dict[int, tuple[int, ...]] = {}
    assigned_by_route: dict[int, RouteAssignment] = {}

    for expert_id, routes in routes_by_expert.items():
        placements = tuple(replicas_by_expert.get(expert_id, ()))
        if not placements:
            raise ReplicaPlanError(f"Expert {expert_id} has Gate routes but no replica placement")
        replica_ids = [placement.replica_id for placement in placements]
        if any(not item or not item.strip() for item in replica_ids):
            raise ReplicaPlanError(f"Expert {expert_id} has an empty replica id")
        if len(set(replica_ids)) != len(replica_ids):
            raise ReplicaPlanError(f"Expert {expert_id} has duplicate replica ids")
        if any(not placement.device or not placement.device.strip() for placement in placements):
            raise ReplicaPlanError(f"Expert {expert_id} has an empty replica device")

        if quotas_by_expert is None or expert_id not in quotas_by_expert:
            base, remainder = divmod(len(routes), len(placements))
            quotas = tuple(base + (index < remainder) for index in range(len(placements)))
        else:
            raw_quotas = tuple(quotas_by_expert[expert_id])
            if len(raw_quotas) != len(placements):
                raise ReplicaPlanError(
                    f"Expert {expert_id} quota width {len(raw_quotas)} does not match "
                    f"replica count {len(placements)}"
                )
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in raw_quotas):
                raise ReplicaPlanError(f"Expert {expert_id} quotas must be nonnegative integers")
            quotas = raw_quotas
            if sum(quotas) != len(routes):
                raise ReplicaPlanError(
                    f"Expert {expert_id} quotas sum to {sum(quotas)}, "
                    f"but Gate produced {len(routes)} routes"
                )

        normalized_placements[expert_id] = placements
        normalized_quotas[expert_id] = tuple(int(value) for value in quotas)
        stable_routes = sorted(routes, key=lambda item: (item[1], item[2], item[0]))
        route_offset = 0
        for placement, quota in zip(placements, quotas):
            for route_index, token_index, route_slot in stable_routes[route_offset : route_offset + quota]:
                assigned_by_route[route_index] = RouteAssignment(
                    route_index=route_index,
                    token_index=token_index,
                    route_slot=route_slot,
                    expert_id=expert_id,
                    replica_id=placement.replica_id,
                    device=placement.device,
                )
            route_offset += quota

    assignments = tuple(assigned_by_route[index] for index in range(len(expert_ids)))
    if len(assignments) != len(expert_ids):
        raise ReplicaPlanError("internal error: a Gate route was not assigned exactly once")
    return ReplicaDispatchPlan(
        assignments=assignments,
        quotas_by_expert=normalized_quotas,
        placements_by_expert=normalized_placements,
    )
