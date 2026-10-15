"""Optional C++17 fixed-Gate route schedule builder for the frozen cloud executor.

The shared library is compiled separately once with the documented g++ command.
Importing this bridge never compiles code and never falls back to the Python plan.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence


class NativeScheduleError(ValueError):
    """Native scheduling is unavailable or the supplied fixed-Gate plan is invalid."""


class LazyReplicaDispatchPlan:
    """Compact plan metadata; RouteAssignment objects are built only on explicit access."""

    def __init__(self, expert_ids, token_ids, route_slots, replica_by_route,
                 quotas_by_expert, placements_by_expert):
        self._expert_ids = expert_ids
        self._token_ids = token_ids
        self._route_slots = route_slots
        self._replica_by_route = replica_by_route
        self.quotas_by_expert = quotas_by_expert
        self.placements_by_expert = placements_by_expert
        self._materialized = None

    @property
    def route_count(self) -> int:
        return int(self._expert_ids.numel())

    @property
    def assignments(self):
        return self.materialize().assignments

    def materialize(self):
        if self._materialized is None:
            from mfg_hedge.granite_moe_replica_plan_v1 import ReplicaDispatchPlan, RouteAssignment

            experts = self._expert_ids.tolist()
            tokens = self._token_ids.tolist()
            slots = self._route_slots.tolist()
            owners = self._replica_by_route.tolist()
            assignments = tuple(
                RouteAssignment(
                    route_index=route,
                    token_index=int(tokens[route]),
                    route_slot=int(slots[route]),
                    expert_id=int(experts[route]),
                    replica_id=self.placements_by_expert[int(experts[route])][owners[route]].replica_id,
                    device=self.placements_by_expert[int(experts[route])][owners[route]].device,
                )
                for route in range(len(experts))
            )
            self._materialized = ReplicaDispatchPlan(
                assignments=assignments,
                quotas_by_expert=self.quotas_by_expert,
                placements_by_expert=self.placements_by_expert,
            )
        return self._materialized

    def groups(self):
        return self.materialize().groups()


@lru_cache(maxsize=4)
def _load_library(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        raise NativeScheduleError(f"native schedule library does not exist: {path}")
    library = ctypes.CDLL(str(path))
    fn = library.granite_cloud_build_schedule_v1
    i64p, i32p = ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [
        ctypes.c_int64, i64p, i64p, i64p, ctypes.c_int32, ctypes.c_int32,
        i32p, i64p, i64p, i64p, i32p, ctypes.POINTER(ctypes.c_char), ctypes.c_int64,
    ]
    fn.restype = ctypes.c_int
    return library


def _pointer(tensor, ctype):
    if int(tensor.numel()) == 0:
        return ctypes.POINTER(ctype)()
    return ctypes.cast(ctypes.c_void_p(int(tensor.data_ptr())), ctypes.POINTER(ctype))


def _require_cpu_i64(tensor, name: str):
    try:
        import torch
    except ImportError as exc:
        raise NativeScheduleError("the bridge requires the existing PyTorch runtime") from exc
    if not isinstance(tensor, torch.Tensor):
        raise NativeScheduleError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 1 or tensor.dtype != torch.int64 or tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise NativeScheduleError(f"{name} must be a contiguous CPU int64 vector")
    return tensor


def build_native_execution_schedule(
    expert_ids,
    token_ids,
    route_slots,
    layout,
    quotas_by_expert: Mapping[int, Sequence[int]],
    library_path: str | Path,
):
    """Build a CloudExecutionSchedule-compatible value without RouteAssignment churn.

    CPU transfer of captured CUDA route columns, if needed, belongs in the caller's
    C timer. This function accepts only contiguous CPU int64 tensors. The C-owned
    route-order buffer is converted to the executor's tuple[int] interface here, so
    callers include that transfer in their C timing.
    """
    from mfg_hedge.granite_cloud_executor_v1 import (
        CloudExecutionSchedule, SUPPORTED_DEVICE_COUNTS, _logical_gpu_index,
    )

    expert_ids = _require_cpu_i64(expert_ids, "expert_ids")
    token_ids = _require_cpu_i64(token_ids, "token_ids")
    route_slots = _require_cpu_i64(route_slots, "route_slots")
    n = int(expert_ids.numel())
    if int(token_ids.numel()) != n or int(route_slots.numel()) != n:
        raise NativeScheduleError("route columns must have the same length")
    if layout.device_count not in SUPPORTED_DEVICE_COUNTS:
        raise NativeScheduleError("native cloud schedule supports the frozen 4/6/8-GPU rings")
    placement_flat: list[int] = []
    active_placements = {}
    for expert in range(32):
        placements = tuple(layout.placements_by_expert.get(expert, ()))
        if len(placements) != 2:
            raise NativeScheduleError(f"Expert {expert} must have exactly two ring placements")
        if any(not p.replica_id or not p.replica_id.strip() for p in placements):
            raise NativeScheduleError(f"Expert {expert} has an empty replica ID")
        if len({p.replica_id for p in placements}) != 2:
            raise NativeScheduleError(f"Expert {expert} has duplicate replica IDs")
        gpu_pair = tuple(_logical_gpu_index(p.device, layout) for p in placements)
        if gpu_pair[0] == gpu_pair[1] or any(g < 0 or g >= layout.device_count for g in gpu_pair):
            raise NativeScheduleError(f"Expert {expert} replicas must occupy distinct logical GPUs")
        placement_flat.extend(gpu_pair)
        active_placements[expert] = placements
    if set(quotas_by_expert) != set(range(32)):
        raise NativeScheduleError("native schedule requires exactly 32 Expert quota entries")
    quota_values: list[int] = []
    normalized_quotas = {}
    for expert in range(32):
        raw = tuple(quotas_by_expert[expert])
        if len(raw) != 2 or any(type(value) is not int or value < 0 for value in raw):
            raise NativeScheduleError(f"Expert {expert} quota must contain two nonnegative integers")
        normalized_quotas[expert] = raw
        quota_values.extend(raw)

    try:
        import torch
    except ImportError as exc:
        raise NativeScheduleError("the bridge requires the existing PyTorch runtime") from exc
    order = torch.empty((n,), dtype=torch.int64, device="cpu")
    counts = torch.empty((layout.device_count * 32,), dtype=torch.int64, device="cpu")
    owner = torch.empty((n,), dtype=torch.int32, device="cpu")
    placement_array = (ctypes.c_int32 * 64)(*placement_flat)
    quota_array = (ctypes.c_int64 * 64)(*quota_values)
    error = ctypes.create_string_buffer(512)
    library = _load_library(str(Path(library_path).resolve()))
    status = library.granite_cloud_build_schedule_v1(
        n, _pointer(expert_ids, ctypes.c_int64), _pointer(token_ids, ctypes.c_int64),
        _pointer(route_slots, ctypes.c_int64), int(layout.device_count), 2,
        placement_array, quota_array, _pointer(order, ctypes.c_int64),
        _pointer(counts, ctypes.c_int64), _pointer(owner, ctypes.c_int32),
        error, len(error),
    )
    if status != 0:
        raise NativeScheduleError(error.value.decode("utf-8", errors="replace") or f"native status {status}")
    count_rows = tuple(tuple(int(x) for x in counts[g * 32:(g + 1) * 32].tolist()) for g in range(layout.device_count))
    route_order = tuple(order.tolist())
    active_experts = {e for e in range(32) if sum(count_rows[g][e] for g in range(layout.device_count)) > 0}
    # Preserve the old eager planner's snapshot semantics without allocating one
    # Python RouteAssignment object per route on the measured path.
    plan_experts, plan_tokens, plan_slots = expert_ids.clone(), token_ids.clone(), route_slots.clone()
    plan = LazyReplicaDispatchPlan(
        plan_experts, plan_tokens, plan_slots, owner.clone(),
        {e: normalized_quotas[e] for e in active_experts},
        {e: active_placements[e] for e in active_experts},
    )
    return CloudExecutionSchedule(
        plan=plan,
        route_indices_by_execution_order=route_order,
        counts_by_gpu_expert=count_rows,
        device_count=layout.device_count,
        route_count=n,
    )


def assert_schedule_equivalent(native_schedule, python_schedule) -> None:
    """Full, intentionally expensive differential check; call outside timed runs."""
    native_order = tuple(native_schedule.route_indices_by_execution_order)
    if native_order != tuple(python_schedule.route_indices_by_execution_order):
        raise NativeScheduleError("native route order differs from Python schedule")
    if native_schedule.counts_by_gpu_expert != python_schedule.counts_by_gpu_expert:
        raise NativeScheduleError("native GPU/Expert counts differ from Python schedule")
    if native_schedule.plan.materialize() != python_schedule.plan:
        raise NativeScheduleError("native RouteAssignment membership differs from Python plan")
