"""Common fixed-Gate execution for the Granite three-policy cloud comparison.

The policy adapters in this module only produce same-Expert replica quotas.
Every policy then uses the same stable route schedule and distributed
dispatch/Expert/combine path. Importing this module does not load Transformers
or a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import time
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional cloud runtime
    raise ImportError("Granite cloud execution requires PyTorch") from exc

from .granite_moe_replica_plan_v1 import (
    ReplicaDispatchPlan,
    ReplicaPlacement,
    build_replica_dispatch_plan,
)
from .granite_moe_ticket11_adapters_v1 import topology_masked_sinkhorn_like
from .granite_moe_eplb_lplb_ticket13_approx_v1 import lplb_inspired_approximate
from .granite_structured_solution_map_v1 import (
    RingLayout,
    integerize_equal_integral_loads,
    integerize_group_balanced,
    solve_structured_solution_map,
)


EXPERT_COUNT = 32
SUPPORTED_DEVICE_COUNTS = (4, 6, 8)
POLICY_NAMES = (
    "structured_group_balanced",
    "lplb_inspired_approx_lr",
    "topology_masked_sinkhorn_like_nonexact",
)


class GraniteCloudExecutorError(ValueError):
    """Invalid policy, schedule, weight, or distributed executor input."""


@dataclass(frozen=True)
class PolicyPlan:
    strategy: str
    shares_by_expert: Mapping[int, tuple[float, ...]]
    quotas_by_expert: Mapping[int, tuple[int, ...]]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CloudExecutionSchedule:
    """One conserved route map, stored in destination/Expert/native order."""

    plan: ReplicaDispatchPlan
    route_indices_by_execution_order: tuple[int, ...]
    counts_by_gpu_expert: tuple[tuple[int, ...], ...]
    device_count: int
    route_count: int


def make_ring_placements(device_count: int) -> dict[int, tuple[ReplicaPlacement, ReplicaPlacement]]:
    """Return the frozen adjacent cyclic two-replica layout for every Expert."""
    if type(device_count) is not int or device_count not in SUPPORTED_DEVICE_COUNTS:
        raise GraniteCloudExecutorError(
            f"ring placement requires device_count in {SUPPORTED_DEVICE_COUNTS}"
        )
    return {
        expert: (
            ReplicaPlacement(f"expert-{expert}-replica-0", f"cuda:{expert % device_count}"),
            ReplicaPlacement(
                f"expert-{expert}-replica-1",
                f"cuda:{(expert + 1) % device_count}",
            ),
        )
        for expert in range(EXPERT_COUNT)
    }


def _normalize_counts(counts: Mapping[int, int] | Sequence[int]) -> dict[int, int]:
    if isinstance(counts, Mapping):
        if set(counts) != set(range(EXPERT_COUNT)):
            raise GraniteCloudExecutorError("expert counts must have exactly Expert IDs 0..31")
        values = {expert: counts[expert] for expert in range(EXPERT_COUNT)}
    else:
        if len(counts) != EXPERT_COUNT:
            raise GraniteCloudExecutorError("expert count sequence must have 32 entries")
        values = {expert: counts[expert] for expert in range(EXPERT_COUNT)}
    for expert, value in values.items():
        if type(value) is not int or value < 0:
            raise GraniteCloudExecutorError(
                f"route count for Expert {expert} must be a nonnegative integer"
            )
    return values


def plan_policy(
    strategy: str,
    counts: Mapping[int, int] | Sequence[int],
    layout: RingLayout,
) -> PolicyPlan:
    """Apply one frozen planner and preserve its native integerization/status."""
    if strategy not in POLICY_NAMES:
        raise GraniteCloudExecutorError(f"unknown cloud policy {strategy!r}")
    if not isinstance(layout, RingLayout):
        raise GraniteCloudExecutorError("layout must be prepared by prepare_ring_layout")
    normalized = _normalize_counts(counts)
    rates = {device: 10.0 for device in layout.devices}
    if strategy == "structured_group_balanced":
        solution = solve_structured_solution_map(normalized, layout)
        quotas = integerize_equal_integral_loads(normalized, layout, solution)
        integerizer = "integerize_equal_integral_loads"
        if quotas is None:
            quotas = integerize_group_balanced(normalized, layout, solution)
            integerizer = "integerize_group_balanced"
        return PolicyPlan(
            strategy,
            dict(solution.shares_by_expert),
            dict(quotas),
            {
                "planner": "frozen Ticket18 structured ring solution map",
                "integerizer": integerizer,
                "fw_gap_ms": solution.fw_gap_ms,
                "fw_gate_pass": solution.fw_gate_pass,
                "kkt_violation_routes": solution.kkt_violation_routes,
                "load_scale": solution.load_scale,
                "exact_generic_solver_called": False,
            },
        )
    if strategy == "lplb_inspired_approx_lr":
        result = lplb_inspired_approximate(
            normalized,
            layout.placements_by_expert,
            rates,
        )
    else:
        result = topology_masked_sinkhorn_like(
            normalized,
            layout.placements_by_expert,
            rates,
        )
    return PolicyPlan(
        strategy,
        dict(result.shares_by_expert),
        dict(result.quotas_by_expert),
        dict(result.diagnostics),
    )


def _logical_gpu_index(device: str, layout: RingLayout) -> int:
    text = str(device)
    if text in layout.devices:
        return layout.devices.index(text)
    if text.startswith("cuda:"):
        suffix = text[5:]
        if suffix.isdigit():
            return int(suffix)
    if text.isdigit():
        return int(text)
    raise GraniteCloudExecutorError(f"cannot map placement device {text!r} to a GPU rank")


def build_execution_schedule(
    expert_ids: Sequence[int],
    token_ids: Sequence[int],
    route_slots: Sequence[int],
    layout: RingLayout,
    quotas_by_expert: Mapping[int, Sequence[int]],
) -> CloudExecutionSchedule:
    """Materialize the common destination/Expert schedule from native Gate rows."""
    if not isinstance(layout, RingLayout):
        raise GraniteCloudExecutorError("layout must be prepared by prepare_ring_layout")
    device_count = layout.device_count
    if device_count not in SUPPORTED_DEVICE_COUNTS:
        raise GraniteCloudExecutorError("cloud schedule requires the frozen four/six/eight-GPU ring")
    plan = build_replica_dispatch_plan(
        expert_ids,
        token_ids,
        route_slots,
        layout.placements_by_expert,
        quotas_by_expert=quotas_by_expert,
    )
    route_groups: list[list[list[int]]] = [
        [[] for _ in range(EXPERT_COUNT)] for _ in range(device_count)
    ]
    for assignment in plan.assignments:
        gpu = _logical_gpu_index(assignment.device, layout)
        if not 0 <= gpu < device_count:
            raise GraniteCloudExecutorError(
                f"placement GPU {gpu} is outside the prepared {device_count}-GPU layout"
            )
        route_groups[gpu][assignment.expert_id].append(assignment.route_index)
    counts = tuple(
        tuple(len(route_groups[gpu][expert]) for expert in range(EXPERT_COUNT))
        for gpu in range(device_count)
    )
    execution_order = tuple(
        route_index
        for gpu in range(device_count)
        for expert in range(EXPERT_COUNT)
        for route_index in route_groups[gpu][expert]
    )
    if len(execution_order) != plan.route_count or sorted(execution_order) != list(
        range(plan.route_count)
    ):
        raise GraniteCloudExecutorError("schedule did not conserve every original route exactly once")
    if sum(map(sum, counts)) != plan.route_count:
        raise GraniteCloudExecutorError("schedule count matrix does not conserve the route total")
    return CloudExecutionSchedule(
        plan=plan,
        route_indices_by_execution_order=execution_order,
        counts_by_gpu_expert=counts,
        device_count=device_count,
        route_count=plan.route_count,
    )


def _check_counts32(counts32: Sequence[int], row_count: int) -> tuple[int, ...]:
    if len(counts32) != EXPERT_COUNT:
        raise GraniteCloudExecutorError("local Expert counts must have 32 entries")
    checked: list[int] = []
    for expert, value in enumerate(counts32):
        if type(value) is not int or value < 0:
            raise GraniteCloudExecutorError(
                f"local route count for Expert {expert} must be a nonnegative integer"
            )
        checked.append(value)
    if sum(checked) != row_count:
        raise GraniteCloudExecutorError(
            f"local Expert counts sum to {sum(checked)} but input has {row_count} rows"
        )
    return tuple(checked)


@torch.inference_mode()
def execute_local_experts(
    inputs: torch.Tensor,
    counts32: Sequence[int],
    localweights: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Run packed Granite Expert blocks in Expert-ID order on one physical GPU."""
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise GraniteCloudExecutorError("local Expert input must be a rank-2 tensor")
    if inputs.dtype != torch.float32:
        raise GraniteCloudExecutorError("local Granite Expert compute is frozen to FP32")
    counts = _check_counts32(counts32, int(inputs.shape[0]))
    outputs = torch.empty_like(inputs)
    offset = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        if expert not in localweights:
            raise GraniteCloudExecutorError(
                f"this rank received {count} routes for Expert {expert} but owns no weights"
            )
        weight_in, weight_out = localweights[expert]
        if weight_in.ndim != 2 or weight_out.ndim != 2:
            raise GraniteCloudExecutorError("packed Granite Expert weights must be matrices")
        if weight_in.dtype != inputs.dtype or weight_out.dtype != inputs.dtype:
            raise GraniteCloudExecutorError(f"Expert {expert} weights must match the FP32 input dtype")
        if weight_in.shape[1] != inputs.shape[1] or weight_in.shape[0] % 2:
            raise GraniteCloudExecutorError(
                f"Expert {expert} input weight is incompatible with hidden size"
            )
        intermediate = weight_in.shape[0] // 2
        if tuple(weight_out.shape) != (inputs.shape[1], intermediate):
            raise GraniteCloudExecutorError(
                f"Expert {expert} output weight must have shape (hidden, intermediate)"
            )
        block = inputs.narrow(0, offset, count)
        projected = F.linear(block, weight_in)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        outputs.narrow(0, offset, count).copy_(F.linear(activated, weight_out))
        offset += count
    return outputs


def _as_route_tensor(values: Any, *, name: str, dtype, device) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    if tensor.ndim != 1:
        raise GraniteCloudExecutorError(f"{name} must be a rank-1 route vector")
    if dtype is torch.long and tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise GraniteCloudExecutorError(f"{name} must contain integer token indices")
    return tensor.to(device=device, dtype=dtype).contiguous()


def _weighted_native_scatter(
    flat_hidden: torch.Tensor,
    token_ids: torch.Tensor,
    gate_weights: torch.Tensor,
    route_outputs_native_order: torch.Tensor,
) -> torch.Tensor:
    route_count, hidden_size = route_outputs_native_order.shape
    if token_ids.numel() != route_count or gate_weights.numel() != route_count:
        raise GraniteCloudExecutorError("native token/gate vectors do not match output route count")
    if token_ids.numel() and (
        int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= flat_hidden.shape[0]
    ):
        raise GraniteCloudExecutorError("native token index is outside the captured hidden rows")
    weighted = route_outputs_native_order * gate_weights.reshape(-1, 1)
    output = torch.zeros(
        (flat_hidden.shape[0], hidden_size),
        dtype=flat_hidden.dtype,
        device=flat_hidden.device,
    )
    output.index_add_(0, token_ids, weighted)
    return output


@torch.inference_mode()
def loopback_execute(
    flat_hidden: torch.Tensor,
    token_ids: Sequence[int] | torch.Tensor,
    gate_weights: Sequence[float] | torch.Tensor,
    schedule: CloudExecutionSchedule,
    local_weights: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run all logical GPU groups sequentially on one device for functional smoke."""
    if flat_hidden.ndim != 2:
        raise GraniteCloudExecutorError("captured hidden rows must be a rank-2 tensor")
    if flat_hidden.dtype != torch.float32:
        raise GraniteCloudExecutorError("Granite loopback smoke is frozen to FP32")
    if schedule.route_count != len(schedule.route_indices_by_execution_order):
        raise GraniteCloudExecutorError("schedule route order has the wrong length")
    start = time.perf_counter()
    route_indices = torch.as_tensor(
        schedule.route_indices_by_execution_order,
        dtype=torch.long,
        device=flat_hidden.device,
    )
    native_tokens = _as_route_tensor(
        token_ids, name="token_ids", dtype=torch.long, device=flat_hidden.device
    )
    raw_gates = gate_weights if isinstance(gate_weights, torch.Tensor) else torch.as_tensor(gate_weights)
    if raw_gates.dtype != torch.float32:
        raise GraniteCloudExecutorError("captured Gate weights must be FP32")
    native_gates = _as_route_tensor(
        gate_weights, name="gate_weights", dtype=flat_hidden.dtype, device=flat_hidden.device
    )
    if native_tokens.numel() != schedule.route_count or native_gates.numel() != schedule.route_count:
        raise GraniteCloudExecutorError("root route vectors do not match the loopback schedule")
    if schedule.route_count:
        execution_tokens = native_tokens.index_select(0, route_indices)
        packed_inputs = flat_hidden.index_select(0, execution_tokens)
    else:
        packed_inputs = flat_hidden.new_empty((0, flat_hidden.shape[1]))
    packed_outputs = torch.empty_like(packed_inputs)
    offset = 0
    for gpu in range(schedule.device_count):
        counts = schedule.counts_by_gpu_expert[gpu]
        count = sum(counts)
        if count:
            packed_outputs.narrow(0, offset, count).copy_(
                execute_local_experts(
                    packed_inputs.narrow(0, offset, count),
                    counts,
                    local_weights,
                )
            )
        offset += count
    if schedule.route_count:
        native_outputs = torch.empty_like(packed_outputs)
        native_outputs.index_copy_(0, route_indices, packed_outputs)
    else:
        native_outputs = packed_outputs
    output = _weighted_native_scatter(flat_hidden, native_tokens, native_gates, native_outputs)
    if flat_hidden.device.type == "cuda":
        torch.cuda.synchronize(flat_hidden.device)
    elapsed = (time.perf_counter() - start) * 1000.0
    return output, {
        "execution_mode": "single_device_loopback_functional_control",
        "timing_scope": "sequential logical-GPU replay; not distributed performance",
        "F_wall_ms": elapsed,
        "route_count": schedule.route_count,
        "device_count": schedule.device_count,
    }


def _require_distributed(rank: int, world: int, backend: str) -> None:
    if type(rank) is not int or type(world) is not int or world < 1 or not 0 <= rank < world:
        raise GraniteCloudExecutorError("rank/world must be valid native integers")
    if not dist.is_available() or not dist.is_initialized():
        raise GraniteCloudExecutorError("torch.distributed must be initialized by the cloud runner")
    if dist.get_rank() != rank or dist.get_world_size() != world:
        raise GraniteCloudExecutorError("executor rank/world disagree with initialized process group")
    actual = str(dist.get_backend()).lower()
    if actual != str(backend).lower():
        raise GraniteCloudExecutorError(
            f"executor backend {backend!r} differs from initialized backend {actual!r}"
        )


class DistributedExecutor:
    """One process per GPU all-to-all executor with rank-0 native Gate combine."""

    def __init__(
        self,
        devices: str | Sequence[str],
        backend: str,
        rank: int,
        world: int,
        *,
        root: int = 0,
        process_group=None,
    ) -> None:
        _require_distributed(rank, world, backend)
        if root != 0:
            raise GraniteCloudExecutorError("the frozen cloud executor uses global rank 0 as root")
        if process_group is not None:
            if dist.get_rank(process_group) != rank or dist.get_world_size(process_group) != world:
                raise GraniteCloudExecutorError("executor rank/world disagree with process_group")
        if isinstance(devices, str):
            selected = devices
        else:
            device_list = tuple(str(item) for item in devices)
            if len(device_list) == 1:
                selected = device_list[0]
            elif len(device_list) == world:
                selected = device_list[rank]
            else:
                raise GraniteCloudExecutorError(
                    "devices must be one process-local device or one device per rank"
                )
        self.device = torch.device(selected)
        self.backend = str(backend).lower()
        self.rank = rank
        self.world = world
        self.root = root
        self.process_group = process_group
        if self.backend == "nccl" and self.device.type != "cuda":
            raise GraniteCloudExecutorError("NCCL execution requires one CUDA device per rank")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _metadata(self, root_flat_hidden, root_token_ids, root_gate_weights, root_schedule):
        meta = None
        if self.rank == self.root:
            try:
                if not isinstance(root_flat_hidden, torch.Tensor) or root_flat_hidden.ndim != 2:
                    raise GraniteCloudExecutorError("rank 0 must provide flat hidden rows [tokens, hidden]")
                if root_flat_hidden.dtype != torch.float32:
                    raise GraniteCloudExecutorError("distributed Granite execution is frozen to FP32")
                if not isinstance(root_schedule, CloudExecutionSchedule):
                    raise GraniteCloudExecutorError("rank 0 must provide a CloudExecutionSchedule")
                if root_schedule.device_count != self.world:
                    raise GraniteCloudExecutorError("schedule device count must equal distributed world size")
                tokens = _as_route_tensor(
                    root_token_ids, name="root_token_ids", dtype=torch.long, device=self.device
                )
                raw_gates = (
                    root_gate_weights
                    if isinstance(root_gate_weights, torch.Tensor)
                    else torch.as_tensor(root_gate_weights)
                )
                if raw_gates.dtype != torch.float32:
                    raise GraniteCloudExecutorError("captured Gate weights must be FP32")
                gates = _as_route_tensor(
                    root_gate_weights,
                    name="root_gate_weights",
                    dtype=root_flat_hidden.dtype,
                    device=self.device,
                )
                if tokens.numel() != root_schedule.route_count or gates.numel() != root_schedule.route_count:
                    raise GraniteCloudExecutorError(
                        "root native token/gate vectors do not match schedule route count"
                    )
                if tokens.numel() and (
                    int(tokens.min().item()) < 0
                    or int(tokens.max().item()) >= root_flat_hidden.shape[0]
                ):
                    raise GraniteCloudExecutorError("root token index is outside hidden rows")
                if len(root_schedule.counts_by_gpu_expert) != self.world:
                    raise GraniteCloudExecutorError("schedule count matrix does not match world size")
                counts = tuple(
                    _check_counts32(row, sum(row)) for row in root_schedule.counts_by_gpu_expert
                )
                if sum(map(sum, counts)) != root_schedule.route_count:
                    raise GraniteCloudExecutorError("schedule count matrix lost routes")
                route_order = root_schedule.route_indices_by_execution_order
                if len(route_order) != root_schedule.route_count or sorted(route_order) != list(
                    range(root_schedule.route_count)
                ):
                    raise GraniteCloudExecutorError("schedule route order is not a route permutation")
                meta = {
                    "error": None,
                    "counts": counts,
                    "route_count": root_schedule.route_count,
                    "hidden_size": int(root_flat_hidden.shape[1]),
                    "token_count": int(root_flat_hidden.shape[0]),
                    "dtype": str(root_flat_hidden.dtype).removeprefix("torch."),
                    "device_count": root_schedule.device_count,
                }
                self._root_tokens = tokens
                self._root_gates = gates
            except Exception as exc:
                meta = {"error": f"{type(exc).__name__}: {exc}"}
        objects = [meta]
        dist.broadcast_object_list(
            objects,
            src=self.root,
            group=self.process_group,
            device=self.device,
        )
        result = objects[0]
        if not isinstance(result, dict) or result.get("error"):
            reason = result.get("error") if isinstance(result, dict) else "malformed root metadata"
            raise GraniteCloudExecutorError(f"distributed root metadata rejected: {reason}")
        return result

    def _validate_local_weights(self, counts, hidden_size, dtype, local_weights) -> str | None:
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            if expert not in local_weights:
                return f"rank {self.rank} lacks weights for Expert {expert}"
            pair = local_weights[expert]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                return f"rank {self.rank} has malformed weights for Expert {expert}"
            weight_in, weight_out = pair
            if not isinstance(weight_in, torch.Tensor) or not isinstance(weight_out, torch.Tensor):
                return f"rank {self.rank} has non-tensor weights for Expert {expert}"
            if (
                weight_in.ndim != 2
                or weight_out.ndim != 2
                or weight_in.device != self.device
                or weight_out.device != self.device
                or weight_in.dtype != dtype
                or weight_out.dtype != dtype
                or weight_in.shape[1] != hidden_size
                or weight_in.shape[0] % 2
                or tuple(weight_out.shape) != (hidden_size, weight_in.shape[0] // 2)
            ):
                return f"rank {self.rank} has incompatible tensors for Expert {expert}"
        return None

    @torch.inference_mode()
    def execute(
        self,
        root_flat_hidden: torch.Tensor | None = None,
        root_token_ids: Sequence[int] | torch.Tensor | None = None,
        root_gate_weights: Sequence[float] | torch.Tensor | None = None,
        root_schedule: CloudExecutionSchedule | None = None,
        *,
        local_weights: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Dispatch, compute, return routes, and combine on rank 0.

        Root inputs are the native sorted-route vectors from the selected
        Granite Gate. Every process supplies only its already distributed
        same-Expert weights. Metadata handoff is inside the measured wall.
        """
        if self.rank != self.root and any(
            value is not None
            for value in (root_flat_hidden, root_token_ids, root_gate_weights, root_schedule)
        ):
            raise GraniteCloudExecutorError("only rank 0 may supply captured root tensors and schedule")
        dist.barrier(group=self.process_group)
        self._sync()
        wall_start = time.perf_counter()
        metadata_start = time.perf_counter()
        meta = self._metadata(
            root_flat_hidden,
            root_token_ids,
            root_gate_weights,
            root_schedule,
        )
        self._sync()
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
        if meta["device_count"] != self.world or len(meta["counts"]) != self.world:
            raise GraniteCloudExecutorError("broadcast execution metadata has the wrong world size")
        dtype = getattr(torch, meta["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise GraniteCloudExecutorError(f"unsupported hidden tensor dtype {meta['dtype']!r}")

        preflight_start = time.perf_counter()
        local_error = self._validate_local_weights(
            meta["counts"][self.rank],
            meta["hidden_size"],
            dtype,
            local_weights,
        )
        validation = torch.tensor(
            [1 if local_error else 0],
            dtype=torch.int32,
            device=self.device if self.backend == "nccl" else torch.device("cpu"),
        )
        dist.all_reduce(validation, op=dist.ReduceOp.MAX, group=self.process_group)
        if int(validation.item()):
            raise GraniteCloudExecutorError(
                local_error or "another rank has incompatible or missing local Expert weights"
            )
        preflight_ms = (time.perf_counter() - preflight_start) * 1000.0

        per_gpu = tuple(sum(row) for row in meta["counts"])
        hidden_size = meta["hidden_size"]
        route_count = meta["route_count"]
        dispatch_start = time.perf_counter()
        if self.rank == self.root:
            order = torch.as_tensor(
                root_schedule.route_indices_by_execution_order,
                dtype=torch.long,
                device=self.device,
            )
            if route_count:
                ordered_token_ids = self._root_tokens.index_select(0, order)
                dispatch_input = root_flat_hidden.to(self.device).index_select(0, ordered_token_ids)
            else:
                dispatch_input = torch.empty((0, hidden_size), dtype=dtype, device=self.device)
            send_splits = list(per_gpu)
            receive_splits = [per_gpu[self.rank]] + [0] * (self.world - 1)
        else:
            dispatch_input = torch.empty((0, hidden_size), dtype=dtype, device=self.device)
            send_splits = [0] * self.world
            receive_splits = [per_gpu[self.rank]] + [0] * (self.world - 1)
            order = None
        received = torch.empty((per_gpu[self.rank], hidden_size), dtype=dtype, device=self.device)
        dist.all_to_all_single(
            received,
            dispatch_input,
            output_split_sizes=receive_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )
        self._sync()
        dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0

        compute_start = time.perf_counter()
        local_output = execute_local_experts(
            received,
            meta["counts"][self.rank],
            local_weights,
        )
        self._sync()
        expert_compute_ms = (time.perf_counter() - compute_start) * 1000.0

        return_start = time.perf_counter()
        if self.rank == self.root:
            return_receive_splits = list(per_gpu)
            return_send_splits = [per_gpu[self.root]] + [0] * (self.world - 1)
            returned = torch.empty((route_count, hidden_size), dtype=dtype, device=self.device)
        else:
            return_receive_splits = [0] * self.world
            return_send_splits = [per_gpu[self.rank]] + [0] * (self.world - 1)
            returned = torch.empty((0, hidden_size), dtype=dtype, device=self.device)
        dist.all_to_all_single(
            returned,
            local_output,
            output_split_sizes=return_receive_splits,
            input_split_sizes=return_send_splits,
            group=self.process_group,
        )
        self._sync()
        return_ms = (time.perf_counter() - return_start) * 1000.0

        combine_start = time.perf_counter()
        output = None
        if self.rank == self.root:
            if route_count:
                native_outputs = torch.empty_like(returned)
                native_outputs.index_copy_(0, order, returned)
            else:
                native_outputs = returned
            output = _weighted_native_scatter(
                root_flat_hidden.to(self.device),
                self._root_tokens,
                self._root_gates,
                native_outputs,
            )
        self._sync()
        combine_ms = (time.perf_counter() - combine_start) * 1000.0
        # Empty workers still wait here; F ends only when every rank/device is ready.
        dist.barrier(group=self.process_group)
        self._sync()
        output_ready_perf_counter_s = time.perf_counter()
        f_wall_ms = (output_ready_perf_counter_s - wall_start) * 1000.0

        diagnostic_start = time.perf_counter()
        local_stages = torch.tensor(
            [metadata_ms, preflight_ms, dispatch_ms, expert_compute_ms, return_ms, combine_ms],
            dtype=torch.float64,
            device=self.device if self.backend == "nccl" else torch.device("cpu"),
        )
        gathered = [torch.empty_like(local_stages) for _ in range(self.world)]
        dist.all_gather(gathered, local_stages, group=self.process_group)
        per_rank = [row.cpu().tolist() for row in gathered]
        post_output_diagnostic_ms = (time.perf_counter() - diagnostic_start) * 1000.0
        stage_names = (
            "metadata_broadcast_ms",
            "preflight_validation_ms",
            "dispatch_ms",
            "expert_compute_ms",
            "return_ms",
            "combine_ms",
        )
        profile: dict[str, Any] = {
            "execution_mode": (
                "distributed_one_rank_per_gpu"
                if self.backend == "nccl"
                else "distributed_cpu_gloo_functional_control"
            ),
            "timing_scope": (
                "measured one-rank-per-GPU distributed layer call"
                if self.backend == "nccl"
                else "CPU/Gloo numerical control; not GPU performance"
            ),
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world,
            "route_count": route_count,
            "device_count": meta["device_count"],
            "routes_by_gpu": list(per_gpu),
            "F_wall_ms": f_wall_ms,
            "output_ready_elapsed_ms": f_wall_ms,
            "output_ready_perf_counter_s": output_ready_perf_counter_s,
            "post_output_diagnostic_ms": post_output_diagnostic_ms,
            "per_rank_stage_ms": [
                {name: float(value) for name, value in zip(stage_names, values)}
                for values in per_rank
            ],
            "critical_path_stage_ms": {
                name: max(float(values[index]) for values in per_rank)
                for index, name in enumerate(stage_names)
            },
            "stages_are_not_additive_wall_estimate": True,
        }
        return output, profile


def _weights_from_source(root_moe_or_weights, expert: int):
    if isinstance(root_moe_or_weights, Mapping):
        pair = root_moe_or_weights[expert]
        return pair[0], pair[1]
    try:
        return (
            root_moe_or_weights.input_linear.weight[expert],
            root_moe_or_weights.output_linear.weight[expert],
        )
    except (AttributeError, IndexError, KeyError) as exc:
        raise GraniteCloudExecutorError(
            "rank 0 weights must be a Granite MoE layer or Expert-to-(W1,W2) mapping"
        ) from exc


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().view(torch.uint8).cpu()
    payload = (
        ctypes.string_at(contiguous.data_ptr(), contiguous.numel())
        if contiguous.numel()
        else b""
    )
    shape_and_type = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
    return hashlib.sha256(shape_and_type + payload).hexdigest()


def _layout_owner_experts(layout: RingLayout, rank: int) -> tuple[int, ...]:
    owned: list[int] = []
    for expert in range(EXPERT_COUNT):
        owner_ranks = tuple(
            _logical_gpu_index(place.device, layout)
            for place in layout.placements_by_expert[expert]
        )
        if rank in owner_ranks:
            owned.append(expert)
    return tuple(owned)


def distribute_replica_weights(
    root_moe_or_weights,
    layout: RingLayout,
    *,
    devices: str | Sequence[str],
    backend: str,
    rank: int,
    world: int,
    process_group=None,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, dict[str, Any]]]:
    """Broadcast FP32 weights and retain only each rank's same-Expert replicas.

    Rank 0 may pass the full captured Granite MoE module or an
    Expert-to-(W1,W2) mapping. Other ranks pass None. Each Expert's two tensors
    are broadcast once through temporary per-rank buffers; only ranks listed in
    the common placement retain clones. SHA256 exchange verifies that copies
    of each Expert match. Call this during setup, outside repeated layer timing.
    """
    _require_distributed(rank, world, backend)
    if not isinstance(layout, RingLayout) or layout.device_count != world:
        raise GraniteCloudExecutorError("weight layout device count must equal distributed world size")
    if isinstance(devices, str):
        selected = devices
    else:
        device_list = tuple(str(item) for item in devices)
        if len(device_list) == 1:
            selected = device_list[0]
        elif len(device_list) == world:
            selected = device_list[rank]
        else:
            raise GraniteCloudExecutorError(
                "devices must be one process-local device or one device per rank"
            )
    device = torch.device(selected)
    if str(backend).lower() == "nccl" and device.type != "cuda":
        raise GraniteCloudExecutorError("NCCL weight broadcast requires CUDA devices")
    if str(backend).lower() == "nccl":
        torch.cuda.set_device(device)
    owned_experts = set(_layout_owner_experts(layout, rank))
    local_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    identities: dict[int, dict[str, Any]] = {}

    header = [None]
    if rank == 0:
        try:
            first_in, first_out = _weights_from_source(root_moe_or_weights, 0)
            if first_in.dtype != torch.float32 or first_out.dtype != torch.float32:
                raise GraniteCloudExecutorError("cloud preparation freezes Expert weights to FP32")
            header[0] = {
                "error": None,
                "input_shape": tuple(first_in.shape),
                "output_shape": tuple(first_out.shape),
            }
        except Exception as exc:
            header[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(header, src=0, group=process_group, device=device)
    if not isinstance(header[0], dict) or header[0].get("error"):
        reason = header[0].get("error") if isinstance(header[0], dict) else "invalid weight header"
        raise GraniteCloudExecutorError(f"distributed weight header rejected: {reason}")
    input_shape = tuple(header[0]["input_shape"])
    output_shape = tuple(header[0]["output_shape"])
    if len(input_shape) != 2 or len(output_shape) != 2:
        raise GraniteCloudExecutorError("packed input/output Expert weights must be matrices")

    for expert in range(EXPERT_COUNT):
        input_buffer = torch.empty(input_shape, dtype=torch.float32, device=device)
        output_buffer = torch.empty(output_shape, dtype=torch.float32, device=device)
        root_error = None
        if rank == 0:
            try:
                source_in, source_out = _weights_from_source(root_moe_or_weights, expert)
                if (
                    tuple(source_in.shape) != input_shape
                    or tuple(source_out.shape) != output_shape
                    or source_in.dtype != torch.float32
                    or source_out.dtype != torch.float32
                ):
                    raise GraniteCloudExecutorError(
                        f"Expert {expert} differs from the frozen FP32 packed shape"
                    )
                input_buffer.copy_(source_in.detach().to(device=device))
                output_buffer.copy_(source_out.detach().to(device=device))
            except Exception as exc:
                root_error = f"{type(exc).__name__}: {exc}"
        errors = [root_error]
        dist.broadcast_object_list(errors, src=0, group=process_group, device=device)
        if errors[0]:
            raise GraniteCloudExecutorError(f"weight source rejected for Expert {expert}: {errors[0]}")
        dist.broadcast(input_buffer, src=0, group=process_group)
        dist.broadcast(output_buffer, src=0, group=process_group)
        if expert in owned_experts:
            input_copy = input_buffer.clone(memory_format=torch.contiguous_format)
            output_copy = output_buffer.clone(memory_format=torch.contiguous_format)
            local_weights[expert] = (input_copy, output_copy)
            identities[expert] = {
                "input_sha256": _tensor_sha256(input_copy),
                "output_sha256": _tensor_sha256(output_copy),
                "device": str(device),
                "semantics": "independent_rank_local_copy",
            }
        del input_buffer, output_buffer

    all_identities: list[Any] = [None] * world
    dist.all_gather_object(
        all_identities,
        identities,
        group=process_group,
    )
    copies_by_expert: dict[int, list[tuple[str, str]]] = {
        expert: [] for expert in range(EXPERT_COUNT)
    }
    for rank_rows in all_identities:
        for expert_text, row in rank_rows.items():
            copies_by_expert[int(expert_text)].append(
                (row["input_sha256"], row["output_sha256"])
            )
    for expert, copies in copies_by_expert.items():
        expected_owners = len(layout.placements_by_expert[expert])
        if len(copies) != expected_owners or len(set(copies)) != 1:
            raise GraniteCloudExecutorError(
                f"same-Expert replica weight identity failed for Expert {expert}"
            )
    return local_weights, identities


__all__ = [
    "CloudExecutionSchedule",
    "DistributedExecutor",
    "EXPERT_COUNT",
    "GraniteCloudExecutorError",
    "POLICY_NAMES",
    "PolicyPlan",
    "SUPPORTED_DEVICE_COUNTS",
    "build_execution_schedule",
    "distribute_replica_weights",
    "execute_local_experts",
    "loopback_execute",
    "make_ring_placements",
    "plan_policy",
]
