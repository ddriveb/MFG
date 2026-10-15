"""Trainable distributed Granite MoE Expert-replica routing backend.

The adapter leaves Granite's native Router in the autograd graph, replaces each
packed full Expert table with only the adjacent-ring Experts resident on this
rank, and runs dispatch/return through autograd-aware all-to-all collectives.
Planning and route scheduling are injected native interfaces; this module has no
Python planning or scheduling fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional model runtime
    raise ImportError("Granite trainable replica routing requires PyTorch") from exc


EXPERT_COUNT = 32
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
_TIMING_KEYS = (
    "planner_ms",
    "schedule_ms",
    "dispatch_ms",
    "return_ms",
    "expert_compute_ms",
    "combine_ms",
    "dispatch_backward_ms",
    "return_backward_ms",
    "dense_sync_ms",
    "expert_sync_ms",
    "routed_moe_calls",
    "routes",
    "routes_received",
)


class GraniteTrainingReplicaError(RuntimeError):
    """Invalid runtime configuration or an unsupported Granite ABI."""


@dataclass(frozen=True)
class _NativeScheduleView:
    """The small portion of the native schedule consumed by the adapter."""

    route_indices_by_execution_order: tuple[int, ...]
    counts_by_gpu_expert: tuple[tuple[int, ...], ...]
    route_count: int


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ring_layout(world_size: int):
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise GraniteTrainingReplicaError(
            f"native training ring requires world size in {SUPPORTED_WORLD_SIZES}; got {world_size}"
        )
    from .granite_cloud_executor_v1 import make_ring_placements
    from .granite_structured_solution_map_v1 import prepare_ring_layout

    # The frozen schedule ABI accepts only 4/6/8 logical GPUs. A two-rank
    # Gloo test embeds its adjacent 2-cycle in the 4-cycle, then folds logical
    # destinations 0/2 and 1/3 onto physical ranks 0/1 in _schedule().
    logical_world_size = 4 if world_size == 2 else world_size
    placements = make_ring_placements(logical_world_size)
    rates = {f"cuda:{rank}": 10.0 for rank in range(logical_world_size)}
    return prepare_ring_layout(placements, rates)


def _native_schedule(
    schedule_library: Any,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    route_slots: torch.Tensor,
    layout: Any,
    quotas_by_expert: Mapping[int, Sequence[int]],
):
    """Call the frozen native schedule wrapper or an explicitly supplied adapter.

    Production callers pass the compiled schedule-library path. The callable
    form is useful for independent test adapters; it is explicit dependency
    injection and is never selected as a fallback.
    """
    if callable(schedule_library):
        value = schedule_library(
            expert_ids, token_ids, route_slots, layout, quotas_by_expert
        )
    else:
        if schedule_library is None:
            raise GraniteTrainingReplicaError(
                "schedule_library is required; native schedule fallback is disabled"
            )
        from .granite_cloud_schedule_native_v1 import build_native_execution_schedule

        value = build_native_execution_schedule(
            expert_ids,
            token_ids,
            route_slots,
            layout,
            quotas_by_expert,
            Path(schedule_library),
        )
    required = ("route_indices_by_execution_order", "counts_by_gpu_expert", "route_count")
    if any(not hasattr(value, name) for name in required):
        raise GraniteTrainingReplicaError("native schedule result is missing required fields")
    return value


def _autograd_all_to_all(
    tensor: torch.Tensor,
    send_splits: Sequence[int],
    recv_splits: Sequence[int],
    group: Any,
    handle: "TrainingRoutingHandle",
    timing_key: str,
) -> torch.Tensor:
    class _AllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, send, recv, process_group, owner, key):
            ctx.send_splits = tuple(int(item) for item in send)
            ctx.recv_splits = tuple(int(item) for item in recv)
            ctx.group = process_group
            ctx.owner = owner
            ctx.key = key
            result = value.new_empty((sum(ctx.recv_splits), *value.shape[1:]))
            started = time.perf_counter()
            dist.all_to_all_single(
                result,
                value.contiguous(),
                output_split_sizes=list(ctx.recv_splits),
                input_split_sizes=list(ctx.send_splits),
                group=process_group,
            )
            owner._add_time(key, (time.perf_counter() - started) * 1000.0)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.new_empty((sum(ctx.send_splits), *grad_output.shape[1:]))
            started = time.perf_counter()
            dist.all_to_all_single(
                grad_input,
                grad_output.contiguous(),
                output_split_sizes=list(ctx.send_splits),
                input_split_sizes=list(ctx.recv_splits),
                group=ctx.group,
            )
            ctx.owner._add_time(
                "dispatch_backward_ms" if ctx.key == "dispatch_ms" else "return_backward_ms",
                (time.perf_counter() - started) * 1000.0,
            )
            return grad_input, None, None, None, None, None

    return _AllToAll.apply(
        tensor,
        tuple(int(item) for item in send_splits),
        tuple(int(item) for item in recv_splits),
        group,
        handle,
        timing_key,
    )


def _all_to_all_metadata(
    tensor: torch.Tensor,
    send_splits: Sequence[int],
    recv_splits: Sequence[int],
    group: Any,
) -> torch.Tensor:
    result = tensor.new_empty((sum(int(item) for item in recv_splits), *tensor.shape[1:]))
    dist.all_to_all_single(
        result,
        tensor.contiguous(),
        output_split_sizes=[int(item) for item in recv_splits],
        input_split_sizes=[int(item) for item in send_splits],
        group=group,
    )
    return result


def _copy_quota_plan(planner: Any, counts: Sequence[int], world_size: int):
    """Call the native planner and normalize its left-replica quotas."""
    if planner is None or not callable(getattr(planner, "plan", None)):
        raise GraniteTrainingReplicaError("a native training planner with plan(counts, world_size) is required")
    started = time.perf_counter()
    result = planner.plan(tuple(int(value) for value in counts), world_size)
    elapsed = (time.perf_counter() - started) * 1000.0
    if not hasattr(result, "quota_left"):
        raise GraniteTrainingReplicaError("native planner result must expose quota_left")
    left = tuple(result.quota_left)
    if len(left) != EXPERT_COUNT:
        raise GraniteTrainingReplicaError(
            f"native planner returned {len(left)} left quotas; expected {EXPERT_COUNT}"
        )
    for expert, (value, count) in enumerate(zip(left, counts)):
        if type(value) is not int or not 0 <= value <= int(count):
            raise GraniteTrainingReplicaError(
                f"native quota_left[{expert}]={value!r} is outside [0, {count}]"
            )
    return left, elapsed, result


class _RoutedGraniteMoE(nn.Module):
    """Granite MoE with native Gate and only this rank's resident Experts."""

    def __init__(
        self,
        original: nn.Module,
        handle: "TrainingRoutingHandle",
        layer_name: str,
    ) -> None:
        super().__init__()
        if not all(hasattr(original, name) for name in ("router", "input_linear", "output_linear", "activation")):
            raise GraniteTrainingReplicaError(f"unsupported Granite MoE at {layer_name!r}")
        input_weight = original.input_linear.weight
        output_weight = original.output_linear.weight
        if input_weight.ndim != 3 or output_weight.ndim != 3:
            raise GraniteTrainingReplicaError("Granite Experts must use packed rank-3 weights")
        if input_weight.shape[0] != EXPERT_COUNT or output_weight.shape[0] != EXPERT_COUNT:
            raise GraniteTrainingReplicaError(
                f"training ring expects {EXPERT_COUNT} Experts, got {input_weight.shape[0]}"
            )
        self.input_size = int(original.input_size)
        self.hidden_size = int(original.hidden_size)
        self.activation = original.activation
        self.router = original.router
        self.handle = handle
        self.layer_name = layer_name
        self.top_k = int(self.router.top_k)
        if self.top_k < 1:
            raise GraniteTrainingReplicaError("Granite router top_k must be positive")
        local_ids = handle.resident_experts
        self.local_expert_w1 = nn.ParameterDict()
        self.local_expert_w2 = nn.ParameterDict()
        for expert in local_ids:
            if handle.loopback:
                # Parameter(Tensor-view) keeps the original packed checkpoint storage.
                w1 = input_weight[expert]
                w2 = output_weight[expert]
            else:
                w1 = input_weight[expert].detach().clone()
                w2 = output_weight[expert].detach().clone()
            self.local_expert_w1[str(expert)] = nn.Parameter(
                w1, requires_grad=input_weight.requires_grad
            )
            self.local_expert_w2[str(expert)] = nn.Parameter(
                w2, requires_grad=output_weight.requires_grad
            )
            handle._register_expert_parameter(expert, self.local_expert_w1[str(expert)])
            handle._register_expert_parameter(expert, self.local_expert_w2[str(expert)])

        self._handle_moe = handle
        self._original_dtype = input_weight.dtype
        self._original_device = input_weight.device
        # original's full packed Expert tables become unreachable when its parent
        # replaces it with this wrapper.

    def _expert_mlp(self, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        result = inputs * 0
        for expert in self.handle.resident_experts:
            indices = torch.nonzero(expert_ids == expert, as_tuple=False).flatten()
            if not indices.numel():
                continue
            expert_inputs = inputs.index_select(0, indices)
            projected = F.linear(expert_inputs, self.local_expert_w1[str(expert)])
            gate, value = projected.chunk(2, dim=-1)
            activated = self.activation(gate) * value
            expert_outputs = F.linear(activated, self.local_expert_w2[str(expert)])
            result = result.index_copy(0, indices, expert_outputs)
        if expert_ids.numel() and bool(torch.any(~torch.isin(
            expert_ids,
            torch.as_tensor(self.handle.resident_experts, device=expert_ids.device, dtype=expert_ids.dtype),
        ))):
            missing = sorted(set(expert_ids.detach().cpu().tolist()) - set(self.handle.resident_experts))
            raise GraniteTrainingReplicaError(
                f"rank {self.handle.rank} received nonresident Experts {missing}"
            )
        return result

    def forward(self, layer_input: torch.Tensor):
        if layer_input.ndim != 3:
            raise GraniteTrainingReplicaError(
                f"{self.layer_name} expected [batch, sequence, hidden], got {tuple(layer_input.shape)}"
            )
        batch_size, sequence_length, emb_size = layer_input.shape
        if emb_size != self.input_size:
            raise GraniteTrainingReplicaError("Granite MoE hidden width changed after installation")
        flat = layer_input.reshape(-1, emb_size)
        route_permutation, token_indices, gate_weights, expert_sizes, router_logits = self.router(flat)
        if len(expert_sizes) != EXPERT_COUNT:
            raise GraniteTrainingReplicaError("Granite router Expert-count width changed")
        device = flat.device
        self.handle._check_equal_source_tokens(int(flat.shape[0]), device)
        local_counts = tuple(int(value) for value in expert_sizes)
        count_tensor = torch.tensor(local_counts, dtype=torch.int64, device=device)
        rank_counts = self.handle._gather_counts(count_tensor)
        global_counts = tuple(sum(row[expert] for row in rank_counts) for expert in range(EXPERT_COUNT))
        if self.handle.loopback:
            prefix = (0,) * EXPERT_COUNT
        else:
            prefix = tuple(
                sum(rank_counts[source][expert] for source in range(self.handle.rank))
                for expert in range(EXPERT_COUNT)
            )
        quota_left, planner_ms, planner_result = self.handle._global_quotas(global_counts)
        self.handle._add_time("planner_ms", planner_ms)
        local_quotas: dict[int, tuple[int, int]] = {}
        for expert in range(EXPERT_COUNT):
            local_count = local_counts[expert]
            left = max(0, min(local_count, quota_left[expert] - prefix[expert]))
            local_quotas[expert] = (left, local_count - left)
        expert_ids = torch.repeat_interleave(
            torch.arange(EXPERT_COUNT, dtype=torch.int64, device=device),
            torch.as_tensor(local_counts, dtype=torch.int64, device=device),
        )
        route_slots = route_permutation.remainder(self.top_k)

        # Route schedule creation consumes only discrete Gate structure. Gate
        # weights and all hidden-state values stay attached to the autograd graph.
        started = time.perf_counter()
        schedule = self.handle._schedule(expert_ids, token_indices, route_slots, local_quotas)
        self.handle._add_time("schedule_ms", (time.perf_counter() - started) * 1000.0)
        if int(schedule.route_count) != int(token_indices.numel()):
            raise GraniteTrainingReplicaError("native schedule did not conserve every Gate route")
        order = torch.as_tensor(
            schedule.route_indices_by_execution_order, dtype=torch.int64, device=device
        )
        ordered_tokens = token_indices.index_select(0, order)
        ordered_experts = expert_ids.index_select(0, order)
        send_splits = tuple(
            sum(int(value) for value in schedule.counts_by_gpu_expert[gpu])
            for gpu in range(self.handle.world_size)
        )
        if sum(send_splits) != int(order.numel()):
            raise GraniteTrainingReplicaError("native schedule send splits do not conserve routes")

        if self.handle.loopback:
            ordered_inputs = flat.index_select(0, ordered_tokens)
            start = time.perf_counter()
            ordered_outputs = self._expert_mlp(ordered_inputs, ordered_experts)
            self.handle._add_time("expert_compute_ms", (time.perf_counter() - start) * 1000.0)
            returned = ordered_outputs
        else:
            recv_splits = self.handle._exchange_splits(send_splits, device)
            self.handle._add_time("routes_received", sum(recv_splits))
            ordered_inputs = flat.index_select(0, ordered_tokens)
            received_inputs = _autograd_all_to_all(
                ordered_inputs,
                send_splits,
                recv_splits,
                self.handle.world_group,
                self.handle,
                "dispatch_ms",
            )
            received_experts = _all_to_all_metadata(
                ordered_experts,
                send_splits,
                recv_splits,
                self.handle.world_group,
            )
            start = time.perf_counter()
            received_outputs = self._expert_mlp(received_inputs, received_experts)
            self.handle._add_time("expert_compute_ms", (time.perf_counter() - start) * 1000.0)
            returned = _autograd_all_to_all(
                received_outputs,
                recv_splits,
                send_splits,
                self.handle.world_group,
                self.handle,
                "return_ms",
            )

        start = time.perf_counter()
        route_outputs = flat.new_zeros((token_indices.numel(), self.input_size)).index_copy(
            0, order, returned
        )
        weighted = route_outputs * gate_weights.to(route_outputs.dtype).unsqueeze(-1)
        layer_output = flat.new_zeros((flat.shape[0], self.input_size)).index_add(
            0, token_indices, weighted
        )
        layer_output = layer_output.view(batch_size, sequence_length, self.input_size)
        self.handle._add_time("combine_ms", (time.perf_counter() - start) * 1000.0)
        self.handle._add_time("routed_moe_calls", 1)
        self.handle._add_time("routes", int(token_indices.numel()))
        if planner_result is not None:
            self.handle._record_planner_result(planner_result)
        return layer_output, router_logits


class TrainingRoutingHandle:
    """Installed backend controls, categorized parameters, and diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        planner: Any,
        schedule_library: Any,
        world_size: int,
        rank: int,
        *,
        loopback: bool,
    ) -> None:
        if type(world_size) is not int or world_size not in SUPPORTED_WORLD_SIZES:
            raise GraniteTrainingReplicaError(
                f"world_size must be one of {SUPPORTED_WORLD_SIZES}"
            )
        if type(rank) is not int or not 0 <= rank < world_size:
            raise GraniteTrainingReplicaError("rank must be in [0, world_size)")
        if loopback:
            if _distributed_ready():
                raise GraniteTrainingReplicaError("loopback mode cannot run inside a process group")
        else:
            if not _distributed_ready():
                raise GraniteTrainingReplicaError("distributed training requires an initialized process group")
            if dist.get_world_size() != world_size or dist.get_rank() != rank:
                raise GraniteTrainingReplicaError("declared rank/world_size do not match torch.distributed")
        if planner is None or not callable(getattr(planner, "plan", None)):
            raise GraniteTrainingReplicaError("planner with native plan(counts, world_size) is required")
        if schedule_library is None:
            raise GraniteTrainingReplicaError("schedule_library is required; native fallback is disabled")
        self.model = model
        self.planner = planner
        self.schedule_library = schedule_library
        self.world_size = world_size
        self.rank = rank
        self.loopback = loopback
        self.world_group = dist.group.WORLD if not loopback else None
        self.layout = _ring_layout(world_size)
        self.schedule_world_size = int(self.layout.device_count)
        self.resident_experts = tuple(
            range(EXPERT_COUNT)
            if loopback
            else sorted(
                expert
                for expert in range(EXPERT_COUNT)
                if rank in (expert % world_size, (expert + 1) % world_size)
            )
        )
        self._expert_parameters: dict[int, list[nn.Parameter]] = {
            expert: [] for expert in range(EXPERT_COUNT)
        }
        self._timings = {key: 0.0 for key in _TIMING_KEYS}
        self._planner_results: list[Any] = []
        self._replica_groups: dict[int, Any] = {}
        self._native_layouts = 0
        self._closed = False
        if not loopback:
            # Every process creates every pair group in the same global order.
            for expert in range(EXPERT_COUNT):
                ranks = [expert % world_size, (expert + 1) % world_size]
                self._replica_groups[expert] = dist.new_group(ranks=ranks)
        self._install_modules()

    def _register_expert_parameter(self, expert: int, parameter: nn.Parameter) -> None:
        self._expert_parameters[expert].append(parameter)

    def _install_modules(self) -> None:
        targets = [
            (name, module)
            for name, module in self.model.named_modules()
            if name and all(
                hasattr(module, key)
                for key in ("router", "input_linear", "output_linear", "activation")
            )
            and type(module).__name__ == "GraniteMoeMoE"
        ]
        if not targets:
            raise GraniteTrainingReplicaError("model contains no supported GraniteMoeMoE modules")
        for name, original in targets:
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            setattr(parent, child_name, _RoutedGraniteMoE(original, self, name))
        self.routed_layer_names = tuple(name for name, _ in targets)
        expert_ids = {id(parameter) for rows in self._expert_parameters.values() for parameter in rows}
        self.dense_router_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in expert_ids
        )
        self.expert_parameters = tuple(
            parameter
            for expert in range(EXPERT_COUNT)
            for parameter in self._expert_parameters[expert]
            if parameter.requires_grad
        )
        self.owned_parameters = self.dense_router_parameters + self.expert_parameters
        # Every rank must own only the two ring-neighbor Expert parameter sets.
        expected_per_moe = len(self.resident_experts) * 2
        for name in self.routed_layer_names:
            local_wrapper = self.model.get_submodule(name)
            if len(local_wrapper.local_expert_w1) + len(local_wrapper.local_expert_w2) != expected_per_moe:
                raise GraniteTrainingReplicaError(f"rank {self.rank} Expert residency is incomplete at {name}")

    def _gather_counts(self, local: torch.Tensor) -> tuple[tuple[int, ...], ...]:
        if local.shape != (EXPERT_COUNT,):
            raise GraniteTrainingReplicaError("local Gate counts must contain 32 entries")
        if self.loopback:
            return (tuple(int(value) for value in local.detach().cpu().tolist()),)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local.contiguous(), group=self.world_group)
        return tuple(tuple(int(value) for value in row.detach().cpu().tolist()) for row in gathered)

    def _check_equal_source_tokens(self, local_tokens: int, device: torch.device) -> None:
        if self.loopback:
            return
        token_count = torch.tensor([local_tokens], dtype=torch.int64, device=device)
        gathered = [torch.empty_like(token_count) for _ in range(self.world_size)]
        dist.all_gather(gathered, token_count, group=self.world_group)
        values = [int(item.item()) for item in gathered]
        if len(set(values)) != 1:
            raise GraniteTrainingReplicaError(
                f"source ranks must use equal Token counts for mean-loss synchronization; got {values}"
            )

    def _global_quotas(self, counts: Sequence[int]):
        if self.loopback:
            return _copy_quota_plan(self.planner, counts, self.world_size)
        count_tensor = torch.tensor(tuple(counts), dtype=torch.int64, device=self._collective_device())
        quota = torch.zeros((EXPERT_COUNT,), dtype=torch.int64, device=count_tensor.device)
        planner_ms = 0.0
        result = None
        failure = None
        planner_ok = torch.ones((1,), dtype=torch.int32, device=count_tensor.device)
        if self.rank == 0:
            try:
                left, planner_ms, result = _copy_quota_plan(
                    self.planner, tuple(int(value) for value in counts), self.world_size
                )
                quota.copy_(torch.as_tensor(left, dtype=torch.int64, device=quota.device))
            except Exception as exc:
                planner_ok.zero_()
                failure = f"{type(exc).__name__}: {exc}"
        # Every rank learns about a planner error before anyone waits for quota
        # broadcast, avoiding a rank-0 exception plus timeout on the others.
        dist.broadcast(planner_ok, src=0, group=self.world_group)
        if not int(planner_ok.item()):
            detail = failure if self.rank == 0 else "see rank 0 planner diagnostic"
            raise GraniteTrainingReplicaError(f"rank 0 native planner failed: {detail}")
        dist.broadcast(quota, src=0, group=self.world_group)
        left = tuple(int(value) for value in quota.detach().cpu().tolist())
        for expert, (value, count) in enumerate(zip(left, counts)):
            if not 0 <= value <= int(count):
                raise GraniteTrainingReplicaError(
                    f"broadcast native quota for Expert {expert} is outside its global count"
                )
        return left, planner_ms, result

    def _collective_device(self):
        if dist.get_backend(self.world_group) == "nccl":
            for parameter in self.model.parameters():
                return parameter.device
            raise GraniteTrainingReplicaError("NCCL model has no parameters to select a device")
        return torch.device("cpu")

    def _schedule(self, expert_ids, token_ids, route_slots, quotas_by_expert):
        columns = tuple(
            tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()
            for tensor in (expert_ids, token_ids, route_slots)
        )
        schedule = None
        failure = None
        try:
            raw = _native_schedule(
                self.schedule_library,
                *columns,
                self.layout,
                quotas_by_expert,
            )
            logical_counts = tuple(tuple(int(value) for value in row) for row in raw.counts_by_gpu_expert)
            logical_order = tuple(int(value) for value in raw.route_indices_by_execution_order)
            if len(logical_counts) != self.schedule_world_size or any(
                len(row) != EXPERT_COUNT for row in logical_counts
            ):
                raise GraniteTrainingReplicaError("native schedule returned an invalid GPU/Expert count matrix")
            if int(raw.route_count) != int(columns[0].numel()) or len(logical_order) != int(raw.route_count):
                raise GraniteTrainingReplicaError("native schedule did not conserve every source route")
            if sum(sum(row) for row in logical_counts) != int(raw.route_count):
                raise GraniteTrainingReplicaError("native schedule count matrix does not conserve routes")
            if self.schedule_world_size == self.world_size:
                schedule = raw
            else:
                # Fold logical destination rows into the physical Gloo ranks
                # while preserving the C builder's Expert order within each row.
                row_groups: list[list[tuple[int, ...]]] = [
                    [tuple() for _ in range(EXPERT_COUNT)] for _ in range(self.schedule_world_size)
                ]
                cursor = 0
                for logical_rank in range(self.schedule_world_size):
                    for expert in range(EXPERT_COUNT):
                        count = logical_counts[logical_rank][expert]
                        row_groups[logical_rank][expert] = logical_order[cursor : cursor + count]
                        cursor += count
                if cursor != len(logical_order):
                    raise GraniteTrainingReplicaError(
                        "native schedule count matrix does not match its route order"
                    )
                physical_counts = [[0] * EXPERT_COUNT for _ in range(self.world_size)]
                physical_order: list[int] = []
                for physical_rank in range(self.world_size):
                    for logical_rank in range(physical_rank, self.schedule_world_size, self.world_size):
                        for expert in range(EXPERT_COUNT):
                            routes = row_groups[logical_rank][expert]
                            physical_counts[physical_rank][expert] += len(routes)
                            physical_order.extend(routes)
                schedule = _NativeScheduleView(
                    route_indices_by_execution_order=tuple(physical_order),
                    counts_by_gpu_expert=tuple(tuple(row) for row in physical_counts),
                    route_count=int(raw.route_count),
                )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        if not self.loopback:
            schedule_ok = torch.tensor(
                [int(failure is None)], dtype=torch.int32, device=self._collective_device()
            )
            dist.all_reduce(schedule_ok, op=dist.ReduceOp.MIN, group=self.world_group)
            if not int(schedule_ok.item()):
                messages: list[Any] = [None] * self.world_size
                dist.all_gather_object(messages, failure, group=self.world_group)
                first = next((item for item in messages if item), "native scheduler failed on a source rank")
                raise GraniteTrainingReplicaError(f"native schedule failed: {first}")
        elif failure is not None:
            raise GraniteTrainingReplicaError(f"native schedule failed: {failure}")
        assert schedule is not None
        self._native_layouts += 1
        return schedule

    def _exchange_splits(self, send_splits: Sequence[int], device: torch.device):
        if len(send_splits) != self.world_size or any(int(value) < 0 for value in send_splits):
            raise GraniteTrainingReplicaError("native schedule returned invalid all-to-all splits")
        values = torch.tensor(send_splits, dtype=torch.int64, device=device)
        incoming = torch.empty_like(values)
        dist.all_to_all_single(incoming, values, group=self.world_group)
        return tuple(int(value) for value in incoming.detach().cpu().tolist())

    def _add_time(self, key: str, value: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + float(value)

    def _record_planner_result(self, result: Any) -> None:
        if self.rank == 0 or self.loopback:
            self._planner_results.append(result)

    @property
    def dense_parameters(self) -> tuple[nn.Parameter, ...]:
        """Trainable dense and Router parameters, in stable model order."""
        return self.dense_router_parameters

    @property
    def local_expert_parameters(self) -> tuple[nn.Parameter, ...]:
        """Only trainable Expert Parameters resident on this rank."""
        return self.expert_parameters

    @property
    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters to pass to AdamW; no cold Expert weights are registered."""
        return self.owned_parameters

    def sync_gradients(self) -> None:
        """Apply the canonical global-mean gradient scaling once per optimizer step.

        Dense/Router gradients use world SUM/world_size. Each Expert pair uses
        pair SUM/world_size, intentionally not pair averaging. Missing Expert
        gradients become explicit zero contributions so collective order is fixed.
        """
        if self._closed:
            raise GraniteTrainingReplicaError("training routing handle is closed")
        if self.loopback:
            for parameter in self.expert_parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            return
        started = time.perf_counter()
        for parameter in self.dense_router_parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=self.world_group)
            parameter.grad.div_(self.world_size)
        self._add_time("dense_sync_ms", (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        for expert in range(EXPERT_COUNT):
            ranks = (expert % self.world_size, (expert + 1) % self.world_size)
            if self.rank not in ranks:
                continue
            for parameter in self._expert_parameters[expert]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(
                    parameter.grad,
                    op=dist.ReduceOp.SUM,
                    group=self._replica_groups[expert],
                )
                parameter.grad.div_(self.world_size)
        self._add_time("expert_sync_ms", (time.perf_counter() - started) * 1000.0)

    @torch.no_grad()
    def check_replica_identity(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        raise_on_failure: bool = True,
    ) -> bool:
        """Collectively verify Expert weights and optional optimizer states."""
        if self._closed:
            raise GraniteTrainingReplicaError("training routing handle is closed")
        if self.loopback:
            return True
        local_ok = torch.ones((), dtype=torch.int32, device=self._collective_device())
        for expert in range(EXPERT_COUNT):
            left_rank = expert % self.world_size
            if self.rank not in (left_rank, (expert + 1) % self.world_size):
                continue
            for parameter in self._expert_parameters[expert]:
                reference = torch.empty_like(parameter)
                if self.rank == left_rank:
                    reference.copy_(parameter)
                dist.broadcast(reference, src=left_rank, group=self._replica_groups[expert])
                if not torch.equal(reference, parameter):
                    local_ok.zero_()
                if optimizer is None:
                    continue
                state = optimizer.state.get(parameter, {})
                gathered_keys: list[Any] = [None, None]
                dist.all_gather_object(
                    gathered_keys,
                    tuple(sorted(state)),
                    group=self._replica_groups[expert],
                )
                if gathered_keys[0] != gathered_keys[1]:
                    local_ok.zero_()
                    continue
                for key in gathered_keys[0]:
                    value = state[key]
                    if torch.is_tensor(value):
                        value_on_collective_device = value.to(parameter.device)
                        state_reference = torch.empty_like(value_on_collective_device)
                        if self.rank == left_rank:
                            state_reference.copy_(value_on_collective_device)
                        dist.broadcast(
                            state_reference,
                            src=left_rank,
                            group=self._replica_groups[expert],
                        )
                        if not torch.equal(state_reference, value_on_collective_device):
                            local_ok.zero_()
                    else:
                        state_values: list[Any] = [None, None]
                        dist.all_gather_object(
                            state_values,
                            value,
                            group=self._replica_groups[expert],
                        )
                        if state_values[0] != state_values[1]:
                            local_ok.zero_()
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN, group=self.world_group)
        passed = bool(local_ok.item())
        if raise_on_failure and not passed:
            raise GraniteTrainingReplicaError("same-Expert parameter replicas diverged")
        return passed

    def reset_timing_diagnostics(self) -> None:
        for key in _TIMING_KEYS:
            self._timings[key] = 0.0
        self._planner_results.clear()

    def read_timing_diagnostics(self) -> dict[str, Any]:
        return {
            **{key: float(value) for key, value in self._timings.items()},
            "timing_clock": (
                "host perf_counter around calls; CUDA compute/collective phases may be "
                "asynchronous, so the runner must use synchronized boundaries for wall time"
            ),
            "world_size": self.world_size,
            "rank": self.rank,
            "loopback": self.loopback,
            "routed_layer_names": list(self.routed_layer_names),
            "planner_diagnostics": [
                getattr(result, "diagnostics", None) for result in self._planner_results
            ],
            "native_schedule_calls": int(self._native_layouts),
        }

    def close(self) -> None:
        """Release pair groups after every rank has left the training loop.

        All ranks must call close in the same order because NCCL group teardown
        follows the same ordering rule as group creation.
        """
        if self._closed:
            return
        if not self.loopback and _distributed_ready():
            for expert in range(EXPERT_COUNT):
                group = self._replica_groups.get(expert)
                if group is not None:
                    dist.destroy_process_group(group)
        self._replica_groups.clear()
        self._closed = True


def install_training_routing(
    model: nn.Module,
    planner: Any,
    schedule_library: Any,
    world_size: int,
    rank: int,
    *,
    loopback: bool = False,
) -> TrainingRoutingHandle:
    """Replace every Granite MoE with the trainable adjacent-ring backend.

    ``planner`` must provide the native ``plan(global_counts, world_size)`` API.
    ``schedule_library`` is normally the compiled C ABI library path. In tests,
    an explicit callable schedule adapter can be injected. Set ``loopback=True``
    for a single process that simulates a virtual 4/6/8-device placement using
    shared views of the model's canonical Expert weights.
    """
    return TrainingRoutingHandle(
        model,
        planner,
        schedule_library,
        world_size,
        rank,
        loopback=loopback,
    )

