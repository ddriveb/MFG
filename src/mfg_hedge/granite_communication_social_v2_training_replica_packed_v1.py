"""Opt-in packed Expert execution for communication-social-v2 training.

The existing routing handle, Gate, source quotas, schedule, collectives, and
parameter ownership are reused unchanged. This module only moves stable Expert
grouping, input packing, and inverse restoration outside checkpoint recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from . import granite_communication_social_v2_training_replica_v1 as _base
from .granite_training_numeric_kernels_v1 import fixed_tile_linear


GraniteTrainingReplicaError = _base.GraniteTrainingReplicaError
EXPERT_COUNT = _base.EXPERT_COUNT
TrainingRoutingHandle = _base.TrainingRoutingHandle


@dataclass(frozen=True)
class PackedExpertLayout:
    """Per-invocation stable grouping metadata.

    permutation maps packed positions to original positions;
    inverse_permutation maps original positions back to packed positions.
    These tensors are created once by build_expert_layout and treated as
    read-only. The checkpoint closure captures only this invocation's frozen
    layout and never stores it on a module.
    """

    resident_experts: tuple[int, ...]
    counts: tuple[int, ...]
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor


def _normalize_resident(resident_experts) -> tuple[int, ...]:
    resident = tuple(int(value) for value in resident_experts)
    if len(set(resident)) != len(resident):
        raise GraniteTrainingReplicaError("resident Experts must be unique")
    if tuple(sorted(resident)) != resident:
        raise GraniteTrainingReplicaError("resident Experts must be in ascending order")
    if any(expert < 0 or expert >= EXPERT_COUNT for expert in resident):
        raise GraniteTrainingReplicaError("resident Expert id is out of range")
    return resident


def build_expert_layout(
    expert_ids: torch.Tensor,
    resident_experts,
    *,
    rank: int | None = None,
) -> PackedExpertLayout:
    """Build one stable Expert sort and its inverse for a route batch."""
    if not isinstance(expert_ids, torch.Tensor) or expert_ids.ndim != 1:
        raise GraniteTrainingReplicaError("Expert ids must be a rank-one tensor")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise GraniteTrainingReplicaError("Expert ids must use an integer dtype")
    resident = _normalize_resident(resident_experts)
    if expert_ids.numel():
        allowed = torch.as_tensor(resident, dtype=expert_ids.dtype, device=expert_ids.device)
        invalid = ~torch.isin(expert_ids, allowed)
        if bool(torch.any(invalid)):
            missing = sorted(set(int(value) for value in expert_ids[invalid].detach().cpu().tolist()) - set(resident))
            prefix = f"rank {rank} " if rank is not None else ""
            raise GraniteTrainingReplicaError(
                f"{prefix}received nonresident Experts {missing}"
            )

    permutation = torch.argsort(expert_ids, stable=True)
    inverse = torch.empty_like(permutation)
    if permutation.numel():
        inverse.scatter_(
            0,
            permutation,
            torch.arange(permutation.numel(), dtype=permutation.dtype, device=permutation.device),
        )
    if expert_ids.numel():
        counts_tensor = torch.bincount(expert_ids.to(torch.int64), minlength=EXPERT_COUNT)
        counts_by_expert = counts_tensor.detach().cpu().tolist()
        counts = tuple(int(counts_by_expert[expert]) for expert in resident)
    else:
        counts = (0,) * len(resident)
    return PackedExpertLayout(resident, counts, permutation, inverse)


class PackedRoutedGraniteMoE(_base._RoutedGraniteMoE):
    """Existing routed wrapper with per-call packed Expert activation execution."""

    def _expert_mlp(self, packed_inputs: torch.Tensor, layout: PackedExpertLayout) -> torch.Tensor:
        if packed_inputs.ndim != 2:
            raise GraniteTrainingReplicaError("packed Expert inputs must be rank two")
        if len(layout.resident_experts) != len(layout.counts):
            raise GraniteTrainingReplicaError("packed Expert layout dimensions are inconsistent")
        if sum(layout.counts) != int(packed_inputs.shape[0]):
            raise GraniteTrainingReplicaError("packed Expert counts do not cover the input batch")

        outputs = []
        offset = 0
        for expert, count in zip(layout.resident_experts, layout.counts):
            next_offset = offset + count
            if count:
                expert_inputs = packed_inputs[offset:next_offset]
                projected = fixed_tile_linear(
                    expert_inputs,
                    self.local_expert_w1[str(expert)],
                )
                gate, value = projected.chunk(2, dim=-1)
                activated = self.activation(gate) * value
                outputs.append(
                    fixed_tile_linear(
                        activated,
                        self.local_expert_w2[str(expert)],
                    )
                )
            offset = next_offset
        if not outputs:
            return packed_inputs * 0
        return torch.cat(outputs, dim=0)

    def _run_expert_mlp(self, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or expert_ids.ndim != 1 or inputs.shape[0] != expert_ids.numel():
            raise GraniteTrainingReplicaError("Expert input and id rows do not match")
        if not inputs.numel():
            return inputs * 0

        layout = build_expert_layout(
            expert_ids,
            self.handle.resident_experts,
            rank=self.handle.rank,
        )
        packed_inputs = inputs.index_select(0, layout.permutation)

        # This closure belongs to one forward invocation. Recompute receives
        # the already-packed activation and reuses the same frozen layout.
        def pure_packed_mlp(packed_activation: torch.Tensor) -> torch.Tensor:
            return self._expert_mlp(packed_activation, layout)

        if self.training and torch.is_grad_enabled() and inputs.numel():
            packed_outputs = checkpoint(
                pure_packed_mlp,
                packed_inputs,
                use_reentrant=False,
            )
        else:
            packed_outputs = pure_packed_mlp(packed_inputs)
        return packed_outputs.index_select(0, layout.inverse_permutation)


class PackedTrainingRoutingHandle(_base.TrainingRoutingHandle):
    """Existing source-quota handle installing the packed Expert wrapper."""

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
            setattr(parent, child_name, PackedRoutedGraniteMoE(original, self, name))
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
        expected_per_moe = len(self.resident_experts) * 2
        for name in self.routed_layer_names:
            local_wrapper = self.model.get_submodule(name)
            if len(local_wrapper.local_expert_w1) + len(local_wrapper.local_expert_w2) != expected_per_moe:
                raise GraniteTrainingReplicaError(
                    f"rank {self.rank} Expert residency is incomplete at {name}"
                )


def install_packed_training_routing(
    model,
    planner: Any,
    schedule_library: Any,
    world_size: int,
    rank: int,
    *,
    loopback: bool = False,
) -> PackedTrainingRoutingHandle:
    """Install the opt-in packed executor with the v2 backend's frozen API."""
    return PackedTrainingRoutingHandle(
        model,
        planner,
        schedule_library,
        world_size,
        rank,
        loopback=loopback,
    )


__all__ = [
    "GraniteTrainingReplicaError",
    "PackedExpertLayout",
    "PackedRoutedGraniteMoE",
    "PackedTrainingRoutingHandle",
    "TrainingRoutingHandle",
    "build_expert_layout",
    "install_packed_training_routing",
]
