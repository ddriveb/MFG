"""Opt-in shared Expert GEMM-granularity candidates for the packed executor.

This module changes only the per-Expert linear call shapes. Routing, stable
packing, checkpoint recomputation, weights, activation, and parameter ownership
remain inherited from the existing packed executor. FP32/TF32 policy is chosen
by the caller exactly as for the historical path.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .granite_communication_social_v2_training_replica_packed_v1 import (
    GraniteTrainingReplicaError,
    PackedExpertLayout,
    PackedRoutedGraniteMoE,
)
from .granite_training_numeric_kernels_v1 import TILE_ROWS, fixed_tile_linear

CHUNK_ROWS = 1024
EXECUTOR_MODES = ("fixed128", "chunk1024", "whole_expert")


def _chunk_linear(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply F.linear in 1024-row chunks, padding only a final partial tile."""
    if inputs.ndim != 2 or weight.ndim != 2:
        raise GraniteTrainingReplicaError("Expert linear inputs and weights must be rank two")
    if inputs.shape[0] == 0:
        return F.linear(inputs, weight)

    pieces = []
    for start in range(0, inputs.shape[0], CHUNK_ROWS):
        block = inputs[start : start + CHUNK_ROWS]
        count = int(block.shape[0])
        remainder = count % TILE_ROWS
        padded_rows = count if remainder == 0 else count + TILE_ROWS - remainder
        if padded_rows != count:
            block = F.pad(block, (0, 0, 0, padded_rows - count))
        pieces.append(F.linear(block, weight)[:count])
    return torch.cat(pieces, dim=0)


class GranularityRoutedGraniteMoE(PackedRoutedGraniteMoE):
    """Packed routed Expert wrapper with an explicitly selected GEMM granularity.

    ``fixed128`` calls the exact existing 128-row reference implementation.
    ``chunk1024`` uses 1024-row chunks and pads only each Expert's final partial
    chunk to a multiple of 128. ``whole_expert`` sends each nonempty Expert's
    full unpadded batch to one ``F.linear`` call per projection.
    """

    def __init__(self, *args, executor_mode: str = "fixed128", **kwargs):
        if executor_mode not in EXECUTOR_MODES:
            raise ValueError(f"executor_mode must be one of {EXECUTOR_MODES}")
        self.executor_mode = executor_mode
        super().__init__(*args, **kwargs)

    def _linear(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.executor_mode == "fixed128":
            return fixed_tile_linear(inputs, weight)
        if self.executor_mode == "chunk1024":
            return _chunk_linear(inputs, weight)
        return F.linear(inputs, weight)

    def _expert_mlp(self, packed_inputs: torch.Tensor, layout: PackedExpertLayout) -> torch.Tensor:
        if self.executor_mode == "fixed128":
            return super()._expert_mlp(packed_inputs, layout)
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
                projected = self._linear(expert_inputs, self.local_expert_w1[str(expert)])
                gate, value = projected.chunk(2, dim=-1)
                activated = self.activation(gate) * value
                outputs.append(self._linear(activated, self.local_expert_w2[str(expert)]))
            offset = next_offset
        if not outputs:
            return packed_inputs * 0
        return torch.cat(outputs, dim=0)


__all__ = ["CHUNK_ROWS", "EXECUTOR_MODES", "GranularityRoutedGraniteMoE"]
