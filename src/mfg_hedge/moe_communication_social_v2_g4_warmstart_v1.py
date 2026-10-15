"""Fixed-topology Torch warm starts and fixed-budget Mirror-Prox unroll.

This module is deliberately limited to the single-node four-GPU topology in
Ticket04.  It preserves the physical five-block max objective; Torch is used
only to differentiate the predicted feasible initial state and unrolled steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .moe_communication_social_v2 import BLOCK_NAMES, SparseSocialProblem


TOPOLOGY_WIDTHS = {
    "single_node_g4": (8, 1, 4, 8, 1),
}
FEATURE_COUNT = 16
VARIABLE_COUNT = 32
ROW_COUNT = 16
ROUTES_PER_SOURCE_FEATURE_SCALE = 8192.0


@dataclass(frozen=True)
class TorchSocialBatch:
    """Dense float64 batches of the existing sparse K operators.

    ``matrices[j]`` has shape ``[batch, resources_j, 32]``.  Coefficients
    already include row demand, exactly as in ``SparseSocialProblem``.
    """

    problems: tuple[SparseSocialProblem, ...]
    matrices: tuple[torch.Tensor, ...]
    f_reference: torch.Tensor
    gamma: torch.Tensor
    startup: torch.Tensor
    topology_id: str
    row_offsets: tuple[int, ...]
    block_widths: tuple[int, ...]

    @classmethod
    def from_problems(
        cls,
        problems: Sequence[SparseSocialProblem],
        *,
        device: torch.device | str = "cpu",
    ) -> "TorchSocialBatch":
        selected = tuple(problems)
        if not selected:
            raise ValueError("at least one problem is required")
        first = selected[0]
        if len(first.config.gpu_nodes) != 4 or len(first.rows) != ROW_COUNT or first.variable_count != VARIABLE_COUNT:
            raise ValueError("neural warm-start core requires the reduced G4 16-row/32-variable model")
        if first.row_offsets != tuple(range(0, VARIABLE_COUNT + 1, 2)):
            raise ValueError("reduced G4 rows must each have exactly two ordered slots")
        if tuple(block.name for block in first.blocks) != BLOCK_NAMES:
            raise ValueError("the full five objective blocks in frozen order are required")
        nodes = first.config.gpu_nodes
        if nodes == (0, 0, 0, 0):
            topology_id = "single_node_g4"
        else:
            raise ValueError("only the single-node G4 topology is supported")
        widths = tuple(len(block.resource_names) for block in first.blocks)
        if widths != TOPOLOGY_WIDTHS[topology_id]:
            raise ValueError("resource block widths differ from the frozen topology contract")

        for problem in selected:
            if problem.config.gpu_nodes != nodes:
                raise ValueError("a packed batch cannot mix topologies")
            if problem.row_offsets != first.row_offsets or problem.variable_gpus != first.variable_gpus:
                raise ValueError("a packed batch must retain identical row and slot order")
            if tuple(block.name for block in problem.blocks) != BLOCK_NAMES:
                raise ValueError("every problem must contain the frozen five blocks")
            if tuple(len(block.resource_names) for block in problem.blocks) != widths:
                raise ValueError("resource dimensions differ within the packed batch")
            if tuple(block.resource_names for block in problem.blocks) != tuple(
                block.resource_names for block in first.blocks
            ):
                raise ValueError("resource names/order differ within the packed batch")

        target = torch.device(device)
        matrices: list[torch.Tensor] = []
        for block_index, block in enumerate(first.blocks):
            per_problem = []
            for problem in selected:
                rows = [[0.0] * VARIABLE_COUNT for _ in problem.blocks[block_index].resource_names]
                for variable_index, column in enumerate(problem.blocks[block_index].columns):
                    for resource_index, coefficient in column:
                        rows[resource_index][variable_index] += coefficient
                per_problem.append(rows)
            matrices.append(torch.tensor(per_problem, dtype=torch.float64, device=target))

        references = []
        gammas = []
        startup = []
        for problem in selected:
            uniform = problem.uniform_shares()
            reference = problem.score(uniform).total_ms
            normalization = reference if reference > 0 else 1.0
            norm_bound = problem.operator_norm_bound(normalization, BLOCK_NAMES)
            references.append(normalization)
            gammas.append(0.0 if norm_bound == 0.0 else 0.5 / norm_bound)
            startup.append(problem.fixed_startup_ms)
        return cls(
            problems=selected,
            matrices=tuple(matrices),
            f_reference=torch.tensor(references, dtype=torch.float64, device=target),
            gamma=torch.tensor(gammas, dtype=torch.float64, device=target),
            startup=torch.tensor(startup, dtype=torch.float64, device=target),
            topology_id=topology_id,
            row_offsets=first.row_offsets,
            block_widths=widths,
        )

    @property
    def batch_size(self) -> int:
        return len(self.problems)

    @property
    def device(self) -> torch.device:
        return self.f_reference.device

    def select(self, indices: Sequence[int]) -> "TorchSocialBatch":
        index = torch.as_tensor(tuple(indices), dtype=torch.long, device=self.device)
        if index.numel() == 0:
            raise ValueError("cannot select an empty batch")
        return TorchSocialBatch(
            problems=tuple(self.problems[i] for i in indices),
            matrices=tuple(matrix.index_select(0, index) for matrix in self.matrices),
            f_reference=self.f_reference.index_select(0, index),
            gamma=self.gamma.index_select(0, index),
            startup=self.startup.index_select(0, index),
            topology_id=self.topology_id,
            row_offsets=self.row_offsets,
            block_widths=self.block_widths,
        )

    def apply(self, shares: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _check_batch_vector(shares, self.batch_size, VARIABLE_COUNT, "shares")
        return tuple(torch.bmm(matrix, shares.unsqueeze(-1)).squeeze(-1) for matrix in self.matrices)

    def adjoint(self, dual: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(dual) != len(BLOCK_NAMES):
            raise ValueError("dual must contain five resource-simplex blocks")
        result = torch.zeros((self.batch_size, VARIABLE_COUNT), dtype=torch.float64, device=self.device)
        for name, matrix, values, width in zip(BLOCK_NAMES, self.matrices, dual, self.block_widths):
            _check_batch_vector(values, self.batch_size, width, f"dual[{name}]")
            result = result + torch.bmm(matrix.transpose(1, 2), values.unsqueeze(-1)).squeeze(-1)
        return result


class FixedBudgetWarmStartNet(nn.Module):
    """Topology-specific 16→64→64 MLP with feasible primal/dual heads."""

    def __init__(self, topology_id: str) -> None:
        super().__init__()
        try:
            widths = TOPOLOGY_WIDTHS[topology_id]
        except KeyError as exc:
            raise ValueError(f"unsupported frozen G4 topology: {topology_id}") from exc
        self.topology_id = topology_id
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.primal_head = nn.Linear(64, VARIABLE_COUNT)
        self.dual_heads = nn.ModuleDict(
            {name: nn.Linear(64, width) for name, width in zip(BLOCK_NAMES, widths) if width > 1}
        )
        self.block_widths = widths

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if features.ndim != 2 or features.shape[-1] != FEATURE_COUNT:
            raise ValueError("features must have shape [batch, 16]")
        if features.dtype != torch.float64:
            raise ValueError("neural warm-start features must use float64")
        hidden = self.trunk(features)
        primal = torch.softmax(self.primal_head(hidden).reshape(-1, ROW_COUNT, 2), dim=-1).reshape(-1, VARIABLE_COUNT)
        dual = []
        for name, width in zip(BLOCK_NAMES, self.block_widths):
            if width == 1:
                dual.append(torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device))
            else:
                dual.append(torch.softmax(self.dual_heads[name](hidden), dim=-1))
        return primal, tuple(dual)


def project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean projection onto one or more last-axis simplexes."""

    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("simplex projection needs a non-empty last dimension")
    if not torch.isfinite(values).all().item():
        raise ValueError("simplex projection input must be finite")
    width = values.shape[-1]
    if width == 1:
        return torch.ones_like(values)
    ordered = torch.sort(values, dim=-1, descending=True).values
    ranks = torch.arange(1, width + 1, dtype=values.dtype, device=values.device)
    candidates = (ordered.cumsum(dim=-1) - 1.0) / ranks
    active = ordered > candidates
    rho = active.sum(dim=-1).clamp_min(1)
    theta = candidates.gather(-1, (rho - 1).unsqueeze(-1)).squeeze(-1)
    projected = torch.clamp(values - theta.unsqueeze(-1), min=0.0)
    total = projected.sum(dim=-1, keepdim=True)
    fallback = torch.nn.functional.one_hot(values.argmax(dim=-1), num_classes=width).to(values.dtype)
    return torch.where(total > 0.0, projected / total.clamp_min(torch.finfo(values.dtype).tiny), fallback)


def _project_primal(values: torch.Tensor) -> torch.Tensor:
    pairs = values.reshape(values.shape[0], ROW_COUNT, 2)
    left = ((pairs[..., 0] - pairs[..., 1] + 1.0) * 0.5).clamp(0.0, 1.0)
    return torch.stack((left, 1.0 - left), dim=-1).reshape(values.shape[0], VARIABLE_COUNT)


def _project_dual(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(project_simplex(block) for block in values)


def uniform_primal(batch: TorchSocialBatch) -> torch.Tensor:
    return torch.full((batch.batch_size, VARIABLE_COUNT), 0.5, dtype=torch.float64, device=batch.device)


def tied_max_uniform_dual(batch: TorchSocialBatch, shares: torch.Tensor) -> tuple[torch.Tensor, ...]:
    loads = batch.apply(shares)
    result = []
    for block_loads in loads:
        winners = block_loads == block_loads.max(dim=-1, keepdim=True).values
        result.append(winners.to(torch.float64) / winners.sum(dim=-1, keepdim=True))
    return tuple(result)


def fixed_mirror_prox(
    batch: TorchSocialBatch,
    initial_shares: torch.Tensor,
    initial_dual: Sequence[torch.Tensor],
    *,
    budgets: Sequence[int] = (4, 8, 16),
) -> dict[int, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
    """Run fixed safe-step Mirror-Prox and average synchronized predictor points."""

    steps = tuple(budgets)
    if not steps or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("budgets must contain positive integer iteration counts")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("budgets must be strictly increasing")
    _check_feasible_shares(initial_shares, batch.batch_size)
    if len(initial_dual) != len(BLOCK_NAMES):
        raise ValueError("initial_dual must contain five blocks")
    for name, values, width in zip(BLOCK_NAMES, initial_dual, batch.block_widths):
        _check_feasible_simplex(values, batch.batch_size, width, f"initial_dual[{name}]")

    current_shares = initial_shares
    current_dual = tuple(initial_dual)
    average_shares = torch.zeros_like(current_shares)
    average_dual = tuple(torch.zeros_like(values) for values in current_dual)
    outputs: dict[int, tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = {}
    step_scale = batch.gamma / batch.f_reference

    for iteration in range(1, steps[-1] + 1):
        current_gradient = batch.adjoint(current_dual)
        trial_shares = _project_primal(current_shares - step_scale[:, None] * current_gradient)
        current_loads = batch.apply(current_shares)
        trial_dual = _project_dual(
            tuple(value + step_scale[:, None] * load for value, load in zip(current_dual, current_loads))
        )
        trial_gradient = batch.adjoint(trial_dual)
        next_shares = _project_primal(current_shares - step_scale[:, None] * trial_gradient)
        trial_loads = batch.apply(trial_shares)
        next_dual = _project_dual(
            tuple(value + step_scale[:, None] * load for value, load in zip(current_dual, trial_loads))
        )

        average_shares = average_shares + trial_shares
        average_dual = tuple(total + trial for total, trial in zip(average_dual, trial_dual))
        current_shares, current_dual = next_shares, next_dual
        if iteration in steps:
            outputs[iteration] = (
                average_shares / iteration,
                tuple(values / iteration for values in average_dual),
            )
    return outputs


def physical_certificate(
    batch: TorchSocialBatch,
    shares: torch.Tensor,
    dual: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return physical primal UB, dual LB and raw UB−LB in model milliseconds."""

    _check_feasible_shares(shares, batch.batch_size)
    if len(dual) != len(BLOCK_NAMES):
        raise ValueError("dual must contain five blocks")
    for name, weights, width in zip(BLOCK_NAMES, dual, batch.block_widths):
        _check_feasible_simplex(weights, batch.batch_size, width, f"dual[{name}]")
    loads = batch.apply(shares)
    upper = batch.startup.clone()
    for block_loads in loads:
        upper = upper + block_loads.max(dim=-1).values
    lower = batch.startup + batch.adjoint(dual).reshape(batch.batch_size, ROW_COUNT, 2).min(dim=-1).values.sum(dim=-1)
    return upper, lower, upper - lower


def _check_batch_vector(values: torch.Tensor, batch_size: int, width: int, name: str) -> None:
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (batch_size, width):
        raise ValueError(f"{name} must have shape [{batch_size}, {width}]")
    if values.dtype != torch.float64:
        raise ValueError(f"{name} must use float64")
    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} must be finite")


def _check_feasible_simplex(values: torch.Tensor, batch_size: int, width: int, name: str) -> None:
    _check_batch_vector(values, batch_size, width, name)
    detached = values.detach()
    if torch.any(detached < -1e-12).item() or torch.any((detached.sum(-1) - 1.0).abs() > 1e-9).item():
        raise ValueError(f"{name} must lie on its unit simplex")


def _check_feasible_shares(values: torch.Tensor, batch_size: int) -> None:
    _check_batch_vector(values, batch_size, VARIABLE_COUNT, "shares")
    pairs = values.detach().reshape(batch_size, ROW_COUNT, 2)
    if torch.any(pairs < -1e-12).item() or torch.any((pairs.sum(-1) - 1.0).abs() > 1e-9).item():
        raise ValueError("shares must lie on the 16 row simplexes")
