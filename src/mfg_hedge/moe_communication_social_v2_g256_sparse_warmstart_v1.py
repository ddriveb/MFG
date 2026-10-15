"""Generic sparse float64 warm-start model and fixed-budget social Mirror-Prox.

This module is independent of the frozen G6-only warm-start implementation.
All five physical operators stay in sparse incidence form; no dense
resource-by-variable matrix is materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from .moe_communication_social_v2 import BLOCK_NAMES, SparseSocialProblem


ROW_FEATURE_COUNT = 7
EDGE_FEATURE_COUNT = 12
RESOURCE_FEATURE_COUNT = 7
DEFAULT_WIDTH = 64
DEFAULT_MESSAGE_ROUNDS = 2
ROW_FEATURE_NAMES = (
    "log1p_demand_over_log1p_max_demand",
    "demand_over_mean_demand",
    "mask_width_over_four",
    "source_gpu_is_legal_destination",
    "fraction_of_mask_destinations_on_source_node",
    "cyclic_expert_minus_source_offset_over_gpu_count",
    "mean_row_operator_mass_over_uniform_F_reference",
)
EDGE_FEATURE_NAMES = (
    "destination_is_source_gpu",
    "destination_is_on_source_node",
    "destination_is_on_different_node",
    "cyclic_destination_minus_source_offset_over_gpu_count",
    "cyclic_destination_minus_expert_gpu_offset_over_gpu_count",
    "cyclic_expert_minus_source_offset_over_gpu_count",
    "slot_index_over_mask_width_minus_one",
    "dispatch_intra_coefficient_mass_over_uniform_F_reference",
    "dispatch_inter_coefficient_mass_over_uniform_F_reference",
    "compute_coefficient_mass_over_uniform_F_reference",
    "return_intra_coefficient_mass_over_uniform_F_reference",
    "return_inter_coefficient_mass_over_uniform_F_reference",
)
RESOURCE_FEATURE_NAMES = (
    "dispatch_intra_block_indicator",
    "dispatch_inter_block_indicator",
    "compute_block_indicator",
    "return_intra_block_indicator",
    "return_inter_block_indicator",
    "incidence_degree_over_variable_count",
    "coefficient_mass_over_uniform_F_reference",
)


def _group_ids(values: Sequence[int]) -> tuple[tuple[int, ...], int]:
    mapping: dict[int, int] = {}
    result = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return tuple(result), len(mapping)


def _normalised_delta(delta: int, modulus: int) -> float:
    if modulus <= 1:
        return 0.0
    return (delta % modulus) / float(modulus)


@dataclass(frozen=True)
class SparseSocialTorchGraph:
    """One variable-size sparse problem and symmetry-safe graph features."""

    problem: SparseSocialProblem
    row_features: torch.Tensor
    edge_features: torch.Tensor
    resource_features: torch.Tensor
    row_index: torch.Tensor
    row_slot_indices: torch.Tensor
    row_slot_mask: torch.Tensor
    source_group: torch.Tensor
    source_group_count: int
    expert_group: torch.Tensor
    expert_group_count: int
    destination_group: torch.Tensor
    destination_group_count: int
    incidence_resource: torch.Tensor
    incidence_variable: torch.Tensor
    resource_to_variable_weight: torch.Tensor
    variable_to_resource_weight: torch.Tensor
    incidence_coefficients: torch.Tensor
    block_resource_offsets: tuple[int, ...]
    block_incidence_offsets: tuple[int, ...]
    f_reference: torch.Tensor
    step_scale: torch.Tensor

    @classmethod
    def from_problem(
        cls,
        problem: SparseSocialProblem,
        *,
        device: torch.device | str = "cpu",
    ) -> "SparseSocialTorchGraph":
        if tuple(block.name for block in problem.blocks) != BLOCK_NAMES:
            raise ValueError("the five physical objective blocks in frozen order are required")
        row_count = len(problem.rows)
        variable_count = problem.variable_count
        if row_count <= 0 or variable_count <= 0:
            raise ValueError("problem must contain rows and legal destination variables")
        gpu_count = len(problem.config.gpu_nodes)
        if gpu_count <= 0:
            raise ValueError("problem topology must contain GPUs")
        target = torch.device(device)

        max_slots = max(problem.row_offsets[i + 1] - problem.row_offsets[i] for i in range(row_count))
        slot_indices = [[0] * max_slots for _ in range(row_count)]
        slot_mask = [[False] * max_slots for _ in range(row_count)]
        row_index = [0] * variable_count
        for row_i in range(row_count):
            start, end = problem.row_offsets[row_i : row_i + 2]
            for slot, variable_i in enumerate(range(start, end)):
                slot_indices[row_i][slot] = variable_i
                slot_mask[row_i][slot] = True
                row_index[variable_i] = row_i

        source_ids = tuple(row.source_gpu for row in problem.rows)
        expert_ids = tuple(row.expert_id for row in problem.rows)
        source_groups, source_group_count = _group_ids(source_ids)
        expert_groups, expert_group_count = _group_ids(expert_ids)
        del source_group_count, expert_group_count
        destination_ids = tuple(problem.variable_gpus)
        destination_groups, _destination_group_count = _group_ids(destination_ids)

        uniform = problem.uniform_shares()
        f_reference = float(problem.score(uniform).total_ms)
        if not math.isfinite(f_reference) or f_reference <= 0.0:
            raise ValueError("uniform reference objective must be finite and positive")
        norm_bound = problem.operator_norm_bound(f_reference, BLOCK_NAMES)
        step_scale = 0.0 if norm_bound == 0.0 else 0.5 / norm_bound / f_reference

        # The graph features contain only scale-free or relative quantities.
        # Resource/Expert/GPU identifiers are used for sparse aggregation only,
        # never as learned embeddings or raw feature columns.
        max_demand = max((row.demand for row in problem.rows), default=1)
        mean_demand = max(sum(row.demand for row in problem.rows) / row_count, 1.0)
        row_feats: list[list[float]] = []
        edge_feats: list[list[float]] = [[0.0] * EDGE_FEATURE_COUNT for _ in range(variable_count)]
        variable_block_mass = [[0.0] * len(BLOCK_NAMES) for _ in range(variable_count)]
        for block_i, block in enumerate(problem.blocks):
            for variable_i, column in enumerate(block.columns):
                variable_block_mass[variable_i][block_i] = sum(
                    coefficient for _resource_i, coefficient in column
                ) / f_reference

        for row_i, row in enumerate(problem.rows):
            start, end = problem.row_offsets[row_i : row_i + 2]
            width = end - start
            local_count = sum(
                problem.config.gpu_nodes[row.source_gpu]
                == problem.config.gpu_nodes[problem.variable_gpus[variable_i]]
                for variable_i in range(start, end)
            )
            source_expert_delta = _normalised_delta(row.expert_id - row.source_gpu, gpu_count)
            row_mass = sum(sum(variable_block_mass[v]) for v in range(start, end)) / width
            row_feats.append(
                [
                    math.log1p(row.demand) / max(math.log1p(max_demand), 1.0),
                    row.demand / mean_demand,
                    width / 4.0,
                    float(row.source_gpu in row.allowed_gpus),
                    local_count / width,
                    source_expert_delta,
                    row_mass,
                ]
            )
            for slot, variable_i in enumerate(range(start, end)):
                destination = problem.variable_gpus[variable_i]
                source_node = problem.config.gpu_nodes[row.source_gpu]
                destination_node = problem.config.gpu_nodes[destination]
                expert_gpu = row.expert_id % gpu_count
                edge_feats[variable_i] = [
                    float(destination == row.source_gpu),
                    float(destination_node == source_node),
                    float(destination_node != source_node),
                    _normalised_delta(destination - row.source_gpu, gpu_count),
                    _normalised_delta(destination - expert_gpu, gpu_count),
                    source_expert_delta,
                    slot / max(width - 1, 1),
                    *variable_block_mass[variable_i],
                ]

        block_offsets = [0]
        incidence_offsets = [0]
        incidence_resources: list[int] = []
        incidence_variables: list[int] = []
        incidence_coefficients: list[float] = []
        resource_features: list[list[float]] = []
        for block_i, block in enumerate(problem.blocks):
            resource_offset = block_offsets[-1]
            resource_count = len(block.resource_names)
            resource_coeff_sum = [0.0] * resource_count
            resource_degree = [0] * resource_count
            variable_coeff_sum = [0.0] * variable_count
            for variable_i, column in enumerate(block.columns):
                for resource_i, coefficient in column:
                    resource_coeff_sum[resource_i] += coefficient
                    resource_degree[resource_i] += 1
                    variable_coeff_sum[variable_i] += coefficient
                    incidence_resources.append(resource_offset + resource_i)
                    incidence_variables.append(variable_i)
                    incidence_coefficients.append(coefficient)
            for resource_i in range(resource_count):
                one_hot = [0.0] * len(BLOCK_NAMES)
                one_hot[block_i] = 1.0
                resource_features.append(
                    [
                        *one_hot,
                        resource_degree[resource_i] / max(variable_count, 1),
                        resource_coeff_sum[resource_i] / f_reference,
                    ]
                )
            block_offsets.append(resource_offset + resource_count)
            incidence_offsets.append(len(incidence_resources))

        res_weight = []
        var_weight = []
        incidence_res_total = [0.0] * block_offsets[-1]
        incidence_var_total = [0.0] * variable_count
        for resource_i, variable_i, coefficient in zip(
            incidence_resources, incidence_variables, incidence_coefficients
        ):
            incidence_res_total[resource_i] += coefficient
            incidence_var_total[variable_i] += coefficient
        for resource_i, variable_i, coefficient in zip(
            incidence_resources, incidence_variables, incidence_coefficients
        ):
            res_weight.append(
                coefficient / incidence_res_total[resource_i]
                if incidence_res_total[resource_i] > 0.0 else 0.0
            )
            var_weight.append(
                coefficient / incidence_var_total[variable_i]
                if incidence_var_total[variable_i] > 0.0 else 0.0
            )

        def tensor(data, *, dtype=torch.float64):
            return torch.tensor(data, dtype=dtype, device=target)

        return cls(
            problem=problem,
            row_features=tensor(row_feats),
            edge_features=tensor(edge_feats),
            resource_features=tensor(resource_features),
            row_index=tensor(row_index, dtype=torch.long),
            row_slot_indices=tensor(slot_indices, dtype=torch.long),
            row_slot_mask=tensor(slot_mask, dtype=torch.bool),
            source_group=tensor(source_groups, dtype=torch.long),
            source_group_count=len(set(source_groups)),
            expert_group=tensor(expert_groups, dtype=torch.long),
            expert_group_count=len(set(expert_groups)),
            destination_group=tensor(destination_groups, dtype=torch.long),
            destination_group_count=len(set(destination_groups)),
            incidence_resource=tensor(incidence_resources, dtype=torch.long),
            incidence_variable=tensor(incidence_variables, dtype=torch.long),
            resource_to_variable_weight=tensor(res_weight),
            variable_to_resource_weight=tensor(var_weight),
            incidence_coefficients=tensor(incidence_coefficients),
            block_resource_offsets=tuple(block_offsets),
            block_incidence_offsets=tuple(incidence_offsets),
            f_reference=tensor(f_reference),
            step_scale=tensor(step_scale),
        )

    @property
    def row_count(self) -> int:
        return len(self.problem.rows)

    @property
    def variable_count(self) -> int:
        return self.problem.variable_count

    @property
    def device(self) -> torch.device:
        return self.row_features.device

    @property
    def block_widths(self) -> tuple[int, ...]:
        return tuple(
            self.block_resource_offsets[i + 1] - self.block_resource_offsets[i]
            for i in range(len(BLOCK_NAMES))
        )

    def to(self, device: torch.device | str) -> "SparseSocialTorchGraph":
        target = torch.device(device)
        return SparseSocialTorchGraph(
            problem=self.problem,
            row_features=self.row_features.to(target),
            edge_features=self.edge_features.to(target),
            resource_features=self.resource_features.to(target),
            row_index=self.row_index.to(target),
            row_slot_indices=self.row_slot_indices.to(target),
            row_slot_mask=self.row_slot_mask.to(target),
            source_group=self.source_group.to(target),
            source_group_count=self.source_group_count,
            expert_group=self.expert_group.to(target),
            expert_group_count=self.expert_group_count,
            destination_group=self.destination_group.to(target),
            destination_group_count=self.destination_group_count,
            incidence_resource=self.incidence_resource.to(target),
            incidence_variable=self.incidence_variable.to(target),
            resource_to_variable_weight=self.resource_to_variable_weight.to(target),
            variable_to_resource_weight=self.variable_to_resource_weight.to(target),
            incidence_coefficients=self.incidence_coefficients.to(target),
            block_resource_offsets=self.block_resource_offsets,
            block_incidence_offsets=self.block_incidence_offsets,
            f_reference=self.f_reference.to(target),
            step_scale=self.step_scale.to(target),
        )

    def apply(self, shares: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _check_vector(shares, self.variable_count, "shares", self.device, finite=False)
        all_loads = torch.zeros(
            self.block_resource_offsets[-1], dtype=torch.float64, device=self.device
        ).scatter_add(
            0,
            self.incidence_resource,
            self.incidence_coefficients * shares[self.incidence_variable],
        )
        return tuple(
            all_loads[self.block_resource_offsets[i] : self.block_resource_offsets[i + 1]]
            for i in range(len(BLOCK_NAMES))
        )

    def adjoint(self, dual: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(dual) != len(BLOCK_NAMES):
            raise ValueError("dual must contain five resource simplexes")
        if any(
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.float64
            or values.device != self.device
            or tuple(values.shape) != (self.block_widths[i],)
            for i, values in enumerate(dual)
        ):
            raise ValueError("dual blocks must be correctly shaped float64 tensors on the graph device")
        flat_dual = torch.cat(tuple(dual))
        return torch.zeros(self.variable_count, dtype=torch.float64, device=self.device).scatter_add(
            0,
            self.incidence_variable,
            self.incidence_coefficients * flat_dual[self.incidence_resource],
        )


class SparseGraphPrimalDualWarmStartNet(nn.Module):
    """Shared width-64 sparse message-passing network with feasible heads."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        message_rounds: int = DEFAULT_MESSAGE_ROUNDS,
    ) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        if type(message_rounds) is not int or message_rounds != 2:
            raise ValueError("Ticket24 freezes exactly two shared message rounds")
        self.width = width
        self.message_rounds = message_rounds
        self.row_encoder = nn.Sequential(nn.Linear(ROW_FEATURE_COUNT, width), nn.SiLU())
        self.edge_encoder = nn.Sequential(nn.Linear(EDGE_FEATURE_COUNT, width), nn.SiLU())
        self.resource_encoder = nn.Sequential(
            nn.Linear(RESOURCE_FEATURE_COUNT, width), nn.SiLU()
        )
        self.edge_update = nn.Sequential(
            nn.Linear(width * 5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.row_update = nn.Sequential(
            nn.Linear(width * 5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.resource_update = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.primal_head = nn.Linear(width, 1)
        self.dual_head = nn.Linear(width, 1)
        self.double()

    def forward(
        self, graph: SparseSocialTorchGraph
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if graph.row_features.dtype != torch.float64 or graph.edge_features.dtype != torch.float64:
            raise ValueError("G256 warm-start graph features must use float64")
        if graph.device != next(self.parameters()).device:
            raise ValueError("model and graph must be on the same device")
        row_state = self.row_encoder(graph.row_features)
        edge_state = self.edge_encoder(graph.edge_features)
        resource_base = self.resource_encoder(graph.resource_features)
        resource_state = resource_base
        width = self.width

        row_count = graph.row_count
        source_count = graph.source_group_count
        expert_count = graph.expert_group_count
        destination_count = graph.destination_group_count
        variable_count = graph.variable_count
        resource_count = graph.resource_features.shape[0]

        for _ in range(self.message_rounds):
            resource_messages = torch.zeros(
                (resource_count, width), dtype=torch.float64, device=graph.device
            )
            if graph.incidence_resource.numel():
                resource_messages = resource_messages.index_add(
                    0,
                    graph.incidence_resource,
                    edge_state[graph.incidence_variable]
                    * graph.resource_to_variable_weight[:, None],
                )
            resource_state = self.resource_update(
                torch.cat((resource_base, resource_messages), dim=-1)
            )
            edge_resource = torch.zeros(
                (variable_count, width), dtype=torch.float64, device=graph.device
            )
            if graph.incidence_resource.numel():
                edge_resource = edge_resource.index_add(
                    0,
                    graph.incidence_variable,
                    resource_state[graph.incidence_resource]
                    * graph.variable_to_resource_weight[:, None],
                )

            destination_messages = torch.zeros(
                (destination_count, width), dtype=torch.float64, device=graph.device
            ).index_add(0, graph.destination_group, edge_state)
            destination_counts = torch.bincount(
                graph.destination_group, minlength=destination_count
            ).to(torch.float64).clamp_min(1.0)
            destination_context = destination_messages / destination_counts[:, None]

            edge_by_row = torch.zeros(
                (row_count, width), dtype=torch.float64, device=graph.device
            ).index_add(0, graph.row_index, edge_state)
            row_degree = torch.bincount(graph.row_index, minlength=row_count).to(
                torch.float64
            ).clamp_min(1.0)
            edge_by_row = edge_by_row / row_degree[:, None]

            source_messages = torch.zeros(
                (source_count, width), dtype=torch.float64, device=graph.device
            ).index_add(0, graph.source_group, row_state)
            source_counts = torch.bincount(
                graph.source_group, minlength=source_count
            ).to(torch.float64).clamp_min(1.0)
            source_context = source_messages / source_counts[:, None]

            expert_messages = torch.zeros(
                (expert_count, width), dtype=torch.float64, device=graph.device
            ).index_add(0, graph.expert_group, row_state)
            expert_counts = torch.bincount(
                graph.expert_group, minlength=expert_count
            ).to(torch.float64).clamp_min(1.0)
            expert_context = expert_messages / expert_counts[:, None]
            global_context = row_state.mean(dim=0, keepdim=True).expand(row_count, -1)

            edge_state = self.edge_update(
                torch.cat(
                    (
                        edge_state,
                        row_state[graph.row_index],
                        destination_context[graph.destination_group],
                        edge_resource,
                        global_context[graph.row_index],
                    ),
                    dim=-1,
                )
            )
            row_state = self.row_update(
                torch.cat(
                    (
                        row_state,
                        edge_by_row,
                        source_context[graph.source_group],
                        expert_context[graph.expert_group],
                        global_context,
                    ),
                    dim=-1,
                )
            )

        # Refresh resource messages from final edge states for the dual head.
        resource_messages = torch.zeros(
            (resource_count, width), dtype=torch.float64, device=graph.device
        )
        if graph.incidence_resource.numel():
            resource_messages = resource_messages.index_add(
                0,
                graph.incidence_resource,
                edge_state[graph.incidence_variable]
                * graph.resource_to_variable_weight[:, None],
            )
        resource_state = self.resource_update(torch.cat((resource_base, resource_messages), dim=-1))

        flat_logits = self.primal_head(edge_state).squeeze(-1)
        padded_logits = flat_logits[graph.row_slot_indices]
        masked_logits = padded_logits.masked_fill(~graph.row_slot_mask, -torch.inf)
        primal_padded = torch.softmax(masked_logits, dim=-1)
        primal = primal_padded[graph.row_slot_mask]

        dual_logits = self.dual_head(resource_state).squeeze(-1)
        dual: list[torch.Tensor] = []
        for block_i in range(len(BLOCK_NAMES)):
            start, end = graph.block_resource_offsets[block_i : block_i + 2]
            dual.append(torch.softmax(dual_logits[start:end], dim=0))
        return primal, tuple(dual)


def _project_simplex(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("simplex projection expects a nonempty vector")
    if values.numel() == 1:
        return torch.ones_like(values)
    ordered = torch.sort(values, descending=True).values
    ranks = torch.arange(1, values.numel() + 1, dtype=values.dtype, device=values.device)
    candidates = (ordered.cumsum(dim=0) - 1.0) / ranks
    active = ordered > candidates
    rho = active.sum().clamp_min(1)
    theta = candidates[rho - 1]
    projected = torch.clamp(values - theta, min=0.0)
    total = projected.sum()
    return projected / total.clamp_min(torch.finfo(values.dtype).tiny)


def project_row_simplexes(values: torch.Tensor, graph: SparseSocialTorchGraph) -> torch.Tensor:
    _check_vector(values, graph.variable_count, "shares", graph.device)
    padded = values[graph.row_slot_indices]
    mask = graph.row_slot_mask
    ordered = torch.sort(padded.masked_fill(~mask, -torch.inf), dim=-1, descending=True).values
    sorted_mask = torch.sort(mask.to(torch.int8), dim=-1, descending=True).values.to(torch.bool)
    ranks = torch.arange(1, padded.shape[-1] + 1, dtype=values.dtype, device=values.device)
    candidates = (ordered.cumsum(dim=-1) - 1.0) / ranks
    active = sorted_mask & (ordered > candidates)
    rho = active.sum(dim=-1).clamp_min(1)
    theta = candidates.gather(1, (rho - 1)[:, None])
    projected = torch.clamp(padded - theta, min=0.0).masked_fill(~mask, 0.0)
    projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return projected[mask]


def _project_dual(dual: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(_project_simplex(values) for values in dual)


def _check_vector(
    values: torch.Tensor,
    width: int,
    name: str,
    device: torch.device,
    *,
    finite: bool = True,
) -> None:
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (width,):
        raise ValueError(f"{name} must have shape [{width}]")
    if values.dtype != torch.float64 or values.device != device:
        raise ValueError(f"{name} must be float64 on {device}")
    if finite and not torch.isfinite(values).all().item():
        raise ValueError(f"{name} must be finite")


def _check_row_simplexes(values: torch.Tensor, graph: SparseSocialTorchGraph, name: str) -> None:
    padded = values[graph.row_slot_indices]
    masked_values = padded.masked_fill(~graph.row_slot_mask, 0.0)
    if not torch.isfinite(masked_values).all().item():
        raise ValueError(f"{name} must be finite")
    if torch.any(masked_values < -1e-12).item():
        raise ValueError(f"{name} must be non-negative")
    row_sums = masked_values.sum(dim=-1)
    if torch.any((row_sums - 1.0).abs() > 1e-9).item():
        raise ValueError(f"{name} must lie on the row simplexes")


def _check_simplex(values: torch.Tensor, name: str) -> None:
    if torch.any(values < -1e-12).item() or abs(float(values.sum().item()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must lie on the unit simplex")


def fixed_mirror_prox(
    graph: SparseSocialTorchGraph,
    initial_shares: torch.Tensor,
    initial_dual: Sequence[torch.Tensor],
    *,
    budgets: Sequence[int] = (16, 24, 36),
) -> dict[int, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
    """Run exact fixed-count projected Mirror-Prox predictor averaging."""

    steps = tuple(budgets)
    if not steps or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("budgets must contain positive integer iteration counts")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("budgets must be strictly increasing")
    _check_vector(initial_shares, graph.variable_count, "initial_shares", graph.device)
    _check_row_simplexes(initial_shares, graph, "initial_shares")
    if len(initial_dual) != len(BLOCK_NAMES):
        raise ValueError("initial_dual must contain five blocks")
    for block_i, values in enumerate(initial_dual):
        _check_vector(values, graph.block_widths[block_i], f"initial_dual[{block_i}]", graph.device)
        _check_simplex(values, f"initial_dual[{block_i}]")

    current_shares = initial_shares
    current_dual = tuple(initial_dual)
    average_shares = torch.zeros_like(current_shares)
    average_dual = tuple(torch.zeros_like(values) for values in current_dual)
    outputs: dict[int, tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = {}

    for iteration in range(1, steps[-1] + 1):
        current_gradient = graph.adjoint(current_dual)
        trial_shares = project_row_simplexes(
            current_shares - graph.step_scale * current_gradient, graph
        )
        current_loads = graph.apply(current_shares)
        trial_dual = _project_dual(
            tuple(
                value + graph.step_scale * load
                for value, load in zip(current_dual, current_loads)
            )
        )
        trial_gradient = graph.adjoint(trial_dual)
        next_shares = project_row_simplexes(
            current_shares - graph.step_scale * trial_gradient, graph
        )
        trial_loads = graph.apply(trial_shares)
        next_dual = _project_dual(
            tuple(
                value + graph.step_scale * load
                for value, load in zip(current_dual, trial_loads)
            )
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
    graph: SparseSocialTorchGraph,
    shares: torch.Tensor,
    dual: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the unchanged physical primal UB, dual LB and raw gap."""

    _check_vector(shares, graph.variable_count, "shares", graph.device)
    _check_row_simplexes(shares, graph, "shares")
    if len(dual) != len(BLOCK_NAMES):
        raise ValueError("dual must contain five resource simplexes")
    for block_i, values in enumerate(dual):
        _check_vector(values, graph.block_widths[block_i], f"dual[{block_i}]", graph.device)
        _check_simplex(values, f"dual[{block_i}]")
    loads = graph.apply(shares)
    upper = torch.as_tensor(
        graph.problem.fixed_startup_ms, dtype=torch.float64, device=graph.device
    )
    gradient = graph.adjoint(dual)
    for block_loads in loads:
        upper = upper + block_loads.max()
    padded_gradient = gradient[graph.row_slot_indices]
    row_minima = padded_gradient.masked_fill(~graph.row_slot_mask, torch.inf).min(dim=-1).values
    lower = torch.as_tensor(
        graph.problem.fixed_startup_ms, dtype=torch.float64, device=graph.device
    ) + row_minima.sum()
    return upper, lower, upper - lower


def normalized_gap(
    graph: SparseSocialTorchGraph,
    shares: torch.Tensor,
    dual: Sequence[torch.Tensor],
) -> torch.Tensor:
    """True physical primal-dual gap normalized by the per-instance F reference."""

    _upper, _lower, gap = physical_certificate(graph, shares, dual)
    return gap / graph.f_reference

