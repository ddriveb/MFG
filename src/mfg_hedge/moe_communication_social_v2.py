"""Sparse source-aware convex communication social model and Mirror-Prox solver.

Rows retain the (source GPU, logical Expert) identity.  A variable is one
legal destination slot in that row's fixed mask; the row variables form a
simplex of shares.  All sparse operator coefficients already include the
row demand, so callers must not multiply route counts a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Mapping, Sequence


BLOCK_NAMES = (
    "dispatch_intra",
    "dispatch_inter",
    "compute",
    "return_intra",
    "return_inter",
)
_COMMUNICATION_BLOCKS = frozenset(
    ("dispatch_intra", "dispatch_inter", "return_intra", "return_inter")
)


@dataclass(frozen=True)
class RouteRow:
    """Demand and legal destination mask for one (source, Expert) row."""

    source_gpu: int
    expert_id: int
    demand: int
    allowed_gpus: tuple[int, ...]


@dataclass(frozen=True)
class CommunicationConfig:
    """Fixed topology, homogeneous service rate, traffic sizes and link rates.

    Bandwidths use decimal units.  For example, 200 Gbps is converted to
    25,000,000 bytes/ms per GPU NIC.  A node's directional capacity is its
    represented GPU count times that per-GPU NIC rate.
    """

    gpu_nodes: tuple[int, ...]
    mu_routes_per_ms: float = 10.0
    dispatch_bytes_per_route: float = 1.0
    return_bytes_per_route: float = 1.0
    intra_bandwidth_bytes_per_ms: float = 50_000_000.0
    nic_bandwidth_gbps: float = 200.0
    alpha_dispatch_ms: float = 0.02
    alpha_return_ms: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_nodes, tuple) or not self.gpu_nodes:
            raise ValueError("gpu_nodes must be a non-empty tuple indexed by GPU")
        if any(type(node) is not int or node < 0 for node in self.gpu_nodes):
            raise ValueError("node IDs must be non-negative integers")
        positive = (
            ("mu_routes_per_ms", self.mu_routes_per_ms),
            ("intra_bandwidth_bytes_per_ms", self.intra_bandwidth_bytes_per_ms),
            ("nic_bandwidth_gbps", self.nic_bandwidth_gbps),
        )
        nonnegative = (
            ("dispatch_bytes_per_route", self.dispatch_bytes_per_route),
            ("return_bytes_per_route", self.return_bytes_per_route),
            ("alpha_dispatch_ms", self.alpha_dispatch_ms),
            ("alpha_return_ms", self.alpha_return_ms),
        )
        for name, value in positive:
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in nonnegative:
            if not _is_finite_number(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def nic_bandwidth_bytes_per_ms(self) -> float:
        return self.nic_bandwidth_gbps * 1_000_000.0 / 8.0


@dataclass(frozen=True)
class SparseBlock:
    """One max block stored by sparse variable-column incidences."""

    name: str
    resource_names: tuple[str, ...]
    columns: tuple[tuple[tuple[int, float], ...], ...]

    @property
    def incidence_count(self) -> int:
        return sum(len(column) for column in self.columns)


@dataclass(frozen=True)
class ObjectiveScore:
    dispatch_ms: float
    compute_ms: float
    return_ms: float
    total_ms: float
    dispatch_intra_ms: float
    dispatch_inter_ms: float
    return_intra_ms: float
    return_inter_ms: float
    fixed_startup_ms: float
    resource_loads_ms: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class MirrorProxResult:
    shares: tuple[float, ...]
    dual_weights: Mapping[str, tuple[float, ...]]
    status: str
    iterations: int
    upper_bound_ms: float
    lower_bound_ms: float
    gap_ms: float
    step_size: float
    operator_norm_bound: float
    normalization_ms: float
    objective_blocks: tuple[str, ...]
    elapsed_ms: float


class SparseSocialProblem:
    """A fixed-placement, source-row masked-simplex optimization problem."""

    def __init__(
        self,
        rows: tuple[RouteRow, ...],
        config: CommunicationConfig,
        row_offsets: tuple[int, ...],
        variable_rows: tuple[int, ...],
        variable_gpus: tuple[int, ...],
        blocks: tuple[SparseBlock, ...],
    ) -> None:
        self.rows = rows
        self.config = config
        self.row_offsets = row_offsets
        self.variable_rows = variable_rows
        self.variable_gpus = variable_gpus
        self.blocks = blocks
        self._blocks_by_name = {block.name: block for block in blocks}

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[RouteRow],
        config: CommunicationConfig,
    ) -> "SparseSocialProblem":
        """Validate rows and build only route-to-resource sparse incidences."""

        if not isinstance(config, CommunicationConfig):
            raise TypeError("config must be CommunicationConfig")
        normalized_rows = tuple(rows)
        if not normalized_rows:
            raise ValueError("at least one source-Expert row is required")

        gpu_count = len(config.gpu_nodes)
        seen_rows: set[tuple[int, int]] = set()
        expert_masks: dict[int, frozenset[int]] = {}
        for row in normalized_rows:
            if not isinstance(row, RouteRow):
                raise TypeError("each row must be RouteRow")
            if type(row.source_gpu) is not int or not 0 <= row.source_gpu < gpu_count:
                raise ValueError("row source_gpu must name a configured GPU")
            if type(row.expert_id) is not int or row.expert_id < 0:
                raise ValueError("expert_id must be a non-negative integer")
            if type(row.demand) is not int or row.demand < 0:
                raise ValueError("row demand must be a non-negative integer")
            if not isinstance(row.allowed_gpus, tuple) or not 1 <= len(row.allowed_gpus) <= 4:
                raise ValueError("each row needs a tuple destination mask of size 1..4")
            if any(type(gpu) is not int or not 0 <= gpu < gpu_count for gpu in row.allowed_gpus):
                raise ValueError("destination mask contains an unknown GPU")
            if len(set(row.allowed_gpus)) != len(row.allowed_gpus):
                raise ValueError("destination mask contains duplicate GPUs")
            mask = frozenset(row.allowed_gpus)
            previous_mask = expert_masks.setdefault(row.expert_id, mask)
            if previous_mask != mask:
                raise ValueError("all source rows for one Expert must share its fixed replica set")
            key = (row.source_gpu, row.expert_id)
            if key in seen_rows:
                raise ValueError("source-Expert rows must be unique")
            seen_rows.add(key)

        variable_rows: list[int] = []
        variable_gpus: list[int] = []
        row_offsets = [0]
        pending: dict[str, list[list[tuple[str, float]]]] = {
            name: [] for name in BLOCK_NAMES
        }
        resource_names: dict[str, set[str]] = {name: set() for name in BLOCK_NAMES}
        gpu_counts_by_node: dict[int, int] = {}
        for node in config.gpu_nodes:
            gpu_counts_by_node[node] = gpu_counts_by_node.get(node, 0) + 1
        nic_bw = config.nic_bandwidth_bytes_per_ms
        if not math.isfinite(nic_bw) or nic_bw <= 0:
            raise ValueError("converted NIC bandwidth is outside the finite numeric range")

        for row_index, row in enumerate(normalized_rows):
            for destination_gpu in row.allowed_gpus:
                variable_rows.append(row_index)
                variable_gpus.append(destination_gpu)
                column_terms: dict[str, list[tuple[str, float]]] = {
                    name: [] for name in BLOCK_NAMES
                }
                if row.demand:
                    compute_coeff = row.demand / config.mu_routes_per_ms
                    _add_term(column_terms, resource_names, "compute", f"gpu:{destination_gpu}", compute_coeff)
                    source_node = config.gpu_nodes[row.source_gpu]
                    destination_node = config.gpu_nodes[destination_gpu]
                    if row.source_gpu != destination_gpu:
                        if source_node == destination_node:
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_intra",
                                f"gpu:{row.source_gpu}:tx",
                                row.demand * config.dispatch_bytes_per_route / config.intra_bandwidth_bytes_per_ms,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_intra",
                                f"gpu:{destination_gpu}:rx",
                                row.demand * config.dispatch_bytes_per_route / config.intra_bandwidth_bytes_per_ms,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_intra",
                                f"gpu:{destination_gpu}:tx",
                                row.demand * config.return_bytes_per_route / config.intra_bandwidth_bytes_per_ms,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_intra",
                                f"gpu:{row.source_gpu}:rx",
                                row.demand * config.return_bytes_per_route / config.intra_bandwidth_bytes_per_ms,
                            )
                        else:
                            source_node_bw = gpu_counts_by_node[source_node] * nic_bw
                            destination_node_bw = gpu_counts_by_node[destination_node] * nic_bw
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_inter",
                                f"gpu:{row.source_gpu}:tx",
                                row.demand * config.dispatch_bytes_per_route / nic_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_inter",
                                f"gpu:{destination_gpu}:rx",
                                row.demand * config.dispatch_bytes_per_route / nic_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_inter",
                                f"node:{source_node}:tx",
                                row.demand * config.dispatch_bytes_per_route / source_node_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "dispatch_inter",
                                f"node:{destination_node}:rx",
                                row.demand * config.dispatch_bytes_per_route / destination_node_bw,
                            )
                            # Return payload reverses the dispatch direction.
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_inter",
                                f"gpu:{destination_gpu}:tx",
                                row.demand * config.return_bytes_per_route / nic_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_inter",
                                f"gpu:{row.source_gpu}:rx",
                                row.demand * config.return_bytes_per_route / nic_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_inter",
                                f"node:{destination_node}:tx",
                                row.demand * config.return_bytes_per_route / destination_node_bw,
                            )
                            _add_term(
                                column_terms,
                                resource_names,
                                "return_inter",
                                f"node:{source_node}:rx",
                                row.demand * config.return_bytes_per_route / source_node_bw,
                            )
                for name in BLOCK_NAMES:
                    pending[name].append(column_terms[name])
            row_offsets.append(len(variable_rows))

        blocks: list[SparseBlock] = []
        for name in BLOCK_NAMES:
            names = tuple(sorted(resource_names[name]))
            if not names:
                # A one-element zero block keeps each saddle simplex defined.
                names = ("dummy:zero",)
            index_by_name = {resource: index for index, resource in enumerate(names)}
            columns: list[tuple[tuple[int, float], ...]] = []
            for column_terms in pending[name]:
                terms = tuple(
                    (index_by_name[resource], coefficient)
                    for resource, coefficient in column_terms
                    if coefficient != 0.0
                )
                columns.append(terms)
            blocks.append(SparseBlock(name, names, tuple(columns)))

        return cls(
            normalized_rows,
            config,
            tuple(row_offsets),
            tuple(variable_rows),
            tuple(variable_gpus),
            tuple(blocks),
        )

    @property
    def variable_count(self) -> int:
        return len(self.variable_gpus)

    @property
    def incidence_count(self) -> int:
        return sum(block.incidence_count for block in self.blocks)

    @property
    def resource_count(self) -> int:
        return sum(len(block.resource_names) for block in self.blocks)

    def uniform_shares(self) -> tuple[float, ...]:
        return tuple(
            1.0 / (self.row_offsets[index + 1] - self.row_offsets[index])
            for index in range(len(self.rows))
            for _ in range(self.row_offsets[index + 1] - self.row_offsets[index])
        )

    def source_local_shares(self) -> tuple[float, ...]:
        """Use the source slot when legal; otherwise choose its nearest topology."""

        shares = [0.0] * self.variable_count
        for row_index, row in enumerate(self.rows):
            start, end = self.row_offsets[row_index : row_index + 2]
            if row.source_gpu in row.allowed_gpus:
                selected = row.allowed_gpus.index(row.source_gpu)
            else:
                source_node = self.config.gpu_nodes[row.source_gpu]
                selected = min(
                    range(end - start),
                    key=lambda slot: (
                        self.config.gpu_nodes[self.variable_gpus[start + slot]] != source_node,
                        self.variable_gpus[start + slot],
                    ),
                )
            shares[start + selected] = 1.0
        return tuple(shares)

    def project_shares(self, values: Sequence[float]) -> tuple[float, ...]:
        """Euclidean projection onto the product of row masked simplexes."""

        vector = _validated_vector(values, self.variable_count, "shares")
        projected = [0.0] * self.variable_count
        for row_index in range(len(self.rows)):
            start, end = self.row_offsets[row_index : row_index + 2]
            projected[start:end] = _project_simplex(vector[start:end])
        return tuple(projected)

    def validate_shares(self, shares: Sequence[float], tolerance: float = 1e-9) -> tuple[float, ...]:
        vector = _validated_vector(shares, self.variable_count, "shares")
        for row_index in range(len(self.rows)):
            start, end = self.row_offsets[row_index : row_index + 2]
            row_values = vector[start:end]
            if any(value < -tolerance for value in row_values):
                raise ValueError("shares must be non-negative")
            if abs(sum(row_values) - 1.0) > tolerance:
                raise ValueError("each source-Expert row must sum to one")
        return vector

    def apply_operator(self, shares: Sequence[float]) -> dict[str, tuple[float, ...]]:
        """Return all five sparse load vectors K_s p in milliseconds."""

        vector = _validated_vector(shares, self.variable_count, "shares")
        outputs: dict[str, tuple[float, ...]] = {}
        for block in self.blocks:
            loads = [0.0] * len(block.resource_names)
            for share, column in zip(vector, block.columns):
                if share:
                    for resource, coefficient in column:
                        loads[resource] += coefficient * share
            outputs[block.name] = tuple(loads)
        return outputs

    def apply_adjoint(
        self,
        weights: Mapping[str, Sequence[float]],
    ) -> tuple[float, ...]:
        """Compute K^T lambda; row demand is already inside each coefficient."""

        if not isinstance(weights, Mapping):
            raise TypeError("weights must map block names to resource weights")
        gradients = [0.0] * self.variable_count
        for block in self.blocks:
            values = weights.get(block.name)
            if values is None:
                continue
            lambdas = _validated_vector(values, len(block.resource_names), block.name)
            for column_index, column in enumerate(block.columns):
                gradients[column_index] += sum(
                    coefficient * lambdas[resource]
                    for resource, coefficient in column
                )
        return tuple(gradients)

    def score(self, shares: Sequence[float]) -> ObjectiveScore:
        """Score the hard sum-of-maxima model, including fixed startup once."""

        vector = self.validate_shares(shares)
        loads = self.apply_operator(vector)
        maxima = {name: max(resource_loads, default=0.0) for name, resource_loads in loads.items()}
        alpha_dispatch = self.config.alpha_dispatch_ms
        alpha_return = self.config.alpha_return_ms
        dispatch = alpha_dispatch + maxima["dispatch_intra"] + maxima["dispatch_inter"]
        compute = maxima["compute"]
        returned = alpha_return + maxima["return_intra"] + maxima["return_inter"]
        return ObjectiveScore(
            dispatch_ms=dispatch,
            compute_ms=compute,
            return_ms=returned,
            total_ms=dispatch + compute + returned,
            dispatch_intra_ms=maxima["dispatch_intra"],
            dispatch_inter_ms=maxima["dispatch_inter"],
            return_intra_ms=maxima["return_intra"],
            return_inter_ms=maxima["return_inter"],
            fixed_startup_ms=alpha_dispatch + alpha_return,
            resource_loads_ms=loads,
        )

    def integer_score(self, quotas: Sequence[Sequence[int]]) -> ObjectiveScore:
        if len(quotas) != len(self.rows):
            raise ValueError("quotas must contain one vector per source-Expert row")
        shares: list[float] = []
        for row, row_quotas in zip(self.rows, quotas):
            if len(row_quotas) != len(row.allowed_gpus):
                raise ValueError("quota vector length must match the row mask")
            if any(type(value) is not int or value < 0 for value in row_quotas):
                raise ValueError("quotas must be non-negative integers")
            if sum(row_quotas) != row.demand:
                raise ValueError("integer quotas must conserve each row demand")
            if row.demand:
                shares.extend(value / row.demand for value in row_quotas)
            else:
                shares.extend(1.0 / len(row_quotas) for _ in row_quotas)
        return self.score(shares)

    def objective_value(
        self,
        shares: Sequence[float],
        objective_blocks: Sequence[str] = BLOCK_NAMES,
    ) -> float:
        vector = self.validate_shares(shares)
        active = _validate_objective_blocks(objective_blocks)
        loads = self.apply_operator(vector)
        return self.fixed_startup_ms + sum(max(loads[name], default=0.0) for name in active)

    @property
    def fixed_startup_ms(self) -> float:
        return self.config.alpha_dispatch_ms + self.config.alpha_return_ms

    def dual_lower_bound(
        self,
        dual_weights: Mapping[str, Sequence[float]],
        objective_blocks: Sequence[str] = BLOCK_NAMES,
    ) -> float:
        active = _validate_objective_blocks(objective_blocks)
        validated: dict[str, tuple[float, ...]] = {}
        for name in active:
            if name not in dual_weights:
                raise ValueError(f"dual weights are missing block {name}")
            block = self._blocks_by_name[name]
            values = _validated_vector(dual_weights[name], len(block.resource_names), name)
            if any(value < -1e-10 for value in values) or abs(sum(values) - 1.0) > 1e-8:
                raise ValueError(f"dual weights for {name} must lie on its simplex")
            validated[name] = values
        gradient = self.apply_adjoint(validated)
        bound = self.fixed_startup_ms
        for row_index in range(len(self.rows)):
            start, end = self.row_offsets[row_index : row_index + 2]
            bound += min(gradient[start:end])
        return bound

    def operator_norm_bound(
        self,
        normalization_ms: float,
        objective_blocks: Sequence[str] = BLOCK_NAMES,
    ) -> float:
        """Sparse induced-norm bound sqrt(||K||_1 ||K||_infinity)."""

        active = _validate_objective_blocks(objective_blocks)
        if not _is_finite_number(normalization_ms) or normalization_ms <= 0:
            raise ValueError("normalization_ms must be finite and positive")
        max_column_sum = 0.0
        active_by_name = {name: self._blocks_by_name[name] for name in active}
        for variable_index in range(self.variable_count):
            total = sum(
                coefficient / normalization_ms
                for block in active_by_name.values()
                for _, coefficient in block.columns[variable_index]
            )
            max_column_sum = max(max_column_sum, total)
        max_row_sum = 0.0
        for block in active_by_name.values():
            sums = [0.0] * len(block.resource_names)
            for column in block.columns:
                for resource, coefficient in column:
                    sums[resource] += coefficient / normalization_ms
            max_row_sum = max(max_row_sum, max(sums, default=0.0))
        return math.sqrt(max_column_sum * max_row_sum)


def integerize_rows(
    rows: Sequence[RouteRow],
    shares: Sequence[float],
) -> tuple[tuple[int, ...], ...]:
    """Conservative per-row largest-remainder rounding, without optimality claims."""

    normalized_rows = tuple(rows)
    expected_variables = sum(len(row.allowed_gpus) for row in normalized_rows)
    vector = _validated_vector(shares, expected_variables, "shares")
    allocations: list[tuple[int, ...]] = []
    offset = 0
    for row in normalized_rows:
        width = len(row.allowed_gpus)
        row_shares = vector[offset : offset + width]
        offset += width
        if any(value < -1e-9 for value in row_shares) or abs(sum(row_shares) - 1.0) > 1e-8:
            raise ValueError("shares must be non-negative row simplexes")
        if row.demand == 0:
            allocations.append(tuple(0 for _ in row_shares))
            continue
        exact = [max(0.0, share) * row.demand for share in row_shares]
        floors = [math.floor(value) for value in exact]
        remaining = row.demand - sum(floors)
        if remaining < 0 or remaining > width:
            raise ArithmeticError("floating row rounding violated its remainder bound")
        order = sorted(range(width), key=lambda slot: (-(exact[slot] - floors[slot]), slot))
        for slot in order[:remaining]:
            floors[slot] += 1
        allocations.append(tuple(floors))
    return tuple(allocations)


def solve_projected_mirror_prox(
    problem: SparseSocialProblem,
    *,
    max_iterations: int = 1000,
    absolute_gap_ms: float = 1e-4,
    relative_gap: float = 1e-7,
    initial_shares: Sequence[float] | None = None,
    objective_blocks: Sequence[str] = BLOCK_NAMES,
) -> MirrorProxResult:
    """Solve the frozen saddle model with projected Euclidean Mirror-Prox.

    Returned bounds use physical milliseconds.  ``iteration_cap`` means the
    requested cap was reached before the absolute-plus-relative certificate
    threshold was met; it never denotes exact convergence.
    """

    if not isinstance(problem, SparseSocialProblem):
        raise TypeError("problem must be SparseSocialProblem")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not _is_finite_number(absolute_gap_ms) or absolute_gap_ms < 0:
        raise ValueError("absolute_gap_ms must be finite and non-negative")
    if not _is_finite_number(relative_gap) or relative_gap < 0:
        raise ValueError("relative_gap must be finite and non-negative")
    active = _validate_objective_blocks(objective_blocks)
    if initial_shares is None:
        current_shares = problem.uniform_shares()
    else:
        current_shares = problem.validate_shares(initial_shares)
    initial_loads = problem.apply_operator(current_shares)
    current_dual = {}
    for name in active:
        loads = initial_loads[name]
        selected = max(range(len(loads)), key=lambda index: (loads[index], -index))
        one_hot = [0.0] * len(loads)
        one_hot[selected] = 1.0
        current_dual[name] = tuple(one_hot)

    normalization_ms = problem.score(problem.uniform_shares()).total_ms
    if normalization_ms <= 0:
        normalization_ms = 1.0
    norm_bound = problem.operator_norm_bound(normalization_ms, active)
    if norm_bound == 0.0:
        upper = problem.objective_value(current_shares, active)
        lower = problem.fixed_startup_ms
        for row_index in range(len(problem.rows)):
            start, end = problem.row_offsets[row_index : row_index + 2]
            lower += min(0.0 for _ in range(end - start))
        dual = {name: tuple(values) for name, values in current_dual.items()}
        return MirrorProxResult(
            shares=tuple(current_shares),
            dual_weights=dual,
            status="converged",
            iterations=0,
            upper_bound_ms=upper,
            lower_bound_ms=lower,
            gap_ms=upper - lower,
            step_size=0.0,
            operator_norm_bound=0.0,
            normalization_ms=normalization_ms,
            objective_blocks=active,
            elapsed_ms=0.0,
        )
    step_size = 0.5 / norm_bound

    start_time = time.perf_counter()
    average_share_sums = [0.0] * problem.variable_count
    average_dual_sums = {
        name: [0.0] * len(problem._blocks_by_name[name].resource_names)
        for name in active
    }
    best_shares = tuple(current_shares)
    best_upper = problem.objective_value(best_shares, active)
    best_dual = {name: tuple(values) for name, values in current_dual.items()}
    best_lower = problem.dual_lower_bound(best_dual, active)
    certificate_error = False
    checkpoint_interval = max(1, min(50, max_iterations // 100))
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        current_gradient = problem.apply_adjoint(current_dual)
        trial_shares = problem.project_shares(
            tuple(
                value - step_size * gradient / normalization_ms
                for value, gradient in zip(current_shares, current_gradient)
            )
        )

        current_loads = problem.apply_operator(current_shares)
        trial_dual: dict[str, tuple[float, ...]] = {}
        for name in active:
            trial_dual[name] = _project_simplex(
                tuple(
                    value + step_size * load / normalization_ms
                    for value, load in zip(current_dual[name], current_loads[name])
                )
            )

        trial_gradient = problem.apply_adjoint(trial_dual)
        next_shares = problem.project_shares(
            tuple(
                value - step_size * gradient / normalization_ms
                for value, gradient in zip(current_shares, trial_gradient)
            )
        )
        trial_loads = problem.apply_operator(trial_shares)
        next_dual = {
            name: _project_simplex(
                tuple(
                    value + step_size * load / normalization_ms
                    for value, load in zip(current_dual[name], trial_loads[name])
                )
            )
            for name in active
        }

        for index, value in enumerate(trial_shares):
            average_share_sums[index] += value
        for name in active:
            for index, value in enumerate(trial_dual[name]):
                average_dual_sums[name][index] += value
        current_shares, current_dual = next_shares, next_dual
        iterations = iteration

        if iteration % checkpoint_interval == 0 or iteration == max_iterations:
            averaged_shares = tuple(value / iteration for value in average_share_sums)
            averaged_dual = {
                name: tuple(value / iteration for value in values)
                for name, values in average_dual_sums.items()
            }
            primal_candidates = (current_shares, trial_shares, next_shares, averaged_shares)
            dual_candidates = (current_dual, trial_dual, next_dual, averaged_dual)
            for candidate_shares in primal_candidates:
                candidate_upper = problem.objective_value(candidate_shares, active)
                if candidate_upper < best_upper:
                    best_upper = candidate_upper
                    best_shares = tuple(candidate_shares)
            for candidate_dual in dual_candidates:
                candidate_lower = problem.dual_lower_bound(candidate_dual, active)
                if candidate_lower > best_lower:
                    best_lower = candidate_lower
                    best_dual = {
                        name: tuple(values) for name, values in candidate_dual.items()
                    }
            gap = best_upper - best_lower
            numeric_tolerance = 1e-10 * max(1.0, abs(best_upper), abs(best_lower))
            if gap < -numeric_tolerance:
                certificate_error = True
                break
            threshold = absolute_gap_ms + relative_gap * max(abs(best_upper), abs(best_lower))
            if gap <= threshold:
                break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    gap = best_upper - best_lower
    threshold = absolute_gap_ms + relative_gap * max(abs(best_upper), abs(best_lower))
    return MirrorProxResult(
        shares=best_shares,
        dual_weights=best_dual,
        status=("certificate_error" if certificate_error else
                "converged" if gap <= threshold else "iteration_cap"),
        iterations=iterations,
        upper_bound_ms=best_upper,
        lower_bound_ms=best_lower,
        gap_ms=gap,
        step_size=step_size,
        operator_norm_bound=norm_bound,
        normalization_ms=normalization_ms,
        objective_blocks=active,
        elapsed_ms=elapsed_ms,
    )


def _project_simplex(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("simplex projection requires at least one coordinate")
    vector = _validated_vector(values, len(values), "simplex values")
    if len(vector) == 1:
        return (1.0,)
    ordered = sorted(vector, reverse=True)
    cumulative = 0.0
    threshold = 0.0
    for position, value in enumerate(ordered, 1):
        cumulative += value
        candidate = (cumulative - 1.0) / position
        if value > candidate:
            threshold = candidate
    projected = tuple(max(value - threshold, 0.0) for value in vector)
    total = sum(projected)
    if total == 0.0:
        # This branch is only a guard against extreme floating-point inputs.
        largest = max(range(len(vector)), key=lambda index: vector[index])
        result = [0.0] * len(vector)
        result[largest] = 1.0
        return tuple(result)
    return tuple(value / total for value in projected)


def _validated_vector(values: Sequence[float], expected_length: int, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite numeric sequence") from exc
    if len(vector) != expected_length:
        raise ValueError(f"{name} has length {len(vector)}; expected {expected_length}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_objective_blocks(objective_blocks: Sequence[str]) -> tuple[str, ...]:
    active = tuple(objective_blocks)
    if not active:
        raise ValueError("at least one objective block must be active")
    if len(set(active)) != len(active) or any(name not in BLOCK_NAMES for name in active):
        raise ValueError("objective_blocks must be unique names from BLOCK_NAMES")
    return active


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _add_term(
    columns: dict[str, list[tuple[str, float]]],
    resource_names: dict[str, set[str]],
    block: str,
    resource: str,
    coefficient: float,
) -> None:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("resource incidence coefficient is outside finite non-negative range")
    if coefficient == 0.0:
        return
    columns[block].append((resource, coefficient))
    resource_names[block].add(resource)
