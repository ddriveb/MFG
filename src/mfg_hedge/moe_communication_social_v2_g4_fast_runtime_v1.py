"""Cached native runtime for Ticket07 communication-social v2 warm starts.

The topology-specific network remains installed Torch CPU float64. It predicts
feasible primal/dual initials; a resident C++17 runtime performs fixed-budget
Mirror-Prox, source/mask expansion, and original-row integerization.
"""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Mapping, Sequence


from .moe_communication_social_v2 import (
    BLOCK_NAMES, CommunicationConfig, RouteRow, SparseSocialProblem,
)
ROUTES_PER_SOURCE_FEATURE_SCALE = 8128.0
TOPOLOGY_WIDTHS = {"single_node_g4": (8, 1, 4, 8, 1)}
from .moe_communication_social_v2_quality_runner_v1 import (
    _ARM_IDS, _MAX_SAFE_ROUTE_COUNT, _QUALITY_DLL, _COMPILER,
    _retain_compiler_dll_directory, local_first_source_quotas, materialize_source_quotas,
)
from .moe_communication_social_v2_reduction_v1 import reduce_equal_source_masks


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().with_name("moe_communication_social_v2_g4_fast_runtime_v1.cpp")
_DEFAULT_LIBRARY = (
    _ROOT / ".scratch" / "communication-aware-social-v2" / "native-cache"
    / f"moe_communication_social_v2_fast_runtime_v1-{hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]}.dll"
)
_MAX_SNAPSHOTS = 16
_DLL_DIRECTORY_HANDLES: list[object] = []


@dataclass(frozen=True)
class FastCorrectionPlan:
    reduced_shares: tuple[float, ...]
    dual_weights: Mapping[str, tuple[float, ...]]
    quotas_by_row: tuple[tuple[int, ...], ...]
    reduced_demands: tuple[int, ...]
    normalization_ms: float
    operator_norm_bound: float
    setup_ms: float
    model_ms: float
    native_ms: float
    return_ms: float


def compile_fast_runtime_library(
    output_path: str | Path | None = None, *, compiler_path: str | Path | None = None,
) -> Path:
    """Compile the isolated C++17 runtime using the approved GCC toolchain."""
    target = Path(output_path) if output_path is not None else _DEFAULT_LIBRARY
    compiler = Path(compiler_path) if compiler_path is not None else _COMPILER
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"native fast runtime source is missing: {_SOURCE}")
    if not compiler.is_file():
        raise FileNotFoundError(f"configured GCC compiler is missing: {compiler}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler), "-std=c++17", "-O3", "-DNDEBUG", "-fno-fast-math",
        "-ffp-contract=off", "-Wall", "-Wextra", "-shared",
        "-static-libgcc", "-static-libstdc++", str(_SOURCE), "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "native fast runtime compilation failed\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return target


class G4FastWarmStartRuntime:
    """Single-thread-owned runtime for a fixed G4 topology and 128-row layout."""

    def __init__(
        self, rows: Sequence[RouteRow], config: CommunicationConfig, *,
        library_path: str | Path | None = None, compiler_path: str | Path | None = None,
    ) -> None:
        cache_start = time.perf_counter()
        self._owner_thread = threading.get_ident()
        self._closed = False
        self._rows = tuple(rows)
        self._config = config
        self._validate_fixed_layout()
        self._reduction = reduce_equal_source_masks(
            SparseSocialProblem.from_rows(self._rows, config)
        )
        self._row_groups = self._reduction.row_groups
        self._slot_maps = self._reduction.original_slot_to_group_slot
        unit_layout = self._reduction.problem
        self._group_count = len(unit_layout.rows)
        self._variable_count = unit_layout.variable_count
        if self._group_count != 16 or self._variable_count != 32:
            raise ValueError("native runtime requires the reduced G4 16-row/32-variable layout")
        self._topology_id = self._topology_for(config)
        self._block_widths = tuple(len(block.resource_names) for block in unit_layout.blocks)
        if self._block_widths != TOPOLOGY_WIDTHS[self._topology_id]:
            raise ValueError("resource block widths differ from the frozen G4 topology")
        self._dual_offsets = [0]
        for width in self._block_widths:
            self._dual_offsets.append(self._dual_offsets[-1] + width)
        self._dual_count = self._dual_offsets[-1]

        # Rebuild once with unit demand so incidence factors and masks are
        # independent of the constructor rows' initial demand snapshot.
        canonical_unit_rows = tuple(
            RouteRow(row.source_gpu, index, 1, row.allowed_gpus)
            for index, row in enumerate(unit_layout.rows)
        )
        unit_problem = SparseSocialProblem.from_rows(canonical_unit_rows, config)
        packed = self._pack_unit_layout(unit_problem)
        target = Path(library_path) if library_path is not None else _DEFAULT_LIBRARY
        compile_ms = 0.0
        if not target.is_file():
            start = time.perf_counter()
            compile_fast_runtime_library(target, compiler_path=compiler_path)
            compile_ms = (time.perf_counter() - start) * 1000.0
        self.library_path = target
        self._retain_dll_directory(compiler_path)
        self._native = ctypes.CDLL(str(target))
        self._set_native_signatures(self._native)
        self._error_buffer = ctypes.create_string_buffer(1024)
        self._row_offsets = _uint64_array(unit_problem.row_offsets)
        self._dual_offset_array = _uint64_array(packed["dual_offsets"])
        self._column_offsets = _uint64_array(packed["column_offsets"])
        self._resource_indices = _uint64_array(packed["resource_indices"])
        self._numerators = _double_array(packed["numerators"])
        self._denominators = _double_array(packed["denominators"])
        self._original_row_offsets = _uint64_array(self._reduction.original.row_offsets)
        self._row_group_array = _uint64_array(self._row_groups)
        self._slot_map_array = _uint64_array(
            value for slots in self._slot_maps for value in slots
        )
        self._handle = self._native.mfg_fast_create_v1(
            ctypes.c_uint64(len(unit_problem.rows)),
            ctypes.c_uint64(unit_problem.variable_count),
            ctypes.c_uint64(len(unit_problem.blocks)),
            self._row_offsets, self._dual_offset_array, self._column_offsets,
            self._resource_indices if packed["resource_indices"] else None,
            self._numerators if packed["numerators"] else None,
            self._denominators if packed["denominators"] else None,
            ctypes.c_uint64(len(packed["resource_indices"])),
            ctypes.c_double(unit_problem.fixed_startup_ms),
            ctypes.c_uint64(len(self._rows)),
            ctypes.c_uint64(len(self._reduction.original.variable_gpus)),
            self._original_row_offsets, self._row_group_array, self._slot_map_array,
            self._error_buffer, ctypes.c_size_t(len(self._error_buffer)),
        )
        if not self._handle:
            raise RuntimeError(f"native fast runtime setup failed: {self._error_text()}")
        self._demand_buffer = _double_array((0.0,) * self._group_count)
        self._original_demand_buffer = (ctypes.c_uint64 * len(self._rows))()
        self._primal_input = _double_array((0.0,) * self._variable_count)
        self._dual_input = _double_array((0.0,) * self._dual_count)
        self._step_buffer = (ctypes.c_uint64 * _MAX_SNAPSHOTS)()
        self._output_primal = (ctypes.c_double * (_MAX_SNAPSHOTS * self._variable_count))()
        self._output_dual = (ctypes.c_double * (_MAX_SNAPSHOTS * self._dual_count))()
        self._output_quotas = (
            ctypes.c_int64 * (_MAX_SNAPSHOTS * len(self._reduction.original.variable_gpus))
        )()
        self._output_reference = ctypes.c_double()
        self._output_norm = ctypes.c_double()
        self._feature_stats = None
        self._feature_mean_tensor = None
        self._feature_std_tensor = None
        self._cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0
        self._compile_ms = compile_ms

    @classmethod
    def from_rows(
        cls, rows: Sequence[RouteRow], config: CommunicationConfig, *,
        library_path: str | Path | None = None, compiler_path: str | Path | None = None,
    ) -> "G4FastWarmStartRuntime":
        return cls(rows, config, library_path=library_path, compiler_path=compiler_path)

    @property
    def topology_id(self) -> str:
        return self._topology_id

    @property
    def cache_setup_ms(self) -> float:
        return self._cache_setup_ms

    @property
    def compile_ms(self) -> float:
        return self._compile_ms

    @property
    def is_open(self) -> bool:
        return not self._closed

    def aggregate_demands(self, demands: Sequence[int]) -> tuple[int, ...]:
        self._check_owner_open()
        values = self._validated_demands(demands)
        grouped = [0] * self._group_count
        for demand, group in zip(values, self._row_groups):
            grouped[group] += demand
            if grouped[group] > _MAX_SAFE_ROUTE_COUNT:
                raise ValueError("reduced group demand exceeds the frozen exact-count range")
        return tuple(grouped)

    def expand_shares(self, reduced_shares: Sequence[float]) -> tuple[float, ...]:
        """Expand using the cached explicit original-slot map."""
        self._check_owner_open()
        values = _finite_values(reduced_shares, self._variable_count, "reduced_shares")
        expanded = []
        for row_index, group in enumerate(self._row_groups):
            group_start = self._reduction.problem.row_offsets[group]
            for group_slot in self._slot_maps[row_index]:
                expanded.append(values[group_start + group_slot])
        return _validate_simplexes(
            expanded, self._reduction.original.row_offsets, "expanded shares"
        )

    def correct(
        self, demands: Sequence[int], initial_shares: Sequence[float],
        initial_dual: Mapping[str, Sequence[float]] | Sequence[Sequence[float]], *,
        steps: int | Sequence[int],
    ) -> dict[int, tuple[tuple[float, ...], dict[str, tuple[float, ...]]]]:
        """Correct fresh feasible initials for current original-row demands."""
        result, _quotas, _ref, _norm = self._run_correction(
            demands, initial_shares, initial_dual, steps
        )
        return result

    def plan(
        self, demands: Sequence[int], model: FixedBudgetWarmStartNet,
        feature_mean: Sequence[float], feature_std: Sequence[float], *,
        steps: int = 4,
    ) -> FastCorrectionPlan:
        """Run Torch inference, fixed correction, and native original-row quotas."""
        import torch
        from .moe_communication_social_v2_g4_warmstart_v1 import FixedBudgetWarmStartNet
        self._check_owner_open()
        setup_start = time.perf_counter()
        original_demands = self._validated_demands(demands)
        reduced = [0] * self._group_count
        for demand, group in zip(original_demands, self._row_groups):
            reduced[group] += demand
            if reduced[group] > _MAX_SAFE_ROUTE_COUNT:
                raise ValueError("reduced group demand exceeds the frozen exact-count range")
        mean = _finite_values(feature_mean, 16, "feature_mean")
        scale = _finite_values(feature_std, 16, "feature_std")
        if any(value <= 0.0 for value in scale):
            raise ValueError("feature_std values must be positive")
        if type(steps) is not int or not 0 <= steps <= 1000000:
            raise ValueError("steps must be a non-negative integer no greater than 1000000")
        if not isinstance(model, FixedBudgetWarmStartNet):
            raise TypeError("model must be a FixedBudgetWarmStartNet checkpoint")
        if model.topology_id != self._topology_id:
            raise ValueError("model topology differs from the cached runtime")
        if any(p.device.type != "cpu" or p.dtype != torch.float64 for p in model.parameters()):
            raise ValueError("warm-start model parameters must remain on CPU in float64")
        if torch.get_num_threads() != 1:
            raise ValueError("warm-start inference requires Torch CPU thread count 1")
        if self._feature_stats != (mean, scale):
            self._feature_mean_tensor = torch.tensor(mean, dtype=torch.float64, device="cpu")
            self._feature_std_tensor = torch.tensor(scale, dtype=torch.float64, device="cpu")
            self._feature_stats = (mean, scale)
        raw = torch.tensor(
            [value / ROUTES_PER_SOURCE_FEATURE_SCALE for value in reduced],
            dtype=torch.float64, device="cpu",
        )
        setup_ms = (time.perf_counter() - setup_start) * 1000.0

        model_start = time.perf_counter()
        features = (
            (raw - self._feature_mean_tensor) / self._feature_std_tensor
        ).reshape(1, 16)
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad():
                initial_p_tensor, dual_tensors = model(features)
        finally:
            if was_training:
                model.train()
        model_ms = (time.perf_counter() - model_start) * 1000.0
        if tuple(initial_p_tensor.shape) != (1, 32) or len(dual_tensors) != 5:
            raise ValueError("warm-start model returned an invalid primal/dual shape")
        initial_p = tuple(initial_p_tensor[0].tolist())
        initial_dual = {
            name: tuple(values[0].tolist())
            for name, values in zip(BLOCK_NAMES, dual_tensors)
        }
        native_start = time.perf_counter()
        output, quota_snapshots, reference, norm = self._run_correction(
            original_demands, initial_p, initial_dual, steps
        )
        native_ms = (time.perf_counter() - native_start) * 1000.0
        return_start = time.perf_counter()
        reduced_shares, dual_weights = output[steps]
        quota_flat = quota_snapshots[steps]
        quotas = tuple(
            tuple(quota_flat[i * 2 : i * 2 + 2]) for i in range(len(self._rows))
        )
        return FastCorrectionPlan(
            reduced_shares=reduced_shares,
            dual_weights=dual_weights,
            quotas_by_row=quotas,
            reduced_demands=tuple(reduced),
            normalization_ms=reference,
            operator_norm_bound=norm,
            setup_ms=setup_ms,
            model_ms=model_ms,
            native_ms=native_ms,
            return_ms=(time.perf_counter() - return_start) * 1000.0,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._check_owner_thread()
        if self._handle:
            self._native.mfg_fast_destroy_v1(self._handle)
            self._handle = None
        self._native = None
        self._closed = True

    def __enter__(self) -> "G4FastWarmStartRuntime":
        self._check_owner_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _run_correction(self, demands, initial_shares, initial_dual, steps):
        self._check_owner_open()
        original = self._validated_demands(demands)
        reduced = self.aggregate_demands(original)
        budgets = _normalize_steps(steps)
        primal = _validate_simplexes(
            _finite_values(initial_shares, self._variable_count, "initial_shares"),
            self._reduction.problem.row_offsets, "initial_shares",
        )
        dual_blocks = _dual_blocks(initial_dual, self._block_widths)
        flat_dual = tuple(value for block in dual_blocks for value in block)
        for i, value in enumerate(original):
            self._original_demand_buffer[i] = value
        for i, value in enumerate(reduced):
            self._demand_buffer[i] = float(value)
        for i, value in enumerate(primal):
            self._primal_input[i] = value
        for i, value in enumerate(flat_dual):
            self._dual_input[i] = value
        for i, value in enumerate(budgets):
            self._step_buffer[i] = value
        self._error_buffer.value = b""
        code = self._native.mfg_fast_run_v1(
            self._handle, self._demand_buffer, self._original_demand_buffer,
            self._primal_input, self._dual_input, self._step_buffer,
            ctypes.c_uint64(len(budgets)), self._output_primal,
            self._output_dual, self._output_quotas,
            ctypes.byref(self._output_reference), ctypes.byref(self._output_norm),
            self._error_buffer, ctypes.c_size_t(len(self._error_buffer)),
        )
        if code != 0:
            raise RuntimeError(f"native fixed-budget correction failed ({code}): {self._error_text()}")
        outputs, quota_snapshots = {}, {}
        slot_count = len(self._reduction.original.variable_gpus)
        for snapshot, step in enumerate(budgets):
            p_start = snapshot * self._variable_count
            d_start = snapshot * self._dual_count
            p = tuple(float(self._output_primal[p_start + i]) for i in range(self._variable_count))
            p = _validate_simplexes(p, self._reduction.problem.row_offsets, "native shares")
            flat = tuple(float(self._output_dual[d_start + i]) for i in range(self._dual_count))
            blocks = {}
            for block_index, name in enumerate(BLOCK_NAMES):
                start, end = self._dual_offsets[block_index : block_index + 2]
                block = flat[start:end]
                if any(not math.isfinite(x) or x < -1e-12 for x in block) or abs(sum(block) - 1.0) > 1e-9:
                    raise ArithmeticError(f"native dual[{name}] is not a feasible simplex")
                blocks[name] = block
            q_start = snapshot * slot_count
            q_flat = tuple(int(self._output_quotas[q_start + i]) for i in range(slot_count))
            row_quotas = tuple(q_flat[i * 2 : i * 2 + 2] for i in range(len(self._rows)))
            for demand, row_quota in zip(original, row_quotas):
                if any(value < 0 for value in row_quota) or sum(row_quota) != demand:
                    raise ArithmeticError("native original-row rounding failed conservation")
            outputs[step] = (p, blocks)
            quota_snapshots[step] = q_flat
        return (
            outputs, quota_snapshots,
            float(self._output_reference.value), float(self._output_norm.value),
        )

    def _validate_fixed_layout(self) -> None:
        if not isinstance(self._config, CommunicationConfig):
            raise TypeError("config must be CommunicationConfig")
        if len(self._config.gpu_nodes) != 4 or len(self._rows) != 128:
            raise ValueError("native warm-start runtime requires G4 and 128 source rows")
        seen = set()
        for row in self._rows:
            if not isinstance(row, RouteRow):
                raise TypeError("each row must be RouteRow")
            if type(row.source_gpu) is not int or not 0 <= row.source_gpu < 4:
                raise ValueError("row source_gpu must name one of four GPUs")
            if type(row.expert_id) is not int or not 0 <= row.expert_id < 32:
                raise ValueError("expert_id must name one of 32 frozen Experts")
            if type(row.demand) is not int or row.demand < 0:
                raise ValueError("row demand must be a non-negative integer")
            key = (row.source_gpu, row.expert_id)
            if key in seen:
                raise ValueError("source-Expert rows must be unique")
            seen.add(key)
            if frozenset(row.allowed_gpus) != frozenset(
                (row.expert_id % 4, (row.expert_id + 1) % 4)
            ):
                raise ValueError("row legal mask differs from frozen adjacent-ring placement")
        if len(seen) != 128:
            raise ValueError("fixed layout must contain every source-Expert row")
        self._topology_for(self._config)

    @staticmethod
    def _topology_for(config: CommunicationConfig) -> str:
        if config.gpu_nodes == (0, 0, 0, 0):
            return "single_node_g4"
        raise ValueError("only the frozen G4 topologies are supported")

    def _pack_unit_layout(self, problem: SparseSocialProblem) -> dict[str, list]:
        offsets, indices, numerators, denominators = [], [], [], []
        for block in problem.blocks:
            offsets.append(len(indices))
            for variable, column in enumerate(block.columns):
                row = problem.rows[problem.variable_rows[variable]]
                destination = problem.variable_gpus[variable]
                for resource_index, _coefficient in column:
                    numerator, denominator = self._coefficient_factors(
                        block.name, row.source_gpu, destination,
                        block.resource_names[resource_index],
                    )
                    indices.append(resource_index)
                    numerators.append(numerator)
                    denominators.append(denominator)
                offsets.append(len(indices))
        return {
            "dual_offsets": [0, *self._dual_offsets[1:]],
            "column_offsets": offsets,
            "resource_indices": indices,
            "numerators": numerators,
            "denominators": denominators,
        }

    def _coefficient_factors(self, block, source_gpu, destination_gpu, resource_name):
        config = self._config
        if block == "compute":
            return 1.0, config.mu_routes_per_ms
        if block == "dispatch_intra":
            return config.dispatch_bytes_per_route, config.intra_bandwidth_bytes_per_ms
        if block == "return_intra":
            return config.return_bytes_per_route, config.intra_bandwidth_bytes_per_ms
        if block in ("dispatch_inter", "return_inter"):
            numerator = config.dispatch_bytes_per_route if block == "dispatch_inter" else config.return_bytes_per_route
            if resource_name.startswith("node:"):
                node_id = int(resource_name.split(":")[1])
                node_width = sum(node == node_id for node in config.gpu_nodes)
                denominator = node_width * config.nic_bandwidth_bytes_per_ms
            else:
                denominator = config.nic_bandwidth_bytes_per_ms
            return numerator, denominator
        raise ValueError(f"unknown resource block {block!r}")

    def _validated_demands(self, demands: Sequence[int]) -> tuple[int, ...]:
        try:
            values = tuple(demands)
        except TypeError as exc:
            raise TypeError("demands must be a sequence of source-row integers") from exc
        if len(values) != len(self._rows):
            raise ValueError("demands must contain one count per fixed source row")
        if any(type(value) is not int or value < 0 or value > _MAX_SAFE_ROUTE_COUNT for value in values):
            raise ValueError("demands must be non-negative exact integers in the frozen range")
        if sum(values) > _MAX_SAFE_ROUTE_COUNT:
            raise ValueError("global route total exceeds the frozen exact-count range")
        return values

    def _set_native_signatures(self, native):
        u64p = ctypes.POINTER(ctypes.c_uint64)
        dp = ctypes.POINTER(ctypes.c_double)
        native.mfg_fast_create_v1.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            u64p, u64p, u64p, u64p, dp, dp, ctypes.c_uint64, ctypes.c_double,
            ctypes.c_uint64, ctypes.c_uint64, u64p, u64p, u64p,
            ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.mfg_fast_create_v1.restype = ctypes.c_void_p
        native.mfg_fast_destroy_v1.argtypes = [ctypes.c_void_p]
        native.mfg_fast_destroy_v1.restype = None
        native.mfg_fast_run_v1.argtypes = [
            ctypes.c_void_p, dp, u64p, dp, dp, u64p, ctypes.c_uint64,
            dp, dp, ctypes.POINTER(ctypes.c_int64), dp, dp,
            ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        native.mfg_fast_run_v1.restype = ctypes.c_int

    def _retain_dll_directory(self, compiler_path):
        if hasattr(os, "add_dll_directory"):
            compiler = Path(compiler_path) if compiler_path is not None else _COMPILER
            if compiler.parent.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(compiler.parent)))

    def _error_text(self):
        return self._error_buffer.value.decode("utf-8", errors="replace")

    def _check_owner_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("native warm-start runtime is owned by another thread")

    def _check_owner_open(self):
        self._check_owner_thread()
        if self._closed:
            raise RuntimeError("native warm-start runtime is closed")


def _dual_blocks(initial_dual, widths):
    if isinstance(initial_dual, Mapping):
        if set(initial_dual) != set(BLOCK_NAMES):
            raise ValueError("initial_dual must contain exactly the five objective blocks")
        raw = tuple(initial_dual[name] for name in BLOCK_NAMES)
    else:
        raw = tuple(initial_dual)
        if len(raw) != len(BLOCK_NAMES):
            raise ValueError("initial_dual must contain five resource-simplex blocks")
    blocks = []
    for name, values, width in zip(BLOCK_NAMES, raw, widths):
        block = _finite_values(values, width, f"initial_dual[{name}]")
        if any(value < -1e-12 for value in block) or abs(sum(block) - 1.0) > 1e-9:
            raise ValueError(f"initial_dual[{name}] must lie on its unit simplex")
        blocks.append(block)
    return tuple(blocks)


def _finite_values(values, expected, name):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain finite numeric values") from exc
    if len(result) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_simplexes(values, row_offsets, name):
    result = tuple(values)
    for row, (start, end) in enumerate(zip(row_offsets, row_offsets[1:])):
        simplex = result[start:end]
        if any(value < -1e-12 for value in simplex) or abs(sum(simplex) - 1.0) > 1e-9:
            raise ValueError(f"{name} row {row} must be a unit simplex")
    return result


def _normalize_steps(steps):
    if type(steps) is int:
        budgets = (steps,)
    else:
        try:
            budgets = tuple(steps)
        except TypeError as exc:
            raise TypeError("steps must be an integer or increasing integer sequence") from exc
    if not budgets:
        raise ValueError("at least one fixed budget is required")
    if any(type(value) is not int or value < 0 or value > 1000000 for value in budgets):
        raise ValueError("fixed budgets must be integers from zero through 1000000")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("fixed budgets must be strictly increasing")
    if len(budgets) > _MAX_SNAPSHOTS:
        raise ValueError(f"at most {_MAX_SNAPSHOTS} fixed budgets can share one native call")
    return budgets


def _uint64_array(values):
    items = tuple(values)
    return (ctypes.c_uint64 * len(items))(*items)


def _double_array(values):
    items = tuple(values)
    return (ctypes.c_double * len(items))(*items)

