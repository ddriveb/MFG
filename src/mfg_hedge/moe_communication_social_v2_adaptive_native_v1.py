"""Buffered C++17 Mirror-Prox adapter with an optional local line search.

The Python v2 module remains the model and certificate reference. This module
packs one validated sparse problem, calls the separate adaptive-native ABI,
and independently replays returned primal/dual certificates.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

from .moe_communication_social_v2 import (
    BLOCK_NAMES,
    MirrorProxResult,
    SparseSocialProblem,
    _validate_objective_blocks,
)


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().with_suffix(".cpp")
_DEFAULT_COMPILER = (
    _ROOT / ".scratch" / "granite-moe-replica-adapter" / "toolchain"
    / "winlibs-gcc14.2-ucrt-r3" / "mingw64" / "bin" / "g++.exe"
)
_DEFAULT_LIBRARY = (
    _ROOT / "artifacts" / "communication-social-v2-adaptive-20261005-a"
    / "moe_communication_social_v2_adaptive_native_v1.dll"
)
_STATUS_NAMES = {
    0: "converged",
    1: "iteration_cap",
    2: "certificate_error",
    3: "line_search_exhausted",
}
_DLL_DIRECTORY_HANDLES: list[object] = []


@dataclass(frozen=True)
class AdaptiveNativeMirrorProxResult:
    """Native solver result, certificate diagnostics, and call boundaries."""

    result: MirrorProxResult
    adaptive_line_search: bool
    accepted_trials: int
    rejected_trials: int
    total_weights: float
    max_delta: float
    max_accepted_delta: float
    max_floating_slack: float
    initial_step_size: float
    final_step_size: float
    max_step_size: float
    weighted_average_shares: tuple[float, ...]
    weighted_average_dual_weights: Mapping[str, tuple[float, ...]]
    line_search_exhausted: bool
    preparation_ms: float
    native_call_ms: float
    native_solver_ms: float
    compile_ms: float
    full_call_ms: float
    initialization_mode: str
    initial_certificate_gap_ms: float
    initial_certificate_passed: bool


def compile_adaptive_native_library(
    output_path: str | Path | None = None,
    *,
    compiler_path: str | Path | None = None,
) -> Path:
    """Compile the standalone float64 C++ solver using the configured GCC."""

    target = Path(output_path) if output_path is not None else _DEFAULT_LIBRARY
    compiler = Path(compiler_path) if compiler_path is not None else _DEFAULT_COMPILER
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"adaptive native source is missing: {_SOURCE}")
    if not compiler.is_file():
        raise FileNotFoundError(f"C++ compiler is missing: {compiler}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler), "-std=c++17", "-O3", "-DNDEBUG", "-fno-fast-math",
        "-Wall", "-Wextra", "-shared", "-static-libgcc", "-static-libstdc++",
        str(_SOURCE), "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "adaptive native Mirror-Prox compilation failed\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return target


def solve_projected_mirror_prox_adaptive_native(
    problem: SparseSocialProblem,
    *,
    max_iterations: int = 1000,
    absolute_gap_ms: float = 1e-4,
    relative_gap: float = 1e-7,
    initial_shares: Sequence[float] | None = None,
    objective_blocks: Sequence[str] = BLOCK_NAMES,
    initialization_mode: str = "reference",
    initial_certificate_check: bool = False,
    adaptive_line_search: bool = False,
    library_path: str | Path | None = None,
    compiler_path: str | Path | None = None,
) -> AdaptiveNativeMirrorProxResult:
    """Solve the frozen v2 model with the reusable-buffer native solver.

    Fixed mode matches the Python reference's one-hot dual initialization,
    step, averaging and checkpoint behavior. Adaptive mode uses the separate
    local-condition backtracking policy and gamma-weighted ergodic average.
    """

    total_start = time.perf_counter()
    compile_ms = 0.0
    if not isinstance(problem, SparseSocialProblem):
        raise TypeError("problem must be SparseSocialProblem")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not _finite_nonnegative(absolute_gap_ms):
        raise ValueError("absolute_gap_ms must be finite and non-negative")
    if not _finite_nonnegative(relative_gap):
        raise ValueError("relative_gap must be finite and non-negative")
    if initialization_mode not in ("reference", "tied_max_uniform"):
        raise ValueError("initialization_mode must be 'reference' or 'tied_max_uniform'")
    if initial_certificate_check and initialization_mode != "tied_max_uniform":
        raise ValueError("initial certificate check requires tied_max_uniform initialization")
    if type(adaptive_line_search) is not bool:
        raise ValueError("adaptive_line_search must be bool")

    target = Path(library_path) if library_path is not None else _DEFAULT_LIBRARY
    if not target.is_file():
        compile_start = time.perf_counter()
        compile_adaptive_native_library(target, compiler_path=compiler_path)
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

    preparation_start = time.perf_counter()
    active = _validate_objective_blocks(objective_blocks)
    shares = problem.uniform_shares() if initial_shares is None else problem.validate_shares(initial_shares)
    normalization_ms = problem.score(problem.uniform_shares()).total_ms
    if normalization_ms <= 0.0:
        normalization_ms = 1.0
    norm_bound = problem.operator_norm_bound(normalization_ms, active)

    block_resource_offsets = [0]
    packed_column_offsets: list[int] = []
    packed_resource_indices: list[int] = []
    packed_coefficients: list[float] = []
    for name in active:
        block = problem._blocks_by_name[name]
        packed_column_offsets.append(len(packed_resource_indices))
        for column in block.columns:
            for resource_index, coefficient in column:
                packed_resource_indices.append(resource_index)
                packed_coefficients.append(coefficient)
            packed_column_offsets.append(len(packed_resource_indices))
        block_resource_offsets.append(block_resource_offsets[-1] + len(block.resource_names))

    row_offsets_array = _uint64_array(problem.row_offsets)
    dual_offsets_array = _uint64_array(block_resource_offsets)
    column_offsets_array = _uint64_array(packed_column_offsets)
    resource_indices_array = _uint64_array(packed_resource_indices)
    coefficients_array = _double_array(packed_coefficients)
    initial_shares_array = _double_array(shares)
    _retain_compiler_dll_directory(compiler_path)
    native_library = ctypes.CDLL(str(target))
    function = native_library.mfg_mirror_prox_solve_adaptive_v1
    _set_adaptive_abi_signature(function)
    preparation_ms = (time.perf_counter() - preparation_start) * 1000.0

    variable_count = problem.variable_count
    total_resources = block_resource_offsets[-1]
    output_shares = (ctypes.c_double * variable_count)()
    output_dual = (ctypes.c_double * total_resources)()
    output_average_shares = (ctypes.c_double * variable_count)()
    output_average_dual = (ctypes.c_double * total_resources)()
    output_status = ctypes.c_int()
    output_iterations = ctypes.c_uint64()
    output_accepted = ctypes.c_uint64()
    output_rejected = ctypes.c_uint64()
    output_doubles = [ctypes.c_double() for _ in range(14)]
    (upper, lower, gap, step_size, output_norm, elapsed, initial_gap,
     total_weights, max_delta, max_accepted_delta, max_slack, initial_step,
     final_step, max_step) = output_doubles
    output_initial_passed = ctypes.c_int()
    error_buffer = ctypes.create_string_buffer(1024)

    call_start = time.perf_counter()
    return_code = function(
        ctypes.c_uint64(len(problem.rows)),
        ctypes.c_uint64(variable_count),
        ctypes.c_uint64(len(active)),
        row_offsets_array,
        dual_offsets_array,
        column_offsets_array,
        resource_indices_array if packed_resource_indices else None,
        coefficients_array if packed_coefficients else None,
        initial_shares_array,
        ctypes.c_double(normalization_ms),
        ctypes.c_double(problem.fixed_startup_ms),
        ctypes.c_double(norm_bound),
        ctypes.c_uint64(max_iterations),
        ctypes.c_double(absolute_gap_ms),
        ctypes.c_double(relative_gap),
        ctypes.c_int(0 if initialization_mode == "reference" else 1),
        ctypes.c_int(1 if initial_certificate_check else 0),
        ctypes.c_int(1 if adaptive_line_search else 0),
        output_shares,
        output_dual,
        output_average_shares,
        output_average_dual,
        ctypes.byref(output_status),
        ctypes.byref(output_iterations),
        ctypes.byref(output_accepted),
        ctypes.byref(output_rejected),
        ctypes.byref(upper),
        ctypes.byref(lower),
        ctypes.byref(gap),
        ctypes.byref(step_size),
        ctypes.byref(output_norm),
        ctypes.byref(elapsed),
        ctypes.byref(initial_gap),
        ctypes.byref(total_weights),
        ctypes.byref(max_delta),
        ctypes.byref(max_accepted_delta),
        ctypes.byref(max_slack),
        ctypes.byref(initial_step),
        ctypes.byref(final_step),
        ctypes.byref(max_step),
        ctypes.byref(output_initial_passed),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    native_call_ms = (time.perf_counter() - call_start) * 1000.0
    if return_code != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"adaptive native Mirror-Prox failed ({return_code}): {message}")
    if output_status.value not in _STATUS_NAMES:
        raise RuntimeError(f"adaptive native solver returned unknown status {output_status.value}")

    native_shares = tuple(output_shares)
    native_dual_values = tuple(output_dual)
    average_shares = tuple(output_average_shares)
    average_dual_values = tuple(output_average_dual)
    dual_weights = _unpack_blocks(native_dual_values, active, block_resource_offsets)
    average_dual_weights = _unpack_blocks(average_dual_values, active, block_resource_offsets)

    # Replay the best certificate and ergodic feasible points in the Python model.
    problem.validate_shares(native_shares)
    problem.validate_shares(average_shares)
    upper_replay = problem.objective_value(native_shares, active)
    lower_replay = problem.dual_lower_bound(dual_weights, active)
    # Replay the averaged dual through the same lower-bound routine as a
    # feasibility/numerical check; the averaged point is diagnostic only.
    problem.dual_lower_bound(average_dual_weights, active)
    scale = max(1.0, abs(upper_replay), abs(lower_replay))
    replay_tolerance = 2e-10 * scale
    if abs(upper_replay - upper.value) > replay_tolerance:
        raise ArithmeticError("adaptive native primal upper bound failed Python replay")
    if abs(lower_replay - lower.value) > replay_tolerance:
        raise ArithmeticError("adaptive native dual lower bound failed Python replay")
    if abs((upper_replay - lower_replay) - gap.value) > 2.0 * replay_tolerance:
        raise ArithmeticError("adaptive native primal-dual gap failed Python replay")
    expected_initial_step = 0.5 / norm_bound if norm_bound else 0.0
    if initial_step.value != expected_initial_step:
        raise ArithmeticError("adaptive native solver returned an unexpected initial step")
    if adaptive_line_search:
        if step_size.value != final_step.value:
            raise ArithmeticError("adaptive native solver's final step diagnostic disagrees")
        if step_size.value < 0.0 or step_size.value > 1e4 * expected_initial_step:
            raise ArithmeticError("adaptive native solver returned an invalid final step")
        if output_accepted.value == 0 and total_weights.value != 0.0:
            raise ArithmeticError("adaptive native solver returned weight without accepted steps")
    elif step_size.value != expected_initial_step:
        raise ArithmeticError("fixed native solver returned an unexpected reference step")
    diagnostics = (total_weights.value, max_delta.value, max_accepted_delta.value,
                   max_slack.value, initial_step.value, final_step.value, max_step.value)
    if any(not math.isfinite(value) for value in diagnostics):
        raise ArithmeticError("adaptive native solver returned non-finite diagnostics")

    result = MirrorProxResult(
        shares=native_shares,
        dual_weights=dual_weights,
        status=_STATUS_NAMES[output_status.value],
        iterations=output_iterations.value,
        upper_bound_ms=upper.value,
        lower_bound_ms=lower.value,
        gap_ms=gap.value,
        step_size=step_size.value,
        operator_norm_bound=output_norm.value,
        normalization_ms=normalization_ms,
        objective_blocks=active,
        elapsed_ms=elapsed.value,
    )
    full_call_ms = (time.perf_counter() - total_start) * 1000.0
    return AdaptiveNativeMirrorProxResult(
        result=result,
        adaptive_line_search=adaptive_line_search,
        accepted_trials=output_accepted.value,
        rejected_trials=output_rejected.value,
        total_weights=total_weights.value,
        max_delta=max_delta.value,
        max_accepted_delta=max_accepted_delta.value,
        max_floating_slack=max_slack.value,
        initial_step_size=initial_step.value,
        final_step_size=final_step.value,
        max_step_size=max_step.value,
        weighted_average_shares=average_shares,
        weighted_average_dual_weights=average_dual_weights,
        line_search_exhausted=result.status == "line_search_exhausted",
        preparation_ms=preparation_ms,
        native_call_ms=native_call_ms,
        native_solver_ms=elapsed.value,
        compile_ms=compile_ms,
        full_call_ms=full_call_ms,
        initialization_mode=initialization_mode,
        initial_certificate_gap_ms=initial_gap.value,
        initial_certificate_passed=bool(output_initial_passed.value),
    )


def _unpack_blocks(values, active, offsets):
    return {
        name: tuple(values[offsets[index]:offsets[index + 1]])
        for index, name in enumerate(active)
    }


def _set_adaptive_abi_signature(function: ctypes._CFuncPtr) -> None:
    u64p = ctypes.POINTER(ctypes.c_uint64)
    dp = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        u64p, u64p, u64p, u64p, dp, dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint64, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        dp, dp, dp, dp, ctypes.POINTER(ctypes.c_int),
        u64p, u64p, u64p, dp, dp, dp, dp, dp, dp, dp, dp, dp, dp,
        dp, dp, dp, dp, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int


def _uint64_array(values: Sequence[int]) -> ctypes.Array:
    return (ctypes.c_uint64 * len(values))(*values)


def _double_array(values: Sequence[float]) -> ctypes.Array:
    return (ctypes.c_double * len(values))(*values)


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _retain_compiler_dll_directory(compiler_path: str | Path | None) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    compiler = Path(compiler_path) if compiler_path is not None else _DEFAULT_COMPILER
    if compiler.parent.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(compiler.parent)))
