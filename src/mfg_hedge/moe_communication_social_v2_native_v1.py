"""C++17 accelerator for the frozen communication-social-v2 Mirror-Prox loop.

The Python v2 module remains the mathematical reference. This adapter packs a
single validated SparseSocialProblem, invokes the native solver once, then
replays the returned primal and dual certificates through the reference model.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence

from .moe_communication_social_v2 import (
    BLOCK_NAMES,
    MirrorProxResult,
    SparseSocialProblem,
    _validate_objective_blocks,
)


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Path(__file__).resolve().with_suffix(".cpp")
_DEFAULT_COMPILER = (
    _ROOT
    / ".scratch"
    / "granite-moe-replica-adapter"
    / "toolchain"
    / "winlibs-gcc14.2-ucrt-r3"
    / "mingw64"
    / "bin"
    / "g++.exe"
)
_DEFAULT_LIBRARY = (
    _ROOT
    / "artifacts"
    / "communication-social-v2-native-20261005-a"
    / "moe_communication_social_v2_native_v1.dll"
)
_STATUS_NAMES = {0: "converged", 1: "iteration_cap", 2: "certificate_error"}
_DLL_DIRECTORY_HANDLES: list[object] = []


@dataclass(frozen=True)
class NativeMirrorProxResult:
    """Native solver result with setup and call timing boundaries."""

    result: MirrorProxResult
    preparation_ms: float
    native_call_ms: float
    native_solver_ms: float
    compile_ms: float
    full_call_ms: float
    initialization_mode: str
    initial_certificate_gap_ms: float
    initial_certificate_passed: bool


def compile_native_library(
    output_path: str | Path | None = None,
    *,
    compiler_path: str | Path | None = None,
) -> Path:
    """Compile the standalone C++17 solver with the supplied WinLibs GCC."""

    target = Path(output_path) if output_path is not None else _DEFAULT_LIBRARY
    compiler = Path(compiler_path) if compiler_path is not None else _DEFAULT_COMPILER
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"native Mirror-Prox source is missing: {_SOURCE}")
    if not compiler.is_file():
        raise FileNotFoundError(f"C++ compiler is missing: {compiler}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler),
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
        "-shared",
        "-static-libgcc",
        "-static-libstdc++",
        str(_SOURCE),
        "-o",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "native Mirror-Prox compilation failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return target


def solve_projected_mirror_prox_native(
    problem: SparseSocialProblem,
    *,
    max_iterations: int = 1000,
    absolute_gap_ms: float = 1e-4,
    relative_gap: float = 1e-7,
    initial_shares: Sequence[float] | None = None,
    objective_blocks: Sequence[str] = BLOCK_NAMES,
    initialization_mode: str = "reference",
    initial_certificate_check: bool = False,
    library_path: str | Path | None = None,
    compiler_path: str | Path | None = None,
) -> NativeMirrorProxResult:
    """Solve the current v2 model with a one-call packed sparse C++ ABI.

    ``reference`` uses the original one-hot maximum-load dual initialization
    and no initial early stop, matching the Python solver's default behavior.
    ``tied_max_uniform`` is an optional feasible dual seed over exactly tied
    maximum-load resources. Its initial certificate check may return at zero
    iterations, but it uses the unchanged physical objective and tolerance.
    """

    total_start = time.perf_counter()
    compile_ms = 0.0
    if not isinstance(problem, SparseSocialProblem):
        raise TypeError("problem must be SparseSocialProblem")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not _is_finite_nonnegative(absolute_gap_ms):
        raise ValueError("absolute_gap_ms must be finite and non-negative")
    if not _is_finite_nonnegative(relative_gap):
        raise ValueError("relative_gap must be finite and non-negative")
    if initialization_mode not in ("reference", "tied_max_uniform"):
        raise ValueError("initialization_mode must be 'reference' or 'tied_max_uniform'")
    if initial_certificate_check and initialization_mode != "tied_max_uniform":
        raise ValueError("initial certificate check requires tied_max_uniform initialization")

    target_path = Path(library_path) if library_path is not None else _DEFAULT_LIBRARY
    if not target_path.is_file():
        compile_start = time.perf_counter()
        compile_native_library(target_path, compiler_path=compiler_path)
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

    preparation_start = time.perf_counter()
    active = _validate_objective_blocks(objective_blocks)
    if initial_shares is None:
        shares = problem.uniform_shares()
    else:
        shares = problem.validate_shares(initial_shares)
    normalization_ms = problem.score(problem.uniform_shares()).total_ms
    if normalization_ms <= 0.0:
        normalization_ms = 1.0
    norm_bound = problem.operator_norm_bound(normalization_ms, active)

    row_offsets = tuple(problem.row_offsets)
    block_resource_offsets = [0]
    packed_column_offsets: list[int] = []
    packed_resource_indices: list[int] = []
    packed_coefficients: list[float] = []
    for name in active:
        block = problem._blocks_by_name[name]
        incidence_cursor = len(packed_resource_indices)
        packed_column_offsets.append(incidence_cursor)
        for column in block.columns:
            for resource_index, coefficient in column:
                packed_resource_indices.append(resource_index)
                packed_coefficients.append(coefficient)
            incidence_cursor = len(packed_resource_indices)
            packed_column_offsets.append(incidence_cursor)
        block_resource_offsets.append(block_resource_offsets[-1] + len(block.resource_names))

    row_offsets_array = _uint64_array(row_offsets)
    dual_offsets_array = _uint64_array(block_resource_offsets)
    column_offsets_array = _uint64_array(packed_column_offsets)
    resource_indices_array = _uint64_array(packed_resource_indices)
    coefficients_array = _double_array(packed_coefficients)
    initial_shares_array = _double_array(shares)
    _retain_compiler_dll_directory(compiler_path)
    native_library = ctypes.CDLL(str(target_path))
    function = native_library.mfg_mirror_prox_solve_v1
    _set_abi_signature(function)
    preparation_ms = (time.perf_counter() - preparation_start) * 1000.0

    variable_count = problem.variable_count
    resource_count = block_resource_offsets[-1]
    output_shares = (ctypes.c_double * variable_count)()
    output_dual_weights = (ctypes.c_double * resource_count)()
    output_status = ctypes.c_int()
    output_iterations = ctypes.c_uint64()
    output_upper = ctypes.c_double()
    output_lower = ctypes.c_double()
    output_gap = ctypes.c_double()
    output_step = ctypes.c_double()
    output_norm = ctypes.c_double()
    output_elapsed = ctypes.c_double()
    output_initial_gap = ctypes.c_double()
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
        output_shares,
        output_dual_weights,
        ctypes.byref(output_status),
        ctypes.byref(output_iterations),
        ctypes.byref(output_upper),
        ctypes.byref(output_lower),
        ctypes.byref(output_gap),
        ctypes.byref(output_step),
        ctypes.byref(output_norm),
        ctypes.byref(output_elapsed),
        ctypes.byref(output_initial_gap),
        ctypes.byref(output_initial_passed),
        error_buffer,
        ctypes.c_size_t(len(error_buffer)),
    )
    native_call_ms = (time.perf_counter() - call_start) * 1000.0
    if return_code != 0:
        message = error_buffer.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"native Mirror-Prox failed ({return_code}): {message}")
    if output_status.value not in _STATUS_NAMES:
        raise RuntimeError(f"native Mirror-Prox returned unknown status {output_status.value}")

    native_shares = tuple(output_shares)
    native_dual_values = tuple(output_dual_weights)
    dual_weights: dict[str, tuple[float, ...]] = {}
    for block_index, name in enumerate(active):
        start = block_resource_offsets[block_index]
        end = block_resource_offsets[block_index + 1]
        dual_weights[name] = native_dual_values[start:end]

    # Independently replay feasibility and both certificates in the Python
    # reference model before exposing a successful native result.
    problem.validate_shares(native_shares)
    upper_replay = problem.objective_value(native_shares, active)
    lower_replay = problem.dual_lower_bound(dual_weights, active)
    scale = max(1.0, abs(upper_replay), abs(lower_replay))
    replay_tolerance = 2e-10 * scale
    if abs(upper_replay - output_upper.value) > replay_tolerance:
        raise ArithmeticError("native primal upper bound failed Python replay")
    if abs(lower_replay - output_lower.value) > replay_tolerance:
        raise ArithmeticError("native dual lower bound failed Python replay")
    if abs((upper_replay - lower_replay) - output_gap.value) > 2.0 * replay_tolerance:
        raise ArithmeticError("native primal-dual gap failed Python replay")
    if output_step.value != (0.5 / norm_bound if norm_bound else 0.0):
        raise ArithmeticError("native solver returned an unexpected safe step size")

    result = MirrorProxResult(
        shares=native_shares,
        dual_weights=dual_weights,
        status=_STATUS_NAMES[output_status.value],
        iterations=output_iterations.value,
        upper_bound_ms=output_upper.value,
        lower_bound_ms=output_lower.value,
        gap_ms=output_gap.value,
        step_size=output_step.value,
        operator_norm_bound=output_norm.value,
        normalization_ms=normalization_ms,
        objective_blocks=active,
        elapsed_ms=output_elapsed.value,
    )
    full_call_ms = (time.perf_counter() - total_start) * 1000.0
    return NativeMirrorProxResult(
        result=result,
        preparation_ms=preparation_ms,
        native_call_ms=native_call_ms,
        native_solver_ms=output_elapsed.value,
        compile_ms=compile_ms,
        full_call_ms=full_call_ms,
        initialization_mode=initialization_mode,
        initial_certificate_gap_ms=output_initial_gap.value,
        initial_certificate_passed=bool(output_initial_passed.value),
    )


def _set_abi_signature(function: ctypes._CFuncPtr) -> None:
    uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
        uint64_pointer,
        double_pointer,
        double_pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        double_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_int),
        uint64_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int


def _uint64_array(values: Sequence[int]) -> ctypes.Array:
    array_type = ctypes.c_uint64 * len(values)
    return array_type(*values)


def _double_array(values: Sequence[float]) -> ctypes.Array:
    array_type = ctypes.c_double * len(values)
    return array_type(*values)


def _is_finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _retain_compiler_dll_directory(compiler_path: str | Path | None) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    compiler = Path(compiler_path) if compiler_path is not None else _DEFAULT_COMPILER
    directory = compiler.parent
    if directory.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
