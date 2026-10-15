"""Run the frozen five-arm quality comparison under one common v2 evaluator.

The legacy planners return fixed Expert-level integer quotas. Those quotas are
realized into source rows with the existing local-first rule before scoring.
The communication-social v2 arm retains its own source-row shares and integer
quotas. This module is intentionally separate from historical runners.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from .moe_communication_social_v2 import (
    BLOCK_NAMES,
    CommunicationConfig,
    RouteRow,
    SparseSocialProblem,
    integerize_rows,
)
from .moe_communication_social_v2_adaptive_native_v1 import (
    compile_adaptive_native_library,
    solve_projected_mirror_prox_adaptive_native,
)
from .moe_communication_social_v2_reduction_v1 import reduce_equal_source_masks
from .moe_communication_social_v2_runner import _make_case_rows, _rows_hash


_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = _ROOT / "artifacts" / "communication-social-v2-quality-topologies-20261005-a"
_PROTOCOL_PATH = _ARTIFACT_DIR / "protocol.json"
_ADDENDUM_PATH = _ARTIFACT_DIR / "protocol_addendum.json"
_PROTOCOL_SHA256 = "712ae6e439c8bd3ddf2faa9643e56ba569299769cb36d89710031b02efc40b69"
_ADDENDUM_SHA256 = "df0fc923ffa8c0a0667658c415937077adab90d91efc723f03709e55919b3ab4"
_QUALITY_CPP = Path(__file__).resolve().with_name(
    "moe_communication_social_v2_quality_adapters_v1.cpp"
)
_QUALITY_DLL = _ARTIFACT_DIR / "moe_communication_social_v2_quality_adapters_v1.dll"
_ADAPTIVE_DLL = _ARTIFACT_DIR / "moe_communication_social_v2_adaptive_native_v1.dll"
_COMPILER = (
    _ROOT / ".scratch" / "granite-moe-replica-adapter" / "toolchain"
    / "winlibs-gcc14.2-ucrt-r3" / "mingw64" / "bin" / "g++.exe"
)
_ARM_IDS = {
    "chain_v1": 0,
    "lazarus_local_first": 1,
    "tempo_paper_inspired_no_lp": 2,
    "sinkhorn_relaxed": 3,
}
_DLL_DIRECTORY_HANDLES: list[object] = []
_MAX_SAFE_ROUTE_COUNT = (1 << 53) - 1


@dataclass(frozen=True)
class LegacyQuotaPlan:
    """Frozen native status and fixed left/right Expert quotas."""

    arm: str
    quota_left_by_expert: tuple[int, ...]
    quota_right_by_expert: tuple[int, ...]
    status: str
    iterations: int
    tempo_metadata: Mapping[str, int]
    planner_diagnostics: Mapping[str, int]


def load_frozen_protocol() -> dict[str, Any]:
    """Load and verify both frozen protocol files and every protected input."""

    protocol_bytes = _PROTOCOL_PATH.read_bytes()
    addendum_bytes = _ADDENDUM_PATH.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    addendum_sha = hashlib.sha256(addendum_bytes).hexdigest()
    if protocol_sha != _PROTOCOL_SHA256:
        raise ValueError(f"frozen protocol hash mismatch: {protocol_sha}")
    if addendum_sha != _ADDENDUM_SHA256:
        raise ValueError(f"frozen protocol addendum hash mismatch: {addendum_sha}")
    protocol = json.loads(protocol_bytes)
    addendum = json.loads(addendum_bytes)
    if addendum.get("original_protocol_sha256") != _PROTOCOL_SHA256:
        raise ValueError("protocol addendum does not bind to the frozen protocol")
    if addendum.get("v2_solver", {}).get("adaptive_line_search") is not True:
        raise ValueError("frozen addendum must enable adaptive_line_search")
    if protocol.get("arms") != [
        "chain_v1", "lazarus_local_first", "tempo_paper_inspired_no_lp",
        "sinkhorn_relaxed", "communication_social_v2",
    ]:
        raise ValueError("the frozen five-arm order changed")
    for relative, expected in protocol.get("protected_inputs_sha256", {}).items():
        path = _ROOT / relative
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"protected frozen input changed: {relative} ({actual})")
    return protocol


def frozen_instances(protocol: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    """Expand the frozen topologies and seeds into nine fixed input cases."""

    protocol = load_frozen_protocol() if protocol is None else protocol
    workload = protocol["workload"]
    instances: list[dict[str, Any]] = []
    for frozen_case in protocol["cases"]:
        for seed in frozen_case["seeds"]:
            case = {
                "case_id": frozen_case["case_id"],
                "gpu_count": frozen_case["gpu_count"],
                "expert_count": frozen_case["expert_count"],
                "gpu_nodes": tuple(frozen_case["gpu_nodes"]),
                "seed": seed,
                "tokens_per_gpu": workload["tokens_per_gpu"],
                "top_k": workload["top_k"],
                "kind": "synthetic top-k route-count workload",
            }
            if frozen_case["gpu_count"] == 256:
                case["active_experts_per_source"] = 8
                case["repeat_selected_set_for_each_token"] = True
            else:
                case["repeat_selected_set_for_each_token"] = False
            instances.append(case)
    if len(instances) != 9 or len({(item["case_id"], item["seed"]) for item in instances}) != 9:
        raise AssertionError("frozen protocol must expand to nine unique input cases")
    return tuple(instances)


def build_case_rows(case: Mapping[str, Any]) -> tuple[RouteRow, ...]:
    """Use the project's existing frozen per-case count generator."""

    return _make_case_rows(case)


def communication_config(
    case: Mapping[str, Any], protocol: Mapping[str, Any] | None = None,
) -> CommunicationConfig:
    """Build the exact fixed evaluator configuration for one protocol case."""

    protocol = load_frozen_protocol() if protocol is None else protocol
    model = protocol["model"]
    return CommunicationConfig(
        gpu_nodes=tuple(case["gpu_nodes"]),
        mu_routes_per_ms=float(model["mu_routes_per_ms"]),
        dispatch_bytes_per_route=float(model["dispatch_bytes_per_route"]),
        return_bytes_per_route=float(model["return_bytes_per_route"]),
        intra_bandwidth_bytes_per_ms=float(model["intra_bandwidth_bytes_per_ms"]),
        nic_bandwidth_gbps=float(model["nic_bandwidth_gbps"]),
        alpha_dispatch_ms=float(model["alpha_dispatch_ms"]),
        alpha_return_ms=float(model["alpha_return_ms"]),
    )


def aggregate_source_counts(
    rows: Sequence[RouteRow], *, gpu_count: int, expert_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return source-major count rows, including implicit zero entries."""

    counts = [[0] * expert_count for _ in range(gpu_count)]
    for row in rows:
        if not 0 <= row.source_gpu < gpu_count or not 0 <= row.expert_id < expert_count:
            raise ValueError("source row lies outside frozen dimensions")
        if type(row.demand) is not int or row.demand < 0:
            raise ValueError("source-row demand must be a non-negative integer")
        expected = (row.expert_id % gpu_count, (row.expert_id + 1) % gpu_count)
        if row.allowed_gpus != expected:
            raise ValueError("source row does not use the frozen adjacent-ring placement")
        counts[row.source_gpu][row.expert_id] += row.demand
    return tuple(tuple(row) for row in counts)


def local_first_source_quotas(
    rank_counts: Sequence[Sequence[int]], quota_left: Sequence[int], *, world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Translate the frozen count-level local_first_left_quotas helper.

    The returned matrix is the number of each source's routes assigned to the
    left destination (Expert id modulo world size). The right amount is each
    source count minus this value.
    """

    if len(rank_counts) != world_size or any(len(row) != len(quota_left) for row in rank_counts):
        raise ValueError("source-count dimensions differ from placement")
    expert_count = len(quota_left)
    result = [[0] * expert_count for _ in range(world_size)]
    for expert in range(expert_count):
        counts = [rank_counts[source][expert] for source in range(world_size)]
        left = quota_left[expert]
        total = sum(counts)
        right = total - left
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("source counts must be non-negative integers")
        if type(left) is not int or left < 0 or right < 0:
            raise ValueError("left Expert quota lies outside total demand")
        left_gpu = expert % world_size
        right_gpu = (expert + 1) % world_size
        take_left = min(counts[left_gpu], left)
        take_right = min(counts[right_gpu], right)
        result[left_gpu][expert] = take_left
        remaining = counts.copy()
        remaining[left_gpu] -= take_left
        remaining[right_gpu] -= take_right
        left -= take_left
        right -= take_right
        for source, available in enumerate(remaining):
            to_left = min(available, left)
            to_right = available - to_left
            result[source][expert] += to_left
            left -= to_left
            right -= to_right
        if left != 0 or right != 0:
            raise ValueError("local-first source allocation did not conserve quota")
    return tuple(tuple(row) for row in result)


def materialize_source_quotas(
    rows: Sequence[RouteRow],
    source_counts: Sequence[Sequence[int]],
    source_left_quotas: Sequence[Sequence[int]],
    *, gpu_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return integer destination quotas in each source row's frozen slot order."""

    if len(source_counts) != gpu_count or len(source_left_quotas) != gpu_count:
        raise ValueError("source quota matrices differ from GPU count")
    expert_count = len(source_counts[0]) if source_counts else 0
    if any(len(counts) != expert_count for counts in source_counts) or any(
        len(counts) != expert_count for counts in source_left_quotas
    ):
        raise ValueError("source quota matrices are not rectangular")
    quotas: list[tuple[int, ...]] = []
    observed = [[0] * expert_count for _ in range(gpu_count)]
    for row in rows:
        source, expert = row.source_gpu, row.expert_id
        observed[source][expert] += row.demand
        left = source_left_quotas[source][expert]
        right = source_counts[source][expert] - left
        if left < 0 or right < 0 or left + right != row.demand:
            raise ValueError("source-aware quota does not conserve its input row")
        left_gpu, right_gpu = expert % gpu_count, (expert + 1) % gpu_count
        slot_quota = tuple(left if gpu == left_gpu else right if gpu == right_gpu else -1 for gpu in row.allowed_gpus)
        if any(value < 0 for value in slot_quota):
            raise ValueError("source row mask and fixed ring placement disagree")
        quotas.append(slot_quota)
    if tuple(tuple(row) for row in observed) != tuple(tuple(row) for row in source_counts):
        raise ValueError("source count matrix differs from input rows")
    return tuple(quotas)


def compile_quality_adapters_library(
    output_path: str | Path | None = None, *, compiler_path: str | Path | None = None,
) -> Path:
    """Compile the thin generic C++ adapter into this frozen artifact directory."""

    target = Path(output_path) if output_path is not None else _QUALITY_DLL
    compiler = Path(compiler_path) if compiler_path is not None else _COMPILER
    if not _QUALITY_CPP.is_file():
        raise FileNotFoundError(f"quality adapter source is missing: {_QUALITY_CPP}")
    if not compiler.is_file():
        raise FileNotFoundError(f"configured GCC compiler is missing: {compiler}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(compiler), "-std=c++17", "-O2", "-DNDEBUG", "-fno-fast-math",
        "-Wall", "-Wextra", "-shared", "-static-libgcc", "-static-libstdc++",
        str(_QUALITY_CPP), "-o", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "generic quality adapter compilation failed\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return target


def plan_legacy_adapter(
    arm: str,
    counts: Sequence[int],
    source_counts: Sequence[Sequence[int]],
    *,
    library_path: str | Path | None = None,
    compiler_path: str | Path | None = None,
) -> LegacyQuotaPlan:
    """Call one frozen generic policy and return its per-Expert integer quotas."""

    if arm not in _ARM_IDS:
        raise ValueError(f"unsupported legacy arm {arm!r}")
    gpu_count, expert_count = len(source_counts), len(counts)
    if not 4 <= gpu_count <= 512 or not 1 <= expert_count <= 4096:
        raise ValueError("generic adapter dimensions outside frozen core limits")
    if any(len(row) != expert_count for row in source_counts):
        raise ValueError("source-count matrix is not rectangular with the Expert vector")
    if any(type(value) is not int or value < 0 or value > _MAX_SAFE_ROUTE_COUNT for value in counts):
        raise ValueError("Expert counts must be non-negative safe integers")
    if any(
        type(value) is not int or value < 0 or value > _MAX_SAFE_ROUTE_COUNT
        for row in source_counts for value in row
    ):
        raise ValueError("source-Expert counts must be non-negative safe integers")
    if sum(counts) > _MAX_SAFE_ROUTE_COUNT:
        raise ValueError("global route total exceeds the frozen exact-count range")
    matrix = aggregate_source_counts(
        tuple(
            RouteRow(source, expert, value, (expert % gpu_count, (expert + 1) % gpu_count))
            for source, source_row in enumerate(source_counts)
            for expert, value in enumerate(source_row)
        ),
        gpu_count=gpu_count,
        expert_count=expert_count,
    )
    if tuple(sum(matrix[source][expert] for source in range(gpu_count)) for expert in range(expert_count)) != tuple(counts):
        raise ValueError("global Expert totals differ from source counts")
    target = Path(library_path) if library_path is not None else _QUALITY_DLL
    if not target.is_file():
        compile_quality_adapters_library(target, compiler_path=compiler_path)
    _retain_compiler_dll_directory(compiler_path)
    native = ctypes.CDLL(str(target))
    function = native.mfg_quality_plan_v1
    i64p = ctypes.POINTER(ctypes.c_int64)
    function.argtypes = [
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        i64p, i64p, i64p, i64p,
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), i64p,
        ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    counts_array = _int64_array(counts)
    source_array = _int64_array(value for row in matrix for value in row)
    left_array = (ctypes.c_int64 * expert_count)()
    right_array = (ctypes.c_int64 * expert_count)()
    status_buffer = ctypes.create_string_buffer(128)
    error_buffer = ctypes.create_string_buffer(1024)
    output_iterations = ctypes.c_uint64()
    diagnostics = (ctypes.c_int64 * 9)()
    code = function(
        ctypes.c_int32(gpu_count), ctypes.c_int32(expert_count), ctypes.c_int32(_ARM_IDS[arm]),
        counts_array, source_array, left_array, right_array,
        status_buffer, ctypes.c_size_t(len(status_buffer)),
        ctypes.byref(output_iterations), diagnostics,
        error_buffer, ctypes.c_size_t(len(error_buffer)),
    )
    if code != 0:
        raise RuntimeError(f"generic legacy adapter failed ({code}): {error_buffer.value.decode(errors='replace')}")
    tempo = {
        "stage3_iterations": int(diagnostics[0]),
        "stage2_accepted": int(diagnostics[1]),
        "stage3_accepted": int(diagnostics[2]),
        "fallback_switched": int(diagnostics[3]),
    }
    planner = {
        "compaction_group_updates": int(diagnostics[4]),
        "exchange_diagnostic_0": int(diagnostics[5]),
        "exchange_diagnostic_1": int(diagnostics[6]),
        "exchange_diagnostic_2": int(diagnostics[7]),
        "exchange_diagnostic_3": int(diagnostics[8]),
    }
    left = tuple(int(value) for value in left_array)
    right = tuple(int(value) for value in right_array)
    for expert, (left_value, right_value, demand) in enumerate(zip(left, right, counts)):
        if type(demand) is not int or demand < 0 or left_value < 0 or right_value < 0 or left_value + right_value != demand:
            raise ArithmeticError(f"native adapter returned illegal quota for Expert {expert}")
    return LegacyQuotaPlan(
        arm=arm,
        quota_left_by_expert=left,
        quota_right_by_expert=right,
        status=status_buffer.value.decode("utf-8", errors="replace"),
        iterations=int(output_iterations.value),
        tempo_metadata=tempo,
        planner_diagnostics=planner,
    )


def score_integer_plan(
    problem: SparseSocialProblem, row_quotas: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Score a feasible integer plan and retain direct route/byte components."""

    if len(row_quotas) != len(problem.rows):
        raise ValueError("row quota count differs from problem rows")
    same = intra = inter = 0
    for row, quotas in zip(problem.rows, row_quotas):
        if len(quotas) != len(row.allowed_gpus) or sum(quotas) != row.demand:
            raise ValueError("integer plan violates row conservation or slot count")
        for gpu, routes in zip(row.allowed_gpus, quotas):
            if type(routes) is not int or routes < 0:
                raise ValueError("integer plan contains an invalid route count")
            if gpu == row.source_gpu:
                same += routes
            elif problem.config.gpu_nodes[gpu] == problem.config.gpu_nodes[row.source_gpu]:
                intra += routes
            else:
                inter += routes
    score = problem.integer_score(row_quotas)
    if not math.isfinite(score.total_ms):
        raise ArithmeticError("integer evaluator returned a non-finite F")
    dispatch_bytes = problem.config.dispatch_bytes_per_route
    return_bytes = problem.config.return_bytes_per_route
    return {
        "F_ms": score.total_ms,
        "compute_ms": score.compute_ms,
        "dispatch_ms": score.dispatch_ms,
        "dispatch_intra_ms": score.dispatch_intra_ms,
        "dispatch_inter_ms": score.dispatch_inter_ms,
        "return_ms": score.return_ms,
        "return_intra_ms": score.return_intra_ms,
        "return_inter_ms": score.return_inter_ms,
        "fixed_startup_ms": score.fixed_startup_ms,
        "dispatch_startup_ms": problem.config.alpha_dispatch_ms,
        "return_startup_ms": problem.config.alpha_return_ms,
        "same_gpu_routes": same,
        "intra_node_routes": intra,
        "inter_node_routes": inter,
        "dispatch_intra_bytes": intra * dispatch_bytes,
        "dispatch_inter_bytes": inter * dispatch_bytes,
        "return_intra_bytes": intra * return_bytes,
        "return_inter_bytes": inter * return_bytes,
        "resource_loads_ms": {name: list(values) for name, values in score.resource_loads_ms.items()},
    }


def run_v2_solver(
    problem: SparseSocialProblem,
    *,
    protocol: Mapping[str, Any] | None = None,
    adaptive_library_path: str | Path | None = None,
    compiler_path: str | Path | None = None,
):
    """Solve the original/reduced sparse model with the exact frozen v2 settings."""

    protocol = load_frozen_protocol() if protocol is None else protocol
    solver_cfg = protocol["v2_solver"]
    addendum = json.loads(_ADDENDUM_PATH.read_text(encoding="utf-8"))
    if addendum["v2_solver"]["adaptive_line_search"] is not True:
        raise ValueError("quality protocol requires adaptive local line search")
    reduction = reduce_equal_source_masks(problem)
    library_path = Path(adaptive_library_path) if adaptive_library_path is not None else _ADAPTIVE_DLL
    if not library_path.is_file():
        compile_adaptive_native_library(library_path, compiler_path=compiler_path)
    native = solve_projected_mirror_prox_adaptive_native(
        reduction.problem,
        max_iterations=int(solver_cfg["max_iterations"]),
        absolute_gap_ms=float(solver_cfg["absolute_gap_ms"]),
        relative_gap=float(solver_cfg["relative_gap"]),
        objective_blocks=BLOCK_NAMES,
        initialization_mode=solver_cfg["initialization_mode"],
        initial_certificate_check=bool(solver_cfg["initial_certificate_check"]),
        adaptive_line_search=bool(addendum["v2_solver"]["adaptive_line_search"]),
        library_path=library_path,
        compiler_path=compiler_path,
    )
    expanded_shares = reduction.expand(native.result.shares)
    problem.validate_shares(expanded_shares)
    return native, tuple(expanded_shares), reduction


def run_frozen_quality_matrix(
    output_path: str | Path | None = None,
    *, compiler_path: str | Path | None = None,
) -> Path:
    """Execute the frozen 45-cell quality-only comparison once.

    This entry point is intentionally not called by focused tests. It refuses
    to overwrite an existing result artifact.
    """

    protocol = load_frozen_protocol()
    destination = Path(output_path) if output_path is not None else _ARTIFACT_DIR / "quality_results.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite quality result: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cases_out: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    inputs: list[tuple[dict[str, Any], tuple[RouteRow, ...], tuple[tuple[int, ...], ...], CommunicationConfig, SparseSocialProblem]] = []

    for case in frozen_instances(protocol):
        rows = build_case_rows(case)
        source_counts = aggregate_source_counts(
            rows, gpu_count=case["gpu_count"], expert_count=case["expert_count"]
        )
        config = communication_config(case, protocol)
        problem = SparseSocialProblem.from_rows(rows, config)
        row_payload = [_row_dict(row) for row in rows]
        counts_payload = [list(row) for row in source_counts]
        cases_out.append({
            "case_id": case["case_id"],
            "seed": case["seed"],
            "rows": row_payload,
            "source_counts_by_gpu_expert": counts_payload,
            "communication_config": _json_value(asdict(config)),
            "row_sha256": _rows_hash(rows),
            "source_counts_sha256": _canonical_sha256(counts_payload),
                "config_sha256": _canonical_sha256(_json_value(asdict(config))),
        })
        inputs.append((case, rows, source_counts, config, problem))

    for case, rows, source_counts, _config, problem in inputs:
        gpu_count, expert_count = case["gpu_count"], case["expert_count"]
        counts = tuple(sum(source_counts[source][expert] for source in range(gpu_count)) for expert in range(expert_count))
        for arm in protocol["arms"]:
            if arm == "communication_social_v2":
                native, shares, _reduction = run_v2_solver(
                    problem, protocol=protocol, compiler_path=compiler_path,
                )
                quotas = integerize_rows(problem.rows, shares)
                record = score_integer_plan(problem, quotas)
                continuous = problem.score(shares)
                row_trace = _share_trace(problem, shares)
                solver = native.result
                record.update({
                    "case_id": case["case_id"],
                    "seed": case["seed"],
                    "arm": arm,
                    "native_status": solver.status,
                    "native_iterations": solver.iterations,
                    "adaptive_line_search": native.adaptive_line_search,
                    "initial_certificate_passed": native.initial_certificate_passed,
                    "initial_certificate_gap_ms": native.initial_certificate_gap_ms,
                    "continuous_F_ms": continuous.total_ms,
                    "continuous_compute_ms": continuous.compute_ms,
                    "continuous_dispatch_ms": continuous.dispatch_ms,
                    "continuous_return_ms": continuous.return_ms,
                    "continuous_UB_ms": solver.upper_bound_ms,
                    "continuous_LB_ms": solver.lower_bound_ms,
                    "continuous_gap_ms": solver.gap_ms,
                    "integer_minus_continuous_F_ms": record["F_ms"] - continuous.total_ms,
                    "continuous_row_shares": row_trace,
                    "dual_weights": {name: list(values) for name, values in solver.dual_weights.items()},
                    "integer_quotas_by_row": [list(row_quota) for row_quota in quotas],
                    "source_quotas_slot_order": _quota_trace(problem.rows, quotas),
                })
                records.append(record)
            else:
                plan = plan_legacy_adapter(
                    arm, counts, source_counts, compiler_path=compiler_path,
                )
                left_by_source = local_first_source_quotas(
                    source_counts, plan.quota_left_by_expert, world_size=gpu_count,
                )
                quotas = materialize_source_quotas(
                    rows, source_counts, left_by_source, gpu_count=gpu_count,
                )
                record = score_integer_plan(problem, quotas)
                record.update({
                    "case_id": case["case_id"],
                    "seed": case["seed"],
                    "arm": arm,
                    "native_status": plan.status,
                    "native_iterations": plan.iterations,
                    "quota_left_by_expert": list(plan.quota_left_by_expert),
                    "quota_right_by_expert": list(plan.quota_right_by_expert),
                    "source_left_quotas_by_gpu_expert": [list(row) for row in left_by_source],
                    "integer_quotas_by_row": [list(row_quota) for row_quota in quotas],
                    "source_quotas_slot_order": _quota_trace(problem.rows, quotas),
                    "tempo_metadata": dict(plan.tempo_metadata),
                    "planner_diagnostics": dict(plan.planner_diagnostics),
                })
                records.append(record)

    if len(records) != 45:
        raise AssertionError(f"frozen quality matrix produced {len(records)} cells, expected 45")
    payload = {
        "protocol_sha256": _PROTOCOL_SHA256,
        "protocol_addendum_sha256": _ADDENDUM_SHA256,
        "protocol": protocol,
        "cases": cases_out,
        "records": records,
        "paired_medians_F_ms": _paired_medians(records),
        "claim_boundary": protocol["workload"]["claim_boundary"],
        "baseline_label_boundary": protocol["baselines"]["labels"],
        "decision_time": protocol["metrics"]["decision_time"],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def _quota_trace(rows: Sequence[RouteRow], quotas: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {
            "source_gpu": row.source_gpu,
            "expert_id": row.expert_id,
            "demand": row.demand,
            "allowed_gpus": list(row.allowed_gpus),
            "slot_quotas": list(row_quota),
        }
        for row, row_quota in zip(rows, quotas)
    ]


def _share_trace(problem: SparseSocialProblem, shares: Sequence[float]) -> list[dict[str, Any]]:
    return [
        {
            "source_gpu": row.source_gpu,
            "expert_id": row.expert_id,
            "demand": row.demand,
            "allowed_gpus": list(row.allowed_gpus),
            "slot_shares": list(shares[problem.row_offsets[index]:problem.row_offsets[index + 1]]),
        }
        for index, row in enumerate(problem.rows)
    ]


def _paired_medians(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for record in records:
        groups.setdefault((record["case_id"], record["arm"]), []).append(float(record["F_ms"]))
    return [
        {
            "case_id": case_id,
            "arm": arm,
            "seed_count": len(values),
            "median_F_ms": statistics.median(values),
            "per_seed_F_ms": values,
        }
        for (case_id, arm), values in sorted(groups.items())
    ]


def _row_dict(row: RouteRow) -> dict[str, Any]:
    return {
        "source_gpu": row.source_gpu,
        "expert_id": row.expert_id,
        "demand": row.demand,
        "allowed_gpus": list(row.allowed_gpus),
    }


def _int64_array(values: Sequence[int] | Any) -> ctypes.Array:
    items = tuple(values)
    if any(type(value) is not int or value < 0 or value > _MAX_SAFE_ROUTE_COUNT for value in items):
        raise ValueError("ctypes route counts must be non-negative exact integers")
    return (ctypes.c_int64 * len(items))(*items)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _retain_compiler_dll_directory(compiler_path: str | Path | None) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    compiler = Path(compiler_path) if compiler_path is not None else _COMPILER
    if compiler.parent.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(compiler.parent)))


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="new results JSON path; existing paths are refused")
    args = parser.parse_args()
    result_path = run_frozen_quality_matrix(args.output)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
