"""Frozen, bounded synthetic evaluation for communication-social v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from .moe_communication_social_v2 import (
    BLOCK_NAMES,
    CommunicationConfig,
    RouteRow,
    SparseSocialProblem,
    integerize_rows,
    solve_projected_mirror_prox,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/communication-aware-social-v2-20261005-a/bounded-matrix-v2")
MANIFEST_FILENAME = "matrix_manifest.json"
RESULT_FILENAME = "matrix_results.json"


def frozen_manifest() -> dict[str, Any]:
    """Return the exact workload and solver budget frozen before execution."""

    payload: dict[str, Any] = {
        "protocol": "communication-aware-social-v2 bounded matrix v2",
        "freeze_date": "2026-10-05",
        "model": {
            "service_rate_routes_per_ms": 10.0,
            "dispatch_bytes_per_route": 8192.0,
            "return_bytes_per_route": 8192.0,
            "intra_bandwidth_decimal_gb_per_s": 50.0,
            "startup_dispatch_ms": 0.02,
            "startup_return_ms": 0.02,
            "topology": "ring two-replica masks; fixed placement",
            "scoring": "serial dispatch + compute + return; no overlap",
            "objective": "sum of five hard resource maxima plus fixed startup",
        },
        "arms": ["uniform", "source_local", "count_only_same_solver", "communication_aware"],
        "profiles": [200.0, 400.0],
        "cases": [
            {
                "case_id": "analytic_two_gpu_slow",
                "kind": "analytical fixture",
                "gpu_count": 2,
                "expert_count": 1,
                "tokens_per_gpu": None,
                "top_k": None,
                "seed": None,
                "nic_gbps": 0.000004,
                "demand_routes": 100,
                "dispatch_bytes_per_route": 1.0,
                "return_bytes_per_route": 1.0,
                "startup_each_ms": 0.125,
                "max_iterations": 1000,
                "absolute_gap_ms": 0.001,
            },
            {
                "case_id": "analytic_two_gpu_fast",
                "kind": "analytical fixture",
                "gpu_count": 2,
                "expert_count": 1,
                "tokens_per_gpu": None,
                "top_k": None,
                "seed": None,
                "nic_gbps": 0.0008,
                "demand_routes": 100,
                "dispatch_bytes_per_route": 1.0,
                "return_bytes_per_route": 1.0,
                "startup_each_ms": 0.125,
                "max_iterations": 1000,
                "absolute_gap_ms": 0.001,
            },
            {
                "case_id": "synthetic_ring_g6_e32",
                "kind": "synthetic top-k route-count workload",
                "gpu_count": 6,
                "expert_count": 32,
                "tokens_per_gpu": 256,
                "top_k": 8,
                "seed": 60205,
                "nic_gbps": 200.0,
                "max_iterations": 2000,
                "absolute_gap_ms": 0.05,
                "relative_gap": 0.0,
            },
            {
                "case_id": "synthetic_sparse_g256_e256",
                "kind": "synthetic top-k route-count workload",
                "gpu_count": 256,
                "expert_count": 256,
                "tokens_per_gpu": 1024,
                "top_k": 8,
                "seed": 102105,
                "active_experts_per_source": 8,
                "repeat_selected_set_for_each_token": True,
                "nic_gbps": 200.0,
                "max_iterations": 1000,
                "absolute_gap_ms": 0.1,
                "relative_gap": 0.0,
            },
        ],
        "synthetic_gate": {
            "g6": "uniform distinct Experts sampled without replacement per Token; each selected Expert gets one route",
            "g256": "uniform 8 distinct Experts sampled once per source, then the same 8 are repeated for all 1024 Tokens on that source; only positive-demand source-Expert rows are materialized",
            "claim_boundary": "synthetic fixed-Gate counts; G256 has M=4096 route variables, not a real model, dense E=1024 benchmark, or general Top8 traffic distribution",
        },
        "integerization": "per-source-Expert row largest remainder; feasible only, not integer optimal",
        "measurement": {
            "setup_ms": "fresh sparse operator construction for the frozen count rows",
            "solver_ms": "wall time around the complete solver call, including norm setup",
            "rounding_ms": "row quota construction only",
            "local_complete_cpu_c_ms": "setup + policy/solver + rounding; synthetic count generation excluded",
            "model_F_units": "analytical milliseconds; not device or NCCL timing",
        },
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    return payload


def freeze_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / MANIFEST_FILENAME
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {destination}")
    manifest = frozen_manifest()
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def run_frozen_matrix(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"freeze the manifest before running: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied_hash = manifest.pop("manifest_sha256", None)
    if supplied_hash != _manifest_hash(manifest):
        raise ValueError("frozen manifest digest does not match its content")
    expected = frozen_manifest()
    if supplied_hash != expected["manifest_sha256"]:
        raise ValueError("manifest differs from the frozen runner contract")
    output_path = output_dir / RESULT_FILENAME
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output_path}")

    workloads: dict[str, tuple[RouteRow, ...]] = {}
    workload_build_ms: dict[str, float] = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        build_start = time.perf_counter()
        workloads[case_id] = _make_case_rows(case)
        workload_build_ms[case_id] = (time.perf_counter() - build_start) * 1000.0

    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        rows = workloads[case_id]
        profiles = (case["nic_gbps"],) if case["kind"] == "analytical fixture" else tuple(manifest["profiles"])
        for nic_gbps in profiles:
            config = _make_config(case, manifest, nic_gbps)
            setup_start = time.perf_counter()
            problem = SparseSocialProblem.from_rows(rows, config)
            setup_ms = (time.perf_counter() - setup_start) * 1000.0
            topology_id = f"{len(config.gpu_nodes)}gpu_{nic_gbps:g}gbps"
            row_hash = _rows_hash(rows)
            for arm in manifest["arms"]:
                record = _run_arm(
                    problem,
                    arm,
                    case,
                    case_id,
                    nic_gbps,
                    setup_ms,
                    workload_build_ms[case_id],
                    row_hash,
                )
                record["topology_id"] = topology_id
                records.append(record)

    results = {
        "manifest_sha256": supplied_hash,
        "result_schema": "communication-aware-social-v2 matrix row v2",
        "records": records,
        "interpretation": [
            "Synthetic model milliseconds are not measured communication or GPU time.",
            "Every arm is scored with the full five-block v2 objective.",
            "The 256-GPU case is sparse E=256 synthetic Top8, not dense E=1024 or a physical cluster.",
            "Integer row rounding is feasible and conservative, with no integer-global-optimum claim.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def replay_solver_traces(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Repeat only solver arms and save shares/quotas/duals for independent audit."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied_hash = manifest.pop("manifest_sha256", None)
    if supplied_hash != _manifest_hash(manifest) or supplied_hash != frozen_manifest()["manifest_sha256"]:
        raise ValueError("trace manifest digest does not match the frozen runner contract")
    output_path = output_dir / "matrix_solver_trace.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit replay: {output_path}")

    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        rows = _make_case_rows(case)
        profiles = (case["nic_gbps"],) if case["kind"] == "analytical fixture" else tuple(manifest["profiles"])
        for nic_gbps in profiles:
            problem = SparseSocialProblem.from_rows(rows, _make_config(case, manifest, nic_gbps))
            for arm in ("count_only_same_solver", "communication_aware"):
                objective_blocks = ("compute",) if arm == "count_only_same_solver" else BLOCK_NAMES
                start = time.perf_counter()
                result = solve_projected_mirror_prox(
                    problem,
                    max_iterations=case["max_iterations"],
                    absolute_gap_ms=case["absolute_gap_ms"],
                    relative_gap=case.get("relative_gap", 1e-7),
                    objective_blocks=objective_blocks,
                )
                replay_wall_ms = (time.perf_counter() - start) * 1000.0
                quotas = integerize_rows(problem.rows, result.shares)
                row_trace = []
                for row_index, row in enumerate(problem.rows):
                    begin, end = problem.row_offsets[row_index : row_index + 2]
                    destinations = []
                    for variable_index in range(begin, end):
                        slot = variable_index - begin
                        destinations.append({
                            "gpu": problem.variable_gpus[variable_index],
                            "share": result.shares[variable_index],
                            "integer_routes": quotas[row_index][slot],
                        })
                    row_trace.append({
                        "source_gpu": row.source_gpu,
                        "expert_id": row.expert_id,
                        "demand": row.demand,
                        "allowed_gpus": list(row.allowed_gpus),
                        "destinations": destinations,
                    })
                score = problem.score(result.shares)
                records.append({
                    "case_id": case["case_id"],
                    "arm": arm,
                    "nic_gbps": nic_gbps,
                    "objective_blocks": list(result.objective_blocks),
                    "solver_status": result.status,
                    "solver_iterations": result.iterations,
                    "solver_upper_bound_ms": result.upper_bound_ms,
                    "solver_lower_bound_ms": result.lower_bound_ms,
                    "solver_gap_ms": result.gap_ms,
                    "normalization_ms": result.normalization_ms,
                    "step_size": result.step_size,
                    "operator_norm_bound": result.operator_norm_bound,
                    "solver_replay_wall_ms_not_formal": replay_wall_ms,
                    "full_model_score_ms": {
                        "D_dispatch": score.dispatch_ms,
                        "K_compute": score.compute_ms,
                        "C_return": score.return_ms,
                        "F": score.total_ms,
                    },
                    "model_config": {
                        "gpu_nodes": list(problem.config.gpu_nodes),
                        "mu_routes_per_ms": problem.config.mu_routes_per_ms,
                        "dispatch_bytes_per_route": problem.config.dispatch_bytes_per_route,
                        "return_bytes_per_route": problem.config.return_bytes_per_route,
                        "intra_bandwidth_bytes_per_ms": problem.config.intra_bandwidth_bytes_per_ms,
                        "nic_bandwidth_gbps": problem.config.nic_bandwidth_gbps,
                        "alpha_dispatch_ms": problem.config.alpha_dispatch_ms,
                        "alpha_return_ms": problem.config.alpha_return_ms,
                    },
                    "M_variables": problem.variable_count,
                    "Z_sparse_incidences": problem.incidence_count,
                    "Q_by_block": {block.name: len(block.resource_names) for block in problem.blocks},
                    "dual_weights": {
                        name: [
                            {"resource": resource, "weight": weight}
                            for resource, weight in zip(
                                problem._blocks_by_name[name].resource_names,
                                result.dual_weights[name],
                            )
                        ]
                        for name in objective_blocks
                    },
                    "source_expert_rows": row_trace,
                })
    replay = {
        "manifest_sha256": supplied_hash,
        "purpose": "independent certificate and row-quota audit; separate rerun, not formal timing",
        "records": records,
    }
    output_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="write the frozen manifest; no experiment runs")
    mode.add_argument("--run", action="store_true", help="run the formal matrix against the frozen manifest")
    mode.add_argument("--trace", action="store_true", help="repeat solver arms and save auditable primal/dual traces")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.freeze:
        destination = freeze_manifest(args.output_dir)
    elif args.trace:
        destination = replay_solver_traces(args.output_dir)
    else:
        destination = run_frozen_matrix(args.output_dir)
    print(destination)
    return 0


def _run_arm(
    problem: SparseSocialProblem,
    arm: str,
    case: Mapping[str, Any],
    case_id: str,
    nic_gbps: float,
    setup_ms: float,
    workload_build_ms: float,
    row_hash: str,
) -> dict[str, Any]:
    result = None
    policy_start = time.perf_counter()
    solver_ms = 0.0
    objective_blocks: tuple[str, ...] = ()
    if arm == "uniform":
        shares = problem.uniform_shares()
        policy_build_ms = (time.perf_counter() - policy_start) * 1000.0
    elif arm == "source_local":
        shares = problem.source_local_shares()
        policy_build_ms = (time.perf_counter() - policy_start) * 1000.0
    elif arm in ("count_only_same_solver", "communication_aware"):
        objective_blocks = ("compute",) if arm == "count_only_same_solver" else BLOCK_NAMES
        solver_start = time.perf_counter()
        result = solve_projected_mirror_prox(
            problem,
            max_iterations=case["max_iterations"],
            absolute_gap_ms=case["absolute_gap_ms"],
            relative_gap=case.get("relative_gap", 1e-7),
            objective_blocks=objective_blocks,
        )
        solver_ms = (time.perf_counter() - solver_start) * 1000.0
        shares = result.shares
        policy_build_ms = 0.0
    else:
        raise ValueError(f"unknown arm: {arm}")

    continuous = problem.score(shares)
    rounding_start = time.perf_counter()
    quotas = integerize_rows(problem.rows, shares)
    rounding_ms = (time.perf_counter() - rounding_start) * 1000.0
    integer_score = problem.integer_score(quotas)
    row_conservation = all(sum(allocation) == row.demand for row, allocation in zip(problem.rows, quotas))
    support_legal = all(len(allocation) == len(row.allowed_gpus) for row, allocation in zip(problem.rows, quotas))
    if not row_conservation or not support_legal:
        raise AssertionError("row integerization failed conservation or mask shape")
    continuous_traffic = _traffic_summary(problem, shares)
    integer_traffic = _traffic_summary(problem, tuple(
        quota / row.demand if row.demand else 1.0 / len(row_quotas)
        for row, row_quotas in zip(problem.rows, quotas)
        for quota in row_quotas
    ))

    return {
        "case_id": case_id,
        "arm": arm,
        "seed": case.get("seed"),
        "nic_gbps": nic_gbps,
        "gpu_count": len(problem.config.gpu_nodes),
        "expert_count": case["expert_count"],
        "tokens_per_gpu": case.get("tokens_per_gpu"),
        "top_k": case.get("top_k"),
        "rows": len(problem.rows),
        "routes": sum(row.demand for row in problem.rows),
        "row_hash_sha256": row_hash,
        "M_variables": problem.variable_count,
        "Z_sparse_incidences": problem.incidence_count,
        "Q_resources_total": problem.resource_count,
        "Q_by_block": {block.name: len(block.resource_names) for block in problem.blocks},
        "D_dispatch_ms": continuous.dispatch_ms,
        "K_compute_ms": continuous.compute_ms,
        "C_return_ms": continuous.return_ms,
        "F_continuous_full_v2_ms": continuous.total_ms,
        "F_integer_full_v2_ms": integer_score.total_ms,
        "integer_rounding_gap_ms": integer_score.total_ms - continuous.total_ms,
        "dispatch_intra_ms": continuous.dispatch_intra_ms,
        "dispatch_inter_ms": continuous.dispatch_inter_ms,
        "return_intra_ms": continuous.return_intra_ms,
        "return_inter_ms": continuous.return_inter_ms,
        "fixed_startup_ms": continuous.fixed_startup_ms,
        "continuous_local_routes": continuous_traffic["local_routes"],
        "continuous_intra_routes": continuous_traffic["intra_routes"],
        "continuous_inter_routes": continuous_traffic["inter_routes"],
        "integer_local_routes": integer_traffic["local_routes"],
        "integer_intra_routes": integer_traffic["intra_routes"],
        "integer_inter_routes": integer_traffic["inter_routes"],
        "dispatch_payload_bytes_all_routes": sum(row.demand for row in problem.rows) * problem.config.dispatch_bytes_per_route,
        "return_payload_bytes_all_routes": sum(row.demand for row in problem.rows) * problem.config.return_bytes_per_route,
        "intra_dispatch_payload_bytes": continuous_traffic["intra_routes"] * problem.config.dispatch_bytes_per_route,
        "inter_dispatch_payload_bytes": continuous_traffic["inter_routes"] * problem.config.dispatch_bytes_per_route,
        "total_network_dispatch_payload_bytes": (continuous_traffic["intra_routes"] + continuous_traffic["inter_routes"]) * problem.config.dispatch_bytes_per_route,
        "intra_return_payload_bytes": continuous_traffic["intra_routes"] * problem.config.return_bytes_per_route,
        "inter_return_payload_bytes": continuous_traffic["inter_routes"] * problem.config.return_bytes_per_route,
        "total_network_return_payload_bytes": (continuous_traffic["intra_routes"] + continuous_traffic["inter_routes"]) * problem.config.return_bytes_per_route,
        "integer_dispatch_payload_bytes_all_routes": sum(row.demand for row in problem.rows) * problem.config.dispatch_bytes_per_route,
        "integer_return_payload_bytes_all_routes": sum(row.demand for row in problem.rows) * problem.config.return_bytes_per_route,
        "integer_intra_dispatch_payload_bytes": integer_traffic["intra_routes"] * problem.config.dispatch_bytes_per_route,
        "integer_inter_dispatch_payload_bytes": integer_traffic["inter_routes"] * problem.config.dispatch_bytes_per_route,
        "integer_total_network_dispatch_payload_bytes": (integer_traffic["intra_routes"] + integer_traffic["inter_routes"]) * problem.config.dispatch_bytes_per_route,
        "integer_intra_return_payload_bytes": integer_traffic["intra_routes"] * problem.config.return_bytes_per_route,
        "integer_inter_return_payload_bytes": integer_traffic["inter_routes"] * problem.config.return_bytes_per_route,
        "integer_total_network_return_payload_bytes": (integer_traffic["intra_routes"] + integer_traffic["inter_routes"]) * problem.config.return_bytes_per_route,
        "row_conservation": row_conservation,
        "support_legal": support_legal,
        "solver_status": result.status if result else "not_applicable",
        "solver_iterations": result.iterations if result else 0,
        "solver_gap_ms": result.gap_ms if result else None,
        "solver_upper_bound_ms": result.upper_bound_ms if result else None,
        "solver_lower_bound_ms": result.lower_bound_ms if result else None,
        "solver_certificate_scope": "+".join(objective_blocks) if result else None,
        "solver_loop_ms": result.elapsed_ms if result else 0.0,
        "setup_ms": setup_ms,
        "workload_build_ms_outside_C": workload_build_ms,
        "solver_wall_ms": solver_ms,
        "policy_setup_ms": policy_build_ms,
        "rounding_ms": rounding_ms,
        "local_complete_cpu_C_ms": setup_ms + solver_ms + policy_build_ms + rounding_ms,
    }


def _make_case_rows(case: Mapping[str, Any]) -> tuple[RouteRow, ...]:
    if case["kind"] == "analytical fixture":
        return (RouteRow(0, 0, case["demand_routes"], (0, 1)),)
    gpu_count = case["gpu_count"]
    expert_count = case["expert_count"]
    top_k = case["top_k"]
    rng = random.Random(case["seed"])
    counts = [[0] * expert_count for _ in range(gpu_count)]
    if case.get("repeat_selected_set_for_each_token"):
        for source_gpu in range(gpu_count):
            for expert_id in rng.sample(range(expert_count), case["active_experts_per_source"]):
                counts[source_gpu][expert_id] = case["tokens_per_gpu"]
    else:
        for source_gpu in range(gpu_count):
            for _ in range(case["tokens_per_gpu"]):
                for expert_id in rng.sample(range(expert_count), top_k):
                    counts[source_gpu][expert_id] += 1
    rows: list[RouteRow] = []
    for source_gpu in range(gpu_count):
        for expert_id in range(expert_count):
            if case.get("repeat_selected_set_for_each_token") and counts[source_gpu][expert_id] == 0:
                continue
            mask = (expert_id % gpu_count, (expert_id + 1) % gpu_count)
            rows.append(RouteRow(source_gpu, expert_id, counts[source_gpu][expert_id], mask))
    return tuple(rows)


def _make_config(
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    nic_gbps: float,
) -> CommunicationConfig:
    if case["kind"] == "analytical fixture":
        gpu_nodes = (0, 1)
        startup = case["startup_each_ms"]
        input_bytes = case["dispatch_bytes_per_route"]
        output_bytes = case["return_bytes_per_route"]
        intra_bw = 1.0
    else:
        gpu_count = case["gpu_count"]
        gpus_per_node = 2 if gpu_count == 6 else 8
        gpu_nodes = tuple(gpu // gpus_per_node for gpu in range(gpu_count))
        model = manifest["model"]
        startup = model["startup_dispatch_ms"]
        input_bytes = model["dispatch_bytes_per_route"]
        output_bytes = model["return_bytes_per_route"]
        intra_bw = model["intra_bandwidth_decimal_gb_per_s"] * 1_000_000.0
    return CommunicationConfig(
        gpu_nodes=gpu_nodes,
        mu_routes_per_ms=manifest["model"]["service_rate_routes_per_ms"],
        dispatch_bytes_per_route=input_bytes,
        return_bytes_per_route=output_bytes,
        intra_bandwidth_bytes_per_ms=intra_bw,
        nic_bandwidth_gbps=nic_gbps,
        alpha_dispatch_ms=startup,
        alpha_return_ms=startup if case["kind"] == "analytical fixture" else manifest["model"]["startup_return_ms"],
    )


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rows_hash(rows: Sequence[RouteRow]) -> str:
    values = [
        [row.source_gpu, row.expert_id, row.demand, list(row.allowed_gpus)]
        for row in rows
    ]
    canonical = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _traffic_summary(problem: SparseSocialProblem, shares: Sequence[float]) -> dict[str, float]:
    local = intra = inter = 0.0
    for row_index, row in enumerate(problem.rows):
        start, end = problem.row_offsets[row_index : row_index + 2]
        for variable_index in range(start, end):
            routes = row.demand * shares[variable_index]
            destination = problem.variable_gpus[variable_index]
            if row.source_gpu == destination:
                local += routes
            elif problem.config.gpu_nodes[row.source_gpu] == problem.config.gpu_nodes[destination]:
                intra += routes
            else:
                inter += routes
    return {"local_routes": local, "intra_routes": intra, "inter_routes": inter}


if __name__ == "__main__":
    raise SystemExit(main())
