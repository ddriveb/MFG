from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(V2 / "tools"))

from mfg_hedge.moe_communication_social_v2 import (  # noqa: E402
    BLOCK_NAMES,
    CommunicationConfig,
    RouteRow,
    SparseBlock,
    SparseSocialProblem,
)

EVIDENCE = V2 / "evidence"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def origin_path(origin: str) -> Path:
    origin = origin.replace("\\", "/")
    if origin.startswith("artifacts/"):
        return EVIDENCE / origin
    return EVIDENCE / "artifacts" / origin


def plan_problem(plan: dict) -> SparseSocialProblem:
    cfg_data = dict(plan["communication_config"])
    cfg_data["gpu_nodes"] = tuple(cfg_data["gpu_nodes"])
    config = CommunicationConfig(**cfg_data)
    rows = tuple(
        RouteRow(
            row["source_gpu"], row["expert_id"], row["demand"], tuple(row["allowed_gpus"])
        )
        for row in plan["rows"]
    )
    return SparseSocialProblem.from_rows(rows, config)


def assert_plan(plan: dict, summary: dict | None = None) -> None:
    problem = plan_problem(plan)
    quotas = plan["quotas"]
    assert len(quotas) == len(problem.rows), (plan.get("state_id"), len(quotas))
    for quota, row in zip(quotas, problem.rows):
        assert len(quota) == len(row.allowed_gpus), (plan.get("state_id"), row)
        assert all(type(value) is int and value >= 0 for value in quota)
        assert sum(quota) == row.demand, (plan.get("state_id"), row, quota)
    score = problem.integer_score(quotas).total_ms
    saved = plan.get("integer_F_ms", plan.get("integer_model_F_ms"))
    if saved is not None:
        assert abs(score - saved) < 1e-9, (plan.get("state_id"), score, saved)
    if summary is not None and "integer_model_F_ms" in summary:
        assert abs(score - summary["integer_model_F_ms"]) < 1e-9


def verify_manifest() -> int:
    manifest = read_json(V2 / "evidence-manifest.json")
    count = 0
    for row in manifest["files"]:
        path = V2 / row["path"]
        data = path.read_bytes()
        assert len(data) == row["bytes"], row["path"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        count += 1
    return count


def replay_fresh_gates() -> dict:
    from gate_generation import FAMILIES, g64_gate, hotspot_gate, hotspot_support, multiscale_gate

    g64_dir = EVIDENCE / "artifacts/moe-g64-three-arm-aicb150-20261014-a"
    g64_protocol = read_json(EVIDENCE / "artifacts/moe-g64-depth-pressure-20261013-a/r2/protocol.json")
    g64_support = tuple(tuple(row) for row in g64_protocol["support"])
    g64_records = read_json(g64_dir / "counts.json")
    assert len(g64_records) == 150
    for index, record in enumerate(g64_records):
        family = FAMILIES[index % len(FAMILIES)]
        assert record["family"] == family
        regenerated = g64_gate(g64_support, record["seed"], family)
        counts = [value for row in regenerated.counts for value in row]
        assert counts == record["counts"], ("g64", record["seed"])

    multi_dir = EVIDENCE / "artifacts/moe-gnn-multiscale-execution-20261014-a"
    multi_protocol = read_json(multi_dir / "protocol.json")
    multi_records = read_json(multi_dir / "counts.json")
    assert len(multi_records) == 192
    for record in multi_records:
        support = tuple(tuple(row) for row in record["support"])
        counts, patterns, strength = multiscale_gate(
            support, record["seed"], record["family"], record["P"]
        )
        assert counts == record["counts"], ("multiscale", record["seed"])
        assert patterns == record["patterns"], ("multiscale patterns", record["seed"])
        assert strength == record["strength"], ("multiscale strength", record["seed"])
    assert multi_protocol["support_seed"] == 1325001

    hot_dir = EVIDENCE / "artifacts/moe-gnn-global-hotspot-execution-20261014-a"
    hot_protocol = read_json(hot_dir / "protocol.json")
    hot_records = read_json(hot_dir / "counts.json")
    assert len(hot_records) == 96
    support_cache = {}
    for record in hot_records:
        ranks = record["P"]
        if ranks not in support_cache:
            support_cache[ranks] = hotspot_support(ranks, hot_protocol["support_seed"])
        support, hot = support_cache[ranks]
        assert [list(row) for row in support] == record["support"]
        assert list(hot) == record["hot_experts"]
        counts, patterns, actual = hotspot_gate(
            support, hot, record["seed"], record["rho"], ranks
        )
        assert counts == record["counts"], ("hotspot", record["seed"])
        assert patterns == record["patterns"], ("hotspot patterns", record["seed"])
        assert actual == record["actual_hot_route_fraction"]
    return {"g64": len(g64_records), "multiscale": len(multi_records), "global_hotspot": len(hot_records)}


def replay_g64() -> dict:
    base = EVIDENCE / "artifacts/moe-g64-three-arm-aicb150-20261014-a"
    rows = read_json(base / "comparison_rows.json")
    saved = read_json(base / "session_report_zh.json")["Artifacts"]["summary"]
    assert len(rows) == 450
    plans = 0
    for row in rows:
        plan = read_json(base / "plans" / f"g64-{row['seed']}-{row['arm']}.json")
        assert plan["rows"] and plan["arm"] == row["arm"]
        assert_plan(plan, row)
        plans += 1
    for arm in ("gnn4", "sinkhorn_fixed", "lazarus_inspired"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        assert len(arm_rows) == 150
        for key in ("C_ms", "integer_model_F_ms", "F_C_ms", "common_integer_gap_ms"):
            mean = statistics.mean(row[key] for row in arm_rows)
            assert abs(mean - saved[arm][key]["mean"]) < 1e-9, (arm, key, mean)
    return {"states": 150, "plans": plans}


def replay_multiscale() -> dict:
    base = EVIDENCE / "artifacts/moe-gnn-multiscale-execution-20261014-a"
    result = read_json(base / "results.json")
    rows = result["rows"]
    assert len(rows) == 576
    for row in rows:
        plan = read_json(base / "plans" / f"{row['state_id']}-{row['arm']}.json")
        assert_plan(plan, row)
    for size in ("16", "32", "64"):
        for arm, stats in result["by_scale"][size].items():
            subset = [r for r in rows if str(r["P"]) == size and r["arm"] == arm]
            assert len(subset) == 64
            for key in ("C_ms", "F_C_ms", "integer_model_F_ms", "common_integer_gap_ms"):
                assert abs(statistics.mean(r[key] for r in subset) - stats[key]["mean"]) < 1e-9
    return {"states": 192, "plans": len(rows)}


def replay_hotspot() -> dict:
    base = EVIDENCE / "artifacts/moe-gnn-global-hotspot-execution-20261014-a"
    rows = read_json(base / "results.json")["rows"]
    assert len(rows) == 384
    states: dict[str, list[dict]] = {}
    for row in rows:
        plan = read_json(base / "plans" / f"{row['state_id']}-{row['arm']}.json")
        assert_plan(plan, row)
        states.setdefault(row["state_id"], []).append(plan)
    assert len(states) == 96 and all(len(plans) == 4 for plans in states.values())
    for state, plans in states.items():
        first = plans[0]
        assert all(p["rows"] == first["rows"] for p in plans)
        assert all(p["communication_config"] == first["communication_config"] for p in plans)
    return {"states": len(states), "plans": len(rows)}


def replay_g4() -> tuple[dict, dict]:
    archive = EVIDENCE / "artifacts/g4-v100-timing100-20261014-a/optimized-final-1791914591.tar.gz"
    physical: dict[tuple[int, str], dict] = {}
    rank_steps: dict[tuple[int, str, int], float] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for rank in range(4):
            member = bundle.extractfile(
                f"results/optimized-three-arm-20261014-a/raw-rank-{rank}.jsonl"
            )
            assert member is not None
            for line in member:
                record = json.loads(line)
                sample = record["sample"]
                if not 2 <= sample <= 151:
                    continue
                assert record["integrity_passed"] and record["error"] is None
                rank_steps[sample, record["arm"], rank] = record["full_step_ms"]
                if rank == 0:
                    assert len(record["planner"]) == 24
                    for plan in record["planner"]:
                        for demands, quotas in zip(plan["source_counts"], plan["source_quotas"]):
                            assert len(demands) == len(quotas)
                            assert all(sum(quota) == demand and min(quota) >= 0
                                       for demand, quota in zip(demands, quotas))
                    physical[sample, record["arm"]] = record
    assert len(physical) == 450 and len(rank_steps) == 1800
    for (sample, arm), record in physical.items():
        record["full_step_ms"] = max(rank_steps[sample, arm, rank] for rank in range(4))
    stats = read_json(EVIDENCE / "artifacts/v2-paper-gnn-results-integration-20261014-a/g4-first150-summary.json")["stats"]
    for arm in ("gnn", "sinkhorn", "lazarus"):
        subset = [physical[sample, arm] for sample in range(2, 152)]
        assert abs(statistics.mean(v["full_step_ms"] for v in subset) - stats[arm]["full_step_mean_ms"]) < 1e-8
        integer = statistics.mean(
            layer["integer_model_ms"] for record in subset for layer in record["planner"]
        )
        assert abs(integer - stats[arm]["integer_model_mean_ms"]) < 1e-8
    for sample in range(2, 152):
        assert len({physical[sample, arm]["gate_sha256"] for arm in ("gnn", "sinkhorn", "lazarus")}) == 1
    return physical, {"groups": 150, "rank_records": len(rank_steps), "quota_layers": 10800}


def replay_price_certificates(physical: dict[tuple[int, str], dict]) -> tuple[int, int]:
    old = read_json(EVIDENCE / "artifacts/paper-evidence-chain-20261014-a/reference_certificates.json")
    assert len(old) == 24
    for certificate in old:
        if certificate["domain"] == "G64":
            g64 = EVIDENCE / "artifacts/moe-g64-three-arm-aicb150-20261014-a"
            plan = read_json(g64 / "plans" / Path(certificate["plan_file"]).name)
            problem = plan_problem(plan)
        else:
            record = physical[certificate["sample"], "gnn"]["planner"][certificate["layer"]]
            counts = [value for row in record["source_counts"] for value in row]
            config = CommunicationConfig(
                (0, 0, 0, 0), mu_routes_per_ms=2513.238,
                dispatch_bytes_per_route=4096, return_bytes_per_route=4096,
                intra_bandwidth_bytes_per_ms=900000000.0, nic_bandwidth_gbps=200.0,
            )
            unit = SparseSocialProblem.from_rows(
                tuple(RouteRow(source, expert, 1, (expert % 4, (expert + 1) % 4))
                      for source in range(4) for expert in range(32)), config
            )
            blocks = tuple(
                SparseBlock(
                    block.name,
                    block.resource_names,
                    tuple(tuple((resource, coefficient * counts[unit.variable_rows[index]])
                                for resource, coefficient in column)
                          for index, column in enumerate(block.columns)),
                )
                for block in unit.blocks
            )
            rows = tuple(dataclasses.replace(row, demand=demand)
                         for row, demand in zip(unit.rows, counts))
            problem = SparseSocialProblem(
                rows, config, unit.row_offsets, unit.variable_rows,
                unit.variable_gpus, blocks,
            )
        upper = problem.objective_value(certificate["shares"])
        lower = problem.dual_lower_bound(certificate["dual_weights"])
        assert abs(upper - certificate["reference_upper"]) < 1e-8
        assert abs(lower - certificate["reference_lower"]) < 1e-8
        assert upper - lower <= 0.001 + 1e-9

    new = read_json(EVIDENCE / "artifacts/paper-global-hotspot-closeout-20261014-a/price-reference-certificates.json")["records"]
    assert len(new) == 12
    price_vectors = 0
    for certificate in new:
        plan = read_json(origin_path(certificate["plan_file"]))
        problem = plan_problem(plan)
        upper = problem.objective_value(certificate["reference_shares"])
        lower = problem.dual_lower_bound(certificate["reference_prices"])
        assert abs(upper - certificate["reference_upper_ms"]) < 1e-9
        assert abs(lower - certificate["reference_lower_ms"]) < 1e-9
        assert upper - lower <= 0.001 + 1e-9 and certificate["status"] == "converged"
        for model in certificate["models"].values():
            prices = model["prices"]
            assert all(min(row) >= 0 and abs(sum(row) - 1) < 1e-12 for row in prices.values())
            learned = problem.dual_lower_bound(prices)
            assert abs(learned - model["learned_lower_ms"]) < 1e-9
            assert learned <= certificate["reference_lower_ms"] + 1e-9
            price_vectors += 1
    return len(old) + len(new), price_vectors


def numeric_summary() -> dict:
    g64_base = EVIDENCE / "artifacts/moe-g64-three-arm-aicb150-20261014-a"
    g64 = read_json(g64_base / "comparison_rows.json")
    by_seed: dict[int, dict[str, dict]] = {}
    for row in g64:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    g64_arms = {}
    for arm in ("gnn4", "sinkhorn_fixed", "lazarus_inspired"):
        subset = [v[arm] for v in by_seed.values()]
        g64_arms[arm] = {
            "states": len(subset),
            "C_ms_mean": statistics.mean(v["C_ms"] for v in subset),
            "F_C_ms_mean": statistics.mean(v["F_C_ms"] for v in subset),
            "integer_F_ms_mean": statistics.mean(v["integer_model_F_ms"] for v in subset),
            "integer_gap_ms_mean": statistics.mean(v["common_integer_gap_ms"] for v in subset),
        }
    g64_path_comparison = {}
    for baseline in ("sinkhorn_fixed", "lazarus_inspired"):
        baseline_mean = g64_arms[baseline]["F_C_ms_mean"]
        gnn_mean = g64_arms["gnn4"]["F_C_ms_mean"]
        g64_path_comparison[baseline] = {
            "paired_wins": sum(v["gnn4"]["F_C_ms"] < v[baseline]["F_C_ms"] for v in by_seed.values()),
            "states": len(by_seed),
            "mean_saving_ms": baseline_mean - gnn_mean,
            "saving_percent": (baseline_mean - gnn_mean) / baseline_mean * 100.0,
        }

    focus = read_json(EVIDENCE / "artifacts/paper-global-hotspot-closeout-20261014-a/saved-data-analysis.json")
    focus_headline = {}
    for baseline, values in focus["headline"].items():
        sign = values["sign_test"]
        focus_headline[baseline] = {
            "states": sign["n_nonzero"],
            "wins": sign["wins"],
            "losses": sign["losses"],
            "gnn_mean_ms": values["gnn_mean_ms"],
            "baseline_mean_ms": values["baseline_mean_ms"],
            "saving_ms": values["saving_ms"],
            "saving_percent": values["saving_percent"],
            "two_sided_p": sign["two_sided_p"],
        }

    hot_rows = read_json(EVIDENCE / "artifacts/moe-gnn-global-hotspot-execution-20261014-a/results.json")["rows"]
    hot_cells = {}
    for ranks in (32, 64):
        for ratio in (0.4, 0.6, 0.8):
            cell = [r for r in hot_rows if r["P"] == ranks and abs(r["rho"] - ratio) < 1e-12]
            grouped = {arm: [r for r in cell if r["arm"] == arm] for arm in
                       ("gnn_mixed", "gnn_hot", "sinkhorn_fixed", "lazarus_inspired")}
            assert all(len(values) == 16 for values in grouped.values())
            saved_cell = focus["adaptation"][str(ranks)]["cells"][str(ratio)]["arms"]
            means = {}
            for arm, values in grouped.items():
                t_mean = statistics.mean(r["T_eval_ms"] for r in values)
                c_mean = statistics.mean(r["C_ms"] for r in values)
                int_mean = statistics.mean(r["integer_model_F_ms"] for r in values)
                assert abs(t_mean - saved_cell[arm]["T_eval_ms"]["mean"]) < 1e-9
                means[arm] = {"T_eval_ms_mean": t_mean, "C_ms_mean": c_mean, "integer_F_ms_mean": int_mean}
            specialist_saving = statistics.mean(r["T_eval_ms"] for r in grouped["gnn_mixed"]) - statistics.mean(r["T_eval_ms"] for r in grouped["gnn_hot"])
            hot_by_seed = {r["seed"]: r for r in grouped["gnn_hot"]}
            mixed_by_seed = {r["seed"]: r for r in grouped["gnn_mixed"]}
            hot_cells[f"p{ranks}_rho{int(ratio * 100)}"] = {
                "arms": means,
                "specialist_paired_wins": sum(hot_by_seed[s]["T_eval_ms"] < mixed_by_seed[s]["T_eval_ms"] for s in hot_by_seed),
                "states": len(hot_by_seed),
                "specialist_saving_ms": specialist_saving,
                "specialist_saving_percent": specialist_saving / means["gnn_mixed"]["T_eval_ms_mean"] * 100.0,
            }

    g4 = read_json(EVIDENCE / "artifacts/v2-paper-gnn-results-integration-20261014-a/g4-first150-summary.json")
    g4_summary = {
        arm: {
            "full_step_mean_ms": values["full_step_mean_ms"],
            "integer_model_mean_ms": values["integer_model_mean_ms"],
            "integer_gap_mean_ms": values["integer_gap_mean_ms"],
            "gap_le_0_5": values["gap_le_0_5"],
        }
        for arm, values in g4["stats"].items()
    }
    return {
        "g64_150_state": {"arms": g64_arms, "paired_path_comparison": g64_path_comparison},
        "historical_focal_38": focus_headline,
        "global_hotspot_specialist_six_cells": hot_cells,
        "g4_first150": g4_summary,
        "timing_note": "Reported timing values are frozen evidence; replay validates their records and metrics but does not predict new hardware wall time.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay frozen V2 evidence and certificates.")
    parser.add_argument("--scope", choices=("all",), default="all")
    parser.add_argument("--skip-hashes", action="store_true", help="skip full copied-file SHA-256 checks")
    parser.add_argument("--output", help="write the JSON report to a new path under v2/artifacts/")
    args = parser.parse_args()
    import mfg_hedge.moe_communication_social_v2 as domain
    package_root = (V2 / "src").resolve()
    imported = {
        name: Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("mfg_hedge.") and getattr(module, "__file__", None)
    }
    assert imported and all(path.is_relative_to(package_root) for path in imported.values()), imported
    package_file = Path(domain.__file__).resolve()
    manifest_files = 0 if args.skip_hashes else verify_manifest()
    fresh_gates = replay_fresh_gates()
    campaigns = {
        "g64": replay_g64(),
        "multiscale": replay_multiscale(),
        "global_hotspot": replay_hotspot(),
    }
    physical, g4 = replay_g4()
    certs, prices = replay_price_certificates(physical)
    report = {
        "passed": True,
        "imported_domain_module": str(package_file.relative_to(V2)),
        "manifest_files_verified": manifest_files,
        "campaigns": {**campaigns, "g4": g4},
        "price_certificates_replayed": certs,
        "learned_price_vectors_replayed": prices,
        "fresh_gates": fresh_gates,
        "numeric_summary": numeric_summary(),
    }
    encoded = json.dumps(report, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        if not destination.is_absolute():
            destination = V2 / destination
        destination = destination.resolve()
        assert destination.is_relative_to((V2 / "artifacts").resolve()), "report output must remain under v2/artifacts/"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite replay output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

