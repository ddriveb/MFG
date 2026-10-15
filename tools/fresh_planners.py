"""Regenerate V2 Gate inputs and compare fresh GPU planner quotas to saved plans."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(V2 / "tools"))

from gate_generation import (  # noqa: E402
    FAMILIES,
    g64_gate,
    hotspot_gate,
    hotspot_support,
    multiscale_gate,
)
from mfg_hedge.moe_communication_social_v2 import (  # noqa: E402
    BLOCK_NAMES,
    CommunicationConfig,
    RouteRow,
    SparseSocialProblem,
)

EVIDENCE = V2 / "evidence" / "artifacts"
CHECKPOINTS = V2 / "models" / "checkpoints"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def new_output(raw_path: str | None) -> Path:
    root = (V2 / "artifacts").resolve()
    if raw_path:
        destination = Path(raw_path).expanduser()
        if not destination.is_absolute():
            destination = V2 / destination
        destination = destination.resolve()
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = root / f"fresh-planners-{stamp}"
    if not destination.is_relative_to(root):
        raise ValueError("fresh planner output must be under v2/artifacts/")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite fresh planner output: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def generated_cases(selected: set[str], limit: int | None):
    if "g64" in selected:
        base = EVIDENCE / "moe-g64-three-arm-aicb150-20261014-a"
        records = load_json(base / "counts.json")
        support_protocol = load_json(EVIDENCE / "moe-g64-depth-pressure-20261013-a" / "r2" / "protocol.json")
        support = tuple(tuple(row) for row in support_protocol["support"])
        assert len(records) == 150
        for index, record in enumerate(records[:limit]):
            family = FAMILIES[index % len(FAMILIES)]
            assert record["family"] == family
            snapshot = g64_gate(support, record["seed"], family)
            counts = [n for row in snapshot.counts for n in row]
            patterns = snapshot.patterns
            assert counts == record["counts"]
            seed = record["seed"]
            state_id = f"g64-{seed}"
            plans = {
                arm: load_json(base / "plans" / f"{state_id}-{arm}.json")
                for arm in ("gnn4", "sinkhorn_fixed", "lazarus_inspired")
            }
            yield make_case("g64", state_id, 64, counts, support, plans, {
                "gnn4": "g64_r4",
            }, family=family, fresh_pattern_rows=len(patterns) * 64)

    if "multiscale" in selected:
        base = EVIDENCE / "moe-gnn-multiscale-execution-20261014-a"
        for record in load_json(base / "counts.json")[:limit]:
            support = tuple(tuple(row) for row in record["support"])
            counts, patterns, strength = multiscale_gate(
                support, record["seed"], record["family"], record["P"]
            )
            assert counts == record["counts"]
            assert patterns == record["patterns"] and strength == record["strength"]
            state_id = f"p{record['P']}-{record['family']}-{record['seed']}"
            plans = {
                arm: load_json(base / "plans" / f"{state_id}-{arm}.json")
                for arm in ("gnn4", "sinkhorn_fixed", "lazarus_inspired")
            }
            yield make_case("multiscale", state_id, record["P"], counts, support, plans, {
                "gnn4": "multiscale",
            }, family=record["family"], fresh_pattern_rows=len(patterns) * 64)

    if "global_hotspot" in selected:
        base = EVIDENCE / "moe-gnn-global-hotspot-execution-20261014-a"
        protocol = load_json(base / "protocol.json")
        for record in load_json(base / "counts.json")[:limit]:
            ranks = record["P"]
            support, hot = hotspot_support(ranks, protocol["support_seed"])
            assert [list(row) for row in support] == record["support"]
            assert list(hot) == record["hot_experts"]
            counts, patterns, actual = hotspot_gate(
                support, hot, record["seed"], record["rho"], ranks
            )
            assert counts == record["counts"]
            assert patterns == record["patterns"]
            assert actual == record["actual_hot_route_fraction"]
            state_id = record["state_id"]
            plans = {
                arm: load_json(base / "plans" / f"{state_id}-{arm}.json")
                for arm in ("gnn_mixed", "gnn_hot", "sinkhorn_fixed", "lazarus_inspired")
            }
            yield make_case("global_hotspot", state_id, ranks, counts, support, plans, {
                "gnn_mixed": "multiscale",
                "gnn_hot": "global_hotspot",
            }, family=record["family"], hot_route_fraction=actual,
               fresh_pattern_rows=len(patterns) * 64)


def make_case(campaign, state_id, ranks, counts, support, plans, gnn_checkpoints, **metadata):
    reference = next(iter(plans.values()))
    config_data = dict(reference["communication_config"])
    config_data["gpu_nodes"] = tuple(config_data["gpu_nodes"])
    config = CommunicationConfig(**config_data)
    rows = []
    index = 0
    for source, candidates in enumerate(support):
        assert len(candidates) == 16
        for expert in candidates:
            saved = reference["rows"][index]
            assert saved["source_gpu"] == source and saved["expert_id"] == expert
            allowed = tuple(saved["allowed_gpus"])
            assert allowed == (expert, (expert + 1) % ranks)
            rows.append(RouteRow(source, expert, counts[index], allowed))
            index += 1
    assert index == len(reference["rows"]) == len(counts)
    problem = SparseSocialProblem.from_rows(tuple(rows), config)
    for plan in plans.values():
        assert plan["rows"] == [
            {"source_gpu": row.source_gpu, "expert_id": row.expert_id,
             "demand": row.demand, "allowed_gpus": list(row.allowed_gpus)}
            for row in rows
        ]
    return {
        "campaign": campaign,
        "state_id": state_id,
        "ranks": ranks,
        "counts": counts,
        "support": support,
        "plans": plans,
        "problem": problem,
        "gnn_checkpoints": gnn_checkpoints,
        "metadata": metadata,
    }


def run(args) -> dict:
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fresh GNN/Sinkhorn/Lazarus parity requires an available CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from mfg_hedge.moe_fixed_sinkhorn_gpu_v1 import build_dispatcher as build_sinkhorn
    from mfg_hedge.moe_gpu_cached_warmstart_v1 import CachedGpuWarmStart
    from mfg_hedge.moe_gnn_allocation_features_v1 import build_allocation_feature_refresh
    from mfg_hedge.moe_gnn_allocation_fused_v1 import build_fused_allocation
    from mfg_hedge.moe_lazarus_ring_gpu_v1 import build_dispatcher as build_lazarus
    from mfg_hedge.moe_load_feedback_gnn_v1 import build_load_feedback_gnn

    model_names = set()
    if "g64" in args.campaigns:
        model_names.add("g64_r4")
    if "multiscale" in args.campaigns:
        model_names.add("multiscale")
    if "global_hotspot" in args.campaigns:
        model_names.update(("multiscale", "global_hotspot"))
    models = {}
    for name in sorted(model_names):
        model = build_load_feedback_gnn(feedback_rounds=4).to(device)
        checkpoint = CHECKPOINTS / name / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
        model.eval()
        models[name] = model

    cases = generated_cases(set(args.campaigns), args.limit)
    cache_by_topology = {}
    gnn_models = models
    operator_cache = {}
    checked = {"g64": 0, "multiscale": 0, "global_hotspot": 0}
    arm_counts = {}
    changed = []
    for case in cases:
        problem = case["problem"]
        counts = torch.tensor(case["counts"], device=device, dtype=torch.int64).contiguous()
        support_key = tuple(tuple(row) for row in case["support"])
        topology_key = (case["ranks"], support_key, tuple(problem.config.gpu_nodes))
        row_sources = tuple(row.source_gpu for row in problem.rows)
        row_experts = tuple(row.expert_id for row in problem.rows)
        expected = {arm: plan["quotas"] for arm, plan in case["plans"].items()}

        for arm, checkpoint_name in case["gnn_checkpoints"].items():
            cache_key = (checkpoint_name, topology_key)
            if cache_key not in cache_by_topology:
                cache = CachedGpuWarmStart(problem, gnn_models[checkpoint_name], device=str(device), projection="active")
                cache_by_topology[cache_key] = cache
                fused = build_fused_allocation(gnn_models[checkpoint_name], cache.graph, feedback_rounds=4)
                refresh = build_allocation_feature_refresh(cache)
                operator_cache[cache_key] = (fused, refresh)
            cache = cache_by_topology[cache_key]
            fused, refresh = operator_cache[cache_key]
            cache.update_features(counts)
            cache.model.prepare(cache.graph)
            with torch.no_grad():
                reference_share, reference_prices = cache.model(cache.graph)
                reference_quota = cache.runtime.round_shares(reference_share).clone()
                reference_upper = problem.objective_value(reference_share.detach().cpu().tolist())
                price_mapping = {
                    name: values.detach().cpu().tolist()
                    for name, values in zip(BLOCK_NAMES, reference_prices)
                }
                reference_lower = problem.dual_lower_bound(price_mapping)
            refresh(counts)
            fused_share = fused()
            torch.cuda.synchronize(device)
            fused_quota = cache.runtime.round_shares(fused_share).clone()
            max_share_delta = float((reference_share - fused_share).abs().max().item())
            assert max_share_delta <= 1e-9, (case["state_id"], arm, max_share_delta)
            assert torch.equal(reference_quota, fused_quota), (case["state_id"], arm, "fused-vs-framework integer mismatch")
            observed = fused_quota.cpu().tolist()
            assert observed == expected[arm], (case["state_id"], arm, "saved integer quota differs")
            score = problem.integer_score(observed).total_ms
            saved_score = case["plans"][arm].get("integer_F_ms", case["plans"][arm].get("integer_model_F_ms"))
            assert saved_score is None or abs(score - saved_score) < 1e-9
            if "gnn_native" in case["plans"][arm]:
                saved_upper = case["plans"][arm]["gnn_native"][arm]["upper_ms"]
                saved_lower = case["plans"][arm]["gnn_native"][arm]["lower_ms"]
            else:
                saved_upper = case["plans"][arm]["gnn_upper_ms"]
                saved_lower = case["plans"][arm]["gnn_lower_ms"]
            assert abs(reference_upper - saved_upper) < 1e-8, (case["state_id"], arm, reference_upper, saved_upper)
            assert abs(reference_lower - saved_lower) < 1e-8, (case["state_id"], arm, reference_lower, saved_lower)
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            changed.append({"campaign": case["campaign"], "state_id": case["state_id"], "arm": arm,
                            "max_share_delta_vs_framework": max_share_delta, "integer_F_ms": score,
                            "gnn_upper_ms": reference_upper, "gnn_lower_ms": reference_lower})

        for arm, builder in (("sinkhorn_fixed", build_sinkhorn), ("lazarus_inspired", build_lazarus)):
            cache_key = (arm, topology_key)
            if cache_key not in operator_cache:
                operator_cache[cache_key] = builder(row_sources, row_experts, ranks=case["ranks"], device=str(device))
            dispatcher = operator_cache[cache_key]
            observed = dispatcher(counts).clone().cpu().tolist()
            assert observed == expected[arm], (case["state_id"], arm, "saved integer quota differs")
            score = problem.integer_score(observed).total_ms
            saved_score = case["plans"][arm].get("integer_F_ms", case["plans"][arm].get("integer_model_F_ms"))
            assert saved_score is None or abs(score - saved_score) < 1e-9
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            changed.append({"campaign": case["campaign"], "state_id": case["state_id"], "arm": arm,
                            "integer_F_ms": score})
        checked[case["campaign"]] += 1

    import mfg_hedge.moe_communication_social_v2 as domain
    package_root = (V2 / "src").resolve()
    imported = {
        name: Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("mfg_hedge.") and getattr(module, "__file__", None)
    }
    assert imported and all(path.is_relative_to(package_root) for path in imported.values()), imported
    return {
        "passed": True,
        "mode": "fresh_seeded_gate_and_gpu_planner_inference",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "full_campaign_counts": checked,
        "planner_quota_comparisons": arm_counts,
        "model_checkpoints": sorted(model_names),
        "model_checkpoint_sha256": {
            name: hashlib.sha256((CHECKPOINTS / name / "best.pt").read_bytes()).hexdigest()
            for name in sorted(model_names)
        },
        "imported_domain_module": str(Path(domain.__file__).resolve().relative_to(V2)),
        "loaded_mfg_hedge_module_count": len(imported),
        "plan_results": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("all", "g64", "multiscale", "global_hotspot"), default="all")
    parser.add_argument("--limit", type=int, help="run only the first N states per selected campaign (GPU smoke mode)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for FP64 GNN and planner kernels")
    parser.add_argument("--output", help="new report directory under v2/artifacts/")
    parser.add_argument("--python-include", help="optional external C/Python headers include directory for Triton")
    parser.add_argument("--triton-cache", help="optional Triton cache directory; defaults inside the new report directory")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.campaigns = ("g64", "multiscale", "global_hotspot") if args.campaign == "all" else (args.campaign,)
    if args.python_include:
        include = Path(args.python_include).expanduser().resolve()
        if not include.is_dir():
            raise FileNotFoundError(include)
        os.environ["CPATH"] = str(include)
    output = new_output(args.output)
    if args.triton_cache:
        cache = Path(args.triton_cache).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
    else:
        cache = output / "triton-cache"
        cache.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    report = run(args)
    report["output_directory"] = str(output.relative_to(V2))
    report["run_kind"] = "smoke" if args.limit is not None else "full"
    report["timing_note"] = "This reports new inference results; saved wall-time values remain frozen evidence and are not inferred."
    (output / "fresh-planner-results.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "plan_results"}, sort_keys=True))
    print(f"Plan-level results: {output.relative_to(V2) / 'fresh-planner-results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
