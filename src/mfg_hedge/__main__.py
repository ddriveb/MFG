"""CLI entry points: environment check and the Healthy + No Hedge slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Sequence
import sys

from .artifacts import write_run_directory, write_summary_json
from .calibration import build_calibration_table
from .campaign import run_multiseed_campaign
from .common_state import CommonState
from .common_state_metrics import build_common_state_summary
from .common_state_simulation import simulate_common_state_no_hedge
from .config import (
    load_config,
    load_config_with_sha256,
    load_common_state_config_with_sha256,
    load_paired_config_with_sha256,
)
from .mfg_solver import state_capacity
from .metrics import build_summary
from .paired import run_paired_comparison, trace_identity_fingerprint
from .paired_metrics import build_arm_summary, build_comparison
from .simulation import simulate_healthy_no_hedge
from .scaled_experiment import (
    run_frozen_scaled_development,
    scaled_manifest,
    write_scaled_artifact,
)
from .stress import load_artifact_name, run_load_stress_diagnostics
from .transient_search import (
    frozen_manifest,
    run_frozen_development_search,
    write_development_search_artifact,
)
from .workload import generate_workload, generate_workload_with_replay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfg-hedge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate environment and configuration")
    check.add_argument("--config", required=True, type=Path)
    simulate = subparsers.add_parser(
        "simulate-healthy-no-hedge",
        help="run the minimal Healthy + No Hedge queue simulation",
    )
    simulate.add_argument("--config", required=True, type=Path)
    simulate.add_argument("--tokens", required=True, type=int)
    simulate.add_argument(
        "--run-id",
        default=None,
        help="artifact directory name; defaults to a timestamped identifier",
    )
    simulate.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    simulate_common = subparsers.add_parser(
        "simulate-common-state-no-hedge",
        help="run the Common State + No Hedge failure simulation",
    )
    simulate_common.add_argument("--config", required=True, type=Path)
    simulate_common.add_argument("--tokens", required=True, type=int)
    simulate_common.add_argument(
        "--run-id",
        default=None,
        help="artifact directory name; defaults to a timestamped identifier",
    )
    simulate_common.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    compare = subparsers.add_parser(
        "compare-mfg-hedge-vs-no-hedge",
        help="run the paired MFG-Hedge vs No-Hedge comparison on one trace",
    )
    compare.add_argument("--config", required=True, type=Path)
    compare.add_argument("--tokens", required=True, type=int)
    compare.add_argument(
        "--run-id",
        default=None,
        help="artifact directory name; defaults to a timestamped identifier",
    )
    compare.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    campaign = subparsers.add_parser(
        "campaign-mfg-hedge-vs-no-hedge",
        help="run the configured paired multi-seed campaign",
    )
    campaign.add_argument("--config", required=True, type=Path)
    campaign.add_argument(
        "--run-id",
        default=None,
        help="artifact directory name; defaults to a timestamped identifier",
    )
    campaign.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    stress = subparsers.add_parser(
        "diagnose-mfg-hedge-loads",
        help="run bounded solver diagnostics at explicitly listed stress loads",
    )
    stress.add_argument("--config", required=True, type=Path)
    stress.add_argument("--loads", required=True, type=float, nargs="+")
    stress.add_argument("--run-id", default=None)
    stress.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    transient = subparsers.add_parser(
        "search-transient-rules",
        help="run the frozen development-only 81-rule transient-control search",
    )
    transient.add_argument("--config", required=True, type=Path)
    transient.add_argument("--run-id", default=None)
    transient.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    scaled = subparsers.add_parser(
        "search-scaled-multi-backup",
        help="run the frozen 8-Expert / dual-Backup development experiment",
    )
    scaled.add_argument("--config", required=True, type=Path)
    scaled.add_argument("--run-id", default=None)
    scaled.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    return parser


def _default_run_id(
    experiment_name: str, token_count: int, scenario: str = "healthy-no-hedge"
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{experiment_name}-{scenario}-t{token_count}-{stamp}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "check":
        config = load_config(args.config)
        result = {
            "status": "ok",
            "python": platform.python_version(),
            "experiment": config.experiment_name,
            "single_domain_failure_load": config.single_domain_failure_load,
            "audit_findings": list(config.audit_findings()),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "search-transient-rules":
        try:
            config, config_sha256 = load_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name, 64, scenario="transient-control-81-rule-fit"
            )
            search = run_frozen_development_search(config)
            from . import __version__

            manifest = frozen_manifest(config, config_sha256, __version__)
            run_dir = write_development_search_artifact(
                args.artifacts_root, run_id, manifest, search
            )
        except (ValueError, RuntimeError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "ok", "run_id": run_id,
                          "run_dir": str(run_dir),
                          "selected_rule": search["selected_rule"],
                          "feasible_rule_count": search["feasible_rule_count"],
                          "selected_vs_baseline": search["selected_vs_baseline"]},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "search-scaled-multi-backup":
        try:
            config, config_sha256 = load_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name, 150000, scenario="scaled-8e-dual-backup-fit"
            )
            def progress(done, total, rule):
                if done == 1 or done % 9 == 0 or done == total:
                    print(f"progress: dual rule {done}/{total} ({rule})",
                          file=sys.stderr, flush=True)
            result = run_frozen_scaled_development(config, progress=progress)
            from . import __version__

            manifest = scaled_manifest(config, config_sha256, __version__)
            run_dir = write_scaled_artifact(
                args.artifacts_root, run_id, manifest, result
            )
        except (ValueError, RuntimeError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"status": "ok", "run_id": run_id,
                          "run_dir": str(run_dir),
                          "token_count": result["token_count"],
                          "dual_selected_rule": result["dual_backup_search"]["selected_rule"],
                          "comparisons": result["comparisons"]},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "simulate-healthy-no-hedge":
        config, config_sha256 = load_config_with_sha256(args.config)
        run_id = args.run_id or _default_run_id(config.experiment_name, args.tokens)
        trace = generate_workload(config, args.tokens)
        result = simulate_healthy_no_hedge(
            trace, replica_count=config.replicas_per_expert
        )
        summary = build_summary(
            config, trace, result, run_id=run_id, config_sha256=config_sha256
        )
        try:
            summary_path = write_summary_json(args.artifacts_root, run_id, summary)
        except FileExistsError:
            print(
                f"error: run directory already exists: {args.artifacts_root / run_id}",
                file=sys.stderr,
            )
            return 2
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"status": "ok", "run_id": run_id, "summary": str(summary_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "simulate-common-state-no-hedge":
        try:
            config, config_sha256 = load_common_state_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name, args.tokens, scenario="common-state-no-hedge"
            )
            trace = generate_workload_with_replay(config.base_config(), args.tokens)
            last_arrival = trace.tokens[-1].arrival_time
            if not last_arrival > config.recovered_start:
                raise ValueError(
                    f"last arrival {last_arrival:.6f} does not exceed "
                    f"recovered_start {config.recovered_start}; increase --tokens"
                )
            result = simulate_common_state_no_hedge(
                trace,
                timeline=config.timeline(),
                degraded_slowdown=config.degraded_slowdown,
                replica_count=config.replicas_per_expert,
            )
            summary = build_common_state_summary(
                config, trace, result, run_id=run_id, config_sha256=config_sha256
            )
            summary_path = write_summary_json(args.artifacts_root, run_id, summary)
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"status": "ok", "run_id": run_id, "summary": str(summary_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "compare-mfg-hedge-vs-no-hedge":
        try:
            config, config_sha256 = load_paired_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name, args.tokens, scenario="paired-mfg-vs-no-hedge"
            )
            table = build_calibration_table(config.base_config())
            run = run_paired_comparison(config, table, args.tokens)
            timeline = config.timeline()
            capacities = {
                "H": state_capacity(config.base_config(), CommonState.HEALTHY),
                "D": state_capacity(config.base_config(), CommonState.DEGRADED),
                "F": state_capacity(config.base_config(), CommonState.FAILED),
                "R": state_capacity(config.base_config(), CommonState.HEALTHY),
            }
            summary_a = build_arm_summary(
                run.arm_a,
                timeline,
                arm="no_hedge",
                projection=None,
                bin_width=config.storm_bin_width,
                capacities=capacities,
            )
            summary_b = build_arm_summary(
                run.arm_b,
                timeline,
                arm="mfg_hedge",
                projection=run.projection,
                bin_width=config.storm_bin_width,
                capacities=capacities,
            )
            comparison = build_comparison(summary_a, summary_b)
            for arm_name, summary in (("A", summary_a), ("B", summary_b)):
                broken = [
                    name
                    for name, ok in summary["invariants"].items()
                    if ok is not True
                ]
                if broken:
                    raise ValueError(
                        f"invariants violated in arm {arm_name}: {broken}"
                    )
            comparison["trace_identity"] = trace_identity_fingerprint(run.trace)
            comparison["config"] = {
                "schema_version": config.schema_version,
                "config_sha256": config_sha256,
                "resolved_config": {
                    name: getattr(config, name)
                    for name in type(config).__dataclass_fields__
                },
            }
            from . import __version__

            comparison["simulator_version"] = __version__
            comparison["tau0"] = run.tau0
            solution_mapping = run.solution.to_mapping()
            comparison["mfg_solution"] = solution_mapping["states"]
            comparison["solver_parameters"] = solution_mapping["solver_parameters"]
            comparison["invariants"] = {
                "arm_a": summary_a["invariants"],
                "arm_b": summary_b["invariants"],
            }
            manifest = {
                "run_id": run_id,
                "scenario": "paired_mfg_vs_no_hedge",
                "simulator_version": __version__,
                "token_count": args.tokens,
                "config_sha256": config_sha256,
                "config_schema_version": config.schema_version,
                "trace_identity": comparison["trace_identity"],
                "tau0": run.tau0,
                "runtime_costs": {
                    "incremental_work_cost": config.incremental_work_cost,
                    "wasted_work_cost": config.wasted_work_cost,
                },
                "solver_status": {
                    state: block["status"]
                    for state, block in comparison["mfg_solution"].items()
                },
            }
            run_dir = write_run_directory(
                args.artifacts_root,
                run_id,
                {
                    "manifest.json": manifest,
                    "no_hedge_summary.json": summary_a,
                    "mfg_hedge_summary.json": summary_b,
                    "comparison.json": comparison,
                },
            )
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"status": "ok", "run_id": run_id, "run_dir": str(run_dir)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "campaign-mfg-hedge-vs-no-hedge":
        try:
            config, config_sha256 = load_paired_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name,
                config.tokens_per_run,
                scenario=f"paired-campaign-s{config.seed_count}",
            )
            table = build_calibration_table(config.base_config())
            campaign = run_multiseed_campaign(config, table)
            from . import __version__

            resolved_config = {
                name: getattr(config, name)
                for name in type(config).__dataclass_fields__
            }
            manifest = {
                "run_id": run_id,
                "scenario": "paired_mfg_vs_no_hedge_multiseed_campaign",
                "simulator_version": __version__,
                "config_sha256": config_sha256,
                "config_schema_version": config.schema_version,
                "resolved_config": resolved_config,
                "token_count_per_seed": config.tokens_per_run,
                "evaluation_seeds": [
                    record["evaluation_seed"] for record in campaign.records
                ],
                "calibration_provenance": dict(table.provenance),
                "mfg_solution": campaign.solution.to_mapping()["states"],
                "artifact_files": [
                    "manifest.json",
                    "aggregate.json",
                    *[
                        f"seed_{record['evaluation_seed']}.json"
                        for record in campaign.records
                    ],
                ],
            }
            files = {
                "manifest.json": manifest,
                "aggregate.json": campaign.aggregate,
            }
            files.update(
                {
                    f"seed_{record['evaluation_seed']}.json": record
                    for record in campaign.records
                }
            )
            run_dir = write_run_directory(args.artifacts_root, run_id, files)
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"status": "ok", "run_id": run_id, "run_dir": str(run_dir)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "diagnose-mfg-hedge-loads":
        try:
            config, config_sha256 = load_paired_config_with_sha256(args.config)
            run_id = args.run_id or _default_run_id(
                config.experiment_name,
                config.smoke_tokens,
                scenario="mfg-load-stress-diagnostics",
            )
            records = run_load_stress_diagnostics(config, args.loads)
            from . import __version__

            manifest = {
                "run_id": run_id,
                "scenario": "mfg_load_stress_diagnostics",
                "classification": "stress_diagnostic_only",
                "simulator_version": __version__,
                "base_config_sha256": config_sha256,
                "base_healthy_offered_load": config.healthy_offered_load,
                "diagnostic_loads": [
                    record["healthy_offered_load"] for record in records
                ],
                "paired_simulations_run": 0,
            }
            files = {"manifest.json": manifest}
            files.update(
                {
                    load_artifact_name(record["healthy_offered_load"]): record
                    for record in records
                }
            )
            run_dir = write_run_directory(args.artifacts_root, run_id, files)
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"status": "ok", "run_id": run_id, "run_dir": str(run_dir)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
