"""Launch the archived optimized G4 planners in a fresh, parameterized run."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(V2 / "external" / "physical_g4" / "optimized130"))


def configure_runtime_environment(environ=None) -> None:
    if environ is None:
        environ = os.environ
    expected = ":4096:8"
    existing = environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, expected):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 for the frozen G4 protocol")
    environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        environ[name] = "1"
    environ.setdefault("OMP_NUM_THREADS", "1")
    environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def validate_external_assets(model_dir: Path, schedule_library: Path, manifest_data: dict) -> None:
    model_dir = Path(model_dir).expanduser().resolve()
    schedule_library = Path(schedule_library).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Granite model directory does not exist: {model_dir}")
    files = manifest_data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Granite model manifest has no file inventory")
    for row in files:
        name = row.get("name")
        size = row.get("bytes")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Granite model manifest contains an unsafe relative path")
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"required local Granite asset is missing: {path}")
        if path.stat().st_size != size:
            raise ValueError(f"local Granite asset size differs from the pinned manifest: {path}")
    if not schedule_library.is_file():
        raise ValueError(f"compiled native schedule library does not exist: {schedule_library}")


@contextmanager
def install_optimized_planners(port_module=None):
    if port_module is None:
        port_module = importlib.import_module("optimized130_port")
    planner_module = importlib.import_module("mfg_hedge.granite_g4_feedback_planner_v1")
    replay_module = importlib.import_module("mfg_hedge.granite_g4_feedback_replay_v1")
    old_planner = planner_module.DenseSupportPlanner
    old_baseline = planner_module.baseline_quotas
    old_replay_baseline = replay_module.baseline_quotas
    planner_module.DenseSupportPlanner = port_module.OptimizedPlanner
    planner_module.baseline_quotas = port_module.optimized_baseline
    replay_module.baseline_quotas = port_module.optimized_baseline
    try:
        if planner_module.DenseSupportPlanner is not port_module.OptimizedPlanner:
            raise AssertionError("optimized DenseSupportPlanner hook did not install")
        if planner_module.baseline_quotas is not port_module.optimized_baseline:
            raise AssertionError("optimized baseline hook did not install")
        if replay_module.baseline_quotas is not port_module.optimized_baseline:
            raise AssertionError("replay baseline hook did not install")
        yield planner_module, replay_module
    finally:
        planner_module.DenseSupportPlanner = old_planner
        planner_module.baseline_quotas = old_baseline
        replay_module.baseline_quotas = old_replay_baseline


def transformed_runner():
    from mfg_hedge import granite_g4_feedback_runner_v1 as runner
    from optimized_full130 import build_source, validate_timing

    source = build_source(inspect.getsource(runner))
    if source.count("dense.quotas(c)") != 1:
        raise ValueError("unexpected reference runner refresh call count")
    source = source.replace("dense.quotas(c)", "dense.refresh(c)")
    source = source.replace(
        "execution_profile='deterministic_whole_expert_fp32'",
        "execution_profile='fused_features_gnn4_sinkhorn50_lazarus_cuda_graph_whole_expert_fp32'",
    )
    source = source.replace(
        "completed_groups=completed_samples,",
        "planner_profile='fused_cuda_graph',completed_groups=completed_samples,",
    )
    compile(source, "portable-optimized-g4-runner", "exec")
    return source, validate_timing


def selftest() -> dict:
    source, validate_timing = transformed_runner()
    class StubPlanner:
        pass

    def stub_baseline(*args, **kwargs):
        return None

    class StubPorts:
        OptimizedPlanner = StubPlanner
        optimized_baseline = staticmethod(stub_baseline)

    with install_optimized_planners(StubPorts) as (planner_module, replay_module):
        if planner_module.DenseSupportPlanner is not StubPlanner:
            raise AssertionError("planner module did not receive stub optimized port")
        if replay_module.baseline_quotas is not stub_baseline:
            raise AssertionError("replay module did not receive stub optimized baseline")
    if "paired_numerics_passed=False" not in source:
        raise AssertionError("strict-fail timing qualification marker was removed")
    if "strict_numerics_passed=False" not in source:
        raise AssertionError("strict full-model numerical failure marker was removed")
    if "dense.refresh(c)" not in source or "fused_cuda_graph" not in source:
        raise AssertionError("optimized G4 planner integration was not applied")
    try:
        validate_timing([], samples=1, layers=24)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete four-rank input was accepted")
    return {
        "passed": True,
        "runner_compiles": True,
        "fused_planner_hook_stub_tested": True,
        "incomplete_rank_rejected": True,
        "strict_full_model_failure_retained": True,
        "gpu_execution": False,
    }


def validate_portable_config(config_data: dict) -> None:
    if config_data.get("schema") != "g4_feedback_paired_v1":
        raise ValueError("unsupported G4 configuration schema")
    if config_data.get("rounds") != 4:
        raise ValueError("optimized G4 ports support rounds=4 only")
    if config_data.get("measured_steps") != 150 or config_data.get("warmup_steps") != 2:
        raise ValueError("fresh optimized G4 config must specify 2 warmup and 150 measured groups")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="compile and validate the runner without CUDA")
    parser.add_argument("--config", default=str(V2 / "configs/g4-optimized-paired.json"))
    parser.add_argument("--output", help="new run directory under v2/artifacts/")
    parser.add_argument("--model-dir", help="caller-provided local Granite model directory")
    parser.add_argument("--schedule-library", help="caller-provided compiled native schedule library")
    parser.add_argument("--local-tiny", action="store_true", help="unsupported by the fused CUDA ports; rejected")
    parser.add_argument("--init-file", help="unsupported by the fused CUDA ports; rejected")
    parser.add_argument("--deadline", help="optional new ISO-8601 deadline, including timezone; default is no lease deadline")
    args = parser.parse_args(argv)
    try:
        configure_runtime_environment()
    except ValueError as exc:
        parser.error(str(exc))
    if args.selftest:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    if args.local_tiny or args.init_file:
        parser.error("--local-tiny and --init-file are unsupported by the fused CUDA ports; use --selftest for CPU-only checks")
    config = Path(args.config).expanduser().resolve()
    if not config.is_file():
        parser.error(f"configuration file does not exist: {config}")
    try:
        config_data = json.loads(config.read_text(encoding="utf-8"))
        validate_portable_config(config_data)
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if not args.output:
        parser.error("--output is required for a run")
    if not args.model_dir or not args.schedule_library:
        parser.error("full runs require caller-provided --model-dir and --schedule-library")
    try:
        model_manifest = json.loads((V2 / "external" / "granite_model_manifest.json").read_text(encoding="utf-8"))
        validate_external_assets(Path(args.model_dir), Path(args.schedule_library), model_manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = V2 / output
    output = output.resolve()
    artifacts = (V2 / "artifacts").resolve()
    if not output.is_relative_to(artifacts):
        parser.error("--output must be under v2/artifacts/")
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")

    deadline = float("inf")
    if args.deadline:
        try:
            parsed = dt.datetime.fromisoformat(args.deadline)
        except ValueError as exc:
            parser.error(f"invalid ISO-8601 --deadline: {exc}")
        if parsed.tzinfo is None:
            parser.error("--deadline must include a timezone")
        deadline = parsed.timestamp()
        if deadline <= time.time():
            parser.error("--deadline must be in the future")

    source, validate_timing = transformed_runner()
    namespace = {
        "__name__": "mfg_hedge.portable_optimized_g4_runner",
        "__package__": "mfg_hedge",
        "validate_timing": validate_timing,
        "DEADLINE": deadline,
    }
    exec(compile(source, "portable-optimized-g4-runner", "exec"), namespace)
    runner_args = ["--config", str(config), "--output", str(output)]
    if args.model_dir:
        runner_args.extend(("--model-dir", str(Path(args.model_dir).expanduser().resolve())))
    if args.schedule_library:
        runner_args.extend(("--schedule-library", str(Path(args.schedule_library).expanduser().resolve())))
    with install_optimized_planners():
        namespace["main"](runner_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
