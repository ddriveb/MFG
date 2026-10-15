"""Portable parity runner for the archived optimized G4 fused planner ports.

This checks the resident G4 planners against their Torch reference on one
visible CUDA device. It does not launch the historical four-rank training
timing runner or claim a new physical training measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))
sys.path.insert(0, str(V2 / "external" / "physical_g4" / "optimized130"))


def new_output(raw: str | None) -> Path:
    root = (V2 / "artifacts").resolve()
    if raw:
        destination = Path(raw).expanduser()
        if not destination.is_absolute():
            destination = V2 / destination
        destination = destination.resolve()
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = root / f"physical-g4-parity-{stamp}"
    if not destination.is_relative_to(root):
        raise ValueError("physical G4 output must be under v2/artifacts/")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def recorded_counts(limit_states: int | None):
    archive = V2 / "evidence" / "artifacts/g4-v100-timing100-20261014-a/optimized-final-1791914591.tar.gz"
    cases = []
    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile("results/optimized-three-arm-20261014-a/raw-rank-0.jsonl")
        if stream is None:
            raise FileNotFoundError("rank-zero G4 evidence was not found in the frozen archive")
        for line in stream:
            row = json.loads(line)
            if row["arm"] != "gnn" or not 2 <= row["sample"] <= 151:
                continue
            if limit_states is not None and row["sample"] >= 2 + limit_states:
                continue
            for layer, plan in enumerate(row["planner"]):
                counts = [value for source in plan["source_counts"] for value in source]
                cases.append((row["sample"], layer, counts))
    expected = 24 * (limit_states if limit_states is not None else 150)
    if len(cases) != expected:
        raise AssertionError(f"expected {expected} recorded G4 count vectors, got {len(cases)}")
    return archive, cases


def run_parity(device_name: str, limit_states: int | None, output: str | None) -> dict:
    import torch

    from mfg_hedge.granite_g4_feedback_planner_v1 import (
        DenseSupportPlanner,
        baseline_quotas,
        default_model_config,
        load_checkpoint,
    )
    from optimized130_port import OptimizedPlanner

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("recorded G4 fused parity requires a visible CUDA device")
    torch.cuda.set_device(device)
    if torch.cuda.device_count() < device.index + 1:
        raise RuntimeError(f"CUDA device is unavailable: {device}")

    archive, cases = recorded_counts(limit_states)
    archived_config = json.loads((V2 / "external/physical_g4/optimized130/config.json").read_text(encoding="utf-8"))
    checkpoint = V2 / "models/checkpoints/g4/best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != archived_config["checkpoint_sha256"]:
        raise ValueError("packaged G4 checkpoint does not match the frozen optimized-port config")
    reference = DenseSupportPlanner(
        load_checkpoint(checkpoint, device, checkpoint_hash), default_model_config(), device=device
    )
    fused = OptimizedPlanner(
        load_checkpoint(checkpoint, device, checkpoint_hash), default_model_config(), device=device
    )

    comparisons = 0
    maximum_share_difference = 0.0
    with torch.no_grad():
        for sample, layer, values in cases:
            counts = torch.tensor(values, device=device, dtype=torch.int64)
            expected = {
                "gnn": reference.quotas(counts).clone(),
                "sinkhorn": baseline_quotas(counts.reshape(4, 32), "sinkhorn").reshape(128, 2),
                "lazarus": baseline_quotas(counts.reshape(4, 32), "lazarus").reshape(128, 2),
            }
            for arm in ("gnn", "sinkhorn", "lazarus"):
                actual = fused.dispatch(counts, arm)
                if not torch.equal(actual, expected[arm]):
                    difference = int((actual != expected[arm]).sum().item())
                    raise AssertionError((sample, layer, arm, difference))
                if not torch.equal(actual.sum(-1), counts) or bool((actual < 0).any().item()):
                    raise AssertionError((sample, layer, arm, "quota conservation"))
                comparisons += 1
            fused.count_buffer.copy_(counts)
            fused.fused_refresh(fused.count_buffer)
            share_delta = float((fused.fused() - reference.shares).abs().max().item())
            maximum_share_difference = max(maximum_share_difference, share_delta)
            if share_delta > 1e-10:
                raise AssertionError((sample, layer, "GNN share parity", share_delta))

    result = {
        "passed": True,
        "execution": "single visible CUDA device; fused planner parity only; no four-rank timing claim",
        "states": limit_states if limit_states is not None else 150,
        "count_vectors": len(cases),
        "exact_quota_comparisons": comparisons,
        "maximum_gnn_share_absolute_difference": maximum_share_difference,
        "checkpoint_sha256": checkpoint_hash,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(device),
    }
    destination = new_output(output)
    (destination / "qualification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(destination.relative_to(V2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero-template", "recorded-planner"), default="zero-template")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-states", type=int, help="qualify the first N saved 24-layer states")
    parser.add_argument(
        "--python-include",
        action="append",
        help="existing Python header directory needed by Triton; repeat for a multi-directory include tree",
    )
    parser.add_argument("--output", help="new output path under v2/artifacts/")
    args = parser.parse_args(argv)
    if args.limit_states is not None and args.limit_states <= 0:
        parser.error("--limit-states must be positive")
    if args.mode == "zero-template":
        from physical_g4_zero_template import check_zero_template

        report = check_zero_template()
    else:
        if args.python_include:
            includes = [Path(value).expanduser().resolve() for value in args.python_include]
            missing = [path for path in includes if not path.is_dir()]
            if missing:
                parser.error(f"--python-include is not a directory: {missing[0]}")
            os.environ["CPATH"] = os.pathsep.join(map(str, includes))
        report = run_parity(args.device, args.limit_states, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
