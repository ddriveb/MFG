"""Ticket 15 engine benchmark and r2 cross-check.

Timing-only harness: consumes deterministic traces rebuilt from the frozen r2
identities; generates no new experiment data and publishes no scientific
conclusion. Modes:
  --phase before|after   write artifacts/engine-equivalence-benchmark-20260911/<phase>.json
  --crosscheck           write artifacts/engine-equivalence-benchmark-20260911/r2_crosscheck.json
  --compare              print the before/after speedup table
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mfg_hedge.population_estimator import BIN_SCHEMA_V1, run_population_forward
from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
from mfg_hedge.token_mfg_formal_backend import FormalQualificationBackend
from mfg_hedge.token_mfg_qualification_backend import (
    RoutingPolicySeed,
    _PolicyStateAdapter,
)
from mfg_hedge.token_mfg_qualification_campaign import QualificationPlan
from mfg_hedge.token_mfg_qualification_state_machine import _pack, _unpack
from mfg_hedge.token_mfg_streaming import build_streaming_panel
from mfg_hedge.token_mfg_streaming_formal import streaming_iteration_episode_worker
from mfg_hedge.token_population_response import (
    action_buckets_for_token,
    evaluate_finite_k_deviation,
)

ARTIFACT = ROOT / "artifacts" / "engine-equivalence-benchmark-20260911"
RUN_DIR = ROOT / ".scratch" / "reliability-aware-token-mfg" / "qualification-runs" / "reliability-aware-token-mfg-qualification-20260911-r2"


def _canonical(value):
    import dataclasses
    if dataclasses.is_dataclass(value):
        return {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    return value


def _digest(value):
    import hashlib
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _target_cost(result, target_token_id):
    token = next(row for row in result.physical.token_results if row.token_id == target_token_id)
    completed = [row for row in token.attempts if row.terminal_status == "completed"]
    return completed[-1].terminal_time if completed else 0.0


def _adapter_factory():
    return lambda: _PolicyStateAdapter(RoutingPolicySeed(kind="uniform"), BIN_SCHEMA_V1)


def _timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def _summarize(times):
    return {
        "n": len(times),
        "mean_s": statistics.fmean(times),
        "worst_s": max(times),
    }


def run_phase(phase: str) -> None:
    plan = QualificationPlan.r2()
    report = {
        "phase": phase,
        "python": platform.python_version(),
        "host": platform.platform(),
        "k8": {},
        "k64": {},
    }

    for key, k, count, deviation_count in (("k8", 8, 32, 8), ("k64", 64, 4, 4)):
        panel = build_streaming_panel(plan, k=k, count=count)
        forward_times = []
        deviation_times = []
        for spec in panel.specs:
            trace = spec.generate_trace()
            elapsed, _ = _timed(lambda: simulate_simultaneous_routing(
                trace, _adapter_factory()(), policy_name="benchmark",
            ))
            forward_times.append(elapsed)
            if len(deviation_times) < deviation_count:
                factory = _adapter_factory()
                episode = run_population_forward(trace, factory)
                batch = episode.batches[0]
                target = batch.eta.token_ids[0]
                buckets = action_buckets_for_token(batch, target)

                def run_dev():
                    return evaluate_finite_k_deviation(
                        trace, episode, factory, target, buckets, _target_cost,
                        cost_model_id="latency_v1",
                    )
                elapsed, _ = _timed(run_dev)
                deviation_times.append(elapsed)
        report[key] = {
            "forward_call_s": _summarize(forward_times),
            "deviation_panel_s": _summarize(deviation_times),
        }

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT / f"{phase}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written: {out}")


def run_crosscheck() -> None:
    plan = QualificationPlan.r2()
    panel = build_streaming_panel(plan, k=8)
    checkpoint = json.loads((RUN_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    records = {
        key: rec for key, rec in checkpoint["records"].items()
        if rec.get("status") == "complete" and "complete-panel" in key
    }
    policy = FormalQualificationBackend.initial_seed("uniform")
    rows = {}
    mismatches = []
    for key, rec in sorted(records.items()):
        episode_index = int(key.rsplit(":", 1)[1])
        spec = panel.specs[episode_index]
        blob = json.loads((RUN_DIR / "blobs" / rec["output"]["__content_addressed_blob__"]).read_text(encoding="utf-8"))
        payload = {
            "episode_spec": spec,
            "policy": policy,
            "model_id": "unpriced_mfg",
            "iteration": 0,
            "mean_field_namespace": plan.continuation_namespace,
            "mean_field_macro_seed": plan.continuation_macro_seed,
            "finite_k_namespace": plan.finite_k_namespace,
            "finite_k_macro_seed": plan.finite_k_macro_seed,
        }
        worker_output = streaming_iteration_episode_worker(payload)
        output = worker_output["output"]
        checks = {
            "trace_fingerprint": output["trace_fingerprint"] == blob["trace_fingerprint"],
            "target_state": output["target_state"] == blob["target_state"],
            "episode_identity": list(output["episode_identity"]) == list(blob["episode_identity"]),
            "snapshot": _digest(_unpack(output["snapshot"])) == _digest(_unpack(blob["snapshot"])),
            "compact_tokens": _digest(_unpack(output["compact_tokens"])) == _digest(_unpack(blob["compact_tokens"])),
            "continuation_rows": _digest(_unpack(output["continuation_rows"])) == _digest(_unpack(blob["continuation_rows"])),
            "finite_rows": _digest(_unpack(output["finite_rows"])) == _digest(_unpack(blob["finite_rows"])),
            "completed_calls": worker_output["completed_calls"] == rec["completed_calls"],
        }
        rows[key] = {"all_match": all(checks.values()), "checks": checks}
        if not all(checks.values()):
            mismatches.append(key)
    report = {
        "cases": len(rows),
        "matched": len(rows) - len(mismatches),
        "mismatched": mismatches,
        "rows": rows,
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT / "r2_crosscheck.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"cross-check: {report['matched']}/{report['cases']} matched; written: {out}")


def run_compare() -> None:
    before = json.loads((ARTIFACT / "before.json").read_text(encoding="utf-8"))
    after = json.loads((ARTIFACT / "after.json").read_text(encoding="utf-8"))
    for key in ("k8", "k64"):
        for section in ("forward_call_s", "deviation_panel_s"):
            b = before[key][section]
            a = after[key][section]
            speedup = b["mean_s"] / a["mean_s"] if a["mean_s"] else float("inf")
            print(
                f"{key} {section}: mean {b['mean_s']:.4f}s -> {a['mean_s']:.4f}s "
                f"(speedup {speedup:.2f}x); worst {b['worst_s']:.4f}s -> {a['worst_s']:.4f}s"
            )
    # K=64 per-iteration projection with the post-fix per-episode wall.
    a = after["k64"]["deviation_panel_s"]["mean_s"]
    for n_episodes in (32, 280):
        projected_worker = a * n_episodes / 8
        print(
            f"K=64 projection: {n_episodes} episodes/iteration at {a:.3f}s/episode "
            f"with 8 workers -> {projected_worker:.0f}s = {projected_worker/3600:.2f}h per iteration"
        )


if __name__ == "__main__":
    if "--phase" in sys.argv:
        run_phase(sys.argv[sys.argv.index("--phase") + 1])
    elif "--crosscheck" in sys.argv:
        run_crosscheck()
    elif "--compare" in sys.argv:
        run_compare()
    else:
        raise SystemExit("usage: engine_benchmark.py --phase before|after | --crosscheck | --compare")
