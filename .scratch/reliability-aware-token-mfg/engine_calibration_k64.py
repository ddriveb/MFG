"""Step-0 K=64 calibration: measured per-episode full-cost timing.

Consumes deterministic traces rebuilt from the frozen r2 identities via the
real production worker (streaming_iteration_episode_worker). Timing-only:
no new experiment data, no scientific conclusion, no ticket-06 execution.
Writes artifacts/engine-calibration-k64-20260912/calibration.json.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
from mfg_hedge.token_mfg_formal_backend import FormalQualificationBackend
from mfg_hedge.token_mfg_qualification_backend import (
    RoutingPolicySeed,
    _PolicyStateAdapter,
)
from mfg_hedge.population_estimator import BIN_SCHEMA_V1
from mfg_hedge.token_mfg_qualification_campaign import QualificationPlan
from mfg_hedge.token_mfg_streaming import build_streaming_panel
from mfg_hedge.token_mfg_streaming_formal import streaming_iteration_episode_worker

ARTIFACT = ROOT / "artifacts" / "engine-calibration-k64-20260912"
K = 64
N_EPISODES = 32


def main() -> None:
    plan = QualificationPlan.r2()
    panel = build_streaming_panel(plan, k=K, count=N_EPISODES)
    policy = FormalQualificationBackend.initial_seed("uniform")

    forward_times = []
    episode_times = []
    episode_calls = []
    for spec in panel.specs:
        start = time.perf_counter()
        trace = spec.generate_trace()
        simulate_simultaneous_routing(
            trace, _PolicyStateAdapter(RoutingPolicySeed(kind="uniform"), BIN_SCHEMA_V1),
            policy_name="calibration",
        )
        forward_times.append(time.perf_counter() - start)

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
        start = time.perf_counter()
        worker_output = streaming_iteration_episode_worker(payload)
        episode_times.append(time.perf_counter() - start)
        episode_calls.append(int(worker_output["completed_calls"]))

    def summary(times):
        ordered = sorted(times)
        return {
            "n": len(times),
            "mean_s": statistics.fmean(times),
            "median_s": statistics.median(times),
            "worst_s": max(times),
            "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    per_episode_mean = summary(episode_times)["mean_s"]
    report = {
        "purpose": "k64 per-episode full-cost calibration for a 10h-budget setting",
        "python": platform.python_version(),
        "host": platform.platform(),
        "k": K,
        "episodes": N_EPISODES,
        "forward_call_s": summary(forward_times),
        "episode_full_worker_s": summary(episode_times),
        "episode_calls": {
            "mean": statistics.fmean(episode_calls),
            "min": min(episode_calls),
            "max": max(episode_calls),
        },
        "projection_per_iteration_s": {
            str(n): round(per_episode_mean * n / 8, 1) for n in (64, 280, 460, 600)
        },
        "projection_note": "8-worker aggregate; single-process measured wall here",
        "total_wall_s": sum(episode_times) + sum(forward_times),
    }
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT / "calibration.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
