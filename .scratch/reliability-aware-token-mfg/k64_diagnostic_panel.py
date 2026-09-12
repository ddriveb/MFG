"""K=64 large-N diagnostic panel driver (overnight diagnostic, NOT qualification).

Runs iteration-0 (uniform start, unpriced_mfg) per-episode production workers at
K=64 with N chosen at launch time, in a separate diagnostic namespace. Per-episode
outputs are written as content-addressed blobs with a manifest; restart skips
completed episodes. Produces an occupancy/floor/UCB summary for the morning
decision. Publishes no qualification, Nash, or MFG claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mfg_hedge.token_mfg_formal_backend import FormalQualificationBackend
from mfg_hedge.token_mfg_qualification_campaign import QualificationPlan
from mfg_hedge.token_mfg_qualification_state_machine import _unpack
from mfg_hedge.token_mfg_streaming import build_streaming_panel
from mfg_hedge.token_mfg_streaming_formal import streaming_iteration_episode_worker

DIAG_NAMESPACE = "reliability-aware-token-mfg:v1:k64-diagnostic"
DIAG_MACRO_SEED = 20260930
K = 64
WORKERS = 8

CONFIG_PATH = ROOT / "artifacts" / "engine-calibration-k64-20260912" / "calibration.json"


def _load_n() -> int:
    """N from the step-0 calibration: fill 80% of a 10h budget, cap 600, /4."""
    calib = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    per_episode = float(calib["episode_full_worker_s"]["mean_s"])
    budget_s = 0.8 * 10 * 3600
    n = int(budget_s * WORKERS / per_episode)
    n = min(600, n)
    n -= n % 4
    return max(4, n)


def _blob_name(payload: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"{digest}.json"


def _run_one(args):
    spec, plan, policy = args
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
    wall = time.perf_counter() - start
    out = worker_output["output"]
    record = {
        "episode_identity": [str(x) for x in out["episode_identity"]],
        "trace_fingerprint": out["trace_fingerprint"],
        "completed_calls": int(worker_output["completed_calls"]),
        "wall_s": wall,
        "output": out,
    }
    return record


def main() -> None:
    n = _load_n()
    run_id = f"k64-diagnostic-panel-20260912-n{n}"
    out_dir = ROOT / "artifacts" / run_id
    blob_dir = out_dir / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)

    plan = QualificationPlan.r2()
    panel = build_streaming_panel(
        plan, k=K, count=n, namespace=DIAG_NAMESPACE, macro_seed=DIAG_MACRO_SEED,
    )
    policy = FormalQualificationBackend.initial_seed("uniform")

    def key_of(identity) -> str:
        return hashlib.sha256(
            json.dumps([str(x) for x in identity]).encode("utf-8")
        ).hexdigest()[:16]

    todo = []
    for spec in panel.specs:
        key = key_of(spec.identity)
        if (blob_dir / f"{key}.json").exists():
            continue
        todo.append((spec, key))

    print(f"panel N={n}, todo={len(todo)}, workers={WORKERS}", flush=True)
    started = time.perf_counter()
    with Pool(WORKERS) as pool:
        for record in pool.imap_unordered(
            _run_one, [(spec, plan, policy) for spec, _ in todo]
        ):
            key = key_of(record["episode_identity"])
            (blob_dir / f"{key}.json").write_text(
                json.dumps(record, default=str), encoding="utf-8"
            )
            print(f"done {record['episode_identity'][-1]} in {record['wall_s']:.2f}s", flush=True)
    wall_total = time.perf_counter() - started

    rows = []
    for path in sorted(blob_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    target_states: Counter = Counter()
    finite_counts: Counter = Counter()
    finite_pairs: dict = {}
    token_state_counts: Counter = Counter()
    calls = []
    walls = []
    for rec in rows:
        out = rec["output"]
        target_states[out["target_state"]] += 1
        calls.append(rec["completed_calls"])
        walls.append(rec["wall_s"])
        for row in _unpack(out["compact_tokens"]):
            token_state_counts[row[0]] += 1
        for row in _unpack(out["finite_rows"]):
            cell = (out["target_state"], row.candidate_bucket.fingerprint)
            finite_counts[cell] += 1
            finite_pairs.setdefault(cell, []).append(
                (-float(row.pathwise_difference), float(row.baseline_cost))
            )

    cells = sorted(finite_counts)
    cell_counts = [finite_counts[c] for c in cells]
    hist = Counter(cell_counts)
    normalized_uppers = []
    for cell in cells:
        pairs = finite_pairs[cell]
        gains = [g for g, _ in pairs]
        baselines = [b for _, b in pairs]
        m = len(gains)
        mean = math.fsum(gains) / m
        se = (
            math.sqrt(math.fsum((g - mean) ** 2 for g in gains) / (m - 1) / m)
            if m > 1
            else 0.0
        )
        baseline = math.fsum(baselines) / m
        normalized_uppers.append((mean + 2.0 * se) / baseline)

    summary = {
        "run_id": run_id,
        "label": "diagnostic panel only; no qualification/Nash/MFG claim",
        "k": K,
        "namespace": DIAG_NAMESPACE,
        "macro_seed": DIAG_MACRO_SEED,
        "episodes_completed": len(rows),
        "wall_seconds_total": wall_total,
        "per_episode_wall_s": {
            "mean": statistics.fmean(walls),
            "median": statistics.median(walls),
            "worst": max(walls),
        },
        "per_episode_calls": {
            "mean": statistics.fmean(calls),
            "min": min(calls),
            "max": max(calls),
        },
        "distinct_target_states": len(target_states),
        "distinct_cells": len(cells),
        "cell_count_histogram": dict(sorted(hist.items())),
        "cells_below_floor_8": sum(1 for c in cell_counts if c < 8),
        "max_normalized_ucb": max(normalized_uppers) if normalized_uppers else None,
        "panel_fingerprint": panel.fingerprint,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
