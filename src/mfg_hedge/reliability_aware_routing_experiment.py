"""Frozen three-round finite routing evaluation for ADR-0036.

The runner keeps episode generation, physical simulation, selection, and
cluster inference separate.  It is intentionally an internal experiment
module; importing it does not run a campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
from typing import Mapping, Sequence

from .reliability_aware_routing import (
    POLICY_NAMES,
    SCENARIO_NAMES,
    generate_routing_trace,
    simulate_routing,
)


FIXED_BASELINE_ARMS = (
    "uniform_rr", "jsq", "loew", "reliability_only", "lazarus_algorithm1",
)
ROUND1_WINDOWS = (25.0, 50.0, 100.0)
ROUND1_GAMMAS = ((2.0, 3.0), (4.0, 6.0), (8.0, 12.0))
ROUND1_EPISODES = 256
ROUND3_EPISODES = 1024
ROUND1_CALLS = ROUND1_EPISODES * 4 * (5 + 9)
ROUND3_CALLS = ROUND3_EPISODES * 4 * 6
BOOTSTRAP_REPLICATES = 4096
MAX_PREFLIGHT_CALLS = 24
MAX_SCHEDULER_CALLS = 60000
R1_NAMESPACE = "reliability-aware-routing:v1:development:r1"
R2_NAMESPACE = "reliability-aware-routing:v1:development:r2"
HOLDOUT_NAMESPACE = "reliability-aware-routing:v1:holdout:r1"
R1_SEED = 20260910
R2_SEED = 20260911
HOLDOUT_SEED = 20260912
CORRECTED_R1_NAMESPACE = "reliability-aware-routing:v1:corrected:development:r1"
CORRECTED_R2_NAMESPACE = "reliability-aware-routing:v1:corrected:development:r2"
CORRECTED_HOLDOUT_NAMESPACE = "reliability-aware-routing:v1:corrected:holdout:r1"
CORRECTED_R1_SEED = 20260913
CORRECTED_R2_SEED = 20260914
CORRECTED_HOLDOUT_SEED = 20260915
CORRECTED_ARTIFACT_ROOT = "artifacts/reliability-aware-routing-corrected-evaluation-20260910"


@dataclass(frozen=True)
class CandidateConfig:
    history_window: float
    gamma_regular: float
    gamma_urgent: float

    @property
    def arm_name(self) -> str:
        return f"risk_aware_jsq_w{int(self.history_window)}_g{int(self.gamma_regular)}_{int(self.gamma_urgent)}"


@dataclass(frozen=True)
class RoundTask:
    scenario: str
    episode: int
    arm: str
    candidate_config: CandidateConfig | None
    is_candidate: bool


@dataclass(frozen=True)
class DevelopmentSelection:
    status: str
    config: CandidateConfig | None
    reason: str
    scores: tuple[tuple[str, float], ...]


def candidate_grid() -> tuple[CandidateConfig, ...]:
    return tuple(
        CandidateConfig(window, regular, urgent)
        for window in ROUND1_WINDOWS
        for regular, urgent in ROUND1_GAMMAS
    )


def build_round1_plan() -> tuple[RoundTask, ...]:
    tasks: list[RoundTask] = []
    candidates = candidate_grid()
    for scenario in SCENARIO_NAMES:
        for episode in range(ROUND1_EPISODES):
            for arm in FIXED_BASELINE_ARMS:
                tasks.append(RoundTask(scenario, episode, arm, None, False))
            for config in candidates:
                tasks.append(RoundTask(scenario, episode, config.arm_name, config, True))
    return tuple(tasks)


def _group_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int, str], Mapping[str, object]]:
    grouped: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row["scenario"]), int(row["episode"]), str(row["arm"]))
        if key in grouped:
            raise ValueError(f"duplicate panel row {key}")
        grouped[key] = row
    return grouped


def validate_complete_panel(
    rows: Sequence[Mapping[str, object]], required_arms: Sequence[str],
    episode_count: int, scenario_count: int,
) -> None:
    if any(not bool(row.get("complete", False)) for row in rows):
        raise ValueError("incomplete physical panel")
    grouped = _group_rows(rows)
    scenarios = tuple(sorted({key[0] for key in grouped}))
    episodes = tuple(sorted({key[1] for key in grouped}))
    if len(scenarios) != scenario_count or len(episodes) != episode_count:
        raise ValueError("panel scenario or episode shape is incomplete")
    expected = {
        (scenario, episode, arm)
        for scenario in scenarios for episode in episodes for arm in required_arms
    }
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise ValueError(f"panel identity mismatch missing={missing[:3]} extra={extra[:3]}")


def _mean_metric(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    values = [float(row["metrics"][metric]) for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"invalid metric {metric}")
    return math.fsum(values) / len(values)


def select_development_candidate(
    rows: Sequence[Mapping[str, object]], configs: Sequence[CandidateConfig],
    *, baseline_loew_arm: str = "loew",
) -> DevelopmentSelection:
    config_by_arm = {config.arm_name: config for config in configs}
    scenarios = tuple(sorted({str(row["scenario"]) for row in rows}))
    if set(scenarios) != set(SCENARIO_NAMES):
        raise ValueError("development selection requires S0-S3")
    baseline_rows = [row for row in rows if row["arm"] == baseline_loew_arm]
    if not baseline_rows:
        raise ValueError("LOEW baseline is required for development guardrail")
    loew_by_scenario = {
        scenario: [row for row in baseline_rows if row["scenario"] == scenario]
        for scenario in SCENARIO_NAMES
    }
    candidates: list[tuple[CandidateConfig, list[Mapping[str, object]]]] = []
    for config in configs:
        candidate_rows = [row for row in rows if row["arm"] == config.arm_name]
        if len(candidate_rows) != len(baseline_rows):
            raise ValueError(f"candidate panel incomplete: {config.arm_name}")
        guard_ok = True
        for scenario in ("S0",):
            candidate_scenario = [row for row in candidate_rows if row["scenario"] == scenario]
            loew_scenario = loew_by_scenario[scenario]
            for metric in ("overall_mean", "overall_cvar95"):
                if _mean_metric(candidate_scenario, metric) > 1.02 * _mean_metric(loew_scenario, metric):
                    guard_ok = False
        if guard_ok:
            candidates.append((config, candidate_rows))
    if not candidates:
        return DevelopmentSelection("development_no_candidate", None, "no candidate passed S0 guardrail", ())

    score_rows: list[tuple[CandidateConfig, float, float, float]] = []
    for config, candidate_rows in candidates:
        s23 = [row for row in candidate_rows if row["scenario"] in {"S2", "S3"}]
        score_rows.append((
            config,
            _mean_metric(s23, "overall_cvar95"),
            _mean_metric(s23, "urgent_deadline_miss_rate"),
            _mean_metric(s23, "lost_work"),
        ))
    best_cvar = min(row[1] for row in score_rows)
    cvar_tied = [row for row in score_rows if (row[1] - best_cvar) / max(abs(best_cvar), 1e-12) < 0.005]
    selected = min(cvar_tied, key=lambda row: (row[2], row[3], row[0].history_window,
                                                 row[0].gamma_regular, row[0].gamma_urgent))
    return DevelopmentSelection(
        "development_candidate", selected[0], "passed S0 guardrail and frozen tie-break",
        tuple((row[0].arm_name, row[1]) for row in sorted(score_rows, key=lambda row: row[0].arm_name)),
    )


def _paired_values(
    rows: Sequence[Mapping[str, object]], candidate_arm: str, comparison_arm: str,
    metric: str,
) -> dict[str, list[float]]:
    def metric_value(row: Mapping[str, object]) -> float:
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            return float(metrics[metric])
        # Metric-only rows are used by the small bootstrap helper and by no
        # scheduler path; production episode rows always use ``metrics``.
        return float(row[metric])

    grouped = _group_rows(rows)
    result: dict[str, list[float]] = {scenario: [] for scenario in SCENARIO_NAMES}
    keys = {(scenario, episode) for scenario, episode, arm in grouped if arm == candidate_arm}
    for scenario, episode in sorted(keys):
        candidate = grouped[(scenario, episode, candidate_arm)]
        comparison = grouped[(scenario, episode, comparison_arm)]
        if candidate.get("complete", True) is False or comparison.get("complete", True) is False:
            raise ValueError("incomplete paired row")
        result[scenario].append(
            metric_value(candidate) - metric_value(comparison)
        )
    if any(not values for values in result.values()):
        raise ValueError("missing scenario cluster")
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot quantile empty values")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def compute_paired_max_t(
    rows: Sequence[Mapping[str, object]], *, candidate_arm: str,
    comparison_arms: Sequence[str], metric: str = "metric",
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES, seed: int = HOLDOUT_SEED,
) -> dict[str, object]:
    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive true int")
    contrasts = {
        arm: _paired_values(rows, candidate_arm, arm, metric)
        for arm in comparison_arms
    }
    points = {
        arm: math.fsum(math.fsum(values) / len(values) for values in by_scenario.values()) / len(by_scenario)
        for arm, by_scenario in contrasts.items()
    }
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = {arm: [] for arm in comparison_arms}
    for _ in range(bootstrap_replicates):
        for arm, by_scenario in contrasts.items():
            scenario_means = []
            for values in by_scenario.values():
                sampled = [values[rng.randrange(len(values))] for _ in values]
                scenario_means.append(math.fsum(sampled) / len(sampled))
            bootstrap[arm].append(math.fsum(scenario_means) / len(scenario_means))
    ses = {
        arm: statistics.stdev(values) if len(values) > 1 else 0.0
        for arm, values in bootstrap.items()
    }
    max_t_values = []
    for index in range(bootstrap_replicates):
        max_t_values.append(max(
            (abs(bootstrap[arm][index] - points[arm]) / ses[arm]) if ses[arm] > 0.0 else 0.0
            for arm in comparison_arms
        ))
    critical = _quantile(max_t_values, 0.95)
    result = {}
    for arm in comparison_arms:
        half_width = critical * ses[arm]
        result[arm] = {
            "point": points[arm],
            "se": ses[arm],
            "simultaneous_95_ci": (points[arm] - half_width, points[arm] + half_width),
            "bootstrap_max_t_critical": critical,
        }
    return {
        "metric": metric,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "method": "paired_cluster_bootstrap_max_t",
        "contrasts": result,
    }


def _rss_bytes() -> int | None:
    try:
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                        ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
                        ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
                        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t), ("quota_non_paged_pool_usage", ctypes.c_size_t),
                        ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return int(counters.peak_working_set_size) if ok else None
    except Exception:
        return None


def run_timing_preflight() -> dict[str, object]:
    calls = 0
    start = time.perf_counter()
    trace = generate_routing_trace("S0", 0, namespace="reliability-aware-routing:v1:preflight", macro_seed=20260910)
    for policy in FIXED_BASELINE_ARMS:
        simulate_routing(trace, policy)
        calls += 1
    config = candidate_grid()[0]
    for _ in range(19):
        simulate_routing(trace, "risk_aware_jsq", history_window=config.history_window,
                         gamma_regular=config.gamma_regular, gamma_urgent=config.gamma_urgent)
        calls += 1
    elapsed = time.perf_counter() - start
    calls_per_second = calls / elapsed if elapsed > 0.0 else 0.0
    return {
        "calls": calls,
        "elapsed_seconds": elapsed,
        "calls_per_second": calls_per_second,
        "projected_53248_seconds": 53248.0 / calls_per_second if calls_per_second else math.inf,
        "projected_53248_hours": 53248.0 / calls_per_second / 3600.0 if calls_per_second else math.inf,
        "peak_rss_bytes": _rss_bytes(),
        "namespace": "reliability-aware-routing:v1:preflight",
        "formal_identity_consumed": False,
    }


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "src" / "mfg_hedge" / "reliability_aware_routing.py",
        root / "src" / "mfg_hedge" / "reliability_aware_routing_experiment.py",
        root / "docs" / "adr" / "0035-reliability-aware-multi-replica-routing.md",
        root / "docs" / "adr" / "0036-three-round-reliability-aware-routing-evaluation.md",
        root / ".scratch" / "reliability-aware-multi-replica-routing" / "spec.md",
    )
    payload = [(str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _round_protocol(
    round_name: str, *,
    r1_namespace: str = R1_NAMESPACE,
    r2_namespace: str = R2_NAMESPACE,
    holdout_namespace: str = HOLDOUT_NAMESPACE,
    r1_macro_seed: int = R1_SEED,
    r2_macro_seed: int = R2_SEED,
    holdout_macro_seed: int = HOLDOUT_SEED,
) -> dict[str, object]:
    return {
        "round": round_name,
        "r1_namespace": r1_namespace,
        "r2_namespace": r2_namespace,
        "holdout_namespace": holdout_namespace,
        "r1_macro_seed": r1_macro_seed,
        "r2_macro_seed": r2_macro_seed,
        "holdout_macro_seed": holdout_macro_seed,
        "r1_episodes": ROUND1_EPISODES,
        "holdout_episodes": ROUND3_EPISODES,
        "r1_calls": ROUND1_CALLS,
        "holdout_calls": ROUND3_CALLS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "claim_boundary": "finite_routing_evidence_only_no_mfg_or_equilibrium_claim",
    }


def _write_round_artifact(
    root: Path, run_name: str, summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]], paired: Mapping[str, object],
    protocol: Mapping[str, object], analysis: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / run_name
    if final.exists():
        raise FileExistsError(f"artifact already exists: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_name}-", dir=str(root)))
    try:
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        (staging / "paired_statistics.json").write_text(json.dumps(paired, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        (staging / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "run_name": run_name, "source_fingerprint": _source_fingerprint(),
            "transaction": "committed_after_complete_panel", "row_count": len(rows),
        }, indent=2, sort_keys=True), encoding="utf-8")
        with (staging / "episode_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (staging / "analysis.md").write_text(analysis, encoding="utf-8")
        try:
            staging.replace(final)
        except PermissionError:
            shutil.move(str(staging), str(final))
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _simulate_arm(trace, arm: str, config: CandidateConfig | None):
    if config is None:
        return simulate_routing(trace, arm)
    return simulate_routing(trace, "risk_aware_jsq", history_window=config.history_window,
                            gamma_regular=config.gamma_regular, gamma_urgent=config.gamma_urgent)


def _row(scenario: str, episode: int, arm: str, result, config: CandidateConfig | None) -> dict[str, object]:
    return {
        "scenario": scenario, "episode": episode, "arm": arm,
        "candidate_config": None if config is None else {
            "history_window": config.history_window,
            "gamma_regular": config.gamma_regular,
            "gamma_urgent": config.gamma_urgent,
        },
        "trace_fingerprint": result.trace_fingerprint,
        "token_fingerprint": result.token_fingerprint,
        "metrics": dict(result.metrics),
        "invariants": dict(result.invariants),
        "replica_audit": {str(key): dict(value) for key, value in result.replica_audit.items()},
        "complete": result.complete_drain,
    }


def run_round1(
    artifact_root: str | Path, *, namespace: str = R1_NAMESPACE,
    macro_seed: int = R1_SEED,
    run_name: str = "reliability-aware-routing-development-r1-20260910",
    protocol: Mapping[str, object] | None = None,
) -> tuple[DevelopmentSelection, dict[str, object], Path]:
    root = Path(artifact_root)
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    calls = 0
    configs = candidate_grid()
    for scenario in SCENARIO_NAMES:
        for episode in range(ROUND1_EPISODES):
            trace = generate_routing_trace(scenario, episode, namespace=namespace, macro_seed=macro_seed)
            for arm in FIXED_BASELINE_ARMS:
                result = _simulate_arm(trace, arm, None)
                rows.append(_row(scenario, episode, arm, result, None))
                calls += 1
            for config in configs:
                result = _simulate_arm(trace, config.arm_name, config)
                rows.append(_row(scenario, episode, config.arm_name, result, config))
                calls += 1
    validate_complete_panel(rows, FIXED_BASELINE_ARMS + tuple(config.arm_name for config in configs), ROUND1_EPISODES, 4)
    selection = select_development_candidate(rows, configs)
    baseline_means = {
        arm: {
            scenario: _mean_metric([row for row in rows if row["arm"] == arm and row["scenario"] == scenario], "overall_cvar95")
            for scenario in SCENARIO_NAMES
        }
        for arm in FIXED_BASELINE_ARMS
    }
    summary = {
        "status": selection.status, "calls": calls, "failures": 0,
        "elapsed_seconds": time.perf_counter() - start, "peak_rss_bytes": _rss_bytes(),
        "selected_config": None if selection.config is None else selection.config.__dict__,
        "selection_reason": selection.reason, "scores": selection.scores,
        "baseline_cvar95_by_scenario": baseline_means,
        "claim_boundary": "development_candidate_or_no_candidate_only",
    }
    paired = {arm: {
        "overall_cvar95": {
            comparison: _mean_metric([row for row in rows if row["arm"] == arm], "overall_cvar95")
            - _mean_metric([row for row in rows if row["arm"] == comparison], "overall_cvar95")
        } for comparison in FIXED_BASELINE_ARMS
    } for arm in (config.arm_name for config in configs)}
    analysis = "# Round 1 analysis\n\n" + json.dumps({"status": selection.status, "selection": selection.reason, "scores": selection.scores}, indent=2, sort_keys=True)
    path = _write_round_artifact(root, run_name, summary, rows, paired,
                                 protocol if protocol is not None else _round_protocol("round1"), analysis)
    return selection, summary, path


def run_round2(
    artifact_root: str | Path, selection: DevelopmentSelection, *,
    namespace: str = R2_NAMESPACE,
    macro_seed: int = R2_SEED,
    run_name: str = "reliability-aware-routing-development-r2-20260911",
    protocol: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], Path]:
    if selection.config is None:
        raise ValueError("Round 2 robustness requires a frozen development config")
    cells = (
        ("s2_early_load_low", "S2", 0.9 * 5.6, 10.0, 0.03),
        ("s2_late_load_high", "S2", 1.1 * 5.6, 30.0, 0.03),
        ("s3_shock_low", "S3", 5.6, 20.0, 0.024),
        ("s3_shock_high", "S3", 5.6, 20.0, 0.036),
    )
    arms = FIXED_BASELINE_ARMS + (selection.config.arm_name,)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    calls = 0
    for cell, scenario, arrival_rate, switch, domain_hazard in cells:
        for episode in range(ROUND1_EPISODES):
            trace = generate_routing_trace(
                scenario, episode, namespace=namespace + ":" + cell,
                macro_seed=macro_seed, arrival_rate=arrival_rate,
                s2_switch_time=switch, domain_hazard=domain_hazard,
            )
            for arm in FIXED_BASELINE_ARMS:
                rows.append(_row(cell, episode, arm, _simulate_arm(trace, arm, None), None))
                calls += 1
            rows.append(_row(cell, episode, selection.config.arm_name,
                             _simulate_arm(trace, selection.config.arm_name, selection.config), selection.config))
            calls += 1
    # The robustness panel uses four explicit cells, each with a complete six-arm panel.
    grouped = _group_rows(rows)
    if len(grouped) != len(cells) * ROUND1_EPISODES * len(arms):
        raise ValueError("Round 2 panel is incomplete")
    candidate_worse = False
    for cell, _, _, _, _ in cells:
        candidate = _mean_metric([row for row in rows if row["scenario"] == cell and row["arm"] == selection.config.arm_name], "overall_cvar95")
        loew = _mean_metric([row for row in rows if row["scenario"] == cell and row["arm"] == "loew"], "overall_cvar95")
        candidate_worse = candidate_worse or candidate > 1.02 * loew
    status = "candidate_not_robust" if candidate_worse else "candidate_qualified_for_holdout"
    summary = {"status": status, "calls": calls, "failures": 0,
               "elapsed_seconds": time.perf_counter() - start, "peak_rss_bytes": _rss_bytes(),
               "frozen_candidate": selection.config.__dict__, "cells": [cell for cell, *_ in cells]}
    paired = {"candidate_vs_loew_cvar95": {
        cell: _mean_metric([row for row in rows if row["scenario"] == cell and row["arm"] == selection.config.arm_name], "overall_cvar95")
        - _mean_metric([row for row in rows if row["scenario"] == cell and row["arm"] == "loew"], "overall_cvar95")
        for cell, *_ in cells
    }}
    analysis = "# Round 2 analysis\n\n" + json.dumps(summary, indent=2, sort_keys=True)
    path = _write_round_artifact(Path(artifact_root), run_name, summary, rows, paired,
                                 protocol if protocol is not None else _round_protocol("round2"), analysis)
    return summary, path


def run_round3(
    artifact_root: str | Path, selection: DevelopmentSelection, *,
    namespace: str = HOLDOUT_NAMESPACE,
    macro_seed: int = HOLDOUT_SEED,
    run_name: str = "reliability-aware-routing-holdout-r1-20260912",
    protocol: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], Path]:
    config = selection.config
    candidate_arm = config.arm_name if config is not None else "frozen_best_explainable"
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    calls = 0
    for scenario in SCENARIO_NAMES:
        for episode in range(ROUND3_EPISODES):
            trace = generate_routing_trace(scenario, episode, namespace=namespace, macro_seed=macro_seed)
            for arm in FIXED_BASELINE_ARMS:
                rows.append(_row(scenario, episode, arm, _simulate_arm(trace, arm, None), None))
                calls += 1
            if config is None:
                # A no-candidate confirmation still runs a frozen, explicit fallback.
                fallback = min(candidate_grid(), key=lambda row: (row.history_window, row.gamma_regular, row.gamma_urgent))
                result = _simulate_arm(trace, fallback.arm_name, fallback)
                rows.append(_row(scenario, episode, candidate_arm, result, fallback))
            else:
                rows.append(_row(scenario, episode, candidate_arm, _simulate_arm(trace, candidate_arm, config), config))
            calls += 1
    validate_complete_panel(rows, FIXED_BASELINE_ARMS + (candidate_arm,), ROUND3_EPISODES, 4)
    paired: dict[str, object] = {}
    for metric in ("overall_cvar95", "urgent_deadline_miss_rate", "overall_mean", "overall_p99",
                   "replay_rate", "lost_work", "drain_duration"):
        metric_rows = [
            {"scenario": row["scenario"], "episode": row["episode"], "arm": row["arm"],
             "metric": row["metrics"][metric], "complete": row["complete"]}
            for row in rows
        ]
        paired[metric] = compute_paired_max_t(
            metric_rows, candidate_arm=candidate_arm,
            comparison_arms=("jsq", "loew", "reliability_only", "lazarus_algorithm1"),
            bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=macro_seed,
        )
    primary = paired["overall_cvar95"]["contrasts"]
    loew_ci = primary["loew"]["simultaneous_95_ci"]
    jsq_point = primary["jsq"]["point"]
    s0_rows = [row for row in rows if row["scenario"] == "S0"]
    candidate_s0_mean = _mean_metric([row for row in s0_rows if row["arm"] == candidate_arm], "overall_mean")
    loew_s0_mean = _mean_metric([row for row in s0_rows if row["arm"] == "loew"], "overall_mean")
    s0_cvar = _mean_metric([row for row in s0_rows if row["arm"] == candidate_arm], "overall_cvar95")
    loew_s0_cvar = _mean_metric([row for row in s0_rows if row["arm"] == "loew"], "overall_cvar95")
    candidate_replay = _mean_metric([row for row in rows if row["arm"] == candidate_arm], "replay_rate")
    loew_replay = _mean_metric([row for row in rows if row["arm"] == "loew"], "replay_rate")
    candidate_lost = _mean_metric([row for row in rows if row["arm"] == candidate_arm], "lost_work")
    loew_lost = _mean_metric([row for row in rows if row["arm"] == "loew"], "lost_work")
    success = (loew_ci[1] < 0.0 and jsq_point < 0.0 and
               candidate_s0_mean <= 1.02 * loew_s0_mean and candidate_s0_cvar <= 1.02 * loew_s0_cvar and
               (candidate_replay <= loew_replay or candidate_lost <= loew_lost))
    summary = {
        "status": "holdout_pass" if success else "holdout_not_improving",
        "calls": calls, "failures": 0, "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_bytes": _rss_bytes(), "candidate_arm": candidate_arm,
        "candidate_config": None if config is None else config.__dict__,
        "primary_loew_ci": loew_ci, "primary_jsq_point": jsq_point,
        "s0_mean_candidate": candidate_s0_mean, "s0_mean_loew": loew_s0_mean,
        "s0_cvar_candidate": s0_cvar, "s0_cvar_loew": loew_s0_cvar,
        "candidate_replay_rate": candidate_replay, "loew_replay_rate": loew_replay,
        "candidate_lost_work": candidate_lost, "loew_lost_work": loew_lost,
        "claim_boundary": "finite_holdout_routing_result_no_mfg_or_equilibrium_claim",
    }
    analysis = "# Round 3 analysis\n\n" + json.dumps(summary, indent=2, sort_keys=True)
    path = _write_round_artifact(Path(artifact_root), run_name, summary, rows, paired,
                                 protocol if protocol is not None else _round_protocol("round3"), analysis)
    return summary, path


def run_corrected_three_rounds(artifact_root: str | Path = CORRECTED_ARTIFACT_ROOT) -> dict[str, object]:
    """Run the corrected three-round evaluation under fresh identities.

    This is an explicit internal entry point for the exposure-horizon
    correction.  It preserves the historical runner defaults and never
    reuses the old development or holdout identities.
    """
    protocol_args = {
        "r1_namespace": CORRECTED_R1_NAMESPACE,
        "r2_namespace": CORRECTED_R2_NAMESPACE,
        "holdout_namespace": CORRECTED_HOLDOUT_NAMESPACE,
        "r1_macro_seed": CORRECTED_R1_SEED,
        "r2_macro_seed": CORRECTED_R2_SEED,
        "holdout_macro_seed": CORRECTED_HOLDOUT_SEED,
    }
    selection, round1_summary, round1_path = run_round1(
        artifact_root, namespace=CORRECTED_R1_NAMESPACE,
        macro_seed=CORRECTED_R1_SEED,
        run_name="reliability-aware-routing-corrected-development-r1-20260913",
        protocol=_round_protocol("round1", **protocol_args),
    )
    round2_summary, round2_path = run_round2(
        artifact_root, selection, namespace=CORRECTED_R2_NAMESPACE,
        macro_seed=CORRECTED_R2_SEED,
        run_name="reliability-aware-routing-corrected-development-r2-20260914",
        protocol=_round_protocol("round2", **protocol_args),
    )
    round3_summary, round3_path = run_round3(
        artifact_root, selection, namespace=CORRECTED_HOLDOUT_NAMESPACE,
        macro_seed=CORRECTED_HOLDOUT_SEED,
        run_name="reliability-aware-routing-corrected-holdout-r1-20260915",
        protocol=_round_protocol("round3", **protocol_args),
    )
    return {
        "selection": selection,
        "round1": {"summary": round1_summary, "artifact_dir": str(round1_path)},
        "round2": {"summary": round2_summary, "artifact_dir": str(round2_path)},
        "round3": {"summary": round3_summary, "artifact_dir": str(round3_path)},
    }


__all__ = [
    "BOOTSTRAP_REPLICATES", "MAX_SCHEDULER_CALLS", "ROUND1_CALLS", "ROUND3_CALLS",
    "CandidateConfig", "DevelopmentSelection", "RoundTask", "build_round1_plan",
    "candidate_grid", "compute_paired_max_t", "run_round1", "run_round2", "run_round3",
    "run_timing_preflight", "select_development_candidate", "validate_complete_panel",
]
