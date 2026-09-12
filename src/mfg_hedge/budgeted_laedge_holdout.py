"""Confirmed-mapping Budgeted LÆDGE holdout execution.

This module is deliberately separate from the development calibration runner.
It consumes one confirmed v2 mapping, never recalibrates it, and runs the
fresh v2 holdout panel with paired episode-cluster bootstrap statistics.  The
result is a finite-system descriptive Pareto panel only.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Callable, Mapping, Sequence

from .artifacts import _validate_run_id
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeTrace,
    generate_episode_trace,
)
from .budgeted_laedge import (
    BudgetedLaedgeResult,
    simulate_budgeted_laedge_episode,
)
from .budgeted_laedge_campaign import (
    BUDGETED_LAEDGE_ARM_KEYS,
    BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
    BUDGETED_LAEDGE_HEDGE_DELAY,
    BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
    BUDGETED_LAEDGE_HOLDOUT_EPISODES,
    BUDGETED_LAEDGE_HOLDOUT_LABEL,
    BUDGETED_LAEDGE_STORM_BIN_WIDTH,
    BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED,
    BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE,
    BudgetedLaedgeCampaignError,
    _V2_PROTOCOL,
    _digest,
    _outcome,
    _protocol_payload,
    _simulation_for_arm,
    _trace_digest,
)
from .config import ExperimentConfig, load_config
from .metrics import percentile
from .token_t3a_execution import validate_frozen_config


BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS = BUDGETED_LAEDGE_ARM_KEYS
BUDGETED_LAEDGE_HOLDOUT_NAMESPACE = BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE
BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v2:holdout:paired-bootstrap"
)
BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_SEED = 20260914
BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES = 4096
BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v2:holdout:preflight"
)
BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_MACRO_SEED = 20260915
BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_EPISODES = 2
BUDGETED_LAEDGE_HOLDOUT_PROTOCOL_LABEL = "finite_budgeted_laedge_holdout_pareto_v2"
BUDGETED_LAEDGE_HOLDOUT_MAPPING_FINGERPRINT = (
    "c716b57c8167d16aa03af7cc6992407ddd04b7e0b17638d487ebcb963f92976d"
)
_TARGET_RATE_KEYS = ("0.0", "0.01", "0.03", "0.05", "0.08", "0.12", "0.18")
_CONFIRMED_RATES = {
    "0.0": 0.0,
    "0.01": 0.029794334898597338,
    "0.03": 0.06752794601883591,
    "0.05": 0.09304612180547656,
    "0.08": 0.17872563347189496,
    "0.12": 0.2770264334252587,
    "0.18": 4.0,
}
_METRICS = (
    "overall_mean",
    "overall_p95",
    "overall_p99",
    "df_mean",
    "df_p95",
    "df_p99",
    "df_cvar95",
    "df_miss_rate",
    "hr_p99",
    "replay_rate",
    "total_work",
    "incremental_hedge_work",
    "wasted_work",
    "hedge_launches",
    "hedge_winner_rate",
    "hedge_suppressed",
    "storm_peak",
    "drain_duration",
)
_BUDGET_METRICS = (
    "planned_budget",
    "committed_work",
    "launch_cohort_realized_work",
    "execution_time_realized_work",
    "overshoot",
    "unused_budget",
)


class BudgetedLaedgeHoldoutError(RuntimeError):
    """A confirmed-mapping holdout or artifact contract violation."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetedLaedgeHoldoutError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise BudgetedLaedgeHoldoutError(f"{name} must be finite")
    return result


def confirmed_v2_rates() -> dict[str, float]:
    """Return a fresh copy of the immutable, confirmed target-to-rate map."""

    return dict(_CONFIRMED_RATES)


def _physics_fingerprint(protocol: Mapping[str, object]) -> str:
    payload = {
        "protocol_schema_id": protocol.get("schema_id"),
        "episode_protocol": {
            "degraded_start": ATTRIBUTION_V1_PROTOCOL.timeline.degraded_start,
            "failed_start": ATTRIBUTION_V1_PROTOCOL.timeline.failed_start,
            "recovered_start": ATTRIBUTION_V1_PROTOCOL.timeline.recovered_start,
            "arrival_cutoff": ATTRIBUTION_V1_PROTOCOL.arrival_cutoff,
        },
        "degraded_slowdown": BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
        "hedge_delay": BUDGETED_LAEDGE_HEDGE_DELAY,
        "storm_bin_width": BUDGETED_LAEDGE_STORM_BIN_WIDTH,
        "budget_window_width": 25.0,
        "expected_work_per_hedge": 1.0,
        "cancellation": "conservative_primary_result_preemptive_oracle_only",
        "complete_drain": True,
        "crn_attempt_keys": [0, 1, 2],
    }
    return _digest(payload)


def _verify_mapping(development_result: Mapping[str, object]) -> Mapping[str, object]:
    mapping = development_result.get("mapping")
    if not isinstance(mapping, Mapping):
        raise BudgetedLaedgeHoldoutError("confirmed development mapping is missing")
    mapping_fingerprint = development_result.get("mapping_fingerprint")
    mapping_payload = dict(mapping)
    embedded_fingerprint = mapping_payload.pop("mapping_fingerprint", None)
    if embedded_fingerprint != mapping_fingerprint or mapping_fingerprint != _digest(mapping_payload):
        raise BudgetedLaedgeHoldoutError("development mapping fingerprint is stale")
    if mapping_fingerprint != BUDGETED_LAEDGE_HOLDOUT_MAPPING_FINGERPRINT:
        raise BudgetedLaedgeHoldoutError("development mapping is not the confirmed v2 mapping")
    targets = mapping.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(_TARGET_RATE_KEYS):
        raise BudgetedLaedgeHoldoutError("confirmed mapping has the wrong target set")
    for key in _TARGET_RATE_KEYS:
        row = targets[key]
        if not isinstance(row, Mapping):
            raise BudgetedLaedgeHoldoutError("confirmed mapping target is malformed")
        rate = _finite(row.get("planned_budget_rate"), f"rate {key}")
        if rate != _CONFIRMED_RATES[key]:
            raise BudgetedLaedgeHoldoutError(f"confirmed rate for {key} was changed")
    if not bool(targets["0.18"].get("saturation_limited")):
        raise BudgetedLaedgeHoldoutError("18% saturation-limited label is missing")
    return mapping


def _verify_source_bundle(
    development_result: Mapping[str, object],
    project_root: str | Path | None,
) -> str:
    bundle = development_result.get("source_bundle")
    if not isinstance(bundle, Mapping):
        raise BudgetedLaedgeHoldoutError("confirmed source bundle is missing")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BudgetedLaedgeHoldoutError("confirmed source bundle entries are missing")
    if _digest(entries) != bundle.get("fingerprint"):
        raise BudgetedLaedgeHoldoutError("confirmed source bundle fingerprint is stale")
    if project_root is not None:
        root = Path(project_root).resolve()
        current = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise BudgetedLaedgeHoldoutError("confirmed source bundle entry is malformed")
            relative, expected_digest = entry
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise BudgetedLaedgeHoldoutError(f"source bundle file is unavailable: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            current.append([relative, digest])
            if digest != expected_digest:
                raise BudgetedLaedgeHoldoutError(f"source bundle changed: {relative}")
        if current != entries:
            raise BudgetedLaedgeHoldoutError("source bundle ordering changed")
    return str(bundle["fingerprint"])


def validate_confirmed_v2_development(
    development_result: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
) -> dict[str, str]:
    if not isinstance(development_result, Mapping):
        raise BudgetedLaedgeHoldoutError("development result must be a mapping")
    if development_result.get("status") != "confirmed":
        raise BudgetedLaedgeHoldoutError("holdout requires confirmed v2 development")
    protocol = development_result.get("protocol")
    expected_protocol = _protocol_payload(_V2_PROTOCOL)
    if protocol != expected_protocol:
        raise BudgetedLaedgeHoldoutError("development protocol does not match v2 holdout")
    if development_result.get("protocol_fingerprint") != _digest(protocol):
        raise BudgetedLaedgeHoldoutError("development protocol fingerprint is stale")
    _verify_mapping(development_result)
    source_fingerprint = _verify_source_bundle(development_result, project_root)
    confirmed_physics = _physics_fingerprint(protocol)
    current_physics = _physics_fingerprint(expected_protocol)
    if confirmed_physics != current_physics:
        raise BudgetedLaedgeHoldoutError("physics fingerprint does not match development")
    return {
        "mapping_fingerprint": str(development_result["mapping_fingerprint"]),
        "protocol_fingerprint": str(development_result["protocol_fingerprint"]),
        "source_bundle_fingerprint": source_fingerprint,
        "physics_fingerprint": current_physics,
    }


def _bootstrap_indices(sample_count: int) -> array:
    if type(sample_count) is not int or sample_count <= 0:
        raise BudgetedLaedgeHoldoutError("bootstrap sample count must be positive")
    seed_material = (
        f"{BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_NAMESPACE}:"
        f"{BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_SEED}:{sample_count}:"
        f"{BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES}"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    indices = array("I")
    for _ in range(BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES * sample_count):
        indices.append(rng.randrange(sample_count))
    return indices


def _bootstrap_fingerprint(indices: array) -> str:
    return hashlib.sha256(indices.tobytes()).hexdigest()


def paired_cluster_statistics(
    left: Sequence[float],
    right: Sequence[float],
    *,
    metric_name: str,
    indices: array | None = None,
    relative_baseline: Sequence[float] | None = None,
) -> dict[str, object]:
    if len(left) != len(right) or not left:
        raise BudgetedLaedgeHoldoutError("paired samples must have equal positive length")
    left_values = tuple(_finite(value, f"{metric_name}.left") for value in left)
    right_values = tuple(_finite(value, f"{metric_name}.right") for value in right)
    sample_count = len(left_values)
    indices = _bootstrap_indices(sample_count) if indices is None else indices
    if len(indices) != sample_count * BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES:
        raise BudgetedLaedgeHoldoutError("bootstrap index library has the wrong shape")
    differences = tuple(right_value - left_value for left_value, right_value in zip(left_values, right_values))
    point = math.fsum(differences) / sample_count
    paired_se = (
        math.sqrt(math.fsum((value - point) ** 2 for value in differences) / (sample_count - 1))
        / math.sqrt(sample_count)
        if sample_count > 1
        else 0.0
    )
    bootstrap_means = [
        math.fsum(differences[indices[start + offset]] for offset in range(sample_count))
        / sample_count
        for start in range(0, len(indices), sample_count)
    ]
    baseline_values = left_values if relative_baseline is None else tuple(
        _finite(value, f"{metric_name}.baseline") for value in relative_baseline
    )
    if len(baseline_values) != sample_count:
        raise BudgetedLaedgeHoldoutError("relative baseline has the wrong shape")
    baseline = math.fsum(baseline_values) / sample_count
    return {
        "metric": metric_name,
        "point_estimate": math.fsum(right_values) / sample_count,
        "absolute_difference": point,
        "relative_difference": point / abs(baseline) if baseline else None,
        "paired_se": paired_se,
        "ci95": [percentile(bootstrap_means, 2.5), percentile(bootstrap_means, 97.5)],
        "sample_count": sample_count,
        "cluster_unit": "episode_index",
        "bootstrap_replicates": BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES,
        "bootstrap_fingerprint": _bootstrap_fingerprint(indices),
    }


def _peak_rss_bytes() -> dict[str, object]:
    if os.name != "nt":
        return {"status": "unsupported", "peak_rss_bytes": None}
    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    ok = get_process_memory_info(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        wintypes.DWORD(counters.cb),
    )
    if not ok:
        return {"status": "unavailable", "peak_rss_bytes": None}
    return {"status": "ok", "peak_rss_bytes": int(counters.PeakWorkingSetSize)}


def _budget_payload(raw: BudgetedLaedgeResult | None) -> dict[str, object]:
    if raw is None:
        return {
            "planned_budget": None,
            "committed_work": None,
            "launch_cohort_realized_work": None,
            "execution_time_realized_work": None,
            "overshoot": None,
            "unused_budget": None,
            "windows": [],
        }
    windows = [
        {
            **asdict(window),
            "phase": window.phase.value,
        }
        for window in raw.windows
    ]
    return {
        "planned_budget": math.fsum(window.planned_budget for window in raw.windows),
        "committed_work": math.fsum(window.committed_work for window in raw.windows),
        "launch_cohort_realized_work": math.fsum(
            window.launch_cohort_realized_work for window in raw.windows
        ),
        "execution_time_realized_work": math.fsum(
            window.execution_time_realized_work for window in raw.windows
        ),
        "overshoot": math.fsum(window.overshoot for window in raw.windows),
        "unused_budget": math.fsum(window.unused_budget for window in raw.windows),
        "windows": windows,
    }


def _run_holdout_arm(
    episode: EpisodeTrace,
    arm_key: str,
    *,
    planned_budget_rate: float | None,
) -> dict[str, object]:
    before = _trace_digest(episode)
    raw_budget: BudgetedLaedgeResult | None = None
    if arm_key.startswith("budgeted-laedge:"):
        if planned_budget_rate is None:
            raise BudgetedLaedgeHoldoutError("Budgeted arm has no frozen rate")
        raw_budget = simulate_budgeted_laedge_episode(
            episode,
            planned_budget_rate=planned_budget_rate,
            degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            cancel_running_losers=False,
        )
        simulation = raw_budget.simulation
    else:
        simulation = _simulation_for_arm(
            episode,
            arm_key,
            planned_budget_rate=planned_budget_rate,
        )
    after = _trace_digest(episode)
    if before != after:
        raise BudgetedLaedgeHoldoutError(f"arm {arm_key} mutated its episode")
    placement_mode = (
        "fixed_dispatcher"
        if arm_key in {"fixed-dispatcher:no-hedge", "niin:conservative"}
        else "budgeted_idle_release"
    )
    return {
        "arm_key": arm_key,
        "trace_fingerprint": before,
        "outcome": _outcome(simulation, placement_mode=placement_mode),
        "budget": _budget_payload(raw_budget),
    }


def _frozen_rate_for_arm(arm_key: str) -> float | None:
    if not arm_key.startswith("budgeted-laedge:"):
        return None
    text = arm_key.split("=", 1)[1].rstrip("%")
    target_key = str(float(text) / 100.0)
    try:
        return _CONFIRMED_RATES[target_key]
    except KeyError as error:
        raise BudgetedLaedgeHoldoutError(
            f"no confirmed rate for Budgeted LÆDGE arm {arm_key!r}"
        ) from error


def _arm_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise BudgetedLaedgeHoldoutError("cannot summarize an empty arm")
    outcomes = [row["outcome"] for row in rows]
    summary: dict[str, object] = {"episode_count": len(rows)}
    for name in _METRICS:
        values = [outcome.get(name) for outcome in outcomes]
        if any(value is None for value in values):
            summary[name] = None
        else:
            summary[name] = math.fsum(float(value) for value in values) / len(values)
    for name in _BUDGET_METRICS:
        values = [row["budget"].get(name) for row in rows]
        summary[name] = (
            None
            if any(value is None for value in values)
            else math.fsum(float(value) for value in values) / len(values)
        )
    summary["planned_budget_rate"] = (
        None
        if all(row["budget"].get("planned_budget") is None for row in rows)
        else None
    )
    return summary


def _comparison(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    left_arm: str,
    right_arm: str,
    indices: array,
    *,
    relative_baseline: str | None = None,
) -> dict[str, object]:
    left = records[left_arm]
    right = records[right_arm]
    if len(left) != len(right) or not left:
        raise BudgetedLaedgeHoldoutError("comparison arms are incomplete")
    baseline_rows = records[relative_baseline] if relative_baseline else left
    metrics: dict[str, object] = {}
    for name in _METRICS:
        left_values = [row["outcome"].get(name) for row in left]
        right_values = [row["outcome"].get(name) for row in right]
        baseline_values = [row["outcome"].get(name) for row in baseline_rows]
        if any(value is None for value in (*left_values, *right_values, *baseline_values)):
            metrics[name] = {"metric": name, "status": "unavailable", "reason": "non_numeric_metric"}
            continue
        metrics[name] = paired_cluster_statistics(
            left_values,
            right_values,
            metric_name=name,
            indices=indices,
            relative_baseline=baseline_values,
        )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "orientation": "right_minus_left",
        "cluster_unit": "episode_index",
        "metrics": metrics,
    }


def _dominance_and_gain(
    summaries: Mapping[str, Mapping[str, object]],
    *,
    niin_arm: str,
) -> dict[str, object]:
    dimensions = ("df_cvar95", "df_miss_rate", "total_work", "storm_peak")
    points: dict[str, object] = {}
    niin = summaries[niin_arm]
    niin_work = float(niin["total_work"])
    niin_tail = float(niin["df_cvar95"])
    for arm, summary in summaries.items():
        values = {name: summary.get(name) for name in dimensions}
        dominated_by = []
        if all(value is not None for value in values.values()):
            for other, other_summary in summaries.items():
                if other == arm:
                    continue
                other_values = {name: other_summary.get(name) for name in dimensions}
                if all(other_values[name] <= values[name] for name in dimensions) and any(
                    other_values[name] < values[name] for name in dimensions
                ):
                    dominated_by.append(other)
        work = summary.get("total_work")
        tail = summary.get("df_cvar95")
        points[arm] = {
            "status": "dominated" if dominated_by else "pareto_candidate",
            "dominated_by": sorted(dominated_by),
            "tail_gain_per_extra_work": (
                (niin_tail - float(tail)) / (float(work) - niin_work)
                if tail is not None and work is not None and float(work) > niin_work
                else None
            ),
            "realized_delta_vs_niin": (
                float(work) / niin_work - 1.0 if work is not None else None
            ),
        }
    return points


def run_budgeted_laedge_holdout_preflight_v2(
    config: ExperimentConfig,
    development_result: Mapping[str, object],
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    episode_count: int = BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_EPISODES,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    verification = validate_confirmed_v2_development(
        development_result,
        project_root=project_root,
    )
    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= 8:
        raise BudgetedLaedgeHoldoutError("preflight episode count must be in 1..8")
    build = episode_builder or (
        lambda namespace, index: generate_episode_trace(
            config,
            namespace,
            BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )
    )
    start = time.perf_counter()
    calls = 0
    failures = []
    for index in range(episode_count):
        try:
            episode = build(BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE, index)
            before = _trace_digest(episode)
            for arm_key in BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS:
                rate = _frozen_rate_for_arm(arm_key)
                calls += 1
                _run_holdout_arm(episode, arm_key, planned_budget_rate=rate)
                if _trace_digest(episode) != before:
                    raise BudgetedLaedgeHoldoutError("preflight trace fingerprint changed")
        except Exception as error:
            failures.append({
                "episode_index": index,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
    elapsed = time.perf_counter() - start
    calls_per_second = calls / elapsed if elapsed > 0.0 else None
    memory = _peak_rss_bytes()
    return {
        "status": "ok" if not failures else "physical_failed",
        "namespace": BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE,
        "macro_seed": BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_MACRO_SEED,
        "episode_count": episode_count,
        "calls": calls,
        "calls_per_second": calls_per_second,
        "elapsed_seconds": elapsed,
        "estimated_formal_seconds": (
            BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET / calls_per_second
            if calls_per_second
            else None
        ),
        "memory": memory,
        "failures": failures,
        "verification": verification,
    }


def run_budgeted_laedge_holdout_v2(
    config: ExperimentConfig,
    development_result: Mapping[str, object],
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    episode_count: int = BUDGETED_LAEDGE_HOLDOUT_EPISODES,
    progress: Callable[[int, int], None] | None = None,
    project_root: str | Path | None = None,
    preflight_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    verification = validate_confirmed_v2_development(
        development_result,
        project_root=project_root,
    )
    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= BUDGETED_LAEDGE_HOLDOUT_EPISODES:
        raise BudgetedLaedgeHoldoutError("holdout episode count is outside the frozen limit")
    if preflight_result is not None and preflight_result.get("status") != "ok":
        raise BudgetedLaedgeHoldoutError("formal holdout requires a successful preflight")
    build = episode_builder or (
        lambda namespace, index: generate_episode_trace(
            config,
            namespace,
            BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )
    )
    target_rates = {
        f"budgeted-laedge:delta={int(float(key) * 100):d}%": rate
        for key, rate in _CONFIRMED_RATES.items()
    }
    records: dict[str, list[dict[str, object]]] = {arm: [] for arm in BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS}
    episode_rows = []
    failures = []
    calls = 0
    start = time.perf_counter()
    for index in range(episode_count):
        episode_row: dict[str, object] = {
            "episode_index": index,
            "namespace": BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE,
            "status": "completed",
            "arms": {},
        }
        try:
            episode = build(BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE, index)
            if not isinstance(episode, EpisodeTrace):
                raise BudgetedLaedgeHoldoutError("episode builder returned a non-EpisodeTrace")
            trace_fingerprint = _trace_digest(episode)
            episode_row["trace_fingerprint"] = trace_fingerprint
            for arm_key in BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS:
                calls += 1
                rate = target_rates.get(arm_key)
                try:
                    item = _run_holdout_arm(
                        episode,
                        arm_key,
                        planned_budget_rate=rate,
                    )
                    if item["trace_fingerprint"] != trace_fingerprint:
                        raise BudgetedLaedgeHoldoutError("arm trace fingerprint differs")
                    records[arm_key].append(item)
                    episode_row["arms"][arm_key] = {"status": "completed", **item}
                except Exception as error:
                    failures.append({
                        "episode_index": index,
                        "arm_key": arm_key,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    })
                    episode_row["status"] = "physical_failed"
                    episode_row["arms"][arm_key] = {
                        "status": "failed",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
        except Exception as error:
            failures.append({
                "episode_index": index,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
            episode_row["status"] = "physical_failed"
            for arm_key in BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS:
                calls += 1
                episode_row["arms"][arm_key] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        episode_rows.append(episode_row)
        if progress is not None:
            progress(calls, BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET)
    elapsed = time.perf_counter() - start
    if calls != episode_count * len(BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS):
        raise BudgetedLaedgeHoldoutError("holdout call accounting failed")
    complete = not failures and all(len(rows) == episode_count for rows in records.values())
    result: dict[str, object] = {
        "label": BUDGETED_LAEDGE_HOLDOUT_PROTOCOL_LABEL,
        "status": "completed" if complete else "physical_failed",
        "scheduler_calls": calls,
        "formal_scheduler_call_budget": BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
        "episode_count": episode_count,
        "elapsed_seconds": elapsed,
        "calls_per_second": calls / elapsed if elapsed > 0.0 else None,
        "verification": verification,
        "preflight": dict(preflight_result or {}),
        "failures": failures,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": episode_rows,
    }
    if not complete:
        result["paired_statistics"] = {"status": "unavailable", "reason": "physical_failed"}
        return result
    indices = _bootstrap_indices(episode_count)
    summaries = {arm: _arm_summary(rows) for arm, rows in records.items()}
    comparisons = {}
    fixed = "fixed-dispatcher:no-hedge"
    niin = "niin:conservative"
    zero = "budgeted-laedge:delta=0%"
    comparisons["delta0_vs_fixed_no_hedge"] = _comparison(records, fixed, zero, indices)
    for delta in ("1", "3", "5", "8", "12", "18"):
        arm = f"budgeted-laedge:delta={delta}%"
        comparisons[f"{arm}_vs_delta0"] = _comparison(records, zero, arm, indices)
        comparisons[f"{arm}_vs_niin"] = _comparison(records, niin, arm, indices)
    comparisons["rate4_vs_unconstrained_conservative"] = _comparison(
        records,
        "unconstrained-laedge:conservative",
        "budgeted-laedge:delta=18%",
        indices,
    )
    comparisons["unconstrained_preemptive_vs_conservative"] = _comparison(
        records,
        "unconstrained-laedge:conservative",
        "unconstrained-laedge:preemptive-oracle",
        indices,
    )
    result["arm_summaries"] = summaries
    result["mapping_labels"] = {
        f"budgeted-laedge:delta={delta}%": {
            "target_delta": float(delta) / 100.0,
            "planned_budget_rate": _CONFIRMED_RATES[str(float(delta) / 100.0)],
            "saturation_limited": delta == "18",
            "realized_delta_vs_niin": (
                summaries[f"budgeted-laedge:delta={delta}%"]["total_work"]
                / summaries[niin]["total_work"]
                - 1.0
            ),
        }
        for delta in ("0", "1", "3", "5", "8", "12", "18")
    }
    result["paired_statistics"] = {
        "status": "completed",
        "cluster_unit": "episode_index",
        "bootstrap_replicates": BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES,
        "bootstrap_fingerprint": _bootstrap_fingerprint(indices),
        "comparisons": comparisons,
    }
    result["pareto"] = _dominance_and_gain(summaries, niin_arm=niin)
    result["protocol"] = {
        **_protocol_payload(_V2_PROTOCOL),
        "holdout_label": BUDGETED_LAEDGE_HOLDOUT_PROTOCOL_LABEL,
        "bootstrap_namespace": BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_NAMESPACE,
        "bootstrap_seed": BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_SEED,
        "bootstrap_replicates": BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES,
        "preflight_namespace": BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE,
        "preflight_macro_seed": BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_MACRO_SEED,
    }
    result["protocol_fingerprint"] = _digest(result["protocol"])
    result["memory"] = _peak_rss_bytes()
    return result


def write_budgeted_laedge_holdout_artifact(
    artifacts_root: str | Path,
    run_id: str,
    files: Mapping[str, object],
) -> Path:
    _validate_run_id(run_id)
    if not files:
        raise ValueError("files must not be empty")
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = (root / run_id).resolve()
    if final.parent != root or final.exists():
        raise FileExistsError(f"artifact run directory is unavailable: {final}")
    staging = root / f".{run_id}.staging"
    if staging.exists():
        raise FileExistsError(f"artifact staging directory exists: {staging}")
    staging.mkdir()
    try:
        for name, payload in files.items():
            if name == "episode_rows.jsonl":
                text = "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    for row in payload
                )
            elif name.endswith(".json") and "/" not in name and "\\" not in name:
                text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
            else:
                raise ValueError(f"invalid holdout artifact file name: {name!r}")
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def execute_budgeted_laedge_holdout_v2(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    development_artifact: str | Path,
    *,
    preflight_episode_count: int = BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_EPISODES,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    development_root = Path(development_artifact).resolve()
    summary_path = development_root / "summary.json"
    manifest_path = development_root / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise BudgetedLaedgeHoldoutError("confirmed development artifact is incomplete")
    development_result = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "confirmed" or manifest.get("mapping_fingerprint") not in (None, development_result.get("mapping_fingerprint")):
        raise BudgetedLaedgeHoldoutError("confirmed development manifest is stale")
    verification = validate_confirmed_v2_development(
        development_result,
        project_root=project,
    )
    config = load_config(project / "configs" / "v1_minimal.json")
    preflight = run_budgeted_laedge_holdout_preflight_v2(
        config,
        development_result,
        episode_count=preflight_episode_count,
        project_root=project,
    )
    if preflight["status"] != "ok":
        raise BudgetedLaedgeHoldoutError("timing preflight failed; formal holdout not started")
    result = run_budgeted_laedge_holdout_v2(
        config,
        development_result,
        project_root=project,
        preflight_result=preflight,
    )
    result["verification"] = verification
    result["source_bundle"] = development_result["source_bundle"]
    result["development_artifact"] = str(development_root)
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    manifest_payload = {
        "schema_id": "budgeted_laedge_holdout_manifest_v2",
        "run_id": run_id,
        "label": BUDGETED_LAEDGE_HOLDOUT_PROTOCOL_LABEL,
        "status": result["status"],
        "scheduler_calls": result["scheduler_calls"],
        "mapping_fingerprint": verification["mapping_fingerprint"],
        "protocol_fingerprint": verification["protocol_fingerprint"],
        "source_bundle_fingerprint": verification["source_bundle_fingerprint"],
        "physics_fingerprint": verification["physics_fingerprint"],
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    return (
        write_budgeted_laedge_holdout_artifact(
            artifacts_root,
            run_id,
            {
                "summary.json": summary,
                "paired_statistics.json": result.get("paired_statistics", {"status": "unavailable"}),
                "episode_rows.jsonl": result["episode_rows"],
                "manifest.json": manifest_payload,
                "protocol.json": result.get("protocol", _protocol_payload(_V2_PROTOCOL)),
            },
        ),
        result,
    )


__all__ = [
    "BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS",
    "BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES",
    "BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET",
    "BUDGETED_LAEDGE_HOLDOUT_EPISODES",
    "BUDGETED_LAEDGE_HOLDOUT_NAMESPACE",
    "BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE",
    "BudgetedLaedgeHoldoutError",
    "confirmed_v2_rates",
    "execute_budgeted_laedge_holdout_v2",
    "paired_cluster_statistics",
    "run_budgeted_laedge_holdout_preflight_v2",
    "run_budgeted_laedge_holdout_v2",
    "validate_confirmed_v2_development",
    "write_budgeted_laedge_holdout_artifact",
]
