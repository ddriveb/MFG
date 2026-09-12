"""Development validation runner for ADR-0034 Selective Hedge.

The runner freezes the nine normalized-slack candidates, uses one fixed
Budgeted LÆDGE v2 nominal-5% capacity envelope, and compares every candidate
against dynamic No-Hedge on the same immutable episode trace.  This module
does not run holdout automatically and never makes an equilibrium claim.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import asdict
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
from .attribution_episode import ATTRIBUTION_V1_PROTOCOL, EpisodeTrace, generate_episode_trace
from .attribution_metrics import SLODeadlines
from .budgeted_laedge import BudgetedLaedgeResult
from .budgeted_laedge_campaign import (
    BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
    BUDGETED_LAEDGE_STORM_BIN_WIDTH,
    _outcome,
    _simulation_for_arm,
    _trace_digest,
)
from .budgeted_laedge_holdout import (
    _METRICS,
    _bootstrap_fingerprint,
    _comparison,
    _digest,
    _peak_rss_bytes,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .selective_hedging import (
    SELECTIVE_HEDGE_THRESHOLD_GRID,
    SelectiveHedgeResult,
    SelectiveHedgeThresholds,
    selective_arm_key,
    simulate_selective_laedge_episode,
)
from .token_t3a_execution import validate_frozen_config


SELECTIVE_PLANNED_BUDGET_RATE = 0.09304612180547656
SELECTIVE_PLANNED_BUDGET_RATE_SOURCE = "budgeted-laedge-v2-nominal-5-percent-mapping"
SELECTIVE_THRESHOLD_GRID = SELECTIVE_HEDGE_THRESHOLD_GRID
SELECTIVE_DEVELOPMENT_EPISODES = 256
SELECTIVE_DEVELOPMENT_NAMESPACE = (
    "replica-routing-baselines:selective-hedging:v1:development"
)
SELECTIVE_DEVELOPMENT_MACRO_SEED = 20260916
SELECTIVE_HOLDOUT_EPISODES = 1024
SELECTIVE_HOLDOUT_NAMESPACE = (
    "replica-routing-baselines:selective-hedging:v1:holdout"
)
SELECTIVE_HOLDOUT_MACRO_SEED = 20260917
SELECTIVE_DEVELOPMENT_BOOTSTRAP_NAMESPACE = (
    "replica-routing-baselines:selective-hedging:v1:development:paired-bootstrap"
)
SELECTIVE_DEVELOPMENT_BOOTSTRAP_SEED = 20260918
SELECTIVE_BOOTSTRAP_REPLICATES = 4096
SELECTIVE_REFERENCE_ARM_KEYS = (
    "fixed-dispatcher:no-hedge",
    "niin:conservative",
    "budgeted-laedge:delta=0%",
)
SELECTIVE_CANDIDATE_ARM_KEYS = tuple(
    selective_arm_key(SelectiveHedgeThresholds(regular=regular, urgent=urgent))
    for regular in SELECTIVE_THRESHOLD_GRID
    for urgent in SELECTIVE_THRESHOLD_GRID
)
SELECTIVE_DEVELOPMENT_ARM_KEYS = (
    *SELECTIVE_REFERENCE_ARM_KEYS,
    *SELECTIVE_CANDIDATE_ARM_KEYS,
)
SELECTIVE_DEVELOPMENT_CALL_BUDGET = (
    SELECTIVE_DEVELOPMENT_EPISODES * len(SELECTIVE_DEVELOPMENT_ARM_KEYS)
)
SELECTIVE_HOLDOUT_CALL_BUDGET = (
    SELECTIVE_HOLDOUT_EPISODES * (len(SELECTIVE_REFERENCE_ARM_KEYS) + 1)
)
SELECTIVE_DEVELOPMENT_LABEL = "finite_selective_hedging_development_v1"
SELECTIVE_HOLDOUT_LABEL = "finite_selective_hedging_holdout_v1"
SELECTIVE_DEVELOPMENT_PREFLIGHT_NAMESPACE = (
    "replica-routing-baselines:selective-hedging:v1:development:preflight"
)
SELECTIVE_DEVELOPMENT_PREFLIGHT_MACRO_SEED = 20260919
SELECTIVE_DEVELOPMENT_PREFLIGHT_EPISODES = 2


class SelectiveHedgeExperimentError(RuntimeError):
    """Fail-closed development/holdout contract violation."""


def _selective_bootstrap_indices(sample_count: int) -> array:
    """Build the frozen development bootstrap library.

    The paired statistic implementation is shared with the earlier holdout
    runner, but its random-index library is protocol-specific.  Keeping the
    namespace and seed here prevents a development result from silently
    inheriting the Budgeted LÆDGE holdout bootstrap stream.
    """

    if type(sample_count) is not int or sample_count <= 0:
        raise SelectiveHedgeExperimentError(
            "bootstrap sample count must be positive"
        )
    seed_material = (
        f"{SELECTIVE_DEVELOPMENT_BOOTSTRAP_NAMESPACE}:"
        f"{SELECTIVE_DEVELOPMENT_BOOTSTRAP_SEED}:{sample_count}:"
        f"{SELECTIVE_BOOTSTRAP_REPLICATES}"
    )
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest(),
        "big",
    )
    rng = random.Random(seed)
    indices = array("I")
    for _ in range(SELECTIVE_BOOTSTRAP_REPLICATES * sample_count):
        indices.append(rng.randrange(sample_count))
    return indices


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectiveHedgeExperimentError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise SelectiveHedgeExperimentError(f"{name} must be finite")
    return result


def _validate_fixed_rate(rate: object) -> float:
    value = _finite(rate, "planned_budget_rate")
    if value != SELECTIVE_PLANNED_BUDGET_RATE:
        raise SelectiveHedgeExperimentError(
            "planned_budget_rate does not match the frozen ADR-0034 envelope"
        )
    return value


def build_selective_development_protocol() -> dict[str, object]:
    """Return a fresh, timestamp-free development protocol payload."""

    return {
        "schema_id": "selective_hedging_development_protocol_v1",
        "label": SELECTIVE_DEVELOPMENT_LABEL,
        "namespace": SELECTIVE_DEVELOPMENT_NAMESPACE,
        "macro_seed": SELECTIVE_DEVELOPMENT_MACRO_SEED,
        "episode_count": SELECTIVE_DEVELOPMENT_EPISODES,
        "arm_keys": list(SELECTIVE_DEVELOPMENT_ARM_KEYS),
        "threshold_grid": list(SELECTIVE_THRESHOLD_GRID),
        "planned_budget_rate": SELECTIVE_PLANNED_BUDGET_RATE,
        "planned_budget_rate_source": SELECTIVE_PLANNED_BUDGET_RATE_SOURCE,
        "planned_rate_means_achieved_delta": False,
        "underfill_is_valid": True,
        "window_width": 25.0,
        "state_boundary_windows": True,
        "degraded_slowdown": BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
        "cancellation": "conservative",
        "complete_drain": True,
        "crn_fingerprint_schema": "episode_trace_fingerprint_v1",
        "bootstrap_namespace": SELECTIVE_DEVELOPMENT_BOOTSTRAP_NAMESPACE,
        "bootstrap_seed": SELECTIVE_DEVELOPMENT_BOOTSTRAP_SEED,
        "bootstrap_replicates": SELECTIVE_BOOTSTRAP_REPLICATES,
        "scheduler_call_budget": SELECTIVE_DEVELOPMENT_CALL_BUDGET,
        "claim_boundary": {
            "best_response": False,
            "optimality": False,
            "nash": False,
            "mfg": False,
        },
    }


def _thresholds_from_arm_key(arm_key: str) -> SelectiveHedgeThresholds:
    prefix = "selective-laedge:theta_regular="
    if not isinstance(arm_key, str) or not arm_key.startswith(prefix):
        raise SelectiveHedgeExperimentError(f"not a Selective Hedge arm: {arm_key!r}")
    body = arm_key[len(prefix):]
    marker = ":theta_urgent="
    if marker not in body:
        raise SelectiveHedgeExperimentError(f"malformed Selective Hedge arm: {arm_key!r}")
    regular_text, urgent_text = body.split(marker, 1)
    try:
        thresholds = SelectiveHedgeThresholds(float(regular_text), float(urgent_text))
    except (TypeError, ValueError) as error:
        raise SelectiveHedgeExperimentError(f"malformed Selective Hedge arm: {arm_key!r}") from error
    if thresholds.regular not in SELECTIVE_THRESHOLD_GRID or thresholds.urgent not in SELECTIVE_THRESHOLD_GRID:
        raise SelectiveHedgeExperimentError(f"arm is outside frozen threshold grid: {arm_key!r}")
    if selective_arm_key(thresholds) != arm_key:
        raise SelectiveHedgeExperimentError(f"non-canonical Selective Hedge arm: {arm_key!r}")
    return thresholds


def _budget_payload(raw: BudgetedLaedgeResult | SelectiveHedgeResult) -> dict[str, object]:
    windows = [
        {**asdict(window), "phase": window.phase.value}
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


def _selective_audit(raw: SelectiveHedgeResult) -> dict[str, object]:
    tokens = raw.simulation.simulation.tokens
    hedges = tuple(
        attempt
        for token in tokens
        for attempt in token.attempts
        if attempt.attempt_id == 2
    )
    d_hedges = tuple(
        attempt
        for attempt in hedges
        if phase_at(raw.simulation.episode.protocol.timeline, attempt.start_time)
        is Phase.DEGRADED
    )
    reason_counts = Counter(item.reason for item in raw.suppression_reasons)
    d_class_counts = Counter(
        token.token_class.value
        for token in tokens
        if any(attempt.attempt_id == 2 for attempt in token.attempts)
        and phase_at(raw.simulation.episode.protocol.timeline, next(
            attempt.start_time for attempt in token.attempts if attempt.attempt_id == 2
        )) is Phase.DEGRADED
    )
    return {
        "suppression_reason_counts": dict(sorted(reason_counts.items())),
        "d_phase_hedge_launches": len(d_hedges),
        "d_phase_hedge_start_times": [attempt.start_time for attempt in d_hedges],
        "d_phase_hedge_class_counts": dict(sorted(d_class_counts.items())),
        "hedge_launches": len(hedges),
        "hedge_winner_count": sum(
            attempt.status.value == "completed_winner" for attempt in hedges
        ),
        "planned_budget_rate": raw.planned_budget_rate,
        "planned_rate_source": SELECTIVE_PLANNED_BUDGET_RATE_SOURCE,
        "underfill_is_valid": True,
    }


def _run_selective_arm(episode: EpisodeTrace, arm_key: str) -> dict[str, object]:
    thresholds = _thresholds_from_arm_key(arm_key)
    before = _trace_digest(episode)
    raw = simulate_selective_laedge_episode(
        episode,
        planned_budget_rate=_validate_fixed_rate(SELECTIVE_PLANNED_BUDGET_RATE),
        thresholds=thresholds,
        deadlines=SLODeadlines(),
        degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
        cancel_running_losers=False,
    )
    after = _trace_digest(episode)
    if before != after:
        raise SelectiveHedgeExperimentError(f"arm {arm_key} mutated its episode")
    return {
        "arm_key": arm_key,
        "trace_fingerprint": before,
        "outcome": _outcome(raw.simulation, placement_mode="budgeted_idle_release"),
        "budget": _budget_payload(raw),
        "selective_audit": _selective_audit(raw),
    }


def _run_reference_arm(episode: EpisodeTrace, arm_key: str) -> dict[str, object]:
    before = _trace_digest(episode)
    simulation = _simulation_for_arm(
        episode,
        arm_key,
        planned_budget_rate=0.0,
    )
    after = _trace_digest(episode)
    if before != after:
        raise SelectiveHedgeExperimentError(f"arm {arm_key} mutated its episode")
    placement_mode = (
        "budgeted_idle_release"
        if arm_key == "budgeted-laedge:delta=0%"
        else "fixed_dispatcher"
    )
    return {
        "arm_key": arm_key,
        "trace_fingerprint": before,
        "outcome": _outcome(simulation, placement_mode=placement_mode),
        "budget": None,
    }


def _arm_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise SelectiveHedgeExperimentError("cannot summarize empty arm")
    summary: dict[str, object] = {"episode_count": len(rows)}
    for metric in _METRICS:
        values = [row["outcome"].get(metric) for row in rows]
        summary[metric] = (
            None
            if any(value is None for value in values)
            else math.fsum(float(value) for value in values) / len(values)
        )
    audits = [row.get("selective_audit") for row in rows if row.get("selective_audit")]
    if audits:
        summary["suppression_reason_counts"] = dict(
            sorted(
                sum(
                    (Counter(audit["suppression_reason_counts"]) for audit in audits),
                    Counter(),
                ).items()
            )
        )
        summary["d_phase_hedge_launches"] = math.fsum(
            audit["d_phase_hedge_launches"] for audit in audits
        ) / len(audits)
        summary["d_phase_hedge_class_counts"] = dict(
            sorted(
                sum(
                    (Counter(audit["d_phase_hedge_class_counts"]) for audit in audits),
                    Counter(),
                ).items()
            )
        )
    return summary


def select_development_threshold(
    candidate_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    """Select one candidate using only paired development metrics."""

    passing = []
    for arm_key, metrics in candidate_metrics.items():
        try:
            cvar_ci = metrics["df_cvar95"]["ci95"]
            miss_ci = metrics["df_miss_rate"]["ci95"]
            work_relative = float(metrics["total_work"]["relative_difference"])
            storm_raw = metrics["storm_peak"]
            storm = (
                float(storm_raw["point_estimate"])
                if isinstance(storm_raw, Mapping)
                else float(storm_raw)
            )
            cvar_upper = float(cvar_ci[1])
            miss_upper = float(miss_ci[1])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise SelectiveHedgeExperimentError(
                f"candidate metrics are incomplete for {arm_key!r}"
            ) from error
        if (
            (cvar_upper < 0.0 or miss_upper < 0.0)
            and cvar_upper <= 0.0
            and miss_upper <= 0.0
            and work_relative <= 0.05
            and storm <= 2.0
        ):
            passing.append(
                (
                    work_relative,
                    float(metrics["df_cvar95"].get("point_estimate", math.inf)),
                    float(metrics["df_miss_rate"].get("point_estimate", math.inf)),
                    arm_key,
                )
            )
    if not passing:
        return {"status": "non_improving", "arm_key": None, "candidates": []}
    passing.sort()
    selected = passing[0][3]
    thresholds = _thresholds_from_arm_key(selected)
    return {
        "status": "confirmed",
        "arm_key": selected,
        "theta_regular": thresholds.regular,
        "theta_urgent": thresholds.urgent,
        "candidates": [item[3] for item in passing],
    }


def _source_bundle(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    entries = []
    for path in sorted((root / "src" / "mfg_hedge").glob("*.py")):
        entries.append([
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ])
    return {
        "schema_id": "selective_hedging_source_bundle_v1",
        "entries": entries,
        "fingerprint": _digest(entries),
    }


def _run_episode(
    episode: EpisodeTrace,
) -> dict[str, object]:
    before = _trace_digest(episode)
    arms: dict[str, object] = {}
    failures = []
    for arm_key in SELECTIVE_DEVELOPMENT_ARM_KEYS:
        try:
            if arm_key in SELECTIVE_REFERENCE_ARM_KEYS:
                item = _run_reference_arm(episode, arm_key)
            else:
                item = _run_selective_arm(episode, arm_key)
            if item["trace_fingerprint"] != before:
                raise SelectiveHedgeExperimentError("arm trace fingerprint differs")
            arms[arm_key] = {"status": "completed", **item}
        except Exception as error:
            failures.append({
                "arm_key": arm_key,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
            arms[arm_key] = {
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
    return {
        "status": "completed" if not failures else "physical_failed",
        "trace_fingerprint": before,
        "arms": arms,
        "failures": failures,
    }


def run_selective_development(
    config: ExperimentConfig,
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    episode_count: int = SELECTIVE_DEVELOPMENT_EPISODES,
    namespace: str = SELECTIVE_DEVELOPMENT_NAMESPACE,
    macro_seed: int = SELECTIVE_DEVELOPMENT_MACRO_SEED,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the fixed development panel; never runs holdout."""

    validate_frozen_config(config)
    _validate_fixed_rate(SELECTIVE_PLANNED_BUDGET_RATE)
    if type(episode_count) is not int or not 0 < episode_count <= SELECTIVE_DEVELOPMENT_EPISODES:
        raise SelectiveHedgeExperimentError("development episode count is outside the frozen limit")
    build = episode_builder or (
        lambda current_namespace, index: generate_episode_trace(
            config,
            current_namespace,
            macro_seed,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )
    )
    records: dict[str, list[dict[str, object]]] = {
        arm_key: [] for arm_key in SELECTIVE_DEVELOPMENT_ARM_KEYS
    }
    episode_rows = []
    failures = []
    calls = 0
    started = time.perf_counter()
    for index in range(episode_count):
        row = {
            "episode_index": index,
            "namespace": namespace,
            "macro_seed": macro_seed,
            "status": "completed",
            "arms": {},
        }
        try:
            episode = build(namespace, index)
            if not isinstance(episode, EpisodeTrace):
                raise SelectiveHedgeExperimentError("episode builder returned a non-EpisodeTrace")
            row["trace_fingerprint"] = _trace_digest(episode)
            episode_result = _run_episode(episode)
            for arm_key, arm_row in episode_result["arms"].items():
                calls += 1
                row["arms"][arm_key] = arm_row
                if arm_row["status"] == "completed":
                    records[arm_key].append(arm_row)
                else:
                    failures.append({
                        "episode_index": index,
                        **arm_row,
                    })
            row["status"] = episode_result["status"]
        except Exception as error:
            row["status"] = "physical_failed"
            failures.append({
                "episode_index": index,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
            for arm_key in SELECTIVE_DEVELOPMENT_ARM_KEYS:
                calls += 1
                row["arms"][arm_key] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        episode_rows.append(row)
        if progress is not None:
            progress(index + 1, episode_count)
    elapsed = time.perf_counter() - started
    complete = (
        not failures
        and calls == episode_count * len(SELECTIVE_DEVELOPMENT_ARM_KEYS)
        and all(len(rows) == episode_count for rows in records.values())
    )
    result: dict[str, object] = {
        "label": SELECTIVE_DEVELOPMENT_LABEL,
        "status": "physical_failed" if not complete else "pending_selection",
        "namespace": namespace,
        "macro_seed": macro_seed,
        "episode_count": episode_count,
        "scheduler_calls": calls,
        "formal_scheduler_call_budget": episode_count * len(SELECTIVE_DEVELOPMENT_ARM_KEYS),
        "elapsed_seconds": elapsed,
        "calls_per_second": calls / elapsed if elapsed > 0.0 else None,
        "failures": failures,
        "episode_rows": episode_rows,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "protocol": build_selective_development_protocol(),
    }
    if not complete:
        result["paired_statistics"] = {"status": "unavailable", "reason": "physical_failed"}
        result["memory"] = _peak_rss_bytes()
        return result
    indices = _selective_bootstrap_indices(episode_count)
    summaries = {arm_key: _arm_summary(rows) for arm_key, rows in records.items()}
    comparisons = {}
    candidate_metrics = {}
    delta_zero = "budgeted-laedge:delta=0%"
    for arm_key in SELECTIVE_CANDIDATE_ARM_KEYS:
        comparison = _comparison(records, delta_zero, arm_key, indices)
        comparisons[arm_key] = comparison
        candidate_metrics[arm_key] = comparison["metrics"]
    selection = select_development_threshold(candidate_metrics)
    result["status"] = selection["status"]
    result["arm_summaries"] = summaries
    result["candidate_selection"] = selection
    result["paired_statistics"] = {
        "status": "completed",
        "cluster_unit": "episode_index",
        "bootstrap_replicates": SELECTIVE_BOOTSTRAP_REPLICATES,
        "bootstrap_fingerprint": _bootstrap_fingerprint(indices),
        "baseline_arm": delta_zero,
        "comparisons": comparisons,
    }
    result["protocol_fingerprint"] = _digest(result["protocol"])
    result["memory"] = _peak_rss_bytes()
    return result


def run_selective_development_preflight(
    config: ExperimentConfig,
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    episode_count: int = SELECTIVE_DEVELOPMENT_PREFLIGHT_EPISODES,
) -> dict[str, object]:
    """Run only the isolated timing/physics preflight panel."""

    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= 8:
        raise SelectiveHedgeExperimentError(
            "preflight episode count must be in 1..8"
        )
    build = episode_builder or (
        lambda current_namespace, index: generate_episode_trace(
            config,
            current_namespace,
            SELECTIVE_DEVELOPMENT_PREFLIGHT_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )
    )
    started = time.perf_counter()
    calls = 0
    failures = []
    for index in range(episode_count):
        try:
            episode = build(SELECTIVE_DEVELOPMENT_PREFLIGHT_NAMESPACE, index)
            if not isinstance(episode, EpisodeTrace):
                raise SelectiveHedgeExperimentError(
                    "preflight episode builder returned a non-EpisodeTrace"
                )
            before = _trace_digest(episode)
            episode_result = _run_episode(episode)
            calls += len(SELECTIVE_DEVELOPMENT_ARM_KEYS)
            if _trace_digest(episode) != before:
                raise SelectiveHedgeExperimentError(
                    "preflight trace fingerprint changed"
                )
            if episode_result["status"] != "completed":
                failures.extend(
                    {
                        "episode_index": index,
                        **failure,
                    }
                    for failure in episode_result["failures"]
                )
        except Exception as error:
            calls += len(SELECTIVE_DEVELOPMENT_ARM_KEYS)
            failures.append({
                "episode_index": index,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            })
    elapsed = time.perf_counter() - started
    calls_per_second = calls / elapsed if elapsed > 0.0 else None
    return {
        "status": "ok" if not failures else "physical_failed",
        "namespace": SELECTIVE_DEVELOPMENT_PREFLIGHT_NAMESPACE,
        "macro_seed": SELECTIVE_DEVELOPMENT_PREFLIGHT_MACRO_SEED,
        "episode_count": episode_count,
        "scheduler_calls": calls,
        "calls_per_second": calls_per_second,
        "elapsed_seconds": elapsed,
        "estimated_formal_seconds": (
            SELECTIVE_DEVELOPMENT_CALL_BUDGET / calls_per_second
            if calls_per_second
            else None
        ),
        "memory": _peak_rss_bytes(),
        "failures": failures,
        "formal_scheduler_call_budget": SELECTIVE_DEVELOPMENT_CALL_BUDGET,
    }


def execute_selective_development(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    preflight_episode_count: int = SELECTIVE_DEVELOPMENT_PREFLIGHT_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Preflight and write one immutable development artifact."""

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    preflight = run_selective_development_preflight(
        config,
        episode_count=preflight_episode_count,
    )
    if preflight["status"] != "ok":
        raise SelectiveHedgeExperimentError(
            "timing preflight failed; formal development not started"
        )
    result = run_selective_development(config, progress=progress)
    source_bundle = _source_bundle(project)
    result["source_bundle"] = source_bundle
    result["source_bundle_fingerprint"] = source_bundle["fingerprint"]
    result["preflight"] = preflight
    result["physics_fingerprint"] = _digest({
        "degraded_slowdown": BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
        "window_width": 25.0,
        "state_boundary_windows": True,
        "cancellation": "conservative",
        "complete_drain": True,
    })
    result["protocol_fingerprint"] = _digest(result["protocol"])
    summary = {
        key: value for key, value in result.items() if key != "episode_rows"
    }
    manifest = {
        "schema_id": "selective_hedging_development_manifest_v1",
        "run_id": run_id,
        "label": SELECTIVE_DEVELOPMENT_LABEL,
        "status": result["status"],
        "scheduler_calls": result["scheduler_calls"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "source_bundle_fingerprint": source_bundle["fingerprint"],
        "physics_fingerprint": result["physics_fingerprint"],
        "planned_budget_rate": SELECTIVE_PLANNED_BUDGET_RATE,
        "planned_budget_rate_source": SELECTIVE_PLANNED_BUDGET_RATE_SOURCE,
        "planned_rate_means_achieved_delta": False,
        "underfill_is_valid": True,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    return (
        write_selective_hedging_artifact(
            artifacts_root,
            run_id,
            {
                "summary.json": summary,
                "paired_statistics.json": result.get(
                    "paired_statistics", {"status": "unavailable"}
                ),
                "episode_rows.jsonl": result["episode_rows"],
                "manifest.json": manifest,
                "protocol.json": result["protocol"],
            },
        ),
        result,
    )


def write_selective_hedging_artifact(
    artifacts_root: str | Path,
    run_id: str,
    files: Mapping[str, object],
) -> Path:
    """Write a development artifact transactionally without overwriting."""

    _validate_run_id(run_id)
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
                raise ValueError(f"invalid artifact file name: {name!r}")
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


__all__ = [
    "SELECTIVE_CANDIDATE_ARM_KEYS",
    "SELECTIVE_DEVELOPMENT_ARM_KEYS",
    "SELECTIVE_DEVELOPMENT_CALL_BUDGET",
    "SELECTIVE_DEVELOPMENT_EPISODES",
    "SELECTIVE_DEVELOPMENT_NAMESPACE",
    "SELECTIVE_DEVELOPMENT_MACRO_SEED",
    "SELECTIVE_HOLDOUT_CALL_BUDGET",
    "SELECTIVE_HOLDOUT_EPISODES",
    "SELECTIVE_HOLDOUT_NAMESPACE",
    "SELECTIVE_HOLDOUT_MACRO_SEED",
    "SELECTIVE_PLANNED_BUDGET_RATE",
    "SELECTIVE_THRESHOLD_GRID",
    "SelectiveHedgeExperimentError",
    "build_selective_development_protocol",
    "execute_selective_development",
    "run_selective_development",
    "run_selective_development_preflight",
    "select_development_threshold",
    "write_selective_hedging_artifact",
]
