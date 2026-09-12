"""Frozen paired protection-baseline campaign from ADR-0031."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .artifacts import write_run_directory
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeSimulationResult,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
    simulate_episode,
)
from .attribution_metrics import (
    SLODeadlines,
    _latency_views,
    empirical_cvar95,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction
from .laedge_episode import simulate_laedge_episode
from .metrics import percentile
from .token_online import simulate_episode_online
from .token_t3a_execution import (
    _RESERVATION,
    build_source_bundle,
    environment_fingerprint,
    validate_frozen_config,
)
from .token_three_arm_evaluation import (
    BOOTSTRAP_REPLICATES,
    FrozenNIINPolicy,
    REQUESTED_RESERVATION_PRICE,
    THREE_ARM_MACRO_SEED,
    THREE_ARM_NAMESPACE,
    _bootstrap_index_library,
    _comparison_entry,
    load_requested_price_candidate,
)


PROTECTION_LABEL = "finite_protection_baseline_comparison_v1"
PROTECTION_ARMS = (
    "failover_only",
    "p95_delayed_hedge",
    "laedge_work_conserving",
    "pi_NIIN_v1",
    "pi_requested_reservation_price_v1",
)
EVALUATION_EPISODES = 1024
EVALUATION_CALL_LIMIT = EVALUATION_EPISODES * len(PROTECTION_ARMS)
P95_CALIBRATION_NAMESPACE = "replica-routing-baselines:p95-calibration:v1"
P95_CALIBRATION_MACRO_SEED = 20260930
P95_CALIBRATION_EPISODES = 256
DEGRADED_SLOWDOWN = 2.0
NIIN_HEDGE_DELAY = 2.0
STORM_BIN_WIDTH = 1.0
_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)


class ProtectionExperimentError(RuntimeError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _trace_digest(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _evaluation_episode(config: ExperimentConfig, index: int) -> EpisodeTrace:
    return generate_episode_trace(
        config,
        THREE_ARM_NAMESPACE,
        THREE_ARM_MACRO_SEED,
        index,
        ATTRIBUTION_V1_PROTOCOL,
    )


def calibrate_p95_delay(
    config: ExperimentConfig,
    *,
    episode_count: int = P95_CALIBRATION_EPISODES,
) -> dict[str, object]:
    """Estimate one arrival-based P95 timer on a disjoint No-Hedge library."""

    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= P95_CALIBRATION_EPISODES:
        raise ProtectionExperimentError("P95 calibration episode_count must be in 1..256")
    latencies: list[float] = []
    fingerprints: list[str] = []
    for index in range(episode_count):
        episode = generate_episode_trace(
            config,
            P95_CALIBRATION_NAMESPACE,
            P95_CALIBRATION_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )
        before = _trace_digest(episode)
        run = simulate_episode(episode, DEGRADED_SLOWDOWN)
        if _trace_digest(episode) != before:
            raise ProtectionExperimentError("P95 calibration mutated an episode")
        fingerprints.append(before)
        latencies.extend(
            token.latency
            for token in run.simulation.tokens
            if token.arrival_time < episode.protocol.timeline.degraded_start
        )
    if len(set(fingerprints)) != len(fingerprints) or not latencies:
        raise ProtectionExperimentError("P95 calibration library is invalid")
    delay = percentile(latencies, 95.0)
    if not math.isfinite(delay) or delay <= 0.0:
        raise ProtectionExperimentError("P95 calibration produced an invalid delay")
    return {
        "namespace": P95_CALIBRATION_NAMESPACE,
        "macro_seed": P95_CALIBRATION_MACRO_SEED,
        "episode_count": episode_count,
        "scheduler_calls": episode_count,
        "latency_sample_count": len(latencies),
        "cohort": "healthy_arrival_phase",
        "delay": delay,
        "trace_library_fingerprint": _digest(fingerprints),
    }


def _latency_row(tokens: Sequence[object], deadlines: SLODeadlines) -> dict[str, float]:
    if not tokens:
        raise ProtectionExperimentError("required latency cohort is empty")
    values = [float(token.latency) for token in tokens]
    return {
        "mean": math.fsum(values) / len(values),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "cvar95": empirical_cvar95(values),
        "miss_rate": sum(
            token.latency > deadlines.for_class(token.token_class) for token in tokens
        ) / len(tokens),
    }


def _episode_outcomes(run: EpisodeSimulationResult) -> dict[str, float]:
    tokens = tuple(run.simulation.tokens)
    by_phase = {
        phase: tuple(
            token
            for token in tokens
            if phase_at(run.episode.protocol.timeline, token.arrival_time) is phase
        )
        for phase in _PHASES
    }
    df = by_phase[Phase.DEGRADED] + by_phase[Phase.FAILED]
    hr = by_phase[Phase.HEALTHY] + by_phase[Phase.RECOVERED]
    deadlines = SLODeadlines()
    overall = _latency_row(tokens, deadlines)
    df_row = _latency_row(df, deadlines)
    hr_row = _latency_row(hr, deadlines)
    attempts = run.simulation.attempts
    winners = [
        row
        for token in tokens
        for row in token.attempts
        if row.status.value == "completed_winner"
    ]
    if len(winners) != len(tokens) or any(
        row.executed_work < 0.0
        or row.executed_work > row.required_work + 1e-9
        or not math.isfinite(row.executed_work)
        for row in attempts
    ):
        raise ProtectionExperimentError("common physical invariant failure")
    launch_bins: dict[int, int] = {}
    for row in attempts:
        if row.attempt_id == 2 and row.enqueue_time < run.episode.protocol.arrival_cutoff:
            index = int(row.enqueue_time // STORM_BIN_WIDTH)
            launch_bins[index] = launch_bins.get(index, 0) + 1
    total_work = math.fsum(row.executed_work for row in attempts)
    wasted_work = math.fsum(
        row.executed_work
        for row in attempts
        if row.status.value != "completed_winner"
    )
    return {
        "overall_mean": overall["mean"],
        "overall_p95": overall["p95"],
        "overall_p99": overall["p99"],
        "df_cvar95": df_row["cvar95"],
        "df_miss_rate": df_row["miss_rate"],
        "df_p95": df_row["p95"],
        "df_p99": df_row["p99"],
        "hr_p99": hr_row["p99"],
        "replay_rate": sum(token.replay_count > 0 for token in tokens) / len(tokens),
        "total_work": total_work,
        "wasted_work": wasted_work,
        "storm_peak": float(max(launch_bins.values(), default=0)),
    }


def _aggregate(
    runs: Sequence[EpisodeSimulationResult], outcomes: Sequence[Mapping[str, float]]
) -> dict[str, object]:
    tokens = tuple(token for run in runs for token in run.simulation.tokens)
    timeline = runs[0].episode.protocol.timeline
    latency = _latency_views(tokens, timeline, SLODeadlines())
    df = tuple(
        token
        for token in tokens
        if phase_at(timeline, token.arrival_time) in (Phase.DEGRADED, Phase.FAILED)
    )
    hr = tuple(
        token
        for token in tokens
        if phase_at(timeline, token.arrival_time) in (Phase.HEALTHY, Phase.RECOVERED)
    )
    deadlines = SLODeadlines()
    df_row = _latency_row(df, deadlines)
    hr_row = _latency_row(hr, deadlines)
    simulations = [run.simulation for run in runs]
    attempts = tuple(row for result in simulations for row in result.attempts)
    primary_work = math.fsum(row.executed_work for row in attempts if row.attempt_id == 0)
    replay_work = math.fsum(row.executed_work for row in attempts if row.attempt_id == 1)
    hedge_work = math.fsum(row.executed_work for row in attempts if row.attempt_id == 2)
    total_work = primary_work + replay_work + hedge_work
    wasted_work = math.fsum(
        row.executed_work
        for row in attempts
        if row.status.value != "completed_winner"
    )
    launched = sum(result.hedge_launches for result in simulations)
    winners = sum(
        token.winner_attempt_id == 2
        for result in simulations
        for token in result.tokens
    )
    replayed = sum(token.replay_count > 0 for token in tokens)
    return {
        "episode_count": len(runs),
        "token_count": len(tokens),
        "latency": {
            "overall": latency["overall"],
            "df": df_row,
            "hr": hr_row,
            "arrival_phase": latency["arrival_phase"],
        },
        "replay": {"count": replayed, "rate": replayed / len(tokens)},
        "hedge": {
            "requested": sum(result.hedge_requested for result in simulations),
            "launched": launched,
            "winners": winners,
            "win_per_launch": winners / launched if launched else None,
        },
        "work": {
            "executed_primary_work": primary_work,
            "executed_replay_work": replay_work,
            "executed_hedge_work": hedge_work,
            "total_executed_work": total_work,
            "wasted_work": wasted_work,
        },
        "storm": {
            "peak_max": max(row["storm_peak"] for row in outcomes),
            "peak_mean": math.fsum(row["storm_peak"] for row in outcomes) / len(outcomes),
        },
        "common_metrics": {
            "overall_mean": latency["overall"]["mean_latency"],
            "overall_p95": latency["overall"]["latency_p95"],
            "overall_p99": latency["overall"]["latency_p99"],
            "df_cvar95": df_row["cvar95"],
            "df_miss_rate": df_row["miss_rate"],
            "hr_p99": hr_row["p99"],
            "replay_rate": replayed / len(tokens),
            "total_work": total_work,
            "wasted_work": wasted_work,
            "storm_peak": max(row["storm_peak"] for row in outcomes),
        },
        "invariants": {
            "all_episode_invariants_pass": all(
                len(run.simulation.tokens) == len(run.episode.workload.tokens)
                and all(token.completion_time >= token.arrival_time for token in run.simulation.tokens)
                for run in runs
            )
        },
        "episode_outcomes": list(outcomes),
    }


def _run_arm(
    arm: str,
    episode: EpisodeTrace,
    p95_delay: float,
    candidate: FrozenNIINPolicy,
) -> EpisodeSimulationResult:
    if arm == "failover_only":
        return simulate_episode(episode, DEGRADED_SLOWDOWN)
    if arm == "p95_delayed_hedge":
        actions = {
            spec.token_id: ProtectionAction.DELAYED_HEDGE
            for spec in episode.workload.tokens
        }
        return simulate_episode(
            episode,
            DEGRADED_SLOWDOWN,
            actions=actions,
            hedge_delay=p95_delay,
        )
    if arm == "laedge_work_conserving":
        return simulate_laedge_episode(episode, degraded_slowdown=DEGRADED_SLOWDOWN)
    if arm == "pi_NIIN_v1":
        return simulate_episode_online(
            episode,
            FrozenNIINPolicy(),
            _RESERVATION,
            degraded_slowdown=DEGRADED_SLOWDOWN,
            hedge_delay=NIIN_HEDGE_DELAY,
        ).simulation
    if arm == "pi_requested_reservation_price_v1":
        return simulate_episode_online(
            episode,
            candidate,
            _RESERVATION,
            degraded_slowdown=DEGRADED_SLOWDOWN,
            hedge_delay=NIIN_HEDGE_DELAY,
            public_price=REQUESTED_RESERVATION_PRICE,
        ).simulation
    raise ProtectionExperimentError(f"unknown protection arm {arm!r}")


def run_protection_baseline_experiment(
    config: ExperimentConfig,
    *,
    episode_count: int = EVALUATION_EPISODES,
    calibration_episode_count: int = P95_CALIBRATION_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    candidate_policy: FrozenNIINPolicy | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= EVALUATION_EPISODES:
        raise ProtectionExperimentError("evaluation episode_count must be in 1..1024")
    calibration = calibrate_p95_delay(config, episode_count=calibration_episode_count)
    candidate = candidate_policy or load_requested_price_candidate()
    if not candidate.supported_probabilities:
        raise ProtectionExperimentError("requested-price candidate is incomplete")
    build = episode_builder or (lambda index: _evaluation_episode(config, index))
    runs: dict[str, list[EpisodeSimulationResult]] = {arm: [] for arm in PROTECTION_ARMS}
    outcomes: dict[str, list[dict[str, float]]] = {arm: [] for arm in PROTECTION_ARMS}
    rows = []
    seen = set()
    calls = 0
    failures = 0
    for index in range(episode_count):
        episode = build(index)
        before = _trace_digest(episode)
        if before in seen:
            raise ProtectionExperimentError("duplicate evaluation episode fingerprint")
        seen.add(before)
        row = {"episode_index": index, "status": "completed", "trace_fingerprints": {}, "arms": {}}
        completed: list[str] = []
        for arm in PROTECTION_ARMS:
            calls += 1
            try:
                run = _run_arm(arm, episode, float(calibration["delay"]), candidate)
                after = _trace_digest(episode)
                if after != before:
                    raise ProtectionExperimentError(f"arm {arm} mutated the episode")
                outcome = _episode_outcomes(run)
                runs[arm].append(run)
                outcomes[arm].append(outcome)
                completed.append(arm)
                row["trace_fingerprints"][arm] = after
                row["arms"][arm] = outcome
            except Exception as error:
                row["status"] = "failed"
                row["arms"][arm] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        if row["status"] == "failed":
            failures += 1
            for arm in completed:
                runs[arm].pop()
                outcomes[arm].pop()
        rows.append(row)
        if progress is not None:
            progress(index + 1, episode_count)
    if calls != episode_count * len(PROTECTION_ARMS) or calls > EVALUATION_CALL_LIMIT:
        raise ProtectionExperimentError("evaluation call accounting failed")
    complete = episode_count - failures
    result: dict[str, object] = {
        "label": PROTECTION_LABEL,
        "status": "completed" if failures == 0 else "physical_failed",
        "episode_count": episode_count,
        "complete_episode_count": complete,
        "failed_episode_count": failures,
        "evaluation_calls": calls,
        "evaluation_call_limit": EVALUATION_CALL_LIMIT,
        "calibration": calibration,
        "arms": {arm: _aggregate(runs[arm], outcomes[arm]) for arm in PROTECTION_ARMS if runs[arm]},
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": rows,
    }
    if complete:
        indices = _bootstrap_index_library(complete, BOOTSTRAP_REPLICATES)
        comparisons = {}
        metrics = tuple(next(iter(outcomes.values()))[0])
        for arm in PROTECTION_ARMS:
            if arm == "pi_NIIN_v1":
                continue
            comparisons[f"{arm}-minus-pi_NIIN_v1"] = {
                name: _comparison_entry(
                    [row[name] for row in outcomes["pi_NIIN_v1"]],
                    [row[name] for row in outcomes[arm]],
                    index_library=indices,
                )
                for name in metrics
            }
        result["paired_statistics"] = {
            "cluster_unit": "episode_index",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "comparisons": comparisons,
        }
    else:
        result["paired_statistics"] = None
    return result


def write_protection_baseline_artifact(
    artifacts_root: str | Path,
    run_id: str,
    result: Mapping[str, object],
    config: ExperimentConfig,
    *,
    source_bundle: Mapping[str, object] | None = None,
) -> Path:
    if result.get("label") != PROTECTION_LABEL:
        raise ProtectionExperimentError("unexpected protection experiment label")
    for claim in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg"):
        if result.get(claim) is not False:
            raise ProtectionExperimentError(f"{claim} must be false")
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    provenance = {
        "environment_fingerprint": environment_fingerprint(config),
        "evaluation_namespace": THREE_ARM_NAMESPACE,
        "evaluation_macro_seed": THREE_ARM_MACRO_SEED,
        "arms": list(PROTECTION_ARMS),
        "p95_calibration": result["calibration"],
        "source_bundle": dict(source_bundle or {}),
        "adr": "ADR-0031",
    }
    summary["provenance"] = provenance
    manifest = {
        "schema_id": "protection_baseline_manifest_v1",
        "run_id": run_id,
        "label": PROTECTION_LABEL,
        "status": result.get("status"),
        "provenance": provenance,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    return write_run_directory(
        artifacts_root,
        run_id,
        {
            "manifest.json": manifest,
            "summary.json": summary,
            "episode_rows.json": {"rows": result.get("episode_rows", [])},
            "paired_statistics.json": result.get("paired_statistics") or {"status": "unavailable"},
            "metric_contract.json": {
                "schema_id": "protection_baseline_metrics_v1",
                "paired_cluster": "episode_index",
                "primary_reference": "pi_NIIN_v1",
                "resource_metrics": ["total_work", "wasted_work", "storm_peak"],
            },
        },
    )


def execute_protection_baseline_experiment(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    candidate = load_requested_price_candidate(project / "artifacts/token-refined-fixed-grid-20260908-r1/summary.json")
    result = run_protection_baseline_experiment(
        config, candidate_policy=candidate, progress=progress
    )
    directory = write_protection_baseline_artifact(
        artifacts_root,
        run_id,
        result,
        config,
        source_bundle=asdict(build_source_bundle(project)),
    )
    return directory, result


__all__ = [
    "EVALUATION_EPISODES",
    "PROTECTION_ARMS",
    "ProtectionExperimentError",
    "calibrate_p95_delay",
    "execute_protection_baseline_experiment",
    "run_protection_baseline_experiment",
    "write_protection_baseline_artifact",
]
