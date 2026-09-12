"""Paired population A/B diagnostic for the T3B Token candidate.

Both arms run the complete online engine on the same immutable episode trace.
This module reports descriptive finite-population effects only; it does not
iterate policies or make an equilibrium claim.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .artifacts import write_run_directory
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
)
from .attribution_metrics import build_episode_metrics
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction
from .hedge_simulation import HedgeAttemptStatus
from .metrics import percentile
from .token_best_response import (
    CANDIDATE_LABEL,
    CandidateTable,
    load_t3a_runtime_candidate,
    ConditionalTokenCandidatePolicy,
)
from .token_online import OnlineEpisodeResult, simulate_episode_online
from .token_t3a_execution import (
    ENVIRONMENT_ID,
    POPULATION_POLICY_ID,
    TARGET_SELECTION_ID,
    _RESERVATION,
    build_source_bundle,
    environment_fingerprint,
    policy_factory,
    population_policy_fingerprint,
    validate_frozen_config,
)


T3C_LABEL = "population_policy_ab_diagnostic"
T3C_NAMESPACE = "token-mfg-restoration:t3c:population-ab:v1"
T3C_MACRO_SEED = 20260912
T3C_EPISODES = 2048
T3C_CALL_LIMIT = 4096
DEGRADED_SLOWDOWN = 2.0
HEDGE_DELAY = 2.0
STORM_BIN_WIDTH = 1.0
PARENT_T3B_RUN_ID = "token-t3b-qualification-20260907-r1"


class T3CExecutionError(RuntimeError):
    """Fail-closed population A/B protocol violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def population_call_budget(episode_count: int) -> int:
    if type(episode_count) is not int or episode_count < 0:
        raise T3CExecutionError("episode_count must be a non-negative int")
    return episode_count * 2


def latency_summary(values: Iterable[float]) -> dict[str, object]:
    samples = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in samples):
        raise T3CExecutionError("latency samples must be finite")
    if not samples:
        return {"count": 0, "mean": None, "p95": None, "p99": None}
    return {
        "count": len(samples),
        "mean": math.fsum(samples) / len(samples),
        "p95": percentile(samples, 95.0),
        "p99": percentile(samples, 99.0),
    }


def _trace_fingerprint(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _token_identity_fingerprint(run: OnlineEpisodeResult) -> str:
    return _digest(
        [
            [token.token_id, token.arrival_time, token.token_class.value]
            for token in run.simulation.simulation.tokens
        ]
    )


def _phase_latency_values(run: OnlineEpisodeResult) -> dict[str, list[float]]:
    timeline = run.episode.protocol.timeline
    values: dict[str, list[float]] = {phase.value: [] for phase in Phase}
    for token in run.simulation.simulation.tokens:
        values[phase_at(timeline, token.arrival_time).value].append(token.latency)
    return values


def _episode_arm_metrics(run: OnlineEpisodeResult) -> tuple[dict[str, object], dict[str, list[float]]]:
    simulation = run.simulation.simulation
    tokens = simulation.tokens
    attempts = simulation.attempts
    phase_values = _phase_latency_values(run)
    attribution = build_episode_metrics(
        run.simulation,
        DEGRADED_SLOWDOWN,
        storm_bin_width=STORM_BIN_WIDTH,
    )
    replayed_tokens = sum(token.replay_count > 0 for token in tokens)
    replay_executions = sum(
        attempt.attempt_id == 1 and attempt.start_time is not None
        for attempt in attempts
    )
    hedge_attempts = [attempt for attempt in attempts if attempt.attempt_id == 2]
    requested_hedges = sum(
        decision.requested is not ProtectionAction.NORMAL for decision in run.decisions
    )
    applied_hedges = sum(
        decision.applied is not ProtectionAction.NORMAL for decision in run.decisions
    )
    reservation_suppressed = sum(
        decision.reservation_suppressed for decision in run.decisions
    )
    reservation_admitted = sum(
        decision.reservation_admitted for decision in run.decisions
    )
    timer_executor_suppressed = sum(
        decision.timer_status == "executor_suppressed" for decision in run.decisions
    )
    work = {
        "executed_primary_work": math.fsum(
            attempt.executed_work for attempt in attempts if attempt.attempt_id == 0
        ),
        "executed_replay_work": math.fsum(
            attempt.executed_work for attempt in attempts if attempt.attempt_id == 1
        ),
        "executed_hedge_work": math.fsum(
            attempt.executed_work for attempt in attempts if attempt.attempt_id == 2
        ),
        "total_executed_work": math.fsum(attempt.executed_work for attempt in attempts),
        "wasted_work": math.fsum(
            attempt.executed_work
            for attempt in attempts
            if attempt.status is not HedgeAttemptStatus.COMPLETED_WINNER
        ),
    }
    work["waste_ratio"] = (
        work["wasted_work"] / work["total_executed_work"]
        if work["total_executed_work"]
        else None
    )
    storm = attribution["storm"]
    metrics = {
        "token_count": len(tokens),
        "latency": {
            "overall": latency_summary(token.latency for token in tokens),
            "arrival_phase": {
                phase: latency_summary(values) for phase, values in phase_values.items()
            },
        },
        "replay": {
            "replayed_token_count": replayed_tokens,
            "replayed_token_rate": replayed_tokens / len(tokens) if tokens else None,
            "replay_execution_count": replay_executions,
        },
        "hedge": {
            "requested_count": requested_hedges,
            "applied_count": applied_hedges,
            "launched_count": len(hedge_attempts),
            "executor_suppressed_count": simulation.hedge_suppressed,
            "timer_executor_suppressed_count": timer_executor_suppressed,
        },
        "work": work,
        "reservation": {
            "suppressed_count": reservation_suppressed,
            "admitted_count": reservation_admitted,
            "suppression_rate": reservation_suppressed / len(tokens) if tokens else None,
            "reserved_work": math.fsum(decision.charge for decision in run.decisions),
        },
        "storm": {
            "hedge_launch_count": storm["hedge_launch_count"],
            "peak_hedge_launch_rate": storm["peak_hedge_launch_rate"],
            "mean_hedge_launch_rate": storm["mean_hedge_launch_rate"],
            "peak_to_mean_hedge_launch_ratio": storm["peak_to_mean_hedge_launch_ratio"],
            "sustained_overload_duration": storm["sustained_overload_duration"],
            "overloaded_bin_count": storm["overloaded_bin_count"],
        },
        "invariant_count": sum(bool(value) for value in attribution["invariants"].values()),
        "invariant_total": len(attribution["invariants"]),
    }
    return metrics, phase_values


def _new_accumulator() -> dict[str, object]:
    return {
        "episodes": 0,
        "tokens": 0,
        "latencies": defaultdict(list),
        "replayed_tokens": 0,
        "replay_executions": 0,
        "requested_hedges": 0,
        "applied_hedges": 0,
        "launched_hedges": 0,
        "executor_suppressed": 0,
        "timer_executor_suppressed": 0,
        "work": defaultdict(float),
        "reservation_suppressed": 0,
        "reservation_admitted": 0,
        "reserved_work": 0.0,
        "storm_peak_rates": [],
        "storm_mean_rates": [],
        "storm_ratios": [],
        "storm_sustained": [],
        "storm_overloaded_bins": 0,
        "storm_hedge_launches": 0,
        "invariant_pass": 0,
        "invariant_total": 0,
    }


def _accumulate(acc: dict[str, object], metrics: Mapping[str, object], phase_values: Mapping[str, list[float]]) -> None:
    acc["episodes"] += 1
    acc["tokens"] += metrics["token_count"]
    for phase, values in phase_values.items():
        acc["latencies"][phase].extend(values)
    acc["replayed_tokens"] += metrics["replay"]["replayed_token_count"]
    acc["replay_executions"] += metrics["replay"]["replay_execution_count"]
    acc["requested_hedges"] += metrics["hedge"]["requested_count"]
    acc["applied_hedges"] += metrics["hedge"]["applied_count"]
    acc["launched_hedges"] += metrics["hedge"]["launched_count"]
    acc["executor_suppressed"] += metrics["hedge"]["executor_suppressed_count"]
    acc["timer_executor_suppressed"] += metrics["hedge"]["timer_executor_suppressed_count"]
    for name, value in metrics["work"].items():
        if value is not None:
            acc["work"][name] += value
    acc["reservation_suppressed"] += metrics["reservation"]["suppressed_count"]
    acc["reservation_admitted"] += metrics["reservation"]["admitted_count"]
    acc["reserved_work"] += metrics["reservation"]["reserved_work"]
    storm = metrics["storm"]
    acc["storm_peak_rates"].append(storm["peak_hedge_launch_rate"])
    acc["storm_mean_rates"].append(storm["mean_hedge_launch_rate"])
    if storm["peak_to_mean_hedge_launch_ratio"] is not None:
        acc["storm_ratios"].append(storm["peak_to_mean_hedge_launch_ratio"])
    acc["storm_sustained"].append(storm["sustained_overload_duration"])
    acc["storm_overloaded_bins"] += storm["overloaded_bin_count"]
    acc["storm_hedge_launches"] += storm["hedge_launch_count"]
    acc["invariant_pass"] += metrics["invariant_count"]
    acc["invariant_total"] += metrics["invariant_total"]


def _mean_or_none(values: Iterable[float]) -> float | None:
    data = tuple(values)
    return None if not data else math.fsum(data) / len(data)


def _finalize_accumulator(acc: Mapping[str, object]) -> dict[str, object]:
    episodes = acc["episodes"]
    tokens = acc["tokens"]
    latencies = acc["latencies"]
    total_work = acc["work"]["total_executed_work"]
    final = {
        "episode_count": episodes,
        "token_count": tokens,
        "latency": {
            "overall": latency_summary(
                value for values in latencies.values() for value in values
            ),
            "arrival_phase": {
                phase: latency_summary(values) for phase, values in latencies.items()
            },
        },
        "replay": {
            "replayed_token_count": acc["replayed_tokens"],
            "replayed_token_rate": acc["replayed_tokens"] / tokens if tokens else None,
            "replay_execution_count": acc["replay_executions"],
            "replay_execution_mean_per_episode": acc["replay_executions"] / episodes,
        },
        "hedge": {
            "requested_count": acc["requested_hedges"],
            "applied_count": acc["applied_hedges"],
            "launched_count": acc["launched_hedges"],
            "executor_suppressed_count": acc["executor_suppressed"],
            "timer_executor_suppressed_count": acc["timer_executor_suppressed"],
            "launched_mean_per_episode": acc["launched_hedges"] / episodes,
        },
        "work": {
            name: value for name, value in acc["work"].items()
        },
        "reservation": {
            "suppressed_count": acc["reservation_suppressed"],
            "admitted_count": acc["reservation_admitted"],
            "suppression_rate": acc["reservation_suppressed"] / tokens if tokens else None,
            "reserved_work": acc["reserved_work"],
        },
        "storm": {
            "hedge_launch_count": acc["storm_hedge_launches"],
            "peak_hedge_launch_rate_max": max(acc["storm_peak_rates"], default=None),
            "peak_hedge_launch_rate_mean": _mean_or_none(acc["storm_peak_rates"]),
            "mean_hedge_launch_rate_mean": _mean_or_none(acc["storm_mean_rates"]),
            "peak_to_mean_hedge_launch_ratio_max": max(acc["storm_ratios"], default=None),
            "sustained_overload_duration_max": max(acc["storm_sustained"], default=None),
            "overloaded_bin_count_total": acc["storm_overloaded_bins"],
            "overloaded_bin_count_mean": acc["storm_overloaded_bins"] / episodes,
        },
        "invariants": {
            "passed": acc["invariant_pass"],
            "checks": acc["invariant_total"],
            "all_passed": acc["invariant_pass"] == acc["invariant_total"],
        },
    }
    final["work"]["mean_total_executed_work_per_episode"] = total_work / episodes
    final["work"]["mean_wasted_work_per_episode"] = acc["work"]["wasted_work"] / episodes
    final["work"]["waste_ratio"] = (
        acc["work"]["wasted_work"] / total_work if total_work else None
    )
    return final


def _comparison(arms: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    a = arms["A"]
    b = arms["B"]

    def difference(left: object, right: object) -> float | None:
        if left is None or right is None:
            return None
        return float(right) - float(left)

    result = {}
    for phase in ("overall", "D", "F"):
        left = a["latency"]["overall"] if phase == "overall" else a["latency"]["arrival_phase"][phase]
        right = b["latency"]["overall"] if phase == "overall" else b["latency"]["arrival_phase"][phase]
        for statistic in ("mean", "p95", "p99"):
            result[f"{phase}_latency_{statistic}_B_minus_A"] = difference(
                left[statistic], right[statistic]
            )
    result.update(
        {
            "replayed_token_rate_B_minus_A": difference(
                a["replay"]["replayed_token_rate"], b["replay"]["replayed_token_rate"]
            ),
            "mean_hedge_launches_B_minus_A": difference(
                a["hedge"]["launched_mean_per_episode"], b["hedge"]["launched_mean_per_episode"]
            ),
            "mean_total_work_B_minus_A": difference(
                a["work"]["mean_total_executed_work_per_episode"],
                b["work"]["mean_total_executed_work_per_episode"],
            ),
            "mean_wasted_work_B_minus_A": difference(
                a["work"]["mean_wasted_work_per_episode"],
                b["work"]["mean_wasted_work_per_episode"],
            ),
            "reservation_suppression_rate_B_minus_A": difference(
                a["reservation"]["suppression_rate"], b["reservation"]["suppression_rate"]
            ),
            "storm_peak_rate_max_B_minus_A": difference(
                a["storm"]["peak_hedge_launch_rate_max"],
                b["storm"]["peak_hedge_launch_rate_max"],
            ),
            "storm_peak_to_mean_max_B_minus_A": difference(
                a["storm"]["peak_to_mean_hedge_launch_ratio_max"],
                b["storm"]["peak_to_mean_hedge_launch_ratio_max"],
            ),
            "storm_sustained_overload_max_B_minus_A": difference(
                a["storm"]["sustained_overload_duration_max"],
                b["storm"]["sustained_overload_duration_max"],
            ),
        }
    )
    return result


def _episode_factory(config: ExperimentConfig) -> Callable[[int], EpisodeTrace]:
    def create(index: int) -> EpisodeTrace:
        return generate_episode_trace(
            config,
            T3C_NAMESPACE,
            T3C_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )

    return create


def run_population_ab(
    config: ExperimentConfig,
    candidate: CandidateTable,
    *,
    episode_count: int = T3C_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    validate_frozen_config(config)
    if not isinstance(candidate, CandidateTable):
        raise T3CExecutionError("candidate must be CandidateTable")
    if type(episode_count) is not int or not 0 < episode_count <= T3C_EPISODES:
        raise T3CExecutionError("episode_count must be in 1..2048")
    if population_call_budget(episode_count) > T3C_CALL_LIMIT:
        raise T3CExecutionError("population A/B call budget exceeded")
    build = episode_builder or _episode_factory(config)
    source_b = ConditionalTokenCandidatePolicy(candidate)
    accumulators = {"A": _new_accumulator(), "B": _new_accumulator()}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    attempted_calls = 0
    for index in range(episode_count):
        episode = build(index)
        if not isinstance(episode, EpisodeTrace):
            raise T3CExecutionError("episode_builder returned a non-EpisodeTrace")
        trace_fp = _trace_fingerprint(episode)
        if trace_fp in seen:
            raise T3CExecutionError("duplicate A/B episode trace fingerprint")
        seen.add(trace_fp)
        before = _trace_fingerprint(episode)
        try:
            run_a = simulate_episode_online(
                episode,
                policy_factory(),
                _RESERVATION,
                degraded_slowdown=DEGRADED_SLOWDOWN,
                hedge_delay=HEDGE_DELAY,
            )
            attempted_calls += 1
            after_a = _trace_fingerprint(episode)
            if after_a != before:
                raise T3CExecutionError("A simulation mutated its immutable trace")
            run_b = simulate_episode_online(
                episode,
                source_b,
                _RESERVATION,
                degraded_slowdown=DEGRADED_SLOWDOWN,
                hedge_delay=HEDGE_DELAY,
            )
            attempted_calls += 1
            after_b = _trace_fingerprint(episode)
            if after_b != before:
                raise T3CExecutionError("B simulation mutated its immutable trace")
        except Exception as error:
            raise T3CExecutionError(
                f"A/B episode {index} failed after {attempted_calls} calls: {error}"
            ) from error
        metrics_a, phase_a = _episode_arm_metrics(run_a)
        metrics_b, phase_b = _episode_arm_metrics(run_b)
        _accumulate(accumulators["A"], metrics_a, phase_a)
        _accumulate(accumulators["B"], metrics_b, phase_b)
        rows.append(
            {
                "episode_index": index,
                "trace_fingerprint_a": before,
                "trace_fingerprint_b": after_b,
                "token_identity_fingerprint_a": _token_identity_fingerprint(run_a),
                "token_identity_fingerprint_b": _token_identity_fingerprint(run_b),
                "token_count": metrics_a["token_count"],
                "a_decision_count": len(run_a.decisions),
                "b_decision_count": len(run_b.decisions),
                "a": metrics_a,
                "b": metrics_b,
            }
        )
        if progress is not None and ((index + 1) % 64 == 0 or index + 1 == episode_count):
            progress(index + 1, episode_count)
    if attempted_calls != population_call_budget(episode_count):
        raise T3CExecutionError("A/B actual call count is not exactly two per episode")
    arms = {name: _finalize_accumulator(acc) for name, acc in accumulators.items()}
    return {
        "label": T3C_LABEL,
        "candidate_label": CANDIDATE_LABEL,
        "status": "completed",
        "namespace": T3C_NAMESPACE,
        "macro_seed": T3C_MACRO_SEED,
        "episode_count": episode_count,
        "attempted_calls": attempted_calls,
        "max_scheduler_calls": population_call_budget(episode_count),
        "protocol_call_limit": T3C_CALL_LIMIT,
        "candidate_parent_fingerprint": candidate.parent_fingerprint,
        "arms": arms,
        "comparison": _comparison(arms),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": rows,
    }


def write_population_ab_artifact(
    artifacts_root: str | Path,
    run_id: str,
    result: Mapping[str, object],
    candidate: CandidateTable,
    *,
    source_bundle: Mapping[str, object],
    config: ExperimentConfig,
) -> Path:
    if result.get("label") != T3C_LABEL or result.get("status") != "completed":
        raise T3CExecutionError("A/B result is not completed")
    if any(
        result.get(name) is not False
        for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg")
    ):
        raise T3CExecutionError("A/B claim boundary is invalid")
    if result.get("attempted_calls") != result.get("max_scheduler_calls"):
        raise T3CExecutionError("A/B call accounting is incomplete")
    provenance = {
        "environment_id": ENVIRONMENT_ID,
        "environment_fingerprint": environment_fingerprint(config),
        "population_policy_id": POPULATION_POLICY_ID,
        "population_policy_fingerprint": population_policy_fingerprint(),
        "target_selection_id": TARGET_SELECTION_ID,
        "namespace": T3C_NAMESPACE,
        "macro_seed": T3C_MACRO_SEED,
        "degraded_slowdown": DEGRADED_SLOWDOWN,
        "hedge_delay": HEDGE_DELAY,
        "storm_bin_width": STORM_BIN_WIDTH,
        "reservation": asdict(_RESERVATION),
        "candidate_parent_fingerprint": candidate.parent_fingerprint,
        "source_bundle_fingerprint": source_bundle["fingerprint"],
        "episode_count": result["episode_count"],
        "actual_scheduler_calls": result["attempted_calls"],
        "max_scheduler_calls": T3C_CALL_LIMIT,
        "deterministic_order": "episode_index_then_arm_A_then_arm_B",
    }
    manifest = {
        "schema_id": "token_t3c_population_ab_manifest_v1",
        "run_id": run_id,
        "label": T3C_LABEL,
        "source_bundle": dict(source_bundle),
        "parent_t3b_run_id": PARENT_T3B_RUN_ID,
        "parent_t3b_candidate_fingerprint": candidate.parent_fingerprint,
        "provenance": provenance,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    metric_contract = {
        "label": T3C_LABEL,
        "latency_phase_basis": "Token arrival phase",
        "latency_tail": ["mean", "p95", "p99"],
        "work_basis": "executed attempt work; wasted means non-winning executed work",
        "storm_basis": "existing attribution_metrics normalized storm contract, bin_width=1.0",
        "comparison": "descriptive delta_B_minus_A; no statistical or equilibrium claim",
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    summary["provenance"] = provenance
    candidate_payload = candidate.to_dict()
    candidate_payload.update(
        {
            "evidence_label": CANDIDATE_LABEL,
            "parent_t3b_run_id": PARENT_T3B_RUN_ID,
            "claims_best_response": False,
            "claims_regret": False,
            "claims_nash": False,
            "claims_mfg": False,
        }
    )
    return write_run_directory(
        artifacts_root,
        run_id,
        {
            "manifest.json": manifest,
            "candidate_policy.json": candidate_payload,
            "metric_contract.json": metric_contract,
            "summary.json": summary,
            "episode_rows.json": {"rows": result["episode_rows"]},
        },
    )


def execute_population_ab(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    validation_dir: str | Path,
    occupancy_dir: str | Path,
    episode_count: int = T3C_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    candidate = load_t3a_runtime_candidate(validation_dir, occupancy_dir)
    result = run_population_ab(
        config,
        candidate,
        episode_count=episode_count,
        progress=progress,
    )
    directory = write_population_ab_artifact(
        artifacts_root,
        run_id,
        result,
        candidate,
        source_bundle=asdict(bundle),
        config=config,
    )
    return directory, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the paired Token population A/B diagnostic")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--occupancy-dir", required=True)
    args = parser.parse_args(argv)
    try:
        directory, result = execute_population_ab(
            args.project_root,
            args.artifacts_root,
            args.run_id,
            validation_dir=args.validation_dir,
            occupancy_dir=args.occupancy_dir,
            progress=lambda done, total: print(f"A/B progress: {done}/{total}", flush=True),
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"artifact: {directory}")
    print(f"status: {result['status']}")
    print(f"calls: {result['attempted_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "T3CExecutionError",
    "T3C_LABEL",
    "T3C_CALL_LIMIT",
    "T3C_EPISODES",
    "T3C_MACRO_SEED",
    "T3C_NAMESPACE",
    "execute_population_ab",
    "latency_summary",
    "population_call_budget",
    "run_population_ab",
    "write_population_ab_artifact",
]
