"""Frozen paired 2x2 cancellation-semantics ablation.

The panel is a descriptive mechanism experiment.  It compares the existing
one-shot NIIN policy and the existing idle-release LÆDGE scheduler under the
two explicit running-loser semantics from ADR-0032.  It does not select a
policy or make equilibrium claims.
"""

from __future__ import annotations

from dataclasses import asdict
from array import array
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
    EpisodeSimulationResult,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
)
from .attribution_metrics import (
    SLODeadlines,
    _latency_views,
    build_episode_metrics,
    empirical_cvar95,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction
from .metrics import percentile
from .protection_cancellation_ablation import (
    CANCELLATION_ARMS,
    CancellationAblationArm,
    PolicyFamily,
    simulate_cancellation_arm,
)
from .token_online import OnlineEpisodeResult
from .token_t3a_execution import (
    _RESERVATION,
    build_source_bundle,
    environment_fingerprint,
    validate_frozen_config,
)
from .token_three_arm_evaluation import FrozenNIINPolicy


CANCELLATION_LABEL = "finite_cancellation_semantics_ablation_v1"
CANCELLATION_NAMESPACE = "replica-routing-baselines:cancellation-ablation:v1"
CANCELLATION_MACRO_SEED = 20260931
CANCELLATION_EPISODES = 1024
CANCELLATION_ARM_COUNT = 4
CANCELLATION_CALL_LIMIT = CANCELLATION_EPISODES * CANCELLATION_ARM_COUNT
CANCELLATION_PREFLIGHT_EPISODES = 2
CANCELLATION_BOOTSTRAP_NAMESPACE = (
    "replica-routing-baselines:cancellation-ablation:v1:paired-bootstrap"
)
CANCELLATION_BOOTSTRAP_SEED = 20261001
CANCELLATION_BOOTSTRAP_REPLICATES = 4096
DEGRADED_SLOWDOWN = 2.0
NIIN_HEDGE_DELAY = 2.0
STORM_BIN_WIDTH = 1.0
_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)
_METRICS = (
    "overall_mean",
    "overall_p95",
    "overall_p99",
    "df_cvar95",
    "df_miss_rate",
    "hr_p99",
    "replay_rate",
    "total_work",
    "wasted_work",
    "hedge_launches",
    "hedge_winner_rate",
    "storm_peak",
    "drain_duration",
)


class CancellationExperimentError(RuntimeError):
    """A frozen cancellation panel or artifact contract violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _trace_digest(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CancellationExperimentError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise CancellationExperimentError(f"{name} must be finite")
    return result


def _bootstrap_index_library(sample_count: int) -> array:
    if type(sample_count) is not int or sample_count <= 0:
        raise CancellationExperimentError("bootstrap sample_count must be positive")
    seed_bytes = hashlib.sha256(
        f"{CANCELLATION_BOOTSTRAP_NAMESPACE}:{CANCELLATION_BOOTSTRAP_SEED}:"
        f"{sample_count}:{CANCELLATION_BOOTSTRAP_REPLICATES}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    indices = array("I")
    for _ in range(CANCELLATION_BOOTSTRAP_REPLICATES * sample_count):
        indices.append(rng.randrange(sample_count))
    return indices


def _bootstrap_fingerprint(indices: array) -> str:
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _paired_entry(
    left: Sequence[float],
    right: Sequence[float],
    indices: array,
    *,
    relative_baseline: Sequence[float] | None = None,
) -> dict[str, object]:
    if len(left) != len(right) or not left:
        raise CancellationExperimentError("paired inputs must have equal positive length")
    values_left = tuple(_finite(value, "left value") for value in left)
    values_right = tuple(_finite(value, "right value") for value in right)
    differences = tuple(
        right_value - left_value
        for left_value, right_value in zip(values_left, values_right)
    )
    sample_count = len(differences)
    if len(indices) != sample_count * CANCELLATION_BOOTSTRAP_REPLICATES:
        raise CancellationExperimentError("bootstrap index library has wrong shape")
    point = math.fsum(differences) / sample_count
    if sample_count > 1:
        paired_se = math.sqrt(
            math.fsum((value - point) ** 2 for value in differences)
            / (sample_count - 1)
        ) / math.sqrt(sample_count)
    else:
        paired_se = 0.0
    bootstrap_means = []
    for start in range(0, len(indices), sample_count):
        bootstrap_means.append(
            math.fsum(
                differences[indices[start + offset]]
                for offset in range(sample_count)
            )
            / sample_count
        )
    baseline = (
        math.fsum(values_left) / sample_count
        if relative_baseline is None
        else math.fsum(_finite(value, "relative baseline") for value in relative_baseline)
        / sample_count
    )
    return {
        "point": point,
        "absolute_difference": point,
        "relative_change": point / abs(baseline) if baseline else None,
        "paired_se": paired_se,
        "ci95": [percentile(bootstrap_means, 2.5), percentile(bootstrap_means, 97.5)],
        "sample_count": sample_count,
        "cluster_unit": "episode_index",
    }


def _phase_tokens(run: EpisodeSimulationResult, phases: tuple[Phase, ...]):
    return tuple(
        token
        for token in run.simulation.tokens
        if phase_at(run.episode.protocol.timeline, token.arrival_time) in phases
    )


def _latency_summary(tokens, deadlines: SLODeadlines) -> dict[str, float]:
    if not tokens:
        raise CancellationExperimentError("required latency cohort is empty")
    values = tuple(float(token.latency) for token in tokens)
    return {
        "mean": math.fsum(values) / len(values),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "cvar95": empirical_cvar95(values),
        "miss_rate": math.fsum(
            token.latency > deadlines.for_class(token.token_class)
            for token in tokens
        )
        / len(tokens),
    }


def _episode_outcomes(
    run: EpisodeSimulationResult,
    attribution: Mapping[str, object],
    policy_family: PolicyFamily,
) -> dict[str, object]:
    deadlines = SLODeadlines()
    tokens = tuple(run.simulation.tokens)
    if not tokens:
        raise CancellationExperimentError("episode completed without Tokens")
    overall = _latency_summary(tokens, deadlines)
    df = _latency_summary(
        _phase_tokens(run, (Phase.DEGRADED, Phase.FAILED)), deadlines
    )
    hr = _latency_summary(
        _phase_tokens(run, (Phase.HEALTHY, Phase.RECOVERED)), deadlines
    )
    hedge = attribution["hedge"]
    work = attribution["work"]
    storm = attribution["storm"]
    result = run.simulation
    invariants = attribution["invariants"]
    excluded = {
        "dispatcher_assignment_matches_protocol",
        "copy_placement_matches_protocol",
        "boundary_queue_counts_match_legacy_result",
    }
    checked_invariants = {
        name: value
        for name, value in invariants.items()
        if not (policy_family is PolicyFamily.LAEDGE and name in excluded)
    }
    if not all(checked_invariants.values()):
        raise CancellationExperimentError("episode invariant failure")
    return {
        "overall_mean": overall["mean"],
        "overall_p95": overall["p95"],
        "overall_p99": overall["p99"],
        "df_cvar95": df["cvar95"],
        "df_miss_rate": df["miss_rate"],
        "hr_p99": hr["p99"],
        "replay_rate": sum(token.replay_count > 0 for token in tokens) / len(tokens),
        "total_work": work["total_executed_work"],
        "wasted_work": work["wasted_work"],
        "hedge_launches": hedge["launched"],
        "hedge_winners": hedge["winners"],
        "hedge_winner_rate": hedge["win_per_launch"],
        "storm_peak": storm["peak_hedge_launch_rate"],
        "drain_duration": result.drain_duration,
        "post_cutoff_drain_duration": run.post_cutoff_drain_duration,
        "completed_tokens": result.completed_tokens,
        "failed_running_primary_executions": result.failed_running_primary_executions,
        "invalidated_queued_primary_executions": result.invalidated_queued_primary_executions,
        "stale_completion_events_ignored": result.stale_completion_events_ignored,
        "invariants": dict(checked_invariants),
        "invariant_exclusions": sorted(
            excluded if policy_family is PolicyFamily.LAEDGE else ()
        ),
    }


def _decision_signature(run: OnlineEpisodeResult) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            decision.token_id,
            decision.requested.value,
            decision.applied.value,
            decision.window_start,
            decision.window_end,
            decision.cap,
            decision.balance_before,
            decision.charge,
            decision.balance_after,
            decision.reservation_suppressed,
            decision.reservation_admitted,
        )
        for decision in run.decisions
    )


def _run_arm(
    episode: EpisodeTrace,
    arm: CancellationAblationArm,
) -> OnlineEpisodeResult | EpisodeSimulationResult:
    if arm.policy_family is PolicyFamily.NIIN:
        return simulate_cancellation_arm(
            episode,
            arm,
            action_source=FrozenNIINPolicy(),
            reservation_parameters=_RESERVATION,
            degraded_slowdown=DEGRADED_SLOWDOWN,
            hedge_delay=NIIN_HEDGE_DELAY,
            public_price=0.0,
        )
    return simulate_cancellation_arm(
        episode,
        arm,
        degraded_slowdown=DEGRADED_SLOWDOWN,
    )


def _simulation_from_run(
    run: OnlineEpisodeResult | EpisodeSimulationResult,
) -> EpisodeSimulationResult:
    return run.simulation if isinstance(run, OnlineEpisodeResult) else run


def _compact_arm_record(
    run: OnlineEpisodeResult | EpisodeSimulationResult,
    trace_fingerprint: str,
    outcomes: Mapping[str, object],
    attribution: Mapping[str, object],
    policy_family: PolicyFamily,
) -> dict[str, object]:
    simulation = _simulation_from_run(run).simulation
    return {
        "status": "completed",
        "trace_fingerprint": trace_fingerprint,
        "outcomes": dict(outcomes),
        "counters": {
            "completed_tokens": simulation.completed_tokens,
            "primary_executions": simulation.primary_executions,
            "replay_executions": simulation.replay_executions,
            "hedge_requested": simulation.hedge_requested,
            "hedge_launches": simulation.hedge_launches,
            "hedge_suppressed": simulation.hedge_suppressed,
            "hedge_timers_voided": simulation.hedge_timers_voided,
            "cancelled_queued_total": simulation.cancelled_queued_total,
            "completed_loser_total": simulation.completed_loser_total,
            "failed_running_primary_executions": simulation.failed_running_primary_executions,
            "invalidated_queued_primary_executions": simulation.invalidated_queued_primary_executions,
            "stale_completion_events_ignored": simulation.stale_completion_events_ignored,
            "all_invariants_pass": all(outcomes["invariants"].values()),
            "invariant_exclusions": outcomes["invariant_exclusions"],
        },
    }


def _aggregate_arm(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "episode_count": 0,
            "token_count": 0,
            "status": "unavailable",
        }
    outcomes = [record["outcomes"] for record in records]
    tokens = tuple(
        token
        for record in records
        for token in record["run"].simulation.tokens
    )
    first_run = records[0]["run"]
    timeline = first_run.episode.protocol.timeline
    deadlines = SLODeadlines()
    overall = _latency_summary(tokens, deadlines)
    df = _latency_summary(
        tuple(
            token
            for record in records
            for token in record["run"].simulation.tokens
            if phase_at(timeline, token.arrival_time)
            in (Phase.DEGRADED, Phase.FAILED)
        ),
        deadlines,
    )
    hr = _latency_summary(
        tuple(
            token
            for record in records
            for token in record["run"].simulation.tokens
            if phase_at(timeline, token.arrival_time)
            in (Phase.HEALTHY, Phase.RECOVERED)
        ),
        deadlines,
    )
    scalar_means = {
        name: math.fsum(float(outcome[name]) for outcome in outcomes) / len(outcomes)
        for name in _METRICS
    }
    counters = {}
    for name in (
        "completed_tokens",
        "primary_executions",
        "replay_executions",
        "hedge_requested",
        "hedge_launches",
        "hedge_suppressed",
        "hedge_timers_voided",
        "cancelled_queued_total",
        "completed_loser_total",
        "failed_running_primary_executions",
        "invalidated_queued_primary_executions",
        "stale_completion_events_ignored",
    ):
        counters[name] = sum(record["counters"][name] for record in records)
    counters["all_invariants_pass"] = all(
        record["counters"]["all_invariants_pass"] for record in records
    )
    return {
        "episode_count": len(records),
        "token_count": len(tokens),
        "latency": {
            "overall": overall,
            "df": df,
            "hr": hr,
        },
        "common_metrics": scalar_means,
        "work": {
            "total_executed_work": math.fsum(
                outcome["total_work"] for outcome in outcomes
            ),
            "wasted_work": math.fsum(
                outcome["wasted_work"] for outcome in outcomes
            ),
        },
        "hedge": {
            "launches": counters["hedge_launches"],
            "winners": sum(outcome["hedge_winners"] for outcome in outcomes),
            "winner_rate": (
                sum(outcome["hedge_winners"] for outcome in outcomes)
                / counters["hedge_launches"]
                if counters["hedge_launches"]
                else None
            ),
        },
        "storm": {
            "peak_max": max(outcome["storm_peak"] for outcome in outcomes),
            "peak_mean": math.fsum(outcome["storm_peak"] for outcome in outcomes)
            / len(outcomes),
        },
        "drain": {
            "mean": math.fsum(outcome["drain_duration"] for outcome in outcomes)
            / len(outcomes),
            "max": max(outcome["drain_duration"] for outcome in outcomes),
        },
        "counters": counters,
    }


def _comparison(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    left_name: str,
    right_name: str,
    indices: array,
    *,
    relative_baseline_name: str | None = None,
) -> dict[str, object]:
    left = records[left_name]
    right = records[right_name]
    if len(left) != len(right) or not left:
        raise CancellationExperimentError("paired comparisons require complete rows")
    baseline = records[relative_baseline_name] if relative_baseline_name else left
    return {
        "left_arm": left_name,
        "right_arm": right_name,
        "orientation": "right_minus_left",
        "cluster_unit": "episode_index",
        **{
            name: _paired_entry(
                [item["outcomes"][name] for item in left],
                [item["outcomes"][name] for item in right],
                indices,
                relative_baseline=[item["outcomes"][name] for item in baseline],
            )
            for name in _METRICS
        },
        "metrics": {
            name: _paired_entry(
                [item["outcomes"][name] for item in left],
                [item["outcomes"][name] for item in right],
                indices,
                relative_baseline=[item["outcomes"][name] for item in baseline],
            )
            for name in _METRICS
        },
    }


def _difference_in_differences(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    indices: array,
) -> dict[str, object]:
    niin_delta = records["niin:preemptive"]
    niin_base = records["niin:conservative"]
    laedge_delta = records["laedge:preemptive"]
    laedge_base = records["laedge:conservative"]
    if not (len(niin_delta) == len(niin_base) == len(laedge_delta) == len(laedge_base)):
        raise CancellationExperimentError("difference-in-differences rows are incomplete")
    metrics = {}
    for name in _METRICS:
        left = [
            right_item["outcomes"][name] - left_item["outcomes"][name]
            for left_item, right_item in zip(niin_base, niin_delta)
        ]
        right = [
            right_item["outcomes"][name] - left_item["outcomes"][name]
            for left_item, right_item in zip(laedge_base, laedge_delta)
        ]
        metrics[name] = _paired_entry(
            left,
            right,
            indices,
            relative_baseline=[item["outcomes"][name] for item in niin_base],
        )
    return {
        "left_effect": "niin:preemptive-minus-niin:conservative",
        "right_effect": "laedge:preemptive-minus-laedge:conservative",
        "orientation": "right_effect_minus_left_effect",
        "cluster_unit": "episode_index",
        **metrics,
        "metrics": metrics,
    }


def _default_episode_builder(config: ExperimentConfig) -> Callable[[int], EpisodeTrace]:
    def build(index: int) -> EpisodeTrace:
        return generate_episode_trace(
            config,
            CANCELLATION_NAMESPACE,
            CANCELLATION_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )

    return build


def run_cancellation_ablation(
    config: ExperimentConfig,
    *,
    episode_count: int = CANCELLATION_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the four arms on paired immutable episodes in memory."""

    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= CANCELLATION_EPISODES:
        raise CancellationExperimentError("episode_count must be in 1..1024")
    calls_limit = episode_count * CANCELLATION_ARM_COUNT
    if calls_limit > CANCELLATION_CALL_LIMIT:
        raise CancellationExperimentError("cancellation call budget exceeded")
    build = episode_builder or _default_episode_builder(config)
    rows: list[dict[str, object]] = []
    records: dict[str, list[dict[str, object]]] = {
        arm.key: [] for arm in CANCELLATION_ARMS
    }
    seen: set[str] = set()
    attempted_calls = 0
    failed_episode_count = 0
    for index in range(episode_count):
        episode = build(index)
        if not isinstance(episode, EpisodeTrace):
            raise CancellationExperimentError("episode_builder returned a non-EpisodeTrace")
        trace_fingerprint = _trace_digest(episode)
        if trace_fingerprint in seen:
            raise CancellationExperimentError("duplicate episode trace fingerprint")
        seen.add(trace_fingerprint)
        row: dict[str, object] = {
            "episode_index": index,
            "episode_seed": episode.episode_seed,
            "trace_fingerprint": trace_fingerprint,
            "status": "completed",
            "arms": {},
            "mechanism_verification": {},
        }
        completed: dict[str, dict[str, object]] = {}
        failed = False
        for arm in CANCELLATION_ARMS:
            attempted_calls += 1
            try:
                run = _run_arm(episode, arm)
                if _trace_digest(episode) != trace_fingerprint:
                    raise CancellationExperimentError(
                        f"arm {arm.key} mutated the immutable episode"
                    )
                simulation = _simulation_from_run(run)
                attribution = build_episode_metrics(
                    simulation,
                    DEGRADED_SLOWDOWN,
                    storm_bin_width=STORM_BIN_WIDTH,
                    placement_mode=(
                        "idle_release"
                        if arm.policy_family is PolicyFamily.LAEDGE
                        else "fixed_dispatcher"
                    ),
                )
                outcomes = _episode_outcomes(
                    simulation,
                    attribution,
                    arm.policy_family,
                )
                compact = _compact_arm_record(
                    run,
                    trace_fingerprint,
                    outcomes,
                    attribution,
                    arm.policy_family,
                )
                compact["run"] = simulation
                compact["wrapper"] = run
                completed[arm.key] = compact
                row["arms"][arm.key] = {
                    key: value
                    for key, value in compact.items()
                    if key not in {"run", "wrapper"}
                }
            except Exception as error:
                failed = True
                row["status"] = "failed"
                row["arms"][arm.key] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        if failed:
            failed_episode_count += 1
        else:
            for arm in CANCELLATION_ARMS:
                records[arm.key].append(completed[arm.key])
            niin_con = completed["niin:conservative"]["wrapper"]
            niin_pre = completed["niin:preemptive"]["wrapper"]
            row["mechanism_verification"] = {
                "crn_equal": len(
                    {
                        item["trace_fingerprint"]
                        for item in completed.values()
                    }
                )
                == 1,
                "niin_admission_equal": _decision_signature(niin_con)
                == _decision_signature(niin_pre),
                "queued_loser_semantics": "cancelled_queued_zero_work",
                "failure_first_semantics": "fault_before_completion_and_timer",
                "replay_semantics": "at_most_one_replay_with_complete_drain",
                "dispatch_semantics": "unchanged_policy_family_dispatcher",
                "mode_difference": "running_loser_cancellation_only",
            }
        rows.append(row)
        if progress is not None:
            progress(index + 1, episode_count)
    complete_count = episode_count - failed_episode_count
    if attempted_calls != calls_limit or attempted_calls > CANCELLATION_CALL_LIMIT:
        raise CancellationExperimentError("cancellation call accounting failed")
    status = "completed" if failed_episode_count == 0 else "physical_failed"
    result: dict[str, object] = {
        "label": CANCELLATION_LABEL,
        "status": status,
        "namespace": CANCELLATION_NAMESPACE,
        "macro_seed": CANCELLATION_MACRO_SEED,
        "episode_count": episode_count,
        "complete_episode_count": complete_count,
        "failed_episode_count": failed_episode_count,
        "attempted_calls": attempted_calls,
        "formal_call_limit": CANCELLATION_CALL_LIMIT,
        "arms": {
            name: _aggregate_arm(records[name])
            for name in records
            if records[name]
        },
        "episode_rows": rows,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    if status != "completed":
        result["paired_statistics"] = {
            "status": "unavailable",
            "reason": "physical_failed",
            "complete_episode_count": complete_count,
        }
        return result
    indices = _bootstrap_index_library(complete_count)
    comparisons = {
        "niin:preemptive-minus-niin:conservative": _comparison(
            records,
            "niin:conservative",
            "niin:preemptive",
            indices,
        ),
        "laedge:preemptive-minus-laedge:conservative": _comparison(
            records,
            "laedge:conservative",
            "laedge:preemptive",
            indices,
        ),
        "laedge:conservative-minus-niin:conservative": _comparison(
            records,
            "niin:conservative",
            "laedge:conservative",
            indices,
        ),
        "laedge:preemptive-minus-niin:preemptive": _comparison(
            records,
            "niin:preemptive",
            "laedge:preemptive",
            indices,
        ),
        "cancellation_x_policy_family_did": _difference_in_differences(
            records,
            indices,
        ),
    }
    result["paired_statistics"] = {
        "status": "completed",
        "cluster_unit": "episode_index",
        "bootstrap": {
            "namespace": CANCELLATION_BOOTSTRAP_NAMESPACE,
            "seed": CANCELLATION_BOOTSTRAP_SEED,
            "replicates": CANCELLATION_BOOTSTRAP_REPLICATES,
            "index_library_fingerprint": _bootstrap_fingerprint(indices),
            "scheduler_calls": 0,
        },
        "comparisons": comparisons,
    }
    return result


def run_cancellation_timing_preflight(
    config: ExperimentConfig,
    *,
    episode_count: int = CANCELLATION_PREFLIGHT_EPISODES,
) -> dict[str, object]:
    """Run a small fixed-size timing probe without changing formal constants."""

    if type(episode_count) is not int or not 0 < episode_count <= CANCELLATION_PREFLIGHT_EPISODES:
        raise CancellationExperimentError("preflight episode_count must be in 1..2")
    started = time.perf_counter()
    result = run_cancellation_ablation(config, episode_count=episode_count)
    elapsed = time.perf_counter() - started
    return {
        "status": result["status"],
        "episode_count": episode_count,
        "scheduler_calls": result["attempted_calls"],
        "elapsed_seconds": elapsed,
        "calls_per_second": (
            result["attempted_calls"] / elapsed if elapsed > 0.0 else None
        ),
        "formal_episode_count": CANCELLATION_EPISODES,
        "formal_call_limit": CANCELLATION_CALL_LIMIT,
    }


def _write_transactional_artifact(
    artifacts_root: str | Path,
    run_id: str,
    files: Mapping[str, object],
) -> Path:
    _validate_run_id(run_id)
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = (root / run_id).resolve()
    if final.parent != root:
        raise ValueError(f"run directory {final} escapes artifacts root {root}")
    if final.exists():
        raise FileExistsError(f"run directory already exists: {final}")
    staging = root / f".{run_id}.staging"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        for name, payload in files.items():
            if name == "episode_rows.jsonl":
                if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
                    raise ValueError("episode_rows.jsonl payload must be a sequence")
                text = "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in payload
                )
            elif name.endswith(".json") and "/" not in name and "\\" not in name:
                text = json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
            else:
                raise ValueError(f"invalid cancellation artifact file name: {name!r}")
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def write_cancellation_artifact(
    artifacts_root: str | Path,
    run_id: str,
    result: Mapping[str, object],
    config: ExperimentConfig,
    *,
    source_bundle: Mapping[str, object] | None = None,
) -> Path:
    if result.get("label") != CANCELLATION_LABEL:
        raise CancellationExperimentError("unexpected cancellation experiment label")
    if int(result.get("attempted_calls", -1)) > CANCELLATION_CALL_LIMIT:
        raise CancellationExperimentError("artifact call count exceeds frozen limit")
    for claim in (
        "claims_best_response",
        "claims_optimality",
        "claims_nash",
        "claims_mfg",
    ):
        if result.get(claim) is not False:
            raise CancellationExperimentError(f"{claim} must be false")
    validate_frozen_config(config)
    protocol = {
        "schema_id": "cancellation_semantics_ablation_protocol_v1",
        "adr": "ADR-0032",
        "label": CANCELLATION_LABEL,
        "namespace": CANCELLATION_NAMESPACE,
        "macro_seed": CANCELLATION_MACRO_SEED,
        "episode_count": CANCELLATION_EPISODES,
        "scheduler_call_limit": CANCELLATION_CALL_LIMIT,
        "arms": [arm.key for arm in CANCELLATION_ARMS],
        "degraded_slowdown": DEGRADED_SLOWDOWN,
        "hedge_delay": NIIN_HEDGE_DELAY,
        "reservation": "existing _RESERVATION unchanged",
        "niin_policy": "FrozenNIINPolicy unchanged",
        "laedge_policy": "existing idle-release dispatcher unchanged",
        "common_random_numbers": (
            "arrival,class,attempt0,attempt1,attempt2,common fault timeline"
        ),
        "bootstrap": {
            "namespace": CANCELLATION_BOOTSTRAP_NAMESPACE,
            "seed": CANCELLATION_BOOTSTRAP_SEED,
            "replicates": CANCELLATION_BOOTSTRAP_REPLICATES,
            "cluster_unit": "episode_index",
        },
        "preemptive_semantics": "oracle_upper_bound_if_runtime_cannot_cancel",
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    provenance = {
        "environment_fingerprint": environment_fingerprint(config),
        "source_bundle": dict(source_bundle or {}),
        "protocol_fingerprint": _digest(protocol),
    }
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    summary["provenance"] = provenance
    manifest = {
        "schema_id": "cancellation_semantics_ablation_manifest_v1",
        "run_id": run_id,
        "label": CANCELLATION_LABEL,
        "status": result.get("status"),
        "provenance": provenance,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    paired = result.get("paired_statistics") or {"status": "unavailable"}
    return _write_transactional_artifact(
        artifacts_root,
        run_id,
        {
            "summary.json": summary,
            "paired_statistics.json": paired,
            "episode_rows.jsonl": result.get("episode_rows", []),
            "manifest.json": manifest,
            "protocol.json": protocol,
        },
    )


def execute_cancellation_ablation(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    preflight = run_cancellation_timing_preflight(config)
    if preflight["status"] != "completed":
        raise CancellationExperimentError("timing preflight failed closed")
    result = run_cancellation_ablation(config, progress=progress)
    result["timing_preflight"] = preflight
    directory = write_cancellation_artifact(
        artifacts_root,
        run_id,
        result,
        config,
        source_bundle=asdict(build_source_bundle(project)),
    )
    return directory, result


__all__ = [
    "CANCELLATION_ARMS",
    "CANCELLATION_BOOTSTRAP_REPLICATES",
    "CANCELLATION_CALL_LIMIT",
    "CANCELLATION_EPISODES",
    "CANCELLATION_LABEL",
    "CANCELLATION_MACRO_SEED",
    "CANCELLATION_NAMESPACE",
    "CancellationExperimentError",
    "execute_cancellation_ablation",
    "run_cancellation_ablation",
    "run_cancellation_timing_preflight",
    "write_cancellation_artifact",
]
