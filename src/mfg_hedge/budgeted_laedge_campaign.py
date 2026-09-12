"""Development calibration and guarded holdout execution for Budgeted LÆDGE.

The module is intentionally separate from the Budgeted LÆDGE physics kernel.
It freezes a rate grid, estimates a monotone work curve, confirms the chosen
rates on a disjoint development library, and refuses holdout execution until
that mapping is complete.  It does not define prices, best responses, Nash,
or MFG claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence

from .artifacts import _validate_run_id
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeSimulationResult,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
    simulate_episode,
)
from .attribution_metrics import SLODeadlines, build_episode_metrics, empirical_cvar95
from .budgeted_laedge import (
    BUDGETED_LAEDGE_ARM_KEYS,
    simulate_budgeted_laedge_episode,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .hedge_simulation import HedgeAttemptStatus
from .laedge_episode import simulate_laedge_episode
from .metrics import percentile
from .token_online import simulate_episode_online
from .token_three_arm_evaluation import FrozenNIINPolicy
from .token_t3a_execution import _RESERVATION, validate_frozen_config


BUDGETED_LAEDGE_CALIBRATION_EPISODES = 64
BUDGETED_LAEDGE_CONFIRMATION_EPISODES = 192
BUDGETED_LAEDGE_REFERENCE_EPISODES = 256
BUDGETED_LAEDGE_HOLDOUT_EPISODES = 1024
BUDGETED_LAEDGE_CALIBRATION_RATES = (
    0.0,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.12,
    0.18,
    0.25,
    0.35,
    0.50,
    1.00,
)
BUDGETED_LAEDGE_V2_CALIBRATION_RATES = (
    0.0,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.12,
    0.18,
    0.25,
    0.35,
    0.50,
    4.00,
)
BUDGETED_LAEDGE_DELTA_TARGETS = (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18)
BUDGETED_LAEDGE_DEVELOPMENT_MACRO_SEED = 20260910
BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v1:development"
)
BUDGETED_LAEDGE_CALIBRATION_NAMESPACE = BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE + ":calibration"
BUDGETED_LAEDGE_CONFIRMATION_NAMESPACE = BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE + ":confirmation"
BUDGETED_LAEDGE_NIIN_REFERENCE_NAMESPACE = BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE + ":niin-reference"
BUDGETED_LAEDGE_UNCONSTRAINED_REFERENCE_NAMESPACE = (
    BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE + ":unconstrained-reference"
)
BUDGETED_LAEDGE_HOLDOUT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v1:holdout"
)
BUDGETED_LAEDGE_HOLDOUT_MACRO_SEED = 20260911
BUDGETED_LAEDGE_V2_DEVELOPMENT_MACRO_SEED = 20260912
BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v2:development"
)
BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE = (
    "replica-routing-baselines:budgeted-laedge:v2:holdout"
)
BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED = 20260913
BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET = 2816
BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET = 11264
BUDGETED_LAEDGE_CONFIRMATION_OVERFILL_TOLERANCE = 0.0025
BUDGETED_LAEDGE_TARGET_UNDERFILL_TOLERANCE = 0.01
BUDGETED_LAEDGE_UPPER_COVERAGE_TOLERANCE = 0.0025
BUDGETED_LAEDGE_LABEL = "finite_budgeted_laedge_development_calibration_v1"
BUDGETED_LAEDGE_HOLDOUT_LABEL = "finite_budgeted_laedge_holdout_pareto_v1"
BUDGETED_LAEDGE_V2_LABEL = "finite_budgeted_laedge_development_calibration_v2"
BUDGETED_LAEDGE_V2_HOLDOUT_LABEL = "finite_budgeted_laedge_holdout_pareto_v2"
BUDGETED_LAEDGE_DEGRADED_SLOWDOWN = 2.0
BUDGETED_LAEDGE_HEDGE_DELAY = 2.0
BUDGETED_LAEDGE_STORM_BIN_WIDTH = 1.0
_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)


@dataclass(frozen=True)
class _BudgetedLaedgeProtocol:
    label: str
    holdout_label: str
    development_namespace: str
    holdout_namespace: str
    development_macro_seed: int
    holdout_macro_seed: int
    calibration_rates: tuple[float, ...]
    schema_id: str

    @property
    def calibration_namespace(self) -> str:
        return self.development_namespace + ":calibration"

    @property
    def confirmation_namespace(self) -> str:
        return self.development_namespace + ":confirmation"

    @property
    def niin_reference_namespace(self) -> str:
        return self.development_namespace + ":niin-reference"

    @property
    def unconstrained_reference_namespace(self) -> str:
        return self.development_namespace + ":unconstrained-reference"


_V1_PROTOCOL = _BudgetedLaedgeProtocol(
    label=BUDGETED_LAEDGE_LABEL,
    holdout_label=BUDGETED_LAEDGE_HOLDOUT_LABEL,
    development_namespace=BUDGETED_LAEDGE_DEVELOPMENT_NAMESPACE,
    holdout_namespace=BUDGETED_LAEDGE_HOLDOUT_NAMESPACE,
    development_macro_seed=BUDGETED_LAEDGE_DEVELOPMENT_MACRO_SEED,
    holdout_macro_seed=BUDGETED_LAEDGE_HOLDOUT_MACRO_SEED,
    calibration_rates=BUDGETED_LAEDGE_CALIBRATION_RATES,
    schema_id="budgeted_laedge_calibration_protocol_v1",
)
_V2_PROTOCOL = _BudgetedLaedgeProtocol(
    label=BUDGETED_LAEDGE_V2_LABEL,
    holdout_label=BUDGETED_LAEDGE_V2_HOLDOUT_LABEL,
    development_namespace=BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE,
    holdout_namespace=BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE,
    development_macro_seed=BUDGETED_LAEDGE_V2_DEVELOPMENT_MACRO_SEED,
    holdout_macro_seed=BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED,
    calibration_rates=BUDGETED_LAEDGE_V2_CALIBRATION_RATES,
    schema_id="budgeted_laedge_calibration_protocol_v2",
)


class BudgetedLaedgeCampaignError(RuntimeError):
    """A frozen Budgeted LÆDGE campaign or physical execution violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetedLaedgeCampaignError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise BudgetedLaedgeCampaignError(f"{name} must be finite")
    return result


def development_call_plan(
    calibration_rates: Sequence[float] = BUDGETED_LAEDGE_CALIBRATION_RATES,
) -> dict[str, int]:
    return {
        "calibration": BUDGETED_LAEDGE_CALIBRATION_EPISODES
        * len(calibration_rates),
        "confirmation_budgeted": BUDGETED_LAEDGE_CONFIRMATION_EPISODES
        * len(BUDGETED_LAEDGE_DELTA_TARGETS),
        "confirmation_fixed": BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
        "niin_reference": BUDGETED_LAEDGE_REFERENCE_EPISODES,
        "unconstrained_reference": BUDGETED_LAEDGE_REFERENCE_EPISODES,
    }


def assess_confirmation(*, target_delta: float, achieved_delta: float) -> dict[str, float | str]:
    target = _finite(target_delta, "target_delta", nonnegative=True)
    # A confirmation panel can realize less work than the NIIN reference,
    # especially at delta=0.  That is ordinary underfill, not an invalid
    # negative delta.  Only finiteness is required for the achieved value.
    achieved = _finite(achieved_delta, "achieved_delta")
    overfill = achieved - target
    underfill = target - achieved
    if overfill > BUDGETED_LAEDGE_CONFIRMATION_OVERFILL_TOLERANCE:
        raise BudgetedLaedgeCampaignError(
            f"confirmation overfill {overfill!r} exceeds "
            f"{BUDGETED_LAEDGE_CONFIRMATION_OVERFILL_TOLERANCE!r}"
        )
    return {
        "target_delta": target,
        "achieved_delta": achieved,
        "overfill": max(0.0, overfill),
        "underfill": max(0.0, underfill),
        "status": (
            "target_underfilled"
            if underfill > BUDGETED_LAEDGE_TARGET_UNDERFILL_TOLERANCE
            else "confirmed"
        ),
    }


def _validate_rate_results(
    rate_results: Sequence[Mapping[str, object]],
    calibration_rates: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    if len(rate_results) != len(calibration_rates):
        raise BudgetedLaedgeCampaignError("calibration rate curve has wrong size")
    rows = []
    for expected_rate, row in zip(calibration_rates, rate_results):
        if not isinstance(row, Mapping):
            raise BudgetedLaedgeCampaignError("calibration rate row must be a mapping")
        rate = _finite(row.get("rate"), "calibration rate", nonnegative=True)
        if rate != expected_rate:
            raise BudgetedLaedgeCampaignError("calibration rates are not canonical")
        work = _finite(row.get("mean_total_work"), "calibration mean work", nonnegative=True)
        rows.append((rate, work))
    return tuple(rows)


def select_budget_rate_mapping(
    *,
    rate_results: Sequence[Mapping[str, object]],
    niin_mean_total_work: float,
    unconstrained_mean_total_work: float,
    targets: Sequence[float] = BUDGETED_LAEDGE_DELTA_TARGETS,
    calibration_rates: Sequence[float] = BUDGETED_LAEDGE_CALIBRATION_RATES,
) -> dict[str, object]:
    baseline = _finite(niin_mean_total_work, "niin_mean_total_work", nonnegative=True)
    upper_reference = _finite(
        unconstrained_mean_total_work,
        "unconstrained_mean_total_work",
        nonnegative=True,
    )
    if baseline <= 0.0:
        raise BudgetedLaedgeCampaignError("NIIN reference work must be positive")
    curve = _validate_rate_results(rate_results, calibration_rates)
    for (_, left), (_, right) in zip(curve, curve[1:]):
        if right + 1e-12 < left:
            raise BudgetedLaedgeCampaignError("rate-to-work calibration curve is non-monotone")
    upper_shortfall = upper_reference - curve[-1][1]
    if upper_shortfall > BUDGETED_LAEDGE_UPPER_COVERAGE_TOLERANCE * baseline:
        raise BudgetedLaedgeCampaignError("calibration grid does not cover the unconstrained neighborhood")
    if tuple(float(value) for value in targets) != tuple(BUDGETED_LAEDGE_DELTA_TARGETS):
        raise BudgetedLaedgeCampaignError("delta targets are not canonical")
    selected: dict[str, object] = {}
    for target_delta in BUDGETED_LAEDGE_DELTA_TARGETS:
        target_work = baseline * (1.0 + target_delta)
        if target_work + 1e-12 < curve[0][1]:
            raise BudgetedLaedgeCampaignError(
                f"target {target_delta!r} is below the zero-rate work curve"
            )
        if target_delta == 0.0:
            selected[str(target_delta)] = {
                "target_delta": target_delta,
                "target_work": target_work,
                "planned_budget_rate": 0.0,
                "predicted_total_work": curve[0][1],
                "selection_method": "forced_zero_rate_no_hedge",
                "bracketing_rates": None,
                "interpolation_fraction": None,
                "saturation_limited": False,
            }
            continue
        feasible = [index for index, (_, work) in enumerate(curve) if work <= target_work + 1e-12]
        if not feasible:
            raise BudgetedLaedgeCampaignError("no feasible budget rate for target")
        left_index = max(feasible)
        saturation_limited = target_work > upper_reference + 1e-12
        if left_index == len(curve) - 1 or curve[left_index][1] >= target_work - 1e-12:
            selected[str(target_delta)] = {
                "target_delta": target_delta,
                "target_work": target_work,
                "planned_budget_rate": curve[left_index][0],
                "predicted_total_work": curve[left_index][1],
                "selection_method": "grid_max_feasible",
                "bracketing_rates": None,
                "interpolation_fraction": None,
                "saturation_limited": saturation_limited,
            }
            continue
        right_index = left_index + 1
        left_rate, left_work = curve[left_index]
        right_rate, right_work = curve[right_index]
        if right_work <= left_work:
            raise BudgetedLaedgeCampaignError("flat calibration segment cannot be interpolated")
        fraction = (target_work - left_work) / (right_work - left_work)
        rate = left_rate + fraction * (right_rate - left_rate)
        selected[str(target_delta)] = {
            "target_delta": target_delta,
            "target_work": target_work,
            "planned_budget_rate": rate,
            "predicted_total_work": target_work,
            "selection_method": "linear_interpolation",
            "bracketing_rates": [left_rate, right_rate],
            "interpolation_fraction": fraction,
            "saturation_limited": saturation_limited,
        }
    mapping = {
        "schema_id": "budgeted_laedge_rate_mapping_v1",
        "niin_mean_total_work": baseline,
        "unconstrained_mean_total_work": upper_reference,
        "rate_curve": [
            {"rate": rate, "mean_total_work": work} for rate, work in curve
        ],
        "targets": selected,
        "upper_coverage_shortfall": max(0.0, upper_shortfall),
    }
    mapping["mapping_fingerprint"] = _digest(mapping)
    return mapping


def _trace_digest(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _default_builder(
    config: ExperimentConfig,
    namespace: str,
    index: int,
    *,
    macro_seed: int = BUDGETED_LAEDGE_DEVELOPMENT_MACRO_SEED,
) -> EpisodeTrace:
    return generate_episode_trace(
        config,
        namespace,
        macro_seed,
        index,
        ATTRIBUTION_V1_PROTOCOL,
    )


def _simulation_for_arm(
    episode: EpisodeTrace,
    arm_key: str,
    *,
    planned_budget_rate: float | None = None,
) -> EpisodeSimulationResult:
    if arm_key == "fixed-dispatcher:no-hedge":
        return simulate_episode(episode, BUDGETED_LAEDGE_DEGRADED_SLOWDOWN)
    if arm_key == "niin:conservative":
        return simulate_episode_online(
            episode,
            FrozenNIINPolicy(),
            _RESERVATION,
            degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            hedge_delay=BUDGETED_LAEDGE_HEDGE_DELAY,
            cancel_running_losers=False,
        ).simulation
    if arm_key.startswith("budgeted-laedge:"):
        if planned_budget_rate is None:
            raise BudgetedLaedgeCampaignError("Budgeted arm requires a planned rate")
        return simulate_budgeted_laedge_episode(
            episode,
            planned_budget_rate=planned_budget_rate,
            degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            cancel_running_losers=False,
        ).simulation
    if arm_key == "unconstrained-laedge:conservative":
        return simulate_laedge_episode(
            episode,
            degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            cancel_running_losers=False,
        )
    if arm_key == "unconstrained-laedge:preemptive-oracle":
        return simulate_laedge_episode(
            episode,
            degraded_slowdown=BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            cancel_running_losers=True,
        )
    raise BudgetedLaedgeCampaignError(f"unknown Budgeted LÆDGE arm {arm_key!r}")


def _latency_row(tokens: Sequence[object], deadlines: SLODeadlines) -> dict[str, float]:
    if not tokens:
        raise BudgetedLaedgeCampaignError("required latency cohort is empty")
    values = tuple(float(token.latency) for token in tokens)
    return {
        "mean": math.fsum(values) / len(values),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "cvar95": empirical_cvar95(values),
        "miss_rate": math.fsum(
            token.latency > deadlines.for_class(token.token_class) for token in tokens
        )
        / len(tokens),
    }


def _outcome(run: EpisodeSimulationResult, *, placement_mode: str) -> dict[str, object]:
    try:
        metric = build_episode_metrics(
            run,
            BUDGETED_LAEDGE_DEGRADED_SLOWDOWN,
            storm_bin_width=BUDGETED_LAEDGE_STORM_BIN_WIDTH,
            placement_mode=placement_mode,
        )
    except Exception as error:
        raise
    tokens = tuple(run.simulation.tokens)
    timeline = run.episode.protocol.timeline
    deadlines = SLODeadlines()
    overall = _latency_row(tokens, deadlines)
    df = _latency_row(
        tuple(token for token in tokens if phase_at(timeline, token.arrival_time) in (Phase.DEGRADED, Phase.FAILED)),
        deadlines,
    )
    hr = _latency_row(
        tuple(token for token in tokens if phase_at(timeline, token.arrival_time) in (Phase.HEALTHY, Phase.RECOVERED)),
        deadlines,
    )
    attempts = tuple(run.simulation.attempts)
    total_work = math.fsum(attempt.executed_work for attempt in attempts)
    hedge_work = math.fsum(
        attempt.executed_work for attempt in attempts if attempt.attempt_id == 2
    )
    wasted_work = math.fsum(
        attempt.executed_work
        for attempt in attempts
        if attempt.status is not HedgeAttemptStatus.COMPLETED_WINNER
    )
    hedge_attempts = tuple(attempt for attempt in attempts if attempt.attempt_id == 2)
    hedge_winners = sum(
        attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
        for attempt in hedge_attempts
    )
    return {
        "overall_mean": overall["mean"],
        "overall_p95": overall["p95"],
        "overall_p99": overall["p99"],
        "df_mean": df["mean"],
        "df_p95": df["p95"],
        "df_p99": df["p99"],
        "df_cvar95": df["cvar95"],
        "df_miss_rate": df["miss_rate"],
        "hr_p99": hr["p99"],
        "replay_rate": math.fsum(token.replay_count > 0 for token in tokens) / len(tokens),
        "total_work": total_work,
        "incremental_hedge_work": hedge_work,
        "wasted_work": wasted_work,
        "hedge_launches": len(hedge_attempts),
        "hedge_winner_rate": hedge_winners / len(hedge_attempts) if hedge_attempts else None,
        "hedge_suppressed": run.simulation.hedge_suppressed,
        "storm_peak": (
            metric["storm"]["peak_hedge_launch_rate"]
        ),
        "drain_duration": run.simulation.drain_duration,
        "completed_tokens": run.simulation.completed_tokens,
        "invariant_exclusions": (
            ["hedge_requested_counter_matches_actions"]
            if placement_mode == "budgeted_idle_release"
            else []
        ),
    }


def _run_one(
    config: ExperimentConfig,
    namespace: str,
    index: int,
    arm_key: str,
    builder: Callable[[str, int], EpisodeTrace],
    *,
    planned_budget_rate: float | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    episode = builder(namespace, index)
    if not isinstance(episode, EpisodeTrace):
        raise BudgetedLaedgeCampaignError("episode builder returned a non-EpisodeTrace")
    return _run_on_episode(
        episode,
        arm_key,
        planned_budget_rate=planned_budget_rate,
    )


def _run_on_episode(
    episode: EpisodeTrace,
    arm_key: str,
    *,
    planned_budget_rate: float | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(episode, EpisodeTrace):
        raise BudgetedLaedgeCampaignError("episode must be an EpisodeTrace")
    before = _trace_digest(episode)
    run = _simulation_for_arm(
        episode,
        arm_key,
        planned_budget_rate=planned_budget_rate,
    )
    after = _trace_digest(episode)
    if before != after:
        raise BudgetedLaedgeCampaignError(f"arm {arm_key} mutated its episode")
    placement_mode = (
        "budgeted_idle_release"
        if arm_key.startswith("budgeted-laedge:")
        or arm_key.startswith("unconstrained-laedge:")
        else "fixed_dispatcher"
    )
    return (
        {
            "arm_key": arm_key,
            "trace_fingerprint": before,
            "outcome": _outcome(run, placement_mode=placement_mode),
        },
        {"episode": episode, "run": run},
    )


def _mean_outcome(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row["outcome"][key]) for row in rows]
    if not values:
        raise BudgetedLaedgeCampaignError("cannot aggregate an empty outcome set")
    return math.fsum(values) / len(values)


def _run_reference_library(
    config: ExperimentConfig,
    namespace: str,
    count: int,
    arm_key: str,
    builder: Callable[[str, int], EpisodeTrace],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    compact = []
    seen: set[str] = set()
    for index in range(count):
        try:
            episode = builder(namespace, index)
            if not isinstance(episode, EpisodeTrace):
                raise BudgetedLaedgeCampaignError("episode builder returned a non-EpisodeTrace")
            item, _ = _run_on_episode(episode, arm_key)
            fingerprint = item["trace_fingerprint"]
            if fingerprint in seen:
                raise BudgetedLaedgeCampaignError("duplicate episode trace fingerprint")
            seen.add(fingerprint)
            rows.append(item)
            compact.append(
                {
                    "episode_index": index,
                    "namespace": namespace,
                    "status": "completed",
                    "trace_fingerprint": fingerprint,
                    "arms": {arm_key: item["outcome"]},
                }
            )
        except Exception as error:
            compact.append(
                {
                    "episode_index": index,
                    "namespace": namespace,
                    "status": "failed",
                    "arms": {
                        arm_key: {
                            "status": "failed",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        }
                    },
                }
            )
    return rows, compact


def _source_bundle(project_root: str | Path | None = None) -> dict[str, object]:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    paths = sorted((root / "src" / "mfg_hedge").glob("*.py"))
    paths.append(root / "configs" / "v1_minimal.json")
    entries = []
    for path in paths:
        if not path.is_file():
            raise BudgetedLaedgeCampaignError(f"source bundle file is missing: {path}")
        relative = path.relative_to(root).as_posix()
        entries.append(
            [relative, hashlib.sha256(path.read_bytes()).hexdigest()]
        )
    return {
        "schema_id": "budgeted_laedge_source_bundle_v1",
        "entries": entries,
        "fingerprint": _digest(entries),
    }


def _protocol_payload(protocol: _BudgetedLaedgeProtocol = _V1_PROTOCOL) -> dict[str, object]:
    return {
        "schema_id": protocol.schema_id,
        "adr": "ADR-0033",
        "label": protocol.label,
        "development_namespace": protocol.development_namespace,
        "holdout_namespace": protocol.holdout_namespace,
        "development_macro_seed": protocol.development_macro_seed,
        "holdout_macro_seed": protocol.holdout_macro_seed,
        "calibration_rates": list(protocol.calibration_rates),
        "delta_targets": list(BUDGETED_LAEDGE_DELTA_TARGETS),
        "arms": list(BUDGETED_LAEDGE_ARM_KEYS),
        "counts": {
            "calibration": BUDGETED_LAEDGE_CALIBRATION_EPISODES,
            "confirmation": BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
            "reference": BUDGETED_LAEDGE_REFERENCE_EPISODES,
            "holdout": BUDGETED_LAEDGE_HOLDOUT_EPISODES,
        },
        "call_plan": development_call_plan(protocol.calibration_rates),
        "development_call_budget": BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET,
        "holdout_call_budget": BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
        "confirmation_overfill_tolerance": BUDGETED_LAEDGE_CONFIRMATION_OVERFILL_TOLERANCE,
        "target_underfill_tolerance": BUDGETED_LAEDGE_TARGET_UNDERFILL_TOLERANCE,
        "upper_coverage_tolerance": BUDGETED_LAEDGE_UPPER_COVERAGE_TOLERANCE,
        "no_retry_or_supplement": True,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def _run_budgeted_laedge_development(
    config: ExperimentConfig,
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    calibration_episode_count: int = BUDGETED_LAEDGE_CALIBRATION_EPISODES,
    confirmation_episode_count: int = BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
    reference_episode_count: int = BUDGETED_LAEDGE_REFERENCE_EPISODES,
    progress: Callable[[int, int], None] | None = None,
    protocol: _BudgetedLaedgeProtocol = _V1_PROTOCOL,
) -> dict[str, object]:
    validate_frozen_config(config)
    counts = (calibration_episode_count, confirmation_episode_count, reference_episode_count)
    if any(type(value) is not int or value <= 0 for value in counts):
        raise BudgetedLaedgeCampaignError("development episode counts must be positive ints")
    if calibration_episode_count > BUDGETED_LAEDGE_CALIBRATION_EPISODES:
        raise BudgetedLaedgeCampaignError("calibration episode count exceeds frozen limit")
    if confirmation_episode_count > BUDGETED_LAEDGE_CONFIRMATION_EPISODES:
        raise BudgetedLaedgeCampaignError("confirmation episode count exceeds frozen limit")
    if reference_episode_count > BUDGETED_LAEDGE_REFERENCE_EPISODES:
        raise BudgetedLaedgeCampaignError("reference episode count exceeds frozen limit")
    build = episode_builder or (
        lambda namespace, index: _default_builder(
            config,
            namespace,
            index,
            macro_seed=protocol.development_macro_seed,
        )
    )
    rows: list[dict[str, object]] = []
    calls = 0
    calibration_rows_by_rate: dict[float, list[dict[str, object]]] = {
        rate: [] for rate in protocol.calibration_rates
    }
    calibration_failures = 0
    for index in range(calibration_episode_count):
        row = {"episode_index": index, "namespace": protocol.calibration_namespace, "status": "completed", "arms": {}}
        try:
            episode = build(protocol.calibration_namespace, index)
            if not isinstance(episode, EpisodeTrace):
                raise BudgetedLaedgeCampaignError("episode builder returned a non-EpisodeTrace")
            row["trace_fingerprint"] = _trace_digest(episode)
            for rate in protocol.calibration_rates:
                arm_key = f"budgeted-laedge:rate={rate:g}"
                calls += 1
                try:
                    item, _ = _run_on_episode(
                        episode,
                        arm_key,
                        planned_budget_rate=rate,
                    )
                    calibration_rows_by_rate[rate].append(item)
                    row["arms"][arm_key] = {"status": "completed", **item}
                except Exception as error:
                    calibration_failures += 1
                    row["status"] = "failed"
                    row["arms"][arm_key] = {
                        "status": "failed",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
        except Exception as error:
            calibration_failures += len(protocol.calibration_rates)
            row["status"] = "failed"
            row["failure"] = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
            for rate in protocol.calibration_rates:
                calls += 1
                row["arms"][f"budgeted-laedge:rate={rate:g}"] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        rows.append(row)
        if progress is not None:
            progress(calls, BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET)
    nii_rows, nii_compact = _run_reference_library(
        config,
        protocol.niin_reference_namespace,
        reference_episode_count,
        "niin:conservative",
        build,
    )
    calls += reference_episode_count
    unconstrained_rows, unconstrained_compact = _run_reference_library(
        config,
        protocol.unconstrained_reference_namespace,
        reference_episode_count,
        "unconstrained-laedge:conservative",
        build,
    )
    calls += reference_episode_count
    rate_results = [
        {
            "rate": rate,
            "mean_total_work": _mean_outcome(calibration_rows_by_rate[rate], "total_work")
            if len(calibration_rows_by_rate[rate]) == calibration_episode_count
            else math.nan,
            "complete_episode_count": len(calibration_rows_by_rate[rate]),
        }
        for rate in protocol.calibration_rates
    ]
    mapping: dict[str, object] | None = None
    status = "calibration_failed" if calibration_failures else "confirmed"
    failure_reason = None
    try:
        mapping = select_budget_rate_mapping(
            rate_results=rate_results,
            niin_mean_total_work=_mean_outcome(nii_rows, "total_work"),
            unconstrained_mean_total_work=_mean_outcome(unconstrained_rows, "total_work"),
            calibration_rates=protocol.calibration_rates,
        )
    except Exception as error:
        status = "calibration_failed"
        failure_reason = f"{type(error).__name__}: {error}"
    confirmation_rows: list[dict[str, object]] = []
    confirmation_summary: dict[str, object] = {}
    if mapping is not None:
        for index in range(confirmation_episode_count):
            row = {"episode_index": index, "namespace": protocol.confirmation_namespace, "status": "completed", "arms": {}}
            try:
                episode = build(protocol.confirmation_namespace, index)
                if not isinstance(episode, EpisodeTrace):
                    raise BudgetedLaedgeCampaignError("episode builder returned a non-EpisodeTrace")
                row["trace_fingerprint"] = _trace_digest(episode)
                for arm_key in ("fixed-dispatcher:no-hedge", *BUDGETED_LAEDGE_ARM_KEYS[2:9]):
                    calls += 1
                    rate = None
                    if arm_key.startswith("budgeted-laedge:"):
                        delta_text = arm_key.split("=", 1)[1].rstrip("%")
                        delta = float(delta_text) / 100.0
                        rate = float(mapping["targets"][str(delta)]["planned_budget_rate"])
                    try:
                        item, _ = _run_on_episode(
                            episode,
                            arm_key,
                            planned_budget_rate=rate,
                        )
                        row["arms"][arm_key] = {"status": "completed", **item}
                    except Exception as error:
                        row["status"] = "failed"
                        row["arms"][arm_key] = {
                            "status": "failed",
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                        }
            except Exception as error:
                row["status"] = "failed"
                row["failure"] = {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
                for arm_key in ("fixed-dispatcher:no-hedge", *BUDGETED_LAEDGE_ARM_KEYS[2:9]):
                    calls += 1
                    row["arms"][arm_key] = {
                        "status": "failed",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
            confirmation_rows.append(row)
            if progress is not None:
                progress(calls, BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET)
        for delta in BUDGETED_LAEDGE_DELTA_TARGETS:
            arm_key = f"budgeted-laedge:delta={int(delta * 100):d}%"
            outcomes = [row["arms"][arm_key]["outcome"] for row in confirmation_rows if row["arms"].get(arm_key, {}).get("status") == "completed"]
            if len(outcomes) != confirmation_episode_count:
                status = "calibration_failed"
                failure_reason = "confirmation panel is incomplete"
                break
            achieved = (_mean_outcome([{"outcome": outcome} for outcome in outcomes], "total_work") / _mean_outcome(nii_rows, "total_work")) - 1.0
            try:
                confirmation_summary[str(delta)] = {
                    **assess_confirmation(
                    target_delta=delta,
                    achieved_delta=achieved,
                    ),
                    "saturation_limited": bool(
                        mapping["targets"][str(delta)].get("saturation_limited", False)
                    ),
                }
            except Exception as error:
                status = "calibration_failed"
                failure_reason = f"{type(error).__name__}: {error}"
                break
        if status != "calibration_failed":
            status = "confirmed"
    expected_calls = (
        calibration_episode_count * len(protocol.calibration_rates)
        + (
            confirmation_episode_count * (1 + len(BUDGETED_LAEDGE_DELTA_TARGETS))
            if mapping is not None
            else 0
        )
        + 2 * reference_episode_count
    )
    if calls != expected_calls:
        raise BudgetedLaedgeCampaignError(
            f"development call accounting failed: {calls} != {expected_calls}"
        )
    protocol_payload = _protocol_payload(protocol)
    result = {
        "label": protocol.label,
        "status": status,
        "failure_reason": failure_reason,
        "scheduler_calls": calls,
        "formal_scheduler_call_budget": BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET,
        "expected_scheduler_calls_for_this_run": expected_calls,
        "protocol": protocol_payload,
        "protocol_fingerprint": _digest(protocol_payload),
        "rate_curve": rate_results,
        "references": {
            "niin_mean_total_work": _mean_outcome(nii_rows, "total_work") if len(nii_rows) == reference_episode_count else None,
            "unconstrained_mean_total_work": _mean_outcome(unconstrained_rows, "total_work") if len(unconstrained_rows) == reference_episode_count else None,
            "niin_complete_episode_count": len(nii_rows),
            "unconstrained_complete_episode_count": len(unconstrained_rows),
        },
        "mapping": mapping,
        "mapping_fingerprint": mapping.get("mapping_fingerprint") if mapping else None,
        "confirmation": confirmation_summary,
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": rows + nii_compact + unconstrained_compact + confirmation_rows,
    }
    return result


def run_budgeted_laedge_development(
    config: ExperimentConfig,
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    calibration_episode_count: int = BUDGETED_LAEDGE_CALIBRATION_EPISODES,
    confirmation_episode_count: int = BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
    reference_episode_count: int = BUDGETED_LAEDGE_REFERENCE_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    return _run_budgeted_laedge_development(
        config,
        episode_builder=episode_builder,
        calibration_episode_count=calibration_episode_count,
        confirmation_episode_count=confirmation_episode_count,
        reference_episode_count=reference_episode_count,
        progress=progress,
        protocol=_V1_PROTOCOL,
    )


def run_budgeted_laedge_development_v2(
    config: ExperimentConfig,
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    calibration_episode_count: int = BUDGETED_LAEDGE_CALIBRATION_EPISODES,
    confirmation_episode_count: int = BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
    reference_episode_count: int = BUDGETED_LAEDGE_REFERENCE_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    return _run_budgeted_laedge_development(
        config,
        episode_builder=episode_builder,
        calibration_episode_count=calibration_episode_count,
        confirmation_episode_count=confirmation_episode_count,
        reference_episode_count=reference_episode_count,
        progress=progress,
        protocol=_V2_PROTOCOL,
    )


def development_call_plan_v2() -> dict[str, int]:
    return development_call_plan(BUDGETED_LAEDGE_V2_CALIBRATION_RATES)


def protocol_payload_v2() -> dict[str, object]:
    return _protocol_payload(_V2_PROTOCOL)


def select_budget_rate_mapping_v2(
    *,
    rate_results: Sequence[Mapping[str, object]],
    niin_mean_total_work: float,
    unconstrained_mean_total_work: float,
    targets: Sequence[float] = BUDGETED_LAEDGE_DELTA_TARGETS,
) -> dict[str, object]:
    return select_budget_rate_mapping(
        rate_results=rate_results,
        niin_mean_total_work=niin_mean_total_work,
        unconstrained_mean_total_work=unconstrained_mean_total_work,
        targets=targets,
        calibration_rates=BUDGETED_LAEDGE_V2_CALIBRATION_RATES,
    )


def run_budgeted_laedge_holdout(
    config: ExperimentConfig,
    development_result: Mapping[str, object],
    *,
    episode_builder: Callable[[str, int], EpisodeTrace] | None = None,
    episode_count: int = BUDGETED_LAEDGE_HOLDOUT_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    if not isinstance(development_result, Mapping) or development_result.get("status") != "confirmed":
        raise BudgetedLaedgeCampaignError("holdout requires a confirmed development mapping")
    mapping = development_result.get("mapping")
    if not isinstance(mapping, Mapping) or development_result.get("mapping_fingerprint") != mapping.get("mapping_fingerprint"):
        raise BudgetedLaedgeCampaignError("development mapping fingerprint is missing or stale")
    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= BUDGETED_LAEDGE_HOLDOUT_EPISODES:
        raise BudgetedLaedgeCampaignError("holdout episode count is outside the frozen limit")
    build = episode_builder or (lambda namespace, index: generate_episode_trace(config, namespace, BUDGETED_LAEDGE_HOLDOUT_MACRO_SEED, index, ATTRIBUTION_V1_PROTOCOL))
    target_rates = {
        f"budgeted-laedge:delta={int(delta * 100):d}%": float(mapping["targets"][str(delta)]["planned_budget_rate"])
        for delta in BUDGETED_LAEDGE_DELTA_TARGETS
    }
    rows = []
    calls = 0
    for index in range(episode_count):
        episode_row = {"episode_index": index, "status": "completed", "arms": {}}
        try:
            episode = build(BUDGETED_LAEDGE_HOLDOUT_NAMESPACE, index)
            if not isinstance(episode, EpisodeTrace):
                raise BudgetedLaedgeCampaignError("episode builder returned a non-EpisodeTrace")
            episode_row["trace_fingerprint"] = _trace_digest(episode)
            for arm_key in BUDGETED_LAEDGE_ARM_KEYS:
                calls += 1
                rate = target_rates.get(arm_key)
                try:
                    item, _ = _run_on_episode(
                        episode,
                        arm_key,
                        planned_budget_rate=rate,
                    )
                    episode_row["arms"][arm_key] = {"status": "completed", **item}
                except Exception as error:
                    episode_row["status"] = "physical_failed"
                    episode_row["arms"][arm_key] = {
                        "status": "failed",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
        except Exception as error:
            episode_row["status"] = "physical_failed"
            episode_row["failure"] = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
            for arm_key in BUDGETED_LAEDGE_ARM_KEYS:
                calls += 1
                episode_row["arms"][arm_key] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        rows.append(episode_row)
        if progress is not None:
            progress(calls, BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET)
    return {
        "label": BUDGETED_LAEDGE_HOLDOUT_LABEL,
        "status": "completed" if all(row["status"] == "completed" for row in rows) else "physical_failed",
        "scheduler_calls": calls,
        "formal_scheduler_call_budget": BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
        "mapping_fingerprint": mapping["mapping_fingerprint"],
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": rows,
    }


def write_budgeted_laedge_artifact(
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
                if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
                    raise ValueError("episode_rows.jsonl payload must be a sequence")
                text = "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    for row in payload
                )
            elif name.endswith(".json") and "/" not in name and "\\" not in name:
                text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
            else:
                raise ValueError(f"invalid Budgeted LÆDGE artifact file name: {name!r}")
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _execute_budgeted_laedge_development(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    progress: Callable[[int, int], None] | None = None,
    protocol: _BudgetedLaedgeProtocol = _V1_PROTOCOL,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    result = _run_budgeted_laedge_development(
        config,
        progress=progress,
        protocol=protocol,
    )
    source_bundle = _source_bundle(project)
    protocol_payload = result["protocol"]
    result["source_bundle"] = source_bundle
    result["source_bundle_fingerprint"] = source_bundle["fingerprint"]
    result["protocol_fingerprint"] = _digest(protocol_payload)
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    manifest = {
        "schema_id": f"budgeted_laedge_development_manifest_{'v2' if protocol is _V2_PROTOCOL else 'v1'}",
        "run_id": run_id,
        "label": protocol.label,
        "status": result["status"],
        "scheduler_calls": result["scheduler_calls"],
        "protocol_fingerprint": result["protocol_fingerprint"],
        "source_bundle_fingerprint": source_bundle["fingerprint"],
        "claims_best_response": False,
        "claims_optimality": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    return (
        write_budgeted_laedge_artifact(
            artifacts_root,
            run_id,
            {
                "summary.json": summary,
                "episode_rows.jsonl": result["episode_rows"],
                "manifest.json": manifest,
                "protocol.json": protocol_payload,
            },
        ),
        result,
    )


def execute_budgeted_laedge_development(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    return _execute_budgeted_laedge_development(
        project_root,
        artifacts_root,
        run_id,
        progress=progress,
        protocol=_V1_PROTOCOL,
    )


def execute_budgeted_laedge_development_v2(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    return _execute_budgeted_laedge_development(
        project_root,
        artifacts_root,
        run_id,
        progress=progress,
        protocol=_V2_PROTOCOL,
    )


__all__ = [
    "BUDGETED_LAEDGE_CALIBRATION_EPISODES",
    "BUDGETED_LAEDGE_CALIBRATION_RATES",
    "BUDGETED_LAEDGE_V2_CALIBRATION_RATES",
    "BUDGETED_LAEDGE_CONFIRMATION_EPISODES",
    "BUDGETED_LAEDGE_CONFIRMATION_OVERFILL_TOLERANCE",
    "BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET",
    "BUDGETED_LAEDGE_DELTA_TARGETS",
    "BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET",
    "BUDGETED_LAEDGE_HOLDOUT_EPISODES",
    "BUDGETED_LAEDGE_ARM_KEYS",
    "BUDGETED_LAEDGE_V2_DEVELOPMENT_MACRO_SEED",
    "BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE",
    "BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED",
    "BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE",
    "BudgetedLaedgeCampaignError",
    "assess_confirmation",
    "development_call_plan",
    "execute_budgeted_laedge_development",
    "execute_budgeted_laedge_development_v2",
    "development_call_plan_v2",
    "protocol_payload_v2",
    "run_budgeted_laedge_development",
    "run_budgeted_laedge_holdout",
    "select_budget_rate_mapping",
    "select_budget_rate_mapping_v2",
    "write_budgeted_laedge_artifact",
]
