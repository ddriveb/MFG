"""Frozen paired three-arm population evaluation for the requested-price policy.

This module is deliberately a finite-system performance gate.  It runs the
complete online physical engine for No-Hedge, exact NIIN, and the sealed
requested-price candidate on shared immutable episode traces.  It does not
compute a best response, regret, Nash equilibrium, or MFG result.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from array import array
from typing import Callable, Iterable, Mapping, Sequence

from .artifacts import write_run_directory
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
)
from .attribution_metrics import (
    SLODeadlines,
    _execution_work,
    _latency_views,
    _sum_work,
    build_episode_metrics,
    empirical_cvar95,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction, TokenClass
from .metrics import percentile
from .token_online import TokenObservation, simulate_episode_online
from .token_refined_grid import refined_bin_id
from .token_t3a_execution import (
    ENVIRONMENT_ID,
    _RESERVATION,
    build_source_bundle,
    environment_fingerprint,
    validate_frozen_config,
)


THREE_ARM_LABEL = "finite_three_arm_requested_price_population_evaluation"
THREE_ARM_NAMESPACE = "token-mfg-restoration:three-arm-holdout:v1"
THREE_ARM_MACRO_SEED = 20260918
THREE_ARM_EPISODES = 1024
THREE_ARM_COUNT = 3
HOLDOUT_CALL_LIMIT = THREE_ARM_EPISODES * THREE_ARM_COUNT
REQUESTED_RESERVATION_PRICE = 2.0
REQUESTED_PRICE_BETA = 4.0
LATE_AFTER = 50.0
DEGRADED_SLOWDOWN = 2.0
HEDGE_DELAY = 2.0
STORM_BIN_WIDTH = 1.0
BOOTSTRAP_NAMESPACE = (
    "token-mfg-restoration:three-arm-holdout:v1:paired-bootstrap"
)
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPLICATES = 4096
PARENT_SUMMARY = Path("artifacts/token-refined-fixed-grid-20260908-r1/summary.json")
POLICY_DRAW_NAMESPACE = "token-mfg-restoration:three-arm-holdout:v1:policy-draw"
ARM_NAMES = ("A", "B", "C")
_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_PHASES = (Phase.HEALTHY, Phase.DEGRADED, Phase.FAILED, Phase.RECOVERED)


class ThreeArmExecutionError(RuntimeError):
    """A frozen three-arm protocol or physical execution violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThreeArmExecutionError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ThreeArmExecutionError(f"{name} must be finite")
    return result


def stable_policy_uniform(seed: object, policy_key: str) -> float:
    """Return the frozen SHA-256 policy-key uniform in ``[0, 1)``."""

    if not isinstance(policy_key, str) or not policy_key:
        raise ThreeArmExecutionError("policy_key must be a non-empty string")
    raw = hashlib.sha256(
        f"{POLICY_DRAW_NAMESPACE}:{seed}:{policy_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(raw[:8], "big") / float(1 << 64)


def candidate_action_probabilities(
    costs: Mapping[ProtectionAction, float],
    *,
    requested_reservation_price: float = REQUESTED_RESERVATION_PRICE,
    beta: float = REQUESTED_PRICE_BETA,
) -> dict[ProtectionAction, float]:
    """Compute the frozen stable softmax over runtime action costs.

    The requested price is charged to D and I as a policy cost only.  It does
    not modify the online Reservation ledger or physical engine.
    """

    price = _finite(requested_reservation_price, "requested_reservation_price", nonnegative=True)
    temperature = _finite(beta, "beta", nonnegative=True)
    if temperature <= 0.0:
        raise ThreeArmExecutionError("beta must be positive")
    if set(costs) != set(_ACTIONS):
        raise ThreeArmExecutionError("candidate costs must contain exactly N/D/I")
    adjusted = {}
    for action in _ACTIONS:
        base = _finite(costs[action], f"cost[{action.value}]")
        adjusted[action] = base + (price if action is not ProtectionAction.NORMAL else 0.0)
    minimum = min(adjusted.values())
    weights = {
        action: math.exp(-temperature * (value - minimum))
        for action, value in adjusted.items()
    }
    normalizer = math.fsum(weights.values())
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ThreeArmExecutionError("candidate softmax normalizer is invalid")
    return {action: weights[action] / normalizer for action in _ACTIONS}


class FrozenNIINPolicy:
    """Exact NIIN with an optional supported-bin requested-price policy."""

    def __init__(
        self,
        *,
        late_after: float = LATE_AFTER,
        supported_probabilities: Mapping[str, Mapping[ProtectionAction, float]] | None = None,
        policy_seed: int = THREE_ARM_MACRO_SEED,
        candidate_fingerprint: str | None = None,
    ) -> None:
        self.late_after = _finite(late_after, "late_after")
        if type(policy_seed) is not int or policy_seed < 0:
            raise ThreeArmExecutionError("policy_seed must be a non-negative int")
        self.policy_seed = policy_seed
        normalized: dict[str, dict[ProtectionAction, float]] = {}
        for bin_id, probabilities in (supported_probabilities or {}).items():
            if not isinstance(bin_id, str) or not bin_id:
                raise ThreeArmExecutionError("candidate bin_id must be non-empty")
            if set(probabilities) != set(_ACTIONS):
                raise ThreeArmExecutionError("candidate probabilities must contain N/D/I")
            values = {action: _finite(probabilities[action], "candidate probability") for action in _ACTIONS}
            if any(value < 0.0 for value in values.values()) or not math.isclose(
                math.fsum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ThreeArmExecutionError("candidate probabilities must sum to one")
            normalized[bin_id] = values
        self.supported_probabilities = {
            key: dict(value) for key, value in normalized.items()
        }
        self.candidate_fingerprint = candidate_fingerprint

    def new_episode(self) -> "FrozenNIINPolicy":
        return FrozenNIINPolicy(
            late_after=self.late_after,
            supported_probabilities=self.supported_probabilities,
            policy_seed=self.policy_seed,
            candidate_fingerprint=self.candidate_fingerprint,
        )

    def action(
        self, token_class: TokenClass, phase_age: float, primary_replica: int
    ) -> ProtectionAction:
        if not isinstance(token_class, TokenClass):
            raise ThreeArmExecutionError("token_class must be TokenClass")
        if type(primary_replica) is not int or primary_replica not in (0, 1):
            raise ThreeArmExecutionError("primary_replica must be int 0 or 1")
        age = _finite(phase_age, "phase_age")
        if primary_replica != 0:
            return ProtectionAction.NORMAL
        late = age >= self.late_after
        if token_class is TokenClass.REGULAR:
            return ProtectionAction.IMMEDIATE_HEDGE if late else ProtectionAction.NORMAL
        return ProtectionAction.NORMAL if late else ProtectionAction.IMMEDIATE_HEDGE

    def choose_observation(self, observation: TokenObservation | None, policy_key: str = "") -> ProtectionAction:
        if not isinstance(observation, TokenObservation):
            return ProtectionAction.NORMAL
        fallback = self.action(
            observation.token_class,
            observation.phase_age,
            observation.primary_replica,
        )
        if not self.supported_probabilities:
            return fallback
        try:
            bin_id = refined_bin_id(observation)
        except Exception:
            return fallback
        probabilities = self.supported_probabilities.get(bin_id)
        if probabilities is None:
            return fallback
        draw = stable_policy_uniform(self.policy_seed, policy_key)
        cumulative = 0.0
        for action in _ACTIONS:
            cumulative += probabilities[action]
            if draw < cumulative:
                return action
        return _ACTIONS[-1]

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        return self.choose_observation(observation, policy_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": "pi_requested_reservation_price_v1",
            "fallback_policy_id": "pi_NIIN_v1",
            "late_after": self.late_after,
            "requested_reservation_price": REQUESTED_RESERVATION_PRICE,
            "beta": REQUESTED_PRICE_BETA,
            "policy_seed": self.policy_seed,
            "candidate_fingerprint": self.candidate_fingerprint,
            "supported_probabilities": {
                bin_id: {action.value: probabilities[action] for action in _ACTIONS}
                for bin_id, probabilities in sorted(self.supported_probabilities.items())
            },
        }


class NoHedgePolicy:
    def new_episode(self) -> "NoHedgePolicy":
        return NoHedgePolicy()

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        return ProtectionAction.NORMAL


def load_requested_price_candidate(summary_path: str | Path = PARENT_SUMMARY) -> FrozenNIINPolicy:
    path = Path(summary_path)
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ThreeArmExecutionError(f"cannot read sealed candidate summary: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("status") != "completed":
        raise ThreeArmExecutionError("parent candidate summary is not completed")
    for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg"):
        if payload.get(name) is not False:
            raise ThreeArmExecutionError(f"parent candidate claim boundary {name} is invalid")
    grid = payload.get("grid")
    if not isinstance(grid, Mapping):
        raise ThreeArmExecutionError("parent summary lacks grid")
    supported = grid.get("supported_bins")
    bin_stats = grid.get("bin_stats")
    if not isinstance(supported, list) or not isinstance(bin_stats, Mapping):
        raise ThreeArmExecutionError("parent grid support is incomplete")
    probabilities: dict[str, dict[ProtectionAction, float]] = {}
    for bin_id in sorted(supported):
        stats = bin_stats.get(bin_id)
        if not isinstance(stats, Mapping) or not isinstance(stats.get("actions"), Mapping):
            raise ThreeArmExecutionError(f"parent bin {bin_id!r} is incomplete")
        costs = {}
        for action in _ACTIONS:
            action_row = stats["actions"].get(action.value)
            if not isinstance(action_row, Mapping) or "unpriced_cost" not in action_row:
                raise ThreeArmExecutionError(f"parent bin {bin_id!r} lacks {action.value}")
            costs[action] = _finite(action_row["unpriced_cost"], "unpriced_cost")
        probabilities[bin_id] = candidate_action_probabilities(costs)
    candidate_payload = {
        "parent_summary_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "cost_model_id": "token_runtime_requested_reservation_price_v1",
        "requested_reservation_price": REQUESTED_RESERVATION_PRICE,
        "beta": REQUESTED_PRICE_BETA,
        "refined_observation": "ADR-0027",
        "supported_probabilities": {
            key: {action.value: probabilities[key][action] for action in _ACTIONS}
            for key in sorted(probabilities)
        },
    }
    return FrozenNIINPolicy(
        supported_probabilities=probabilities,
        candidate_fingerprint=_digest(candidate_payload),
    )


def _bootstrap_index_library(sample_count: int, replicates: int) -> array:
    if type(sample_count) is not int or sample_count <= 0:
        raise ThreeArmExecutionError("sample_count must be positive")
    if type(replicates) is not int or replicates <= 0:
        raise ThreeArmExecutionError("replicates must be positive")
    seed_bytes = hashlib.sha256(
        f"{BOOTSTRAP_NAMESPACE}:{BOOTSTRAP_SEED}:{sample_count}:{replicates}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    indices = array("I")
    for _ in range(replicates * sample_count):
        indices.append(rng.randrange(sample_count))
    return indices


def _bootstrap_fingerprint(indices: array) -> str:
    return hashlib.sha256(indices.tobytes()).hexdigest()


def paired_cluster_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    index_library: array | None = None,
) -> dict[str, object]:
    """Return paired cluster difference, SE, and deterministic percentile CI."""

    if len(left) != len(right) or not left:
        raise ThreeArmExecutionError("paired bootstrap inputs must have equal positive length")
    values_left = tuple(_finite(value, "left cluster value") for value in left)
    values_right = tuple(_finite(value, "right cluster value") for value in right)
    differences = tuple(right_value - left_value for left_value, right_value in zip(values_left, values_right))
    point = math.fsum(differences) / len(differences)
    if len(differences) > 1:
        mean = point
        paired_se = math.sqrt(
            math.fsum((value - mean) ** 2 for value in differences)
            / (len(differences) - 1)
        ) / math.sqrt(len(differences))
    else:
        paired_se = 0.0
    indices = index_library
    if indices is None:
        indices = _bootstrap_index_library(len(differences), replicates)
    expected_size = len(differences) * replicates
    if len(indices) != expected_size:
        raise ThreeArmExecutionError("bootstrap index library has the wrong shape")
    means = []
    for start in range(0, len(indices), len(differences)):
        means.append(
            math.fsum(differences[indices[start + offset]] for offset in range(len(differences)))
            / len(differences)
        )
    return {
        "namespace": BOOTSTRAP_NAMESPACE,
        "seed": BOOTSTRAP_SEED,
        "replicates": replicates,
        "sample_count": len(differences),
        "point": point,
        "paired_se": paired_se,
        "ci95": [percentile(means, 2.5), percentile(means, 97.5)],
        "index_library_fingerprint": _bootstrap_fingerprint(indices),
    }


def _comparison_entry(
    left: Sequence[float],
    right: Sequence[float],
    *,
    index_library: array,
) -> dict[str, object]:
    result = paired_cluster_bootstrap(
        left, right, replicates=BOOTSTRAP_REPLICATES, index_library=index_library
    )
    baseline = math.fsum(left) / len(left)
    result["relative_change"] = result["point"] / abs(baseline) if baseline else None
    return result


def _gate_field(data: Mapping[str, object], name: str) -> object:
    nested = data.get(name)
    if isinstance(nested, Mapping):
        return nested
    return data.get(name)


def evaluate_gate(comparison: Mapping[str, object], *, invariants_pass: bool) -> bool:
    """Apply the frozen C-versus-NIIN performance gate."""

    def value(name: str) -> float | None:
        raw = _gate_field(comparison, name)
        if isinstance(raw, Mapping):
            raw = raw.get("relative_change", raw.get("point"))
        return None if raw is None else float(raw)

    cvar = _gate_field(comparison, "df_cvar95")
    if isinstance(cvar, Mapping):
        cvar_relative = cvar.get("relative_change")
        cvar_ci = cvar.get("ci95")
    else:
        cvar_relative = comparison.get("df_cvar95_relative_change")
        cvar_ci = comparison.get("df_cvar95_ci")
    if cvar_relative is None or not isinstance(cvar_ci, (list, tuple)) or len(cvar_ci) != 2:
        return False
    conditions = (
        float(cvar_relative) <= -0.10,
        float(cvar_ci[1]) < 0.0,
        (value("df_miss_rate") is not None and value("df_miss_rate") <= 0.0),
        (value("replay_rate") is not None and value("replay_rate") <= 0.0),
        (value("hr_p99") is not None and value("hr_p99") <= 0.02),
        (value("total_work") is not None and value("total_work") <= 0.05),
        (value("storm_peak") is not None and value("storm_peak") <= 0.0),
        invariants_pass,
    )
    return all(conditions)


def _trace_digest(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _latency_scalar(tokens: Sequence[object], deadlines: SLODeadlines) -> dict[str, object]:
    values = [float(token.latency) for token in tokens]
    if not values:
        return {"sample_count": 0, "mean": None, "p99": None, "cvar95": None, "miss_rate": None}
    misses = sum(token.latency > deadlines.for_class(token.token_class) for token in tokens)
    return {
        "sample_count": len(values),
        "mean": math.fsum(values) / len(values),
        "p99": percentile(values, 99.0),
        "cvar95": empirical_cvar95(values),
        "miss_rate": misses / len(values),
    }


def _episode_outcomes(run, deadlines: SLODeadlines) -> dict[str, float]:
    tokens = tuple(run.simulation.simulation.tokens)
    by_phase = {
        phase.value: tuple(
            token
            for token in tokens
            if phase_at(run.episode.protocol.timeline, token.arrival_time) is phase
        )
        for phase in _PHASES
    }
    df = by_phase[Phase.DEGRADED.value] + by_phase[Phase.FAILED.value]
    hr = by_phase[Phase.HEALTHY.value] + by_phase[Phase.RECOVERED.value]
    latency = _latency_scalar(tokens, deadlines)
    df_latency = _latency_scalar(df, deadlines)
    hr_latency = _latency_scalar(hr, deadlines)
    if any(
        value.get("cvar95") is None
        for value in (df_latency, hr_latency, latency)
    ):
        raise ThreeArmExecutionError("episode lacks required phase latency support")
    replay_rate = sum(token.replay_count > 0 for token in tokens) / len(tokens)
    return {
        "df_cvar95": float(df_latency["cvar95"]),
        "df_miss_rate": float(df_latency["miss_rate"]),
        "df_p95": percentile([token.latency for token in df], 95.0),
        "df_p99": float(df_latency["p99"]),
        "overall_mean": float(latency["mean"]),
        "overall_p99": float(latency["p99"]),
        "hr_p99": float(hr_latency["p99"]),
        "replay_rate": replay_rate,
    }


def _episode_record(run) -> tuple[dict[str, object], dict[str, float]]:
    attribution = build_episode_metrics(
        run.simulation,
        DEGRADED_SLOWDOWN,
        storm_bin_width=STORM_BIN_WIDTH,
    )
    if not all(attribution["invariants"].values()):
        raise ThreeArmExecutionError("episode invariant failure")
    outcomes = _episode_outcomes(run, SLODeadlines())
    decisions = tuple(run.decisions)
    requested = sum(decision.requested is not ProtectionAction.NORMAL for decision in decisions)
    applied = sum(decision.applied is not ProtectionAction.NORMAL for decision in decisions)
    reservation_suppressed = sum(decision.reservation_suppressed for decision in decisions)
    admitted = sum(decision.reservation_admitted for decision in decisions)
    windows = {
        (decision.window_start, decision.window_end): decision.cap
        for decision in decisions
        if decision.window_start is not None and decision.window_end is not None
    }
    record = {
        "token_count": len(run.simulation.simulation.tokens),
        "attribution": attribution,
        "decision_counts": {
            "requested_hedges": requested,
            "applied_hedges": applied,
            "reservation_suppressed": reservation_suppressed,
            "reservation_admitted": admitted,
            "executor_suppressed": sum(decision.executor_suppressed for decision in decisions),
            "timer_scheduled": sum(decision.timer_status == "scheduled" for decision in decisions),
            "timer_fired": sum(decision.timer_status == "fired" for decision in decisions),
            "timer_voided": sum(decision.timer_status == "voided" for decision in decisions),
            "requested_reservation_work": requested * _RESERVATION.mean_requirement,
            "admitted_reservation_work": math.fsum(decision.charge for decision in decisions),
            "reservation_capacity": math.fsum(windows.values()),
        },
        "outcomes": outcomes,
    }
    compact = {
        "token_count": record["token_count"],
        "decision_counts": record["decision_counts"],
        "outcomes": outcomes,
        "work": attribution["work"],
        "replay": attribution["replay"],
        "hedge": attribution["hedge"],
        "storm": attribution["storm"],
    }
    return {"record": record, "compact": compact}, outcomes


def _aggregate_arm(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        return {"episode_count": 0, "token_count": 0, "invariants": {"all_passed": False}}
    record_values = [item["record"] for item in records]
    runs = [item["run"] for item in records if "run" in item]
    if not runs:
        raise ThreeArmExecutionError("successful arm records lack runs")
    first = runs[0]
    tokens = tuple(token for run in runs for token in run.simulation.simulation.tokens)
    total_tokens = len(tokens)
    latency = _latency_views(tokens, first.episode.protocol.timeline, SLODeadlines())
    episode_metrics = [record["attribution"] for record in record_values]
    work = _sum_work(episode_metrics)
    decision_names = (
        "requested_hedges",
        "applied_hedges",
        "reservation_suppressed",
        "reservation_admitted",
        "executor_suppressed",
        "timer_scheduled",
        "timer_fired",
        "timer_voided",
        "requested_reservation_work",
        "admitted_reservation_work",
        "reservation_capacity",
    )
    decision_counts = {
        name: math.fsum(item["decision_counts"][name] for item in record_values)
        for name in decision_names
    }
    decision_counts["requested_hedges"] = int(decision_counts["requested_hedges"])
    decision_counts["applied_hedges"] = int(decision_counts["applied_hedges"])
    decision_counts["reservation_suppressed"] = int(decision_counts["reservation_suppressed"])
    decision_counts["reservation_admitted"] = int(decision_counts["reservation_admitted"])
    decision_counts["executor_suppressed"] = int(decision_counts["executor_suppressed"])
    decision_counts["timer_scheduled"] = int(decision_counts["timer_scheduled"])
    decision_counts["timer_fired"] = int(decision_counts["timer_fired"])
    decision_counts["timer_voided"] = int(decision_counts["timer_voided"])
    requested_capacity = decision_counts["reservation_capacity"]
    decision_counts["requested_capacity_ratio"] = (
        decision_counts["requested_reservation_work"] / requested_capacity
        if requested_capacity > 0.0
        else None
    )
    decision_counts["admitted_capacity_ratio"] = (
        decision_counts["admitted_reservation_work"] / requested_capacity
        if requested_capacity > 0.0
        else None
    )
    decision_counts["suppression_rate"] = (
        decision_counts["reservation_suppressed"] / total_tokens
        if total_tokens > 0
        else None
    )
    hedge = {
        name: sum(item["hedge"][name] for item in episode_metrics)
        for name in (
            "requested",
            "launched",
            "winners",
            "cancelled_queued",
            "completed_loser",
            "failed_running",
            "invalidated_queued",
            "executor_suppressed",
        )
    }
    hedge["win_per_launch"] = hedge["winners"] / hedge["launched"] if hedge["launched"] else None
    replayed = sum(item["replay"]["replayed_tokens"] for item in episode_metrics)
    storm = {
        "peak_hedge_launch_rate_max": max(item["storm"]["peak_hedge_launch_rate"] for item in episode_metrics),
        "peak_hedge_launch_rate_mean": math.fsum(item["storm"]["peak_hedge_launch_rate"] for item in episode_metrics) / len(records),
        "mean_hedge_launch_rate_mean": math.fsum(item["storm"]["mean_hedge_launch_rate"] for item in episode_metrics) / len(records),
        "peak_to_mean_hedge_launch_ratio_max": max(
            (item["storm"]["peak_to_mean_hedge_launch_ratio"] for item in episode_metrics if item["storm"]["peak_to_mean_hedge_launch_ratio"] is not None),
            default=None,
        ),
        "sustained_overload_duration_max": max(item["storm"]["sustained_overload_duration"] for item in episode_metrics),
        "hedge_launch_count": sum(item["storm"]["hedge_launch_count"] for item in episode_metrics),
        "overloaded_bin_count_total": sum(item["storm"]["overloaded_bin_count"] for item in episode_metrics),
    }
    invariants = {
        "all_episode_invariants_pass": all(
            all(item["invariants"].values()) for item in episode_metrics
        ),
        "token_count_positive": total_tokens > 0,
    }
    return {
        "episode_count": len(records),
        "token_count": total_tokens,
        "latency": {
            "overall": latency["overall"],
            "arrival_phase": latency["arrival_phase"],
            "completion_phase": latency["completion_phase"],
        },
        "replay": {
            "replayed_tokens": replayed,
            "rate": replayed / total_tokens,
            "replay_execution_count": sum(item["replay"]["replayed_tokens"] for item in episode_metrics),
        },
        "hedge": hedge,
        "reservation": decision_counts,
        "work": work,
        "storm": storm,
        "invariants": invariants,
    }


def _comparison(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    left_name: str,
    right_name: str,
    index_library: array,
) -> dict[str, object]:
    left = records[left_name]
    right = records[right_name]
    if len(left) != len(right) or not left:
        raise ThreeArmExecutionError("paired comparisons require complete episode rows")
    names = (
        "df_cvar95",
        "df_miss_rate",
        "df_p95",
        "df_p99",
        "overall_mean",
        "overall_p99",
        "hr_p99",
        "replay_rate",
    )
    result: dict[str, object] = {
        "left_arm": left_name,
        "right_arm": right_name,
        "cluster_unit": "episode_index",
    }
    for name in names:
        left_values = [item["record"]["outcomes"][name] for item in left]
        right_values = [item["record"]["outcomes"][name] for item in right]
        result[name] = _comparison_entry(left_values, right_values, index_library=index_library)
    for name in ("total_work", "wasted_work", "storm_peak"):
        left_values = []
        right_values = []
        for left_item, right_item in zip(left, right):
            left_attr = left_item["record"]["attribution"]
            right_attr = right_item["record"]["attribution"]
            if name == "total_work":
                left_values.append(left_attr["work"]["total_executed_work"])
                right_values.append(right_attr["work"]["total_executed_work"])
            elif name == "wasted_work":
                left_values.append(left_attr["work"]["wasted_work"])
                right_values.append(right_attr["work"]["wasted_work"])
            else:
                left_values.append(left_attr["storm"]["peak_hedge_launch_rate"])
                right_values.append(right_attr["storm"]["peak_hedge_launch_rate"])
        result[name] = _comparison_entry(left_values, right_values, index_library=index_library)
    return result


def _default_episode_builder(config: ExperimentConfig) -> Callable[[int], EpisodeTrace]:
    def build(index: int) -> EpisodeTrace:
        return generate_episode_trace(
            config,
            THREE_ARM_NAMESPACE,
            THREE_ARM_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )

    return build


def run_three_arm_evaluation(
    config: ExperimentConfig,
    *,
    episode_count: int = THREE_ARM_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    candidate_policy: FrozenNIINPolicy | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= THREE_ARM_EPISODES:
        raise ThreeArmExecutionError("episode_count must be in 1..1024")
    max_calls = episode_count * THREE_ARM_COUNT
    if max_calls > HOLDOUT_CALL_LIMIT:
        raise ThreeArmExecutionError("three-arm call budget exceeded")
    policy_c = candidate_policy or load_requested_price_candidate()
    if not isinstance(policy_c, FrozenNIINPolicy) or not policy_c.supported_probabilities:
        raise ThreeArmExecutionError("requested-price candidate is incomplete")
    build = episode_builder or _default_episode_builder(config)
    sources = {"A": NoHedgePolicy(), "B": FrozenNIINPolicy(), "C": policy_c}
    public_prices = {"A": 0.0, "B": 0.0, "C": REQUESTED_RESERVATION_PRICE}
    records: dict[str, list[dict[str, object]]] = {name: [] for name in ARM_NAMES}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    attempted_calls = 0
    failed_episode_count = 0
    for index in range(episode_count):
        episode = build(index)
        if not isinstance(episode, EpisodeTrace):
            raise ThreeArmExecutionError("episode_builder returned a non-EpisodeTrace")
        trace_fp = _trace_digest(episode)
        if trace_fp in seen:
            raise ThreeArmExecutionError("duplicate episode trace fingerprint")
        seen.add(trace_fp)
        row = {
            "episode_index": index,
            "trace_fingerprints": {},
            "arms": {},
            "status": "completed",
        }
        failed = False
        appended = {arm: False for arm in ARM_NAMES}
        for arm in ARM_NAMES:
            attempted_calls += 1
            try:
                run = simulate_episode_online(
                    episode,
                    sources[arm],
                    _RESERVATION,
                    degraded_slowdown=DEGRADED_SLOWDOWN,
                    hedge_delay=HEDGE_DELAY,
                    public_price=public_prices[arm],
                )
                after = _trace_digest(episode)
                if after != trace_fp:
                    raise ThreeArmExecutionError(f"arm {arm} mutated the immutable trace")
                arm_record, _ = _episode_record(run)
                arm_record["run"] = run
                records[arm].append(arm_record)
                appended[arm] = True
                row["trace_fingerprints"][arm] = after
                row["arms"][arm] = arm_record["compact"]
            except Exception as error:
                failed = True
                row["arms"][arm] = {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
        if failed:
            failed_episode_count += 1
            row["status"] = "failed"
            for arm in ARM_NAMES:
                if appended[arm]:
                    records[arm].pop()
        rows.append(row)
        if progress is not None:
            progress(index + 1, episode_count)
    complete_count = episode_count - failed_episode_count
    if attempted_calls > HOLDOUT_CALL_LIMIT:
        raise ThreeArmExecutionError("actual calls exceeded the frozen limit")
    result: dict[str, object] = {
        "label": THREE_ARM_LABEL,
        "status": "physical_failed" if failed_episode_count else "completed",
        "namespace": THREE_ARM_NAMESPACE,
        "macro_seed": THREE_ARM_MACRO_SEED,
        "episode_count": episode_count,
        "complete_episode_count": complete_count,
        "failed_episode_count": failed_episode_count,
        "attempted_calls": attempted_calls,
        "max_scheduler_calls": max_calls,
        "protocol_call_limit": HOLDOUT_CALL_LIMIT,
        "arms": {name: _aggregate_arm(records[name]) for name in ARM_NAMES if records[name]},
        "candidate_policy": policy_c.to_dict(),
        "candidate_parent_fingerprint": policy_c.candidate_fingerprint,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "episode_rows": rows,
    }
    if complete_count:
        indices = _bootstrap_index_library(complete_count, BOOTSTRAP_REPLICATES)
        comparisons = {
            "C-B": _comparison(records, "B", "C", indices),
            "C-A": _comparison(records, "A", "C", indices),
            "B-A": _comparison(records, "A", "B", indices),
        }
        result["paired_statistics"] = {
            "bootstrap": {
                "namespace": BOOTSTRAP_NAMESPACE,
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "index_library_fingerprint": _bootstrap_fingerprint(indices),
                "scheduler_calls": 0,
            },
            "comparisons": comparisons,
        }
        c_b = comparisons["C-B"]
        result["gate_pass"] = (
            failed_episode_count == 0
            and evaluate_gate(c_b, invariants_pass=all(
                arm.get("invariants", {}).get("all_episode_invariants_pass", False)
                for arm in result["arms"].values()
            ))
        )
    else:
        result["paired_statistics"] = None
        result["gate_pass"] = False
    return result


def write_three_arm_artifact(
    artifacts_root: str | Path,
    run_id: str,
    result: Mapping[str, object],
    config: ExperimentConfig,
    *,
    source_bundle: Mapping[str, object] | None = None,
    parent_summary_sha256: str | None = None,
) -> Path:
    if result.get("label") != THREE_ARM_LABEL:
        raise ThreeArmExecutionError("unexpected three-arm result label")
    for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg"):
        if result.get(name) is not False:
            raise ThreeArmExecutionError(f"claim boundary {name} must be false")
    if int(result.get("attempted_calls", -1)) > HOLDOUT_CALL_LIMIT:
        raise ThreeArmExecutionError("artifact call count exceeds frozen limit")
    bundle = dict(source_bundle or {"schema_id": "unit-test-source-bundle", "fingerprint": "unavailable"})
    provenance = {
        "environment_id": ENVIRONMENT_ID,
        "environment_fingerprint": environment_fingerprint(config),
        "namespace": THREE_ARM_NAMESPACE,
        "macro_seed": THREE_ARM_MACRO_SEED,
        "episode_count": result.get("episode_count"),
        "attempted_calls": result.get("attempted_calls"),
        "max_scheduler_calls": HOLDOUT_CALL_LIMIT,
        "arms": {
            "A": "no_hedge_v1",
            "B": "pi_NIIN_v1",
            "C": "pi_requested_reservation_price_v1",
        },
        "candidate_parent_summary_sha256": parent_summary_sha256,
        "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_bundle": bundle,
    }
    manifest = {
        "schema_id": "token_three_arm_holdout_manifest_v1",
        "run_id": run_id,
        "label": THREE_ARM_LABEL,
        "provenance": provenance,
        "status": result.get("status"),
        "gate_pass": result.get("gate_pass", False),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    summary = {key: value for key, value in result.items() if key != "episode_rows"}
    summary["provenance"] = provenance
    metric_contract = {
        "schema_id": "token_three_arm_holdout_metrics_v1",
        "latency_phase_basis": "Token arrival phase; H/D/F/R and overall",
        "primary_endpoints": ["D/F pooled CVaR95", "D/F deadline miss rate"],
        "paired_cluster": "episode index",
        "bootstrap": {
            "namespace": BOOTSTRAP_NAMESPACE,
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    candidate_payload = dict(result["candidate_policy"])
    candidate_payload.update(
        {
            "parent_summary_sha256": parent_summary_sha256,
            "claims_best_response": False,
            "claims_regret": False,
            "claims_nash": False,
            "claims_mfg": False,
        }
    )
    paired = result.get("paired_statistics") or {"status": "unavailable"}
    return write_run_directory(
        artifacts_root,
        run_id,
        {
            "manifest.json": manifest,
            "summary.json": summary,
            "episode_rows.json": {"rows": result.get("episode_rows", [])},
            "paired_statistics.json": paired,
            "candidate_policy.json": candidate_payload,
            "metric_contract.json": metric_contract,
        },
    )


def execute_three_arm_evaluation(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    episode_count: int = THREE_ARM_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    source_bundle = build_source_bundle(project)
    parent_path = project / PARENT_SUMMARY
    candidate = load_requested_price_candidate(parent_path)
    result = run_three_arm_evaluation(
        config,
        episode_count=episode_count,
        candidate_policy=candidate,
        progress=progress,
    )
    directory = write_three_arm_artifact(
        artifacts_root,
        run_id,
        result,
        config,
        source_bundle=asdict(source_bundle),
        parent_summary_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    )
    return directory, result


__all__ = [
    "BOOTSTRAP_NAMESPACE",
    "BOOTSTRAP_REPLICATES",
    "FrozenNIINPolicy",
    "HOLDOUT_CALL_LIMIT",
    "NoHedgePolicy",
    "REQUESTED_PRICE_BETA",
    "REQUESTED_RESERVATION_PRICE",
    "THREE_ARM_LABEL",
    "ThreeArmExecutionError",
    "candidate_action_probabilities",
    "evaluate_gate",
    "execute_three_arm_evaluation",
    "load_requested_price_candidate",
    "paired_cluster_bootstrap",
    "run_three_arm_evaluation",
    "stable_policy_uniform",
    "write_three_arm_artifact",
]
