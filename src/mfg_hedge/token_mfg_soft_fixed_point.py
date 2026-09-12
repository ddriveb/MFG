"""T4b zero-price soft-response population diagnostic.

The physical engine remains the complete online Token simulator.  This module
only changes the population policy update from a hard argmin table to a
temperature-controlled N/D/I probability function.  It is not an equilibrium
solver and never emits Nash, regret, or MFG claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_run_directory
from .domain import ProtectionAction


T4B_LABEL = "finite_population_token_soft_response_fixed_point_diagnostic"
T4B_NAMESPACE = "token-mfg-restoration:t4b:soft-fixed-point:v1:library"
T4B_MACRO_SEED = 20260914
T4B_BETA = 1.0
T4B_ETA = 0.2
T4B_MAX_ROUNDS = 20
T4B_EPISODE_SLOTS = 2048
T4B_PANEL_FLOOR = 32
T4B_CALLS_PER_SLOT = 5
T4B_CALL_LIMIT_PER_ROUND = T4B_EPISODE_SLOTS * T4B_CALLS_PER_SLOT
T4B_CALL_LIMIT_TOTAL = T4B_MAX_ROUNDS * T4B_CALL_LIMIT_PER_ROUND
T4B_RESIDUAL_TOLERANCE = 0.01

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_ACTION_INDEX = {action: index for index, action in enumerate(_ACTIONS)}


class SoftFixedPointError(ValueError):
    """Raised for a fail-closed T4b input or accounting violation."""


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SoftFixedPointError("value is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SoftFixedPointError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SoftFixedPointError(f"{name} must be finite")
    return result


def _action(action: object) -> ProtectionAction:
    if not isinstance(action, ProtectionAction):
        raise SoftFixedPointError(f"invalid action: {action!r}")
    return action


def _probabilities(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise SoftFixedPointError("probability vector must contain N/D/I")
    result = tuple(_finite(value, "probability", nonnegative=True) for value in values)
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SoftFixedPointError("probabilities must sum to one")
    return result  # type: ignore[return-value]


def _one_hot(action: ProtectionAction) -> tuple[float, float, float]:
    return tuple(1.0 if item is action else 0.0 for item in _ACTIONS)  # type: ignore[return-value]


def soft_best_response(
    means_by_bin: Mapping[str, Mapping[ProtectionAction, float]],
    *,
    beta: float = T4B_BETA,
) -> dict[str, tuple[float, float, float]]:
    """Return a numerically stable N/D/I SoftBR for every supplied bin."""

    beta = _finite(beta, "beta")
    if beta <= 0.0:
        raise SoftFixedPointError("beta must be positive")
    if not isinstance(means_by_bin, Mapping):
        raise SoftFixedPointError("means_by_bin must be a mapping")
    result: dict[str, tuple[float, float, float]] = {}
    for bin_id, means in means_by_bin.items():
        if not isinstance(bin_id, str) or not bin_id or not isinstance(means, Mapping):
            raise SoftFixedPointError("SoftBR input has an invalid bin")
        costs = []
        for action in _ACTIONS:
            if action not in means:
                raise SoftFixedPointError(f"missing Q mean for {bin_id!r}/{action.value}")
            costs.append(_finite(means[action], "Q mean"))
        logits = tuple(-beta * cost for cost in costs)
        peak = max(logits)
        weights = tuple(math.exp(value - peak) for value in logits)
        total = math.fsum(weights)
        result[bin_id] = _probabilities(tuple(value / total for value in weights))
    return result


def _normalise_component(
    mapping: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if not isinstance(mapping, Mapping):
        raise SoftFixedPointError("soft component must be a mapping")
    rows = []
    for bin_id, values in mapping.items():
        if not isinstance(bin_id, str) or not bin_id:
            raise SoftFixedPointError("component bin IDs must be non-empty strings")
        rows.append((bin_id, _probabilities(values)))
    return tuple(sorted(rows))


@dataclass(frozen=True)
class SoftPolicyState:
    """Immutable NIIN-plus-soft-response function mixture."""

    base_weight: float = 1.0
    components: tuple[tuple[float, tuple[tuple[str, tuple[float, float, float]], ...]], ...] = ()

    def __post_init__(self) -> None:
        base_weight = _finite(self.base_weight, "base_weight", nonnegative=True)
        total = base_weight
        clean = []
        if not isinstance(self.components, tuple):
            raise SoftFixedPointError("components must be a tuple")
        for weight, mapping in self.components:
            weight = _finite(weight, "component weight", nonnegative=True)
            clean.append((weight, _normalise_component(dict(mapping))))
            total += weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SoftFixedPointError("policy component weights must sum to one")
        object.__setattr__(self, "base_weight", base_weight)
        object.__setattr__(self, "components", tuple(clean))

    @classmethod
    def initial(cls) -> "SoftPolicyState":
        return cls()

    def updated(
        self,
        soft_response: Mapping[str, Sequence[float]],
        *,
        eta: float = T4B_ETA,
    ) -> "SoftPolicyState":
        eta = _finite(eta, "eta")
        if not 0.0 < eta <= 1.0:
            raise SoftFixedPointError("eta must lie in (0, 1]")
        component = _normalise_component(soft_response)
        retained = tuple((weight * (1.0 - eta), mapping) for weight, mapping in self.components)
        return SoftPolicyState(self.base_weight * (1.0 - eta), retained + ((eta, component),))

    def probabilities_for(
        self,
        bin_id: str,
        *,
        base_action: ProtectionAction = ProtectionAction.NORMAL,
    ) -> tuple[float, float, float]:
        if not isinstance(bin_id, str) or not bin_id:
            raise SoftFixedPointError("bin_id must be non-empty")
        values = [self.base_weight * value for value in _one_hot(_action(base_action))]
        for weight, mapping in self.components:
            component = _one_hot(_action(base_action))
            for candidate_bin, candidate_probabilities in mapping:
                if candidate_bin == bin_id:
                    component = candidate_probabilities
                    break
            for index in range(3):
                values[index] += weight * component[index]
        return _probabilities(tuple(round(value, 15) for value in values))

    def fingerprint(self) -> str:
        return _digest({
            "base_policy": "pi_NIIN_v1",
            "base_weight": self.base_weight,
            "components": self.components,
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "base_policy": "pi_NIIN_v1",
            "base_weight": self.base_weight,
            "components": [
                {
                    "weight": weight,
                    "soft_response": [
                        {"bin_id": bin_id, "probabilities": list(probabilities)}
                        for bin_id, probabilities in mapping
                    ],
                }
                for weight, mapping in self.components
            ],
            "fingerprint": self.fingerprint(),
        }


class SoftPopulationPolicy:
    """Online action source sampled from a frozen probability function."""

    def __init__(self, state: SoftPolicyState | None = None, *, seed: int = T4B_MACRO_SEED):
        if state is not None and not isinstance(state, SoftPolicyState):
            raise SoftFixedPointError("state must be SoftPolicyState")
        if type(seed) is not int or seed < 0:
            raise SoftFixedPointError("seed must be a non-negative int")
        self.state = state or SoftPolicyState.initial()
        self.seed = seed

    @classmethod
    def initial(cls, *, seed: int = T4B_MACRO_SEED) -> "SoftPopulationPolicy":
        return cls(SoftPolicyState.initial(), seed=seed)

    def new_episode(self) -> "SoftPopulationPolicy":
        return SoftPopulationPolicy(self.state, seed=self.seed)

    def _base_action(self, observation: object) -> ProtectionAction:
        phase = getattr(getattr(observation, "phase", None), "value", None)
        primary = getattr(observation, "primary_replica", None)
        token_class = getattr(getattr(observation, "token_class", None), "value", None)
        age = getattr(observation, "phase_age", None)
        if phase == "D" and primary == 0 and (token_class == "U" or (isinstance(age, (int, float)) and age >= 50.0)):
            return ProtectionAction.IMMEDIATE_HEDGE
        return ProtectionAction.NORMAL

    def probabilities_for_observation(self, observation: object) -> tuple[float, float, float]:
        try:
            from .token_continuation import BIN_SCHEMA_V1
            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception:
            return _one_hot(self._base_action(observation))
        return self.state.probabilities_for(bin_id, base_action=self._base_action(observation))

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise SoftFixedPointError("policy_key must be non-empty")
        probabilities = self.probabilities_for_observation(observation)
        digest = hashlib.sha256(
            f"{self.seed}:{self.state.fingerprint()}:{policy_key}".encode("utf-8")
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, probabilities):
            cumulative += probability
            if draw < cumulative or action is _ACTIONS[-1]:
                return action
        raise AssertionError("unreachable action sample")


def paired_action_gap(
    panels: Mapping[tuple[str, ProtectionAction], Sequence[float]],
    bin_id: str,
) -> dict[str, object]:
    rows = []
    for action in _ACTIONS:
        values = tuple(_finite(value, "panel cost") for value in panels.get((bin_id, action), ()))
        if not values:
            raise SoftFixedPointError("paired gap requires a complete action panel")
        rows.append((math.fsum(values) / len(values), _ACTION_INDEX[action], action, values))
    rows.sort(key=lambda row: (row[0], row[1]))
    best_mean, _, best_action, best_values = rows[0]
    runner_mean, _, runner_action, runner_values = rows[1]
    if len(best_values) != len(runner_values):
        raise SoftFixedPointError("paired action samples have different effective n")
    differences = tuple(runner - best for runner, best in zip(runner_values, best_values))
    mean_difference = math.fsum(differences) / len(differences)
    variance = 0.0 if len(differences) == 1 else math.fsum((value - mean_difference) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    return {
        "best_action": best_action,
        "runner_up": runner_action,
        "gap": runner_mean - best_mean,
        "paired_standard_error": standard_error,
        "effective_n": len(differences),
        "gap_over_paired_se": (
            None if standard_error == 0.0
            else (runner_mean - best_mean) / standard_error
        ),
    }


def episode_library_fingerprint(library: Sequence[object]) -> str:
    if not isinstance(library, tuple) or not library:
        raise SoftFixedPointError("episode library must be a non-empty tuple")
    rows = []
    from .attribution_episode import episode_trace_fingerprint
    for episode in library:
        if isinstance(episode, str):
            rows.append(episode)
        else:
            rows.append(episode_trace_fingerprint(episode))
    return _digest(tuple(rows))


def validate_episode_library(library: Sequence[object], fingerprint: str) -> tuple[object, ...]:
    if not isinstance(library, tuple) or not library:
        raise SoftFixedPointError("episode library must be a non-empty tuple")
    if not isinstance(fingerprint, str) or episode_library_fingerprint(library) != fingerprint:
        raise SoftFixedPointError("episode library fingerprint mismatch")
    return library


def soft_stop_status(
    policy_residual: float,
    population_residual: float,
    *,
    comparable_rounds: int,
    tolerance: float = T4B_RESIDUAL_TOLERANCE,
) -> str | None:
    policy_residual = _finite(policy_residual, "policy_residual", nonnegative=True)
    population_residual = _finite(population_residual, "population_residual", nonnegative=True)
    if type(comparable_rounds) is not int or comparable_rounds < 0:
        raise SoftFixedPointError("comparable_rounds must be a non-negative int")
    if comparable_rounds >= 2 and policy_residual <= tolerance and population_residual <= tolerance:
        return "soft_fixed_point"
    return None


class T4BCallLedger:
    def __init__(self, maximum: int):
        if type(maximum) is not int or maximum <= 0:
            raise SoftFixedPointError("maximum calls must be positive int")
        self.maximum = maximum
        self.reserved = 0
        self.attempted = 0

    def reserve(self, count: int) -> None:
        if type(count) is not int or count <= 0 or self.reserved + count > self.maximum:
            raise SoftFixedPointError("call reservation exceeds budget")
        self.reserved += count

    def start(self, count: int = 1) -> None:
        if type(count) is not int or count < 0 or self.attempted + count > self.reserved:
            raise SoftFixedPointError("call settlement exceeds reservation")
        self.attempted += count


def _json_value(value: object) -> object:
    if isinstance(value, ProtectionAction):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_value(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


def write_soft_fixed_point_artifact(
    artifacts_root: str | Path,
    run_id: str,
    summary: Mapping[str, object],
) -> Path:
    return write_run_directory(
        artifacts_root,
        run_id,
        {"summary.json": _json_value(summary)},
    )


class _CaptureSource:
    def __init__(self, source: object, selector: object):
        self.source = source
        self.selector = selector
        self.target_id = None
        self.observation = None

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if self.target_id is None and self.selector.select(observation):
            self.target_id = observation.token_id
            self.observation = observation
        return _action(self.source.choose(observation, policy_key))


def _observation_bin(observation: object) -> str:
    from .token_continuation import BIN_SCHEMA_V1
    return BIN_SCHEMA_V1.bin(observation).bin_id


def _protocol_fingerprint(protocol: object) -> str:
    return _digest({
        "arrival_cutoff": protocol.arrival_cutoff,
        "timeline": {
            "degraded_start": protocol.timeline.degraded_start,
            "failed_start": protocol.timeline.failed_start,
            "recovered_start": protocol.timeline.recovered_start,
        },
    })


def _run_round(
    config: object,
    library: tuple[object, ...],
    library_fp: str,
    retained_bins: tuple[str, ...],
    policy: SoftPopulationPolicy,
    *,
    iteration: int,
    minimum_complete_panels: int,
    max_scheduler_calls: int,
) -> dict[str, object]:
    from .token_deviations import evaluate_token_pathwise_deviations
    from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
    from .token_t3a_execution import T3AAnchorSelector
    from .token_online import simulate_episode_online
    from .transient_control import ReservationParameters

    if len(library) * T4B_CALLS_PER_SLOT > max_scheduler_calls:
        raise SoftFixedPointError("round call budget is too small")
    reservation = ReservationParameters(window_width=25.0, budget_rate=0.45, scale=0.25, mean_requirement=1.0)
    parameters = TokenRuntimeADR0009Parameters()
    quote = ExternalQuote(0.0, "incremental_executed_work")
    ledger = T4BCallLedger(max_scheduler_calls)
    samples: dict[tuple[str, ProtectionAction], list[float]] = {}
    counts = {bin_id: 0 for bin_id in retained_bins}
    counts.update({"missing_target": 0, "other_bin": 0, "selection_failed": 0})
    observations = []
    episode_rows = []
    for index, episode in enumerate(validate_episode_library(library, library_fp)):
        ledger.reserve(1)
        try:
            capture = _CaptureSource(policy.new_episode(), T3AAnchorSelector(index % 4))
            simulate_episode_online(episode, capture, reservation, degraded_slowdown=2.0, hedge_delay=2.0)
            ledger.start(1)
        except Exception as error:
            ledger.start(1)
            counts["selection_failed"] += 1
            episode_rows.append({"episode_index": index, "status": "selection_failed", "attempted_calls": 1, "reason": f"{type(error).__name__}: {error}"})
            continue
        if capture.target_id is None or capture.observation is None:
            counts["missing_target"] += 1
            episode_rows.append({"episode_index": index, "status": "missing_target", "attempted_calls": 1})
            continue
        try:
            bin_id = _observation_bin(capture.observation)
        except Exception as error:
            counts["other_bin"] += 1
            episode_rows.append({"episode_index": index, "status": "missing_bin", "attempted_calls": 1, "reason": f"{type(error).__name__}: {error}"})
            continue
        counts[bin_id if bin_id in counts else "other_bin"] += 1
        observations.append(capture.observation)
        ledger.reserve(4)
        try:
            result = evaluate_token_pathwise_deviations(
                episode,
                policy.new_episode,
                capture.target_id,
                candidates=_ACTIONS,
                parameters=parameters,
                quote=quote,
                reservation=reservation,
                degraded_slowdown=2.0,
                hedge_delay=2.0,
            )
            actual = result.attempted_calls
            ledger.start(actual)
            if result.status != "completed" or len(result.rows) != 3:
                episode_rows.append({"episode_index": index, "status": "panel_failed", "bin_id": bin_id, "attempted_calls": 1 + actual, "reason": result.failure_reason or "incomplete paired panel"})
                continue
            if bin_id in retained_bins:
                for row in result.rows:
                    samples.setdefault((bin_id, row.candidate_requested), []).append(float(row.candidate_payoff.total))
            episode_rows.append({"episode_index": index, "status": "completed" if bin_id in retained_bins else "other_bin", "bin_id": bin_id, "attempted_calls": 1 + actual})
        except Exception as error:
            ledger.start(4)
            episode_rows.append({"episode_index": index, "status": "panel_failed", "bin_id": bin_id, "attempted_calls": 5, "reason": f"{type(error).__name__}: {error}"})

    q_rows = []
    means: dict[str, dict[ProtectionAction, float]] = {}
    incomplete = False
    for bin_id in retained_bins:
        means[bin_id] = {}
        for action in _ACTIONS:
            values = tuple(samples.get((bin_id, action), ()))
            n = len(values)
            if n < minimum_complete_panels:
                incomplete = True
                q_rows.append({"bin_id": bin_id, "requested_action": action.value, "mean": None, "standard_error": None, "effective_n": n, "status": "insufficient"})
                continue
            mean = math.fsum(values) / n
            variance = 0.0 if n == 1 else math.fsum((value - mean) ** 2 for value in values) / (n - 1)
            means[bin_id][action] = mean
            q_rows.append({"bin_id": bin_id, "requested_action": action.value, "mean": mean, "standard_error": math.sqrt(variance / n), "effective_n": n, "status": "completed"})
    gap_rows = []
    if not incomplete:
        for bin_id in retained_bins:
            gap = paired_action_gap(samples, bin_id)
            gap_rows.append({"bin_id": bin_id, **_json_value(gap)})
    return {
        "iteration": iteration,
        "policy_before": policy.state.to_dict(),
        "policy_before_fingerprint": policy.state.fingerprint(),
        "library_fingerprint": library_fp,
        "q_rows": q_rows,
        "action_gap_rows": gap_rows,
        "occupancy_counts": counts,
        "occupancy_vector": list(_occupancy_vector(counts, retained_bins + ("missing_target", "other_bin", "selection_failed"))),
        "episode_rows": episode_rows,
        "observations": observations,
        "q_status": "insufficient" if incomplete else "completed",
        "reserved_calls": ledger.reserved,
        "attempted_calls": ledger.attempted,
        "means": means,
    }


def _occupancy_vector(counts: Mapping[str, int], keys: Sequence[str]) -> tuple[float, ...]:
    total = sum(counts.values())
    return tuple(0.0 for _ in keys) if total == 0 else tuple(counts.get(key, 0) / total for key in keys)


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise SoftFixedPointError("residual vectors differ in length")
    return math.fsum(abs(a - b) for a, b in zip(left, right))


def _hard_br_fingerprint(means: Mapping[str, Mapping[ProtectionAction, float]]) -> str:
    rows = []
    for bin_id, values in sorted(means.items()):
        action = min((_finite(values[item], "Q mean"), _ACTION_INDEX[item], item) for item in _ACTIONS)[2]
        rows.append((bin_id, action.value))
    return _digest(tuple(rows))


def run_soft_fixed_point(
    config: object,
    retained_bins: Sequence[str],
    *,
    episode_count: int = T4B_EPISODE_SLOTS,
    minimum_complete_panels: int = T4B_PANEL_FLOOR,
    max_rounds: int = T4B_MAX_ROUNDS,
    beta: float = T4B_BETA,
    eta: float = T4B_ETA,
) -> dict[str, object]:
    from .attribution_episode import (
        ATTRIBUTION_V1_PROTOCOL,
        episode_trace_fingerprint,
        generate_episode_trace,
    )
    from .token_t3a_execution import validate_frozen_config

    validate_frozen_config(config)
    bins = tuple(retained_bins)
    if not bins or tuple(sorted(set(bins))) != bins:
        raise SoftFixedPointError("retained_bins must be sorted and unique")
    if type(episode_count) is not int or episode_count <= 0 or episode_count > T4B_EPISODE_SLOTS:
        raise SoftFixedPointError("episode_count is outside 1..2048")
    if type(max_rounds) is not int or max_rounds <= 0 or max_rounds > T4B_MAX_ROUNDS:
        raise SoftFixedPointError("max_rounds is outside 1..20")
    _finite(beta, "beta")
    _finite(eta, "eta")
    library = tuple(
        generate_episode_trace(config, T4B_NAMESPACE, T4B_MACRO_SEED, index, ATTRIBUTION_V1_PROTOCOL)
        for index in range(episode_count)
    )
    library_fp = episode_library_fingerprint(library)
    policy = SoftPopulationPolicy.initial(seed=T4B_MACRO_SEED)
    iterations = []
    previous_occupancy = None
    previous_policy = None
    final_status = "not_converged_20"
    final_reason = "maximum iteration count reached"
    for iteration in range(max_rounds):
        result = _run_round(
            config, library, library_fp, bins, policy,
            iteration=iteration,
            minimum_complete_panels=minimum_complete_panels,
            max_scheduler_calls=T4B_CALL_LIMIT_PER_ROUND,
        )
        observations = result.pop("observations")
        result["observed_target_count"] = len(observations)
        if result["q_status"] != "completed":
            result["status"] = "statistics_insufficient"
            result["stop_reason"] = "one or more retained bin/action panels below floor"
            iterations.append(result)
            final_status = "statistics_insufficient"
            final_reason = str(result["stop_reason"])
            break
        soft_response = soft_best_response(result["means"], beta=beta)
        next_policy = SoftPopulationPolicy(SoftPolicyState(policy.state.base_weight, policy.state.components).updated(soft_response, eta=eta), seed=T4B_MACRO_SEED)
        current_occupancy = tuple(result["occupancy_vector"])
        policy_residual = max(
            _l1(policy.probabilities_for_observation(observation), next_policy.probabilities_for_observation(observation))
            for observation in observations
        )
        population_residual = None if previous_occupancy is None else _l1(previous_occupancy, current_occupancy)
        hard_fp = _hard_br_fingerprint(result["means"])
        result.update({
            "soft_response": {bin_id: list(values) for bin_id, values in soft_response.items()},
            "hard_br_fingerprint": hard_fp,
            "policy_after": next_policy.state.to_dict(),
            "policy_after_fingerprint": next_policy.state.fingerprint(),
            "policy_probability_l1_residual": policy_residual,
            "population_occupancy_l1_residual": population_residual,
            "status": "completed",
            "stop_reason": "updated soft probability function",
        })
        stop = None if population_residual is None else soft_stop_status(policy_residual, population_residual, comparable_rounds=2)
        iterations.append(result)
        if stop is not None:
            iterations[-1]["status"] = stop
            iterations[-1]["stop_reason"] = "policy and population residuals within tolerance"
            final_status = stop
            final_reason = str(iterations[-1]["stop_reason"])
            break
        previous_occupancy = current_occupancy
        previous_policy = policy
        policy = next_policy
    total_reserved = sum(int(row["reserved_calls"]) for row in iterations)
    total_attempted = sum(int(row["attempted_calls"]) for row in iterations)
    return {
        "evidence_label": T4B_LABEL,
        "status": final_status,
        "stop_reason": final_reason,
        "environment_id": "theta_token_load0p7_v1",
        "policy_id": "pi_T4b_soft_response_mixture_v1",
        "beta": beta,
        "eta": eta,
        "macro_seed": T4B_MACRO_SEED,
        "namespace": T4B_NAMESPACE,
        "episode_count": episode_count,
        "panel_floor": minimum_complete_panels,
        "max_rounds": max_rounds,
        "library_fingerprint": library_fp,
        "library_rows": [
            {
                "episode_index": index,
                "fingerprint": _digest(episode_trace_fingerprint(episode)),
            }
            for index, episode in enumerate(library)
        ],
        "retained_bins": list(bins),
        "iterations": iterations,
        "attempted_calls": total_attempted,
        "reserved_calls": total_reserved,
        "call_limit_per_round": T4B_CALL_LIMIT_PER_ROUND,
        "call_limit_total": T4B_CALL_LIMIT_TOTAL,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def execute_soft_fixed_point(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    episode_count: int = T4B_EPISODE_SLOTS,
    minimum_complete_panels: int = T4B_PANEL_FLOOR,
    max_rounds: int = T4B_MAX_ROUNDS,
) -> Path:
    from .config import load_config
    from .token_best_response import load_t3a_runtime_candidate
    from .token_t3a_execution import build_source_bundle, environment_fingerprint
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    parent = load_t3a_runtime_candidate(
        project / "artifacts" / "token-t3a-oracle-20260907-r1-validation",
        project / "artifacts" / "token-t3a-oracle-20260907-r1-occupancy",
    )
    result = run_soft_fixed_point(config, parent.retained_bins, episode_count=episode_count, minimum_complete_panels=minimum_complete_panels, max_rounds=max_rounds)
    manifest = {
        "schema_id": "token_t4b_soft_fixed_point_manifest_v1",
        "run_id": run_id,
        "evidence_label": T4B_LABEL,
        "source_bundle": asdict(bundle),
        "environment_fingerprint": environment_fingerprint(config),
        "protocol_fingerprint": _protocol_fingerprint(ATTRIBUTION_V1_PROTOCOL),
        "library_fingerprint": result["library_fingerprint"],
        "namespace": T4B_NAMESPACE,
        "macro_seed": T4B_MACRO_SEED,
        "beta": T4B_BETA,
        "eta": T4B_ETA,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    metric_contract = {
        "schema_id": "token_t4b_metric_contract_v1",
        "soft_response": "exp(-beta*Q)/sum(exp(-beta*Q))",
        "beta": T4B_BETA,
        "eta": T4B_ETA,
        "residual_tolerance": T4B_RESIDUAL_TOLERANCE,
        "stop_statuses": ["soft_fixed_point", "not_converged_20", "statistics_insufficient"],
        "hard_br_repeat_is_diagnostic_only": True,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    q_rows = {
        "rows": [
            {"iteration": iteration["iteration"], **row}
            for iteration in result["iterations"]
            for row in iteration["q_rows"]
        ]
    }
    files = {
        "manifest.json": manifest,
        "metric_contract.json": metric_contract,
        "library.json": {"rows": result["library_rows"], "fingerprint": result["library_fingerprint"]},
        "policy_iterations.json": {"rows": result["iterations"]},
        "q_rows.json": q_rows,
        "summary.json": result,
    }
    return write_run_directory(
        artifacts_root,
        run_id,
        {name: _json_value(payload) for name, payload in files.items()},
    )


__all__ = [
    "SoftFixedPointError",
    "SoftPolicyState",
    "SoftPopulationPolicy",
    "T4BCallLedger",
    "T4B_LABEL",
    "T4B_CALL_LIMIT_PER_ROUND",
    "T4B_CALL_LIMIT_TOTAL",
    "episode_library_fingerprint",
    "execute_soft_fixed_point",
    "paired_action_gap",
    "run_soft_fixed_point",
    "soft_best_response",
    "soft_stop_status",
    "validate_episode_library",
    "write_soft_fixed_point_artifact",
]
