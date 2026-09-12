"""Damped finite-population Token response diagnostic (T4-v1).

This module keeps the T3A physical evaluator as the source of truth.  The
policy layer is an online mixture of functions; it never stores or replays an
action trajectory.  Results from this module are diagnostic population-policy
iterations only and do not assert best response, regret, Nash, or MFG.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_run_directory
from .domain import ProtectionAction


T4_LABEL = "finite_population_token_response_fixed_point_diagnostic"
T4_NAMESPACE = "token-mfg-restoration:t4:fixed-point:v1"
T4_MACRO_SEED = 20260913
T4_ETA = 0.2
T4_MAX_ROUNDS = 10
T4_EPISODE_SLOTS = 2048
T4_PANEL_FLOOR = 32
T4_CALLS_PER_SLOT = 5
T4_CALL_LIMIT_PER_ROUND = T4_EPISODE_SLOTS * T4_CALLS_PER_SLOT
T4_CALL_LIMIT_TOTAL = T4_MAX_ROUNDS * T4_CALL_LIMIT_PER_ROUND
T4_RESIDUAL_TOLERANCE = 0.01

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_ACTION_INDEX = {action: index for index, action in enumerate(_ACTIONS)}


class FixedPointDiagnosticError(ValueError):
    """Raised for a fail-closed T4 contract violation."""


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FixedPointDiagnosticError("payload is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedPointDiagnosticError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise FixedPointDiagnosticError(f"{name} must be finite")
    return result


def _validate_action(action: object) -> ProtectionAction:
    if not isinstance(action, ProtectionAction):
        raise FixedPointDiagnosticError(f"invalid ProtectionAction: {action!r}")
    return action


def _validate_probability_vector(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise FixedPointDiagnosticError("action probabilities must have N/D/I length 3")
    result = tuple(_finite(value, "action probability") for value in values)
    if any(value < 0.0 for value in result) or not math.isclose(
        math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FixedPointDiagnosticError("action probabilities must be non-negative and sum to one")
    return result  # type: ignore[return-value]


def _one_hot(action: ProtectionAction) -> tuple[float, float, float]:
    _validate_action(action)
    return tuple(1.0 if item is action else 0.0 for item in _ACTIONS)  # type: ignore[return-value]


def _normalise_component(mapping: Mapping[str, ProtectionAction]) -> tuple[tuple[str, str], ...]:
    if not isinstance(mapping, Mapping):
        raise FixedPointDiagnosticError("BR component must be a mapping")
    rows = []
    for bin_id, action in mapping.items():
        if not isinstance(bin_id, str) or not bin_id:
            raise FixedPointDiagnosticError("BR bin IDs must be non-empty strings")
        rows.append((bin_id, _validate_action(action).value))
    return tuple(sorted(rows))


@dataclass(frozen=True)
class DampedPolicyState:
    """Immutable function-mixture state, with NIIN as the base component."""

    base_weight: float = 1.0
    components: tuple[tuple[float, tuple[tuple[str, str], ...]], ...] = ()

    def __post_init__(self) -> None:
        base = _finite(self.base_weight, "base_weight", nonnegative=True)
        if not isinstance(self.components, tuple):
            raise FixedPointDiagnosticError("components must be a tuple")
        total = base
        clean = []
        for weight, mapping in self.components:
            weight = _finite(weight, "component weight", nonnegative=True)
            if not isinstance(mapping, tuple):
                raise FixedPointDiagnosticError("component mapping must be a tuple")
            for bin_id, action_value in mapping:
                if not isinstance(bin_id, str) or not bin_id:
                    raise FixedPointDiagnosticError("component bin ID is invalid")
                _validate_action(ProtectionAction(action_value))
            clean.append((weight, tuple(mapping)))
            total += weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise FixedPointDiagnosticError(f"policy component weights must sum to one, got {total}")
        object.__setattr__(self, "base_weight", base)
        object.__setattr__(self, "components", tuple(clean))

    @classmethod
    def initial(cls) -> "DampedPolicyState":
        return cls()

    def updated(
        self,
        best_response: Mapping[str, ProtectionAction],
        *,
        eta: float = T4_ETA,
    ) -> "DampedPolicyState":
        eta = _finite(eta, "eta")
        if not 0.0 < eta <= 1.0:
            raise FixedPointDiagnosticError("eta must lie in (0, 1]")
        component = _normalise_component(best_response)
        retained = tuple((weight * (1.0 - eta), mapping) for weight, mapping in self.components)
        return DampedPolicyState(
            self.base_weight * (1.0 - eta), retained + ((eta, component),)
        )

    def _base_action(self, observation: object | None = None) -> ProtectionAction:
        # NIIN is the frozen T3A policy: I for degraded Primary-A urgent/late
        # cells and N elsewhere.  Keeping this function local avoids storing
        # an action trajectory in the policy state.
        if observation is not None:
            phase = getattr(observation, "phase", None)
            primary = getattr(observation, "primary_replica", None)
            token_class = getattr(observation, "token_class", None)
            age = getattr(observation, "phase_age", None)
            if getattr(phase, "value", phase) == "D" and primary == 0:
                urgent = getattr(token_class, "value", token_class) == "U"
                if urgent or (isinstance(age, (int, float)) and age >= 50.0):
                    return ProtectionAction.IMMEDIATE_HEDGE
        return ProtectionAction.NORMAL

    def probabilities_for(
        self,
        bin_id: str,
        *,
        base_action: ProtectionAction = ProtectionAction.NORMAL,
    ) -> tuple[float, float, float]:
        if not isinstance(bin_id, str) or not bin_id:
            raise FixedPointDiagnosticError("bin_id must be a non-empty string")
        values = [self.base_weight * value for value in _one_hot(base_action)]
        for weight, mapping in self.components:
            # A BR component is defined only on the retained bins.  Outside
            # that support the frozen contract keeps the NIIN base function;
            # it must not silently turn an urgent/late NIIN hedge into N.
            action = base_action
            for candidate_bin, candidate_action in mapping:
                if candidate_bin == bin_id:
                    action = ProtectionAction(candidate_action)
                    break
            one_hot = _one_hot(action)
            for index in range(3):
                values[index] += weight * one_hot[index]
        # Canonicalize the public mixture probabilities so repeated damping
        # has a stable serialized value (the stored component weights remain
        # the exact immutable audit state).
        return _validate_probability_vector(tuple(round(value, 15) for value in values))

    def fingerprint(self) -> str:
        return _digest(
            {
                "base_weight": self.base_weight,
                "components": self.components,
                "base_policy": "pi_NIIN_v1",
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "base_policy": "pi_NIIN_v1",
            "base_weight": self.base_weight,
            "components": [
                {
                    "weight": weight,
                    "best_response": [
                        {"bin_id": bin_id, "action": action}
                        for bin_id, action in mapping
                    ],
                }
                for weight, mapping in self.components
            ],
            "fingerprint": self.fingerprint(),
        }


class MixedPopulationPolicy:
    """Fresh-state online action source for a frozen damped function mixture."""

    def __init__(self, state: DampedPolicyState | None = None, *, seed: int = T4_MACRO_SEED):
        if state is not None and not isinstance(state, DampedPolicyState):
            raise FixedPointDiagnosticError("state must be DampedPolicyState")
        if type(seed) is not int or seed < 0:
            raise FixedPointDiagnosticError("seed must be a non-negative int")
        self.state = state or DampedPolicyState.initial()
        self.seed = seed

    @classmethod
    def initial(cls, *, seed: int = T4_MACRO_SEED) -> "MixedPopulationPolicy":
        return cls(DampedPolicyState.initial(), seed=seed)

    def new_episode(self) -> "MixedPopulationPolicy":
        return MixedPopulationPolicy(self.state, seed=self.seed)

    def _bin_and_base(self, observation: object) -> tuple[str | None, ProtectionAction]:
        try:
            from .token_continuation import BIN_SCHEMA_V1

            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception:
            return None, self.state._base_action(observation)
        return bin_id, self.state._base_action(observation)

    def probabilities_for_observation(self, observation: object) -> tuple[float, float, float]:
        bin_id, base_action = self._bin_and_base(observation)
        if bin_id is None:
            return _one_hot(base_action)
        return self.state.probabilities_for(bin_id, base_action=base_action)

    def action_for_bin(self, bin_id: str) -> ProtectionAction:
        probabilities = self.state.probabilities_for(bin_id)
        return _ACTIONS[max(range(3), key=lambda index: probabilities[index])]

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise FixedPointDiagnosticError("policy_key must be non-empty")
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
        raise AssertionError("unreachable action sampling path")


def build_best_response(
    means: Mapping[tuple[str, ProtectionAction], float],
    bin_ids: Sequence[str],
) -> dict[str, ProtectionAction]:
    if not isinstance(means, Mapping):
        raise FixedPointDiagnosticError("means must be a mapping")
    output: dict[str, ProtectionAction] = {}
    for bin_id in bin_ids:
        if not isinstance(bin_id, str) or not bin_id:
            raise FixedPointDiagnosticError("bin_ids must be non-empty strings")
        candidates = []
        for action in _ACTIONS:
            key = (bin_id, action)
            if key not in means:
                raise FixedPointDiagnosticError(f"missing Q mean for {key!r}")
            value = _finite(means[key], f"Q mean {key!r}")
            candidates.append((value, _ACTION_INDEX[action], action))
        output[bin_id] = min(candidates, key=lambda row: (row[0], row[1]))[2]
    return output


def summarize_q_rows(
    samples: Mapping[tuple[str, ProtectionAction], Sequence[float]],
    bin_ids: Sequence[str],
    *,
    minimum_complete_panels: int = T4_PANEL_FLOOR,
) -> dict[tuple[str, ProtectionAction], dict[str, object]]:
    if type(minimum_complete_panels) is not int or minimum_complete_panels <= 0:
        raise FixedPointDiagnosticError("minimum_complete_panels must be positive int")
    output = {}
    incomplete = []
    for bin_id in bin_ids:
        for action in _ACTIONS:
            values = tuple(_finite(value, "Q sample") for value in samples.get((bin_id, action), ()))
            count = len(values)
            if count < minimum_complete_panels:
                incomplete.append((bin_id, action.value, count))
                output[(bin_id, action)] = {
                    "bin_id": bin_id,
                    "requested_action": action.value,
                    "mean": None,
                    "standard_error": None,
                    "effective_n": count,
                    "status": "insufficient",
                }
                continue
            mean = math.fsum(values) / count
            variance = 0.0 if count == 1 else math.fsum((value - mean) ** 2 for value in values) / (count - 1)
            output[(bin_id, action)] = {
                "bin_id": bin_id,
                "requested_action": action.value,
                "mean": mean,
                "standard_error": math.sqrt(variance / count),
                "effective_n": count,
                "status": "completed",
            }
    if incomplete:
        raise FixedPointDiagnosticError(f"statistics_insufficient: {incomplete}")
    return output


def classify_br_history(history: Sequence[str]) -> tuple[str | None, int | None]:
    seen: dict[str, int] = {}
    for index, fingerprint in enumerate(history):
        if fingerprint in seen and index - seen[fingerprint] > 1:
            return "cycle", index - seen[fingerprint]
        seen[fingerprint] = index
    return None, None


class T4CallLedger:
    def __init__(self, maximum: int):
        if type(maximum) is not int or maximum <= 0:
            raise FixedPointDiagnosticError("maximum calls must be a positive int")
        self.maximum = maximum
        self.reserved = 0
        self.attempted = 0

    def reserve(self, count: int) -> None:
        if type(count) is not int or count <= 0 or self.reserved + count > self.maximum:
            raise FixedPointDiagnosticError("scheduler call reservation exceeds budget")
        self.reserved += count

    def start(self, count: int = 1) -> None:
        if type(count) is not int or count < 0 or self.attempted + count > self.reserved:
            raise FixedPointDiagnosticError("scheduler call settlement exceeds reservation")
        self.attempted += count


def write_fixed_point_artifact(
    artifacts_root: str | Path,
    run_id: str,
    summary: Mapping[str, object],
) -> Path:
    if not isinstance(summary, Mapping):
        raise FixedPointDiagnosticError("summary must be a mapping")
    return write_run_directory(
        artifacts_root,
        run_id,
        {
            "summary.json": dict(summary),
        },
    )


class _TargetCaptureSource:
    """Capture one causal target while delegating every action online."""

    def __init__(self, source: object, selector: object) -> None:
        if not callable(getattr(source, "choose", None)):
            raise FixedPointDiagnosticError("policy source must provide choose")
        if not callable(getattr(selector, "select", None)):
            raise FixedPointDiagnosticError("target selector must provide select")
        self.source = source
        self.selector = selector
        self.target_id: int | None = None
        self.observation: object | None = None

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if self.target_id is None and self.selector.select(observation):
            self.target_id = observation.token_id
            self.observation = observation
        action = self.source.choose(observation, policy_key)
        return _validate_action(action)


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
        return {
            name: _json_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return value


def _observation_bin(observation: object) -> str:
    from .token_continuation import BIN_SCHEMA_V1

    return BIN_SCHEMA_V1.bin(observation).bin_id


def _policy_fingerprint(policy: MixedPopulationPolicy) -> str:
    return _digest(
        {
            "state": policy.state.to_dict(),
            "seed": policy.seed,
            "policy_id": "pi_T4_damped_function_mixture_v1",
        }
    )


def _q_rows(
    samples: Mapping[tuple[str, ProtectionAction], Sequence[float]],
    bin_ids: Sequence[str],
    minimum_complete_panels: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for bin_id in bin_ids:
        for action in _ACTIONS:
            values = tuple(_finite(value, "Q sample") for value in samples.get((bin_id, action), ()))
            count = len(values)
            if count < minimum_complete_panels:
                rows.append({
                    "bin_id": bin_id,
                    "requested_action": action.value,
                    "mean": None,
                    "standard_error": None,
                    "effective_n": count,
                    "status": "insufficient",
                })
                continue
            mean = math.fsum(values) / count
            variance = 0.0 if count == 1 else math.fsum((value - mean) ** 2 for value in values) / (count - 1)
            rows.append({
                "bin_id": bin_id,
                "requested_action": action.value,
                "mean": mean,
                "standard_error": math.sqrt(variance / count),
                "effective_n": count,
                "status": "completed",
            })
    return tuple(rows)


def _mean_map(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, ProtectionAction], float]:
    output = {}
    for row in rows:
        if row.get("status") != "completed" or row.get("mean") is None:
            raise FixedPointDiagnosticError("cannot construct BR from incomplete Q rows")
        output[(str(row["bin_id"]), ProtectionAction(str(row["requested_action"])))]=float(row["mean"])
    return output


def _vector(counts: Mapping[str, int], keys: Sequence[str]) -> tuple[float, ...]:
    total = sum(counts.values())
    if total <= 0:
        return tuple(0.0 for _ in keys)
    return tuple(counts.get(key, 0) / total for key in keys)


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise FixedPointDiagnosticError("residual vectors have different lengths")
    return math.fsum(abs(a - b) for a, b in zip(left, right))


def _br_fingerprint(best_response: Mapping[str, ProtectionAction]) -> str:
    return _digest(tuple((key, best_response[key].value) for key in sorted(best_response)))


def _capture_target(episode: object, policy: MixedPopulationPolicy, selector: object, reservation: object) -> tuple[int | None, object | None]:
    from .token_online import simulate_episode_online

    capture = _TargetCaptureSource(policy.new_episode(), selector)
    simulate_episode_online(
        episode,
        capture,
        reservation,
        degraded_slowdown=2.0,
        hedge_delay=2.0,
    )
    return capture.target_id, capture.observation


def _run_iteration(
    config: object,
    retained_bins: tuple[str, ...],
    policy: MixedPopulationPolicy,
    *,
    iteration: int,
    episode_count: int,
    minimum_complete_panels: int,
    max_scheduler_calls: int,
) -> dict[str, object]:
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL, generate_episode_trace
    from .token_deviations import evaluate_token_pathwise_deviations
    from .token_t3a_execution import T3AAnchorSelector
    from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
    from .transient_control import ReservationParameters

    if type(episode_count) is not int or episode_count <= 0:
        raise FixedPointDiagnosticError("episode_count must be positive int")
    if episode_count * T4_CALLS_PER_SLOT > max_scheduler_calls:
        raise FixedPointDiagnosticError("iteration call budget is too small")
    namespace = f"{T4_NAMESPACE}:iteration:{iteration}"
    reservation = ReservationParameters(
        window_width=25.0, budget_rate=0.45, scale=0.25, mean_requirement=1.0
    )
    parameters = TokenRuntimeADR0009Parameters()
    quote = ExternalQuote(0.0, "incremental_executed_work")
    ledger = T4CallLedger(max_scheduler_calls)
    samples: dict[tuple[str, ProtectionAction], list[float]] = {}
    counts = {bin_id: 0 for bin_id in retained_bins}
    counts.update({"missing_target": 0, "other_bin": 0, "selection_failed": 0})
    observations: list[object] = []
    panel_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    for index in range(episode_count):
        episode = generate_episode_trace(
            config, namespace, T4_MACRO_SEED, index, ATTRIBUTION_V1_PROTOCOL
        )
        ledger.reserve(1)
        try:
            selector = T3AAnchorSelector(index % 4)
            target_id, target_observation = _capture_target(episode, policy, selector, reservation)
            ledger.start(1)
        except Exception as error:
            ledger.start(1)
            counts["selection_failed"] += 1
            episode_rows.append({
                "episode_index": index,
                "status": "selection_failed",
                "reason": f"{type(error).__name__}: {error}",
                "attempted_calls": 1,
            })
            continue
        if target_id is None or target_observation is None:
            counts["missing_target"] += 1
            episode_rows.append({
                "episode_index": index,
                "status": "missing_target",
                "attempted_calls": 1,
            })
            continue
        try:
            bin_id = _observation_bin(target_observation)
        except Exception as error:
            counts["other_bin"] += 1
            episode_rows.append({
                "episode_index": index,
                "status": "missing_bin",
                "reason": f"{type(error).__name__}: {error}",
                "attempted_calls": 1,
            })
            continue
        if bin_id in counts:
            counts[bin_id] += 1
        else:
            counts["other_bin"] += 1
        observations.append(target_observation)
        ledger.reserve(4)
        try:
            physical = evaluate_token_pathwise_deviations(
                episode,
                policy.new_episode,
                target_id,
                candidates=_ACTIONS,
                parameters=parameters,
                quote=quote,
                reservation=reservation,
                degraded_slowdown=2.0,
                hedge_delay=2.0,
            )
            actual = physical.attempted_calls
            ledger.start(actual)
            if physical.status != "completed" or len(physical.rows) != 3:
                episode_rows.append({
                    "episode_index": index,
                    "status": "panel_failed",
                    "bin_id": bin_id,
                    "attempted_calls": 1 + actual,
                    "reason": physical.failure_reason or "incomplete paired panel",
                })
                continue
            if bin_id in retained_bins:
                for row in physical.rows:
                    samples.setdefault((bin_id, row.candidate_requested), []).append(float(row.candidate_payoff.total))
                panel_rows.append({
                    "episode_index": index,
                    "bin_id": bin_id,
                    "target_token_id": target_id,
                    "status": "completed",
                    "actions": [
                        {"requested_action": row.candidate_requested.value, "cost": row.candidate_payoff.total}
                        for row in physical.rows
                    ],
                })
            episode_rows.append({
                "episode_index": index,
                "status": "completed" if bin_id in retained_bins else "other_bin",
                "bin_id": bin_id,
                "attempted_calls": 1 + actual,
            })
        except Exception as error:
            ledger.start(4)
            episode_rows.append({
                "episode_index": index,
                "status": "panel_failed",
                "bin_id": bin_id,
                "attempted_calls": 5,
                "reason": f"{type(error).__name__}: {error}",
            })

    rows = _q_rows(samples, retained_bins, minimum_complete_panels)
    q_status = "completed" if all(row["status"] == "completed" for row in rows) else "insufficient"
    result: dict[str, object] = {
        "iteration": iteration,
        "namespace": namespace,
        "policy_before": policy.state.to_dict(),
        "policy_before_fingerprint": _policy_fingerprint(policy),
        "q_rows": list(rows),
        "occupancy_counts": counts,
        "occupancy_vector": list(_vector(counts, retained_bins + ("missing_target", "other_bin", "selection_failed"))),
        "episode_rows": episode_rows,
        "panel_rows": panel_rows,
        "observations": observations,
        "q_status": q_status,
        "reserved_calls": ledger.reserved,
        "attempted_calls": ledger.attempted,
        "maximum_calls": max_scheduler_calls,
    }
    if q_status != "completed":
        result["status"] = "statistics_insufficient"
        result["stop_reason"] = "one or more retained bin/action panels below floor"
        return result
    result["means_fingerprint"] = _digest(
        tuple((row["bin_id"], row["requested_action"], row["mean"]) for row in rows)
    )
    return result


def run_fixed_point(
    config: object,
    retained_bins: Sequence[str],
    *,
    episode_count: int = T4_EPISODE_SLOTS,
    minimum_complete_panels: int = T4_PANEL_FLOOR,
    max_rounds: int = T4_MAX_ROUNDS,
    eta: float = T4_ETA,
    max_scheduler_calls: int = T4_CALL_LIMIT_PER_ROUND,
    seed: int = T4_MACRO_SEED,
) -> dict[str, object]:
    from .token_t3a_execution import validate_frozen_config

    validate_frozen_config(config)
    bins = tuple(retained_bins)
    if not bins or tuple(sorted(set(bins))) != bins:
        raise FixedPointDiagnosticError("retained_bins must be sorted and unique")
    if type(max_rounds) is not int or not 0 < max_rounds <= T4_MAX_ROUNDS:
        raise FixedPointDiagnosticError("max_rounds must be in 1..10")
    eta = _finite(eta, "eta")
    if not 0.0 < eta <= 1.0:
        raise FixedPointDiagnosticError("eta must lie in (0, 1]")
    policy = MixedPopulationPolicy.initial(seed=seed)
    iterations: list[dict[str, object]] = []
    br_history: list[str] = []
    previous_occupancy: tuple[float, ...] | None = None
    previous_policy = policy
    final_status = "not_converged_10"
    final_reason = "maximum iteration count reached"

    for iteration in range(max_rounds):
        result = _run_iteration(
            config,
            bins,
            policy,
            iteration=iteration,
            episode_count=episode_count,
            minimum_complete_panels=minimum_complete_panels,
            max_scheduler_calls=max_scheduler_calls,
        )
        if result["q_status"] == "insufficient":
            result["status"] = "statistics_insufficient"
            result["stop_reason"] = "q panel floor not met; no BR update"
            iterations.append(result)
            final_status = "statistics_insufficient"
            final_reason = str(result["stop_reason"])
            break
        means = _mean_map(result["q_rows"])
        br = build_best_response(means, bins)
        br_fp = _br_fingerprint(br)
        current_occupancy = tuple(float(value) for value in result["occupancy_vector"])
        policy_after = MixedPopulationPolicy(policy.state.updated(br, eta=eta), seed=seed)
        policy_l1 = None
        br_residual = None
        distribution_l1 = None
        if result["observations"]:
            policy_l1 = max(
                _l1(
                    policy.probabilities_for_observation(observation),
                    policy_after.probabilities_for_observation(observation),
                )
                for observation in result["observations"]
            )
            br_residual = max(
                1.0
                - policy.probabilities_for_observation(observation)[
                    _ACTION_INDEX[br[_observation_bin(observation)]]
                ]
                for observation in result["observations"]
                if _observation_bin(observation) in br
            )
        if previous_occupancy is not None:
            distribution_l1 = _l1(previous_occupancy, current_occupancy)
        result.update({
            "best_response": {bin_id: action.value for bin_id, action in br.items()},
            "best_response_fingerprint": br_fp,
            "policy_after": policy_after.state.to_dict(),
            "policy_after_fingerprint": _policy_fingerprint(policy_after),
            "policy_l1_residual": policy_l1,
            "br_residual": br_residual,
            "distribution_l1_residual": distribution_l1,
            "status": "completed",
            "stop_reason": "updated policy function",
        })
        if br_fp in br_history and len(br_history) - br_history.index(br_fp) > 1:
            period = len(br_history) - br_history.index(br_fp)
            result["status"] = "cycle"
            result["stop_reason"] = f"repeated best-response table with period {period}"
            result["cycle_period"] = period
            iterations.append(result)
            final_status = "cycle"
            final_reason = str(result["stop_reason"])
            break
        iterations.append(result)
        br_history.append(br_fp)
        if (
            policy_l1 is not None
            and br_residual is not None
            and distribution_l1 is not None
            and policy_l1 <= T4_RESIDUAL_TOLERANCE
            and br_residual <= T4_RESIDUAL_TOLERANCE
            and distribution_l1 <= T4_RESIDUAL_TOLERANCE
            and len(iterations) >= 2
        ):
            iterations[-1]["status"] = "pure_fixed_point"
            iterations[-1]["stop_reason"] = "all frozen residuals are within tolerance"
            final_status = "pure_fixed_point"
            final_reason = str(iterations[-1]["stop_reason"])
            break
        previous_occupancy = current_occupancy
        previous_policy = policy
        policy = policy_after
    else:
        final_status = "not_converged_10"
        final_reason = "maximum iteration count reached"

    total_reserved = sum(int(row["reserved_calls"]) for row in iterations)
    total_attempted = sum(int(row["attempted_calls"]) for row in iterations)
    if total_reserved > T4_CALL_LIMIT_TOTAL:
        raise FixedPointDiagnosticError("declared T4 call ceiling exceeded")
    return {
        "evidence_label": T4_LABEL,
        "status": final_status,
        "stop_reason": final_reason,
        "environment_id": "theta_token_load0p7_v1",
        "policy_id": "pi_T4_damped_function_mixture_v1",
        "macro_seed": seed,
        "eta": eta,
        "episode_slots": episode_count,
        "panel_floor": minimum_complete_panels,
        "max_rounds": max_rounds,
        "retained_bins": list(bins),
        "iterations": iterations,
        "reserved_calls": total_reserved,
        "attempted_calls": total_attempted,
        "call_limit_per_round": max_scheduler_calls,
        "call_limit_total": T4_CALL_LIMIT_TOTAL,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def execute_fixed_point(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    episode_count: int = T4_EPISODE_SLOTS,
    minimum_complete_panels: int = T4_PANEL_FLOOR,
    max_rounds: int = T4_MAX_ROUNDS,
) -> Path:
    from dataclasses import asdict
    from .config import load_config
    from .token_best_response import load_t3a_runtime_candidate
    from .token_t3a_execution import build_source_bundle, environment_fingerprint

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    validation = project / "artifacts" / "token-t3a-oracle-20260907-r1-validation"
    occupancy = project / "artifacts" / "token-t3a-oracle-20260907-r1-occupancy"
    parent = load_t3a_runtime_candidate(validation, occupancy)
    bins = tuple(parent.retained_bins)
    result = run_fixed_point(
        config,
        bins,
        episode_count=episode_count,
        minimum_complete_panels=minimum_complete_panels,
        max_rounds=max_rounds,
    )
    manifest = {
        "schema_id": "token_t4_fixed_point_manifest_v1",
        "run_id": run_id,
        "evidence_label": T4_LABEL,
        "source_bundle": asdict(bundle),
        "environment_fingerprint": environment_fingerprint(config),
        "bin_schema_id": "bin_schema_v1",
        "parent_t3a_validation_fingerprint": parent.parent_fingerprint,
        "namespace": T4_NAMESPACE,
        "macro_seed": T4_MACRO_SEED,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    metric_contract = {
        "schema_id": "token_t4_metric_contract_v1",
        "policy_update": "pi_next=(1-eta)*pi_current+eta*BR",
        "eta": T4_ETA,
        "residual_tolerance": T4_RESIDUAL_TOLERANCE,
        "stop_statuses": ["pure_fixed_point", "cycle", "not_converged_10", "statistics_insufficient"],
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    files = {
        "manifest.json": manifest,
        "metric_contract.json": metric_contract,
        "policy_iterations.json": {"rows": result["iterations"]},
        "q_rows.json": {
            "rows": [
                {"iteration": row["iteration"], **q_row}
                for row in result["iterations"]
                for q_row in row["q_rows"]
            ]
        },
        "summary.json": result,
    }
    # Observation objects are retained in-memory for exact residuals, but the
    # durable artifact boundary is canonical JSON and must never depend on a
    # custom encoder or Python object identity.
    return write_run_directory(
        artifacts_root,
        run_id,
        {name: _json_value(payload) for name, payload in files.items()},
    )


__all__ = [
    "DampedPolicyState",
    "FixedPointDiagnosticError",
    "MixedPopulationPolicy",
    "T4CallLedger",
    "T4_LABEL",
    "T4_CALL_LIMIT_PER_ROUND",
    "T4_CALL_LIMIT_TOTAL",
    "execute_fixed_point",
    "run_fixed_point",
    "T4_NAMESPACE",
    "build_best_response",
    "classify_br_history",
    "summarize_q_rows",
    "write_fixed_point_artifact",
]
