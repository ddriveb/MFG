"""T4c support-weighted zero-price Soft-BR diagnostic.

T4c keeps the T4b physical reruns and paired scorer, but stores only stable
episode identities and reconstructs immutable traces on demand.  A positive
but undersupported bin is unresolved; only zero-support bins are inactive.
This module is a finite-population diagnostic and never emits equilibrium
claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import write_run_directory
from .attribution_episode import EpisodeIdentity, EpisodeProtocol, EpisodeTrace
from .domain import ProtectionAction


T4C_LABEL = "finite_population_token_t4c_supported_soft_response_diagnostic"
T4C_NAMESPACE = "token-mfg-restoration:t4c:soft-fixed-point:v1:library"
T4C_MACRO_SEED = 20260915
T4C_BETA = 1.0
T4C_ETA = 0.2
T4C_MAX_ROUNDS = 20
T4C_EPISODE_SLOTS = 4096
T4C_PANEL_FLOOR = 32
T4C_CALLS_PER_SLOT = 5
T4C_CALL_LIMIT_PER_ROUND = T4C_EPISODE_SLOTS * T4C_CALLS_PER_SLOT
T4C_CALL_LIMIT_TOTAL = T4C_MAX_ROUNDS * T4C_CALL_LIMIT_PER_ROUND
T4C_RESIDUAL_TOLERANCE = 0.01
T4C_SUPPORTED_FIXED_POINT_STATUS = "supported_soft_fixed_point"

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_ACTION_INDEX = {action: index for index, action in enumerate(_ACTIONS)}


class T4CError(ValueError):
    """Raised for a T4c contract or accounting violation."""


class _T4CPhysicalFailure(Exception):
    """Fail-closed wrapper for one ordinary physical/invariant exception."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        *,
        iteration: int,
        reserved_calls: int,
        attempted_calls: int,
    ) -> None:
        super().__init__(error_message)
        self.error_type = error_type
        self.error_message = error_message
        self.iteration = iteration
        self.reserved_calls = reserved_calls
        self.attempted_calls = attempted_calls

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        *,
        iteration: int,
        ledger: "T4CCallLedger",
    ) -> "_T4CPhysicalFailure":
        return cls(
            type(error).__name__,
            str(error),
            iteration=iteration,
            reserved_calls=ledger.reserved,
            attempted_calls=ledger.attempted,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "round": self.iteration,
            "exception_type": self.error_type,
            "exception_message": self.error_message,
            "reserved_calls": self.reserved_calls,
            "attempted_calls": self.attempted_calls,
        }


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise T4CError("value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T4CError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise T4CError(f"{name} must be finite")
    return result


def _probabilities(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise T4CError("probability vector must contain N/D/I")
    result = tuple(_finite(value, "probability", nonnegative=True) for value in values)
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise T4CError("probabilities must sum to one")
    return result  # type: ignore[return-value]


def _one_hot(action: ProtectionAction) -> tuple[float, float, float]:
    if not isinstance(action, ProtectionAction):
        raise T4CError("invalid protection action")
    return tuple(1.0 if item is action else 0.0 for item in _ACTIONS)  # type: ignore[return-value]


def _normalise_mapping(
    mapping: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if not isinstance(mapping, Mapping):
        raise T4CError("probability mapping must be a mapping")
    rows = []
    for bin_id, values in mapping.items():
        if not isinstance(bin_id, str) or not bin_id:
            raise T4CError("bin IDs must be non-empty strings")
        rows.append((bin_id, _probabilities(values)))
    if len({bin_id for bin_id, _ in rows}) != len(rows):
        raise T4CError("duplicate bin IDs")
    return tuple(sorted(rows))


@dataclass(frozen=True)
class T4CIdentityLibrary:
    """The compact, immutable library retained between T4c rounds."""

    identities: tuple[EpisodeIdentity, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.identities, tuple) or not self.identities:
            raise T4CError("identity library must be a non-empty tuple")
        if any(not isinstance(identity, EpisodeIdentity) for identity in self.identities):
            raise T4CError("identity library contains a non-identity record")
        if tuple(identity.episode_index for identity in self.identities) != tuple(
            range(len(self.identities))
        ):
            raise T4CError("identity library indices must be dense and ordered")
        expected = identity_library_fingerprint(self.identities)
        if self.fingerprint != expected:
            raise T4CError("identity library fingerprint mismatch")


def identity_library_fingerprint(identities: Sequence[EpisodeIdentity]) -> str:
    if not isinstance(identities, tuple) or not identities:
        raise T4CError("identities must be a non-empty tuple")
    if any(not isinstance(identity, EpisodeIdentity) for identity in identities):
        raise T4CError("identities must contain EpisodeIdentity values")
    return _digest(tuple(identity.key for identity in identities))


def build_identity_library(
    namespace: str,
    macro_seed: int,
    *,
    episode_count: int = T4C_EPISODE_SLOTS,
) -> T4CIdentityLibrary:
    if not isinstance(namespace, str) or not namespace or any(character.isspace() for character in namespace):
        raise T4CError("namespace must be a non-empty string without whitespace")
    if type(macro_seed) is not int or macro_seed < 0:
        raise T4CError("macro_seed must be a non-negative int")
    if type(episode_count) is not int or not 0 < episode_count <= T4C_EPISODE_SLOTS:
        raise T4CError("episode_count is outside 1..4096")
    identities = tuple(
        EpisodeIdentity(namespace, macro_seed, episode_index)
        for episode_index in range(episode_count)
    )
    return T4CIdentityLibrary(identities, identity_library_fingerprint(identities))


def validate_identity_library(
    library: T4CIdentityLibrary,
    fingerprint: str,
) -> T4CIdentityLibrary:
    if not isinstance(library, T4CIdentityLibrary):
        raise T4CError("library must be a T4CIdentityLibrary")
    if not isinstance(fingerprint, str) or library.fingerprint != fingerprint:
        raise T4CError("identity library fingerprint mismatch")
    return library


def reconstruct_episode(
    config: object,
    identity: EpisodeIdentity,
    protocol: EpisodeProtocol,
) -> EpisodeTrace:
    from .attribution_episode import generate_episode_trace

    if not isinstance(identity, EpisodeIdentity):
        raise T4CError("identity must be an EpisodeIdentity")
    if not isinstance(protocol, EpisodeProtocol):
        raise T4CError("protocol must be an EpisodeProtocol")
    return generate_episode_trace(
        config, identity.namespace, identity.macro_seed, identity.episode_index, protocol
    )


def reconstructed_trace_fingerprint(episode: EpisodeTrace) -> str:
    from .attribution_episode import episode_trace_fingerprint

    if not isinstance(episode, EpisodeTrace):
        raise T4CError("episode must be an EpisodeTrace")
    return _digest(episode_trace_fingerprint(episode))


@dataclass(frozen=True)
class T4CPolicyState:
    """Immutable probability function with inactive-bin retention."""

    probabilities_by_bin: tuple[tuple[str, tuple[float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities_by_bin, tuple):
            raise T4CError("probabilities_by_bin must be a tuple")
        clean = _normalise_mapping(dict(self.probabilities_by_bin))
        object.__setattr__(self, "probabilities_by_bin", clean)

    @classmethod
    def initial(cls) -> "T4CPolicyState":
        return cls()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[float]]) -> "T4CPolicyState":
        return cls(_normalise_mapping(mapping))

    def probabilities_for_bin(
        self,
        bin_id: str,
        fallback: Sequence[float],
    ) -> tuple[float, float, float]:
        if not isinstance(bin_id, str) or not bin_id:
            raise T4CError("bin_id must be a non-empty string")
        for candidate, probabilities in self.probabilities_by_bin:
            if candidate == bin_id:
                return probabilities
        return _probabilities(fallback)

    def updated(
        self,
        soft_response: Mapping[str, Sequence[float]],
        *,
        active_bins: Sequence[str],
        fallback_by_bin: Mapping[str, Sequence[float]],
        eta: float = T4C_ETA,
    ) -> "T4CPolicyState":
        eta = _finite(eta, "eta")
        if not 0.0 < eta <= 1.0:
            raise T4CError("eta must lie in (0, 1]")
        active = tuple(active_bins)
        if tuple(sorted(set(active))) != active:
            raise T4CError("active_bins must be sorted and unique")
        current = dict(self.probabilities_by_bin)
        for bin_id in active:
            if bin_id not in soft_response or bin_id not in fallback_by_bin:
                raise T4CError(f"missing active-bin update for {bin_id!r}")
            old = current.get(bin_id, _probabilities(fallback_by_bin[bin_id]))
            target = _probabilities(soft_response[bin_id])
            current[bin_id] = _probabilities(
                tuple((1.0 - eta) * left + eta * right for left, right in zip(old, target))
            )
        return T4CPolicyState.from_mapping(current)

    def fingerprint(self) -> str:
        return _digest({"base_policy": "pi_NIIN_v1", "probabilities_by_bin": self.probabilities_by_bin})

    def to_dict(self) -> dict[str, object]:
        return {
            "base_policy": "pi_NIIN_v1",
            "probabilities_by_bin": [
                {"bin_id": bin_id, "probabilities": list(probabilities)}
                for bin_id, probabilities in self.probabilities_by_bin
            ],
            "fingerprint": self.fingerprint(),
        }


class T4CPopulationPolicy:
    """Online T4c policy; each episode receives a fresh policy object."""

    def __init__(self, state: T4CPolicyState | None = None, *, seed: int = T4C_MACRO_SEED):
        if state is not None and not isinstance(state, T4CPolicyState):
            raise T4CError("state must be a T4CPolicyState")
        if type(seed) is not int or seed < 0:
            raise T4CError("seed must be a non-negative int")
        self.state = state if state is not None else T4CPolicyState.initial()
        self.seed = seed

    def new_episode(self) -> "T4CPopulationPolicy":
        return T4CPopulationPolicy(self.state, seed=self.seed)

    @staticmethod
    def base_action(observation: object) -> ProtectionAction:
        phase = getattr(getattr(observation, "phase", None), "value", None)
        primary = getattr(observation, "primary_replica", None)
        token_class = getattr(getattr(observation, "token_class", None), "value", None)
        age = getattr(observation, "phase_age", None)
        late = isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 50.0
        urgent = token_class == "U"
        # Exact NIIN order: early-Regular N, early-Urgent I,
        # late-Regular I, late-Urgent N.
        if phase == "D" and primary == 0 and late != urgent:
            return ProtectionAction.IMMEDIATE_HEDGE
        return ProtectionAction.NORMAL

    def probabilities_for_observation(self, observation: object) -> tuple[float, float, float]:
        from .token_continuation import BIN_SCHEMA_V1

        try:
            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception:
            return _one_hot(self.base_action(observation))
        return self.state.probabilities_for_bin(bin_id, _one_hot(self.base_action(observation)))

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise T4CError("policy_key must be a non-empty string")
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


def classify_bin_support(
    occupancy_count: int,
    complete_panel_count: int,
    *,
    panel_floor: int = T4C_PANEL_FLOOR,
) -> str:
    if type(occupancy_count) is not int or occupancy_count < 0:
        raise T4CError("occupancy_count must be a non-negative int")
    if type(complete_panel_count) is not int or complete_panel_count < 0:
        raise T4CError("complete_panel_count must be a non-negative int")
    if type(panel_floor) is not int or panel_floor <= 0:
        raise T4CError("panel_floor must be a positive int")
    if occupancy_count == 0:
        return "inactive"
    if complete_panel_count < panel_floor:
        return "unresolved"
    return "active"


def classify_support_rows(
    rows: Mapping[str, tuple[int, int]],
    *,
    panel_floor: int = T4C_PANEL_FLOOR,
) -> dict[str, object]:
    if not isinstance(rows, Mapping):
        raise T4CError("support rows must be a mapping")
    result: dict[str, object] = {}
    unresolved = False
    for bin_id, (occupancy_count, complete_panel_count) in sorted(rows.items()):
        status = classify_bin_support(occupancy_count, complete_panel_count, panel_floor=panel_floor)
        result[bin_id] = status
        unresolved = unresolved or status == "unresolved"
    result["gate_status"] = "statistics_insufficient" if unresolved else "ready"
    return result


def support_gate(
    rows: Mapping[str, tuple[int, int]],
    *,
    panel_floor: int = T4C_PANEL_FLOOR,
) -> dict[str, object]:
    result = classify_support_rows(rows, panel_floor=panel_floor)
    result["status"] = result["gate_status"]
    return result


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise T4CError("residual vectors have different lengths")
    return math.fsum(abs(a - b) for a, b in zip(left, right))


def support_residuals(
    previous: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    occupancy: Mapping[str, float],
    *,
    active_bins: Sequence[str],
) -> dict[str, object]:
    active = tuple(active_bins)
    if tuple(sorted(set(active))) != active or not active:
        raise T4CError("active_bins must be a non-empty sorted tuple")
    residual_by_bin = {}
    weights = []
    for bin_id in active:
        if bin_id not in previous or bin_id not in current or bin_id not in occupancy:
            raise T4CError(f"missing residual row for {bin_id!r}")
        residual = _l1(_probabilities(previous[bin_id]), _probabilities(current[bin_id]))
        weight = _finite(occupancy[bin_id], "occupancy", nonnegative=True)
        residual_by_bin[bin_id] = residual
        weights.append((bin_id, weight))
    total = math.fsum(weight for _, weight in weights)
    if total <= 0.0:
        raise T4CError("active occupancy must be positive")
    return {
        "weighted_policy_l1": math.fsum(
            residual_by_bin[bin_id] * weight / total for bin_id, weight in weights
        ),
        "max_active_policy_l1": max(residual_by_bin.values()),
        "active_bins": list(active),
        "policy_l1_by_bin": residual_by_bin,
    }


class T4CCallLedger:
    def __init__(self, maximum: int):
        if type(maximum) is not int or maximum <= 0:
            raise T4CError("maximum calls must be a positive int")
        self.maximum = maximum
        self.reserved = 0
        self.attempted = 0

    def reserve(self, count: int) -> None:
        if type(count) is not int or count <= 0 or self.reserved + count > self.maximum:
            raise T4CError("call reservation exceeds budget")
        self.reserved += count

    def start(self, count: int = 1) -> None:
        if type(count) is not int or count < 0 or self.attempted + count > self.reserved:
            raise T4CError("call settlement exceeds reservation")
        self.attempted += count


def write_t4c_artifact(
    artifacts_root: str | Path,
    run_id: str,
    summary: Mapping[str, object],
) -> Path:
    return write_run_directory(
        artifacts_root,
        run_id,
        {"summary.json": _json_value(summary)},
    )


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
        return self.source.choose(observation, policy_key)


def _observation_bin(observation: object) -> str:
    from .token_continuation import BIN_SCHEMA_V1

    return BIN_SCHEMA_V1.bin(observation).bin_id


def _protocol_fingerprint(protocol: EpisodeProtocol) -> str:
    return _digest({
        "arrival_cutoff": protocol.arrival_cutoff,
        "timeline": {
            "degraded_start": protocol.timeline.degraded_start,
            "failed_start": protocol.timeline.failed_start,
            "recovered_start": protocol.timeline.recovered_start,
        },
    })


def _hard_br_fingerprint(means: Mapping[str, Mapping[ProtectionAction, float]]) -> str:
    rows = []
    for bin_id, values in sorted(means.items()):
        action = min(
            (_finite(values[item], "Q mean"), _ACTION_INDEX[item], item)
            for item in _ACTIONS
        )[2]
        rows.append((bin_id, action.value))
    return _digest(tuple(rows))


def _occupancy_vector(
    counts: Mapping[str, int],
    bins: Sequence[str],
) -> dict[str, float]:
    total = sum(counts.get(bin_id, 0) for bin_id in bins)
    if total == 0:
        return {bin_id: 0.0 for bin_id in bins}
    return {
        bin_id: counts.get(bin_id, 0) / total
        for bin_id in bins
    }


def aggregate_round_physics_metrics(
    episode_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Aggregate existing attribution metric fields for one T4c round.

    Additive quantities use the same pooled-token or pooled-work arithmetic as
    the attribution metric contract.  Quantile and storm-shape fields retain
    the existing per-episode metric and report its arithmetic episode mean;
    no new physical definition is introduced here.
    """
    rows = tuple(episode_metrics)
    if not rows:
        return {
            "panel_count": 0,
            "latency": {"overall": {"sample_count": 0}},
            "replay": {"replayed_tokens": 0, "generated_tokens": 0, "rate": None},
            "hedge": {"requested": 0, "launched": 0, "winners": 0},
            "work": {
                "nominal_primary_work": 0.0,
                "executed_primary_work": 0.0,
                "executed_hedge_work": 0.0,
                "executed_replay_work": 0.0,
                "total_executed_work": 0.0,
                "wasted_work": 0.0,
                "waste_ratio": None,
            },
            "storm": {
                "hedge_launch_count": 0,
                "post_cutoff_hedge_launch_count": 0,
                "post_cutoff_hedge_launch_work": 0.0,
                "observation_duration": 0.0,
                "mean_hedge_launch_rate": None,
                "peak_hedge_launch_rate": None,
                "peak_to_mean_hedge_launch_ratio": None,
                "overloaded_bin_count": 0,
                "sustained_overload_duration": None,
            },
            "aggregation_contract": "existing_attribution_fields:pooled_additive_episode_mean_shape",
        }

    def total(section: str, field: str) -> float:
        return math.fsum(float(row[section][field]) for row in rows)

    def mean(section: str, field: str) -> float | None:
        values = [row[section][field] for row in rows]
        values = [float(value) for value in values if value is not None]
        return math.fsum(values) / len(values) if values else None

    def latency_mean(field: str) -> float | None:
        values = [row["latency"]["overall"][field] for row in rows]
        values = [float(value) for value in values if value is not None]
        return math.fsum(values) / len(values) if values else None

    overall = [row["latency"]["overall"] for row in rows]
    token_count = sum(int(row["generated_tokens"]) for row in rows)
    latency_sum = math.fsum(float(item["latency_sum"]) for item in overall)
    deadline_miss_count = sum(int(item["deadline_miss_count"]) for item in overall)
    excess_sum = math.fsum(float(item["excess_latency_sum"]) for item in overall)
    latency = {
        "overall": {
            "sample_count": sum(int(item["sample_count"]) for item in overall),
            "latency_sum": latency_sum,
            "mean_latency": latency_sum / token_count if token_count else None,
            "latency_p50": latency_mean("latency_p50"),
            "latency_p95": latency_mean("latency_p95"),
            "latency_p99": latency_mean("latency_p99"),
            "cvar95_latency": latency_mean("cvar95_latency"),
            "deadline_miss_count": deadline_miss_count,
            "deadline_miss_probability": (
                deadline_miss_count / token_count if token_count else None
            ),
            "excess_latency_sum": excess_sum,
            "mean_excess_latency": excess_sum / token_count if token_count else None,
        },
        "quantile_aggregation": "arithmetic_mean_of_existing_episode_metric_fields",
    }

    hedge_fields = (
        "requested",
        "launched",
        "winners",
        "cancelled_queued",
        "completed_loser",
        "failed_running",
        "invalidated_queued",
        "executor_suppressed",
    )
    hedge = {field: sum(int(row["hedge"][field]) for row in rows) for field in hedge_fields}
    hedge["win_per_launch"] = (
        hedge["winners"] / hedge["launched"] if hedge["launched"] else None
    )

    work_fields = (
        "nominal_primary_work",
        "executed_primary_work",
        "executed_hedge_work",
        "executed_replay_work",
        "total_executed_work",
        "wasted_work",
    )
    work = {field: total("work", field) for field in work_fields}
    work["waste_ratio"] = (
        work["wasted_work"] / work["total_executed_work"]
        if work["total_executed_work"]
        else None
    )

    storm = {
        "hedge_launch_count": sum(int(row["storm"]["hedge_launch_count"]) for row in rows),
        "post_cutoff_hedge_launch_count": sum(
            int(row["storm"]["post_cutoff_hedge_launch_count"]) for row in rows
        ),
        "post_cutoff_hedge_launch_work": total(
            "storm", "post_cutoff_hedge_launch_work"
        ),
        "observation_duration": total("storm", "observation_duration"),
        "mean_hedge_launch_rate": mean("storm", "mean_hedge_launch_rate"),
        "peak_hedge_launch_rate": mean("storm", "peak_hedge_launch_rate"),
        "peak_to_mean_hedge_launch_ratio": mean(
            "storm", "peak_to_mean_hedge_launch_ratio"
        ),
        "overloaded_bin_count": sum(
            int(row["storm"]["overloaded_bin_count"]) for row in rows
        ),
        "sustained_overload_duration": mean(
            "storm", "sustained_overload_duration"
        ),
    }
    storm["total_hedge_launch_count"] = (
        storm["hedge_launch_count"] + storm["post_cutoff_hedge_launch_count"]
    )
    return {
        "panel_count": len(rows),
        "latency": latency,
        "replay": {
            "replayed_tokens": sum(int(row["replay"]["replayed_tokens"]) for row in rows),
            "generated_tokens": token_count,
            "rate": (
                sum(int(row["replay"]["replayed_tokens"]) for row in rows) / token_count
                if token_count
                else None
            ),
        },
        "hedge": hedge,
        "work": work,
        "storm": storm,
        "aggregation_contract": "existing_attribution_fields:pooled_additive_episode_mean_shape",
    }


def _run_round(
    config: object,
    library: T4CIdentityLibrary,
    protocol: EpisodeProtocol,
    retained_bins: tuple[str, ...],
    policy: T4CPopulationPolicy,
    *,
    iteration: int,
    minimum_complete_panels: int,
    max_scheduler_calls: int,
) -> dict[str, object]:
    from .attribution_metrics import build_episode_metrics
    from .token_deviations import evaluate_token_pathwise_deviations
    from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
    from .token_t3a_execution import T3AAnchorSelector
    from .token_online import simulate_episode_online
    from .transient_control import ReservationParameters
    from .token_mfg_soft_fixed_point import paired_action_gap

    if len(library.identities) * T4C_CALLS_PER_SLOT > max_scheduler_calls:
        raise T4CError("round call budget is too small")
    reservation = ReservationParameters(
        window_width=25.0,
        budget_rate=0.45,
        scale=0.25,
        mean_requirement=1.0,
    )
    parameters = TokenRuntimeADR0009Parameters()
    quote = ExternalQuote(0.0, "incremental_executed_work")
    ledger = T4CCallLedger(max_scheduler_calls)
    samples: dict[tuple[str, ProtectionAction], list[float]] = {}
    counts = {bin_id: 0 for bin_id in retained_bins}
    complete_panels = {bin_id: 0 for bin_id in retained_bins}
    counts.update({"missing_target": 0, "other_bin": 0, "selection_failed": 0})
    fallback_by_bin: dict[str, tuple[float, float, float]] = {}
    reconstruction_fingerprints = []
    physics_metric_rows: list[dict[str, Any]] = []

    for index, identity in enumerate(library.identities):
        episode = reconstruct_episode(config, identity, protocol)
        reconstruction_fingerprints.append(reconstructed_trace_fingerprint(episode))
        ledger.reserve(1)
        try:
            capture = _CaptureSource(
                policy.new_episode(),
                T3AAnchorSelector(index % 4),
            )
            simulate_episode_online(
                episode,
                capture,
                reservation,
                degraded_slowdown=2.0,
                hedge_delay=2.0,
            )
            ledger.start(1)
        except Exception as error:
            ledger.start(1)
            failure = _T4CPhysicalFailure.from_exception(
                error,
                iteration=iteration,
                ledger=ledger,
            )
            raise failure from error
        if capture.target_id is None or capture.observation is None:
            counts["missing_target"] += 1
            continue
        try:
            bin_id = _observation_bin(capture.observation)
        except Exception:
            counts["other_bin"] += 1
            continue
        counts[bin_id if bin_id in counts else "other_bin"] += 1
        if bin_id in retained_bins:
            fallback_by_bin.setdefault(bin_id, _one_hot(policy.base_action(capture.observation)))
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
            ledger.start(result.attempted_calls)
            if result.status != "completed" or len(result.rows) != 3:
                error = RuntimeError(
                    result.failure_reason
                    or "pathwise deviation did not complete a full panel"
                )
                failure = _T4CPhysicalFailure(
                    "TokenPathwiseDeviationResult",
                    str(error),
                    iteration=iteration,
                    reserved_calls=ledger.reserved,
                    attempted_calls=ledger.attempted,
                )
                raise failure from error
            try:
                physics_metric_rows.append(
                    build_episode_metrics(
                        result.baseline_run.simulation,
                        degraded_slowdown=2.0,
                    )
                )
            except Exception as error:
                failure = _T4CPhysicalFailure.from_exception(
                    error,
                    iteration=iteration,
                    ledger=ledger,
                )
                raise failure from error
            if bin_id in retained_bins:
                complete_panels[bin_id] += 1
                for row in result.rows:
                    samples.setdefault((bin_id, row.candidate_requested), []).append(
                        float(row.candidate_payoff.total)
                    )
        except _T4CPhysicalFailure:
            raise
        except Exception as error:
            ledger.start(4)
            failure = _T4CPhysicalFailure.from_exception(
                error,
                iteration=iteration,
                ledger=ledger,
            )
            raise failure from error

    support_rows = {
        bin_id: (counts[bin_id], complete_panels[bin_id])
        for bin_id in retained_bins
    }
    support = classify_support_rows(
        support_rows,
        panel_floor=minimum_complete_panels,
    )
    active_bins = tuple(
        bin_id for bin_id in retained_bins if support[bin_id] == "active"
    )
    q_rows = []
    means: dict[str, dict[ProtectionAction, float]] = {}
    for bin_id in retained_bins:
        means[bin_id] = {}
        status = support[bin_id]
        for action in _ACTIONS:
            values = tuple(samples.get((bin_id, action), ()))
            n = len(values)
            if status != "active":
                q_rows.append({
                    "bin_id": bin_id,
                    "requested_action": action.value,
                    "mean": None,
                    "standard_error": None,
                    "effective_n": n,
                    "status": status,
                })
                continue
            mean = math.fsum(values) / n
            variance = (
                0.0
                if n == 1
                else math.fsum((value - mean) ** 2 for value in values) / (n - 1)
            )
            means[bin_id][action] = mean
            q_rows.append({
                "bin_id": bin_id,
                "requested_action": action.value,
                "mean": mean,
                "standard_error": math.sqrt(variance / n),
                "effective_n": n,
                "status": "completed",
            })
    means = {bin_id: values for bin_id, values in means.items() if values}
    gap_rows = []
    for bin_id in active_bins:
        gap = paired_action_gap(samples, bin_id)
        gap_rows.append({"bin_id": bin_id, **_json_value(gap)})
    return {
        "iteration": iteration,
        "policy_before": policy.state.to_dict(),
        "policy_before_fingerprint": policy.state.fingerprint(),
        "identity_library_fingerprint": library.fingerprint,
        "reconstruction_fingerprint": _digest(tuple(reconstruction_fingerprints)),
        "support": {
            bin_id: {
                "occupancy_count": counts[bin_id],
                "complete_panel_count": complete_panels[bin_id],
                "status": support[bin_id],
            }
            for bin_id in retained_bins
        },
        "support_gate_status": support["gate_status"],
        "active_bins": list(active_bins),
        "q_rows": q_rows,
        "action_gap_rows": gap_rows,
        "occupancy_counts": counts,
        "occupancy_vector": _occupancy_vector(counts, retained_bins),
        "physics_metrics": aggregate_round_physics_metrics(physics_metric_rows),
        "reserved_calls": ledger.reserved,
        "attempted_calls": ledger.attempted,
        "means": means,
        "fallback_by_bin": fallback_by_bin,
        "q_status": "completed" if support["gate_status"] == "ready" else "insufficient",
    }


def run_t4c(
    config: object,
    retained_bins: Sequence[str],
    *,
    episode_count: int = T4C_EPISODE_SLOTS,
    minimum_complete_panels: int = T4C_PANEL_FLOOR,
    max_rounds: int = T4C_MAX_ROUNDS,
    beta: float = T4C_BETA,
    eta: float = T4C_ETA,
) -> dict[str, object]:
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL
    from .token_t3a_execution import validate_frozen_config
    from .token_mfg_soft_fixed_point import soft_best_response

    validate_frozen_config(config)
    bins = tuple(retained_bins)
    if not bins or tuple(sorted(set(bins))) != bins:
        raise T4CError("retained_bins must be sorted and unique")
    if type(episode_count) is not int or not 0 < episode_count <= T4C_EPISODE_SLOTS:
        raise T4CError("episode_count is outside 1..4096")
    if type(max_rounds) is not int or not 0 < max_rounds <= T4C_MAX_ROUNDS:
        raise T4CError("max_rounds is outside 1..20")
    if type(minimum_complete_panels) is not int or minimum_complete_panels <= 0:
        raise T4CError("minimum_complete_panels must be positive")
    beta = _finite(beta, "beta")
    eta = _finite(eta, "eta")
    execution_profile = (
        "formal_adr0025"
        if episode_count == T4C_EPISODE_SLOTS
        and minimum_complete_panels == T4C_PANEL_FLOOR
        and max_rounds == T4C_MAX_ROUNDS
        else "exploratory"
    )
    profile_call_limit_per_round = episode_count * T4C_CALLS_PER_SLOT
    profile_call_limit_total = max_rounds * profile_call_limit_per_round
    library = build_identity_library(
        T4C_NAMESPACE,
        T4C_MACRO_SEED,
        episode_count=episode_count,
    )
    policy = T4CPopulationPolicy(seed=T4C_MACRO_SEED)
    iterations = []
    previous_occupancy = None
    comparable_rounds = 0
    final_status = f"not_converged_{max_rounds}"
    final_reason = "maximum iteration count reached"
    previous_hard_fp = None
    physical_failure = None
    for iteration in range(max_rounds):
        try:
            result = _run_round(
                config,
                library,
                ATTRIBUTION_V1_PROTOCOL,
                bins,
                policy,
                iteration=iteration,
                minimum_complete_panels=minimum_complete_panels,
                max_scheduler_calls=profile_call_limit_per_round,
            )
        except _T4CPhysicalFailure as failure:
            physical_failure = failure.as_dict()
            result = {
                "iteration": iteration,
                "status": "physical_failed",
                "stop_reason": failure.error_message,
                "physical_failure": physical_failure,
                "reserved_calls": failure.reserved_calls,
                "attempted_calls": failure.attempted_calls,
                "physics_metrics": aggregate_round_physics_metrics(()),
            }
            iterations.append(result)
            final_status = "physical_failed"
            final_reason = failure.error_message
            break
        if result["q_status"] != "completed":
            result["status"] = "statistics_insufficient"
            result["stop_reason"] = "positive-support bin has fewer than panel floor"
            iterations.append(result)
            final_status = "statistics_insufficient"
            final_reason = str(result["stop_reason"])
            break
        active_bins = tuple(result["active_bins"])
        if not active_bins:
            result["status"] = "statistics_insufficient"
            result["stop_reason"] = "no active retained bin has complete support"
            iterations.append(result)
            final_status = "statistics_insufficient"
            final_reason = str(result["stop_reason"])
            break
        soft_response = soft_best_response(result["means"], beta=beta)
        next_state = policy.state.updated(
            soft_response,
            active_bins=active_bins,
            fallback_by_bin=result["fallback_by_bin"],
            eta=eta,
        )
        next_policy = T4CPopulationPolicy(next_state, seed=T4C_MACRO_SEED)
        current_probs = {
            bin_id: policy.state.probabilities_for_bin(
                bin_id,
                result["fallback_by_bin"][bin_id],
            )
            for bin_id in active_bins
        }
        next_probs = {
            bin_id: next_state.probabilities_for_bin(
                bin_id,
                result["fallback_by_bin"][bin_id],
            )
            for bin_id in active_bins
        }
        residuals = support_residuals(
            current_probs,
            next_probs,
            result["occupancy_counts"],
            active_bins=active_bins,
        )
        occupancy = result["occupancy_vector"]
        population_residual = (
            None
            if previous_occupancy is None
            else _l1(
                tuple(previous_occupancy[bin_id] for bin_id in bins),
                tuple(occupancy[bin_id] for bin_id in bins),
            )
        )
        hard_fp = _hard_br_fingerprint(result["means"])
        result.update({
            "soft_response": {
                bin_id: list(values) for bin_id, values in soft_response.items()
            },
            "hard_br_fingerprint": hard_fp,
            "hard_br_repeat": previous_hard_fp == hard_fp if previous_hard_fp is not None else False,
            "policy_after": next_state.to_dict(),
            "policy_after_fingerprint": next_state.fingerprint(),
            "weighted_policy_l1_residual": residuals["weighted_policy_l1"],
            "max_active_policy_l1_residual": residuals["max_active_policy_l1"],
            "policy_l1_by_bin": residuals["policy_l1_by_bin"],
            "population_occupancy_l1_residual": population_residual,
            "status": "completed",
            "stop_reason": "updated active-bin soft probability function",
        })
        iterations.append(result)
        comparable_rounds += 1
        if (
            comparable_rounds >= 2
            and population_residual is not None
            and residuals["weighted_policy_l1"] <= T4C_RESIDUAL_TOLERANCE
            and residuals["max_active_policy_l1"] <= T4C_RESIDUAL_TOLERANCE
            and population_residual <= T4C_RESIDUAL_TOLERANCE
        ):
            iterations[-1]["status"] = T4C_SUPPORTED_FIXED_POINT_STATUS
            iterations[-1]["stop_reason"] = "support-weighted and active-bin residuals within tolerance"
            final_status = T4C_SUPPORTED_FIXED_POINT_STATUS
            final_reason = str(iterations[-1]["stop_reason"])
            break
        previous_occupancy = occupancy
        previous_hard_fp = hard_fp
        policy = next_policy
    total_reserved = sum(int(row["reserved_calls"]) for row in iterations)
    total_attempted = sum(int(row["attempted_calls"]) for row in iterations)
    for result in iterations:
        result.pop("means", None)
        result.pop("fallback_by_bin", None)
    return {
        "evidence_label": T4C_LABEL,
        "execution_profile": execution_profile,
        "status": final_status,
        "stop_reason": final_reason,
        "environment_id": "theta_token_load0p7_v1",
        "policy_id": "pi_T4c_support_weighted_soft_response_v1",
        "beta": beta,
        "eta": eta,
        "macro_seed": T4C_MACRO_SEED,
        "namespace": T4C_NAMESPACE,
        "episode_count": episode_count,
        "panel_floor": minimum_complete_panels,
        "max_rounds": max_rounds,
        "identity_library_fingerprint": library.fingerprint,
        "identity_library_rows": [
            {"episode_index": identity.episode_index, "key": identity.key}
            for identity in library.identities
        ],
        "retained_bins": list(bins),
        "iterations": iterations,
        "attempted_calls": total_attempted,
        "reserved_calls": total_reserved,
        "physical_failure": physical_failure,
        "call_limit_per_round": profile_call_limit_per_round,
        "call_limit_total": profile_call_limit_total,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def execute_t4c(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    episode_count: int = T4C_EPISODE_SLOTS,
    minimum_complete_panels: int = T4C_PANEL_FLOOR,
    max_rounds: int = T4C_MAX_ROUNDS,
) -> Path:
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL
    from .config import load_config
    from .token_best_response import load_t3a_runtime_candidate
    from .token_t3a_execution import build_source_bundle, environment_fingerprint

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    parent = load_t3a_runtime_candidate(
        project / "artifacts" / "token-t3a-oracle-20260907-r1-validation",
        project / "artifacts" / "token-t3a-oracle-20260907-r1-occupancy",
    )
    result = run_t4c(
        config,
        parent.retained_bins,
        episode_count=episode_count,
        minimum_complete_panels=minimum_complete_panels,
        max_rounds=max_rounds,
    )
    manifest = {
        "schema_id": "token_t4c_supported_soft_response_manifest_v1",
        "run_id": run_id,
        "evidence_label": T4C_LABEL,
        "execution_profile": result["execution_profile"],
        "source_bundle": asdict(bundle),
        "environment_fingerprint": environment_fingerprint(config),
        "protocol_fingerprint": _protocol_fingerprint(ATTRIBUTION_V1_PROTOCOL),
        "identity_library_fingerprint": result["identity_library_fingerprint"],
        "namespace": T4C_NAMESPACE,
        "macro_seed": T4C_MACRO_SEED,
        "beta": T4C_BETA,
        "eta": T4C_ETA,
        "episode_count": episode_count,
        "max_rounds": max_rounds,
        "panel_floor": minimum_complete_panels,
        "call_limit_per_round": result["call_limit_per_round"],
        "call_limit_total": result["call_limit_total"],
        "support_rule": "zero occupancy inactive; positive occupancy below floor unresolved",
        "allowed_terminal_statuses": [
            T4C_SUPPORTED_FIXED_POINT_STATUS,
            "statistics_insufficient",
            f"not_converged_{max_rounds}",
            "physical_failed",
        ],
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    metric_contract = {
        "schema_id": "token_t4c_supported_soft_response_metric_contract_v1",
        "execution_profile": result["execution_profile"],
        "soft_response": "exp(-beta*Q)/sum(exp(-beta*Q))",
        "beta": T4C_BETA,
        "eta": T4C_ETA,
        "residual_tolerance": T4C_RESIDUAL_TOLERANCE,
        "residuals": [
            "weighted_policy_l1",
            "max_active_policy_l1",
            "population_occupancy_l1",
        ],
        "hard_br_repeat_is_diagnostic_only": True,
        "per_round_physics_metrics": {
            "source": "mfg_hedge.attribution_metrics.build_episode_metrics",
            "fields": ["latency", "replay", "hedge", "work", "storm"],
            "aggregation": "pooled additive fields and arithmetic episode means for existing shape fields",
        },
        "allowed_terminal_statuses": [
            T4C_SUPPORTED_FIXED_POINT_STATUS,
            "statistics_insufficient",
            f"not_converged_{max_rounds}",
            "physical_failed",
        ],
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
        "identity_library.json": {
            "rows": result["identity_library_rows"],
            "fingerprint": result["identity_library_fingerprint"],
        },
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
    "T4CError",
    "T4CIdentityLibrary",
    "T4CPolicyState",
    "T4CPopulationPolicy",
    "T4CCallLedger",
    "T4C_BETA",
    "T4C_CALL_LIMIT_PER_ROUND",
    "T4C_CALL_LIMIT_TOTAL",
    "T4C_EPISODE_SLOTS",
    "T4C_ETA",
    "T4C_LABEL",
    "T4C_MACRO_SEED",
    "T4C_MAX_ROUNDS",
    "T4C_NAMESPACE",
    "T4C_PANEL_FLOOR",
    "T4C_SUPPORTED_FIXED_POINT_STATUS",
    "aggregate_round_physics_metrics",
    "build_identity_library",
    "classify_bin_support",
    "classify_support_rows",
    "identity_library_fingerprint",
    "reconstruct_episode",
    "reconstructed_trace_fingerprint",
    "run_t4c",
    "execute_t4c",
    "support_gate",
    "support_residuals",
    "validate_identity_library",
    "write_t4c_artifact",
]
