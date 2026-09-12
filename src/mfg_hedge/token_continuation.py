"""T3A observable-bin conditional Token cost oracle.

This module is intentionally an internal implementation of ADR-0022.  It
generates independent exogenous episodes first, selects at most one target
online, bins that episode-local observation, and then performs complete paired
N/D/I T2 reruns.  It never selects an action or computes BR, regret, Nash, or
MFG quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .attribution_episode import (
    EpisodeProtocol,
    EpisodeTrace,
    episode_trace_fingerprint,
)
from .common_state import Phase
from .domain import ProtectionAction, TokenClass
from .token_deviations import evaluate_token_pathwise_deviations
from .token_online import TokenObservation, simulate_episode_online, validate_token_action
from .token_payoff import (
    ADMITTED_RESERVED_WORK,
    ExternalQuote,
    INCREMENTAL_EXECUTED_WORK,
    TokenExtendedReservationParameters,
    TokenInitialPriceParameters,
    TokenRuntimeADR0009Parameters,
    _expected_basis,
)
from .transient_control import ReservationParameters
from .common_state import validate_slowdown


EVIDENCE_LABEL = "observable_bin_conditional_cost_oracle"
ORACLE_MODE = "implementation_validation_only"
BIN_SCHEMA_ID = "bin_schema_v1"
TARGET_SELECTION_RULE_ID = "target_selection_v1:first_degraded_primary_A"
FROZEN_CALIBRATION_EPISODE_COUNT = 16
FROZEN_VALIDATION_EPISODE_COUNT = 16
MIN_COMPLETE_PANELS_PER_BIN = 2
FROZEN_MAX_SCHEDULER_CALLS = 512

_ACTION_ORDER = {
    ProtectionAction.NORMAL: 0,
    ProtectionAction.DELAYED_HEDGE: 1,
    ProtectionAction.IMMEDIATE_HEDGE: 2,
}
_MODEL_ORDER = {
    "token_initial_price_v0": 0,
    "token_runtime_adr0009_v1": 1,
    "token_extended_reservation_v1": 2,
}


class TokenObservableBinOracleError(ValueError):
    """Raised when the frozen T3A input or call budget cannot be honored."""


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TokenObservableBinOracleError(
            f"{name} must be an int >= {minimum}, got {value!r}"
        )
    return value


def _strict_real(value: object, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TokenObservableBinOracleError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise TokenObservableBinOracleError(f"{name} must be finite")
    return result


def _digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TokenObservableBinOracleError("payload is not canonically serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _observation_payload(observation: TokenObservation) -> dict[str, object]:
    return {
        field.name: _json_value(getattr(observation, field.name))
        for field in observation.__dataclass_fields__.values()
    }


def observation_fingerprint(observation: TokenObservation) -> str:
    if not isinstance(observation, TokenObservation):
        raise TokenObservableBinOracleError("observation must be TokenObservation")
    return _digest(_observation_payload(observation))


def _trace_fingerprint(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _bucket_open_upper(value: float, boundaries: tuple[float, ...], name: str) -> str:
    if not boundaries or value < boundaries[0]:
        raise TokenObservableBinOracleError(f"{name} is below its first boundary")
    for index, upper in enumerate(boundaries[1:]):
        if value < upper:
            return f"[{boundaries[index]:g},{upper:g})"
    return f"[{boundaries[-1]:g},+inf)"


def _bucket_finite(value: float, boundaries: tuple[float, ...], name: str) -> str:
    if not boundaries or value < boundaries[0] or value >= boundaries[-1]:
        raise TokenObservableBinOracleError(f"{name} is outside its finite bins")
    for index, upper in enumerate(boundaries[1:]):
        if value < upper:
            return f"[{boundaries[index]:g},{upper:g})"
    raise TokenObservableBinOracleError(f"{name} has no half-open bin")


@dataclass(frozen=True)
class ObservableBin:
    """Canonical key for one episode-local observable bin."""

    schema_id: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_id != BIN_SCHEMA_ID:
            raise TokenObservableBinOracleError("unknown observable-bin schema")
        if not isinstance(self.values, tuple) or not self.values or any(
            not isinstance(value, str) or not value for value in self.values
        ):
            raise TokenObservableBinOracleError("bin values must be non-empty strings")

    @property
    def bin_id(self) -> str:
        return "|".join(self.values)


@dataclass(frozen=True)
class ObservableBinSchema:
    """Frozen schema and half-open boundary contract for ``B(O)``."""

    schema_id: str
    field_order: tuple[str, ...]
    categorical_levels: tuple[tuple[str, tuple[str, ...]], ...]
    _numeric_items: tuple[tuple[str, tuple[float, ...], bool], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_id != BIN_SCHEMA_ID:
            raise TokenObservableBinOracleError("schema_id must be bin_schema_v1")
        if not isinstance(self.field_order, tuple) or not self.field_order:
            raise TokenObservableBinOracleError("field_order must be non-empty")
        categories = dict(self.categorical_levels)
        numeric = {
            name: (bounds, finite)
            for name, bounds, finite in self._numeric_items
        }
        if set(self.field_order) != set(categories) | set(numeric):
            raise TokenObservableBinOracleError("schema field partitions do not match")
        for name, levels in categories.items():
            if not levels or len(set(levels)) != len(levels):
                raise TokenObservableBinOracleError(f"invalid categorical levels for {name}")
        for name, (bounds, finite) in numeric.items():
            if len(bounds) < 2 if finite else len(bounds) < 1:
                raise TokenObservableBinOracleError(f"invalid numeric boundaries for {name}")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in bounds
            ):
                raise TokenObservableBinOracleError(f"non-finite boundary for {name}")
            if any(left >= right for left, right in zip(bounds, bounds[1:])):
                raise TokenObservableBinOracleError(f"non-increasing boundaries for {name}")
        payload = {
            "schema_id": self.schema_id,
            "field_order": self.field_order,
            "categorical_levels": self.categorical_levels,
            "numeric_items": self._numeric_items,
            "endpoint_rule": "left_closed_right_open_final_unbounded_or_finite",
        }
        expected = _digest(payload)
        if self.fingerprint != expected:
            raise TokenObservableBinOracleError("schema fingerprint does not match schema bytes")

    @property
    def numeric_boundaries(self) -> Mapping[str, tuple[float, ...]]:
        return MappingProxyType({name: bounds for name, bounds, _ in self._numeric_items})

    @property
    def numeric_finite(self) -> Mapping[str, bool]:
        return MappingProxyType({name: finite for name, _, finite in self._numeric_items})

    def bin(self, observation: TokenObservation) -> ObservableBin:
        if not isinstance(observation, TokenObservation):
            raise TokenObservableBinOracleError("observation must be TokenObservation")
        queue_load = sum(len(queue) for queue in observation.queue_snapshot) + len(
            observation.running_attempts
        )
        values_by_name: dict[str, str] = {
            "phase": observation.phase.value,
            "token_class": observation.token_class.value,
            "primary_replica": str(observation.primary_replica),
            "arrival_time": _bucket_finite(
                observation.arrival_time,
                self.numeric_boundaries["arrival_time"],
                "arrival_time",
            ),
            "phase_age": _bucket_open_upper(
                observation.phase_age,
                self.numeric_boundaries["phase_age"],
                "phase_age",
            ),
            "queue_load": _bucket_open_upper(
                float(queue_load),
                self.numeric_boundaries["queue_load"],
                "queue_load",
            ),
            "reservation_balance": _bucket_open_upper(
                observation.reservation_balance,
                self.numeric_boundaries["reservation_balance"],
                "reservation_balance",
            ),
        }
        for name, levels in self.categorical_levels:
            if values_by_name[name] not in levels:
                raise TokenObservableBinOracleError(f"invalid categorical value for {name}")
        return ObservableBin(
            self.schema_id,
            tuple(values_by_name[name] for name in self.field_order),
        )


def _schema_fingerprint(
    schema_id: str,
    field_order: tuple[str, ...],
    categorical_levels: tuple[tuple[str, tuple[str, ...]], ...],
    numeric_items: tuple[tuple[str, tuple[float, ...], bool], ...],
) -> str:
    return _digest(
        {
            "schema_id": schema_id,
            "field_order": field_order,
            "categorical_levels": categorical_levels,
            "numeric_items": numeric_items,
            "endpoint_rule": "left_closed_right_open_final_unbounded_or_finite",
        }
    )


_BIN_FIELDS = (
    "phase",
    "token_class",
    "primary_replica",
    "arrival_time",
    "phase_age",
    "queue_load",
    "reservation_balance",
)
_BIN_CATEGORIES = (
    ("phase", ("H", "D", "F", "R")),
    ("token_class", ("R", "U")),
    ("primary_replica", ("0", "1")),
)
_BIN_NUMERIC = (
    ("arrival_time", (0.0, 100.0, 200.0, 220.0, 320.0), True),
    ("phase_age", (0.0, 1.5, 3.0, 10.0), False),
    ("queue_load", (0.0, 1.0, 3.0), False),
    ("reservation_balance", (0.0, 1.0, 2.8125), False),
)
BIN_SCHEMA_V1 = ObservableBinSchema(
    BIN_SCHEMA_ID,
    _BIN_FIELDS,
    _BIN_CATEGORIES,
    _BIN_NUMERIC,
    _schema_fingerprint(BIN_SCHEMA_ID, _BIN_FIELDS, _BIN_CATEGORIES, _BIN_NUMERIC),
)


@dataclass(frozen=True)
class FirstDegradedPrimaryARule:
    rule_id: str = TARGET_SELECTION_RULE_ID

    def select(self, observation: TokenObservation) -> bool:
        if not isinstance(observation, TokenObservation):
            raise TokenObservableBinOracleError("selector requires TokenObservation")
        return observation.phase is Phase.DEGRADED and observation.primary_replica == 0


S_V1 = FirstDegradedPrimaryARule()


@dataclass(frozen=True)
class EpisodeOracleRecord:
    episode_index: int
    episode_fingerprint: str
    status: str
    target_token_id: int | None
    target_observation: TokenObservation | None
    observation_fingerprint: str | None
    bin_id: str | None
    failed_panels: int
    attempted_calls: int
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservableBinEstimate:
    model_id: str
    bin_id: str
    action: ProtectionAction
    mean: float | None
    standard_error: float | None
    effective_n: int
    status: str
    completed_panels: int
    failed_panels: int


@dataclass(frozen=True)
class ObservableBinOracleResult:
    evidence_label: str
    claims_best_response: bool
    claims_regret: bool
    claims_nash: bool
    claims_mfg: bool
    status: str
    split: str
    namespace: str
    macro_seed: int
    schema_id: str
    schema_fingerprint: str
    target_selection_rule_id: str
    episode_count: int
    minimum_complete_panels: int
    attempted_calls: int
    reserved_calls: int
    episodes: tuple[EpisodeOracleRecord, ...]
    estimates: tuple[ObservableBinEstimate, ...]


class _SelectionSource:
    def __init__(self, source: object, selector: FirstDegradedPrimaryARule):
        if not callable(getattr(source, "choose", None)):
            raise TokenObservableBinOracleError("policy factory returned invalid selector source")
        self.source = source
        self.selector = selector
        self.target_token_id: int | None = None
        self.target_observation: TokenObservation | None = None

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if self.target_token_id is None and self.selector.select(observation):
            self.target_token_id = observation.token_id
            self.target_observation = observation
        action = validate_token_action(self.source.choose(observation, policy_key))
        return action


class _CallLedger:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.reserved = 0
        self.attempted = 0

    def reserve(self, count: int) -> None:
        _strict_int(count, "call_count", 1)
        if self.reserved + count > self.maximum:
            raise TokenObservableBinOracleError(
                f"scheduler call budget exceeded: {self.reserved + count} > {self.maximum}"
            )
        self.reserved += count

    def invocation_started(self) -> None:
        if self.attempted >= self.maximum:
            raise TokenObservableBinOracleError("scheduler call budget exhausted before invocation")
        self.attempted += 1


def _validate_models(parameters: Iterable[object] | None) -> tuple[object, ...]:
    values = (
        (
            TokenInitialPriceParameters(),
            TokenRuntimeADR0009Parameters(),
            TokenExtendedReservationParameters(),
        )
        if parameters is None
        else tuple(parameters)
    )
    if not values:
        raise TokenObservableBinOracleError("at least one cost model is required")
    if any(
        type(value)
        not in {
            TokenInitialPriceParameters,
            TokenRuntimeADR0009Parameters,
            TokenExtendedReservationParameters,
        }
        for value in values
    ):
        raise TokenObservableBinOracleError("parameters must be named Token cost models")
    model_ids = [value.model_id for value in values]
    if len(set(model_ids)) != len(model_ids):
        raise TokenObservableBinOracleError("cost model IDs must be unique")
    return tuple(sorted(values, key=lambda value: _MODEL_ORDER[value.model_id]))


def _quote_for(parameters: object) -> ExternalQuote:
    return ExternalQuote(0.0, _expected_basis(parameters))


def _validate_common_inputs(
    episode_factory: Callable[[int, str, EpisodeProtocol], EpisodeTrace],
    policy_factory: Callable[[], object],
    split: str,
    namespace: str,
    macro_seed: int,
    protocol: EpisodeProtocol,
    episode_count: int,
    minimum_complete_panels: int,
    max_scheduler_calls: int,
    degraded_slowdown: float,
    hedge_delay: float | None,
) -> None:
    if not callable(episode_factory):
        raise TokenObservableBinOracleError("episode_factory must be callable")
    if not callable(policy_factory):
        raise TokenObservableBinOracleError("policy_factory must be callable")
    if not isinstance(split, str) or not split:
        raise TokenObservableBinOracleError("split must be non-empty")
    if not isinstance(namespace, str) or not namespace or any(character.isspace() for character in namespace):
        raise TokenObservableBinOracleError("namespace must be non-empty and whitespace-free")
    _strict_int(macro_seed, "macro_seed")
    if not isinstance(protocol, EpisodeProtocol):
        raise TokenObservableBinOracleError("protocol must be EpisodeProtocol")
    _strict_int(episode_count, "episode_count", 1)
    _strict_int(minimum_complete_panels, "minimum_complete_panels", 1)
    _strict_int(max_scheduler_calls, "max_scheduler_calls", 1)
    if max_scheduler_calls > FROZEN_MAX_SCHEDULER_CALLS:
        raise TokenObservableBinOracleError("max_scheduler_calls exceeds frozen T3A maximum")
    validate_slowdown(degraded_slowdown)
    if hedge_delay is not None:
        _strict_real(hedge_delay, "hedge_delay", nonnegative=False)
        if float(hedge_delay) <= 0.0:
            raise TokenObservableBinOracleError("hedge_delay must be positive")


def _selection_pass(
    episode: EpisodeTrace,
    policy_factory: Callable[[], object],
    selector: FirstDegradedPrimaryARule,
    reservation: ReservationParameters | None,
    degraded_slowdown: float,
    hedge_delay: float | None,
    seen_sources: list[object],
) -> tuple[int | None, TokenObservation | None, str | None]:
    source = policy_factory()
    if source is None or not callable(getattr(source, "choose", None)):
        raise TokenObservableBinOracleError("policy factory returned an invalid source")
    if any(source is previous for previous in seen_sources):
        raise TokenObservableBinOracleError("policy factory reused mutable source state")
    seen_sources.append(source)
    runner = _SelectionSource(source, selector)
    before = _trace_fingerprint(episode)
    simulate_episode_online(
        episode,
        runner,
        reservation,
        degraded_slowdown=degraded_slowdown,
        hedge_delay=hedge_delay,
    )
    after = _trace_fingerprint(episode)
    if before != after:
        raise TokenObservableBinOracleError("episode trace changed during target selection")
    return runner.target_token_id, runner.target_observation, None


def _panel(
    episode: EpisodeTrace,
    target_token_id: int,
    policy_factory: Callable[[], object],
    parameters: object,
    quote: ExternalQuote,
    reservation: ReservationParameters | None,
    degraded_slowdown: float,
    hedge_delay: float | None,
    seen_sources: list[object],
) -> tuple[str, int, tuple[tuple[ProtectionAction, float], ...], str | None]:
    def tracked_factory():
        source = policy_factory()
        if source is None or not callable(getattr(source, "choose", None)):
            raise TokenObservableBinOracleError("policy factory returned an invalid source")
        if any(source is previous for previous in seen_sources):
            raise TokenObservableBinOracleError("policy factory reused mutable source state")
        seen_sources.append(source)
        return source

    result = evaluate_token_pathwise_deviations(
        episode,
        tracked_factory,
        target_token_id,
        candidates=tuple(ProtectionAction),
        parameters=parameters,
        quote=quote,
        reservation=reservation,
        degraded_slowdown=degraded_slowdown,
        hedge_delay=hedge_delay,
    )
    if result.status != "completed" or len(result.rows) != 3:
        return "failed", result.attempted_calls, (), result.failure_reason or "counterfactual panel failed"
    rows = tuple(sorted(result.rows, key=lambda row: _ACTION_ORDER[row.candidate_requested]))
    first = rows[0]
    for row in rows[1:]:
        if (
            row.candidate_target_observation_fingerprint != first.candidate_target_observation_fingerprint
            or row.candidate_prefix_fingerprint != first.candidate_prefix_fingerprint
            or row.baseline_target_observation_fingerprint != first.baseline_target_observation_fingerprint
            or row.baseline_prefix_fingerprint != first.baseline_prefix_fingerprint
        ):
            return "failed", result.attempted_calls, (), "target prefix diverged within paired panel"
    values = tuple((row.candidate_requested, float(row.candidate_payoff.total)) for row in rows)
    return "completed", result.attempted_calls, values, None


def _estimate_rows(
    samples: Mapping[tuple[str, str, ProtectionAction], list[float]],
    failures: Mapping[tuple[str, str], int],
    minimum_complete_panels: int,
    model_ids: tuple[str, ...],
    bin_ids: tuple[str, ...],
) -> tuple[ObservableBinEstimate, ...]:
    output: list[ObservableBinEstimate] = []
    for model_id in sorted(model_ids, key=_MODEL_ORDER.__getitem__):
        for bin_id in sorted(bin_ids):
            failed = failures.get((model_id, bin_id), 0)
            for action in sorted(ProtectionAction, key=_ACTION_ORDER.__getitem__):
                values = samples.get((model_id, bin_id, action), [])
                complete = len(values)
                if complete < minimum_complete_panels:
                    status = "failed" if complete == 0 and failed else "insufficient"
                    mean = None
                    standard_error = None
                else:
                    mean = math.fsum(values) / complete
                    variance = math.fsum(
                        (value - mean) ** 2 for value in values
                    ) / (complete - 1)
                    standard_error = math.sqrt(variance / complete)
                    status = "completed"
                output.append(
                    ObservableBinEstimate(
                        model_id=model_id,
                        bin_id=bin_id,
                        action=action,
                        mean=mean,
                        standard_error=standard_error,
                        effective_n=complete,
                        status=status,
                        completed_panels=complete,
                        failed_panels=failed,
                    )
                )
    return tuple(output)


def run_observable_bin_oracle(
    episode_factory: Callable[[int, str, EpisodeProtocol], EpisodeTrace],
    policy_factory: Callable[[], object],
    *,
    split: str,
    namespace: str,
    macro_seed: int,
    protocol: EpisodeProtocol,
    episode_count: int,
    parameters: Iterable[object] | None = None,
    selector: FirstDegradedPrimaryARule = S_V1,
    schema: ObservableBinSchema = BIN_SCHEMA_V1,
    reservation: ReservationParameters | None = None,
    degraded_slowdown: float = 2.0,
    hedge_delay: float | None = None,
    minimum_complete_panels: int = MIN_COMPLETE_PANELS_PER_BIN,
    max_scheduler_calls: int = FROZEN_MAX_SCHEDULER_CALLS,
) -> ObservableBinOracleResult:
    """Run T3A over an episode-indexed exogenous factory.

    The explicit-count form is also used by small deterministic tests.  The
    frozen implementation-validation entry point is
    :func:`run_frozen_observable_bin_oracle` below.
    """

    _validate_common_inputs(
        episode_factory,
        policy_factory,
        split,
        namespace,
        macro_seed,
        protocol,
        episode_count,
        minimum_complete_panels,
        max_scheduler_calls,
        degraded_slowdown,
        hedge_delay,
    )
    if not isinstance(selector, FirstDegradedPrimaryARule):
        raise TokenObservableBinOracleError("selector must be the frozen S_v1 rule type")
    if not isinstance(schema, ObservableBinSchema) or schema.schema_id != BIN_SCHEMA_ID:
        raise TokenObservableBinOracleError("schema must be BIN_SCHEMA_V1")
    models = _validate_models(parameters)
    ledger = _CallLedger(max_scheduler_calls)
    # Retain every source object so CPython cannot recycle an identity while
    # later branches are still subject to the fresh-state contract.
    seen_sources: list[object] = []
    samples: dict[tuple[str, str, ProtectionAction], list[float]] = {}
    failures: dict[tuple[str, str], int] = {}
    model_ids = tuple(model.model_id for model in models)
    observed_bins: set[str] = set()
    records: list[EpisodeOracleRecord] = []
    episode_fingerprints: set[str] = set()
    overall_status = "completed"

    for episode_index in range(episode_count):
        episode = episode_factory(episode_index, namespace, protocol)
        if not isinstance(episode, EpisodeTrace):
            raise TokenObservableBinOracleError("episode_factory must return EpisodeTrace")
        episode_fp = _trace_fingerprint(episode)
        if episode_fp in episode_fingerprints:
            raise TokenObservableBinOracleError("duplicate episode trace fingerprint")
        episode_fingerprints.add(episode_fp)
        ledger.reserve(1)
        ledger.invocation_started()
        try:
            target_id, target_observation, _ = _selection_pass(
                episode,
                policy_factory,
                selector,
                reservation,
                degraded_slowdown,
                hedge_delay,
                seen_sources,
            )
        except Exception as error:
            overall_status = "failed"
            records.append(
                EpisodeOracleRecord(
                    episode_index,
                    episode_fp,
                    "failed",
                    None,
                    None,
                    None,
                    None,
                    0,
                    1,
                    (f"{type(error).__name__}: {error}",),
                )
            )
            continue
        if target_id is None or target_observation is None:
            records.append(
                EpisodeOracleRecord(
                    episode_index,
                    episode_fp,
                    "missing_target",
                    None,
                    None,
                    None,
                    None,
                    0,
                    1,
                )
            )
            continue
        try:
            bin_key = schema.bin(target_observation)
        except Exception as error:
            records.append(
                EpisodeOracleRecord(
                    episode_index,
                    episode_fp,
                    "missing_bin",
                    target_id,
                    target_observation,
                    observation_fingerprint(target_observation),
                    None,
                    0,
                    1,
                    (f"{type(error).__name__}: {error}",),
                )
            )
            continue
        failed_panels = 0
        episode_attempted = 1
        reasons: list[str] = []
        observed_bins.add(bin_key.bin_id)
        for model in models:
            ledger.reserve(4)
            status, calls, values, reason = _panel(
                episode,
                target_id,
                policy_factory,
                model,
                _quote_for(model),
                reservation,
                degraded_slowdown,
                hedge_delay,
                seen_sources,
            )
            ledger.attempted += calls
            ledger.reserved += 0
            episode_attempted += calls
            if status != "completed":
                failed_panels += 1
                reasons.append(f"{model.model_id}: {reason}")
                failures[(model.model_id, bin_key.bin_id)] = (
                    failures.get((model.model_id, bin_key.bin_id), 0) + 1
                )
                continue
            for action, value in values:
                samples.setdefault((model.model_id, bin_key.bin_id, action), []).append(value)
        if failed_panels:
            overall_status = "partial" if overall_status == "completed" else overall_status
        records.append(
            EpisodeOracleRecord(
                episode_index,
                episode_fp,
                "completed" if failed_panels == 0 else "panel_failed",
                target_id,
                target_observation,
                observation_fingerprint(target_observation),
                bin_key.bin_id,
                failed_panels,
                episode_attempted,
                tuple(reasons),
            )
        )

    estimates = _estimate_rows(
        samples,
        failures,
        minimum_complete_panels,
        model_ids,
        tuple(observed_bins),
    )
    return ObservableBinOracleResult(
        evidence_label=EVIDENCE_LABEL,
        claims_best_response=False,
        claims_regret=False,
        claims_nash=False,
        claims_mfg=False,
        status=overall_status,
        split=split,
        namespace=namespace,
        macro_seed=macro_seed,
        schema_id=schema.schema_id,
        schema_fingerprint=schema.fingerprint,
        target_selection_rule_id=selector.rule_id,
        episode_count=episode_count,
        minimum_complete_panels=minimum_complete_panels,
        attempted_calls=ledger.attempted,
        reserved_calls=ledger.reserved,
        episodes=tuple(records),
        estimates=estimates,
    )


def run_frozen_observable_bin_oracle(
    episode_factory: Callable[[int, str, EpisodeProtocol], EpisodeTrace],
    policy_factory: Callable[[], object],
    *,
    split: str,
    protocol: EpisodeProtocol,
    parameters: Iterable[object] | None = None,
    reservation: ReservationParameters | None = None,
    degraded_slowdown: float = 2.0,
    hedge_delay: float | None = 2.0,
) -> ObservableBinOracleResult:
    """Run the frozen 16-episode implementation-validation configuration."""

    if split == "calibration":
        namespace = "token-mfg-restoration:t3a:calibration:v1"
        macro_seed = 20260907
        episode_count = FROZEN_CALIBRATION_EPISODE_COUNT
    elif split == "validation":
        namespace = "token-mfg-restoration:t3a:validation:v1"
        macro_seed = 20260908
        episode_count = FROZEN_VALIDATION_EPISODE_COUNT
    else:
        raise TokenObservableBinOracleError("frozen split must be calibration or validation")
    return run_observable_bin_oracle(
        episode_factory,
        policy_factory,
        split=split,
        namespace=namespace,
        macro_seed=macro_seed,
        protocol=protocol,
        episode_count=episode_count,
        parameters=parameters,
        reservation=reservation,
        degraded_slowdown=degraded_slowdown,
        hedge_delay=hedge_delay,
        minimum_complete_panels=MIN_COMPLETE_PANELS_PER_BIN,
        max_scheduler_calls=FROZEN_MAX_SCHEDULER_CALLS,
    )


__all__ = [
    "BIN_SCHEMA_ID",
    "BIN_SCHEMA_V1",
    "EVIDENCE_LABEL",
    "FROZEN_CALIBRATION_EPISODE_COUNT",
    "FROZEN_MAX_SCHEDULER_CALLS",
    "FROZEN_VALIDATION_EPISODE_COUNT",
    "MIN_COMPLETE_PANELS_PER_BIN",
    "ORACLE_MODE",
    "S_V1",
    "TARGET_SELECTION_RULE_ID",
    "EpisodeOracleRecord",
    "FirstDegradedPrimaryARule",
    "ObservableBin",
    "ObservableBinEstimate",
    "ObservableBinOracleResult",
    "ObservableBinSchema",
    "TokenObservableBinOracleError",
    "bin_observation",
    "observation_fingerprint",
    "run_observable_bin_oracle",
    "run_frozen_observable_bin_oracle",
]


def bin_observation(observation: TokenObservation) -> ObservableBin:
    """Apply the frozen ``bin_schema_v1`` to one causal observation."""

    return BIN_SCHEMA_V1.bin(observation)
