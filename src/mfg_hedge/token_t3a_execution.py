"""Accepted ADR-0023 occupancy and statistical Token-oracle execution.

This module deliberately stops at conditional N/D/I cost estimates.  It does
not select an action or compute a best response, regret, Nash, or MFG result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeProtocol,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
)
from .artifacts import write_run_directory
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction, TokenClass
from .token_continuation import BIN_SCHEMA_V1, EVIDENCE_LABEL, observation_fingerprint
from .token_deviations import evaluate_token_pathwise_deviations
from .token_online import (
    StaticTimeClassActionSource,
    TokenObservation,
    simulate_episode_online,
    validate_token_action,
)
from .token_payoff import (
    ADMITTED_RESERVED_WORK,
    ExternalQuote,
    INCREMENTAL_EXECUTED_WORK,
    TokenExtendedReservationParameters,
    TokenInitialPriceParameters,
    TokenRuntimeADR0009Parameters,
    score_token_payoff,
)
from .transient_control import ReservationParameters, TimeClassRule


SOURCE_BUNDLE_SCHEMA = "token_t3a_source_bundle_v1"
ENVIRONMENT_ID = "theta_token_load0p7_v1"
POPULATION_POLICY_ID = "pi_NIIN_v1"
TARGET_SELECTION_ID = "target_selection_v2:anchored_degraded_primary_A"
OCCUPANCY_NAMESPACE = "token-mfg-restoration:t3a:occupancy:v1"
VALIDATION_NAMESPACE = "token-mfg-restoration:t3a:oracle-validation:v1"
OCCUPANCY_MACRO_SEED = 20260909
VALIDATION_MACRO_SEED = 20260910
OCCUPANCY_EPISODES = 512
OCCUPANCY_FLOOR = 16
PANEL_FLOOR = 32
MAX_VALIDATION_EPISODES = 2048
OCCUPANCY_CALL_LIMIT = 512
VALIDATION_CALL_LIMIT = 10_240
COMBINED_CALL_LIMIT = 10_752
ANCHOR_WINDOWS: tuple[tuple[float, float | None], ...] = (
    (0.0, 1.5),
    (1.5, 3.0),
    (3.0, 10.0),
    (10.0, None),
)
_ACTIONS = tuple(ProtectionAction)
_MODELS = (
    TokenInitialPriceParameters(),
    TokenRuntimeADR0009Parameters(),
    TokenExtendedReservationParameters(),
)
_RESERVATION = ReservationParameters(
    window_width=25.0,
    budget_rate=0.45,
    scale=0.25,
    mean_requirement=1.0,
)
_NIIN_RULE = TimeClassRule(
    ProtectionAction.NORMAL,
    ProtectionAction.IMMEDIATE_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
    ProtectionAction.NORMAL,
)
_SOURCE_ROOTS = (
    "src/mfg_hedge/hedge_simulation.py",
    "src/mfg_hedge/token_online.py",
    "src/mfg_hedge/transient_control.py",
    "src/mfg_hedge/token_payoff.py",
    "src/mfg_hedge/token_deviations.py",
    "src/mfg_hedge/token_continuation.py",
    "src/mfg_hedge/workload.py",
    "src/mfg_hedge/common_state.py",
    "src/mfg_hedge/attribution_episode.py",
    "src/mfg_hedge/token_t3a_execution.py",
)


class T3AExecutionError(RuntimeError):
    """Fail-closed ADR-0023 contract violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _trace_fingerprint(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


@dataclass(frozen=True)
class T3ASourceBundle:
    schema_id: str
    roots: tuple[str, ...]
    entries: tuple[tuple[str, str], ...]
    fingerprint: str

    def verify(self, project_root: str | Path) -> None:
        current = build_source_bundle(project_root)
        if current != self:
            raise T3AExecutionError("source bundle changed after it was sealed")


def build_source_bundle(project_root: str | Path) -> T3ASourceBundle:
    root = Path(project_root).resolve()
    source_root = root / "src" / "mfg_hedge"
    config_path = root / "configs" / "v1_minimal.json"
    if not source_root.is_dir() or not config_path.is_file():
        raise T3AExecutionError("project root lacks source tree or v1_minimal config")
    missing = [name for name in _SOURCE_ROOTS if not (root / Path(name)).is_file()]
    if missing:
        raise T3AExecutionError(f"source-bundle roots are missing: {missing}")
    paths = sorted(source_root.rglob("*.py")) + [config_path]
    entries = tuple(
        (
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    )
    payload = {
        "schema_id": SOURCE_BUNDLE_SCHEMA,
        "roots": _SOURCE_ROOTS,
        "entries": entries,
    }
    return T3ASourceBundle(
        SOURCE_BUNDLE_SCHEMA,
        _SOURCE_ROOTS,
        entries,
        _digest(payload),
    )


def _config_payload(config: ExperimentConfig) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def validate_frozen_config(config: ExperimentConfig) -> None:
    if not isinstance(config, ExperimentConfig):
        raise T3AExecutionError("config must be ExperimentConfig")
    expected = {
        "schema_version": 1,
        "experiment_name": "single_expert_mechanism_check",
        "expert_count": 1,
        "replicas_per_expert": 2,
        "failure_domain_count": 2,
        "regular_token_ratio": 0.8,
        "urgent_token_ratio": 0.2,
        "healthy_service_mean": 1.0,
        "service_time_cv": 0.5,
        "degraded_slowdown": 2.0,
        "healthy_offered_load": 0.7,
        "base_seed": 20260901,
    }
    mismatches = {
        name: (getattr(config, name), value)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise T3AExecutionError(f"config differs from {ENVIRONMENT_ID}: {mismatches}")


def environment_fingerprint(config: ExperimentConfig) -> str:
    validate_frozen_config(config)
    timeline = ATTRIBUTION_V1_PROTOCOL.timeline
    payload = {
        "environment_id": ENVIRONMENT_ID,
        "config": _config_payload(config),
        "arrival_cutoff": ATTRIBUTION_V1_PROTOCOL.arrival_cutoff,
        "timeline": [
            timeline.degraded_start,
            timeline.failed_start,
            timeline.recovered_start,
        ],
        "degraded_slowdown": 2.0,
        "hedge_delay": 2.0,
        "reservation": asdict(_RESERVATION),
        "dispatcher": "token_id_parity_except_failed_primary_B",
        "drain": "all_attempts_including_running_losers",
        "quote": "episode_constant_zero",
        "stream_labels": [
            "arrival",
            "token-class",
            "service:{replica}",
            "service:{replica}:attempt1",
            "service:{replica}:attempt2",
        ],
    }
    return _digest(payload)


def population_policy_fingerprint() -> str:
    return _digest(
        {
            "policy_id": POPULATION_POLICY_ID,
            "source": "token_static_time_class_v1",
            "rule": [action.value for action in _NIIN_RULE.actions],
            "rule_name": "NIIN",
            "late_after": 50.0,
            "fresh_state": "new_episode",
            "randomness": "none",
            "policy_key": "token:{base_seed}:{token_id}",
        }
    )


def policy_factory() -> StaticTimeClassActionSource:
    return StaticTimeClassActionSource(_NIIN_RULE, late_after=50.0)


@dataclass(frozen=True)
class T3AAnchorSelector:
    anchor_id: int

    def __post_init__(self) -> None:
        if type(self.anchor_id) is not int or not 0 <= self.anchor_id < 4:
            raise T3AExecutionError("anchor_id must be a true int in 0..3")

    @property
    def rule_id(self) -> str:
        return f"{TARGET_SELECTION_ID}:anchor:{self.anchor_id}"

    def select(self, observation: TokenObservation) -> bool:
        if not isinstance(observation, TokenObservation):
            raise T3AExecutionError("selector requires TokenObservation")
        lower, upper = ANCHOR_WINDOWS[self.anchor_id]
        return (
            observation.phase.value == "D"
            and observation.primary_replica == 0
            and observation.phase_age >= lower
            and (upper is None or observation.phase_age < upper)
        )


class _CaptureSource:
    def __init__(self, source: object, selector: T3AAnchorSelector) -> None:
        self.source = source
        self.selector = selector
        self.target_id: int | None = None
        self.observation: TokenObservation | None = None

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if self.target_id is None and self.selector.select(observation):
            self.target_id = observation.token_id
            self.observation = observation
        return validate_token_action(self.source.choose(observation, policy_key))


@dataclass(frozen=True)
class T3AProvenance:
    source_bundle_fingerprint: str
    theta_fingerprint: str
    population_policy_fingerprint: str
    bin_schema_fingerprint: str
    target_selection_fingerprint: str


def build_provenance(config: ExperimentConfig, bundle: T3ASourceBundle) -> T3AProvenance:
    return T3AProvenance(
        bundle.fingerprint,
        environment_fingerprint(config),
        population_policy_fingerprint(),
        BIN_SCHEMA_V1.fingerprint,
        _digest({"rule": TARGET_SELECTION_ID, "anchors": ANCHOR_WINDOWS}),
    )


@dataclass(frozen=True)
class OccupancyEpisode:
    episode_index: int
    anchor_id: int
    status: str
    bin_id: str | None
    target_token_id: int | None
    episode_fingerprint: str
    observation_fingerprint: str | None
    attempted_calls: int
    reason: str | None = None


@dataclass(frozen=True)
class OccupancyResult:
    evidence_label: str
    status: str
    namespace: str
    macro_seed: int
    protocol_id: str
    episode_count: int
    attempted_calls: int
    occupancy_floor: int
    counts: tuple[tuple[str, int], ...]
    retained_bins: tuple[str, ...]
    validation_episode_count: int | None
    provenance: T3AProvenance
    episodes: tuple[OccupancyEpisode, ...]
    claims_best_response: bool = False
    claims_regret: bool = False
    claims_nash: bool = False
    claims_mfg: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validation_episode_count(counts: Mapping[str, int]) -> int:
    if not isinstance(counts, Mapping):
        raise T3AExecutionError("occupancy counts must be a mapping")
    retained: list[int] = []
    for bin_id, count in counts.items():
        if not isinstance(bin_id, str) or not bin_id or type(count) is not int or count < 0:
            raise T3AExecutionError("occupancy counts contain an invalid row")
        if count >= OCCUPANCY_FLOOR:
            retained.append(count)
    if not retained:
        raise T3AExecutionError("occupancy screen retained no bins")
    minimum = min(retained)
    result = math.ceil(2 * PANEL_FLOOR * OCCUPANCY_EPISODES / minimum)
    if result > MAX_VALIDATION_EPISODES:
        raise T3AExecutionError("validation allocation exceeds frozen maximum")
    return result


def _episode_factory(config: ExperimentConfig, namespace: str, macro_seed: int):
    def create(index: int) -> EpisodeTrace:
        return generate_episode_trace(
            config,
            namespace,
            macro_seed,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )

    return create


def _select_target(
    episode: EpisodeTrace,
    selector: T3AAnchorSelector,
) -> tuple[int | None, TokenObservation | None]:
    capture = _CaptureSource(policy_factory(), selector)
    simulate_episode_online(
        episode,
        capture,
        _RESERVATION,
        degraded_slowdown=2.0,
        hedge_delay=2.0,
    )
    return capture.target_id, capture.observation


def run_occupancy(
    config: ExperimentConfig,
    bundle: T3ASourceBundle,
    *,
    episode_count: int = OCCUPANCY_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
) -> OccupancyResult:
    validate_frozen_config(config)
    if type(episode_count) is not int or episode_count <= 0 or episode_count > OCCUPANCY_EPISODES:
        raise T3AExecutionError("occupancy episode_count is outside 1..512")
    provenance = build_provenance(config, bundle)
    build = episode_builder or _episode_factory(
        config, OCCUPANCY_NAMESPACE, OCCUPANCY_MACRO_SEED
    )
    records: list[OccupancyEpisode] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    overall = "completed"
    for index in range(episode_count):
        episode = build(index)
        fingerprint = _trace_fingerprint(episode)
        if fingerprint in seen:
            raise T3AExecutionError("duplicate occupancy episode fingerprint")
        seen.add(fingerprint)
        selector = T3AAnchorSelector(index % 4)
        try:
            token_id, observation = _select_target(episode, selector)
            if token_id is None or observation is None:
                records.append(OccupancyEpisode(index, index % 4, "missing_target", None, None, fingerprint, None, 1))
                continue
            try:
                bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
            except Exception as error:
                records.append(OccupancyEpisode(index, index % 4, "missing_bin", None, token_id, fingerprint, observation_fingerprint(observation), 1, f"{type(error).__name__}: {error}"))
                continue
            counts[bin_id] = counts.get(bin_id, 0) + 1
            records.append(OccupancyEpisode(index, index % 4, "completed", bin_id, token_id, fingerprint, observation_fingerprint(observation), 1))
        except Exception as error:
            overall = "failed"
            records.append(OccupancyEpisode(index, index % 4, "failed", None, None, fingerprint, None, 1, f"{type(error).__name__}: {error}"))
    retained = tuple(sorted(key for key, value in counts.items() if value >= OCCUPANCY_FLOOR))
    allocation = validation_episode_count(counts) if retained and episode_count == OCCUPANCY_EPISODES else None
    return OccupancyResult(
        "t3a_occupancy_calibration",
        overall,
        OCCUPANCY_NAMESPACE,
        OCCUPANCY_MACRO_SEED,
        "occupancy_v1",
        episode_count,
        episode_count,
        OCCUPANCY_FLOOR,
        tuple(sorted(counts.items())),
        retained,
        allocation,
        provenance,
        tuple(records),
    )


@dataclass(frozen=True)
class ValidationEpisode:
    episode_index: int
    anchor_id: int
    status: str
    bin_id: str | None
    target_token_id: int | None
    episode_fingerprint: str
    observation_fingerprint: str | None
    attempted_calls: int
    failed_models: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ConditionalCostRow:
    model_id: str
    model_fingerprint: str
    quote_value: float
    quote_basis: str
    bin_id: str
    requested_action: str
    effective_n: int
    mean: float | None
    standard_error: float | None
    status: str
    failed_panels: int


@dataclass(frozen=True)
class ValidationResult:
    evidence_label: str
    status: str
    namespace: str
    macro_seed: int
    protocol_id: str
    episode_count: int
    attempted_calls: int
    panel_floor: int
    retained_bins: tuple[str, ...]
    provenance: T3AProvenance
    occupancy_fingerprint: str | None
    episodes: tuple[ValidationEpisode, ...]
    estimates: tuple[ConditionalCostRow, ...]
    claims_best_response: bool = False
    claims_regret: bool = False
    claims_nash: bool = False
    claims_mfg: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _model_payload(parameters: object) -> dict[str, object]:
    basis = (
        ADMITTED_RESERVED_WORK
        if type(parameters) is TokenExtendedReservationParameters
        else INCREMENTAL_EXECUTED_WORK
    )
    return {
        "model_id": parameters.model_id,
        "schema_id": parameters.schema_id,
        "parameters": {
            field.name: getattr(parameters, field.name)
            for field in fields(parameters)
        },
        "quote": {"value": 0.0, "basis": basis},
    }


def _quote(parameters: object) -> ExternalQuote:
    return ExternalQuote(0.0, _model_payload(parameters)["quote"]["basis"])


def _estimate_validation_rows(
    retained_bins: tuple[str, ...],
    samples: Mapping[tuple[str, str, ProtectionAction], list[float]],
    failures: Mapping[tuple[str, str], int],
    minimum_complete_panels: int,
) -> tuple[ConditionalCostRow, ...]:
    rows: list[ConditionalCostRow] = []
    for parameters in _MODELS:
        payload = _model_payload(parameters)
        model_fingerprint = _digest(payload)
        quote = payload["quote"]
        for bin_id in retained_bins:
            failed = failures.get((parameters.model_id, bin_id), 0)
            for action in _ACTIONS:
                values = samples.get((parameters.model_id, bin_id, action), [])
                count = len(values)
                if count < minimum_complete_panels:
                    mean = None
                    standard_error = None
                    status = "insufficient"
                else:
                    mean = math.fsum(values) / count
                    if count == 1:
                        standard_error = 0.0
                    else:
                        variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
                        standard_error = math.sqrt(variance / count)
                    status = "completed"
                rows.append(
                    ConditionalCostRow(
                        parameters.model_id,
                        model_fingerprint,
                        float(quote["value"]),
                        str(quote["basis"]),
                        bin_id,
                        action.value,
                        count,
                        mean,
                        standard_error,
                        status,
                        failed,
                    )
                )
    return tuple(rows)


def _run_validation(
    config: ExperimentConfig,
    bundle: T3ASourceBundle,
    *,
    retained_bins: tuple[str, ...],
    episode_count: int,
    minimum_complete_panels: int = PANEL_FLOOR,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ValidationResult:
    """Execute a frozen-library validation; count overrides exist for tests only."""

    validate_frozen_config(config)
    if (
        type(episode_count) is not int
        or episode_count <= 0
        or episode_count > MAX_VALIDATION_EPISODES
    ):
        raise T3AExecutionError("validation episode_count is outside 1..2048")
    if (
        type(minimum_complete_panels) is not int
        or minimum_complete_panels <= 0
        or minimum_complete_panels > PANEL_FLOOR
    ):
        raise T3AExecutionError("minimum_complete_panels is outside 1..32")
    if (
        not isinstance(retained_bins, tuple)
        or not retained_bins
        or tuple(sorted(set(retained_bins))) != retained_bins
    ):
        raise T3AExecutionError("retained_bins must be a non-empty sorted unique tuple")
    maximum_calls = episode_count * 5
    if maximum_calls > VALIDATION_CALL_LIMIT:
        raise T3AExecutionError("validation scheduler-call budget would be exceeded")

    provenance = build_provenance(config, bundle)
    build = episode_builder or _episode_factory(
        config, VALIDATION_NAMESPACE, VALIDATION_MACRO_SEED
    )
    samples: dict[tuple[str, str, ProtectionAction], list[float]] = {}
    failures: dict[tuple[str, str], int] = {}
    records: list[ValidationEpisode] = []
    seen: set[str] = set()
    attempted_calls = 0

    for index in range(episode_count):
        episode = build(index)
        episode_fp = _trace_fingerprint(episode)
        if episode_fp in seen:
            raise T3AExecutionError("duplicate validation episode fingerprint")
        seen.add(episode_fp)
        selector = T3AAnchorSelector(index % 4)
        attempted_calls += 1
        try:
            target_id, observation = _select_target(episode, selector)
        except Exception as error:
            records.append(
                ValidationEpisode(
                    index, index % 4, "failed", None, None, episode_fp, None, 1,
                    tuple(model.model_id for model in _MODELS),
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        if target_id is None or observation is None:
            records.append(
                ValidationEpisode(index, index % 4, "missing_target", None, None, episode_fp, None, 1)
            )
            continue
        selected_fp = observation_fingerprint(observation)
        try:
            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception as error:
            records.append(
                ValidationEpisode(index, index % 4, "missing_bin", None, target_id, episode_fp, selected_fp, 1, (), f"{type(error).__name__}: {error}")
            )
            continue

        physical = evaluate_token_pathwise_deviations(
            episode,
            policy_factory,
            target_id,
            candidates=_ACTIONS,
            parameters=_MODELS[0],
            quote=_quote(_MODELS[0]),
            reservation=_RESERVATION,
            degraded_slowdown=2.0,
            hedge_delay=2.0,
        )
        attempted_calls += physical.attempted_calls
        if physical.status != "completed" or len(physical.rows) != 3:
            for model in _MODELS:
                if bin_id in retained_bins:
                    failures[(model.model_id, bin_id)] = failures.get((model.model_id, bin_id), 0) + 1
            records.append(
                ValidationEpisode(
                    index, index % 4, "physical_failed", bin_id, target_id,
                    episode_fp, selected_fp, 1 + physical.attempted_calls,
                    tuple(model.model_id for model in _MODELS),
                    physical.failure_reason or "paired physical panel failed",
                )
            )
            continue
        by_action = {row.candidate_requested: row for row in physical.rows}
        if set(by_action) != set(_ACTIONS) or any(
            row.baseline_target_observation_fingerprint != selected_fp
            or row.candidate_target_observation_fingerprint != selected_fp
            for row in physical.rows
        ):
            raise T3AExecutionError("selection and N/D/I target prefixes disagree")

        failed_models: list[str] = []
        for parameters in _MODELS:
            try:
                values = tuple(
                    (
                        action,
                        score_token_payoff(
                            by_action[action].candidate_run,
                            target_id,
                            parameters,
                            _quote(parameters),
                        ).total,
                    )
                    for action in _ACTIONS
                )
                if any(not math.isfinite(value) for _, value in values):
                    raise T3AExecutionError("scorer returned a non-finite cost")
            except Exception:
                failed_models.append(parameters.model_id)
                if bin_id in retained_bins:
                    failures[(parameters.model_id, bin_id)] = failures.get((parameters.model_id, bin_id), 0) + 1
                continue
            if bin_id in retained_bins:
                for action, value in values:
                    samples.setdefault((parameters.model_id, bin_id, action), []).append(value)
        status = "completed" if not failed_models else "scorer_failed"
        records.append(
            ValidationEpisode(
                index, index % 4, status, bin_id, target_id, episode_fp,
                selected_fp, 1 + physical.attempted_calls, tuple(failed_models),
                None if not failed_models else "one or more pure scorers failed",
            )
        )
        if progress is not None and ((index + 1) % 64 == 0 or index + 1 == episode_count):
            progress(index + 1, episode_count)

    if attempted_calls > maximum_calls or attempted_calls > VALIDATION_CALL_LIMIT:
        raise T3AExecutionError("actual scheduler calls exceeded the frozen budget")
    estimates = _estimate_validation_rows(
        retained_bins, samples, failures, minimum_complete_panels
    )
    status = "completed" if estimates and all(row.status == "completed" for row in estimates) else "insufficient"
    return ValidationResult(
        EVIDENCE_LABEL,
        status,
        VALIDATION_NAMESPACE,
        VALIDATION_MACRO_SEED,
        "oracle-validation_v1",
        episode_count,
        attempted_calls,
        minimum_complete_panels,
        retained_bins,
        provenance,
        None,
        tuple(records),
        estimates,
    )


def run_validation(
    config: ExperimentConfig,
    bundle: T3ASourceBundle,
    occupancy: OccupancyResult,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ValidationResult:
    """Verify the complete occupancy gate, then execute the frozen validation."""

    expected_provenance = build_provenance(config, bundle)
    if (
        not isinstance(occupancy, OccupancyResult)
        or occupancy.status != "completed"
        or occupancy.episode_count != OCCUPANCY_EPISODES
        or occupancy.attempted_calls != OCCUPANCY_CALL_LIMIT
        or occupancy.namespace != OCCUPANCY_NAMESPACE
        or occupancy.macro_seed != OCCUPANCY_MACRO_SEED
        or occupancy.provenance != expected_provenance
    ):
        raise T3AExecutionError("occupancy manifest failed the frozen validation gate")
    counts = dict(occupancy.counts)
    allocation = validation_episode_count(counts)
    retained = tuple(sorted(key for key, count in counts.items() if count >= OCCUPANCY_FLOOR))
    if occupancy.retained_bins != retained or occupancy.validation_episode_count != allocation:
        raise T3AExecutionError("occupancy screen or allocation changed after sealing")
    result = _run_validation(
        config,
        bundle,
        retained_bins=retained,
        episode_count=allocation,
        minimum_complete_panels=PANEL_FLOOR,
        progress=progress,
    )
    return replace(result, occupancy_fingerprint=_digest(occupancy.to_dict()))


def _compact_occupancy(result: OccupancyResult) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for row in result.episodes:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    return {
        "evidence_label": result.evidence_label,
        "status": result.status,
        "episode_count": result.episode_count,
        "attempted_calls": result.attempted_calls,
        "status_counts": statuses,
        "observed_bin_count": len(result.counts),
        "retained_bin_count": len(result.retained_bins),
        "retained_bins": result.retained_bins,
        "validation_episode_count": result.validation_episode_count,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def _compact_validation(result: ValidationResult) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for row in result.episodes:
        statuses[row.status] = statuses.get(row.status, 0) + 1
    completed = sum(row.status == "completed" for row in result.estimates)
    return {
        "evidence_label": result.evidence_label,
        "status": result.status,
        "episode_count": result.episode_count,
        "attempted_calls": result.attempted_calls,
        "status_counts": statuses,
        "retained_bin_count": len(result.retained_bins),
        "estimate_row_count": len(result.estimates),
        "completed_estimate_rows": completed,
        "insufficient_estimate_rows": len(result.estimates) - completed,
        "occupancy_fingerprint": result.occupancy_fingerprint,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def execute_frozen_t3a(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
) -> tuple[Path, Path, OccupancyResult, ValidationResult]:
    """Run both accepted stages, sealing occupancy before validation."""

    if not isinstance(run_id, str) or not run_id:
        raise T3AExecutionError("run_id must be a non-empty string")
    project = Path(project_root).resolve()
    artifacts = Path(artifacts_root).resolve()
    occupancy_id = f"{run_id}-occupancy"
    validation_id = f"{run_id}-validation"
    if (artifacts / occupancy_id).exists() or (artifacts / validation_id).exists():
        raise FileExistsError("T3A run ID already has an occupancy or validation artifact")
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    provenance = build_provenance(config, bundle)
    occupancy = run_occupancy(config, bundle)
    if occupancy.status != "completed" or not occupancy.retained_bins:
        raise T3AExecutionError("occupancy calibration failed closed")
    occupancy_payload = occupancy.to_dict()
    occupancy_fingerprint = _digest(occupancy_payload)
    occupancy_manifest = {
        "schema_id": "token_t3a_occupancy_manifest_v1",
        "run_id": occupancy_id,
        "occupancy_fingerprint": occupancy_fingerprint,
        "source_bundle": asdict(bundle),
        "provenance": asdict(provenance),
    }
    occupancy_dir = write_run_directory(
        artifacts,
        occupancy_id,
        {
            "manifest.json": occupancy_manifest,
            "summary.json": _compact_occupancy(occupancy),
            "occupancy.json": occupancy_payload,
        },
    )
    print(
        f"occupancy sealed: bins={len(occupancy.counts)} "
        f"retained={len(occupancy.retained_bins)} "
        f"validation_episodes={occupancy.validation_episode_count}",
        flush=True,
    )
    sealed = json.loads((occupancy_dir / "occupancy.json").read_text(encoding="utf-8"))
    sealed_manifest = json.loads((occupancy_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        _digest(sealed) != occupancy_fingerprint
        or sealed_manifest.get("occupancy_fingerprint") != occupancy_fingerprint
        or sealed_manifest.get("provenance") != asdict(provenance)
    ):
        raise T3AExecutionError("sealed occupancy artifact failed verification")
    bundle.verify(project)
    validation = run_validation(
        config,
        bundle,
        occupancy,
        progress=lambda done, total: print(
            f"validation progress: {done}/{total}", flush=True
        ),
    )
    if validation.occupancy_fingerprint != occupancy_fingerprint:
        raise T3AExecutionError("validation lost the sealed occupancy identity")
    validation_payload = validation.to_dict()
    validation_manifest = {
        "schema_id": "token_t3a_validation_manifest_v1",
        "run_id": validation_id,
        "occupancy_run_id": occupancy_id,
        "occupancy_fingerprint": occupancy_fingerprint,
        "validation_fingerprint": _digest(validation_payload),
        "source_bundle": asdict(bundle),
        "provenance": asdict(provenance),
    }
    validation_dir = write_run_directory(
        artifacts,
        validation_id,
        {
            "manifest.json": validation_manifest,
            "summary.json": _compact_validation(validation),
            "estimates.json": {"rows": [asdict(row) for row in validation.estimates]},
            "episodes.json": {"rows": [asdict(row) for row in validation.episodes]},
        },
    )
    return occupancy_dir, validation_dir, occupancy, validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run accepted ADR-0023 T3A calibration and validation")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        occupancy_dir, validation_dir, occupancy, validation = execute_frozen_t3a(
            args.project_root, args.artifacts_root, args.run_id
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"occupancy: {occupancy_dir}")
    print(f"retained_bins: {len(occupancy.retained_bins)}")
    print(f"validation: {validation_dir}")
    print(f"validation_status: {validation.status}")
    return 0


__all__ = [
    "ANCHOR_WINDOWS",
    "COMBINED_CALL_LIMIT",
    "ENVIRONMENT_ID",
    "MAX_VALIDATION_EPISODES",
    "OCCUPANCY_EPISODES",
    "OCCUPANCY_FLOOR",
    "PANEL_FLOOR",
    "POPULATION_POLICY_ID",
    "T3AAnchorSelector",
    "T3AExecutionError",
    "T3AProvenance",
    "T3ASourceBundle",
    "ConditionalCostRow",
    "OccupancyEpisode",
    "OccupancyResult",
    "ValidationEpisode",
    "ValidationResult",
    "build_provenance",
    "build_source_bundle",
    "execute_frozen_t3a",
    "environment_fingerprint",
    "policy_factory",
    "population_policy_fingerprint",
    "run_occupancy",
    "run_validation",
    "validate_frozen_config",
    "validation_episode_count",
]


if __name__ == "__main__":
    raise SystemExit(main())
