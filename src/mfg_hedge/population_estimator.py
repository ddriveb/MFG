"""Causal finite-population state and routing-share calibration.

This is an isolated estimator layer for ADR-0037.  It consumes the
simultaneous finite routing engine but never chooses a best response, prices a
Replica, or solves an MFG.  ``mu``, ``nu``, ``eta``, predicted ``x``, and the
post-commit realized share are deliberately represented as separate immutable
records.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from .reliability_aware_routing import (
    DOMAINS,
    EVAL_HORIZON,
    HISTORY_WINDOW,
    REPLICA_IDS,
    RoutingTrace,
    SERVICE_CV,
    SERVICE_MEAN,
)
from .simultaneous_token_routing import (
    ActiveTokenState,
    PopulationDecisionContext,
    TokenDecisionObservation,
    SimultaneousRoutingResult,
    simulate_simultaneous_routing,
)
from .topology import DEFAULT_TOPOLOGY, Topology


PROTOCOL_ID = "reliability-aware-token-mfg:population-v1"
FORWARD_NAMESPACE = "reliability-aware-token-mfg:v1:forward"
CALIBRATION_NAMESPACE = "reliability-aware-token-mfg:v1:share-calibration"
FORWARD_MACRO_SEED = 20260910
CALIBRATION_MACRO_SEED = 20260911
FORWARD_EPISODE_COUNT = 16
CALIBRATION_EPISODE_COUNT = 32
IMPLEMENTATION_CALL_BUDGET = FORWARD_EPISODE_COUNT + CALIBRATION_EPISODE_COUNT
MIN_EPISODE_SAMPLES = 2


class PopulationEstimatorError(ValueError):
    """Raised when a population-state or calibration contract fails."""


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PopulationEstimatorError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PopulationEstimatorError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PopulationEstimatorError(f"{name} must be finite")
    return result


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PopulationEstimatorError("value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _validate_edges(edges: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, f"{name} edge") for value in edges)
    if any(value <= 0.0 for value in result) or tuple(sorted(result)) != result:
        raise PopulationEstimatorError(f"{name} edges must be positive and increasing")
    if len(set(result)) != len(result):
        raise PopulationEstimatorError(f"{name} edges must be unique")
    return result


def _bucket(value: float, edges: tuple[float, ...], prefix: str) -> str:
    value = _finite(value, f"{prefix} value")
    if value < 0.0:
        raise PopulationEstimatorError(f"{prefix} value must be non-negative")
    lower = 0.0
    for upper in edges:
        if lower <= value < upper:
            return f"{prefix}_{_label(lower)}_{_label(upper)}"
        lower = upper
    return f"{prefix}_{_label(lower)}_inf"


def _label(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return format(value, "g").replace(".", "p")


@dataclass(frozen=True)
class BucketSchema:
    """Frozen observable state bucket boundaries for ``bin_schema_v1``."""

    schema_id: str
    token_age_edges: tuple[float, ...]
    retry_edges: tuple[float, ...]
    queue_depth_edges: tuple[float, ...]
    estimated_work_edges: tuple[float, ...]
    running_age_edges: tuple[float, ...]
    hazard_edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_id != "bin_schema_v1":
            raise PopulationEstimatorError("only bin_schema_v1 is supported")
        for name in (
            "token_age_edges",
            "retry_edges",
            "queue_depth_edges",
            "estimated_work_edges",
            "running_age_edges",
            "hazard_edges",
        ):
            edges = getattr(self, name)
            if not isinstance(edges, tuple):
                raise PopulationEstimatorError(f"{name} must be an immutable tuple")
            _validate_edges(edges, name)

    @property
    def fingerprint(self) -> str:
        return _digest({
            "schema_id": self.schema_id,
            "token_age_edges": self.token_age_edges,
            "retry_edges": self.retry_edges,
            "queue_depth_edges": self.queue_depth_edges,
            "estimated_work_edges": self.estimated_work_edges,
            "running_age_edges": self.running_age_edges,
            "hazard_edges": self.hazard_edges,
            "endpoint_rule": "left_closed_right_open_final_unbounded",
        })

    def bucket_age(self, value: float) -> str:
        return _bucket(value, self.token_age_edges, "age")

    def bucket_retry_count(self, value: int) -> str:
        _strict_int(value, "retry_count")
        return _bucket(float(value), self.retry_edges, "retry")

    def bucket_queue_depth(self, value: int) -> str:
        _strict_int(value, "queue_depth")
        return _bucket(float(value), self.queue_depth_edges, "queue")

    def bucket_estimated_work(self, value: float) -> str:
        return _bucket(value, self.estimated_work_edges, "work")

    def bucket_running_age(self, value: float | None) -> str:
        if value is None:
            return "idle"
        return _bucket(value, self.running_age_edges, "service_age")

    def bucket_hazard(self, value: float) -> str:
        return _bucket(value, self.hazard_edges, "hazard")

    def bucket_token(self, observation: TokenDecisionObservation) -> "TokenBucket":
        return TokenBucket(
            observation.token_class,
            self.bucket_age(observation.current_age),
            self.bucket_retry_count(observation.retry_count),
            observation.ingress_rank % 4,
        )

    def bucket_replica(self, observation: object) -> "ReplicaBucket":
        required = (
            "replica_id", "domain_id", "available", "queue_depth",
            "running_age", "estimated_work", "estimated_hazard",
            "estimated_failure_risk",
        )
        if any(not hasattr(observation, name) for name in required):
            raise PopulationEstimatorError("invalid Replica observation")
        return ReplicaBucket(
            _strict_int(observation.replica_id, "replica_id"),
            observation.domain_id,
            "UP" if observation.available else "DOWN",
            self.bucket_queue_depth(observation.queue_depth),
            self.bucket_estimated_work(observation.estimated_work),
            self.bucket_running_age(observation.running_age),
            self.bucket_hazard(observation.estimated_hazard),
            self.bucket_hazard(observation.estimated_failure_risk),
        )


BIN_SCHEMA_V1 = BucketSchema(
    "bin_schema_v1",
    token_age_edges=(1.0, 2.0, 4.0),
    retry_edges=(1.0, 2.0),
    queue_depth_edges=(1.0, 2.0, 4.0, 8.0),
    estimated_work_edges=(1.0, 2.0, 4.0),
    running_age_edges=(1.0, 2.0, 4.0),
    hazard_edges=(0.01, 0.05, 0.10),
)


@dataclass(frozen=True)
class TokenBucket:
    token_class: str
    age_bin: str
    retry_bin: str
    ingress_domain: int

    def __post_init__(self) -> None:
        if self.token_class not in {"Regular", "Urgent"}:
            raise PopulationEstimatorError("invalid Token class bucket")
        if type(self.ingress_domain) is not int or self.ingress_domain not in DOMAINS:
            raise PopulationEstimatorError("invalid ingress domain bucket")


@dataclass(frozen=True)
class ReplicaBucket:
    replica_id: int
    domain_id: int
    health: str
    queue_bin: str
    work_bin: str
    running_age_bin: str
    hazard_bin: str
    failure_risk_bin: str

    def __post_init__(self) -> None:
        if type(self.replica_id) is not int or self.replica_id < 0:
            raise PopulationEstimatorError("invalid Replica bucket ID")
        if type(self.domain_id) is not int or self.domain_id not in DOMAINS:
            raise PopulationEstimatorError("invalid domain bucket ID")
        if self.health not in {"UP", "DOWN"}:
            raise PopulationEstimatorError("invalid health bucket")


@dataclass(frozen=True)
class PopulationCellKey:
    """One Token-local cell under one observable Replica-state signature."""

    token_bucket: TokenBucket
    replica_state: tuple[ReplicaBucket, ...]
    locality_signature: tuple[str, ...]
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Topology):
            raise PopulationEstimatorError("cell topology is invalid")
        replica_ids = self.topology.replica_ids
        if len(self.replica_state) != len(replica_ids):
            raise PopulationEstimatorError("cell must cover every Replica")
        if tuple(row.replica_id for row in self.replica_state) != replica_ids:
            raise PopulationEstimatorError("Replica state must be canonical")
        if len(self.locality_signature) != len(replica_ids):
            raise PopulationEstimatorError("locality signature must cover every Replica")
        if any(value not in {"same_domain", "other_domain"} for value in self.locality_signature):
            raise PopulationEstimatorError("invalid locality signature")


@dataclass(frozen=True)
class MuState:
    """The empirical measure of all arrived, nonterminal Tokens."""

    active_tokens: tuple[ActiveTokenState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.active_tokens, tuple):
            raise PopulationEstimatorError("mu active Tokens must be immutable")
        if any(not isinstance(row, ActiveTokenState) for row in self.active_tokens):
            raise PopulationEstimatorError("mu active Token rows are invalid")
        token_ids = tuple(row.token_id for row in self.active_tokens)
        if len(set(token_ids)) != len(token_ids):
            raise PopulationEstimatorError("mu active Token IDs must be unique")
        if token_ids != tuple(
            row.token_id
            for row in sorted(
                self.active_tokens,
                key=lambda row: (row.arrival_time, row.token_id),
            )
        ):
            raise PopulationEstimatorError("mu active Token order must be canonical")

    @property
    def active_token_count(self) -> int:
        return len(self.active_tokens)

    @property
    def class_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                token_class,
                sum(row.token_class == token_class for row in self.active_tokens),
            )
            for token_class in ("Regular", "Urgent")
        )


@dataclass(frozen=True)
class NuState:
    replica_buckets: tuple[ReplicaBucket, ...]
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Topology):
            raise PopulationEstimatorError("nu topology is invalid")
        replica_ids = self.topology.replica_ids
        if len(self.replica_buckets) != len(replica_ids):
            raise PopulationEstimatorError("nu must cover every Replica")
        if tuple(row.replica_id for row in self.replica_buckets) != replica_ids:
            raise PopulationEstimatorError("nu Replica order must be canonical")


@dataclass(frozen=True)
class EtaState:
    token_ids: tuple[int, ...]
    token_buckets: tuple[TokenBucket, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_buckets):
            raise PopulationEstimatorError("eta IDs and buckets must align")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise PopulationEstimatorError("eta Token IDs must be canonical ints")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise PopulationEstimatorError("eta Token IDs must be unique")


@dataclass(frozen=True)
class XState:
    predicted: tuple[tuple[int, float], ...]
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Topology):
            raise PopulationEstimatorError("x topology is invalid")
        if _normalise_share(self.predicted, "predicted share", self.topology) != self.predicted:
            raise PopulationEstimatorError("predicted share must be canonical")


@dataclass(frozen=True)
class PopulationEpisodeIdentity:
    namespace: str
    macro_seed: int
    episode_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise PopulationEstimatorError("namespace must be non-empty")
        _strict_int(self.macro_seed, "macro_seed")
        _strict_int(self.episode_index, "episode_index")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.namespace, self.macro_seed, self.episode_index

    @property
    def fingerprint(self) -> str:
        return _digest(self.key)


@dataclass(frozen=True)
class TokenShareRecord:
    """One Token's predicted distribution and post-commit assignment."""

    token_id: int
    cell_key: PopulationCellKey
    predicted_share: tuple[tuple[int, float], ...]
    realized_replica_id: int | None

    def __post_init__(self) -> None:
        _strict_int(self.token_id, "share Token ID")
        if not isinstance(self.cell_key, PopulationCellKey):
            raise PopulationEstimatorError("share cell key is invalid")
        if _normalise_share(
            self.predicted_share, "Token predicted share", self.cell_key.topology
        ) != self.predicted_share:
            raise PopulationEstimatorError("Token predicted share must be canonical")
        if self.realized_replica_id is not None and (
            type(self.realized_replica_id) is not int
            or self.realized_replica_id not in self.cell_key.topology.replica_ids
        ):
            raise PopulationEstimatorError("Token realized Replica ID is invalid")


@dataclass(frozen=True)
class PopulationBatchRecord:
    episode_key: tuple[str, int, int]
    batch_id: int
    time: float
    public_state_fingerprint: str
    common_noise_prefix_fingerprint: str
    mu: MuState
    nu: NuState
    eta: EtaState
    x: XState
    realized_share: tuple[tuple[int, float], ...]
    cell_keys: tuple[PopulationCellKey, ...]
    token_rows: tuple[TokenShareRecord, ...]
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        _strict_int(self.batch_id, "batch_id")
        _finite(self.time, "batch time")
        if not isinstance(self.topology, Topology):
            raise PopulationEstimatorError("batch topology is invalid")
        if not isinstance(self.mu, MuState):
            raise PopulationEstimatorError("mu is invalid")
        if not isinstance(self.nu, NuState):
            raise PopulationEstimatorError("nu is invalid")
        if not isinstance(self.eta, EtaState):
            raise PopulationEstimatorError("eta is invalid")
        if not isinstance(self.x, XState):
            raise PopulationEstimatorError("x is invalid")
        if self.mu.active_tokens and any(
            row.topology != self.topology for row in self.mu.active_tokens
        ):
            raise PopulationEstimatorError("mu and batch topology differ")
        if self.nu.topology != self.topology or self.x.topology != self.topology:
            raise PopulationEstimatorError("batch state topologies differ")
        if any(cell.topology != self.topology for cell in self.cell_keys):
            raise PopulationEstimatorError("cell and batch topologies differ")
        if len(self.cell_keys) != len(self.eta.token_ids):
            raise PopulationEstimatorError("cell keys must align with eta")
        if len(self.token_rows) != len(self.eta.token_ids):
            raise PopulationEstimatorError("Token share rows must align with eta")
        if tuple(row.token_id for row in self.token_rows) != self.eta.token_ids:
            raise PopulationEstimatorError("Token share rows must follow eta order")
        if tuple(row.cell_key for row in self.token_rows) != self.cell_keys:
            raise PopulationEstimatorError("Token share cells must align with batch cells")
        if self.x.predicted != _aggregate_predicted_shares(
            tuple(row.predicted_share for row in self.token_rows), self.topology
        ):
            raise PopulationEstimatorError("x must aggregate per-Token predictions")
        if self.realized_share != _aggregate_realized_assignments(
            tuple(row.realized_replica_id for row in self.token_rows), self.topology
        ):
            raise PopulationEstimatorError("realized share must aggregate assignments")
        _normalise_share(self.realized_share, "realized share", self.topology)
        if not isinstance(self.public_state_fingerprint, str) or len(self.public_state_fingerprint) != 64:
            raise PopulationEstimatorError("public state fingerprint must be SHA-256")
        if not isinstance(self.common_noise_prefix_fingerprint, str) or len(self.common_noise_prefix_fingerprint) != 64:
            raise PopulationEstimatorError("common-noise fingerprint must be SHA-256")


@dataclass(frozen=True)
class ForwardEpisode:
    identity: PopulationEpisodeIdentity
    protocol_fingerprint: str
    source_fingerprint: str
    trace_fingerprint: str
    token_fingerprint: str
    batches: tuple[PopulationBatchRecord, ...]
    scenario: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.batches, tuple):
            raise PopulationEstimatorError("episode batches must be immutable")
        if not self.batches:
            raise PopulationEstimatorError("forward episode must contain a decision batch")
        if not isinstance(self.scenario, str) or not self.scenario:
            raise PopulationEstimatorError("forward episode scenario must be non-empty")


@dataclass(frozen=True)
class PopulationCellEstimate:
    key: PopulationCellKey
    episode_count: int
    batch_count: int
    occupancy_mass: float
    predicted_share: tuple[tuple[int, float], ...]
    realized_share: tuple[tuple[int, float], ...]
    residual_l1: float
    episode_keys: tuple[tuple[str, int, int], ...]
    common_noise_prefix_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        _strict_int(self.episode_count, "cell episode_count")
        _strict_int(self.batch_count, "cell batch_count")
        occupancy = _finite(self.occupancy_mass, "occupancy_mass")
        if occupancy <= 0.0 or occupancy > 1.0 + 1e-12:
            raise PopulationEstimatorError("cell occupancy must be in (0, 1]")
        residual = _finite(self.residual_l1, "residual_l1")
        if residual < 0.0:
            raise PopulationEstimatorError("residual must be non-negative")


@dataclass(frozen=True)
class PopulationCalibration:
    protocol_fingerprint: str
    source_fingerprint: str
    namespace: str
    macro_seed: int
    episode_count: int
    batch_count: int
    min_episode_samples: int
    cells: tuple[PopulationCellEstimate, ...]
    complete: bool


class PopulationPolicy(Protocol):
    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        ...

    def predict_token_share(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
    ) -> Sequence[tuple[int, float]]:
        """Return one Token's pre-commit conditional action distribution."""


def _source_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _protocol_fingerprint(
    schema: BucketSchema, topology: Topology = DEFAULT_TOPOLOGY,
) -> str:
    if not isinstance(topology, Topology):
        raise PopulationEstimatorError("topology must be Topology")
    payload: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "schema": schema.fingerprint,
        "replicas": topology.replica_ids,
        "domains": topology.domains,
        "history_window": HISTORY_WINDOW,
        "service_mean": SERVICE_MEAN,
        "service_cv": SERVICE_CV,
        "eval_horizon": EVAL_HORIZON,
        "single_copy": True,
        "hedge": False,
        "price": False,
    }
    # Keep the accepted K=8 protocol fingerprint byte-compatible; larger
    # qualification topologies carry their explicit topology identity.
    if topology != DEFAULT_TOPOLOGY:
        payload["topology"] = topology.fingerprint
    return _digest(payload)


def _normalise_share(
    value: object, name: str, topology: Topology = DEFAULT_TOPOLOGY,
) -> tuple[tuple[int, float], ...]:
    if not isinstance(topology, Topology):
        raise PopulationEstimatorError("share topology must be Topology")
    if not isinstance(value, (tuple, list)):
        raise PopulationEstimatorError(f"{name} must be a sequence of (Replica, share)")
    rows: list[tuple[int, float]] = []
    for item in value:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise PopulationEstimatorError(f"{name} has an invalid share row")
        replica_id, share = item
        if type(replica_id) is not int or replica_id not in topology.replica_ids:
            raise PopulationEstimatorError(f"{name} has an invalid Replica ID")
        share_value = _finite(share, f"{name} share")
        if share_value <= 0.0:
            raise PopulationEstimatorError(f"{name} shares must be positive")
        rows.append((replica_id, share_value))
    if len({replica_id for replica_id, _ in rows}) != len(rows):
        raise PopulationEstimatorError(f"{name} has duplicate Replica IDs")
    rows.sort()
    if rows and abs(math.fsum(share for _, share in rows) - 1.0) > 1e-12:
        raise PopulationEstimatorError(f"{name} shares must sum to one")
    return tuple(rows)


def _share_l1(
    left: tuple[tuple[int, float], ...],
    right: tuple[tuple[int, float], ...],
    topology: Topology = DEFAULT_TOPOLOGY,
) -> float:
    left_map = dict(left)
    right_map = dict(right)
    return math.fsum(abs(left_map.get(replica_id, 0.0) - right_map.get(replica_id, 0.0))
                     for replica_id in topology.replica_ids)


def _common_noise_prefix(trace: RoutingTrace, time: float) -> str:
    events = tuple(
        (event.time, event.kind, event.scope, event.component_id)
        for event in trace.health_events
        if event.time <= time + 1e-12
    )
    return _digest({"events_through_time": events})


class _RecordingPolicy:
    def __init__(self, base: PopulationPolicy, topology: Topology):
        self.base = base
        self.topology = topology
        self.observations: dict[str, list[TokenDecisionObservation]] = {}
        self.contexts: dict[str, PopulationDecisionContext] = {}
        self.predicted: dict[tuple[str, int], tuple[tuple[int, float], ...]] = {}

    def choose(self, observation, context, policy_key):
        fingerprint = observation.public_state.fingerprint
        if fingerprint not in self.contexts:
            self.contexts[fingerprint] = context
            self.observations[fingerprint] = []
        predictor = getattr(self.base, "predict_token_share", None)
        if not callable(predictor):
            raise PopulationEstimatorError(
                "population policy must provide predict_token_share"
            )
        key = (fingerprint, observation.token_id)
        if key in self.predicted:
            raise PopulationEstimatorError("Token prediction was recorded twice")
        self.predicted[key] = _normalise_share(
            predictor(observation, context), "Token predicted share", self.topology,
        )
        self.observations[fingerprint].append(observation)
        return self.base.choose(observation, context, policy_key)

    def prediction(
        self, fingerprint: str, token_id: int,
    ) -> tuple[tuple[int, float], ...]:
        try:
            return self.predicted[(fingerprint, token_id)]
        except KeyError as error:
            raise PopulationEstimatorError(
                "missing per-Token prediction for decision cohort"
            ) from error


def _aggregate_predicted_shares(
    shares: Sequence[tuple[tuple[int, float], ...]],
    topology: Topology = DEFAULT_TOPOLOGY,
) -> tuple[tuple[int, float], ...]:
    if not shares:
        raise PopulationEstimatorError("cannot aggregate an empty prediction cohort")
    return tuple(
        (replica_id, value)
        for replica_id, value in (
            (
                replica_id,
                math.fsum(
                    dict(share).get(replica_id, 0.0) for share in shares
                ) / len(shares),
            )
            for replica_id in topology.replica_ids
        )
        if value > 0.0
    )


def _aggregate_realized_assignments(
    assignments: Sequence[int | None],
    topology: Topology = DEFAULT_TOPOLOGY,
) -> tuple[tuple[int, float], ...]:
    assigned = tuple(value for value in assignments if value is not None)
    if not assigned:
        return ()
    return tuple(
        (replica_id, assigned.count(replica_id) / len(assigned))
        for replica_id in topology.replica_ids
        if replica_id in assigned
    )


def _build_batch_record(
    trace: RoutingTrace,
    identity: PopulationEpisodeIdentity,
    schema: BucketSchema,
    audit: object,
    recorder: _RecordingPolicy,
) -> PopulationBatchRecord:
    topology = trace.topology
    fingerprint = audit.public_state_fingerprint
    observations = tuple(recorder.observations.get(fingerprint, ()))
    context = recorder.contexts.get(fingerprint)
    if context is None or tuple(row.token_id for row in observations) != audit.token_ids:
        raise PopulationEstimatorError("policy observations do not match decision cohort")
    public_state = observations[0].public_state
    if any(row.public_state is not public_state for row in observations):
        raise PopulationEstimatorError("same-batch observations did not share one state")
    replica_buckets = tuple(
        schema.bucket_replica(row) for row in public_state.replica_observations
    )
    nu = NuState(replica_buckets, topology)
    token_buckets = tuple(schema.bucket_token(row) for row in observations)
    eta = EtaState(audit.token_ids, token_buckets)
    mu = MuState(tuple(audit.active_token_states))
    predicted_rows = tuple(
        recorder.prediction(fingerprint, token_id)
        for token_id in audit.token_ids
    )
    predicted = _aggregate_predicted_shares(predicted_rows, topology)
    x = XState(predicted, topology)
    realized = tuple(audit.realized_share)
    locality_by_observation = tuple(
        tuple(
            "same_domain"
            if row.domain_id == trace.topology.domain_of(observation.ingress_rank)
            else "other_domain"
            for row in public_state.replica_observations
        )
        for observation in observations
    )
    cell_keys = tuple(
        PopulationCellKey(token_bucket, replica_buckets, locality, topology)
        for token_bucket, locality in zip(token_buckets, locality_by_observation)
    )
    token_rows = tuple(
        TokenShareRecord(
            token_id,
            cell_key,
            token_prediction,
            assignment,
        )
        for token_id, cell_key, token_prediction, assignment in zip(
            audit.token_ids,
            cell_keys,
            predicted_rows,
            audit.assigned_replica_ids,
        )
    )
    return PopulationBatchRecord(
        identity.key,
        audit.batch_id,
        audit.time,
        fingerprint,
        _common_noise_prefix(trace, audit.time),
        mu,
        nu,
        eta,
        x,
        realized,
        cell_keys,
        token_rows,
        topology,
    )


def run_population_forward(
    trace: RoutingTrace,
    policy_factory: Callable[[], PopulationPolicy],
    *,
    schema: BucketSchema = BIN_SCHEMA_V1,
) -> ForwardEpisode:
    """Run one fresh finite episode and materialize its causal population state."""

    if not isinstance(trace, RoutingTrace):
        raise PopulationEstimatorError("trace must be RoutingTrace")
    if not callable(policy_factory):
        raise PopulationEstimatorError("policy_factory must be callable")
    if not isinstance(schema, BucketSchema):
        raise PopulationEstimatorError("schema must be BucketSchema")
    identity = PopulationEpisodeIdentity(trace.namespace, trace.macro_seed, trace.episode_index)
    try:
        policy = policy_factory()
    except Exception as error:
        raise PopulationEstimatorError("policy factory failed before physics") from error
    if not callable(getattr(policy, "choose", None)) or not callable(
        getattr(policy, "predict_token_share", None)
    ):
        raise PopulationEstimatorError(
            "policy must provide choose and predict_token_share"
        )
    recorder = _RecordingPolicy(policy, trace.topology)
    try:
        result: SimultaneousRoutingResult = simulate_simultaneous_routing(
            trace, recorder, policy_name="population_forward",
        )
        batches = tuple(
            _build_batch_record(trace, identity, schema, audit, recorder)
            for audit in result.batch_audits
        )
    except PopulationEstimatorError:
        raise
    except Exception as error:
        raise PopulationEstimatorError(
            f"population forward physical run failed: {type(error).__name__}: {error}"
        ) from error
    return ForwardEpisode(
        identity,
        _protocol_fingerprint(schema, trace.topology),
        _source_fingerprint(),
        trace.fingerprint,
        trace.token_fingerprint,
        batches,
        trace.scenario,
    )


def run_population_forward_library(
    traces: Iterable[RoutingTrace],
    policy_factory: Callable[[], PopulationPolicy],
    *,
    schema: BucketSchema = BIN_SCHEMA_V1,
) -> tuple[ForwardEpisode, ...]:
    """Run a canonical trace library, creating a new policy per episode."""

    rows = tuple(
        run_population_forward(trace, policy_factory, schema=schema)
        for trace in traces
    )
    keys = tuple(row.identity.key for row in rows)
    if len(set(keys)) != len(keys):
        raise PopulationEstimatorError("forward library has duplicate episode identities")
    return rows


def _validate_episode_library(
    episodes: Sequence[ForwardEpisode],
    min_episode_samples: int,
) -> tuple[str, int, str, str]:
    if not episodes:
        raise PopulationEstimatorError("calibration requires at least one episode")
    if type(min_episode_samples) is not int or min_episode_samples < 1:
        raise PopulationEstimatorError("min_episode_samples must be a positive true int")
    keys = tuple(row.identity.key for row in episodes)
    if len(set(keys)) != len(keys):
        raise PopulationEstimatorError("calibration episode identities must be unique")
    trace_fingerprints = tuple(row.trace_fingerprint for row in episodes)
    if len(set(trace_fingerprints)) != len(trace_fingerprints):
        raise PopulationEstimatorError("calibration episodes reuse one trace")
    namespaces = {row.identity.namespace for row in episodes}
    seeds = {row.identity.macro_seed for row in episodes}
    protocols = {row.protocol_fingerprint for row in episodes}
    sources = {row.source_fingerprint for row in episodes}
    if len(namespaces) != 1 or len(seeds) != 1 or len(protocols) != 1 or len(sources) != 1:
        raise PopulationEstimatorError("calibration episodes do not share one library contract")
    if any(not row.batches for row in episodes):
        raise PopulationEstimatorError("incomplete episode cannot enter calibration")
    return next(iter(namespaces)), next(iter(seeds)), next(iter(protocols)), next(iter(sources))


def calibrate_population_shares(
    episodes: Sequence[ForwardEpisode],
    *,
    required_cells: Sequence[PopulationCellKey] = (),
    min_episode_samples: int = MIN_EPISODE_SAMPLES,
) -> PopulationCalibration:
    """Aggregate predicted/realized shares with episode-level independence."""

    rows = tuple(episodes)
    namespace, macro_seed, protocol_fingerprint, source_fingerprint = _validate_episode_library(
        rows, min_episode_samples,
    )
    required = tuple(required_cells)
    if len(set(required)) != len(required):
        raise PopulationEstimatorError("required cells must be unique")
    cell_episode_rows: dict[
        PopulationCellKey,
        dict[tuple[str, int, int], list[tuple[TokenShareRecord, PopulationBatchRecord]]],
    ] = {}
    for episode in rows:
        for batch in episode.batches:
            if len(batch.token_rows) != len(batch.eta.token_ids):
                raise PopulationEstimatorError("corrupt batch cell alignment")
            for token_row in batch.token_rows:
                cell_episode_rows.setdefault(token_row.cell_key, {}).setdefault(
                    episode.identity.key, []
                ).append((token_row, batch))
    if required and any(cell not in cell_episode_rows for cell in required):
        raise PopulationEstimatorError("required population cell is unsupported")
    selected_cells = tuple(sorted(required or cell_episode_rows, key=repr))
    total_occupancy = sum(
        len(batch.eta.token_ids)
        for episode in rows
        for batch in episode.batches
    )
    if total_occupancy <= 0:
        raise PopulationEstimatorError("calibration has no decision-cohort Tokens")
    estimates: list[PopulationCellEstimate] = []
    for cell in selected_cells:
        topology = cell.topology
        per_episode = cell_episode_rows.get(cell, {})
        if len(per_episode) < min_episode_samples:
            raise PopulationEstimatorError("population cell has insufficient episode samples")
        episode_values: list[tuple[float, tuple[tuple[int, float], ...], tuple[tuple[int, float], ...]]] = []
        batch_count = 0
        occupancy_count = 0
        prefixes: set[str] = set()
        for key, token_batch_rows in sorted(per_episode.items()):
            predicted_values = [row.predicted_share for row, _ in token_batch_rows]
            predicted_map = {
                replica_id: math.fsum(
                    dict(value).get(replica_id, 0.0) for value in predicted_values
                ) / len(predicted_values)
                for replica_id in topology.replica_ids
            }
            assignment_values = [
                row.realized_replica_id for row, _ in token_batch_rows
            ]
            realized_map = {
                replica_id: assignment_values.count(replica_id) / len(assignment_values)
                for replica_id in topology.replica_ids
            }
            predicted = tuple((rid, share) for rid, share in predicted_map.items() if share > 0.0)
            realized = tuple((rid, share) for rid, share in realized_map.items() if share > 0.0)
            episode_values.append((_share_l1(predicted, realized, topology), predicted, realized))
            batch_keys = {
                (batch.episode_key, batch.batch_id)
                for _, batch in token_batch_rows
            }
            batch_count += len(batch_keys)
            occupancy_count += len(token_batch_rows)
            prefixes.update(
                batch.common_noise_prefix_fingerprint
                for _, batch in token_batch_rows
            )
        residual = math.fsum(value[0] for value in episode_values) / len(episode_values)
        predicted = tuple(
            (
                rid,
                math.fsum(dict(value[1]).get(rid, 0.0) for value in episode_values)
                / len(episode_values),
            )
            for rid in topology.replica_ids
        )
        realized = tuple(
            (
                rid,
                math.fsum(dict(value[2]).get(rid, 0.0) for value in episode_values)
                / len(episode_values),
            )
            for rid in topology.replica_ids
        )
        estimates.append(PopulationCellEstimate(
            cell,
            len(per_episode),
            batch_count,
            occupancy_count / total_occupancy,
            tuple((rid, share) for rid, share in predicted if share > 0.0),
            tuple((rid, share) for rid, share in realized if share > 0.0),
            residual,
            tuple(sorted(per_episode)),
            tuple(sorted(prefixes)),
        ))
    return PopulationCalibration(
        protocol_fingerprint,
        source_fingerprint,
        namespace,
        macro_seed,
        len(rows),
        sum(len(row.batches) for row in rows),
        min_episode_samples,
        tuple(estimates),
        True,
    )


__all__ = [
    "ActiveTokenState",
    "BIN_SCHEMA_V1",
    "BucketSchema",
    "CALIBRATION_EPISODE_COUNT",
    "CALIBRATION_MACRO_SEED",
    "CALIBRATION_NAMESPACE",
    "EtaState",
    "FORWARD_EPISODE_COUNT",
    "FORWARD_MACRO_SEED",
    "FORWARD_NAMESPACE",
    "ForwardEpisode",
    "IMPLEMENTATION_CALL_BUDGET",
    "MIN_EPISODE_SAMPLES",
    "MuState",
    "NuState",
    "PopulationBatchRecord",
    "PopulationCalibration",
    "PopulationCellEstimate",
    "PopulationCellKey",
    "PopulationEpisodeIdentity",
    "PopulationEstimatorError",
    "PopulationPolicy",
    "ReplicaBucket",
    "TokenBucket",
    "TokenShareRecord",
    "XState",
    "calibrate_population_shares",
    "run_population_forward",
    "run_population_forward_library",
]
