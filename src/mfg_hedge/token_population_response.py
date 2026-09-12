"""Tagged continuation and finite-K response diagnostics.

This isolated layer keeps two estimands separate:

mean_field_continuation
    Scores a tagged action against an immutable, exogenous population path.

finite_k_deviation
    Re-runs the complete finite routing engine while overriding one tagged
    requested action. Every other Token continues to call its policy online.

Neither output selects an action or makes a BR, regret, Nash, or MFG claim.
The module is intentionally not re-exported from mfg_hedge, so it does not
change the package version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Sequence

from .population_estimator import (
    BIN_SCHEMA_V1,
    BucketSchema,
    ForwardEpisode,
    PopulationBatchRecord,
    ReplicaBucket,
)
from .reliability_aware_routing import DOMAINS, REPLICA_IDS, RoutingTrace
from .simultaneous_token_routing import (
    PopulationDecisionContext,
    SimultaneousPolicy,
    SimultaneousRoutingResult,
    TokenDecisionObservation,
    simulate_simultaneous_routing,
)


MEAN_FIELD_LABEL = "mean_field_continuation"
FINITE_K_LABEL = "finite_k_deviation"
MEAN_FIELD_NAMESPACE = "reliability-aware-token-mfg:v1:mean-field-continuation"
FINITE_K_NAMESPACE = "reliability-aware-token-mfg:v1:finite-k-deviation"
MEAN_FIELD_MACRO_SEED = 20260912
FINITE_K_MACRO_SEED = 20260913
IMPLEMENTATION_EPISODES = 16
MIN_COMPLETE_PANELS = 2
MAX_ACTION_BUCKETS = 8
MAX_CALLS_PER_OUTPUT = IMPLEMENTATION_EPISODES * (1 + MAX_ACTION_BUCKETS)
MAX_TOTAL_CALLS = 2 * MAX_CALLS_PER_OUTPUT


class PopulationResponseError(ValueError):
    """Raised when a tagged-response contract cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        attempted_calls: int = 0,
        original_exception_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempted_calls = attempted_calls
        self.original_exception_type = original_exception_type


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PopulationResponseError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PopulationResponseError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PopulationResponseError(f"{name} must be finite")
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
        raise PopulationResponseError("value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _canonical_dataclass(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


@dataclass(frozen=True)
class ReplicaStateActionBucket:
    """An observable Replica-state action, without a concrete Replica ID."""

    domain_id: int
    health: str
    queue_bin: str
    work_bin: str
    running_age_bin: str
    hazard_bin: str
    failure_risk_bin: str
    locality: str

    def __post_init__(self) -> None:
        _strict_int(self.domain_id, "action bucket domain_id")
        if self.domain_id not in DOMAINS:
            raise PopulationResponseError("action bucket domain_id is invalid")
        if self.health not in {"UP", "DOWN"}:
            raise PopulationResponseError("action bucket health is invalid")
        for name in (
            "queue_bin", "work_bin", "running_age_bin", "hazard_bin",
            "failure_risk_bin",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PopulationResponseError(f"action bucket {name} is invalid")
        if self.locality not in {"same_domain", "other_domain"}:
            raise PopulationResponseError("action bucket locality is invalid")

    @classmethod
    def from_replica(
        cls, replica: ReplicaBucket, locality: str,
    ) -> "ReplicaStateActionBucket":
        if not isinstance(replica, ReplicaBucket):
            raise PopulationResponseError("Replica bucket is invalid")
        return cls(
            replica.domain_id,
            replica.health,
            replica.queue_bin,
            replica.work_bin,
            replica.running_age_bin,
            replica.hazard_bin,
            replica.failure_risk_bin,
            locality,
        )

    @property
    def fingerprint(self) -> str:
        return _digest(asdict(self))

    def matches(self, replica: ReplicaBucket, locality: str) -> bool:
        return self == self.from_replica(replica, locality)


@dataclass(frozen=True)
class MeanFieldEnvironment:
    """One full exogenous population path and one preselected target."""

    episode: ForwardEpisode
    target_batch_id: int
    target_token_id: int
    selection_rule: str = "future_blind_pre_registered"

    def __post_init__(self) -> None:
        if not isinstance(self.episode, ForwardEpisode):
            raise PopulationResponseError("mean-field episode is invalid")
        _strict_int(self.target_batch_id, "target batch ID")
        _strict_int(self.target_token_id, "target Token ID")
        if not isinstance(self.selection_rule, str) or not self.selection_rule:
            raise PopulationResponseError("selection_rule must be non-empty")
        matches = tuple(
            batch for batch in self.episode.batches
            if batch.batch_id == self.target_batch_id
        )
        if len(matches) != 1:
            raise PopulationResponseError("target batch is missing or duplicated")
        if self.target_token_id not in matches[0].eta.token_ids:
            raise PopulationResponseError("target Token is not in the selected cohort")

    @property
    def target_batch(self) -> PopulationBatchRecord:
        return next(
            batch for batch in self.episode.batches
            if batch.batch_id == self.target_batch_id
        )

    @property
    def path_fingerprint(self) -> str:
        return _digest({
            "identity": self.episode.identity.key,
            "protocol": self.episode.protocol_fingerprint,
            "source": self.episode.source_fingerprint,
            "trace": self.episode.trace_fingerprint,
            "token": self.episode.token_fingerprint,
            "batches": tuple(
                _canonical_dataclass(batch) for batch in self.episode.batches
            ),
        })


@dataclass(frozen=True)
class MeanFieldBranchContext:
    environment: MeanFieldEnvironment
    target_batch_id: int
    target_token_id: int
    action_bucket: ReplicaStateActionBucket | None
    selected_replica_id: int | None
    crn_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.environment, MeanFieldEnvironment):
            raise PopulationResponseError("mean-field branch environment is invalid")
        if self.action_bucket is not None and not isinstance(
            self.action_bucket, ReplicaStateActionBucket
        ):
            raise PopulationResponseError("mean-field action bucket is invalid")
        if self.selected_replica_id is not None and (
            type(self.selected_replica_id) is not int
            or self.selected_replica_id < 0
        ):
            raise PopulationResponseError("selected Replica ID is invalid")
        if not isinstance(self.crn_key, str) or not self.crn_key:
            raise PopulationResponseError("CRN key must be non-empty")


@dataclass(frozen=True)
class MeanFieldContinuationRow:
    episode_key: tuple[str, int, int]
    target_batch_id: int
    target_token_id: int
    action_bucket: ReplicaStateActionBucket
    selected_replica_id: int
    crn_key: str
    cost: float
    environment_fingerprint: str

    def __post_init__(self) -> None:
        _strict_int(self.target_batch_id, "mean-field row batch ID")
        _strict_int(self.target_token_id, "mean-field row Token ID")
        _finite(self.cost, "mean-field row cost")
        if type(self.selected_replica_id) is not int or self.selected_replica_id < 0:
            raise PopulationResponseError("mean-field row Replica ID is invalid")
        if len(self.environment_fingerprint) != 64:
            raise PopulationResponseError("mean-field environment fingerprint is invalid")


@dataclass(frozen=True)
class ActionCostSummary:
    action_bucket: ReplicaStateActionBucket
    mean: float
    standard_error: float
    effective_n: int

    def __post_init__(self) -> None:
        _finite(self.mean, "action mean")
        _finite(self.standard_error, "action standard error")
        _strict_int(self.effective_n, "action effective sample size", 1)


@dataclass(frozen=True)
class MeanFieldContinuationResult:
    output_label: str
    namespace: str
    macro_seed: int
    cost_model_id: str
    rows: tuple[MeanFieldContinuationRow, ...]
    action_summaries: tuple[ActionCostSummary, ...]
    baseline_costs: tuple[float, ...]
    attempted_calls: int
    complete: bool
    claim_boundary: str = "conditional_cost_only"

    @property
    def claims_best_action(self) -> bool:
        return False

    @property
    def claims_mfg(self) -> bool:
        return False


@dataclass(frozen=True)
class FiniteKDeviationRow:
    episode_key: tuple[str, int, int]
    target_batch_id: int
    target_token_id: int
    candidate_bucket: ReplicaStateActionBucket
    selected_replica_id: int
    baseline_selected_replica_id: int | None
    baseline_cost: float
    candidate_cost: float
    pathwise_difference: float
    crn_key: str
    trace_fingerprint: str
    baseline_result_fingerprint: str
    candidate_result_fingerprint: str
    baseline_policy_fingerprint: str
    candidate_policy_fingerprint: str

    def __post_init__(self) -> None:
        _strict_int(self.target_batch_id, "finite-K row batch ID")
        _strict_int(self.target_token_id, "finite-K row Token ID")
        if type(self.selected_replica_id) is not int or self.selected_replica_id < 0:
            raise PopulationResponseError("finite-K selected Replica ID is invalid")
        if self.baseline_selected_replica_id is not None and (
            type(self.baseline_selected_replica_id) is not int
            or self.baseline_selected_replica_id < 0
        ):
            raise PopulationResponseError("finite-K baseline Replica ID is invalid")
        _finite(self.baseline_cost, "finite-K baseline cost")
        _finite(self.candidate_cost, "finite-K candidate cost")
        difference = _finite(self.pathwise_difference, "finite-K pathwise difference")
        if abs(difference - (self.candidate_cost - self.baseline_cost)) > 1e-12:
            raise PopulationResponseError("finite-K pathwise difference is inconsistent")
        for name in (
            "trace_fingerprint", "baseline_result_fingerprint",
            "candidate_result_fingerprint", "baseline_policy_fingerprint",
            "candidate_policy_fingerprint",
        ):
            if len(getattr(self, name)) != 64:
                raise PopulationResponseError(f"finite-K {name} is invalid")


@dataclass(frozen=True)
class FiniteKDeviationResult:
    output_label: str
    namespace: str
    macro_seed: int
    cost_model_id: str
    episode_key: tuple[str, int, int]
    trace_fingerprint: str
    target_batch_id: int
    target_token_id: int
    rows: tuple[FiniteKDeviationRow, ...]
    attempted_calls: int
    complete: bool
    claim_boundary: str = "pathwise_deviation_only"

    @property
    def claims_regret(self) -> bool:
        return False

    @property
    def claims_nash(self) -> bool:
        return False


@dataclass(frozen=True)
class FiniteKDeviationCase:
    """One immutable finite-K target specification in a validation panel."""

    trace: RoutingTrace
    episode: ForwardEpisode
    target_batch_id: int
    target_token_id: int
    action_buckets: tuple[ReplicaStateActionBucket, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trace, RoutingTrace):
            raise PopulationResponseError("finite-K case trace is invalid")
        if not isinstance(self.episode, ForwardEpisode):
            raise PopulationResponseError("finite-K case episode is invalid")
        _strict_int(self.target_batch_id, "finite-K case target batch ID")
        _strict_int(self.target_token_id, "finite-K case target Token ID")
        if not isinstance(self.action_buckets, tuple):
            raise PopulationResponseError("finite-K case buckets must be immutable")


@dataclass(frozen=True)
class FiniteKDeviationPanelResult:
    """Compact panel rows for the frozen 16-episode finite-K output."""

    output_label: str
    namespace: str
    macro_seed: int
    cost_model_id: str
    rows: tuple[FiniteKDeviationRow, ...]
    episode_keys: tuple[tuple[str, int, int], ...]
    per_episode_calls: tuple[int, ...]
    attempted_calls: int
    complete: bool
    claim_boundary: str = "pathwise_deviation_only"

    @property
    def claims_regret(self) -> bool:
        return False

    @property
    def claims_nash(self) -> bool:
        return False


def _validate_cost_model(cost_model_id: object, evaluator: object) -> str:
    if not isinstance(cost_model_id, str) or not cost_model_id:
        raise PopulationResponseError("cost_model_id must be non-empty")
    if not callable(evaluator):
        raise PopulationResponseError("cost evaluator must be callable")
    return cost_model_id


def _target_row(
    batch: PopulationBatchRecord, token_id: int,
) -> object:
    rows = tuple(row for row in batch.token_rows if row.token_id == token_id)
    if len(rows) != 1:
        raise PopulationResponseError("target Token row is missing or duplicated")
    return rows[0]


def action_buckets_for_token(
    batch: PopulationBatchRecord,
    token_id: int,
    *,
    schema: BucketSchema = BIN_SCHEMA_V1,
) -> tuple[ReplicaStateActionBucket, ...]:
    """Return supported UP action buckets in canonical fingerprint order."""

    if not isinstance(batch, PopulationBatchRecord):
        raise PopulationResponseError("population batch is invalid")
    if not isinstance(schema, BucketSchema):
        raise PopulationResponseError("bucket schema is invalid")
    row = _target_row(batch, token_id)
    buckets: dict[str, ReplicaStateActionBucket] = {}
    for index, replica in enumerate(batch.nu.replica_buckets):
        if replica.health != "UP":
            continue
        bucket = ReplicaStateActionBucket.from_replica(
            replica, row.cell_key.locality_signature[index],
        )
        buckets[bucket.fingerprint] = bucket
    result = tuple(sorted(buckets.values(), key=lambda bucket: bucket.fingerprint))
    if not result:
        raise PopulationResponseError("target has no supported UP action bucket")
    if len(result) > MAX_ACTION_BUCKETS:
        raise PopulationResponseError("target has more than eight action buckets")
    return result


def select_replica_for_action_bucket(
    batch: PopulationBatchRecord,
    token_id: int,
    bucket: ReplicaStateActionBucket,
    *,
    action_key: str,
) -> int:
    """Select a concrete Replica stably and uniformly inside a bucket."""

    if not isinstance(bucket, ReplicaStateActionBucket) or bucket.health != "UP":
        raise PopulationResponseError("only UP action buckets are supported")
    if not isinstance(action_key, str) or not action_key:
        raise PopulationResponseError("action_key must be non-empty")
    row = _target_row(batch, token_id)
    candidates = tuple(
        replica.replica_id
        for index, replica in enumerate(batch.nu.replica_buckets)
        if replica.health == "UP"
        and bucket.matches(replica, row.cell_key.locality_signature[index])
    )
    if not candidates:
        raise PopulationResponseError("action bucket is unsupported in target state")
    digest = hashlib.sha256(action_key.encode("utf-8")).digest()
    return tuple(sorted(candidates))[int.from_bytes(digest[:8], "big") % len(candidates)]


def _buckets_from_public_state(
    observation: TokenDecisionObservation,
    schema: BucketSchema,
) -> tuple[tuple[int, ReplicaStateActionBucket], ...]:
    rows: list[tuple[int, ReplicaStateActionBucket]] = []
    for replica in observation.public_state.replica_observations:
        bucket = ReplicaStateActionBucket.from_replica(
            schema.bucket_replica(replica),
            "same_domain" if replica.domain_id == observation.ingress_rank % 4
            else "other_domain",
        )
        if bucket.health == "UP":
            rows.append((replica.replica_id, bucket))
    return tuple(rows)


def _select_public_bucket(
    observation: TokenDecisionObservation,
    bucket: ReplicaStateActionBucket,
    action_key: str,
    schema: BucketSchema,
) -> int:
    candidates = tuple(
        replica_id
        for replica_id, candidate in _buckets_from_public_state(observation, schema)
        if candidate == bucket
    )
    if not candidates:
        raise PopulationResponseError("action bucket is not supported at target decision")
    digest = hashlib.sha256(action_key.encode("utf-8")).digest()
    return tuple(sorted(candidates))[int.from_bytes(digest[:8], "big") % len(candidates)]


def _branch_key(
    namespace: str,
    macro_seed: int,
    episode_key: tuple[str, int, int],
    batch_id: int,
    token_id: int,
    bucket: ReplicaStateActionBucket | None,
    policy_iteration: int,
    *,
    scenario: str,
) -> str:
    """Return the stable action CRN identity for one episode branch.

    ``policy_iteration`` remains an input for call-site compatibility and
    validation, but is deliberately absent from this key so a fixed episode
    is paired across solver iterations.
    """
    if not isinstance(scenario, str) or not scenario:
        raise PopulationResponseError("branch scenario must be non-empty")
    bucket_key = "baseline" if bucket is None else bucket.fingerprint
    return (
        f"{namespace}|seed={macro_seed}|scenario={scenario}"
        f"|episode={episode_key[2]}|batch={batch_id}|token={token_id}"
        f"|action={'baseline' if bucket is None else 'bucket'}|bucket={bucket_key}"
    )


# F1 (ticket 15): the deviation-panel baseline branch is bit-identical for a
# fixed (trace, policy, iteration) pair — the intervention-free wrapper never
# changes any action — so it is cached per process. Keys with an unknown
# policy fingerprint simply bypass the cache.
_BASELINE_CACHE: dict = {}


def _baseline_cache_key(
    trace: RoutingTrace,
    policy_fingerprint: str | None,
    policy_iteration: int,
    namespace: str,
    macro_seed: int,
):
    if not isinstance(policy_fingerprint, str) or len(policy_fingerprint) != 64:
        return None
    return (
        trace.fingerprint, policy_fingerprint, policy_iteration, namespace,
        macro_seed,
    )


def _physical_fingerprint(result: SimultaneousRoutingResult) -> str:
    return _digest({
        "trace": result.physical.trace_fingerprint,
        "token": result.physical.token_fingerprint,
        "token_results": tuple(
            _canonical_dataclass(row) for row in result.physical.token_results
        ),
        "starts": tuple(_canonical_dataclass(row) for row in result.physical.starts),
        "metrics": dict(result.physical.metrics),
        "invariants": dict(result.physical.invariants),
        "replica_audit": result.physical.replica_audit,
        "complete_drain": result.physical.complete_drain,
        "batches": tuple(
            _canonical_dataclass(row) for row in result.batch_audits
        ),
    })


@dataclass(frozen=True)
class _PolicyCall:
    policy_key: str
    observation_fingerprint: str
    recommended: int | None
    applied: int | None


class _BucketInterventionPolicy:
    def __init__(
        self,
        base: SimultaneousPolicy,
        *,
        target_token_id: int,
        target_batch_id: int,
        action_bucket: ReplicaStateActionBucket | None,
        branch_key: str,
        schema: BucketSchema,
        seed_calls: Sequence[_PolicyCall] = (),
    ) -> None:
        self.base = base
        self.target_token_id = target_token_id
        self.target_batch_id = target_batch_id
        self.action_bucket = action_bucket
        self.branch_key = branch_key
        self.schema = schema
        self.calls: list[_PolicyCall] = list(seed_calls)
        self.target_calls: list[_PolicyCall] = []
        self.target_observations: list[TokenDecisionObservation] = []

    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        recommended = self.base.choose(observation, context, policy_key)
        target_key = (
            observation.token_id == self.target_token_id
            and f"|batch={self.target_batch_id}|token={self.target_token_id}|" in policy_key
        )
        applied = recommended
        if target_key and self.action_bucket is not None:
            applied = _select_public_bucket(
                observation, self.action_bucket,
                self.branch_key, self.schema,
            )
        record = _PolicyCall(
            policy_key, observation.public_state.fingerprint, recommended, applied,
        )
        self.calls.append(record)
        if target_key:
            self.target_calls.append(record)
            self.target_observations.append(observation)
        return applied

    def call_fingerprint(self) -> str:
        return _digest(tuple(_canonical_dataclass(row) for row in self.calls))

    def target_observation(self) -> TokenDecisionObservation:
        if len(self.target_observations) != 1:
            raise PopulationResponseError("target action was not observed exactly once")
        return self.target_observations[0]


def _validate_episode_for_finite(
    trace: RoutingTrace,
    episode: ForwardEpisode,
    target_batch_id: int,
    target_token_id: int,
) -> PopulationBatchRecord:
    if not isinstance(trace, RoutingTrace):
        raise PopulationResponseError("finite-K trace is invalid")
    if not isinstance(episode, ForwardEpisode):
        raise PopulationResponseError("finite-K forward episode is invalid")
    if trace.fingerprint != episode.trace_fingerprint:
        raise PopulationResponseError("trace and forward episode fingerprints differ")
    if trace.scenario != episode.scenario:
        raise PopulationResponseError("trace and forward episode scenarios differ")
    batches = tuple(
        batch for batch in episode.batches if batch.batch_id == target_batch_id
    )
    if len(batches) != 1:
        raise PopulationResponseError("finite-K target batch is missing or duplicated")
    batch = batches[0]
    if target_token_id not in batch.eta.token_ids:
        raise PopulationResponseError("finite-K target Token is not in target cohort")
    return batch


def _validate_candidate_buckets(
    batch: PopulationBatchRecord,
    token_id: int,
    buckets: Sequence[ReplicaStateActionBucket],
    schema: BucketSchema,
) -> tuple[ReplicaStateActionBucket, ...]:
    rows = tuple(buckets)
    if not rows:
        raise PopulationResponseError("at least one candidate bucket is required")
    if len(rows) > MAX_ACTION_BUCKETS:
        raise PopulationResponseError("more than eight candidate buckets")
    if any(not isinstance(row, ReplicaStateActionBucket) for row in rows):
        raise PopulationResponseError("candidate action bucket is invalid")
    if len({row.fingerprint for row in rows}) != len(rows):
        raise PopulationResponseError("candidate action buckets are duplicated")
    supported = {
        row.fingerprint
        for row in action_buckets_for_token(batch, token_id, schema=schema)
    }
    if any(
        row.health != "UP" or row.fingerprint not in supported
        for row in rows
    ):
        raise PopulationResponseError("candidate action bucket is unsupported")
    return tuple(sorted(rows, key=lambda row: row.fingerprint))


def _validate_namespace(
    namespace: str, expected: str, macro_seed: int, expected_seed: int,
) -> None:
    if namespace != expected or macro_seed != expected_seed:
        raise PopulationResponseError("frozen response namespace or macro seed mismatch")


def _mean_summary(
    bucket: ReplicaStateActionBucket,
    values: Sequence[float],
    *,
    minimum_panels: int = MIN_COMPLETE_PANELS,
) -> ActionCostSummary:
    _strict_int(minimum_panels, "minimum complete panels", 1)
    if len(values) < minimum_panels:
        raise PopulationResponseError("action bucket has insufficient complete panels")
    mean = math.fsum(values) / len(values)
    variance = (
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return ActionCostSummary(
        bucket, mean, math.sqrt(variance / len(values)), len(values),
    )


def evaluate_mean_field_continuation(
    environments: Sequence[MeanFieldEnvironment],
    action_buckets: Sequence[ReplicaStateActionBucket],
    cost_evaluator: Callable[[MeanFieldBranchContext], float],
    *,
    cost_model_id: str,
    namespace: str = MEAN_FIELD_NAMESPACE,
    macro_seed: int = MEAN_FIELD_MACRO_SEED,
    policy_iteration: int = 0,
    schema: BucketSchema = BIN_SCHEMA_V1,
    expected_namespace: str = MEAN_FIELD_NAMESPACE,
    expected_macro_seed: int = MEAN_FIELD_MACRO_SEED,
    expected_episodes: int = IMPLEMENTATION_EPISODES,
    min_complete_panels: int = MIN_COMPLETE_PANELS,
) -> MeanFieldContinuationResult:
    """Score action buckets against complete exogenous population paths."""

    _validate_namespace(
        namespace, expected_namespace, macro_seed, expected_macro_seed,
    )
    _strict_int(expected_episodes, "expected mean-field episodes", 1)
    _strict_int(min_complete_panels, "minimum complete panels", 1)
    _strict_int(policy_iteration, "policy iteration")
    _validate_cost_model(cost_model_id, cost_evaluator)
    if not isinstance(schema, BucketSchema):
        raise PopulationResponseError("bucket schema is invalid")
    envs = tuple(environments)
    if len(envs) != expected_episodes:
        raise PopulationResponseError(
            "mean-field panel has the wrong episode count"
        )
    if any(not isinstance(env, MeanFieldEnvironment) for env in envs):
        raise PopulationResponseError("mean-field environment is invalid")
    if len({env.episode.identity.key for env in envs}) != len(envs):
        raise PopulationResponseError("mean-field episode identity is duplicated")
    if len({env.episode.trace_fingerprint for env in envs}) != len(envs):
        raise PopulationResponseError("mean-field trace identity is duplicated")
    buckets = tuple(action_buckets)
    if not buckets or len(buckets) > MAX_ACTION_BUCKETS:
        raise PopulationResponseError("mean-field action bucket count is invalid")
    if len({bucket.fingerprint for bucket in buckets}) != len(buckets):
        raise PopulationResponseError("mean-field action buckets are duplicated")
    for env in envs:
        _validate_candidate_buckets(
            env.target_batch, env.target_token_id, buckets, schema,
        )

    attempted = 0
    baseline_costs: list[float] = []
    rows: list[MeanFieldContinuationRow] = []
    values: dict[str, list[float]] = {
        bucket.fingerprint: [] for bucket in buckets
    }
    for env in envs:
        base_key = _branch_key(
            namespace, macro_seed, env.episode.identity.key,
            env.target_batch_id, env.target_token_id, None, policy_iteration,
            scenario=env.episode.scenario,
        )
        baseline_context = MeanFieldBranchContext(
            env, env.target_batch_id, env.target_token_id, None, None, base_key,
        )
        attempted += 1
        try:
            baseline = _finite(
                cost_evaluator(baseline_context), "mean-field baseline cost",
            )
        except Exception as error:
            raise PopulationResponseError(
                f"mean-field cost evaluation failed: {type(error).__name__}: {error}",
                attempted_calls=attempted,
                original_exception_type=type(error).__name__,
            ) from error
        if env.path_fingerprint != baseline_context.environment.path_fingerprint:
            raise PopulationResponseError(
                "mean-field environment changed during baseline scoring"
            )
        baseline_costs.append(baseline)
        for bucket in buckets:
            branch_key = _branch_key(
                namespace, macro_seed, env.episode.identity.key,
                env.target_batch_id, env.target_token_id, bucket, policy_iteration,
                scenario=env.episode.scenario,
            )
            selected = select_replica_for_action_bucket(
                env.target_batch, env.target_token_id, bucket,
                action_key=branch_key,
            )
            context = MeanFieldBranchContext(
                env, env.target_batch_id, env.target_token_id,
                bucket, selected, branch_key,
            )
            attempted += 1
            try:
                value = _finite(
                    cost_evaluator(context), "mean-field action cost",
                )
            except Exception as error:
                raise PopulationResponseError(
                    f"mean-field cost evaluation failed: {type(error).__name__}: {error}",
                    attempted_calls=attempted,
                    original_exception_type=type(error).__name__,
                ) from error
            if env.path_fingerprint != context.environment.path_fingerprint:
                raise PopulationResponseError(
                    "mean-field environment changed during action scoring"
                )
            values[bucket.fingerprint].append(value)
            rows.append(MeanFieldContinuationRow(
                env.episode.identity.key, env.target_batch_id, env.target_token_id,
                bucket, selected, branch_key, value, env.path_fingerprint,
            ))
    summaries = tuple(
        _mean_summary(
            bucket, values[bucket.fingerprint],
            minimum_panels=min_complete_panels,
        )
        for bucket in buckets
    )
    return MeanFieldContinuationResult(
        MEAN_FIELD_LABEL, namespace, macro_seed, cost_model_id,
        tuple(rows), summaries, tuple(baseline_costs), attempted, True,
    )


def _run_finite_branch(
    trace: RoutingTrace,
    policy_factory: Callable[[], SimultaneousPolicy],
    *,
    target_token_id: int,
    target_batch_id: int,
    action_bucket: ReplicaStateActionBucket | None,
    branch_key: str,
    schema: BucketSchema,
    policy_name: str,
    snapshot_recorder: Callable[[int, object], None] | None = None,
    fork_snapshot: object | None = None,
    seed_calls: Sequence[object] = (),
    base_policy: object | None = None,
) -> tuple[SimultaneousRoutingResult, _BucketInterventionPolicy, object]:
    if not callable(policy_factory):
        raise PopulationResponseError("policy_factory must be callable")
    if base_policy is None:
        try:
            base = policy_factory()
        except Exception as error:
            raise PopulationResponseError(
                f"policy factory failed: {type(error).__name__}: {error}",
                original_exception_type=type(error).__name__,
            ) from error
    else:
        base = base_policy
    if not callable(getattr(base, "choose", None)):
        raise PopulationResponseError("policy factory did not provide online choose")
    wrapper = _BucketInterventionPolicy(
        base, target_token_id=target_token_id, target_batch_id=target_batch_id,
        action_bucket=action_bucket, branch_key=branch_key, schema=schema,
        seed_calls=seed_calls,
    )
    try:
        result = simulate_simultaneous_routing(
            trace, wrapper, policy_name=policy_name,
            snapshot_recorder=snapshot_recorder,
            fork_snapshot=fork_snapshot,
        )
    except Exception as error:
        raise PopulationResponseError(
            f"finite-K physics failed: {type(error).__name__}: {error}",
            original_exception_type=type(error).__name__,
        ) from error
    if not result.physical.complete_drain:
        raise PopulationResponseError("finite-K branch did not complete drain")
    if result.physical.trace_fingerprint != trace.fingerprint:
        raise PopulationResponseError("finite-K branch trace fingerprint changed")
    wrapper.target_observation()
    return result, wrapper, base


def evaluate_finite_k_deviation(
    trace: RoutingTrace,
    episode: ForwardEpisode,
    policy_factory: Callable[[], SimultaneousPolicy],
    target_token_id: int,
    action_buckets: Sequence[ReplicaStateActionBucket],
    cost_evaluator: Callable[[SimultaneousRoutingResult, int], float],
    *,
    cost_model_id: str,
    target_batch_id: int | None = None,
    namespace: str = FINITE_K_NAMESPACE,
    macro_seed: int = FINITE_K_MACRO_SEED,
    policy_iteration: int = 0,
    schema: BucketSchema = BIN_SCHEMA_V1,
    expected_namespace: str = FINITE_K_NAMESPACE,
    expected_macro_seed: int = FINITE_K_MACRO_SEED,
) -> FiniteKDeviationResult:
    """Re-run complete finite physics for one tagged action intervention."""

    _validate_namespace(
        namespace, expected_namespace, macro_seed, expected_macro_seed,
    )
    _strict_int(target_token_id, "target Token ID")
    _strict_int(policy_iteration, "policy iteration")
    _validate_cost_model(cost_model_id, cost_evaluator)
    if not isinstance(schema, BucketSchema):
        raise PopulationResponseError("bucket schema is invalid")
    if not isinstance(episode, ForwardEpisode):
        raise PopulationResponseError("finite-K forward episode is invalid")
    if target_batch_id is None:
        candidates = tuple(
            batch.batch_id for batch in episode.batches
            if target_token_id in batch.eta.token_ids
        )
        if len(candidates) != 1:
            raise PopulationResponseError(
                "target batch must be explicit when selection is ambiguous"
            )
        target_batch_id = candidates[0]
    _strict_int(target_batch_id, "target batch ID")
    batch = _validate_episode_for_finite(
        trace, episode, target_batch_id, target_token_id,
    )
    buckets = _validate_candidate_buckets(
        batch, target_token_id, action_buckets, schema,
    )

    retained_policies: list[object] = []
    baseline_key = _branch_key(
        namespace, macro_seed, episode.identity.key,
        target_batch_id, target_token_id, None, policy_iteration,
        scenario=trace.scenario,
    )
    # F2: the snapshot-fork path is only valid for policies that are pure
    # functions of (observation, context, key); such policies carry the
    # __engine_stateless__ marker. Everything else falls back to full replay.
    try:
        probe_policy = policy_factory()
    except Exception as error:
        raise PopulationResponseError(
            f"policy factory failed: {type(error).__name__}: {error}",
            attempted_calls=1,
            original_exception_type=type(error).__name__,
        ) from error
    stateless = bool(getattr(probe_policy, "__engine_stateless__", False))
    cache_key = _baseline_cache_key(
        trace, getattr(probe_policy, "fingerprint", None),
        policy_iteration, namespace, macro_seed,
    )
    cached = _BASELINE_CACHE.get(cache_key) if cache_key is not None else None
    held_snapshots: dict[int, object] = {}
    attempted = 1
    if cached is not None:
        baseline_result, baseline_wrapper, baseline_policy, held_snapshots = cached
    else:
        def _record_snapshot(batch_id: int, snapshot: object) -> None:
            if batch_id == target_batch_id:
                held_snapshots[batch_id] = snapshot

        try:
            baseline_result, baseline_wrapper, baseline_policy = _run_finite_branch(
                trace, policy_factory, target_token_id=target_token_id,
                target_batch_id=target_batch_id, action_bucket=None,
                branch_key=baseline_key, schema=schema,
                policy_name="finite_k_baseline",
                snapshot_recorder=_record_snapshot if stateless else None,
                base_policy=probe_policy,
            )
        except PopulationResponseError as error:
            raise PopulationResponseError(
                str(error), attempted_calls=attempted,
                original_exception_type=error.original_exception_type,
            ) from error
        if cache_key is not None:
            _BASELINE_CACHE[cache_key] = (
                baseline_result, baseline_wrapper, baseline_policy,
                held_snapshots,
            )
    retained_policies.append(baseline_policy)
    try:
        baseline_cost = _finite(
            cost_evaluator(baseline_result, target_token_id),
            "finite-K baseline cost",
        )
    except Exception as error:
        raise PopulationResponseError(
            f"finite-K baseline cost failed: {type(error).__name__}: {error}",
            attempted_calls=attempted,
            original_exception_type=type(error).__name__,
        ) from error
    baseline_physical_fp = _physical_fingerprint(baseline_result)
    baseline_policy_fp = baseline_wrapper.call_fingerprint()
    baseline_target = baseline_wrapper.target_calls[0]
    baseline_target_observation = baseline_wrapper.target_observation()

    output_rows: list[FiniteKDeviationRow] = []
    fork = held_snapshots.get(target_batch_id) if stateless else None
    seed_calls: tuple = ()
    if fork is not None:
        target_position = baseline_wrapper.calls.index(
            baseline_wrapper.target_calls[0]
        )
        seed_calls = tuple(baseline_wrapper.calls[:target_position])
    for bucket in buckets:
        branch_key = _branch_key(
            namespace, macro_seed, episode.identity.key,
            target_batch_id, target_token_id, bucket, policy_iteration,
            scenario=trace.scenario,
        )
        attempted += 1
        try:
            candidate_result, candidate_wrapper, candidate_policy = _run_finite_branch(
                trace, policy_factory, target_token_id=target_token_id,
                target_batch_id=target_batch_id, action_bucket=bucket,
                branch_key=branch_key, schema=schema,
                policy_name=f"finite_k_candidate_{bucket.fingerprint[:12]}",
                fork_snapshot=fork,
                seed_calls=seed_calls,
            )
            retained_policies.append(candidate_policy)
            candidate_target = candidate_wrapper.target_calls[0]
            candidate_observation = candidate_wrapper.target_observation()
            if baseline_target_observation != candidate_observation:
                raise PopulationResponseError(
                    "paired branches changed the target prefix observation"
                )
            baseline_prefix = baseline_wrapper.calls[
                :baseline_wrapper.calls.index(baseline_target)
            ]
            candidate_prefix = candidate_wrapper.calls[
                :candidate_wrapper.calls.index(candidate_target)
            ]
            if baseline_prefix != candidate_prefix:
                raise PopulationResponseError(
                    "paired branches changed the pre-target online prefix"
                )
            candidate_cost = _finite(
                cost_evaluator(candidate_result, target_token_id),
                "finite-K candidate cost",
            )
        except PopulationResponseError as error:
            raise PopulationResponseError(
                str(error), attempted_calls=attempted,
                original_exception_type=error.original_exception_type,
            ) from error
        except Exception as error:
            raise PopulationResponseError(
                f"finite-K candidate failed: {type(error).__name__}: {error}",
                attempted_calls=attempted,
                original_exception_type=type(error).__name__,
            ) from error
        candidate_physical_fp = _physical_fingerprint(candidate_result)
        difference = candidate_cost - baseline_cost
        selected = candidate_target.applied
        if type(selected) is not int or selected < 0:
            raise PopulationResponseError("finite-K candidate did not select a Replica")
        output_rows.append(FiniteKDeviationRow(
            episode.identity.key, target_batch_id, target_token_id, bucket,
            selected, baseline_target.applied,
            baseline_cost, candidate_cost, difference, branch_key,
            trace.fingerprint, baseline_physical_fp, candidate_physical_fp,
            baseline_policy_fp, candidate_wrapper.call_fingerprint(),
        ))

    return FiniteKDeviationResult(
        FINITE_K_LABEL, namespace, macro_seed, cost_model_id,
        episode.identity.key, trace.fingerprint, target_batch_id, target_token_id,
        tuple(output_rows), attempted, True,
    )


def evaluate_finite_k_deviation_panel(
    cases: Sequence[FiniteKDeviationCase],
    policy_factory: Callable[[], SimultaneousPolicy],
    cost_evaluator: Callable[[SimultaneousRoutingResult, int], float],
    *,
    cost_model_id: str,
    namespace: str = FINITE_K_NAMESPACE,
    macro_seed: int = FINITE_K_MACRO_SEED,
    policy_iteration: int = 0,
    schema: BucketSchema = BIN_SCHEMA_V1,
    expected_namespace: str = FINITE_K_NAMESPACE,
    expected_macro_seed: int = FINITE_K_MACRO_SEED,
    expected_episodes: int = IMPLEMENTATION_EPISODES,
    max_calls: int = MAX_CALLS_PER_OUTPUT,
) -> FiniteKDeviationPanelResult:
    """Run the frozen 16-episode finite-K panel in canonical order."""

    _validate_namespace(
        namespace, expected_namespace, macro_seed, expected_macro_seed,
    )
    _strict_int(expected_episodes, "expected finite-K episodes", 1)
    _strict_int(max_calls, "finite-K call ceiling", 1)
    _validate_cost_model(cost_model_id, cost_evaluator)
    rows = tuple(cases)
    if len(rows) != expected_episodes:
        raise PopulationResponseError(
            "finite-K implementation panel must contain 16 episodes"
        )
    if any(not isinstance(case, FiniteKDeviationCase) for case in rows):
        raise PopulationResponseError("finite-K panel case is invalid")
    ordered = tuple(
        sorted(rows, key=lambda case: case.episode.identity.key)
    )
    episode_keys = tuple(case.episode.identity.key for case in ordered)
    trace_keys = tuple(case.trace.fingerprint for case in ordered)
    if len(set(episode_keys)) != len(episode_keys):
        raise PopulationResponseError("finite-K panel episode identity is duplicated")
    if len(set(trace_keys)) != len(trace_keys):
        raise PopulationResponseError("finite-K panel trace identity is duplicated")
    if len(ordered) * (1 + MAX_ACTION_BUCKETS) > max_calls:
        raise PopulationResponseError("finite-K panel exceeds its frozen call ceiling")

    compact_rows: list[FiniteKDeviationRow] = []
    per_episode_calls: list[int] = []
    completed_calls = 0
    for case in ordered:
        try:
            result = evaluate_finite_k_deviation(
                case.trace,
                case.episode,
                policy_factory,
                case.target_token_id,
                case.action_buckets,
                cost_evaluator,
                cost_model_id=cost_model_id,
                target_batch_id=case.target_batch_id,
                namespace=namespace,
                macro_seed=macro_seed,
                policy_iteration=policy_iteration,
                schema=schema,
                expected_namespace=expected_namespace,
                expected_macro_seed=expected_macro_seed,
            )
        except PopulationResponseError as error:
            raise PopulationResponseError(
                str(error),
                attempted_calls=completed_calls + error.attempted_calls,
                original_exception_type=error.original_exception_type,
            ) from error
        compact_rows.extend(result.rows)
        per_episode_calls.append(result.attempted_calls)
        completed_calls += result.attempted_calls
    if completed_calls > max_calls:
        raise PopulationResponseError(
            "finite-K panel exceeded its frozen call ceiling",
            attempted_calls=completed_calls,
        )
    return FiniteKDeviationPanelResult(
        FINITE_K_LABEL, namespace, macro_seed, cost_model_id,
        tuple(compact_rows), episode_keys, tuple(per_episode_calls),
        completed_calls, True,
    )


__all__ = [
    "ActionCostSummary",
    "FINITE_K_LABEL",
    "FINITE_K_MACRO_SEED",
    "FINITE_K_NAMESPACE",
    "FiniteKDeviationResult",
    "FiniteKDeviationCase",
    "FiniteKDeviationPanelResult",
    "FiniteKDeviationRow",
    "IMPLEMENTATION_EPISODES",
    "MAX_ACTION_BUCKETS",
    "MAX_CALLS_PER_OUTPUT",
    "MAX_TOTAL_CALLS",
    "MEAN_FIELD_LABEL",
    "MEAN_FIELD_MACRO_SEED",
    "MEAN_FIELD_NAMESPACE",
    "MeanFieldBranchContext",
    "MeanFieldContinuationResult",
    "MeanFieldContinuationRow",
    "MeanFieldEnvironment",
    "MIN_COMPLETE_PANELS",
    "PopulationResponseError",
    "ReplicaStateActionBucket",
    "action_buckets_for_token",
    "evaluate_finite_k_deviation",
    "evaluate_finite_k_deviation_panel",
    "evaluate_mean_field_continuation",
    "select_replica_for_action_bucket",
]
