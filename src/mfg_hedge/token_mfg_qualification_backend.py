"""Concrete bounded adapter from finite routing physics to the MFG response layer.

This module is intentionally a bounded integration slice.  It uses the real
trace, population, continuation, finite-K, and response implementations, but
does not claim that the frozen 51,072-call qualification graph is ready to
run.  The formal campaign remains gated until its dependent work graph can
persist the dynamic solver dependencies without replaying a physical call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .qualification_checkpoint import QualificationCheckpointError, QualificationCheckpointStore
from .population_estimator import (
    BIN_SCHEMA_V1,
    BucketSchema,
    ForwardEpisode,
    PopulationCellKey,
    PopulationEpisodeIdentity,
    PopulationPolicy,
    ReplicaBucket,
    run_population_forward_library,
)
from .reliability_aware_routing import RoutingTrace
from .simultaneous_token_routing import (
    PopulationDecisionContext,
    SimultaneousRoutingResult,
    TokenDecisionObservation,
)
from .token_mfg_qualification import (
    FiniteKGainRow,
    FixedEpisodeLibrary,
    SimultaneousFiniteKGainBound,
)
from .token_mfg_qualification import compute_simultaneous_finite_k_gain_bound
from .token_mfg_qualification_campaign import (
    QualificationCampaignError,
    QualificationPlan,
    QualificationWorkUnit,
    qualification_source_fingerprint,
)
from .token_mfg_response_solver import (
    ActionCostEstimate,
    FinalPolicyConfirmation,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    ForwardStateRow,
    MFGCostModel,
    MFGModelResult,
    MFGSolverConfig,
    PolicyRow,
    PolicyState,
    confirm_final_policy,
    solve_unpriced_priced_mfg,
)
from .token_population_response import (
    FINITE_K_MACRO_SEED,
    FINITE_K_NAMESPACE,
    MEAN_FIELD_MACRO_SEED,
    MEAN_FIELD_NAMESPACE,
    MeanFieldBranchContext,
    MeanFieldEnvironment,
    ReplicaStateActionBucket,
    action_buckets_for_token,
    evaluate_finite_k_deviation,
    evaluate_mean_field_continuation,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_k_simultaneous_bound(
    panel: Sequence[tuple[str, object]],
    *,
    minimum_samples: int,
    critical_value: float = 2.0,
) -> SimultaneousFiniteKGainBound:
    """Aggregate complete pathwise rows into the qualification UCB.

    ``FiniteKDeviationRow.pathwise_difference`` is candidate minus baseline;
    qualification gain is baseline minus candidate, hence the sign reversal.
    The grouping key is the observed state/action support, never a global
    episode mean.  Incomplete support is returned as an explicit
    ``statistics_insufficient`` bound so formal callers fail closed.
    """

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for state_id, row in panel:
        key = (state_id, row.candidate_bucket.fingerprint)
        grouped.setdefault(key, []).append((
            -float(row.pathwise_difference), float(row.baseline_cost),
        ))
    if not grouped:
        raise QualificationCampaignError("finite-K deviation panel has no supported rows")

    gain_rows: list[FiniteKGainRow] = []
    normalized_upper: list[float] = []
    for (state_id, action_id), pairs in sorted(grouped.items()):
        values = tuple(value for value, _ in pairs)
        count = len(values)
        mean = math.fsum(values) / count
        if count > 1:
            variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
            standard_error = math.sqrt(max(0.0, variance / count))
        else:
            standard_error = 0.0
        gain_rows.append(FiniteKGainRow(
            state_id, action_id, mean, standard_error, count,
        ))
        baseline = math.fsum(value for _, value in pairs) / count
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise QualificationCampaignError(
                "finite-K baseline private cost must be finite and positive"
            )
        normalized_upper.append(
            (mean + critical_value * standard_error) / baseline
        )
    ordered = tuple(gain_rows)
    upper = tuple(
        row.gain_mean + critical_value * row.paired_standard_error
        for row in ordered
    )
    if any(row.effective_n < minimum_samples for row in ordered):
        return SimultaneousFiniteKGainBound(
            ordered, critical_value, max(upper), len(ordered), False,
            "statistics_insufficient", max(normalized_upper),
        )
    raw = compute_simultaneous_finite_k_gain_bound(
        ordered,
        critical_value=critical_value,
        min_complete_samples=minimum_samples,
    )
    return replace(raw, normalized_max_upper_bound=max(normalized_upper))


def _bucket_map(batch, token_id: int, schema: BucketSchema) -> dict[int, ReplicaStateActionBucket]:
    row = next(row for row in batch.token_rows if row.token_id == token_id)
    return {
        replica.replica_id: ReplicaStateActionBucket.from_replica(
            replica, row.cell_key.locality_signature[index],
        )
        for index, replica in enumerate(batch.nu.replica_buckets)
        if replica.health == "UP"
    }


def _state_id_from_cell(cell: PopulationCellKey) -> str:
    return _digest(asdict(cell))


def _cell_for_observation(
    observation: TokenDecisionObservation,
    schema: BucketSchema,
) -> PopulationCellKey:
    replica_state = tuple(
        schema.bucket_replica(row)
        for row in observation.public_state.replica_observations
    )
    locality = tuple(
        "same_domain"
        if row.domain_id == observation.ingress_rank % 4
        else "other_domain"
        for row in observation.public_state.replica_observations
    )
    return PopulationCellKey(
        schema.bucket_token(observation), replica_state, locality,
        observation.public_state.topology,
    )


def _stable_unit(policy_key: str, suffix: str = "") -> float:
    digest = hashlib.sha256(f"{policy_key}|{suffix}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


@dataclass(frozen=True)
class RoutingPolicySeed:
    """Checkpointable online seed used only for formal initialization."""

    kind: str
    marginals: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {
            "uniform", "simultaneous_loew_projection",
            "corrected_risk_aware_projection", "scale_marginal_projection",
        }:
            raise ValueError("routing policy seed kind is invalid")
        if self.kind != "scale_marginal_projection" and self.marginals:
            raise ValueError("only a scale seed may carry marginals")
        if self.kind == "scale_marginal_projection":
            if tuple(name for name, _ in self.marginals) != ("Regular", "Urgent"):
                raise ValueError("scale seed must cover Regular and Urgent")
            for _, rows in self.marginals:
                actions = tuple(action for action, _ in rows)
                if (
                    not rows
                    or actions != tuple(sorted(actions))
                    or len(set(actions)) != len(actions)
                ):
                    raise ValueError("scale marginals must be non-empty and canonical")
                values = tuple(value for _, value in rows)
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                    for value in values
                ):
                    raise ValueError("scale marginal values must be finite and nonnegative")
                if not math.isclose(
                    math.fsum(values), 1.0,
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise ValueError("scale marginals must sum to one")

    @property
    def fingerprint(self) -> str:
        return _digest(asdict(self))


def materialize_policy(snapshot: ForwardSnapshot) -> PolicyState:
    """Materialize the policy that actually generated one forward snapshot."""

    return PolicyState(tuple(
        PolicyRow(
            state.state_id,
            tuple(action for action, _ in state.predicted_share),
            tuple(probability for _, probability in state.predicted_share),
        )
        for state in snapshot.states
    ))


class _PolicyStateAdapter:
    """Online population policy backed by a solver PolicyState.

    ``choose`` is a pure function of (observation, context, policy_key): all
    randomness is hash-keyed and no instance state is mutated across calls,
    so the engine's snapshot-fork fast path (ticket 15) is valid for it.
    """

    __engine_stateless__ = True

    @property
    def fingerprint(self) -> str:
        policy = self.policy
        marker = policy.fingerprint if policy is not None else "none"
        return _digest({"policy_state_adapter": marker, "schema": self.schema.schema_id})

    def __init__(
        self, policy: PolicyState | RoutingPolicySeed | None,
        schema: BucketSchema,
    ):
        self.policy = policy
        self.schema = schema

    def _available(
        self, observation: TokenDecisionObservation,
    ) -> dict[str, tuple[ReplicaStateActionBucket, tuple[int, ...]]]:
        state = observation.public_state
        by_bucket: dict[str, tuple[ReplicaStateActionBucket, list[int]]] = {}
        for index, replica in enumerate(state.replica_observations):
            if not replica.available:
                continue
            bucket = ReplicaStateActionBucket.from_replica(
                self.schema.bucket_replica(replica),
                "same_domain"
                if replica.domain_id == observation.ingress_rank % 4
                else "other_domain",
            )
            key = bucket.fingerprint
            if key not in by_bucket:
                by_bucket[key] = (bucket, [])
            by_bucket[key][1].append(replica.replica_id)
        return {
            key: (bucket, tuple(sorted(replicas)))
            for key, (bucket, replicas) in by_bucket.items()
        }

    def _probabilities(
        self,
        observation: TokenDecisionObservation,
        available: Mapping[str, tuple[ReplicaStateActionBucket, tuple[int, ...]]],
    ) -> tuple[tuple[str, float], ...]:
        state_id = _state_id_from_cell(_cell_for_observation(observation, self.schema))
        row = None
        if isinstance(self.policy, PolicyState):
            row = next((row for row in self.policy.rows if row.state_id == state_id), None)
        if isinstance(self.policy, RoutingPolicySeed) and (
            self.policy.kind == "scale_marginal_projection"
        ):
            marginal = dict(dict(self.policy.marginals)[observation.token_class])
            weights = {
                action_id: marginal.get(action_id, 0.0)
                for action_id in available
                if marginal.get(action_id, 0.0) > 0.0
            }
            total = math.fsum(weights.values())
            if total > 0.0:
                return tuple(
                    (action_id, weights[action_id] / total)
                    for action_id in sorted(weights)
                )
        if row is None:
            value = 1.0 / len(available) if available else 0.0
            return tuple((key, value) for key in sorted(available))
        weights = {
            action_id: probability
            for action_id, probability in zip(row.action_ids, row.probabilities)
            if action_id in available and probability > 0.0
        }
        total = math.fsum(weights.values())
        if total <= 0.0:
            value = 1.0 / len(available) if available else 0.0
            return tuple((key, value) for key in sorted(available))
        return tuple(
            (action_id, weights[action_id] / total)
            for action_id in sorted(weights)
        )

    def _scale_has_supported_mass(
        self,
        observation: TokenDecisionObservation,
        available: Mapping[str, tuple[ReplicaStateActionBucket, tuple[int, ...]]],
    ) -> bool:
        if not isinstance(self.policy, RoutingPolicySeed) or (
            self.policy.kind != "scale_marginal_projection"
        ):
            return True
        marginal = dict(dict(self.policy.marginals)[observation.token_class])
        return any(marginal.get(action_id, 0.0) > 0.0 for action_id in available)

    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        available_replicas = tuple(
            row for row in observation.public_state.replica_observations
            if row.available
        )
        if isinstance(self.policy, RoutingPolicySeed) and self.policy.kind in {
            "uniform", "simultaneous_loew_projection",
            "corrected_risk_aware_projection",
        }:
            if not available_replicas:
                return None
            if self.policy.kind == "uniform":
                ordered = tuple(sorted(row.replica_id for row in available_replicas))
                return ordered[int(_stable_unit(policy_key) * len(ordered))]
            if self.policy.kind == "simultaneous_loew_projection":
                return min(
                    available_replicas,
                    key=lambda row: (
                        row.estimated_work, row.queue_depth, row.replica_id,
                    ),
                ).replica_id
            gamma = 12.0 if observation.token_class == "Urgent" else 8.0
            return min(
                available_replicas,
                key=lambda row: (
                    row.estimated_work + gamma * row.estimated_failure_risk,
                    row.queue_depth, row.replica_id,
                ),
            ).replica_id
        available = self._available(observation)
        if available_replicas and not self._scale_has_supported_mass(
            observation, available,
        ):
            ordered = tuple(sorted(row.replica_id for row in available_replicas))
            return ordered[int(_stable_unit(policy_key) * len(ordered))]
        probabilities = self._probabilities(observation, available)
        if not probabilities:
            return None
        draw = _stable_unit(policy_key)
        cumulative = 0.0
        selected = probabilities[-1][0]
        for action_id, probability in probabilities:
            cumulative += probability
            if draw < cumulative:
                selected = action_id
                break
        candidates = available[selected][1]
        return candidates[
            int(_stable_unit(policy_key, selected) * len(candidates))
        ]

    def predict_token_share(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
    ) -> tuple[tuple[int, float], ...]:
        available_replicas = tuple(
            row for row in observation.public_state.replica_observations
            if row.available
        )
        if isinstance(self.policy, RoutingPolicySeed) and self.policy.kind in {
            "uniform", "simultaneous_loew_projection",
            "corrected_risk_aware_projection",
        }:
            if not available_replicas:
                return ()
            if self.policy.kind == "uniform":
                share = 1.0 / len(available_replicas)
                return tuple((row.replica_id, share) for row in available_replicas)
            selected = self.choose(observation, context, "prediction")
            return () if selected is None else ((selected, 1.0),)
        available = self._available(observation)
        if available_replicas and not self._scale_has_supported_mass(
            observation, available,
        ):
            share = 1.0 / len(available_replicas)
            return tuple(
                (row.replica_id, share)
                for row in sorted(available_replicas, key=lambda row: row.replica_id)
            )
        probabilities = self._probabilities(observation, available)
        shares: dict[int, float] = {}
        for action_id, probability in probabilities:
            candidates = available[action_id][1]
            for replica_id in candidates:
                shares[replica_id] = shares.get(replica_id, 0.0) + (
                    probability / len(candidates)
                )
        total = math.fsum(shares.values())
        if total <= 0.0:
            return ()
        return tuple(
            (replica_id, shares[replica_id] / total)
            for replica_id in sorted(shares)
        )


def _share_by_bucket(
    batch,
    token_id: int,
    values: Sequence[tuple[int, float]],
) -> tuple[tuple[str, float], ...]:
    buckets = _bucket_map(batch, token_id, BIN_SCHEMA_V1)
    grouped: dict[str, float] = {}
    for replica_id, share in values:
        bucket = buckets.get(replica_id)
        if bucket is not None:
            grouped[bucket.fingerprint] = grouped.get(bucket.fingerprint, 0.0) + share
    return tuple(sorted(grouped.items()))


@dataclass(frozen=True)
class BoundedBackendResult:
    status: str
    topology_k: int
    trace_library_fingerprint: str
    trace_fingerprint: str
    layers: tuple[str, ...]
    forward_calls: int
    continuation_calls: int
    finite_k_calls: int
    response_calls: int
    confirmation_calls: int
    total_calls: int
    formal_calls: int = 0
    real_physics: bool = True
    dispatched_calls: int = 0
    reused_calls: int = 0

    def to_checkpoint_output(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "topology_k": self.topology_k,
            "trace_library_fingerprint": self.trace_library_fingerprint,
            "trace_fingerprint": self.trace_fingerprint,
            "layers": self.layers,
            "forward_calls": self.forward_calls,
            "continuation_calls": self.continuation_calls,
            "finite_k_calls": self.finite_k_calls,
            "response_calls": self.response_calls,
            "confirmation_calls": self.confirmation_calls,
            "total_calls": self.total_calls,
            "formal_calls": self.formal_calls,
            "real_physics": self.real_physics,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.to_checkpoint_output(), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")


class ConcreteQualificationBackend:
    """Real bounded adapter used before the formal qualification graph."""

    real_physics = True

    def __init__(
        self,
        library: FixedEpisodeLibrary,
        *,
        schema: BucketSchema = BIN_SCHEMA_V1,
        mean_field_namespace: str = MEAN_FIELD_NAMESPACE,
        mean_field_macro_seed: int = MEAN_FIELD_MACRO_SEED,
        finite_k_namespace: str = FINITE_K_NAMESPACE,
        finite_k_macro_seed: int = FINITE_K_MACRO_SEED,
        bounded: bool = True,
        min_complete_panels: int = 2,
    ):
        if not isinstance(library, FixedEpisodeLibrary):
            raise TypeError("library must be FixedEpisodeLibrary")
        if bounded and library.traces[0].topology.K != 8:
            raise ValueError("bounded backend is restricted to K=8")
        if not bounded and library.traces[0].topology.K not in {8, 16, 32, 64}:
            raise ValueError("formal backend K must be one of 8, 16, 32, 64")
        # A streamed work unit intentionally owns one episode.  The aggregate
        # panel, not the worker-local library, is responsible for sample-floor
        # validation.  Keep the historical bounded adapter's guard intact.
        if bounded and len(library.traces) < 2:
            raise ValueError("bounded backend needs at least two episodes")
        if not isinstance(schema, BucketSchema):
            raise TypeError("schema must be BucketSchema")
        for name, value in (
            ("mean_field_namespace", mean_field_namespace),
            ("finite_k_namespace", finite_k_namespace),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (
            ("mean_field_macro_seed", mean_field_macro_seed),
            ("finite_k_macro_seed", finite_k_macro_seed),
        ):
            if type(value) is not int:
                raise ValueError(f"{name} must be a true int")
        if type(bounded) is not bool:
            raise ValueError("bounded must be bool")
        if type(min_complete_panels) is not int or min_complete_panels < 1:
            raise ValueError("min_complete_panels must be a positive true int")
        self.library = library
        self.schema = schema
        self.bounded = bounded
        self.min_complete_panels = min_complete_panels
        self.mean_field_namespace = mean_field_namespace
        self.mean_field_macro_seed = mean_field_macro_seed
        self.finite_k_namespace = finite_k_namespace
        self.finite_k_macro_seed = finite_k_macro_seed
        self.source_fingerprint = qualification_source_fingerprint()
        self._episodes_by_environment: dict[str, tuple[ForwardEpisode, ...]] = {}
        self._last_snapshot: dict[str, ForwardSnapshot] = {}
        self._calls = {"forward": 0, "continuation": 0, "finite_k": 0, "confirmation": 0}

    def build_qualification_work_units(
        self, plan: QualificationPlan,
    ) -> Sequence[QualificationWorkUnit]:
        """Refuse the formal graph until dynamic dependency persistence is added.

        The refusal is deliberate: a static work graph cannot safely pass a
        newly produced PolicyState and ForwardEpisode into later units without
        either replaying physical calls or serializing endogenous state.  The
        bounded adapter below proves the concrete path first.
        """

        raise QualificationCampaignError(
            "bounded concrete backend is not a formal qualification work graph; "
            "dynamic solver dependency persistence is required before dispatch"
        )

    def worker(self, payload: Any) -> Mapping[str, Any]:
        raise QualificationCampaignError(
            "bounded concrete backend has no formal spawn worker"
        )

    def _forward(
        self, policy: PolicyState | RoutingPolicySeed | None,
        model_id: str, iteration: int,
    ) -> ForwardSnapshot:
        episodes = run_population_forward_library(
            self.library.traces,
            lambda: _PolicyStateAdapter(policy, self.schema),
            schema=self.schema,
        )
        self._calls["forward"] += len(episodes)
        state_values: dict[str, list[tuple[str, tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], str]]] = {}
        scale_fallback_count = 0
        for episode in episodes:
            # The bounded adapter uses the first causal decision cohort as its
            # supported state panel.  The underlying forward physics still
            # runs and drains the complete trace; limiting only the exported
            # panel keeps this smoke bounded and explicit.
            for batch in episode.batches[:1]:
                for token_row in batch.token_rows:
                    state_id = _state_id_from_cell(token_row.cell_key)
                    actions = action_buckets_for_token(
                        batch, token_row.token_id, schema=self.schema,
                    )
                    action_ids = tuple(bucket.fingerprint for bucket in actions)
                    if isinstance(policy, RoutingPolicySeed) and (
                        policy.kind == "scale_marginal_projection"
                    ):
                        supported = set(dict(policy.marginals)[
                            token_row.cell_key.token_bucket.token_class
                        ])
                        if not supported.intersection(action_ids):
                            scale_fallback_count += 1
                    predicted = dict(_share_by_bucket(
                        batch, token_row.token_id, token_row.predicted_share,
                    ))
                    realized = dict(_share_by_bucket(
                        batch, token_row.token_id,
                        () if token_row.realized_replica_id is None
                        else ((token_row.realized_replica_id, 1.0),),
                    ))
                    state_values.setdefault(state_id, []).append((
                        state_id,
                        tuple((action_id, predicted.get(action_id, 0.0)) for action_id in action_ids),
                        tuple((action_id, realized.get(action_id, 0.0)) for action_id in action_ids),
                        token_row.cell_key.token_bucket.token_class,
                    ))
        total = sum(len(rows) for rows in state_values.values())
        states = []
        for state_id in sorted(state_values):
            rows = state_values[state_id]
            action_ids = tuple(sorted({action_id for _, predicted, _, _ in rows for action_id, _ in predicted}))
            predicted = tuple(
                (action_id, math.fsum(
                    dict(row[1]).get(action_id, 0.0) for row in rows
                ) / len(rows))
                for action_id in action_ids
            )
            realized = tuple(
                (action_id, math.fsum(
                    dict(row[2]).get(action_id, 0.0) for row in rows
                ) / len(rows))
                for action_id in action_ids
            )
            states.append(ForwardStateRow(
                state_id,
                rows[0][3],
                len(rows) / total,
                predicted,
                realized,
            ))
        state_tuple = tuple(states)
        common = tuple(episode.trace_fingerprint for episode in episodes)
        snapshot = ForwardSnapshot(
            model_id=model_id,
            iteration=iteration,
            policy_fingerprint=(
                policy.fingerprint if policy is not None else _digest("uniform-seed")
            ),
            environment_fingerprint=_digest({
                "library": self.library.fingerprint,
                "episodes": tuple(episode.path_fingerprint if hasattr(episode, "path_fingerprint") else episode.trace_fingerprint for episode in episodes),
                "model": model_id,
            }),
            trace_library_fingerprint=self.library.fingerprint,
            mu_fingerprint=_digest(tuple(asdict(batch.mu) for ep in episodes for batch in ep.batches)),
            nu_fingerprint=_digest(tuple(asdict(batch.nu) for ep in episodes for batch in ep.batches)),
            eta_fingerprint=_digest(tuple(asdict(batch.eta) for ep in episodes for batch in ep.batches)),
            x_fingerprint=_digest(tuple(asdict(batch.x) for ep in episodes for batch in ep.batches)),
            states=state_tuple,
            sample_count=len(episodes),
            complete=True,
            scheduler_calls=len(episodes),
            scale_fallback_count=scale_fallback_count,
        )
        self._episodes_by_environment[snapshot.environment_fingerprint] = episodes
        self._last_snapshot[model_id] = snapshot
        return snapshot

    @staticmethod
    def _target(episode: ForwardEpisode) -> tuple[int, int]:
        for batch in episode.batches:
            if batch.eta.token_ids:
                return batch.batch_id, batch.eta.token_ids[0]
        raise RuntimeError("real episode contains no target cohort")

    @staticmethod
    def _observable_cost(context: MeanFieldBranchContext) -> float:
        batch = context.environment.target_batch
        row = next(row for row in batch.token_rows if row.token_id == context.target_token_id)
        if context.selected_replica_id is None:
            selected = row.realized_replica_id
        else:
            selected = context.selected_replica_id
        if selected is None:
            return batch.time + 1.0
        replica = batch.nu.replica_buckets[selected]
        queue = batch.x.topology.replica_ids.index(selected) if selected in batch.x.topology.replica_ids else 0
        return batch.time + queue + 1.0 + (0.5 if replica.health != "UP" else 0.0)

    @staticmethod
    def _physical_latency(result: SimultaneousRoutingResult, token_id: int) -> float:
        token = next(row for row in result.physical.token_results if row.token_id == token_id)
        if token.completion_time is None:
            raise RuntimeError("finite-K target did not complete")
        return token.completion_time

    def _continuation(
        self, snapshot: ForwardSnapshot, policy: PolicyState,
        model_id: str, iteration: int,
    ) -> tuple[ActionCostEstimate, ...]:
        episodes = self._episodes_by_environment[snapshot.environment_fingerprint]
        grouped: dict[str, list[MeanFieldEnvironment]] = {}
        for episode in episodes:
            batch_id, token_id = self._target(episode)
            batch = next(batch for batch in episode.batches if batch.batch_id == batch_id)
            token_row = next(row for row in batch.token_rows if row.token_id == token_id)
            state_id = _state_id_from_cell(token_row.cell_key)
            grouped.setdefault(state_id, []).append(
                MeanFieldEnvironment(episode, batch_id, token_id)
            )
        estimates: list[ActionCostEstimate] = []
        for state_id in sorted(grouped):
            environments = tuple(grouped[state_id])
            actions = action_buckets_for_token(
                environments[0].target_batch,
                environments[0].target_token_id,
                schema=self.schema,
            )
            result = evaluate_mean_field_continuation(
                environments,
                actions,
                self._observable_cost,
                cost_model_id=model_id,
                namespace=self.mean_field_namespace,
                macro_seed=self.mean_field_macro_seed,
                policy_iteration=iteration,
                schema=self.schema,
                expected_namespace=self.mean_field_namespace,
                expected_macro_seed=self.mean_field_macro_seed,
                expected_episodes=len(environments),
                min_complete_panels=self.min_complete_panels,
            )
            self._calls["continuation"] += result.attempted_calls
            token_class = next(
                row.token_class for row in snapshot.states if row.state_id == state_id
            )
            estimates.extend(
                ActionCostEstimate(
                    state_id,
                    summary.action_bucket.fingerprint,
                    token_class,
                    summary.mean,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    summary.effective_n,
                    summary.standard_error,
                )
                for summary in result.action_summaries
            )
        return tuple(sorted(estimates, key=lambda row: (row.state_id, row.action_id)))

    def _finite_k(
        self, snapshot: ForwardSnapshot, policy: PolicyState,
        model_id: str, iteration: int,
    ) -> FiniteKDeviationDiagnostic:
        episodes = self._episodes_by_environment[snapshot.environment_fingerprint]
        rows: list[tuple[str, object]] = []
        max_calls = 0
        for episode in episodes:
            batch_id, token_id = self._target(episode)
            batch = next(batch for batch in episode.batches if batch.batch_id == batch_id)
            actions = action_buckets_for_token(batch, token_id, schema=self.schema)
            result = evaluate_finite_k_deviation(
                episode_trace := next(
                    trace for trace in self.library.traces
                    if trace.fingerprint == episode.trace_fingerprint
                ),
                episode,
                lambda: _PolicyStateAdapter(policy, self.schema),
                token_id,
                actions,
                self._physical_latency,
                cost_model_id=model_id,
                target_batch_id=batch_id,
                namespace=self.finite_k_namespace,
                macro_seed=self.finite_k_macro_seed,
                policy_iteration=iteration,
                schema=self.schema,
                expected_namespace=self.finite_k_namespace,
                expected_macro_seed=self.finite_k_macro_seed,
            )
            self._calls["finite_k"] += result.attempted_calls
            max_calls += result.attempted_calls
            state_id = _state_id_from_cell(
                next(row for row in batch.token_rows if row.token_id == token_id).cell_key
            )
            rows.extend((state_id, row) for row in result.rows)
        bound = _finite_k_simultaneous_bound(
            rows,
            minimum_samples=self.min_complete_panels,
        )
        return FiniteKDeviationDiagnostic(
            complete=True,
            effective_n=len(episodes),
            max_abs_pathwise_difference=max(
                abs(row.pathwise_difference) for _, row in rows
            ),
            trace_library_fingerprint=self.library.fingerprint,
            scheduler_calls=max_calls,
            simultaneous_ucb=bound.max_upper_bound,
            simultaneous_ucb_status=("complete" if bound.complete else bound.status),
            normalized_simultaneous_ucb=bound.normalized_max_upper_bound,
        )

    def _panel_call_counts(self) -> tuple[int, int, int]:
        """Return the exact bounded forward/continuation/deviation calls."""

        forward_calls = len(self.library.traces)
        first_batches = tuple(episode.batches[0] for episode in self._episodes_by_environment[next(iter(self._episodes_by_environment))]) if self._episodes_by_environment else ()
        continuation_calls = 0
        finite_calls = 0
        if first_batches:
            # Continuation is evaluated once per supported state.  Each state
            # has at most one target per episode, so its baseline plus action
            # branches are counted independently from other states.
            state_counts: dict[str, tuple[int, int]] = {}
            target_rows: list[tuple[object, int, int, int]] = []
            episodes = self._episodes_by_environment[next(iter(self._episodes_by_environment))]
            for episode in episodes:
                batch_id, token_id = self._target(episode)
                batch = next(batch for batch in episode.batches if batch.batch_id == batch_id)
                token_row = next(row for row in batch.token_rows if row.token_id == token_id)
                action_count = len(action_buckets_for_token(
                    batch, token_id, schema=self.schema,
                ))
                if 1 + action_count > 9:
                    raise QualificationCampaignError(
                        "one episode exceeds the frozen nine-call panel bound"
                    )
                target_rows.append((batch, token_id, action_count, _state_id_from_cell(token_row.cell_key)))
                state_id = target_rows[-1][3]
                state_counts[state_id] = (
                    state_counts.get(state_id, (0, 0))[0] + 1,
                    action_count,
                )
            continuation_calls = sum(
                count * (1 + action_count)
                for count, action_count in state_counts.values()
            )
            finite_calls = sum(1 + action_count for _, _, action_count, _ in target_rows)
            if continuation_calls > len(episodes) * 9 or finite_calls > len(episodes) * 9:
                raise QualificationCampaignError(
                    "panel call arithmetic exceeds the frozen per-episode bound"
                )
        return forward_calls, continuation_calls, finite_calls

    def run_bounded_k8(self) -> BoundedBackendResult:
        if not self.bounded or self.library.traces[0].topology.K != 8:
            raise QualificationCampaignError(
                "run_bounded_k8 requires the bounded K=8 backend"
            )
        self._calls = {"forward": 0, "continuation": 0, "finite_k": 0, "confirmation": 0}
        seed = self._forward(None, "unpriced_mfg", 0)
        initial_policy = PolicyState.uniform(seed.states)
        forward_calls, continuation_calls, finite_k_calls = self._panel_call_counts()
        config = MFGSolverConfig(
            max_iterations=1,
            min_complete_samples=2,
            forward_calls=forward_calls,
            continuation_calls=continuation_calls,
            finite_k_calls=finite_k_calls,
            call_budget=2 * (
                forward_calls + continuation_calls + finite_k_calls
            ),
        )
        batch = solve_unpriced_priced_mfg(
            self._forward,
            self._continuation,
            self._finite_k,
            initial_policy=initial_policy,
            config=config,
        )
        confirmations: list[FinalPolicyConfirmation] = []
        for model_result in batch.model_results:
            previous = self._last_snapshot[model_result.model_id]
            confirmation = confirm_final_policy(
                model_result.final_policy,
                MFGCostModel(model_result.model_id, price_per_queue_unit=(0.25 if model_result.model_id == "priced_mfg" else 0.0)),
                self._forward,
                self._continuation,
                self._finite_k,
                previous_snapshot=previous,
                config=config,
            )
            confirmations.append(confirmation)
            self._calls["confirmation"] += confirmation.attempted_calls
        statuses = tuple(row.status for row in confirmations)
        status = "bounded_complete" if all(
            row.status != "physical_failed" for row in confirmations
        ) else "bounded_fail_closed"
        total = sum(self._calls.values())
        return BoundedBackendResult(
            status,
            8,
            self.library.fingerprint,
            self.library.fingerprint,
            (
                "forward_population", "mean_field_continuation",
                "finite_k_deviation", "response_iteration",
                "final_policy_confirmation",
            ),
            self._calls["forward"],
            self._calls["continuation"],
            self._calls["finite_k"],
            len(batch.model_results),
            self._calls["confirmation"],
            total,
        )

    @staticmethod
    def _from_checkpoint_output(output: Mapping[str, Any]) -> BoundedBackendResult:
        return BoundedBackendResult(
            output["status"], output["topology_k"],
            output["trace_library_fingerprint"], output["trace_fingerprint"],
            tuple(output["layers"]), output["forward_calls"],
            output["continuation_calls"], output["finite_k_calls"],
            output["response_calls"], output["confirmation_calls"],
            output["total_calls"], output["formal_calls"],
            output["real_physics"], 0, 1,
        )

    def resume_bounded_from_checkpoint(
        self,
        checkpoint: QualificationCheckpointStore,
        key: str,
        input_fingerprint: str,
    ) -> BoundedBackendResult:
        try:
            output = checkpoint.completed_output(
                key, input_fingerprint=input_fingerprint,
            )
        except QualificationCheckpointError as error:
            raise RuntimeError(str(error)) from error
        if output is None:
            raise RuntimeError("bounded backend checkpoint output is missing")
        return self._from_checkpoint_output(output)


__all__ = ["BoundedBackendResult", "ConcreteQualificationBackend"]
