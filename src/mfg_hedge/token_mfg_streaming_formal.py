"""Memory-bounded formal r2 qualification over streamed episode identities.

Each worker materializes one trace, completes the forward/continuation/finite-K
panel for that episode, and returns compact sufficient statistics.  The parent
never owns a ``RoutingTrace`` or ``ForwardEpisode`` collection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping

from .population_estimator import BIN_SCHEMA_V1
from .qualification_checkpoint import QualificationCheckpointStore
from .token_mfg_formal_backend import FormalModelRun, FormalQualificationBackend
from .token_mfg_qualification_backend import (
    ConcreteQualificationBackend,
    RoutingPolicySeed,
    _PolicyStateAdapter,
    _finite_k_simultaneous_bound,
    _share_by_bucket,
    _state_id_from_cell,
    materialize_policy,
)
from .token_mfg_qualification_campaign import (
    QualificationCampaignError,
    QualificationPlan,
    QualificationTimingReport,
    QualificationWorkUnit,
    execute_canonical_work_units,
    qualification_source_fingerprint,
)
from .token_mfg_response_solver import (
    ActionCostEstimate,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    ForwardStateRow,
    MFGCostModel,
    PolicyState,
)
from .token_mfg_streaming import StreamingEpisodeSpec, build_streaming_panel, _digest
from .token_mfg_qualification_state_machine import _pack, _unpack
from .token_population_response import (
    MeanFieldEnvironment,
    action_buckets_for_token,
    evaluate_finite_k_deviation,
    evaluate_mean_field_continuation,
)


MAX_CALLS_PER_EPISODE = 19  # forward + (baseline + <=8 actions) twice


def _sample_standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance / len(values)))


def streaming_iteration_episode_worker(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one complete real episode and return no trace/episode object."""

    spec = payload.get("episode_spec")
    policy = payload.get("policy")
    model_id = payload.get("model_id")
    iteration = payload.get("iteration")
    if not isinstance(spec, StreamingEpisodeSpec):
        raise QualificationCampaignError("formal streaming episode identity is invalid")
    if not isinstance(policy, (PolicyState, RoutingPolicySeed)):
        raise QualificationCampaignError("formal streaming policy is invalid")
    if model_id not in {"unpriced_mfg", "priced_mfg"}:
        raise QualificationCampaignError("formal streaming model is invalid")
    if type(iteration) is not int or iteration < 0:
        raise QualificationCampaignError("formal streaming iteration is invalid")

    trace = spec.generate_trace()
    from .token_mfg_qualification import FixedEpisodeLibrary

    library = FixedEpisodeLibrary.from_traces((trace,))
    backend = ConcreteQualificationBackend(
        library,
        schema=BIN_SCHEMA_V1,
        mean_field_namespace=payload["mean_field_namespace"],
        mean_field_macro_seed=payload["mean_field_macro_seed"],
        finite_k_namespace=payload["finite_k_namespace"],
        finite_k_macro_seed=payload["finite_k_macro_seed"],
        bounded=False,
        min_complete_panels=1,
    )
    snapshot = backend._forward(policy, model_id, iteration)
    episode = backend._episodes_by_environment[snapshot.environment_fingerprint][0]

    compact_tokens = []
    for batch in episode.batches[:1]:
        for token_row in batch.token_rows:
            state_id = _state_id_from_cell(token_row.cell_key)
            actions = action_buckets_for_token(batch, token_row.token_id, schema=BIN_SCHEMA_V1)
            action_ids = tuple(bucket.fingerprint for bucket in actions)
            predicted = dict(_share_by_bucket(
                batch, token_row.token_id, token_row.predicted_share,
            ))
            realized = dict(_share_by_bucket(
                batch,
                token_row.token_id,
                () if token_row.realized_replica_id is None
                else ((token_row.realized_replica_id, 1.0),),
            ))
            compact_tokens.append((
                state_id,
                token_row.cell_key.token_bucket.token_class,
                tuple((action, predicted.get(action, 0.0)) for action in action_ids),
                tuple((action, realized.get(action, 0.0)) for action in action_ids),
            ))

    batch_id, token_id = backend._target(episode)
    batch = next(row for row in episode.batches if row.batch_id == batch_id)
    token_row = next(row for row in batch.token_rows if row.token_id == token_id)
    target_state = _state_id_from_cell(token_row.cell_key)
    actions = action_buckets_for_token(batch, token_id, schema=BIN_SCHEMA_V1)
    if 1 + len(actions) > 9:
        raise QualificationCampaignError("formal streamed target exceeds nine calls")
    continuation = evaluate_mean_field_continuation(
        (MeanFieldEnvironment(episode, batch_id, token_id),),
        actions,
        backend._observable_cost,
        cost_model_id=model_id,
        namespace=backend.mean_field_namespace,
        macro_seed=backend.mean_field_macro_seed,
        policy_iteration=iteration,
        schema=BIN_SCHEMA_V1,
        expected_namespace=backend.mean_field_namespace,
        expected_macro_seed=backend.mean_field_macro_seed,
        expected_episodes=1,
        min_complete_panels=1,
    )
    finite = evaluate_finite_k_deviation(
        trace,
        episode,
        lambda: _PolicyStateAdapter(policy, BIN_SCHEMA_V1),
        token_id,
        actions,
        backend._physical_latency,
        cost_model_id=model_id,
        target_batch_id=batch_id,
        namespace=backend.finite_k_namespace,
        macro_seed=backend.finite_k_macro_seed,
        policy_iteration=iteration,
        schema=BIN_SCHEMA_V1,
        expected_namespace=backend.finite_k_namespace,
        expected_macro_seed=backend.finite_k_macro_seed,
    )
    completed = 1 + continuation.attempted_calls + finite.attempted_calls
    return {
        "completed_calls": completed,
        "output": {
            "episode_identity": list(spec.identity),
            "trace_fingerprint": trace.fingerprint,
            "snapshot": _pack(snapshot),
            "compact_tokens": _pack(tuple(compact_tokens)),
            "target_state": target_state,
            "continuation_rows": _pack(continuation.rows),
            "finite_rows": _pack(finite.rows),
        },
    }


@dataclass(frozen=True)
class StreamingIterationPanel:
    snapshot: ForwardSnapshot
    estimates: tuple[ActionCostEstimate, ...]
    diagnostic: FiniteKDeviationDiagnostic
    completed_calls: int


def _aggregate_iteration_outputs(
    outputs: tuple[tuple[str, Mapping[str, Any]], ...],
    *,
    model_id: str,
    iteration: int,
    policy: PolicyState | RoutingPolicySeed,
    panel_fingerprint: str,
    minimum_samples: int,
) -> StreamingIterationPanel:
    token_rows = []
    continuation_rows: list[tuple[str, object]] = []
    finite_rows: list[tuple[str, object]] = []
    snapshots = []
    trace_fingerprints = []
    for _, output in outputs:
        snapshot = _unpack(output["snapshot"])
        if not isinstance(snapshot, ForwardSnapshot):
            raise QualificationCampaignError("streaming snapshot boundary is invalid")
        snapshots.append(snapshot)
        trace_fingerprints.append(output["trace_fingerprint"])
        token_rows.extend(_unpack(output["compact_tokens"]))
        state_id = output["target_state"]
        continuation_rows.extend(
            (state_id, row) for row in _unpack(output["continuation_rows"])
        )
        finite_rows.extend((state_id, row) for row in _unpack(output["finite_rows"]))

    grouped_tokens: dict[str, list[tuple[Any, ...]]] = {}
    for row in token_rows:
        grouped_tokens.setdefault(row[0], []).append(row)
    total_tokens = len(token_rows)
    states = []
    for state_id in sorted(grouped_tokens):
        rows = grouped_tokens[state_id]
        action_ids = tuple(action for action, _ in rows[0][2])
        if any(tuple(action for action, _ in row[2]) != action_ids for row in rows):
            raise QualificationCampaignError("streamed forward action support changed within state")
        predicted = tuple((action, math.fsum(dict(row[2])[action] for row in rows) / len(rows)) for action in action_ids)
        realized = tuple((action, math.fsum(dict(row[3])[action] for row in rows) / len(rows)) for action in action_ids)
        states.append(ForwardStateRow(
            state_id, rows[0][1], len(rows) / total_tokens, predicted, realized,
        ))
    policy_fingerprint = policy.fingerprint
    snapshot = ForwardSnapshot(
        model_id,
        iteration,
        policy_fingerprint,
        _digest({"panel": panel_fingerprint, "model": model_id, "policy": policy_fingerprint}),
        panel_fingerprint,
        _digest(tuple(row.mu_fingerprint for row in snapshots)),
        _digest(tuple(row.nu_fingerprint for row in snapshots)),
        _digest(tuple(row.eta_fingerprint for row in snapshots)),
        _digest(tuple(row.x_fingerprint for row in snapshots)),
        tuple(states),
        len(outputs),
        True,
        len(outputs),
        sum(row.scale_fallback_count for row in snapshots),
    )

    grouped_costs: dict[tuple[str, str], list[float]] = {}
    buckets: dict[tuple[str, str], object] = {}
    for state_id, row in continuation_rows:
        key = (state_id, row.action_bucket.fingerprint)
        grouped_costs.setdefault(key, []).append(float(row.cost))
        buckets[key] = row.action_bucket
    state_classes = {row.state_id: row.token_class for row in snapshot.states}
    estimates = []
    for key in sorted(grouped_costs):
        values = tuple(grouped_costs[key])
        if len(values) < minimum_samples:
            continue
        state_id, action_id = key
        estimates.append(ActionCostEstimate(
            state_id, action_id, state_classes[state_id],
            math.fsum(values) / len(values), 0.0, 0.0, 0.0, 0.0,
            len(values), _sample_standard_error(values),
        ))
    bound = _finite_k_simultaneous_bound(
        finite_rows, minimum_samples=minimum_samples,
    )
    diagnostic = FiniteKDeviationDiagnostic(
        True,
        len(outputs),
        max(abs(float(row.pathwise_difference)) for _, row in finite_rows),
        panel_fingerprint,
        sum(1 for _ in finite_rows) + len(outputs),
        bound.max_upper_bound,
        "complete" if bound.complete else bound.status,
        bound.normalized_max_upper_bound,
    )
    return StreamingIterationPanel(snapshot, tuple(estimates), diagnostic, 0)


class StreamingFormalQualificationRunner:
    """Official r2 runner; streams compact per-episode sufficient statistics."""

    def __init__(
        self,
        *,
        plan: QualificationPlan,
        preflight: QualificationTimingReport,
        checkpoint_root: str | Path,
        run_id: str,
        worker_count: int = 8,
    ) -> None:
        if plan.protocol_fingerprint != QualificationPlan.r2().protocol_fingerprint:
            raise QualificationCampaignError("streaming formal runner requires r2")
        plan.require_formal_start(preflight)
        self.plan = plan
        self.worker_count = worker_count
        self.checkpoint = QualificationCheckpointStore(
            checkpoint_root,
            run_id,
            plan_fingerprint=plan.protocol_fingerprint,
            source_fingerprint=qualification_source_fingerprint(),
        )

    def _panel(
        self,
        *,
        k: int,
        start_id: str,
        model_id: str,
        iteration: int,
        policy: PolicyState | RoutingPolicySeed,
    ) -> StreamingIterationPanel:
        panel = build_streaming_panel(self.plan, k=k)
        units = []
        for spec in panel.specs:
            key = (
                f"k={k}:start={start_id}:model={model_id}:iteration={iteration}:"
                f"complete-panel:{spec.scenario}:{spec.episode_index:08d}"
            )
            payload = {
                "episode_spec": spec,
                "policy": policy,
                "model_id": model_id,
                "iteration": iteration,
                "mean_field_namespace": self.plan.continuation_namespace,
                "mean_field_macro_seed": self.plan.continuation_macro_seed,
                "finite_k_namespace": self.plan.finite_k_namespace,
                "finite_k_macro_seed": self.plan.finite_k_macro_seed,
            }
            units.append(QualificationWorkUnit(
                key,
                "formal_streaming_episode",
                _digest({
                    "key": key, "spec": spec.to_dict(), "policy": policy.fingerprint,
                    "plan": self.plan.protocol_fingerprint,
                }),
                MAX_CALLS_PER_EPISODE,
                payload,
            ))
        execution = execute_canonical_work_units(
            tuple(units),
            checkpoint=self.checkpoint,
            worker=streaming_iteration_episode_worker,
            max_workers=self.worker_count,
            content_addressed=True,
        )
        result = _aggregate_iteration_outputs(
            execution.outputs,
            model_id=model_id,
            iteration=iteration,
            policy=policy,
            panel_fingerprint=panel.fingerprint,
            minimum_samples=self.plan.min_complete_samples,
        )
        return replace(result, completed_calls=execution.completed_calls)

    @staticmethod
    def _population_residual(left: ForwardSnapshot | None, right: ForwardSnapshot) -> float | None:
        if left is None:
            return None
        a = {row.state_id: row.occupancy_mass for row in left.states}
        b = {row.state_id: row.occupancy_mass for row in right.states}
        return math.fsum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in set(a) | set(b))

    def run_model(
        self,
        *,
        k: int,
        start_id: str,
        model_id: str,
        initial_policy: PolicyState | RoutingPolicySeed,
    ) -> FormalModelRun:
        policy: PolicyState | RoutingPolicySeed = initial_policy
        previous: ForwardSnapshot | None = None
        final_snapshot: ForwardSnapshot | None = None
        final_diagnostic: FiniteKDeviationDiagnostic | None = None
        total_calls = 0
        status = "not_converged_6"
        iterations = 0
        seen: set[str] = set()
        policy_after: PolicyState | None = None
        for iteration in range(self.plan.max_iterations):
            panel = self._panel(
                k=k, start_id=start_id, model_id=model_id,
                iteration=iteration, policy=policy,
            )
            total_calls += panel.completed_calls
            snapshot = panel.snapshot
            if isinstance(policy, RoutingPolicySeed):
                policy = materialize_policy(snapshot)
                snapshot = replace(snapshot, policy_fingerprint=policy.fingerprint)
            expected = {
                (state.state_id, action) for state in snapshot.states
                for action, _ in state.predicted_share
            }
            actual = {(row.state_id, row.action_id) for row in panel.estimates}
            if actual != expected or panel.diagnostic.simultaneous_ucb_status != "complete":
                status = "statistics_insufficient"
                final_snapshot, final_diagnostic = snapshot, panel.diagnostic
                policy_after = policy
                iterations = iteration + 1
                break
            model = MFGCostModel(
                model_id,
                price_per_queue_unit=FormalQualificationBackend.price_for_model(model_id),
            )
            costs: dict[str, dict[str, float]] = {}
            for estimate in panel.estimates:
                costs.setdefault(estimate.state_id, {})[estimate.action_id] = model.score(estimate)
            policy_after = policy.damped(policy.logit_response(costs))
            policy_residual = policy.residual(policy_after)
            population_residual = self._population_residual(previous, snapshot)
            iterations = iteration + 1
            final_snapshot, final_diagnostic = snapshot, panel.diagnostic
            if (
                policy_residual <= self.plan.residual_tolerance
                and snapshot.share_residual <= self.plan.residual_tolerance
                and population_residual is not None
                and population_residual <= self.plan.residual_tolerance
            ):
                status = "fixed_point"
                break
            composite = _digest((policy_after.fingerprint, snapshot.environment_fingerprint, snapshot.share_residual))
            if composite in seen:
                status = "cycle"
                break
            seen.add(composite)
            previous = snapshot
            policy = policy_after

        if policy_after is None or final_snapshot is None or final_diagnostic is None:
            raise QualificationCampaignError("streaming model produced no complete iteration")
        confirmation = "not_run"
        if status == "fixed_point":
            confirmed = self._panel(
                k=k, start_id=start_id, model_id=model_id,
                iteration=iterations, policy=policy_after,
            )
            total_calls += confirmed.completed_calls
            expected = {
                (state.state_id, action) for state in confirmed.snapshot.states
                for action, _ in state.predicted_share
            }
            actual = {(row.state_id, row.action_id) for row in confirmed.estimates}
            if actual != expected or confirmed.diagnostic.simultaneous_ucb_status != "complete":
                status, confirmation = "statistics_insufficient", "statistics_insufficient"
            else:
                model = MFGCostModel(
                    model_id,
                    price_per_queue_unit=FormalQualificationBackend.price_for_model(model_id),
                )
                costs: dict[str, dict[str, float]] = {}
                for estimate in confirmed.estimates:
                    costs.setdefault(estimate.state_id, {})[estimate.action_id] = model.score(estimate)
                response = policy_after.logit_response(costs)
                response_residual = max(
                    math.fsum(abs(old - new) for old, new in zip(row.probabilities, response[row.state_id]))
                    for row in policy_after.rows
                )
                population_residual = self._population_residual(final_snapshot, confirmed.snapshot)
                if (
                    response_residual <= self.plan.residual_tolerance
                    and confirmed.snapshot.share_residual <= self.plan.residual_tolerance
                    and population_residual is not None
                    and population_residual <= self.plan.residual_tolerance
                ):
                    confirmation = "confirmed"
                else:
                    confirmation = "not_converged"
                final_snapshot, final_diagnostic = confirmed.snapshot, confirmed.diagnostic
        return FormalModelRun(
            k, start_id, model_id, status, iterations, policy_after.fingerprint,
            confirmation, total_calls,
            final_diagnostic.simultaneous_ucb,
            final_diagnostic.simultaneous_ucb_status,
            self._population_residual(previous, final_snapshot),
            final_snapshot.share_residual,
            final_diagnostic.normalized_simultaneous_ucb,
            policy_after, final_snapshot, final_snapshot.scale_fallback_count,
        )

    def run_qualification(self) -> tuple[str, tuple[FormalModelRun, ...]]:
        rows: list[FormalModelRun] = []
        k8: list[FormalModelRun] = []
        for start_id in self.plan.starts:
            for model_id in self.plan.models:
                row = self.run_model(
                    k=8, start_id=start_id, model_id=model_id,
                    initial_policy=FormalQualificationBackend.initial_seed(start_id),
                )
                rows.append(row)
                k8.append(row)
                if row.status == "physical_failed":
                    return "physical_failed", tuple(rows)
                if row.status == "statistics_insufficient":
                    return "statistics_insufficient", tuple(rows)
        for model_id in self.plan.models:
            model_rows = tuple(row for row in k8 if row.model_id == model_id)
            if not all(row.status == "fixed_point" and row.confirmation_status == "confirmed" for row in model_rows):
                return "no_candidate", tuple(rows)
            if len({row.policy_fingerprint for row in model_rows}) != 1:
                return "multiple_candidates", tuple(rows)
        previous = {model: next(row for row in k8 if row.model_id == model) for model in self.plan.models}
        ucb = {model: [max(float(row.normalized_finite_k_ucb) for row in k8 if row.model_id == model)] for model in self.plan.models}
        for k in self.plan.k_values[1:]:
            for model_id in self.plan.models:
                prior = previous[model_id]
                row = self.run_model(
                    k=k, start_id="scale_continuation", model_id=model_id,
                    initial_policy=FormalQualificationBackend.scale_seed(prior.final_snapshot, prior.final_policy),
                )
                rows.append(row)
                if row.status != "fixed_point" or row.confirmation_status != "confirmed":
                    return ("statistics_insufficient" if row.status == "statistics_insufficient" else "not_converged_6"), tuple(rows)
                ucb[model_id].append(float(row.normalized_finite_k_ucb))
                if k == 64 and FormalQualificationBackend.policy_distance(
                    prior.final_snapshot, prior.final_policy, row.final_snapshot, row.final_policy,
                ) > 0.05:
                    return "no_candidate", tuple(rows)
                previous[model_id] = row
        for values in ucb.values():
            if values[-1] > 0.02 or any(b > a + 1e-12 for a, b in zip(values, values[1:])):
                return "no_candidate", tuple(rows)
        return "qualification_pass", tuple(rows)


__all__ = [
    "MAX_CALLS_PER_EPISODE",
    "StreamingFormalQualificationRunner",
    "StreamingIterationPanel",
    "streaming_iteration_episode_worker",
]
