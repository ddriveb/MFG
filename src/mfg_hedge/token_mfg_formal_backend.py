"""Formal r2 backend contract for dynamic qualification orchestration.

This module supplies the K-scale, two-model inputs to the dynamic checkpoint
machine.  It does not launch a campaign; dispatch remains owned by the
qualification runner after the timing and governance gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .population_estimator import BIN_SCHEMA_V1
from .token_mfg_qualification import FixedEpisodeLibrary
from .token_mfg_qualification_backend import ConcreteQualificationBackend
from .token_mfg_qualification_backend import RoutingPolicySeed
from .token_mfg_qualification_campaign import (
    MODEL_IDS,
    START_IDS,
    QualificationCampaignError,
    QualificationPlan,
    QualificationTimingReport,
    QualificationWorkUnit,
    execute_canonical_work_units,
    qualification_source_fingerprint,
)
from .token_mfg_response_solver import (
    FinalPolicyConfirmation,
    ForwardSnapshot,
    MFGSolverConfig,
    PolicyState,
    _population_residual,
)
from .qualification_checkpoint import QualificationCheckpointStore
from .token_mfg_qualification_state_machine import _pack, _stage_worker, _unpack


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class FormalQualificationPhase:
    """One dynamically materialized phase; never a pre-expanded work graph."""

    key: str
    stage: str
    topology_k: int
    start_id: str
    model_id: str
    iteration: int
    expected_calls: int
    input_fingerprint: str


class FormalQualificationBackend(ConcreteQualificationBackend):
    """K-scale backend used by the formal dynamic runner."""

    def __init__(
        self,
        library: FixedEpisodeLibrary,
        *,
        plan: QualificationPlan,
    ) -> None:
        if not isinstance(plan, QualificationPlan):
            raise TypeError("plan must be QualificationPlan")
        if library.traces[0].topology.K not in plan.k_values:
            raise QualificationCampaignError("formal library K is outside the frozen scale")
        if len(library.traces) != plan.forward_episodes:
            raise QualificationCampaignError(
                "formal library must contain exactly the frozen forward panel"
            )
        super().__init__(
            library,
            schema=BIN_SCHEMA_V1,
            mean_field_namespace=plan.continuation_namespace,
            mean_field_macro_seed=plan.continuation_macro_seed,
            finite_k_namespace=plan.finite_k_namespace,
            finite_k_macro_seed=plan.finite_k_macro_seed,
            bounded=False,
            min_complete_panels=plan.min_complete_samples,
        )
        self.plan = plan
        self.source_fingerprint = self.source_fingerprint

    @staticmethod
    def price_for_model(model_id: str) -> float:
        if model_id == "unpriced_mfg":
            return 0.0
        if model_id == "priced_mfg":
            return 0.25
        raise QualificationCampaignError("unknown formal model")

    @staticmethod
    def initial_seed(start_id: str) -> RoutingPolicySeed:
        if start_id not in START_IDS:
            raise QualificationCampaignError("unknown frozen K=8 start")
        return RoutingPolicySeed(start_id)

    @staticmethod
    def policy_marginals(
        snapshot: ForwardSnapshot, policy: PolicyState,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        policy_rows = {row.state_id: row for row in policy.rows}
        output = []
        for token_class in ("Regular", "Urgent"):
            weights: dict[str, float] = {}
            mass = 0.0
            for state in snapshot.states:
                if state.token_class != token_class:
                    continue
                row = policy_rows.get(state.state_id)
                if row is None:
                    raise QualificationCampaignError(
                        "confirmed policy does not cover its forward snapshot"
                    )
                mass += state.occupancy_mass
                for action_id, probability in zip(
                    row.action_ids, row.probabilities,
                ):
                    weights[action_id] = weights.get(action_id, 0.0) + (
                        state.occupancy_mass * probability
                    )
            if mass <= 0.0 or not weights:
                raise QualificationCampaignError(
                    "confirmed policy lacks one Token-class marginal"
                )
            total = math.fsum(weights.values())
            output.append((token_class, tuple(
                (action_id, weights[action_id] / total)
                for action_id in sorted(weights)
            )))
        return tuple(output)

    @classmethod
    def scale_seed(
        cls, snapshot: ForwardSnapshot, policy: PolicyState,
    ) -> RoutingPolicySeed:
        return RoutingPolicySeed(
            "scale_marginal_projection", cls.policy_marginals(snapshot, policy),
        )

    @classmethod
    def policy_distance(
        cls,
        left_snapshot: ForwardSnapshot,
        left_policy: PolicyState,
        right_snapshot: ForwardSnapshot,
        right_policy: PolicyState,
    ) -> float:
        left = {
            token_class: dict(rows)
            for token_class, rows in cls.policy_marginals(left_snapshot, left_policy)
        }
        right = {
            token_class: dict(rows)
            for token_class, rows in cls.policy_marginals(right_snapshot, right_policy)
        }
        return max(
            math.fsum(
                abs(left[token_class].get(action, 0.0) - right[token_class].get(action, 0.0))
                for action in set(left[token_class]) | set(right[token_class])
            )
            for token_class in ("Regular", "Urgent")
        )

    def phase(
        self,
        *,
        start_id: str,
        model_id: str,
        iteration: int,
        stage: str,
        expected_calls: int,
        input_payload: Mapping[str, Any],
    ) -> FormalQualificationPhase:
        if start_id not in (*START_IDS, "scale_continuation"):
            raise QualificationCampaignError("invalid formal start")
        if model_id not in MODEL_IDS:
            raise QualificationCampaignError("invalid formal model")
        if type(iteration) is not int or not 0 <= iteration <= self.plan.max_iterations:
            raise QualificationCampaignError("formal iteration is outside the frozen range")
        if stage not in {
            "forward_population", "mean_field_continuation",
            "finite_k_deviation", "final_policy_confirmation",
        }:
            raise QualificationCampaignError("invalid formal phase")
        if type(expected_calls) is not int or expected_calls < 1:
            raise QualificationCampaignError("formal phase calls must be positive")
        key = (
            f"k={self.library.traces[0].topology.K}/start={start_id}"
            f"/model={model_id}/iteration={iteration}/phase={stage}"
        )
        return FormalQualificationPhase(
            key, stage, self.library.traces[0].topology.K, start_id,
            model_id, iteration, expected_calls, _digest({
                "key": key,
                "library": self.library.fingerprint,
                "payload": {
                    name: (
                        value.fingerprint
                        if hasattr(value, "fingerprint")
                        else value
                        if isinstance(value, (str, int, float, bool, type(None)))
                        else repr(value)
                    )
                    for name, value in sorted(input_payload.items())
                    if name != "library"
                },
                "expected_calls": expected_calls,
            }),
        )

    def validate_panel_counts(self) -> tuple[int, int, int]:
        forward, continuation, finite = self._panel_call_counts()
        if forward != self.plan.forward_episodes:
            raise QualificationCampaignError("formal forward panel count is not 32")
        upper = self.plan.forward_episodes * self.plan.max_actions_per_target
        if continuation > upper or finite > upper:
            raise QualificationCampaignError("formal panel exceeds 32*9 calls")
        return forward, continuation, finite


@dataclass(frozen=True)
class FormalModelRun:
    """One model/start run; it is diagnostic data, not a candidate claim."""

    topology_k: int
    start_id: str
    model_id: str
    status: str
    iterations: int
    policy_fingerprint: str
    confirmation_status: str
    scheduler_calls: int
    finite_k_ucb: float | None = None
    finite_k_ucb_status: str = "not_computed"
    population_residual: float | None = None
    share_residual: float | None = None
    normalized_finite_k_ucb: float | None = None
    final_policy: PolicyState | None = None
    final_snapshot: ForwardSnapshot | None = None
    scale_fallback_count: int = 0


class FormalQualificationRunner:
    """Dynamic, checkpointed formal runner with no pre-expanded work graph."""

    def __init__(
        self,
        *,
        plan: QualificationPlan,
        preflight: QualificationTimingReport,
        checkpoint_root: str,
        run_id: str,
        worker_count: int = 8,
    ) -> None:
        if plan.protocol_fingerprint != QualificationPlan.r2().protocol_fingerprint:
            raise QualificationCampaignError("formal runner requires accepted r2 plan")
        if type(worker_count) is not int or not 1 <= worker_count <= 8:
            raise ValueError("worker_count must be a true int in 1..8")
        plan.require_formal_start(preflight)
        self.plan = plan
        self.worker_count = worker_count
        self.checkpoint = QualificationCheckpointStore(
            checkpoint_root,
            run_id,
            plan_fingerprint=plan.protocol_fingerprint,
            source_fingerprint=qualification_source_fingerprint(),
        )
        self.dispatched_calls = 0
        self.reused_calls = 0

    @staticmethod
    def _model_payload(backend: FormalQualificationBackend, model_id: str) -> dict[str, Any]:
        return {
            "mean_field_namespace": backend.mean_field_namespace,
            "mean_field_macro_seed": backend.mean_field_macro_seed,
            "finite_k_namespace": backend.finite_k_namespace,
            "finite_k_macro_seed": backend.finite_k_macro_seed,
            "bounded": False,
            "min_complete_panels": backend.min_complete_panels,
            "price_per_queue_unit": backend.price_for_model(model_id),
        }

    def _phase(
        self,
        backend: FormalQualificationBackend,
        *,
        start_id: str,
        model_id: str,
        iteration: int,
        stage: str,
        expected_calls: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        phase = backend.phase(
            start_id=start_id, model_id=model_id, iteration=iteration,
            stage=stage, expected_calls=expected_calls,
            input_payload=payload,
        )
        unit = QualificationWorkUnit(
            key=phase.key,
            kind=stage,
            input_fingerprint=phase.input_fingerprint,
            expected_calls=expected_calls,
            payload=dict(payload),
        )
        previous = self.checkpoint.completed_output(
            phase.key, input_fingerprint=phase.input_fingerprint,
        )
        if previous is not None:
            self.reused_calls += 1
            return previous
        if self.checkpoint.reserved_calls + expected_calls > self.plan.qualification_calls:
            raise QualificationCampaignError(
                "formal phase would exceed the frozen qualification call ceiling"
            )
        execution = execute_canonical_work_units(
            (unit,),
            checkpoint=self.checkpoint,
            worker=_stage_worker,
            max_workers=self.worker_count,
        )
        self.dispatched_calls += execution.dispatched_calls
        if len(execution.outputs) != 1:
            raise QualificationCampaignError("formal phase did not commit exactly one output")
        return execution.outputs[0][1]

    def run_qualification(
        self,
        *,
        libraries: Mapping[int, FixedEpisodeLibrary],
    ) -> tuple[str, tuple[FormalModelRun, ...]]:
        """Run the frozen six-run order; holdout is deliberately unreachable.

        Starts and scale handoffs are derived inside committed boundaries.  No
        caller-supplied scientific policy state is accepted.
        """

        if set(libraries) != set(self.plan.k_values):
            raise QualificationCampaignError("qualification libraries must cover every K")
        rows: list[FormalModelRun] = []
        k8_rows: list[FormalModelRun] = []
        for start_id in self.plan.starts:
            for model_id in self.plan.models:
                backend = FormalQualificationBackend(libraries[8], plan=self.plan)
                result = self.run_model(
                    backend,
                    start_id=start_id,
                    model_id=model_id,
                    initial_policy=backend.initial_seed(start_id),
                )
                rows.append(result)
                k8_rows.append(result)
        for model_id in self.plan.models:
            model_rows = tuple(row for row in k8_rows if row.model_id == model_id)
            if any(row.status == "physical_failed" for row in model_rows):
                return "physical_failed", tuple(rows)
            if any(
                row.status == "statistics_insufficient"
                or row.confirmation_status == "statistics_insufficient"
                for row in model_rows
            ):
                return "statistics_insufficient", tuple(rows)
            if not all(
                row.status == "fixed_point" and row.confirmation_status == "confirmed"
                for row in model_rows
            ):
                return "no_candidate", tuple(rows)
            candidates = tuple(row.policy_fingerprint for row in model_rows)
            if len(set(candidates)) != 1:
                return "multiple_candidates", tuple(rows)

        previous: dict[str, FormalModelRun] = {}
        scale_ucb: dict[str, list[float]] = {}
        for model_id in self.plan.models:
            model_rows = tuple(row for row in k8_rows if row.model_id == model_id)
            chosen = model_rows[0]
            if chosen.final_policy is None or chosen.final_snapshot is None:
                raise QualificationCampaignError("K=8 confirmation omitted its policy state")
            previous[model_id] = chosen
            values = tuple(row.normalized_finite_k_ucb for row in model_rows)
            if any(value is None for value in values):
                return "statistics_insufficient", tuple(rows)
            scale_ucb[model_id] = [max(float(value) for value in values)]
        for k in self.plan.k_values[1:]:
            for model_id in self.plan.models:
                backend = FormalQualificationBackend(libraries[k], plan=self.plan)
                prior = previous[model_id]
                if prior.final_policy is None or prior.final_snapshot is None:
                    raise QualificationCampaignError("prior K omitted its confirmed policy")
                result = self.run_model(
                    backend,
                    start_id="scale_continuation",
                    model_id=model_id,
                    initial_policy=backend.scale_seed(
                        prior.final_snapshot, prior.final_policy,
                    ),
                )
                rows.append(result)
                if result.status == "physical_failed":
                    return "physical_failed", tuple(rows)
                if result.status == "statistics_insufficient":
                    return "statistics_insufficient", tuple(rows)
                if result.status != "fixed_point" or result.confirmation_status != "confirmed":
                    return "not_converged_6", tuple(rows)
                if result.normalized_finite_k_ucb is None:
                    return "statistics_insufficient", tuple(rows)
                scale_ucb[model_id].append(result.normalized_finite_k_ucb)
                if k == 64:
                    if prior.topology_k != 32:
                        raise QualificationCampaignError("K=64 did not follow K=32")
                    if result.final_policy is None or result.final_snapshot is None:
                        raise QualificationCampaignError("K=64 omitted its confirmed policy")
                    distance = backend.policy_distance(
                        prior.final_snapshot, prior.final_policy,
                        result.final_snapshot, result.final_policy,
                    )
                    if distance > 0.05:
                        return "no_candidate", tuple(rows)
                previous[model_id] = result
        for model_id, values in scale_ucb.items():
            if values[-1] > 0.02 or any(
                later > earlier + 1e-12
                for earlier, later in zip(values, values[1:])
            ):
                return "no_candidate", tuple(rows)
        return "qualification_pass", tuple(rows)

    @staticmethod
    def require_holdout_gate(qualification_status: str) -> None:
        if qualification_status != "qualification_pass":
            raise QualificationCampaignError(
                "holdout requires a confirmed K=64 qualification pass"
            )

    def run_model(
        self,
        backend: FormalQualificationBackend,
        *,
        start_id: str,
        model_id: str,
        initial_policy: PolicyState | RoutingPolicySeed,
    ) -> FormalModelRun:
        if backend.library.traces[0].topology.K not in self.plan.k_values:
            raise QualificationCampaignError("formal model K is outside the frozen scale")
        policy: PolicyState | RoutingPolicySeed = initial_policy
        previous_snapshot: ForwardSnapshot | None = None
        total_calls = 0
        seen: set[str] = set()
        confirmation_status = "not_run"
        terminal_status = "not_converged_6"
        iterations = 0
        for iteration in range(self.plan.max_iterations):
            forward_calls = self.plan.forward_episodes
            model_payload = self._model_payload(backend, model_id)
            seed_payload = (
                {"initial_seed": _pack(policy)}
                if isinstance(policy, RoutingPolicySeed)
                else {"initial_policy": _pack(policy)}
            )
            forward_output = self._phase(
                backend, start_id=start_id, model_id=model_id,
                iteration=iteration, stage="forward_population",
                expected_calls=forward_calls,
                payload={
                    "stage": "forward_population", "library": backend.library,
                    "model_id": model_id, "iteration": iteration,
                    **seed_payload,
                    **model_payload,
                },
            )
            snapshot = _unpack(forward_output["snapshot"])
            episodes = _unpack(forward_output["episodes"])
            if not isinstance(snapshot, ForwardSnapshot) or not isinstance(episodes, tuple):
                raise QualificationCampaignError("formal forward boundary is invalid")
            backend._episodes_by_environment[snapshot.environment_fingerprint] = episodes
            _, continuation_calls, finite_calls = backend.validate_panel_counts()
            policy = _unpack(forward_output["policy_seed"]) if iteration == 0 else policy
            if not isinstance(policy, PolicyState):
                raise QualificationCampaignError("formal policy boundary is invalid")
            continuation_output = self._phase(
                backend, start_id=start_id, model_id=model_id,
                iteration=iteration, stage="mean_field_continuation",
                expected_calls=continuation_calls,
                payload={
                    "stage": "mean_field_continuation", "library": backend.library,
                    "snapshot": _pack(snapshot), "episodes": _pack(episodes),
                    "policy": _pack(policy), "model_id": model_id,
                    "iteration": iteration, **model_payload,
                },
            )
            estimates = _unpack(continuation_output["estimates"])
            finite_output = self._phase(
                backend, start_id=start_id, model_id=model_id,
                iteration=iteration, stage="finite_k_deviation",
                expected_calls=finite_calls,
                payload={
                    "stage": "finite_k_deviation", "library": backend.library,
                    "snapshot": _pack(snapshot), "episodes": _pack(episodes),
                    "policy": _pack(policy), "estimates": _pack(estimates),
                    "model_id": model_id, "iteration": iteration,
                    **model_payload,
                },
            )
            policy_after = _unpack(finite_output["policy_after"])
            diagnostic = _unpack(finite_output["diagnostic"])
            if not isinstance(policy_after, PolicyState):
                raise QualificationCampaignError("formal policy update boundary is invalid")
            if not hasattr(diagnostic, "simultaneous_ucb_status"):
                raise QualificationCampaignError("finite-K boundary lacks simultaneous UCB")
            if diagnostic.simultaneous_ucb_status == "statistics_insufficient":
                terminal_status = "statistics_insufficient"
            policy_residual = policy.residual(policy_after)
            population_residual = _population_residual(previous_snapshot, snapshot)
            total_calls += forward_calls + continuation_calls + finite_calls
            iterations = iteration + 1
            composite = _digest((policy_after.fingerprint, snapshot.environment_fingerprint, snapshot.share_residual))
            converged = (
                policy_residual <= self.plan.residual_tolerance
                and snapshot.share_residual <= self.plan.residual_tolerance
                and population_residual is not None
                and population_residual <= self.plan.residual_tolerance
                and diagnostic.simultaneous_ucb_status == "complete"
            )
            if terminal_status == "statistics_insufficient":
                break
            if converged:
                terminal_status = "fixed_point"
                break
            if composite in seen:
                terminal_status = "cycle"
                break
            seen.add(composite)
            policy = policy_after
            previous_snapshot = snapshot

        if previous_snapshot is None:
            previous_snapshot = snapshot
        forward_calls, continuation_calls, finite_calls = backend.validate_panel_counts()
        confirmation_config = MFGSolverConfig(
            max_iterations=1,
            min_complete_samples=self.plan.min_complete_samples,
            forward_calls=forward_calls,
            continuation_calls=continuation_calls,
            finite_k_calls=finite_calls,
            call_budget=forward_calls + continuation_calls + finite_calls,
        )
        confirmation_output = self._phase(
            backend, start_id=start_id, model_id=model_id,
            iteration=iterations, stage="final_policy_confirmation",
            expected_calls=forward_calls + continuation_calls + finite_calls,
            payload={
                "stage": "final_policy_confirmation", "library": backend.library,
                "snapshot": _pack(snapshot), "episodes": _pack(episodes),
                "policy": _pack(policy_after),
                "previous_snapshot": _pack(previous_snapshot),
                "model_id": model_id, "iteration": iterations,
                "config": confirmation_config,
                **self._model_payload(backend, model_id),
            },
        )
        confirmation = _unpack(confirmation_output["confirmation"])
        if not isinstance(confirmation, FinalPolicyConfirmation):
            raise QualificationCampaignError("formal confirmation boundary is invalid")
        confirmation_status = confirmation.status
        if confirmation.finite_k_simultaneous_ucb_status == "statistics_insufficient":
            terminal_status = "statistics_insufficient"
        return FormalModelRun(
            backend.library.traces[0].topology.K, start_id, model_id,
            terminal_status, iterations, policy_after.fingerprint,
            confirmation_status, total_calls + confirmation.attempted_calls,
            confirmation.finite_k_simultaneous_ucb,
            confirmation.finite_k_simultaneous_ucb_status,
            confirmation.population_residual,
            confirmation.share_residual,
            confirmation.finite_k_normalized_simultaneous_ucb,
            policy_after,
            confirmation.snapshot,
            0 if confirmation.snapshot is None else confirmation.snapshot.scale_fallback_count,
        )


__all__ = [
    "FormalModelRun",
    "FormalQualificationBackend",
    "FormalQualificationPhase",
    "FormalQualificationRunner",
]
