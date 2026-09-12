"""Dynamic, checkpointed orchestration for real qualification stages.

Unlike the old static work-graph entry point, this state machine creates only
the next phase after the previous phase has committed.  The bounded path is
the first executable slice; it is intentionally limited to one K=8 model and
one response round, but uses the same real backend functions as qualification.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from .qualification_checkpoint import QualificationCheckpointError, QualificationCheckpointStore
from .population_estimator import BIN_SCHEMA_V1
from .token_mfg_qualification import FixedEpisodeLibrary
from .token_mfg_qualification_backend import (
    BoundedBackendResult,
    ConcreteQualificationBackend,
    RoutingPolicySeed,
    materialize_policy,
)
from .token_mfg_qualification_campaign import (
    QualificationCampaignError,
    QualificationPlan,
    QualificationWorkUnit,
    execute_canonical_work_units,
    qualification_source_fingerprint,
)
from .token_mfg_response_solver import (
    ActionCostEstimate,
    FinalPolicyConfirmation,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    MFGCostModel,
    MFGSolverConfig,
    PolicyState,
    confirm_final_policy,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


def _pack(value: object) -> str:
    return base64.b85encode(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")


def _unpack(value: object) -> object:
    if not isinstance(value, str) or not value:
        raise QualificationCampaignError("checkpoint payload is not serialized")
    try:
        return pickle.loads(base64.b85decode(value.encode("ascii")))
    except Exception as error:
        raise QualificationCampaignError("checkpoint payload cannot be decoded") from error


def _stage_worker(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Top-level Windows-spawn worker for one dynamically selected phase."""

    backend = ConcreteQualificationBackend(
        payload["library"],
        schema=BIN_SCHEMA_V1,
        mean_field_namespace=payload["mean_field_namespace"],
        mean_field_macro_seed=payload["mean_field_macro_seed"],
        finite_k_namespace=payload["finite_k_namespace"],
        finite_k_macro_seed=payload["finite_k_macro_seed"],
        bounded=payload.get("bounded", True),
        min_complete_panels=payload.get("min_complete_panels", 2),
    )
    stage = payload["stage"]
    model_id = payload["model_id"]
    iteration = payload["iteration"]
    if stage == "forward_population":
        initial_policy = payload.get("initial_policy")
        initial_seed = payload.get("initial_seed")
        if initial_policy is not None and initial_seed is not None:
            raise QualificationCampaignError("forward input has two policy seeds")
        if initial_seed is not None:
            policy_seed = _unpack(initial_seed)
            if not isinstance(policy_seed, RoutingPolicySeed):
                raise QualificationCampaignError("forward routing seed is invalid")
        elif initial_policy is None:
            policy_seed = None
        else:
            policy_seed = _unpack(initial_policy)
            if not isinstance(policy_seed, PolicyState):
                raise QualificationCampaignError("forward initial policy is invalid")
        snapshot = backend._forward(policy_seed, model_id, iteration)
        episodes = backend._episodes_by_environment[snapshot.environment_fingerprint]
        if not isinstance(policy_seed, PolicyState):
            policy_seed = materialize_policy(snapshot)
        return {
            "completed_calls": len(episodes),
            "output": {
                "snapshot": _pack(snapshot),
                "episodes": _pack(episodes),
                "trace_library_fingerprint": backend.library.fingerprint,
                "policy_seed": _pack(policy_seed),
            },
        }

    snapshot = _unpack(payload["snapshot"])
    episodes = _unpack(payload["episodes"])
    policy = _unpack(payload["policy"])
    if not isinstance(snapshot, ForwardSnapshot):
        raise QualificationCampaignError("dynamic worker snapshot is invalid")
    if not isinstance(episodes, tuple):
        raise QualificationCampaignError("dynamic worker episodes are invalid")
    if not isinstance(policy, PolicyState):
        raise QualificationCampaignError("dynamic worker policy is invalid")
    backend._episodes_by_environment[snapshot.environment_fingerprint] = episodes

    if stage == "mean_field_continuation":
        estimates = backend._continuation(snapshot, policy, model_id, iteration)
        expected = backend._panel_call_counts()[1]
        return {
            "completed_calls": expected,
            "output": {
                "estimates": _pack(estimates),
                "policy_before": _pack(policy),
                "episode_keys": tuple(episode.identity.key for episode in episodes),
                "trace_library_fingerprint": backend.library.fingerprint,
            },
        }
    if stage == "finite_k_deviation":
        diagnostic = backend._finite_k(snapshot, policy, model_id, iteration)
        estimates = _unpack(payload["estimates"])
        if not isinstance(estimates, tuple) or any(
            not isinstance(row, ActionCostEstimate) for row in estimates
        ):
            raise QualificationCampaignError("finite-K policy update input is invalid")
        model = MFGCostModel(
            model_id,
            price_per_queue_unit=payload.get("price_per_queue_unit", 0.0),
        )
        costs: dict[str, dict[str, float]] = {}
        for estimate in estimates:
            costs.setdefault(estimate.state_id, {})[estimate.action_id] = model.score(estimate)
        policy_after = policy.damped(policy.logit_response(costs))
        return {
            "completed_calls": diagnostic.scheduler_calls,
            "output": {
                "diagnostic": _pack(diagnostic),
                "policy_before": _pack(policy),
                # Policy update is committed atomically with the finite-K
                # boundary.  It is therefore derived only from the committed
                # continuation output and becomes the sole input to confirm.
                "policy_after": _pack(policy_after),
                "episode_keys": tuple(episode.identity.key for episode in episodes),
                "trace_library_fingerprint": backend.library.fingerprint,
            },
        }
    if stage == "final_policy_confirmation":
        previous_snapshot = _unpack(payload["previous_snapshot"])
        if not isinstance(previous_snapshot, ForwardSnapshot):
            raise QualificationCampaignError("dynamic confirmation snapshot is invalid")
        config = payload["config"]
        if not isinstance(config, MFGSolverConfig):
            raise QualificationCampaignError("dynamic confirmation config is invalid")

        def forward_provider(current_policy, current_model, current_iteration):
            return backend._forward(current_policy, current_model, current_iteration)

        def continuation_provider(current_snapshot, current_policy, current_model, current_iteration):
            return backend._continuation(
                current_snapshot, current_policy, current_model, current_iteration,
            )

        def deviation_provider(current_snapshot, current_policy, current_model, current_iteration):
            return backend._finite_k(
                current_snapshot, current_policy, current_model, current_iteration,
            )

        confirmation = confirm_final_policy(
            policy,
            MFGCostModel(
                model_id,
                price_per_queue_unit=payload.get("price_per_queue_unit", 0.0),
            ),
            forward_provider,
            continuation_provider,
            deviation_provider,
            previous_snapshot=previous_snapshot,
            config=config,
        )
        return {
            "completed_calls": confirmation.attempted_calls,
            "output": {
                "confirmation": _pack(confirmation),
                "policy": _pack(policy),
                "episode_keys": tuple(episode.identity.key for episode in backend._episodes_by_environment[next(iter(backend._episodes_by_environment))]),
                "trace_library_fingerprint": backend.library.fingerprint,
            },
        }
    raise QualificationCampaignError(f"unsupported dynamic stage: {stage}")


@dataclass(frozen=True)
class DynamicQualificationResult:
    status: str
    run_id: str
    trace_library_fingerprint: str
    scheduler_calls: int
    dispatched_calls: int
    reused_calls: int
    boundaries: int
    committed_boundaries: int
    policy_fingerprint: str
    confirmation_status: str
    checkpoint_fingerprint: str

    def to_bytes(self) -> bytes:
        # Execution metadata such as worker count and resume path is excluded
        # so serial, parallel, and interrupted runs compare scientifically.
        return json.dumps(
            {
                "status": self.status,
                "trace_library_fingerprint": self.trace_library_fingerprint,
                "scheduler_calls": self.scheduler_calls,
                "boundaries": self.boundaries,
                "committed_boundaries": self.committed_boundaries,
                "policy_fingerprint": self.policy_fingerprint,
                "confirmation_status": self.confirmation_status,
            }, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")


class DynamicQualificationStateMachine:
    """Parent-owned bounded phase machine with atomic checkpoint boundaries."""

    _MODEL = "unpriced_mfg"
    _ITERATION = 0
    _BOUNDARY_PHASES = (
        "forward_population",
        "mean_field_continuation",
        "finite_k_deviation",
        "final_policy_confirmation",
    )

    def __init__(
        self,
        backend: ConcreteQualificationBackend,
        *,
        checkpoint_root: str | Path,
        run_id: str,
        worker_count: int = 8,
    ) -> None:
        if not isinstance(backend, ConcreteQualificationBackend):
            raise TypeError("backend must be ConcreteQualificationBackend")
        if type(worker_count) is not int or not 1 <= worker_count <= 8:
            raise ValueError("worker_count must be a true int in 1..8")
        self.backend = backend
        self.worker_count = worker_count
        plan = QualificationPlan()
        self.checkpoint = QualificationCheckpointStore(
            checkpoint_root,
            run_id,
            plan_fingerprint=plan.protocol_fingerprint,
            source_fingerprint=qualification_source_fingerprint(),
        )
        self.run_id = run_id
        self.dispatched_calls = 0
        self.reused_calls = 0

    def _key(self, phase: str) -> str:
        return f"bounded/k8/model={self._MODEL}/iteration={self._ITERATION}/phase={phase}"

    def _protocol_payload(self) -> dict[str, Any]:
        return {
            "mean_field_namespace": self.backend.mean_field_namespace,
            "mean_field_macro_seed": self.backend.mean_field_macro_seed,
            "finite_k_namespace": self.backend.finite_k_namespace,
            "finite_k_macro_seed": self.backend.finite_k_macro_seed,
        }

    def _run_phase(
        self,
        phase: str,
        payload: Mapping[str, Any],
        expected_calls: int,
    ) -> Mapping[str, Any]:
        key = self._key(phase)
        fingerprint = _digest({
            "key": key,
            "phase": phase,
            "library": self.backend.library.fingerprint,
            "payload": {
                name: (
                    value.fingerprint
                    if hasattr(value, "fingerprint") else
                    getattr(value, "policy_fingerprint", None)
                    if hasattr(value, "policy_fingerprint") else
                    _digest(value) if isinstance(value, (str, bytes, bytearray))
                    else str(type(value))
                )
                for name, value in sorted(payload.items())
                if name != "library"
            },
            "expected_calls": expected_calls,
        })
        unit = QualificationWorkUnit(
            key=key,
            kind=phase,
            input_fingerprint=fingerprint,
            expected_calls=expected_calls,
            payload=dict(payload),
        )
        before = self.checkpoint.completed_output(
            key, input_fingerprint=fingerprint,
        )
        if before is not None:
            self.reused_calls += 1
            return before
        execution = execute_canonical_work_units(
            (unit,),
            checkpoint=self.checkpoint,
            worker=_stage_worker,
            max_workers=self.worker_count,
        )
        self.dispatched_calls += execution.dispatched_calls
        if not execution.outputs:
            raise QualificationCampaignError("dynamic phase committed no output")
        return execution.outputs[0][1]

    def run_bounded(
        self, *, stop_after_phase: str | None = None,
    ) -> DynamicQualificationResult:
        if stop_after_phase is not None and stop_after_phase not in self._BOUNDARY_PHASES:
            raise ValueError("stop_after_phase is not a bounded phase")
        forward_output = self._run_phase(
            "forward_population",
            {
                "stage": "forward_population",
                "library": self.backend.library,
                "model_id": self._MODEL,
                "iteration": self._ITERATION,
                **self._protocol_payload(),
            },
            len(self.backend.library.traces),
        )
        snapshot = _unpack(forward_output["snapshot"])
        episodes = _unpack(forward_output["episodes"])
        policy = _unpack(forward_output["policy_seed"])
        if not isinstance(snapshot, ForwardSnapshot) or not isinstance(policy, PolicyState):
            raise QualificationCampaignError("forward boundary is invalid")
        self.backend._episodes_by_environment[snapshot.environment_fingerprint] = episodes
        if stop_after_phase == "forward_population":
            return DynamicQualificationResult(
                "paused", self.run_id, self.backend.library.fingerprint,
                len(self.backend.library.traces), self.dispatched_calls,
                self.reused_calls, len(self._BOUNDARY_PHASES), 1,
                policy.fingerprint, "not_run", self.checkpoint.fingerprint,
            )

        forward_calls, continuation_calls, finite_calls = self.backend._panel_call_counts()
        continuation_output = self._run_phase(
            "mean_field_continuation",
            {
                "stage": "mean_field_continuation",
                "library": self.backend.library,
                "snapshot": _pack(snapshot),
                "episodes": _pack(episodes),
                "policy": _pack(policy),
                "model_id": self._MODEL,
                "iteration": self._ITERATION,
                **self._protocol_payload(),
            },
            continuation_calls,
        )
        estimates = _unpack(continuation_output["estimates"])
        if not isinstance(estimates, tuple):
            raise QualificationCampaignError("continuation boundary is invalid")
        if "policy_after" in continuation_output:
            raise QualificationCampaignError(
                "policy update must not be committed before finite-K"
            )
        if stop_after_phase == "mean_field_continuation":
            return DynamicQualificationResult(
                "paused", self.run_id, self.backend.library.fingerprint,
                forward_calls + continuation_calls, self.dispatched_calls,
                self.reused_calls, len(self._BOUNDARY_PHASES), 2,
                policy.fingerprint, "not_run", self.checkpoint.fingerprint,
            )

        finite_output = self._run_phase(
            "finite_k_deviation",
            {
                "stage": "finite_k_deviation",
                "library": self.backend.library,
                "snapshot": _pack(snapshot),
                "episodes": _pack(episodes),
                "policy": _pack(policy),
                "estimates": continuation_output["estimates"],
                "model_id": self._MODEL,
                "iteration": self._ITERATION,
                **self._protocol_payload(),
            },
            finite_calls,
        )
        diagnostic = _unpack(finite_output["diagnostic"])
        if not isinstance(diagnostic, FiniteKDeviationDiagnostic):
            raise QualificationCampaignError("finite-K boundary is invalid")
        policy_after = _unpack(finite_output["policy_after"])
        if not isinstance(policy_after, PolicyState):
            raise QualificationCampaignError("policy update boundary is invalid")
        if stop_after_phase == "finite_k_deviation":
            return DynamicQualificationResult(
                "paused", self.run_id, self.backend.library.fingerprint,
                forward_calls + continuation_calls + finite_calls,
                self.dispatched_calls, self.reused_calls,
                len(self._BOUNDARY_PHASES), 3,
                policy_after.fingerprint, "not_run", self.checkpoint.fingerprint,
            )

        config = MFGSolverConfig(
            max_iterations=1,
            min_complete_samples=2,
            forward_calls=forward_calls,
            continuation_calls=continuation_calls,
            finite_k_calls=finite_calls,
            call_budget=forward_calls + continuation_calls + finite_calls,
        )
        confirmation_output = self._run_phase(
            "final_policy_confirmation",
            {
                "stage": "final_policy_confirmation",
                "library": self.backend.library,
                "snapshot": _pack(snapshot),
                "episodes": _pack(episodes),
                "policy": _pack(policy_after),
                "previous_snapshot": _pack(snapshot),
                "model_id": self._MODEL,
                "iteration": self._ITERATION + 1,
                "config": config,
                **self._protocol_payload(),
            },
            forward_calls + continuation_calls + finite_calls,
        )
        confirmation = _unpack(confirmation_output["confirmation"])
        if not isinstance(confirmation, FinalPolicyConfirmation):
            raise QualificationCampaignError("confirmation boundary is invalid")
        return DynamicQualificationResult(
            "complete", self.run_id, self.backend.library.fingerprint,
            forward_calls + continuation_calls + finite_calls
            + confirmation.attempted_calls,
            self.dispatched_calls, self.reused_calls,
            len(self._BOUNDARY_PHASES), len(self._BOUNDARY_PHASES),
            policy_after.fingerprint, confirmation.status,
            self.checkpoint.fingerprint,
        )


__all__ = ["DynamicQualificationResult", "DynamicQualificationStateMachine"]
