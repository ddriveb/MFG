"""Bounded, spawn-safe streaming execution for formal qualification panels.

This module deliberately keeps exogenous episode identities in the parent
payload.  A worker materializes exactly one immutable trace, runs the existing
real qualification backend, and returns only its sufficient output.  It is an
execution layer; it does not alter the routing or statistical contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .qualification_checkpoint import QualificationCheckpointStore
from .reliability_aware_routing import SCENARIO_NAMES, generate_routing_trace
from .token_mfg_qualification import FixedEpisodeLibrary
from .token_mfg_qualification_campaign import (
    QualificationCampaignError,
    QualificationExecutionResult,
    QualificationPlan,
    QualificationWorkUnit,
    execute_canonical_work_units,
)
from .topology import Topology


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class StreamingEpisodeSpec:
    """The complete immutable identity needed to regenerate one episode."""

    k: int
    namespace: str
    macro_seed: int
    scenario: str
    episode_index: int

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k not in (8, 16, 32, 64):
            raise ValueError("streaming episode K must be one of 8, 16, 32, 64")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("streaming episode namespace must be non-empty")
        if type(self.macro_seed) is not int:
            raise ValueError("streaming episode macro_seed must be a true int")
        if self.scenario not in SCENARIO_NAMES:
            raise ValueError("streaming episode scenario is invalid")
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("streaming episode index must be a non-negative true int")

    @property
    def identity(self) -> tuple[str, int, str, int]:
        return (self.namespace, self.macro_seed, self.scenario, self.episode_index)

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "namespace": self.namespace,
            "macro_seed": self.macro_seed,
            "scenario": self.scenario,
            "episode_index": self.episode_index,
        }

    def generate_trace(self):
        return generate_routing_trace(
            self.scenario,
            self.episode_index,
            namespace=self.namespace,
            macro_seed=self.macro_seed,
            topology=Topology(self.k),
        )


@dataclass(frozen=True)
class StreamingPanel:
    specs: tuple[StreamingEpisodeSpec, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("streaming panel must not be empty")
        if tuple(spec.identity for spec in self.specs) != tuple(
            sorted(spec.identity for spec in self.specs)
        ):
            raise ValueError("streaming panel identities must be canonical")
        if len({spec.identity for spec in self.specs}) != len(self.specs):
            raise ValueError("streaming panel identities must be unique")
        expected = _digest({"specs": [spec.to_dict() for spec in self.specs]})
        if self.fingerprint != expected:
            raise ValueError("streaming panel fingerprint does not match identities")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "specs": [spec.to_dict() for spec in self.specs],
        }


def build_streaming_panel(
    plan: QualificationPlan,
    *,
    k: int,
    count: int = 32,
    namespace: str | None = None,
    macro_seed: int | None = None,
    episode_start: int = 0,
) -> StreamingPanel:
    """Build only canonical identities; no trace is generated here."""

    if not isinstance(plan, QualificationPlan):
        raise TypeError("plan must be QualificationPlan")
    if type(count) is not int or count < 1:
        raise ValueError("streaming panel count must be a positive true int")
    if type(episode_start) is not int or episode_start < 0:
        raise ValueError("episode_start must be a non-negative true int")
    actual_namespace = namespace if namespace is not None else plan.forward_namespace
    actual_seed = macro_seed if macro_seed is not None else plan.forward_macro_seed
    if count <= plan.forward_episodes:
        scenarios = plan.scenario_schedule[:count]
    elif count % len(SCENARIO_NAMES) == 0:
        per_scenario = count // len(SCENARIO_NAMES)
        scenarios = tuple(
            scenario
            for scenario in SCENARIO_NAMES
            for _ in range(per_scenario)
        )
    else:
        raise ValueError("streaming panel count must balance all scenarios")
    specs = tuple(
        StreamingEpisodeSpec(
            k=k,
            namespace=actual_namespace,
            macro_seed=actual_seed,
            scenario=scenarios[index],
            episode_index=episode_start + index,
        )
        for index in range(count)
    )
    ordered = tuple(sorted(specs, key=lambda spec: spec.identity))
    return StreamingPanel(
        ordered,
        _digest({"specs": [spec.to_dict() for spec in ordered]}),
    )


def streaming_formal_episode_worker(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Windows-spawn-safe worker that materializes at most one trace."""

    spec = payload.get("episode_spec")
    if not isinstance(spec, StreamingEpisodeSpec):
        raise QualificationCampaignError("streaming worker episode identity is invalid")
    trace = spec.generate_trace()
    library = FixedEpisodeLibrary.from_traces((trace,))
    stage_payload = dict(payload)
    stage_payload["library"] = library
    # The library is a worker-local object by construction.  It is never part
    # of a work-unit payload and never returned to the parent.
    stage_payload.pop("episode_spec", None)
    from .token_mfg_qualification_state_machine import _stage_worker

    result = _stage_worker(stage_payload)
    output = result.get("output")
    if not isinstance(output, Mapping) or "library" in output:
        raise QualificationCampaignError("streaming worker returned an invalid payload")
    # These are compact audit fields, not a second trace representation.
    output = dict(output)
    output.update({
        "episode_identity": list(spec.identity),
        "trace_fingerprint": trace.fingerprint,
        "trace_count": 1,
        "live_trace_count": 1,
    })
    result = dict(result)
    result["output"] = output
    return result


def execute_streaming_episode_panel(
    panel: StreamingPanel,
    *,
    checkpoint_root: str | Path,
    run_id: str,
    plan_fingerprint: str,
    source_fingerprint: str,
    worker_count: int,
    phase: str,
    model_id: str,
    iteration: int,
    branch_payload: Mapping[str, Any],
    expected_calls: int = 1,
    start_id: str = "stream",
) -> QualificationExecutionResult:
    """Execute a bounded panel with one canonical worker unit per episode."""

    if not isinstance(panel, StreamingPanel):
        raise TypeError("panel must be StreamingPanel")
    if phase not in {
        "forward_population", "mean_field_continuation",
        "finite_k_deviation", "final_policy_confirmation",
    }:
        raise ValueError("streaming phase is invalid")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("iteration must be a non-negative true int")
    if type(expected_calls) is not int or expected_calls < 1:
        raise ValueError("expected_calls must be a positive true int")
    if not isinstance(branch_payload, Mapping):
        raise TypeError("branch_payload must be a mapping")
    if not isinstance(start_id, str) or not start_id:
        raise ValueError("start_id must be non-empty")

    checkpoint = QualificationCheckpointStore(
        checkpoint_root,
        run_id,
        plan_fingerprint=plan_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    units = []
    for spec in panel.specs:
        payload = dict(branch_payload)
        payload.update({
            "stage": phase,
            "episode_spec": spec,
            "model_id": model_id,
            "iteration": iteration,
        })
        key = (
            f"k={spec.k}:start={start_id}:model={model_id}:iteration={iteration}:"
            f"{phase}:{spec.scenario}:{spec.episode_index:08d}"
        )
        units.append(
            QualificationWorkUnit(
                key=key,
                kind=phase,
                input_fingerprint=_digest({
                    "panel": panel.fingerprint,
                    "spec": spec.to_dict(),
                    "phase": phase,
                    "model_id": model_id,
                    "iteration": iteration,
                    "start_id": start_id,
                    "branch": dict(branch_payload),
                }),
                expected_calls=expected_calls,
                payload=payload,
            )
        )
    return execute_canonical_work_units(
        tuple(units),
        checkpoint=checkpoint,
        worker=streaming_formal_episode_worker,
        max_workers=worker_count,
        content_addressed=True,
    )


__all__ = [
    "StreamingEpisodeSpec",
    "StreamingPanel",
    "build_streaming_panel",
    "execute_streaming_episode_panel",
    "streaming_formal_episode_worker",
]
