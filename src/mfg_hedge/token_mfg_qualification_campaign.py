"""Real finite-system MFG qualification campaign contract.

This module owns the execution protocol and the real-physics preflight for
ticket 06.  It deliberately does not manufacture solver snapshots: a formal
run must be connected to the real simultaneous-routing, population, and
response layers before it can start.  The preflight uses the real routing
engine on isolated identities and never consumes formal episode identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import pickle
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .qualification_checkpoint import (
    QualificationCheckpointError,
    QualificationCheckpointStore,
)

from .reliability_aware_routing import (
    SCENARIO_NAMES,
    RoutingTrace,
    generate_routing_trace,
)
from .simultaneous_token_routing import (
    PopulationDecisionContext,
    SimultaneousRoutingResult,
    TokenDecisionObservation,
    simulate_simultaneous_routing,
)
from .token_mfg_qualification import FixedEpisodeLibrary
from .topology import QUALIFICATION_K, Topology


QUALIFICATION_CALLS = 51_072
HOLDOUT_CALLS = 8_064
PRE_FLIGHT_MAX_CALLS = 32
TOTAL_CALL_LIMIT = 59_168
QUALIFICATION_EPISODES = 32
MAX_ACTIONS_PER_TARGET = 9
MAX_ITERATIONS = 6
MODEL_IDS = ("unpriced_mfg", "priced_mfg")
START_IDS = (
    "uniform",
    "simultaneous_loew_projection",
    "corrected_risk_aware_projection",
)
QUALIFICATION_SPLIT = "qualification"
HOLDOUT_SPLIT = "holdout"
CLAIM_BOUNDARY = "finite_mfg_qualification_campaign"
QUALIFICATION_SCENARIO_SCHEDULE = tuple(
    scenario for scenario in SCENARIO_NAMES for _ in range(8)
)
MAX_K64_CALL_SECONDS = 30.0
MAX_EIGHT_WORKER_SECONDS = 12.0 * 60.0 * 60.0
TIMING_WORKERS = 8

R2_RUN_ID = "reliability-aware-token-mfg-qualification-20260911-r2"
R2_FORWARD_NAMESPACE = "reliability-aware-token-mfg:v1:r2:qualification:forward"
R2_FORWARD_MACRO_SEED = 20260921
R2_CONTINUATION_NAMESPACE = "reliability-aware-token-mfg:v1:r2:qualification:continuation"
R2_CONTINUATION_MACRO_SEED = 20260922
R2_FINITE_K_NAMESPACE = "reliability-aware-token-mfg:v1:r2:qualification:finite-k"
R2_FINITE_K_MACRO_SEED = 20260923
R2_DRY_RUN_NAMESPACE = "reliability-aware-token-mfg:v1:r2:dry-run"
R2_DRY_RUN_MACRO_SEED = 20260926


class QualificationCampaignError(ValueError):
    """Raised when the frozen campaign contract cannot be executed safely."""

    def __init__(self, message: str, *, status: str = "fail_closed") -> None:
        super().__init__(message)
        if status not in {"fail_closed", "physical_failed"}:
            raise ValueError("invalid qualification failure status")
        self.status = status


_WORK_KINDS = {
    "forward_population",
    "mean_field_continuation",
    "finite_k_deviation",
    "final_policy_confirmation",
    "holdout",
    "formal_streaming_episode",
}


@dataclass(frozen=True)
class QualificationWorkUnit:
    """One canonical, independently resumable qualification work unit.

    The payload is an immutable-by-contract, spawn-pickleable input projection.
    It must contain only the inputs needed by the real worker; endogenous
    queues, actions, completions, and scorer state are created inside it.
    """

    key: str
    kind: str
    input_fingerprint: str
    expected_calls: int
    payload: Any

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise QualificationCampaignError("work unit key must be non-empty")
        if self.kind not in _WORK_KINDS:
            raise QualificationCampaignError("work unit kind is unsupported")
        _check_sha256(self.input_fingerprint, "work unit input fingerprint")
        _true_int(self.expected_calls, "work unit expected calls", 1)
        try:
            pickle.dumps(self.payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            raise QualificationCampaignError(
                "work unit payload must be spawn-pickleable"
            ) from error


@dataclass(frozen=True)
class QualificationExecutionResult:
    """Canonical result of a checkpointed batch, independent of completion order."""

    outputs: tuple[tuple[str, Mapping[str, Any]], ...]
    reserved_calls: int
    completed_calls: int
    dispatched_calls: int
    reused_calls: int
    worker_count: int
    status: str = "complete"

    def __post_init__(self) -> None:
        if self.status not in {"complete", "physical_failed"}:
            raise QualificationCampaignError("invalid execution result status")
        keys = tuple(key for key, _ in self.outputs)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise QualificationCampaignError("execution outputs are not canonical")
        if any(not isinstance(output, Mapping) for _, output in self.outputs):
            raise QualificationCampaignError("execution output is not a mapping")
        for name in (
            "reserved_calls", "completed_calls", "dispatched_calls", "reused_calls",
            "worker_count",
        ):
            _true_int(getattr(self, name), f"execution {name}")
        if self.worker_count < 1 or self.worker_count > TIMING_WORKERS:
            raise QualificationCampaignError("execution worker count is outside 1..8")

    def to_bytes(self) -> bytes:
        # This is the scientific result serialization.  Dispatch accounting
        # and worker-count metadata are execution provenance and intentionally
        # do not make serial, parallel, or resumed outputs differ.
        return json.dumps(
            {
                "outputs": [
                    [key, dict(output)] for key, output in self.outputs
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class QualificationCampaignResult:
    """Checkpointed qualification work completion, not an MFG claim."""

    status: str
    plan_fingerprint: str
    source_fingerprint: str
    execution: QualificationExecutionResult

    def __post_init__(self) -> None:
        if self.status not in {"qualification_work_complete", "physical_failed"}:
            raise QualificationCampaignError("invalid qualification campaign status")
        _check_sha256(self.plan_fingerprint, "campaign plan fingerprint")
        _check_sha256(self.source_fingerprint, "campaign source fingerprint")
        if not isinstance(self.execution, QualificationExecutionResult):
            raise QualificationCampaignError("campaign execution result is invalid")

    def to_bytes(self) -> bytes:
        return self.execution.to_bytes()


class QualificationBackendProtocol:
    """Structural contract for the real routing/population/response adapter.

    A backend must expose source identity, build the frozen work graph, and
    provide a top-level spawn-safe worker.  The campaign module does not accept
    arbitrary provider callbacks that could silently stand in for physics.
    """

    real_physics: bool
    source_fingerprint: str

    def build_qualification_work_units(
        self, plan: "QualificationPlan",
    ) -> Sequence[QualificationWorkUnit]:
        raise NotImplementedError

    def worker(self, payload: Any) -> Mapping[str, Any]:
        raise NotImplementedError


def _check_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QualificationCampaignError(f"{name} must be a SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise QualificationCampaignError(
            f"{name} must be a SHA-256 fingerprint"
        ) from error
    return value


def _spawn_worker_entry(
    task: tuple[str, Any, str, str],
) -> tuple[str, int, Mapping[str, Any]]:
    """Top-level Windows-spawn entry point; never captures a local closure."""

    key, payload, module_name, qualname = task
    if "<locals>" in qualname:
        raise QualificationCampaignError("worker must be a top-level callable")
    module = importlib.import_module(module_name)
    worker: Any = module
    for part in qualname.split("."):
        worker = getattr(worker, part)
    if not callable(worker):
        raise QualificationCampaignError("worker entry is not callable")
    result = worker(payload)
    if not isinstance(result, Mapping):
        raise QualificationCampaignError("worker result must be a mapping")
    completed_calls = result.get("completed_calls")
    output = result.get("output")
    if type(completed_calls) is not int or completed_calls < 0:
        raise QualificationCampaignError("worker completed_calls is invalid")
    if not isinstance(output, Mapping):
        raise QualificationCampaignError("worker output must be a mapping")
    return key, completed_calls, dict(output)


def _worker_identity(worker: Callable[[Any], Mapping[str, Any]]) -> tuple[str, str]:
    if not callable(worker):
        raise QualificationCampaignError("worker must be callable")
    module_name = getattr(worker, "__module__", None)
    qualname = getattr(worker, "__qualname__", None)
    if (
        not isinstance(module_name, str)
        or not module_name
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
    ):
        raise QualificationCampaignError(
            "worker must be a top-level importable callable"
        )
    try:
        module = importlib.import_module(module_name)
        candidate: Any = module
        for part in qualname.split("."):
            candidate = getattr(candidate, part)
    except Exception as error:
        raise QualificationCampaignError(
            "worker cannot be resolved from its module and qualname"
        ) from error
    if candidate is not worker:
        raise QualificationCampaignError(
            "worker identity is not stable under import"
        )
    return module_name, qualname


_BLOB_REF_KEY = "__content_addressed_blob__"


def _checkpoint_output(
    checkpoint: QualificationCheckpointStore,
    output: Mapping[str, Any],
    *,
    content_addressed: bool,
) -> Mapping[str, Any]:
    """Materialize or persist one output without changing its science fields.

    The compact form is deliberately a one-key reference.  It is only used by
    the streaming path; the historical executor keeps its inline JSON output.
    """

    if not content_addressed:
        return output
    if not isinstance(output, Mapping):
        raise QualificationCampaignError("worker output is not a mapping")
    digest = checkpoint.put_json_blob(output)
    return {_BLOB_REF_KEY: digest}


def _load_checkpoint_output(
    checkpoint: QualificationCheckpointStore,
    output: Mapping[str, Any],
    *,
    content_addressed: bool,
) -> Mapping[str, Any]:
    if not content_addressed:
        return output
    if set(output) != {_BLOB_REF_KEY} or not isinstance(output.get(_BLOB_REF_KEY), str):
        raise QualificationCampaignError(
            "content-addressed checkpoint output is missing its blob reference"
        )
    try:
        return checkpoint.read_json_blob(output[_BLOB_REF_KEY])
    except QualificationCheckpointError as error:
        raise QualificationCampaignError(str(error)) from error


def execute_canonical_work_units(
    units: Sequence[QualificationWorkUnit],
    *,
    checkpoint: QualificationCheckpointStore,
    worker: Callable[[Any], Mapping[str, Any]],
    max_workers: int = TIMING_WORKERS,
    content_addressed: bool = False,
) -> QualificationExecutionResult:
    """Run checkpointed units with deterministic Windows-spawn semantics.

    All reservations are persisted before any process is dispatched.  Worker
    completion order is deliberately ignored; successful outputs are checked
    and committed in canonical key order.  A worker error leaves reservations
    unresolved, which makes a later invocation fail closed rather than retry.
    """

    if not isinstance(checkpoint, QualificationCheckpointStore):
        raise QualificationCampaignError("checkpoint must be a checkpoint store")
    _true_int(max_workers, "execution worker count", 1)
    if max_workers > TIMING_WORKERS:
        raise QualificationCampaignError("execution worker count exceeds eight")
    rows = tuple(units)
    if not rows or any(not isinstance(row, QualificationWorkUnit) for row in rows):
        raise QualificationCampaignError("work units must be a non-empty sequence")
    ordered = tuple(sorted(rows, key=lambda row: row.key))
    if tuple(row.key for row in ordered) != tuple(sorted(set(row.key for row in ordered))):
        raise QualificationCampaignError("work unit keys must be unique and canonical")
    module_name, qualname = _worker_identity(worker)

    if checkpoint.unresolved_keys:
        raise QualificationCampaignError(
            "checkpoint contains unresolved reserved work; automatic retry is forbidden",
            status="fail_closed",
        )

    reused: dict[str, Mapping[str, Any]] = {}
    reused_completed: dict[str, int] = {}
    pending: list[QualificationWorkUnit] = []
    for unit in ordered:
        try:
            should_dispatch = checkpoint.reserve(
                unit.key,
                input_fingerprint=unit.input_fingerprint,
                expected_calls=unit.expected_calls,
            )
        except QualificationCheckpointError as error:
            raise QualificationCampaignError(str(error)) from error
        if should_dispatch:
            pending.append(unit)
        else:
            try:
                output = checkpoint.completed_output(
                    unit.key, input_fingerprint=unit.input_fingerprint,
                )
            except QualificationCheckpointError as error:
                raise QualificationCampaignError(str(error)) from error
            if output is None:
                raise QualificationCampaignError("completed work has no output")
            reused[unit.key] = _load_checkpoint_output(
                checkpoint, output, content_addressed=content_addressed,
            )
            try:
                reused_completed[unit.key] = checkpoint.completed_call_count(
                    unit.key, input_fingerprint=unit.input_fingerprint,
                )
            except QualificationCheckpointError as error:
                raise QualificationCampaignError(str(error)) from error

    worker_outputs: dict[str, tuple[int, Mapping[str, Any]]] = {}
    if pending:
        tasks = tuple(
            (unit.key, unit.payload, module_name, qualname)
            for unit in pending
        )
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=context,
            ) as executor:
                futures = tuple(executor.submit(_spawn_worker_entry, task) for task in tasks)
                # Read in canonical task order, never in completion order.
                for unit, future in zip(pending, futures):
                    key, completed_calls, output = future.result()
                    if key != unit.key:
                        raise QualificationCampaignError(
                            "worker returned a non-canonical work key"
                        )
                    if completed_calls > unit.expected_calls:
                        raise QualificationCampaignError(
                            "worker exceeded its reserved scheduler calls"
                        )
                    worker_outputs[key] = (completed_calls, output)
        except QualificationCampaignError:
            raise
        except Exception as error:
            raise QualificationCampaignError(
                f"worker execution failed: {type(error).__name__}: {error}",
                status="physical_failed",
            ) from error

    # Commit only after every pending worker returned successfully, preserving
    # the exact same canonical order on every run and every worker count.
    for unit in pending:
        completed_calls, output = worker_outputs[unit.key]
        try:
            persisted_output = _checkpoint_output(
                checkpoint, output, content_addressed=content_addressed,
            )
            checkpoint.commit(
                unit.key,
                input_fingerprint=unit.input_fingerprint,
                completed_calls=completed_calls,
                output=persisted_output,
            )
        except QualificationCheckpointError as error:
            raise QualificationCampaignError(str(error)) from error

    outputs = tuple(
        (
            unit.key,
            reused[unit.key] if unit.key in reused else worker_outputs[unit.key][1],
        )
        for unit in ordered
    )
    reserved_calls = sum(unit.expected_calls for unit in ordered)
    completed_calls = sum(
        reused_completed[unit.key]
        if unit.key in reused
        else worker_outputs[unit.key][0]
        for unit in ordered
    )
    return QualificationExecutionResult(
        outputs=outputs,
        reserved_calls=reserved_calls,
        completed_calls=completed_calls,
        dispatched_calls=sum(unit.expected_calls for unit in pending),
        reused_calls=len(reused),
        worker_count=max_workers,
    )


def _validate_qualification_backend(
    backend: Any,
    plan: "QualificationPlan",
) -> tuple[tuple[QualificationWorkUnit, ...], Callable[[Any], Mapping[str, Any]]]:
    if backend is None:
        raise QualificationCampaignError(
            "formal qualification requires a real routing/population backend"
        )
    if getattr(backend, "real_physics", False) is not True:
        raise QualificationCampaignError(
            "synthetic qualification backends are permanently refused"
        )
    source = getattr(backend, "source_fingerprint", None)
    expected_source = qualification_source_fingerprint()
    if source != expected_source:
        raise QualificationCampaignError("qualification backend source fingerprint mismatch")
    builder = getattr(backend, "build_qualification_work_units", None)
    worker = getattr(backend, "worker", None)
    if not callable(builder) or not callable(worker):
        raise QualificationCampaignError(
            "qualification backend lacks the typed work-graph/worker interface"
        )
    try:
        units = tuple(builder(plan))
    except Exception as error:
        raise QualificationCampaignError(
            f"qualification work graph construction failed: {type(error).__name__}: {error}"
        ) from error
    if not units or any(not isinstance(unit, QualificationWorkUnit) for unit in units):
        raise QualificationCampaignError("qualification backend returned invalid work units")
    if any(unit.kind == "holdout" for unit in units):
        raise QualificationCampaignError(
            "holdout work is forbidden in the qualification work graph"
        )
    kinds = {unit.kind for unit in units}
    required = {
        "forward_population", "mean_field_continuation",
        "finite_k_deviation", "final_policy_confirmation",
    }
    if not required.issubset(kinds):
        raise QualificationCampaignError(
            "qualification work graph is missing a required work-unit kind"
        )
    if sum(unit.expected_calls for unit in units) != plan.qualification_calls:
        raise QualificationCampaignError(
            "qualification work graph call total differs from frozen 51,072 calls"
        )
    _worker_identity(worker)
    return units, worker


def run_qualification_campaign(
    plan: "QualificationPlan",
    *,
    preflight: "QualificationTimingReport",
    backend: QualificationBackendProtocol,
    checkpoint_root: str | Path,
    run_id: str,
    max_workers: int = TIMING_WORKERS,
) -> QualificationCampaignResult:
    """Run the frozen real qualification work graph with restart safety.

    This is an execution entry point only.  It never selects a candidate or
    publishes an MFG result.  The caller must provide the real adapter that
    connects simultaneous routing, population estimation, continuation,
    finite-K deviation, and final-policy confirmation.
    """

    if not isinstance(plan, QualificationPlan):
        raise QualificationCampaignError("plan must be QualificationPlan")
    plan.require_formal_start(preflight)
    units, worker = _validate_qualification_backend(backend, plan)
    source = qualification_source_fingerprint()
    try:
        checkpoint = QualificationCheckpointStore(
            checkpoint_root,
            run_id,
            plan_fingerprint=plan.protocol_fingerprint,
            source_fingerprint=source,
        )
        execution = execute_canonical_work_units(
            units,
            checkpoint=checkpoint,
            worker=worker,
            max_workers=max_workers,
        )
    except (QualificationCheckpointError, QualificationCampaignError):
        raise
    except Exception as error:
        raise QualificationCampaignError(
            f"qualification execution failed: {type(error).__name__}: {error}"
        ) from error
    return QualificationCampaignResult(
        "qualification_work_complete",
        plan.protocol_fingerprint,
        source,
        execution,
    )


def _true_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationCampaignError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationCampaignError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise QualificationCampaignError(f"{name} must be finite")
    return result


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover
        raise QualificationCampaignError("value cannot be fingerprinted") from error
    return hashlib.sha256(payload).hexdigest()


def qualification_source_fingerprint() -> str:
    """Fingerprint the complete local execution closure used by ticket 06."""

    root = Path(__file__).resolve().parent
    names = (
        "reliability_aware_routing.py",
        "simultaneous_token_routing.py",
        "population_estimator.py",
        "token_population_response.py",
        "token_mfg_response_solver.py",
        "token_mfg_qualification.py",
        "topology.py",
        "qualification_checkpoint.py",
        "token_mfg_qualification_campaign.py",
        "token_mfg_qualification_backend.py",
        "token_mfg_qualification_state_machine.py",
        "token_mfg_formal_backend.py",
        "token_mfg_streaming.py",
        "token_mfg_streaming_formal.py",
    )
    rows = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise QualificationCampaignError(f"qualification source is missing: {name}")
        rows.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return _digest(rows)


@dataclass(frozen=True)
class QualificationPlan:
    """Immutable ticket-06 execution plan.

    The defaults are the already-frozen qualification protocol.  Any change
    is rejected rather than silently creating a different experiment.
    """

    k_values: tuple[int, ...] = QUALIFICATION_K
    models: tuple[str, ...] = MODEL_IDS
    starts: tuple[str, ...] = START_IDS
    forward_episodes: int = 32
    continuation_episodes: int = 32
    finite_k_episodes: int = 32
    max_actions_per_target: int = MAX_ACTIONS_PER_TARGET
    max_iterations: int = MAX_ITERATIONS
    min_complete_samples: int = 8
    residual_tolerance: float = 0.01
    preflight_max_calls: int = PRE_FLIGHT_MAX_CALLS
    forward_namespace: str = (
        "reliability-aware-token-mfg:v1:qualification:forward"
    )
    forward_macro_seed: int = 20260916
    continuation_namespace: str = (
        "reliability-aware-token-mfg:v1:qualification:continuation"
    )
    continuation_macro_seed: int = 20260917
    finite_k_namespace: str = (
        "reliability-aware-token-mfg:v1:qualification:finite-k"
    )
    finite_k_macro_seed: int = 20260918
    holdout_namespace: str = (
        "reliability-aware-token-mfg:v1:qualification:holdout"
    )
    holdout_macro_seed: int = 20260919
    preflight_namespace: str = (
        "reliability-aware-token-mfg:v1:qualification:preflight"
    )
    preflight_macro_seed: int = 20260920
    holdout_episodes_per_k: int = 224
    holdout_arms: tuple[str, ...] = (
        "simultaneous_rr",
        "simultaneous_jsq",
        "simultaneous_loew",
        "reliability_only",
        "corrected_risk_aware",
        "lazarus",
        "unpriced_mfg",
        "priced_mfg",
        "sequential_loew_centralized",
    )
    qualification_calls: int = QUALIFICATION_CALLS
    holdout_calls: int = HOLDOUT_CALLS
    total_call_limit: int = TOTAL_CALL_LIMIT
    scenario_schedule: tuple[str, ...] = QUALIFICATION_SCENARIO_SCHEDULE
    max_k64_call_seconds: float = MAX_K64_CALL_SECONDS
    max_eight_worker_seconds: float = MAX_EIGHT_WORKER_SECONDS
    timing_workers: int = TIMING_WORKERS

    @classmethod
    def r2(cls) -> "QualificationPlan":
        """Return the fresh qualification plan after r1 namespace retirement."""

        return cls(
            forward_namespace=R2_FORWARD_NAMESPACE,
            forward_macro_seed=R2_FORWARD_MACRO_SEED,
            continuation_namespace=R2_CONTINUATION_NAMESPACE,
            continuation_macro_seed=R2_CONTINUATION_MACRO_SEED,
            finite_k_namespace=R2_FINITE_K_NAMESPACE,
            finite_k_macro_seed=R2_FINITE_K_MACRO_SEED,
        )

    def __post_init__(self) -> None:
        if self.k_values != QUALIFICATION_K:
            raise QualificationCampaignError("qualification K values are frozen")
        if self.models != MODEL_IDS:
            raise QualificationCampaignError("qualification models are frozen")
        if self.starts != START_IDS:
            raise QualificationCampaignError("qualification starts are frozen")
        for name in (
            "forward_episodes", "continuation_episodes", "finite_k_episodes",
            "max_actions_per_target", "max_iterations", "min_complete_samples",
            "preflight_max_calls", "holdout_episodes_per_k",
            "qualification_calls", "holdout_calls", "total_call_limit",
        ):
            _true_int(getattr(self, name), name, 1)
        if self.forward_episodes != 32 or self.continuation_episodes != 32:
            raise QualificationCampaignError("qualification episode counts are frozen")
        if self.finite_k_episodes != 32 or self.max_actions_per_target != 9:
            raise QualificationCampaignError("qualification panel counts are frozen")
        if self.max_iterations != 6 or self.min_complete_samples != 8:
            raise QualificationCampaignError("qualification iteration/floor is frozen")
        if self.preflight_max_calls != PRE_FLIGHT_MAX_CALLS:
            raise QualificationCampaignError("preflight call limit is frozen")
        if self.holdout_episodes_per_k != 224 or len(self.holdout_arms) != 9:
            raise QualificationCampaignError("holdout panel is frozen")
        if self.qualification_calls != QUALIFICATION_CALLS:
            raise QualificationCampaignError("qualification call budget is frozen")
        if self.holdout_calls != HOLDOUT_CALLS:
            raise QualificationCampaignError("holdout call budget is frozen")
        if self.total_call_limit != TOTAL_CALL_LIMIT:
            raise QualificationCampaignError("total call budget is frozen")
        if self.scenario_schedule != QUALIFICATION_SCENARIO_SCHEDULE:
            raise QualificationCampaignError("qualification scenario schedule is frozen")
        if len(self.scenario_schedule) != self.forward_episodes or any(
            self.scenario_schedule.count(scenario) != 8
            for scenario in SCENARIO_NAMES
        ):
            raise QualificationCampaignError("qualification scenarios must be balanced")
        if not math.isclose(self.max_k64_call_seconds, MAX_K64_CALL_SECONDS):
            raise QualificationCampaignError("K=64 timing gate is frozen")
        if not math.isclose(self.max_eight_worker_seconds, MAX_EIGHT_WORKER_SECONDS):
            raise QualificationCampaignError("qualification timing gate is frozen")
        if self.timing_workers != TIMING_WORKERS:
            raise QualificationCampaignError("qualification worker count is frozen")
        if not math.isclose(self.residual_tolerance, 0.01, rel_tol=0.0, abs_tol=1e-15):
            raise QualificationCampaignError("residual tolerance is frozen")
        namespaces = (
            self.forward_namespace, self.continuation_namespace,
            self.finite_k_namespace, self.holdout_namespace, self.preflight_namespace,
        )
        if any(not isinstance(value, str) or not value for value in namespaces):
            raise QualificationCampaignError("campaign namespace is invalid")
        if len(set(namespaces)) != len(namespaces):
            raise QualificationCampaignError("campaign namespaces must be disjoint")
        for name in (
            "forward_macro_seed", "continuation_macro_seed", "finite_k_macro_seed",
            "holdout_macro_seed", "preflight_macro_seed",
        ):
            _true_int(getattr(self, name), name)
        seeds = (
            self.forward_macro_seed, self.continuation_macro_seed,
            self.finite_k_macro_seed, self.holdout_macro_seed, self.preflight_macro_seed,
        )
        if len(set(seeds)) != len(seeds):
            raise QualificationCampaignError("campaign macro seeds must be disjoint")
        expected_qualification = 2 * 7 * (
            self.forward_episodes
            + self.continuation_episodes * self.max_actions_per_target
            + self.finite_k_episodes * self.max_actions_per_target
        )
        # Three K=8 starts plus one continuation run for each of K=16, 32,
        # and 64 are the six frozen solver runs.
        if expected_qualification * 6 != self.qualification_calls:
            raise QualificationCampaignError("qualification call arithmetic is inconsistent")
        expected_holdout = self.holdout_episodes_per_k * len(self.k_values) * len(self.holdout_arms)
        if expected_holdout != self.holdout_calls:
            raise QualificationCampaignError("holdout call arithmetic is inconsistent")
        if self.qualification_calls + self.holdout_calls + self.preflight_max_calls != self.total_call_limit:
            raise QualificationCampaignError("total call arithmetic is inconsistent")

    @property
    def protocol_fingerprint(self) -> str:
        return _digest((
            self.k_values, self.models, self.starts, self.forward_episodes,
            self.continuation_episodes, self.finite_k_episodes,
            self.max_actions_per_target, self.max_iterations,
            self.min_complete_samples, self.residual_tolerance,
            self.forward_namespace, self.forward_macro_seed,
            self.continuation_namespace, self.continuation_macro_seed,
            self.finite_k_namespace, self.finite_k_macro_seed,
            self.holdout_namespace, self.holdout_macro_seed,
            self.preflight_namespace, self.preflight_macro_seed,
            self.holdout_episodes_per_k, self.holdout_arms,
            self.qualification_calls, self.holdout_calls, self.total_call_limit,
            self.scenario_schedule, self.max_k64_call_seconds,
            self.max_eight_worker_seconds, self.timing_workers,
        ))

    def require_formal_start(self, preflight: object) -> None:
        if not isinstance(preflight, QualificationTimingReport):
            raise QualificationCampaignError(
                "formal qualification requires the complete K-scale timing report"
            )
        if preflight.plan_fingerprint != self.protocol_fingerprint:
            raise QualificationCampaignError("preflight plan fingerprint mismatch")
        if preflight.scheduler_calls > self.preflight_max_calls:
            raise QualificationCampaignError("preflight exceeded its isolated call budget")
        if preflight.formal_calls_consumed != 0:
            raise QualificationCampaignError("preflight consumed formal calls")
        if not preflight.complete:
            raise QualificationCampaignError("timing preflight did not complete")
        if preflight.worst_k64_call_seconds > self.max_k64_call_seconds:
            raise QualificationCampaignError("K=64 single-call timing gate failed")
        if preflight.projected_eight_worker_seconds > self.max_eight_worker_seconds:
            raise QualificationCampaignError("eight-worker qualification timing gate failed")
        if not preflight.rss_supported:
            raise QualificationCampaignError("timing report lacks supported RSS measurements")


@dataclass(frozen=True)
class TimingPreflight:
    plan_fingerprint: str
    namespace: str
    macro_seed: int
    k: int
    scenario: str
    episodes: int
    scheduler_calls: int
    formal_calls_consumed: int
    wall_seconds: float
    scheduler_wall_seconds: float
    preparation_seconds: float
    cpu_seconds: float
    serialization_bytes: int
    peak_rss_bytes: int | None
    rss_supported: bool
    trace_fingerprint: str
    complete: bool
    claim_boundary: str = "real_physics_timing_preflight"

    def __post_init__(self) -> None:
        if not isinstance(self.plan_fingerprint, str) or len(self.plan_fingerprint) != 64:
            raise QualificationCampaignError("preflight plan fingerprint is invalid")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise QualificationCampaignError("preflight namespace is invalid")
        _true_int(self.macro_seed, "preflight macro seed")
        _true_int(self.k, "preflight K", 1)
        if self.scenario not in SCENARIO_NAMES:
            raise QualificationCampaignError("preflight scenario is invalid")
        _true_int(self.episodes, "preflight episodes", 1)
        _true_int(self.scheduler_calls, "preflight scheduler calls")
        _true_int(self.formal_calls_consumed, "preflight formal calls")
        _finite(self.wall_seconds, "preflight wall seconds", nonnegative=True)
        _finite(self.scheduler_wall_seconds, "scheduler wall seconds", nonnegative=True)
        _finite(self.preparation_seconds, "preparation seconds", nonnegative=True)
        _finite(self.cpu_seconds, "preflight CPU seconds", nonnegative=True)
        _true_int(self.serialization_bytes, "preflight serialization bytes", 1)
        if self.peak_rss_bytes is not None:
            _true_int(self.peak_rss_bytes, "preflight peak RSS", 1)
        if type(self.rss_supported) is not bool or type(self.complete) is not bool:
            raise QualificationCampaignError("preflight boolean field is invalid")
        if not isinstance(self.trace_fingerprint, str) or len(self.trace_fingerprint) != 64:
            raise QualificationCampaignError("preflight trace fingerprint is invalid")


@dataclass(frozen=True)
class QualificationTimingReport:
    plan_fingerprint: str
    rows: tuple[TimingPreflight, ...]
    scheduler_calls: int
    formal_calls_consumed: int
    worst_k64_call_seconds: float
    projected_eight_worker_seconds: float
    rss_supported: bool
    complete: bool
    throughput_rows: tuple[TimingPreflight, ...]
    worker_count: int
    parent_wait_seconds: float
    k64_parallel_elapsed_seconds: float
    k64_throughput_calls_per_second: float
    parent_peak_rss_bytes: int | None

    def __post_init__(self) -> None:
        _true_int(self.worker_count, "timing worker count", 1)
        expected = tuple((k, scenario) for k in QUALIFICATION_K for scenario in SCENARIO_NAMES)
        actual = tuple((row.k, row.scenario) for row in self.rows)
        if actual != expected:
            raise QualificationCampaignError("timing report must cover every K/scenario pair")
        if any(row.plan_fingerprint != self.plan_fingerprint for row in self.rows):
            raise QualificationCampaignError("timing report mixes plan fingerprints")
        _true_int(self.scheduler_calls, "timing report scheduler calls")
        _true_int(self.formal_calls_consumed, "timing report formal calls")
        _finite(self.worst_k64_call_seconds, "worst K=64 seconds", nonnegative=True)
        _finite(
            self.projected_eight_worker_seconds,
            "projected eight-worker seconds",
            nonnegative=True,
        )
        if type(self.rss_supported) is not bool or type(self.complete) is not bool:
            raise QualificationCampaignError("timing report flags must be bool")
        if len(self.throughput_rows) != self.worker_count:
            raise QualificationCampaignError("timing throughput row count is incomplete")
        if any(row.k != 64 for row in self.throughput_rows):
            raise QualificationCampaignError("timing throughput must use K=64")
        if any(
            row.plan_fingerprint != self.plan_fingerprint
            for row in self.throughput_rows
        ):
            raise QualificationCampaignError("throughput rows mix plan fingerprints")
        if self.scheduler_calls != sum(
            row.scheduler_calls for row in self.rows + self.throughput_rows
        ):
            raise QualificationCampaignError("timing report call accounting is inconsistent")
        _finite(self.parent_wait_seconds, "parent wait seconds", nonnegative=True)
        _finite(
            self.k64_parallel_elapsed_seconds,
            "K=64 parallel elapsed seconds",
            nonnegative=True,
        )
        throughput = _finite(
            self.k64_throughput_calls_per_second,
            "K=64 throughput",
            nonnegative=True,
        )
        if throughput <= 0.0:
            raise QualificationCampaignError("K=64 throughput must be positive")
        if self.parent_peak_rss_bytes is not None:
            _true_int(self.parent_peak_rss_bytes, "parent peak RSS", 1)


class _UniformRealPolicy:
    """Deterministic online policy used only by the real physics preflight."""

    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        if not isinstance(policy_key, str) or not policy_key:
            raise QualificationCampaignError("preflight policy key is invalid")
        available = tuple(
            row.replica_id
            for row in observation.public_state.replica_observations
            if row.available
        )
        if not available:
            return None
        digest = hashlib.sha256(policy_key.encode("utf-8")).digest()
        return tuple(sorted(available))[int.from_bytes(digest[:8], "big") % len(available)]

    def predict_token_share(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
    ) -> tuple[tuple[int, float], ...]:
        topology = observation.public_state.topology
        available = tuple(
            row.replica_id
            for row in observation.public_state.replica_observations
            if row.available
        )
        if not available:
            return ()
        share = 1.0 / len(available)
        return tuple((replica_id, share) for replica_id in sorted(available))


def _peak_rss_bytes() -> tuple[int | None, bool]:
    """Return real process RSS/working-set bytes, never a Python heap proxy."""

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("Psapi.dll")
            kernel = ctypes.WinDLL("Kernel32.dll")
            get_current_process = kernel.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = (
                wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD,
            )
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(
                get_current_process(), ctypes.byref(counters), counters.cb,
            )
            if ok:
                return int(counters.peak_working_set_size), True
        except Exception:
            return None, False
        return None, False
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name != "nt":
            value *= 1024
        return value, True
    except Exception:
        return None, False


def build_qualification_trace_library(
    plan: QualificationPlan,
    *,
    k: int,
    count: int,
    scenario: str | None = None,
    namespace: str | None = None,
    macro_seed: int | None = None,
    episode_start: int = 0,
) -> FixedEpisodeLibrary:
    """Build a real immutable trace library for one scenario and topology."""

    if not isinstance(plan, QualificationPlan):
        raise QualificationCampaignError("plan must be QualificationPlan")
    _true_int(k, "K", 1)
    if k not in plan.k_values:
        raise QualificationCampaignError("K is outside the frozen qualification scale")
    _true_int(count, "trace count", 1)
    _true_int(episode_start, "episode start")
    if scenario is not None and scenario not in SCENARIO_NAMES:
        raise QualificationCampaignError("scenario is outside the routing protocol")
    chosen_namespace = plan.forward_namespace if namespace is None else namespace
    chosen_seed = plan.forward_macro_seed if macro_seed is None else macro_seed
    if not isinstance(chosen_namespace, str) or not chosen_namespace:
        raise QualificationCampaignError("trace namespace is invalid")
    _true_int(chosen_seed, "trace macro seed")
    topology = Topology(k)
    if scenario is not None:
        scenarios = (scenario,) * count
    elif count == plan.forward_episodes:
        scenarios = plan.scenario_schedule
    elif count == 1:
        scenarios = ("S0",)
    elif count % len(SCENARIO_NAMES) == 0:
        per_scenario = count // len(SCENARIO_NAMES)
        scenarios = tuple(
            current for current in SCENARIO_NAMES for _ in range(per_scenario)
        )
    else:
        raise QualificationCampaignError(
            "multi-scenario library count must be balanced across four scenarios"
        )
    traces = tuple(
        generate_routing_trace(
            current_scenario,
            episode_index,
            namespace=chosen_namespace,
            macro_seed=chosen_seed,
            topology=topology,
        )
        for episode_index, current_scenario in enumerate(
            scenarios, start=episode_start,
        )
    )
    return FixedEpisodeLibrary.from_traces(traces)


def run_timing_preflight(
    plan: QualificationPlan,
    *,
    k: int = 8,
    episodes: int = 1,
    scenario: str = "S0",
    episode_start: int = 0,
) -> TimingPreflight:
    """Run real routing physics on isolated preflight identities."""

    if not isinstance(plan, QualificationPlan):
        raise QualificationCampaignError("plan must be QualificationPlan")
    _true_int(episodes, "preflight episodes", 1)
    _true_int(episode_start, "preflight episode start")
    if episodes > plan.preflight_max_calls:
        raise QualificationCampaignError("preflight exceeds its frozen call limit")
    if k not in plan.k_values:
        raise QualificationCampaignError("preflight K is outside the qualification scale")
    if scenario not in SCENARIO_NAMES:
        raise QualificationCampaignError("preflight scenario is outside the routing protocol")
    total_start = time.perf_counter()
    cpu_start = time.process_time()
    preparation_start = time.perf_counter()
    serialized = pickle.dumps(
        (
            plan.protocol_fingerprint, k, scenario, episode_start, episodes,
            plan.preflight_namespace, plan.preflight_macro_seed,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    library = build_qualification_trace_library(
        plan,
        k=k,
        count=episodes,
        scenario=scenario,
        namespace=plan.preflight_namespace,
        macro_seed=plan.preflight_macro_seed,
        episode_start=episode_start,
    )
    preparation_seconds = time.perf_counter() - preparation_start
    start = time.perf_counter()
    for trace in library.traces:
        result: SimultaneousRoutingResult = simulate_simultaneous_routing(
            trace, _UniformRealPolicy(), policy_name="qualification_preflight_uniform",
        )
        if result.physical.trace_fingerprint != trace.fingerprint:
            raise QualificationCampaignError("preflight trace fingerprint changed")
        if not result.physical.complete_drain:
            raise QualificationCampaignError("preflight did not complete drain")
        invariant_counters = {
            "down_starts", "future_work_observations",
            "completed_without_winner", "duplicate_live_attempts",
        }
        if any(result.physical.invariants.get(key, 0) != 0
               for key in invariant_counters):
            raise QualificationCampaignError("preflight physical invariant failed")
    scheduler_elapsed = time.perf_counter() - start
    elapsed = time.perf_counter() - total_start
    cpu_elapsed = time.process_time() - cpu_start
    peak_rss, rss_supported = _peak_rss_bytes()
    trace_fingerprint = _digest(tuple(trace.fingerprint for trace in library.traces))
    return TimingPreflight(
        plan.protocol_fingerprint,
        plan.preflight_namespace,
        plan.preflight_macro_seed,
        k,
        scenario,
        episodes,
        episodes,
        0,
        elapsed,
        scheduler_elapsed,
        preparation_seconds,
        cpu_elapsed,
        len(serialized),
        peak_rss,
        rss_supported,
        trace_fingerprint,
        True,
    )


def _timing_worker(
    task: tuple[QualificationPlan, int, str, int],
) -> TimingPreflight:
    plan, k, scenario, episode_start = task
    return run_timing_preflight(
        plan, k=k, episodes=1, scenario=scenario,
        episode_start=episode_start,
    )


def run_k_scale_timing_preflight(
    plan: QualificationPlan,
) -> QualificationTimingReport:
    """Measure every K/scenario pair and compute the frozen worst-case gate."""

    if not isinstance(plan, QualificationPlan):
        raise QualificationCampaignError("plan must be QualificationPlan")
    grid_tasks = tuple(
        (plan, k, scenario, 0)
        for k in plan.k_values
        for scenario in SCENARIO_NAMES
    )
    throughput_tasks = tuple(
        (plan, 64, SCENARIO_NAMES[index % len(SCENARIO_NAMES)], index + 1)
        for index in range(plan.timing_workers)
    )
    context = multiprocessing.get_context("spawn")
    parent_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=plan.timing_workers, mp_context=context,
    ) as executor:
        rows = tuple(executor.map(_timing_worker, grid_tasks))
        throughput_start = time.perf_counter()
        throughput_rows = tuple(executor.map(_timing_worker, throughput_tasks))
        k64_parallel_elapsed = time.perf_counter() - throughput_start
    parent_wait = time.perf_counter() - parent_start
    calls_per_solver_run = plan.qualification_calls // 6
    runs_by_k = {8: 3, 16: 1, 32: 1, 64: 1}
    worst_by_k = {
        k: max(row.scheduler_wall_seconds for row in rows if row.k == k)
        for k in plan.k_values
    }
    projected_non_k64 = math.fsum(
        runs_by_k[k] * calls_per_solver_run * worst_by_k[k]
        for k in plan.k_values if k != 64
    ) / plan.timing_workers
    throughput = len(throughput_rows) / k64_parallel_elapsed
    projected = (
        projected_non_k64
        + runs_by_k[64] * calls_per_solver_run / throughput
    )
    parent_peak, parent_rss_supported = _peak_rss_bytes()
    return QualificationTimingReport(
        plan.protocol_fingerprint,
        rows,
        sum(row.scheduler_calls for row in rows + throughput_rows),
        0,
        worst_by_k[64],
        projected,
        all(row.rss_supported for row in rows + throughput_rows)
        and parent_rss_supported,
        all(row.complete for row in rows + throughput_rows),
        throughput_rows,
        plan.timing_workers,
        parent_wait,
        k64_parallel_elapsed,
        throughput,
        parent_peak,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "HOLDOUT_CALLS",
    "MAX_ACTIONS_PER_TARGET",
    "MODEL_IDS",
    "MAX_EIGHT_WORKER_SECONDS",
    "MAX_K64_CALL_SECONDS",
    "PRE_FLIGHT_MAX_CALLS",
    "R2_CONTINUATION_MACRO_SEED",
    "R2_CONTINUATION_NAMESPACE",
    "R2_DRY_RUN_MACRO_SEED",
    "R2_DRY_RUN_NAMESPACE",
    "R2_FINITE_K_MACRO_SEED",
    "R2_FINITE_K_NAMESPACE",
    "R2_FORWARD_MACRO_SEED",
    "R2_FORWARD_NAMESPACE",
    "R2_RUN_ID",
    "QUALIFICATION_CALLS",
    "QUALIFICATION_EPISODES",
    "QUALIFICATION_SCENARIO_SCHEDULE",
    "QualificationTimingReport",
    "QualificationCampaignError",
    "QualificationBackendProtocol",
    "QualificationCampaignResult",
    "QualificationPlan",
    "QualificationExecutionResult",
    "QualificationWorkUnit",
    "QualificationTimingReport",
    "TimingPreflight",
    "TOTAL_CALL_LIMIT",
    "build_qualification_trace_library",
    "execute_canonical_work_units",
    "qualification_source_fingerprint",
    "run_k_scale_timing_preflight",
    "run_qualification_campaign",
    "run_k_scale_timing_preflight",
    "run_timing_preflight",
]
