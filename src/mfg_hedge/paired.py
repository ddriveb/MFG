"""Paired A/B pipeline: one shared trace, two arms, verified identity.

Spec: `.scratch/mfg-hedge-paired-comparison/spec.md` sections 2 and 15. The
pipeline completes every simulation and invariant check in memory before any
artifact is written (the CLI commits transactionally afterwards).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import NormalDist

from .calibration import CalibrationTable
from .common_state import CommonStateTimeline
from .config import PairedExperimentConfig
from .domain import ProtectionAction
from .hedge_simulation import HedgeSimulationResult, simulate_hedge_common_state
from .mfg_solver import MFGSolution, SolverParameters, SolverStatus, solve_mfg
from .quota import QuotaProjectionResult, project_actions
from .workload import WorkloadTrace, generate_workload_with_hedge


@dataclass(frozen=True)
class PairedRunResult:
    trace: WorkloadTrace
    solution: MFGSolution
    projection: QuotaProjectionResult
    arm_a: HedgeSimulationResult
    arm_b: HedgeSimulationResult
    tau0: float


def verify_trace_identity(first: WorkloadTrace, second: WorkloadTrace) -> None:
    """Fail loudly unless two traces are identical in every stream."""
    differences = []
    if first.tokens != second.tokens:
        differences.append("tokens(arrival/class)")
    if first.service_times != second.service_times:
        differences.append("attempt-0 service draws")
    if first.replay_service_times != second.replay_service_times:
        differences.append("attempt-1 replay draws")
    if first.hedge_service_times != second.hedge_service_times:
        differences.append("attempt-2 hedge draws")
    if differences:
        raise ValueError("trace identity mismatch: " + ", ".join(differences))


def trace_identity_fingerprint(trace: WorkloadTrace) -> dict[str, str]:
    """Per-stream SHA-256 fingerprints for manifest/comparison provenance."""

    def digest(payload) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    return {
        "tokens": digest(
            [
                [spec.token_id, spec.arrival_time, spec.token_class.value]
                for spec in trace.tokens
            ]
        ),
        "attempt0_service_times": digest([list(s) for s in trace.service_times]),
        "attempt1_replay_times": digest(
            [list(s) for s in trace.replay_service_times or ()]
        ),
        "attempt2_hedge_times": digest(
            [list(s) for s in trace.hedge_service_times or ()]
        ),
    }


def compute_tau0(config) -> float:
    """tau0 = exp(mu + sigma * z_q) from the Healthy service distribution."""
    from .workload import lognormal_parameters

    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )
    return math.exp(mu + sigma * NormalDist().inv_cdf(config.hedge_delay_quantile))


def solve_paired_policy(
    config: PairedExperimentConfig, table: CalibrationTable
) -> MFGSolution:
    """Solve and enforce the official H/D/F gate once for paired evaluation."""
    base = config.base_config()
    params = SolverParameters(
        inverse_temperature=base.softmax_inverse_temperature,
        policy_damping=base.policy_damping,
        price_step=base.price_step,
        price_damping=base.price_damping,
        replay_penalty_regular=config.replay_penalty_regular,
        replay_penalty_urgent=config.replay_penalty_urgent,
        incremental_work_cost=config.incremental_work_cost,
        wasted_work_cost=config.wasted_work_cost,
    )
    solution = solve_mfg(base, table, params)
    for state in solution.solutions.values():
        if state.status not in (SolverStatus.CONVERGED, SolverStatus.FORCED_NORMAL):
            raise ValueError(
                f"official gate failed: state {state.state.value} has status "
                f"{state.status.value} ({state.reason})"
            )
    return solution


def _assert_result_matches_trace(result: HedgeSimulationResult, trace: WorkloadTrace) -> None:
    """Guard that an arm consumed the shared trace untampered."""
    for token in result.tokens:
        spec = trace.tokens[token.token_id]
        if (
            token.arrival_time != spec.arrival_time
            or token.token_class is not spec.token_class
        ):
            raise ValueError(
                f"trace identity mismatch at Token {token.token_id}: "
                "arrival/class differ from the shared trace"
            )
        primary = token.attempts[0]
        if (
            primary.required_work
            != trace.service_times[token.primary_replica][token.token_id]
        ):
            raise ValueError(
                f"trace identity mismatch at Token {token.token_id}: "
                "attempt-0 work differs from the shared trace"
            )


def run_paired_comparison(
    config: PairedExperimentConfig,
    table: CalibrationTable,
    token_count: int,
    *,
    evaluation_seed: int | None = None,
    solution: MFGSolution | None = None,
) -> PairedRunResult:
    """Run both arms on one shared trace with a verified MFG solution."""
    if isinstance(token_count, bool) or token_count <= 0:
        raise ValueError(f"token_count must be a positive int, got {token_count!r}")
    base = config.base_config()
    timeline = config.timeline()
    if evaluation_seed is not None and (
        isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int)
    ):
        raise ValueError(f"evaluation_seed must be an int, got {evaluation_seed!r}")
    trace = generate_workload_with_hedge(base, token_count, base_seed=evaluation_seed)
    last_arrival = trace.tokens[-1].arrival_time
    if not last_arrival > timeline.recovered_start:
        raise ValueError(
            f"last arrival {last_arrival:.6f} does not exceed "
            f"recovered_start {timeline.recovered_start}; increase --tokens"
        )
    solution = solve_paired_policy(config, table) if solution is None else solution
    for state in solution.solutions.values():
        if state.status not in (SolverStatus.CONVERGED, SolverStatus.FORCED_NORMAL):
            raise ValueError(
                f"official gate failed: state {state.state.value} has status "
                f"{state.status.value} ({state.reason})"
            )
    projection = project_actions(
        trace, solution, timeline, window_width=config.control_window
    )
    tau0 = compute_tau0(base)
    arm_a = simulate_hedge_common_state(
        trace,
        timeline=timeline,
        degraded_slowdown=base.degraded_slowdown,
        actions={},
    )
    arm_b = simulate_hedge_common_state(
        trace,
        timeline=timeline,
        degraded_slowdown=base.degraded_slowdown,
        actions=projection.to_action_plan(),
        hedge_delay=tau0,
    )
    _assert_result_matches_trace(arm_a, trace)
    _assert_result_matches_trace(arm_b, trace)
    return PairedRunResult(
        trace=trace,
        solution=solution,
        projection=projection,
        arm_a=arm_a,
        arm_b=arm_b,
        tau0=tau0,
    )
