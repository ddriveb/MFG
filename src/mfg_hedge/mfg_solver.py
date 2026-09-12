"""Capacity-feasible finite-action MFG solver (ADR-0007/0008/0009).

Prices the total incremental resource (Hedge + Replay work). A cold-started
policy/load fixed point is used exactly once at price zero. If capacity binds,
rho is pinned to the target utilization and only the resulting one-dimensional
price equation is bracketed and bisected. Reads only a CalibrationTable and
configuration — never evaluation traces or A/B results. Deterministic and
timestamp-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Callable, Mapping

from .calibration import CalibrationTable
from .common_state import CommonState
from .config import ExperimentConfig
from .domain import ActionStats, ProtectionAction, TokenClass


DEFAULT_TOLERANCE = 1e-6
DEFAULT_INNER_MAX_ITERATIONS = 2000
DEFAULT_BRACKET_MAX_DOUBLINGS = 60
DEFAULT_BISECTION_MAX_ITERATIONS = 80

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_HEDGE_ACTIONS = (ProtectionAction.DELAYED_HEDGE, ProtectionAction.IMMEDIATE_HEDGE)
_CLASSES = (TokenClass.REGULAR, TokenClass.URGENT)
_STATES = (CommonState.HEALTHY, CommonState.DEGRADED, CommonState.FAILED)


class SolverStatus(str, Enum):
    CONVERGED = "converged"
    FORCED_NORMAL = "forced_normal"
    INFEASIBLE = "infeasible"
    NONCONVERGED = "nonconverged"


def softmax_stable(logits: Mapping) -> dict:
    """Numerically stable softmax: subtract the max before exp."""
    if not logits:
        raise ValueError("softmax requires at least one action")
    for value in logits.values():
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"logits must be finite numbers, got {value!r}")
    peak = max(logits.values())
    weights = {key: math.exp(value - peak) for key, value in logits.items()}
    total = math.fsum(weights.values())
    return {key: value / total for key, value in weights.items()}


def arrival_rate_of(config: ExperimentConfig) -> float:
    """The single arrival-rate formula shared with the workload generator."""
    return (
        config.healthy_offered_load
        * config.replicas_per_expert
        / config.healthy_service_mean
    )


def state_capacity(config: ExperimentConfig, state: CommonState) -> float:
    if state is CommonState.HEALTHY:
        return config.replicas_per_expert / config.healthy_service_mean
    if state is CommonState.DEGRADED:
        return (1.0 + 1.0 / config.degraded_slowdown) / config.healthy_service_mean
    if state is CommonState.FAILED:
        return 1.0 / config.healthy_service_mean
    raise ValueError(f"state must be H, D, or F, got {state!r}")


def validate_capacity_acceptance(
    price: float,
    capacity_gap: float,
    policy_residual: float,
    load_residual: float,
    *,
    capacity_tolerance: float,
    complementarity_tolerance: float,
    residual_tolerance: float,
) -> tuple[str, ...]:
    """Pure acceptance predicate; returns violation codes (empty = accept)."""
    violations = []
    if price < 0.0:
        violations.append("negative_price")
    if capacity_gap > capacity_tolerance:
        violations.append("capacity_gap")
    if abs(price * capacity_gap) > complementarity_tolerance:
        violations.append("complementarity")
    if policy_residual >= residual_tolerance:
        violations.append("policy_residual")
    if load_residual >= residual_tolerance:
        violations.append("load_residual")
    return tuple(violations)


@dataclass(frozen=True)
class SolverParameters:
    inverse_temperature: float
    policy_damping: float
    price_step: float
    price_damping: float  # retained for config/provenance; not consumed
    replay_penalty_regular: float = 1.0
    replay_penalty_urgent: float = 5.0
    incremental_work_cost: float = 0.0
    wasted_work_cost: float = 0.0
    tolerance: float = DEFAULT_TOLERANCE
    capacity_tolerance: float = DEFAULT_TOLERANCE
    complementarity_tolerance: float = DEFAULT_TOLERANCE
    inner_max_iterations: int = DEFAULT_INNER_MAX_ITERATIONS
    bracket_max_doublings: int = DEFAULT_BRACKET_MAX_DOUBLINGS
    bisection_max_iterations: int = DEFAULT_BISECTION_MAX_ITERATIONS

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "SolverParameters":
        return cls(
            inverse_temperature=config.softmax_inverse_temperature,
            policy_damping=config.policy_damping,
            price_step=config.price_step,
            price_damping=config.price_damping,
        )

    def __post_init__(self) -> None:
        for name in (
            "inverse_temperature",
            "policy_damping",
            "price_step",
            "price_damping",
            "replay_penalty_regular",
            "replay_penalty_urgent",
            "tolerance",
            "capacity_tolerance",
            "complementarity_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        for name in ("incremental_work_cost", "wasted_work_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        for name in (
            "inner_max_iterations",
            "bracket_max_doublings",
            "bisection_max_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")


@dataclass(frozen=True)
class StatePolicySolution:
    state: CommonState
    status: SolverStatus
    reason: str
    grid_clamped: bool
    probabilities: Mapping[TokenClass, Mapping[ProtectionAction, float]]
    price: float
    rho: float
    primary_work_rate: float
    hedge_work_rate: float
    replay_work_rate: float
    total_work_rate: float
    target_work_rate: float
    capacity_gap: float
    capacity_violation: float
    capacity_slack: float
    complementarity_residual: float
    hedge_budget_rate: float
    policy_residual: float
    price_residual: float
    load_residual: float
    iterations: int
    outer_evaluations: int
    price_evaluations: tuple[tuple[float, float], ...]
    structural_capacity_violation: bool
    final_stats: Mapping[TokenClass, Mapping[ProtectionAction, ActionStats]]

    @property
    def converged(self) -> bool:
        return self.status is SolverStatus.CONVERGED


@dataclass(frozen=True)
class MFGSolution:
    solutions: Mapping[CommonState, StatePolicySolution]
    parameters: SolverParameters
    class_weights: Mapping[TokenClass, float] = field(
        default_factory=lambda: {TokenClass.REGULAR: 0.8, TokenClass.URGENT: 0.2}
    )

    def for_state(self, state: CommonState) -> StatePolicySolution:
        """Recovered phase reuses the H solution; R is never solved."""
        if state is CommonState.HEALTHY:
            return self.solutions[CommonState.HEALTHY]
        return self.solutions[state]

    def to_mapping(self) -> dict:
        states = {}
        for state in _STATES:
            solution = self.solutions[state]
            states[state.value] = {
                "status": solution.status.value,
                "reason": solution.reason,
                "grid_clamped": solution.grid_clamped,
                "probabilities": {
                    cls.value: {
                        action.value: solution.probabilities[cls][action]
                        for action in _ACTIONS
                    }
                    for cls in _CLASSES
                },
                "price": solution.price,
                "rho": solution.rho,
                "primary_work_rate": solution.primary_work_rate,
                "hedge_work_rate": solution.hedge_work_rate,
                "replay_work_rate": solution.replay_work_rate,
                "total_work_rate": solution.total_work_rate,
                "target_work_rate": solution.target_work_rate,
                "capacity_gap": solution.capacity_gap,
                "capacity_violation": solution.capacity_violation,
                "capacity_slack": solution.capacity_slack,
                "complementarity_residual": solution.complementarity_residual,
                "hedge_budget_rate": solution.hedge_budget_rate,
                "policy_residual": solution.policy_residual,
                "price_residual": solution.price_residual,
                "load_residual": solution.load_residual,
                "iterations": solution.iterations,
                "outer_evaluations": solution.outer_evaluations,
                "price_evaluations": [
                    [price, gap] for price, gap in solution.price_evaluations
                ],
                "structural_capacity_violation": solution.structural_capacity_violation,
                "final_stats": {
                    cls.value: {
                        action.value: {
                            "mean_latency": solution.final_stats[cls][action].mean_latency,
                            "replay_probability": solution.final_stats[cls][action].replay_probability,
                            "deadline_miss_probability": solution.final_stats[cls][action].deadline_miss_probability,
                            "expected_hedge_work": solution.final_stats[cls][action].expected_hedge_work,
                            "expected_replay_work": solution.final_stats[cls][action].expected_replay_work,
                            "expected_wasted_work": solution.final_stats[cls][action].expected_wasted_work,
                        }
                        for action in _ACTIONS
                    }
                    for cls in _CLASSES
                },
            }
        return {
            "solver_parameters": {
                "inverse_temperature": self.parameters.inverse_temperature,
                "policy_damping": self.parameters.policy_damping,
                "price_step": self.parameters.price_step,
                "price_damping": self.parameters.price_damping,
                "replay_penalty_regular": self.parameters.replay_penalty_regular,
                "replay_penalty_urgent": self.parameters.replay_penalty_urgent,
                "incremental_work_cost": self.parameters.incremental_work_cost,
                "wasted_work_cost": self.parameters.wasted_work_cost,
                "tolerance": self.parameters.tolerance,
                "capacity_tolerance": self.parameters.capacity_tolerance,
                "complementarity_tolerance": self.parameters.complementarity_tolerance,
                "inner_max_iterations": self.parameters.inner_max_iterations,
                "bracket_max_doublings": self.parameters.bracket_max_doublings,
                "bisection_max_iterations": self.parameters.bisection_max_iterations,
            },
            "class_weights": {
                cls.value: self.class_weights[cls] for cls in _CLASSES
            },
            "states": states,
        }

    def to_json_bytes(self) -> bytes:
        payload = self.to_mapping()
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")


class _GridClamped(Exception):
    def __init__(self, rho: float) -> None:
        super().__init__(f"calibration grid clamped at rho={rho}")
        self.rho = rho


@dataclass
class _InnerResult:
    probabilities: dict
    rho: float
    stats: dict
    primary: float
    hedge: float
    replay: float
    total: float
    iterations: int
    converged: bool


def _query_stats(table, state, rho, feasible) -> dict:
    result = {}
    clamped = False
    for cls in _CLASSES:
        result[cls] = {}
        for action in _ACTIONS:
            if action in feasible:
                estimate = table.estimate(state, rho, cls, action)
                clamped = clamped or estimate.clamped
                result[cls][action] = estimate.stats
            else:
                result[cls][action] = ActionStats(
                    mean_latency=0.0,
                    replay_probability=0.0,
                    deadline_miss_probability=0.0,
                    expected_hedge_work=0.0,
                    expected_replay_work=0.0,
                )
    if clamped:
        raise _GridClamped(rho)
    return result


def _work_rates(config, probabilities, stats) -> tuple[float, float, float]:
    arrival_rate = arrival_rate_of(config)
    weights = {
        TokenClass.REGULAR: config.regular_token_ratio,
        TokenClass.URGENT: config.urgent_token_ratio,
    }
    primary = arrival_rate * config.healthy_service_mean
    hedge = 0.0
    replay = 0.0
    for cls in _CLASSES:
        for action in _ACTIONS:
            probability = probabilities[cls][action]
            if probability == 0.0:
                continue
            hedge += probability * stats[cls][action].expected_hedge_work * weights[cls]
            replay += probability * stats[cls][action].expected_replay_work * weights[cls]
    return primary, hedge * arrival_rate, replay * arrival_rate


def _class_weights(config) -> dict:
    return {
        TokenClass.REGULAR: config.regular_token_ratio,
        TokenClass.URGENT: config.urgent_token_ratio,
    }


def _soft_best_response(params, stats, price, feasible, cls) -> dict:
    penalty = (
        params.replay_penalty_regular
        if cls is TokenClass.REGULAR
        else params.replay_penalty_urgent
    )
    costs = {}
    for action in feasible:
        cell = stats[cls][action]
        incremental = cell.expected_hedge_work + cell.expected_replay_work
        costs[action] = (
            cell.mean_latency
            + penalty * cell.replay_probability
            + (price + params.incremental_work_cost) * incremental
            + params.wasted_work_cost * cell.expected_wasted_work
        )
    logits = {action: -params.inverse_temperature * costs[action] for action in feasible}
    response = softmax_stable(logits)
    return {action: response.get(action, 0.0) for action in _ACTIONS}


def _inner_solve(
    config, table, params, state, price
) -> _InnerResult:
    """Cold-started deterministic policy/load fixed point at a fixed price."""
    capacity = state_capacity(config, state)
    primary = arrival_rate_of(config) * config.healthy_service_mean
    feasible = (ProtectionAction.NORMAL,) if state is CommonState.FAILED else _ACTIONS
    probabilities = {
        cls: {action: 1.0 if action is ProtectionAction.NORMAL else 0.0 for action in _ACTIONS}
        for cls in _CLASSES
    }
    rho = primary / capacity
    iterations = 0
    converged = False
    stats = None
    for iteration in range(1, params.inner_max_iterations + 1):
        iterations = iteration
        stats = _query_stats(table, state, rho, feasible)
        policy_residual = 0.0
        candidate = {}
        for cls in _CLASSES:
            best = _soft_best_response(params, stats, price, feasible, cls)
            candidate[cls] = {
                action: (1.0 - params.policy_damping) * probabilities[cls][action]
                + params.policy_damping * best[action]
                for action in _ACTIONS
            }
            for action in _ACTIONS:
                policy_residual = max(
                    policy_residual, abs(best[action] - probabilities[cls][action])
                )
        primary, hedge, replay = _work_rates(config, candidate, stats)
        total = primary + hedge + replay
        rho_candidate = total / capacity
        load_residual = abs(rho_candidate - rho)
        probabilities = candidate
        rho = rho_candidate
        if policy_residual < params.tolerance and load_residual < params.tolerance:
            converged = True
            break
    primary, hedge, replay = _work_rates(config, probabilities, stats)
    return _InnerResult(
        probabilities=probabilities,
        rho=rho,
        stats=stats,
        primary=primary,
        hedge=hedge,
        replay=replay,
        total=primary + hedge + replay,
        iterations=iterations,
        converged=converged,
    )


def _finalize(
    config, table, params, state, result: _InnerResult, price: float,
    status: SolverStatus, reason: str, grid_clamped: bool,
    inner_iterations: int, outer_evaluations: int,
    price_evaluations: tuple, structural_violation: bool,
    price_residual: float = 0.0,
    stats_override: dict | None = None,
) -> StatePolicySolution:
    """Recompute every derived quantity from the final state."""
    capacity = state_capacity(config, state)
    target = config.target_max_utilization * capacity
    stats = stats_override
    if stats is None:
        stats = _query_stats(
            table,
            state,
            result.rho,
            _ACTIONS if state is not CommonState.FAILED else (ProtectionAction.NORMAL,),
        )
    primary, hedge, replay = _work_rates(config, result.probabilities, stats)
    total = primary + hedge + replay
    policy_residual = 0.0
    for cls in _CLASSES:
        best = _soft_best_response(
            params, stats, price,
            (ProtectionAction.NORMAL,) if state is CommonState.FAILED else _ACTIONS,
            cls,
        )
        for action in _ACTIONS:
            policy_residual = max(
                policy_residual, abs(best[action] - result.probabilities[cls][action])
            )
    load_residual = abs(total / capacity - result.rho)
    capacity_gap = total - target
    complementarity = abs(price * capacity_gap)
    hedge_budget = max(0.0, target - primary - replay)
    return StatePolicySolution(
        state=state,
        status=status,
        reason=reason,
        grid_clamped=grid_clamped,
        probabilities=result.probabilities,
        price=price,
        rho=result.rho,
        primary_work_rate=primary,
        hedge_work_rate=hedge,
        replay_work_rate=replay,
        total_work_rate=total,
        target_work_rate=target,
        capacity_gap=capacity_gap,
        capacity_violation=max(0.0, capacity_gap),
        capacity_slack=max(0.0, -capacity_gap),
        complementarity_residual=complementarity,
        hedge_budget_rate=hedge_budget,
        policy_residual=policy_residual,
        price_residual=price_residual,
        load_residual=load_residual,
        iterations=inner_iterations,
        outer_evaluations=outer_evaluations,
        price_evaluations=price_evaluations,
        structural_capacity_violation=structural_violation,
        final_stats=stats,
    )


def _failed_solution(state, status, reason, grid_clamped, primary, target, capacity):
    zero_stats = {
        cls: {
            action: ActionStats(0.0, 0.0, 0.0, 0.0, 0.0) for action in _ACTIONS
        }
        for cls in _CLASSES
    }
    gap = primary - target
    return StatePolicySolution(
        state=state,
        status=status,
        reason=reason,
        grid_clamped=grid_clamped,
        probabilities={
            cls: {ProtectionAction.NORMAL: 1.0, ProtectionAction.DELAYED_HEDGE: 0.0, ProtectionAction.IMMEDIATE_HEDGE: 0.0}
            for cls in _CLASSES
        },
        price=0.0,
        rho=primary / capacity,
        primary_work_rate=primary,
        hedge_work_rate=0.0,
        replay_work_rate=0.0,
        total_work_rate=primary,
        target_work_rate=target,
        capacity_gap=gap,
        capacity_violation=max(0.0, gap),
        capacity_slack=max(0.0, -gap),
        complementarity_residual=0.0,
        hedge_budget_rate=max(0.0, target - primary),
        policy_residual=math.inf,
        price_residual=0.0,
        load_residual=math.inf,
        iterations=0,
        outer_evaluations=0,
        price_evaluations=(),
        structural_capacity_violation=False,
        final_stats=zero_stats,
    )


def _boundary_result(config, params, stats, price: float) -> _InnerResult:
    """Direct soft best response at the pinned capacity boundary."""
    probabilities = {
        cls: _soft_best_response(params, stats, price, _ACTIONS, cls)
        for cls in _CLASSES
    }
    primary, hedge, replay = _work_rates(config, probabilities, stats)
    return _InnerResult(
        probabilities=probabilities,
        rho=config.target_max_utilization,
        stats=stats,
        primary=primary,
        hedge=hedge,
        replay=replay,
        total=primary + hedge + replay,
        iterations=0,
        converged=True,
    )


def _minimum_boundary_result(config, stats) -> _InnerResult:
    """Hard per-class resource minimum used only as an infeasibility witness."""
    probabilities = {}
    for cls in _CLASSES:
        action = min(
            _ACTIONS,
            key=lambda candidate: (
                stats[cls][candidate].expected_hedge_work
                + stats[cls][candidate].expected_replay_work,
                _ACTIONS.index(candidate),
            ),
        )
        probabilities[cls] = {
            candidate: 1.0 if candidate is action else 0.0
            for candidate in _ACTIONS
        }
    primary, hedge, replay = _work_rates(config, probabilities, stats)
    return _InnerResult(
        probabilities=probabilities,
        rho=config.target_max_utilization,
        stats=stats,
        primary=primary,
        hedge=hedge,
        replay=replay,
        total=primary + hedge + replay,
        iterations=0,
        converged=True,
    )


def _solve_hd(config, table, params, state) -> StatePolicySolution:
    capacity = state_capacity(config, state)
    primary = arrival_rate_of(config) * config.healthy_service_mean
    target = config.target_max_utilization * capacity
    try:
        base = _inner_solve(config, table, params, state, 0.0)
    except _GridClamped:
        return _failed_solution(state, SolverStatus.NONCONVERGED, "grid_clamped", True, primary, target, capacity)
    if not base.converged:
        return _failed_solution(state, SolverStatus.NONCONVERGED, "inner_iteration_cap", False, primary, target, capacity)

    gap0 = base.total - target
    if gap0 <= params.capacity_tolerance:
        evaluations = ((0.0, gap0),)
        solution = _finalize(
            config, table, params, state, base, 0.0,
            SolverStatus.CONVERGED, "price_zero_slack", False,
            base.iterations, 1, evaluations, False,
        )
        violations = validate_capacity_acceptance(
            solution.price, solution.capacity_gap,
            solution.policy_residual, solution.load_residual,
            capacity_tolerance=params.capacity_tolerance,
            complementarity_tolerance=params.complementarity_tolerance,
            residual_tolerance=params.tolerance,
        )
        if violations:
            return _failed_solution(state, SolverStatus.NONCONVERGED, "acceptance_failed:" + ",".join(violations), False, primary, target, capacity)
        return solution

    # Capacity binds. Query the calibration table once at the boundary; every
    # price candidate below reuses these fixed statistics.
    try:
        boundary_stats = _query_stats(
            table,
            state,
            config.target_max_utilization,
            _ACTIONS,
        )
    except _GridClamped:
        return _failed_solution(state, SolverStatus.NONCONVERGED, "grid_clamped", True, primary, target, capacity)

    minimum = _minimum_boundary_result(config, boundary_stats)
    if minimum.total > target + params.capacity_tolerance:
        return _finalize(
            config,
            table,
            params,
            state,
            minimum,
            0.0,
            SolverStatus.INFEASIBLE,
            "boundary_minimum_exceeds_capacity",
            False,
            base.iterations,
            0,
            (),
            False,
            stats_override=boundary_stats,
        )

    boundary_zero = _boundary_result(config, params, boundary_stats, 0.0)
    boundary_gap0 = boundary_zero.total - target
    evaluations: list[tuple[float, float]] = [(0.0, boundary_gap0)]
    if boundary_gap0 < -params.capacity_tolerance:
        return _failed_solution(
            state,
            SolverStatus.NONCONVERGED,
            "binding_branch_inconsistent",
            False,
            primary,
            target,
            capacity,
        )
    if abs(boundary_gap0) <= params.capacity_tolerance:
        solution = _finalize(
            config,
            table,
            params,
            state,
            boundary_zero,
            0.0,
            SolverStatus.CONVERGED,
            "capacity_boundary_root",
            False,
            base.iterations,
            1,
            tuple(evaluations),
            False,
            stats_override=boundary_stats,
        )
        return solution

    # Signed bracket expansion over the direct boundary response.
    low = 0.0
    high = params.price_step
    high_gap = math.inf
    for _ in range(params.bracket_max_doublings):
        high_result = _boundary_result(config, params, boundary_stats, high)
        high_gap = high_result.total - target
        evaluations.append((high, high_gap))
        if high_gap <= 0.0:
            break
        low = high
        high *= 2.0
    if high_gap > 0.0:
        return _failed_solution(state, SolverStatus.NONCONVERGED, "bracket_failed", False, primary, target, capacity)

    # Bisection on the signed capacity gap.
    for _ in range(params.bisection_max_iterations):
        mid = (low + high) / 2.0
        result = _boundary_result(config, params, boundary_stats, mid)
        gap = result.total - target
        evaluations.append((mid, gap))
        if abs(gap) <= params.capacity_tolerance and abs(mid * gap) <= params.complementarity_tolerance:
            solution = _finalize(
                config, table, params, state, result, mid,
                SolverStatus.CONVERGED, "capacity_boundary_root", False,
                base.iterations, len(evaluations), tuple(evaluations), False,
                price_residual=(high - low) / 2.0,
                stats_override=boundary_stats,
            )
            violations = validate_capacity_acceptance(
                solution.price, solution.capacity_gap,
                solution.policy_residual, solution.load_residual,
                capacity_tolerance=params.capacity_tolerance,
                complementarity_tolerance=params.complementarity_tolerance,
                residual_tolerance=params.tolerance,
            )
            if violations:
                return _failed_solution(state, SolverStatus.NONCONVERGED, "acceptance_failed:" + ",".join(violations), False, primary, target, capacity)
            if solution.hedge_work_rate > solution.hedge_budget_rate + params.capacity_tolerance:
                return _failed_solution(state, SolverStatus.NONCONVERGED, "hedge_budget_exceeded", False, primary, target, capacity)
            return solution
        if gap > 0.0:
            low = mid
        else:
            high = mid
    return _failed_solution(state, SolverStatus.NONCONVERGED, "outer_iteration_cap", False, primary, target, capacity)


def _solve_forced_normal(config, table, params) -> StatePolicySolution:
    state = CommonState.FAILED
    capacity = state_capacity(config, state)
    primary = arrival_rate_of(config) * config.healthy_service_mean
    target = config.target_max_utilization * capacity
    probabilities = {
        cls: {ProtectionAction.NORMAL: 1.0, ProtectionAction.DELAYED_HEDGE: 0.0, ProtectionAction.IMMEDIATE_HEDGE: 0.0}
        for cls in _CLASSES
    }
    rho = primary / capacity
    iterations = 0
    try:
        for iteration in range(1, params.inner_max_iterations + 1):
            iterations = iteration
            stats = _query_stats(table, state, rho, (ProtectionAction.NORMAL,))
            _, _, replay = _work_rates(config, probabilities, stats)
            total = primary + replay
            rho_candidate = total / capacity
            load_residual = abs(rho_candidate - rho)
            rho = rho_candidate
            if load_residual < params.tolerance:
                break
        else:
            return _failed_solution(state, SolverStatus.NONCONVERGED, "inner_iteration_cap", False, primary, target, capacity)
    except _GridClamped:
        return _failed_solution(state, SolverStatus.NONCONVERGED, "grid_clamped", True, primary, target, capacity)

    result = _InnerResult(
        probabilities=probabilities, rho=rho, stats=stats,
        primary=primary, hedge=0.0, replay=replay, total=primary + replay,
        iterations=iterations, converged=True,
    )
    gap = result.total - target
    return _finalize(
        config, table, params, state, result, 0.0,
        SolverStatus.FORCED_NORMAL, "forced_normal_state", False,
        iterations, 1, ((0.0, gap),), gap > params.capacity_tolerance,
    )


def solve_mfg(
    config: ExperimentConfig,
    table: CalibrationTable,
    parameters: SolverParameters | None = None,
) -> MFGSolution:
    """Solve each Common State independently; R reuses H (never solved)."""
    if config.replicas_per_expert != 2:
        raise ValueError("this slice fixes exactly two Replicas")
    params = parameters if parameters is not None else SolverParameters.from_config(config)
    solutions = {
        CommonState.HEALTHY: _solve_hd(config, table, params, CommonState.HEALTHY),
        CommonState.DEGRADED: _solve_hd(config, table, params, CommonState.DEGRADED),
        CommonState.FAILED: _solve_forced_normal(config, table, params),
    }
    return MFGSolution(
        solutions=solutions,
        parameters=params,
        class_weights=_class_weights(config),
    )
