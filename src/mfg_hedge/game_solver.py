"""Stage 4 bounded symmetric pure-strategy Pi256 solver.

This module is intentionally a small state machine around the completed
Stage 3 finite evaluator.  The default evaluator reruns every candidate from
the same immutable scenarios, so the shared pool is endogenous on every
update.  The optional evaluator hook exists only for deterministic state
machine tests; it is not a second physics implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
import time
from typing import Callable, Iterable, Mapping

from .expert_game import (
    ExpertEpisodeRun,
    score_expert,
    score_population,
    shared_trace_fingerprint,
)
from .game_deviations import (
    DeviationScenario,
    DeviationRow,
    GameRuleLike,
    _make_row,
    _policy_name,
    _require_complete,
    _rule,
    _run_profile,
    _validate_profile,
    _validate_scenarios,
)
from .game_workload import ActionRule, enumerate_action_rules


FROZEN_FIT_SCENARIO_COUNT = 32
FROZEN_VALIDATION_SCENARIO_COUNT = 64
FROZEN_START_COUNT = 3
FROZEN_MAX_ROUNDS = 32
FROZEN_CALL_BUDGET = 1_000_000
FROZEN_PRECISION_RATIO = 0.005


class SolverDynamicsStatus(str, Enum):
    PURE_FIXED_POINT = "pure_fixed_point"
    RULE_CYCLE = "rule_cycle"
    NOT_CONVERGED_32 = "not_converged_32"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EXECUTION_FAILURE = "execution_failure"


class SolverPrecisionStatus(str, Enum):
    SUFFICIENT = "sufficient"
    STATISTICS_INSUFFICIENT = "statistics_insufficient"
    SIMULTANEOUS_BOUND_PENDING = "simultaneous_bound_pending"


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _rule_sort_key(name: str) -> tuple[int, int, int, int]:
    order = {"N": 0, "D": 1, "S": 2, "X": 3}
    return tuple(order[code] for code in name)  # type: ignore[return-value]


def cluster_jackknife_se(values: Iterable[float | None]) -> float:
    """SE from exact delete-one-common-path paired estimates."""

    samples = tuple(None if value is None else float(value) for value in values)
    if any(value is None for value in samples):
        # A missing empirical cluster is an observed insufficiency, not a
        # zero-valued observation.  Preserve it as an infinite precision
        # error so the solver can retain point-estimate rows while refusing a
        # precision claim.
        return math.inf
    finite_samples = tuple(value for value in samples if value is not None)
    if any(not math.isfinite(value) for value in finite_samples):
        raise ValueError("cluster difference must be finite")
    if len(finite_samples) < 2:
        return math.inf
    count = len(finite_samples)
    mean = math.fsum(finite_samples) / count
    return math.sqrt(
        ((count - 1) / count)
        * math.fsum((value - mean) ** 2 for value in finite_samples)
    )


@dataclass(frozen=True)
class SolverRuleRow:
    """One complete candidate row retained for every solver update."""

    rule: ActionRule
    objective: float
    executed_work: float
    paired_delete_one_deltas: tuple[float | None, ...]
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rule, ActionRule):
            raise ValueError("SolverRuleRow.rule must be ActionRule")
        object.__setattr__(self, "objective", _finite(self.objective, "objective"))
        object.__setattr__(self, "executed_work", _finite(self.executed_work, "executed_work"))
        deltas = tuple(None if value is None else float(value)
                       for value in self.paired_delete_one_deltas)
        if any(value is not None and not math.isfinite(value) for value in deltas):
            raise ValueError("paired_delete_one_deltas must be finite or null")
        object.__setattr__(self, "paired_delete_one_deltas", deltas)
        if not isinstance(self.complete, bool):
            raise ValueError("SolverRuleRow.complete must be bool")

    @property
    def rule_name(self) -> str:
        return self.rule.name

    @property
    def paired_jackknife_se(self) -> float:
        return cluster_jackknife_se(self.paired_delete_one_deltas)

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule.name,
            "objective": self.objective,
            "executed_work": self.executed_work,
            "paired_delete_one_deltas": list(self.paired_delete_one_deltas),
            "paired_jackknife_se": self.paired_jackknife_se,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class BankEvaluation:
    """Validated complete Pi256 bank evaluation for one current rule."""

    current_rule: str
    rows: tuple[SolverRuleRow, ...]
    environment_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.current_rule) is not str or len(self.current_rule) != 4:
            raise ValueError("current_rule must be a four-code rule name")
        rows = tuple(self.rows)
        expected = tuple(rule.name for rule in enumerate_action_rules())
        if any(not isinstance(row, SolverRuleRow) for row in rows):
            raise ValueError("BankEvaluation rows must be SolverRuleRow values")
        names = tuple(row.rule_name for row in rows)
        if len(rows) != 256 or names != expected:
            raise ValueError("BankEvaluation must retain the canonical 256 rows")
        cluster_lengths = {len(row.paired_delete_one_deltas) for row in rows}
        if len(cluster_lengths) != 1:
            raise ValueError("all candidate rows must use the same cluster set")
        if self.current_rule not in names:
            raise ValueError("current_rule is not in the canonical bank")
        if type(self.environment_fingerprint) is not str or not self.environment_fingerprint:
            raise ValueError("environment_fingerprint must be nonempty")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class IterationRecord:
    """All auditable state produced by one best-response update."""

    round_index: int
    current_rule: str
    rule_rows: tuple[SolverRuleRow, ...]
    selected_rule: str
    runner_up_rule: str
    selected_objective: float
    selected_work: float
    selected_gap: float
    selected_runner_up_delete_one_deltas: tuple[float | None, ...]
    selected_jackknife_se: float
    max_candidate_jackknife_se: float
    precision_target: float
    precision_status: SolverPrecisionStatus
    precision_reason: str
    environment_fingerprint: str
    call_count_start: int
    call_count_end: int
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError("invalid solver iteration identity or rule rows")
        rows = tuple(self.rule_rows)
        if len(rows) != 256 or any(not isinstance(row, SolverRuleRow) for row in rows):
            raise ValueError("solver iteration must retain 256 SolverRuleRow values")
        names = {row.rule_name for row in rows}
        if self.selected_rule not in names or self.runner_up_rule not in names:
            raise ValueError("selected and runner-up rules must be retained rows")
        if self.runner_up_rule == self.selected_rule:
            raise ValueError("selected and runner-up rules must differ")
        if (type(self.current_rule) is not str
                or type(self.environment_fingerprint) is not str
                or not self.environment_fingerprint
                or type(self.precision_reason) is not str):
            raise ValueError("iteration identity and audit fields are invalid")
        if not isinstance(self.precision_status, SolverPrecisionStatus):
            raise ValueError("precision_status must be SolverPrecisionStatus")
        if self.stop_reason is not None and type(self.stop_reason) is not str:
            raise ValueError("stop_reason must be a string or None")
        if (type(self.call_count_start) is not int
                or type(self.call_count_end) is not int
                or self.call_count_start < 0
                or self.call_count_end < self.call_count_start):
            raise ValueError("iteration call counts must be monotone integers")
        if len(self.selected_runner_up_delete_one_deltas) != len(rows[0].paired_delete_one_deltas):
            raise ValueError("selected/runner-up paired differences are incomplete")
        object.__setattr__(self, "rule_rows", rows)
        for name in (
            "selected_objective", "selected_work", "selected_gap",
            "precision_target",
        ):
            _finite(getattr(self, name), name)
        if self.selected_jackknife_se < 0.0 and not math.isinf(self.selected_jackknife_se):
            raise ValueError("selected_jackknife_se must be nonnegative or infinity")
        if (self.max_candidate_jackknife_se < 0.0
                and not math.isinf(self.max_candidate_jackknife_se)):
            raise ValueError("max_candidate_jackknife_se must be nonnegative or infinity")

    @property
    def environment_fingerprint_alias(self) -> str:
        return self.environment_fingerprint

    def as_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "current_rule": self.current_rule,
            "rule_rows": [row.as_dict() for row in self.rule_rows],
            "selected_rule": self.selected_rule,
            "runner_up_rule": self.runner_up_rule,
            "selected_objective": self.selected_objective,
            "selected_work": self.selected_work,
            "selected_gap": self.selected_gap,
            "selected_runner_up_delete_one_deltas": list(
                self.selected_runner_up_delete_one_deltas
            ),
            "selected_jackknife_se": self.selected_jackknife_se,
            "max_candidate_jackknife_se": self.max_candidate_jackknife_se,
            "precision_target": self.precision_target,
            "precision_status": self.precision_status.value,
            "precision_reason": self.precision_reason,
            "environment_fingerprint": self.environment_fingerprint,
            "call_count_start": self.call_count_start,
            "call_count_end": self.call_count_end,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class SolverStartResult:
    start_rule: str
    iterations: tuple[IterationRecord, ...]
    visited_rules: tuple[str, ...]
    candidate_rule: str
    dynamics_status: SolverDynamicsStatus
    precision_status: SolverPrecisionStatus
    stop_reason: str
    cycle_first_repeat: int | None = None
    cycle_period: int | None = None

    def __post_init__(self) -> None:
        if not self.visited_rules:
            raise ValueError("a solver start must retain at least one visited rule")
        if self.dynamics_status is SolverDynamicsStatus.RULE_CYCLE:
            if self.cycle_first_repeat is None or self.cycle_period is None:
                raise ValueError("cycle result must retain first repeat and period")
        elif self.cycle_first_repeat is not None or self.cycle_period is not None:
            raise ValueError("non-cycle result cannot retain cycle metadata")

    @property
    def status(self) -> SolverDynamicsStatus:
        return self.dynamics_status

    def as_dict(self) -> dict[str, object]:
        return {
            "start_rule": self.start_rule,
            "iterations": [iteration.as_dict() for iteration in self.iterations],
            "visited_rules": list(self.visited_rules),
            "candidate_rule": self.candidate_rule,
            "dynamics_status": self.dynamics_status.value,
            "precision_status": self.precision_status.value,
            "stop_reason": self.stop_reason,
            "cycle_first_repeat": self.cycle_first_repeat,
            "cycle_period": self.cycle_period,
        }


@dataclass(frozen=True)
class Stage4SolverResult:
    starts: tuple[SolverStartResult, ...]
    call_count: int
    call_budget: int
    budget_exhausted: bool
    stop_reason: str
    precision_denominator: float
    claims_nash: bool = False
    claims_mfg: bool = False

    @property
    def start_count(self) -> int:
        return len(self.starts)

    @property
    def candidate_rules(self) -> tuple[str, ...]:
        return tuple(start.candidate_rule for start in self.starts)

    @property
    def remaining_budget(self) -> int:
        return self.call_budget - self.call_count

    def as_dict(self) -> dict[str, object]:
        return {
            "starts": [start.as_dict() for start in self.starts],
            "call_count": self.call_count,
            "call_budget": self.call_budget,
            "remaining_budget": self.remaining_budget,
            "budget_exhausted": self.budget_exhausted,
            "stop_reason": self.stop_reason,
            "precision_denominator": self.precision_denominator,
            "claims_nash": self.claims_nash,
            "claims_mfg": self.claims_mfg,
        }


@dataclass(frozen=True)
class Stage4Preflight:
    fit_scenario_count: int
    validation_scenario_count: int
    expert_count: int
    starts: int
    max_rounds: int
    fit_calls: int
    validation_calls: int
    total_calls: int
    call_budget: int
    remaining_budget: int
    within_budget: bool
    measured_seconds_per_call: float | None = None
    estimated_seconds: float | None = None


@dataclass(frozen=True)
class SchedulerTiming:
    runtime_seconds: float
    scheduler_calls: int
    trace_fingerprint: str
    fault_fingerprint: str


@dataclass(frozen=True)
class FrozenValidationResult:
    candidate_rule: str
    rows: tuple[DeviationRow, ...]
    call_count: int
    call_budget: int
    expert_count: int
    all_experts: bool
    complete: bool
    budget_exhausted: bool
    precision_status: SolverPrecisionStatus
    fit_trace_fingerprints: tuple[str, ...]
    validation_trace_fingerprints: tuple[str, ...]
    fit_fault_fingerprints: tuple[str, ...]
    validation_fault_fingerprints: tuple[str, ...]
    claims_nash: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)


class _BudgetExceeded(RuntimeError):
    pass


class _CallBudget:
    def __init__(self, limit: int):
        if type(limit) is not int or limit <= 0:
            raise ValueError("call_budget must be a positive int")
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise _BudgetExceeded("Stage 4 scheduler call budget exhausted")
        self.used += 1


def _validate_count(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def preflight_stage4(
    *,
    fit_scenario_count: int = FROZEN_FIT_SCENARIO_COUNT,
    validation_scenario_count: int = FROZEN_VALIDATION_SCENARIO_COUNT,
    expert_count: int = 8,
    starts: int = FROZEN_START_COUNT,
    max_rounds: int = FROZEN_MAX_ROUNDS,
    call_budget: int = FROZEN_CALL_BUDGET,
    measured_seconds_per_call: float | None = None,
) -> Stage4Preflight:
    """Close the call arithmetic before any fit/validation execution."""

    fit_count = _validate_count(fit_scenario_count, "fit_scenario_count")
    validation_count = _validate_count(validation_scenario_count, "validation_scenario_count")
    n_experts = _validate_count(expert_count, "expert_count", minimum=1)
    n_starts = _validate_count(starts, "starts", minimum=1)
    rounds = _validate_count(max_rounds, "max_rounds", minimum=1)
    budget = _validate_count(call_budget, "call_budget", minimum=1)
    fit_calls = n_starts * rounds * fit_count * 257
    validation_calls = validation_count * n_experts * 257
    total = fit_calls + validation_calls
    measured = None
    estimated = None
    if measured_seconds_per_call is not None:
        measured = _finite(measured_seconds_per_call, "measured_seconds_per_call", positive=True)
        estimated = measured * total
    return Stage4Preflight(
        fit_scenario_count=fit_count,
        validation_scenario_count=validation_count,
        expert_count=n_experts,
        starts=n_starts,
        max_rounds=rounds,
        fit_calls=fit_calls,
        validation_calls=validation_calls,
        total_calls=total,
        call_budget=budget,
        remaining_budget=budget - total,
        within_budget=total <= budget,
        measured_seconds_per_call=measured,
        estimated_seconds=estimated,
    )


def measure_scheduler_timing(
    scenario: DeviationScenario,
    *,
    profile: Mapping[int, GameRuleLike],
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
) -> SchedulerTiming:
    """Measure one complete scheduler call without producing an artifact."""

    scenarios = _validate_scenarios((scenario,))
    profile_map = _validate_profile(profile, scenarios[0].trace.expert_count)
    started = time.perf_counter()
    _run_profile(
        scenarios, profile_map, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    elapsed = time.perf_counter() - started
    return SchedulerTiming(
        runtime_seconds=elapsed,
        scheduler_calls=1,
        trace_fingerprint=shared_trace_fingerprint(scenario.trace),
        fault_fingerprint=scenario.fault_fingerprint,
    )


def _environment_fingerprint(
    current_rule: ActionRule,
    runs: tuple[ExpertEpisodeRun, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(current_rule.name.encode("ascii"))
    for run in runs:
        digest.update(run.episode_key.encode("utf-8"))
        digest.update(run.fault_fingerprint.encode("ascii"))
        digest.update(shared_trace_fingerprint(run.trace).encode("ascii"))
        for interval in run.result.pool_intervals:
            digest.update(repr(interval).encode("ascii"))
    return digest.hexdigest()


def _cluster_runs(runs: tuple[ExpertEpisodeRun, ...]) -> tuple[tuple[str, tuple[ExpertEpisodeRun, ...]], ...]:
    grouped: dict[str, list[ExpertEpisodeRun]] = {}
    for run in runs:
        grouped.setdefault(run.fault_fingerprint, []).append(run)
    return tuple(
        (key, tuple(sorted(value, key=lambda run: run.episode_key)))
        for key, value in sorted(grouped.items())
    )


def _delete_one_scores(
    runs: tuple[ExpertEpisodeRun, ...], expert_id: int,
) -> tuple[tuple[str, float | None], ...]:
    clusters = _cluster_runs(runs)
    return tuple(
        (
            omitted,
            (score_expert(retained, expert_id).total if retained else None),
        )
        for omitted, _group in clusters
        for retained in (
            tuple(run for key, group in clusters if key != omitted for run in group),
        )
    )


def _validate_bank_evaluation(evaluation: BankEvaluation, current: ActionRule) -> None:
    if evaluation.current_rule != current.name:
        raise ValueError("bank evaluator current rule mismatch")
    if any(not row.complete for row in evaluation.rows):
        raise ValueError("incomplete candidate row cannot enter solver selection")


def _rank_rows(rows: tuple[SolverRuleRow, ...]) -> tuple[SolverRuleRow, ...]:
    return tuple(sorted(
        rows,
        key=lambda row: (
            row.objective, row.executed_work, _rule_sort_key(row.rule_name)
        ),
    ))


def _iteration_record(
    evaluation: BankEvaluation,
    *,
    round_index: int,
    call_count_start: int,
    call_count_end: int,
    precision_denominator: float,
    precision_ratio: float,
) -> IterationRecord:
    ranked = _rank_rows(evaluation.rows)
    selected, runner_up = ranked[0], ranked[1]
    if len(selected.paired_delete_one_deltas) != len(runner_up.paired_delete_one_deltas):
        raise ValueError("best and runner-up rows do not share cluster observations")
    paired = tuple(
        (None if best is None or runner is None else best - runner)
        for best, runner in zip(
            selected.paired_delete_one_deltas, runner_up.paired_delete_one_deltas
        )
    )
    selected_se = cluster_jackknife_se(paired)
    max_candidate_se = max(row.paired_jackknife_se for row in ranked)
    target = precision_ratio * precision_denominator
    # Keep the comparison explicit: this is the paired best-versus-runner-up
    # cluster difference, never sqrt(SE_best^2 + SE_runner_up^2).
    sufficient = (
        math.isfinite(target)
        and target > 0.0
        and math.isfinite(max_candidate_se)
        and max_candidate_se <= target
        and math.isfinite(selected_se)
        and selected.objective < runner_up.objective - 2.0 * selected_se
    )
    return IterationRecord(
        round_index=round_index,
        current_rule=evaluation.current_rule,
        rule_rows=evaluation.rows,
        selected_rule=selected.rule_name,
        runner_up_rule=runner_up.rule_name,
        selected_objective=selected.objective,
        selected_work=selected.executed_work,
        selected_gap=runner_up.objective - selected.objective,
        selected_runner_up_delete_one_deltas=paired,
        selected_jackknife_se=selected_se,
        max_candidate_jackknife_se=max_candidate_se,
        precision_target=target,
        precision_status=(
            SolverPrecisionStatus.SUFFICIENT
            if sufficient else SolverPrecisionStatus.STATISTICS_INSUFFICIENT
        ),
        precision_reason="ok" if sufficient else "paired_jackknife_se_or_gap",
        environment_fingerprint=evaluation.environment_fingerprint,
        call_count_start=call_count_start,
        call_count_end=call_count_end,
    )


def _real_bank_evaluator(
    scenarios: tuple[DeviationScenario, ...],
    *,
    expert_count: int,
    c_b: float,
    degraded_slowdown: float,
    hedge_delay: float,
    budget: _CallBudget,
) -> Callable[[ActionRule], BankEvaluation]:
    bank = enumerate_action_rules()
    representative = 0

    def evaluate(current: ActionRule) -> BankEvaluation:
        current_profile = {expert: current for expert in range(expert_count)}
        base_runs = _run_profile(
            scenarios, current_profile, c_b=c_b,
            degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
            call_observer=budget.consume,
        )
        base_population = score_population(base_runs)
        _require_complete(base_population)
        rows = []
        base_delete_one = dict(_delete_one_scores(base_runs, representative))
        for candidate in bank:
            candidate_profile = dict(current_profile)
            candidate_profile[representative] = candidate
            candidate_runs = _run_profile(
                scenarios, candidate_profile, c_b=c_b,
                degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
                call_observer=budget.consume,
            )
            candidate_population = score_population(candidate_runs)
            _require_complete(candidate_population)
            candidate_score = candidate_population.expert_scores[representative]
            candidate_delete_one = dict(
                _delete_one_scores(candidate_runs, representative)
            )
            deltas = []
            for key, base_value in base_delete_one.items():
                candidate_value = candidate_delete_one[key]
                deltas.append(
                    None if base_value is None or candidate_value is None
                    else candidate_value - base_value
                )
            if candidate_score.total is None:
                raise ValueError("candidate scorer returned incomplete objective")
            rows.append(SolverRuleRow(
                rule=candidate,
                objective=candidate_score.total,
                executed_work=candidate_score.raw_work_totals["executed"],
                paired_delete_one_deltas=tuple(deltas),
            ))
        return BankEvaluation(
            current_rule=current.name,
            rows=tuple(rows),
            environment_fingerprint=_environment_fingerprint(current, base_runs),
        )

    return evaluate


def solve_symmetric_pi256(
    *,
    fit_scenarios: Iterable[DeviationScenario] | None = None,
    expert_count: int = 8,
    starts: Iterable[ActionRule] | None = None,
    max_rounds: int = FROZEN_MAX_ROUNDS,
    precision_denominator: float | None = None,
    precision_ratio: float = FROZEN_PRECISION_RATIO,
    call_budget: int = FROZEN_CALL_BUDGET,
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    candidate_evaluator: Callable[[ActionRule], BankEvaluation] | None = None,
) -> Stage4SolverResult:
    """Run the bounded symmetric pure-rule state machine."""

    n_experts = _validate_count(expert_count, "expert_count", minimum=1)
    rounds = _validate_count(max_rounds, "max_rounds", minimum=1)
    ratio = _finite(precision_ratio, "precision_ratio", positive=True)
    denominator = (
        _finite(precision_denominator, "precision_denominator", positive=True)
        if precision_denominator is not None else None
    )
    budget = _CallBudget(call_budget)
    bank = enumerate_action_rules()
    bank_names = {rule.name for rule in bank}
    if starts is None:
        start_rules = (_rule("NNNN"), _rule("NSSN"), _rule("XXXX"))
    else:
        start_rules = tuple(starts)
    if not start_rules:
        raise ValueError("starts must be nonempty")
    start_names = tuple(_policy_name(rule) for rule in start_rules)
    if len(set(start_names)) != len(start_names):
        raise ValueError("starts must be distinct")
    if any(name not in bank_names for name in start_names):
        raise ValueError("starts must be canonical Pi256 rules")

    scenarios = None
    if candidate_evaluator is None:
        if fit_scenarios is None:
            raise ValueError("real solver requires fit_scenarios")
        scenarios = _validate_scenarios(fit_scenarios)
        if any(s.trace.expert_count != n_experts for s in scenarios):
            raise ValueError("fit scenario Expert count does not match solver")
        evaluator = _real_bank_evaluator(
            scenarios, expert_count=n_experts, c_b=c_b,
            degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
            budget=budget,
        )
    else:
        evaluator = candidate_evaluator

    results = []
    global_precision_denominator = denominator
    global_budget_exhausted = False
    for start in start_rules:
        current = start
        visited: list[str] = []
        seen: dict[str, int] = {}
        iterations: list[IterationRecord] = []
        precision_statuses: list[SolverPrecisionStatus] = []
        dynamics_status = SolverDynamicsStatus.NOT_CONVERGED_32
        stop_reason = "not_converged_32"
        candidate_name = current.name
        cycle_first = None
        cycle_period = None
        for round_index in range(rounds):
            seen[current.name] = len(visited)
            visited.append(current.name)
            call_start = budget.used
            try:
                evaluation = evaluator(current)
                _validate_bank_evaluation(evaluation, current)
                if global_precision_denominator is None:
                    nnnn = next(row for row in evaluation.rows if row.rule_name == "NNNN")
                    global_precision_denominator = nnnn.objective
                    _finite(global_precision_denominator, "precision_denominator", positive=True)
                record = _iteration_record(
                    evaluation,
                    round_index=round_index,
                    call_count_start=call_start,
                    call_count_end=budget.used,
                    precision_denominator=global_precision_denominator,
                    precision_ratio=ratio,
                )
            except _BudgetExceeded:
                global_budget_exhausted = True
                dynamics_status = SolverDynamicsStatus.BUDGET_EXHAUSTED
                stop_reason = "call_budget_exhausted"
                results.append(SolverStartResult(
                    start_rule=start.name,
                    iterations=tuple(iterations),
                    visited_rules=tuple(visited),
                    candidate_rule=candidate_name,
                    dynamics_status=dynamics_status,
                    precision_status=SolverPrecisionStatus.STATISTICS_INSUFFICIENT,
                    stop_reason=stop_reason,
                ))
                for remaining in start_rules[len(results):]:
                    results.append(SolverStartResult(
                        start_rule=remaining.name,
                        iterations=(),
                        visited_rules=(remaining.name,),
                        candidate_rule=remaining.name,
                        dynamics_status=SolverDynamicsStatus.BUDGET_EXHAUSTED,
                        precision_status=SolverPrecisionStatus.STATISTICS_INSUFFICIENT,
                        stop_reason=stop_reason,
                    ))
                break
            iterations.append(record)
            precision_statuses.append(record.precision_status)
            candidate_name = record.selected_rule
            selected = next(row.rule for row in record.rule_rows
                            if row.rule_name == record.selected_rule)
            if record.selected_rule == current.name:
                dynamics_status = SolverDynamicsStatus.PURE_FIXED_POINT
                stop_reason = "pure_fixed_point"
                iterations[-1] = replace(record, stop_reason=stop_reason)
                break
            if record.selected_rule in seen:
                dynamics_status = SolverDynamicsStatus.RULE_CYCLE
                stop_reason = "rule_cycle"
                cycle_first = seen[record.selected_rule]
                cycle_period = len(visited) - cycle_first
                iterations[-1] = replace(record, stop_reason=stop_reason)
                break
            current = selected
        else:
            dynamics_status = SolverDynamicsStatus.NOT_CONVERGED_32
            stop_reason = "not_converged_32"
            iterations[-1] = replace(iterations[-1], stop_reason=stop_reason)
        if global_budget_exhausted:
            break
        aggregate_precision = (
            SolverPrecisionStatus.STATISTICS_INSUFFICIENT
            if any(status is SolverPrecisionStatus.STATISTICS_INSUFFICIENT
                   for status in precision_statuses)
            else SolverPrecisionStatus.SUFFICIENT
        )
        results.append(SolverStartResult(
            start_rule=start.name,
            iterations=tuple(iterations),
            visited_rules=tuple(visited),
            candidate_rule=candidate_name,
            dynamics_status=dynamics_status,
            precision_status=aggregate_precision,
            stop_reason=stop_reason,
            cycle_first_repeat=cycle_first,
            cycle_period=cycle_period,
        ))

    if global_precision_denominator is None:
        global_precision_denominator = 1.0
    return Stage4SolverResult(
        starts=tuple(results),
        call_count=budget.used,
        call_budget=budget.limit,
        budget_exhausted=global_budget_exhausted,
        stop_reason=("call_budget_exhausted" if global_budget_exhausted else "completed"),
        precision_denominator=global_precision_denominator,
    )


def validate_frozen_candidate(
    *,
    candidate_rule: ActionRule,
    fit_scenarios: Iterable[DeviationScenario],
    validation_scenarios: Iterable[DeviationScenario],
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    call_budget: int = FROZEN_CALL_BUDGET,
) -> FrozenValidationResult:
    """Run all Expert/all-256 deviations on a disjoint frozen validation set."""

    if not isinstance(candidate_rule, ActionRule):
        raise ValueError("candidate_rule must be ActionRule")
    fit = _validate_scenarios(fit_scenarios)
    validation = _validate_scenarios(validation_scenarios)
    expert_count = validation[0].trace.expert_count
    if any(s.trace.expert_count != expert_count for s in fit + validation):
        raise ValueError("fit/validation Expert counts do not match")
    fit_trace_fingerprints = tuple(shared_trace_fingerprint(s.trace) for s in fit)
    validation_trace_fingerprints = tuple(
        shared_trace_fingerprint(s.trace) for s in validation
    )
    fit_fault_fingerprints = tuple(s.fault_fingerprint for s in fit)
    validation_fault_fingerprints = tuple(s.fault_fingerprint for s in validation)
    if (set(fit_trace_fingerprints) & set(validation_trace_fingerprints)
            or set(fit_fault_fingerprints) & set(validation_fault_fingerprints)):
        raise ValueError("fit and validation trace/fault fingerprints overlap")

    bank = enumerate_action_rules()
    budget = _CallBudget(call_budget)
    rows = []
    budget_exhausted = False
    for expert_id in range(expert_count):
        base_profile = {expert: candidate_rule for expert in range(expert_count)}
        try:
            base_runs = _run_profile(
                validation, base_profile, c_b=c_b,
                degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
                call_observer=budget.consume,
            )
            base_population = score_population(base_runs)
            _require_complete(base_population)
            for candidate in bank:
                candidate_profile = dict(base_profile)
                candidate_profile[expert_id] = candidate
                candidate_runs = _run_profile(
                    validation, candidate_profile, c_b=c_b,
                    degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
                    call_observer=budget.consume,
                )
                candidate_population = score_population(candidate_runs)
                _require_complete(candidate_population)
                rows.append(_make_row(
                    validation, base_runs, base_population, candidate_runs,
                    candidate_population, expert_id, candidate_rule.name,
                    candidate.name,
                ))
        except _BudgetExceeded:
            budget_exhausted = True
            break
    rows = tuple(sorted(rows, key=lambda row: (
        row.deviating_expert, _rule_sort_key(row.candidate_rule)
    )))
    complete = (
        not budget_exhausted
        and len(rows) == expert_count * 256
        and all(row.complete for row in rows)
    )
    all_experts = (
        not budget_exhausted
        and {row.deviating_expert for row in rows} == set(range(expert_count))
    )
    return FrozenValidationResult(
        candidate_rule=candidate_rule.name,
        rows=rows,
        call_count=budget.used,
        call_budget=budget.limit,
        expert_count=expert_count,
        all_experts=all_experts,
        complete=complete,
        budget_exhausted=budget_exhausted,
        precision_status=SolverPrecisionStatus.SIMULTANEOUS_BOUND_PENDING,
        fit_trace_fingerprints=fit_trace_fingerprints,
        validation_trace_fingerprints=validation_trace_fingerprints,
        fit_fault_fingerprints=fit_fault_fingerprints,
        validation_fault_fingerprints=validation_fault_fingerprints,
    )


__all__ = [
    "BankEvaluation",
    "FROZEN_CALL_BUDGET",
    "FROZEN_FIT_SCENARIO_COUNT",
    "FROZEN_MAX_ROUNDS",
    "FROZEN_PRECISION_RATIO",
    "FROZEN_START_COUNT",
    "FROZEN_VALIDATION_SCENARIO_COUNT",
    "FrozenValidationResult",
    "IterationRecord",
    "SchedulerTiming",
    "SolverDynamicsStatus",
    "SolverPrecisionStatus",
    "SolverRuleRow",
    "SolverStartResult",
    "Stage4Preflight",
    "Stage4SolverResult",
    "cluster_jackknife_se",
    "measure_scheduler_timing",
    "preflight_stage4",
    "solve_symmetric_pi256",
    "validate_frozen_candidate",
]
