"""Frozen Stage 4 ``Pi_256`` campaign orchestration.

This module is the execution boundary above the exact scheduler and the
parallel deviation runner.  It owns no alternate physics.  Every baseline and
deviation is a fresh shared-backup simulation from an immutable scenario;
workers only reduce tagged results to scorer sufficient rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .artifacts import write_run_directory
from .campaign_execution import (
    DeviationBatchResult,
    DeviationBatchRow,
    DeviationTask,
    run_deviation_batch,
)
from .common_state import Phase
from .expert_game import (
    ExpertEpisodeRun,
    score_expert,
    score_population,
    shared_trace_fingerprint,
)
from .game_deviations import DeviationScenario, _validate_scenarios
from .game_solver import (
    FROZEN_CALL_BUDGET,
    FROZEN_FIT_SCENARIO_COUNT,
    FROZEN_MAX_ROUNDS,
    FROZEN_PRECISION_RATIO,
    FROZEN_START_COUNT,
    FROZEN_VALIDATION_SCENARIO_COUNT,
    BankEvaluation,
    IterationRecord,
    SolverDynamicsStatus,
    SolverPrecisionStatus,
    SolverRuleRow,
    SolverStartResult,
    _environment_fingerprint,
    _iteration_record,
    _rule_sort_key,
    cluster_jackknife_se,
)
from .game_workload import (
    ActionRule,
    build_common_fault_path,
    build_nested_population,
    enumerate_action_rules,
)
from .shared_backup import (
    SharedAction,
    simulate_shared_backup_optimized,
)


FIT_CALLS = FROZEN_START_COUNT * FROZEN_MAX_ROUNDS * FROZEN_FIT_SCENARIO_COUNT * 257
VALIDATION_CALLS = FROZEN_VALIDATION_SCENARIO_COUNT * 8 * 257
TOTAL_PLANNED_CALLS = FIT_CALLS + VALIDATION_CALLS
HARD_CALL_LIMIT = 1_000_000
VALIDATION_REGRET_RATIO = 0.01
SIMULTANEOUS_BOOTSTRAP_SEED = 20260906
SIMULTANEOUS_BOOTSTRAP_REPLICATES = 4096


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_set_fingerprint(root: Path, relative_paths: Sequence[str]) -> tuple[list[dict[str, str]], str]:
    files: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"provenance source is missing: {relative}")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(content)
    return files, aggregate.hexdigest()


def _stage4_provenance() -> dict[str, object]:
    """Build the deterministic provenance frozen before a Stage 4 run."""

    from . import __version__

    root = Path(__file__).resolve().parents[2]
    source_paths = (
        "src/mfg_hedge/shared_backup.py",
        "src/mfg_hedge/game_workload.py",
        "src/mfg_hedge/expert_game.py",
        "src/mfg_hedge/game_deviations.py",
        "src/mfg_hedge/game_solver.py",
        "src/mfg_hedge/campaign_execution.py",
        "src/mfg_hedge/stage4_campaign.py",
    )
    source_files, source_aggregate = _file_set_fingerprint(root, source_paths)

    config_path = root / "configs/v1_minimal.json"
    if not config_path.is_file():
        raise RuntimeError("Stage 4 provenance configuration is missing")
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    physical = {
        "expert_count": 8,
        "replicas_per_expert": 2,
        "c_B": 0.5,
        "degraded_slowdown": 2.0,
        "hedge_delay": 1.5,
        "tariff": 0.0,
        "action_alphabet": "NDSX",
        "canonical_rule_count": 256,
    }

    protocol_paths = (
        "docs/adr/0015-shared-backup-expert-game.md",
        "docs/adr/0016-finite-256-rule-game-solver-protocol.md",
        "docs/adr/0018-pooled-objective-cluster-inference.md",
        ".scratch/shared-backup-game/spec.md",
    )
    protocol_files, protocol_fingerprint = _file_set_fingerprint(
        root, protocol_paths
    )
    provenance: dict[str, object] = {
        "schema_version": "stage4-provenance-v1",
        "package_version": __version__,
        "source": {
            "files": source_files,
            "aggregate_sha256": source_aggregate,
        },
        "configuration": {
            "path": "configs/v1_minimal.json",
            "sha256": _file_sha256(config_path),
            "raw": configuration,
            "physics": physical,
        },
        "inference": {
            "protocol_files": protocol_files,
            "protocol_fingerprint": protocol_fingerprint,
            "objective": "pooled_nonlinear_objective",
            "delete_one_unit": "one_common_fault_path",
            "jackknife": "paired_common_path_pseudo_values",
            "max_t_pivot": "(pseudo_mean-bootstrap_mean)/standard_error",
            "confidence_level": 0.95,
            "bootstrap_seed": SIMULTANEOUS_BOOTSTRAP_SEED,
            "bootstrap_replicates": SIMULTANEOUS_BOOTSTRAP_REPLICATES,
            "fit_precision_ratio": FROZEN_PRECISION_RATIO,
            "validation_regret_ratio": VALIDATION_REGRET_RATIO,
            "fit_calls": FIT_CALLS,
            "validation_calls": VALIDATION_CALLS,
            "hard_call_limit": HARD_CALL_LIMIT,
        },
        "attempt_binding": {
            "single": "default_destination:attempt2",
            "dual_attempts": [
                "default_destination:attempt2",
                "other:attempt3",
            ],
        },
        "event_order": [
            "health_state_change",
            "A_failure_invalidation_replay_enqueue",
            "valid_completion_batch",
            "hedge_timer",
            "token_arrival",
            "dispatch_start",
            "shared_speed_completion_schedule",
        ],
    }
    provenance["provenance_fingerprint"] = _canonical_sha256(provenance)
    return provenance


def _finite(value: object, name: str, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or (not allow_negative and number < 0.0):
        raise ValueError(f"{name} must be finite")
    return number


def _rule_key(rule: ActionRule) -> tuple[int, int, int, int]:
    return _rule_sort_key(rule.name)


def build_frozen_stage4_scenarios(
    namespace: str,
    macro_seed: int,
    *,
    common_path_count: int,
    population_ids: Sequence[int | str] = (0, 1),
    expert_count: int = 8,
) -> tuple[DeviationScenario, ...]:
    """Build the declared nested scenarios in deterministic key order."""

    if namespace not in (
        "shared-backup-game:v1:stage4-fit",
        "shared-backup-game:v1:stage4-validation",
    ):
        raise ValueError("Stage 4 namespace is not frozen")
    if type(macro_seed) is not int or macro_seed < 0:
        raise ValueError("macro_seed must be a non-negative int")
    if type(common_path_count) is not int or common_path_count < 1:
        raise ValueError("common_path_count must be positive")
    if tuple(population_ids) != (0, 1):
        raise ValueError("Stage 4 requires population IDs 0 and 1")
    if type(expert_count) is not int or expert_count != 8:
        raise ValueError("Stage 4 requires N=8")

    scenarios: list[DeviationScenario] = []
    for common_id in range(common_path_count):
        fault = build_common_fault_path(namespace, macro_seed, common_id)
        for population_id in (0, 1):
            population = build_nested_population(fault, population_id, expert_count)
            scenarios.append(DeviationScenario(
                episode_key=(
                    f"{namespace}:common-{common_id:02d}:population-{population_id}"
                ),
                trace=population,
                fault_fingerprint=population.common_fault_fingerprint,
            ))
    result = _validate_scenarios(scenarios)
    if len(result) != common_path_count * 2:
        raise ValueError("frozen scenario count mismatch")
    return result


def _scenario_fingerprints(
    scenarios: Sequence[DeviationScenario],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    traces = tuple(
        shared_trace_fingerprint(scenario.trace)
        for scenario in scenarios
    )
    faults = tuple(str(scenario.fault_fingerprint) for scenario in scenarios)
    return traces, faults


@dataclass(frozen=True)
class FrozenStage4Plan:
    fit_scenario_count: int
    validation_scenario_count: int
    expert_count: int
    starts: int
    max_rounds: int
    fit_calls: int
    validation_calls: int
    total_calls: int
    call_budget: int
    within_budget: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "fit_scenario_count": self.fit_scenario_count,
            "validation_scenario_count": self.validation_scenario_count,
            "expert_count": self.expert_count,
            "starts": self.starts,
            "max_rounds": self.max_rounds,
            "fit_calls": self.fit_calls,
            "validation_calls": self.validation_calls,
            "total_calls": self.total_calls,
            "call_budget": self.call_budget,
            "within_budget": self.within_budget,
        }


def preflight_frozen_stage4(
    *,
    fit_common_path_count: int = 16,
    validation_common_path_count: int = 32,
    expert_count: int = 8,
    starts: int = 3,
    max_rounds: int = 32,
    call_budget: int = FROZEN_CALL_BUDGET,
) -> FrozenStage4Plan:
    """Validate the exact frozen arithmetic before dispatching any call."""

    values = {
        "fit_common_path_count": fit_common_path_count,
        "validation_common_path_count": validation_common_path_count,
        "expert_count": expert_count,
        "starts": starts,
        "max_rounds": max_rounds,
        "call_budget": call_budget,
    }
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError("frozen Stage 4 plan values must be positive ints")
    if (fit_common_path_count, validation_common_path_count, expert_count,
            starts, max_rounds) != (16, 32, 8, 3, 32):
        raise ValueError("Stage 4 plan cannot change frozen dimensions")
    fit_scenarios = fit_common_path_count * 2
    validation_scenarios = validation_common_path_count * 2
    fit_calls = starts * max_rounds * fit_scenarios * 257
    validation_calls = validation_scenarios * expert_count * 257
    total = fit_calls + validation_calls
    return FrozenStage4Plan(
        fit_scenario_count=fit_scenarios,
        validation_scenario_count=validation_scenarios,
        expert_count=expert_count,
        starts=starts,
        max_rounds=max_rounds,
        fit_calls=fit_calls,
        validation_calls=validation_calls,
        total_calls=total,
        call_budget=call_budget,
        within_budget=total <= call_budget <= HARD_CALL_LIMIT,
    )


class _RuleSource:
    def __init__(self, rule: ActionRule):
        self.rule = rule

    @property
    def name(self) -> str:
        return self.rule.name

    def request(self, observation: object) -> SharedAction:
        return self.rule.request(observation)


def _full_profile_runs(
    scenarios: tuple[DeviationScenario, ...],
    profile: Mapping[int, ActionRule],
    *,
    c_b: float,
    degraded_slowdown: float,
    hedge_delay: float,
    call_observer: Callable[[], None] | None = None,
) -> tuple[ExpertEpisodeRun, ...]:
    runs: list[ExpertEpisodeRun] = []
    for scenario in scenarios:
        sources = {
            expert_id: _RuleSource(rule)
            for expert_id, rule in profile.items()
        }
        if call_observer is not None:
            call_observer()
        result = simulate_shared_backup_optimized(
            scenario.trace,
            c_b,
            degraded_slowdown=degraded_slowdown,
            hedge_delay=hedge_delay,
            action_sources=sources,
        )
        runs.append(ExpertEpisodeRun(
            episode_key=scenario.episode_key,
            trace=scenario.trace,
            result=result,
            fault_fingerprint=scenario.fault_fingerprint,
        ))
    return tuple(runs)


def _delete_one_objectives(
    runs: tuple[ExpertEpisodeRun, ...], expert_id: int,
) -> tuple[tuple[str, float | None], ...]:
    grouped: dict[str, list[ExpertEpisodeRun]] = {}
    for run in runs:
        grouped.setdefault(run.fault_fingerprint, []).append(run)
    result: list[tuple[str, float | None]] = []
    for omitted in sorted(grouped):
        retained = tuple(
            run for fault, group in sorted(grouped.items())
            if fault != omitted for run in group
        )
        score = score_expert(retained, expert_id).total if retained else None
        result.append((omitted, score))
    return tuple(result)


def jackknife_pseudo_values(
    full_estimate: float, delete_one_estimates: Sequence[float | None],
) -> tuple[float | None, ...]:
    """Return paired common-path jackknife pseudo-values, or fail closed."""

    full = _finite(full_estimate, "full_estimate", allow_negative=True)
    values = tuple(delete_one_estimates)
    if len(values) < 2 or any(value is None for value in values):
        # Preserve the cluster shape so the inference connector can classify
        # the row as incomplete instead of failing on a length mismatch.
        return (None,) * len(values)
    finite = tuple(
        _finite(value, "delete_one_estimate", allow_negative=True)
        for value in values
    )
    count = len(finite)
    return tuple(count * full - (count - 1) * value for value in finite)


def _batch_environment_fingerprint(
    current_rule: ActionRule,
    baseline_runs: tuple[ExpertEpisodeRun, ...],
) -> str:
    """Use the reference baseline environment fingerprint for an iteration."""

    return _environment_fingerprint(current_rule, baseline_runs)


def _bank_from_batch(
    batch: DeviationBatchResult,
    *,
    current_rule: ActionRule,
    baseline_runs: tuple[ExpertEpisodeRun, ...],
    scenarios: tuple[DeviationScenario, ...],
    representative: int = 0,
) -> BankEvaluation:
    bank = enumerate_action_rules()
    expected = tuple(rule.name for rule in bank)
    if not batch.complete or batch.status != "complete":
        raise RuntimeError(f"incomplete deviation batch: {batch.failure_reason}")
    rows_by_name = {row.candidate_rule: row for row in batch.rows}
    if tuple(rows_by_name) != expected or len(rows_by_name) != len(expected):
        raise RuntimeError("parallel batch did not return the complete canonical bank")
    baseline_delete_one = dict(
        _delete_one_objectives(baseline_runs, representative)
    )
    solver_rows: list[SolverRuleRow] = []
    for rule in bank:
        row = rows_by_name[rule.name]
        if row.aggregate_score is None or row.aggregate_score.total is None:
            raise RuntimeError(f"candidate {rule.name} has no complete aggregate score")
        delete_one_values = dict(row.delete_one_objectives)
        if set(delete_one_values) != set(baseline_delete_one):
            raise RuntimeError(f"candidate {rule.name} has an incomplete delete-one set")
        deltas = tuple(
            (None if delete_one_values[key] is None or baseline_delete_one[key] is None
             else delete_one_values[key] - baseline_delete_one[key])
            for key in sorted(baseline_delete_one)
        )
        work = dict(row.aggregate_score.raw_work_totals).get("executed")
        if work is None:
            raise RuntimeError(f"candidate {rule.name} has no executed-work total")
        solver_rows.append(SolverRuleRow(
            rule=rule,
            objective=row.aggregate_score.total,
            executed_work=work,
            paired_delete_one_deltas=deltas,
        ))
    return BankEvaluation(
        current_rule=current_rule.name,
        rows=tuple(solver_rows),
        environment_fingerprint=_batch_environment_fingerprint(
            current_rule, baseline_runs
        ),
    )


@dataclass(frozen=True)
class ParallelFitResult:
    starts: tuple[SolverStartResult, ...]
    call_count: int
    call_budget: int
    fit_denominators: tuple[tuple[int, float], ...]
    complete: bool
    status: str
    failure_reason: str

    @property
    def candidate_rules(self) -> tuple[str, ...]:
        return tuple(start.candidate_rule for start in self.starts)

    def as_dict(self) -> dict[str, object]:
        return {
            "starts": [start.as_dict() for start in self.starts],
            "call_count": self.call_count,
            "call_budget": self.call_budget,
            "fit_denominators": {str(key): value for key, value in self.fit_denominators},
            "complete": self.complete,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }


def _fit_failure_start(
    start_rule: ActionRule,
    *,
    iterations: Sequence[IterationRecord],
    visited: Sequence[str],
    reason: str,
) -> SolverStartResult:
    return SolverStartResult(
        start_rule=start_rule.name,
        iterations=tuple(iterations),
        visited_rules=tuple(visited) or (start_rule.name,),
        candidate_rule=(visited[-1] if visited else start_rule.name),
        dynamics_status=(
            SolverDynamicsStatus.BUDGET_EXHAUSTED
            if reason == "call_budget_exhausted"
            else SolverDynamicsStatus.EXECUTION_FAILURE
        ),
        precision_status=SolverPrecisionStatus.STATISTICS_INSUFFICIENT,
        stop_reason=reason,
    )


def run_parallel_fit(
    scenarios: Iterable[DeviationScenario],
    *,
    expert_count: int = 8,
    starts: Iterable[ActionRule] | None = None,
    max_rounds: int = FROZEN_MAX_ROUNDS,
    precision_denominator: float | None = None,
    precision_ratio: float = FROZEN_PRECISION_RATIO,
    fit_call_budget: int = FIT_CALLS,
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    max_workers: int = 8,
) -> ParallelFitResult:
    """Run the frozen best-response state machine using exact batches."""

    materialized = _validate_scenarios(scenarios)
    if any(item.trace.expert_count != expert_count for item in materialized):
        raise ValueError("fit scenario Expert count does not match")
    if type(fit_call_budget) is not int or fit_call_budget < 1:
        raise ValueError("fit_call_budget must be a positive int")
    if type(max_rounds) is not int or not 1 <= max_rounds <= FROZEN_MAX_ROUNDS:
        raise ValueError("max_rounds must be in 1..32")
    if not math.isfinite(float(precision_ratio)) or precision_ratio <= 0.0:
        raise ValueError("precision_ratio must be positive and finite")
    bank = enumerate_action_rules()
    bank_by_name = {rule.name: rule for rule in bank}
    start_rules = tuple(starts) if starts is not None else (
        bank_by_name["NNNN"], bank_by_name["NSSN"], bank_by_name["XXXX"]
    )
    if tuple(rule.name for rule in start_rules) != ("NNNN", "NSSN", "XXXX"):
        raise ValueError("Stage 4 requires the three frozen starts in order")

    call_count = 0
    denominator = precision_denominator
    fit_denominators: dict[int, float] = {}
    results: list[SolverStartResult] = []
    failure_reason = ""
    for start_index, start in enumerate(start_rules):
        current = start
        visited: list[str] = []
        seen: dict[str, int] = {}
        iterations: list[IterationRecord] = []
        precision_statuses: list[SolverPrecisionStatus] = []
        candidate_name = current.name
        dynamics = SolverDynamicsStatus.NOT_CONVERGED_32
        stop_reason = "not_converged_32"
        cycle_first = cycle_period = None

        for round_index in range(max_rounds):
            seen[current.name] = len(visited)
            visited.append(current.name)
            call_start = call_count
            required_baseline = len(materialized)
            if call_count + required_baseline > fit_call_budget:
                failure_reason = "call budget prevents baseline dispatch"
                current_result = _fit_failure_start(
                    start, iterations=iterations, visited=visited,
                    reason="call_budget_exhausted",
                )
                results.append(current_result)
                for remaining in start_rules[start_index + 1:]:
                    results.append(_fit_failure_start(
                        remaining, iterations=(), visited=(remaining.name,),
                        reason="call_budget_exhausted",
                    ))
                return ParallelFitResult(
                    tuple(results), call_count, fit_call_budget,
                    tuple(sorted(fit_denominators.items())), False,
                    "budget_exhausted", failure_reason,
                )
            try:
                profile = {expert: current for expert in range(expert_count)}
                def count_baseline_call() -> None:
                    nonlocal call_count
                    call_count += 1

                baseline_runs = _full_profile_runs(
                    materialized, profile, c_b=c_b,
                    degraded_slowdown=degraded_slowdown,
                    hedge_delay=hedge_delay,
                    call_observer=count_baseline_call,
                )
                baseline_population = score_population(baseline_runs)
                if any(score.total is None for score in baseline_population.expert_scores):
                    raise RuntimeError("fit baseline has an incomplete Expert cohort")
                if denominator is None:
                    denominator = baseline_population.expert_scores[0].total
                    if denominator is None or denominator <= 0.0:
                        raise RuntimeError("NNNN precision denominator is not positive")
                if not fit_denominators and current.name == "NNNN":
                    fit_denominators = {
                        score.expert_id: score.total
                        for score in baseline_population.expert_scores
                        if score.expert_id is not None and score.total is not None
                    }

                tasks = tuple(
                    DeviationTask(0, candidate, current)
                    for candidate in bank
                )
                batch = run_deviation_batch(
                    materialized,
                    tasks,
                    expert_count=expert_count,
                    c_b=c_b,
                    degraded_slowdown=degraded_slowdown,
                    hedge_delay=hedge_delay,
                    max_workers=max_workers,
                    call_budget=fit_call_budget - call_count,
                    rule_bank=bank,
                )
                call_count += batch.reserved_calls
                if not batch.complete or batch.reserved_calls != 256 * len(materialized):
                    raise RuntimeError(
                        f"fit deviation batch incomplete: {batch.failure_reason}"
                    )
                evaluation = _bank_from_batch(
                    batch, current_rule=current, baseline_runs=baseline_runs,
                    scenarios=materialized,
                )
                record = _iteration_record(
                    evaluation,
                    round_index=round_index,
                    call_count_start=call_start,
                    call_count_end=call_count,
                    precision_denominator=denominator,
                    precision_ratio=precision_ratio,
                )
            except Exception as exc:
                failure_reason = repr(exc)
                results.append(_fit_failure_start(
                    start, iterations=iterations, visited=visited,
                    reason="execution_failure",
                ))
                for remaining in start_rules[start_index + 1:]:
                    results.append(_fit_failure_start(
                        remaining, iterations=(), visited=(remaining.name,),
                        reason="execution_failure",
                    ))
                return ParallelFitResult(
                    tuple(results), call_count, fit_call_budget,
                    tuple(sorted(fit_denominators.items())), False,
                    "execution_failed", failure_reason,
                )
            iterations.append(record)
            precision_statuses.append(record.precision_status)
            candidate_name = record.selected_rule
            if record.selected_rule == current.name:
                dynamics = SolverDynamicsStatus.PURE_FIXED_POINT
                stop_reason = "pure_fixed_point"
                iterations[-1] = IterationRecord(
                    **{**record.__dict__, "stop_reason": stop_reason}
                )
                break
            if record.selected_rule in seen:
                dynamics = SolverDynamicsStatus.RULE_CYCLE
                stop_reason = "rule_cycle"
                cycle_first = seen[record.selected_rule]
                cycle_period = len(visited) - cycle_first
                iterations[-1] = IterationRecord(
                    **{**record.__dict__, "stop_reason": stop_reason}
                )
                break
            current = bank_by_name[record.selected_rule]
        else:
            dynamics = SolverDynamicsStatus.NOT_CONVERGED_32
            stop_reason = "not_converged_32"
            if iterations:
                record = iterations[-1]
                iterations[-1] = IterationRecord(
                    **{**record.__dict__, "stop_reason": stop_reason}
                )

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
            dynamics_status=dynamics,
            precision_status=aggregate_precision,
            stop_reason=stop_reason,
            cycle_first_repeat=cycle_first,
            cycle_period=cycle_period,
        ))

    return ParallelFitResult(
        tuple(results), call_count, fit_call_budget,
        tuple(sorted(fit_denominators.items())), True, "complete", "",
    )


@dataclass(frozen=True)
class ValidationDeviationRow:
    deviating_expert: int
    incumbent_rule: str
    candidate_rule: str
    base_objective: float
    deviated_objective: float
    improvement: float
    paired_delete_one_improvements: tuple[float | None, ...]
    jackknife_pseudo_values: tuple[float | None, ...]
    target: float
    trace_fingerprints: tuple[str, ...]
    fault_fingerprints: tuple[str, ...]
    invariant_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("base_objective", "deviated_objective", "target"):
            _finite(getattr(self, name), name)
        _finite(self.improvement, "improvement", allow_negative=True)
        improvements = tuple(
            None if value is None else _finite(
                value, "paired cluster improvement", allow_negative=True
            )
            for value in self.paired_delete_one_improvements
        )
        object.__setattr__(self, "paired_delete_one_improvements", improvements)
        pseudo = tuple(
            None if value is None else _finite(
                value, "jackknife pseudo-value", allow_negative=True
            )
            for value in self.jackknife_pseudo_values
        )
        object.__setattr__(self, "jackknife_pseudo_values", pseudo)

    @property
    def regret_contribution(self) -> float:
        return max(0.0, self.improvement)

    def as_dict(self) -> dict[str, object]:
        return {
            "deviating_expert": self.deviating_expert,
            "incumbent_rule": self.incumbent_rule,
            "candidate_rule": self.candidate_rule,
            "base_objective": self.base_objective,
            "deviated_objective": self.deviated_objective,
            "improvement": self.improvement,
            "regret_contribution": self.regret_contribution,
            "paired_delete_one_improvements": list(
                self.paired_delete_one_improvements
            ),
            "jackknife_pseudo_values": list(self.jackknife_pseudo_values),
            "target": self.target,
            "trace_fingerprints": list(self.trace_fingerprints),
            "fault_fingerprints": list(self.fault_fingerprints),
            "invariant_fingerprints": list(self.invariant_fingerprints),
        }


@dataclass(frozen=True)
class SimultaneousBoundResult:
    status: str
    cluster_count: int
    bootstrap_replicates: int
    seed: int
    observed_max_regret: float | None
    simultaneous_upper_bound: float | None
    upper_bounds: Mapping[tuple[int, str], float]
    targets: Mapping[int, float]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "statistics_insufficient"):
            raise ValueError("invalid simultaneous-bound status")
        object.__setattr__(self, "upper_bounds", dict(self.upper_bounds))
        object.__setattr__(self, "targets", dict(self.targets))

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cluster_count": self.cluster_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "seed": self.seed,
            "observed_max_regret": self.observed_max_regret,
            "simultaneous_upper_bound": self.simultaneous_upper_bound,
            "upper_bounds": {
                f"expert-{expert}:{rule}": value
                for (expert, rule), value in sorted(self.upper_bounds.items())
            },
            "targets": {str(key): value for key, value in sorted(self.targets.items())},
            "reason": self.reason,
            "method": (
                "paired common-path jackknife pseudo-value bootstrap "
                "max-t one-sided 95pct"
            ),
        }


def compute_simultaneous_regret_bound(
    observed_improvements: Mapping[tuple[int, str], float],
    jackknife_pseudo_rows: Mapping[
        tuple[int, str], Sequence[float | None]
    ],
    *,
    targets: Mapping[int, float],
    bootstrap_replicates: int = SIMULTANEOUS_BOOTSTRAP_REPLICATES,
    seed: int = SIMULTANEOUS_BOOTSTRAP_SEED,
) -> SimultaneousBoundResult:
    """Bootstrap max-t jointly over paired common-path pseudo-values."""

    observed = {
        key: _finite(value, "observed improvement", allow_negative=True)
        for key, value in observed_improvements.items()
    }
    rows = {
        key: tuple(
            None if value is None else _finite(
                value, "jackknife pseudo-value", allow_negative=True
            )
            for value in values
        )
        for key, values in jackknife_pseudo_rows.items()
    }
    if not rows or set(rows) != set(observed):
        raise ValueError("simultaneous bound requires matching observed and pseudo rows")
    cluster_counts = {len(values) for values in rows.values()}
    if len(cluster_counts) != 1:
        # An absent/short row is an incomplete statistical result, not an
        # executable inference shape.  Fail closed so a late Validation
        # defect is reported as insufficient precision rather than raised
        # after the expensive reruns have completed.
        return SimultaneousBoundResult(
            "statistics_insufficient", max(cluster_counts), bootstrap_replicates,
            seed, None, None, {}, targets,
            "one or more paired common-path jackknife rows have incomplete shape",
        )
    cluster_count = next(iter(cluster_counts))
    if any(value is None for values in rows.values() for value in values):
        return SimultaneousBoundResult(
            "statistics_insufficient", cluster_count, bootstrap_replicates, seed,
            None, None, {}, targets,
            "one or more paired common-path jackknife rows are incomplete",
        )
    if cluster_count < 2:
        return SimultaneousBoundResult(
            "statistics_insufficient", cluster_count, bootstrap_replicates, seed,
            max(max(0.0, value) for value in observed.values()),
            None, {}, targets, "fewer than two common-path clusters",
        )
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 32:
        raise ValueError("bootstrap_replicates must be an int >= 32")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative int")
    for expert, target in targets.items():
        if type(expert) is not int or _finite(target, "target") <= 0.0:
            raise ValueError("targets must be positive finite Expert values")

    keys = tuple(sorted(rows))
    pseudo_means = {
        key: math.fsum(rows[key]) / cluster_count for key in keys
    }
    standard_errors: dict[tuple[int, str], float] = {}
    for key in keys:
        values = rows[key]
        mean = pseudo_means[key]
        variance = math.fsum((value - mean) ** 2 for value in values)
        standard_errors[key] = math.sqrt(
            variance / (cluster_count * (cluster_count - 1))
        )
    observed_max = max(max(0.0, observed[key]) for key in keys)

    rng = random.Random(seed)
    max_t_values: list[float] = []
    for _ in range(bootstrap_replicates):
        counts = [0] * cluster_count
        for _ in range(cluster_count):
            counts[rng.randrange(cluster_count)] += 1
        max_t = 0.0
        for key in keys:
            values = rows[key]
            bootstrap_mean = math.fsum(
                count * value for count, value in zip(counts, values)
            ) / cluster_count
            se = standard_errors[key]
            statistic = (
                0.0 if se == 0.0 and bootstrap_mean == pseudo_means[key]
                else math.inf if se == 0.0 and pseudo_means[key] > bootstrap_mean
                else -math.inf if se == 0.0
                else (pseudo_means[key] - bootstrap_mean) / se
            )
            max_t = max(max_t, statistic)
        max_t_values.append(max_t)
    max_t_values.sort()
    quantile_index = min(
        len(max_t_values) - 1,
        max(0, math.ceil(0.95 * len(max_t_values)) - 1),
    )
    quantile = max_t_values[quantile_index]
    upper_bounds = {
        key: observed[key] + quantile * standard_errors[key]
        for key in keys
    }
    status = "pass"
    for key, upper in upper_bounds.items():
        target = targets.get(key[0])
        if target is None:
            raise ValueError(f"missing target for Expert {key[0]}")
        if not math.isfinite(upper) or upper > target:
            status = "fail"
            break
    return SimultaneousBoundResult(
        status=status,
        cluster_count=cluster_count,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        observed_max_regret=observed_max,
        simultaneous_upper_bound=max(upper_bounds.values()),
        upper_bounds=upper_bounds,
        targets=targets,
        reason=("all simultaneous upper bounds below targets"
                if status == "pass" else "at least one simultaneous upper bound exceeds target"),
    )


@dataclass(frozen=True)
class ParallelValidationResult:
    candidate_rule: str
    rows: tuple[ValidationDeviationRow, ...]
    call_count: int
    call_budget: int
    complete: bool
    status: str
    failure_reason: str
    fit_trace_fingerprints: tuple[str, ...]
    validation_trace_fingerprints: tuple[str, ...]
    fit_fault_fingerprints: tuple[str, ...]
    validation_fault_fingerprints: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_rule": self.candidate_rule,
            "rows": [row.as_dict() for row in self.rows],
            "call_count": self.call_count,
            "call_budget": self.call_budget,
            "complete": self.complete,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "fit_trace_fingerprints": list(self.fit_trace_fingerprints),
            "validation_trace_fingerprints": list(self.validation_trace_fingerprints),
            "fit_fault_fingerprints": list(self.fit_fault_fingerprints),
            "validation_fault_fingerprints": list(self.validation_fault_fingerprints),
        }


def _require_disjoint_scenarios(
    fit: tuple[DeviationScenario, ...],
    validation: tuple[DeviationScenario, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    fit_trace, fit_fault = _scenario_fingerprints(fit)
    validation_trace, validation_fault = _scenario_fingerprints(validation)
    if set(fit_trace) & set(validation_trace) or set(fit_fault) & set(validation_fault):
        raise ValueError("Fit and Validation trace/fault fingerprints overlap")
    return fit_trace, validation_trace, fit_fault, validation_fault


def run_parallel_validation(
    candidate_rule: ActionRule,
    fit_scenarios: Iterable[DeviationScenario],
    validation_scenarios: Iterable[DeviationScenario],
    *,
    fit_denominators: Mapping[int, float],
    validation_call_budget: int = VALIDATION_CALLS,
    expert_count: int = 8,
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    max_workers: int = 8,
) -> ParallelValidationResult:
    """Recompute every validation deviation with frozen opponents."""

    if not isinstance(candidate_rule, ActionRule):
        raise ValueError("candidate_rule must be ActionRule")
    fit = _validate_scenarios(fit_scenarios)
    validation = _validate_scenarios(validation_scenarios)
    if any(item.trace.expert_count != expert_count for item in fit + validation):
        raise ValueError("Fit/Validation Expert counts do not match")
    fit_trace, validation_trace, fit_fault, validation_fault = (
        _require_disjoint_scenarios(fit, validation)
    )
    bank = enumerate_action_rules()
    call_count = 0
    rows: list[ValidationDeviationRow] = []
    failure_reason = ""

    def count_validation_call() -> None:
        nonlocal call_count
        call_count += 1

    try:
        for expert_id in range(expert_count):
            if call_count + len(validation) > validation_call_budget:
                raise RuntimeError("validation budget prevents baseline dispatch")
            base_profile = {expert: candidate_rule for expert in range(expert_count)}
            baseline_runs = _full_profile_runs(
                validation, base_profile, c_b=c_b,
                degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
                call_observer=count_validation_call,
            )
            baseline_score = score_expert(baseline_runs, expert_id)
            if baseline_score.total is None:
                raise RuntimeError("validation baseline score is incomplete")
            baseline_delete_one = dict(
                _delete_one_objectives(baseline_runs, expert_id)
            )
            denominator = fit_denominators.get(expert_id)
            if denominator is None or denominator <= 0.0:
                raise RuntimeError(f"missing positive Fit NNNN denominator for Expert {expert_id}")
            target = VALIDATION_REGRET_RATIO * denominator
            tasks = tuple(
                DeviationTask(expert_id, rule, candidate_rule)
                for rule in bank
            )
            batch = run_deviation_batch(
                validation,
                tasks,
                expert_count=expert_count,
                c_b=c_b,
                degraded_slowdown=degraded_slowdown,
                hedge_delay=hedge_delay,
                max_workers=max_workers,
                call_budget=validation_call_budget - call_count,
                rule_bank=bank,
            )
            call_count += batch.reserved_calls
            if not batch.complete or batch.reserved_calls != len(validation) * 256:
                raise RuntimeError(f"validation deviation batch incomplete: {batch.failure_reason}")
            for row in batch.rows:
                if row.aggregate_score is None or row.aggregate_score.total is None:
                    raise RuntimeError("validation candidate score is incomplete")
                delete_one_values = dict(row.delete_one_objectives)
                if set(delete_one_values) != set(baseline_delete_one):
                    raise RuntimeError("validation delete-one set is incomplete")
                improvements = tuple(
                    (None if baseline_delete_one[key] is None
                     or delete_one_values[key] is None
                     else baseline_delete_one[key] - delete_one_values[key])
                    for key in sorted(baseline_delete_one)
                )
                full_improvement = baseline_score.total - row.aggregate_score.total
                invariant_fingerprints = tuple(
                    repr(score.invariant_counters) for score in row.scenario_scores
                )
                rows.append(ValidationDeviationRow(
                    deviating_expert=expert_id,
                    incumbent_rule=candidate_rule.name,
                    candidate_rule=row.candidate_rule,
                    base_objective=baseline_score.total,
                    deviated_objective=row.aggregate_score.total,
                    improvement=full_improvement,
                    paired_delete_one_improvements=improvements,
                    jackknife_pseudo_values=jackknife_pseudo_values(
                        full_improvement, improvements
                    ),
                    target=target,
                    trace_fingerprints=validation_trace,
                    fault_fingerprints=validation_fault,
                    invariant_fingerprints=invariant_fingerprints,
                ))
    except Exception as exc:
        failure_reason = repr(exc)
    complete = not failure_reason and len(rows) == expert_count * 256
    return ParallelValidationResult(
        candidate_rule=candidate_rule.name,
        rows=tuple(sorted(rows, key=lambda row: (
            row.deviating_expert, _rule_key(ActionRule(row.candidate_rule, tuple(row.candidate_rule)))
        ))),
        call_count=call_count,
        call_budget=validation_call_budget,
        complete=complete,
        status="complete" if complete else "execution_failed",
        failure_reason=failure_reason,
        fit_trace_fingerprints=fit_trace,
        validation_trace_fingerprints=validation_trace,
        fit_fault_fingerprints=fit_fault,
        validation_fault_fingerprints=validation_fault,
    )


@dataclass(frozen=True)
class Stage4CampaignResult:
    status: str
    fit: ParallelFitResult
    candidate_rule: str | None
    candidate_fingerprint: str | None
    validation: ParallelValidationResult | None
    simultaneous_bound: SimultaneousBoundResult | None
    fit_calls: int
    validation_calls: int
    total_calls: int
    elapsed_seconds: float
    artifact_run_dir: str | None
    failure_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "fit": self.fit.as_dict(),
            "candidate_rule": self.candidate_rule,
            "candidate_fingerprint": self.candidate_fingerprint,
            "validation": self.validation.as_dict() if self.validation else None,
            "simultaneous_bound": (
                self.simultaneous_bound.as_dict()
                if self.simultaneous_bound else None
            ),
            "fit_calls": self.fit_calls,
            "validation_calls": self.validation_calls,
            "total_calls": self.total_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "artifact_run_dir": self.artifact_run_dir,
            "failure_reason": self.failure_reason,
            "claim_boundary": (
                "restricted Pi256 approximate equilibrium candidate only"
                if self.status == "pass"
                else "no restricted Pi256 approximate equilibrium candidate"
            ),
            "claims_unrestricted_nash": False,
            "claims_mfg": False,
        }


def validate_stage4_campaign_result(result: Stage4CampaignResult) -> None:
    """Fail closed unless a result satisfies the Stage 4 completeness schema."""

    if not isinstance(result, Stage4CampaignResult):
        raise ValueError("result must be Stage4CampaignResult")
    if result.total_calls > HARD_CALL_LIMIT:
        raise ValueError("campaign result exceeds the hard call limit")
    if result.fit_calls != result.fit.call_count:
        raise ValueError("Fit call accounting does not reconcile")
    if result.validation is None:
        if result.candidate_rule is not None and result.status == "pass":
            raise ValueError("passing result requires Validation")
        return
    validation = result.validation
    if not validation.complete or len(validation.rows) != 8 * 256:
        raise ValueError("Validation result is incomplete")
    keys = tuple((row.deviating_expert, row.candidate_rule) for row in validation.rows)
    expected = tuple(
        (expert, rule.name)
        for expert in range(8)
        for rule in enumerate_action_rules()
    )
    if keys != expected:
        raise ValueError("Validation rows are not complete canonical rows")
    if result.validation_calls != validation.call_count:
        raise ValueError("Validation call accounting does not reconcile")
    if result.simultaneous_bound is None:
        raise ValueError("complete Validation requires a simultaneous bound")


def _candidate_fingerprint(
    candidate: str,
    fit: ParallelFitResult,
    scenarios: tuple[DeviationScenario, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(candidate.encode("ascii"))
    for start in fit.starts:
        digest.update(repr(start.as_dict()).encode("utf-8"))
    trace, fault = _scenario_fingerprints(scenarios)
    for value in trace + fault:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _select_candidate(fit: ParallelFitResult) -> str | None:
    if not fit.complete or len(fit.starts) != 3:
        return None
    if any(start.dynamics_status is not SolverDynamicsStatus.PURE_FIXED_POINT
           or start.precision_status is not SolverPrecisionStatus.SUFFICIENT
           for start in fit.starts):
        return None
    names = {start.candidate_rule for start in fit.starts}
    return next(iter(names)) if len(names) == 1 else None


def run_frozen_stage4_campaign(
    *,
    artifacts_root: str | Path | None = None,
    run_id: str | None = None,
    max_workers: int = 8,
) -> Stage4CampaignResult:
    """Run the frozen real Stage 4 campaign after all in-memory gates pass."""

    plan = preflight_frozen_stage4()
    if not plan.within_budget:
        raise ValueError("frozen Stage 4 call plan exceeds the hard budget")
    fit_scenarios = build_frozen_stage4_scenarios(
        "shared-backup-game:v1:stage4-fit", 20260905, common_path_count=16,
    )
    validation_scenarios = build_frozen_stage4_scenarios(
        "shared-backup-game:v1:stage4-validation", 20260906, common_path_count=32,
    )
    provenance = _stage4_provenance()
    started = time.perf_counter()
    fit = run_parallel_fit(
        fit_scenarios,
        expert_count=8,
        max_rounds=32,
        fit_call_budget=FIT_CALLS,
        c_b=0.5,
        degraded_slowdown=2.0,
        hedge_delay=1.5,
        max_workers=max_workers,
    )
    candidate = _select_candidate(fit)
    if candidate is None:
        result = Stage4CampaignResult(
            status="fit_no_candidate" if fit.complete else fit.status,
            fit=fit,
            candidate_rule=None,
            candidate_fingerprint=None,
            validation=None,
            simultaneous_bound=None,
            fit_calls=fit.call_count,
            validation_calls=0,
            total_calls=fit.call_count,
            elapsed_seconds=time.perf_counter() - started,
            artifact_run_dir=None,
            failure_reason=(
                "no common precise pure fixed point across all starts"
                if fit.complete else fit.failure_reason
            ),
        )
        # A failed or partial Fit is not a scientific run result.  In
        # particular, do not leave a directory containing a partial trace
        # that could be mistaken for a complete frozen campaign.  A complete
        # Fit with no admissible candidate is a valid negative result and may
        # be written transactionally.
        if artifacts_root is not None and fit.complete:
            if run_id is None:
                raise ValueError("run_id is required when artifacts_root is provided")
            run_dir = write_run_directory(
                artifacts_root, run_id,
                {
                    "manifest.json": {
                        "scenario": "shared_backup_stage4_frozen",
                        "run_id": run_id,
                        "status": result.status,
                        "protocol": plan.as_dict(),
                        "provenance": provenance,
                        "fit_trace_fingerprints": list(_scenario_fingerprints(fit_scenarios)[0]),
                        "validation_trace_fingerprints": list(_scenario_fingerprints(validation_scenarios)[0]),
                    },
                    "fit.json": fit.as_dict(),
                    "result.json": result.as_dict(),
                },
            )
            result = Stage4CampaignResult(**{
                **result.__dict__, "artifact_run_dir": str(run_dir)
            })
        return result

    candidate_rule = next(rule for rule in enumerate_action_rules() if rule.name == candidate)
    validation = run_parallel_validation(
        candidate_rule, fit_scenarios, validation_scenarios,
        fit_denominators=dict(fit.fit_denominators),
        validation_call_budget=VALIDATION_CALLS,
        expert_count=8,
        c_b=0.5,
        degraded_slowdown=2.0,
        hedge_delay=1.5,
        max_workers=max_workers,
    )
    bound = None
    status = validation.status
    failure_reason = validation.failure_reason
    if validation.complete:
        observed = {
            (row.deviating_expert, row.candidate_rule): row.improvement
            for row in validation.rows
        }
        pseudo_rows = {
            (row.deviating_expert, row.candidate_rule): row.jackknife_pseudo_values
            for row in validation.rows
        }
        targets = {
            expert: VALIDATION_REGRET_RATIO * denominator
            for expert, denominator in fit.fit_denominators
        }
        bound = compute_simultaneous_regret_bound(
            observed, pseudo_rows, targets=targets
        )
        status = bound.status
        failure_reason = bound.reason if status != "pass" else ""
    result = Stage4CampaignResult(
        status=status,
        fit=fit,
        candidate_rule=candidate,
        candidate_fingerprint=_candidate_fingerprint(candidate, fit, fit_scenarios),
        validation=validation,
        simultaneous_bound=bound,
        fit_calls=fit.call_count,
        validation_calls=validation.call_count,
        total_calls=fit.call_count + validation.call_count,
        elapsed_seconds=time.perf_counter() - started,
        artifact_run_dir=None,
        failure_reason=failure_reason,
    )
    validate_stage4_campaign_result(result)
    if result.total_calls > HARD_CALL_LIMIT or result.fit_calls > FIT_CALLS or result.validation_calls > VALIDATION_CALLS:
        raise RuntimeError("Stage 4 call accounting exceeded frozen limits")
    if artifacts_root is not None:
        if run_id is None:
            raise ValueError("run_id is required when artifacts_root is provided")
        run_dir = write_run_directory(
            artifacts_root, run_id,
            {
                "manifest.json": {
                    "scenario": "shared_backup_stage4_frozen",
                    "run_id": run_id,
                    "status": result.status,
                    "protocol": plan.as_dict(),
                    "provenance": provenance,
                    "candidate_rule": candidate,
                    "candidate_fingerprint": result.candidate_fingerprint,
                    "fit_trace_fingerprints": list(result.validation.fit_trace_fingerprints),
                    "validation_trace_fingerprints": list(result.validation.validation_trace_fingerprints),
                    "fit_fault_fingerprints": list(result.validation.fit_fault_fingerprints),
                    "validation_fault_fingerprints": list(result.validation.validation_fault_fingerprints),
                },
                "fit.json": fit.as_dict(),
                "validation.json": validation.as_dict(),
                "uncertainty.json": bound.as_dict() if bound else {
                    "status": "statistics_insufficient",
                    "reason": "validation did not produce a bound",
                },
                "result.json": result.as_dict(),
            },
        )
        result = Stage4CampaignResult(**{
            **result.__dict__, "artifact_run_dir": str(run_dir)
        })
    return result


__all__ = [
    "FIT_CALLS",
    "HARD_CALL_LIMIT",
    "SIMULTANEOUS_BOOTSTRAP_REPLICATES",
    "SIMULTANEOUS_BOOTSTRAP_SEED",
    "TOTAL_PLANNED_CALLS",
    "VALIDATION_CALLS",
    "FrozenStage4Plan",
    "ParallelFitResult",
    "ParallelValidationResult",
    "SimultaneousBoundResult",
    "Stage4CampaignResult",
    "ValidationDeviationRow",
    "build_frozen_stage4_scenarios",
    "cluster_jackknife_se",
    "compute_simultaneous_regret_bound",
    "jackknife_pseudo_values",
    "preflight_frozen_stage4",
    "run_frozen_stage4_campaign",
    "run_parallel_fit",
    "run_parallel_validation",
    "validate_stage4_campaign_result",
]
