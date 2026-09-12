"""Deterministic, spawn-safe execution primitives for the Stage 4 gate.

This module deliberately stops below campaign selection.  It batches complete
counterfactual reruns and returns compact scorer sufficient data; it does not
choose a best rule, write artifacts, or launch Fit/Validation.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import ctypes
import math
import multiprocessing
import os
import time
from typing import Iterable

from .common_state import CommonState, Phase
from .domain import TokenClass
from .expert_game import ExpertScore, TaggedExpertEpisodeRun, score_expert, score_tagged_result
from .game_workload import ActionRule, enumerate_action_rules
from .shared_backup import (
    PreparedTrace,
    SharedAction,
    SharedBackupTrace,
    prepare_shared_trace,
    simulate_shared_backup_tagged,
)


MAX_WORKERS = 8
MAX_IN_FLIGHT = 8
WORKER_RSS_LIMIT = 512 * 1024 * 1024
PARENT_RSS_LIMIT = 4 * 1024 * 1024 * 1024


def _rule_key(rule: ActionRule) -> tuple[int, int, int, int]:
    if not isinstance(rule, ActionRule):
        raise ValueError("campaign tasks require ActionRule values")
    if type(rule.name) is not str or len(rule.name) != 4:
        raise ValueError("ActionRule names must contain exactly four codes")
    if any(code not in "NDSX" for code in rule.name):
        raise ValueError("ActionRule names must contain only N, D, S, or X")
    order = {"N": 0, "D": 1, "S": 2, "X": 3}
    return tuple(order[code] for code in rule.name)  # type: ignore[return-value]


def validate_canonical_rule_bank(
    rule_bank: Iterable[ActionRule],
) -> tuple[ActionRule, ...]:
    """Reject missing, duplicate, noncanonical, or reordered rule banks."""

    materialized = tuple(rule_bank)
    expected = enumerate_action_rules()
    if materialized != expected:
        raise ValueError("rule bank must be the exact canonical 256-rule bank")
    return materialized


@dataclass(frozen=True)
class DeviationTask:
    """One fixed-grain ``(deviating Expert, candidate rule)`` task."""

    deviating_expert: int
    candidate_rule: ActionRule
    incumbent_rule: ActionRule

    @property
    def key(self) -> tuple[int, str]:
        return self.deviating_expert, self.candidate_rule.name

    def __post_init__(self) -> None:
        if type(self.deviating_expert) is not int or self.deviating_expert < 0:
            raise ValueError("deviating_expert must be a non-negative int")
        _rule_key(self.candidate_rule)
        _rule_key(self.incumbent_rule)


@dataclass(frozen=True)
class PreparedScenario:
    """One validated exogenous scenario shared by all worker tasks."""

    episode_key: str
    prepared_trace: PreparedTrace
    fault_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be a non-empty string")
        if not isinstance(self.prepared_trace, PreparedTrace):
            raise ValueError("prepared_trace must be PreparedTrace")
        if type(self.fault_fingerprint) is not str or not self.fault_fingerprint:
            raise ValueError("fault_fingerprint must be non-empty")


@dataclass(frozen=True)
class TaggedScenarioScore:
    """Compact, immutable scorer sufficient data for one scenario."""

    episode_key: str
    trace_fingerprint: str
    fault_fingerprint: str
    expert_id: int
    total: float | None
    complete: bool
    generated_token_count: int
    phase_losses: tuple[tuple[str, float | None], ...]
    cvar95: tuple[tuple[str, float | None], ...]
    class_cvar95: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...]
    raw_work_totals: tuple[tuple[str, float], ...]
    components: tuple[tuple[str, float | None], ...]
    missing_cohorts: tuple[tuple[str, str], ...]
    invariant_counters: tuple[tuple[str, int], ...] = ()
    pool_audit_summary: tuple[tuple[str, float | int], ...] = ()
    drain_end_time: float | None = None


@dataclass(frozen=True)
class DeviationBatchRow:
    """One candidate row with no materialized non-tagged Expert outputs."""

    deviating_expert: int
    candidate_rule: str
    incumbent_rule: str
    scenario_scores: tuple[TaggedScenarioScore, ...]
    aggregate_score: TaggedScenarioScore | None = None
    delete_one_objectives: tuple[tuple[str, float | None], ...] = ()

    @property
    def mean_objective(self) -> float | None:
        values = tuple(score.total for score in self.scenario_scores)
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None) / len(values)


@dataclass(frozen=True)
class DeviationBatchResult:
    """Parent-owned accounting and deterministic rows for one batch."""

    rows: tuple[DeviationBatchRow, ...]
    merge_keys: tuple[tuple[int, str], ...]
    reserved_calls: int
    completed_calls: int
    failed_calls: int
    complete: bool
    status: str
    failure_reason: str
    worker_count: int
    start_method: str
    task_granularity: str
    deterministic_merge_order: str
    elapsed_seconds: float
    worker_peak_rss_bytes: tuple[int, ...]
    parent_peak_rss_bytes: int | None
    rss_supported: bool

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    def __post_init__(self) -> None:
        for name in ("reserved_calls", "completed_calls", "failed_calls", "worker_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.completed_calls + self.failed_calls > self.reserved_calls:
            raise ValueError("batch call settlement exceeds reservation")
        if self.status not in ("complete", "partial", "failed"):
            raise ValueError("invalid batch status")
        if self.complete != (self.status == "complete"):
            raise ValueError("batch complete flag disagrees with status")
        if type(self.failure_reason) is not str:
            raise ValueError("failure_reason must be a string")
        if self.parent_peak_rss_bytes is not None and self.parent_peak_rss_bytes < 0:
            raise ValueError("parent RSS must be non-negative")
        if type(self.rss_supported) is not bool:
            raise ValueError("rss_supported must be bool")


@dataclass(frozen=True)
class _WorkerParameters:
    expert_count: int
    c_b: float
    degraded_slowdown: float
    hedge_delay: float


@dataclass(frozen=True)
class _WorkerTaskResult:
    row: DeviationBatchRow
    peak_rss_bytes: int | None


_WORKER_SCENARIOS: tuple[PreparedScenario, ...] = ()
_WORKER_PARAMETERS: _WorkerParameters | None = None


def _rss_bytes() -> int | None:
    """Return real process RSS, or None when the platform API is unavailable."""

    if os.name != "nt":
        return None
    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        error = ctypes.get_last_error()
        raise OSError(error, "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def _compact_score(
    score: ExpertScore,
    *,
    scenario: PreparedScenario,
    tagged_result: object | None = None,
) -> TaggedScenarioScore:
    phase_losses = tuple(sorted(score.phase_losses.items()))
    cvar95 = tuple(sorted(score.cvar95.items()))
    class_cvar95 = tuple(
        (phase, tuple(sorted(values.items())))
        for phase, values in sorted(score.class_cvar95.items())
    )
    counters = ()
    pool_summary = ()
    drain_end = None
    if tagged_result is not None:
        counter_values = tagged_result.counters
        counters = tuple(
            (name, getattr(counter_values, name))
            for name in counter_values.__dataclass_fields__
        )
        summary = tagged_result.pool_audit_summary
        pool_summary = (
            ("interval_count", summary.interval_count),
            ("total_executed_work", summary.total_executed_work),
            ("total_capacity_bound", summary.total_capacity_bound),
            ("max_active_heads", summary.max_active_heads),
            ("constrained_interval_count", summary.constrained_interval_count),
        )
        drain_end = tagged_result.drain_end_time
    return TaggedScenarioScore(
        episode_key=scenario.episode_key,
        trace_fingerprint=scenario.prepared_trace.trace_fingerprint,
        fault_fingerprint=scenario.fault_fingerprint,
        expert_id=score.expert_id if score.expert_id is not None else -1,
        total=score.total,
        complete=score.complete,
        generated_token_count=score.generated_token_count,
        phase_losses=phase_losses,
        cvar95=cvar95,
        class_cvar95=class_cvar95,
        raw_work_totals=tuple(sorted(score.raw_work_totals.items())),
        components=tuple(sorted(score.components.items())),
        missing_cohorts=score.missing_cohorts,
        invariant_counters=counters,
        pool_audit_summary=pool_summary,
        drain_end_time=drain_end,
    )


def _worker_init(
    scenarios: tuple[PreparedScenario, ...], parameters: _WorkerParameters
) -> None:
    global _WORKER_SCENARIOS, _WORKER_PARAMETERS
    _WORKER_SCENARIOS = scenarios
    _WORKER_PARAMETERS = parameters


class _BatchRuleSource:
    def __init__(self, rule: ActionRule):
        self.rule = rule

    @property
    def name(self) -> str:
        return self.rule.name

    def request(self, observation: object):
        return self.rule.request(observation)

    def request_compact(
        self, *, common_state, phase, phase_age, token_class, primary_replica
    ):
        # Preserve custom ActionRule subclass semantics.  Only the frozen,
        # exact base rule is proven to depend on this compact projection.
        if type(self.rule) is not ActionRule:
            return NotImplemented
        if (common_state is not CommonState.DEGRADED
                or phase is not Phase.DEGRADED
                or primary_replica != 0):
            return SharedAction.NORMAL
        late = int(phase_age >= 50.0)
        urgent = int(token_class is TokenClass.URGENT)
        return SharedAction(self.rule.actions[2 * late + urgent])


def _run_deviation_task(task: DeviationTask) -> DeviationBatchRow:
    """Top-level spawn-safe worker; it never writes or mutates parent state."""

    if _WORKER_PARAMETERS is None or not _WORKER_SCENARIOS:
        raise RuntimeError("worker was not initialized with prepared scenarios")
    parameters = _WORKER_PARAMETERS
    scores: list[TaggedScenarioScore] = []
    runs: list[TaggedExpertEpisodeRun] = []
    peak = _rss_bytes()
    for scenario in _WORKER_SCENARIOS:
        profile = {
            expert: _BatchRuleSource(task.incumbent_rule)
            for expert in range(parameters.expert_count)
        }
        profile[task.deviating_expert] = _BatchRuleSource(task.candidate_rule)
        tagged = simulate_shared_backup_tagged(
            scenario.prepared_trace,
            parameters.c_b,
            degraded_slowdown=parameters.degraded_slowdown,
            hedge_delay=parameters.hedge_delay,
            action_sources=profile,
            tagged_expert_id=task.deviating_expert,
        )
        score = score_tagged_result(
            tagged,
            trace=scenario.prepared_trace.trace,
            expert_id=task.deviating_expert,
            episode_key=scenario.episode_key,
            fault_fingerprint=scenario.fault_fingerprint,
        )
        scores.append(_compact_score(score, scenario=scenario, tagged_result=tagged))
        runs.append(TaggedExpertEpisodeRun(
            episode_key=scenario.episode_key,
            trace=scenario.prepared_trace.trace,
            result=tagged,
            fault_fingerprint=scenario.fault_fingerprint,
        ))
        current = _rss_bytes()
        if current is not None:
            peak = max(peak or current, current)
            if peak > WORKER_RSS_LIMIT:
                raise MemoryError("worker RSS exceeded 512 MiB")
    clusters: dict[str, list[TaggedExpertEpisodeRun]] = {}
    for run in runs:
        clusters.setdefault(run.fault_fingerprint, []).append(run)
    delete_one_objectives = []
    for omitted in sorted(clusters):
        retained = tuple(
            run for fault, group in sorted(clusters.items())
            if fault != omitted for run in group
        )
        score = (
            score_expert(retained, task.deviating_expert).total
            if retained else None
        )
        delete_one_objectives.append((omitted, score))
    return _WorkerTaskResult(
        row=DeviationBatchRow(
            deviating_expert=task.deviating_expert,
            candidate_rule=task.candidate_rule.name,
            incumbent_rule=task.incumbent_rule.name,
            scenario_scores=tuple(scores),
            aggregate_score=_compact_score(
                score_expert(tuple(runs), task.deviating_expert),
                scenario=_WORKER_SCENARIOS[0],
            ),
            delete_one_objectives=tuple(delete_one_objectives),
        ),
        peak_rss_bytes=peak,
    )


def _prepare_scenarios(scenarios: Iterable[object]) -> tuple[PreparedScenario, ...]:
    prepared: list[PreparedScenario] = []
    for scenario in scenarios:
        if isinstance(scenario, PreparedScenario):
            prepared.append(scenario)
            continue
        episode_key = getattr(scenario, "episode_key", None)
        raw_trace = getattr(scenario, "trace", None)
        fault = getattr(scenario, "fault_fingerprint", None)
        if type(episode_key) is not str or not episode_key:
            raise ValueError("each scenario must expose a non-empty episode_key")
        if isinstance(raw_trace, PreparedTrace):
            projection = raw_trace
        else:
            projection = prepare_shared_trace(raw_trace)
        if type(fault) is not str or not fault:
            fault = projection.fault_fingerprint
        prepared.append(PreparedScenario(episode_key, projection, fault))
    if not prepared:
        raise ValueError("at least one scenario is required")
    prepared.sort(key=lambda scenario: scenario.episode_key)
    keys = tuple(scenario.episode_key for scenario in prepared)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate scenario episode_key")
    return tuple(prepared)


def _validate_tasks(
    tasks: Iterable[DeviationTask], expert_count: int
) -> tuple[DeviationTask, ...]:
    materialized = tuple(tasks)
    if not materialized:
        raise ValueError("at least one deviation task is required")
    for task in materialized:
        if not isinstance(task, DeviationTask):
            raise ValueError("tasks must be DeviationTask rows")
        if task.deviating_expert >= expert_count:
            raise ValueError("task Expert is outside the population")
    keys = tuple(task.key for task in materialized)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate deviation task key")
    return tuple(sorted(materialized, key=lambda task: (task.deviating_expert, _rule_key(task.candidate_rule))))


def run_deviation_batch(
    scenarios: Iterable[object],
    tasks: Iterable[DeviationTask],
    *,
    expert_count: int,
    c_b: float,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    max_workers: int = MAX_WORKERS,
    call_budget: int = 1_000_000,
    rule_bank: Iterable[ActionRule] | None = None,
) -> DeviationBatchResult:
    """Run fixed-grain exact tagged reruns with deterministic parent merge."""

    if type(expert_count) is not int or expert_count < 1:
        raise ValueError("expert_count must be a positive int")
    if type(call_budget) is not int or call_budget < 0:
        raise ValueError("call_budget must be a non-negative int")
    if call_budget > 1_000_000:
        raise ValueError("call_budget exceeds the one-million hard limit")
    for name, value in (
        ("c_b", c_b), ("degraded_slowdown", degraded_slowdown),
        ("hedge_delay", hedge_delay),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite real number")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(degraded_slowdown) < 1.0:
        raise ValueError("degraded_slowdown must be >= 1")
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_WORKERS:
        raise ValueError("max_workers must be in 1..8")
    if rule_bank is not None:
        validate_canonical_rule_bank(rule_bank)
    prepared = _prepare_scenarios(scenarios)
    if any(item.prepared_trace.expected_expert_count != expert_count for item in prepared):
        raise ValueError("scenario Expert counts do not match expert_count")
    validated_tasks = _validate_tasks(tasks, expert_count)
    per_task_calls = len(prepared)
    parameters = _WorkerParameters(
        expert_count, float(c_b), float(degraded_slowdown), float(hedge_delay)
    )
    start = time.perf_counter()
    reserved = completed = failed = 0
    rows_by_key: dict[tuple[int, str], DeviationBatchRow] = {}
    pending: dict[object, tuple[DeviationTask, int]] = {}
    next_task = 0
    failure_reasons: list[str] = []
    worker_peaks: list[int] = []
    parent_peak = _rss_bytes()

    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(prepared, parameters),
    )
    try:
        while next_task < len(validated_tasks) or pending:
            while next_task < len(validated_tasks) and len(pending) < min(max_workers, MAX_IN_FLIGHT):
                task = validated_tasks[next_task]
                if reserved + per_task_calls > call_budget:
                    failure_reasons.append("call budget prevents dispatch of the next task")
                    next_task = len(validated_tasks)
                    break
                reserved += per_task_calls
                try:
                    future = executor.submit(_run_deviation_task, task)
                except Exception as exc:
                    failed += per_task_calls
                    failure_reasons.append(f"worker dispatch failed: {exc!r}")
                    next_task += 1
                    continue
                pending[future] = (task, per_task_calls)
                next_task += 1
            if not pending:
                continue
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task, quota = pending.pop(future)
                try:
                    worker_result = future.result()
                except Exception as exc:
                    failed += quota
                    failure_reasons.append(
                        f"task {task.key!r} failed without retry: {exc!r}"
                    )
                    continue
                if (
                    not isinstance(worker_result, _WorkerTaskResult)
                    or worker_result.row.deviating_expert != task.deviating_expert
                    or worker_result.row.candidate_rule != task.candidate_rule.name
                    or len(worker_result.row.scenario_scores) != per_task_calls
                ):
                    failed += quota
                    failure_reasons.append(
                        f"task {task.key!r} returned a duplicate or missing result"
                    )
                    continue
                completed += quota
                rows_by_key[task.key] = worker_result.row
                if worker_result.peak_rss_bytes is not None:
                    worker_peaks.append(worker_result.peak_rss_bytes)
            current_parent = _rss_bytes()
            if current_parent is not None:
                parent_peak = max(parent_peak or current_parent, current_parent)
                if parent_peak > PARENT_RSS_LIMIT:
                    failure_reasons.append("parent RSS exceeded 4 GiB")
                    for future in pending:
                        future.cancel()
                    break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    elapsed = time.perf_counter() - start
    ordered_keys = tuple(task.key for task in validated_tasks if task.key in rows_by_key)
    rows = tuple(rows_by_key[key] for key in ordered_keys)
    if failure_reasons:
        status = "partial" if rows else "failed"
    elif len(rows) != len(validated_tasks):
        status = "partial"
    else:
        status = "complete"
    complete = status == "complete"
    return DeviationBatchResult(
        rows=rows,
        merge_keys=ordered_keys,
        reserved_calls=reserved,
        completed_calls=completed,
        failed_calls=failed,
        complete=complete,
        status=status,
        failure_reason="; ".join(failure_reasons),
        worker_count=max_workers,
        start_method=context.get_start_method(),
        task_granularity="(deviating_expert, candidate_rule)",
        deterministic_merge_order="(expert_id, canonical rule index)",
        elapsed_seconds=elapsed,
        worker_peak_rss_bytes=tuple(sorted(worker_peaks)),
        parent_peak_rss_bytes=parent_peak,
        rss_supported=parent_peak is not None,
    )


__all__ = [
    "DeviationBatchResult",
    "DeviationBatchRow",
    "DeviationTask",
    "MAX_IN_FLIGHT",
    "MAX_WORKERS",
    "PreparedScenario",
    "TaggedScenarioScore",
    "validate_canonical_rule_bank",
    "run_deviation_batch",
]
