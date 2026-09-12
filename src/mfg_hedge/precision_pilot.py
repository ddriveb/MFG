"""Accepted Stage 4b precision pilot.

This module is deliberately isolated from the historical Stage 4 campaign.
It owns only the pilot library, exact four-profile reruns, pooled raw-payload
scoring, the declared common-path and nested diagnostics, and transactional
pilot output.  It never selects a candidate and never claims Nash or MFG.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .artifacts import _validate_run_id
from .common_state import Phase, phase_at
from .domain import TokenClass
from .game_workload import (
    ActionRule,
    build_common_fault_path,
    build_nested_population,
)
from .shared_backup import (
    AttemptStatus,
    PreparedTrace,
    SharedAction,
    prepare_shared_trace,
    simulate_shared_backup_tagged,
)


PILOT_NAMESPACE = "shared-backup-game:v1:stage4b-precision-pilot"
PILOT_MACRO_SEED = 20260907
PILOT_RESAMPLING_SEED = 20260908
PILOT_COMMON_PATHS = 64
PILOT_POPULATION_REPLICATES = 8
PILOT_C_VALUES = (16, 32, 64)
PILOT_P_VALUES = (2, 4, 8)
PILOT_EXPERT_COUNT = 8
PILOT_C_B = 0.5
PILOT_SLOWDOWN = 2.0
PILOT_HEDGE_DELAY = 1.5
PILOT_ARRIVAL_CUTOFF = 360.0
B_OUTER = 4096
B_WITHIN = 256
PILOT_WIDTH = 0.20
PREFLIGHT_MAX_SECONDS = 3600.0
_Z95 = 1.959964
_PROFILE_ORDER = {
    "NSNS/NSNS": 0,
    "NSNS/NSSS": 1,
    "NSNS/NSND": 2,
    "NSNS/NNDX": 3,
}
PILOT_PROFILES = (
    ("NSNS/NSNS", "NSNS", "NSNS"),
    ("NSNS/NSSS", "NSSS", "NSNS"),
    ("NSNS/NSND", "NSND", "NSNS"),
    ("NSNS/NNDX", "NNDX", "NSNS"),
)
PILOT_DEVIATIONS = ("NSSS", "NSND", "NNDX")
_PHASES = ("H", "D", "F", "R")
_CLASSES = ("R", "U")
_ACTION_CODES = frozenset("NDSX")
_ATTEMPT_STATUSES = frozenset(status.value for status in AttemptStatus)
_CLASS_WEIGHT = {"R": 0.8, "U": 0.2}
_DEADLINE = {"R": 3.0, "U": 2.0}
_PENALTY = {"R": 1.0, "U": 5.0}


def _finite(value: object, name: str, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (not allow_negative and result < 0.0):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _freeze(mapping: Mapping[object, object]) -> MappingProxyType:
    return MappingProxyType(dict(mapping))


def _freeze_nested(mapping: Mapping[object, Mapping[object, object]]) -> MappingProxyType:
    return MappingProxyType({key: _freeze(value) for key, value in mapping.items()})


@dataclass(frozen=True)
class PilotConfig:
    """All scientific and sampling inputs accepted by the frozen pilot."""

    namespace: str = PILOT_NAMESPACE
    macro_seed: int = PILOT_MACRO_SEED
    resampling_seed: int = PILOT_RESAMPLING_SEED
    max_common_paths: int = PILOT_COMMON_PATHS
    max_population_replicates: int = PILOT_POPULATION_REPLICATES
    c_values: tuple[int, ...] = PILOT_C_VALUES
    p_values: tuple[int, ...] = PILOT_P_VALUES
    expert_count: int = PILOT_EXPERT_COUNT
    c_b: float = PILOT_C_B
    degraded_slowdown: float = PILOT_SLOWDOWN
    hedge_delay: float = PILOT_HEDGE_DELAY
    arrival_cutoff: float = PILOT_ARRIVAL_CUTOFF
    epsilon_nash: float = PILOT_WIDTH
    pilot_width_target: float = PILOT_WIDTH
    b_outer: int = B_OUTER
    b_within: int = B_WITHIN

    def __post_init__(self) -> None:
        if self.namespace != PILOT_NAMESPACE:
            raise ValueError("Stage 4b namespace is frozen")
        if self.macro_seed != PILOT_MACRO_SEED:
            raise ValueError("Stage 4b macro seed is frozen")
        if self.resampling_seed != PILOT_RESAMPLING_SEED:
            raise ValueError("Stage 4b resampling seed is frozen")
        if self.max_common_paths != PILOT_COMMON_PATHS:
            raise ValueError("Stage 4b common-path count is frozen")
        if self.max_population_replicates != PILOT_POPULATION_REPLICATES:
            raise ValueError("Stage 4b population count is frozen")
        if tuple(self.c_values) != PILOT_C_VALUES or tuple(self.p_values) != PILOT_P_VALUES:
            raise ValueError("Stage 4b C/P grid is frozen")
        if self.expert_count != PILOT_EXPERT_COUNT:
            raise ValueError("Stage 4b requires N=8")
        for name, expected in (
            ("c_b", PILOT_C_B), ("degraded_slowdown", PILOT_SLOWDOWN),
            ("hedge_delay", PILOT_HEDGE_DELAY),
            ("arrival_cutoff", PILOT_ARRIVAL_CUTOFF),
            ("epsilon_nash", PILOT_WIDTH),
            ("pilot_width_target", PILOT_WIDTH),
        ):
            if not math.isclose(_positive(getattr(self, name), name), expected):
                raise ValueError(f"Stage 4b {name} is frozen")
        if self.b_outer != B_OUTER or self.b_within != B_WITHIN:
            raise ValueError("Stage 4b bootstrap counts are frozen")

    def as_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "macro_seed": self.macro_seed,
            "resampling_seed": self.resampling_seed,
            "max_common_paths": self.max_common_paths,
            "max_population_replicates": self.max_population_replicates,
            "c_values": list(self.c_values),
            "p_values": list(self.p_values),
            "expert_count": self.expert_count,
            "c_b": self.c_b,
            "degraded_slowdown": self.degraded_slowdown,
            "hedge_delay": self.hedge_delay,
            "arrival_cutoff": self.arrival_cutoff,
            "epsilon_nash": self.epsilon_nash,
            "pilot_width_target": self.pilot_width_target,
            "b_outer": self.b_outer,
            "b_within": self.b_within,
        }


@dataclass(frozen=True)
class PilotScenario:
    common_path_id: int
    population_id: int
    episode_key: str
    prepared_trace: PreparedTrace
    fault_fingerprint: str

    def __post_init__(self) -> None:
        _int(self.common_path_id, "common_path_id")
        _int(self.population_id, "population_id")
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be a non-empty string")
        if not isinstance(self.prepared_trace, PreparedTrace):
            raise ValueError("prepared_trace must be PreparedTrace")
        if type(self.fault_fingerprint) is not str or not self.fault_fingerprint:
            raise ValueError("fault_fingerprint must be a non-empty string")

    @property
    def trace(self):
        return self.prepared_trace.trace

    @property
    def key(self) -> tuple[int, int]:
        return self.common_path_id, self.population_id


@dataclass(frozen=True)
class PilotLibrary:
    scenarios: tuple[PilotScenario, ...]
    generation_count: int = 1
    library_fingerprint: str = ""

    def __post_init__(self) -> None:
        scenarios = tuple(self.scenarios)
        if len(scenarios) != PILOT_COMMON_PATHS * PILOT_POPULATION_REPLICATES:
            raise ValueError("Stage 4b library must contain exactly 64x8 scenarios")
        expected = tuple(
            (common, population)
            for common in range(PILOT_COMMON_PATHS)
            for population in range(PILOT_POPULATION_REPLICATES)
        )
        if tuple(scenario.key for scenario in scenarios) != expected:
            raise ValueError("Stage 4b library order must be common/population canonical")
        if self.generation_count != 1:
            raise ValueError("Stage 4b library must be generated exactly once")
        digest = hashlib.sha256()
        for scenario in scenarios:
            digest.update(str(scenario.key).encode("ascii"))
            digest.update(scenario.prepared_trace.trace_fingerprint.encode("ascii"))
            digest.update(scenario.fault_fingerprint.encode("ascii"))
        canonical = digest.hexdigest()
        if self.library_fingerprint and self.library_fingerprint != canonical:
            raise ValueError("library fingerprint is not canonical")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "library_fingerprint", canonical)

    def as_dict(self) -> dict[str, object]:
        return {
            "generation_count": self.generation_count,
            "library_fingerprint": self.library_fingerprint,
            "scenario_count": len(self.scenarios),
            "scenario_keys": [list(scenario.key) for scenario in self.scenarios],
            "trace_fingerprints": [
                scenario.prepared_trace.trace_fingerprint for scenario in self.scenarios
            ],
            "fault_fingerprints": [scenario.fault_fingerprint for scenario in self.scenarios],
        }


def build_pilot_library(config: PilotConfig | None = None) -> PilotLibrary:
    """Generate the 64x8 immutable exogenous library once in canonical order."""

    config = config or PilotConfig()
    if not isinstance(config, PilotConfig):
        raise ValueError("config must be PilotConfig")
    scenarios: list[PilotScenario] = []
    for common_id in range(config.max_common_paths):
        fault = build_common_fault_path(
            config.namespace, config.macro_seed, common_id,
            mode="stochastic", arrival_cutoff=config.arrival_cutoff,
        )
        for population_id in range(config.max_population_replicates):
            population = build_nested_population(
                fault, population_id, config.expert_count,
            )
            prepared = prepare_shared_trace(population.trace)
            scenarios.append(PilotScenario(
                common_path_id=common_id,
                population_id=population_id,
                episode_key=(
                    f"{config.namespace}:common-{common_id:02d}:population-{population_id}"
                ),
                prepared_trace=prepared,
                fault_fingerprint=population.common_fault_fingerprint,
            ))
    return PilotLibrary(tuple(scenarios))


def select_prefix_cell(
    library: PilotLibrary, common_paths: int, population_replicates: int,
) -> tuple[PilotScenario, ...]:
    if not isinstance(library, PilotLibrary):
        raise ValueError("library must be PilotLibrary")
    if common_paths not in PILOT_C_VALUES or population_replicates not in PILOT_P_VALUES:
        raise ValueError("C/P must be in the frozen Stage 4b grid")
    return tuple(
        scenario for scenario in library.scenarios
        if scenario.common_path_id < common_paths
        and scenario.population_id < population_replicates
    )


def pilot_call_count(common_paths: int, population_replicates: int) -> int:
    _int(common_paths, "common_paths", 1)
    _int(population_replicates, "population_replicates", 1)
    return 4 * common_paths * population_replicates


def profile_names() -> tuple[str, ...]:
    return tuple(name for name, _deviation, _opponent in PILOT_PROFILES)


@dataclass(frozen=True)
class PilotAttemptPayload:
    attempt_id: int
    replica_id: int
    status: str
    required_work: float
    executed_work: float

    def __post_init__(self) -> None:
        _int(self.attempt_id, "attempt_id")
        if self.attempt_id > 3:
            raise ValueError("attempt_id must be in 0..3")
        if self.replica_id not in (0, 1, 2):
            raise ValueError("replica_id must be 0, 1, or 2")
        if self.status not in _ATTEMPT_STATUSES:
            raise ValueError("invalid attempt status")
        required = _positive(self.required_work, "required_work")
        executed = _finite(self.executed_work, "executed_work")
        if executed > required + 1e-12:
            raise ValueError("executed_work exceeds required_work")
        object.__setattr__(self, "required_work", required)
        object.__setattr__(self, "executed_work", executed)

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "replica_id": self.replica_id,
            "status": self.status,
            "required_work": self.required_work,
            "executed_work": self.executed_work,
        }


@dataclass(frozen=True)
class PilotTokenPayload:
    global_token_id: int
    local_token_id: int
    phase: str
    token_class: str
    arrival_time: float
    latency: float
    replay_count: int
    requested_action: str
    applied_action: str
    attempts: tuple[PilotAttemptPayload, ...]

    def __post_init__(self) -> None:
        _int(self.global_token_id, "global_token_id")
        _int(self.local_token_id, "local_token_id")
        if self.phase not in _PHASES or self.token_class not in _CLASSES:
            raise ValueError("invalid phase or Token class")
        _finite(self.arrival_time, "arrival_time")
        _finite(self.latency, "latency")
        if type(self.replay_count) is not int or self.replay_count not in (0, 1):
            raise ValueError("replay_count must be 0 or 1")
        if self.requested_action not in _ACTION_CODES or self.applied_action not in _ACTION_CODES:
            raise ValueError("invalid requested/applied action")
        if not self.attempts:
            raise ValueError("Token payload requires attempts")
        attempts = tuple(self.attempts)
        if any(not isinstance(attempt, PilotAttemptPayload) for attempt in attempts):
            raise ValueError("invalid attempt payload")
        if len({(a.replica_id, a.attempt_id) for a in attempts}) != len(attempts):
            raise ValueError("duplicate attempt identity")
        object.__setattr__(self, "arrival_time", float(self.arrival_time))
        object.__setattr__(self, "latency", float(self.latency))
        object.__setattr__(self, "attempts", attempts)

    def as_dict(self) -> dict[str, object]:
        return {
            "global_token_id": self.global_token_id,
            "local_token_id": self.local_token_id,
            "phase": self.phase,
            "token_class": self.token_class,
            "arrival_time": self.arrival_time,
            "latency": self.latency,
            "replay_count": self.replay_count,
            "requested_action": self.requested_action,
            "applied_action": self.applied_action,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class PilotScenarioResult:
    """Compact exact tagged result for one profile and one scenario."""

    profile: str
    common_path_id: int
    population_id: int
    episode_key: str
    trace_fingerprint: str
    fault_fingerprint: str
    tokens: tuple[PilotTokenPayload, ...]
    counters: tuple[tuple[str, int], ...]
    pool_total_executed: float
    pool_total_capacity: float
    drain_end_time: float

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_ORDER:
            raise ValueError("unknown Stage 4b profile")
        _int(self.common_path_id, "common_path_id")
        _int(self.population_id, "population_id")
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be non-empty")
        for name in ("trace_fingerprint", "fault_fingerprint"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        tokens = tuple(self.tokens)
        if not tokens or any(not isinstance(token, PilotTokenPayload) for token in tokens):
            raise ValueError("scenario result requires Token payloads")
        ids = tuple(token.global_token_id for token in tokens)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Token payloads must be sorted by unique global ID")
        counters = tuple(self.counters)
        if any(type(name) is not str or type(value) is not int or value < 0
               for name, value in counters):
            raise ValueError("counters must be non-negative named integers")
        _finite(self.pool_total_executed, "pool_total_executed")
        _finite(self.pool_total_capacity, "pool_total_capacity")
        _finite(self.drain_end_time, "drain_end_time")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "counters", counters)

    @property
    def key(self) -> tuple[str, int, int]:
        return self.profile, self.common_path_id, self.population_id

    def as_dict(self, *, include_tokens: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile": self.profile,
            "common_path_id": self.common_path_id,
            "population_id": self.population_id,
            "episode_key": self.episode_key,
            "trace_fingerprint": self.trace_fingerprint,
            "fault_fingerprint": self.fault_fingerprint,
            "token_count": len(self.tokens),
            "counters": {name: value for name, value in self.counters},
            "pool_total_executed": self.pool_total_executed,
            "pool_total_capacity": self.pool_total_capacity,
            "drain_end_time": self.drain_end_time,
        }
        if include_tokens:
            payload["tokens"] = [token.as_dict() for token in self.tokens]
        return payload


def _result_from_tagged(profile: str, scenario: PilotScenario, tagged) -> PilotScenarioResult:
    token_rows = []
    for token in tagged.tokens:
        phase = phase_at(scenario.trace.timeline, token.arrival_time).value
        attempts = tuple(
            PilotAttemptPayload(
                attempt_id=attempt.attempt_id,
                replica_id=attempt.replica_id,
                status=attempt.status.value,
                required_work=attempt.required_work,
                executed_work=attempt.executed_work,
            )
            for attempt in token.attempts
        )
        token_rows.append(PilotTokenPayload(
            global_token_id=token.global_token_id,
            local_token_id=token.local_token_id,
            phase=phase,
            token_class=token.token_class.value,
            arrival_time=token.arrival_time,
            latency=token.latency,
            replay_count=token.replay_count,
            requested_action=token.requested_action.value,
            applied_action=token.action.value,
            attempts=attempts,
        ))
    counters = tuple(
        (name, getattr(tagged.counters, name))
        for name in tagged.counters.__dataclass_fields__
    )
    summary = tagged.pool_audit_summary
    return PilotScenarioResult(
        profile=profile,
        common_path_id=scenario.common_path_id,
        population_id=scenario.population_id,
        episode_key=scenario.episode_key,
        trace_fingerprint=scenario.prepared_trace.trace_fingerprint,
        fault_fingerprint=scenario.fault_fingerprint,
        tokens=tuple(token_rows),
        counters=counters,
        pool_total_executed=summary.total_executed_work,
        pool_total_capacity=summary.total_capacity_bound,
        drain_end_time=tagged.drain_end_time,
    )


class _PilotRuleSource:
    def __init__(self, rule: ActionRule):
        self.rule = rule

    def request(self, observation):
        return self.rule.request(observation)


def _rule(name: str) -> ActionRule:
    return ActionRule(name, tuple(name))


def _profile_sources(profile: str) -> dict[int, _PilotRuleSource]:
    definitions = {name: (deviation, opponent) for name, deviation, opponent in PILOT_PROFILES}
    deviation, opponent = definitions[profile]
    return {
        expert: _PilotRuleSource(_rule(deviation if expert == 0 else opponent))
        for expert in range(PILOT_EXPERT_COUNT)
    }


def _simulate_profile_scenario(profile: str, scenario: PilotScenario) -> PilotScenarioResult:
    tagged = simulate_shared_backup_tagged(
        scenario.prepared_trace,
        PILOT_C_B,
        degraded_slowdown=PILOT_SLOWDOWN,
        hedge_delay=PILOT_HEDGE_DELAY,
        action_sources=_profile_sources(profile),
        tagged_expert_id=0,
    )
    return _result_from_tagged(profile, scenario, tagged)


@dataclass(frozen=True)
class PilotTask:
    profile: str
    start_index: int
    end_index: int

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_ORDER:
            raise ValueError("unknown pilot profile")
        _int(self.start_index, "start_index")
        _int(self.end_index, "end_index")
        if self.end_index <= self.start_index:
            raise ValueError("pilot task must contain at least one scenario")

    @property
    def scenario_count(self) -> int:
        return self.end_index - self.start_index


@dataclass(frozen=True)
class _PilotTaskResult:
    task: PilotTask
    results: tuple[PilotScenarioResult, ...]
    attempted_calls: int
    complete: bool
    failure_reason: str
    peak_rss_bytes: int | None


_WORKER_SCENARIOS: tuple[PilotScenario, ...] = ()


def _rss_bytes() -> int | None:
    """Read actual Windows peak working set, never Python heap size."""

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
        ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def _pilot_worker_init(scenarios: tuple[PilotScenario, ...]) -> None:
    global _WORKER_SCENARIOS
    _WORKER_SCENARIOS = scenarios


def _run_pilot_task(task: PilotTask) -> _PilotTaskResult:
    if not _WORKER_SCENARIOS:
        raise RuntimeError("pilot worker was not initialized")
    started = _rss_bytes()
    results: list[PilotScenarioResult] = []
    try:
        for index in range(task.start_index, task.end_index):
            results.append(_simulate_profile_scenario(task.profile, _WORKER_SCENARIOS[index]))
        peak = _rss_bytes()
        if peak is not None and started is not None:
            peak = max(peak, started)
        if peak is not None and peak > 512 * 1024 * 1024:
            raise MemoryError("pilot worker RSS exceeded 512 MiB")
        return _PilotTaskResult(
            task, tuple(results), len(results), True, "", peak,
        )
    except BaseException as exc:
        peak = _rss_bytes()
        if peak is not None and started is not None:
            peak = max(peak, started)
        return _PilotTaskResult(
            task, tuple(results), len(results), False, repr(exc), peak,
        )


@dataclass(frozen=True)
class PilotRun:
    results: tuple[PilotScenarioResult, ...]
    reserved_calls: int
    completed_calls: int
    failed_calls: int
    expected_calls: int
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
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def scheduler_calls(self) -> int:
        return self.completed_calls + self.failed_calls

    def lookup(self) -> dict[tuple[str, int, int], PilotScenarioResult]:
        return {result.key: result for result in self.results}

    def as_dict(self, *, include_tokens: bool = False) -> dict[str, object]:
        return {
            "reserved_calls": self.reserved_calls,
            "completed_calls": self.completed_calls,
            "failed_calls": self.failed_calls,
            "scheduler_calls": self.scheduler_calls,
            "expected_calls": self.expected_calls,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "worker_count": self.worker_count,
            "start_method": self.start_method,
            "task_granularity": self.task_granularity,
            "deterministic_merge_order": self.deterministic_merge_order,
            "elapsed_seconds": self.elapsed_seconds,
            "worker_peak_rss_bytes": list(self.worker_peak_rss_bytes),
            "parent_peak_rss_bytes": self.parent_peak_rss_bytes,
            "rss_supported": self.rss_supported,
            "results": [result.as_dict(include_tokens=include_tokens) for result in self.results],
        }


def _validate_library(library: PilotLibrary) -> None:
    if not isinstance(library, PilotLibrary):
        raise ValueError("library must be PilotLibrary")
    expected = {
        (profile, scenario.common_path_id, scenario.population_id)
        for profile in profile_names()
        for scenario in library.scenarios
    }
    if len(expected) != len(profile_names()) * len(library.scenarios):
        raise ValueError("pilot library scenario keys are not unique")


def _pilot_tasks(scenario_count: int, max_workers: int) -> tuple[PilotTask, ...]:
    parts = max(1, max_workers // len(PILOT_PROFILES))
    tasks: list[PilotTask] = []
    for profile, _deviation, _opponent in PILOT_PROFILES:
        for part in range(parts):
            start = (scenario_count * part) // parts
            end = (scenario_count * (part + 1)) // parts
            if end > start:
                tasks.append(PilotTask(profile, start, end))
    return tuple(tasks)


def run_pilot_scheduler(
    library: PilotLibrary,
    *,
    max_workers: int = 8,
) -> PilotRun:
    """Run all four profiles exactly once per immutable library scenario."""

    _validate_library(library)
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be in 1..8")
    tasks = _pilot_tasks(len(library.scenarios), max_workers)
    expected_calls = len(library.scenarios) * len(PILOT_PROFILES)
    started = time.perf_counter()
    reserved = completed = failed = 0
    results_by_key: dict[tuple[str, int, int], PilotScenarioResult] = {}
    failures: list[str] = []
    worker_peaks: list[int] = []
    parent_peak = _rss_bytes()
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_pilot_worker_init,
        initargs=(library.scenarios,),
    )
    pending: dict[object, PilotTask] = {}
    next_task = 0
    try:
        while next_task < len(tasks) or pending:
            while next_task < len(tasks) and len(pending) < min(8, max_workers):
                task = tasks[next_task]
                if reserved + task.scenario_count > expected_calls:
                    failures.append("call reservation exceeded frozen 2,048 calls")
                    next_task = len(tasks)
                    break
                reserved += task.scenario_count
                pending[executor.submit(_run_pilot_task, task)] = task
                next_task += 1
            if not pending:
                continue
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    worker_result = future.result()
                except BaseException as exc:
                    failed += task.scenario_count
                    failures.append(f"task {task.profile}:{task.start_index} failed without retry: {exc!r}")
                    continue
                completed += worker_result.attempted_calls
                failed += task.scenario_count - worker_result.attempted_calls
                if worker_result.peak_rss_bytes is not None:
                    worker_peaks.append(worker_result.peak_rss_bytes)
                if not worker_result.complete:
                    failures.append(
                        f"task {task.profile}:{task.start_index} failed without retry: "
                        f"{worker_result.failure_reason}"
                    )
                for result in worker_result.results:
                    key = result.key
                    if key in results_by_key:
                        failures.append(f"duplicate pilot result {key!r}")
                    results_by_key[key] = result
            current_parent = _rss_bytes()
            if current_parent is not None:
                parent_peak = max(parent_peak or current_parent, current_parent)
                if parent_peak > 4 * 1024 * 1024 * 1024:
                    failures.append("parent RSS exceeded 4 GiB")
                    for future in pending:
                        future.cancel()
                    break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    elapsed = time.perf_counter() - started
    ordered = tuple(
        results_by_key[key]
        for profile in profile_names()
        for scenario in library.scenarios
        for key in ((profile, scenario.common_path_id, scenario.population_id),)
        if key in results_by_key
    )
    if failures:
        status = "partial" if ordered else "failed"
    elif len(ordered) != expected_calls:
        status = "partial"
        failures.append("missing pilot result rows")
    else:
        status = "complete"
    return PilotRun(
        results=ordered,
        reserved_calls=reserved,
        completed_calls=completed,
        failed_calls=failed,
        expected_calls=expected_calls,
        status=status,
        failure_reason="; ".join(failures),
        worker_count=max_workers,
        start_method=context.get_start_method(),
        task_granularity="(profile, scenario chunk)",
        deterministic_merge_order="(profile order, common_path_id, population_id)",
        elapsed_seconds=elapsed,
        worker_peak_rss_bytes=tuple(sorted(worker_peaks)),
        parent_peak_rss_bytes=parent_peak,
        rss_supported=parent_peak is not None,
    )


@dataclass(frozen=True)
class PooledObjective:
    total: float | None
    complete: bool
    missing_cohorts: tuple[tuple[str, str], ...]
    generated_token_count: int
    phase_class_counts: Mapping[str, Mapping[str, int]]
    phase_totals: Mapping[str, int]
    phase_losses: Mapping[str, float | None]
    cvar95: Mapping[str, float | None]
    class_cvar95: Mapping[str, Mapping[str, float | None]]
    effective_tail_mass: Mapping[str, float]
    raw_work_totals: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_class_counts", _freeze_nested(self.phase_class_counts))
        object.__setattr__(self, "phase_totals", _freeze(self.phase_totals))
        object.__setattr__(self, "phase_losses", _freeze(self.phase_losses))
        object.__setattr__(self, "cvar95", _freeze(self.cvar95))
        object.__setattr__(self, "class_cvar95", _freeze_nested(self.class_cvar95))
        object.__setattr__(self, "effective_tail_mass", _freeze(self.effective_tail_mass))
        object.__setattr__(self, "raw_work_totals", _freeze(self.raw_work_totals))
        if self.complete != (not self.missing_cohorts):
            raise ValueError("objective complete flag disagrees with missing cohorts")
        if self.complete and self.total is None:
            raise ValueError("complete objective requires a total")
        if not self.complete and self.total is not None:
            raise ValueError("incomplete objective must have null total")

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "complete": self.complete,
            "missing_cohorts": [list(value) for value in self.missing_cohorts],
            "generated_token_count": self.generated_token_count,
            "phase_class_counts": {
                phase: dict(values) for phase, values in self.phase_class_counts.items()
            },
            "phase_totals": dict(self.phase_totals),
            "phase_losses": dict(self.phase_losses),
            "cvar95": dict(self.cvar95),
            "class_cvar95": {
                phase: dict(values) for phase, values in self.class_cvar95.items()
            },
            "effective_tail_mass": dict(self.effective_tail_mass),
            "raw_work_totals": dict(self.raw_work_totals),
        }


def _weighted_cvar(values: Sequence[tuple[float, int]]) -> float:
    if not values:
        raise ValueError("CVaR requires at least one value")
    total_count = sum(weight for _value, weight in values)
    if total_count <= 0:
        raise ValueError("CVaR requires positive multiplicity")
    ordered = sorted(values, key=lambda item: item[0], reverse=True)
    tail_mass = total_count / 20.0
    remaining = tail_mass
    terms: list[float] = []
    for value, multiplicity in ordered:
        take = min(float(multiplicity), remaining)
        if take > 0.0:
            terms.append(value * take)
            remaining -= take
        if remaining <= 1e-15:
            break
    if remaining > 1e-12:
        raise RuntimeError("weighted CVaR tail construction underflowed")
    return math.fsum(terms) / tail_mass


def score_pooled_payloads(
    results: Iterable[PilotScenarioResult],
    *,
    multiplicities: Mapping[tuple[int, int], int] | None = None,
) -> PooledObjective:
    """Score raw Token/attempt payloads pooled over scenarios exactly."""

    rows = tuple(sorted(results, key=lambda row: row.key))
    if not rows or any(not isinstance(row, PilotScenarioResult) for row in rows):
        raise ValueError("pooled scoring requires nonempty PilotScenarioResult rows")
    keys = tuple((row.common_path_id, row.population_id) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("pooled scoring rejects duplicate scenario keys")
    if multiplicities is None:
        weights = {key: 1 for key in keys}
    else:
        weights = {}
        for key, value in multiplicities.items():
            if type(key) is not tuple or len(key) != 2 or type(value) is not int or value < 0:
                raise ValueError("multiplicities must map keys to non-negative integers")
            weights[key] = value
        if set(weights) != set(keys):
            raise ValueError("multiplicities must cover exactly the pooled scenario keys")
        if not any(weights.values()):
            raise ValueError("pooled multiplicities must have positive total")

    rows_by_cohort: dict[tuple[str, str], list[tuple[float, int]]] = {
        (phase, cls): [] for phase in _PHASES for cls in _CLASSES
    }
    cvar_by_phase: dict[str, list[tuple[float, int]]] = {phase: [] for phase in _PHASES}
    phase_class_counts = {phase: {cls: 0 for cls in _CLASSES} for phase in _PHASES}
    phase_totals = {phase: 0 for phase in _PHASES}
    replay_counts = {phase: {cls: 0 for cls in _CLASSES} for phase in _PHASES}
    attempts: list[tuple[PilotAttemptPayload, int]] = []
    selected_count = 0
    for row in rows:
        weight = weights[(row.common_path_id, row.population_id)]
        if weight == 0:
            continue
        selected_count += weight * len(row.tokens)
        for token in row.tokens:
            phase_class_counts[token.phase][token.token_class] += weight
            phase_totals[token.phase] += weight
            replay_counts[token.phase][token.token_class] += weight * token.replay_count
            rows_by_cohort[(token.phase, token.token_class)].append((token.latency, weight))
            cvar_by_phase[token.phase].append((token.latency, weight))
            attempts.extend((attempt, weight) for attempt in token.attempts)

    if selected_count <= 0:
        raise ValueError("pooled scoring selected no Token payloads")
    missing = tuple(
        (phase, cls)
        for phase in _PHASES for cls in _CLASSES
        if phase_class_counts[phase][cls] == 0
    )
    phase_losses: dict[str, float | None] = {}
    class_cvar95: dict[str, dict[str, float | None]] = {
        phase: {cls: None for cls in _CLASSES} for phase in _PHASES
    }
    for phase in _PHASES:
        for cls in _CLASSES:
            values = rows_by_cohort[(phase, cls)]
            if values:
                class_cvar95[phase][cls] = _weighted_cvar(values)
        if all(phase_class_counts[phase][cls] > 0 for cls in _CLASSES):
            phase_losses[phase] = math.fsum(
                _CLASS_WEIGHT[cls] * (
                    math.fsum(value * weight for value, weight in rows_by_cohort[(phase, cls)])
                    / phase_class_counts[phase][cls]
                    + _PENALTY[cls] * math.fsum(
                        max(0.0, value - _DEADLINE[cls]) * weight
                        for value, weight in rows_by_cohort[(phase, cls)]
                    ) / phase_class_counts[phase][cls]
                    + _PENALTY[cls] * replay_counts[phase][cls]
                    / phase_class_counts[phase][cls]
                )
                for cls in _CLASSES
            )
        else:
            phase_losses[phase] = None

    cvar95 = {
        phase: (_weighted_cvar(cvar_by_phase[phase]) if cvar_by_phase[phase] else None)
        for phase in _PHASES
    }
    effective_tail_mass = {
        phase: 0.05 * count for phase, count in phase_totals.items()
    }
    effective_tail_mass.update({
        f"{phase}/{cls}": 0.05 * phase_class_counts[phase][cls]
        for phase in _PHASES for cls in _CLASSES
    })
    executed = math.fsum(attempt.executed_work * weight for attempt, weight in attempts)
    wasted = math.fsum(
        attempt.executed_work * weight
        for attempt, weight in attempts
        if attempt.status != AttemptStatus.COMPLETED_WINNER.value
    )
    raw_work = {
        "executed": executed,
        "wasted": wasted,
        "winner_executed": math.fsum(
            attempt.executed_work * weight
            for attempt, weight in attempts
            if attempt.status == AttemptStatus.COMPLETED_WINNER.value
        ),
        "attempt_count": float(sum(weight for _attempt, weight in attempts)),
    }
    phase_component = (
        math.fsum(phase_losses[phase] for phase in _PHASES) / 4.0
        if not missing else None
    )
    fault_tail = (
        math.fsum(cvar95[phase] for phase in ("D", "F")) / 2.0
        if not missing else None
    )
    work_per_token = executed / selected_count
    waste_per_token = wasted / selected_count
    total = (
        math.fsum((phase_component, fault_tail, work_per_token, waste_per_token))
        if not missing else None
    )
    return PooledObjective(
        total=total,
        complete=not missing,
        missing_cohorts=missing,
        generated_token_count=selected_count,
        phase_class_counts=phase_class_counts,
        phase_totals=phase_totals,
        phase_losses=phase_losses,
        cvar95=cvar95,
        class_cvar95=class_cvar95,
        effective_tail_mass=effective_tail_mass,
        raw_work_totals=raw_work,
    )


def jackknife_se(values: Sequence[float | None]) -> float:
    if len(values) < 2 or any(value is None for value in values):
        return math.inf
    samples = tuple(_finite(value, "delete-one contrast", allow_negative=True) for value in values)
    mean = math.fsum(samples) / len(samples)
    return math.sqrt(
        (len(samples) - 1) / len(samples)
        * math.fsum((value - mean) ** 2 for value in samples)
    )


@dataclass(frozen=True)
class NestedVariance:
    v_between_raw: float
    v_within: float
    v_between: float
    v_total: float
    se_nested: float
    percentile_width: float | None
    repeated_path_positions: tuple[tuple[int, ...], ...] = ()
    independent_inner_by_position: bool = True
    paired_rule_indices: bool = True
    invalid_replicates: int = 0
    complete: bool = True

    def __post_init__(self) -> None:
        for name in ("v_between_raw", "v_within", "v_between", "v_total", "se_nested"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.percentile_width is not None:
            _finite(self.percentile_width, "percentile_width")
        if self.invalid_replicates < 0:
            raise ValueError("invalid_replicates must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "v_between_raw": self.v_between_raw,
            "v_within": self.v_within,
            "v_between": self.v_between,
            "v_total": self.v_total,
            "se_nested": self.se_nested,
            "percentile_width": self.percentile_width,
            "repeated_path_positions": [list(row) for row in self.repeated_path_positions],
            "independent_inner_by_position": self.independent_inner_by_position,
            "paired_rule_indices": self.paired_rule_indices,
            "invalid_replicates": self.invalid_replicates,
            "complete": self.complete,
        }


def corrected_nested_variance(
    outer_means: Sequence[float],
    within_variances: Sequence[float],
    *,
    b_within: int,
    repeated_path_positions: Sequence[Sequence[int]] = (),
    paired_rule_indices: bool = True,
    percentile_values: Sequence[float] | None = None,
    invalid_replicates: int = 0,
) -> NestedVariance:
    if type(b_within) is not int or b_within <= 0:
        raise ValueError("b_within must be a positive int")
    if len(outer_means) < 2 or len(outer_means) != len(within_variances):
        raise ValueError("nested variance requires paired outer rows")
    means = tuple(_finite(value, "outer mean", allow_negative=True) for value in outer_means)
    within = tuple(_finite(value, "within variance") for value in within_variances)
    mean_of_means = math.fsum(means) / len(means)
    raw = math.fsum((value - mean_of_means) ** 2 for value in means) / (len(means) - 1)
    v_within = math.fsum(within) / len(within)
    v_between = max(0.0, raw - v_within / b_within)
    v_total = v_between + v_within
    percentile_width = None
    if percentile_values:
        ordered = sorted(_finite(value, "nested percentile value", allow_negative=True)
                         for value in percentile_values)
        def quantile(q: float) -> float:
            position = q * (len(ordered) - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                return ordered[lower]
            fraction = position - lower
            return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        percentile_width = quantile(0.975) - quantile(0.025)
    return NestedVariance(
        v_between_raw=raw,
        v_within=v_within,
        v_between=v_between,
        v_total=v_total,
        se_nested=math.sqrt(v_total),
        percentile_width=percentile_width,
        repeated_path_positions=tuple(tuple(row) for row in repeated_path_positions),
        independent_inner_by_position=True,
        paired_rule_indices=paired_rule_indices,
        invalid_replicates=invalid_replicates,
    )


def _child_rng(config: PilotConfig, common_paths: int, population_replicates: int,
               layer: str, replicate: int) -> random.Random:
    payload = json.dumps(
        [config.namespace, config.resampling_seed, common_paths,
         population_replicates, layer, replicate],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _nested_diagnostics(
    results_by_profile: Mapping[str, Mapping[tuple[int, int], PilotScenarioResult]],
    config: PilotConfig,
    common_paths: int,
    population_replicates: int,
    *,
    b_outer: int,
    b_within: int,
) -> dict[str, NestedVariance]:
    required_profiles = set(profile_names())
    if set(results_by_profile) != required_profiles:
        raise ValueError("nested diagnostics require all four profiles")
    all_keys = {
        (common, population)
        for common in range(common_paths)
        for population in range(population_replicates)
    }
    for profile in profile_names():
        if set(results_by_profile[profile]) != all_keys:
            raise ValueError("nested diagnostics require a complete rectangular result set")
    outer_values = {rule: [] for rule in PILOT_DEVIATIONS}
    within_values = {rule: [] for rule in PILOT_DEVIATIONS}
    percentile_values = {rule: [] for rule in PILOT_DEVIATIONS}
    invalid = {rule: 0 for rule in PILOT_DEVIATIONS}
    repeated_positions: list[tuple[int, ...]] = []
    for outer_index in range(b_outer):
        outer_rng = _child_rng(config, common_paths, population_replicates, "outer", outer_index)
        selected_common = tuple(outer_rng.randrange(common_paths) for _ in range(common_paths))
        counts_by_inner: list[dict[tuple[int, int], int]] = []
        for inner_index in range(b_within):
            inner_rng = _child_rng(
                config, common_paths, population_replicates,
                f"inner:{outer_index}", inner_index,
            )
            counts: dict[tuple[int, int], int] = {key: 0 for key in all_keys}
            for common in selected_common:
                # Each path position gets a fresh draw, including repeated IDs.
                for _ in range(population_replicates):
                    population = inner_rng.randrange(population_replicates)
                    counts[(common, population)] += 1
            counts_by_inner.append(counts)
        repeated = tuple(common for common in sorted(set(selected_common))
                         if selected_common.count(common) > 1)
        if repeated:
            repeated_positions.append(repeated)
        inner_values = {rule: [] for rule in PILOT_DEVIATIONS}
        for counts in counts_by_inner:
            scores: dict[str, float] = {}
            base = score_pooled_payloads(
                tuple(results_by_profile["NSNS/NSNS"].values()),
                multiplicities=counts,
            )
            for rule in PILOT_DEVIATIONS:
                deviation_profile = {
                    "NSSS": "NSNS/NSSS",
                    "NSND": "NSNS/NSND",
                    "NNDX": "NSNS/NNDX",
                }[rule]
                dev = score_pooled_payloads(
                    tuple(results_by_profile[deviation_profile].values()),
                    multiplicities=counts,
                )
                if base.total is None or dev.total is None:
                    invalid[rule] += 1
                else:
                    scores[rule] = base.total - dev.total
            for rule in PILOT_DEVIATIONS:
                if rule in scores:
                    inner_values[rule].append(scores[rule])
                    percentile_values[rule].append(scores[rule])
        for rule in PILOT_DEVIATIONS:
            block = inner_values[rule]
            if len(block) == b_within:
                outer_values[rule].append(math.fsum(block) / len(block))
                within_values[rule].append(math.fsum(
                    (value - math.fsum(block) / len(block)) ** 2 for value in block
                ) / (len(block) - 1) if len(block) > 1 else 0.0)
            else:
                invalid[rule] += 1

    output: dict[str, NestedVariance] = {}
    for rule in PILOT_DEVIATIONS:
        if len(outer_values[rule]) < 2 or len(within_values[rule]) != len(outer_values[rule]):
            output[rule] = NestedVariance(
                v_between_raw=0.0, v_within=0.0, v_between=0.0,
                v_total=0.0, se_nested=0.0, percentile_width=None,
                repeated_path_positions=tuple(repeated_positions),
                paired_rule_indices=True, invalid_replicates=invalid[rule],
                complete=False,
            )
            continue
        output[rule] = corrected_nested_variance(
            outer_values[rule], within_values[rule], b_within=b_within,
            repeated_path_positions=repeated_positions,
            paired_rule_indices=True,
            percentile_values=percentile_values[rule],
            invalid_replicates=invalid[rule],
        )
        if invalid[rule]:
            output[rule] = replace(output[rule], complete=False)
    return output


@dataclass(frozen=True)
class PilotPreflight:
    scheduler_preflight_calls: int
    scheduler_seconds: float
    pooled_score_seconds: float
    nested_benchmark_seconds: float
    nested_benchmark_objective_calls: int
    nested_objective_calls: int
    estimated_nested_seconds: float
    rss_supported: bool
    parent_peak_rss_bytes: int | None
    feasible: bool
    blocker: str

    def as_dict(self) -> dict[str, object]:
        return {
            "scheduler_preflight_calls": self.scheduler_preflight_calls,
            "scheduler_seconds": self.scheduler_seconds,
            "pooled_score_seconds": self.pooled_score_seconds,
            "nested_benchmark_seconds": self.nested_benchmark_seconds,
            "nested_benchmark_objective_calls": self.nested_benchmark_objective_calls,
            "nested_objective_calls": self.nested_objective_calls,
            "estimated_nested_seconds": self.estimated_nested_seconds,
            "rss_supported": self.rss_supported,
            "parent_peak_rss_bytes": self.parent_peak_rss_bytes,
            "feasible": self.feasible,
            "blocker": self.blocker,
        }


def _clone_result(result: PilotScenarioResult, profile: str, common: int, population: int) -> PilotScenarioResult:
    return replace(
        result,
        profile=profile,
        common_path_id=common,
        population_id=population,
        episode_key=f"preflight:{profile}:{common}:{population}",
        trace_fingerprint=f"preflight-trace:{common}:{population}",
        fault_fingerprint=f"preflight-fault:{common}",
    )


def run_pilot_preflight(
    library: PilotLibrary,
    config: PilotConfig | None = None,
) -> PilotPreflight:
    """Measure scheduler/scorer/nested exact work before formal pilot calls."""

    config = config or PilotConfig()
    _validate_library(library)
    parent_peak = _rss_bytes()
    profile = "NSNS/NSNS"
    first = library.scenarios[0]
    _simulate_profile_scenario(profile, first)  # warm-up scheduler call
    started = time.perf_counter()
    sample = _simulate_profile_scenario(profile, first)
    scheduler_seconds = time.perf_counter() - started
    parent_peak = max(parent_peak or 0, _rss_bytes() or 0) or parent_peak

    pooled_rows = tuple(
        _clone_result(sample, profile, index // 2, index % 2)
        for index in range(32)
    )
    score_started = time.perf_counter()
    score_pooled_payloads(pooled_rows)
    score_pooled_payloads(pooled_rows)
    pooled_seconds = time.perf_counter() - score_started

    # A full-size 64x8 synthetic payload measures exact scorer work without
    # dispatching formal scheduler calls or writing a pilot artifact.
    profiles: dict[str, dict[tuple[int, int], PilotScenarioResult]] = {}
    for profile_name_value in profile_names():
        profiles[profile_name_value] = {}
        for common in range(PILOT_COMMON_PATHS):
            for population in range(PILOT_POPULATION_REPLICATES):
                profiles[profile_name_value][(common, population)] = _clone_result(
                    sample, profile_name_value, common, population,
                )
    nested_started = time.perf_counter()
    _nested_diagnostics(
        profiles, config, PILOT_COMMON_PATHS, PILOT_POPULATION_REPLICATES,
        b_outer=2, b_within=2,
    )
    nested_seconds = time.perf_counter() - nested_started
    benchmark_calls = 2 * 2 * 4
    nested_calls = len(PILOT_C_VALUES) * len(PILOT_P_VALUES) * config.b_outer * config.b_within * 4
    per_call = nested_seconds / benchmark_calls if nested_seconds > 0.0 else math.inf
    estimated = per_call * nested_calls
    rss_supported = parent_peak is not None
    blocker = ""
    feasible = True
    if not rss_supported:
        feasible = False
        blocker = "real Windows RSS measurement is unsupported"
    elif estimated > PREFLIGHT_MAX_SECONDS:
        feasible = False
        blocker = (
            f"exact nested resampling estimated at {estimated:.3f}s, above the "
            f"{PREFLIGHT_MAX_SECONDS:.1f}s operational preflight limit"
        )
    return PilotPreflight(
        scheduler_preflight_calls=2,
        scheduler_seconds=scheduler_seconds,
        pooled_score_seconds=pooled_seconds,
        nested_benchmark_seconds=nested_seconds,
        nested_benchmark_objective_calls=benchmark_calls,
        nested_objective_calls=nested_calls,
        estimated_nested_seconds=estimated,
        rss_supported=rss_supported,
        parent_peak_rss_bytes=parent_peak,
        feasible=feasible,
        blocker=blocker,
    )


@dataclass(frozen=True)
class PilotRuleRow:
    common_paths: int
    population_replicates: int
    rule: str
    baseline_objective: float | None
    deviation_objective: float | None
    gain_g: float | None
    delete_one_g: tuple[float | None, ...]
    common_jackknife_se: float
    diagnostic_width: float
    nested: NestedVariance | None
    phase_class_counts: Mapping[str, Mapping[str, int]]
    phase_totals: Mapping[str, int]
    effective_tail_mass: Mapping[str, float]
    largest_path_influence_share: float | None
    top_two_path_influence_share: float | None
    max_path_variance_contribution: float | None
    rank_flip_count: int
    calls: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "common_paths": self.common_paths,
            "population_replicates": self.population_replicates,
            "rule": self.rule,
            "baseline_objective": self.baseline_objective,
            "deviation_objective": self.deviation_objective,
            "gain_g": self.gain_g,
            "delete_one_g": list(self.delete_one_g),
            "common_jackknife_se": self.common_jackknife_se,
            "diagnostic_width": self.diagnostic_width,
            "nested": self.nested.as_dict() if self.nested else None,
            "phase_class_counts": {
                phase: dict(values) for phase, values in self.phase_class_counts.items()
            },
            "phase_totals": dict(self.phase_totals),
            "effective_tail_mass": dict(self.effective_tail_mass),
            "largest_path_influence_share": self.largest_path_influence_share,
            "top_two_path_influence_share": self.top_two_path_influence_share,
            "max_path_variance_contribution": self.max_path_variance_contribution,
            "rank_flip_count": self.rank_flip_count,
            "calls": self.calls,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class PilotCellSummary:
    common_paths: int
    population_replicates: int
    se_max: float
    width_max: float
    calls: int
    complete: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "common_paths": self.common_paths,
            "population_replicates": self.population_replicates,
            "se_max": self.se_max,
            "width_max": self.width_max,
            "calls": self.calls,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class PilotAnalysis:
    rows: tuple[PilotRuleRow, ...]
    cells: tuple[PilotCellSummary, ...]
    marginal_efficiency: tuple[Mapping[str, object], ...]
    disposition: str
    recommended_cell: tuple[int, int] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "cells": [cell.as_dict() for cell in self.cells],
            "marginal_efficiency": [dict(row) for row in self.marginal_efficiency],
            "disposition": self.disposition,
            "recommended_cell": list(self.recommended_cell) if self.recommended_cell else None,
            "claims_candidate": False,
            "claims_nash": False,
            "claims_mfg": False,
        }


def _rank_g(values: Mapping[str, float | None]) -> tuple[str, ...]:
    if any(value is None for value in values.values()):
        return ()
    return tuple(sorted(values, key=lambda rule: (-float(values[rule]), PILOT_DEVIATIONS.index(rule))))


def _analyze_cell(
    run: PilotRun,
    config: PilotConfig,
    common_paths: int,
    population_replicates: int,
    *,
    include_nested: bool,
) -> tuple[tuple[PilotRuleRow, ...], PilotCellSummary]:
    lookup = run.lookup()
    # Cell analysis derives the stable rectangular key prefix directly from
    # the one completed maximum-library run; it never regenerates a trace.
    keys = tuple(
        (common, population)
        for common in range(common_paths)
        for population in range(population_replicates)
    )
    base = {
        key: lookup[("NSNS/NSNS", *key)] for key in keys
    }
    deviations = {
        rule: {key: lookup[(f"NSNS/{rule}", *key)] for key in keys}
        for rule in PILOT_DEVIATIONS
    }
    base_rows = tuple(base.values())
    baseline = score_pooled_payloads(base_rows)
    delete_one_by_rule: dict[str, list[float | None]] = {rule: [] for rule in PILOT_DEVIATIONS}
    for common in range(common_paths):
        retained_keys = tuple(key for key in keys if key[0] != common)
        base_minus = score_pooled_payloads(tuple(base[key] for key in retained_keys))
        for rule in PILOT_DEVIATIONS:
            dev_minus = score_pooled_payloads(tuple(deviations[rule][key] for key in retained_keys))
            delete_one_by_rule[rule].append(
                None if base_minus.total is None or dev_minus.total is None
                else base_minus.total - dev_minus.total
            )
    nested_by_rule = None
    if include_nested:
        nested_by_rule = _nested_diagnostics(
            {
                "NSNS/NSNS": base,
                "NSNS/NSSS": deviations["NSSS"],
                "NSNS/NSND": deviations["NSND"],
                "NSNS/NNDX": deviations["NNDX"],
            }, config, common_paths, population_replicates,
            b_outer=config.b_outer, b_within=config.b_within,
        )
    full_g: dict[str, float | None] = {}
    for rule in PILOT_DEVIATIONS:
        dev = score_pooled_payloads(tuple(deviations[rule].values()))
        full_g[rule] = (
            None if baseline.total is None or dev.total is None
            else baseline.total - dev.total
        )
    full_rank = _rank_g(full_g)
    rows: list[PilotRuleRow] = []
    for rule in PILOT_DEVIATIONS:
        delete_one = tuple(delete_one_by_rule[rule])
        se = jackknife_se(delete_one)
        q = []
        mean = math.fsum(value for value in delete_one if value is not None) / len(delete_one) \
            if delete_one and all(value is not None for value in delete_one) else None
        if mean is not None:
            q = [(value - mean) ** 2 for value in delete_one if value is not None]
        qsum = math.fsum(q) if q else 0.0
        influence = sorted(q, reverse=True)
        rank_flips = 0
        for omitted_index in range(common_paths):
            ranks = {
                candidate: delete_one_by_rule[candidate][omitted_index]
                for candidate in PILOT_DEVIATIONS
            }
            if full_rank and _rank_g(ranks) and _rank_g(ranks) != full_rank:
                rank_flips += 1
        dev_objective = None
        if full_g[rule] is not None and baseline.total is not None:
            dev_objective = baseline.total - full_g[rule]
        rows.append(PilotRuleRow(
            common_paths=common_paths,
            population_replicates=population_replicates,
            rule=rule,
            baseline_objective=baseline.total,
            deviation_objective=dev_objective,
            gain_g=full_g[rule],
            delete_one_g=delete_one,
            common_jackknife_se=se,
            diagnostic_width=2.0 * _Z95 * se,
            nested=nested_by_rule[rule] if nested_by_rule else None,
            phase_class_counts=baseline.phase_class_counts,
            phase_totals=baseline.phase_totals,
            effective_tail_mass=baseline.effective_tail_mass,
            largest_path_influence_share=(influence[0] / qsum if qsum else 0.0) if q else None,
            top_two_path_influence_share=(math.fsum(influence[:2]) / qsum if qsum else 0.0) if q else None,
            max_path_variance_contribution=(max(q) / qsum if qsum else 0.0) if q else None,
            rank_flip_count=rank_flips,
            calls=pilot_call_count(common_paths, population_replicates),
            elapsed_seconds=run.elapsed_seconds * pilot_call_count(common_paths, population_replicates) / run.expected_calls,
        ))
    se_values = tuple(row.common_jackknife_se for row in rows)
    widths = tuple(row.diagnostic_width for row in rows)
    complete = all(math.isfinite(value) for value in se_values) and all(
        row.nested is not None and row.nested.invalid_replicates == 0 for row in rows
    ) if include_nested else all(math.isfinite(value) for value in se_values)
    return tuple(rows), PilotCellSummary(
        common_paths, population_replicates,
        max(se_values) if se_values else math.inf,
        max(widths) if widths else math.inf,
        pilot_call_count(common_paths, population_replicates),
        complete,
    )


def analyze_pilot(
    run: PilotRun,
    library: PilotLibrary,
    config: PilotConfig | None = None,
    *,
    include_nested: bool = True,
) -> PilotAnalysis:
    config = config or PilotConfig()
    if not run.complete:
        raise ValueError("cannot analyze an incomplete pilot scheduler run")
    _validate_library(library)
    if run.scheduler_calls != pilot_call_count(PILOT_COMMON_PATHS, PILOT_POPULATION_REPLICATES):
        raise ValueError("pilot scheduler call count is not exactly 2,048")
    rows: list[PilotRuleRow] = []
    cells: list[PilotCellSummary] = []
    for common in PILOT_C_VALUES:
        for population in PILOT_P_VALUES:
            cell_rows, cell = _analyze_cell(
                run, config, common, population, include_nested=include_nested,
            )
            rows.extend(cell_rows)
            cells.append(cell)
    cell_by_key = {(cell.common_paths, cell.population_replicates): cell for cell in cells}
    margins: list[Mapping[str, object]] = []
    for population in PILOT_P_VALUES:
        for old, new in ((16, 32), (32, 64)):
            a, b = cell_by_key[(old, population)], cell_by_key[(new, population)]
            margins.append({
                "axis": "C", "population_replicates": population,
                "old": old, "new": new,
                "se_reduction_per_1000": 1000.0 * (a.se_max - b.se_max) / (b.calls - a.calls),
                "width_reduction_per_1000": 1000.0 * (a.width_max - b.width_max) / (b.calls - a.calls),
            })
    for common in PILOT_C_VALUES:
        for old, new in ((2, 4), (4, 8)):
            a, b = cell_by_key[(common, old)], cell_by_key[(common, new)]
            margins.append({
                "axis": "P", "common_paths": common,
                "old": old, "new": new,
                "se_reduction_per_1000": 1000.0 * (a.se_max - b.se_max) / (b.calls - a.calls),
                "width_reduction_per_1000": 1000.0 * (a.width_max - b.width_max) / (b.calls - a.calls),
            })
    eligible = [cell for cell in cells if cell.complete and cell.width_max <= config.pilot_width_target]
    recommended = min(eligible, key=lambda cell: cell.calls) if eligible else None
    return PilotAnalysis(
        rows=tuple(rows),
        cells=tuple(cells),
        marginal_efficiency=tuple(margins),
        disposition="recommended_configuration" if recommended else "precision_plan_infeasible",
        recommended_cell=(recommended.common_paths, recommended.population_replicates)
        if recommended else None,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pilot_manifest(
    config: PilotConfig,
    library: PilotLibrary,
    *,
    preflight: PilotPreflight | None = None,
    scheduler: PilotRun | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    source_paths = (
        "src/mfg_hedge/precision_pilot.py",
        "src/mfg_hedge/shared_backup.py",
        "src/mfg_hedge/game_workload.py",
        "src/mfg_hedge/expert_game.py",
        "src/mfg_hedge/campaign_execution.py",
        "src/mfg_hedge/game_deviations.py",
        "src/mfg_hedge/game_solver.py",
        "src/mfg_hedge/stage4_campaign.py",
    )
    source_files = [
        {"path": path, "sha256": _file_sha256(root / path)} for path in source_paths
    ]
    config_path = root / "configs/v1_minimal.json"
    protocol_paths = tuple(
        f"docs/adr/{number}-{name}"
        for number, name in (
            ("0015", "shared-backup-expert-game.md"),
            ("0016", "finite-256-rule-game-solver-protocol.md"),
            ("0017", "stage4-campaign-performance.md"),
            ("0018", "pooled-objective-cluster-inference.md"),
            ("0019", "stage4b-precision-pilot-protocol.md"),
        )
    ) + (".scratch/shared-backup-game/spec.md",)
    protocol_files = [
        {"path": path, "sha256": _file_sha256(root / path)} for path in protocol_paths
    ]
    from . import __version__
    manifest: dict[str, object] = {
        "schema_version": "stage4b-precision-pilot-manifest-v1",
        "package_version": __version__,
        "configuration": {
            "path": "configs/v1_minimal.json",
            "sha256": _file_sha256(config_path),
            "raw": json.loads(config_path.read_text(encoding="utf-8")),
        },
        "source": {"files": source_files, "aggregate_sha256": _canonical_sha256(source_files)},
        "protocol": {"files": protocol_files, "fingerprint": _canonical_sha256(protocol_files)},
        "pilot": config.as_dict(),
        "profiles": [list(row) for row in PILOT_PROFILES],
        "physical": {
            "N": 8, "c_B": 0.5, "degraded_slowdown": 2.0,
            "hedge_delay": 1.5, "tariff": 0.0, "arrival_cutoff": 360.0,
            "complete_drain": True, "canonical_rule_count": 256,
        },
        "inference_protocol": {
            "gain": "G=J0(NSNS,NSNS_-0)-J0(r,NSNS_-0)",
            "delete_one": "one common path and all P populations",
            "jackknife": "sqrt((C-1)/C * sum((G_minus_c-mean)^2))",
            "nested": {
                "V_between_raw": "Var_b(mu_b)",
                "V_within": "mean_b(s_b^2)",
                "V_between": "max(0,V_between_raw-V_within/B_within)",
                "V_total": "V_between+V_within",
            },
            "resampling_seed": config.resampling_seed,
            "B_outer": config.b_outer,
            "B_within": config.b_within,
            "epsilon_nash": config.epsilon_nash,
            "pilot_width_target": config.pilot_width_target,
        },
        "library": library.as_dict(),
        "preflight": preflight.as_dict() if preflight else None,
        "scheduler": scheduler.as_dict(include_tokens=False) if scheduler else None,
    }
    manifest["manifest_fingerprint"] = _canonical_sha256(manifest)
    return manifest


def write_pilot_artifacts(
    artifacts_root: str | Path,
    run_id: str,
    manifest: Mapping[str, object],
    pilot: Mapping[str, object],
    report: Mapping[str, object],
) -> Path:
    """Commit the four pilot files atomically into a fresh directory."""

    _validate_run_id(run_id)
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("pilot run directory escapes artifacts root")
    if run_dir.exists():
        raise FileExistsError(f"pilot run directory already exists: {run_dir}")
    staging = root / f".{run_id}.staging"
    if staging.exists():
        raise FileExistsError(f"pilot staging directory already exists: {staging}")
    staging.mkdir()
    try:
        payloads = {
            "manifest.json": manifest,
            "pilot.json": pilot,
            "report.json": report,
        }
        for name, payload in payloads.items():
            text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        report_text = "# Stage 4b precision pilot\n\n" + json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n"
        temporary = staging / "report.md.tmp"
        temporary.write_text(report_text, encoding="utf-8")
        os.replace(temporary, staging / "report.md")
        os.replace(staging, run_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_dir


class PilotDisposition:
    """Claim-boundary descriptors, not result-bearing mutable state."""

    class _Value:
        def __init__(self, value: str, claims_nash: bool, claims_mfg: bool):
            self.value = value
            self.claims_nash = claims_nash
            self.claims_mfg = claims_mfg

    PRECISION_PLAN_INFEASIBLE = _Value("precision_plan_infeasible", False, False)
    RECOMMENDED_CONFIGURATION = _Value("recommended_configuration", False, False)


@dataclass(frozen=True)
class PilotExecution:
    status: str
    disposition: str
    preflight: PilotPreflight
    scheduler: PilotRun | None
    analysis: PilotAnalysis | None
    artifact_run_dir: str | None
    failure_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "disposition": self.disposition,
            "preflight": self.preflight.as_dict(),
            "scheduler": self.scheduler.as_dict() if self.scheduler else None,
            "analysis": self.analysis.as_dict() if self.analysis else None,
            "artifact_run_dir": self.artifact_run_dir,
            "failure_reason": self.failure_reason,
            "claims_candidate": False,
            "claims_nash": False,
            "claims_mfg": False,
        }


def run_precision_pilot(
    *,
    artifacts_root: str | Path | None = None,
    run_id: str | None = None,
    max_workers: int = 8,
) -> PilotExecution:
    """Run preflight, then the frozen pilot only when exact work is feasible."""

    config = PilotConfig()
    library = build_pilot_library(config)
    preflight = run_pilot_preflight(library, config)
    if not preflight.feasible:
        return PilotExecution(
            status="preflight_blocked",
            disposition="precision_plan_infeasible",
            preflight=preflight,
            scheduler=None,
            analysis=None,
            artifact_run_dir=None,
            failure_reason=preflight.blocker,
        )
    scheduler = run_pilot_scheduler(library, max_workers=max_workers)
    if not scheduler.complete:
        return PilotExecution(
            status="execution_failed",
            disposition="precision_plan_infeasible",
            preflight=preflight,
            scheduler=scheduler,
            analysis=None,
            artifact_run_dir=None,
            failure_reason=scheduler.failure_reason,
        )
    analysis = analyze_pilot(scheduler, library, config, include_nested=True)
    artifact_dir = None
    if artifacts_root is not None:
        if run_id is None:
            raise ValueError("run_id is required when artifacts_root is provided")
        manifest = build_pilot_manifest(
            config, library, preflight=preflight, scheduler=scheduler,
        )
        pilot = {
            "config": config.as_dict(),
            "preflight": preflight.as_dict(),
            "scheduler": scheduler.as_dict(include_tokens=False),
            "disposition": analysis.disposition,
            "recommended_cell": list(analysis.recommended_cell)
            if analysis.recommended_cell else None,
            "claims_candidate": False, "claims_nash": False, "claims_mfg": False,
        }
        artifact_dir = str(write_pilot_artifacts(
            artifacts_root, run_id, manifest, pilot, analysis.as_dict(),
        ))
    return PilotExecution(
        status="completed",
        disposition=analysis.disposition,
        preflight=preflight,
        scheduler=scheduler,
        analysis=analysis,
        artifact_run_dir=artifact_dir,
        failure_reason="",
    )


__all__ = [
    "B_OUTER",
    "B_WITHIN",
    "PILOT_COMMON_PATHS",
    "PILOT_MACRO_SEED",
    "PILOT_NAMESPACE",
    "PILOT_POPULATION_REPLICATES",
    "PILOT_PROFILES",
    "PILOT_RESAMPLING_SEED",
    "PilotAnalysis",
    "PilotAttemptPayload",
    "PilotCellSummary",
    "PilotConfig",
    "PilotExecution",
    "PilotLibrary",
    "PilotPreflight",
    "PilotRuleRow",
    "PilotRun",
    "PilotScenario",
    "PilotScenarioResult",
    "PilotTokenPayload",
    "build_pilot_library",
    "build_pilot_manifest",
    "corrected_nested_variance",
    "jackknife_se",
    "pilot_call_count",
    "profile_names",
    "run_pilot_preflight",
    "run_pilot_scheduler",
    "run_precision_pilot",
    "score_pooled_payloads",
    "select_prefix_cell",
    "write_pilot_artifacts",
]
