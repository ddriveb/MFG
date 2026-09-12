"""Stage 3 objective accounting for the finite Shared-Backup Expert game.

The objects in this module are deliberately separate from the historical
experiment metrics.  A scored episode couples one immutable Stage 1 trace to
one immutable simulation result; the scorer then pools raw Token and attempt
rows across episodes.  It never averages episode scalars and never fills a
missing cohort with zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import struct
from types import MappingProxyType
from typing import Iterable, Mapping

from .common_state import CommonState, CommonStateTimeline, Phase, phase_at
from .domain import TokenClass
from .shared_backup import (
    AttemptRecord,
    AttemptStatus,
    SharedBackupTrace,
    SharedSimulationResult,
    SharedTokenResult,
    TaggedSharedSimulationResult,
)
from .transient_objective import cvar95_fractional_tail


_PHASES = tuple(phase.value for phase in Phase)
_CLASSES = tuple(token_class.value for token_class in TokenClass)
_CLASS_WEIGHTS = {TokenClass.REGULAR.value: 0.8, TokenClass.URGENT.value: 0.2}
_DEADLINES = {TokenClass.REGULAR.value: 3.0, TokenClass.URGENT.value: 2.0}
_SLO_COEFFICIENTS = {TokenClass.REGULAR.value: 1.0, TokenClass.URGENT.value: 5.0}
_REPLAY_COEFFICIENTS = {TokenClass.REGULAR.value: 1.0, TokenClass.URGENT.value: 5.0}


def _finite(value: object, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _freeze(mapping: Mapping[object, object]) -> MappingProxyType:
    return MappingProxyType(dict(mapping))


def _freeze_nested(mapping: Mapping[object, Mapping[object, object]]) -> MappingProxyType:
    return MappingProxyType({key: _freeze(value) for key, value in mapping.items()})


def shared_trace_fingerprint(trace: SharedBackupTrace) -> str:
    """Return a deterministic digest of every exogenous Stage 1 input row."""

    if not isinstance(trace, SharedBackupTrace):
        raise ValueError("trace must be SharedBackupTrace")
    digest = hashlib.sha256()
    digest.update(struct.pack(
        ">qdddd", trace.expert_count, trace.timeline.degraded_start,
        trace.timeline.failed_start, trace.timeline.recovered_start,
        trace.arrival_cutoff,
    ))
    for token, draw in zip(trace.tokens, trace.work):
        digest.update(struct.pack(
            ">qqdq", token.global_token_id, token.expert_id,
            token.arrival_time, token.local_token_id,
        ))
        digest.update(token.token_class.value.encode("ascii"))
        digest.update(struct.pack(">q", token.destination_replica))
        digest.update(struct.pack(
            ">12d", *(value for attempt in (
                draw.attempt0, draw.attempt1, draw.attempt2, draw.attempt3
            ) for value in attempt)
        ))
    return digest.hexdigest()


def fault_timeline_fingerprint(timeline: CommonStateTimeline, cutoff: float) -> str:
    """Fingerprint a common fault path without adding population identity."""

    if not isinstance(timeline, CommonStateTimeline):
        raise ValueError("timeline must be CommonStateTimeline")
    cutoff_value = _finite(cutoff, "cutoff")
    digest = hashlib.sha256()
    digest.update(struct.pack(
        ">4d", timeline.degraded_start, timeline.failed_start,
        timeline.recovered_start, cutoff_value,
    ))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExpertEpisodeRun:
    """Immutable trace/result pair accepted by the Stage 3 scorer."""

    episode_key: str
    trace: SharedBackupTrace
    result: SharedSimulationResult
    fault_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be a non-empty string")
        if not isinstance(self.trace, SharedBackupTrace):
            raise ValueError("trace must be SharedBackupTrace")
        if not isinstance(self.result, SharedSimulationResult):
            raise ValueError("result must be SharedSimulationResult")
        if self.result.completed_tokens != len(self.trace.tokens):
            raise ValueError("trace/result Token counts do not match")
        for expected, actual in zip(self.trace.tokens, self.result.tokens):
            if (
                expected.global_token_id != actual.global_token_id
                or expected.expert_id != actual.expert_id
                or expected.local_token_id != actual.local_token_id
                or expected.token_class is not actual.token_class
                or expected.arrival_time != actual.arrival_time
                or actual.primary_replica != (
                    expected.destination_replica
                    if phase_at(self.trace.timeline, expected.arrival_time) is Phase.FAILED
                    else 0
                )
            ):
                raise ValueError("trace/result Token identity mismatch")
        expected_by_key = set()
        for attempt in self.result.attempts:
            if not isinstance(attempt, AttemptRecord):
                raise ValueError("result contains a non-AttemptRecord row")
            if not 0 <= attempt.global_token_id < len(self.trace.tokens):
                raise ValueError("attempt global Token ID is outside the trace")
            token = self.trace.tokens[attempt.global_token_id]
            if (
                attempt.expert_id != token.expert_id
                or attempt.local_token_id != token.local_token_id
                or attempt.key in expected_by_key
            ):
                raise ValueError("attempt ownership or identity is invalid")
            expected_by_key.add(attempt.key)
            draw = self.trace.work[attempt.global_token_id]
            expected_work = getattr(draw, f"attempt{attempt.attempt_id}")[attempt.replica_id]
            if attempt.required_work != expected_work:
                raise ValueError("attempt required work does not match immutable trace")
            _finite(attempt.executed_work, "attempt.executed_work")
            _finite(attempt.remaining_work, "attempt.remaining_work")
            if attempt.status in (
                AttemptStatus.CANCELLED_QUEUED,
                AttemptStatus.INVALIDATED_QUEUED,
            ) and attempt.executed_work != 0.0:
                raise ValueError("queued cancellation cannot contain executed work")
        for token in self.result.tokens:
            if token.replay_count not in (0, 1):
                raise ValueError("Replay count must be at most one")
            if phase_at(self.trace.timeline, token.arrival_time) is Phase.FAILED:
                if any(
                    attempt.replica_id == 0 and attempt.executed_work != 0.0
                    for attempt in token.attempts
                ):
                    raise ValueError("F-arrival A-domain executed work must be zero")
        fingerprint = self.fault_fingerprint
        if fingerprint is None:
            fingerprint = fault_timeline_fingerprint(
                self.trace.timeline, self.trace.arrival_cutoff
            )
        if type(fingerprint) is not str or not fingerprint:
            raise ValueError("fault_fingerprint must be a non-empty string")
        object.__setattr__(self, "fault_fingerprint", fingerprint)


@dataclass(frozen=True)
class TaggedExpertEpisodeRun:
    """Trace plus an exact tagged output accepted by the Expert scorer."""

    episode_key: str
    trace: SharedBackupTrace
    result: TaggedSharedSimulationResult
    fault_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be a non-empty string")
        if not isinstance(self.trace, SharedBackupTrace):
            raise ValueError("trace must be SharedBackupTrace")
        if not isinstance(self.result, TaggedSharedSimulationResult):
            raise ValueError("result must be TaggedSharedSimulationResult")
        expert_id = self.result.tagged_expert_id
        if not 0 <= expert_id < self.trace.expert_count:
            raise ValueError("tagged Expert is outside the trace population")
        expected_tokens = tuple(
            token for token in self.trace.tokens if token.expert_id == expert_id
        )
        if tuple(token.global_token_id for token in self.result.tokens) != tuple(
            token.global_token_id for token in expected_tokens
        ):
            raise ValueError("tagged result Token ordering or identity mismatch")
        for expected, actual in zip(expected_tokens, self.result.tokens):
            if (
                expected.local_token_id != actual.local_token_id
                or expected.token_class is not actual.token_class
                or expected.arrival_time != actual.arrival_time
                or actual.primary_replica != (
                    expected.destination_replica
                    if phase_at(self.trace.timeline, expected.arrival_time) is Phase.FAILED
                    else 0
                )
            ):
                raise ValueError("tagged result Token identity mismatch")
            if actual.replay_count not in (0, 1):
                raise ValueError("Replay count must be at most one")
            for attempt in actual.attempts:
                if attempt.expert_id != expert_id:
                    raise ValueError("tagged attempt ownership mismatch")
                draw = self.trace.work[attempt.global_token_id]
                expected_work = getattr(draw, f"attempt{attempt.attempt_id}")[attempt.replica_id]
                if attempt.required_work != expected_work:
                    raise ValueError("tagged attempt work does not match trace")
                if attempt.status in (
                    AttemptStatus.CANCELLED_QUEUED,
                    AttemptStatus.INVALIDATED_QUEUED,
                ) and attempt.executed_work != 0.0:
                    raise ValueError("queued cancellation cannot contain executed work")
            if phase_at(self.trace.timeline, actual.arrival_time) is Phase.FAILED:
                if any(
                    attempt.replica_id == 0 and attempt.executed_work != 0.0
                    for attempt in actual.attempts
                ):
                    raise ValueError("F-arrival A-domain executed work must be zero")
        expected_fingerprint = shared_trace_fingerprint(self.trace)
        if self.result.trace_fingerprint != expected_fingerprint:
            raise ValueError("tagged trace fingerprint mismatch")
        fingerprint = self.fault_fingerprint
        if fingerprint is None:
            fingerprint = fault_timeline_fingerprint(
                self.trace.timeline, self.trace.arrival_cutoff
            )
        if type(fingerprint) is not str or not fingerprint:
            raise ValueError("fault_fingerprint must be a non-empty string")
        object.__setattr__(self, "fault_fingerprint", fingerprint)


@dataclass(frozen=True)
class CohortMetrics:
    """Pooled raw statistics for one arrival-phase/Token-class cohort."""

    phase: str
    token_class: str
    count: int
    total_latency: float
    total_excess: float
    miss_count: int
    replay_count: int
    mean_latency: float
    mean_excess: float
    miss_rate: float
    replay_rate: float
    cvar95: float

    @property
    def slo_excess(self) -> float:
        return self.mean_excess

    @property
    def weight(self) -> float:
        return _CLASS_WEIGHTS[self.token_class]

    @property
    def deadline(self) -> float:
        return _DEADLINES[self.token_class]

    @property
    def excess_penalty(self) -> float:
        return _SLO_COEFFICIENTS[self.token_class]

    @property
    def replay_penalty(self) -> float:
        return _REPLAY_COEFFICIENTS[self.token_class]

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "token_class": self.token_class,
            "weight": self.weight,
            "deadline": self.deadline,
            "excess_penalty": self.excess_penalty,
            "replay_penalty": self.replay_penalty,
            "count": self.count,
            "total_latency": self.total_latency,
            "total_excess": self.total_excess,
            "miss_count": self.miss_count,
            "replay_count": self.replay_count,
            "mean_latency": self.mean_latency,
            "mean_excess": self.mean_excess,
            "miss_rate": self.miss_rate,
            "replay_rate": self.replay_rate,
            "cvar95": self.cvar95,
        }


@dataclass(frozen=True)
class ExpertScore:
    """Complete individual objective and its auditable components."""

    expert_id: int | None
    episode_count: int
    generated_token_count: int
    cohorts: Mapping[str, Mapping[str, CohortMetrics]]
    phase_losses: Mapping[str, float | None]
    cvar95: Mapping[str, float | None]
    class_cvar95: Mapping[str, Mapping[str, float | None]]
    raw_work_totals: Mapping[str, float]
    components: Mapping[str, float | None]
    missing_cohorts: tuple[tuple[str, str], ...]
    complete: bool
    total: float | None

    @property
    def objective(self) -> float | None:
        return self.total

    @property
    def status(self) -> str:
        return "complete" if self.complete else "incomplete"

    @property
    def is_complete(self) -> bool:
        return self.complete

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohorts", _freeze_nested(self.cohorts))
        object.__setattr__(self, "phase_losses", _freeze(self.phase_losses))
        object.__setattr__(self, "cvar95", _freeze(self.cvar95))
        object.__setattr__(self, "class_cvar95", _freeze_nested(self.class_cvar95))
        object.__setattr__(self, "raw_work_totals", _freeze(self.raw_work_totals))
        object.__setattr__(self, "components", _freeze(self.components))
        if self.complete != (not self.missing_cohorts):
            raise ValueError("complete flag disagrees with missing cohorts")
        if self.complete and self.total is None:
            raise ValueError("complete Expert score requires a total")
        if not self.complete and self.total is not None:
            raise ValueError("incomplete Expert score must have null total")

    def as_dict(self) -> dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "episode_count": self.episode_count,
            "generated_token_count": self.generated_token_count,
            "phase_class_counts": {
                phase: {token_class: cohort.count
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "cohorts": {
                phase: {token_class: cohort.as_dict()
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "mean_latency": {
                phase: {token_class: cohort.mean_latency
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "mean_excess": {
                phase: {token_class: cohort.mean_excess
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "miss_rate": {
                phase: {token_class: cohort.miss_rate
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "replay_rate": {
                phase: {token_class: cohort.replay_rate
                        for token_class, cohort in cells.items()}
                for phase, cells in self.cohorts.items()
            },
            "cvar95": dict(self.cvar95),
            "class_cvar95": {
                phase: dict(classes) for phase, classes in self.class_cvar95.items()
            },
            "phase_losses": dict(self.phase_losses),
            "raw_work_totals": dict(self.raw_work_totals),
            "components": dict(self.components),
            "missing_cohorts": list(self.missing_cohorts),
            "complete": self.complete,
            "total": self.total,
        }


@dataclass(frozen=True)
class PopulationScore:
    """Individual scores plus equal-weight social and pooled diagnostics."""

    expert_scores: tuple[ExpertScore, ...]
    social_cost_mean_expert_objective: float | None
    pooled_across_experts_objective: float | None
    pooled_score: ExpertScore

    def __post_init__(self) -> None:
        if not self.expert_scores:
            raise ValueError("PopulationScore requires at least one Expert")

    @property
    def pooled_cvar95(self) -> Mapping[str, float | None]:
        return self.pooled_score.cvar95

    @property
    def social_cost(self) -> float | None:
        return self.social_cost_mean_expert_objective

    @property
    def complete(self) -> bool:
        return all(score.complete for score in self.expert_scores)

    def as_dict(self) -> dict[str, object]:
        return {
            "experts": [score.as_dict() for score in self.expert_scores],
            "social_cost_mean_expert_objective": self.social_cost_mean_expert_objective,
            "pooled_across_experts_objective": self.pooled_across_experts_objective,
            "pooled": self.pooled_score.as_dict(),
            "complete": self.complete,
        }


EpisodeRun = ExpertEpisodeRun | TaggedExpertEpisodeRun


def _validate_runs(runs: Iterable[EpisodeRun]) -> tuple[EpisodeRun, ...]:
    materialized = tuple(runs)
    if not materialized or any(
        not isinstance(run, (ExpertEpisodeRun, TaggedExpertEpisodeRun))
        for run in materialized
    ):
        raise ValueError("scoring requires nonempty episode run rows")
    keys = tuple(run.episode_key for run in materialized)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode_key in score input")
    counts = {run.trace.expert_count for run in materialized}
    if len(counts) != 1:
        raise ValueError("episode Expert counts do not match")
    return tuple(sorted(materialized, key=lambda run: run.episode_key))


def _score_scope(runs: tuple[EpisodeRun, ...], expert_id: int | None) -> ExpertScore:
    expert_count = runs[0].trace.expert_count
    if expert_id is not None and (type(expert_id) is not int or not 0 <= expert_id < expert_count):
        raise ValueError("expert_id is outside the trace population")

    rows: dict[tuple[str, str], list[SharedTokenResult]] = {
        (phase, token_class): []
        for phase in _PHASES for token_class in _CLASSES
    }
    selected_count = 0
    attempts: list[AttemptRecord] = []
    for run in runs:
        for token in run.result.tokens:
            if expert_id is not None and token.expert_id != expert_id:
                continue
            phase = phase_at(run.trace.timeline, token.arrival_time).value
            rows[(phase, token.token_class.value)].append(token)
            selected_count += 1
        attempts.extend(
            attempt for attempt in run.result.attempts
            if expert_id is None or attempt.expert_id == expert_id
        )

    cohorts: dict[str, dict[str, CohortMetrics]] = {phase: {} for phase in _PHASES}
    missing: list[tuple[str, str]] = []
    phase_losses: dict[str, float | None] = {}
    class_cvar95: dict[str, dict[str, float | None]] = {
        phase: {token_class: None for token_class in _CLASSES}
        for phase in _PHASES
    }
    cvar95: dict[str, float | None] = {}
    for phase in _PHASES:
        for token_class in _CLASSES:
            values = rows[(phase, token_class)]
            if not values:
                missing.append((phase, token_class))
                continue
            deadline = _DEADLINES[token_class]
            coefficient = _SLO_COEFFICIENTS[token_class]
            latencies = tuple(_finite(token.latency, "token.latency") for token in values)
            excesses = tuple(max(0.0, latency - deadline) for latency in latencies)
            misses = sum(latency > deadline for latency in latencies)
            replays = sum(token.replay_count > 0 for token in values)
            cohort = CohortMetrics(
                phase=phase,
                token_class=token_class,
                count=len(values),
                total_latency=math.fsum(latencies),
                total_excess=math.fsum(excesses),
                miss_count=misses,
                replay_count=replays,
                mean_latency=math.fsum(latencies) / len(values),
                mean_excess=math.fsum(excesses) / len(values),
                miss_rate=misses / len(values),
                replay_rate=replays / len(values),
                cvar95=cvar95_fractional_tail(latencies),
            )
            cohorts[phase][token_class] = cohort
            class_cvar95[phase][token_class] = cohort.cvar95
        if all(token_class in cohorts[phase] for token_class in _CLASSES):
            phase_losses[phase] = math.fsum(
                _CLASS_WEIGHTS[token_class] * (
                    cohorts[phase][token_class].mean_latency
                    + _SLO_COEFFICIENTS[token_class]
                    * cohorts[phase][token_class].mean_excess
                    + _REPLAY_COEFFICIENTS[token_class]
                    * cohorts[phase][token_class].replay_rate
                )
                for token_class in _CLASSES
            )
        else:
            phase_losses[phase] = None
        if rows[(phase, TokenClass.REGULAR.value)] or rows[(phase, TokenClass.URGENT.value)]:
            cvar95[phase] = cvar95_fractional_tail(
                token.latency for token_class in _CLASSES
                for token in rows[(phase, token_class)]
            )
        else:
            cvar95[phase] = None

    executed = math.fsum(_finite(a.executed_work, "attempt.executed_work") for a in attempts)
    wasted = math.fsum(
        _finite(a.executed_work, "attempt.executed_work")
        for a in attempts if a.status is not AttemptStatus.COMPLETED_WINNER
    )
    raw_work = {
        "executed": executed,
        "wasted": wasted,
        "attempt_count": float(len(attempts)),
        "winner_executed": math.fsum(
            a.executed_work for a in attempts
            if a.status is AttemptStatus.COMPLETED_WINNER
        ),
        "failed_executed": math.fsum(
            a.executed_work for a in attempts
            if a.status is AttemptStatus.FAILED_RUNNING
        ),
        "replay_executions": float(sum(a.attempt_id == 1 for a in attempts)),
        "hedge_executions": float(sum(a.attempt_id >= 2 for a in attempts)),
    }
    complete = not missing
    phase_component = (
        math.fsum(phase_losses[phase] for phase in _PHASES) / 4.0
        if complete else None
    )
    fault_tail = (
        math.fsum(cvar95[phase] for phase in (Phase.DEGRADED.value, Phase.FAILED.value)) / 2.0
        if complete else None
    )
    work_per_token = executed / selected_count if selected_count else None
    waste_per_token = wasted / selected_count if selected_count else None
    components: dict[str, float | None] = {
        "phase_loss": phase_component,
        "fault_tail": fault_tail,
        "executed_work_per_token": work_per_token,
        "wasted_work_per_token": waste_per_token,
        "work_per_token": work_per_token,
        "waste_per_token": waste_per_token,
    }
    total = (
        math.fsum((phase_component, fault_tail, work_per_token, waste_per_token))
        if complete else None
    )
    return ExpertScore(
        expert_id=expert_id,
        episode_count=len(runs),
        generated_token_count=selected_count,
        cohorts=cohorts,
        phase_losses=phase_losses,
        cvar95=cvar95,
        class_cvar95=class_cvar95,
        raw_work_totals=raw_work,
        components=components,
        missing_cohorts=tuple(missing),
        complete=complete,
        total=total,
    )


def score_expert(runs: Iterable[EpisodeRun], expert_id: int) -> ExpertScore:
    """Pool one Expert's raw rows across every supplied episode."""

    materialized = _validate_runs(runs)
    return _score_scope(materialized, expert_id)


def score_population(runs: Iterable[EpisodeRun]) -> PopulationScore:
    """Return individual scores, equal-weight social cost, and pooled diagnostic."""

    materialized = _validate_runs(runs)
    if any(isinstance(run, TaggedExpertEpisodeRun) for run in materialized):
        raise ValueError("population scoring requires full episode outputs")
    expert_count = materialized[0].trace.expert_count
    individual = tuple(_score_scope(materialized, expert_id) for expert_id in range(expert_count))
    pooled = _score_scope(materialized, None)
    social = (
        math.fsum(score.total for score in individual) / expert_count
        if all(score.complete for score in individual) else None
    )
    pooled_total = pooled.total if pooled.complete else None
    return PopulationScore(individual, social, pooled_total, pooled)


def score_tagged_result(
    result: TaggedSharedSimulationResult,
    *,
    trace: SharedBackupTrace,
    expert_id: int,
    episode_key: str,
    fault_fingerprint: str | None = None,
) -> ExpertScore:
    """Score a tagged output with the exact Stage 3 Expert formula."""

    run = TaggedExpertEpisodeRun(
        episode_key=episode_key,
        trace=trace,
        result=result,
        fault_fingerprint=fault_fingerprint,
    )
    if expert_id != result.tagged_expert_id:
        raise ValueError("expert_id does not match tagged result")
    return score_expert((run,), expert_id)


score_experts = score_population
build_expert_objective = score_expert
build_population_objective = score_population
build_expert_score = score_expert
build_population_score = score_population


__all__ = [
    "CohortMetrics",
    "ExpertEpisodeRun",
    "TaggedExpertEpisodeRun",
    "ExpertScore",
    "PopulationScore",
    "fault_timeline_fingerprint",
    "build_expert_objective",
    "build_population_objective",
    "build_expert_score",
    "build_population_score",
    "score_expert",
    "score_experts",
    "score_population",
    "score_tagged_result",
    "shared_trace_fingerprint",
]
