"""Isolated deterministic Shared-Backup Expert game physics.

This module is the first implementation slice of ADR-0015.  It deliberately
does not widen the historical engines: all work is keyed by the new trace
contract and all B/C heads are scheduled by one finite equal-sharing pool.
There is no random sampling, solver, deviation evaluator, CLI, or artifact
writer here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum, Enum
import heapq
import itertools
import math
from typing import Mapping, Protocol

from .common_state import CommonStateTimeline, Phase, phase_at, state_at
from .domain import CommonState, TokenClass


_REPLICA_COUNT = 3
_BACKUP_REPLICAS = (1, 2)
_WINDOW_WIDTH = Decimal("25")
_WINDOW_CAP = Decimal("2.8125")


def _int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {relation}, got {value!r}")
    return result


class SharedAction(str, Enum):
    """The isolated four-action alphabet N/D/S/X."""

    NORMAL = "N"
    DELAYED = "D"
    SINGLE = "S"
    DUAL = "X"
    DELAYED_SINGLE = "D"
    IMMEDIATE_SINGLE = "S"
    IMMEDIATE_DUAL = "X"


class AttemptStatus(str, Enum):
    """Dynamic and terminal copy lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED_WINNER = "completed_winner"
    COMPLETED_LOSER = "completed_loser"
    CANCELLED_QUEUED = "cancelled_queued"
    FAILED_RUNNING = "failed_running"
    INVALIDATED_QUEUED = "invalidated_queued"


@dataclass(frozen=True)
class AttemptKey:
    """Stable identity for one Expert/Token/Replica/attempt copy."""

    expert_id: int
    global_token_id: int
    replica_id: int
    attempt_id: int

    def __post_init__(self) -> None:
        _int(self.expert_id, "expert_id")
        _int(self.global_token_id, "global_token_id")
        _int(self.replica_id, "replica_id")
        _int(self.attempt_id, "attempt_id")
        if self.replica_id >= _REPLICA_COUNT:
            raise ValueError(f"replica_id must be 0, 1, or 2, got {self.replica_id!r}")
        if self.attempt_id > 3:
            raise ValueError(f"attempt_id must be in 0..3, got {self.attempt_id!r}")

    @property
    def token_id(self) -> int:
        """Short compatibility spelling for the global Token identity."""

        return self.global_token_id

    def label(self) -> str:
        return (
            f"expert={self.expert_id}/token={self.global_token_id}/"
            f"replica={self.replica_id}/attempt={self.attempt_id}"
        )


@dataclass(frozen=True)
class SharedTokenSpec:
    """Immutable global/local Token identity and destination mark."""

    global_token_id: int
    arrival_time: float
    token_class: TokenClass
    expert_id: int
    local_token_id: int
    destination_replica: int

    def __post_init__(self) -> None:
        _int(self.global_token_id, "global_token_id")
        object.__setattr__(self, "arrival_time", _real(self.arrival_time, "arrival_time"))
        if not isinstance(self.token_class, TokenClass):
            raise ValueError(f"token_class must be TokenClass, got {self.token_class!r}")
        _int(self.expert_id, "expert_id")
        _int(self.local_token_id, "local_token_id")
        _int(self.destination_replica, "destination_replica")
        if self.destination_replica not in _BACKUP_REPLICAS:
            raise ValueError(
                "destination_replica must be the B/C mark 1 or 2, "
                f"got {self.destination_replica!r}"
            )

    @property
    def token_id(self) -> int:
        return self.global_token_id


@dataclass(frozen=True)
class SharedWorkDraw:
    """Attempt-0/1/2/3 requirements for replicas A/B/C."""

    attempt0: tuple[float, float, float]
    attempt1: tuple[float, float, float]
    attempt2: tuple[float, float, float]
    attempt3: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("attempt0", "attempt1", "attempt2", "attempt3"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != _REPLICA_COUNT:
                raise ValueError(f"{name} must contain exactly three Replica draws")
            object.__setattr__(
                self,
                name,
                tuple(
                    _real(value, f"{name}[{replica}]", positive=True)
                    for replica, value in enumerate(values)
                ),
            )


@dataclass(frozen=True)
class SharedBackupTrace:
    """Immutable deterministic input for one finite Expert population."""

    expert_count: int
    timeline: CommonStateTimeline
    arrival_cutoff: float
    tokens: tuple[SharedTokenSpec, ...]
    work: tuple[SharedWorkDraw, ...]

    def __post_init__(self) -> None:
        _int(self.expert_count, "expert_count", 1)
        if not isinstance(self.timeline, CommonStateTimeline):
            raise ValueError(f"timeline must be CommonStateTimeline, got {self.timeline!r}")
        cutoff = _real(self.arrival_cutoff, "arrival_cutoff", positive=True)
        object.__setattr__(self, "arrival_cutoff", cutoff)
        if not self.tokens or len(self.tokens) != len(self.work):
            raise ValueError("trace requires nonempty one-to-one Token/work rows")
        previous_arrival = -1.0
        expected_local = [0] * self.expert_count
        for expected_global, token in enumerate(self.tokens):
            if not isinstance(token, SharedTokenSpec):
                raise ValueError("every trace Token must be SharedTokenSpec")
            if token.global_token_id != expected_global:
                raise ValueError("global Token IDs must be dense and ordered")
            if token.expert_id >= self.expert_count:
                raise ValueError(
                    f"Token {token.global_token_id} has Expert {token.expert_id} "
                    f"outside 0..{self.expert_count - 1}"
                )
            if token.local_token_id != expected_local[token.expert_id]:
                raise ValueError(
                    f"Expert {token.expert_id} local Token IDs must be dense and ordered"
                )
            if token.arrival_time < previous_arrival:
                raise ValueError("trace arrivals must be non-decreasing")
            if token.arrival_time >= cutoff:
                raise ValueError("trace arrivals must be before arrival_cutoff")
            expected_local[token.expert_id] += 1
            previous_arrival = token.arrival_time
        if any(not isinstance(draw, SharedWorkDraw) for draw in self.work):
            raise ValueError("every trace work row must be SharedWorkDraw")


@dataclass(frozen=True)
class PreparedTrace:
    """Validated, immutable exogenous projection reusable across reruns.

    This object intentionally contains no action, budget, queue, completion,
    timer, winner, or scorer state.  Each simulation creates a new endogenous
    engine even when the same projection is reused.
    """

    trace: SharedBackupTrace
    token_indices_by_expert: tuple[tuple[int, ...], ...]
    arrival_events: tuple[tuple[float, int], ...]
    token_classes: tuple[TokenClass, ...]
    destinations: tuple[int, ...]
    attempt_work: tuple[tuple[tuple[float, ...], ...], ...]
    trace_fingerprint: str
    fault_fingerprint: str
    key_schema_version: str
    expected_token_count: int
    expected_expert_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace, SharedBackupTrace):
            raise ValueError("PreparedTrace.trace must be SharedBackupTrace")
        if self.key_schema_version != "shared-backup-game:key-v2":
            raise ValueError("unsupported PreparedTrace key schema")
        _int(self.expected_token_count, "expected_token_count")
        _int(self.expected_expert_count, "expected_expert_count", 1)
        if self.expected_token_count != len(self.trace.tokens):
            raise ValueError("PreparedTrace token count mismatch")
        if self.expected_expert_count != self.trace.expert_count:
            raise ValueError("PreparedTrace Expert count mismatch")
        if len(self.token_indices_by_expert) != self.expected_expert_count:
            raise ValueError("PreparedTrace local index count mismatch")
        if len(self.arrival_events) != self.expected_token_count:
            raise ValueError("PreparedTrace arrival projection mismatch")
        if len(self.token_classes) != self.expected_token_count:
            raise ValueError("PreparedTrace class projection mismatch")
        if len(self.destinations) != self.expected_token_count:
            raise ValueError("PreparedTrace destination projection mismatch")
        if len(self.attempt_work) != self.expected_token_count:
            raise ValueError("PreparedTrace work projection mismatch")
        flattened = []
        for expert, indices in enumerate(self.token_indices_by_expert):
            if not isinstance(indices, tuple):
                raise ValueError("PreparedTrace local indices must be tuples")
            expected_local = []
            for index in indices:
                _int(index, "token index")
                if index >= self.expected_token_count:
                    raise ValueError("PreparedTrace token index is outside trace")
                token = self.trace.tokens[index]
                if token.expert_id != expert or token.local_token_id != len(expected_local):
                    raise ValueError("PreparedTrace local index ordering changed")
                expected_local.append(index)
            flattened.extend(indices)
        if tuple(sorted(flattened)) != tuple(range(self.expected_token_count)):
            raise ValueError("PreparedTrace local indices are not a partition")
        if self.arrival_events != tuple(
            (token.arrival_time, token.global_token_id) for token in self.trace.tokens
        ):
            raise ValueError("PreparedTrace arrival ordering changed")
        if self.token_classes != tuple(token.token_class for token in self.trace.tokens):
            raise ValueError("PreparedTrace token classes changed")
        if self.destinations != tuple(token.destination_replica for token in self.trace.tokens):
            raise ValueError("PreparedTrace destinations changed")
        expected_work = tuple(
            tuple(
                getattr(self.trace.work[index], f"attempt{attempt}")
                for attempt in (0, 1, 2, 3)
            )
            for index in range(self.expected_token_count)
        )
        if self.attempt_work != expected_work:
            raise ValueError("PreparedTrace attempt work changed")
        for name in ("trace_fingerprint", "fault_fingerprint"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"PreparedTrace {name} must be non-empty")
        from .expert_game import fault_timeline_fingerprint, shared_trace_fingerprint
        if self.trace_fingerprint != shared_trace_fingerprint(self.trace):
            raise ValueError("PreparedTrace trace fingerprint is not canonical")
        if self.fault_fingerprint != fault_timeline_fingerprint(
            self.trace.timeline, self.trace.arrival_cutoff
        ):
            raise ValueError("PreparedTrace fault fingerprint is not canonical")


def prepare_shared_trace(raw_trace: SharedBackupTrace | PreparedTrace) -> PreparedTrace:
    """Validate and project an exogenous trace exactly once."""

    if isinstance(raw_trace, PreparedTrace):
        return raw_trace
    if not isinstance(raw_trace, SharedBackupTrace):
        raise ValueError(f"trace must be SharedBackupTrace, got {raw_trace!r}")
    # Runtime imports avoid a module cycle: expert_game uses the shared trace
    # types, while preparation only needs its pure fingerprint functions.
    from .expert_game import fault_timeline_fingerprint, shared_trace_fingerprint

    local = tuple(
        tuple(token.global_token_id for token in raw_trace.tokens if token.expert_id == expert)
        for expert in range(raw_trace.expert_count)
    )
    arrival_events = tuple(
        (token.arrival_time, token.global_token_id) for token in raw_trace.tokens
    )
    attempt_work = tuple(
        tuple(
            tuple(getattr(draw, f"attempt{attempt}") for attempt in (0, 1, 2, 3))
        )
        for draw in raw_trace.work
    )
    return PreparedTrace(
        trace=raw_trace,
        token_indices_by_expert=local,
        arrival_events=arrival_events,
        token_classes=tuple(token.token_class for token in raw_trace.tokens),
        destinations=tuple(token.destination_replica for token in raw_trace.tokens),
        attempt_work=attempt_work,
        trace_fingerprint=shared_trace_fingerprint(raw_trace),
        fault_fingerprint=fault_timeline_fingerprint(
            raw_trace.timeline, raw_trace.arrival_cutoff
        ),
        key_schema_version="shared-backup-game:key-v2",
        expected_token_count=len(raw_trace.tokens),
        expected_expert_count=raw_trace.expert_count,
    )


@dataclass(frozen=True)
class BudgetObservation:
    window_start: float | None
    window_end: float | None
    cap: float
    balance: float


@dataclass(frozen=True)
class QueueObservation:
    """Observable local queue state; no required or remaining work is exposed."""

    replica_id: int
    queued_attempts: tuple[AttemptKey, ...]
    running_attempt: AttemptKey | None
    executed_work: float


@dataclass(frozen=True)
class AttemptObservation:
    key: AttemptKey
    status: AttemptStatus
    executed_work: float


@dataclass(frozen=True)
class ActionObservation:
    """Causal arrival observation supplied to one Expert's ActionSource."""

    current_time: float
    common_state: CommonState
    phase: Phase
    phase_age: float
    token_class: TokenClass
    expert_id: int
    local_token_id: int
    queues: tuple[QueueObservation, ...]
    executed_progress: tuple[tuple[AttemptKey, float], ...]
    budget: BudgetObservation
    public_active_heads: int
    public_expert_count: int
    winner_history: tuple[tuple[int, int], ...]
    timer_history: tuple[tuple[int, str], ...]
    failure_history: tuple[tuple[int, str], ...]
    primary_replica: int = 0
    k_n: float = 0.0
    public_fault: object | None = None

    @property
    def k_N(self) -> float:
        return self.k_n

    @property
    def observed_phase_start(self) -> float:
        if self.public_fault is None:
            return self.current_time - self.phase_age
        return self.public_fault.observed_phase_start

    @property
    def observed_transitions(self) -> tuple[tuple[str, float], ...]:
        if self.public_fault is None:
            return ()
        return self.public_fault.transitions


class ActionSource(Protocol):
    """Minimal online policy contract used at the arrival event only."""

    def request(self, observation: ActionObservation) -> SharedAction:
        ...


@dataclass(frozen=True)
class ActionDecision:
    global_token_id: int
    expert_id: int
    local_token_id: int
    token_class: TokenClass
    arrival_time: float
    eligible: bool
    requested: SharedAction
    applied: SharedAction
    window_start: float | None
    window_end: float | None
    cap: float
    balance_before: float
    charge: float
    balance_after: float
    budget_suppressed: bool


@dataclass(frozen=True)
class AttemptRecord:
    global_token_id: int
    expert_id: int
    local_token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    start_time: float | None
    terminal_time: float
    required_work: float
    executed_work: float
    remaining_work: float
    status: AttemptStatus
    queue_delay: float | None

    @property
    def key(self) -> AttemptKey:
        return AttemptKey(
            self.expert_id, self.global_token_id, self.replica_id, self.attempt_id
        )

    @property
    def token_id(self) -> int:
        return self.global_token_id


@dataclass(frozen=True)
class PoolIntervalAudit:
    start_time: float
    end_time: float
    active_heads: int
    per_head_speed: float
    executed_work: float
    capacity_bound: float
    contributing_attempt_keys: tuple[str, ...]

    @property
    def per_head_speeds(self) -> tuple[tuple[str, float], ...]:
        """Explicit per-head view; equal sharing gives every key this speed."""

        return tuple((key, self.per_head_speed)
                     for key in self.contributing_attempt_keys)


@dataclass(frozen=True)
class QueueAuditEvent:
    time: float
    expert_id: int
    replica_id: int
    remaining_start: float
    enqueued_work: float
    executed_work: float
    cancelled_work: float
    discarded_work: float
    remaining_end: float
    conservation_holds: bool

    @property
    def enqueued(self) -> float:
        return self.enqueued_work

    @property
    def executed(self) -> float:
        return self.executed_work

    @property
    def cancelled(self) -> float:
        return self.cancelled_work

    @property
    def discarded(self) -> float:
        return self.discarded_work


@dataclass(frozen=True)
class QueueAudit:
    expert_id: int
    replica_id: int
    enqueued_work: float
    executed_work: float
    cancelled_work: float
    discarded_work: float
    remaining_work: float

    @property
    def enqueued(self) -> float:
        return self.enqueued_work

    @property
    def executed(self) -> float:
        return self.executed_work

    @property
    def cancelled(self) -> float:
        return self.cancelled_work

    @property
    def discarded(self) -> float:
        return self.discarded_work


@dataclass(frozen=True)
class SharedInvariantCounters:
    token_arrivals: int
    primary_enqueues: int
    replay_enqueues: int
    hedge_requested: int
    hedge_applied: int
    hedge_launches: int
    hedge_suppressed: int
    budget_suppressed: int
    hedge_timers_scheduled: int
    hedge_timers_fired: int
    hedge_timers_voided: int
    completion_winners: int
    completed_losers: int
    cancelled_queued: int
    failed_running_attempts: int
    invalidated_queued_attempts: int
    stale_completion_events_ignored: int


@dataclass(frozen=True)
class SharedTokenResult:
    global_token_id: int
    expert_id: int
    local_token_id: int
    token_class: TokenClass
    arrival_time: float
    primary_replica: int
    requested_action: SharedAction
    action: SharedAction
    attempts: tuple[AttemptRecord, ...]
    winner: AttemptKey
    winner_attempt_id: int
    completion_time: float
    completion_phase: Phase
    replay_count: int
    timer_scheduled: bool
    timer_fired: bool
    timer_voided: bool
    latency: float

    @property
    def token_id(self) -> int:
        return self.global_token_id


@dataclass(frozen=True)
class SharedSimulationResult:
    """Complete immutable state, lifecycle, pool and conservation audit."""

    tokens: tuple[SharedTokenResult, ...]
    attempts: tuple[AttemptRecord, ...]
    action_decisions: tuple[ActionDecision, ...]
    pool_intervals: tuple[PoolIntervalAudit, ...]
    queue_audits: tuple[QueueAudit, ...]
    queue_audit_events: tuple[QueueAuditEvent, ...]
    counters: SharedInvariantCounters
    last_arrival_time: float
    drain_end_time: float
    trace_fingerprint: str = ""
    fault_fingerprint: str = ""

    @property
    def completed_tokens(self) -> int:
        return len(self.tokens)

    @property
    def invariant_counters(self) -> SharedInvariantCounters:
        return self.counters

    @property
    def drain_duration(self) -> float:
        return self.drain_end_time - self.last_arrival_time

    @property
    def primary_executions(self) -> int:
        return self.counters.primary_enqueues

    @property
    def replay_executions(self) -> int:
        return self.counters.replay_enqueues

    @property
    def hedge_requested(self) -> int:
        return self.counters.hedge_requested

    @property
    def hedge_launches(self) -> int:
        return self.counters.hedge_launches

    @property
    def stale_completion_events_ignored(self) -> int:
        return self.counters.stale_completion_events_ignored

    @property
    def total_executed_work(self) -> float:
        return math.fsum(a.executed_work for a in self.attempts)

    @property
    def wasted_work(self) -> float:
        return math.fsum(
            a.executed_work
            for a in self.attempts
            if a.status is not AttemptStatus.COMPLETED_WINNER
        )

    def __post_init__(self) -> None:
        if not self.tokens or len(self.tokens) != self.counters.token_arrivals:
            raise RuntimeError("result Token/counter invariant failed")
        if tuple(t.global_token_id for t in self.tokens) != tuple(range(len(self.tokens))):
            raise RuntimeError("result Token IDs are not dense")
        expected_attempts = tuple(a for t in self.tokens for a in t.attempts)
        if self.attempts != expected_attempts:
            raise RuntimeError("flattened attempt rows do not match Token rows")
        if len(self.action_decisions) != len(self.tokens):
            raise RuntimeError("one action decision is required per arrival")
        if self.counters.hedge_requested != sum(
            decision.eligible and decision.requested is not SharedAction.NORMAL
            for decision in self.action_decisions
        ):
            raise RuntimeError("requested action counter mismatch")
        if self.counters.hedge_applied != sum(
            decision.applied is not SharedAction.NORMAL
            for decision in self.action_decisions
        ):
            raise RuntimeError("applied action counter mismatch")
        if self.counters.primary_enqueues != len(self.tokens):
            raise RuntimeError("Primary enqueue counter mismatch")
        if self.counters.replay_enqueues != sum(
            attempt.attempt_id == 1 for attempt in self.attempts
        ):
            raise RuntimeError("Replay enqueue counter mismatch")
        if self.counters.hedge_launches != sum(
            attempt.attempt_id >= 2 for attempt in self.attempts
        ):
            raise RuntimeError("Hedge launch counter mismatch")
        if self.counters.hedge_timers_scheduled != sum(
            token.timer_scheduled for token in self.tokens
        ):
            raise RuntimeError("timer schedule counter mismatch")
        if self.counters.hedge_timers_fired != sum(
            token.timer_fired for token in self.tokens
        ):
            raise RuntimeError("timer fire counter mismatch")
        if self.counters.completion_winners != len(self.tokens):
            raise RuntimeError("winner counter mismatch")
        if self.counters.completed_losers != sum(
            attempt.status is AttemptStatus.COMPLETED_LOSER
            for attempt in self.attempts
        ):
            raise RuntimeError("completed loser counter mismatch")
        if self.counters.cancelled_queued != sum(
            attempt.status is AttemptStatus.CANCELLED_QUEUED
            for attempt in self.attempts
        ):
            raise RuntimeError("queued cancellation counter mismatch")
        for token in self.tokens:
            winners = [
                a for a in token.attempts
                if a.status is AttemptStatus.COMPLETED_WINNER
            ]
            if len(winners) != 1 or winners[0].key != token.winner:
                raise RuntimeError("each Token must have exactly one winner")
            if token.winner_attempt_id != token.winner.attempt_id:
                raise RuntimeError("winner attempt id mismatch")
            if token.replay_count not in (0, 1):
                raise RuntimeError("Replay count must be 0 or 1")
            if token.completion_time != winners[0].terminal_time:
                raise RuntimeError("winner completion time mismatch")
            if token.latency != token.completion_time - token.arrival_time:
                raise RuntimeError("Token latency mismatch")
            if sum(a.attempt_id == 1 for a in token.attempts) != token.replay_count:
                raise RuntimeError("Replay lifecycle mismatch")
            if len({a.attempt_id for a in token.attempts}) != len(token.attempts):
                raise RuntimeError("duplicate attempt id for Token")
            for attempt in token.attempts:
                if (
                    not math.isfinite(attempt.required_work)
                    or not math.isfinite(attempt.executed_work)
                    or attempt.required_work <= 0.0
                    or attempt.executed_work < 0.0
                    or attempt.executed_work > attempt.required_work + 1e-9
                ):
                    raise RuntimeError("attempt work invariant failed")
                if attempt.status is AttemptStatus.CANCELLED_QUEUED:
                    if attempt.executed_work != 0.0 or attempt.start_time is not None:
                        raise RuntimeError("queued cancellation must have zero work")
                if attempt.status in (
                    AttemptStatus.COMPLETED_WINNER,
                    AttemptStatus.COMPLETED_LOSER,
                ) and not math.isclose(
                    attempt.executed_work, attempt.required_work,
                    rel_tol=1e-9, abs_tol=1e-9,
                ):
                    raise RuntimeError("completed attempt must have full work")
        if not math.isclose(
            self.drain_end_time,
            max(a.terminal_time for a in self.attempts),
            rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise RuntimeError("drain end does not include every terminal attempt")


@dataclass(frozen=True)
class PoolAuditSummary:
    """Compact global processor-sharing audit retained by tagged runs."""

    interval_count: int
    total_executed_work: float
    total_capacity_bound: float
    max_active_heads: int
    constrained_interval_count: int

    def __post_init__(self) -> None:
        _int(self.interval_count, "interval_count")
        _int(self.max_active_heads, "max_active_heads")
        _int(self.constrained_interval_count, "constrained_interval_count")
        for name in ("total_executed_work", "total_capacity_bound"):
            value = _real(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.constrained_interval_count > self.interval_count:
            raise ValueError("constrained interval count exceeds interval count")


@dataclass(frozen=True)
class TaggedSharedSimulationResult:
    """Exact physical result with only one Expert's materialized rows.

    The counters and pool summary remain global.  Other Experts participate in
    the event loop and online policy calls but their per-Token rows are omitted
    from this immutable output.
    """

    tagged_expert_id: int
    tokens: tuple[SharedTokenResult, ...]
    attempts: tuple[AttemptRecord, ...]
    action_decisions: tuple[ActionDecision, ...]
    queue_audits: tuple[QueueAudit, ...]
    queue_audit_events: tuple[QueueAuditEvent, ...]
    pool_audit_summary: PoolAuditSummary
    counters: SharedInvariantCounters
    last_arrival_time: float
    drain_end_time: float
    trace_fingerprint: str
    fault_fingerprint: str

    @property
    def completed_tokens(self) -> int:
        return len(self.tokens)

    @property
    def invariant_counters(self) -> SharedInvariantCounters:
        return self.counters

    @property
    def total_executed_work(self) -> float:
        return math.fsum(attempt.executed_work for attempt in self.attempts)

    @property
    def wasted_work(self) -> float:
        return math.fsum(
            attempt.executed_work for attempt in self.attempts
            if attempt.status is not AttemptStatus.COMPLETED_WINNER
        )

    def __post_init__(self) -> None:
        _int(self.tagged_expert_id, "tagged_expert_id")
        if not isinstance(self.pool_audit_summary, PoolAuditSummary):
            raise ValueError("pool_audit_summary must be PoolAuditSummary")
        if not isinstance(self.counters, SharedInvariantCounters):
            raise ValueError("counters must be SharedInvariantCounters")
        if not self.tokens:
            raise RuntimeError("tagged result requires at least one Token")
        if any(token.expert_id != self.tagged_expert_id for token in self.tokens):
            raise RuntimeError("tagged Token result contains another Expert")
        if any(attempt.expert_id != self.tagged_expert_id for attempt in self.attempts):
            raise RuntimeError("tagged attempt result contains another Expert")
        if any(decision.expert_id != self.tagged_expert_id
               for decision in self.action_decisions):
            raise RuntimeError("tagged action result contains another Expert")
        if self.attempts != tuple(a for token in self.tokens for a in token.attempts):
            raise RuntimeError("tagged flattened attempts do not match Token rows")
        token_ids = tuple(token.global_token_id for token in self.tokens)
        if token_ids != tuple(sorted(set(token_ids))):
            raise RuntimeError("tagged Token IDs must be unique and canonical")
        if tuple(decision.global_token_id for decision in self.action_decisions) != token_ids:
            raise RuntimeError("tagged action rows do not match Token rows")
        if self.action_decisions != tuple(sorted(
            self.action_decisions, key=lambda decision: decision.global_token_id
        )):
            raise RuntimeError("tagged action decisions are not canonical")
        if len(self.action_decisions) != len(self.tokens):
            raise RuntimeError("tagged output requires one action per tagged Token")
        if any(not isinstance(value, str) or not value
               for value in (self.trace_fingerprint, self.fault_fingerprint)):
            raise ValueError("tagged fingerprints must be non-empty strings")
        if (
            not math.isfinite(self.last_arrival_time)
            or not math.isfinite(self.drain_end_time)
            or self.drain_end_time < self.last_arrival_time
        ):
            raise ValueError("tagged times must be finite")


class _EventType(IntEnum):
    STATE = 1
    COMPLETE = 2
    TIMER = 3
    ARRIVAL = 4


@dataclass(frozen=True)
class _Event:
    event_time: float
    event_type: _EventType
    sequence: int
    global_token_id: int | None = None
    expert_id: int | None = None
    replica_id: int | None = None
    attempt_id: int | None = None
    generation: int = 0


@dataclass
class _Queued:
    key: AttemptKey
    enqueue_time: float
    required_work: float
    cancelled: bool = False


@dataclass
class _Running:
    key: AttemptKey
    enqueue_time: float
    start_time: float
    required_work: float
    executed_work: float
    last_update_time: float
    current_speed: float


@dataclass
class _TokenState:
    spec: SharedTokenSpec
    requested_action: SharedAction = SharedAction.NORMAL
    action: SharedAction = SharedAction.NORMAL
    primary_replica: int | None = None
    replay_count: int = 0
    timer_pending: bool = False
    timer_scheduled: bool = False
    timer_fired: bool = False
    timer_voided: bool = False
    winner: AttemptKey | None = None
    completion_time: float | None = None
    live: dict[int, _Queued | _Running] = field(default_factory=dict)
    records: list[AttemptRecord] = field(default_factory=list)


@dataclass
class _Budget:
    window_start: Decimal | None = None
    window_end: Decimal | None = None
    balance: Decimal = Decimal("0")


class _Engine:
    def __init__(
        self,
        trace: SharedBackupTrace,
        c_b: float,
        slowdown: float,
        hedge_delay: float,
        action_sources: Mapping[int, object] | object | None,
        *,
        prepared: PreparedTrace | None = None,
        optimized: bool = False,
        evaluation_mode: str = "full",
        tagged_expert_id: int | None = None,
    ) -> None:
        self.trace = trace
        self.prepared = prepared
        self.optimized = optimized
        if evaluation_mode not in ("full", "tagged"):
            raise ValueError("evaluation_mode must be 'full' or 'tagged'")
        if evaluation_mode == "tagged":
            _int(tagged_expert_id, "tagged_expert_id")
            if tagged_expert_id >= trace.expert_count:
                raise ValueError("tagged_expert_id is outside the trace population")
        elif tagged_expert_id is not None:
            raise ValueError("tagged_expert_id requires tagged evaluation mode")
        self.evaluation_mode = evaluation_mode
        self.tagged_expert_id = tagged_expert_id
        self.expert_count = trace.expert_count
        self.c_b = c_b
        self.slowdown = slowdown
        self.hedge_delay = hedge_delay
        self.queues = [
            [deque() for _ in range(_REPLICA_COUNT)]
            for _ in range(self.expert_count)
        ]
        self.running: list[list[_Running | None]] = [
            [None] * _REPLICA_COUNT for _ in range(self.expert_count)
        ]
        self.generations = [
            [0] * _REPLICA_COUNT for _ in range(self.expert_count)
        ]
        self.events: list[tuple[float, int, int, _Event]] = []
        self.sequence = itertools.count()
        self.tokens = [_TokenState(spec) for spec in trace.tokens]
        self.sources = self._validate_sources(action_sources)
        self.budgets = [_Budget() for _ in range(self.expert_count)]
        self.action_decisions: list[ActionDecision] = []
        self.pool_intervals: list[PoolIntervalAudit] = []
        self.queue_audit_events: list[QueueAuditEvent] = []
        self.queue_totals = {
            (expert, replica): {
                "enqueued": 0.0, "executed": 0.0,
                "cancelled": 0.0, "discarded": 0.0,
            }
            for expert in range(self.expert_count)
            for replica in range(_REPLICA_COUNT)
        }
        self.last_remaining = {
            (expert, replica): 0.0
            for expert in range(self.expert_count)
            for replica in range(_REPLICA_COUNT)
        }
        self.remaining_work = dict(self.last_remaining)
        self.last_audit_totals = {
            key: {name: 0.0 for name in ("enqueued", "executed", "cancelled", "discarded")}
            for key in self.queue_totals
        }
        self.last_time = 0.0
        self.last_active_keys: tuple[AttemptKey, ...] = ()
        self.active_backup_heads = 0
        self.pool_interval_count = 0
        self.pool_total_executed = 0.0
        self.pool_total_capacity = 0.0
        self.pool_max_active_heads = 0
        self.pool_constrained_intervals = 0
        self.drain_end_time = 0.0
        self.stale_ignored = 0
        self.counters = {
            name: 0 for name in (
                "token_arrivals", "primary_enqueues", "replay_enqueues",
                "hedge_requested", "hedge_applied", "hedge_launches",
                "hedge_suppressed", "budget_suppressed",
                "hedge_timers_scheduled", "hedge_timers_fired",
                "hedge_timers_voided", "completion_winners", "completed_losers",
                "cancelled_queued", "failed_running_attempts",
                "invalidated_queued_attempts",
            )
        }

    # -- validation and event plumbing ---------------------------------

    def _validate_sources(self, supplied: Mapping[int, object] | object | None):
        if supplied is None:
            supplied = {}
        if isinstance(supplied, Mapping):
            source_map = dict(supplied)
            for expert in source_map:
                if type(expert) is not int or expert < 0 or expert >= self.expert_count:
                    raise ValueError(f"unknown Expert id in action_sources: {expert!r}")
            sources = [source_map.get(expert, _NormalSource())
                       for expert in range(self.expert_count)]
        else:
            if isinstance(supplied, (list, tuple, set)):
                raise ValueError("action_sources must be online source objects, not an action list")
            sources = [supplied] * self.expert_count
        for source in sources:
            if not callable(getattr(source, "request", source)):
                raise ValueError("each ActionSource must be callable or expose request()")
        return tuple(sources)

    def _push(self, event: _Event) -> None:
        heapq.heappush(
            self.events,
            (event.event_time, int(event.event_type), event.sequence, event),
        )

    def _schedule_initial_events(self) -> None:
        for boundary in (
            self.trace.timeline.degraded_start,
            self.trace.timeline.failed_start,
            self.trace.timeline.recovered_start,
        ):
            self._push(_Event(boundary, _EventType.STATE, next(self.sequence)))
        arrivals = (
            self.prepared.arrival_events
            if self.prepared is not None
            else tuple((token.arrival_time, token.global_token_id)
                       for token in self.trace.tokens)
        )
        for arrival_time, global_token_id in arrivals:
            self._push(_Event(
                arrival_time, _EventType.ARRIVAL, next(self.sequence),
                global_token_id=global_token_id,
            ))

    def run(self) -> None:
        self._schedule_initial_events()
        while self.events:
            first = heapq.heappop(self.events)
            batch = [first]
            while self.events and self.events[0][0] == first[0]:
                batch.append(heapq.heappop(self.events))
            now = first[0]
            self._advance_to(now)
            state_events = [row[3] for row in batch if row[3].event_type is _EventType.STATE]
            completion_events = [row[3] for row in batch if row[3].event_type is _EventType.COMPLETE]
            timer_events = [row[3] for row in batch if row[3].event_type is _EventType.TIMER]
            arrival_events = [row[3] for row in batch if row[3].event_type is _EventType.ARRIVAL]

            for event in state_events:
                self._handle_state_change(event)
            completion_events.sort(key=lambda e: (
                e.replica_id if e.replica_id is not None else -1,
                e.attempt_id if e.attempt_id is not None else -1,
                e.global_token_id if e.global_token_id is not None else -1,
                e.sequence,
            ))
            for event in completion_events:
                self._handle_complete(event)
            timer_events.sort(key=lambda e: e.sequence)
            for event in timer_events:
                self._handle_timer(event)
            arrival_events.sort(key=lambda e: e.global_token_id)
            for event in arrival_events:
                self._handle_arrival(event)
            self._dispatch_pass(now)
            self._reschedule_pool(now)
            self._audit_queues(now)

        for token in self.tokens:
            if token.winner is None or token.completion_time is None:
                raise RuntimeError(
                    f"event loop drained without a winner for Token {token.spec.global_token_id}"
                )
        if any(self.running[expert][replica] is not None
               for expert in range(self.expert_count)
               for replica in range(_REPLICA_COUNT)):
            raise RuntimeError("event loop ended with a running attempt")
        if self.evaluation_mode == "tagged":
            self._verify_terminal_queue_conservation()

    def _advance_to(self, now: float) -> None:
        if now < self.last_time:
            raise RuntimeError("event time moved backwards")
        dt = now - self.last_time
        active = tuple(
            running.key
            for expert in range(self.expert_count)
            for replica in _BACKUP_REPLICAS
            if (running := self.running[expert][replica]) is not None
        )
        if dt > 0.0:
            speed = self.running[active[0].expert_id][active[0].replica_id].current_speed if active else 0.0
            deltas: list[float] = []
            for expert in range(self.expert_count):
                for replica in range(_REPLICA_COUNT):
                    running = self.running[expert][replica]
                    if running is None:
                        continue
                    old = running.executed_work
                    amount = min(
                        running.required_work - old,
                        max(0.0, dt * running.current_speed),
                    )
                    running.executed_work = old + amount
                    running.last_update_time = now
                    self.queue_totals[(expert, replica)]["executed"] += amount
                    self._ledger_subtract((expert, replica), amount)
                    if replica in _BACKUP_REPLICAS:
                        deltas.append(amount)
            capacity_bound = self.expert_count * self.c_b * dt
            executed_work = math.fsum(deltas)
            interval = PoolIntervalAudit(
                start_time=self.last_time,
                end_time=now,
                active_heads=len(active),
                per_head_speed=speed,
                executed_work=executed_work,
                capacity_bound=capacity_bound,
                contributing_attempt_keys=tuple(key.label() for key in active),
            )
            self.pool_interval_count += 1
            self.pool_total_executed += executed_work
            self.pool_total_capacity += capacity_bound
            self.pool_max_active_heads = max(self.pool_max_active_heads, len(active))
            if len(active) > self.expert_count * self.c_b:
                self.pool_constrained_intervals += 1
            if self.evaluation_mode == "full":
                self.pool_intervals.append(interval)
        self.last_time = now

    def _handle_state_change(self, event: _Event) -> None:
        now = event.event_time
        timeline = self.trace.timeline
        if now == timeline.degraded_start:
            for expert in range(self.expert_count):
                running = self.running[expert][0]
                if running is None:
                    continue
                running.current_speed = 1.0 / self.slowdown
                if running.required_work - running.executed_work > 1e-12:
                    self.generations[expert][0] += 1
                    self._schedule_completion(running, now)
        elif now == timeline.failed_start:
            self._enter_failed(now)
        elif now == timeline.recovered_start:
            # F has already removed every A head.  New post-recovery arrivals
            # dispatch at A speed 1; no old attempt is restored.
            return

    def _enter_failed(self, now: float) -> None:
        lost_primaries: list[int] = []
        for expert in range(self.expert_count):
            running = self.running[expert][0]
            if running is not None:
                self.generations[expert][0] += 1
                self.running[expert][0] = None
                token = self.tokens[running.key.global_token_id]
                token.live.pop(running.key.attempt_id, None)
                self._record(running, AttemptStatus.FAILED_RUNNING, now)
                discarded = max(0.0, running.required_work - running.executed_work)
                self.queue_totals[(expert, 0)]["discarded"] += discarded
                self._ledger_subtract((expert, 0), discarded)
                self.counters["failed_running_attempts"] += 1
                if running.key.attempt_id == 0:
                    lost_primaries.append(running.key.global_token_id)
            queued_rows = list(self.queues[expert][0])
            self.queues[expert][0].clear()
            for queued in queued_rows:
                if queued.cancelled:
                    continue
                token = self.tokens[queued.key.global_token_id]
                token.live.pop(queued.key.attempt_id, None)
                self._record(queued, AttemptStatus.INVALIDATED_QUEUED, now)
                self.queue_totals[(expert, 0)]["discarded"] += queued.required_work
                self._ledger_subtract((expert, 0), queued.required_work)
                self.counters["invalidated_queued_attempts"] += 1
                if queued.key.attempt_id == 0:
                    lost_primaries.append(queued.key.global_token_id)

        for token_id in sorted(lost_primaries):
            token = self.tokens[token_id]
            if token.winner is not None or token.replay_count or token.live:
                continue
            token.replay_count = 1
            if token.timer_pending:
                token.timer_pending = False
                token.timer_voided = True
            replica = token.spec.destination_replica
            self._enqueue(token_id, 1, replica, now)
            self.counters["replay_enqueues"] += 1

    # -- completion, timer, arrival ------------------------------------

    def _handle_complete(self, event: _Event) -> None:
        expert = event.expert_id
        replica = event.replica_id
        if expert is None or replica is None:
            return
        if event.generation != self.generations[expert][replica]:
            self.stale_ignored += 1
            return
        running = self.running[expert][replica]
        if running is None or running.key != AttemptKey(
            expert, event.global_token_id, replica, event.attempt_id
        ):
            return
        self.running[expert][replica] = None
        if replica in _BACKUP_REPLICAS and self.optimized:
            self.active_backup_heads -= 1
            if self.active_backup_heads < 0:
                raise RuntimeError("optimized active B/C head counter underflow")
        token = self.tokens[running.key.global_token_id]
        token.live.pop(running.key.attempt_id, None)
        now = event.event_time
        if token.winner is not None:
            self._record(running, AttemptStatus.COMPLETED_LOSER, now)
            self.counters["completed_losers"] += 1
            return
        token.winner = running.key
        token.completion_time = now
        if token.timer_pending:
            token.timer_voided = True
        token.timer_pending = False
        self._record(running, AttemptStatus.COMPLETED_WINNER, now)
        self.counters["completion_winners"] += 1
        for attempt_id, other in list(token.live.items()):
            if isinstance(other, _Queued) and not other.cancelled:
                other.cancelled = True
                self._record(other, AttemptStatus.CANCELLED_QUEUED, now)
                self.queue_totals[(other.key.expert_id, other.key.replica_id)]["cancelled"] += other.required_work
                self._ledger_subtract(
                    (other.key.expert_id, other.key.replica_id), other.required_work
                )
                self.counters["cancelled_queued"] += 1
                token.live.pop(attempt_id, None)

    def _handle_timer(self, event: _Event) -> None:
        token = self.tokens[event.global_token_id]
        if token.winner is not None or not token.timer_pending:
            token.timer_voided = True
            self.counters["hedge_timers_voided"] += 1
            return
        token.timer_pending = False
        token.timer_fired = True
        self.counters["hedge_timers_fired"] += 1
        if state_at(self.trace.timeline, event.event_time) is CommonState.FAILED:
            token.timer_voided = True
            self.counters["hedge_suppressed"] += 1
            return
        self._launch_hedge(event.global_token_id, event.event_time, (token.spec.destination_replica,))

    def _handle_arrival(self, event: _Event) -> None:
        token = self.tokens[event.global_token_id]
        spec = token.spec
        now = event.event_time
        state = state_at(self.trace.timeline, now)
        primary = 0 if state is not CommonState.FAILED else spec.destination_replica
        token.primary_replica = primary
        eligible = state is CommonState.DEGRADED and primary == 0
        raw_action = self._request_for_arrival(
            self.sources[spec.expert_id], spec, now, state, primary
        )
        requested = raw_action if eligible else SharedAction.NORMAL
        decision = self._project_action(token, requested, eligible)
        token.requested_action = decision.requested
        token.action = decision.applied
        if self.evaluation_mode == "full" or spec.expert_id == self.tagged_expert_id:
            self.action_decisions.append(decision)
        self.counters["token_arrivals"] += 1
        self._enqueue(spec.global_token_id, 0, primary, now)
        self.counters["primary_enqueues"] += 1
        if token.action is SharedAction.SINGLE:
            self._launch_hedge(spec.global_token_id, now, (spec.destination_replica,))
        elif token.action is SharedAction.DUAL:
            # Attempt 2 is the Single/default Backup copy.  Dual adds the
            # other Backup as attempt 3, preserving the declared CRN binding
            # when the default destination is C (replica 2).
            other = next(
                replica for replica in _BACKUP_REPLICAS
                if replica != spec.destination_replica
            )
            self._launch_hedge(
                spec.global_token_id, now,
                (spec.destination_replica, other),
            )
        elif token.action is SharedAction.DELAYED:
            token.timer_pending = True
            token.timer_scheduled = True
            self.counters["hedge_timers_scheduled"] += 1
            self._push(_Event(
                now + self.hedge_delay, _EventType.TIMER, next(self.sequence),
                global_token_id=spec.global_token_id,
            ))

    def _request(self, source: object, observation: ActionObservation) -> SharedAction:
        method = getattr(source, "request", source)
        action = method(observation)
        if not isinstance(action, SharedAction):
            raise ValueError(f"ActionSource must return SharedAction, got {action!r}")
        return action

    def _request_for_arrival(
        self,
        source: object,
        spec: SharedTokenSpec,
        now: float,
        state: CommonState,
        primary_replica: int,
    ) -> SharedAction:
        """Use Pi256's exact causal projection or the general observation API."""

        compact = getattr(source, "request_compact", None)
        if callable(compact):
            phase = phase_at(self.trace.timeline, now)
            if phase is Phase.HEALTHY:
                phase_start = 0.0
            elif phase is Phase.DEGRADED:
                phase_start = self.trace.timeline.degraded_start
            elif phase is Phase.FAILED:
                phase_start = self.trace.timeline.failed_start
            else:
                phase_start = self.trace.timeline.recovered_start
            action = compact(
                common_state=state,
                phase=phase,
                phase_age=now - phase_start,
                token_class=spec.token_class,
                primary_replica=primary_replica,
            )
            if action is NotImplemented:
                return self._request(source, self._observation(spec, now))
            if not isinstance(action, SharedAction):
                raise ValueError(f"ActionSource must return SharedAction, got {action!r}")
            return action
        return self._request(source, self._observation(spec, now))

    # -- action admission and observations ------------------------------

    def _budget_view(self, expert: int, now: float) -> BudgetObservation:
        budget = self.budgets[expert]
        d_start = Decimal(str(self.trace.timeline.degraded_start))
        moment = Decimal(str(now))
        if moment < d_start:
            return BudgetObservation(None, None, 0.0, 0.0)
        index = int((moment - d_start) // _WINDOW_WIDTH)
        window_start = d_start + index * _WINDOW_WIDTH
        window_end = window_start + _WINDOW_WIDTH
        if budget.window_start != window_start:
            budget.window_start = window_start
            budget.window_end = window_end
            budget.balance = _WINDOW_CAP
        return BudgetObservation(
            float(budget.window_start), float(budget.window_end),
            float(_WINDOW_CAP), float(budget.balance),
        )

    def _project_action(
        self, token: _TokenState, requested: SharedAction, eligible: bool
    ) -> ActionDecision:
        spec = token.spec
        now = spec.arrival_time
        view = self._budget_view(spec.expert_id, now)
        before = Decimal(str(view.balance))
        charge_dec = {
            SharedAction.NORMAL: Decimal("0"),
            SharedAction.DELAYED: Decimal("1"),
            SharedAction.SINGLE: Decimal("1"),
            SharedAction.DUAL: Decimal("2"),
        }[requested]
        applied = requested
        suppressed = False
        if eligible and requested is not SharedAction.NORMAL:
            self.counters["hedge_requested"] += 1
            if before < charge_dec:
                applied = SharedAction.NORMAL
                suppressed = True
                self.counters["budget_suppressed"] += 1
            else:
                self.budgets[spec.expert_id].balance -= charge_dec
                self.counters["hedge_applied"] += 1
        after = self.budgets[spec.expert_id].balance if view.window_start is not None else before
        if not eligible:
            applied = SharedAction.NORMAL
            charge_dec = Decimal("0")
            after = before
        if eligible and requested is not SharedAction.NORMAL and applied is SharedAction.NORMAL:
            after = before
        return ActionDecision(
            global_token_id=spec.global_token_id,
            expert_id=spec.expert_id,
            local_token_id=spec.local_token_id,
            token_class=spec.token_class,
            arrival_time=now,
            eligible=eligible,
            requested=requested,
            applied=applied,
            window_start=view.window_start,
            window_end=view.window_end,
            cap=view.cap,
            balance_before=float(before),
            charge=float(charge_dec if applied is not SharedAction.NORMAL else Decimal("0")),
            balance_after=float(after),
            budget_suppressed=suppressed,
        )

    def _observation(self, spec: SharedTokenSpec, now: float) -> ActionObservation:
        # Import lazily so the Stage 1 scheduler remains usable as the lower
        # layer of the isolated Stage 2 workload module without an import
        # cycle.  Only the returned projection crosses the policy boundary.
        from .game_workload import public_fault_observation_from_timeline

        state = state_at(self.trace.timeline, now)
        phase = phase_at(self.trace.timeline, now)
        phase_start = {
            Phase.HEALTHY: 0.0,
            Phase.DEGRADED: self.trace.timeline.degraded_start,
            Phase.FAILED: self.trace.timeline.failed_start,
            Phase.RECOVERED: self.trace.timeline.recovered_start,
        }[phase]
        queues = []
        progress = []
        history: list[AttemptObservation] = []
        winners: list[tuple[int, int]] = []
        timers: list[tuple[int, str]] = []
        failures: list[tuple[int, str]] = []
        token_indices = (
            self.prepared.token_indices_by_expert[spec.expert_id]
            if self.prepared is not None
            else tuple(
                token.spec.global_token_id for token in self.tokens
                if token.spec.expert_id == spec.expert_id
            )
        )
        for token_index in token_indices:
            token = self.tokens[token_index]
            if token.winner is not None:
                winners.append((token.spec.global_token_id, token.winner.attempt_id))
            if token.timer_scheduled:
                timer_status = (
                    "voided" if token.timer_voided
                    else "fired" if token.timer_fired
                    else "pending"
                )
                timers.append((token.spec.global_token_id, timer_status))
            for record in token.records:
                if record.status in (AttemptStatus.FAILED_RUNNING,
                                     AttemptStatus.INVALIDATED_QUEUED):
                    failures.append((token.spec.global_token_id, record.status.value))
                history.append(AttemptObservation(record.key, record.status,
                                                  record.executed_work))
        for replica in range(_REPLICA_COUNT):
            queued = tuple(item.key for item in self.queues[spec.expert_id][replica]
                           if not item.cancelled)
            running = self.running[spec.expert_id][replica]
            running_key = running.key if running is not None else None
            executed = self.queue_totals[(spec.expert_id, replica)]["executed"]
            queues.append(QueueObservation(replica, queued, running_key, executed))
            if running is not None:
                progress.append((running.key, running.executed_work))
        active = (
            self.active_backup_heads if self.optimized else sum(
                self.running[expert][replica] is not None
                for expert in range(self.expert_count)
                for replica in _BACKUP_REPLICAS
            )
        )
        budget = self._budget_view(spec.expert_id, now)
        primary = 0 if state is not CommonState.FAILED else spec.destination_replica
        public_fault = public_fault_observation_from_timeline(
            # The helper projects only history through ``now``; the complete
            # timeline remains private to this engine.
            self.trace.timeline,
            now,
            active / self.expert_count,
        )
        return ActionObservation(
            current_time=now,
            common_state=state,
            phase=phase,
            phase_age=now - phase_start,
            token_class=spec.token_class,
            expert_id=spec.expert_id,
            local_token_id=spec.local_token_id,
            queues=tuple(queues),
            executed_progress=tuple(progress),
            budget=budget,
            public_active_heads=active,
            public_expert_count=self.expert_count,
            winner_history=tuple(winners),
            timer_history=tuple(timers),
            failure_history=tuple(failures),
            primary_replica=primary,
            k_n=active / self.expert_count,
            public_fault=public_fault,
        )

    # -- queue dispatch and pool rescheduling ---------------------------

    def _work(self, token_id: int, attempt_id: int, replica: int) -> float:
        if self.prepared is not None:
            return self.prepared.attempt_work[token_id][attempt_id][replica]
        return getattr(self.trace.work[token_id], f"attempt{attempt_id}")[replica]

    def _enqueue(self, token_id: int, attempt_id: int, replica: int, now: float) -> None:
        token = self.tokens[token_id]
        key = AttemptKey(token.spec.expert_id, token_id, replica, attempt_id)
        queued = _Queued(key, now, self._work(token_id, attempt_id, replica))
        self.queues[key.expert_id][replica].append(queued)
        token.live[attempt_id] = queued
        self.queue_totals[(key.expert_id, replica)]["enqueued"] += queued.required_work
        self.remaining_work[(key.expert_id, replica)] += queued.required_work

    def _launch_hedge(
        self, token_id: int, now: float, replicas: tuple[int, ...]
    ) -> None:
        token = self.tokens[token_id]
        if token.winner is not None:
            return
        if any(attempt_id >= 2 for attempt_id in token.live):
            return
        self.counters["hedge_launches"] += len(replicas)
        for index, replica in enumerate(replicas):
            self._enqueue(token_id, 2 + index, replica, now)

    def _dispatch_pass(self, now: float) -> None:
        failed = state_at(self.trace.timeline, now) is CommonState.FAILED
        for expert in range(self.expert_count):
            for replica in range(_REPLICA_COUNT):
                if replica == 0 and failed:
                    continue
                if self.running[expert][replica] is not None:
                    continue
                while self.queues[expert][replica]:
                    queued = self.queues[expert][replica].popleft()
                    if queued.cancelled:
                        continue
                    key = queued.key
                    speed = 1.0 if replica in _BACKUP_REPLICAS else (
                        1.0 / self.slowdown if state_at(self.trace.timeline, now) is CommonState.DEGRADED else 1.0
                    )
                    running = _Running(
                        key, queued.enqueue_time, now, queued.required_work,
                        0.0, now, speed,
                    )
                    self.running[expert][replica] = running
                    self.tokens[key.global_token_id].live[key.attempt_id] = running
                    if replica in _BACKUP_REPLICAS and self.optimized:
                        self.active_backup_heads += 1
                    if key.attempt_id == 1:
                        # The counter is derived from terminal rows only in the
                        # result, so no separate mutable replay counter is needed.
                        pass
                    if replica == 0:
                        self._schedule_completion(running, now)
                    break

    def _reschedule_pool(self, now: float) -> None:
        active = tuple(
            self.running[expert][replica].key
            for expert in range(self.expert_count)
            for replica in _BACKUP_REPLICAS
            if self.running[expert][replica] is not None
        )
        if active == self.last_active_keys:
            return
        k = len(active)
        speed = 0.0 if k == 0 else min(1.0, self.expert_count * self.c_b / k)
        for expert in range(self.expert_count):
            for replica in _BACKUP_REPLICAS:
                running = self.running[expert][replica]
                if running is None:
                    continue
                self.generations[expert][replica] += 1
                running.current_speed = speed
                self._schedule_completion(running, now)
        self.last_active_keys = active

    def _schedule_completion(self, running: _Running, now: float) -> None:
        remaining = running.required_work - running.executed_work
        if remaining <= 1e-12:
            when = now
        else:
            when = now + remaining / running.current_speed
        self._push(_Event(
            when,
            _EventType.COMPLETE,
            next(self.sequence),
            global_token_id=running.key.global_token_id,
            expert_id=running.key.expert_id,
            replica_id=running.key.replica_id,
            attempt_id=running.key.attempt_id,
            generation=self.generations[running.key.expert_id][running.key.replica_id],
        ))

    # -- records and audits ---------------------------------------------

    def _record(
        self,
        attempt: _Queued | _Running,
        status: AttemptStatus,
        terminal_time: float,
    ) -> None:
        token = self.tokens[attempt.key.global_token_id]
        if any(record.key == attempt.key for record in token.records):
            raise RuntimeError(f"duplicate terminal record for {attempt.key.label()}")
        start = attempt.start_time if isinstance(attempt, _Running) else None
        executed = attempt.executed_work if isinstance(attempt, _Running) else 0.0
        record = AttemptRecord(
            global_token_id=attempt.key.global_token_id,
            expert_id=attempt.key.expert_id,
            local_token_id=token.spec.local_token_id,
            attempt_id=attempt.key.attempt_id,
            replica_id=attempt.key.replica_id,
            enqueue_time=attempt.enqueue_time,
            start_time=start,
            terminal_time=terminal_time,
            required_work=attempt.required_work,
            executed_work=executed,
            remaining_work=attempt.required_work - executed,
            status=status,
            queue_delay=None if start is None else start - attempt.enqueue_time,
        )
        token.records.append(record)
        self.drain_end_time = max(self.drain_end_time, terminal_time)

    def _remaining(self, expert: int, replica: int) -> float:
        total = math.fsum(
            item.required_work for item in self.queues[expert][replica]
            if not item.cancelled
        )
        running = self.running[expert][replica]
        if running is not None:
            total += max(0.0, running.required_work - running.executed_work)
        return total

    def _ledger_subtract(self, key: tuple[int, int], amount: float) -> None:
        """Apply an O(1) remaining-work decrease with fail-fast validation."""

        if amount < 0.0 or not math.isfinite(amount):
            raise RuntimeError("remaining-work ledger received an invalid decrease")
        old = self.remaining_work[key]
        new = old - amount
        if new < -1e-12:
            raise RuntimeError(
                f"remaining-work ledger underflow for {key}: {old!r} - {amount!r}"
            )
        # A tiny negative can only be floating-point cancellation at a terminal
        # boundary.  It is not used to hide a material conservation failure.
        self.remaining_work[key] = 0.0 if new < 0.0 else new

    def _audit_queues(self, now: float) -> None:
        experts = (
            (self.tagged_expert_id,)
            if self.evaluation_mode == "tagged"
            else range(self.expert_count)
        )
        for expert in experts:
            for replica in range(_REPLICA_COUNT):
                key = (expert, replica)
                totals = self.queue_totals[key]
                previous = self.last_audit_totals[key]
                deltas = {
                    name: totals[name] - previous[name]
                    for name in ("enqueued", "executed", "cancelled", "discarded")
                }
                end = (
                    self.remaining_work[key]
                    if self.optimized else self._remaining(expert, replica)
                )
                if end < -1e-12:
                    raise RuntimeError("remaining-work ledger became negative")
                start = self.last_remaining[key]
                holds = math.isclose(
                    end,
                    start + deltas["enqueued"] - deltas["executed"]
                    - deltas["cancelled"] - deltas["discarded"],
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                if self.evaluation_mode == "full" or expert == self.tagged_expert_id:
                    self.queue_audit_events.append(QueueAuditEvent(
                        now, expert, replica, start,
                        deltas["enqueued"], deltas["executed"],
                        deltas["cancelled"], deltas["discarded"], end, holds,
                    ))
                self.last_remaining[key] = end
                self.last_audit_totals[key] = dict(totals)

    def _verify_terminal_queue_conservation(self) -> None:
        """Check every physical queue once after an exact tagged drain."""

        for expert in range(self.expert_count):
            for replica in range(_REPLICA_COUNT):
                key = (expert, replica)
                totals = self.queue_totals[key]
                ledger = self.remaining_work[key]
                physical = self._remaining(expert, replica)
                expected = (
                    totals["enqueued"] - totals["executed"]
                    - totals["cancelled"] - totals["discarded"]
                )
                if not (
                    math.isclose(ledger, physical, rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(
                        ledger, expected, rel_tol=1e-12, abs_tol=1e-12
                    )
                ):
                    raise RuntimeError(
                        "terminal queue conservation failed for "
                        f"{key}: ledger={ledger!r}, physical={physical!r}, "
                        f"expected={expected!r}"
                    )

    def _materialize_token(self, state: _TokenState) -> SharedTokenResult:
        if state.winner is None or state.completion_time is None:
            raise RuntimeError("cannot materialize incomplete Token result")
        attempts = tuple(sorted(state.records, key=lambda row: row.attempt_id))
        return SharedTokenResult(
                global_token_id=state.spec.global_token_id,
                expert_id=state.spec.expert_id,
                local_token_id=state.spec.local_token_id,
                token_class=state.spec.token_class,
                arrival_time=state.spec.arrival_time,
                primary_replica=state.primary_replica,
                requested_action=state.requested_action,
                action=state.action,
                attempts=attempts,
                winner=state.winner,
                winner_attempt_id=state.winner.attempt_id,
                completion_time=state.completion_time,
                completion_phase=phase_at(self.trace.timeline, state.completion_time),
                replay_count=state.replay_count,
                timer_scheduled=state.timer_scheduled,
                timer_fired=state.timer_fired,
                timer_voided=state.timer_voided,
                latency=state.completion_time - state.spec.arrival_time,
            )

    def result(self) -> SharedSimulationResult | TaggedSharedSimulationResult:
        prepared = self.prepared or prepare_shared_trace(self.trace)
        if self.evaluation_mode == "tagged":
            token_rows = tuple(
                self._materialize_token(state)
                for state in self.tokens
                if state.spec.expert_id == self.tagged_expert_id
            )
        else:
            token_rows = tuple(self._materialize_token(state) for state in self.tokens)
        attempts = tuple(a for token in token_rows for a in token.attempts)
        counters = SharedInvariantCounters(
            **self.counters,
            stale_completion_events_ignored=self.stale_ignored,
        )
        audits = tuple(
            QueueAudit(
                expert_id=expert,
                replica_id=replica,
                enqueued_work=self.queue_totals[(expert, replica)]["enqueued"],
                executed_work=self.queue_totals[(expert, replica)]["executed"],
                cancelled_work=self.queue_totals[(expert, replica)]["cancelled"],
                discarded_work=self.queue_totals[(expert, replica)]["discarded"],
                remaining_work=(
                    self.remaining_work[(expert, replica)]
                    if self.optimized else self._remaining(expert, replica)
                ),
            )
            for expert in range(self.expert_count)
            for replica in range(_REPLICA_COUNT)
            if self.evaluation_mode == "full" or expert == self.tagged_expert_id
        )
        if self.evaluation_mode == "tagged":
            return TaggedSharedSimulationResult(
                tagged_expert_id=self.tagged_expert_id,
                tokens=token_rows,
                attempts=attempts,
                action_decisions=tuple(self.action_decisions),
                queue_audits=audits,
                queue_audit_events=tuple(self.queue_audit_events),
                pool_audit_summary=PoolAuditSummary(
                    interval_count=self.pool_interval_count,
                    total_executed_work=self.pool_total_executed,
                    total_capacity_bound=self.pool_total_capacity,
                    max_active_heads=self.pool_max_active_heads,
                    constrained_interval_count=self.pool_constrained_intervals,
                ),
                counters=counters,
                last_arrival_time=max(token.spec.arrival_time for token in self.tokens),
                drain_end_time=self.drain_end_time,
                trace_fingerprint=prepared.trace_fingerprint,
                fault_fingerprint=prepared.fault_fingerprint,
            )
        return SharedSimulationResult(
            tokens=token_rows,
            attempts=attempts,
            action_decisions=tuple(self.action_decisions),
            pool_intervals=tuple(self.pool_intervals),
            queue_audits=audits,
            queue_audit_events=tuple(self.queue_audit_events),
            counters=counters,
            last_arrival_time=max(token.spec.arrival_time for token in self.tokens),
            drain_end_time=self.drain_end_time,
            trace_fingerprint=prepared.trace_fingerprint,
            fault_fingerprint=prepared.fault_fingerprint,
        )


class _NormalSource:
    def request(self, observation: ActionObservation) -> SharedAction:
        return SharedAction.NORMAL


def _validate_inputs(
    trace: SharedBackupTrace, c_b: object, slowdown: object, hedge_delay: object
) -> tuple[float, float, float]:
    if not isinstance(trace, SharedBackupTrace):
        raise ValueError(f"trace must be SharedBackupTrace, got {trace!r}")
    capacity = _real(c_b, "c_b", positive=True)
    degraded_slowdown = _real(slowdown, "degraded_slowdown", positive=True)
    if degraded_slowdown < 1.0:
        raise ValueError("degraded_slowdown must be >= 1")
    delay = _real(hedge_delay, "hedge_delay", positive=True)
    return capacity, degraded_slowdown, delay


def simulate_shared_backup(
    trace: SharedBackupTrace,
    c_b: float,
    *,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    action_sources: Mapping[int, object] | object | None = None,
) -> SharedSimulationResult:
    """Run deterministic Shared-Backup physics and complete the full drain.

    ``action_sources`` is either a mapping from Expert ID to an online source
    or one source shared by all Experts.  It is queried only from the arrival
    handler, after the observation has been constructed.  A mapping/list of
    pre-generated per-Token actions is intentionally rejected.
    """

    capacity, slowdown, delay = _validate_inputs(trace, c_b, degraded_slowdown, hedge_delay)
    engine = _Engine(trace, capacity, slowdown, delay, action_sources)
    engine.run()
    return engine.result()


# Explicit name used by equivalence harnesses.  The historical entry point
# above remains the unmodified reference path.
simulate_shared_backup_reference = simulate_shared_backup


def simulate_shared_backup_optimized(
    trace: SharedBackupTrace | PreparedTrace,
    c_b: float,
    *,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    action_sources: Mapping[int, object] | object | None = None,
) -> SharedSimulationResult:
    """Run the exact scheduler with prepared-input and incremental indexes."""

    prepared = prepare_shared_trace(trace)
    capacity, slowdown, delay = _validate_inputs(
        prepared.trace, c_b, degraded_slowdown, hedge_delay
    )
    engine = _Engine(
        prepared.trace, capacity, slowdown, delay, action_sources,
        prepared=prepared, optimized=True,
    )
    engine.run()
    result = engine.result()
    if not isinstance(result, SharedSimulationResult):
        raise RuntimeError("optimized full mode returned tagged output")
    return result


def simulate_shared_backup_tagged(
    trace: SharedBackupTrace | PreparedTrace,
    c_b: float,
    *,
    tagged_expert_id: int,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
    action_sources: Mapping[int, object] | object | None = None,
) -> TaggedSharedSimulationResult:
    """Run exact all-Expert physics while materializing one Expert's rows."""

    prepared = prepare_shared_trace(trace)
    capacity, slowdown, delay = _validate_inputs(
        prepared.trace, c_b, degraded_slowdown, hedge_delay
    )
    engine = _Engine(
        prepared.trace, capacity, slowdown, delay, action_sources,
        prepared=prepared, optimized=True, evaluation_mode="tagged",
        tagged_expert_id=tagged_expert_id,
    )
    engine.run()
    result = engine.result()
    if not isinstance(result, TaggedSharedSimulationResult):
        raise RuntimeError("tagged mode returned full output")
    return result


__all__ = [
    "ActionDecision",
    "ActionObservation",
    "ActionSource",
    "AttemptKey",
    "AttemptObservation",
    "AttemptRecord",
    "AttemptStatus",
    "BudgetObservation",
    "QueueAudit",
    "QueueAuditEvent",
    "QueueObservation",
    "PoolIntervalAudit",
    "PoolAuditSummary",
    "PreparedTrace",
    "SharedAction",
    "SharedBackupTrace",
    "SharedInvariantCounters",
    "SharedSimulationResult",
    "TaggedSharedSimulationResult",
    "SharedTokenResult",
    "SharedTokenSpec",
    "SharedWorkDraw",
    "prepare_shared_trace",
    "simulate_shared_backup",
    "simulate_shared_backup_reference",
    "simulate_shared_backup_optimized",
    "simulate_shared_backup_tagged",
]
