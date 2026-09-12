"""Deterministic Failure/Replay event engine for Common State + No Hedge.

Spec: `.scratch/common-state-no-hedge/spec.md` sections 2-4 and 6. The engine
consumes an immutable WorkloadTrace and draws no random numbers. Same-time
semantics come from explicit type ranks and ordered sub-steps, never from heap
insertion order. This module records raw facts only; derived metrics belong to
the metrics layer (ticket 03).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
import heapq
import itertools

from .common_state import (
    CommonStateTimeline,
    Phase,
    phase_at,
    speed_at,
    state_at,
    validate_slowdown,
)
from .domain import CommonState, TokenClass
from .workload import (
    WorkloadTrace,
    validate_replay_stream_shape,
    validate_trace_shape,
)


class AttemptStatus(str, Enum):
    COMPLETED = "completed"
    FAILED_RUNNING = "failed_running"
    INVALIDATED_QUEUED = "invalidated_queued"


class _EventType(IntEnum):
    """Heap rank at equal event_time; values are the spec's same-time order."""

    DOMAIN_STATE_CHANGE = 1
    COMPLETE = 4
    TOKEN_ARRIVAL = 5


@dataclass(frozen=True)
class _Event:
    event_time: float
    event_type: _EventType
    sequence: int
    token_id: int | None = None
    replica_id: int | None = None
    attempt_id: int | None = None
    generation: int = 0


@dataclass(frozen=True)
class AttemptRecord:
    """Terminal record of one execution attempt (Primary=0, Replay=1)."""

    token_id: int
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


@dataclass(frozen=True)
class CommonStateTokenResult:
    token_id: int
    token_class: TokenClass
    arrival_time: float
    primary_replica: int
    attempts: tuple[AttemptRecord, ...]
    completion_time: float
    completion_phase: Phase
    replay_count: int
    latency: float


@dataclass(frozen=True)
class CommonStateSimulationResult:
    """Raw engine facts; derived metrics are computed by the metrics layer."""

    tokens: tuple[CommonStateTokenResult, ...]
    attempts: tuple[AttemptRecord, ...]
    primary_executions: int
    replay_executions: int
    hedge_launches: int
    failed_running_primary_executions: int
    invalidated_queued_primary_executions: int
    last_arrival_time: float
    drain_end_time: float
    drain_duration: float
    stale_completion_events_ignored: int
    queue_length_at_failed_start: tuple[int, int]
    queue_length_at_recovered_start: tuple[int, int]

    @property
    def completed_tokens(self) -> int:
        return len(self.tokens)

    @property
    def failed_primary_executions(self) -> int:
        return (
            self.failed_running_primary_executions
            + self.invalidated_queued_primary_executions
        )


@dataclass(frozen=True)
class _QueuedAttempt:
    token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    required_work: float


@dataclass
class _RunningAttempt:
    token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    start_time: float
    required_work: float
    executed_work: float
    last_update_time: float
    current_speed: float


class _Engine:
    """Mutable evaluation state; one instance per run, discarded afterwards."""

    def __init__(
        self,
        trace: WorkloadTrace,
        timeline: CommonStateTimeline,
        degraded_slowdown: float,
        replica_count: int,
    ) -> None:
        self.trace = trace
        self.timeline = timeline
        self.slowdown = degraded_slowdown
        self.replica_count = replica_count
        self.queues = [deque() for _ in range(replica_count)]
        self.running: list[_RunningAttempt | None] = [None] * replica_count
        self.generations = [0] * replica_count
        self.events: list[tuple[float, int, int, _Event]] = []
        self.sequence = itertools.count()

        token_count = len(trace.tokens)
        self.replay_counts = [0] * token_count
        self.primary_replicas: list[int | None] = [None] * token_count
        self.completion_times: list[float | None] = [None] * token_count
        self.attempts_by_token: list[list[AttemptRecord]] = [
            [] for _ in range(token_count)
        ]

        self.primary_executions = 0
        self.replay_executions = 0
        self.failed_running = 0
        self.invalidated_queued = 0
        self.stale_ignored = 0
        self.drain_end_time = 0.0
        self.queue_length_at_failed_start = (0, 0)
        self.queue_length_at_recovered_start = (0, 0)

    def _push(self, event: _Event) -> None:
        heapq.heappush(
            self.events,
            (event.event_time, int(event.event_type), event.sequence, event),
        )

    def _schedule_initial_events(self) -> None:
        for boundary in (
            self.timeline.degraded_start,
            self.timeline.failed_start,
            self.timeline.recovered_start,
        ):
            self._push(
                _Event(
                    event_time=boundary,
                    event_type=_EventType.DOMAIN_STATE_CHANGE,
                    sequence=next(self.sequence),
                )
            )
        for spec in self.trace.tokens:
            self._push(
                _Event(
                    event_time=spec.arrival_time,
                    event_type=_EventType.TOKEN_ARRIVAL,
                    sequence=next(self.sequence),
                    token_id=spec.token_id,
                )
            )

    def run(self) -> None:
        self._schedule_initial_events()
        while self.events:
            first = heapq.heappop(self.events)
            batch = [first]
            while self.events and self.events[0][0] == first[0]:
                batch.append(heapq.heappop(self.events))
            batch.sort(key=lambda item: (item[1], item[2]))
            now = first[0]
            for _, _, _, event in batch:
                if event.event_type is _EventType.DOMAIN_STATE_CHANGE:
                    self._handle_state_change(event)
                elif event.event_type is _EventType.COMPLETE:
                    self._handle_complete(event)
                else:
                    self._handle_arrival(event)
            self._dispatch_pass(now)

    # -- state changes -------------------------------------------------

    def _handle_state_change(self, event: _Event) -> None:
        moment = event.event_time
        if moment == self.timeline.degraded_start:
            self._enter_degraded(moment)
        elif moment == self.timeline.failed_start:
            self._enter_failed(moment)
        else:
            self._enter_recovered(moment)

    def _settle(self, running: _RunningAttempt, now: float) -> None:
        running.executed_work += (
            now - running.last_update_time
        ) * running.current_speed
        running.last_update_time = now

    def _enter_degraded(self, now: float) -> None:
        running = self.running[0]
        if running is None:
            return
        self._settle(running, now)
        remaining = running.required_work - running.executed_work
        if remaining <= 0.0:
            # Physical completion is exactly at this boundary; the already
            # scheduled COMPLETE stays valid and finishes in this batch.
            return
        self.generations[0] += 1
        running.current_speed = speed_at(0, CommonState.DEGRADED, self.slowdown)
        self._schedule_completion(running, now + remaining / running.current_speed)

    def _enter_failed(self, now: float) -> None:
        self.generations[0] += 1
        lost: list[int] = []
        running = self.running[0]
        if running is not None:
            self._settle(running, now)
            # Fault-first: the execution fails even if its remaining work is
            # exactly zero at this boundary.
            self._record_attempt(
                running,
                status=AttemptStatus.FAILED_RUNNING,
                terminal_time=now,
                executed_work=running.executed_work,
            )
            self.failed_running += 1
            lost.append(running.token_id)
            self.running[0] = None
        for queued in self.queues[0]:
            self._record_attempt(
                queued,
                status=AttemptStatus.INVALIDATED_QUEUED,
                terminal_time=now,
                executed_work=0.0,
            )
            self.invalidated_queued += 1
            lost.append(queued.token_id)
        self.queues[0].clear()
        for token_id in lost:
            if self.replay_counts[token_id] != 0:
                continue
            self.replay_counts[token_id] = 1
            self.queues[1].append(
                _QueuedAttempt(
                    token_id=token_id,
                    attempt_id=1,
                    replica_id=1,
                    enqueue_time=now,
                    required_work=self.trace.replay_service_times[1][token_id],
                )
            )
        self.queue_length_at_failed_start = (
            len(self.queues[0]),
            len(self.queues[1]),
        )

    def _enter_recovered(self, now: float) -> None:
        # Replica 0 returns empty; nothing running or queued there to handle.
        self.queue_length_at_recovered_start = (
            len(self.queues[0]),
            len(self.queues[1]),
        )

    # -- completions and arrivals ---------------------------------------

    def _handle_complete(self, event: _Event) -> None:
        replica = event.replica_id
        if event.generation != self.generations[replica]:
            self.stale_ignored += 1
            return
        running = self.running[replica]
        if running is None or (running.token_id, running.attempt_id) != (
            event.token_id,
            event.attempt_id,
        ):
            return
        if self.completion_times[event.token_id] is not None:
            return
        self.running[replica] = None
        self._record_attempt(
            running,
            status=AttemptStatus.COMPLETED,
            terminal_time=event.event_time,
            executed_work=running.required_work,
        )
        self.completion_times[event.token_id] = event.event_time
        self.drain_end_time = max(self.drain_end_time, event.event_time)

    def _handle_arrival(self, event: _Event) -> None:
        spec = self.trace.tokens[event.token_id]
        state = state_at(self.timeline, event.event_time)
        replica = 1 if state is CommonState.FAILED else spec.token_id % 2
        self.primary_replicas[spec.token_id] = replica
        self.queues[replica].append(
            _QueuedAttempt(
                token_id=spec.token_id,
                attempt_id=0,
                replica_id=replica,
                enqueue_time=event.event_time,
                required_work=self.trace.service_times[replica][spec.token_id],
            )
        )
        self.primary_executions += 1

    # -- dispatch --------------------------------------------------------

    def _dispatch_pass(self, now: float) -> None:
        for replica in range(self.replica_count):
            if self.running[replica] is None and self.queues[replica]:
                self._start_next(replica, now)

    def _start_next(self, replica: int, now: float) -> None:
        queued = self.queues[replica].popleft()
        speed = speed_at(replica, state_at(self.timeline, now), self.slowdown)
        if speed <= 0.0:
            raise AssertionError("dispatch must never start on a Failed Replica")
        self.generations[replica] += 1
        running = _RunningAttempt(
            token_id=queued.token_id,
            attempt_id=queued.attempt_id,
            replica_id=replica,
            enqueue_time=queued.enqueue_time,
            start_time=now,
            required_work=queued.required_work,
            executed_work=0.0,
            last_update_time=now,
            current_speed=speed,
        )
        self.running[replica] = running
        if queued.attempt_id == 1:
            self.replay_executions += 1
        self._schedule_completion(running, now + queued.required_work / speed)

    def _schedule_completion(self, running: _RunningAttempt, when: float) -> None:
        self._push(
            _Event(
                event_time=when,
                event_type=_EventType.COMPLETE,
                sequence=next(self.sequence),
                token_id=running.token_id,
                replica_id=running.replica_id,
                attempt_id=running.attempt_id,
                generation=self.generations[running.replica_id],
            )
        )

    # -- recording ---------------------------------------------------------

    def _record_attempt(
        self,
        attempt: _RunningAttempt | _QueuedAttempt,
        status: AttemptStatus,
        terminal_time: float,
        executed_work: float,
    ) -> None:
        start_time = (
            attempt.start_time if isinstance(attempt, _RunningAttempt) else None
        )
        record = AttemptRecord(
            token_id=attempt.token_id,
            attempt_id=attempt.attempt_id,
            replica_id=attempt.replica_id,
            enqueue_time=attempt.enqueue_time,
            start_time=start_time,
            terminal_time=terminal_time,
            required_work=attempt.required_work,
            executed_work=executed_work,
            remaining_work=attempt.required_work - executed_work,
            status=status,
            queue_delay=(
                None if start_time is None else start_time - attempt.enqueue_time
            ),
        )
        self.attempts_by_token[attempt.token_id].append(record)


def simulate_common_state_no_hedge(
    trace: WorkloadTrace,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
    replica_count: int = 2,
) -> CommonStateSimulationResult:
    """Evaluate ``trace`` under the common-state failure timeline.

    Runs until every Token has exactly one terminal outcome; all Tokens
    complete because Replica 1 never fails. Draws no random numbers.
    """
    if (
        isinstance(replica_count, bool)
        or not isinstance(replica_count, int)
        or replica_count != 2
    ):
        raise ValueError(
            f"replica_count must be the int 2 for this slice, got {replica_count!r}"
        )
    if not isinstance(timeline, CommonStateTimeline):
        raise ValueError(
            f"timeline must be a CommonStateTimeline, got {timeline!r}"
        )
    slowdown = validate_slowdown(degraded_slowdown)
    validate_trace_shape(trace, replica_count)
    validate_replay_stream_shape(trace, replica_count)

    engine = _Engine(trace, timeline, slowdown, replica_count)
    engine.run()

    tokens: list[CommonStateTokenResult] = []
    for spec in trace.tokens:
        token_id = spec.token_id
        completion = engine.completion_times[token_id]
        if completion is None:
            raise RuntimeError(
                f"engine terminated with Token {token_id} incomplete"
            )
        attempts = tuple(
            sorted(engine.attempts_by_token[token_id], key=lambda a: a.attempt_id)
        )
        tokens.append(
            CommonStateTokenResult(
                token_id=token_id,
                token_class=spec.token_class,
                arrival_time=spec.arrival_time,
                primary_replica=engine.primary_replicas[token_id],
                attempts=attempts,
                completion_time=completion,
                completion_phase=phase_at(timeline, completion),
                replay_count=engine.replay_counts[token_id],
                latency=completion - spec.arrival_time,
            )
        )

    last_arrival_time = max(spec.arrival_time for spec in trace.tokens)
    return CommonStateSimulationResult(
        tokens=tuple(tokens),
        attempts=tuple(record for token in tokens for record in token.attempts),
        primary_executions=engine.primary_executions,
        replay_executions=engine.replay_executions,
        hedge_launches=0,
        failed_running_primary_executions=engine.failed_running,
        invalidated_queued_primary_executions=engine.invalidated_queued,
        last_arrival_time=last_arrival_time,
        drain_end_time=engine.drain_end_time,
        drain_duration=engine.drain_end_time - last_arrival_time,
        stale_completion_events_ignored=engine.stale_ignored,
        queue_length_at_failed_start=engine.queue_length_at_failed_start,
        queue_length_at_recovered_start=engine.queue_length_at_recovered_start,
    )
