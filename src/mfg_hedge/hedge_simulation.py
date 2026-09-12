"""Hedge-capable deterministic event engine for the paired comparison.

Spec: `.scratch/mfg-hedge-paired-comparison/spec.md` sections 2-8; copy
lifecycle per ADR-0005. The engine supports Normal, Delayed Hedge, and
Immediate Hedge through an explicit per-Token action plan, draws no random
numbers, and in all-Normal mode reproduces `simulate_common_state_no_hedge`
Token-for-Token. MFG policy solving, calibration, and quota projection are out
of scope here (tickets 02/03).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import heapq
import itertools
import math
from typing import Mapping

from .common_state import (
    CommonStateTimeline,
    Phase,
    phase_at,
    speed_at,
    state_at,
    validate_slowdown,
)
from .domain import CommonState, ProtectionAction, TokenClass
from .workload import (
    WorkloadTrace,
    validate_hedge_stream_shape,
    validate_replay_stream_shape,
    validate_stream_values,
    validate_token_specs,
    validate_trace_shape,
)


class HedgeAttemptStatus(str, Enum):
    COMPLETED_WINNER = "completed_winner"
    COMPLETED_LOSER = "completed_loser"
    CANCELLED_QUEUED = "cancelled_queued"
    CANCELLED_RUNNING = "cancelled_running"
    FAILED_RUNNING = "failed_running"
    INVALIDATED_QUEUED = "invalidated_queued"


class _EventType(IntEnum):
    """Heap rank at equal event_time; the spec's fixed same-time order."""

    DOMAIN_STATE_CHANGE = 1
    COMPLETE = 2
    HEDGE_TIMER = 3
    TOKEN_ARRIVAL = 4


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
class HedgeAttemptRecord:
    """Terminal record of one copy: Primary=0, Replay=1, Hedge Backup=2."""

    token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    start_time: float | None
    terminal_time: float
    required_work: float
    executed_work: float
    remaining_work: float
    status: HedgeAttemptStatus
    queue_delay: float | None


@dataclass(frozen=True)
class HedgeTokenResult:
    token_id: int
    token_class: TokenClass
    arrival_time: float
    primary_replica: int
    action: ProtectionAction
    attempts: tuple[HedgeAttemptRecord, ...]
    winner_attempt_id: int
    completion_time: float
    completion_phase: Phase
    replay_count: int
    latency: float


@dataclass(frozen=True)
class HedgeSimulationResult:
    """Raw engine facts; derived paired metrics are computed elsewhere."""

    tokens: tuple[HedgeTokenResult, ...]
    attempts: tuple[HedgeAttemptRecord, ...]
    primary_executions: int
    replay_executions: int
    hedge_requested: int
    hedge_launches: int
    hedge_suppressed: int
    hedge_timers_voided: int
    cancelled_queued_total: int
    completed_loser_total: int
    failed_running_primary_executions: int
    invalidated_queued_primary_executions: int
    last_arrival_time: float
    drain_end_time: float
    drain_duration: float
    stale_completion_events_ignored: int
    queue_length_at_failed_start: tuple[int, int]
    queue_length_at_recovered_start: tuple[int, int]
    hedge_suppressed_events: tuple[tuple[float, int], ...] = ()

    @property
    def completed_tokens(self) -> int:
        return len(self.tokens)

    @property
    def failed_primary_executions(self) -> int:
        return (
            self.failed_running_primary_executions
            + self.invalidated_queued_primary_executions
        )


@dataclass
class _QueuedAttempt:
    token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    required_work: float
    cancelled: bool = False


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


@dataclass
class _TokenState:
    token_id: int
    action: ProtectionAction
    requested_action: ProtectionAction | None = None
    decision_made: bool = False
    primary_replica: int | None = None
    replay_count: int = 0
    timer_pending: bool = False
    winner_set: bool = False
    live: dict[int, _QueuedAttempt | _RunningAttempt] = field(default_factory=dict)


class _Engine:
    """Mutable evaluation state; one instance per run, discarded afterwards."""

    def __init__(
        self,
        trace: WorkloadTrace,
        timeline: CommonStateTimeline,
        degraded_slowdown: float,
        actions: Mapping[int, ProtectionAction],
        hedge_delay: float | None,
        replica_count: int,
        online_source: object | None = None,
        reservation_ledger: object | None = None,
        public_price: float = 0.0,
        cancel_running_losers: bool = False,
    ) -> None:
        self.trace = trace
        self.timeline = timeline
        self.slowdown = degraded_slowdown
        self.hedge_delay = hedge_delay
        self.replica_count = replica_count
        self.online_source = online_source
        self.reservation_ledger = reservation_ledger
        if type(cancel_running_losers) is not bool:
            raise ValueError("cancel_running_losers must be bool")
        self.cancel_running_losers = cancel_running_losers
        if (
            isinstance(public_price, bool)
            or not isinstance(public_price, (int, float))
            or not math.isfinite(float(public_price))
            or float(public_price) < 0.0
        ):
            raise ValueError("public_price must be finite and non-negative")
        self.public_price = float(public_price)
        if (online_source is None) != (reservation_ledger is None):
            raise ValueError(
                "online_source and reservation_ledger must be supplied together"
            )
        self.queues: list[deque[_QueuedAttempt]] = [
            deque() for _ in range(replica_count)
        ]
        self.running: list[_RunningAttempt | None] = [None] * replica_count
        self.generations = [0] * replica_count
        self.events: list[tuple[float, int, int, _Event]] = []
        self.sequence = itertools.count()

        token_count = len(trace.tokens)
        self.tokens = [
            _TokenState(
                token_id=spec.token_id,
                action=actions.get(spec.token_id, ProtectionAction.NORMAL),
                requested_action=(
                    actions.get(spec.token_id, ProtectionAction.NORMAL)
                    if online_source is None else None
                ),
                decision_made=online_source is None,
            )
            for spec in trace.tokens
        ]
        self.completion_times: list[float | None] = [None] * token_count
        self.winner_attempt_ids: list[int | None] = [None] * token_count
        self.attempts_by_token: list[list[HedgeAttemptRecord]] = [
            [] for _ in range(token_count)
        ]

        self.primary_executions = 0
        self.replay_executions = 0
        self.hedge_requested = (
            sum(1 for a in actions.values() if a is not ProtectionAction.NORMAL)
            if online_source is None else 0
        )
        self.hedge_launches = 0
        self.hedge_suppressed = 0
        self.hedge_timers_voided = 0
        self.cancelled_queued_total = 0
        self.completed_loser_total = 0
        self.failed_running = 0
        self.invalidated_queued = 0
        self.stale_ignored = 0
        self.drain_end_time = 0.0
        self.queue_length_at_failed_start = (0, 0)
        self.queue_length_at_recovered_start = (0, 0)
        self.live_attempts_at_failed_start = (0, 0)
        self.live_attempts_at_recovered_start = (0, 0)
        self.remaining_work_at_failed_start = (0.0, 0.0)
        self.remaining_work_at_recovered_start = (0.0, 0.0)
        self.suppressed_events: list[tuple[float, int]] = []
        self.online_decisions: list[object] = []
        self.observed_history: list[str] = []

    # -- event plumbing -------------------------------------------------

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
                elif event.event_type is _EventType.HEDGE_TIMER:
                    self._handle_timer(event)
                else:
                    self._handle_arrival(event)
            self._dispatch_pass(now)

    # -- state changes ---------------------------------------------------

    def _online_observation(self, spec: TokenSpec, now: float, phase, state,
                            primary_replica: int):
        """Build the causal arrival snapshot without exposing trace physics."""
        from .token_online import TokenObservation

        phase_start = {
            Phase.HEALTHY: 0.0,
            Phase.DEGRADED: self.timeline.degraded_start,
            Phase.FAILED: self.timeline.failed_start,
            Phase.RECOVERED: self.timeline.recovered_start,
        }[phase_at(self.timeline, now)]
        queue_snapshot = tuple(
            tuple(
                (queued.token_id, queued.attempt_id)
                for queued in queue
                if not queued.cancelled
            )
            for queue in self.queues
        )
        running_attempts = tuple(
            (running.token_id, running.attempt_id, running.replica_id)
            for running in self.running
            if running is not None
        )
        preview = self.reservation_ledger.preview(
            now, spec.token_class, primary_replica
        )
        return TokenObservation(
            token_id=spec.token_id,
            token_class=spec.token_class,
            arrival_time=now,
            phase=phase_at(self.timeline, now),
            phase_age=now - phase_start,
            primary_replica=primary_replica,
            queue_snapshot=queue_snapshot,
            running_attempts=running_attempts,
            observed_history=tuple(self.observed_history),
            reservation_window=preview.window,
            reservation_cap=preview.cap,
            reservation_balance=preview.balance_before,
            public_price=self.public_price,
            common_state=state,
        )

    def _handle_state_change(self, event: _Event) -> None:
        moment = event.event_time
        if moment == self.timeline.degraded_start:
            self._enter_degraded(moment)
            self.observed_history.append(f"state:D@{moment!r}")
        elif moment == self.timeline.failed_start:
            self._enter_failed(moment)
            self.observed_history.append(f"state:F@{moment!r}")
        else:
            self._enter_recovered(moment)
            self.observed_history.append(f"state:R@{moment!r}")

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
            return  # physically completes exactly at this boundary
        self.generations[0] += 1
        running.current_speed = speed_at(0, CommonState.DEGRADED, self.slowdown)
        self._schedule_completion(running, now + remaining / running.current_speed)

    def _live_queue_length(self, replica: int) -> int:
        return sum(1 for queued in self.queues[replica] if not queued.cancelled)

    def _boundary_snapshot(
        self, now: float
    ) -> tuple[tuple[int, int], tuple[float, float]]:
        """Read live attempt counts and remaining work without settling state.

        The call site is inside the state-change handler, after its
        failure/Replay substeps and before same-time completion, timer,
        arrival, or dispatch handling.  Cancelled queue tombstones and pending
        timers are not live attempts.  Projecting a running attempt is a pure
        read so instrumentation cannot perturb the legacy simulation result.
        """
        live: list[int] = []
        remaining: list[float] = []
        for replica in range(self.replica_count):
            queued = [item for item in self.queues[replica] if not item.cancelled]
            running = self.running[replica]
            replica_live = len(queued) + int(running is not None)
            replica_remaining = math.fsum(item.required_work for item in queued)
            if running is not None:
                projected_executed = running.executed_work + (
                    now - running.last_update_time
                ) * running.current_speed
                replica_remaining += max(
                    0.0, running.required_work - projected_executed
                )
            live.append(replica_live)
            remaining.append(replica_remaining)
        return (live[0], live[1]), (remaining[0], remaining[1])

    def _enter_failed(self, now: float) -> None:
        self.generations[0] += 1
        lost_primaries: list[int] = []
        running = self.running[0]
        if running is not None:
            self._settle(running, now)
            self.running[0] = None
            token = self.tokens[running.token_id]
            token.live.pop(running.attempt_id, None)
            self._record(
                running,
                HedgeAttemptStatus.FAILED_RUNNING,
                terminal_time=now,
                executed_work=running.executed_work,
            )
            if running.attempt_id == 0:
                self.failed_running += 1
                lost_primaries.append(running.token_id)
        for queued in list(self.queues[0]):
            if queued.cancelled:
                continue  # already terminal as cancelled_queued
            token = self.tokens[queued.token_id]
            token.live.pop(queued.attempt_id, None)
            self._record(
                queued,
                HedgeAttemptStatus.INVALIDATED_QUEUED,
                terminal_time=now,
                executed_work=0.0,
            )
            if queued.attempt_id == 0:
                self.invalidated_queued += 1
                lost_primaries.append(queued.token_id)
        self.queues[0].clear()
        for token_id in lost_primaries:
            token = self.tokens[token_id]
            if token.winner_set or 2 in token.live or token.replay_count != 0:
                continue
            token.replay_count = 1
            token.timer_pending = False
            self._enqueue(
                _QueuedAttempt(
                    token_id=token_id,
                    attempt_id=1,
                    replica_id=1,
                    enqueue_time=now,
                    required_work=self.trace.replay_service_times[1][token_id],
                )
            )
        self.queue_length_at_failed_start = (
            self._live_queue_length(0),
            self._live_queue_length(1),
        )
        (
            self.live_attempts_at_failed_start,
            self.remaining_work_at_failed_start,
        ) = self._boundary_snapshot(now)

    def _enter_recovered(self, now: float) -> None:
        self.queue_length_at_recovered_start = (
            self._live_queue_length(0),
            self._live_queue_length(1),
        )
        (
            self.live_attempts_at_recovered_start,
            self.remaining_work_at_recovered_start,
        ) = self._boundary_snapshot(now)

    # -- completions, timers, arrivals ------------------------------------

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
        self.running[replica] = None
        token = self.tokens[running.token_id]
        token.live.pop(running.attempt_id, None)
        now = event.event_time
        if token.winner_set:
            self._record(
                running,
                HedgeAttemptStatus.COMPLETED_LOSER,
                terminal_time=now,
                executed_work=running.required_work,
            )
            self.completed_loser_total += 1
            self.observed_history.append(
                f"complete:loser:{running.token_id}:{running.attempt_id}@{now!r}"
            )
            return
        token.winner_set = True
        token.timer_pending = False
        self._record(
            running,
            HedgeAttemptStatus.COMPLETED_WINNER,
            terminal_time=now,
            executed_work=running.required_work,
        )
        self.completion_times[running.token_id] = now
        self.winner_attempt_ids[running.token_id] = running.attempt_id
        for attempt_id, other in list(token.live.items()):
            if isinstance(other, _QueuedAttempt) and not other.cancelled:
                other.cancelled = True
                self._record(
                    other,
                    HedgeAttemptStatus.CANCELLED_QUEUED,
                    terminal_time=now,
                    executed_work=0.0,
                )
                self.cancelled_queued_total += 1
                token.live.pop(attempt_id, None)
            elif (
                self.cancel_running_losers
                and isinstance(other, _RunningAttempt)
            ):
                self._settle(other, now)
                self.running[other.replica_id] = None
                self.generations[other.replica_id] += 1
                token.live.pop(attempt_id, None)
                self._record(
                    other,
                    HedgeAttemptStatus.CANCELLED_RUNNING,
                    terminal_time=now,
                    executed_work=other.executed_work,
                )
        self.observed_history.append(
            f"complete:winner:{running.token_id}:{running.attempt_id}@{now!r}"
        )

    def _handle_timer(self, event: _Event) -> None:
        token = self.tokens[event.token_id]
        if token.winner_set or not token.timer_pending:
            self.hedge_timers_voided += 1
            self.observed_history.append(f"timer:voided:{event.token_id}@{event.event_time!r}")
            return
        token.timer_pending = False
        if state_at(self.timeline, event.event_time) is CommonState.FAILED:
            self.hedge_suppressed += 1
            self.suppressed_events.append((event.event_time, event.token_id))
            self.observed_history.append(f"timer:suppressed:{event.token_id}@{event.event_time!r}")
            return
        self._launch_hedge(token, event.event_time)
        self.observed_history.append(f"timer:fired:{event.token_id}@{event.event_time!r}")

    def _handle_arrival(self, event: _Event) -> None:
        spec = self.trace.tokens[event.token_id]
        token = self.tokens[spec.token_id]
        now = event.event_time
        state = state_at(self.timeline, now)
        primary_replica = 1 if state is CommonState.FAILED else spec.token_id % 2
        token.primary_replica = primary_replica
        if self.online_source is not None:
            if token.decision_made:
                raise RuntimeError(f"Token {spec.token_id} received a second decision")
            from .token_online import validate_token_action

            observation = self._online_observation(
                spec, now, phase_at(self.timeline, now), state, primary_replica
            )
            requested = validate_token_action(
                self.online_source.choose(
                    observation,
                    f"token:{self.trace.base_seed}:{spec.token_id}",
                )
            )
            admission = self.reservation_ledger.admit(
                spec.token_id, now, spec.token_class, primary_replica, requested
            )
            if admission.applied is ProtectionAction.DELAYED_HEDGE and self.hedge_delay is None:
                raise ValueError("hedge_delay is required when an admitted online D is used")
            if admission.applied is not ProtectionAction.NORMAL:
                from .workload import validate_hedge_stream_shape, validate_stream_values

                validate_hedge_stream_shape(self.trace, self.replica_count)
                validate_stream_values(self.trace.hedge_service_times, 2, "hedge")
            token.requested_action = admission.requested
            token.action = admission.applied
            token.decision_made = True
            self.online_decisions.append(admission)
            self.hedge_requested += int(token.action is not ProtectionAction.NORMAL)
        self._enqueue(
            _QueuedAttempt(
                token_id=spec.token_id,
                attempt_id=0,
                replica_id=primary_replica,
                enqueue_time=now,
                required_work=self.trace.service_times[primary_replica][spec.token_id],
            )
        )
        self.primary_executions += 1
        if token.action is ProtectionAction.IMMEDIATE_HEDGE:
            if state is CommonState.FAILED:
                self.hedge_suppressed += 1
                self.suppressed_events.append((now, spec.token_id))
            else:
                self._launch_hedge(token, now)
        elif token.action is ProtectionAction.DELAYED_HEDGE:
            token.timer_pending = True
            self._push(
                _Event(
                    event_time=now + self.hedge_delay,
                    event_type=_EventType.HEDGE_TIMER,
                    sequence=next(self.sequence),
                    token_id=spec.token_id,
                )
            )
        self.observed_history.append(f"arrival:{spec.token_id}@{now!r}")

    # -- dispatch ----------------------------------------------------------

    def _launch_hedge(self, token_state: _TokenState, now: float) -> None:
        backup = 1 - token_state.primary_replica
        self.hedge_launches += 1
        self._enqueue(
            _QueuedAttempt(
                token_id=token_state.token_id,
                attempt_id=2,
                replica_id=backup,
                enqueue_time=now,
                required_work=self.trace.hedge_service_times[backup][
                    token_state.token_id
                ],
            )
        )

    def _enqueue(self, attempt: _QueuedAttempt) -> None:
        self.queues[attempt.replica_id].append(attempt)
        self.tokens[attempt.token_id].live[attempt.attempt_id] = attempt

    def _dispatch_pass(self, now: float) -> None:
        for replica in range(self.replica_count):
            while self.running[replica] is None and self.queues[replica]:
                queued = self.queues[replica].popleft()
                if queued.cancelled:
                    continue
                self._start(replica, queued, now)

    def _start(self, replica: int, queued: _QueuedAttempt, now: float) -> None:
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
        self.tokens[queued.token_id].live[queued.attempt_id] = running
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

    def _record(
        self,
        attempt: _RunningAttempt | _QueuedAttempt,
        status: HedgeAttemptStatus,
        terminal_time: float,
        executed_work: float,
    ) -> None:
        start_time = (
            attempt.start_time if isinstance(attempt, _RunningAttempt) else None
        )
        record = HedgeAttemptRecord(
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
        self.drain_end_time = max(self.drain_end_time, terminal_time)


def _simulate_hedge_common_state_with_engine(
    trace: WorkloadTrace,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
    actions: Mapping[int, ProtectionAction] | None = None,
    hedge_delay: float | None = None,
    replica_count: int = 2,
    online_source: object | None = None,
    reservation_ledger: object | None = None,
    public_price: float = 0.0,
    cancel_running_losers: bool = False,
) -> tuple[HedgeSimulationResult, _Engine]:
    """Evaluate ``trace`` with per-Token protection actions.

    ``actions`` maps Token ids to ProtectionAction; missing ids are Normal.
    ``hedge_delay`` is tau0 for Delayed Hedge timers, measured from arrival.
    Runs until every Token has exactly one winner and one completion; draws no
    random numbers.
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
    validate_token_specs(trace)
    validate_stream_values(trace.service_times, 0, "service")
    validate_stream_values(trace.replay_service_times, 1, "replay")

    if online_source is not None and actions is not None:
        raise ValueError("online_source cannot be combined with an action plan")
    if online_source is not None and reservation_ledger is None:
        raise ValueError("online_source requires a reservation_ledger")
    plan = dict(actions) if actions is not None else {}
    token_ids = set(range(len(trace.tokens)))
    for token_id, action in plan.items():
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id not in token_ids
        ):
            raise ValueError(f"unknown Token id in action plan: {token_id!r}")
        if not isinstance(action, ProtectionAction):
            raise ValueError(f"action must be a ProtectionAction, got {action!r}")
    needs_hedge = any(a is not ProtectionAction.NORMAL for a in plan.values())
    if needs_hedge:
        validate_hedge_stream_shape(trace, replica_count)
        validate_stream_values(trace.hedge_service_times, 2, "hedge")
    needs_timer = any(
        a is ProtectionAction.DELAYED_HEDGE for a in plan.values()
    )
    if hedge_delay is not None or needs_timer:
        if (
            isinstance(hedge_delay, bool)
            or not isinstance(hedge_delay, (int, float))
            or not math.isfinite(float(hedge_delay))
            or float(hedge_delay) <= 0.0
        ):
            raise ValueError(
                f"hedge_delay (tau0) must be finite and > 0, got {hedge_delay!r}"
            )
        hedge_delay = float(hedge_delay)

    engine = _Engine(
        trace, timeline, slowdown, plan, hedge_delay, replica_count,
        online_source=online_source,
        reservation_ledger=reservation_ledger,
        public_price=public_price,
        cancel_running_losers=cancel_running_losers,
    )
    engine.run()

    tokens: list[HedgeTokenResult] = []
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
            HedgeTokenResult(
                token_id=token_id,
                token_class=spec.token_class,
                arrival_time=spec.arrival_time,
                primary_replica=engine.tokens[token_id].primary_replica,
                action=engine.tokens[token_id].action,
                attempts=attempts,
                winner_attempt_id=engine.winner_attempt_ids[token_id],
                completion_time=completion,
                completion_phase=phase_at(timeline, completion),
                replay_count=engine.tokens[token_id].replay_count,
                latency=completion - spec.arrival_time,
            )
        )

    last_arrival_time = max(spec.arrival_time for spec in trace.tokens)
    result = HedgeSimulationResult(
        tokens=tuple(tokens),
        attempts=tuple(record for token in tokens for record in token.attempts),
        primary_executions=engine.primary_executions,
        replay_executions=engine.replay_executions,
        hedge_requested=engine.hedge_requested,
        hedge_launches=engine.hedge_launches,
        hedge_suppressed=engine.hedge_suppressed,
        hedge_timers_voided=engine.hedge_timers_voided,
        cancelled_queued_total=engine.cancelled_queued_total,
        completed_loser_total=engine.completed_loser_total,
        failed_running_primary_executions=engine.failed_running,
        invalidated_queued_primary_executions=engine.invalidated_queued,
        last_arrival_time=last_arrival_time,
        drain_end_time=engine.drain_end_time,
        drain_duration=engine.drain_end_time - last_arrival_time,
        stale_completion_events_ignored=engine.stale_ignored,
        queue_length_at_failed_start=engine.queue_length_at_failed_start,
        queue_length_at_recovered_start=engine.queue_length_at_recovered_start,
        hedge_suppressed_events=tuple(engine.suppressed_events),
    )
    return result, engine


def simulate_hedge_common_state(
    trace: WorkloadTrace,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
    actions: Mapping[int, ProtectionAction] | None = None,
    hedge_delay: float | None = None,
    replica_count: int = 2,
    cancel_running_losers: bool = False,
) -> HedgeSimulationResult:
    """Evaluate ``trace`` with per-Token protection actions.

    ``actions`` maps Token ids to ProtectionAction; missing ids are Normal.
    ``hedge_delay`` is tau0 for Delayed Hedge timers, measured from arrival.
    Runs until every Token has exactly one winner and one completion; draws no
    random numbers.  Episode instrumentation uses the same private run and
    leaves this public result contract unchanged.
    """
    result, _ = _simulate_hedge_common_state_with_engine(
        trace,
        timeline,
        degraded_slowdown,
        actions=actions,
        hedge_delay=hedge_delay,
        replica_count=replica_count,
        cancel_running_losers=cancel_running_losers,
    )
    return result


def simulate_hedge_common_state_online(
    trace: WorkloadTrace,
    timeline: CommonStateTimeline,
    degraded_slowdown: float,
    *,
    action_source: object,
    reservation_ledger: object,
    hedge_delay: float | None = None,
    replica_count: int = 2,
    public_price: float = 0.0,
    cancel_running_losers: bool = False,
) -> HedgeSimulationResult:
    """Run the same A/B event engine with one causal arrival-time source.

    This is an additive Token restoration entry point.  The legacy static
    ``actions`` mapping path above remains unchanged and is still the
    historical reference path.
    """
    if action_source is None:
        raise ValueError("action_source is required")
    result, _ = _simulate_hedge_common_state_with_engine(
        trace,
        timeline,
        degraded_slowdown,
        actions=None,
        hedge_delay=hedge_delay,
        replica_count=replica_count,
        online_source=action_source,
        reservation_ledger=reservation_ledger,
        public_price=public_price,
        cancel_running_losers=cancel_running_losers,
    )
    return result
