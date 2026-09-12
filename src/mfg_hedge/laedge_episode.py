"""Work-conserving two-Replica LÆDGE episode executor.

The executor is isolated from the N/D/I engine because LÆDGE makes a new
decision whenever a replica becomes idle. It preserves the repository's
fault-first common-state semantics and immutable attempt-keyed CRN streams.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
import heapq
import itertools
import math

from .attribution_episode import BoundarySnapshot, EpisodeSimulationResult, EpisodeTrace
from .common_state import Phase, phase_at, speed_at, state_at, validate_slowdown
from .domain import CommonState, ProtectionAction
from .hedge_simulation import (
    HedgeAttemptRecord,
    HedgeAttemptStatus,
    HedgeSimulationResult,
    HedgeTokenResult,
)
from .workload import (
    validate_hedge_stream_shape,
    validate_replay_stream_shape,
    validate_stream_values,
    validate_token_specs,
    validate_trace_shape,
)


class _Kind(IntEnum):
    STATE = 1
    COMPLETE = 2
    ARRIVAL = 4


@dataclass(frozen=True)
class _Event:
    time: float
    kind: _Kind
    sequence: int
    token_id: int | None = None
    replica_id: int | None = None
    generation: int = 0


@dataclass(frozen=True)
class _Pending:
    token_id: int
    attempt_id: int
    enqueue_time: float
    pinned_replica: int | None = None


@dataclass
class _Running:
    token_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    start_time: float
    required_work: float
    executed_work: float
    last_update: float
    speed: float


@dataclass
class _Token:
    arrival_time: float
    primary_replica: int | None = None
    replay_count: int = 0
    hedge_launched: bool = False
    winner: int | None = None
    completion_time: float | None = None
    records: list[HedgeAttemptRecord] = field(default_factory=list)


@dataclass(frozen=True)
class _IdleReleaseBoundaryAudit:
    """Online-only boundary state for the idle-release executor.

    ``BoundarySnapshot`` can represent work pinned to a Replica and work that
    is currently running, but it intentionally cannot assign an unbound
    waiting item to either Replica.  Keep that distinction in this private
    audit instead of reconstructing it from terminal attempt records.
    """

    snapshot: BoundarySnapshot
    running_attempt_keys: tuple[tuple[int, int, int], ...]
    pinned_queued_attempt_keys: tuple[tuple[int, int, int], ...]
    unassigned_waiting_attempt_keys: tuple[tuple[int, int], ...]

    @property
    def unassigned_waiting_attempts(self) -> int:
        return len(self.unassigned_waiting_attempt_keys)


class _LAEdgeEngine:
    def __init__(
        self,
        episode: EpisodeTrace,
        slowdown: float,
        *,
        cancel_running_losers: bool,
    ) -> None:
        self.episode = episode
        self.trace = episode.workload
        self.timeline = episode.protocol.timeline
        self.slowdown = slowdown
        if type(cancel_running_losers) is not bool:
            raise ValueError("cancel_running_losers must be bool")
        self.cancel_running_losers = cancel_running_losers
        self.events: list[tuple[float, int, int, _Event]] = []
        self.sequence = itertools.count()
        self.waiting: deque[_Pending] = deque()
        self.running: list[_Running | None] = [None, None]
        self.generations = [0, 0]
        self.tokens = [_Token(spec.arrival_time) for spec in self.trace.tokens]
        self.replay_executions = 0
        self.hedge_launches = 0
        self.completed_loser_total = 0
        self.failed_running_primaries = 0
        self.stale = 0
        self.boundary_audits: dict[str, _IdleReleaseBoundaryAudit] = {}

    def _push(self, event: _Event) -> None:
        heapq.heappush(
            self.events, (event.time, int(event.kind), event.sequence, event)
        )

    def _schedule(self) -> None:
        for moment in (
            self.timeline.degraded_start,
            self.timeline.failed_start,
            self.timeline.recovered_start,
        ):
            self._push(_Event(moment, _Kind.STATE, next(self.sequence)))
        for spec in self.trace.tokens:
            self._push(
                _Event(
                    spec.arrival_time,
                    _Kind.ARRIVAL,
                    next(self.sequence),
                    token_id=spec.token_id,
                )
            )

    def run(self) -> None:
        self._schedule()
        while self.events:
            first = heapq.heappop(self.events)
            batch = [first]
            while self.events and self.events[0][0] == first[0]:
                batch.append(heapq.heappop(self.events))
            batch.sort(key=lambda row: (row[1], row[2]))
            now = first[0]
            for _, _, _, event in batch:
                if event.kind is _Kind.STATE:
                    self._state_change(now)
                elif event.kind is _Kind.COMPLETE:
                    self._complete(event)
                else:
                    self.waiting.append(_Pending(event.token_id, 0, now))
            self._release(now)

    def _settle(self, running: _Running, now: float) -> None:
        running.executed_work = min(
            running.required_work,
            running.executed_work + (now - running.last_update) * running.speed,
        )
        running.last_update = now

    def _state_change(self, now: float) -> None:
        if now == self.timeline.degraded_start:
            running = self.running[0]
            if running is None:
                return
            self._settle(running, now)
            remaining = running.required_work - running.executed_work
            if remaining <= 0.0:
                return
            self.generations[0] += 1
            running.speed = speed_at(0, CommonState.DEGRADED, self.slowdown)
            self._schedule_completion(running, now + remaining / running.speed)
            return
        if now == self.timeline.failed_start:
            self.generations[0] += 1
            running = self.running[0]
            if running is not None:
                self._settle(running, now)
                self.running[0] = None
                token = self.tokens[running.token_id]
                self._record(running, HedgeAttemptStatus.FAILED_RUNNING, now)
                if running.attempt_id == 0:
                    self.failed_running_primaries += 1
                if token.winner is None and not self._live_replicas(running.token_id):
                    if token.replay_count:
                        raise RuntimeError("LÆDGE attempted more than one Replay")
                    token.replay_count = 1
                    self.waiting.append(_Pending(running.token_id, 1, now, 1))
            # Fault handling, including Replay admission, is complete before
            # the same-time completion/arrival/release batch is processed.
            self._capture_boundary("failed", now)
            return
        if now == self.timeline.recovered_start:
            # Recovery itself has no release side effect.  Capture before the
            # rest of this timestamp's batch can start waiting work on A.
            self._capture_boundary("recovered", now)

    def _capture_boundary(self, name: str, now: float) -> None:
        queued = [0, 0]
        live = [0, 0]
        remaining = [0.0, 0.0]
        running_keys: list[tuple[int, int, int]] = []
        pinned_keys: list[tuple[int, int, int]] = []
        unassigned_keys: list[tuple[int, int]] = []

        for replica, running in enumerate(self.running):
            if running is None:
                continue
            live[replica] += 1
            running_keys.append(
                (running.token_id, running.attempt_id, running.replica_id)
            )
            progressed = running.executed_work + (
                now - running.last_update
            ) * running.speed
            remaining[replica] += max(0.0, running.required_work - progressed)

        for pending in self.waiting:
            token = self.tokens[pending.token_id]
            if token.winner is not None:
                continue
            if pending.pinned_replica is None:
                unassigned_keys.append((pending.token_id, pending.attempt_id))
                continue
            replica = pending.pinned_replica
            queued[replica] += 1
            live[replica] += 1
            pinned_keys.append((pending.token_id, pending.attempt_id, replica))
            remaining[replica] += self._required(
                pending.attempt_id, replica, pending.token_id
            )

        snapshot = BoundarySnapshot(
            tuple(queued), tuple(live), tuple(remaining)
        )
        self.boundary_audits[name] = _IdleReleaseBoundaryAudit(
            snapshot=snapshot,
            running_attempt_keys=tuple(running_keys),
            pinned_queued_attempt_keys=tuple(pinned_keys),
            unassigned_waiting_attempt_keys=tuple(unassigned_keys),
        )

    def _available(self, replica: int, now: float) -> bool:
        return not (
            replica == 0 and state_at(self.timeline, now) is CommonState.FAILED
        )

    def _idle(self, now: float) -> tuple[int, ...]:
        return tuple(
            replica
            for replica in (0, 1)
            if self.running[replica] is None and self._available(replica, now)
        )

    def _live_replicas(self, token_id: int) -> tuple[int, ...]:
        return tuple(
            replica
            for replica, running in enumerate(self.running)
            if running is not None and running.token_id == token_id
        )

    def _release(self, now: float) -> None:
        while self.waiting and self._idle(now):
            idle = self._idle(now)
            selected = next(
                (
                    index
                    for index, item in enumerate(self.waiting)
                    if item.pinned_replica is None or item.pinned_replica in idle
                ),
                None,
            )
            if selected is None:
                break
            pending = self.waiting[selected]
            del self.waiting[selected]
            if self.tokens[pending.token_id].winner is not None:
                continue
            replica = (
                pending.pinned_replica
                if pending.pinned_replica is not None
                else idle[0]
            )
            self._start(pending, replica, now)
        while self._idle(now):
            candidates = [
                token_id
                for token_id, token in enumerate(self.tokens)
                if token.winner is None
                and not token.hedge_launched
                and len(self._live_replicas(token_id)) == 1
            ]
            if not candidates:
                break
            token_id = min(candidates, key=lambda value: (self.tokens[value].arrival_time, value))
            replica = self._idle(now)[0]
            if replica in self._live_replicas(token_id):
                break
            self.tokens[token_id].hedge_launched = True
            self.hedge_launches += 1
            self._start(_Pending(token_id, 2, now), replica, now)

    def _required(self, attempt_id: int, replica: int, token_id: int) -> float:
        streams = {
            0: self.trace.service_times,
            1: self.trace.replay_service_times,
            2: self.trace.hedge_service_times,
        }
        return streams[attempt_id][replica][token_id]

    def _start(self, pending: _Pending, replica: int, now: float) -> None:
        token = self.tokens[pending.token_id]
        if pending.attempt_id == 0 and token.primary_replica is None:
            token.primary_replica = replica
        if pending.attempt_id == 1:
            self.replay_executions += 1
        speed = speed_at(replica, state_at(self.timeline, now), self.slowdown)
        if speed <= 0.0:
            raise AssertionError("LÆDGE started work on an unavailable Replica")
        self.generations[replica] += 1
        running = _Running(
            token_id=pending.token_id,
            attempt_id=pending.attempt_id,
            replica_id=replica,
            enqueue_time=pending.enqueue_time,
            start_time=now,
            required_work=self._required(pending.attempt_id, replica, pending.token_id),
            executed_work=0.0,
            last_update=now,
            speed=speed,
        )
        self.running[replica] = running
        self._schedule_completion(
            running, now + running.required_work / running.speed
        )

    def _schedule_completion(self, running: _Running, when: float) -> None:
        self._push(
            _Event(
                when,
                _Kind.COMPLETE,
                next(self.sequence),
                token_id=running.token_id,
                replica_id=running.replica_id,
                generation=self.generations[running.replica_id],
            )
        )

    def _complete(self, event: _Event) -> None:
        replica = event.replica_id
        if event.generation != self.generations[replica]:
            self.stale += 1
            return
        running = self.running[replica]
        if running is None or running.token_id != event.token_id:
            self.stale += 1
            return
        now = event.time
        self._settle(running, now)
        running.executed_work = running.required_work
        self.running[replica] = None
        token = self.tokens[running.token_id]
        if token.winner is not None:
            self.running[replica] = None
            self._record(running, HedgeAttemptStatus.COMPLETED_LOSER, now)
            self.completed_loser_total += 1
            return
        token.winner = running.attempt_id
        token.completion_time = now
        self._record(running, HedgeAttemptStatus.COMPLETED_WINNER, now)
        for other_replica in self._live_replicas(running.token_id):
            loser = self.running[other_replica]
            if not self.cancel_running_losers:
                continue
            self._settle(loser, now)
            self.running[other_replica] = None
            self.generations[other_replica] += 1
            self._record(loser, HedgeAttemptStatus.CANCELLED_RUNNING, now)

    def _record(
        self, running: _Running, status: HedgeAttemptStatus, terminal_time: float
    ) -> None:
        executed = running.executed_work
        self.tokens[running.token_id].records.append(
            HedgeAttemptRecord(
                token_id=running.token_id,
                attempt_id=running.attempt_id,
                replica_id=running.replica_id,
                enqueue_time=running.enqueue_time,
                start_time=running.start_time,
                terminal_time=terminal_time,
                required_work=running.required_work,
                executed_work=executed,
                remaining_work=max(0.0, running.required_work - executed),
                status=status,
                queue_delay=running.start_time - running.enqueue_time,
            )
        )


def _snapshot(
    records: tuple[HedgeAttemptRecord, ...],
    boundary: float,
    episode: EpisodeTrace,
    slowdown: float,
    *,
    failed: bool,
) -> BoundarySnapshot:
    from .common_state import work_executed_between

    queued = [0, 0]
    live = [0, 0]
    remaining = [0.0, 0.0]
    for attempt in records:
        created = attempt.enqueue_time < boundary or (
            failed and attempt.attempt_id == 1 and attempt.enqueue_time == boundary
        )
        if not created or attempt.terminal_time < boundary:
            continue
        if failed and attempt.terminal_time == boundary and attempt.status is HedgeAttemptStatus.FAILED_RUNNING:
            continue
        replica = attempt.replica_id
        live[replica] += 1
        if attempt.start_time is None or attempt.start_time >= boundary:
            queued[replica] += 1
            remaining[replica] += attempt.required_work
        else:
            used = work_executed_between(
                replica, attempt.start_time, boundary, episode.protocol.timeline, slowdown
            )
            remaining[replica] += max(0.0, attempt.required_work - used)
    return BoundarySnapshot(tuple(queued), tuple(live), tuple(remaining))


def simulate_laedge_episode(
    episode: EpisodeTrace,
    *,
    degraded_slowdown: float = 2.0,
    cancel_running_losers: bool = True,
) -> EpisodeSimulationResult:
    """Run generalized LÆDGE on one immutable two-Replica episode."""

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    slowdown = validate_slowdown(degraded_slowdown)
    trace = episode.workload
    validate_trace_shape(trace, 2)
    validate_replay_stream_shape(trace, 2)
    validate_hedge_stream_shape(trace, 2)
    validate_token_specs(trace)
    validate_stream_values(trace.service_times, 0, "service")
    validate_stream_values(trace.replay_service_times, 1, "replay")
    validate_stream_values(trace.hedge_service_times, 2, "hedge")
    engine = _LAEdgeEngine(
        episode,
        slowdown,
        cancel_running_losers=cancel_running_losers,
    )
    engine.run()
    token_rows = []
    for spec, state in zip(trace.tokens, engine.tokens):
        if state.completion_time is None or state.winner is None or state.primary_replica is None:
            raise RuntimeError(f"LÆDGE terminated with Token {spec.token_id} incomplete")
        attempts = tuple(sorted(state.records, key=lambda row: row.attempt_id))
        token_rows.append(
            HedgeTokenResult(
                token_id=spec.token_id,
                token_class=spec.token_class,
                arrival_time=spec.arrival_time,
                primary_replica=state.primary_replica,
                action=(
                    ProtectionAction.IMMEDIATE_HEDGE
                    if state.hedge_launched
                    else ProtectionAction.NORMAL
                ),
                attempts=attempts,
                winner_attempt_id=state.winner,
                completion_time=state.completion_time,
                completion_phase=phase_at(episode.protocol.timeline, state.completion_time),
                replay_count=state.replay_count,
                latency=state.completion_time - spec.arrival_time,
            )
        )
    tokens = tuple(token_rows)
    attempts = tuple(row for token in tokens for row in token.attempts)
    drain_end = max(row.terminal_time for row in attempts)
    last_arrival = max(spec.arrival_time for spec in trace.tokens)
    result = HedgeSimulationResult(
        tokens=tokens,
        attempts=attempts,
        primary_executions=len(tokens),
        replay_executions=engine.replay_executions,
        hedge_requested=engine.hedge_launches,
        hedge_launches=engine.hedge_launches,
        hedge_suppressed=0,
        hedge_timers_voided=0,
        cancelled_queued_total=0,
        completed_loser_total=engine.completed_loser_total,
        failed_running_primary_executions=engine.failed_running_primaries,
        invalidated_queued_primary_executions=0,
        last_arrival_time=last_arrival,
        drain_end_time=drain_end,
        drain_duration=drain_end - last_arrival,
        stale_completion_events_ignored=engine.stale,
        queue_length_at_failed_start=(0, 0),
        queue_length_at_recovered_start=(0, 0),
    )
    try:
        failed = engine.boundary_audits["failed"].snapshot
        recovered = engine.boundary_audits["recovered"].snapshot
    except KeyError as exc:
        raise RuntimeError("LÆDGE did not capture both boundary states") from exc
    result = EpisodeSimulationResult(
        episode=episode,
        simulation=result,
        failed_start_snapshot=failed,
        recovered_start_snapshot=recovered,
        post_cutoff_drain_duration=max(
            0.0, result.drain_end_time - episode.protocol.arrival_cutoff
        ),
        degraded_slowdown=slowdown,
        hedge_delay=None,
    )
    # Keep the richer distinction private to idle-release audit consumers; the
    # historical EpisodeSimulationResult schema remains unchanged.
    object.__setattr__(result, "_idle_release_boundary_audit", dict(engine.boundary_audits))
    return result


__all__ = ["simulate_laedge_episode"]
