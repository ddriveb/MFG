"""Isolated simultaneous-batch finite Token routing physics.

This module is the first implementation slice of the reliability-aware Token
MFG design.  It deliberately does not estimate a population law or solve a
response problem.  It only enforces the information boundary at a batch
decision: all new/Replay Tokens receive one immutable pre-batch public state,
choose without seeing same-batch actions, and commit their single Primary
assignments atomically.
"""

from __future__ import annotations

# 阅读提示：批次内每个 Token 看见同一份 pre-batch public state；不能把第一个
# Token 的选择立即写回队列，再让后续 Token 观察到它。那会把同时博弈变成顺序调度。
# 本模块仅定义物理执行；population 估计、响应求解与 qualification 在独立模块中。

from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Callable, Protocol, Sequence

from .reliability_aware_routing import (
    REPLICA_IDS,
    AttemptResult,
    ReplicaObservation,
    RoutingResult,
    RoutingToken,
    RoutingTrace,
    StartRecord,
    TokenResult,
    _result_metrics,
    _risk_observations,
    _state_at,
)
from .topology import DEFAULT_TOPOLOGY, Topology


class SimultaneousRoutingError(ValueError):
    """Raised when a simultaneous finite-routing contract is violated."""


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SimultaneousRoutingError("value is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SimultaneousRoutingError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimultaneousRoutingError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise SimultaneousRoutingError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PreBatchPublicState:
    """One immutable public state shared by every decision in a batch."""

    time: float
    health: tuple[bool, ...]
    queue_depths: tuple[int, ...]
    running_ages: tuple[float | None, ...]
    replica_observations: tuple[ReplicaObservation, ...]
    published_realized_share: tuple[tuple[int, float], ...]
    fingerprint: str
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        _finite(self.time, "public state time")
        if self.health != tuple(bool(value) for value in self.health):
            raise SimultaneousRoutingError("health must be a boolean tuple")
        if not isinstance(self.topology, Topology):
            raise SimultaneousRoutingError("topology must be Topology")
        replica_count = len(self.topology.replica_ids)
        if len(self.health) != replica_count:
            raise SimultaneousRoutingError("health must cover every Replica")
        if len(self.queue_depths) != replica_count or any(
            type(value) is not int or value < 0 for value in self.queue_depths
        ):
            raise SimultaneousRoutingError("queue_depths must be non-negative ints")
        if len(self.running_ages) != replica_count:
            raise SimultaneousRoutingError("running_ages must cover every Replica")
        for value in self.running_ages:
            if value is not None:
                _finite(value, "running age")
                if value < 0.0:
                    raise SimultaneousRoutingError("running age must be non-negative")
        if len(self.replica_observations) != replica_count:
            raise SimultaneousRoutingError("replica observations must cover every Replica")
        if not isinstance(self.published_realized_share, tuple):
            raise SimultaneousRoutingError("published share must be a tuple")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64:
            raise SimultaneousRoutingError("public-state fingerprint must be SHA-256")


@dataclass(frozen=True)
class TokenDecisionObservation:
    """A Token-local observation plus the shared pre-batch public state."""

    token_id: int
    token_class: str
    arrival_time: float
    deadline: float
    ingress_rank: int
    current_age: float
    retry_count: int
    public_state: PreBatchPublicState

    def __post_init__(self) -> None:
        _strict_int(self.token_id, "token_id")
        if self.token_class not in {"Regular", "Urgent"}:
            raise SimultaneousRoutingError("token_class is invalid")
        _finite(self.arrival_time, "arrival_time")
        deadline = _finite(self.deadline, "deadline")
        if deadline <= 0.0:
            raise SimultaneousRoutingError("deadline must be positive")
        _strict_int(self.ingress_rank, "ingress_rank")
        _finite(self.current_age, "current_age")
        if self.current_age < 0.0:
            raise SimultaneousRoutingError("current_age must be non-negative")
        _strict_int(self.retry_count, "retry_count")
        if not isinstance(self.public_state, PreBatchPublicState):
            raise SimultaneousRoutingError("public_state is invalid")


@dataclass(frozen=True)
class ActiveTokenState:
    """Observable state of one arrived, nonterminal Token at a batch boundary."""

    token_id: int
    token_class: str
    arrival_time: float
    deadline: float
    ingress_rank: int
    current_age: float
    retry_count: int
    status: str
    attempt_id: int | None
    replica_id: int | None
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        _strict_int(self.token_id, "active token_id")
        if self.token_class not in {"Regular", "Urgent"}:
            raise SimultaneousRoutingError("active token_class is invalid")
        _finite(self.arrival_time, "active arrival_time")
        deadline = _finite(self.deadline, "active deadline")
        if deadline <= 0.0:
            raise SimultaneousRoutingError("active deadline must be positive")
        _strict_int(self.ingress_rank, "active ingress_rank")
        _finite(self.current_age, "active current_age")
        if self.current_age < 0.0:
            raise SimultaneousRoutingError("active current_age must be non-negative")
        _strict_int(self.retry_count, "active retry_count")
        if not isinstance(self.topology, Topology):
            raise SimultaneousRoutingError("active topology is invalid")
        if self.status not in {"waiting", "queued", "running", "decision"}:
            raise SimultaneousRoutingError("active Token status is invalid")
        if self.status in {"waiting", "decision"}:
            if self.attempt_id is not None or self.replica_id is not None:
                raise SimultaneousRoutingError(
                    "unassigned active Token cannot have an attempt"
                )
        else:
            _strict_int(self.attempt_id, "active attempt_id")
            if type(self.replica_id) is not int or self.replica_id not in self.topology.replica_ids:
                raise SimultaneousRoutingError("active replica_id is invalid")


@dataclass(frozen=True)
class PopulationDecisionContext:
    """Causal population context published before a decision cohort commits."""

    active_token_count: int
    replica_state_fingerprint: str
    cohort_token_count: int
    cohort_class_counts: tuple[tuple[str, int], ...]
    active_token_class_counts: tuple[tuple[str, int], ...]
    predicted_share: tuple[tuple[int, float], ...]
    published_realized_share: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        _strict_int(self.active_token_count, "active_token_count")
        if not isinstance(self.replica_state_fingerprint, str) or len(
            self.replica_state_fingerprint
        ) != 64:
            raise SimultaneousRoutingError("replica-state fingerprint must be SHA-256")
        _strict_int(self.cohort_token_count, "cohort_token_count")
        if not isinstance(self.cohort_class_counts, tuple):
            raise SimultaneousRoutingError("cohort class counts must be a tuple")
        for token_class, count in self.cohort_class_counts:
            if token_class not in {"Regular", "Urgent"}:
                raise SimultaneousRoutingError("invalid cohort Token class")
            _strict_int(count, "cohort class count")
        if not isinstance(self.active_token_class_counts, tuple):
            raise SimultaneousRoutingError("active Token class counts must be a tuple")
        for token_class, count in self.active_token_class_counts:
            if token_class not in {"Regular", "Urgent"}:
                raise SimultaneousRoutingError("invalid active Token class")
            _strict_int(count, "active Token class count")
        if not isinstance(self.predicted_share, tuple):
            raise SimultaneousRoutingError("predicted share must be a tuple")
        if not isinstance(self.published_realized_share, tuple):
            raise SimultaneousRoutingError("published realized share must be a tuple")


class SimultaneousPolicy(Protocol):
    """Online policy contract for one Token decision."""

    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        """Return one eligible same-Expert Replica ID or None to keep waiting."""


@dataclass(frozen=True)
class SimultaneousBatchAudit:
    """Audit of one atomic decision cohort and its pre/post queue state."""

    batch_id: int
    time: float
    token_ids: tuple[int, ...]
    public_state_fingerprint: str
    pre_batch_queue_depths: tuple[int, ...]
    assigned_replica_ids: tuple[int | None, ...]
    queue_depths_after_commit: tuple[int, ...]
    realized_share: tuple[tuple[int, float], ...]
    active_token_states: tuple[ActiveTokenState, ...]
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        _strict_int(self.batch_id, "batch_id")
        _finite(self.time, "batch time")
        if not isinstance(self.token_ids, tuple) or len(set(self.token_ids)) != len(self.token_ids):
            raise SimultaneousRoutingError("batch token IDs must be unique")
        if len(self.token_ids) != len(self.assigned_replica_ids):
            raise SimultaneousRoutingError("assignment audit length mismatch")
        if not isinstance(self.topology, Topology):
            raise SimultaneousRoutingError("topology must be Topology")
        replica_count = len(self.topology.replica_ids)
        if len(self.pre_batch_queue_depths) != replica_count or len(
            self.queue_depths_after_commit
        ) != replica_count:
            raise SimultaneousRoutingError("queue audit must cover every Replica")
        if any(
            type(value) is not int or value < 0
            for value in (*self.pre_batch_queue_depths, *self.queue_depths_after_commit)
        ):
            raise SimultaneousRoutingError("queue audit depths must be non-negative ints")
        if not isinstance(self.public_state_fingerprint, str) or len(
            self.public_state_fingerprint
        ) != 64:
            raise SimultaneousRoutingError("batch public fingerprint must be SHA-256")
        if not isinstance(self.active_token_states, tuple):
            raise SimultaneousRoutingError("active Token states must be immutable")
        active_ids = tuple(row.token_id for row in self.active_token_states)
        if len(set(active_ids)) != len(active_ids):
            raise SimultaneousRoutingError("active Token IDs must be unique")
        if active_ids != tuple(
            row.token_id
            for row in sorted(
                self.active_token_states,
                key=lambda row: (row.arrival_time, row.token_id),
            )
        ):
            raise SimultaneousRoutingError("active Token states must be canonical")


@dataclass(frozen=True)
class SimultaneousRoutingResult:
    """Physical result plus auditable simultaneous decision batches."""

    physical: RoutingResult
    batch_audits: tuple[SimultaneousBatchAudit, ...]


@dataclass
class _MutableAttempt:
    token_id: int
    attempt_id: int
    replica_id: int
    required_work: float
    executed_work: float = 0.0
    start_time: float | None = None
    terminal_time: float | None = None
    terminal_status: str = "queued"
    lost_work: float = 0.0


@dataclass
class EngineSnapshot:
    """Deep copy of every mutable engine field at one batch boundary.

    Hazard/history estimates are NOT stored: they are prefix-recomputable
    from the exogenous trace at any time, so restoring ``event_index`` and
    ``time`` suffices.
    """

    time: float
    arrival_index: int
    event_index: int
    batch_id: int
    new_ids: tuple[int, ...]
    health: dict
    queues: dict
    running: dict
    attempts: dict
    replay_counts: dict
    waiting: deque
    starts: list
    batch_audits: list
    total_work: float
    lost_work: float
    no_eligible_time: float
    down_starts: int
    max_live: int
    terminal: set
    seen_token_ids: set
    published_realized_share: tuple
    replica_audit: dict


def _copy_attempt_map(attempts, running):
    """Deep-copy attempts once so ``attempts`` and ``running`` stay consistent."""

    copied: dict[int, _MutableAttempt] = {}

    def copy_one(attempt):
        key = id(attempt)
        if key not in copied:
            copied[key] = _MutableAttempt(
                attempt.token_id,
                attempt.attempt_id,
                attempt.replica_id,
                attempt.required_work,
                attempt.executed_work,
                attempt.start_time,
                attempt.terminal_time,
                attempt.terminal_status,
                attempt.lost_work,
            )
        return copied[key]

    attempts_copy = {
        token_id: [copy_one(row) for row in rows]
        for token_id, rows in attempts.items()
    }
    running_copy = {
        replica_id: copy_one(attempt)
        for replica_id, attempt in running.items()
    }
    return attempts_copy, running_copy


def _public_state(
    trace: RoutingTrace,
    now: float,
    health: dict[int, bool],
    queues: dict[int, deque[int]],
    running: dict[int, _MutableAttempt],
    published_realized_share: tuple[tuple[int, float], ...],
    history_window: float,
) -> PreBatchPublicState:
    replica_ids = trace.replica_ids
    observations = _risk_observations(
        trace,
        now,
        health,
        queues,
        {replica_id: attempt.token_id for replica_id, attempt in running.items()},
        {
            replica_id: attempt.start_time if attempt.start_time is not None else now
            for replica_id, attempt in running.items()
        },
        history_window=history_window,
        include_arrival_service=True,
    )
    queue_depths = tuple(row.queue_depth for row in observations)
    running_ages = tuple(row.running_age for row in observations)
    fingerprint = _digest(
        {
            "time": now,
            "health": tuple(health[replica_id] for replica_id in replica_ids),
            "queue_depths": queue_depths,
            "running_ages": running_ages,
            "replicas": [
                {
                    "replica_id": row.replica_id,
                    "domain_id": row.domain_id,
                    "available": row.available,
                    "queue_depth": row.queue_depth,
                    "running_age": row.running_age,
                    "estimated_work": row.estimated_work,
                    "estimated_hazard": row.estimated_hazard,
                    "estimated_failure_risk": row.estimated_failure_risk,
                }
                for row in observations
            ],
            "published_realized_share": published_realized_share,
        }
    )
    return PreBatchPublicState(
        time=now,
        health=tuple(health[replica_id] for replica_id in replica_ids),
        queue_depths=queue_depths,
        running_ages=running_ages,
        replica_observations=observations,
        published_realized_share=published_realized_share,
        fingerprint=fingerprint,
        topology=trace.topology,
    )


def _population_context(
    active_token_count: int,
    active_token_ids: set[int],
    public_state: PreBatchPublicState,
    tokens: Sequence[RoutingToken],
    cohort_ids: Sequence[int],
    predicted_share: tuple[tuple[int, float], ...],
) -> PopulationDecisionContext:
    counts = {
        token_class: sum(
            1 for token_id in cohort_ids
            if next(token for token in tokens if token.token_id == token_id).token_class
            == token_class
        )
        for token_class in ("Regular", "Urgent")
    }
    active_counts = {
        token_class: sum(
            1 for token in tokens
            if token.token_id in active_token_ids and token.token_class == token_class
        )
        for token_class in ("Regular", "Urgent")
    }
    replica_payload = tuple(
        (
            row.replica_id,
            row.domain_id,
            row.available,
            row.queue_depth,
            row.running_age,
            row.estimated_work,
            row.estimated_hazard,
            row.estimated_failure_risk,
        )
        for row in public_state.replica_observations
    )
    return PopulationDecisionContext(
        active_token_count=active_token_count,
        replica_state_fingerprint=_digest(replica_payload),
        cohort_token_count=len(cohort_ids),
        cohort_class_counts=tuple((token_class, counts[token_class]) for token_class in ("Regular", "Urgent")),
        active_token_class_counts=tuple(
            (token_class, active_counts[token_class])
            for token_class in ("Regular", "Urgent")
        ),
        predicted_share=predicted_share,
        published_realized_share=public_state.published_realized_share,
    )


def _validate_trace(trace: RoutingTrace) -> dict[int, RoutingToken]:
    if not isinstance(trace, RoutingTrace):
        raise SimultaneousRoutingError("trace must be RoutingTrace")
    token_by_id = {token.token_id: token for token in trace.tokens}
    if len(token_by_id) != len(trace.tokens):
        raise SimultaneousRoutingError("trace Token IDs must be unique")
    if tuple(sorted(trace.tokens, key=lambda token: (token.arrival_time, token.token_id))) != trace.tokens:
        raise SimultaneousRoutingError("trace Tokens must be in canonical arrival order")
    return token_by_id


def _validate_policy(policy: object) -> None:
    if not callable(getattr(policy, "choose", None)):
        raise SimultaneousRoutingError("policy must provide online choose")


def _validated_choice(
    choice: object,
    public_state: PreBatchPublicState,
) -> int | None:
    if choice is None:
        return None
    if type(choice) is not int or choice not in public_state.topology.replica_ids:
        raise SimultaneousRoutingError("policy must return a true Replica ID or None")
    if not public_state.health[choice]:
        raise SimultaneousRoutingError("policy selected a DOWN Replica")
    return choice


def simulate_simultaneous_routing(
    trace: RoutingTrace,
    policy: SimultaneousPolicy,
    *,
    policy_name: str = "simultaneous_custom",
    history_window: float = 50.0,
    snapshot_recorder: Callable[[int, EngineSnapshot], None] | None = None,
    fork_snapshot: EngineSnapshot | None = None,
) -> SimultaneousRoutingResult:
    """Run complete single-copy physics with atomic same-time decisions.

    ``snapshot_recorder`` receives one deep-copied EngineSnapshot at every
    non-empty decision cohort (keyed by the batch ID the cohort will get).
    ``fork_snapshot`` restores a previously recorded boundary and re-executes
    only its pending cohort onward, which is bit-identical to a full replay
    when the policy is a pure function of (observation, context, key).
    """

    token_by_id = _validate_trace(trace)
    topology = trace.topology
    replica_ids = topology.replica_ids
    _validate_policy(policy)
    if not isinstance(policy_name, str) or not policy_name:
        raise SimultaneousRoutingError("policy_name must be non-empty")
    history_window = _finite(history_window, "history_window")
    if history_window <= 0.0:
        raise SimultaneousRoutingError("history_window must be positive")
    if snapshot_recorder is not None and not callable(snapshot_recorder):
        raise SimultaneousRoutingError("snapshot_recorder must be callable")
    if fork_snapshot is not None and not isinstance(fork_snapshot, EngineSnapshot):
        raise SimultaneousRoutingError("fork_snapshot must be an EngineSnapshot")

    if fork_snapshot is None:
        health = {
            replica_id: (
                _state_at(trace.health_events, "replica", replica_id, 0.0)
                and _state_at(trace.health_events, "domain", topology.domain_of(replica_id), 0.0)
            )
            for replica_id in replica_ids
        }
        queues: dict[int, deque[int]] = {replica_id: deque() for replica_id in replica_ids}
        running: dict[int, _MutableAttempt] = {}
        attempts: dict[int, list[_MutableAttempt]] = {token_id: [] for token_id in token_by_id}
        replay_counts = {token_id: 0 for token_id in token_by_id}
        waiting: deque[int] = deque()
        starts: list[StartRecord] = []
        batch_audits: list[SimultaneousBatchAudit] = []
        total_work = 0.0
        lost_work = 0.0
        no_eligible_time = 0.0
        down_starts = 0
        max_live = 0
        current_time = 0.0
        arrival_index = 0
        event_index = next(
            (index for index, event in enumerate(trace.health_events) if event.time >= 0.0),
            len(trace.health_events),
        )
        terminal: set[int] = set()
        seen_token_ids: set[int] = set()
        published_realized_share: tuple[tuple[int, float], ...] = ()
        batch_id = 0
        replica_audit: dict[int, dict[str, object]] = {
            replica_id: {
                "assignments": 0,
                "starts": 0,
                "completions": 0,
                "running_failures": 0,
                "queued_displacements": 0,
                "peak_queue_depth": 0,
                "busy_time": 0.0,
                "available_time": 0.0,
                "hazard_samples": [],
            }
            for replica_id in replica_ids
        }
    else:
        health = dict(fork_snapshot.health)
        queues = {
            replica_id: deque(fork_snapshot.queues[replica_id])
            for replica_id in replica_ids
        }
        attempts, running = _copy_attempt_map(
            fork_snapshot.attempts, fork_snapshot.running
        )
        replay_counts = dict(fork_snapshot.replay_counts)
        waiting = deque(fork_snapshot.waiting)
        starts = list(fork_snapshot.starts)
        batch_audits = list(fork_snapshot.batch_audits)
        total_work = fork_snapshot.total_work
        lost_work = fork_snapshot.lost_work
        no_eligible_time = fork_snapshot.no_eligible_time
        down_starts = fork_snapshot.down_starts
        max_live = fork_snapshot.max_live
        current_time = fork_snapshot.time
        arrival_index = fork_snapshot.arrival_index
        event_index = fork_snapshot.event_index
        terminal = set(fork_snapshot.terminal)
        seen_token_ids = set(fork_snapshot.seen_token_ids)
        published_realized_share = fork_snapshot.published_realized_share
        batch_id = fork_snapshot.batch_id
        replica_audit = copy.deepcopy(fork_snapshot.replica_audit)

    def add_waiting(token_id: int, *, front: bool = False) -> None:
        if front:
            waiting.appendleft(token_id)
        else:
            waiting.append(token_id)

    def capture_state(new_ids: Sequence[int], time: float) -> EngineSnapshot:
        attempts_copy, running_copy = _copy_attempt_map(attempts, running)
        return EngineSnapshot(
            time=time,
            arrival_index=arrival_index,
            event_index=event_index,
            batch_id=batch_id,
            new_ids=tuple(new_ids),
            health=dict(health),
            queues={
                replica_id: deque(queues[replica_id]) for replica_id in replica_ids
            },
            running=running_copy,
            attempts=attempts_copy,
            replay_counts=dict(replay_counts),
            waiting=deque(waiting),
            starts=list(starts),
            batch_audits=list(batch_audits),
            total_work=total_work,
            lost_work=lost_work,
            no_eligible_time=no_eligible_time,
            down_starts=down_starts,
            max_live=max_live,
            terminal=set(terminal),
            seen_token_ids=set(seen_token_ids),
            published_realized_share=published_realized_share,
            replica_audit=copy.deepcopy(replica_audit),
        )

    def invalidate_failed_replicas(failed_replicas: set[int], time: float) -> None:
        nonlocal lost_work
        displaced_running: list[int] = []
        displaced_queued: list[int] = []
        for replica_id in sorted(failed_replicas):
            if replica_id in running:
                attempt = running.pop(replica_id)
                attempt.terminal_time = time
                attempt.terminal_status = "failed"
                attempt.lost_work = attempt.executed_work
                lost_work += attempt.executed_work
                replay_counts[attempt.token_id] += 1
                replica_audit[replica_id]["running_failures"] += 1
                displaced_running.append(attempt.token_id)
            if queues[replica_id]:
                queued_tokens = list(queues[replica_id])
                for token_id in queued_tokens:
                    queued_attempt = next(
                        row
                        for row in attempts[token_id]
                        if row.replica_id == replica_id and row.terminal_status == "queued"
                    )
                    queued_attempt.terminal_time = time
                    queued_attempt.terminal_status = "discarded"
                    replica_audit[replica_id]["queued_displacements"] += 1
                displaced_queued.extend(queued_tokens)
                queues[replica_id].clear()
        for token_id in sorted(
            displaced_running,
            key=lambda current: (token_by_id[current].arrival_time, current),
        ):
            add_waiting(token_id)
        for token_id in sorted(
            displaced_queued,
            key=lambda current: (token_by_id[current].arrival_time, current),
        ):
            add_waiting(token_id)

    def start_available(time: float) -> None:
        nonlocal down_starts
        for replica_id in replica_ids:
            if not health[replica_id] or replica_id in running or not queues[replica_id]:
                continue
            token_id = queues[replica_id].popleft()
            attempt = next(
                row
                for row in attempts[token_id]
                if row.replica_id == replica_id and row.terminal_status == "queued"
            )
            attempt.start_time = time
            attempt.terminal_status = "running"
            running[replica_id] = attempt
            starts.append(StartRecord(token_id, attempt.attempt_id, replica_id, time))
            replica_audit[replica_id]["starts"] += 1
            if not health[replica_id]:
                down_starts += 1

    def active_token_snapshot(time: float) -> tuple[ActiveTokenState, ...]:
        """Capture only arrived, nonterminal Token state before commit."""

        waiting_ids = set(waiting)
        queued_by_token: dict[int, tuple[int, _MutableAttempt]] = {}
        for replica_id in replica_ids:
            for token_id in queues[replica_id]:
                queued_attempt = next(
                    row
                    for row in attempts[token_id]
                    if row.replica_id == replica_id
                    and row.terminal_status == "queued"
                )
                queued_by_token[token_id] = (replica_id, queued_attempt)
        running_by_token = {
            attempt.token_id: (replica_id, attempt)
            for replica_id, attempt in running.items()
        }
        rows: list[ActiveTokenState] = []
        active_ids = seen_token_ids - terminal
        for token in sorted(
            (token_by_id[token_id] for token_id in active_ids),
            key=lambda row: (row.arrival_time, row.token_id),
        ):
            if token.token_id in waiting_ids:
                status = "waiting"
                attempt_id = None
                replica_id = None
            elif token.token_id in running_by_token:
                replica_id, attempt = running_by_token[token.token_id]
                status = "running"
                attempt_id = attempt.attempt_id
            elif token.token_id in queued_by_token:
                replica_id, attempt = queued_by_token[token.token_id]
                status = "queued"
                attempt_id = attempt.attempt_id
            else:
                status = "decision"
                attempt_id = None
                replica_id = None
            rows.append(
                ActiveTokenState(
                    token.token_id,
                    token.token_class,
                    token.arrival_time,
                    token.deadline,
                    token.ingress_rank,
                    time - token.arrival_time,
                    replay_counts[token.token_id],
                    status,
                    attempt_id,
                    replica_id,
                    topology,
                )
            )
        return tuple(rows)

    def commit_decision_cohort(time: float, new_token_ids: Sequence[int]) -> None:
        nonlocal batch_id, published_realized_share
        cohort_ids = tuple(waiting) + tuple(new_token_ids)
        if not cohort_ids:
            start_available(time)
            return
        if snapshot_recorder is not None:
            snapshot_recorder(batch_id, capture_state(new_token_ids, time))
        active_token_states = active_token_snapshot(time)
        waiting.clear()
        public_state = _public_state(
            trace,
            time,
            health,
            queues,
            running,
            published_realized_share,
            history_window,
        )
        context = _population_context(
            len(seen_token_ids - terminal),
            seen_token_ids - terminal,
            public_state,
            trace.tokens,
            cohort_ids,
            published_realized_share,
        )
        choices: list[int | None] = []
        for token_id in cohort_ids:
            token = token_by_id[token_id]
            observation = TokenDecisionObservation(
                token_id=token.token_id,
                token_class=token.token_class,
                arrival_time=token.arrival_time,
                deadline=token.deadline,
                ingress_rank=token.ingress_rank,
                current_age=time - token.arrival_time,
                retry_count=replay_counts[token_id],
                public_state=public_state,
            )
            policy_key = (
                f"{trace.namespace}|seed={trace.macro_seed}|scenario={trace.scenario}"
                f"|episode={trace.episode_index}|batch={batch_id}|token={token_id}"
                "|action=base|bucket=base"
            )
            choice = _validated_choice(
                policy.choose(observation, context, policy_key),
                public_state,
            )
            choices.append(choice)

        assigned_counts = {replica_id: 0 for replica_id in replica_ids}
        for token_id, choice in zip(cohort_ids, choices):
            if choice is None:
                waiting.append(token_id)
                continue
            attempt_id = len(attempts[token_id])
            attempts[token_id].append(
                _MutableAttempt(
                    token_id,
                    attempt_id,
                    choice,
                    trace.work_for(token_id, choice, attempt_id),
                )
            )
            queues[choice].append(token_id)
            assigned_counts[choice] += 1
            replica_audit[choice]["assignments"] += 1
        queue_depths_after_commit = tuple(
            len(queues[replica_id]) + (1 if replica_id in running else 0)
            for replica_id in replica_ids
        )
        for replica_id in replica_ids:
            replica_audit[replica_id]["peak_queue_depth"] = max(
                int(replica_audit[replica_id]["peak_queue_depth"]),
                queue_depths_after_commit[replica_id],
            )
        assigned_total = sum(assigned_counts.values())
        realized = (
            tuple(
                (replica_id, assigned_counts[replica_id] / assigned_total)
                for replica_id in replica_ids
                if assigned_counts[replica_id]
            )
            if assigned_total
            else ()
        )
        batch_audits.append(
            SimultaneousBatchAudit(
                batch_id=batch_id,
                time=time,
                token_ids=tuple(cohort_ids),
                public_state_fingerprint=public_state.fingerprint,
                pre_batch_queue_depths=public_state.queue_depths,
                assigned_replica_ids=tuple(choices),
                queue_depths_after_commit=queue_depths_after_commit,
                realized_share=realized,
                active_token_states=active_token_states,
                topology=topology,
            )
        )
        published_realized_share = realized
        batch_id += 1
        start_available(time)

    def advance(to_time: float) -> None:
        nonlocal total_work, current_time, no_eligible_time
        dt = to_time - current_time
        if dt < -1e-12:
            raise SimultaneousRoutingError("event time moved backwards")
        for replica_id, attempt in running.items():
            increment = min(attempt.required_work - attempt.executed_work, dt)
            increment = max(0.0, increment)
            attempt.executed_work += increment
            total_work += increment
        for replica_id in replica_ids:
            replica_audit[replica_id]["available_time"] += (
                dt if health[replica_id] else 0.0
            )
            replica_audit[replica_id]["busy_time"] += (
                dt if replica_id in running else 0.0
            )
        if waiting and not any(health.values()):
            no_eligible_time += dt
        current_time = to_time

    if fork_snapshot is not None:
        commit_decision_cohort(fork_snapshot.time, list(fork_snapshot.new_ids))

    while True:
        next_arrival = (
            trace.tokens[arrival_index].arrival_time
            if arrival_index < len(trace.tokens)
            else math.inf
        )
        next_health = (
            trace.health_events[event_index].time
            if event_index < len(trace.health_events)
            else math.inf
        )
        next_completion = math.inf
        for attempt in running.values():
            remaining = max(0.0, attempt.required_work - attempt.executed_work)
            next_completion = min(next_completion, current_time + remaining)
        next_time = min(next_arrival, next_health, next_completion)
        if math.isinf(next_time):
            if not waiting and not running and arrival_index >= len(trace.tokens):
                break
            raise SimultaneousRoutingError("finite fault horizon did not drain global queue")
        advance(next_time)
        same_events = []
        while (
            event_index < len(trace.health_events)
            and abs(trace.health_events[event_index].time - current_time) <= 1e-12
        ):
            if trace.health_events[event_index].time >= 0.0:
                same_events.append(trace.health_events[event_index])
            event_index += 1
        old_health = dict(health)
        for event in same_events:
            if event.kind == "recovery":
                if event.scope == "replica":
                    health[event.component_id] = True
                else:
                    for replica_id in replica_ids:
                        if topology.domain_of(replica_id) == event.component_id:
                            health[replica_id] = True
        for event in same_events:
            if event.kind == "failure":
                if event.scope == "replica":
                    health[event.component_id] = False
                else:
                    for replica_id in replica_ids:
                        if topology.domain_of(replica_id) == event.component_id:
                            health[replica_id] = False
        failed = {
            replica_id
            for replica_id in replica_ids
            if old_health[replica_id] and not health[replica_id]
        }
        invalidate_failed_replicas(failed, current_time)
        completed_replicas = [
            replica_id
            for replica_id, attempt in running.items()
            if attempt.required_work - attempt.executed_work <= 1e-12
        ]
        for replica_id in sorted(completed_replicas):
            attempt = running.pop(replica_id)
            attempt.executed_work = attempt.required_work
            attempt.terminal_time = current_time
            attempt.terminal_status = "completed"
            replica_audit[replica_id]["completions"] += 1
            terminal.add(attempt.token_id)
        new_ids: list[int] = []
        while (
            arrival_index < len(trace.tokens)
            and abs(trace.tokens[arrival_index].arrival_time - current_time) <= 1e-12
        ):
            new_ids.append(trace.tokens[arrival_index].token_id)
            arrival_index += 1
        seen_token_ids.update(new_ids)
        commit_decision_cohort(current_time, new_ids)
        max_live = max(max_live, 1 if running else 0)

    token_results: list[TokenResult] = []
    for token in sorted(trace.tokens, key=lambda row: row.token_id):
        rows = tuple(
            AttemptResult(
                row.token_id,
                row.attempt_id,
                row.replica_id,
                row.start_time,
                row.terminal_time,
                row.terminal_status,
                row.required_work,
                row.executed_work,
                row.lost_work,
            )
            for row in attempts[token.token_id]
        )
        completed_rows = tuple(row for row in rows if row.terminal_status == "completed")
        winner = completed_rows[-1].replica_id if completed_rows else None
        completion = completed_rows[-1].terminal_time if completed_rows else None
        token_results.append(
            TokenResult(
                token.token_id,
                rows,
                replay_counts[token.token_id],
                winner,
                completion,
                token.token_class,
            )
        )
    invariants = {
        "down_starts": down_starts,
        "max_live_attempts_per_token": max_live,
        "future_work_observations": 0,
        "completed_without_winner": sum(
            row.winner_replica_id is None for row in token_results
        ),
        "duplicate_live_attempts": 0,
        "executed_work": total_work,
        "lost_work": lost_work,
    }
    if invariants["down_starts"] or invariants["completed_without_winner"]:
        raise SimultaneousRoutingError("simultaneous routing invariant failed")
    metrics = _result_metrics(trace, token_results, total_work, lost_work, current_time)
    metrics["no_eligible_time"] = no_eligible_time
    for replica_id in replica_ids:
        samples = replica_audit[replica_id].pop("hazard_samples")
        replica_audit[replica_id]["mean_estimated_hazard"] = (
            math.fsum(samples) / len(samples) if samples else 0.0
        )
        available_time = float(replica_audit[replica_id]["available_time"])
        replica_audit[replica_id]["utilization_if_available"] = (
            float(replica_audit[replica_id]["busy_time"]) / available_time
            if available_time > 0.0
            else 0.0
        )
    physical = RoutingResult(
        policy_name,
        trace.fingerprint,
        trace.token_fingerprint,
        tuple(token_results),
        tuple(starts),
        metrics,
        invariants,
        replica_audit,
        True,
    )
    return SimultaneousRoutingResult(physical, tuple(batch_audits))


__all__ = [
    "ActiveTokenState",
    "EngineSnapshot",
    "PopulationDecisionContext",
    "PreBatchPublicState",
    "SimultaneousBatchAudit",
    "SimultaneousPolicy",
    "SimultaneousRoutingError",
    "SimultaneousRoutingResult",
    "TokenDecisionObservation",
    "simulate_simultaneous_routing",
]
