"""ADR-0014 isolated eight-Expert, fixed-Primary, dual-Backup experiment.

This module deliberately does not widen the historical two-Replica public
engine. It reuses its verified event plumbing and copy lifecycle internally,
while freezing a distinct topology, CRN namespace and development claim.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
from typing import Callable, Iterable, Mapping

from . import hedge_simulation as hs
from .artifacts import write_run_directory
from .attribution_episode import EpisodeIdentity, EpisodeProtocol, derive_episode_seed
from .common_state import CommonStateTimeline, Phase, phase_at, state_at
from .config import ExperimentConfig
from .domain import CommonState, ProtectionAction as Action, TokenClass
from .metrics import percentile
from .paired import compute_tau0
from .transient_control import ReservationParameters, TimeClassRule
from .transient_objective import cvar95_fractional_tail
from .transient_search import assess_development_safety, enumerate_time_class_rules
from .workload import TokenSpec, WorkloadTrace, _stream_seed, lognormal_parameters


SCALED_NAMESPACE = "scaled-control:v1:fit"
SCALED_MACRO_SEED = 20260905
SCALED_EPISODE_COUNT = 128
SCALED_EXPERT_COUNT = 8
SCALED_REPLICA_COUNT = 3
SCALED_PROTOCOL = EpisodeProtocol(CommonStateTimeline(100, 200, 220), 320)


def _true_int(value: object, name: str, *, lower: int = 0) -> int:
    if type(value) is not int or value < lower:
        raise ValueError(f"{name} must be an int >= {lower}, got {value!r}")
    return value


def _positive(value: object, name: str, *, zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class ScaledTokenSpec:
    token_id: int
    arrival_time: float
    token_class: TokenClass
    expert_id: int
    expert_token_id: int

    def __post_init__(self) -> None:
        _true_int(self.token_id, "token_id")
        object.__setattr__(self, "arrival_time", _positive(self.arrival_time, "arrival_time", zero=True))
        if not isinstance(self.token_class, TokenClass):
            raise ValueError("token_class must be TokenClass")
        _true_int(self.expert_id, "expert_id")
        _true_int(self.expert_token_id, "expert_token_id")


@dataclass(frozen=True)
class ScaledWorkDraw:
    """Work keyed by global Token, its immutable Expert, Replica and attempt."""

    attempt0: tuple[float, float, float]
    attempt1: tuple[float, float, float]
    attempt2: tuple[float, float, float]
    attempt3: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("attempt0", "attempt1", "attempt2", "attempt3"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != SCALED_REPLICA_COUNT:
                raise ValueError(f"{name} must have three Replica draws")
            converted = tuple(_positive(value, f"{name}[{replica}]")
                              for replica, value in enumerate(values))
            object.__setattr__(self, name, converted)


@dataclass(frozen=True)
class ScaledEpisodeTrace:
    identity: EpisodeIdentity
    protocol: EpisodeProtocol
    expert_count: int
    arrival_rate: float
    tokens: tuple[ScaledTokenSpec, ...]
    work: tuple[ScaledWorkDraw, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise ValueError("identity must be EpisodeIdentity")
        if not isinstance(self.protocol, EpisodeProtocol):
            raise ValueError("protocol must be EpisodeProtocol")
        if _true_int(self.expert_count, "expert_count", lower=1) != SCALED_EXPERT_COUNT:
            raise ValueError("scaled trace requires exactly eight Experts")
        object.__setattr__(self, "arrival_rate", _positive(self.arrival_rate, "arrival_rate"))
        if not self.tokens or len(self.tokens) != len(self.work):
            raise ValueError("scaled trace requires nonempty one-to-one Token/work rows")
        expected_local = [0] * self.expert_count
        previous = -1.0
        for expected_id, token in enumerate(self.tokens):
            if not isinstance(token, ScaledTokenSpec) or token.token_id != expected_id:
                raise ValueError("global Token IDs must be dense and ordered")
            if token.arrival_time < previous or token.arrival_time >= self.protocol.arrival_cutoff:
                raise ValueError("arrivals must be ordered inside the half-open episode horizon")
            previous = token.arrival_time
            if token.expert_id >= self.expert_count:
                raise ValueError("expert_id outside scaled topology")
            if token.expert_token_id != expected_local[token.expert_id]:
                raise ValueError("per-Expert Token IDs must be dense and ordered")
            expected_local[token.expert_id] += 1
        if any(not isinstance(draw, ScaledWorkDraw) for draw in self.work):
            raise ValueError("every work row must be ScaledWorkDraw")


def _validate_scaled_config(config: ExperimentConfig) -> None:
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be ExperimentConfig")
    expected = {"expert_count": 8, "replicas_per_expert": 3,
                "failure_domain_count": 3, "healthy_offered_load": .45}
    for name, value in expected.items():
        observed = getattr(config, name)
        if observed != value:
            raise ValueError(f"scaled experiment requires {name}={value}, got {observed}")


def generate_scaled_episode(
    config: ExperimentConfig, namespace: str, macro_seed: int, episode_index: int,
    protocol: EpisodeProtocol = SCALED_PROTOCOL,
) -> ScaledEpisodeTrace:
    """Generate one policy-independent global arrival/Gate/class/work trace."""
    _validate_scaled_config(config)
    identity = EpisodeIdentity(namespace, macro_seed, episode_index)
    if not isinstance(protocol, EpisodeProtocol):
        raise ValueError("protocol must be EpisodeProtocol")
    seed = derive_episode_seed(config.base_seed, identity)
    arrival_rate = config.expert_count * config.healthy_offered_load / config.healthy_service_mean
    arrival_rng = random.Random(_stream_seed(seed, "scaled:arrival"))
    class_rng = random.Random(_stream_seed(seed, "scaled:token-class"))
    expert_rng = random.Random(_stream_seed(seed, "scaled:expert-id"))
    mu, sigma = lognormal_parameters(config.healthy_service_mean, config.service_time_cv)
    streams = {
        (expert, attempt, replica): random.Random(_stream_seed(
            seed, f"scaled:expert:{expert}:replica:{replica}:attempt:{attempt}"))
        for expert in range(config.expert_count)
        for attempt in range(4)
        for replica in range(config.replicas_per_expert)
    }
    tokens, work = [], []
    local_ids = [0] * config.expert_count
    now = 0.0
    while True:
        now += arrival_rng.expovariate(arrival_rate)
        if now >= protocol.arrival_cutoff:
            break
        token_id = len(tokens)
        expert = expert_rng.randrange(config.expert_count)
        cls = TokenClass.REGULAR if class_rng.random() < config.regular_token_ratio else TokenClass.URGENT
        tokens.append(ScaledTokenSpec(token_id, now, cls, expert, local_ids[expert]))
        local_ids[expert] += 1
        attempt_draws = []
        for attempt in range(4):
            attempt_draws.append(tuple(
                streams[(expert, attempt, replica)].lognormvariate(mu, sigma)
                for replica in range(config.replicas_per_expert)))
        work.append(ScaledWorkDraw(*attempt_draws))
    if not tokens or any(count == 0 for count in local_ids):
        raise ValueError("scaled episode must contain at least one Token for every Expert")
    return ScaledEpisodeTrace(identity, protocol, config.expert_count, arrival_rate,
                              tuple(tokens), tuple(work))


def scaled_trace_fingerprint(trace: ScaledEpisodeTrace) -> str:
    """Hash topology, episode identity and all exogenous values without JSON floats."""
    if not isinstance(trace, ScaledEpisodeTrace):
        raise ValueError("trace must be ScaledEpisodeTrace")
    digest = hashlib.sha256()
    header = {"identity": trace.identity.key, "expert_count": trace.expert_count,
              "arrival_rate": trace.arrival_rate,
              "protocol": [trace.protocol.timeline.degraded_start,
                           trace.protocol.timeline.failed_start,
                           trace.protocol.timeline.recovered_start,
                           trace.protocol.arrival_cutoff]}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for token, draw in zip(trace.tokens, trace.work):
        digest.update(struct.pack(">qdqq", token.token_id, token.arrival_time,
                                  token.expert_id, token.expert_token_id))
        digest.update(token.token_class.value.encode())
        digest.update(struct.pack(">12d", *(value for attempt in (
            draw.attempt0, draw.attempt1, draw.attempt2, draw.attempt3)
                                            for value in attempt)))
    return digest.hexdigest()


@dataclass(frozen=True)
class ScaledAdmissionDecision:
    token_id: int
    expert_id: int
    token_class: TokenClass
    arrival_time: float
    requested: Action
    applied: Action
    backup_count: int
    window_start: float | None
    window_end: float | None
    cap: float
    balance_before: float
    charge: float
    balance_after: float
    quota_suppressed: bool


def _decision_counts(decisions) -> dict:
    rows = tuple(decisions)
    requested = sum(d.requested is not Action.NORMAL for d in rows)
    applied = sum(d.applied is not Action.NORMAL for d in rows)
    suppressed = sum(d.quota_suppressed for d in rows)
    if requested != applied + suppressed:
        raise RuntimeError("scaled requested/applied/suppressed identity failed")
    return {"token_count": len(rows), "requested_hedges": requested,
            "applied_hedges": applied, "quota_suppressed": suppressed,
            "applied_backup_copies": sum(d.backup_count for d in rows
                                          if d.applied is not Action.NORMAL),
            "reserved_work": math.fsum(d.charge for d in rows)}


@dataclass(frozen=True)
class ScaledAdmissionPlan:
    decisions: tuple[ScaledAdmissionDecision, ...]
    backup_count: int
    parameters: ReservationParameters

    def audit(self) -> dict:
        return {"contract": "per_expert_nonrefundable_mean_work_reservation_v1",
                "backup_count": self.backup_count,
                "charge_per_admitted_request": self.backup_count * self.parameters.mean_requirement,
                "pathwise_executed_work_cap": False,
                "global": _decision_counts(self.decisions),
                "by_expert": {str(expert): _decision_counts(
                    d for d in self.decisions if d.expert_id == expert)
                              for expert in range(SCALED_EXPERT_COUNT)},
                "by_class": {cls.name.lower(): _decision_counts(
                    d for d in self.decisions if d.token_class is cls)
                             for cls in TokenClass}}


def plan_scaled_actions(
    trace: ScaledEpisodeTrace, rule: TimeClassRule, backup_count: int,
    parameters: ReservationParameters | None = None,
) -> ScaledAdmissionPlan:
    """Causal metadata-only rule with an independent ledger for every Expert."""
    if not isinstance(trace, ScaledEpisodeTrace):
        raise ValueError("trace must be ScaledEpisodeTrace")
    if type(rule) is not TimeClassRule:
        raise ValueError("rule must be TimeClassRule")
    _true_int(backup_count, "backup_count")
    if backup_count not in (0, 1, 2):
        raise ValueError("backup_count must be 0, 1, or 2")
    if backup_count == 0 and any(action is not Action.NORMAL for action in rule.actions):
        raise ValueError("zero-Backup arm must request exact Normal")
    params = ReservationParameters() if parameters is None else parameters
    if not isinstance(params, ReservationParameters):
        raise ValueError("parameters must be ReservationParameters")
    timeline = trace.protocol.timeline
    active_window = [None] * trace.expert_count
    balances = [Decimal(0)] * trace.expert_count
    rows = []
    with localcontext() as context:
        context.prec = 80
        dec = lambda value: Decimal(str(value))
        d_start, f_start = dec(timeline.degraded_start), dec(timeline.failed_start)
        midpoint = (d_start + f_start) / 2
        width = dec(params.window_width)
        rate = dec(params.budget_rate) * dec(params.scale)
        charge = dec(params.mean_requirement) * backup_count
        for token in trace.tokens:
            requested = applied = Action.NORMAL
            ws_out = we_out = None
            cap = before = after = debit = Decimal(0)
            if phase_at(timeline, token.arrival_time) is Phase.DEGRADED:
                now = dec(token.arrival_time)
                index = (now - d_start) // width
                ws, we = d_start + index * width, min(f_start, d_start + (index + 1) * width)
                cap = rate * (we - ws)
                expert = token.expert_id
                if active_window[expert] != ws:
                    active_window[expert], balances[expert] = ws, cap
                before = balances[expert]
                late = int(now >= midpoint)
                urgent = int(token.token_class is TokenClass.URGENT)
                requested = rule.actions[2 * late + urgent]
                if requested is not Action.NORMAL and balances[expert] >= charge:
                    debit = charge
                    balances[expert] -= debit
                    applied = requested
                after = balances[expert]
                ws_out, we_out = float(ws), float(we)
            rows.append(ScaledAdmissionDecision(
                token.token_id, token.expert_id, token.token_class, token.arrival_time,
                requested, applied, backup_count, ws_out, we_out, float(cap),
                float(before), float(debit), float(after),
                requested is not Action.NORMAL and applied is Action.NORMAL))
    return ScaledAdmissionPlan(tuple(rows), backup_count, params)


@dataclass(frozen=True)
class ScaledAttemptRecord:
    token_id: int
    expert_id: int
    attempt_id: int
    replica_id: int
    enqueue_time: float
    start_time: float | None
    terminal_time: float
    required_work: float
    executed_work: float
    status: hs.HedgeAttemptStatus


@dataclass(frozen=True)
class ScaledTokenResult:
    token_id: int
    expert_id: int
    token_class: TokenClass
    arrival_time: float
    primary_replica: int
    action: Action
    attempts: tuple[ScaledAttemptRecord, ...]
    winner_attempt_id: int
    completion_time: float
    completion_phase: Phase
    replay_count: int
    latency: float


@dataclass(frozen=True)
class ScaledSimulationResult:
    tokens: tuple[ScaledTokenResult, ...]
    attempts: tuple[ScaledAttemptRecord, ...]
    primary_executions: int
    replay_executions: int
    hedge_requested: int
    hedge_launches: int
    hedge_timers_voided: int
    hedge_winners: int
    drain_end_time: float

    def __post_init__(self) -> None:
        if not self.tokens or len(self.tokens) != self.primary_executions:
            raise RuntimeError("scaled result must contain one Primary and result per Token")
        if tuple(token.token_id for token in self.tokens) != tuple(range(len(self.tokens))):
            raise RuntimeError("scaled result Token IDs are not dense")
        if tuple(self.attempts) != tuple(a for token in self.tokens for a in token.attempts):
            raise RuntimeError("scaled flattened attempt rows mismatch Token rows")
        for token in self.tokens:
            if len({a.attempt_id for a in token.attempts}) != len(token.attempts):
                raise RuntimeError("duplicate copy attempt for scaled Token")
            winners = [a for a in token.attempts if a.status is hs.HedgeAttemptStatus.COMPLETED_WINNER]
            if (len(winners) != 1 or winners[0].attempt_id != token.winner_attempt_id
                    or winners[0].terminal_time != token.completion_time
                    or token.latency != token.completion_time - token.arrival_time
                    or token.replay_count not in (0, 1)):
                raise RuntimeError("scaled Token winner/lifecycle invariant failed")
            for attempt in token.attempts:
                if (attempt.executed_work < 0 or attempt.executed_work > attempt.required_work
                        or not math.isfinite(attempt.executed_work)
                        or not math.isfinite(attempt.required_work)):
                    raise RuntimeError("scaled attempt work invariant failed")
                if (attempt.status is hs.HedgeAttemptStatus.CANCELLED_QUEUED
                        and attempt.executed_work != 0):
                    raise RuntimeError("cancelled scaled copy executed work")
                if (attempt.status in (hs.HedgeAttemptStatus.COMPLETED_WINNER,
                                       hs.HedgeAttemptStatus.COMPLETED_LOSER)
                        and attempt.executed_work != attempt.required_work):
                    raise RuntimeError("completed scaled copy work mismatch")
        if self.hedge_winners != sum(a.attempt_id >= 2 and
                                     a.status is hs.HedgeAttemptStatus.COMPLETED_WINNER
                                     for a in self.attempts):
            raise RuntimeError("scaled Hedge winner counter mismatch")
        if self.hedge_requested != sum(t.action is not Action.NORMAL for t in self.tokens):
            raise RuntimeError("scaled Hedge request counter mismatch")
        if self.hedge_launches != sum(a.attempt_id >= 2 for a in self.attempts):
            raise RuntimeError("scaled Hedge launch counter mismatch")
        if self.replay_executions != sum(a.attempt_id == 1 for a in self.attempts):
            raise RuntimeError("scaled Replay counter mismatch")
        if self.drain_end_time != max(a.terminal_time for a in self.attempts):
            raise RuntimeError("scaled drain_end_time mismatch")


class _ScaledExpertEngine(hs._Engine):
    def __init__(self, trace, timeline, slowdown, actions, delay, global_ids,
                 backup_count, attempt3):
        self.global_ids = tuple(global_ids)
        self.backup_count = backup_count
        self.attempt3 = tuple(attempt3)
        super().__init__(trace, timeline, slowdown, actions, delay, 3)

    def _boundary_snapshot(self, now):
        live, remaining = [], []
        for replica in range(3):
            queued = [item for item in self.queues[replica] if not item.cancelled]
            running = self.running[replica]
            live.append(len(queued) + int(running is not None))
            value = math.fsum(item.required_work for item in queued)
            if running is not None:
                executed = running.executed_work + (now-running.last_update_time)*running.current_speed
                value += max(0, running.required_work-executed)
            remaining.append(value)
        return tuple(live), tuple(remaining)

    def _enter_failed(self, now):
        self.generations[0] += 1
        lost = []
        running = self.running[0]
        if running is not None:
            self._settle(running, now)
            self.running[0] = None
            token = self.tokens[running.token_id]
            token.live.pop(running.attempt_id, None)
            self._record(running, hs.HedgeAttemptStatus.FAILED_RUNNING, now,
                         running.executed_work)
            if running.attempt_id == 0:
                self.failed_running += 1
                lost.append(running.token_id)
        for queued in list(self.queues[0]):
            if queued.cancelled:
                continue
            token = self.tokens[queued.token_id]
            token.live.pop(queued.attempt_id, None)
            self._record(queued, hs.HedgeAttemptStatus.INVALIDATED_QUEUED, now, 0)
            if queued.attempt_id == 0:
                self.invalidated_queued += 1
                lost.append(queued.token_id)
        self.queues[0].clear()
        for local_id in lost:
            token = self.tokens[local_id]
            if token.winner_set or any(attempt >= 2 for attempt in token.live) or token.replay_count:
                continue
            token.replay_count = 1
            token.timer_pending = False
            destination = 1 + self.global_ids[local_id] % 2
            self._enqueue(hs._QueuedAttempt(local_id, 1, destination, now,
                                            self.trace.replay_service_times[destination][local_id]))
        self.queue_length_at_failed_start = tuple(self._live_queue_length(r) for r in range(3))
        self.live_attempts_at_failed_start, self.remaining_work_at_failed_start = self._boundary_snapshot(now)

    def _enter_recovered(self, now):
        self.queue_length_at_recovered_start = tuple(self._live_queue_length(r) for r in range(3))
        self.live_attempts_at_recovered_start, self.remaining_work_at_recovered_start = self._boundary_snapshot(now)

    def _handle_arrival(self, event):
        spec = self.trace.tokens[event.token_id]
        token = self.tokens[spec.token_id]
        now = event.event_time
        state = state_at(self.timeline, now)
        primary = 1 + self.global_ids[spec.token_id] % 2 if state is CommonState.FAILED else 0
        token.primary_replica = primary
        self._enqueue(hs._QueuedAttempt(spec.token_id, 0, primary, now,
                                        self.trace.service_times[primary][spec.token_id]))
        self.primary_executions += 1
        if token.action is Action.IMMEDIATE_HEDGE:
            if state is CommonState.FAILED:
                self.hedge_suppressed += 1
                self.suppressed_events.append((now, spec.token_id))
            else:
                self._launch_hedge(token, now)
        elif token.action is Action.DELAYED_HEDGE:
            token.timer_pending = True
            self._push(hs._Event(now + self.hedge_delay, hs._EventType.HEDGE_TIMER,
                                 next(self.sequence), token_id=spec.token_id))

    def _launch_hedge(self, token, now):
        global_id = self.global_ids[token.token_id]
        destinations = ((1 + global_id % 2, 2),) if self.backup_count == 1 else ((1, 2), (2, 3))
        for replica, attempt in destinations:
            work = (self.trace.hedge_service_times[replica][token.token_id]
                    if attempt == 2 else self.attempt3[token.token_id])
            self.hedge_launches += 1
            self._enqueue(hs._QueuedAttempt(token.token_id, attempt, replica, now, work))

    def _start(self, replica, queued, now):
        state = state_at(self.timeline, now)
        speed = 1.0
        if replica == 0:
            speed = 0.0 if state is CommonState.FAILED else (1/self.slowdown if state is CommonState.DEGRADED else 1.0)
        if speed <= 0:
            raise AssertionError("scaled dispatcher started Failed Primary")
        self.generations[replica] += 1
        running = hs._RunningAttempt(queued.token_id, queued.attempt_id, replica,
                                     queued.enqueue_time, now, queued.required_work,
                                     0.0, now, speed)
        self.running[replica] = running
        self.tokens[queued.token_id].live[queued.attempt_id] = running
        if queued.attempt_id == 1:
            self.replay_executions += 1
        self._schedule_completion(running, now + queued.required_work/speed)


def _expert_result(trace: ScaledEpisodeTrace, expert: int, plan: ScaledAdmissionPlan,
                   hedge_delay: float, slowdown: float):
    global_ids = [token.token_id for token in trace.tokens if token.expert_id == expert]
    if not global_ids:
        return (), None
    local_tokens = tuple(TokenSpec(local, trace.tokens[global_id].arrival_time,
                                   trace.tokens[global_id].token_class)
                         for local, global_id in enumerate(global_ids))
    def streams(attempt):
        return tuple(tuple(getattr(trace.work[g], f"attempt{attempt}")[replica]
                           for g in global_ids) for replica in range(3))
    local_trace = WorkloadTrace(
        trace.identity.macro_seed, trace.arrival_rate/trace.expert_count,
        local_tokens, streams(0), streams(1), streams(2))
    action_by_global = {d.token_id: d.applied for d in plan.decisions if d.expert_id == expert}
    actions = {local: action_by_global[global_id] for local, global_id in enumerate(global_ids)}
    attempt3 = tuple(trace.work[g].attempt3[2] for g in global_ids)
    engine = _ScaledExpertEngine(local_trace, trace.protocol.timeline, slowdown,
                                 actions, hedge_delay, global_ids, plan.backup_count, attempt3)
    engine.run()
    converted_tokens = []
    for local, global_id in enumerate(global_ids):
        completion = engine.completion_times[local]
        if completion is None:
            raise RuntimeError(f"scaled Expert {expert} left Token {global_id} incomplete")
        attempts = tuple(ScaledAttemptRecord(
            global_id, expert, record.attempt_id, record.replica_id,
            record.enqueue_time, record.start_time, record.terminal_time,
            record.required_work, record.executed_work, record.status)
            for record in sorted(engine.attempts_by_token[local], key=lambda a: a.attempt_id))
        state = engine.tokens[local]
        converted_tokens.append(ScaledTokenResult(
            global_id, expert, local_trace.tokens[local].token_class,
            local_trace.tokens[local].arrival_time, state.primary_replica, state.action,
            attempts, engine.winner_attempt_ids[local], completion,
            phase_at(trace.protocol.timeline, completion), state.replay_count,
            completion-local_trace.tokens[local].arrival_time))
    return tuple(converted_tokens), engine


@dataclass(frozen=True)
class ScaledEpisodeResult:
    trace: ScaledEpisodeTrace
    rule: TimeClassRule
    backup_count: int
    plan: ScaledAdmissionPlan
    simulation: ScaledSimulationResult
    hedge_delay: float


def simulate_scaled_episode(
    trace: ScaledEpisodeTrace, rule: TimeClassRule, backup_count: int,
    parameters: ReservationParameters | None = None, *,
    degraded_slowdown: float = 2.0, hedge_delay: float = 1.5,
) -> ScaledEpisodeResult:
    if not isinstance(trace, ScaledEpisodeTrace):
        raise ValueError("trace must be ScaledEpisodeTrace")
    slowdown = _positive(degraded_slowdown, "degraded_slowdown")
    delay = _positive(hedge_delay, "hedge_delay")
    plan = plan_scaled_actions(trace, rule, backup_count, parameters)
    tokens, engines = [], []
    for expert in range(trace.expert_count):
        expert_tokens, engine = _expert_result(trace, expert, plan, delay, slowdown)
        tokens.extend(expert_tokens)
        if engine is not None:
            engines.append(engine)
    tokens.sort(key=lambda token: token.token_id)
    attempts = tuple(attempt for token in tokens for attempt in token.attempts)
    result = ScaledSimulationResult(
        tuple(tokens), attempts, sum(e.primary_executions for e in engines),
        sum(e.replay_executions for e in engines), sum(e.hedge_requested for e in engines),
        sum(e.hedge_launches for e in engines), sum(e.hedge_timers_voided for e in engines),
        sum(a.attempt_id >= 2 and a.status is hs.HedgeAttemptStatus.COMPLETED_WINNER
            for a in attempts), max(e.drain_end_time for e in engines))
    audit = _decision_counts(plan.decisions)
    if result.hedge_requested != audit["applied_hedges"]:
        raise RuntimeError("scaled engine requests do not match admitted requests")
    if result.hedge_launches > audit["applied_backup_copies"]:
        raise RuntimeError("scaled engine launched more Backup copies than admitted")
    return ScaledEpisodeResult(trace, rule, backup_count, plan, result, delay)


def _scaled_objective(runs: Iterable[ScaledEpisodeResult]) -> dict:
    episodes = tuple(runs)
    if not episodes:
        raise ValueError("scaled objective requires episodes")
    keys = [run.trace.identity.key for run in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scaled episode in objective")
    timeline = episodes[0].trace.protocol.timeline
    if any(run.trace.protocol != episodes[0].trace.protocol or
           run.backup_count != episodes[0].backup_count or
           run.hedge_delay != episodes[0].hedge_delay for run in episodes):
        raise ValueError("scaled objective contract mismatch")
    tokens = tuple(t for run in episodes for t in run.simulation.tokens)
    attempts = tuple(a for run in episodes for a in run.simulation.attempts)
    cohorts, phase_losses, tails, missing = {}, {}, {}, []
    for phase in Phase:
        phase_tokens = [t for t in tokens if phase_at(timeline, t.arrival_time) is phase]
        cells = {}
        for cls, weight, deadline, penalty in (
                (TokenClass.REGULAR, .8, 3., 1.), (TokenClass.URGENT, .2, 2., 5.)):
            rows = [t for t in phase_tokens if t.token_class is cls]
            n = len(rows)
            mean = math.fsum(t.latency for t in rows)/n if n else None
            excess = math.fsum(max(0, t.latency-deadline) for t in rows)/n if n else None
            replay = sum(t.replay_count for t in rows)/n if n else None
            loss = mean + penalty*excess + penalty*replay if n else None
            cells[cls.name.lower()] = {"sample_count": n, "mean_latency": mean,
                "mean_slo_excess": excess, "replay_rate": replay, "loss": loss,
                "weight": weight, "deadline": deadline}
            if not n:
                missing.append(f"{phase.value}/{cls.name.lower()}")
        cohorts[phase.value] = cells
        phase_losses[phase.value] = (math.fsum(c["weight"]*c["loss"] for c in cells.values())
                                     if all(c["sample_count"] for c in cells.values()) else None)
        if phase in (Phase.DEGRADED, Phase.FAILED):
            tails[phase.value] = {"sample_count": len(phase_tokens),
                "cvar95_fractional_tail": cvar95_fractional_tail(t.latency for t in phase_tokens)
                if phase_tokens else None}
    work = math.fsum(a.executed_work for a in attempts)
    waste = math.fsum(a.executed_work for a in attempts
                      if a.status is not hs.HedgeAttemptStatus.COMPLETED_WINNER)
    components = {"phase_loss": math.fsum(phase_losses.values())/4 if not missing else None,
                  "fault_tail": math.fsum(x["cvar95_fractional_tail"] for x in tails.values())/2
                  if all(x["sample_count"] for x in tails.values()) else None,
                  "executed_work": work/len(tokens), "wasted_work": waste/len(tokens)}
    total = math.fsum(components.values()) if not missing else None
    return {"status": "incomplete" if missing else "complete", "total": total,
            "episode_count": len(episodes), "generated_tokens": len(tokens),
            "missing_cohorts": missing, "cohorts": cohorts,
            "phase_losses": phase_losses, "fault_tails": tails,
            "components": components,
            "raw_work": {"total_executed_work": work, "non_winner_executed_work": waste}}


def _resources(runs) -> dict:
    timeline = runs[0].trace.protocol.timeline
    tokens = tuple(t for run in runs for t in run.simulation.tokens)
    result = {}
    for phase in (Phase.HEALTHY, Phase.RECOVERED):
        values = [t.latency for t in tokens if phase_at(timeline, t.arrival_time) is phase]
        result[phase.value] = {"sample_count": len(values),
                               "mean": math.fsum(values)/len(values),
                               "p95": percentile(values, 95)}
    result["work"] = math.fsum(a.executed_work for run in runs for a in run.simulation.attempts)
    return result


def _arm(episodes, rule, backup_count, reservation, delay, baseline_resources=None):
    runs = tuple(simulate_scaled_episode(e, rule, backup_count, reservation,
                                         hedge_delay=delay) for e in episodes)
    objective = _scaled_objective(runs)
    resources = _resources(runs)
    safety = assess_development_safety(resources, baseline_resources or resources)
    audits = [run.plan.audit() for run in runs]
    audit_blocks = [audit["global"] for audit in audits]
    admission = {name: math.fsum(a[name] for a in audit_blocks) if name == "reserved_work"
                 else sum(a[name] for a in audit_blocks)
                 for name in ("token_count", "requested_hedges", "applied_hedges",
                              "quota_suppressed", "applied_backup_copies", "reserved_work")}
    admission["by_expert"] = {
        str(expert): {
            name: (math.fsum(audit["by_expert"][str(expert)][name] for audit in audits)
                   if name == "reserved_work" else
                   sum(audit["by_expert"][str(expert)][name] for audit in audits))
            for name in ("requested_hedges", "applied_hedges", "quota_suppressed",
                         "applied_backup_copies", "reserved_work")
        } for expert in range(SCALED_EXPERT_COUNT)}
    admission["by_class"] = {
        cls.name.lower(): {
            name: (math.fsum(audit["by_class"][cls.name.lower()][name] for audit in audits)
                   if name == "reserved_work" else
                   sum(audit["by_class"][cls.name.lower()][name] for audit in audits))
            for name in ("requested_hedges", "applied_hedges", "quota_suppressed",
                         "applied_backup_copies", "reserved_work")
        } for cls in TokenClass}
    simulation = {"replay_executions": sum(r.simulation.replay_executions for r in runs),
                  "hedge_requested": sum(r.simulation.hedge_requested for r in runs),
                  "hedge_launches": sum(r.simulation.hedge_launches for r in runs),
                  "hedge_winners": sum(r.simulation.hedge_winners for r in runs),
                  "hedge_timers_voided": sum(r.simulation.hedge_timers_voided for r in runs)}
    return {"rule": "".join(a.value for a in rule.actions), "backup_count": backup_count,
            "objective": objective, "resources": resources, "safety": safety,
            "feasible": objective["status"] == "complete" and safety["feasible"],
            "admission": admission, "simulation": simulation}


def _comparison(candidate, baseline):
    baseline_waste = baseline["objective"]["raw_work"]["non_winner_executed_work"]
    return {"objective_ratio": candidate["objective"]["total"]/baseline["objective"]["total"],
            "objective_delta": candidate["objective"]["total"]-baseline["objective"]["total"],
            "D_cvar95_ratio": candidate["objective"]["fault_tails"]["D"]["cvar95_fractional_tail"] /
                              baseline["objective"]["fault_tails"]["D"]["cvar95_fractional_tail"],
            "F_cvar95_ratio": candidate["objective"]["fault_tails"]["F"]["cvar95_fractional_tail"] /
                              baseline["objective"]["fault_tails"]["F"]["cvar95_fractional_tail"],
            "total_work_ratio": candidate["resources"]["work"]/baseline["resources"]["work"],
            "wasted_work_ratio": (candidate["objective"]["raw_work"]["non_winner_executed_work"] /
                                   baseline_waste if baseline_waste > 0 else None),
            "wasted_work_delta": candidate["objective"]["raw_work"]["non_winner_executed_work"] -
                                 baseline_waste}


def _validate_scaled_bank(episodes, require_frozen):
    rows = tuple(sorted(episodes, key=lambda e: e.identity.episode_index))
    if not rows or any(not isinstance(e, ScaledEpisodeTrace) for e in rows):
        raise ValueError("scaled bank requires ScaledEpisodeTrace rows")
    keys = [e.identity.key for e in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scaled episode")
    first = rows[0]
    if first.identity.namespace not in (SCALED_NAMESPACE, "scaled-control:v1:test"):
        raise ValueError("scaled search accepts fit or test namespace only")
    if any(e.identity.namespace != first.identity.namespace or
           e.identity.macro_seed != first.identity.macro_seed or
           e.protocol != first.protocol for e in rows):
        raise ValueError("scaled episode bank identity/protocol mismatch")
    if require_frozen:
        if len(rows) != SCALED_EPISODE_COUNT:
            raise ValueError(f"frozen scaled fit requires exactly {SCALED_EPISODE_COUNT} episodes")
        if (first.identity.namespace != SCALED_NAMESPACE or
                first.identity.macro_seed != SCALED_MACRO_SEED or
                first.protocol != SCALED_PROTOCOL or
                [e.identity.episode_index for e in rows] != list(range(SCALED_EPISODE_COUNT))):
            raise ValueError("scaled bank does not match frozen fit identity")
    return rows


def evaluate_scaled_development(
    episodes: Iterable[ScaledEpisodeTrace],
    reservation: ReservationParameters | None = None, *,
    hedge_delay: float = 1.5, require_frozen: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    rows = _validate_scaled_bank(episodes, require_frozen)
    reservation = ReservationParameters() if reservation is None else reservation
    delay = _positive(hedge_delay, "hedge_delay")
    fingerprints = {e.identity.key: scaled_trace_fingerprint(e) for e in rows}
    baseline = _arm(rows, TimeClassRule(), 0, reservation, delay)
    frozen_rule = TimeClassRule(Action.NORMAL, Action.IMMEDIATE_HEDGE,
                                Action.IMMEDIATE_HEDGE, Action.NORMAL)
    single = _arm(rows, frozen_rule, 1, reservation, delay, baseline["resources"])
    bank_rows = []
    rules = enumerate_time_class_rules()
    for index, rule in enumerate(rules):
        arm = _arm(rows, rule, 2, reservation, delay, baseline["resources"])
        arm["lexical_index"] = index
        bank_rows.append(arm)
        if progress is not None:
            progress(index + 1, len(rules), arm["rule"])
    feasible = [row for row in bank_rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("no feasible dual-Backup rule, including NNNN")
    selected = min(feasible, key=lambda row: (row["objective"]["total"],
                                               row["resources"]["work"],
                                               row["lexical_index"]))
    return {"schema_version": 1,
            "scenario": "eight_expert_three_replica_scaled_development",
            "claim_boundary": "development_only_not_MFG_or_qualification",
            "namespace": rows[0].identity.namespace,
            "macro_seed": rows[0].identity.macro_seed,
            "episode_count": len(rows),
            "token_count": sum(len(e.tokens) for e in rows),
            "topology": {"expert_count": 8, "replicas_per_expert": 3,
                         "primary_replicas_per_expert": 1,
                         "backup_replicas_per_expert": 2,
                         "failure_domains": ["A", "B", "C"],
                         "failed_replica": 0},
            "reservation": {name: getattr(reservation, name)
                            for name in reservation.__dataclass_fields__},
            "hedge_delay": delay, "trace_fingerprints": fingerprints,
            "arms": {"no_hedge": baseline, "single_backup_NIIN": single,
                     "dual_backup_selected": selected},
            "comparisons": {"single_vs_no_hedge": _comparison(single, baseline),
                            "dual_selected_vs_no_hedge": _comparison(selected, baseline),
                            "dual_selected_vs_single": _comparison(selected, single)},
            "dual_backup_search": {"rule_count": 81,
                                   "feasible_rule_count": len(feasible),
                                   "selected_rule": selected["rule"],
                                   "rules": bank_rows}}


def run_frozen_scaled_development(config: ExperimentConfig, progress=None) -> dict:
    _validate_scaled_config(config)
    episodes = tuple(generate_scaled_episode(config, SCALED_NAMESPACE,
                                             SCALED_MACRO_SEED, index)
                     for index in range(SCALED_EPISODE_COUNT))
    return evaluate_scaled_development(episodes, hedge_delay=compute_tau0(config),
                                       require_frozen=True, progress=progress)


def scaled_manifest(config: ExperimentConfig, config_sha256: str, version: str) -> dict:
    _validate_scaled_config(config)
    return {"scenario": "eight_expert_three_replica_scaled_development",
            "development_only": True, "qualification_or_holdout": False,
            "simulator_version": version, "config_sha256": config_sha256,
            "resolved_config": {f.name: getattr(config, f.name) for f in fields(config)},
            "namespace": SCALED_NAMESPACE, "macro_seed": SCALED_MACRO_SEED,
            "episode_indices": list(range(SCALED_EPISODE_COUNT)),
            "attempt_ids": {"primary": 0, "replay": 1,
                            "first_hedge": 2, "second_hedge": 3},
            "stream_contract": "sha256_stable_expert_replica_attempt_streams",
            "arms": ["no_hedge", "frozen_single_backup_NIIN", "dual_backup_81_search"]}


def write_scaled_artifact(root: str | Path, run_id: str, manifest: Mapping,
                          result: Mapping) -> Path:
    search = result.get("dual_backup_search", {})
    expected = ["".join(a.value for a in rule.actions) for rule in enumerate_time_class_rules()]
    observed = [row.get("rule") for row in search.get("rules", ())]
    if search.get("rule_count") != 81 or observed != expected or len(set(observed)) != 81:
        raise ValueError("scaled artifact requires complete unique lexical 81-rule rows")
    manifest_payload = dict(manifest)
    manifest_payload["run_id"] = run_id
    summary = {key: result[key] for key in (
        "schema_version", "scenario", "claim_boundary", "namespace", "macro_seed",
        "episode_count", "token_count", "topology", "reservation", "hedge_delay",
        "arms", "comparisons")}
    summary["dual_backup_search"] = {key: search[key] for key in (
        "rule_count", "feasible_rule_count", "selected_rule")}
    return write_run_directory(root, run_id, {
        "manifest.json": manifest_payload,
        "rule_results.json": {"trace_fingerprints": result["trace_fingerprints"],
                              "dual_backup_rules": search["rules"]},
        "summary.json": summary})
