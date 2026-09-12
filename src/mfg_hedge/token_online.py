"""Causal online Token entry and pooled Reservation for the T1 restoration.

This module is intentionally smaller than an engine: physical lifecycle,
failure ordering, CRN streams, loser handling, and drain remain in
``hedge_simulation._Engine``.  The module supplies only the arrival
observation, one-shot action source contract, and the Reservation projection
used by that engine.

T1 does not implement a Token payoff, deviation evaluator, price feedback,
solver, forward model, or MFG claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import math
import random
from typing import Mapping, Protocol

from .attribution_episode import (
    BoundarySnapshot,
    EpisodeSimulationResult,
    EpisodeTrace,
)
from .common_state import CommonStateTimeline, Phase, phase_at
from .domain import CommonState, ProtectionAction, TokenClass
from .hedge_simulation import _simulate_hedge_common_state_with_engine
from .transient_control import ReservationParameters, TimeClassRule


def _finite(value: object, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite")
    return result


def _token_id(value: object, name: str = "token_id") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _replica_id(value: object, name: str = "primary_replica") -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be the int Replica ID 0 or 1")
    return value


def validate_token_action(action: object) -> ProtectionAction:
    """Accept only the three domain enum values; never coerce strings/bools."""

    if not isinstance(action, ProtectionAction):
        raise ValueError(
            "online Token action must be ProtectionAction.NORMAL, "
            "DELAYED_HEDGE, or IMMEDIATE_HEDGE"
        )
    return action


@dataclass(frozen=True)
class TokenObservation:
    """The complete information available at one Token's arrival decision."""

    token_id: int
    token_class: TokenClass
    arrival_time: float
    phase: Phase
    phase_age: float
    primary_replica: int
    queue_snapshot: tuple[tuple[tuple[int, int], ...], ...]
    running_attempts: tuple[tuple[int, int, int], ...]
    observed_history: tuple[str, ...]
    reservation_window: tuple[float, float] | None
    reservation_cap: float
    reservation_balance: float
    public_price: float
    common_state: CommonState | None = None

    def __post_init__(self) -> None:
        _token_id(self.token_id)
        if not isinstance(self.token_class, TokenClass):
            raise ValueError("token_class must be TokenClass")
        arrival = _finite(self.arrival_time, "arrival_time")
        age = _finite(self.phase_age, "phase_age")
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be Phase")
        if self.common_state is None:
            derived_state = {
                Phase.HEALTHY: CommonState.HEALTHY,
                Phase.DEGRADED: CommonState.DEGRADED,
                Phase.FAILED: CommonState.FAILED,
                Phase.RECOVERED: CommonState.HEALTHY,
            }[self.phase]
            object.__setattr__(self, "common_state", derived_state)
        elif not isinstance(self.common_state, CommonState):
            raise ValueError("common_state must be CommonState")
        else:
            expected_state = {
                Phase.HEALTHY: CommonState.HEALTHY,
                Phase.DEGRADED: CommonState.DEGRADED,
                Phase.FAILED: CommonState.FAILED,
                Phase.RECOVERED: CommonState.HEALTHY,
            }[self.phase]
            if self.common_state is not expected_state:
                raise ValueError("phase and common_state disagree")
        _replica_id(self.primary_replica)
        if (not isinstance(self.queue_snapshot, tuple)
                or len(self.queue_snapshot) != 2):
            raise ValueError("queue_snapshot must contain two Replica queues")
        for queue in self.queue_snapshot:
            if not isinstance(queue, tuple):
                raise ValueError("queue_snapshot entries must be tuples")
            for item in queue:
                if (not isinstance(item, tuple) or len(item) != 2
                        or type(item[0]) is not int or type(item[1]) is not int):
                    raise ValueError("queue entries must be (token_id, attempt_id)")
        if not isinstance(self.running_attempts, tuple):
            raise ValueError("running_attempts must be a tuple")
        for item in self.running_attempts:
            if (not isinstance(item, tuple) or len(item) != 3
                    or any(type(value) is not int for value in item)):
                raise ValueError("running attempts must be integer triples")
        if (not isinstance(self.observed_history, tuple)
                or any(not isinstance(item, str) for item in self.observed_history)):
            raise ValueError("observed_history must be a tuple of strings")
        if self.reservation_window is not None:
            if (not isinstance(self.reservation_window, tuple)
                    or len(self.reservation_window) != 2):
                raise ValueError("reservation_window must be a start/end tuple")
            start, end = (
                _finite(value, "reservation_window", nonnegative=True)
                for value in self.reservation_window
            )
            if start >= end:
                raise ValueError("reservation_window must be increasing")
        object.__setattr__(self, "arrival_time", arrival)
        object.__setattr__(self, "phase_age", age)
        object.__setattr__(self, "reservation_cap", _finite(self.reservation_cap, "reservation_cap"))
        object.__setattr__(self, "reservation_balance", _finite(self.reservation_balance, "reservation_balance"))
        object.__setattr__(self, "public_price", _finite(self.public_price, "public_price"))


class TokenActionSource(Protocol):
    """Online one-shot source; ``new_episode`` returns fresh private state."""

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        ...

    def new_episode(self) -> "TokenActionSource":
        ...


class StaticActionSource:
    """Compatibility adapter for the legacy ``actions`` mapping."""

    def __init__(self, actions: Mapping[int, ProtectionAction]):
        if not isinstance(actions, Mapping):
            raise ValueError("actions must be a mapping")
        self._actions = {}
        for token_id, action in actions.items():
            _token_id(token_id)
            self._actions[token_id] = validate_token_action(action)

    def new_episode(self) -> "StaticActionSource":
        return StaticActionSource(self._actions)

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        return self._actions.get(observation.token_id, ProtectionAction.NORMAL)


class StaticTimeClassActionSource:
    """Adapter for the historical four-cell TimeClassRule.

    ``late_after`` is a predeclared policy parameter.  It is not read from a
    trace or an arrival list and is intended only for the scheduled transient
    equivalence fixture.
    """

    def __init__(self, rule: TimeClassRule, *, late_after: float):
        if type(rule) is not TimeClassRule:
            raise ValueError("rule must be TimeClassRule")
        self.rule = rule
        self.late_after = _finite(late_after, "late_after")

    def new_episode(self) -> "StaticTimeClassActionSource":
        return StaticTimeClassActionSource(self.rule, late_after=self.late_after)

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if observation.phase is not Phase.DEGRADED or observation.primary_replica != 0:
            return ProtectionAction.NORMAL
        late = int(observation.phase_age >= self.late_after)
        urgent = int(observation.token_class is TokenClass.URGENT)
        return self.rule.actions[2 * late + urgent]


class StableRandomActionSource:
    """Small optional source with a stable private per-Token decision stream."""

    def __init__(self, seed: int, probabilities: tuple[float, float, float]):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative int")
        if len(probabilities) != 3:
            raise ValueError("probabilities must contain N/D/I")
        values = tuple(_finite(value, "probability") for value in probabilities)
        if any(value < 0.0 for value in values) or not math.isclose(
            math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("probabilities must be non-negative and sum to one")
        self.seed = seed
        self.probabilities = values

    def new_episode(self) -> "StableRandomActionSource":
        return StableRandomActionSource(self.seed, self.probabilities)

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise ValueError("policy_key must be a non-empty string")
        digest = hashlib.sha256(f"{self.seed}:{policy_key}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        draw = rng.random()
        if draw < self.probabilities[0]:
            return ProtectionAction.NORMAL
        if draw < self.probabilities[0] + self.probabilities[1]:
            return ProtectionAction.DELAYED_HEDGE
        return ProtectionAction.IMMEDIATE_HEDGE


@dataclass(frozen=True)
class ReservationPreview:
    phase: Phase
    eligible: bool
    window: tuple[float, float] | None
    cap: float
    balance_before: float


@dataclass(frozen=True)
class OnlineAdmissionDecision:
    token_id: int
    arrival_time: float
    token_class: TokenClass
    phase: Phase
    primary_replica: int
    requested: ProtectionAction
    applied: ProtectionAction
    window_start: float | None
    window_end: float | None
    cap: float
    balance_before: float
    charge: float
    balance_after: float
    reservation_suppressed: bool
    reservation_admitted: bool = False
    executor_suppressed: bool = False
    timer_status: str = "not_scheduled"

    def __post_init__(self) -> None:
        _token_id(self.token_id)
        _finite(self.arrival_time, "arrival_time")
        if not isinstance(self.token_class, TokenClass) or not isinstance(self.phase, Phase):
            raise ValueError("invalid admission enum")
        _replica_id(self.primary_replica)
        validate_token_action(self.requested)
        validate_token_action(self.applied)
        for name in ("cap", "balance_before", "charge", "balance_after"):
            _finite(getattr(self, name), name)
        if self.charge > self.cap + 1e-12:
            raise ValueError("charge cannot exceed the declared window cap")
        if self.requested is ProtectionAction.NORMAL and self.applied is not ProtectionAction.NORMAL:
            raise ValueError("Normal request cannot become a Hedge")
        if self.requested is not self.applied and not self.reservation_suppressed:
            raise ValueError("non-equal actions must be reservation-suppressed")
        if self.reservation_admitted == self.reservation_suppressed:
            raise ValueError("reservation admission and suppression must be complementary")
        if self.timer_status not in {"not_scheduled", "scheduled", "fired", "voided", "executor_suppressed"}:
            raise ValueError("invalid timer status")


class ReservationLedger:
    """Mutable per-episode Decimal Reservation state.

    Its arithmetic and window construction mirror
    ``transient_control.plan_transient_actions``.  The engine owns one ledger
    per episode; no state is shared between runs.
    """

    def __init__(
        self,
        timeline: CommonStateTimeline,
        parameters: ReservationParameters | None = None,
    ):
        if not isinstance(timeline, CommonStateTimeline):
            raise ValueError("timeline must be CommonStateTimeline")
        self.timeline = timeline
        self.parameters = parameters or ReservationParameters()
        if not isinstance(self.parameters, ReservationParameters):
            raise ValueError("parameters must be ReservationParameters")
        self._active_window: Decimal | None = None
        self._balance = Decimal(0)
        self._last_arrival = -math.inf
        self._decisions: list[OnlineAdmissionDecision] = []

    @property
    def decisions(self) -> tuple[OnlineAdmissionDecision, ...]:
        return tuple(self._decisions)

    def _window(self, now: float) -> tuple[Decimal, Decimal, Decimal]:
        with localcontext() as context:
            context.prec = 80
            start = Decimal(str(self.timeline.degraded_start))
            width = Decimal(str(self.parameters.window_width))
            rate = Decimal(str(self.parameters.budget_rate)) * Decimal(str(self.parameters.scale))
            moment = Decimal(str(now))
            index = (moment - start) // width
            window_start = start + index * width
            # The online contract receives the nominal cap at window opening.
            # An unrevealed future failure must not prorate this observation;
            # the engine stops admission when the failure event is observed.
            window_end = window_start + width
            cap = rate * (window_end - window_start)
            if self._active_window != window_start:
                self._active_window = window_start
                self._balance = cap
            return window_start, window_end, cap

    def preview(self, arrival_time: float, token_class: TokenClass,
                primary_replica: int) -> ReservationPreview:
        now = _finite(arrival_time, "arrival_time")
        if not isinstance(token_class, TokenClass):
            raise ValueError("invalid arrival metadata")
        _replica_id(primary_replica)
        if now < self._last_arrival:
            raise ValueError("online arrivals must be nondecreasing")
        phase = phase_at(self.timeline, now)
        if phase is not Phase.DEGRADED:
            return ReservationPreview(phase, False, None, 0.0, 0.0)
        with localcontext() as context:
            context.prec = 80
            window_start, window_end, cap = self._window(now)
            return ReservationPreview(
                phase,
                primary_replica == 0,
                (float(window_start), float(window_end)),
                float(cap),
                float(self._balance),
            )

    def admit(self, token_id: int, arrival_time: float, token_class: TokenClass,
              primary_replica: int, requested: ProtectionAction) -> OnlineAdmissionDecision:
        _token_id(token_id)
        now = _finite(arrival_time, "arrival_time")
        requested = validate_token_action(requested)
        preview = self.preview(now, token_class, primary_replica)
        charge_dec = Decimal(0)
        applied = ProtectionAction.NORMAL
        suppressed = requested is not ProtectionAction.NORMAL
        with localcontext() as context:
            context.prec = 80
            charge = Decimal(str(self.parameters.mean_requirement))
            if requested is ProtectionAction.NORMAL:
                suppressed = False
            elif preview.eligible and self._balance >= charge:
                charge_dec = charge
                self._balance -= charge
                applied = requested
                suppressed = False
        self._last_arrival = now
        decision = OnlineAdmissionDecision(
            token_id=token_id,
            arrival_time=now,
            token_class=token_class,
            phase=preview.phase,
            primary_replica=primary_replica,
            requested=requested,
            applied=applied,
            window_start=(preview.window[0] if preview.window else None),
            window_end=(preview.window[1] if preview.window else None),
            cap=preview.cap,
            balance_before=preview.balance_before,
            charge=float(charge_dec),
            balance_after=float(self._balance) if preview.window else 0.0,
            reservation_suppressed=suppressed,
            reservation_admitted=not suppressed,
        )
        self._decisions.append(decision)
        return decision

    def finalize(self, simulation) -> tuple[OnlineAdmissionDecision, ...]:
        """Attach executor/timer outcomes without changing the reservation."""

        suppressed = {token_id for _time, token_id in simulation.hedge_suppressed_events}
        by_token = {token.token_id: token for token in simulation.tokens}
        finalized = []
        for decision in self._decisions:
            timer_status = "not_scheduled"
            executor_suppressed = decision.token_id in suppressed
            token = by_token[decision.token_id]
            if decision.applied is ProtectionAction.DELAYED_HEDGE:
                timer_status = "executor_suppressed" if executor_suppressed else (
                    "fired" if any(attempt.attempt_id == 2 for attempt in token.attempts)
                    else "voided"
                )
            elif decision.applied is ProtectionAction.IMMEDIATE_HEDGE:
                timer_status = "not_scheduled"
            finalized.append(replace_decision(
                decision,
                executor_suppressed=executor_suppressed,
                timer_status=timer_status,
            ))
        self._decisions = finalized
        return tuple(finalized)


def replace_decision(decision: OnlineAdmissionDecision, **changes) -> OnlineAdmissionDecision:
    values = {
        field: getattr(decision, field)
        for field in OnlineAdmissionDecision.__dataclass_fields__
    }
    values.update(changes)
    return OnlineAdmissionDecision(**values)


@dataclass(frozen=True)
class OnlineEpisodeResult:
    episode: EpisodeTrace
    simulation: EpisodeSimulationResult
    decisions: tuple[OnlineAdmissionDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, EpisodeTrace):
            raise ValueError("episode must be EpisodeTrace")
        if not isinstance(self.simulation, EpisodeSimulationResult):
            raise ValueError("simulation must be EpisodeSimulationResult")
        decisions = tuple(self.decisions)
        if len(decisions) != len(self.episode.workload.tokens):
            raise ValueError("one online decision is required per Token")
        if tuple(row.token_id for row in decisions) != tuple(
            token.token_id for token in self.episode.workload.tokens
        ):
            raise ValueError("online decisions must follow canonical Token order")
        object.__setattr__(self, "decisions", decisions)

    @property
    def applied_actions(self) -> dict[int, ProtectionAction]:
        return {row.token_id: row.applied for row in self.decisions}

    def audit(self) -> dict[str, object]:
        requested = sum(row.requested is not ProtectionAction.NORMAL for row in self.decisions)
        applied = sum(row.applied is not ProtectionAction.NORMAL for row in self.decisions)
        reservation_suppressed = sum(row.reservation_suppressed for row in self.decisions)
        reservation_admitted_hedges = sum(
            row.requested is not ProtectionAction.NORMAL and row.reservation_admitted
            for row in self.decisions
        )
        executor_suppressed = sum(row.executor_suppressed for row in self.decisions)
        if requested != applied + reservation_suppressed:
            raise RuntimeError("requested/applied/reservation identity failed")
        return {
            "requested_hedges": requested,
            "applied_hedges": applied,
            "reservation_suppressed": reservation_suppressed,
            "reservation_admitted_hedges": reservation_admitted_hedges,
            "executor_suppressed": executor_suppressed,
            "timer_scheduled": sum(row.timer_status != "not_scheduled" for row in self.decisions),
            "timer_fired": sum(row.timer_status == "fired" for row in self.decisions),
            "timer_voided": sum(row.timer_status == "voided" for row in self.decisions),
            "timer_executor_suppressed": sum(
                row.timer_status == "executor_suppressed" for row in self.decisions
            ),
            "reserved_work": math.fsum(row.charge for row in self.decisions),
        }


def _fresh_source(source: object) -> object:
    choose = getattr(source, "choose", None)
    if not callable(choose):
        raise ValueError("action_source must provide choose(observation, policy_key)")
    reset = getattr(source, "new_episode", None)
    if reset is None:
        return source
    if not callable(reset):
        raise ValueError("new_episode must be callable when present")
    fresh = reset()
    if fresh is None or not callable(getattr(fresh, "choose", None)):
        raise ValueError("new_episode must return a fresh action source")
    return fresh


def _episode_result(episode: EpisodeTrace, result, engine) -> EpisodeSimulationResult:
    failed = BoundarySnapshot(
        queued_attempts=engine.queue_length_at_failed_start,
        live_attempts=engine.live_attempts_at_failed_start,
        remaining_work=engine.remaining_work_at_failed_start,
    )
    recovered = BoundarySnapshot(
        queued_attempts=engine.queue_length_at_recovered_start,
        live_attempts=engine.live_attempts_at_recovered_start,
        remaining_work=engine.remaining_work_at_recovered_start,
    )
    return EpisodeSimulationResult(
        episode=episode,
        simulation=result,
        failed_start_snapshot=failed,
        recovered_start_snapshot=recovered,
        post_cutoff_drain_duration=max(
            0.0, result.drain_end_time - episode.protocol.arrival_cutoff
        ),
        degraded_slowdown=engine.slowdown,
        hedge_delay=engine.hedge_delay,
    )


def simulate_episode_online(
    episode: EpisodeTrace,
    action_source: TokenActionSource,
    parameters: ReservationParameters | None = None,
    *,
    degraded_slowdown: float = 2.0,
    hedge_delay: float | None = None,
    public_price: float = 0.0,
    cancel_running_losers: bool = False,
) -> OnlineEpisodeResult:
    """Run one fresh A/B episode and call the policy at each arrival exactly once."""

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    source = _fresh_source(action_source)
    ledger = ReservationLedger(episode.protocol.timeline, parameters)
    result, engine = _simulate_hedge_common_state_with_engine(
        episode.workload,
        episode.protocol.timeline,
        degraded_slowdown,
        actions=None,
        hedge_delay=hedge_delay,
        replica_count=2,
        online_source=source,
        reservation_ledger=ledger,
        public_price=public_price,
        cancel_running_losers=cancel_running_losers,
    )
    decisions = ledger.finalize(result)
    return OnlineEpisodeResult(
        episode=episode,
        simulation=_episode_result(episode, result, engine),
        decisions=decisions,
    )


__all__ = [
    "OnlineAdmissionDecision",
    "OnlineEpisodeResult",
    "ReservationLedger",
    "ReservationPreview",
    "StableRandomActionSource",
    "StaticActionSource",
    "StaticTimeClassActionSource",
    "TokenActionSource",
    "TokenObservation",
    "simulate_episode_online",
    "validate_token_action",
]
