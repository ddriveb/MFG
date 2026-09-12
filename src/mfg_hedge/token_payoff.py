"""Strict, pathwise private-cost scoring for the restored Token player.

This module scores a completed T1 online episode.  It never runs the event
engine, calls a policy, infers a model from numeric fields, or aggregates
multiple episodes.  The three model IDs are deliberately separate provenance
identities even when two parameterizations happen to produce the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .attribution_episode import episode_trace_fingerprint
from .domain import ProtectionAction, TokenClass
from .hedge_simulation import HedgeAttemptStatus
from .token_online import OnlineEpisodeResult


MODEL_INITIAL = "token_initial_price_v0"
MODEL_RUNTIME = "token_runtime_adr0009_v1"
MODEL_EXTENDED = "token_extended_reservation_v1"
WITHDRAWN_MODEL = "token_original_runtime_v1"

INCREMENTAL_EXECUTED_WORK = "incremental_executed_work"
ADMITTED_RESERVED_WORK = "admitted_reserved_work"


def _real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite and non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _token_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("token_id must be a non-negative int")
    return value


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


@dataclass(frozen=True)
class ExternalQuote:
    """Immutable episode-constant external price and its model-specific unit."""

    value: float = 0.0
    basis: str = INCREMENTAL_EXECUTED_WORK

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _real(self.value, "quote.value"))
        if self.basis not in {INCREMENTAL_EXECUTED_WORK, ADMITTED_RESERVED_WORK}:
            raise ValueError("quote.basis is not a supported price basis")


@dataclass(frozen=True)
class TokenInitialPriceParameters:
    gamma_regular: float = 1.0
    gamma_urgent: float = 5.0

    @property
    def model_id(self) -> str:
        return MODEL_INITIAL

    @property
    def schema_id(self) -> str:
        return "token_private_cost_schema_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma_regular", _real(self.gamma_regular, "gamma_regular"))
        object.__setattr__(self, "gamma_urgent", _real(self.gamma_urgent, "gamma_urgent"))


@dataclass(frozen=True)
class TokenRuntimeADR0009Parameters:
    gamma_regular: float = 1.0
    gamma_urgent: float = 5.0
    c_inc: float = 1.0
    c_waste: float = 1.0

    @property
    def model_id(self) -> str:
        return MODEL_RUNTIME

    @property
    def schema_id(self) -> str:
        return "token_private_cost_schema_v1"

    def __post_init__(self) -> None:
        for name in ("gamma_regular", "gamma_urgent", "c_inc", "c_waste"):
            object.__setattr__(self, name, _real(getattr(self, name), name))


@dataclass(frozen=True)
class TokenExtendedReservationParameters:
    gamma_regular: float = 1.0
    gamma_urgent: float = 5.0
    alpha_regular: float = 1.0
    alpha_urgent: float = 5.0
    deadline_regular: float = 3.0
    deadline_urgent: float = 2.0
    c_W: float = 1.0
    c_waste: float = 1.0

    @property
    def model_id(self) -> str:
        return MODEL_EXTENDED

    @property
    def schema_id(self) -> str:
        return "token_private_cost_schema_v1"

    def __post_init__(self) -> None:
        for name in (
            "gamma_regular",
            "gamma_urgent",
            "alpha_regular",
            "alpha_urgent",
            "c_W",
            "c_waste",
        ):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        for name in ("deadline_regular", "deadline_urgent"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))


# Short aliases make the model names readable at call sites without merging
# their provenance identities.
InitialPriceParameters = TokenInitialPriceParameters
RuntimeADR0009Parameters = TokenRuntimeADR0009Parameters
ExtendedReservationParameters = TokenExtendedReservationParameters


def build_token_cost_parameters(model_id: str, **kwargs: Any) -> object:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if model_id == MODEL_INITIAL:
        return TokenInitialPriceParameters(**kwargs)
    if model_id == MODEL_RUNTIME:
        return TokenRuntimeADR0009Parameters(**kwargs)
    if model_id == MODEL_EXTENDED:
        return TokenExtendedReservationParameters(**kwargs)
    raise ValueError(f"unknown or withdrawn Token cost model: {model_id!r}")


def _validate_parameters(parameters: object) -> None:
    if type(parameters) not in {
        TokenInitialPriceParameters,
        TokenRuntimeADR0009Parameters,
        TokenExtendedReservationParameters,
    }:
        raise ValueError("parameters must be one of the three named Token cost models")


def _expected_basis(parameters: object) -> str:
    return (
        ADMITTED_RESERVED_WORK
        if type(parameters) is TokenExtendedReservationParameters
        else INCREMENTAL_EXECUTED_WORK
    )


def _parameter_payload(parameters: object) -> dict[str, object]:
    _validate_parameters(parameters)
    fields = {
        name: getattr(parameters, name)
        for name in parameters.__dataclass_fields__
    }
    return {
        "model_id": parameters.model_id,
        "schema_id": parameters.schema_id,
        "parameters": fields,
    }


def _gamma(parameters: object, token_class: TokenClass) -> float:
    if not isinstance(token_class, TokenClass):
        raise ValueError("token_class must be TokenClass")
    return (
        parameters.gamma_urgent
        if token_class is TokenClass.URGENT
        else parameters.gamma_regular
    )


@dataclass(frozen=True)
class TokenPayoff:
    model_id: str
    schema_id: str
    provenance_fingerprint: str
    input_fingerprint: str
    token_id: int
    token_class: TokenClass
    arrival_time: float
    requested_action: ProtectionAction
    applied_action: ProtectionAction
    reservation_charge: float
    reservation_suppressed: bool
    executor_suppressed: bool
    timer_status: str
    winner_attempt_id: int
    completion_time: float
    settlement_time: float
    latency: float
    replay_count: int
    slo_excess: float | None
    w_primary: float
    w_replay: float
    w_hedge: float
    w_all: float
    w_lose: float
    latency_cost: float
    slo_cost: float
    replay_cost: float
    persistent_work_cost: float
    waste_cost: float
    price_cost: float
    total: float
    price_value: float
    price_basis: str
    attempts: tuple[object, ...]
    completed: bool = True

    @property
    def total_cost(self) -> float:
        return self.total


def _attempt_payload(attempt: object) -> dict[str, object]:
    return {
        "token_id": attempt.token_id,
        "attempt_id": attempt.attempt_id,
        "replica_id": attempt.replica_id,
        "enqueue_time": attempt.enqueue_time,
        "start_time": attempt.start_time,
        "terminal_time": attempt.terminal_time,
        "required_work": attempt.required_work,
        "executed_work": attempt.executed_work,
        "remaining_work": attempt.remaining_work,
        "status": _enum_value(attempt.status),
        "queue_delay": attempt.queue_delay,
    }


def _validate_completed_run(run: object, token_id: int):
    if not isinstance(run, OnlineEpisodeResult):
        raise ValueError("completed_run must be an OnlineEpisodeResult")
    episode = run.episode
    tokens = {token.token_id: token for token in episode.workload.tokens}
    if token_id not in tokens:
        raise ValueError("token_id is absent from the completed run")
    if len(run.decisions) != len(tokens):
        raise ValueError("completed run must contain one decision per Token")
    decisions = {decision.token_id: decision for decision in run.decisions}
    if set(decisions) != set(tokens):
        raise ValueError("completed run decisions do not match the input Tokens")
    simulation = run.simulation.simulation
    token = next((row for row in simulation.tokens if row.token_id == token_id), None)
    if token is None:
        raise ValueError("completed run is missing the target Token result")
    decision = decisions[token_id]
    if (
        token.token_class is not decision.token_class
        or token.arrival_time != decision.arrival_time
        or token.primary_replica != decision.primary_replica
        or token.action is not decision.applied
    ):
        raise ValueError("Token result and online admission disagree")
    if token.winner_attempt_id not in {attempt.attempt_id for attempt in token.attempts}:
        raise ValueError("winner attempt is absent from the target attempts")
    winner_count = sum(
        attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
        for attempt in token.attempts
    )
    if winner_count != 1:
        raise ValueError("completed Token must have exactly one winner")
    winner_record = next(
        attempt
        for attempt in token.attempts
        if attempt.status is HedgeAttemptStatus.COMPLETED_WINNER
    )
    if winner_record.attempt_id != token.winner_attempt_id:
        raise ValueError("winner_attempt_id does not identify the completed winner")
    if not type(token.replay_count) is int or token.replay_count not in (0, 1):
        raise ValueError("replay_count must be 0 or 1")
    attempt_ids = [attempt.attempt_id for attempt in token.attempts]
    if len(set(attempt_ids)) != len(attempt_ids) or any(
        type(attempt_id) is not int or attempt_id not in (0, 1, 2)
        for attempt_id in attempt_ids
    ):
        raise ValueError("attempt identity is invalid or duplicated")
    if 0 not in attempt_ids or (token.replay_count and 1 not in attempt_ids):
        raise ValueError("required Primary/Replay attempt is missing")
    global_attempts = sorted(
        (_attempt_payload(attempt) for attempt in simulation.attempts if attempt.token_id == token_id),
        key=lambda row: row["attempt_id"],
    )
    local_attempts = sorted(
        (_attempt_payload(attempt) for attempt in token.attempts),
        key=lambda row: row["attempt_id"],
    )
    if global_attempts != local_attempts:
        raise ValueError("per-Token and global attempt records disagree")
    for attempt in token.attempts:
        if not isinstance(attempt.status, HedgeAttemptStatus):
            raise ValueError("attempt status is not a terminal HedgeAttemptStatus")
        for name in ("enqueue_time", "terminal_time", "required_work", "executed_work", "remaining_work"):
            _real(getattr(attempt, name), f"attempt.{name}")
        if attempt.terminal_time < attempt.enqueue_time:
            raise ValueError("attempt terminal time precedes enqueue")
        if attempt.start_time is not None:
            _real(attempt.start_time, "attempt.start_time")
            if attempt.start_time < attempt.enqueue_time:
                raise ValueError("attempt start time precedes enqueue")
        if attempt.executed_work > attempt.required_work + 1e-12:
            raise ValueError("executed work exceeds required work")
        if abs(attempt.remaining_work - (attempt.required_work - attempt.executed_work)) > 1e-12:
            raise ValueError("attempt work accounting is inconsistent")
        if attempt.start_time is None and (
            attempt.status not in {
                HedgeAttemptStatus.CANCELLED_QUEUED,
                HedgeAttemptStatus.INVALIDATED_QUEUED,
            }
            or attempt.executed_work != 0.0
        ):
            raise ValueError("queued terminal attempt has invalid status/work")
    if token.replay_count == 0 and 1 in attempt_ids:
        raise ValueError("Replay attempt exists with replay_count zero")
    if token.replay_count == 1 and not any(attempt.attempt_id == 1 for attempt in token.attempts):
        raise ValueError("Replay count does not have a Replay attempt")
    settlement = max(attempt.terminal_time for attempt in token.attempts)
    if simulation.drain_end_time + 1e-12 < settlement:
        raise ValueError("drain ends before Token settlement")
    _real(token.completion_time, "completion_time")
    _real(token.latency, "latency")
    if abs(token.completion_time - winner_record.terminal_time) > 1e-12:
        raise ValueError("completion_time disagrees with the winner terminal time")
    if abs(token.latency - (token.completion_time - token.arrival_time)) > 1e-12:
        raise ValueError("latency disagrees with arrival and winner completion")
    return episode, token, decision, tuple(sorted(token.attempts, key=lambda row: row.attempt_id))


def score_token_payoff(
    completed_run: OnlineEpisodeResult,
    token_id: int,
    parameters: object,
    quote: ExternalQuote | None = None,
) -> TokenPayoff:
    """Score one completely drained Token without performing a simulation."""

    token_id = _token_id(token_id)
    _validate_parameters(parameters)
    expected_basis = _expected_basis(parameters)
    quote = ExternalQuote(0.0, expected_basis) if quote is None else quote
    if not isinstance(quote, ExternalQuote):
        raise ValueError("quote must be ExternalQuote")
    if quote.basis != expected_basis:
        raise ValueError("quote basis is incompatible with the selected model")
    episode, token, decision, attempts = _validate_completed_run(completed_run, token_id)
    input_fingerprint = _digest(episode_trace_fingerprint(episode))

    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    winner = by_id[token.winner_attempt_id]
    w_primary = by_id[0].executed_work
    w_replay = by_id.get(1).executed_work if 1 in by_id else 0.0
    w_hedge = by_id.get(2).executed_work if 2 in by_id else 0.0
    w_all = math.fsum((w_primary, w_replay, w_hedge))
    w_lose = math.fsum(
        attempt.executed_work
        for attempt in attempts
        if attempt.attempt_id != token.winner_attempt_id
    )
    latency = _real(token.latency, "latency")
    settlement = max(attempt.terminal_time for attempt in attempts)
    replay_cost = _gamma(parameters, token.token_class) * token.replay_count
    latency_cost = latency
    slo_excess: float | None = None
    slo_cost = 0.0
    persistent_work_cost = 0.0
    waste_cost = 0.0
    if type(parameters) is TokenRuntimeADR0009Parameters:
        persistent_work_cost = parameters.c_inc * math.fsum((w_hedge, w_replay))
        waste_cost = parameters.c_waste * w_lose
    elif type(parameters) is TokenExtendedReservationParameters:
        deadline = (
            parameters.deadline_urgent
            if token.token_class is TokenClass.URGENT
            else parameters.deadline_regular
        )
        slo_excess = max(0.0, latency - deadline)
        alpha = (
            parameters.alpha_urgent
            if token.token_class is TokenClass.URGENT
            else parameters.alpha_regular
        )
        slo_cost = alpha * slo_excess
        persistent_work_cost = parameters.c_W * w_all
        waste_cost = parameters.c_waste * w_lose
    price_cost = quote.value * (
        decision.charge if type(parameters) is TokenExtendedReservationParameters
        else math.fsum((w_hedge, w_replay))
    )
    total = math.fsum((
        latency_cost,
        slo_cost,
        replay_cost,
        persistent_work_cost,
        waste_cost,
        price_cost,
    ))
    payload = {
        "parameters": _parameter_payload(parameters),
        "quote": {"value": quote.value, "basis": quote.basis},
        "input_fingerprint": input_fingerprint,
        "token": {
            "token_id": token_id,
            "class": token.token_class.value,
            "arrival": token.arrival_time,
            "requested": decision.requested.value,
            "applied": decision.applied.value,
            "charge": decision.charge,
            "winner": token.winner_attempt_id,
            "settlement": settlement,
        },
    }
    return TokenPayoff(
        model_id=parameters.model_id,
        schema_id=parameters.schema_id,
        provenance_fingerprint=_digest(payload),
        input_fingerprint=input_fingerprint,
        token_id=token_id,
        token_class=token.token_class,
        arrival_time=token.arrival_time,
        requested_action=decision.requested,
        applied_action=decision.applied,
        reservation_charge=decision.charge,
        reservation_suppressed=decision.reservation_suppressed,
        executor_suppressed=decision.executor_suppressed,
        timer_status=decision.timer_status,
        winner_attempt_id=token.winner_attempt_id,
        completion_time=token.completion_time,
        settlement_time=settlement,
        latency=latency,
        replay_count=token.replay_count,
        slo_excess=slo_excess,
        w_primary=w_primary,
        w_replay=w_replay,
        w_hedge=w_hedge,
        w_all=w_all,
        w_lose=w_lose,
        latency_cost=latency_cost,
        slo_cost=slo_cost,
        replay_cost=replay_cost,
        persistent_work_cost=persistent_work_cost,
        waste_cost=waste_cost,
        price_cost=price_cost,
        total=total,
        price_value=quote.value,
        price_basis=quote.basis,
        attempts=attempts,
    )


__all__ = [
    "ADMITTED_RESERVED_WORK",
    "ExternalQuote",
    "ExtendedReservationParameters",
    "INCREMENTAL_EXECUTED_WORK",
    "InitialPriceParameters",
    "MODEL_EXTENDED",
    "MODEL_INITIAL",
    "MODEL_RUNTIME",
    "RuntimeADR0009Parameters",
    "TokenExtendedReservationParameters",
    "TokenInitialPriceParameters",
    "TokenPayoff",
    "TokenRuntimeADR0009Parameters",
    "WITHDRAWN_MODEL",
    "build_token_cost_parameters",
    "score_token_payoff",
]
