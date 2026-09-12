"""Finite, pathwise requested-action interventions for one Token.

Every branch is a fresh complete T1 episode.  The runner changes only the
target Token's requested action after calling the base policy, while every
other Token continues to use its online policy function on the same immutable
CRN trace.  This module deliberately has no best-response, regret, or
equilibrium operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from typing import Callable, Iterable

from .attribution_episode import EpisodeTrace, episode_trace_fingerprint
from .domain import ProtectionAction
from .token_online import (
    OnlineEpisodeResult,
    TokenObservation,
    simulate_episode_online,
    validate_token_action,
)
from .token_payoff import (
    ADMITTED_RESERVED_WORK,
    ExternalQuote,
    TokenExtendedReservationParameters,
    TokenInitialPriceParameters,
    TokenPayoff,
    TokenRuntimeADR0009Parameters,
    _expected_basis,
    score_token_payoff,
)
from .transient_control import ReservationParameters
from .common_state import validate_slowdown


EVIDENCE_LABEL = "finite_token_pathwise_deviation"
_ACTION_ORDER = {
    ProtectionAction.NORMAL: 0,
    ProtectionAction.DELAYED_HEDGE: 1,
    ProtectionAction.IMMEDIATE_HEDGE: 2,
}


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _observation_payload(observation: TokenObservation) -> dict[str, object]:
    return {
        field.name: _json_value(getattr(observation, field.name))
        for field in fields(observation)
    }


def _json_value(value: object) -> object:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _observation_fingerprint(observation: TokenObservation) -> str:
    return _digest(_observation_payload(observation))


class _RunnerSource:
    """Runner-owned quote and one-target request adapter."""

    def __init__(
        self,
        source: object,
        target_token_id: int,
        override: ProtectionAction | None,
        quote: ExternalQuote,
    ) -> None:
        if not callable(getattr(source, "choose", None)):
            raise ValueError("policy factory must return an action source with choose")
        self.source = source
        self.target_token_id = target_token_id
        self.override = override
        self.quote = quote
        self.calls: list[tuple[TokenObservation, str, ProtectionAction, ProtectionAction]] = []

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        priced = replace(observation, public_price=self.quote.value)
        recommendation = validate_token_action(self.source.choose(priced, policy_key))
        applied_request = (
            self.override
            if self.override is not None and observation.token_id == self.target_token_id
            else recommendation
        )
        self.calls.append((priced, policy_key, recommendation, applied_request))
        return applied_request


@dataclass(frozen=True)
class TokenPathwiseDeviationRow:
    candidate_requested: ProtectionAction
    baseline_requested: ProtectionAction
    baseline_applied: ProtectionAction
    candidate_applied: ProtectionAction
    baseline_payoff: TokenPayoff
    candidate_payoff: TokenPayoff
    delta_cost: float
    pathwise_gain: float
    input_fingerprint: str
    baseline_target_observation_fingerprint: str
    candidate_target_observation_fingerprint: str
    baseline_prefix_fingerprint: str
    candidate_prefix_fingerprint: str
    policy_key: str
    baseline_other_actions: tuple[tuple[int, ProtectionAction], ...]
    candidate_other_actions: tuple[tuple[int, ProtectionAction], ...]
    baseline_run: OnlineEpisodeResult
    candidate_run: OnlineEpisodeResult
    status: str = "completed"


@dataclass(frozen=True)
class TokenPathwiseDeviationResult:
    evidence_label: str
    claims_best_response: bool
    claims_regret: bool
    claims_nash: bool
    claims_mfg: bool
    scope: str
    status: str
    episode: EpisodeTrace
    target_token_id: int
    candidate_actions: tuple[ProtectionAction, ...]
    baseline_run: OnlineEpisodeResult | None
    baseline_payoff: TokenPayoff | None
    rows: tuple[TokenPathwiseDeviationRow, ...]
    attempted_calls: int
    input_fingerprint: str
    baseline_run_fingerprint: str
    failure_reason: str | None = None
    failed_cost: None = None

    def row_for(self, action: ProtectionAction) -> TokenPathwiseDeviationRow:
        action = validate_token_action(action)
        for row in self.rows:
            if row.candidate_requested is action:
                return row
        raise KeyError(action)


def _validate_candidates(candidates: Iterable[ProtectionAction] | None) -> tuple[ProtectionAction, ...]:
    values = (
        tuple(ProtectionAction)
        if candidates is None
        else tuple(validate_token_action(action) for action in candidates)
    )
    if not values:
        raise ValueError("candidates must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError("candidate actions must be unique")
    return tuple(sorted(values, key=_ACTION_ORDER.__getitem__))


def _validate_inputs(
    episode: EpisodeTrace,
    policy_factory: Callable[[], object],
    target_token_id: int,
    candidates: Iterable[ProtectionAction] | None,
    parameters: object,
    quote: ExternalQuote | None,
    reservation: ReservationParameters | None,
    degraded_slowdown: float,
    hedge_delay: float | None,
) -> tuple[tuple[ProtectionAction, ...], ExternalQuote, str]:
    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    if not callable(policy_factory):
        raise ValueError("policy_factory must be callable and return fresh state")
    if type(target_token_id) is not int or target_token_id < 0:
        raise ValueError("target_token_id must be a non-negative int")
    if target_token_id not in {token.token_id for token in episode.workload.tokens}:
        raise ValueError("target_token_id is absent from episode")
    candidate_actions = _validate_candidates(candidates)
    if type(parameters) not in {
        TokenInitialPriceParameters,
        TokenRuntimeADR0009Parameters,
        TokenExtendedReservationParameters,
    }:
        raise ValueError("parameters must be one of the three named Token cost models")
    expected_basis = _expected_basis(parameters)
    quote = ExternalQuote(0.0, expected_basis) if quote is None else quote
    if not isinstance(quote, ExternalQuote) or quote.basis != expected_basis:
        raise ValueError("quote basis is incompatible with parameters")
    if reservation is not None and not isinstance(reservation, ReservationParameters):
        raise ValueError("reservation must be ReservationParameters")
    validate_slowdown(degraded_slowdown)
    if hedge_delay is not None:
        if isinstance(hedge_delay, bool) or not isinstance(hedge_delay, (int, float)):
            raise ValueError("hedge_delay must be a finite positive real")
        if not math.isfinite(float(hedge_delay)) or float(hedge_delay) <= 0.0:
            raise ValueError("hedge_delay must be a finite positive real")
    if ProtectionAction.DELAYED_HEDGE in candidate_actions and hedge_delay is None:
        raise ValueError("hedge_delay is required when D is a candidate")
    return candidate_actions, quote, _digest(episode_trace_fingerprint(episode))


def _run_branch(
    episode: EpisodeTrace,
    policy_factory: Callable[[], object],
    used_sources: list[object],
    target_token_id: int,
    override: ProtectionAction | None,
    quote: ExternalQuote,
    parameters: object,
    reservation: ReservationParameters | None,
    degraded_slowdown: float,
    hedge_delay: float | None,
    input_fingerprint: str,
) -> tuple[OnlineEpisodeResult, TokenPayoff, _RunnerSource]:
    source = policy_factory()
    if source is None or not callable(getattr(source, "choose", None)):
        raise ValueError("policy factory returned an invalid action source")
    if any(source is previous for previous in used_sources):
        raise ValueError("policy factory reused mutable source state across branches")
    used_sources.append(source)
    runner = _RunnerSource(source, target_token_id, override, quote)
    before = _digest(episode_trace_fingerprint(episode))
    run = simulate_episode_online(
        episode,
        runner,
        reservation,
        degraded_slowdown=degraded_slowdown,
        hedge_delay=hedge_delay,
    )
    after = _digest(episode_trace_fingerprint(episode))
    if before != after or before != input_fingerprint:
        raise ValueError("episode trace changed during counterfactual run")
    payoff = score_token_payoff(run, target_token_id, parameters, quote)
    return run, payoff, runner


def _target_call(runner: _RunnerSource, target_token_id: int):
    target = [call for call in runner.calls if call[0].token_id == target_token_id]
    if len(target) != 1:
        raise ValueError("target Token must receive exactly one policy decision")
    return target[0]


def _prefix_fingerprint(runner: _RunnerSource, target_token_id: int) -> str:
    target_index = next(
        index for index, call in enumerate(runner.calls)
        if call[0].token_id == target_token_id
    )
    return _digest(
        [
            {
                "observation": _observation_payload(call[0]),
                "policy_key": call[1],
                "recommendation": call[2].value,
                "request": call[3].value,
            }
            for call in runner.calls[:target_index]
        ]
    )


def evaluate_token_pathwise_deviations(
    episode: EpisodeTrace,
    policy_factory: Callable[[], object],
    target_token_id: int,
    candidates: Iterable[ProtectionAction] | None = None,
    parameters: object | None = None,
    quote: ExternalQuote | None = None,
    reservation: ReservationParameters | None = None,
    degraded_slowdown: float = 2.0,
    hedge_delay: float | None = None,
) -> TokenPathwiseDeviationResult:
    """Run baseline plus canonical requested-action interventions for one Token."""

    parameters = TokenInitialPriceParameters() if parameters is None else parameters
    candidate_actions, quote, input_fingerprint = _validate_inputs(
        episode,
        policy_factory,
        target_token_id,
        candidates,
        parameters,
        quote,
        reservation,
        degraded_slowdown,
        hedge_delay,
    )
    used_sources: list[object] = []
    attempted = 0
    baseline_run = None
    baseline_payoff = None
    baseline_runner = None
    rows: list[TokenPathwiseDeviationRow] = []
    failure_reason = None

    for override in (None, *candidate_actions):
        attempted += 1
        try:
            run, payoff, runner = _run_branch(
                episode,
                policy_factory,
                used_sources,
                target_token_id,
                override,
                quote,
                parameters,
                reservation,
                degraded_slowdown,
                hedge_delay,
                input_fingerprint,
            )
        except Exception as error:  # retain attempted call and fail closed
            failure_reason = f"{type(error).__name__}: {error}"
            break
        if override is None:
            baseline_run, baseline_payoff, baseline_runner = run, payoff, runner
            continue
        if baseline_run is None or baseline_payoff is None or baseline_runner is None:
            failure_reason = "baseline must complete before candidate scoring"
            break
        baseline_target = _target_call(baseline_runner, target_token_id)
        candidate_target = _target_call(runner, target_token_id)
        if candidate_target[1] != baseline_target[1]:
            failure_reason = "candidate changed the stable target policy key"
            break
        same_action = override is baseline_target[2]
        if same_action and run != baseline_run:
            failure_reason = "same-action intervention did not reproduce baseline"
            break
        baseline_other = tuple(
            (call[0].token_id, call[3])
            for call in baseline_runner.calls
            if call[0].token_id != target_token_id
        )
        candidate_other = tuple(
            (call[0].token_id, call[3])
            for call in runner.calls
            if call[0].token_id != target_token_id
        )
        rows.append(
            TokenPathwiseDeviationRow(
                candidate_requested=override,
                baseline_requested=baseline_target[2],
                baseline_applied=baseline_payoff.applied_action,
                candidate_applied=payoff.applied_action,
                baseline_payoff=baseline_payoff,
                candidate_payoff=payoff,
                delta_cost=payoff.total - baseline_payoff.total,
                pathwise_gain=baseline_payoff.total - payoff.total,
                input_fingerprint=input_fingerprint,
                baseline_target_observation_fingerprint=_observation_fingerprint(baseline_target[0]),
                candidate_target_observation_fingerprint=_observation_fingerprint(candidate_target[0]),
                baseline_prefix_fingerprint=_prefix_fingerprint(baseline_runner, target_token_id),
                candidate_prefix_fingerprint=_prefix_fingerprint(runner, target_token_id),
                policy_key=baseline_target[1],
                baseline_other_actions=baseline_other,
                candidate_other_actions=candidate_other,
                baseline_run=baseline_run,
                candidate_run=run,
            )
        )

    status = "completed" if failure_reason is None else "failed"
    return TokenPathwiseDeviationResult(
        evidence_label=EVIDENCE_LABEL,
        claims_best_response=False,
        claims_regret=False,
        claims_nash=False,
        claims_mfg=False,
        scope="one_episode:one_token:requested_action",
        status=status,
        episode=episode,
        target_token_id=target_token_id,
        candidate_actions=candidate_actions,
        baseline_run=baseline_run,
        baseline_payoff=baseline_payoff,
        rows=tuple(rows),
        attempted_calls=attempted,
        input_fingerprint=input_fingerprint,
        baseline_run_fingerprint=input_fingerprint,
        failure_reason=failure_reason,
    )


__all__ = [
    "EVIDENCE_LABEL",
    "TokenPathwiseDeviationResult",
    "TokenPathwiseDeviationRow",
    "evaluate_token_pathwise_deviations",
]
