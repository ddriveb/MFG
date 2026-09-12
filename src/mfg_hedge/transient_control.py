"""Restricted causal development control, separate from historical quota.

ADR-0012: time/class requests followed by a pooled prospective reservation.
The metadata-only prepass is equivalent to online admission for these fixed
rules: no decision uses queue state, service draws, or a future arrival. This
does NOT implement general feedback control, policy search, or an MFG solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from typing import Iterable

from .attribution_episode import EpisodeTrace, EpisodeSimulationResult, simulate_episode
from .common_state import CommonStateTimeline, Phase, phase_at
from .domain import ProtectionAction as Action, TokenClass
from .workload import TokenSpec


def _real(value: float, name: str, *, zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (value == 0 and not zero)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} real")
    return float(value)


@dataclass(frozen=True)
class ReservationParameters:
    window_width: float = 25.0
    budget_rate: float = 0.45
    scale: float = 0.25
    mean_requirement: float = 1.0

    def __post_init__(self) -> None:
        for name in ("window_width", "budget_rate", "scale", "mean_requirement"):
            object.__setattr__(self, name, _real(getattr(self, name), name, zero=name == "scale"))
        if self.scale > 1:
            raise ValueError("scale must be <= 1")


@dataclass(frozen=True)
class TimeClassRule:
    """One fixed rule; declaration order is early R/U, late R/U. No callbacks."""

    early_regular: Action = Action.NORMAL
    early_urgent: Action = Action.NORMAL
    late_regular: Action = Action.NORMAL
    late_urgent: Action = Action.NORMAL

    def __post_init__(self) -> None:
        for action in self.actions:
            if not isinstance(action, Action):
                raise ValueError("rule entries must be ProtectionAction values")

    @property
    def actions(self) -> tuple[Action, ...]:
        return (self.early_regular, self.early_urgent, self.late_regular, self.late_urgent)


@dataclass(frozen=True)
class AdmissionDecision:
    token_id: int
    arrival_time: float
    token_class: TokenClass
    phase: Phase
    primary_replica: int
    requested: Action
    applied: Action
    window_start: float | None
    window_end: float | None
    cap: float
    balance_before: float
    charge: float
    balance_after: float
    quota_suppressed: bool


def _counts(decisions: Iterable[AdmissionDecision]) -> dict:
    rows = tuple(decisions)
    requested = {a.value: sum(d.requested is a for d in rows) for a in Action}
    applied = {a.value: sum(d.applied is a for d in rows) for a in Action}
    req = requested["D"] + requested["I"]
    app = applied["D"] + applied["I"]
    suppressed = sum(d.quota_suppressed for d in rows)
    if req != app + suppressed:
        raise RuntimeError("requested/applied/quota_suppressed identity failed")
    return {"token_count": len(rows), "requested": requested, "applied": applied,
            "requested_hedges": req, "applied_hedges": app,
            "quota_suppressed": suppressed,
            "reserved_work": math.fsum(d.charge for d in rows)}


@dataclass(frozen=True)
class AdmissionPlan:
    decisions: tuple[AdmissionDecision, ...]
    parameters: ReservationParameters

    def applied_actions(self) -> dict[int, Action]:
        return {d.token_id: d.applied for d in self.decisions}

    def audit(self) -> dict:
        """Fresh JSON-ready audit; empty D windows are not materialized."""
        def classes(rows):
            return {k.name.lower(): _counts(d for d in rows if d.token_class is k)
                    for k in TokenClass}
        windows = []
        starts = sorted({d.window_start for d in self.decisions if d.window_start is not None})
        for start in starts:
            rows = [d for d in self.decisions if d.window_start == start]
            windows.append({"start": start, "end": rows[0].window_end,
                            "cap": rows[0].cap, "unused_at_close": rows[-1].balance_after,
                            "counts": _counts(rows), "by_class": classes(rows)})
        return {"contract": "pooled_nonrefundable_mean_work_reservation_v1",
                "pathwise_executed_work_cap": False,
                "window_coverage": "windows_with_at_least_one_D_arrival",
                "global": _counts(self.decisions), "by_class": classes(self.decisions),
                "windows": windows}


def plan_transient_actions(
    tokens: Iterable[TokenSpec], timeline: CommonStateTimeline, rule: TimeClassRule,
    parameters: ReservationParameters | None = None,
) -> AdmissionPlan:
    """Consume arrival metadata in order; never receive a WorkloadTrace.

    Timelines are scenario metadata known in advance. Only the current Token
    enters each decision. Equal arrivals retain dense Token-ID FIFO order.
    Decimal debit math is isolated from the caller's decimal context.
    """
    if not isinstance(timeline, CommonStateTimeline):
        raise ValueError("timeline must be CommonStateTimeline")
    if type(rule) is not TimeClassRule:
        raise ValueError("rule must be a fixed TimeClassRule, not a policy callback")
    params = ReservationParameters() if parameters is None else parameters
    if not isinstance(params, ReservationParameters):
        raise ValueError("parameters must be ReservationParameters")
    rows = []
    previous = -1.0
    with localcontext() as context:
        context.prec = 80
        dec = lambda x: Decimal(str(x))
        start, stop = dec(timeline.degraded_start), dec(timeline.failed_start)
        width = dec(params.window_width)
        rate = dec(params.budget_rate) * dec(params.scale)
        charge = dec(params.mean_requirement)
        active_window = None
        balance = Decimal(0)
        for expected_id, token in enumerate(tokens):
            if (not isinstance(token, TokenSpec) or type(token.token_id) is not int
                    or token.token_id != expected_id):
                raise ValueError("Token IDs must be true ints in dense arrival order")
            now = _real(token.arrival_time, "arrival_time", zero=True)
            if now < previous or not isinstance(token.token_class, TokenClass):
                raise ValueError("arrivals must be nondecreasing and class must be TokenClass")
            previous = now
            phase = phase_at(timeline, now)
            primary = 1 if phase is Phase.FAILED else token.token_id % 2
            requested = Action.NORMAL
            window_start = window_end = None
            cap = before = after = debit = Decimal(0)
            if phase is Phase.DEGRADED:
                time = dec(now)
                index = (time - start) // width
                ws = start + index * width
                we = min(stop, ws + width)
                cap = rate * (we - ws)
                if not math.isfinite(float(cap)):
                    raise ValueError("window reservation cap overflow")
                if active_window != ws:
                    active_window, balance = ws, cap
                window_start, window_end = float(ws), float(we)
                before = balance
                if primary == 0:
                    late = int(time >= (start + stop) / 2)
                    urgent = int(token.token_class is TokenClass.URGENT)
                    requested = rule.actions[2 * late + urgent]
                if requested is not Action.NORMAL and balance >= charge:
                    debit = charge
                    balance -= debit
                after = balance
            applied = requested if debit else Action.NORMAL
            rows.append(AdmissionDecision(
                token.token_id, now, token.token_class, phase, primary, requested, applied,
                window_start, window_end, float(cap), float(before), float(debit),
                float(after), requested is not Action.NORMAL and applied is Action.NORMAL,
            ))
    return AdmissionPlan(tuple(rows), params)


@dataclass(frozen=True)
class TransientEpisodeResult:
    rule: TimeClassRule
    plan: AdmissionPlan
    run: EpisodeSimulationResult


def simulate_transient_episode(
    episode: EpisodeTrace, rule: TimeClassRule,
    parameters: ReservationParameters | None = None, *,
    degraded_slowdown: float = 2.0, hedge_delay: float | None = None,
) -> TransientEpisodeResult:
    """Apply the composite rule/ledger, then evaluate its complete physical path.

    For Delayed rules callers supply the predeclared delay (compute_tau0 from
    the service distribution in the standard development scenario). Never infer
    a delay, charge, or budget from the sampled service streams.
    """
    if not isinstance(episode, EpisodeTrace):
        raise ValueError("episode must be EpisodeTrace")
    if episode.identity.namespace not in ("transient-control:v1:fit", "transient-control:v1:test"):
        raise ValueError("namespace must be transient-control:v1:fit or transient-control:v1:test")
    plan = plan_transient_actions(episode.workload.tokens, episode.protocol.timeline, rule, parameters)
    if Action.DELAYED_HEDGE in rule.actions and hedge_delay is None:
        raise ValueError("hedge_delay must be declared for a Delayed rule")
    run = simulate_episode(episode, degraded_slowdown, plan.applied_actions(), hedge_delay)
    if run.simulation.hedge_requested != plan.audit()["global"]["applied_hedges"]:
        raise RuntimeError("engine hedge_requested must equal applied Hedge count")
    return TransientEpisodeResult(rule, plan, run)
