"""Test-only observation of accepted physics; never a production controller.

The observer integrates busy time before events, records admitted/removal work,
and compares that independent ledger with live queue state after each handler.
It never settles or changes the engine's running-copy state. Monkeypatching is
scoped to one synchronous test call and the full public result is compared to
an unobserved run. Private-hook coupling is deliberate and limited to tests.
"""

from dataclasses import dataclass
import math
from unittest.mock import patch

import mfg_hedge.hedge_simulation as simulation
from mfg_hedge.domain import ProtectionAction as Action, TokenClass
from mfg_hedge.workload import TokenSpec, WorkloadTrace


@dataclass(frozen=True)
class AuditFrame:
    time: float
    label: str
    admitted: tuple
    executed: tuple
    cancelled: tuple
    discarded: tuple
    remaining: tuple
    queued: tuple
    running: tuple


class AuditedEngine(simulation._Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.admitted = [0.0, 0.0]
        self.executed = [0.0, 0.0]
        self.cancelled = [0.0, 0.0]
        self.discarded = [0.0, 0.0]
        self.service_by_copy = {}
        self.service_by_phase_replica = {}
        self.frames = []
        self._audit_time = 0.0

    def frame(self, time, label):
        matches = [f for f in self.frames if f.time == time and f.label == label]
        if not matches:
            raise AssertionError(f"no audit frame {time}/{label}")
        return matches[0]

    def _advance_audit(self, now):
        dt = now - self._audit_time
        if dt < 0:
            raise AssertionError("audit time went backwards")
        if dt:
            t = self._audit_time
            phase = ("H" if t < self.timeline.degraded_start else
                     "D" if t < self.timeline.failed_start else
                     "F" if t < self.timeline.recovered_start else "R")
            for replica, running in enumerate(self.running):
                if running is None:
                    continue
                speed = (1.0 if replica == 1 or phase in ("H", "R") else
                         1.0 / self.slowdown if phase == "D" else 0.0)
                if speed == 0:
                    raise AssertionError("service on failed Replica A")
                work = speed * dt
                key = (running.token_id, running.attempt_id)
                self.service_by_copy[key] = self.service_by_copy.get(key, 0.0) + work
                cell = (phase, replica)
                self.service_by_phase_replica[cell] = (
                    self.service_by_phase_replica.get(cell, 0.0) + work)
                self.executed[replica] += work
        self._audit_time = now

    def _check_audit(self, now, label):
        remaining, queues, running_keys = [], [], []
        live_keys = set()
        for replica in range(2):
            queue = [q for q in self.queues[replica] if not q.cancelled]
            queue_keys = tuple((q.token_id, q.attempt_id) for q in queue)
            queues.append(queue_keys)
            work = math.fsum(q.required_work for q in queue)
            running = self.running[replica]
            current_key = None
            if running is not None:
                current_key = (running.token_id, running.attempt_id)
                # Pure projection: do not call the engine's _settle method.
                residual = (running.required_work - running.executed_work
                            - (now - running.last_update_time) * running.current_speed)
                if residual < -1e-9:
                    raise AssertionError("negative running residual")
                work += residual
            running_keys.append(current_key)
            for key in queue_keys + (() if current_key is None else (current_key,)):
                if key in live_keys:
                    raise AssertionError("duplicate live copy")
                live_keys.add(key)
            ledger = (self.admitted[replica] - self.executed[replica]
                      - self.cancelled[replica] - self.discarded[replica])
            if not math.isclose(ledger, work, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(
                    f"conservation at {now}/{label}/replica={replica}: "
                    f"ledger={ledger}, live={work}")
            remaining.append(work)
        token_live = {(t.token_id, attempt_id) for t in self.tokens
                      for attempt_id in t.live}
        if token_live != live_keys:
            raise AssertionError("Token/copy live links disagree with queues")
        self.frames.append(AuditFrame(now, label, tuple(self.admitted),
                                     tuple(self.executed), tuple(self.cancelled),
                                     tuple(self.discarded), tuple(remaining),
                                     tuple(queues), tuple(running_keys)))

    def _observe(self, name, label, event):
        now = event.event_time
        self._advance_audit(now)
        self._check_audit(now, f"before:{label}")
        getattr(super(), name)(event)
        self._check_audit(now, f"after:{label}")

    def _handle_state_change(self, event):
        self._observe("_handle_state_change", "state", event)

    def _handle_complete(self, event):
        self._observe("_handle_complete", "complete", event)

    def _handle_timer(self, event):
        self._observe("_handle_timer", "timer", event)

    def _handle_arrival(self, event):
        self._observe("_handle_arrival", "arrival", event)

    def _dispatch_pass(self, now):
        self._advance_audit(now)
        super()._dispatch_pass(now)
        self._check_audit(now, "after:dispatch")

    def _enqueue(self, copy):
        self.admitted[copy.replica_id] += copy.required_work
        super()._enqueue(copy)

    def _record(self, copy, status, terminal_time, executed_work):
        observed = self.service_by_copy.get((copy.token_id, copy.attempt_id), 0.0)
        if not math.isclose(observed, executed_work, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"executed work mismatch: {observed} != {executed_work}")
        remainder = copy.required_work - executed_work
        if status is simulation.HedgeAttemptStatus.CANCELLED_QUEUED:
            self.cancelled[copy.replica_id] += remainder
        elif status in (simulation.HedgeAttemptStatus.FAILED_RUNNING,
                        simulation.HedgeAttemptStatus.INVALIDATED_QUEUED):
            self.discarded[copy.replica_id] += remainder
        super()._record(copy, status, terminal_time, executed_work)


def audited_run(trace, timeline, actions=None, hedge_delay=None, *,
                engine_class=AuditedEngine):
    args = (trace, timeline, 2.0)
    kwargs = dict(actions=actions, hedge_delay=hedge_delay)
    baseline = simulation.simulate_hedge_common_state(*args, **kwargs)
    with patch.object(simulation, "_Engine", engine_class):
        observed, engine = simulation._simulate_hedge_common_state_with_engine(*args, **kwargs)
    if observed != baseline:
        raise AssertionError("observer changed public simulation result")
    for replica in range(2):
        recorded = math.fsum(a.executed_work for a in observed.attempts
                             if a.replica_id == replica)
        split = math.fsum(work for (phase, r), work
                          in engine.service_by_phase_replica.items() if r == replica)
        if not math.isclose(recorded, split, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("lifecycle/phase service decomposition mismatch")
    return observed, engine


def make_trace(arrivals, primary_a, primary_b, *, replay_b=None, hedge_b=None):
    filler = (1.0,) * len(arrivals)
    return WorkloadTrace(
        base_seed=1, arrival_rate=0.9,
        tokens=tuple(TokenSpec(i, float(t), TokenClass.REGULAR)
                     for i, t in enumerate(arrivals)),
        service_times=(tuple(primary_a), tuple(primary_b)),
        replay_service_times=(filler, filler if replay_b is None else tuple(replay_b)),
        hedge_service_times=(filler, filler if hedge_b is None else tuple(hedge_b)),
    )


@dataclass(frozen=True)
class ReservationDecision:
    token_id: int
    requested: Action
    applied: Action
    charge: float
    remaining: float
    suppressed: bool


def reserve_requests(requests, timeline, *, cap, window_width=25.0):
    """Minimal test fixture for mean-one admission; takes no outcome/draw input.

    This is not the general production quota API or adoption of ADR-0011.
    Requests are a causally ordered sequence of (id, arrival, action) tuples.
    """
    decisions = []
    active_window = None
    remaining = 0.0
    for token_id, time, requested in requests:
        eligible = timeline.degraded_start <= time < timeline.failed_start and token_id % 2 == 0
        applied, charge, suppressed = Action.NORMAL, 0.0, False
        if eligible:
            window = math.floor((time - timeline.degraded_start) / window_width)
            if window != active_window:
                active_window, remaining = window, cap
            if requested is not Action.NORMAL:
                if remaining >= 1.0:
                    applied, charge = requested, 1.0
                    remaining -= charge
                else:
                    suppressed = True
        decisions.append(ReservationDecision(token_id, requested, applied, charge,
                                             remaining, suppressed))
    return tuple(decisions)
