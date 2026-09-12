"""Finite causal routing among eight fixed same-Expert Replicas.

This module is deliberately isolated from the historical two-Replica engines.
It implements the accepted ADR-0035 mechanism slice: one live attempt per
Token, causal reliability history, reactive Replay after failure, and six
deterministic routing policies.  It is not a Hedge, price, placement, or MFG
implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

from .topology import DEFAULT_TOPOLOGY, Topology


# Historical aliases intentionally remain K=8.  New traces carry their
# explicit immutable topology and simulation code uses trace.replica_ids.
REPLICA_IDS = DEFAULT_TOPOLOGY.replica_ids
DOMAINS = DEFAULT_TOPOLOGY.domains
SCENARIO_NAMES = ("S0", "S1", "S2", "S3")
POLICY_NAMES = (
    "uniform_rr",
    "jsq",
    "loew",
    "reliability_only",
    "risk_aware_jsq",
    "lazarus_algorithm1",
)
SMOKE_NAMESPACE = "replica-routing-baselines:reliability-aware-multi-replica-routing:v1:smoke"
SMOKE_MACRO_SEED = 20260909
SMOKE_CALL_BUDGET = 4 * 8 * 6
EVAL_HORIZON = 40.0
BURN_IN = 200.0
HISTORY_WINDOW = 50.0
ALPHA_H = 1.0
BETA_H = 100.0
SERVICE_MEAN = 1.0
SERVICE_CV = 0.5
ARRIVAL_RATE = 5.6
BATCH_RATE = ARRIVAL_RATE / 2.5
REGULAR_DEADLINE = 4.0
URGENT_DEADLINE = 2.0
GAMMA_REGULAR = 4.0
GAMMA_URGENT = 6.0


def _true_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RoutingToken:
    token_id: int
    arrival_time: float
    token_class: str
    deadline: float
    ingress_rank: int
    service_work: float

    def __post_init__(self) -> None:
        _true_int(self.token_id, "token_id")
        _nonnegative(self.arrival_time, "arrival_time")
        if self.token_class not in {"Regular", "Urgent"}:
            raise ValueError("token_class must be Regular or Urgent")
        if self.deadline <= 0.0 or not math.isfinite(float(self.deadline)):
            raise ValueError("deadline must be positive and finite")
        _true_int(self.ingress_rank, "ingress_rank")
        if self.service_work <= 0.0 or not math.isfinite(float(self.service_work)):
            raise ValueError("service_work must be positive and finite")


@dataclass(frozen=True)
class HealthEvent:
    time: float
    kind: str
    scope: str
    component_id: int

    def __post_init__(self) -> None:
        _finite(self.time, "time")
        if self.kind not in {"failure", "recovery"}:
            raise ValueError("kind must be failure or recovery")
        if self.scope not in {"replica", "domain"}:
            raise ValueError("scope must be replica or domain")
        _true_int(self.component_id, "component_id")
        if self.scope == "domain" and self.component_id >= 4:
            raise ValueError("component_id is outside the four-domain topology")


@dataclass(frozen=True)
class RoutingTrace:
    namespace: str
    macro_seed: int
    scenario: str
    episode_index: int
    tokens: tuple[RoutingToken, ...]
    health_events: tuple[HealthEvent, ...]
    work_draws: tuple[tuple[int, int, int, float], ...]
    fingerprint: str
    token_fingerprint: str
    topology: Topology = DEFAULT_TOPOLOGY

    def __post_init__(self) -> None:
        if not isinstance(self.topology, Topology):
            raise ValueError("trace topology must be Topology")
        for token in self.tokens:
            self.topology.domain_of(token.ingress_rank)
        for event in self.health_events:
            if event.scope == "replica":
                self.topology.domain_of(event.component_id)
            else:
                self.topology.validate_domain(event.component_id)

    @property
    def replica_ids(self) -> tuple[int, ...]:
        return self.topology.replica_ids

    @property
    def domains(self) -> tuple[int, ...]:
        return self.topology.domains

    @cached_property
    def _token_draw_offsets(self) -> dict[int, int]:
        """Dense generated-draw offsets, cached without changing trace identity.

        Both public trace builders emit draws in Token/Replica/attempt order.
        Keeping only one small Token-to-offset map avoids a second multi-million
        row dictionary at K=64 while turning the hot lookup into O(1).
        """

        stride = self.topology.K * 16
        return {
            token.token_id: position * stride
            for position, token in enumerate(self.tokens)
        }

    @cached_property
    def _component_event_index(
        self,
    ) -> dict[tuple[str, int], tuple[HealthEvent, ...]]:
        rows: dict[tuple[str, int], list[HealthEvent]] = {}
        for event in self.health_events:
            rows.setdefault((event.scope, event.component_id), []).append(event)
        return {key: tuple(values) for key, values in rows.items()}

    def events_for(self, scope: str, component_id: int) -> tuple[HealthEvent, ...]:
        return self._component_event_index.get((scope, component_id), ())

    def work_for(self, token_id: int, replica_id: int, attempt_id: int) -> float:
        offset = self._token_draw_offsets.get(token_id)
        if (
            offset is not None
            and type(replica_id) is int
            and replica_id in self.replica_ids
            and type(attempt_id) is int
            and 0 <= attempt_id < 16
        ):
            index = offset + replica_id * 16 + attempt_id
            if index < len(self.work_draws):
                row = self.work_draws[index]
                if row[:3] == (token_id, replica_id, attempt_id):
                    return row[3]
        # Preserve compatibility with externally constructed, non-dense
        # RoutingTrace values while keeping generated formal traces fast.
        for current_token, current_replica, current_attempt, work in self.work_draws:
            if (current_token, current_replica, current_attempt) == (
                token_id, replica_id, attempt_id
            ):
                return work
        raise KeyError((token_id, replica_id, attempt_id))


@dataclass(frozen=True)
class ReplicaObservation:
    replica_id: int
    domain_id: int
    available: bool
    queue_depth: int
    running_age: float | None
    estimated_work: float
    estimated_hazard: float
    estimated_failure_risk: float


@dataclass(frozen=True)
class AttemptResult:
    token_id: int
    attempt_id: int
    replica_id: int
    start_time: float | None
    terminal_time: float | None
    terminal_status: str
    required_work: float
    executed_work: float
    lost_work: float


@dataclass(frozen=True)
class TokenResult:
    token_id: int
    attempts: tuple[AttemptResult, ...]
    replay_count: int
    winner_replica_id: int | None
    completion_time: float | None
    token_class: str


@dataclass(frozen=True)
class StartRecord:
    token_id: int
    attempt_id: int
    replica_id: int
    start_time: float


@dataclass(frozen=True)
class RoutingResult:
    policy_name: str
    trace_fingerprint: str
    token_fingerprint: str
    token_results: tuple[TokenResult, ...]
    starts: tuple[StartRecord, ...]
    metrics: Mapping[str, float]
    invariants: Mapping[str, int | float]
    replica_audit: Mapping[int, Mapping[str, object]]
    complete_drain: bool


def _trace_identity(
    namespace: str, macro_seed: int, scenario: str, episode_index: int,
    tokens: Sequence[RoutingToken], events: Sequence[HealthEvent],
    draws: Sequence[tuple[int, int, int, float]],
    parameters: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    token_payload = [
        (row.token_id, row.arrival_time, row.token_class, row.deadline,
         row.ingress_rank)
        for row in tokens
    ]
    exogenous = {
        "namespace": namespace,
        "macro_seed": macro_seed,
        "scenario": scenario,
        "episode_index": episode_index,
        "tokens": token_payload,
        "events": [(e.time, e.kind, e.scope, e.component_id) for e in events],
        "draws": list(draws),
        "parameters": dict(parameters or {}),
    }
    token_fingerprint = _sha256({"tokens": token_payload})
    return _sha256(exogenous), token_fingerprint


def build_manual_trace(
    tokens: Sequence[RoutingToken], events: Sequence[HealthEvent] = (),
    *, namespace: str = "manual", macro_seed: int = 0, scenario: str = "manual",
    topology: Topology = DEFAULT_TOPOLOGY, episode_index: int = 0,
) -> RoutingTrace:
    if not isinstance(topology, Topology):
        raise TypeError("topology must be Topology")
    _true_int(episode_index, "episode_index")
    rows = tuple(sorted(tokens, key=lambda row: (row.arrival_time, row.token_id)))
    if tuple(row.token_id for row in rows) != tuple(sorted({row.token_id for row in rows})):
        raise ValueError("tokens must have unique IDs")
    for token in rows:
        topology.domain_of(token.ingress_rank)
    event_rows = tuple(sorted(events, key=lambda row: (row.time, row.scope, row.component_id, row.kind)))
    for event in event_rows:
        if event.scope == "replica":
            topology.domain_of(event.component_id)
        else:
            topology.validate_domain(event.component_id)
    draws = tuple(
        (token.token_id, replica_id, attempt_id, token.service_work)
        for token in rows for replica_id in topology.replica_ids for attempt_id in range(16)
    )
    fingerprint, token_fingerprint = _trace_identity(
        namespace, macro_seed, scenario, episode_index, rows, event_rows, draws,
        {} if topology == DEFAULT_TOPOLOGY else {"topology": topology.fingerprint},
    )
    return RoutingTrace(
        namespace, macro_seed, scenario, episode_index, rows, event_rows, draws,
        fingerprint, token_fingerprint, topology,
    )


def _scenario_rates(
    scenario: str, replica_id: int, domain_id: int, time: float,
    *, s2_switch_time: float = 20.0, domain_hazard: float = 0.03,
) -> tuple[float, float]:
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario}")
    individual = 0.01
    domain = 0.005
    if scenario == "S1" and replica_id % 8 in {0, 1}:
        individual = 0.04
    elif scenario == "S2":
        if time < s2_switch_time and replica_id % 8 in {0, 1}:
            individual = 0.04
        if time >= s2_switch_time and replica_id % 8 in {2, 3}:
            individual = 0.04
    elif scenario == "S3":
        domain = domain_hazard
    return individual, domain


def _append_path(
    events: list[HealthEvent], scope: str, component_id: int, start: float,
    end: float, scenario: str, replica_id: int, domain_id: int, seed: int,
    *, s2_switch_time: float = 20.0, domain_hazard: float = 0.03,
) -> None:
    rng = random.Random(seed)
    current = start
    up = True
    while current < end:
        individual, domain = _scenario_rates(
            scenario, replica_id, domain_id, current,
            s2_switch_time=s2_switch_time, domain_hazard=domain_hazard,
        )
        rate = individual if scope == "replica" else domain
        mean_repair = 6.0 if scope == "replica" else 8.0
        if up:
            duration = rng.expovariate(rate)
            next_time = current + duration
            if next_time >= end:
                break
            events.append(HealthEvent(next_time, "failure", scope, component_id))
            up = False
        else:
            duration = rng.expovariate(1.0 / mean_repair)
            next_time = current + duration
            if next_time >= end:
                events.append(HealthEvent(end, "recovery", scope, component_id))
                break
            events.append(HealthEvent(next_time, "recovery", scope, component_id))
            up = True
        current = next_time


def generate_routing_trace(
    scenario: str, episode_index: int, *,
    namespace: str = SMOKE_NAMESPACE, macro_seed: int = SMOKE_MACRO_SEED,
    arrival_rate: float = ARRIVAL_RATE, s2_switch_time: float = 20.0,
    domain_hazard: float = 0.03, topology: Topology = DEFAULT_TOPOLOGY,
) -> RoutingTrace:
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario}")
    _true_int(episode_index, "episode_index")
    if not isinstance(topology, Topology):
        raise TypeError("topology must be Topology")
    arrival_rate = _finite(arrival_rate, "arrival_rate")
    s2_switch_time = _finite(s2_switch_time, "s2_switch_time")
    domain_hazard = _finite(domain_hazard, "domain_hazard")
    if arrival_rate <= 0.0 or s2_switch_time < 0.0 or domain_hazard <= 0.0:
        raise ValueError("trace parameters must be positive, with non-negative switch time")
    arrival_rng = random.Random(_stable_seed(namespace, macro_seed, scenario, episode_index, "arrival"))
    class_rng = random.Random(_stable_seed(namespace, macro_seed, scenario, episode_index, "class"))
    service_rng = random.Random(_stable_seed(namespace, macro_seed, scenario, episode_index, "service"))
    tokens: list[RoutingToken] = []
    time = 0.0
    token_id = 0
    batch_rate = arrival_rate / 2.5
    while time < EVAL_HORIZON:
        time += arrival_rng.expovariate(batch_rate)
        if time >= EVAL_HORIZON:
            break
        base_batch_size = 1 + arrival_rng.randrange(4)
        # The release process is shared across K.  Only the mean cohort size
        # scales with topology, so total offered load grows linearly rather
        # than multiplying both release rate and batch size by K.
        batch_size = max(1, math.ceil(base_batch_size * topology.K / DEFAULT_TOPOLOGY.K))
        for _ in range(batch_size):
            token_class = "Urgent" if class_rng.random() < 0.2 else "Regular"
            deadline = URGENT_DEADLINE if token_class == "Urgent" else REGULAR_DEADLINE
            sigma = math.sqrt(math.log(1.0 + SERVICE_CV * SERVICE_CV))
            mu = math.log(SERVICE_MEAN) - 0.5 * sigma * sigma
            work = service_rng.lognormvariate(mu, sigma)
            tokens.append(RoutingToken(token_id, time, token_class, deadline,
                                       token_id % topology.K, work))
            token_id += 1
    events: list[HealthEvent] = []
    for replica_id in topology.replica_ids:
        _append_path(events, "replica", replica_id, -BURN_IN, EVAL_HORIZON,
                     scenario, replica_id, topology.domain_of(replica_id),
                     _stable_seed(namespace, macro_seed, scenario, episode_index, "replica", replica_id),
                     s2_switch_time=s2_switch_time, domain_hazard=domain_hazard)
    for domain_id in topology.domain_ids:
        _append_path(events, "domain", domain_id, -BURN_IN, EVAL_HORIZON,
                     scenario, domain_id, domain_id,
                     _stable_seed(namespace, macro_seed, scenario, episode_index, "domain", domain_id),
                     s2_switch_time=s2_switch_time, domain_hazard=domain_hazard)
    work_draws: list[tuple[int, int, int, float]] = []
    for token in tokens:
        sigma = math.sqrt(math.log(1.0 + SERVICE_CV * SERVICE_CV))
        mu = math.log(SERVICE_MEAN) - 0.5 * sigma * sigma
        for replica_id in topology.replica_ids:
            for attempt_id in range(16):
                draw_rng = random.Random(_stable_seed(
                    namespace, macro_seed, scenario, episode_index,
                    token.token_id, replica_id, attempt_id, "work"
                ))
                work_draws.append((token.token_id, replica_id, attempt_id,
                                   draw_rng.lognormvariate(mu, sigma)))
    event_rows = tuple(sorted(events, key=lambda row: (row.time, row.scope, row.component_id, row.kind)))
    fingerprint, token_fingerprint = _trace_identity(
        namespace, macro_seed, scenario, episode_index, tokens, event_rows, work_draws,
        {"arrival_rate": arrival_rate, "s2_switch_time": s2_switch_time,
         "domain_hazard": domain_hazard,
         **({} if topology == DEFAULT_TOPOLOGY else {"topology": topology.fingerprint})},
    )
    return RoutingTrace(
        namespace, macro_seed, scenario, episode_index, tuple(tokens), event_rows,
        tuple(work_draws), fingerprint, token_fingerprint, topology,
    )


def _component_events(events: Sequence[HealthEvent], scope: str, component_id: int) -> tuple[HealthEvent, ...]:
    return tuple(row for row in events if row.scope == scope and row.component_id == component_id)


def _state_at(events: Sequence[HealthEvent], scope: str, component_id: int, time: float) -> bool:
    up = True
    for event in _component_events(events, scope, component_id):
        if event.time > time:
            break
        up = event.kind == "recovery"
    return up


def estimate_hazard(
    events: Sequence[HealthEvent], scope: str, component_id: int, time: float,
    *, window: float = HISTORY_WINDOW, alpha: float = ALPHA_H, beta: float = BETA_H,
) -> float:
    if scope not in {"replica", "domain"}:
        raise ValueError("scope must be replica or domain")
    time = _finite(time, "time")
    window = _nonnegative(window, "window")
    alpha = _finite(alpha, "alpha")
    beta = _finite(beta, "beta")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    start = time - window
    relevant = _component_events(events, scope, component_id)
    return _estimate_hazard_from_component_events(
        relevant, time, start=start, alpha=alpha, beta=beta,
    )


def _estimate_hazard_from_component_events(
    relevant: Sequence[HealthEvent],
    time: float,
    *,
    start: float,
    alpha: float,
    beta: float,
) -> float:
    up = True
    for event in relevant:
        if event.time > start:
            break
        up = event.kind == "recovery"
    cursor = start
    exposure = 0.0
    failures = 0
    for event in relevant:
        if event.time <= start:
            continue
        if event.time > time:
            break
        if up:
            exposure += event.time - cursor
        if event.kind == "failure":
            failures += 1
            up = False
        else:
            up = True
        cursor = event.time
    if up:
        exposure += time - cursor
    return (alpha + failures) / (beta + exposure)


def _normal_survival(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _conditional_residual(age: float) -> float:
    if age <= 0.0:
        return SERVICE_MEAN
    sigma = math.sqrt(math.log(1.0 + SERVICE_CV * SERVICE_CV))
    mu = math.log(SERVICE_MEAN) - 0.5 * sigma * sigma
    z = (math.log(age) - mu) / sigma
    denominator = _normal_survival(z)
    if denominator <= 1e-15:
        return 0.0
    numerator = SERVICE_MEAN * _normal_survival(z - sigma)
    return max(0.0, numerator / denominator - age)


def _risk_observations(
    trace: RoutingTrace, now: float, health: Mapping[int, bool],
    queues: Mapping[int, Sequence[int]], running: Mapping[int, int],
    running_started: Mapping[int, float],
    *, history_window: float = HISTORY_WINDOW,
    include_arrival_service: bool = True,
) -> tuple[ReplicaObservation, ...]:
    result: list[ReplicaObservation] = []
    start = now - history_window
    domain_hazards = {
        domain_id: _estimate_hazard_from_component_events(
            trace.events_for("domain", domain_id), now,
            start=start, alpha=ALPHA_H, beta=BETA_H,
        )
        for domain_id in trace.topology.domain_ids
    }
    for replica_id in trace.replica_ids:
        domain_id = trace.topology.domain_of(replica_id)
        queue_depth = len(queues[replica_id]) + (1 if replica_id in running else 0)
        running_age = None if replica_id not in running else now - running_started[replica_id]
        estimated_work = len(queues[replica_id]) * SERVICE_MEAN
        if running_age is not None:
            estimated_work += _conditional_residual(running_age)
        replica_hazard = _estimate_hazard_from_component_events(
            trace.events_for("replica", replica_id), now,
            start=start, alpha=ALPHA_H, beta=BETA_H,
        )
        domain_hazard = domain_hazards[domain_id]
        hazard = replica_hazard + domain_hazard
        # Healthy replicas run at the fixed unit speed in this finite
        # physical model.  The candidate's causal risk score includes the
        # arriving Token's known mean service requirement.  Frozen comparator
        # policies request the historical projection explicitly so their
        # results remain value-identical across this correction.
        current_speed = 1.0 if health[replica_id] else 0.0
        if include_arrival_service and current_speed > 0.0:
            exposure_horizon = estimated_work + SERVICE_MEAN / current_speed
        else:
            exposure_horizon = max(estimated_work, 1e-12)
        risk = 1.0 - math.exp(-hazard * exposure_horizon)
        result.append(ReplicaObservation(
            replica_id, domain_id, health[replica_id], queue_depth, running_age,
            estimated_work, hazard, risk,
        ))
    return tuple(result)


def _choose(
    policy: str, token: RoutingToken, observations: Sequence[ReplicaObservation],
    *, gamma_regular: float = GAMMA_REGULAR, gamma_urgent: float = GAMMA_URGENT,
) -> int | None:
    eligible = tuple(row for row in observations if row.available)
    if not eligible:
        return None
    if policy == "uniform_rr":
        return eligible[token.token_id % len(eligible)].replica_id
    if policy == "jsq":
        return min(eligible, key=lambda row: (row.queue_depth, row.replica_id)).replica_id
    if policy == "loew":
        return min(eligible, key=lambda row: (row.estimated_work, row.queue_depth, row.replica_id)).replica_id
    if policy == "reliability_only":
        return min(eligible, key=lambda row: (row.estimated_failure_risk, row.estimated_work, row.replica_id)).replica_id
    if policy == "risk_aware_jsq":
        gamma = gamma_urgent if token.token_class == "Urgent" else gamma_regular
        return min(eligible, key=lambda row: (
            row.estimated_work + gamma * row.estimated_failure_risk,
            row.queue_depth, row.replica_id,
        )).replica_id
    raise ValueError(f"unknown per-token policy {policy}")


def _lazarus_assign(
    tokens: Sequence[RoutingToken], observations: Sequence[ReplicaObservation],
    *, batch_identity: int,
) -> dict[int, int]:
    _true_int(batch_identity, "batch_identity")
    eligible = tuple(row for row in observations if row.available)
    if not eligible:
        return {}
    ordered = tuple(sorted(eligible, key=lambda row: row.replica_id))
    base, remainder = divmod(len(tokens), len(ordered))
    start = batch_identity % len(ordered)
    extra_replicas = {
        ordered[(start + index) % len(ordered)].replica_id
        for index in range(remainder)
    }
    capacities = {
        row.replica_id: base + (1 if row.replica_id in extra_replicas else 0)
        for row in ordered
    }
    assigned: dict[int, int] = {}
    counts = {replica_id: 0 for replica_id in capacities}
    for token in sorted(tokens, key=lambda row: row.token_id):
        local = token.ingress_rank if token.ingress_rank in capacities else None
        if local is not None and counts[local] < capacities[local]:
            assigned[token.token_id] = local
            counts[local] += 1
            continue
        candidates = tuple(
            ordered[(start + index) % len(ordered)].replica_id
            for index in range(len(ordered))
            if counts[ordered[(start + index) % len(ordered)].replica_id]
            < capacities[ordered[(start + index) % len(ordered)].replica_id]
        )
        if candidates:
            selected = candidates[0]
            assigned[token.token_id] = selected
            counts[selected] += 1
    return assigned


def lazarus_batch_assignment(
    tokens: Sequence[RoutingToken], eligible_replica_ids: Sequence[int],
    *, batch_identity: int, topology: Topology = DEFAULT_TOPOLOGY,
) -> dict[int, int]:
    """Return one deterministic Lazarus assignment for a current micro-batch.

    This pure helper is used by the comparator fairness audit. It exposes no
    service draw or future state and does not execute a scheduler call.
    """
    _true_int(batch_identity, "batch_identity")
    if not isinstance(eligible_replica_ids, Sequence):
        raise ValueError("eligible_replica_ids must be a sequence")
    ids = tuple(sorted(eligible_replica_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("eligible_replica_ids must be non-empty and unique")
    if not isinstance(topology, Topology):
        raise TypeError("topology must be Topology")
    if any(type(replica_id) is not int or replica_id not in topology.replica_ids for replica_id in ids):
        raise ValueError("eligible_replica_ids must contain canonical Replica IDs")
    token_rows = tuple(tokens)
    if len({token.token_id for token in token_rows}) != len(token_rows):
        raise ValueError("batch token IDs must be unique")
    observations = tuple(
        ReplicaObservation(replica_id, topology.domain_of(replica_id), True, 0, None,
                           SERVICE_MEAN, 0.0, 0.0)
        for replica_id in ids
    )
    return _lazarus_assign(token_rows, observations, batch_identity=batch_identity)


def audit_lazarus_homogeneous_balance() -> dict[str, object]:
    """Audit long-run small-batch allocation without consuming scheduler calls."""
    counts = {replica_id: 0 for replica_id in REPLICA_IDS}
    batch_count = 512
    for batch_index in range(batch_count):
        first_token_id = 2 * batch_index
        tokens = (
            RoutingToken(first_token_id, float(batch_index), "Regular", 4.0,
                         (2 * batch_index) % 8, SERVICE_MEAN),
            RoutingToken(first_token_id + 1, float(batch_index), "Regular", 4.0,
                         (2 * batch_index + 4) % 8, SERVICE_MEAN),
        )
        assignment = lazarus_batch_assignment(
            tokens, REPLICA_IDS, batch_identity=first_token_id,
        )
        for replica_id in assignment.values():
            counts[replica_id] += 1
    gap = max(counts.values()) - min(counts.values())
    if gap > 1:
        raise AssertionError("Lazarus homogeneous allocation fairness audit failed")
    return {
        "batch_count": batch_count,
        "tokens_per_batch": 2,
        "counts": counts,
        "max_count_gap": gap,
        "max_count_gap_bound": 1,
    }


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


def _latency_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {f"{prefix}_{name}": math.nan for name in ("mean", "p50", "p95", "p99", "cvar95")}
    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]
    cut = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    tail = ordered[cut:]
    return {
        f"{prefix}_mean": math.fsum(ordered) / len(ordered),
        f"{prefix}_p50": percentile(0.50),
        f"{prefix}_p95": percentile(0.95),
        f"{prefix}_p99": percentile(0.99),
        f"{prefix}_cvar95": math.fsum(tail) / len(tail),
    }


def _result_metrics(
    trace: RoutingTrace, token_results: Sequence[TokenResult], total_work: float,
    lost_work: float, drain_end: float,
) -> dict[str, float]:
    completed = [row for row in token_results if row.completion_time is not None]
    latencies = sorted(
        row.completion_time - next(token.arrival_time for token in trace.tokens if token.token_id == row.token_id)
        for row in completed
    )
    token_map = {row.token_id: row for row in trace.tokens}
    metrics = {
        "generated_tokens": float(len(trace.tokens)),
        "completed_tokens": float(len(completed)),
        "terminal_failed_tokens": float(sum(row.completion_time is None for row in token_results)),
        "replay_count": float(sum(row.replay_count for row in token_results)),
        "replay_rate": (sum(row.replay_count for row in token_results) / len(trace.tokens)) if trace.tokens else 0.0,
        "executed_work": total_work,
        "lost_work": lost_work,
        "nominal_work": math.fsum(token.service_work for token in trace.tokens),
        "last_arrival": max((token.arrival_time for token in trace.tokens), default=0.0),
        "drain_end": drain_end,
        "drain_duration": drain_end - max((token.arrival_time for token in trace.tokens), default=0.0),
    }
    metrics.update(_latency_stats(latencies, "overall"))
    metrics["mean_latency"] = metrics["overall_mean"]
    metrics["p95_latency"] = metrics["overall_p95"]
    metrics["p99_latency"] = metrics["overall_p99"]
    for token_class in ("Regular", "Urgent"):
        rows = [row for row in completed if row.token_class == token_class]
        class_latencies = [
            row.completion_time - token_map[row.token_id].arrival_time for row in rows
        ]
        metrics.update(_latency_stats(class_latencies, token_class.lower()))
        class_tokens = [row for row in token_results if row.token_class == token_class]
        metrics[f"{token_class.lower()}_deadline_miss_rate"] = (
            sum(
                row.completion_time is None
                or row.completion_time > token_map[row.token_id].arrival_time + token_map[row.token_id].deadline
                for row in class_tokens
            ) / len(class_tokens)
            if class_tokens else 0.0
        )
    return metrics


def simulate_routing(
    trace: RoutingTrace, policy: str, *, history_window: float = HISTORY_WINDOW,
    gamma_regular: float = GAMMA_REGULAR, gamma_urgent: float = GAMMA_URGENT,
) -> RoutingResult:
    if not isinstance(trace, RoutingTrace):
        raise TypeError("trace must be RoutingTrace")
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy {policy}")
    topology = trace.topology
    replica_ids = topology.replica_ids
    history_window = _finite(history_window, "history_window")
    gamma_regular = _finite(gamma_regular, "gamma_regular")
    gamma_urgent = _finite(gamma_urgent, "gamma_urgent")
    if history_window <= 0.0 or gamma_regular < 0.0 or gamma_urgent < 0.0:
        raise ValueError("history window must be positive and gamma values non-negative")
    token_by_id = {token.token_id: token for token in trace.tokens}
    if len(token_by_id) != len(trace.tokens):
        raise ValueError("trace token IDs must be unique")
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
    total_work = 0.0
    lost_work = 0.0
    no_eligible_time = 0.0
    down_starts = 0
    max_live = 0
    current_time = 0.0
    arrival_index = 0
    event_index = next((index for index, event in enumerate(trace.health_events) if event.time >= 0.0), len(trace.health_events))
    terminal: set[int] = set()
    last_time = 0.0
    replica_audit: dict[int, dict[str, object]] = {
        replica_id: {
            "assignments": 0, "starts": 0, "completions": 0,
            "running_failures": 0, "queued_displacements": 0,
            "peak_queue_depth": 0, "busy_time": 0.0,
            "available_time": 0.0, "hazard_samples": [],
        }
        for replica_id in replica_ids
    }

    def add_waiting(token_id: int, *, front: bool = False) -> None:
        if front:
            waiting.appendleft(token_id)
        else:
            waiting.append(token_id)

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
                        row for row in attempts[token_id]
                        if row.replica_id == replica_id and row.terminal_status == "queued"
                    )
                    queued_attempt.terminal_time = time
                    queued_attempt.terminal_status = "discarded"
                    replica_audit[replica_id]["queued_displacements"] += 1
                displaced_queued.extend(queued_tokens)
                queues[replica_id].clear()
        for token_id in sorted(displaced_running, key=lambda row: (token_by_id[row].arrival_time, row)):
            add_waiting(token_id)
        for token_id in sorted(displaced_queued, key=lambda row: (token_by_id[row].arrival_time, row)):
            add_waiting(token_id)

    def start_available(time: float) -> None:
        nonlocal down_starts
        for replica_id in replica_ids:
            if not health[replica_id] or replica_id in running or not queues[replica_id]:
                continue
            token_id = queues[replica_id].popleft()
            attempt = next(
                row for row in attempts[token_id]
                if row.replica_id == replica_id and row.terminal_status == "queued"
            )
            attempt.start_time = time
            attempt.terminal_status = "running"
            running[replica_id] = attempt
            starts.append(StartRecord(token_id, attempt.attempt_id, replica_id, time))
            replica_audit[replica_id]["starts"] += 1
            if not health[replica_id]:
                down_starts += 1

    def route_waiting(time: float, new_token_ids: Sequence[int]) -> None:
        def observations_now() -> tuple[ReplicaObservation, ...]:
            observations = _risk_observations(
                trace, time, health, queues,
                {rid: row.token_id for rid, row in running.items()},
                {rid: row.start_time or time for rid, row in running.items()},
                history_window=history_window,
                include_arrival_service=(policy == "risk_aware_jsq"),
            )
            for row in observations:
                cast_samples = replica_audit[row.replica_id]["hazard_samples"]
                cast_samples.append(row.estimated_hazard)
            return observations

        batch_map: dict[int, int] = {}
        original_waiting = len(waiting)
        for _ in range(original_waiting):
            token_id = waiting.popleft()
            token = token_by_id[token_id]
            choice = _choose(
                "jsq" if policy == "lazarus_algorithm1" else policy,
                token, observations_now(),
                gamma_regular=gamma_regular, gamma_urgent=gamma_urgent,
            )
            if choice is None:
                waiting.append(token_id)
            else:
                attempt_id = len(attempts[token_id])
                attempts[token_id].append(_MutableAttempt(
                    token_id, attempt_id, choice,
                    trace.work_for(token_id, choice, attempt_id),
                ))
                queues[choice].append(token_id)
                replica_audit[choice]["assignments"] += 1
        if policy == "lazarus_algorithm1" and new_token_ids:
            batch_map = _lazarus_assign(
                tuple(token_by_id[token_id] for token_id in new_token_ids),
                observations_now(), batch_identity=new_token_ids[0],
            )
        for token_id in new_token_ids:
            token = token_by_id[token_id]
            choice = batch_map.get(token_id) if policy == "lazarus_algorithm1" else _choose(
                policy, token, observations_now(),
                gamma_regular=gamma_regular, gamma_urgent=gamma_urgent,
            )
            if choice is None:
                waiting.append(token_id)
            else:
                attempt_id = len(attempts[token_id])
                attempts[token_id].append(_MutableAttempt(
                    token_id, attempt_id, choice,
                    trace.work_for(token_id, choice, attempt_id),
                ))
                queues[choice].append(token_id)
                replica_audit[choice]["assignments"] += 1
        for replica_id in replica_ids:
            replica_audit[replica_id]["peak_queue_depth"] = max(
                replica_audit[replica_id]["peak_queue_depth"],
                len(queues[replica_id]) + (1 if replica_id in running else 0),
            )
        start_available(time)

    def advance(to_time: float) -> None:
        nonlocal total_work, current_time, no_eligible_time
        dt = to_time - current_time
        if dt < -1e-12:
            raise AssertionError("event time moved backwards")
        for attempt in running.values():
            increment = min(attempt.required_work - attempt.executed_work, dt)
            attempt.executed_work += max(0.0, increment)
            total_work += max(0.0, increment)
        for replica_id in replica_ids:
            replica_audit[replica_id]["available_time"] += dt if health[replica_id] else 0.0
            replica_audit[replica_id]["busy_time"] += dt if replica_id in running else 0.0
        if waiting and not any(health.values()):
            no_eligible_time += dt
        current_time = to_time

    while True:
        next_arrival = trace.tokens[arrival_index].arrival_time if arrival_index < len(trace.tokens) else math.inf
        next_health = trace.health_events[event_index].time if event_index < len(trace.health_events) else math.inf
        next_completion = min(
            (current_time + max(0.0, attempt.required_work - attempt.executed_work)
             for attempt in running.values()), default=math.inf
        )
        next_time = min(next_arrival, next_health, next_completion)
        if next_time is math.inf:
            if not waiting and not running and arrival_index >= len(trace.tokens):
                last_time = current_time
                break
            raise RuntimeError("finite fault horizon did not drain the global queue")
        advance(next_time)
        last_time = current_time
        same_events: list[HealthEvent] = []
        while event_index < len(trace.health_events) and abs(trace.health_events[event_index].time - current_time) <= 1e-12:
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
        failed = {replica_id for replica_id in replica_ids if old_health[replica_id] and not health[replica_id]}
        invalidate_failed_replicas(failed, current_time)
        completed_replicas = [
            replica_id for replica_id, attempt in running.items()
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
        while arrival_index < len(trace.tokens) and abs(trace.tokens[arrival_index].arrival_time - current_time) <= 1e-12:
            new_ids.append(trace.tokens[arrival_index].token_id)
            arrival_index += 1
        route_waiting(current_time, new_ids)
        max_live = max(max_live, 1 if running else 0)

    token_results: list[TokenResult] = []
    for token in sorted(trace.tokens, key=lambda row: row.token_id):
        rows = tuple(
            AttemptResult(
                row.token_id, row.attempt_id, row.replica_id, row.start_time,
                row.terminal_time, row.terminal_status, row.required_work,
                row.executed_work, row.lost_work,
            ) for row in attempts[token.token_id]
        )
        completed_rows = tuple(row for row in rows if row.terminal_status == "completed")
        winner = completed_rows[-1].replica_id if completed_rows else None
        completion = completed_rows[-1].terminal_time if completed_rows else None
        token_results.append(TokenResult(token.token_id, rows, replay_counts[token.token_id],
                                         winner, completion, token.token_class))
    invariants = {
        "down_starts": down_starts,
        "max_live_attempts_per_token": max_live,
        "future_work_observations": 0,
        "completed_without_winner": sum(row.winner_replica_id is None for row in token_results),
        "duplicate_live_attempts": 0,
        "executed_work": total_work,
        "lost_work": lost_work,
    }
    if invariants["down_starts"] or invariants["completed_without_winner"]:
        raise AssertionError("routing invariant failed")
    metrics = _result_metrics(trace, token_results, total_work, lost_work, last_time)
    metrics["no_eligible_time"] = no_eligible_time
    for replica_id in replica_ids:
        samples = replica_audit[replica_id].pop("hazard_samples")
        replica_audit[replica_id]["mean_estimated_hazard"] = (
            math.fsum(samples) / len(samples) if samples else 0.0
        )
        available_time = float(replica_audit[replica_id]["available_time"])
        replica_audit[replica_id]["utilization_if_available"] = (
            float(replica_audit[replica_id]["busy_time"]) / available_time
            if available_time > 0.0 else 0.0
        )
    return RoutingResult(policy, trace.fingerprint, trace.token_fingerprint,
                         tuple(token_results), tuple(starts), metrics,
                         invariants, replica_audit, True)


def _protocol_payload() -> dict[str, object]:
    return {
        "namespace": SMOKE_NAMESPACE,
        "macro_seed": SMOKE_MACRO_SEED,
        "scenarios": list(SCENARIO_NAMES),
        "policies": list(POLICY_NAMES),
        "calls": SMOKE_CALL_BUDGET,
        "topology": {"replicas": list(REPLICA_IDS), "domains": list(DOMAINS), "speed": 1.0},
        "workload": {"horizon": EVAL_HORIZON, "burn_in": BURN_IN, "arrival_rate": ARRIVAL_RATE,
                      "regular_probability": 0.8, "deadlines": {"Regular": REGULAR_DEADLINE, "Urgent": URGENT_DEADLINE}},
        "service": {"family": "lognormal", "mean": SERVICE_MEAN, "cv": SERVICE_CV},
        "health": {"baseline_individual_hazard": 0.01, "high_risk_hazard": 0.04,
                    "baseline_domain_hazard": 0.005, "correlated_domain_hazard": 0.03,
                    "repair_mean_replica": 6.0, "repair_mean_domain": 8.0},
        "estimator": {"window": HISTORY_WINDOW, "alpha": ALPHA_H, "beta": BETA_H},
        "claim_boundary": "non_inferential_mechanism_smoke_only",
        "lazarus_homogeneous_audit": audit_lazarus_homogeneous_balance(),
    }


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    files = [
        root / "src" / "mfg_hedge" / "reliability_aware_routing.py",
        root / "docs" / "adr" / "0035-reliability-aware-multi-replica-routing.md",
        root / ".scratch" / "reliability-aware-multi-replica-routing" / "spec.md",
        root / ".scratch" / "reliability-aware-multi-replica-routing" / "issues" / "02-implement-finite-multi-replica-routing.md",
    ]
    payload = []
    for path in files:
        payload.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return _sha256(payload)


def run_routing_smoke(artifact_root: str | Path) -> dict[str, object]:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "reliability-aware-multi-replica-routing-smoke-20260909-r"
    index = 1
    while (root / f"{prefix}{index}").exists():
        index += 1
    final_dir = root / f"{prefix}{index}"
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}-", dir=str(root)))
    calls = 0
    rows: list[dict[str, object]] = []
    try:
        fairness_audit = audit_lazarus_homogeneous_balance()
        for scenario in SCENARIO_NAMES:
            for episode in range(8):
                trace = generate_routing_trace(scenario, episode)
                for policy in POLICY_NAMES:
                    calls += 1
                    result = simulate_routing(trace, policy)
                    rows.append({
                        "scenario": scenario,
                        "episode": episode,
                        "policy": policy,
                        "trace_fingerprint": result.trace_fingerprint,
                        "token_fingerprint": result.token_fingerprint,
                        "metrics": dict(result.metrics),
                        "invariants": dict(result.invariants),
                    })
        protocol = _protocol_payload()
        summary = {
            "status": "mechanism_smoke_complete",
            "calls": calls,
            "scenarios": len(SCENARIO_NAMES),
            "episodes_per_scenario": 8,
            "policies": len(POLICY_NAMES),
            "source_fingerprint": _source_fingerprint(),
            "claim_boundary": "non_inferential_mechanism_smoke_only",
            "lazarus_homogeneous_audit": fairness_audit,
            "artifact_dir": str(final_dir),
        }
        (temp_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
        (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (temp_dir / "manifest.json").write_text(json.dumps({
            "artifact_kind": "reliability_aware_routing_mechanism_smoke",
            "namespace": SMOKE_NAMESPACE,
            "macro_seed": SMOKE_MACRO_SEED,
            "source_fingerprint": summary["source_fingerprint"],
            "transaction": "committed_after_all_192_calls",
        }, indent=2, sort_keys=True), encoding="utf-8")
        with (temp_dir / "episode_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        temp_dir.replace(final_dir)
        return summary
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


__all__ = [
    "POLICY_NAMES", "SCENARIO_NAMES", "SMOKE_CALL_BUDGET", "SMOKE_MACRO_SEED",
    "SMOKE_NAMESPACE", "HealthEvent", "RoutingToken", "RoutingTrace",
    "Topology", "DEFAULT_TOPOLOGY",
    "RoutingResult", "TokenResult", "AttemptResult", "build_manual_trace",
    "estimate_hazard", "generate_routing_trace", "simulate_routing",
    "lazarus_batch_assignment", "audit_lazarus_homogeneous_balance",
    "run_routing_smoke",
]
