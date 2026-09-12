"""Stage 2 common noise, nested workloads, and causal rule contracts.

This module is deliberately separate from the historical workload generators
and from the Stage 1 event scheduler.  It constructs immutable inputs for the
isolated shared-backup API; policy code sees only :class:`PublicFaultObservation`
and never receives this module's full fault path or workload trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import random
import struct
from types import MappingProxyType
from typing import Mapping

from .common_state import CommonStateTimeline, Phase, phase_at, state_at
from .domain import CommonState, TokenClass


_ARRIVAL_CUTOFF = 360.0
_ARRIVAL_RATE = 0.45
_REGULAR_RATIO = 0.8
_WORK_MEAN = 1.0
_WORK_CV = 0.5
_ACTION_CODES = ("N", "D", "S", "X")
_ATTEMPTS = (0, 1, 2, 3)
_REPLICAS = (0, 1, 2)
_PACKAGE_VERSION = "0.20.0"


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _strict_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {relation}, got {value!r}")
    return result


def _identifier(value: object, name: str) -> str | int:
    if type(value) is int:
        return _strict_int(value, name)
    if type(value) is str and value:
        return value
    raise ValueError(f"{name} must be a non-empty string or int, got {value!r}")


def _stable_seed(*parts: object) -> int:
    """Derive a stream seed without Python's process-randomized ``hash``."""

    payload = json.dumps(
        list(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _lognormal_parameters(mean: float, cv: float) -> tuple[float, float]:
    sigma2 = math.log(1.0 + cv * cv)
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


@dataclass(frozen=True)
class CommonFaultIdentity:
    """Stable common-noise identity; it intentionally has no population/Expert."""

    namespace: str
    macro_seed: int
    common_episode_id: int

    def __post_init__(self) -> None:
        if type(self.namespace) is not str or not self.namespace:
            raise ValueError("namespace must be a non-empty string")
        _strict_int(self.macro_seed, "macro_seed")
        _strict_int(self.common_episode_id, "common_episode_id")

    @property
    def key(self) -> tuple[str, int, int]:
        return self.namespace, self.macro_seed, self.common_episode_id

    def stream_key(self, component: str) -> tuple[object, ...]:
        if type(component) is not str or not component:
            raise ValueError("common fault component must be a non-empty string")
        return (*self.key, "common_fault", component)


@dataclass(frozen=True)
class CommonFaultPath:
    """A complete realized fault path held by the simulator, not a policy."""

    identity: CommonFaultIdentity
    timeline: CommonStateTimeline
    degraded_duration: float
    failed_duration: float
    arrival_cutoff: float = _ARRIVAL_CUTOFF
    mode: str = "stochastic"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CommonFaultIdentity):
            raise ValueError("identity must be CommonFaultIdentity")
        if not isinstance(self.timeline, CommonStateTimeline):
            raise ValueError("timeline must be CommonStateTimeline")
        degraded = _strict_real(self.degraded_duration, "degraded_duration", positive=True)
        failed = _strict_real(self.failed_duration, "failed_duration", positive=True)
        cutoff = _strict_real(self.arrival_cutoff, "arrival_cutoff", positive=True)
        if self.mode not in ("scheduled", "stochastic"):
            raise ValueError("mode must be 'scheduled' or 'stochastic'")
        if not math.isclose(self.timeline.degraded_start, 100.0):
            raise ValueError("Stage 2 common faults must start degradation at 100")
        if not math.isclose(self.timeline.failed_start, 100.0 + degraded):
            raise ValueError("failed_start must equal 100 + degraded_duration")
        if not math.isclose(self.timeline.recovered_start, 100.0 + degraded + failed):
            raise ValueError("recovered_start must equal failed_start + failed_duration")
        object.__setattr__(self, "degraded_duration", degraded)
        object.__setattr__(self, "failed_duration", failed)
        object.__setattr__(self, "arrival_cutoff", cutoff)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.identity.key, separators=(",", ":")).encode())
        digest.update(struct.pack(">4d", self.degraded_duration, self.failed_duration,
                                 self.timeline.failed_start, self.timeline.recovered_start))
        digest.update(self.mode.encode("ascii"))
        return digest.hexdigest()


def build_common_fault_path(
    namespace: str,
    macro_seed: int,
    common_episode_id: int,
    *,
    mode: str = "stochastic",
    degraded_duration: float | None = None,
    failed_duration: float | None = None,
    arrival_cutoff: float = _ARRIVAL_CUTOFF,
    expert_count: int | None = None,
    population_id: object | None = None,
) -> CommonFaultPath:
    """Build one scheduled or stochastic common path.

    ``expert_count`` and ``population_id`` are accepted only as compatibility
    probes and are validated then ignored.  They are intentionally absent from
    the common stream key.
    """

    identity = CommonFaultIdentity(namespace, macro_seed, common_episode_id)
    if mode not in ("scheduled", "stochastic"):
        raise ValueError("mode must be 'scheduled' or 'stochastic'")
    if expert_count is not None:
        _strict_int(expert_count, "expert_count", 1)
    if population_id is not None:
        _identifier(population_id, "population_id")
    if mode == "scheduled":
        degraded = 100.0 if degraded_duration is None else degraded_duration
        failed = 20.0 if failed_duration is None else failed_duration
    else:
        degraded = (
            random.Random(_stable_seed(*identity.stream_key("degraded_duration")))
            .uniform(75.0, 125.0)
            if degraded_duration is None else degraded_duration
        )
        failed = (
            random.Random(_stable_seed(*identity.stream_key("failed_duration")))
            .uniform(10.0, 30.0)
            if failed_duration is None else failed_duration
        )
    # Validate explicit values before constructing the timeline, including
    # bool/NaN/Inf and the strict finite ordering contract.
    degraded_value = _strict_real(degraded, "degraded_duration", positive=True)
    failed_value = _strict_real(failed, "failed_duration", positive=True)
    timeline = CommonStateTimeline(100.0, 100.0 + degraded_value,
                                   100.0 + degraded_value + failed_value)
    return CommonFaultPath(identity, timeline, degraded_value, failed_value,
                           arrival_cutoff, mode)


def scheduled_common_fault_path(
    namespace: str,
    macro_seed: int,
    common_episode_id: int,
    *,
    arrival_cutoff: float = _ARRIVAL_CUTOFF,
) -> CommonFaultPath:
    return build_common_fault_path(namespace, macro_seed, common_episode_id,
                                   mode="scheduled", arrival_cutoff=arrival_cutoff)


# Short aliases make the isolated contract easy to discover without creating
# a second implementation or touching historical workload APIs.
generate_common_fault_path = build_common_fault_path
generate_fault_path = build_common_fault_path


@dataclass(frozen=True)
class PublicFaultObservation:
    """Only the public fault history available to an online controller."""

    current_time: float
    health: CommonState
    phase: Phase
    observed_phase_start: float
    phase_age: float
    transitions: tuple[tuple[str, float], ...]
    k_n: float

    def __post_init__(self) -> None:
        now = _strict_real(self.current_time, "current_time")
        start = _strict_real(self.observed_phase_start, "observed_phase_start")
        age = _strict_real(self.phase_age, "phase_age")
        k_n = _strict_real(self.k_n, "k_n")
        if k_n > 2.0:
            raise ValueError("k_n must be in [0, 2]")
        if not isinstance(self.health, CommonState):
            raise ValueError("health must be CommonState")
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be Phase")
        if start > now or not math.isclose(age, now - start, abs_tol=1e-10):
            raise ValueError("public phase age must be current_time - observed_phase_start")
        if not isinstance(self.transitions, tuple):
            raise ValueError("transitions must be an immutable tuple")
        previous = -1.0
        for label, time in self.transitions:
            if label not in ("D", "F", "R"):
                raise ValueError("public transitions must use D/F/R labels")
            boundary = _strict_real(time, "transition time")
            if boundary > now or boundary <= previous:
                raise ValueError("public transitions must contain only ordered past events")
            previous = boundary
        object.__setattr__(self, "current_time", now)
        object.__setattr__(self, "observed_phase_start", start)
        object.__setattr__(self, "phase_age", age)
        object.__setattr__(self, "k_n", k_n)

    @property
    def current_state(self) -> CommonState:
        return self.health

    @property
    def observed_transitions(self) -> tuple[tuple[str, float], ...]:
        return self.transitions

    @property
    def k_N(self) -> float:
        return self.k_n


def public_fault_observation(
    path: CommonFaultPath,
    current_time: float,
    k_n: float,
) -> PublicFaultObservation:
    """Project the full path to the information available through ``current_time``."""

    if not isinstance(path, CommonFaultPath):
        raise ValueError("path must be CommonFaultPath")
    return public_fault_observation_from_timeline(path.timeline, current_time, k_n)


def public_fault_observation_from_timeline(
    timeline: CommonStateTimeline,
    current_time: float,
    k_n: float,
) -> PublicFaultObservation:
    """Project a validated timeline for deterministic Stage 1 traces."""

    if not isinstance(timeline, CommonStateTimeline):
        raise ValueError("timeline must be CommonStateTimeline")
    now = _strict_real(current_time, "current_time")
    k_value = _strict_real(k_n, "k_n")
    phase = phase_at(timeline, now)
    health = state_at(timeline, now)
    starts = {
        Phase.HEALTHY: 0.0,
        Phase.DEGRADED: timeline.degraded_start,
        Phase.FAILED: timeline.failed_start,
        Phase.RECOVERED: timeline.recovered_start,
    }
    transitions = tuple(
        (label, boundary)
        for label, boundary in (
            ("D", timeline.degraded_start),
            ("F", timeline.failed_start),
            ("R", timeline.recovered_start),
        )
        if boundary <= now
    )
    start = starts[phase]
    return PublicFaultObservation(now, health, phase, start, now - start,
                                  transitions, k_value)


@dataclass(frozen=True)
class LocalTokenSpec:
    """Per-Expert Token identity before global merge."""

    local_token_id: int
    arrival_time: float
    token_class: TokenClass
    destination_replica: int

    def __post_init__(self) -> None:
        _strict_int(self.local_token_id, "local_token_id")
        object.__setattr__(self, "arrival_time", _strict_real(self.arrival_time, "arrival_time"))
        if not isinstance(self.token_class, TokenClass):
            raise ValueError("token_class must be TokenClass")
        _strict_int(self.destination_replica, "destination_replica")
        if self.destination_replica not in (1, 2):
            raise ValueError("destination_replica must be 1 or 2")


@dataclass(frozen=True)
class LocalWorkDraw:
    """All four attempt streams for one local Token, keyed before merge."""

    attempt0: tuple[float, float, float]
    attempt1: tuple[float, float, float]
    attempt2: tuple[float, float, float]
    attempt3: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("attempt0", "attempt1", "attempt2", "attempt3"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 3:
                raise ValueError(f"{name} must contain exactly three draws")
            object.__setattr__(
                self, name,
                tuple(_strict_real(value, f"{name}[{index}]", positive=True)
                      for index, value in enumerate(values)),
            )


@dataclass(frozen=True)
class ExpertWorkloadTrace:
    expert_id: int
    arrival_rate: float
    arrival_cutoff: float
    tokens: tuple[LocalTokenSpec, ...]
    work: tuple[LocalWorkDraw, ...]

    def __post_init__(self) -> None:
        _strict_int(self.expert_id, "expert_id")
        object.__setattr__(self, "arrival_rate",
                           _strict_real(self.arrival_rate, "arrival_rate", positive=True))
        cutoff = _strict_real(self.arrival_cutoff, "arrival_cutoff", positive=True)
        object.__setattr__(self, "arrival_cutoff", cutoff)
        if len(self.tokens) != len(self.work):
            raise ValueError("each Expert trace requires one-to-one Token/work rows")
        previous = -1.0
        for local_id, (token, draw) in enumerate(zip(self.tokens, self.work)):
            if not isinstance(token, LocalTokenSpec) or not isinstance(draw, LocalWorkDraw):
                raise ValueError("Expert trace rows have invalid types")
            if token.local_token_id != local_id:
                raise ValueError("local Token IDs must be dense and ordered")
            if token.arrival_time < previous or token.arrival_time >= cutoff:
                raise ValueError("local arrivals must be ordered in [0, cutoff)")
            previous = token.arrival_time

    @property
    def local_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack(">qdd", self.expert_id, self.arrival_rate,
                                  self.arrival_cutoff))
        for token, draw in zip(self.tokens, self.work):
            digest.update(struct.pack(">qdq", token.local_token_id,
                                      token.arrival_time, token.destination_replica))
            digest.update(token.token_class.value.encode("ascii"))
            digest.update(struct.pack(">12d", *(value for attempt in (
                draw.attempt0, draw.attempt1, draw.attempt2, draw.attempt3)
                for value in attempt)))
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.local_fingerprint


@dataclass(frozen=True)
class NestedPopulationTrace:
    """One idiosyncratic population attached to one common fault path."""

    fault_path: CommonFaultPath
    population_id: str | int
    expert_count: int
    experts: tuple[ExpertWorkloadTrace, ...]
    merged_tokens: tuple[tuple[int, LocalTokenSpec], ...]
    merged_work: tuple[LocalWorkDraw, ...]
    merged_expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fault_path, CommonFaultPath):
            raise ValueError("fault_path must be CommonFaultPath")
        _identifier(self.population_id, "population_id")
        _strict_int(self.expert_count, "expert_count", 1)
        if len(self.experts) != self.expert_count:
            raise ValueError("nested population Expert count mismatch")
        if any(expert.expert_id != index
               for index, expert in enumerate(self.experts)):
            raise ValueError("Expert IDs must be dense and ordered")
        if (len(self.merged_tokens) != len(self.merged_work)
                or len(self.merged_tokens) != len(self.merged_expert_ids)
                or not self.merged_tokens):
            raise ValueError("merged population must contain Token/work rows")
        for global_id, token in enumerate(self.merged_tokens):
            if not isinstance(global_id, int) or global_id != token[0]:
                raise ValueError("merged global IDs must be dense")
            if not isinstance(token[1], LocalTokenSpec):
                raise ValueError("merged Token row must contain LocalTokenSpec")
            if not isinstance(self.merged_work[global_id], LocalWorkDraw):
                raise ValueError("merged work row must contain LocalWorkDraw")
            if self.merged_expert_ids[global_id] not in range(self.expert_count):
                raise ValueError("merged Expert IDs must be inside the population")

    @property
    def common_fault_fingerprint(self) -> str:
        return self.fault_path.fingerprint

    @property
    def local_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for expert in self.experts:
            digest.update(expert.local_fingerprint.encode("ascii"))
        return digest.hexdigest()

    @property
    def trace_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps((self.population_id, self.expert_count),
                                 sort_keys=True, separators=(",", ":")).encode())
        digest.update(self.common_fault_fingerprint.encode("ascii"))
        for global_id, token in self.merged_tokens:
            digest.update(struct.pack(">qqdq", global_id, token.local_token_id,
                                      token.arrival_time, token.destination_replica))
            digest.update(token.token_class.value.encode("ascii"))
            draw = self.merged_work[global_id]
            digest.update(struct.pack(">12d", *(value for attempt in (
                draw.attempt0, draw.attempt1, draw.attempt2, draw.attempt3)
                for value in attempt)))
        return digest.hexdigest()

    @property
    def audit_identity(self) -> Mapping[str, object]:
        return MappingProxyType({
            "namespace": self.fault_path.identity.namespace,
            "macro_seed": self.fault_path.identity.macro_seed,
            "common_episode_id": self.fault_path.identity.common_episode_id,
            "population_id": self.population_id,
            "expert_count": self.expert_count,
            "fault_fingerprint": self.common_fault_fingerprint,
            "local_fingerprint": self.local_fingerprint,
            "key_schema_version": "shared-backup-game:key-v2",
            "simulator_version": _PACKAGE_VERSION,
        })

    def local_stream_key(
        self,
        expert_id: int,
        local_token_id: int,
        replica_id: int,
        attempt_id: int,
    ) -> tuple[object, ...]:
        """Return the stable idiosyncratic key used for one service draw."""

        expert = _strict_int(expert_id, "expert_id")
        local = _strict_int(local_token_id, "local_token_id")
        replica = _strict_int(replica_id, "replica_id")
        attempt = _strict_int(attempt_id, "attempt_id")
        if expert >= self.expert_count or replica not in _REPLICAS or attempt not in _ATTEMPTS:
            raise ValueError("local stream key contains an out-of-range component")
        return (
            self.fault_path.identity.namespace,
            self.fault_path.identity.macro_seed,
            self.fault_path.identity.common_episode_id,
            self.population_id,
            expert,
            local,
            replica,
            attempt,
        )

    @property
    def trace(self):
        """Lazy Stage 1 adapter; importing it here avoids a module cycle."""

        return self.to_shared_trace()

    def to_shared_trace(self):
        from .shared_backup import SharedBackupTrace, SharedTokenSpec, SharedWorkDraw

        tokens = tuple(
            SharedTokenSpec(
                global_id,
                token.arrival_time,
                token.token_class,
                self.merged_expert_ids[global_id],
                token.local_token_id,
                token.destination_replica,
            )
            for global_id, token in self.merged_tokens
        )
        work = tuple(
            SharedWorkDraw(draw.attempt0, draw.attempt1, draw.attempt2, draw.attempt3)
            for draw in self.merged_work
        )
        return SharedBackupTrace(
            expert_count=self.expert_count,
            timeline=self.fault_path.timeline,
            arrival_cutoff=self.fault_path.arrival_cutoff,
            tokens=tokens,
            work=work,
        )

def _generate_expert(
    fault_path: CommonFaultPath,
    population_id: str | int,
    expert_id: int,
) -> ExpertWorkloadTrace:
    identity = fault_path.identity
    arrival_rng = random.Random(_stable_seed(
        identity.namespace, identity.macro_seed, identity.common_episode_id,
        population_id, expert_id, "arrival",
    ))
    mu, sigma = _lognormal_parameters(_WORK_MEAN, _WORK_CV)
    tokens: list[LocalTokenSpec] = []
    draws: list[LocalWorkDraw] = []
    now = 0.0
    while True:
        now += arrival_rng.expovariate(_ARRIVAL_RATE)
        if now >= fault_path.arrival_cutoff:
            break
        local_id = len(tokens)
        cls = (
            TokenClass.REGULAR
            if random.Random(_stable_seed(identity.namespace, identity.macro_seed,
                                          identity.common_episode_id, population_id,
                                          expert_id, local_id, "class")).random()
            < _REGULAR_RATIO else TokenClass.URGENT
        )
        destination = 1 + int(random.Random(_stable_seed(
            identity.namespace, identity.macro_seed, identity.common_episode_id,
            population_id, expert_id, local_id, "destination",
        )).random() >= 0.5)
        tokens.append(LocalTokenSpec(local_id, now, cls, destination))
        attempt_draws = []
        for attempt in _ATTEMPTS:
            attempt_draws.append(tuple(
                random.Random(_stable_seed(
                    identity.namespace, identity.macro_seed,
                    identity.common_episode_id, population_id, expert_id,
                    local_id, replica, attempt, "service",
                )).lognormvariate(mu, sigma)
                for replica in _REPLICAS
            ))
        draws.append(LocalWorkDraw(*attempt_draws))
    return ExpertWorkloadTrace(expert_id, _ARRIVAL_RATE, fault_path.arrival_cutoff,
                               tuple(tokens), tuple(draws))


def build_nested_population(
    fault_path: CommonFaultPath,
    population_id: str | int,
    expert_count: int,
) -> NestedPopulationTrace:
    """Generate independent local Expert streams, then merge for output IDs."""

    if not isinstance(fault_path, CommonFaultPath):
        raise ValueError("fault_path must be CommonFaultPath")
    pop_id = _identifier(population_id, "population_id")
    count = _strict_int(expert_count, "expert_count", 1)
    experts = tuple(_generate_expert(fault_path, pop_id, expert) for expert in range(count))
    local_rows = [
        (token.arrival_time, expert.expert_id, token.local_token_id, token, draw)
        for expert in experts
        for token, draw in zip(expert.tokens, expert.work)
    ]
    local_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    merged_tokens = tuple(
        (global_id, row[3]) for global_id, row in enumerate(local_rows)
    )
    merged_work = tuple(row[4] for row in local_rows)
    merged_expert_ids = tuple(row[1] for row in local_rows)
    return NestedPopulationTrace(fault_path, pop_id, count, experts,
                                 merged_tokens, merged_work, merged_expert_ids)


generate_nested_population = build_nested_population


def build_population_replicates(
    fault_path: CommonFaultPath,
    population_ids: tuple[str | int, ...] | list[str | int],
    expert_count: int,
) -> tuple[NestedPopulationTrace, ...]:
    if not isinstance(population_ids, (tuple, list)) or not population_ids:
        raise ValueError("population_ids must be a nonempty tuple or list")
    return tuple(build_nested_population(fault_path, population_id, expert_count)
                 for population_id in population_ids)


@dataclass(frozen=True)
class ActionRule:
    """One causal four-position N/D/S/X rule, without any solver/search."""

    name: str
    actions: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if type(self.name) is not str or len(self.name) != 4:
            raise ValueError("rule name must contain four action codes")
        if not isinstance(self.actions, tuple) or len(self.actions) != 4:
            raise ValueError("rule actions must contain four entries")
        normalized = tuple(
            action.value if hasattr(action, "value") else action
            for action in self.actions
        )
        if normalized != tuple(self.name):
            raise ValueError("rule name and action entries disagree")
        if any(action not in _ACTION_CODES for action in normalized):
            raise ValueError("rule actions must be N, D, S, or X")
        object.__setattr__(self, "actions", normalized)

    @property
    def action_codes(self) -> tuple[str, str, str, str]:
        return self.actions

    def request(self, observation):
        """Select from the current observation; admission remains scheduler-owned."""

        from .shared_backup import SharedAction

        if (observation.common_state is not CommonState.DEGRADED
                or observation.phase is not Phase.DEGRADED
                or getattr(observation, "primary_replica", 0) != 0):
            return SharedAction.NORMAL
        late = int(observation.phase_age >= 50.0)
        urgent = int(observation.token_class is TokenClass.URGENT)
        return SharedAction(self.actions[2 * late + urgent])


FixedActionRule = ActionRule


def enumerate_action_rules() -> tuple[ActionRule, ...]:
    return tuple(
        ActionRule("".join(actions), actions)
        for actions in itertools.product(_ACTION_CODES, repeat=4)
    )


enumerate_rules = enumerate_action_rules


def select_online_action(rule: ActionRule, observation):
    if not isinstance(rule, ActionRule):
        raise ValueError("rule must be ActionRule")
    return rule.request(observation)


__all__ = [
    "ActionRule",
    "CommonFaultIdentity",
    "CommonFaultPath",
    "ExpertWorkloadTrace",
    "FixedActionRule",
    "LocalTokenSpec",
    "LocalWorkDraw",
    "NestedPopulationTrace",
    "PublicFaultObservation",
    "build_common_fault_path",
    "build_nested_population",
    "build_population_replicates",
    "enumerate_action_rules",
    "enumerate_rules",
    "generate_common_fault_path",
    "generate_fault_path",
    "generate_nested_population",
    "public_fault_observation",
    "public_fault_observation_from_timeline",
    "scheduled_common_fault_path",
    "select_online_action",
]
