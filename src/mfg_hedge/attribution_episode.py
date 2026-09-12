"""Independent fixed-horizon episodes for the attribution study.

An episode owns its random namespace and a fresh event-engine instance.  New
arrivals are generated on the half-open interval ``[0, arrival_cutoff)``;
accepted work is then drained to terminal state.  Arm names and quota settings
are deliberately absent from the identity so every arm can reuse one immutable
CRN trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping

from .common_state import CommonStateTimeline, validate_slowdown
from .config import ExperimentConfig
from .domain import ProtectionAction
from .hedge_simulation import (
    HedgeSimulationResult,
    _simulate_hedge_common_state_with_engine,
)
from .paired import trace_identity_fingerprint
from .workload import (
    WorkloadTrace,
    _stream_seed,
    generate_workload_with_hedge,
    validate_hedge_stream_shape,
    validate_replay_stream_shape,
    validate_stream_values,
    validate_token_specs,
    validate_trace_shape,
)


def _require_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


@dataclass(frozen=True)
class EpisodeIdentity:
    """Stable identity of one episode inside one macro-seed."""

    namespace: str
    macro_seed: int
    episode_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace.strip()
            or any(character.isspace() for character in self.namespace)
        ):
            raise ValueError(
                "namespace must be a non-empty string without whitespace, "
                f"got {self.namespace!r}"
            )
        _require_non_negative_int(self.macro_seed, "macro_seed")
        _require_non_negative_int(self.episode_index, "episode_index")

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.macro_seed}:episode:{self.episode_index}"


@dataclass(frozen=True)
class EpisodeProtocol:
    """Failure timeline and half-open arrival horizon for one episode."""

    timeline: CommonStateTimeline
    arrival_cutoff: float

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, CommonStateTimeline):
            raise ValueError(
                "timeline must be a CommonStateTimeline, "
                f"got {self.timeline!r}"
            )
        value = self.arrival_cutoff
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= self.timeline.recovered_start
        ):
            raise ValueError(
                "arrival_cutoff must be finite and strictly later than "
                f"recovered_start, got {value!r}"
            )
        object.__setattr__(self, "arrival_cutoff", float(value))


# Frozen confirmatory episode clock from ADR-0010/spec section 9.  Tests may
# construct smaller protocols under test-only namespaces, but production code
# can import this single canonical value rather than restating four numbers.
ATTRIBUTION_V1_PROTOCOL = EpisodeProtocol(
    CommonStateTimeline(100.0, 200.0, 220.0), arrival_cutoff=320.0
)


@dataclass(frozen=True)
class EpisodeTrace:
    """An immutable workload plus all identity needed to audit its CRN key."""

    identity: EpisodeIdentity
    episode_seed: int
    protocol: EpisodeProtocol
    workload: WorkloadTrace

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EpisodeIdentity):
            raise ValueError(
                f"identity must be an EpisodeIdentity, got {self.identity!r}"
            )
        _require_non_negative_int(self.episode_seed, "episode_seed")
        if not isinstance(self.protocol, EpisodeProtocol):
            raise ValueError(
                f"protocol must be an EpisodeProtocol, got {self.protocol!r}"
            )
        if not isinstance(self.workload, WorkloadTrace):
            raise ValueError(
                f"workload must be a WorkloadTrace, got {self.workload!r}"
            )
        if self.workload.base_seed != self.episode_seed:
            raise ValueError(
                "workload base_seed must equal episode_seed, got "
                f"{self.workload.base_seed!r} and {self.episode_seed!r}"
            )
        validate_trace_shape(self.workload, 2)
        validate_replay_stream_shape(self.workload, 2)
        validate_hedge_stream_shape(self.workload, 2)
        validate_token_specs(self.workload)
        validate_stream_values(self.workload.service_times, 0, "service")
        validate_stream_values(
            self.workload.replay_service_times, 1, "replay"
        )
        validate_stream_values(
            self.workload.hedge_service_times, 2, "hedge"
        )
        offending = next(
            (
                spec
                for spec in self.workload.tokens
                if spec.arrival_time >= self.protocol.arrival_cutoff
            ),
            None,
        )
        if offending is not None:
            raise ValueError(
                "episode arrivals must satisfy arrival_time < arrival_cutoff; "
                f"Token {offending.token_id} arrives at "
                f"{offending.arrival_time!r} with cutoff "
                f"{self.protocol.arrival_cutoff!r}"
            )


@dataclass(frozen=True)
class BoundarySnapshot:
    """Pure-read resource state at a domain-state transition."""

    queued_attempts: tuple[int, int]
    live_attempts: tuple[int, int]
    remaining_work: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("queued_attempts", "live_attempts", "remaining_work"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 2:
                raise ValueError(f"{name} must be a two-Replica tuple, got {values!r}")
        for replica, (queued, live, remaining) in enumerate(
            zip(self.queued_attempts, self.live_attempts, self.remaining_work)
        ):
            for value, name in ((queued, "queued"), (live, "live")):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"Replica {replica} {name} attempt count must be a "
                        f"non-negative int, got {value!r}"
                    )
            if queued > live or live - queued not in (0, 1):
                raise ValueError(
                    f"Replica {replica} requires queued <= live <= queued + 1, "
                    f"got queued={queued}, live={live}"
                )
            if (
                isinstance(remaining, bool)
                or not isinstance(remaining, (int, float))
                or not math.isfinite(float(remaining))
                or float(remaining) < 0.0
            ):
                raise ValueError(
                    f"Replica {replica} remaining_work must be finite and "
                    f"non-negative, got {remaining!r}"
                )
            if live == 0 and float(remaining) != 0.0:
                raise ValueError(
                    f"Replica {replica} has no live attempts but nonzero "
                    f"remaining_work {remaining!r}"
                )


@dataclass(frozen=True)
class EpisodeSimulationResult:
    """Legacy engine result augmented only with episode-level observations."""

    episode: EpisodeTrace
    simulation: HedgeSimulationResult
    failed_start_snapshot: BoundarySnapshot
    recovered_start_snapshot: BoundarySnapshot
    post_cutoff_drain_duration: float
    degraded_slowdown: float
    hedge_delay: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.episode, EpisodeTrace):
            raise ValueError(f"episode must be an EpisodeTrace, got {self.episode!r}")
        if not isinstance(self.simulation, HedgeSimulationResult):
            raise ValueError(
                "simulation must be a HedgeSimulationResult, "
                f"got {self.simulation!r}"
            )
        for name in ("failed_start_snapshot", "recovered_start_snapshot"):
            if not isinstance(getattr(self, name), BoundarySnapshot):
                raise ValueError(f"{name} must be a BoundarySnapshot")
        duration = self.post_cutoff_drain_duration
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise ValueError(
                "post_cutoff_drain_duration must be finite and non-negative, "
                f"got {duration!r}"
            )
        object.__setattr__(self, "post_cutoff_drain_duration", float(duration))
        object.__setattr__(
            self, "degraded_slowdown", validate_slowdown(self.degraded_slowdown)
        )
        if self.hedge_delay is not None:
            delay = self.hedge_delay
            if (
                isinstance(delay, bool)
                or not isinstance(delay, (int, float))
                or not math.isfinite(float(delay))
                or float(delay) <= 0.0
            ):
                raise ValueError(
                    f"hedge_delay must be finite and > 0, got {delay!r}"
                )
            object.__setattr__(self, "hedge_delay", float(delay))


def derive_episode_seed(base_seed: int, identity: EpisodeIdentity) -> int:
    """Derive one episode seed directly from its stable, collision-safe key."""
    root = _require_non_negative_int(base_seed, "base_seed")
    if not isinstance(identity, EpisodeIdentity):
        raise ValueError(
            f"identity must be an EpisodeIdentity, got {identity!r}"
        )
    encoded = json.dumps(
        {
            "base_seed": root,
            "namespace": identity.namespace,
            "macro_seed": identity.macro_seed,
            "episode_index": identity.episode_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


def _arrival_count_before(
    config: ExperimentConfig, episode_seed: int, arrival_cutoff: float
) -> int:
    arrival_rate = config.healthy_offered_load * (
        config.replicas_per_expert / config.healthy_service_mean
    )
    arrival_rng = random.Random(_stream_seed(episode_seed, "arrival"))
    count = 0
    now = 0.0
    while True:
        now += arrival_rng.expovariate(arrival_rate)
        if now >= arrival_cutoff:
            break
        count += 1
    if count == 0:
        raise ValueError(
            "episode contains no arrivals before arrival_cutoff; the stable "
            "episode key is retained and must not be silently replaced"
        )
    return count


def generate_episode_trace(
    config: ExperimentConfig,
    namespace: str,
    macro_seed: int,
    episode_index: int,
    protocol: EpisodeProtocol,
) -> EpisodeTrace:
    """Generate the exact existing-CRN prefix whose arrivals are below cutoff."""
    if not isinstance(config, ExperimentConfig):
        raise ValueError(f"config must be an ExperimentConfig, got {config!r}")
    identity = EpisodeIdentity(namespace, macro_seed, episode_index)
    if not isinstance(protocol, EpisodeProtocol):
        raise ValueError(f"protocol must be an EpisodeProtocol, got {protocol!r}")
    episode_seed = derive_episode_seed(config.base_seed, identity)
    token_count = _arrival_count_before(
        config, episode_seed, protocol.arrival_cutoff
    )
    workload = generate_workload_with_hedge(
        config, token_count, base_seed=episode_seed
    )
    return EpisodeTrace(identity, episode_seed, protocol, workload)


def generate_macro_seed_episodes(
    config: ExperimentConfig,
    namespace: str,
    macro_seed: int,
    episode_count: int,
    protocol: EpisodeProtocol,
) -> tuple[EpisodeTrace, ...]:
    """Generate dense, independently keyed episodes for one macro-seed."""
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
    ):
        raise ValueError(
            f"episode_count must be a positive int, got {episode_count!r}"
        )
    return tuple(
        generate_episode_trace(
            config, namespace, macro_seed, episode_index, protocol
        )
        for episode_index in range(episode_count)
    )


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def episode_trace_fingerprint(episode: EpisodeTrace) -> dict[str, str]:
    """Fingerprint episode identity/protocol and all three attempt streams."""
    if not isinstance(episode, EpisodeTrace):
        raise ValueError(f"episode must be an EpisodeTrace, got {episode!r}")
    stream_hashes = trace_identity_fingerprint(episode.workload)
    protocol_hash = _digest(
        {
            "identity": episode.identity.key,
            "episode_seed": episode.episode_seed,
            "arrival_rate": episode.workload.arrival_rate,
            "arrival_cutoff": episode.protocol.arrival_cutoff,
            "timeline": {
                "degraded_start": episode.protocol.timeline.degraded_start,
                "failed_start": episode.protocol.timeline.failed_start,
                "recovered_start": episode.protocol.timeline.recovered_start,
            },
        }
    )
    return {"episode_protocol": protocol_hash, **stream_hashes}


def simulate_episode(
    episode: EpisodeTrace,
    degraded_slowdown: float,
    actions: Mapping[int, ProtectionAction] | None = None,
    hedge_delay: float | None = None,
) -> EpisodeSimulationResult:
    """Run one fresh engine, stop arrivals at cutoff, and drain accepted work."""
    if not isinstance(episode, EpisodeTrace):
        raise ValueError(f"episode must be an EpisodeTrace, got {episode!r}")
    result, engine = _simulate_hedge_common_state_with_engine(
        episode.workload,
        episode.protocol.timeline,
        degraded_slowdown,
        actions=actions,
        hedge_delay=hedge_delay,
        replica_count=2,
    )
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
