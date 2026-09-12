"""Deterministic workload generation, independent of any policy code.

The workload trace is generated once from ``base_seed`` before any policy or
dispatcher runs. Arrival, Token-class, and per-Replica service-time draws come
from separate random streams so future policy comparisons can reuse common
random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import random

from .config import ExperimentConfig
from .domain import TokenClass


def lognormal_parameters(mean: float, cv: float) -> tuple[float, float]:
    """Return ``(mu, sigma)`` matching the configured arithmetic mean and CV."""
    if mean <= 0.0:
        raise ValueError("mean must be positive")
    if cv <= 0.0:
        raise ValueError("cv must be positive")
    sigma2 = math.log(1.0 + cv**2)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, sigma


def _stream_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class TokenSpec:
    token_id: int
    arrival_time: float
    token_class: TokenClass


@dataclass(frozen=True)
class WorkloadTrace:
    """Immutable pre-generated workload shared by every compared policy.

    ``service_times[replica][token_id]`` is the lognormal draw bound to the
    stable key ``(token_id, replica_id, attempt=0)`` — the Primary execution
    of that Token on that Replica. A simulator indexes streams by this key, so
    a change of dispatch or a future Hedge/Replay attempt never shifts a
    service draw to another Token copy; later attempts will extend the key
    with ``attempt > 0`` draws instead of reusing these.

    ``replay_service_times[replica][token_id]`` is the independent draw bound
    to ``(token_id, replica_id, attempt=1)`` — the single Replay allowed after
    a lost Primary. ``hedge_service_times[replica][token_id]`` is the draw
    bound to ``(token_id, replica_id, attempt=2)`` — the Hedge Backup. Both are
    ``None`` for traces without their support; paths that do not Hedge never
    read attempt-2.
    """

    base_seed: int
    arrival_rate: float
    tokens: tuple[TokenSpec, ...]
    service_times: tuple[tuple[float, ...], ...]
    replay_service_times: tuple[tuple[float, ...], ...] | None = None
    hedge_service_times: tuple[tuple[float, ...], ...] | None = None


def generate_workload(
    config: ExperimentConfig,
    token_count: int,
    base_seed: int | None = None,
) -> WorkloadTrace:
    """Generate ``token_count`` Poisson arrivals for the Healthy slice.

    The arrival rate is ``healthy_offered_load * aggregate_service_capacity``,
    where aggregate capacity is ``replicas_per_expert / healthy_service_mean``.
    """
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    seed = config.base_seed if base_seed is None else base_seed
    arrival_rate = config.healthy_offered_load * (
        config.replicas_per_expert / config.healthy_service_mean
    )

    arrival_rng = random.Random(_stream_seed(seed, "arrival"))
    class_rng = random.Random(_stream_seed(seed, "token-class"))

    tokens: list[TokenSpec] = []
    now = 0.0
    for token_id in range(token_count):
        now += arrival_rng.expovariate(arrival_rate)
        token_class = (
            TokenClass.REGULAR
            if class_rng.random() < config.regular_token_ratio
            else TokenClass.URGENT
        )
        tokens.append(TokenSpec(token_id=token_id, arrival_time=now, token_class=token_class))

    mu, sigma = lognormal_parameters(config.healthy_service_mean, config.service_time_cv)
    service_times = []
    for replica_id in range(config.replicas_per_expert):
        replica_rng = random.Random(_stream_seed(seed, f"service:{replica_id}"))
        service_times.append(
            tuple(replica_rng.lognormvariate(mu, sigma) for _ in range(token_count))
        )

    return WorkloadTrace(
        base_seed=seed,
        arrival_rate=arrival_rate,
        tokens=tuple(tokens),
        service_times=tuple(service_times),
    )


def generate_workload_with_replay(
    config: ExperimentConfig,
    token_count: int,
    base_seed: int | None = None,
) -> WorkloadTrace:
    """Generate a trace carrying both attempt-0 and attempt-1 service draws.

    The attempt-0 streams, arrivals, and Token classes are produced by
    `generate_workload` itself, so they are value-identical to a plain trace
    with the same config and seed. Attempt-1 draws come from separate streams
    labeled ``service:{replica}:attempt1`` and never perturb the base streams.
    """
    trace = generate_workload(config, token_count, base_seed)
    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )
    replay_streams = []
    for replica_id in range(config.replicas_per_expert):
        replica_rng = random.Random(
            _stream_seed(trace.base_seed, f"service:{replica_id}:attempt1")
        )
        replay_streams.append(
            tuple(replica_rng.lognormvariate(mu, sigma) for _ in range(token_count))
        )
    return replace(trace, replay_service_times=tuple(replay_streams))


def generate_workload_with_hedge(
    config: ExperimentConfig,
    token_count: int,
    base_seed: int | None = None,
) -> WorkloadTrace:
    """Generate a trace carrying attempt-0, attempt-1, and attempt-2 draws.

    Built on `generate_workload_with_replay`, so arrivals, Token classes,
    attempt-0, and attempt-1 streams are value-identical to the plain
    generators at the same config and seed. Attempt-2 (Hedge Backup) draws use
    the independent label ``service:{replica}:attempt2``.
    """
    trace = generate_workload_with_replay(config, token_count, base_seed)
    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )
    hedge_streams = []
    for replica_id in range(config.replicas_per_expert):
        replica_rng = random.Random(
            _stream_seed(trace.base_seed, f"service:{replica_id}:attempt2")
        )
        hedge_streams.append(
            tuple(replica_rng.lognormvariate(mu, sigma) for _ in range(token_count))
        )
    return replace(trace, hedge_service_times=tuple(hedge_streams))


def validate_hedge_stream_shape(trace: WorkloadTrace, replica_count: int) -> None:
    """Validate attempt-2 stream shape at the hedge engine's boundary."""
    streams = trace.hedge_service_times
    if streams is None:
        raise ValueError("trace carries no hedge service-time streams")
    if len(streams) != replica_count:
        raise ValueError(
            f"trace carries {len(streams)} hedge service-time streams but "
            f"replica_count is {replica_count}"
        )
    max_token_id = len(trace.tokens) - 1
    for replica_id, stream in enumerate(streams):
        if len(stream) <= max_token_id:
            raise ValueError(
                f"hedge service-time stream of Replica {replica_id} has "
                f"{len(stream)} values but the largest Token id is {max_token_id}"
            )


def validate_stream_values(
    streams: tuple[tuple[float, ...], ...],
    attempt_id: int,
    kind: str,
) -> None:
    """Require every draw to be a real, finite, positive number.

    Applied by the hedge-capable engine before its event loop to every stream
    the run may consume; errors locate the (attempt, replica, Token) key.
    """
    for replica_id, stream in enumerate(streams):
        for token_id, value in enumerate(stream):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{kind} draw for (token_id={token_id}, "
                    f"replica_id={replica_id}, attempt_id={attempt_id}) must be "
                    f"finite and > 0, got {value!r}"
                )


def validate_token_specs(trace: WorkloadTrace) -> None:
    """Require finite, non-negative, non-decreasing arrivals and valid classes."""
    previous = 0.0
    for spec in trace.tokens:
        arrival = spec.arrival_time
        if (
            isinstance(arrival, bool)
            or not isinstance(arrival, (int, float))
            or not math.isfinite(arrival)
            or arrival < 0.0
        ):
            raise ValueError(
                f"arrival_time of Token {spec.token_id} must be finite and "
                f">= 0, got {arrival!r}"
            )
        if arrival < previous:
            raise ValueError(
                f"arrival times must be non-decreasing; Token "
                f"{spec.token_id} arrives at {arrival} after {previous}"
            )
        if not isinstance(spec.token_class, TokenClass):
            raise ValueError(
                f"token_class of Token {spec.token_id} must be a TokenClass, "
                f"got {spec.token_class!r}"
            )
        previous = arrival


def validate_trace_shape(trace: WorkloadTrace, replica_count: int) -> None:
    """Validate attempt-0 stream shape and dense Token ids before evaluation."""
    if len(trace.service_times) != replica_count:
        raise ValueError(
            f"trace carries {len(trace.service_times)} per-Replica service-time "
            f"streams but replica_count is {replica_count}"
        )
    if not trace.tokens:
        raise ValueError("trace contains no Tokens")
    token_ids = tuple(spec.token_id for spec in trace.tokens)
    expected_token_ids = tuple(range(len(trace.tokens)))
    if token_ids != expected_token_ids:
        raise ValueError(
            "Token ids must be unique, dense, non-negative, and ordered exactly "
            f"as 0..{len(trace.tokens) - 1}; got {token_ids!r}"
        )
    max_token_id = len(trace.tokens) - 1
    for replica_id, stream in enumerate(trace.service_times):
        if len(stream) <= max_token_id:
            raise ValueError(
                f"service-time stream of Replica {replica_id} has {len(stream)} "
                f"values but the largest Token id is {max_token_id}"
            )


def validate_replay_stream_shape(trace: WorkloadTrace, replica_count: int) -> None:
    """Validate attempt-1 stream shape at the engine's consumption boundary.

    The Common State event engine must call this once before evaluation; the
    Healthy simulator must not (it never reads attempt-1 streams).
    """
    if (
        isinstance(replica_count, bool)
        or not isinstance(replica_count, int)
        or replica_count <= 0
    ):
        raise ValueError(
            f"replica_count must be a positive int, got {replica_count!r}"
        )
    streams = trace.replay_service_times
    if streams is None:
        raise ValueError("trace carries no replay service-time streams")
    if len(streams) != replica_count:
        raise ValueError(
            f"trace carries {len(streams)} replay service-time streams but "
            f"replica_count is {replica_count}"
        )
    max_token_id = len(trace.tokens) - 1
    for replica_id, stream in enumerate(streams):
        if len(stream) <= max_token_id:
            raise ValueError(
                f"replay service-time stream of Replica {replica_id} has "
                f"{len(stream)} values but the largest Token id is {max_token_id}"
            )
