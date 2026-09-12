"""Healthy + No Hedge event evaluation over a pre-generated workload.

This module only evaluates: it consumes an immutable WorkloadTrace and never
draws random numbers. Every Token takes the Normal protection action, so each
Token has exactly one Primary execution, no Backup launch, and no Replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import CommonState, ProtectionAction, TokenClass
from .interfaces import PolicyContext, PolicyDistribution
from .workload import WorkloadTrace, validate_trace_shape


class RoundRobinDispatcher:
    """Fixed deterministic Dispatcher: Token ``k`` goes to Replica ``k % n``.

    Round-robin is the documented fixed dispatch choice for this slice; it
    keeps per-Replica assignment counts balanced within one Token.
    """

    def __init__(self, replica_count: int) -> None:
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        self.replica_count = replica_count

    def assign(self, token_id: int) -> int:
        return token_id % self.replica_count


class NoHedgePolicy:
    """Assigns probability 1 to the Normal action for every TokenClass.

    Implements the ProtectionPolicy interface and consumes no randomness, so
    it cannot perturb the pre-generated workload streams.
    """

    def distribution(self, context: PolicyContext) -> PolicyDistribution:
        return {
            token_class: {ProtectionAction.NORMAL: 1.0} for token_class in TokenClass
        }


@dataclass(frozen=True)
class TokenResult:
    token_id: int
    token_class: TokenClass
    arrival_time: float
    primary_replica: int
    start_time: float
    service_time: float
    completion_time: float
    queue_delay: float
    latency: float


@dataclass(frozen=True)
class InvariantCounters:
    primary_executions: int
    hedge_launches: int
    replay_executions: int


@dataclass(frozen=True)
class SimulationResult:
    tokens: tuple[TokenResult, ...]
    assigned_counts: tuple[int, ...]
    busy_times: tuple[float, ...]
    horizon: float
    counters: InvariantCounters


def _require_normal_only(policy: NoHedgePolicy) -> None:
    # The policy in this slice is context-independent, so one query against the
    # initial Healthy context pins down every future decision.
    context = PolicyContext(
        expert_id=0,
        common_state=CommonState.HEALTHY,
        utilization=0.0,
        hedge_price=0.0,
    )
    distribution = policy.distribution(context)
    for token_class in TokenClass:
        class_distribution = distribution.get(token_class, {})
        if not (
            len(class_distribution) == 1
            and class_distribution.get(ProtectionAction.NORMAL) == 1.0
        ):
            raise ValueError(
                "Healthy + No Hedge requires a Normal-only distribution; "
                f"TokenClass.{token_class.name} maps to {dict(class_distribution)}"
            )


def simulate_healthy_no_hedge(
    trace: WorkloadTrace,
    replica_count: int,
    policy: NoHedgePolicy | None = None,
) -> SimulationResult:
    """Run per-Replica FCFS queueing over ``trace`` with Normal-only actions.

    Each Token's service time is bound by the stable key
    ``trace.service_times[replica][token_id]`` (attempt 0, the Primary), so a
    change of dispatch or future Hedge attempts never shifts a service draw to
    another Token copy.
    """
    validate_trace_shape(trace, replica_count)
    _require_normal_only(policy if policy is not None else NoHedgePolicy())
    dispatcher = RoundRobinDispatcher(replica_count)

    available_at = [0.0] * replica_count
    busy_times = [0.0] * replica_count
    assigned_counts = [0] * replica_count
    results: list[TokenResult] = []
    primary_executions = 0
    hedge_launches = 0
    replay_executions = 0

    for spec in trace.tokens:
        replica = dispatcher.assign(spec.token_id)
        service_time = trace.service_times[replica][spec.token_id]
        start_time = max(spec.arrival_time, available_at[replica])
        completion_time = start_time + service_time
        available_at[replica] = completion_time
        busy_times[replica] += service_time
        assigned_counts[replica] += 1
        primary_executions += 1
        results.append(
            TokenResult(
                token_id=spec.token_id,
                token_class=spec.token_class,
                arrival_time=spec.arrival_time,
                primary_replica=replica,
                start_time=start_time,
                service_time=service_time,
                completion_time=completion_time,
                queue_delay=start_time - spec.arrival_time,
                latency=completion_time - spec.arrival_time,
            )
        )

    horizon = max(available_at) if results else 0.0
    return SimulationResult(
        tokens=tuple(results),
        assigned_counts=tuple(assigned_counts),
        busy_times=tuple(busy_times),
        horizon=horizon,
        counters=InvariantCounters(
            primary_executions=primary_executions,
            hedge_launches=hedge_launches,
            replay_executions=replay_executions,
        ),
    )
