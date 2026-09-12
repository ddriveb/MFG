from pathlib import Path
import unittest

from mfg_hedge.config import load_config
from mfg_hedge.domain import CommonState, ProtectionAction, TokenClass
from mfg_hedge.interfaces import PolicyContext
from mfg_hedge.simulation import (
    NoHedgePolicy,
    RoundRobinDispatcher,
    TokenResult,
    simulate_healthy_no_hedge,
)
from mfg_hedge.workload import TokenSpec, WorkloadTrace, generate_workload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config():
    return load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")


def build_manual_trace() -> WorkloadTrace:
    tokens = tuple(
        TokenSpec(token_id=index, arrival_time=arrival, token_class=TokenClass.REGULAR)
        for index, arrival in enumerate((0.0, 0.1, 0.2, 0.3, 0.4))
    )
    # Full-length per-Replica streams; entries are indexed by token_id so the
    # filler values 50/60/70/80/90 must never be consumed by this dispatch.
    service_times = (
        (1.0, 50.0, 2.0, 60.0, 3.0),
        (70.0, 0.5, 80.0, 1.5, 90.0),
    )
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.0,
        tokens=tokens,
        service_times=service_times,
    )


def build_trace_with_token_ids(token_ids: tuple[int, ...]) -> WorkloadTrace:
    tokens = tuple(
        TokenSpec(token_id=token_id, arrival_time=float(position), token_class=TokenClass.REGULAR)
        for position, token_id in enumerate(token_ids)
    )
    stream_length = max((max(token_ids, default=0) + 1), len(token_ids), 1)
    service_times = tuple(
        tuple(1.0 for _ in range(stream_length)) for _ in range(2)
    )
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.0,
        tokens=tokens,
        service_times=service_times,
    )


class RoundRobinDispatcherTests(unittest.TestCase):
    def test_alternates_between_replicas(self) -> None:
        dispatcher = RoundRobinDispatcher(replica_count=2)
        assigned = [dispatcher.assign(token_id) for token_id in range(6)]
        self.assertEqual(assigned, [0, 1, 0, 1, 0, 1])


class NoHedgePolicyTests(unittest.TestCase):
    def test_implements_protection_policy_interface(self) -> None:
        policy = NoHedgePolicy()
        context = PolicyContext(
            expert_id=0,
            common_state=CommonState.HEALTHY,
            utilization=0.0,
            hedge_price=0.0,
        )
        distribution = policy.distribution(context)
        for token_class in TokenClass:
            self.assertEqual(
                dict(distribution[token_class]), {ProtectionAction.NORMAL: 1.0}
            )

    def test_simulator_rejects_non_normal_policy(self) -> None:
        class AlwaysImmediateHedge:
            def distribution(self, context: PolicyContext):
                return {
                    token_class: {ProtectionAction.IMMEDIATE_HEDGE: 1.0}
                    for token_class in TokenClass
                }

        with self.assertRaisesRegex(ValueError, "Normal"):
            simulate_healthy_no_hedge(
                build_manual_trace(), replica_count=2, policy=AlwaysImmediateHedge()
            )


class ManualTraceSimulationTests(unittest.TestCase):
    def test_fcfs_start_and_completion_times(self) -> None:
        result = simulate_healthy_no_hedge(build_manual_trace(), replica_count=2)
        expected = [
            # (replica, start, completion) using service times bound by token_id
            (0, 0.0, 1.0),
            (1, 0.1, 0.6),
            (0, 1.0, 3.0),
            (1, 0.6, 2.1),
            (0, 3.0, 6.0),
        ]
        self.assertEqual(len(result.tokens), 5)
        for token, (replica, start, completion) in zip(result.tokens, expected):
            self.assertEqual(token.primary_replica, replica)
            self.assertAlmostEqual(token.start_time, start)
            self.assertAlmostEqual(token.completion_time, completion)
            self.assertAlmostEqual(token.completion_time, token.start_time + token.service_time)
            self.assertAlmostEqual(token.latency, token.completion_time - token.arrival_time)
            self.assertAlmostEqual(token.queue_delay, token.start_time - token.arrival_time)

    def test_service_time_is_bound_to_token_and_replica(self) -> None:
        trace = build_manual_trace()
        result = simulate_healthy_no_hedge(trace, replica_count=2)
        for token in result.tokens:
            self.assertEqual(
                token.service_time,
                trace.service_times[token.primary_replica][token.token_id],
            )

    def test_assigned_counts_and_busy_time(self) -> None:
        result = simulate_healthy_no_hedge(build_manual_trace(), replica_count=2)
        self.assertEqual(result.assigned_counts, (3, 2))
        self.assertAlmostEqual(result.busy_times[0], 6.0)
        self.assertAlmostEqual(result.busy_times[1], 2.0)
        self.assertAlmostEqual(result.horizon, 6.0)

    def test_token_result_carries_required_fields(self) -> None:
        result = simulate_healthy_no_hedge(build_manual_trace(), replica_count=2)
        token = result.tokens[0]
        self.assertIsInstance(token, TokenResult)
        self.assertEqual(token.token_id, 0)
        self.assertEqual(token.token_class, TokenClass.REGULAR)
        self.assertAlmostEqual(token.arrival_time, 0.0)

    def test_replica_count_mismatch_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream"):
            simulate_healthy_no_hedge(build_manual_trace(), replica_count=3)

    def test_undersized_service_stream_raises_clear_error(self) -> None:
        trace = build_manual_trace()
        short = WorkloadTrace(
            base_seed=trace.base_seed,
            arrival_rate=trace.arrival_rate,
            tokens=trace.tokens,
            service_times=(trace.service_times[0][:2], trace.service_times[1]),
        )
        with self.assertRaisesRegex(ValueError, "stream"):
            simulate_healthy_no_hedge(short, replica_count=2)

    def test_negative_token_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Token ids"):
            simulate_healthy_no_hedge(
                build_trace_with_token_ids((-1, 0)), replica_count=2
            )

    def test_duplicate_token_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Token ids"):
            simulate_healthy_no_hedge(
                build_trace_with_token_ids((0, 0)), replica_count=2
            )

    def test_sparse_token_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Token ids"):
            simulate_healthy_no_hedge(
                build_trace_with_token_ids((0, 2)), replica_count=2
            )

    def test_out_of_order_token_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Token ids"):
            simulate_healthy_no_hedge(
                build_trace_with_token_ids((1, 0)), replica_count=2
            )


class GeneratedTraceSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_default_config()
        self.token_count = 500
        self.trace = generate_workload(config, self.token_count)
        self.result = simulate_healthy_no_hedge(
            self.trace, replica_count=config.replicas_per_expert
        )

    def test_completed_tokens_equal_generated_tokens(self) -> None:
        self.assertEqual(len(self.result.tokens), self.token_count)

    def test_queue_equations_hold_for_every_token(self) -> None:
        for token in self.result.tokens:
            self.assertGreaterEqual(token.start_time, token.arrival_time)
            self.assertEqual(
                token.completion_time, token.start_time + token.service_time
            )
            self.assertEqual(
                token.latency, token.completion_time - token.arrival_time
            )
            self.assertEqual(token.queue_delay, token.start_time - token.arrival_time)

    def test_service_times_match_the_stable_key_binding(self) -> None:
        for token in self.result.tokens:
            self.assertEqual(
                token.service_time,
                self.trace.service_times[token.primary_replica][token.token_id],
            )

    def test_round_robin_assignment_differs_by_at_most_one(self) -> None:
        counts = self.result.assigned_counts
        self.assertEqual(sum(counts), self.token_count)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_invariant_counters_show_no_hedge_no_replay(self) -> None:
        counters = self.result.counters
        self.assertEqual(counters.primary_executions, self.token_count)
        self.assertEqual(counters.hedge_launches, 0)
        self.assertEqual(counters.replay_executions, 0)

    def test_utilization_stays_within_unit_interval(self) -> None:
        for busy in self.result.busy_times:
            utilization = busy / self.result.horizon
            self.assertGreaterEqual(utilization, 0.0)
            self.assertLessEqual(utilization, 1.0)


if __name__ == "__main__":
    unittest.main()
