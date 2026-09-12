"""Deterministic Failure/Replay event-engine tests (spec section 8)."""

import math
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.common_state_simulation import (
    AttemptStatus,
    simulate_common_state_no_hedge,
)
from mfg_hedge.config import load_config
from mfg_hedge.domain import TokenClass
from mfg_hedge.simulation import simulate_healthy_no_hedge
from mfg_hedge.workload import (
    TokenSpec,
    WorkloadTrace,
    generate_workload_with_replay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMELINE = CommonStateTimeline(
    degraded_start=10.0, failed_start=20.0, recovered_start=30.0
)
SLOWDOWN = 2.0


def build_manual_trace(
    arrivals: tuple[float, ...],
    service_r0: tuple[float, ...],
    service_r1: tuple[float, ...],
    replay_r0: tuple[float, ...],
    replay_r1: tuple[float, ...],
) -> WorkloadTrace:
    count = len(arrivals)
    for name, stream in (
        ("service_r0", service_r0),
        ("service_r1", service_r1),
        ("replay_r0", replay_r0),
        ("replay_r1", replay_r1),
    ):
        if len(stream) != count:
            raise ValueError(f"{name} must cover every Token id")
    tokens = tuple(
        TokenSpec(token_id=index, arrival_time=arrival, token_class=TokenClass.REGULAR)
        for index, arrival in enumerate(arrivals)
    )
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.0,
        tokens=tokens,
        service_times=(service_r0, service_r1),
        replay_service_times=(replay_r0, replay_r1),
    )


def run(trace: WorkloadTrace, timeline: CommonStateTimeline = TIMELINE):
    return simulate_common_state_no_hedge(
        trace, timeline=timeline, degraded_slowdown=SLOWDOWN
    )


class Scenario01HealthyCompletionUnaffectedTests(unittest.TestCase):
    def test_completion_inside_h_is_unaffected_by_later_states(self) -> None:
        result = run(build_manual_trace((2.0,), (4.0,), (9.0,), (9.0,), (9.0,)))
        token = result.tokens[0]
        self.assertEqual(token.completion_time, 6.0)
        self.assertEqual(token.latency, 4.0)
        self.assertEqual(token.completion_phase, Phase.HEALTHY)
        self.assertEqual(token.replay_count, 0)
        self.assertEqual(len(token.attempts), 1)
        attempt = token.attempts[0]
        self.assertEqual(attempt.status, AttemptStatus.COMPLETED)
        self.assertEqual(attempt.executed_work, 4.0)
        self.assertEqual(attempt.remaining_work, 0.0)


class Scenario02DegradedSlowdownTests(unittest.TestCase):
    def test_crossing_into_degraded_continues_remaining_work_at_half_speed(self) -> None:
        result = run(build_manual_trace((9.0,), (2.0,), (9.0,), (9.0,), (9.0,)))
        token = result.tokens[0]
        self.assertAlmostEqual(token.completion_time, 12.0)
        self.assertAlmostEqual(token.latency, 3.0)
        self.assertEqual(token.completion_phase, Phase.DEGRADED)
        attempt = token.attempts[0]
        self.assertAlmostEqual(attempt.executed_work, 2.0)
        self.assertEqual(attempt.status, AttemptStatus.COMPLETED)


class Scenario03CompletionBeforeFailureTests(unittest.TestCase):
    def test_completion_before_failed_start_is_valid(self) -> None:
        result = run(build_manual_trace((15.0,), (2.0,), (9.0,), (9.0,), (9.0,)))
        token = result.tokens[0]
        self.assertAlmostEqual(token.completion_time, 19.0)
        self.assertEqual(token.attempts[0].status, AttemptStatus.COMPLETED)
        self.assertEqual(token.replay_count, 0)


class Scenario04BoundaryCompletionFailsTests(unittest.TestCase):
    def test_completion_exactly_at_failed_start_fails_fault_first(self) -> None:
        # Starts at 15 in D (speed 0.5): physical completion is exactly 20.0.
        trace = build_manual_trace((15.0,), (2.5,), (9.0,), (9.0,), (3.0,))
        result = run(trace)
        token = result.tokens[0]
        primary = token.attempts[0]
        self.assertEqual(primary.status, AttemptStatus.FAILED_RUNNING)
        self.assertAlmostEqual(primary.terminal_time, 20.0)
        self.assertAlmostEqual(primary.executed_work, 2.5)
        self.assertAlmostEqual(primary.remaining_work, 0.0)
        replay = token.attempts[1]
        self.assertEqual(replay.attempt_id, 1)
        self.assertEqual(replay.replica_id, 1)
        self.assertAlmostEqual(replay.start_time, 20.0)
        self.assertAlmostEqual(replay.required_work, 3.0)
        self.assertAlmostEqual(token.completion_time, 23.0)
        self.assertEqual(token.completion_phase, Phase.FAILED)
        self.assertGreaterEqual(result.stale_completion_events_ignored, 1)


class Scenario05RunningFailureReplaysTests(unittest.TestCase):
    def test_running_primary_fails_and_replays_on_replica_one(self) -> None:
        trace = build_manual_trace((18.0,), (2.0,), (9.0,), (9.0,), (0.75,))
        result = run(trace)
        token = result.tokens[0]
        primary, replay = token.attempts
        self.assertEqual(primary.status, AttemptStatus.FAILED_RUNNING)
        self.assertAlmostEqual(primary.executed_work, 1.0)
        self.assertAlmostEqual(primary.remaining_work, 1.0)
        self.assertAlmostEqual(primary.terminal_time, 20.0)
        self.assertEqual(replay.status, AttemptStatus.COMPLETED)
        self.assertAlmostEqual(replay.start_time, 20.0)
        self.assertAlmostEqual(replay.completion_time if hasattr(replay, "completion_time") else replay.terminal_time, 20.75)
        self.assertAlmostEqual(token.completion_time, 20.75)


class Scenario06QueuedInvalidationTests(unittest.TestCase):
    def test_queued_primary_is_invalidated_with_zero_work(self) -> None:
        # tok0 runs on R0 from 18 (D), tok1 completes on R1 at 19, tok2 queues on R0.
        trace = build_manual_trace(
            (18.0, 18.5, 19.0),
            (8.0, 9.0, 3.0),
            (9.0, 0.5, 9.0),
            (9.0, 9.0, 9.0),
            (1.0, 9.0, 2.0),
        )
        result = run(trace)
        tok2 = result.tokens[2]
        primary = tok2.attempts[0]
        self.assertEqual(primary.status, AttemptStatus.INVALIDATED_QUEUED)
        self.assertIsNone(primary.start_time)
        self.assertIsNone(primary.queue_delay)
        self.assertEqual(primary.executed_work, 0.0)
        self.assertEqual(primary.remaining_work, 3.0)
        self.assertAlmostEqual(primary.terminal_time, 20.0)
        # Replays: running Token 0 first, then queued Token 2.
        self.assertAlmostEqual(result.tokens[0].attempts[1].start_time, 20.0)
        self.assertAlmostEqual(tok2.attempts[1].start_time, 21.0)
        self.assertAlmostEqual(result.tokens[0].completion_time, 21.0)
        self.assertAlmostEqual(tok2.completion_time, 23.0)
        self.assertEqual(result.failed_running_primary_executions, 1)
        self.assertEqual(result.invalidated_queued_primary_executions, 1)
        self.assertEqual(result.queue_length_at_failed_start, (0, 2))


class Scenario07ArrivalDuringFailureTests(unittest.TestCase):
    def test_arrival_during_failure_goes_to_replica_one_as_primary(self) -> None:
        result = run(build_manual_trace((25.0,), (9.0,), (1.5,), (9.0,), (9.0,)))
        token = result.tokens[0]
        self.assertEqual(token.primary_replica, 1)
        self.assertEqual(token.replay_count, 0)
        self.assertEqual(len(token.attempts), 1)
        self.assertEqual(token.attempts[0].attempt_id, 0)
        self.assertAlmostEqual(token.completion_time, 26.5)
        self.assertEqual(token.completion_phase, Phase.FAILED)


class Scenario08RecoveryDispatchTests(unittest.TestCase):
    def test_arrival_after_recovery_reenters_replica_zero(self) -> None:
        result = run(build_manual_trace((35.0,), (2.0,), (9.0,), (9.0,), (9.0,)))
        token = result.tokens[0]
        self.assertEqual(token.primary_replica, 0)
        self.assertAlmostEqual(token.completion_time, 37.0)
        self.assertEqual(token.completion_phase, Phase.RECOVERED)


class Scenario09ReplayDrawBindingTests(unittest.TestCase):
    def test_replay_uses_attempt_one_draw_of_replica_one(self) -> None:
        sentinel = 7.25
        trace = build_manual_trace((18.0,), (2.0,), (9.0,), (9.0,), (sentinel,))
        result = run(trace)
        replay = result.tokens[0].attempts[1]
        self.assertEqual(replay.required_work, sentinel)
        self.assertAlmostEqual(replay.terminal_time, 20.0 + sentinel)


class Scenario10StaleCompletionTests(unittest.TestCase):
    def test_stale_completion_is_ignored_and_token_completes_once(self) -> None:
        result = run(build_manual_trace((9.0,), (2.0,), (9.0,), (9.0,), (9.0,)))
        self.assertEqual(result.stale_completion_events_ignored, 1)
        token = result.tokens[0]
        completed = [a for a in token.attempts if a.status is AttemptStatus.COMPLETED]
        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(token.completion_time, 12.0)


class Scenario11DeterminismTests(unittest.TestCase):
    def test_same_inputs_reproduce_identical_results(self) -> None:
        trace = build_manual_trace(
            (18.0, 18.5, 19.0),
            (8.0, 9.0, 3.0),
            (9.0, 0.5, 9.0),
            (9.0, 9.0, 9.0),
            (1.0, 9.0, 2.0),
        )
        self.assertEqual(run(trace), run(trace))


class Scenario121314InvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        self.count = 300
        trace = generate_workload_with_replay(config, self.count)
        self.result = simulate_common_state_no_hedge(
            trace,
            timeline=CommonStateTimeline(100.0, 200.0, 220.0),
            degraded_slowdown=config.degraded_slowdown,
        )

    def test_every_token_completes_exactly_once(self) -> None:
        self.assertEqual(self.result.completed_tokens, self.count)
        for token in self.result.tokens:
            completed = [
                a for a in token.attempts if a.status is AttemptStatus.COMPLETED
            ]
            self.assertEqual(len(completed), 1, msg=f"token {token.token_id}")

    def test_every_token_replays_at_most_once(self) -> None:
        for token in self.result.tokens:
            self.assertLessEqual(token.replay_count, 1)
        replayed = sum(token.replay_count for token in self.result.tokens)
        self.assertEqual(self.result.replay_executions, replayed)
        lost = (
            self.result.failed_running_primary_executions
            + self.result.invalidated_queued_primary_executions
        )
        self.assertEqual(lost, replayed)

    def test_hedge_launches_always_zero(self) -> None:
        self.assertEqual(self.result.hedge_launches, 0)
        self.assertEqual(self.result.primary_executions, self.count)


class Scenario15HealthyRegressionTests(unittest.TestCase):
    def test_never_trigger_timeline_matches_healthy_simulator(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_replay(config, 200)
        healthy = simulate_healthy_no_hedge(trace, replica_count=2)
        horizon = healthy.horizon
        # Boundaries beyond the Healthy baseline's last completion, not merely
        # beyond the last arrival.
        timeline = CommonStateTimeline(horizon + 100.0, horizon + 200.0, horizon + 300.0)
        result = simulate_common_state_no_hedge(
            trace, timeline=timeline, degraded_slowdown=config.degraded_slowdown
        )
        self.assertEqual(result.replay_executions, 0)
        self.assertEqual(result.failed_running_primary_executions, 0)
        self.assertEqual(result.invalidated_queued_primary_executions, 0)
        self.assertEqual(result.stale_completion_events_ignored, 0)
        self.assertEqual(len(result.tokens), len(healthy.tokens))
        for token, healthy_token in zip(result.tokens, healthy.tokens):
            projection = (
                token.token_id,
                token.token_class,
                token.arrival_time,
                token.primary_replica,
                token.attempts[0].start_time,
                token.attempts[0].required_work,
                token.completion_time,
                token.attempts[0].queue_delay,
                token.latency,
            )
            healthy_projection = (
                healthy_token.token_id,
                healthy_token.token_class,
                healthy_token.arrival_time,
                healthy_token.primary_replica,
                healthy_token.start_time,
                healthy_token.service_time,
                healthy_token.completion_time,
                healthy_token.queue_delay,
                healthy_token.latency,
            )
            self.assertEqual(projection, healthy_projection, msg=f"token {token.token_id}")


class Scenario16ReplayOrderingTests(unittest.TestCase):
    def test_replay_order_at_failure_instant(self) -> None:
        # R1 already busy with tok1 (18.5-23.5) and queued tok3 when F hits.
        trace = build_manual_trace(
            (18.0, 18.5, 19.0, 19.5, 20.0),
            (8.0, 9.0, 3.0, 9.0, 9.0),
            (9.0, 5.0, 9.0, 1.0, 1.5),
            (9.0, 9.0, 9.0, 9.0, 9.0),
            (1.0, 9.0, 2.0, 9.0, 9.0),
        )
        result = run(trace)
        tokens = result.tokens
        self.assertAlmostEqual(tokens[1].attempts[0].start_time, 18.5)
        self.assertAlmostEqual(tokens[3].attempts[0].start_time, 23.5)
        # Replays: running tok0 first, then queued tok2, behind pre-existing tok3.
        self.assertAlmostEqual(tokens[0].attempts[1].start_time, 24.5)
        self.assertAlmostEqual(tokens[2].attempts[1].start_time, 25.5)
        # tok4 arrives exactly at failed_start and lands after the Replays.
        self.assertEqual(tokens[4].primary_replica, 1)
        self.assertEqual(tokens[4].replay_count, 0)
        self.assertAlmostEqual(tokens[4].attempts[0].start_time, 27.5)
        self.assertAlmostEqual(tokens[4].completion_time, 29.0)
        self.assertEqual(result.queue_length_at_failed_start, (0, 3))


class EngineContractTests(unittest.TestCase):
    def test_tokens_are_sorted_by_token_id(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_replay(config, 250)
        result = simulate_common_state_no_hedge(
            trace,
            timeline=CommonStateTimeline(100.0, 200.0, 220.0),
            degraded_slowdown=config.degraded_slowdown,
        )
        self.assertEqual(
            [token.token_id for token in result.tokens], list(range(250))
        )

    def test_drain_fields_are_consistent(self) -> None:
        result = run(
            build_manual_trace(
                (18.0, 18.5, 19.0, 19.5, 20.0),
                (8.0, 9.0, 3.0, 9.0, 9.0),
                (9.0, 5.0, 9.0, 1.0, 1.5),
                (9.0, 9.0, 9.0, 9.0, 9.0),
                (1.0, 9.0, 2.0, 9.0, 9.0),
            )
        )
        self.assertEqual(result.last_arrival_time, 20.0)
        self.assertAlmostEqual(result.drain_end_time, 29.0)
        self.assertAlmostEqual(
            result.drain_duration,
            result.drain_end_time - result.last_arrival_time,
        )
        self.assertGreaterEqual(result.drain_end_time, result.last_arrival_time)

    def test_input_trace_is_unchanged_by_simulation(self) -> None:
        trace = build_manual_trace((18.0,), (2.0,), (9.0,), (9.0,), (0.75,))
        reference = build_manual_trace((18.0,), (2.0,), (9.0,), (9.0,), (0.75,))
        run(trace)
        self.assertEqual(trace, reference)

    def test_attempts_flat_view_covers_every_attempt(self) -> None:
        result = run(
            build_manual_trace(
                (18.0, 18.5, 19.0),
                (8.0, 9.0, 3.0),
                (9.0, 0.5, 9.0),
                (9.0, 9.0, 9.0),
                (1.0, 9.0, 2.0),
            )
        )
        flat = [(a.token_id, a.attempt_id) for a in result.attempts]
        nested = [
            (a.token_id, a.attempt_id) for t in result.tokens for a in t.attempts
        ]
        self.assertEqual(flat, nested)
        self.assertEqual(flat, sorted(flat))


class EngineValidationTests(unittest.TestCase):
    def trace(self) -> WorkloadTrace:
        return build_manual_trace((1.0,), (1.0,), (1.0,), (1.0,), (1.0,))

    def test_replica_count_must_be_the_int_two(self) -> None:
        for bad in (3, 0, 2.0, True, "2", None):
            with self.assertRaises(ValueError, msg=f"replica_count={bad!r}"):
                simulate_common_state_no_hedge(
                    self.trace(), TIMELINE, SLOWDOWN, replica_count=bad
                )

    def test_slowdown_must_be_finite_and_at_least_one(self) -> None:
        for bad in (0.5, math.nan, math.inf, True, "2"):
            with self.assertRaises(ValueError, msg=f"slowdown={bad!r}"):
                simulate_common_state_no_hedge(self.trace(), TIMELINE, bad)

    def test_timeline_must_be_a_common_state_timeline(self) -> None:
        with self.assertRaises(ValueError):
            simulate_common_state_no_hedge(self.trace(), (10.0, 20.0, 30.0), SLOWDOWN)

    def test_missing_replay_streams_are_rejected(self) -> None:
        trace = self.trace()
        plain = WorkloadTrace(
            base_seed=trace.base_seed,
            arrival_rate=trace.arrival_rate,
            tokens=trace.tokens,
            service_times=trace.service_times,
        )
        with self.assertRaisesRegex(ValueError, "replay"):
            simulate_common_state_no_hedge(plain, TIMELINE, SLOWDOWN)

    def test_undersized_replay_streams_are_rejected(self) -> None:
        trace = self.trace()
        broken = WorkloadTrace(
            base_seed=trace.base_seed,
            arrival_rate=trace.arrival_rate,
            tokens=trace.tokens,
            service_times=trace.service_times,
            replay_service_times=((), (1.0,)),
        )
        with self.assertRaisesRegex(ValueError, "replay"):
            simulate_common_state_no_hedge(broken, TIMELINE, SLOWDOWN)

    def test_non_dense_duplicate_and_out_of_order_ids_are_rejected(self) -> None:
        def trace_with_ids(ids: tuple[int, ...]) -> WorkloadTrace:
            tokens = tuple(
                TokenSpec(token_id=i, arrival_time=float(p), token_class=TokenClass.REGULAR)
                for p, i in enumerate(ids)
            )
            width = max(max(ids) + 1, len(ids), 1)
            stream = tuple(1.0 for _ in range(width))
            return WorkloadTrace(
                base_seed=1,
                arrival_rate=1.0,
                tokens=tokens,
                service_times=(stream, stream),
                replay_service_times=(stream, stream),
            )

        for ids in ((-1, 0), (0, 0), (0, 2), (1, 0)):
            with self.assertRaisesRegex(ValueError, "Token ids"):
                simulate_common_state_no_hedge(trace_with_ids(ids), TIMELINE, SLOWDOWN)


if __name__ == "__main__":
    unittest.main()
