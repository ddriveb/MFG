"""Hedge-capable engine tests (paired-comparison spec section 18, scenarios 1-15)."""

from dataclasses import replace
import math
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.common_state_simulation import simulate_common_state_no_hedge
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import (
    HedgeAttemptStatus,
    simulate_hedge_common_state,
)
from mfg_hedge.workload import (
    TokenSpec,
    WorkloadTrace,
    generate_workload_with_hedge,
    generate_workload_with_replay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMELINE = CommonStateTimeline(
    degraded_start=10.0, failed_start=20.0, recovered_start=30.0
)
SLOWDOWN = 2.0
N = ProtectionAction.NORMAL
D = ProtectionAction.DELAYED_HEDGE
I = ProtectionAction.IMMEDIATE_HEDGE


def build_trace(arrivals, service_r0, service_r1, replay_r1, hedge_r0, hedge_r1):
    count = len(arrivals)
    filler = (9.0,) * count
    tokens = tuple(
        TokenSpec(token_id=i, arrival_time=a, token_class=TokenClass.REGULAR)
        for i, a in enumerate(arrivals)
    )
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.4,
        tokens=tokens,
        service_times=(service_r0, service_r1),
        replay_service_times=(filler, replay_r1),
        hedge_service_times=(hedge_r0, hedge_r1),
    )


def run(trace, actions=None, hedge_delay=None, timeline=TIMELINE):
    return simulate_hedge_common_state(
        trace,
        timeline=timeline,
        degraded_slowdown=SLOWDOWN,
        actions=actions,
        hedge_delay=hedge_delay,
    )


class Scenario01ImmediateHedgeTests(unittest.TestCase):
    def test_immediate_enqueues_both_copies_at_arrival(self) -> None:
        trace = build_trace((5.0,), (2.0,), (9.0,), (9.0,), (9.0,), (3.0,))
        result = run(trace, actions={0: I})
        token = result.tokens[0]
        self.assertEqual([a.attempt_id for a in token.attempts], [0, 2])
        primary, hedge = token.attempts
        self.assertEqual(primary.replica_id, 0)
        self.assertEqual(hedge.replica_id, 1)
        self.assertEqual(primary.enqueue_time, 5.0)
        self.assertEqual(hedge.enqueue_time, 5.0)
        self.assertEqual(primary.status, HedgeAttemptStatus.COMPLETED_WINNER)
        self.assertEqual(hedge.status, HedgeAttemptStatus.COMPLETED_LOSER)
        self.assertAlmostEqual(hedge.terminal_time, 8.0)
        self.assertAlmostEqual(hedge.executed_work, 3.0)
        self.assertAlmostEqual(token.completion_time, 7.0)
        self.assertEqual(result.hedge_launches, 1)
        self.assertEqual(result.hedge_requested, 1)


class Scenario02TimerSuppressedByCompletionTests(unittest.TestCase):
    def test_timer_void_when_primary_completes_first(self) -> None:
        trace = build_trace((1.0,), (2.0,), (9.0,), (9.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: D}, hedge_delay=5.0)
        self.assertAlmostEqual(result.tokens[0].completion_time, 3.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_timers_voided, 1)


class Scenario03TimerFiresTests(unittest.TestCase):
    def test_timer_fires_and_hedge_wins(self) -> None:
        # The Hedge wins at 4.5; the Primary is left running (conservative
        # cancellation), crosses into D, and is killed by F at 20.0 with its
        # consumed work counted: (10-1)*1.0 + (20-10)*0.5 = 14.0.
        trace = build_trace((1.0,), (100.0,), (9.0,), (9.0,), (9.0,), (1.5,))
        result = run(trace, actions={0: D}, hedge_delay=2.0)
        token = result.tokens[0]
        primary, hedge = token.attempts
        self.assertAlmostEqual(hedge.start_time, 3.0)
        self.assertEqual(hedge.status, HedgeAttemptStatus.COMPLETED_WINNER)
        self.assertAlmostEqual(token.completion_time, 4.5)
        self.assertEqual(token.winner_attempt_id, 2)
        self.assertEqual(primary.status, HedgeAttemptStatus.FAILED_RUNNING)
        self.assertAlmostEqual(primary.terminal_time, 20.0)
        self.assertAlmostEqual(primary.executed_work, 14.0)
        self.assertEqual(token.replay_count, 0)
        self.assertEqual(result.hedge_launches, 1)
        self.assertAlmostEqual(result.drain_end_time, 20.0)


class Scenario04TimerVersusCompletionTests(unittest.TestCase):
    def test_completion_at_the_same_instant_wins_over_timer(self) -> None:
        trace = build_trace((1.0,), (3.0,), (9.0,), (9.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: D}, hedge_delay=3.0)
        self.assertAlmostEqual(result.tokens[0].completion_time, 4.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_timers_voided, 1)
        self.assertEqual(result.stale_completion_events_ignored, 0)


class Scenario05TimerVersusFailureTests(unittest.TestCase):
    def test_5a_primary_on_b_backup_domain_failed_suppresses(self) -> None:
        # tok0 runs on R0 and dies at F; tok1 (Primary on B) has a timer at 20.0.
        trace = build_trace(
            (17.0, 18.0),
            (100.0, 9.0),
            (9.0, 100.0),
            (2.0, 9.0),
            (9.0, 9.0),
            (9.0, 9.0),
        )
        result = run(trace, actions={1: D}, hedge_delay=2.0)
        tok0, tok1 = result.tokens
        self.assertEqual(tok0.attempts[0].status, HedgeAttemptStatus.FAILED_RUNNING)
        self.assertAlmostEqual(tok0.attempts[0].executed_work, 1.5)
        self.assertEqual(tok0.attempts[1].attempt_id, 1)
        self.assertAlmostEqual(tok0.attempts[1].start_time, 118.0)
        self.assertAlmostEqual(tok0.completion_time, 120.0)
        self.assertEqual(len(tok1.attempts), 1)
        self.assertAlmostEqual(tok1.completion_time, 118.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_suppressed, 1)

    def test_5b_primary_on_a_failure_creates_replay_and_voids_timer(self) -> None:
        trace = build_trace((18.0,), (100.0,), (9.0,), (2.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: D}, hedge_delay=2.0)
        token = result.tokens[0]
        self.assertEqual([a.attempt_id for a in token.attempts], [0, 1])
        self.assertEqual(token.attempts[0].status, HedgeAttemptStatus.FAILED_RUNNING)
        self.assertEqual(token.attempts[1].status, HedgeAttemptStatus.COMPLETED_WINNER)
        self.assertAlmostEqual(token.completion_time, 22.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_timers_voided, 1)


class Scenario06QueuedLoserCancelledTests(unittest.TestCase):
    def test_queued_loser_is_cancelled_with_zero_work(self) -> None:
        trace = build_trace(
            (4.8, 4.9, 5.0),
            (20.0, 9.0, 2.0),
            (9.0, 10.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 0.5),
        )
        result = run(trace, actions={2: I})
        token = result.tokens[2]
        primary, hedge = token.attempts
        self.assertAlmostEqual(hedge.start_time, 14.9)
        self.assertEqual(hedge.status, HedgeAttemptStatus.COMPLETED_WINNER)
        self.assertAlmostEqual(token.completion_time, 15.4)
        self.assertEqual(primary.status, HedgeAttemptStatus.CANCELLED_QUEUED)
        self.assertIsNone(primary.start_time)
        self.assertAlmostEqual(primary.executed_work, 0.0)
        self.assertAlmostEqual(primary.terminal_time, 15.4)
        self.assertEqual(result.cancelled_queued_total, 1)


class Scenario07RunningLoserTests(unittest.TestCase):
    def test_running_loser_completes_and_its_work_is_counted(self) -> None:
        trace = build_trace((5.0,), (2.0,), (9.0,), (9.0,), (9.0,), (3.0,))
        result = run(trace, actions={0: I})
        token = result.tokens[0]
        self.assertAlmostEqual(token.completion_time, 7.0)
        self.assertAlmostEqual(token.latency, 2.0)
        self.assertEqual(result.completed_loser_total, 1)
        self.assertAlmostEqual(result.drain_end_time, 8.0)


class Scenario08PrimaryLostBackupAliveTests(unittest.TestCase):
    def test_primary_lost_with_hedge_alive_creates_no_replay(self) -> None:
        trace = build_trace((12.0,), (100.0,), (9.0,), (9.0,), (9.0,), (100.0,))
        result = run(trace, actions={0: I})
        token = result.tokens[0]
        self.assertEqual(token.attempts[0].status, HedgeAttemptStatus.FAILED_RUNNING)
        self.assertAlmostEqual(token.attempts[0].executed_work, 4.0)
        self.assertEqual(token.attempts[1].status, HedgeAttemptStatus.COMPLETED_WINNER)
        self.assertAlmostEqual(token.completion_time, 112.0)
        self.assertEqual(result.replay_executions, 0)
        self.assertEqual(token.replay_count, 0)


class Scenario09PrimaryLostBeforeTimerTests(unittest.TestCase):
    def test_replay_only_no_late_hedge(self) -> None:
        trace = build_trace((18.0,), (100.0,), (9.0,), (2.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: D}, hedge_delay=5.0)
        token = result.tokens[0]
        self.assertEqual([a.attempt_id for a in token.attempts], [0, 1])
        self.assertAlmostEqual(token.completion_time, 22.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_timers_voided, 1)


class Scenario10BackupDomainFailedTests(unittest.TestCase):
    def test_immediate_hedge_during_failure_is_suppressed(self) -> None:
        trace = build_trace((25.0,), (9.0,), (1.0,), (9.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: I})
        token = result.tokens[0]
        self.assertEqual(token.primary_replica, 1)
        self.assertEqual(len(token.attempts), 1)
        self.assertAlmostEqual(token.completion_time, 26.0)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_suppressed, 1)
        self.assertEqual(result.hedge_requested, 1)


class Scenario111213InvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_hedge(config, 300)
        actions = {}
        for spec in trace.tokens:
            if spec.token_id % 7 == 0:
                actions[spec.token_id] = I
            elif spec.token_id % 11 == 0:
                actions[spec.token_id] = D
        self.result = simulate_hedge_common_state(
            trace,
            timeline=CommonStateTimeline(100.0, 200.0, 220.0),
            degraded_slowdown=config.degraded_slowdown,
            actions=actions,
            hedge_delay=1.5,
        )

    def test_exactly_one_winner_and_completion_per_token(self) -> None:
        for token in self.result.tokens:
            winners = [
                a
                for a in token.attempts
                if a.status is HedgeAttemptStatus.COMPLETED_WINNER
            ]
            self.assertEqual(len(winners), 1, msg=f"token {token.token_id}")
            self.assertLessEqual(token.replay_count, 1)
            self.assertLessEqual(
                sum(1 for a in token.attempts if a.attempt_id == 2), 1
            )

    def test_counter_algebra_is_consistent(self) -> None:
        result = self.result
        attempt2 = [a for a in result.attempts if a.attempt_id == 2]
        attempt1 = [a for a in result.attempts if a.attempt_id == 1]
        self.assertEqual(result.hedge_launches, len(attempt2))
        self.assertEqual(result.replay_executions, len(attempt1))
        requested = sum(1 for t in result.tokens if t.action is not N)
        self.assertEqual(result.hedge_requested, requested)
        self.assertEqual(
            requested,
            result.hedge_launches
            + result.hedge_suppressed
            + result.hedge_timers_voided,
        )
        total_work = math.fsum(a.executed_work for a in result.attempts)
        winner_work = math.fsum(
            a.executed_work
            for a in result.attempts
            if a.status is HedgeAttemptStatus.COMPLETED_WINNER
        )
        self.assertGreaterEqual(total_work, winner_work)


class Scenario13StaleEventTests(unittest.TestCase):
    def test_stale_completion_ignored_exactly_once(self) -> None:
        trace = build_trace((9.0,), (2.0,), (9.0,), (9.0,), (9.0,), (9.0,))
        result = run(trace)
        self.assertEqual(result.stale_completion_events_ignored, 1)
        self.assertAlmostEqual(result.tokens[0].completion_time, 12.0)

    def test_timer_voided_exactly_once(self) -> None:
        trace = build_trace((1.0,), (3.0,), (9.0,), (9.0,), (9.0,), (9.0,))
        result = run(trace, actions={0: D}, hedge_delay=3.0)
        self.assertEqual(result.hedge_timers_voided, 1)


class Scenario14Attempt2CrnTests(unittest.TestCase):
    def test_attempt2_stream_does_not_perturb_earlier_streams(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        replay_trace = generate_workload_with_replay(config, 100)
        hedge_trace = generate_workload_with_hedge(config, 100)
        self.assertEqual(replay_trace.tokens, hedge_trace.tokens)
        self.assertEqual(replay_trace.service_times, hedge_trace.service_times)
        self.assertEqual(
            replay_trace.replay_service_times, hedge_trace.replay_service_times
        )
        self.assertIsNotNone(hedge_trace.hedge_service_times)
        self.assertNotEqual(
            hedge_trace.hedge_service_times[0], hedge_trace.service_times[0]
        )

    def test_no_hedge_mode_never_reads_attempt2(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_hedge(config, 200)
        corrupted = replace(
            trace,
            hedge_service_times=tuple(
                tuple(math.nan for _ in stream)
                for stream in trace.hedge_service_times
            ),
        )
        timeline = CommonStateTimeline(100.0, 200.0, 220.0)
        clean = simulate_hedge_common_state(
            trace, timeline=timeline, degraded_slowdown=2.0
        )
        dirty = simulate_hedge_common_state(
            corrupted, timeline=timeline, degraded_slowdown=2.0
        )
        self.assertEqual(clean, dirty)


class Scenario15NoHedgeRegressionTests(unittest.TestCase):
    def test_all_normal_matches_common_state_engine_token_for_token(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_hedge(config, 300)
        timeline = CommonStateTimeline(100.0, 200.0, 220.0)
        reference = simulate_common_state_no_hedge(
            trace, timeline=timeline, degraded_slowdown=config.degraded_slowdown
        )
        result = simulate_hedge_common_state(
            trace, timeline=timeline, degraded_slowdown=config.degraded_slowdown
        )
        self.assertEqual(len(result.tokens), len(reference.tokens))
        for token, ref in zip(result.tokens, reference.tokens):
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
            reference_projection = (
                ref.token_id,
                ref.token_class,
                ref.arrival_time,
                ref.primary_replica,
                ref.attempts[0].start_time,
                ref.attempts[0].required_work,
                ref.completion_time,
                ref.attempts[0].queue_delay,
                ref.latency,
            )
            self.assertEqual(projection, reference_projection, msg=f"token {token.token_id}")
        self.assertEqual(result.hedge_launches, 0)


class EngineValidationTests(unittest.TestCase):
    def simple_trace(self) -> WorkloadTrace:
        return build_trace((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,))

    def test_hedge_stream_required_for_hedge_actions(self) -> None:
        trace = replace(self.simple_trace(), hedge_service_times=None)
        with self.assertRaisesRegex(ValueError, "hedge"):
            run(trace, actions={0: I})
        with self.assertRaisesRegex(ValueError, "hedge"):
            run(trace, actions={0: D}, hedge_delay=1.0)

    def test_hedge_stream_shape_is_validated(self) -> None:
        trace = replace(
            self.simple_trace(), hedge_service_times=((1.0,),)
        )
        with self.assertRaisesRegex(ValueError, "hedge"):
            run(trace, actions={0: I})

    def test_illegal_actions_and_tau0_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run(self.simple_trace(), actions={0: "hedge"})
        with self.assertRaises(ValueError):
            run(self.simple_trace(), actions={7: N})
        for bad_tau in (0.0, -1.0, math.nan, math.inf, True, None):
            with self.assertRaises(ValueError, msg=f"tau0={bad_tau!r}"):
                run(self.simple_trace(), actions={0: D}, hedge_delay=bad_tau)

    def test_illegal_replica_count_and_trace_ids(self) -> None:
        for bad in (3, 2.0, True, None):
            with self.assertRaises(ValueError):
                simulate_hedge_common_state(
                    self.simple_trace(), TIMELINE, SLOWDOWN, replica_count=bad
                )
        tokens = (
            TokenSpec(token_id=1, arrival_time=0.5, token_class=TokenClass.REGULAR),
            TokenSpec(token_id=0, arrival_time=0.6, token_class=TokenClass.REGULAR),
        )
        stream = (1.0, 1.0)
        bad_trace = WorkloadTrace(
            base_seed=1,
            arrival_rate=1.0,
            tokens=tokens,
            service_times=(stream, stream),
            replay_service_times=(stream, stream),
            hedge_service_times=(stream, stream),
        )
        with self.assertRaisesRegex(ValueError, "Token ids"):
            run(bad_trace)

    def test_determinism_and_input_immutability(self) -> None:
        trace = build_trace(
            (4.8, 4.9, 5.0),
            (20.0, 9.0, 2.0),
            (9.0, 10.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 0.5),
        )
        reference = build_trace(
            (4.8, 4.9, 5.0),
            (20.0, 9.0, 2.0),
            (9.0, 10.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 0.5),
        )
        first = run(trace, actions={2: I})
        second = run(trace, actions={2: I})
        self.assertEqual(first, second)
        self.assertEqual(trace, reference)


class SnapshotTombstoneTests(unittest.TestCase):
    """Cancelled queued attempts must not count in queue snapshots."""

    def trace_with_replica_one_tombstone(self) -> WorkloadTrace:
        # tok2's Primary wins on R0 at 7.0; its Hedge sits cancelled behind
        # tok1's long R1 run (2.0-102.0) as a tombstone in R1's deque.
        return build_trace(
            (1.0, 2.0, 3.0),
            (5.0, 9.0, 1.0),
            (9.0, 100.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 7.0),
        )

    def trace_with_replica_zero_tombstone(self) -> WorkloadTrace:
        # tok2's Hedge wins on R1 at 6.0; its Primary sits cancelled behind
        # tok0's long R0 run as a tombstone in R0's deque.
        return build_trace(
            (1.0, 2.0, 3.0),
            (100.0, 9.0, 2.0),
            (9.0, 3.0, 9.0),
            (2.0, 9.0, 9.0),
            (9.0, 9.0, 9.0),
            (9.0, 9.0, 1.0),
        )

    def test_failed_start_snapshot_excludes_replica_one_tombstone(self) -> None:
        result = run(self.trace_with_replica_one_tombstone(), actions={2: I})
        hedge = result.tokens[2].attempts[1]
        self.assertEqual(hedge.status, HedgeAttemptStatus.CANCELLED_QUEUED)
        self.assertEqual(result.queue_length_at_failed_start, (0, 0))

    def test_recovered_start_snapshot_excludes_replica_one_tombstone(self) -> None:
        result = run(self.trace_with_replica_one_tombstone(), actions={2: I})
        self.assertEqual(result.queue_length_at_recovered_start, (0, 0))

    def test_failed_start_snapshot_excludes_replica_zero_tombstone(self) -> None:
        result = run(self.trace_with_replica_zero_tombstone(), actions={2: I})
        primary = result.tokens[2].attempts[0]
        self.assertEqual(primary.status, HedgeAttemptStatus.CANCELLED_QUEUED)
        # R0 counts only live entries (0, not the tombstone); R1 holds tok0's
        # freshly enqueued Replay (the snapshot is taken after Replay enqueue).
        self.assertEqual(result.queue_length_at_failed_start, (0, 1))

    def test_recovered_start_snapshot_excludes_replica_zero_tombstone(self) -> None:
        result = run(self.trace_with_replica_zero_tombstone(), actions={2: I})
        self.assertEqual(result.queue_length_at_recovered_start, (0, 0))


class ServiceValueValidationTests(unittest.TestCase):
    def trace_with_hedge_value(self, value) -> WorkloadTrace:
        return build_trace((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (value,))

    def test_illegal_attempt2_values_rejected_when_hedging(self) -> None:
        for bad in (math.nan, math.inf, -math.inf, 0.0, -1.0, True):
            trace = self.trace_with_hedge_value(bad)
            with self.assertRaisesRegex(ValueError, "attempt_id=2", msg=f"value={bad!r}"):
                run(trace, actions={0: I})

    def test_nan_attempt2_is_still_ignored_in_all_normal_mode(self) -> None:
        trace = self.trace_with_hedge_value(math.nan)
        result = run(trace)
        self.assertEqual(result.hedge_launches, 0)
        self.assertTrue(math.isfinite(result.tokens[0].completion_time))

    def test_illegal_attempt0_and_attempt1_values_rejected(self) -> None:
        for attempt, bad in (("service", math.nan), ("replay", 0.0), ("replay", True)):
            trace = build_trace((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,))
            if attempt == "service":
                trace = replace(
                    trace, service_times=((bad,), trace.service_times[1])
                )
            else:
                trace = replace(
                    trace, replay_service_times=((bad,), trace.replay_service_times[1])
                )
            with self.assertRaisesRegex(ValueError, "attempt_id=", msg=f"{attempt}={bad!r}"):
                run(trace)


class TraceBoundaryValidationTests(unittest.TestCase):
    def trace_with_arrivals(self, arrivals) -> WorkloadTrace:
        return build_trace(
            arrivals,
            *((1.0,) * len(arrivals) for _ in range(5)),
        )

    def test_negative_nan_and_decreasing_arrivals_rejected(self) -> None:
        for bad_arrivals in ((-1.0,), (math.nan,), (3.0, 2.0), (True,)):
            trace = self.trace_with_arrivals(bad_arrivals)
            with self.assertRaises(ValueError, msg=f"arrivals={bad_arrivals!r}"):
                run(trace)

    def test_same_time_arrivals_are_allowed(self) -> None:
        result = run(self.trace_with_arrivals((2.0, 2.0)))
        self.assertEqual(result.completed_tokens, 2)

    def test_illegal_token_class_rejected(self) -> None:
        trace = build_trace((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,))
        bad_tokens = (TokenSpec(token_id=0, arrival_time=1.0, token_class="R"),)
        trace = replace(trace, tokens=bad_tokens)
        with self.assertRaises(ValueError):
            run(trace)


if __name__ == "__main__":
    unittest.main()
