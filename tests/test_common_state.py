import math
import unittest

from mfg_hedge.common_state import (
    CommonStateTimeline,
    ExecutionOutcome,
    Phase,
    completion_after,
    phase_at,
    speed_at,
    state_at,
    work_executed_between,
)
from mfg_hedge.domain import CommonState, ProtectionAction


TIMELINE = CommonStateTimeline(
    degraded_start=10.0, failed_start=20.0, recovered_start=30.0
)
SLOWDOWN = 2.0


class CommonStateTimelineTests(unittest.TestCase):
    def test_valid_timeline_stores_floats(self) -> None:
        timeline = CommonStateTimeline(
            degraded_start=10, failed_start=20, recovered_start=30
        )
        self.assertEqual(timeline.degraded_start, 10.0)
        self.assertIsInstance(timeline.failed_start, float)

    def test_rejects_misordered_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            CommonStateTimeline(degraded_start=-1.0, failed_start=20.0, recovered_start=30.0)
        with self.assertRaises(ValueError):
            CommonStateTimeline(degraded_start=20.0, failed_start=20.0, recovered_start=30.0)
        with self.assertRaises(ValueError):
            CommonStateTimeline(degraded_start=10.0, failed_start=30.0, recovered_start=20.0)

    def test_rejects_non_finite_and_non_numeric_boundaries(self) -> None:
        for bad in (math.nan, math.inf, -math.inf, True, "10"):
            with self.assertRaises(ValueError, msg=f"boundary={bad!r}"):
                CommonStateTimeline(
                    degraded_start=bad, failed_start=20.0, recovered_start=30.0
                )


class StateAndPhaseTests(unittest.TestCase):
    def test_state_at_boundaries_belong_to_the_interval_they_open(self) -> None:
        cases = [
            (0.0, CommonState.HEALTHY),
            (9.999, CommonState.HEALTHY),
            (10.0, CommonState.DEGRADED),
            (19.999, CommonState.DEGRADED),
            (20.0, CommonState.FAILED),
            (29.999, CommonState.FAILED),
            (30.0, CommonState.HEALTHY),
            (100.0, CommonState.HEALTHY),
        ]
        for moment, expected in cases:
            self.assertEqual(state_at(TIMELINE, moment), expected, msg=f"t={moment}")

    def test_phase_at_distinguishes_recovered_from_initial_healthy(self) -> None:
        cases = [
            (0.0, Phase.HEALTHY),
            (10.0, Phase.DEGRADED),
            (20.0, Phase.FAILED),
            (29.999, Phase.FAILED),
            (30.0, Phase.RECOVERED),
            (100.0, Phase.RECOVERED),
        ]
        for moment, expected in cases:
            self.assertEqual(phase_at(TIMELINE, moment), expected, msg=f"t={moment}")
        self.assertEqual(state_at(TIMELINE, 30.0), CommonState.HEALTHY)
        self.assertEqual(phase_at(TIMELINE, 30.0), Phase.RECOVERED)
        self.assertNotEqual(phase_at(TIMELINE, 5.0), phase_at(TIMELINE, 35.0))

    def test_negative_and_non_finite_times_are_rejected(self) -> None:
        for bad in (-0.1, math.nan, math.inf, True):
            with self.assertRaises(ValueError, msg=f"t={bad!r}"):
                state_at(TIMELINE, bad)
            with self.assertRaises(ValueError, msg=f"t={bad!r}"):
                phase_at(TIMELINE, bad)


class SpeedAtTests(unittest.TestCase):
    def test_replica_zero_follows_the_state(self) -> None:
        self.assertEqual(speed_at(0, CommonState.HEALTHY, SLOWDOWN), 1.0)
        self.assertEqual(speed_at(0, CommonState.DEGRADED, SLOWDOWN), 0.5)
        self.assertEqual(speed_at(0, CommonState.FAILED, SLOWDOWN), 0.0)

    def test_replica_one_is_always_full_speed(self) -> None:
        for state in CommonState:
            self.assertEqual(speed_at(1, state, SLOWDOWN), 1.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        for bad_replica in (2, -1, 0.5, True):
            with self.assertRaises(ValueError, msg=f"replica={bad_replica!r}"):
                speed_at(bad_replica, CommonState.HEALTHY, SLOWDOWN)
        for bad_state in ("H", ProtectionAction.NORMAL, None, True):
            with self.assertRaises(ValueError, msg=f"state={bad_state!r}"):
                speed_at(0, bad_state, SLOWDOWN)
        for bad_slowdown in (0.5, 0.0, math.nan, math.inf, True, "2"):
            with self.assertRaises(ValueError, msg=f"slowdown={bad_slowdown!r}"):
                speed_at(0, CommonState.DEGRADED, bad_slowdown)


class WorkExecutedBetweenTests(unittest.TestCase):
    def test_single_state_segment(self) -> None:
        self.assertAlmostEqual(
            work_executed_between(0, 2.0, 8.0, TIMELINE, SLOWDOWN), 6.0
        )

    def test_piecewise_integration_across_healthy_degraded(self) -> None:
        # [8,10) at speed 1.0 -> 2.0; [10,12) at speed 0.5 -> 1.0
        self.assertAlmostEqual(
            work_executed_between(0, 8.0, 12.0, TIMELINE, SLOWDOWN), 3.0
        )

    def test_failed_segment_contributes_zero_on_replica_zero(self) -> None:
        # [15,20) at 0.5 -> 2.5; [20,25) at 0.0 -> 0.0
        self.assertAlmostEqual(
            work_executed_between(0, 15.0, 25.0, TIMELINE, SLOWDOWN), 2.5
        )

    def test_full_span_replica_zero(self) -> None:
        # 5*1.0 + 10*0.5 + 10*0.0 + 5*1.0
        self.assertAlmostEqual(
            work_executed_between(0, 5.0, 35.0, TIMELINE, SLOWDOWN), 15.0
        )

    def test_replica_one_integrates_at_full_speed_across_all_boundaries(self) -> None:
        self.assertAlmostEqual(
            work_executed_between(1, 18.0, 35.0, TIMELINE, SLOWDOWN), 17.0
        )

    def test_zero_length_interval_is_zero(self) -> None:
        self.assertEqual(work_executed_between(0, 15.0, 15.0, TIMELINE, SLOWDOWN), 0.0)

    def test_invalid_intervals_are_rejected(self) -> None:
        for t0, t1 in ((12.0, 8.0), (-1.0, 5.0), (0.0, math.nan), (math.inf, 20.0)):
            with self.assertRaises(ValueError, msg=f"[{t0}, {t1})"):
                work_executed_between(0, t0, t1, TIMELINE, SLOWDOWN)


class CompletionAfterTests(unittest.TestCase):
    def test_spec_example_crossing_into_degraded(self) -> None:
        outcome = completion_after(0, 9.0, 2.0, TIMELINE, SLOWDOWN)
        self.assertIsInstance(outcome, ExecutionOutcome)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 12.0)
        self.assertAlmostEqual(outcome.executed_work, 2.0)
        self.assertAlmostEqual(outcome.remaining_work, 0.0)

    def test_completion_before_failure_boundary(self) -> None:
        outcome = completion_after(0, 8.0, 1.0, TIMELINE, SLOWDOWN)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 9.0)

    def test_healthy_then_degraded_then_failed_reports_failure(self) -> None:
        # banks 2.0 in [8,10), 5.0 in [10,20), then F kills it at t=20
        outcome = completion_after(0, 8.0, 10.0, TIMELINE, SLOWDOWN)
        self.assertFalse(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 20.0)
        self.assertAlmostEqual(outcome.executed_work, 7.0)
        self.assertAlmostEqual(outcome.remaining_work, 3.0)

    def test_degraded_then_failed_reports_failure(self) -> None:
        outcome = completion_after(0, 15.0, 4.0, TIMELINE, SLOWDOWN)
        self.assertFalse(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 20.0)
        self.assertAlmostEqual(outcome.executed_work, 2.5)
        self.assertAlmostEqual(outcome.remaining_work, 1.5)

    def test_starting_during_failed_fails_immediately(self) -> None:
        outcome = completion_after(0, 25.0, 1.0, TIMELINE, SLOWDOWN)
        self.assertFalse(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 25.0)
        self.assertAlmostEqual(outcome.executed_work, 0.0)
        self.assertAlmostEqual(outcome.remaining_work, 1.0)

    def test_starting_after_recovery_completes_normally(self) -> None:
        outcome = completion_after(0, 35.0, 2.0, TIMELINE, SLOWDOWN)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 37.0)

    def test_completion_exactly_at_failure_boundary_is_reported_completed(self) -> None:
        # The pure function reports the physical completion instant; the event
        # engine's fault-first ordering decides that this execution fails.
        outcome = completion_after(0, 19.0, 0.5, TIMELINE, SLOWDOWN)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 20.0)

    def test_failed_execution_does_not_resume_after_recovery(self) -> None:
        outcome = completion_after(0, 8.0, 10.0, TIMELINE, SLOWDOWN)
        self.assertFalse(outcome.completed)
        self.assertLess(outcome.terminal_time, 30.0)

    def test_replica_one_ignores_domain_a_timeline(self) -> None:
        outcome = completion_after(1, 15.0, 10.0, TIMELINE, SLOWDOWN)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 25.0)
        self.assertAlmostEqual(outcome.executed_work, 10.0)

    def test_invalid_work_and_start_are_rejected(self) -> None:
        for bad_work in (0.0, -1.0, math.nan, math.inf, True):
            with self.assertRaises(ValueError, msg=f"work={bad_work!r}"):
                completion_after(0, 0.0, bad_work, TIMELINE, SLOWDOWN)
        for bad_start in (-0.1, math.nan, math.inf, True):
            with self.assertRaises(ValueError, msg=f"start={bad_start!r}"):
                completion_after(0, bad_start, 1.0, TIMELINE, SLOWDOWN)


class ReplicaIdValidationTests(unittest.TestCase):
    def test_float_zero_and_one_are_rejected_everywhere(self) -> None:
        for bad in (0.0, 1.0):
            with self.assertRaises(ValueError, msg=f"speed_at replica={bad!r}"):
                speed_at(bad, CommonState.HEALTHY, SLOWDOWN)
            with self.assertRaises(ValueError, msg=f"work_executed_between replica={bad!r}"):
                work_executed_between(bad, 0.0, 1.0, TIMELINE, SLOWDOWN)
            with self.assertRaises(ValueError, msg=f"completion_after replica={bad!r}"):
                completion_after(bad, 0.0, 1.0, TIMELINE, SLOWDOWN)

    def test_bool_str_none_and_out_of_range_are_rejected_everywhere(self) -> None:
        for bad in (True, False, "0", None, 2, -1):
            with self.assertRaises(ValueError, msg=f"speed_at replica={bad!r}"):
                speed_at(bad, CommonState.HEALTHY, SLOWDOWN)
            with self.assertRaises(ValueError, msg=f"work_executed_between replica={bad!r}"):
                work_executed_between(bad, 0.0, 1.0, TIMELINE, SLOWDOWN)
            with self.assertRaises(ValueError, msg=f"completion_after replica={bad!r}"):
                completion_after(bad, 0.0, 1.0, TIMELINE, SLOWDOWN)

    def test_legitimate_int_results_are_unchanged(self) -> None:
        self.assertEqual(speed_at(0, CommonState.DEGRADED, SLOWDOWN), 0.5)
        self.assertEqual(speed_at(1, CommonState.FAILED, SLOWDOWN), 1.0)
        self.assertAlmostEqual(
            work_executed_between(0, 8.0, 12.0, TIMELINE, SLOWDOWN), 3.0
        )
        outcome = completion_after(0, 9.0, 2.0, TIMELINE, SLOWDOWN)
        self.assertTrue(outcome.completed)
        self.assertAlmostEqual(outcome.terminal_time, 12.0)


if __name__ == "__main__":
    unittest.main()
