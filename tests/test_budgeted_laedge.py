from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import unittest

from mfg_hedge.attribution_episode import (
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import TokenClass
from mfg_hedge.hedge_simulation import HedgeAttemptStatus
from mfg_hedge.attribution_metrics import build_episode_metrics
from mfg_hedge.workload import TokenSpec, WorkloadTrace
from mfg_hedge.budgeted_laedge import (
    BUDGETED_LAEDGE_ARM_KEYS,
    BUDGETED_LAEDGE_DELTA_POINTS,
    BudgetWindowAudit,
    simulate_budgeted_laedge_episode,
)


def make_episode(
    arrivals=(0.0,),
    primary=((100.0,), (100.0,)),
    replay=((10.0,), (10.0,)),
    hedge=((10.0,), (10.0,)),
    timeline=None,
):
    tokens = tuple(
        TokenSpec(index, float(arrival), TokenClass.REGULAR)
        for index, arrival in enumerate(arrivals)
    )
    trace = WorkloadTrace(
        base_seed=20260909,
        arrival_rate=1.4,
        tokens=tokens,
        service_times=tuple(tuple(float(value) for value in row) for row in primary),
        replay_service_times=tuple(tuple(float(value) for value in row) for row in replay),
        hedge_service_times=tuple(tuple(float(value) for value in row) for row in hedge),
    )
    return EpisodeTrace(
        EpisodeIdentity("budgeted-laedge:test", 20260909, len(tokens)),
        20260909,
        EpisodeProtocol(
            timeline or CommonStateTimeline(1000.0, 2000.0, 2200.0),
            3000.0,
        ),
        trace,
    )


class BudgetedLaedgeContractTests(unittest.TestCase):
    def test_later_protocol_has_exactly_eleven_unique_arms_and_one_delta_zero(self):
        self.assertEqual(
            BUDGETED_LAEDGE_DELTA_POINTS,
            (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18),
        )
        self.assertEqual(len(BUDGETED_LAEDGE_ARM_KEYS), 11)
        self.assertEqual(len(set(BUDGETED_LAEDGE_ARM_KEYS)), 11)
        self.assertEqual(
            sum(key.startswith("budgeted-laedge:") for key in BUDGETED_LAEDGE_ARM_KEYS),
            7,
        )
        self.assertIn("budgeted-laedge:delta=0%", BUDGETED_LAEDGE_ARM_KEYS)

    def test_delta_zero_is_dynamic_no_hedge_and_does_not_duplicate_a_fixed_arm(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(primary=((100.0,), (100.0,)), hedge=((0.1,), (0.1,))),
            planned_budget_rate=0.0,
        )
        self.assertEqual(result.simulation.simulation.hedge_launches, 0)
        self.assertEqual(result.simulation.simulation.hedge_suppressed, 1)
        self.assertEqual(
            result.simulation.simulation.tokens[0].attempts,
            (result.simulation.simulation.tokens[0].attempts[0],),
        )
        self.assertEqual(result.arm_key, "budgeted-laedge:delta=0%")

    def test_nonrefundable_charge_is_not_released_after_hedge_completion(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(
                arrivals=(0.0, 1.0),
                primary=((0.2, 0.2), (100.0, 100.0)),
                replay=((10.0, 10.0), (10.0, 10.0)),
                hedge=((0.1, 0.1), (0.1, 0.1)),
            ),
            planned_budget_rate=0.04,
        )
        first_window = result.windows[0]
        self.assertAlmostEqual(first_window.planned_budget, 1.0)
        self.assertAlmostEqual(first_window.nonrefundable_charged_work, 1.0)
        self.assertEqual(first_window.hedge_launches, 1)
        self.assertGreaterEqual(first_window.hedge_suppressed, 1)
        self.assertEqual(result.simulation.simulation.hedge_launches, 1)

    def test_cross_window_work_has_launch_and_execution_time_views(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(
                primary=((100.0,), (100.0,)),
                hedge=((100.0,), (50.0,)),
            ),
            planned_budget_rate=0.04,
        )
        first, second = result.windows[0], result.windows[1]
        self.assertEqual((first.start, first.end), (0.0, 25.0))
        self.assertEqual((second.start, second.end), (25.0, 50.0))
        self.assertAlmostEqual(first.launch_cohort_realized_work, 50.0)
        self.assertAlmostEqual(first.execution_time_realized_work, 25.0)
        self.assertAlmostEqual(second.execution_time_realized_work, 25.0)
        self.assertAlmostEqual(second.launch_cohort_realized_work, 0.0)
        self.assertAlmostEqual(first.overshoot, 49.0)
        self.assertAlmostEqual(second.unused_budget, 1.0)
        self.assertAlmostEqual(
            first.signed_realized_minus_outstanding,
            first.observed_realized_hedge_work - first.outstanding_expected_work,
        )

    def test_state_boundaries_force_windows_even_before_width_limit(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(
                primary=((100.0,), (100.0,)),
                replay=((100.0,), (100.0,)),
                hedge=((100.0,), (100.0,)),
                timeline=CommonStateTimeline(10.0, 30.0, 55.0),
            ),
            planned_budget_rate=0.0,
        )
        self.assertEqual(
            [(window.start, window.end) for window in result.windows[:6]],
            [
                (0.0, 10.0),
                (10.0, 30.0),
                (30.0, 55.0),
                (55.0, 80.0),
                (80.0, 105.0),
                (105.0, 130.0),
            ],
        )
        self.assertEqual(result.simulation.simulation.tokens[0].replay_count, 1)

    def test_admission_does_not_read_the_future_hedge_draw(self):
        short = make_episode(
            primary=((100.0,), (100.0,)),
            hedge=((100.0,), (0.5,)),
        )
        long = make_episode(
            primary=((100.0,), (100.0,)),
            hedge=((100.0,), (50.0,)),
        )
        short_result = simulate_budgeted_laedge_episode(
            short, planned_budget_rate=0.04
        )
        long_result = simulate_budgeted_laedge_episode(
            long, planned_budget_rate=0.04
        )
        self.assertEqual(
            short_result.simulation.simulation.hedge_launches,
            long_result.simulation.simulation.hedge_launches,
        )
        self.assertEqual(
            short_result.simulation.simulation.hedge_suppressed,
            long_result.simulation.simulation.hedge_suppressed,
        )
        self.assertNotEqual(
            short_result.windows[0].launch_cohort_realized_work,
            long_result.windows[0].launch_cohort_realized_work,
        )

    def test_budgeted_result_passes_existing_idle_release_attribution_invariants(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(
                arrivals=(0.0, 0.1, 0.2),
                primary=((100.0, 100.0, 100.0), (100.0, 100.0, 100.0)),
                replay=((10.0, 10.0, 10.0), (10.0, 10.0, 10.0)),
                hedge=((100.0, 100.0, 100.0), (1.0, 1.0, 1.0)),
            ),
            planned_budget_rate=0.04,
        )
        metrics = build_episode_metrics(
            result.simulation,
            2.0,
            storm_bin_width=1.0,
            placement_mode="idle_release",
        )
        excluded = {
            "dispatcher_assignment_matches_protocol",
            "copy_placement_matches_protocol",
            "boundary_queue_counts_match_legacy_result",
        }
        self.assertTrue(
            all(
                value
                for name, value in metrics["invariants"].items()
                if name not in excluded
            )
        )

    def test_conservative_running_loser_is_completed_and_counted(self):
        result = simulate_budgeted_laedge_episode(
            make_episode(
                primary=((10.0,), (10.0,)),
                hedge=((10.0,), (1.0,)),
            ),
            planned_budget_rate=0.04,
            cancel_running_losers=False,
        )
        statuses = {
            attempt.status for attempt in result.simulation.simulation.tokens[0].attempts
        }
        self.assertIn(HedgeAttemptStatus.COMPLETED_LOSER, statuses)
        self.assertGreater(result.simulation.simulation.attempts[0].executed_work, 0.0)

    def test_result_is_deterministic_and_input_is_immutable(self):
        episode = make_episode()
        before = episode
        first = simulate_budgeted_laedge_episode(episode, planned_budget_rate=0.04)
        second = simulate_budgeted_laedge_episode(episode, planned_budget_rate=0.04)
        self.assertEqual(first, second)
        self.assertEqual(episode, before)
        with self.assertRaises(FrozenInstanceError):
            first.windows[0].start = 1.0

    def test_budget_rate_and_cancellation_inputs_are_strict(self):
        episode = make_episode()
        for bad in (-1.0, math.nan, math.inf, True, "0.04"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    simulate_budgeted_laedge_episode(
                        episode,
                        planned_budget_rate=bad,
                    )
        with self.assertRaises(ValueError):
            simulate_budgeted_laedge_episode(
                episode,
                planned_budget_rate=0.04,
                cancel_running_losers=1,
            )


if __name__ == "__main__":
    unittest.main()
