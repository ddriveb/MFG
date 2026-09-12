from dataclasses import replace
import math
import unittest

from mfg_hedge.attribution_episode import EpisodeIdentity, EpisodeProtocol, EpisodeTrace
from mfg_hedge.attribution_metrics import SLODeadlines
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import TokenClass
from mfg_hedge.hedge_simulation import HedgeAttemptStatus
from mfg_hedge.selective_hedging import (
    _SelectiveLAEdgeEngine,
    SELECTIVE_HEDGE_THRESHOLD_GRID,
    SelectiveHedgeThresholds,
    normalized_slack,
    selective_arm_key,
    selective_candidate_key,
    simulate_selective_laedge_episode,
)
from mfg_hedge.budgeted_laedge import simulate_budgeted_laedge_episode
from mfg_hedge.laedge_episode import _Pending, _Running
from mfg_hedge.workload import TokenSpec, WorkloadTrace


def make_episode(
    arrivals=(0.0,),
    classes=None,
    primary=((100.0,), (100.0,)),
    replay=((10.0,), (10.0,)),
    hedge=((10.0,), (10.0,)),
    timeline=None,
    cutoff=3000.0,
):
    classes = classes or (TokenClass.REGULAR,) * len(arrivals)
    tokens = tuple(
        TokenSpec(index, float(arrival), token_class)
        for index, (arrival, token_class) in enumerate(zip(arrivals, classes))
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
        EpisodeIdentity("selective-hedging:test", 20260909, len(tokens)),
        20260909,
        EpisodeProtocol(
            timeline or CommonStateTimeline(1000.0, 2000.0, 2200.0),
            cutoff,
        ),
        trace,
    )


class SelectiveHedgeContractTests(unittest.TestCase):
    def test_normalized_slack_is_unitless_and_thresholds_are_strict(self):
        deadlines = SLODeadlines(regular=30.0, urgent=20.0)
        token = TokenSpec(7, 10.0, TokenClass.REGULAR)
        self.assertAlmostEqual(normalized_slack(token, 25.0, deadlines), 0.5)
        self.assertEqual(
            selective_candidate_key(token, 25.0, deadlines, class_priority=1),
            (0.5, 1, 10.0, 7),
        )
        with self.assertRaises(ValueError):
            SelectiveHedgeThresholds(regular=math.nan, urgent=0.3)
        with self.assertRaises(ValueError):
            SelectiveHedgeThresholds(regular=1.1, urgent=0.3)
        self.assertEqual(SELECTIVE_HEDGE_THRESHOLD_GRID, (0.15, 0.30, 0.50))
        self.assertEqual(
            selective_arm_key(SelectiveHedgeThresholds(regular=0.15, urgent=0.5)),
            "selective-laedge:theta_regular=0.15:theta_urgent=0.5",
        )

    def test_degraded_primary_a_can_hedge_only_to_idle_healthy_b(self):
        episode = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 15.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        )
        result = simulate_selective_laedge_episode(
            episode,
            planned_budget_rate=1.0,
            deadlines=SLODeadlines(regular=30.0, urgent=20.0),
            thresholds=SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        )
        self.assertEqual(result.simulation.simulation.hedge_launches, 1)
        token = result.simulation.simulation.tokens[0]
        hedge = next(attempt for attempt in token.attempts if attempt.attempt_id == 2)
        self.assertEqual(hedge.replica_id, 1)
        self.assertEqual(result.suppression_reasons, ())

    def test_h_f_and_r_never_start_selective_hedge(self):
        healthy = make_episode(
            primary=((100.0,), (100.0,)),
            hedge=((0.1,), (0.1,)),
            timeline=CommonStateTimeline(100.0, 200.0, 220.0),
        )
        recovered = make_episode(
            arrivals=(40.0,),
            primary=((100.0,), (100.0,)),
            hedge=((0.1,), (0.1,)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=50.0,
        )
        failed = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 9.9)),
            replay=((1.0, 1.0), (1.0, 1.0)),
            hedge=((0.1, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(5.0, 10.0, 20.0),
        )
        kwargs = {
            "planned_budget_rate": 1.0,
            "deadlines": SLODeadlines(regular=30.0, urgent=20.0),
            "thresholds": SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        }
        self.assertEqual(
            simulate_selective_laedge_episode(healthy, **kwargs).simulation.simulation.hedge_launches,
            0,
        )
        self.assertEqual(
            simulate_selective_laedge_episode(recovered, **kwargs).simulation.simulation.hedge_launches,
            0,
        )
        failed_result = simulate_selective_laedge_episode(failed, **kwargs)
        self.assertEqual(failed_result.simulation.simulation.hedge_launches, 0)
        self.assertEqual(failed_result.simulation.simulation.tokens[0].replay_count, 1)

    def test_primary_or_replay_waiting_work_precedes_hedge(self):
        episode = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 100.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        )
        engine = _SelectiveLAEdgeEngine(
            episode,
            2.0,
            planned_budget_rate=1.0,
            cancel_running_losers=False,
            deadlines=SLODeadlines(regular=30.0, urgent=20.0),
            thresholds=SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        )
        engine.tokens[0].primary_replica = 0
        engine.running[0] = _Running(
            token_id=0,
            attempt_id=0,
            replica_id=0,
            enqueue_time=0.0,
            start_time=0.0,
            required_work=100.0,
            executed_work=0.0,
            last_update=15.0,
            speed=0.5,
        )
        engine.waiting.append(_Pending(1, 0, 15.0, pinned_replica=1))
        engine._release(15.0)
        self.assertEqual(engine.hedge_launches, 0)
        self.assertIsNotNone(engine.running[1])
        self.assertEqual(engine.running[1].token_id, 1)

    def test_budget_suppression_is_causal_and_does_not_read_hedge_draw(self):
        episode = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 15.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        )
        kwargs = {
            "planned_budget_rate": 0.0,
            "deadlines": SLODeadlines(regular=30.0, urgent=20.0),
            "thresholds": SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        }
        short = simulate_selective_laedge_episode(episode, **kwargs)
        long_episode = replace(
            episode,
            workload=replace(
                episode.workload,
                hedge_service_times=((100.0, 50.0), (50.0, 50.0)),
            ),
        )
        long = simulate_selective_laedge_episode(long_episode, **kwargs)
        self.assertEqual(short.simulation.simulation.hedge_launches, 0)
        self.assertEqual(short.simulation.simulation.hedge_suppressed, 1)
        self.assertEqual(short.suppression_reasons[0].reason, "budget")
        self.assertEqual(short.suppression_reasons, long.suppression_reasons)

    def test_same_episode_is_deterministic_and_input_is_unchanged(self):
        episode = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 15.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        )
        result1 = simulate_selective_laedge_episode(
            episode,
            planned_budget_rate=1.0,
            deadlines=SLODeadlines(regular=30.0, urgent=20.0),
            thresholds=SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        )
        result2 = simulate_selective_laedge_episode(
            episode,
            planned_budget_rate=1.0,
            deadlines=SLODeadlines(regular=30.0, urgent=20.0),
            thresholds=SelectiveHedgeThresholds(regular=0.5, urgent=0.5),
        )
        self.assertEqual(result1, result2)
        self.assertEqual(episode, make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 15.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        ))

    def test_threshold_zero_preserves_dynamic_no_hedge_physics(self):
        episode = make_episode(
            arrivals=(0.0, 0.1),
            primary=((100.0, 100.0), (100.0, 15.0)),
            replay=((10.0, 10.0), (10.0, 10.0)),
            hedge=((100.0, 0.1), (0.1, 0.1)),
            timeline=CommonStateTimeline(10.0, 20.0, 30.0),
            cutoff=40.0,
        )
        selective = simulate_selective_laedge_episode(
            episode,
            planned_budget_rate=1.0,
            deadlines=SLODeadlines(regular=30.0, urgent=20.0),
            thresholds=SelectiveHedgeThresholds(regular=0.0, urgent=0.0),
        )
        no_hedge = simulate_budgeted_laedge_episode(
            episode,
            planned_budget_rate=0.0,
        )
        self.assertEqual(
            selective.simulation.simulation.tokens,
            no_hedge.simulation.simulation.tokens,
        )
        self.assertEqual(
            selective.simulation.simulation.attempts,
            no_hedge.simulation.simulation.attempts,
        )
        self.assertEqual(
            selective.simulation.simulation.drain_end_time,
            no_hedge.simulation.simulation.drain_end_time,
        )


if __name__ == "__main__":
    unittest.main()
