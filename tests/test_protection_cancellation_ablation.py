import unittest

from mfg_hedge.attribution_episode import (
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import HedgeAttemptStatus
from mfg_hedge.laedge_episode import _snapshot as legacy_laedge_snapshot
from mfg_hedge.laedge_episode import simulate_laedge_episode
from mfg_hedge.protection_cancellation_ablation import (
    CANCELLATION_ARMS,
    CancellationAblationArm,
    PolicyFamily,
    simulate_cancellation_arm,
)
from mfg_hedge.workload import TokenSpec, WorkloadTrace


def make_episode(
    *,
    arrivals=(1.0,),
    primary=((10.0,), (10.0,)),
    replay=((1.0,), (1.0,)),
    hedge=((1.0,), (1.0,)),
    timeline=None,
):
    timeline = timeline or CommonStateTimeline(0.0, 100.0, 120.0)
    count = len(arrivals)
    tokens = tuple(
        TokenSpec(index, float(arrival), TokenClass.REGULAR)
        for index, arrival in enumerate(arrivals)
    )
    def expand(rows):
        return tuple(tuple(float(value) for value in row) for row in rows)
    trace = WorkloadTrace(
        base_seed=20260909,
        arrival_rate=1.4,
        tokens=tokens,
        service_times=expand(primary),
        replay_service_times=expand(replay),
        hedge_service_times=expand(hedge),
    )
    return EpisodeTrace(
        EpisodeIdentity("replica-cancel-ablation:test", 20260909, count),
        20260909,
        EpisodeProtocol(timeline, 200.0),
        trace,
    )


class CancellationAblationTests(unittest.TestCase):
    def test_laedge_boundary_does_not_backfill_unassigned_waiting_to_future_replica(self):
        episode = make_episode(
            arrivals=(0.0, 0.0, 0.0, 0.0),
            primary=((100.0, 10.0, 10.0, 10.0), (100.0, 10.0, 10.0, 10.0)),
            replay=((10.0, 10.0, 10.0, 10.0), (10.0, 10.0, 10.0, 10.0)),
            hedge=((100.0, 10.0, 10.0, 10.0), (100.0, 10.0, 10.0, 10.0)),
            timeline=CommonStateTimeline(0.0, 2.0, 4.0),
        )

        run = simulate_laedge_episode(
            episode,
            degraded_slowdown=2.0,
            cancel_running_losers=False,
        )
        failed = run.failed_start_snapshot
        legacy = legacy_laedge_snapshot(
            run.simulation.attempts,
            episode.protocol.timeline.failed_start,
            episode,
            2.0,
            failed=True,
        )

        # Tokens 1 and 2 wait without a Replica binding before failure and are
        # later started on A.  Terminal reconstruction therefore falsely
        # backfills them into the historical failed-start snapshot.
        self.assertGreater(legacy.live_attempts[0], 0)
        self.assertEqual(failed.live_attempts[0], 0)
        self.assertEqual(failed.queued_attempts[0], 0)
        audit = getattr(run, "_idle_release_boundary_audit")
        self.assertGreaterEqual(audit["failed"].unassigned_waiting_attempts, 2)

    def test_four_arms_are_explicit_cartesian_product(self):
        self.assertEqual(
            tuple(arm.key for arm in CANCELLATION_ARMS),
            (
                "niin:conservative",
                "niin:preemptive",
                "laedge:conservative",
                "laedge:preemptive",
            ),
        )
        self.assertTrue(all(isinstance(arm, CancellationAblationArm) for arm in CANCELLATION_ARMS))

    def test_niin_preemptive_cancels_running_loser_and_ignores_stale_completion(self):
        episode = make_episode(
            primary=((10.0,), (10.0,)),
            hedge=((1.0,), (1.0,)),
        )
        conservative = simulate_cancellation_arm(
            episode,
            CancellationAblationArm(PolicyFamily.NIIN, False),
            action_source={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        preemptive = simulate_cancellation_arm(
            episode,
            CancellationAblationArm(PolicyFamily.NIIN, True),
            action_source={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        conservative_token = conservative.simulation.simulation.tokens[0]
        preemptive_token = preemptive.simulation.simulation.tokens[0]
        conservative_loser = next(
            row for row in conservative_token.attempts if row.attempt_id == 0
        )
        preemptive_loser = next(
            row for row in preemptive_token.attempts if row.attempt_id == 0
        )
        self.assertEqual(conservative_loser.status, HedgeAttemptStatus.COMPLETED_LOSER)
        self.assertEqual(preemptive_loser.status, HedgeAttemptStatus.CANCELLED_RUNNING)
        self.assertAlmostEqual(preemptive_loser.executed_work, 0.5)
        self.assertEqual(preemptive.simulation.simulation.stale_completion_events_ignored, 1)
        self.assertEqual(conservative.simulation.simulation.completed_loser_total, 1)
        self.assertEqual(preemptive.simulation.simulation.completed_loser_total, 0)

    def test_laedge_conservative_variant_retains_running_loser(self):
        episode = make_episode(
            primary=((10.0,), (10.0,)),
            hedge=((1.0,), (1.0,)),
        )
        conservative = simulate_cancellation_arm(
            episode,
            CancellationAblationArm(PolicyFamily.LAEDGE, False),
        )
        preemptive = simulate_cancellation_arm(
            episode,
            CancellationAblationArm(PolicyFamily.LAEDGE, True),
        )
        conservative_token = conservative.simulation.tokens[0]
        preemptive_token = preemptive.simulation.tokens[0]
        self.assertEqual(
            next(row for row in conservative_token.attempts if row.attempt_id == 0).status,
            HedgeAttemptStatus.COMPLETED_LOSER,
        )
        loser = next(row for row in preemptive_token.attempts if row.attempt_id == 0)
        self.assertEqual(loser.status, HedgeAttemptStatus.CANCELLED_RUNNING)
        self.assertAlmostEqual(loser.executed_work, 0.5)
        self.assertEqual(conservative.simulation.completed_loser_total, 1)
        self.assertEqual(preemptive.simulation.completed_loser_total, 0)

    def test_queued_loser_remains_zero_work_in_both_modes(self):
        episode = make_episode(
            arrivals=(0.0, 0.5),
            primary=((10.0, 100.0), (10.0, 100.0)),
            replay=((1.0, 1.0), (1.0, 1.0)),
            hedge=((1.0, 1.0), (1.0, 1.0)),
        )
        for mode in (False, True):
            result = simulate_cancellation_arm(
                episode,
                CancellationAblationArm(PolicyFamily.NIIN, mode),
                action_source={0: ProtectionAction.DELAYED_HEDGE},
                hedge_delay=1.0,
            )
            token = result.simulation.simulation.tokens[0]
            queued = next(row for row in token.attempts if row.attempt_id == 2)
            self.assertEqual(queued.status, HedgeAttemptStatus.CANCELLED_QUEUED)
            self.assertEqual(queued.executed_work, 0.0)

    def test_failure_replay_and_crn_streams_are_unchanged_by_mode(self):
        episode = make_episode(
            timeline=CommonStateTimeline(0.0, 2.0, 4.0),
            primary=((10.0,), (10.0,)),
            replay=((1.5,), (1.5,)),
            hedge=((1.0,), (1.0,)),
        )
        rows = []
        for mode in (False, True):
            result = simulate_cancellation_arm(
                episode,
                CancellationAblationArm(PolicyFamily.NIIN, mode),
                action_source={0: ProtectionAction.NORMAL},
            )
            token = result.simulation.simulation.tokens[0]
            rows.append(token)
            self.assertEqual(token.replay_count, 1)
            self.assertEqual(token.winner_attempt_id, 1)
            self.assertEqual(
                next(row for row in token.attempts if row.attempt_id == 0).status,
                HedgeAttemptStatus.FAILED_RUNNING,
            )
        self.assertEqual(
            tuple(row.required_work for row in rows[0].attempts),
            tuple(row.required_work for row in rows[1].attempts),
        )


if __name__ == "__main__":
    unittest.main()
