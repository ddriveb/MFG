from pathlib import Path
import tempfile
import unittest

from mfg_hedge.attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
    generate_episode_trace,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import TokenClass
from mfg_hedge.hedge_simulation import HedgeAttemptStatus
from mfg_hedge.protection_baseline_experiment import (
    PROTECTION_ARMS,
    P95_CALIBRATION_MACRO_SEED,
    P95_CALIBRATION_NAMESPACE,
    calibrate_p95_delay,
    run_protection_baseline_experiment,
    write_protection_baseline_artifact,
)
from mfg_hedge.laedge_episode import simulate_laedge_episode
from mfg_hedge.workload import TokenSpec, WorkloadTrace


PROJECT = Path(__file__).resolve().parents[1]


def make_episode(arrivals, primary, hedge, replay=None, *, timeline=None):
    timeline = timeline or CommonStateTimeline(100.0, 200.0, 220.0)
    replay = replay or primary
    trace = WorkloadTrace(
        base_seed=7,
        arrival_rate=1.0,
        tokens=tuple(
            TokenSpec(i, value, TokenClass.REGULAR)
            for i, value in enumerate(arrivals)
        ),
        service_times=tuple(tuple(row) for row in primary),
        replay_service_times=tuple(tuple(row) for row in replay),
        hedge_service_times=tuple(tuple(row) for row in hedge),
    )
    return EpisodeTrace(
        EpisodeIdentity("test:laedge", 1, 0),
        7,
        EpisodeProtocol(timeline, 300.0),
        trace,
    )


class LAEdgeEpisodeTests(unittest.TestCase):
    def test_release_serves_oldest_unserved_before_hedging(self):
        item = make_episode(
            (0.0, 0.1, 0.2),
            ((10.0, 2.0, 2.0), (1.0, 2.0, 2.0)),
            ((10.0, 2.0, 2.0), (1.0, 2.0, 2.0)),
        )
        run = simulate_laedge_episode(item, degraded_slowdown=2.0)
        first = run.simulation.tokens[0]
        self.assertEqual(first.winner_attempt_id, 2)
        cancelled = next(row for row in first.attempts if row.attempt_id == 0)
        self.assertEqual(cancelled.status, HedgeAttemptStatus.CANCELLED_RUNNING)
        self.assertAlmostEqual(cancelled.executed_work, 1.0)
        token1 = run.simulation.tokens[1]
        token2 = run.simulation.tokens[2]
        self.assertEqual(token1.attempts[0].start_time, 1.0)
        self.assertEqual(token2.attempts[0].start_time, 1.0)
        self.assertTrue(
            all(row.attempt_id == 0 for row in token1.attempts + token2.attempts)
        )

    def test_fault_first_replays_token_with_no_surviving_copy(self):
        item = make_episode(
            (0.0, 0.1, 0.2),
            ((10.0, 300.0, 20.0), (1.0, 300.0, 300.0)),
            ((10.0, 300.0, 20.0), (1.0, 300.0, 300.0)),
            replay=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            timeline=CommonStateTimeline(100.0, 200.0, 220.0),
        )
        run = simulate_laedge_episode(item, degraded_slowdown=2.0)
        token1 = run.simulation.tokens[1]
        self.assertEqual(token1.replay_count, 1)
        replay = next(row for row in token1.attempts if row.attempt_id == 1)
        self.assertEqual(replay.replica_id, 1)
        self.assertGreaterEqual(
            run.simulation.drain_end_time, run.simulation.last_arrival_time
        )
        self.assertEqual(len(run.simulation.tokens), 3)

    def test_trace_is_immutable_and_streams_bind_to_replica(self):
        item = make_episode(
            (0.0,),
            ((9.0,), (8.0,)),
            ((7.0,), (0.5,)),
        )
        before = item
        run = simulate_laedge_episode(item, degraded_slowdown=2.0)
        self.assertEqual(item, before)
        attempts = {
            row.attempt_id: row for row in run.simulation.tokens[0].attempts
        }
        self.assertEqual(attempts[0].required_work, 9.0)
        self.assertEqual(attempts[2].required_work, 0.5)


class ProtectionCampaignTests(unittest.TestCase):
    def test_p95_calibration_is_disjoint_and_positive(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        result = calibrate_p95_delay(config, episode_count=2)
        expected = sum(
            spec.arrival_time < ATTRIBUTION_V1_PROTOCOL.timeline.degraded_start
            for index in range(2)
            for spec in generate_episode_trace(
                config,
                P95_CALIBRATION_NAMESPACE,
                P95_CALIBRATION_MACRO_SEED,
                index,
                ATTRIBUTION_V1_PROTOCOL,
            ).workload.tokens
        )
        self.assertGreater(result["delay"], 0.0)
        self.assertEqual(result["scheduler_calls"], 2)
        self.assertEqual(result["latency_sample_count"], expected)
        self.assertEqual(result["cohort"], "healthy_arrival_phase")
        self.assertIn("p95-calibration", result["namespace"])

    def test_smoke_uses_five_arms_on_identical_trace(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        result = run_protection_baseline_experiment(
            config, episode_count=1, calibration_episode_count=2
        )
        self.assertEqual(tuple(result["arms"]), PROTECTION_ARMS)
        self.assertEqual(result["evaluation_calls"], 5)
        self.assertEqual(result["calibration"]["scheduler_calls"], 2)
        row = result["episode_rows"][0]
        self.assertEqual(len(set(row["trace_fingerprints"].values())), 1)
        self.assertFalse(result["claims_best_response"])
        self.assertFalse(result["claims_mfg"])

    def test_artifact_is_transactional_and_non_overwriting(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        result = run_protection_baseline_experiment(
            config, episode_count=1, calibration_episode_count=2
        )
        with tempfile.TemporaryDirectory() as root:
            path = write_protection_baseline_artifact(root, "run-1", result, config)
            self.assertTrue((path / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                write_protection_baseline_artifact(root, "run-1", result, config)


if __name__ == "__main__":
    unittest.main()
