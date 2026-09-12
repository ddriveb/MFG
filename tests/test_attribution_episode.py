"""Fixed-horizon attribution episode protocol tests."""

from dataclasses import replace
from pathlib import Path
import unittest

from mfg_hedge.attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
    derive_episode_seed,
    episode_trace_fingerprint,
    generate_episode_trace,
    generate_macro_seed_episodes,
    simulate_episode,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.workload import (
    TokenSpec,
    WorkloadTrace,
    generate_workload_with_hedge,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
TIMELINE = CommonStateTimeline(10.0, 20.0, 30.0)
PROTOCOL = EpisodeProtocol(TIMELINE, arrival_cutoff=40.0)
NAMESPACE = "attribution:v1:development"


class EpisodeIdentityTests(unittest.TestCase):
    def test_key_and_seed_are_stable_and_episode_scoped(self) -> None:
        identity = EpisodeIdentity(NAMESPACE, macro_seed=7, episode_index=3)
        self.assertEqual(
            identity.key,
            "attribution:v1:development:7:episode:3",
        )
        first = derive_episode_seed(CONFIG.base_seed, identity)
        second = derive_episode_seed(CONFIG.base_seed, identity)
        other = derive_episode_seed(
            CONFIG.base_seed,
            EpisodeIdentity(NAMESPACE, macro_seed=7, episode_index=4),
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_holdout_key_matches_the_frozen_namespace_format(self) -> None:
        self.assertEqual(
            EpisodeIdentity("attribution:v1:holdout", 17, 4).key,
            "attribution:v1:holdout:17:episode:4",
        )

    def test_identity_and_seed_reject_bool_and_invalid_labels(self) -> None:
        for namespace in ("", "   ", "bad\nnamespace", 7, None):
            with self.subTest(namespace=namespace):
                with self.assertRaises(ValueError):
                    EpisodeIdentity(namespace, 0, 0)
        for value in (True, 1.0, "1", None, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EpisodeIdentity(NAMESPACE, value, 0)
                with self.assertRaises(ValueError):
                    EpisodeIdentity(NAMESPACE, 0, value)
        with self.assertRaises(ValueError):
            derive_episode_seed(True, EpisodeIdentity(NAMESPACE, 0, 0))


class FixedHorizonGenerationTests(unittest.TestCase):
    def test_frozen_v1_protocol_is_100_200_220_320(self) -> None:
        self.assertEqual(
            ATTRIBUTION_V1_PROTOCOL.timeline,
            CommonStateTimeline(100.0, 200.0, 220.0),
        )
        self.assertEqual(ATTRIBUTION_V1_PROTOCOL.arrival_cutoff, 320.0)
        episode = generate_episode_trace(
            CONFIG,
            "attribution:v1:test",
            0,
            0,
            ATTRIBUTION_V1_PROTOCOL,
        )
        self.assertTrue(
            all(token.arrival_time < 320.0 for token in episode.workload.tokens)
        )

    def test_half_open_trace_is_exact_prefix_of_existing_crn_streams(self) -> None:
        episode = generate_episode_trace(
            CONFIG,
            namespace=NAMESPACE,
            macro_seed=11,
            episode_index=2,
            protocol=PROTOCOL,
        )
        trace = episode.workload
        self.assertTrue(trace.tokens)
        self.assertTrue(all(token.arrival_time < 40.0 for token in trace.tokens))

        full = generate_workload_with_hedge(
            CONFIG,
            len(trace.tokens) + 1,
            base_seed=episode.episode_seed,
        )
        self.assertEqual(trace.tokens, full.tokens[:-1])
        self.assertGreaterEqual(full.tokens[-1].arrival_time, 40.0)
        for actual, expected in zip(trace.service_times, full.service_times):
            self.assertEqual(actual, expected[:-1])
        for actual, expected in zip(
            trace.replay_service_times, full.replay_service_times
        ):
            self.assertEqual(actual, expected[:-1])
        for actual, expected in zip(
            trace.hedge_service_times, full.hedge_service_times
        ):
            self.assertEqual(actual, expected[:-1])

    def test_same_key_repeats_and_different_episode_changes_every_namespace(self) -> None:
        kwargs = dict(
            config=CONFIG,
            namespace=NAMESPACE,
            macro_seed=4,
            protocol=PROTOCOL,
        )
        first = generate_episode_trace(episode_index=0, **kwargs)
        repeated = generate_episode_trace(episode_index=0, **kwargs)
        other = generate_episode_trace(episode_index=1, **kwargs)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.episode_seed, other.episode_seed)
        self.assertNotEqual(first.workload.tokens, other.workload.tokens)
        self.assertNotEqual(first.workload.service_times, other.workload.service_times)

    def test_macro_seed_has_dense_unique_episode_keys(self) -> None:
        episodes = generate_macro_seed_episodes(
            CONFIG,
            namespace=NAMESPACE,
            macro_seed=9,
            episode_count=3,
            protocol=PROTOCOL,
        )
        self.assertEqual(len(episodes), 3)
        self.assertEqual([e.identity.episode_index for e in episodes], [0, 1, 2])
        self.assertEqual(len({e.identity.key for e in episodes}), 3)
        self.assertEqual(len({e.episode_seed for e in episodes}), 3)

    def test_protocol_cutoff_and_episode_count_are_validated(self) -> None:
        for cutoff in (True, 0.0, -1.0, float("nan"), float("inf"), "40"):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    EpisodeProtocol(TIMELINE, cutoff)
        for count in (True, 0, -1, 2.0, "2"):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    generate_macro_seed_episodes(
                        CONFIG,
                        NAMESPACE,
                        0,
                        episode_count=count,
                        protocol=PROTOCOL,
                    )

    def test_fingerprint_covers_identity_protocol_and_all_crn_streams(self) -> None:
        episode = generate_episode_trace(CONFIG, NAMESPACE, 1, 0, PROTOCOL)
        fingerprint = episode_trace_fingerprint(episode)
        self.assertEqual(
            set(fingerprint),
            {
                "episode_protocol",
                "tokens",
                "attempt0_service_times",
                "attempt1_replay_times",
                "attempt2_hedge_times",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_episode_requires_complete_finite_attempt_streams_for_all_arms(self) -> None:
        valid = generate_workload_with_hedge(CONFIG, 2, base_seed=77)
        identity = EpisodeIdentity(NAMESPACE, 2, 0)
        with self.assertRaisesRegex(ValueError, "no hedge"):
            EpisodeTrace(
                identity,
                77,
                PROTOCOL,
                replace(valid, hedge_service_times=None),
            )
        invalid_hedge = (
            valid.hedge_service_times[0],
            (float("nan"), valid.hedge_service_times[1][1]),
        )
        with self.assertRaisesRegex(ValueError, "attempt_id=2"):
            EpisodeTrace(
                identity,
                77,
                PROTOCOL,
                replace(valid, hedge_service_times=invalid_hedge),
            )


class EpisodeSimulationTests(unittest.TestCase):
    def test_episode_stops_arrivals_then_drains_every_attempt(self) -> None:
        episode = generate_episode_trace(
            CONFIG, NAMESPACE, 3, 0, PROTOCOL
        )
        run = simulate_episode(
            episode,
            degraded_slowdown=2.0,
        )
        result = run.simulation
        self.assertEqual(result.completed_tokens, len(episode.workload.tokens))
        self.assertLess(result.last_arrival_time, 40.0)
        self.assertGreaterEqual(result.drain_end_time, result.last_arrival_time)
        self.assertEqual(
            run.post_cutoff_drain_duration,
            max(0.0, result.drain_end_time - 40.0),
        )
        self.assertTrue(all(a.terminal_time <= result.drain_end_time for a in result.attempts))

    def test_each_call_uses_a_fresh_engine_and_same_episode_can_feed_two_arms(self) -> None:
        episodes = generate_macro_seed_episodes(
            CONFIG, NAMESPACE, 5, episode_count=2, protocol=PROTOCOL
        )
        first_second_run = simulate_episode(episodes[1], 2.0)
        simulate_episode(episodes[0], 2.0)
        repeated_second_run = simulate_episode(episodes[1], 2.0)
        self.assertEqual(first_second_run, repeated_second_run)

        normal = simulate_episode(episodes[0], 2.0)
        hedged = simulate_episode(
            episodes[0],
            2.0,
            actions={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        self.assertEqual(normal.episode, hedged.episode)
        self.assertEqual(
            [t.arrival_time for t in normal.simulation.tokens],
            [t.arrival_time for t in hedged.simulation.tokens],
        )
        self.assertEqual(
            [t.token_class for t in normal.simulation.tokens],
            [t.token_class for t in hedged.simulation.tokens],
        )

    def test_timeline_must_leave_a_nonempty_recovered_arrival_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "arrival_cutoff"):
            EpisodeProtocol(TIMELINE, arrival_cutoff=30.0)

    def test_cutoff_rejects_new_arrival_but_drains_a_running_loser(self) -> None:
        trace = WorkloadTrace(
            base_seed=1,
            arrival_rate=1.0,
            tokens=(TokenSpec(0, 39.0, TokenClass.REGULAR),),
            service_times=((1.0,), (10.0,)),
            replay_service_times=((9.0,), (9.0,)),
            hedge_service_times=((9.0,), (10.0,)),
        )
        episode = EpisodeTrace(
            EpisodeIdentity(NAMESPACE, 0, 0), 1, PROTOCOL, trace
        )
        run = simulate_episode(
            episode,
            2.0,
            actions={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        self.assertEqual(run.simulation.tokens[0].completion_time, 40.0)
        self.assertEqual(run.simulation.drain_end_time, 49.0)
        self.assertEqual(run.simulation.drain_duration, 10.0)
        self.assertEqual(run.post_cutoff_drain_duration, 9.0)
        direct = simulate_hedge_common_state(
            trace,
            TIMELINE,
            2.0,
            actions={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        self.assertEqual(run.simulation, direct)

        at_cutoff = replace(
            trace,
            tokens=(TokenSpec(0, 40.0, TokenClass.REGULAR),),
        )
        with self.assertRaisesRegex(ValueError, "arrival_time < arrival_cutoff"):
            EpisodeTrace(
                EpisodeIdentity(NAMESPACE, 0, 1), 1, PROTOCOL, at_cutoff
            )
        # The cutoff belongs only to the new episode wrapper; the legacy engine
        # remains a general single-trace evaluator.
        legacy = simulate_hedge_common_state(at_cutoff, TIMELINE, 2.0)
        self.assertEqual(legacy.completed_tokens, 1)


if __name__ == "__main__":
    unittest.main()
