import dataclasses
import unittest

import mfg_hedge.reliability_aware_routing as routing
from mfg_hedge.reliability_aware_routing import RoutingToken, build_manual_trace
from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
from mfg_hedge.token_population_response import (
    ReplicaStateActionBucket,
    _branch_key,
)


class _KeyRecordingPolicy:
    def __init__(self):
        self.keys = []

    def choose(self, observation, context, policy_key):
        self.keys.append(policy_key)
        return 0


def _one_token_trace(*, scenario="S0", episode_index=0):
    return build_manual_trace(
        (RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),),
        namespace="action-crn-correction",
        macro_seed=20260910,
        scenario=scenario,
        episode_index=episode_index,
    )


class KScaleActionCRNCorrectionTests(unittest.TestCase):
    def test_high_risk_type_repeats_at_fixed_fraction_for_each_supported_k(self):
        for K in (8, 16, 32, 64):
            high_s1 = tuple(
                replica_id for replica_id in range(K)
                if routing._scenario_rates("S1", replica_id, replica_id % 4, 0.0)[0]
                == 0.04
            )
            high_s2_early = tuple(
                replica_id for replica_id in range(K)
                if routing._scenario_rates("S2", replica_id, replica_id % 4, 0.0)[0]
                == 0.04
            )
            high_s2_late = tuple(
                replica_id for replica_id in range(K)
                if routing._scenario_rates("S2", replica_id, replica_id % 4, 20.0)[0]
                == 0.04
            )
            self.assertEqual(len(high_s1), K // 4)
            self.assertEqual(len(high_s2_early), K // 4)
            self.assertEqual(len(high_s2_late), K // 4)
            self.assertEqual(high_s1, tuple(i for i in range(K) if i % 8 in {0, 1}))
            self.assertEqual(
                high_s2_late, tuple(i for i in range(K) if i % 8 in {2, 3})
            )

    def test_action_key_is_episode_specific_and_repeatable(self):
        first = _KeyRecordingPolicy()
        simulate_simultaneous_routing(_one_token_trace(), first)
        repeated = _KeyRecordingPolicy()
        simulate_simultaneous_routing(_one_token_trace(), repeated)
        other_episode = _KeyRecordingPolicy()
        simulate_simultaneous_routing(
            _one_token_trace(episode_index=1), other_episode
        )
        other_scenario = _KeyRecordingPolicy()
        simulate_simultaneous_routing(
            _one_token_trace(scenario="S1"), other_scenario
        )

        self.assertEqual(first.keys, repeated.keys)
        self.assertNotEqual(first.keys, other_episode.keys)
        self.assertNotEqual(first.keys, other_scenario.keys)
        self.assertIn("action-crn-correction", first.keys[0])
        self.assertIn("S0", first.keys[0])
        self.assertIn("episode=0", first.keys[0])
        self.assertIn("bucket=base", first.keys[0])

    def test_population_branch_key_pairs_iterations_but_separates_scenario_and_bucket(self):
        baseline = _branch_key(
            "response", 9, ("response", 9, 2), 3, 7, None, 0,
            scenario="S0",
        )
        repeated_iteration = _branch_key(
            "response", 9, ("response", 9, 2), 3, 7, None, 19,
            scenario="S0",
        )
        other_scenario = _branch_key(
            "response", 9, ("response", 9, 2), 3, 7, None, 0,
            scenario="S1",
        )
        bucket = ReplicaStateActionBucket(
            0, "UP", "empty", "zero", "zero", "zero", "zero", "same_domain"
        )
        action_branch = _branch_key(
            "response", 9, ("response", 9, 2), 3, 7, bucket, 0,
            scenario="S0",
        )

        self.assertEqual(baseline, repeated_iteration)
        self.assertNotEqual(baseline, other_scenario)
        self.assertNotEqual(baseline, action_branch)
        self.assertNotIn("iteration=", baseline)

    def test_k8_trace_identity_remains_deterministic(self):
        left = _one_token_trace()
        right = dataclasses.replace(_one_token_trace(), episode_index=0)
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
