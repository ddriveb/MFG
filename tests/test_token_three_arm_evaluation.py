from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.attribution_episode import generate_episode_trace
from mfg_hedge.attribution_episode import ATTRIBUTION_V1_PROTOCOL
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction as A, TokenClass
from mfg_hedge.token_three_arm_evaluation import (
    BOOTSTRAP_NAMESPACE,
    BOOTSTRAP_REPLICATES,
    HOLDOUT_CALL_LIMIT,
    FrozenNIINPolicy,
    ThreeArmExecutionError,
    candidate_action_probabilities,
    evaluate_gate,
    paired_cluster_bootstrap,
    run_three_arm_evaluation,
    stable_policy_uniform,
    write_three_arm_artifact,
)


PROJECT = Path(__file__).resolve().parents[1]


class TokenThreeArmEvaluationTests(unittest.TestCase):
    def test_niin_is_exactly_n_i_i_n(self):
        policy = FrozenNIINPolicy(late_after=50.0)
        self.assertEqual(policy.action(TokenClass.REGULAR, 10.0, 0), A.NORMAL)
        self.assertEqual(policy.action(TokenClass.URGENT, 10.0, 0), A.IMMEDIATE_HEDGE)
        self.assertEqual(policy.action(TokenClass.REGULAR, 60.0, 0), A.IMMEDIATE_HEDGE)
        self.assertEqual(policy.action(TokenClass.URGENT, 60.0, 0), A.NORMAL)
        self.assertEqual(policy.action(TokenClass.REGULAR, 10.0, 1), A.NORMAL)

    def test_requested_price_softmax_hand_calculation(self):
        probabilities = candidate_action_probabilities(
            {A.NORMAL: 1.0, A.DELAYED_HEDGE: 2.0, A.IMMEDIATE_HEDGE: 2.0},
            requested_reservation_price=2.0,
            beta=4.0,
        )
        expected_n = 1.0 / (1.0 + 2.0 * __import__("math").exp(-12.0))
        self.assertAlmostEqual(probabilities[A.NORMAL], expected_n, places=15)
        self.assertAlmostEqual(
            probabilities[A.DELAYED_HEDGE], (1.0 - expected_n) / 2.0, places=15
        )
        self.assertAlmostEqual(probabilities[A.IMMEDIATE_HEDGE], (1.0 - expected_n) / 2.0, places=15)

    def test_unsupported_candidate_falls_back_to_exact_niin(self):
        policy = FrozenNIINPolicy(late_after=50.0, supported_probabilities={})
        self.assertEqual(policy.choose_observation(None), A.NORMAL)

    def test_stable_policy_uniform_is_sha256_deterministic(self):
        first = stable_policy_uniform("seed", "token:4")
        second = stable_policy_uniform("seed", "token:4")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_holdout_uses_three_calls_and_shared_trace(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        result = run_three_arm_evaluation(config, episode_count=1)
        self.assertEqual(result["attempted_calls"], 3)
        self.assertEqual(result["max_scheduler_calls"], 3)
        self.assertEqual(len(result["episode_rows"]), 1)
        row = result["episode_rows"][0]
        self.assertEqual(len(set(row["trace_fingerprints"].values())), 1)

    def test_failed_arm_invalidates_whole_episode_without_zero_fill(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        original = generate_episode_trace(
            config,
            "token-mfg-restoration:three-arm-holdout:v1",
            20260918,
            0,
            ATTRIBUTION_V1_PROTOCOL,
        )

        def fail_on_second(*args, **kwargs):
            fail_on_second.calls += 1
            if fail_on_second.calls == 2:
                raise RuntimeError("synthetic physical failure")
            raise RuntimeError("stop after first call")

        fail_on_second.calls = 0
        with patch("mfg_hedge.token_three_arm_evaluation.simulate_episode_online", side_effect=fail_on_second):
            result = run_three_arm_evaluation(
                config, episode_count=1, episode_builder=lambda _: original
            )
        self.assertEqual(result["status"], "physical_failed")
        self.assertEqual(result["attempted_calls"], 3)
        self.assertEqual(result["complete_episode_count"], 0)
        self.assertEqual(result["failed_episode_count"], 1)

    def test_paired_bootstrap_reuses_one_index_library(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (0.0, 1.0, 2.0, 3.0)
        first = paired_cluster_bootstrap(a, b, replicates=16)
        second = paired_cluster_bootstrap(a, b, replicates=16)
        self.assertEqual(first, second)
        self.assertEqual(first["namespace"], BOOTSTRAP_NAMESPACE)
        self.assertEqual(first["replicates"], 16)

    def test_gate_boundary_hand_calculation(self):
        metrics = {
            "C_vs_B": {
                "df_cvar95": {"relative_change": -0.10, "ci95": (-0.20, -0.01)},
                "df_miss_rate": {"relative_change": 0.0},
                "replay_rate": {"relative_change": 0.0},
                "hr_p99": {"relative_change": 0.02},
                "total_work": {"relative_change": 0.05},
                "storm_peak": {"relative_change": 0.0},
            }
        }
        self.assertTrue(evaluate_gate(metrics["C_vs_B"], invariants_pass=True))

    def test_call_limit_and_claim_boundary_are_frozen(self):
        self.assertEqual(HOLDOUT_CALL_LIMIT, 3072)
        self.assertEqual(BOOTSTRAP_REPLICATES, 4096)
        with self.assertRaises(ThreeArmExecutionError):
            run_three_arm_evaluation(
                load_config(PROJECT / "configs/v1_minimal.json"), episode_count=1025
            )

    def test_artifact_is_atomic_and_non_overwriting(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        result = run_three_arm_evaluation(config, episode_count=1)
        with tempfile.TemporaryDirectory() as temp:
            first = write_three_arm_artifact(temp, "test-three-arm", result, config)
            self.assertTrue((first / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                write_three_arm_artifact(temp, "test-three-arm", result, config)


if __name__ == "__main__":
    unittest.main()
