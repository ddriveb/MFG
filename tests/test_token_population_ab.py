from pathlib import Path
import tempfile
import unittest

from mfg_hedge.config import load_config
from mfg_hedge.token_best_response import load_t3a_runtime_candidate
from mfg_hedge.token_population_ab import (
    T3C_LABEL,
    execute_population_ab,
    latency_summary,
    population_call_budget,
    run_population_ab,
)


PROJECT = Path(__file__).resolve().parents[1]
VALIDATION = PROJECT / "artifacts/token-t3a-oracle-20260907-r1-validation"
OCCUPANCY = PROJECT / "artifacts/token-t3a-oracle-20260907-r1-occupancy"


class TokenPopulationABTests(unittest.TestCase):
    def test_call_budget_is_two_complete_runs_per_episode(self):
        self.assertEqual(population_call_budget(2048), 4096)
        self.assertEqual(population_call_budget(7), 14)

    def test_latency_summary_hand_calculation(self):
        result = latency_summary((1.0, 2.0, 4.0, 8.0))
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mean"], 3.75)
        self.assertAlmostEqual(result["p95"], 7.4)
        self.assertAlmostEqual(result["p99"], 7.88)

    def test_small_ab_uses_same_trace_and_calls_every_token_online(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_population_ab(config, candidate, episode_count=2)
        self.assertEqual(result["attempted_calls"], 4)
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(len(result["episode_rows"]), 2)
        for row in result["episode_rows"]:
            self.assertEqual(row["trace_fingerprint_a"], row["trace_fingerprint_b"])
            self.assertEqual(row["a_decision_count"], row["token_count"])
            self.assertEqual(row["b_decision_count"], row["token_count"])
            self.assertEqual(row["token_identity_fingerprint_a"], row["token_identity_fingerprint_b"])

    def test_ab_result_is_deterministic(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        first = run_population_ab(config, candidate, episode_count=2)
        second = run_population_ab(config, candidate, episode_count=2)
        self.assertEqual(first, second)

    def test_metrics_include_overall_d_f_and_storm_contract(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_population_ab(config, candidate, episode_count=1)
        for arm in ("A", "B"):
            metrics = result["arms"][arm]
            self.assertIn("overall", metrics["latency"])
            self.assertIn("D", metrics["latency"]["arrival_phase"])
            self.assertIn("F", metrics["latency"]["arrival_phase"])
            self.assertIn("replay", metrics)
            self.assertIn("hedge", metrics)
            self.assertIn("work", metrics)
            self.assertIn("reservation", metrics)
            self.assertIn("storm", metrics)

    def test_claim_boundary_is_descriptive_only(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_population_ab(config, candidate, episode_count=1)
        self.assertEqual(result["label"], T3C_LABEL)
        for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg"):
            self.assertFalse(result[name])

    def test_artifact_is_transactional_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            first, _ = execute_population_ab(
                PROJECT,
                temp,
                "t3c-test-run",
                validation_dir=VALIDATION,
                occupancy_dir=OCCUPANCY,
                episode_count=1,
            )
            self.assertTrue((first / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                execute_population_ab(
                    PROJECT,
                    temp,
                    "t3c-test-run",
                    validation_dir=VALIDATION,
                    occupancy_dir=OCCUPANCY,
                    episode_count=1,
                )


if __name__ == "__main__":
    unittest.main()
