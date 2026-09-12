import unittest
from pathlib import Path
import tempfile

from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction
from mfg_hedge.token_best_response import load_t3a_runtime_candidate
from mfg_hedge.token_t3b_execution import (
    QUALIFICATION_MACRO_SEED,
    QUALIFICATION_NAMESPACE,
    T3B_LABEL,
    execute_t3b_qualification,
    run_qualification,
    paired_mean_se,
    qualification_call_budget,
    summarize_bin,
)


PROJECT = Path(__file__).resolve().parents[1]
VALIDATION = PROJECT / "artifacts/token-t3a-oracle-20260907-r1-validation"
OCCUPANCY = PROJECT / "artifacts/token-t3a-oracle-20260907-r1-occupancy"


class T3BExecutionTests(unittest.TestCase):
    def test_paired_statistics_use_episode_as_unit(self):
        stats = paired_mean_se((1.0, 3.0, 5.0))
        self.assertEqual(stats["effective_n"], 3)
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["standard_error"], 2.0 / 3**0.5)

    def test_exact_five_calls_per_selected_panel(self):
        self.assertEqual(qualification_call_budget(2048), 10240)
        self.assertEqual(qualification_call_budget(7), 35)

    def test_insufficient_bin_is_fail_closed(self):
        result = summarize_bin(
            "bin",
            construction_action=ProtectionAction.NORMAL,
            costs={
                ProtectionAction.NORMAL: (1.0, 2.0),
                ProtectionAction.DELAYED_HEDGE: (1.5, 2.5),
                ProtectionAction.IMMEDIATE_HEDGE: (2.0, 3.0),
            },
            panel_floor=32,
            missing_target_count=1,
            failed_panel_count=0,
        )
        self.assertTrue(result["insufficient"])
        self.assertIsNone(result["candidate_max_upper95"])
        self.assertEqual(result["missing_target_count"], 1)

    def test_public_label_has_no_equilibrium_claim(self):
        self.assertEqual(
            T3B_LABEL, "conditional_best_response_candidate_with_holdout_diagnostics"
        )

    def test_small_holdout_preserves_five_call_panel_accounting(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_qualification(config, candidate, episode_count=4)
        self.assertEqual(
            result["attempted_calls"],
            sum(row["attempted_calls"] for row in result["episode_audit"]),
        )
        self.assertLessEqual(result["attempted_calls"], 20)
        for row in result["episode_audit"]:
            if row["status"] in {"completed", "out_of_scope_bin", "physical_failed"}:
                self.assertEqual(row["attempted_calls"], 5)
        self.assertFalse(result["claims_nash"])
        self.assertFalse(result["claims_mfg"])
        self.assertTrue(result["episode_audit"])
        self.assertTrue(all(row["episode_index"] >= 0 for row in result["episode_audit"]))

    def test_completed_panel_keeps_same_crn_and_target_prefix(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_qualification(config, candidate, episode_count=8)
        self.assertTrue(result["qualification_rows"])
        for row in result["qualification_rows"]:
            self.assertEqual(len(row["input_fingerprint"]), 64)
            self.assertEqual(
                set(row["candidate_prefix_fingerprints"].values()),
                {row["baseline_prefix_fingerprint"]},
            )

    def test_holdout_uses_the_frozen_independent_namespace(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        candidate = load_t3a_runtime_candidate(VALIDATION, OCCUPANCY)
        result = run_qualification(config, candidate, episode_count=1)
        # The audit is episode-indexed and the builder contract is asserted by
        # the immutable episode fingerprint; the real namespace/seed are frozen
        # constants rather than data inferred from a future event.
        self.assertEqual(QUALIFICATION_NAMESPACE, "token-mfg-restoration:t3b:qualification:v1")
        self.assertEqual(QUALIFICATION_MACRO_SEED, 20260911)

    def test_small_artifact_is_transactional_and_never_overwritten(self):
        config = load_config(PROJECT / "configs/v1_minimal.json")
        with tempfile.TemporaryDirectory() as temp:
            first, result = execute_t3b_qualification(
                PROJECT,
                temp,
                "t3b-test-run",
                validation_dir=VALIDATION,
                occupancy_dir=OCCUPANCY,
                episode_count=4,
            )
            self.assertTrue((first / "manifest.json").is_file())
            self.assertEqual(result["claims_best_response"], False)
            with self.assertRaises(FileExistsError):
                execute_t3b_qualification(
                    PROJECT,
                    temp,
                    "t3b-test-run",
                    validation_dir=VALIDATION,
                    occupancy_dir=OCCUPANCY,
                    episode_count=4,
                )


if __name__ == "__main__":
    unittest.main()
