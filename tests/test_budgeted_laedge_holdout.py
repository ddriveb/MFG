import tempfile
import unittest
from pathlib import Path

from mfg_hedge.budgeted_laedge_holdout import (
    BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS,
    BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES,
    BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
    BUDGETED_LAEDGE_HOLDOUT_EPISODES,
    BUDGETED_LAEDGE_HOLDOUT_NAMESPACE,
    BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE,
    BudgetedLaedgeHoldoutError,
    confirmed_v2_rates,
    paired_cluster_statistics,
    validate_confirmed_v2_development,
    write_budgeted_laedge_holdout_artifact,
)


class BudgetedLaedgeHoldoutTests(unittest.TestCase):
    def test_frozen_holdout_shape_and_rates(self):
        self.assertEqual(BUDGETED_LAEDGE_HOLDOUT_EPISODES, 1024)
        self.assertEqual(len(BUDGETED_LAEDGE_HOLDOUT_ARM_KEYS), 11)
        self.assertEqual(BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET, 11264)
        self.assertEqual(BUDGETED_LAEDGE_HOLDOUT_BOOTSTRAP_REPLICATES, 4096)
        self.assertIn(":v2:", BUDGETED_LAEDGE_HOLDOUT_NAMESPACE)
        self.assertIn(":preflight", BUDGETED_LAEDGE_HOLDOUT_PREFLIGHT_NAMESPACE)
        self.assertEqual(confirmed_v2_rates()["0.18"], 4.0)

    def test_development_fingerprint_guard_fails_before_execution(self):
        with self.assertRaises(BudgetedLaedgeHoldoutError):
            validate_confirmed_v2_development({"status": "calibration_failed"})

    def test_mapping_fingerprint_and_protocol_are_required(self):
        development = {
            "status": "confirmed",
            "mapping": {"targets": {}},
            "mapping_fingerprint": "wrong",
            "protocol": {},
            "protocol_fingerprint": "wrong",
            "source_bundle": {},
        }
        with self.assertRaises(BudgetedLaedgeHoldoutError):
            validate_confirmed_v2_development(development)

    def test_paired_bootstrap_is_deterministic_and_reports_required_fields(self):
        result = paired_cluster_statistics(
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            metric_name="total_work",
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["absolute_difference"], 2.0)
        self.assertIn("relative_difference", result)
        self.assertIn("paired_se", result)
        self.assertEqual(result["ci95"], paired_cluster_statistics(
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            metric_name="total_work",
        )["ci95"])

    def test_nonfinite_bootstrap_input_fails_closed(self):
        with self.assertRaises(BudgetedLaedgeHoldoutError):
            paired_cluster_statistics([1.0, float("nan")], [2.0, 3.0], metric_name="x")

    def test_artifact_is_transactional_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "summary.json": {"status": "physical_failed"},
                "paired_statistics.json": {"status": "unavailable"},
                "episode_rows.jsonl": [{"episode_index": 0}],
                "manifest.json": {"schema_id": "test"},
                "protocol.json": {"schema_id": "test"},
            }
            path = write_budgeted_laedge_holdout_artifact(root, "holdout-r1", files)
            self.assertTrue((path / "summary.json").exists())
            with self.assertRaises(FileExistsError):
                write_budgeted_laedge_holdout_artifact(root, "holdout-r1", files)


if __name__ == "__main__":
    unittest.main()
