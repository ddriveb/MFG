import tempfile
from pathlib import Path
import unittest

from mfg_hedge.budgeted_laedge_campaign import (
    BUDGETED_LAEDGE_CALIBRATION_EPISODES,
    BUDGETED_LAEDGE_CALIBRATION_RATES,
    BUDGETED_LAEDGE_CONFIRMATION_EPISODES,
    BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET,
    BUDGETED_LAEDGE_DELTA_TARGETS,
    BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET,
    BUDGETED_LAEDGE_HOLDOUT_EPISODES,
    BUDGETED_LAEDGE_ARM_KEYS,
    BudgetedLaedgeCampaignError,
    assess_confirmation,
    development_call_plan,
    select_budget_rate_mapping,
    run_budgeted_laedge_holdout,
    write_budgeted_laedge_artifact,
)


class BudgetedLaedgeCampaignContractTests(unittest.TestCase):
    def test_frozen_split_and_call_arithmetic(self):
        plan = development_call_plan()
        self.assertEqual(BUDGETED_LAEDGE_CALIBRATION_EPISODES, 64)
        self.assertEqual(BUDGETED_LAEDGE_CONFIRMATION_EPISODES, 192)
        self.assertEqual(BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET, 2816)
        self.assertEqual(BUDGETED_LAEDGE_HOLDOUT_EPISODES, 1024)
        self.assertEqual(BUDGETED_LAEDGE_HOLDOUT_CALL_BUDGET, 11264)
        self.assertEqual(plan["calibration"], 64 * 12)
        self.assertEqual(plan["confirmation_budgeted"], 192 * 7)
        self.assertEqual(plan["confirmation_fixed"], 192)
        self.assertEqual(plan["niin_reference"], 256)
        self.assertEqual(plan["unconstrained_reference"], 256)
        self.assertEqual(sum(plan.values()), 2816)

    def test_rate_grid_and_delta_targets_are_canonical(self):
        self.assertEqual(
            BUDGETED_LAEDGE_CALIBRATION_RATES,
            (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 1.0),
        )
        self.assertEqual(BUDGETED_LAEDGE_DELTA_TARGETS, (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18))
        self.assertEqual(len(BUDGETED_LAEDGE_ARM_KEYS), 11)

    def test_mapping_uses_interpolated_largest_rate_without_overfill(self):
        mapping = select_budget_rate_mapping(
            rate_results=(
                {"rate": 0.0, "mean_total_work": 100.0},
                {"rate": 0.01, "mean_total_work": 100.0},
                {"rate": 0.02, "mean_total_work": 100.5},
                {"rate": 0.04, "mean_total_work": 100.8},
                {"rate": 0.06, "mean_total_work": 100.9},
                {"rate": 0.08, "mean_total_work": 101.0},
                {"rate": 0.12, "mean_total_work": 105.0},
                {"rate": 0.18, "mean_total_work": 105.0},
                {"rate": 0.25, "mean_total_work": 105.0},
                {"rate": 0.35, "mean_total_work": 105.0},
                {"rate": 0.5, "mean_total_work": 105.0},
                {"rate": 1.0, "mean_total_work": 105.0},
            ),
            niin_mean_total_work=100.0,
            unconstrained_mean_total_work=105.0,
        )
        row = mapping["targets"]["0.03"]
        self.assertEqual(row["selection_method"], "linear_interpolation")
        self.assertAlmostEqual(row["planned_budget_rate"], 0.1)
        self.assertEqual(row["bracketing_rates"], [0.08, 0.12])

    def test_mapping_rejects_nonmonotone_curve(self):
        with self.assertRaises(BudgetedLaedgeCampaignError):
            select_budget_rate_mapping(
                rate_results=(
                    {"rate": 0.0, "mean_total_work": 100.0},
                    {"rate": 0.01, "mean_total_work": 99.0},
                    *(
                        {"rate": rate, "mean_total_work": 105.0}
                        for rate in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 1.0)
                    ),
                ),
                niin_mean_total_work=100.0,
                unconstrained_mean_total_work=105.0,
                targets=(0.01,),
            )

    def test_mapping_rejects_insufficient_upper_endpoint_coverage(self):
        with self.assertRaises(BudgetedLaedgeCampaignError):
            select_budget_rate_mapping(
                rate_results=(
                    {"rate": 0.0, "mean_total_work": 100.0},
                    *(
                        {"rate": rate, "mean_total_work": 100.1}
                        for rate in (0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 1.0)
                    ),
                ),
                niin_mean_total_work=100.0,
                unconstrained_mean_total_work=105.0,
                targets=(0.01,),
            )

    def test_confirmation_upper_tolerance_and_underfill_are_distinct(self):
        passed = assess_confirmation(
            target_delta=0.03,
            achieved_delta=0.0324,
        )
        self.assertEqual(passed["status"], "confirmed")
        underfilled = assess_confirmation(
            target_delta=0.03,
            achieved_delta=0.018,
        )
        self.assertEqual(underfilled["status"], "target_underfilled")
        with self.assertRaises(BudgetedLaedgeCampaignError):
            assess_confirmation(target_delta=0.03, achieved_delta=0.0326)

    def test_holdout_refuses_unconfirmed_development_before_builder(self):
        called = []

        def builder(namespace, index):
            called.append((namespace, index))
            raise AssertionError("holdout builder must not be called")

        with self.assertRaises(BudgetedLaedgeCampaignError):
            run_budgeted_laedge_holdout(
                object(),
                {"status": "calibration_failed"},
                episode_builder=builder,
                episode_count=1,
            )
        self.assertEqual(called, [])

    def test_artifact_write_is_transactional_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "summary.json": {"status": "calibration_failed"},
                "episode_rows.jsonl": [{"episode_index": 0}],
                "manifest.json": {"schema_id": "test"},
                "protocol.json": {"schema_id": "test"},
            }
            written = write_budgeted_laedge_artifact(root, "dev-r1", files)
            self.assertTrue((written / "summary.json").exists())
            with self.assertRaises(FileExistsError):
                write_budgeted_laedge_artifact(root, "dev-r1", files)


if __name__ == "__main__":
    unittest.main()
