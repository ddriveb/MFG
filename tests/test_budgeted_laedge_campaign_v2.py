import unittest

from mfg_hedge.budgeted_laedge_campaign import (
    BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET,
    BUDGETED_LAEDGE_DELTA_TARGETS,
    BUDGETED_LAEDGE_V2_CALIBRATION_RATES,
    BUDGETED_LAEDGE_V2_DEVELOPMENT_MACRO_SEED,
    BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE,
    BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED,
    BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE,
    BudgetedLaedgeCampaignError,
    assess_confirmation,
    development_call_plan_v2,
    protocol_payload_v2,
    select_budget_rate_mapping_v2,
)


class BudgetedLaedgeCampaignV2Tests(unittest.TestCase):
    def test_v2_replaces_only_the_upper_endpoint_and_keeps_budget(self):
        self.assertEqual(
            BUDGETED_LAEDGE_V2_CALIBRATION_RATES,
            (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 4.0),
        )
        self.assertEqual(development_call_plan_v2(), {
            "calibration": 768,
            "confirmation_budgeted": 1344,
            "confirmation_fixed": 192,
            "niin_reference": 256,
            "unconstrained_reference": 256,
        })
        self.assertEqual(sum(development_call_plan_v2().values()), 2816)
        self.assertEqual(BUDGETED_LAEDGE_DEVELOPMENT_CALL_BUDGET, 2816)

    def test_v2_has_fresh_identities_and_no_v1_namespace(self):
        self.assertEqual(BUDGETED_LAEDGE_V2_DEVELOPMENT_MACRO_SEED, 20260912)
        self.assertEqual(BUDGETED_LAEDGE_V2_HOLDOUT_MACRO_SEED, 20260913)
        self.assertIn(":v2:", BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE)
        self.assertIn(":v2:", BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE)
        self.assertNotIn(":v1:", BUDGETED_LAEDGE_V2_DEVELOPMENT_NAMESPACE)
        self.assertNotIn(":v1:", BUDGETED_LAEDGE_V2_HOLDOUT_NAMESPACE)
        payload = protocol_payload_v2()
        self.assertEqual(payload["calibration_rates"][-1], 4.0)
        self.assertEqual(payload["development_macro_seed"], 20260912)
        self.assertNotIn("budgeted-laedge:v1", payload["development_namespace"])

    def test_v2_rejects_r1_rate_curve_concatenation(self):
        rows = [
            {"rate": rate, "mean_total_work": 100.0 + index}
            for index, rate in enumerate(BUDGETED_LAEDGE_V2_CALIBRATION_RATES)
        ]
        rows[-1] = {"rate": 1.0, "mean_total_work": 120.0}
        with self.assertRaises(BudgetedLaedgeCampaignError):
            select_budget_rate_mapping_v2(
                rate_results=rows,
                niin_mean_total_work=100.0,
                unconstrained_mean_total_work=119.9,
            )

    def test_v2_marks_delta_18_as_saturation_limited(self):
        rows = [
            {"rate": rate, "mean_total_work": 100.0 + 1.5 * index}
            for index, rate in enumerate(BUDGETED_LAEDGE_V2_CALIBRATION_RATES)
        ]
        rows[-1] = {"rate": 4.0, "mean_total_work": 117.8}
        mapping = select_budget_rate_mapping_v2(
            rate_results=rows,
            niin_mean_total_work=100.0,
            unconstrained_mean_total_work=117.8,
        )
        target = mapping["targets"][str(BUDGETED_LAEDGE_DELTA_TARGETS[-1])]
        self.assertTrue(target["saturation_limited"])
        self.assertEqual(target["planned_budget_rate"], 4.0)
        self.assertAlmostEqual(target["target_delta"], 0.18)

    def test_v2_still_rejects_upper_coverage_shortfall(self):
        rows = [
            {"rate": rate, "mean_total_work": 100.0 + index}
            for index, rate in enumerate(BUDGETED_LAEDGE_V2_CALIBRATION_RATES)
        ]
        rows[-1] = {"rate": 4.0, "mean_total_work": 117.0}
        with self.assertRaises(BudgetedLaedgeCampaignError):
            select_budget_rate_mapping_v2(
                rate_results=rows,
                niin_mean_total_work=100.0,
                unconstrained_mean_total_work=117.8,
            )

    def test_negative_realized_delta_is_valid_underfill(self):
        result = assess_confirmation(target_delta=0.0, achieved_delta=-0.011)
        self.assertEqual(result["status"], "target_underfilled")
        self.assertAlmostEqual(result["underfill"], 0.011)


if __name__ == "__main__":
    unittest.main()
