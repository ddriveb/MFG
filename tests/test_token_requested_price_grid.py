import unittest


class TokenRequestedPriceGridTests(unittest.TestCase):
    def physical_grid(self):
        from mfg_hedge.token_refined_grid import summarize_fixed_grid

        rows = []
        for episode in range(8):
            rows.extend((
                {"episode_index": episode, "bin_id": "b", "action": "N", "applied_action": "N", "unpriced_cost": 2.0, "incremental_work": 0.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
                {"episode_index": episode, "bin_id": "b", "action": "D", "applied_action": "N", "unpriced_cost": 1.0, "incremental_work": 0.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
                {"episode_index": episode, "bin_id": "b", "action": "I", "applied_action": "I", "unpriced_cost": 1.5, "incremental_work": 1.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
            ))
        return summarize_fixed_grid(rows, betas=(1.0,), prices=(0.0,), panel_floor=8)

    def test_price_zero_exactly_matches_parent_grid(self):
        from mfg_hedge.token_requested_price_grid import summarize_requested_price_grid

        parent = self.physical_grid()
        child = summarize_requested_price_grid(parent, betas=(1.0,), prices=(0.0,))
        expected = parent["cells"][0]
        actual = child["cells"][0]
        self.assertEqual(actual["action_probabilities"], expected["action_probabilities"])
        self.assertEqual(actual["unpriced_expected_cost"], expected["unpriced_expected_cost"])
        self.assertEqual(actual["niin_unpriced_cost"], expected["niin_unpriced_cost"])

    def test_denied_request_still_pays_requested_unit(self):
        from mfg_hedge.token_requested_price_grid import summarize_requested_price_grid

        child = summarize_requested_price_grid(self.physical_grid(), betas=(1.0,), prices=(1.0,))
        stats = child["bin_stats"]["b"]["actions"]
        self.assertEqual(stats["N"]["requested_reservation_units"], 0.0)
        self.assertEqual(stats["D"]["requested_reservation_units"], 1.0)
        self.assertEqual(stats["D"]["applied_hedge_fraction"], 0.0)
        self.assertEqual(stats["I"]["requested_reservation_units"], 1.0)

    def test_requested_probability_falls_as_price_rises(self):
        from mfg_hedge.token_requested_price_grid import summarize_requested_price_grid

        child = summarize_requested_price_grid(
            self.physical_grid(), betas=(4.0,), prices=(0.0, 1.0, 4.0)
        )
        probabilities = [row["requested_hedge_probability"] for row in child["cells"]]
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])

    def test_result_is_zero_call_diagnostic_with_false_claims(self):
        from mfg_hedge.token_requested_price_grid import result_envelope

        result = result_envelope({"cells": []}, parent_sha256="a" * 64)
        self.assertEqual(result["scheduler_calls"], 0)
        self.assertEqual(result["cost_model_id"], "token_runtime_requested_reservation_price_v1")
        self.assertFalse(result["claims_best_response"])
        self.assertFalse(result["claims_regret"])
        self.assertFalse(result["claims_nash"])
        self.assertFalse(result["claims_mfg"])


if __name__ == "__main__":
    unittest.main()
