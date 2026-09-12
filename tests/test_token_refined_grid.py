import unittest


class TokenRefinedGridTests(unittest.TestCase):
    def observation(self, queue_a, queue_b, running=(), *, age=30.0, balance=2.0):
        from mfg_hedge.common_state import Phase
        from mfg_hedge.domain import TokenClass
        from mfg_hedge.token_online import TokenObservation

        return TokenObservation(
            token_id=0,
            token_class=TokenClass.REGULAR,
            arrival_time=100.0 + age,
            phase=Phase.DEGRADED,
            phase_age=age,
            primary_replica=0,
            queue_snapshot=(tuple(queue_a), tuple(queue_b)),
            running_attempts=tuple(running),
            observed_history=("D",),
            reservation_window=(125.0, 150.0),
            reservation_cap=2.8125,
            reservation_balance=balance,
            public_price=0.0,
        )

    def test_refined_bin_separates_primary_and_backup_load_direction(self):
        from mfg_hedge.token_refined_grid import refined_bin_id

        primary_busy = self.observation(((1, 0), (2, 0), (3, 0)), ())
        backup_busy = self.observation((), ((1, 2), (2, 2), (3, 2)))
        self.assertNotEqual(refined_bin_id(primary_busy), refined_bin_id(backup_busy))
        self.assertIn("primary=[3,+inf)", refined_bin_id(primary_busy))
        self.assertIn("backup=[3,+inf)", refined_bin_id(backup_busy))

    def test_selector_threshold_is_deterministic_and_inside_assigned_stratum(self):
        from mfg_hedge.token_refined_grid import RandomThresholdSelector

        first = RandomThresholdSelector.for_episode(5)
        second = RandomThresholdSelector.for_episode(5)
        other = RandomThresholdSelector.for_episode(9)
        self.assertEqual(first, second)
        self.assertEqual(first.stratum, 1)
        self.assertGreaterEqual(first.threshold, 25.0)
        self.assertLess(first.threshold, 50.0)
        self.assertNotEqual(first.threshold, other.threshold)

    def test_selector_ignores_eligible_tokens_before_predeclared_threshold(self):
        from mfg_hedge.token_refined_grid import RandomThresholdSelector

        selector = RandomThresholdSelector(0, 20.0)
        self.assertFalse(selector.select(self.observation((), (), age=19.9)))
        self.assertTrue(selector.select(self.observation((), (), age=20.0)))

    def test_fixed_grid_reuses_rows_and_prices_incremental_work(self):
        from mfg_hedge.token_refined_grid import summarize_fixed_grid

        rows = []
        for episode in range(8):
            rows.extend((
                {"episode_index": episode, "bin_id": "b", "action": "N", "applied_action": "N", "unpriced_cost": 2.0, "incremental_work": 0.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
                {"episode_index": episode, "bin_id": "b", "action": "D", "applied_action": "N", "unpriced_cost": 1.0, "incremental_work": 2.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
                {"episode_index": episode, "bin_id": "b", "action": "I", "applied_action": "I", "unpriced_cost": 1.5, "incremental_work": 1.0, "baseline_action": "N", "baseline_unpriced_cost": 2.0},
            ))
        result = summarize_fixed_grid(rows, betas=(1.0,), prices=(0.0, 1.0), panel_floor=8)
        self.assertEqual(result["physical_panel_count"], 8)
        self.assertEqual(len(result["cells"]), 2)
        zero, priced = result["cells"]
        self.assertGreater(zero["action_probabilities"]["D"], zero["action_probabilities"]["N"])
        self.assertGreater(priced["action_probabilities"]["N"], priced["action_probabilities"]["D"])
        self.assertEqual(priced["price"], 1.0)
        self.assertEqual(zero["requested_hedge_probability"], zero["action_probabilities"]["D"] + zero["action_probabilities"]["I"])
        self.assertEqual(zero["expected_applied_hedge_probability"], zero["action_probabilities"]["I"])
        self.assertGreater(zero["requested_hedge_probability"], zero["expected_applied_hedge_probability"])

    def test_result_claims_are_false_and_call_limit_is_bounded(self):
        from mfg_hedge.token_refined_grid import result_envelope

        result = result_envelope([], [], attempted_calls=0)
        self.assertEqual(result["call_limit"], 5120)
        self.assertEqual(result["population_policy_id"], "pi_NIIN_v1")
        self.assertFalse(result["claims_best_response"])
        self.assertFalse(result["claims_regret"])
        self.assertFalse(result["claims_nash"])
        self.assertFalse(result["claims_mfg"])


if __name__ == "__main__":
    unittest.main()
