import unittest


class _RecordingNormal:
    def __init__(self):
        self.prices = []

    def choose(self, observation, policy_key):
        from mfg_hedge.domain import ProtectionAction

        self.prices.append(observation.public_price)
        return ProtectionAction.NORMAL


class TokenPriceFeedbackTests(unittest.TestCase):
    def config(self):
        from mfg_hedge.config import load_config

        return load_config("configs/v1_minimal.json")

    def observation(self, *, urgent=False, age=10.0, price=0.0):
        from mfg_hedge.common_state import Phase
        from mfg_hedge.domain import TokenClass
        from mfg_hedge.token_online import TokenObservation

        return TokenObservation(
            token_id=0,
            token_class=TokenClass.URGENT if urgent else TokenClass.REGULAR,
            arrival_time=100.0 + age,
            phase=Phase.DEGRADED,
            phase_age=age,
            primary_replica=0,
            queue_snapshot=((), ()),
            running_attempts=(),
            observed_history=("D",),
            reservation_window=(100.0, 125.0),
            reservation_cap=2.8125,
            reservation_balance=2.0,
            public_price=price,
        )

    def test_public_price_reaches_every_online_observation(self):
        from mfg_hedge.attribution_episode import (
            ATTRIBUTION_V1_PROTOCOL,
            generate_episode_trace,
        )
        from mfg_hedge.token_online import simulate_episode_online

        episode = generate_episode_trace(
            self.config(), "price-observation-test", 91, 0, ATTRIBUTION_V1_PROTOCOL
        )
        source = _RecordingNormal()
        simulate_episode_online(episode, source, public_price=1.25)
        self.assertTrue(source.prices)
        self.assertEqual(set(source.prices), {1.25})

    def test_feedback_uses_requested_work_before_projection(self):
        from mfg_hedge.token_price_feedback import update_public_price

        row = update_public_price(0.5, requested_work=15.0, capacity=10.0, alpha=0.25)
        self.assertEqual(row["demand_pressure"], 0.5)
        self.assertEqual(row["next_price"], 0.625)
        self.assertEqual(row["price_residual"], 0.125)

    def test_feedback_price_is_nonnegative(self):
        from mfg_hedge.token_price_feedback import update_public_price

        row = update_public_price(0.1, requested_work=0.0, capacity=10.0, alpha=0.25)
        self.assertEqual(row["next_price"], 0.0)

    def test_requested_price_charges_denied_d_or_i_but_not_n(self):
        from mfg_hedge.domain import ProtectionAction
        from mfg_hedge.token_price_feedback import requested_price_cost

        self.assertEqual(requested_price_cost(2.0, ProtectionAction.NORMAL, 3.0), 2.0)
        self.assertEqual(requested_price_cost(2.0, ProtectionAction.DELAYED_HEDGE, 3.0), 5.0)
        self.assertEqual(requested_price_cost(2.0, ProtectionAction.IMMEDIATE_HEDGE, 3.0), 5.0)

    def test_population_fallback_is_exact_niin(self):
        from mfg_hedge.domain import ProtectionAction
        from mfg_hedge.token_price_feedback import RefinedPopulationPolicy

        policy = RefinedPopulationPolicy()
        expected = {
            (False, 10.0): ProtectionAction.NORMAL,
            (True, 10.0): ProtectionAction.IMMEDIATE_HEDGE,
            (False, 60.0): ProtectionAction.IMMEDIATE_HEDGE,
            (True, 60.0): ProtectionAction.NORMAL,
        }
        for (urgent, age), action in expected.items():
            with self.subTest(urgent=urgent, age=age):
                self.assertEqual(policy.base_action(self.observation(urgent=urgent, age=age)), action)

    def test_small_feedback_run_is_bounded_and_has_no_equilibrium_claim(self):
        from mfg_hedge.token_price_feedback import run_price_feedback

        result = run_price_feedback(
            self.config(), episode_count=8, max_rounds=2, panel_floor=1
        )
        self.assertLessEqual(result["attempted_calls"], 80)
        self.assertEqual(len(result["iterations"]), 2)
        self.assertTrue(all(row["public_price"] >= 0 for row in result["iterations"]))
        self.assertFalse(result["claims_best_response"])
        self.assertFalse(result["claims_regret"])
        self.assertFalse(result["claims_nash"])
        self.assertFalse(result["claims_mfg"])

    def test_small_requested_price_feedback_uses_distinct_model(self):
        from mfg_hedge.token_price_feedback import run_requested_price_feedback

        result = run_requested_price_feedback(
            self.config(), episode_count=8, max_rounds=1, panel_floor=1
        )
        self.assertEqual(result["price_basis"], "requested_reserved_work")
        self.assertEqual(result["cost_model_id"], "token_runtime_requested_reservation_price_v1")
        self.assertLessEqual(result["attempted_calls"], 40)


if __name__ == "__main__":
    unittest.main()
