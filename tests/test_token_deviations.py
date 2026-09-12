"""Red tests for ADR-0021 finite requested-action deviations."""

import unittest

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction as A
from mfg_hedge.token_deviations import evaluate_token_pathwise_deviations
from mfg_hedge.token_payoff import (
    ADMITTED_RESERVED_WORK,
    ExternalQuote,
    TokenExtendedReservationParameters,
    TokenInitialPriceParameters,
    INCREMENTAL_EXECUTED_WORK,
)
from tests.token_t2_support import make_episode, source_factory


class TokenDeviationContractTests(unittest.TestCase):
    def test_nonzero_quote_reaches_policy_and_scorer_in_every_branch(self):
        calls = []

        def chooser(observation, policy_key):
            calls.append((observation.public_price, policy_key))
            return A.NORMAL

        result = evaluate_token_pathwise_deviations(
            make_episode(),
            lambda: source_factory(chooser=chooser)(),
            0,
            parameters=TokenInitialPriceParameters(),
            quote=ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK),
            hedge_delay=2.0,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual({price for price, _ in calls}, {0.5})
        self.assertEqual(result.baseline_payoff.price_cost, 0.0)

    def test_same_action_intervention_reproduces_baseline_exactly(self):
        result = evaluate_token_pathwise_deviations(
            make_episode(),
            source_factory(A.NORMAL),
            0,
            parameters=TokenInitialPriceParameters(),
            hedge_delay=2.0,
        )
        row = result.row_for(A.NORMAL)
        self.assertEqual(result.attempted_calls, 4)
        self.assertEqual(row.delta_cost, 0.0)
        self.assertEqual(row.pathwise_gain, 0.0)
        self.assertEqual(row.baseline_run, row.candidate_run)

    def test_other_online_token_reacts_to_changed_queue(self):
        def chooser(observation, policy_key):
            if observation.token_id == 0:
                return A.NORMAL
            return A.IMMEDIATE_HEDGE if observation.queue_snapshot[1] else A.NORMAL

        result = evaluate_token_pathwise_deviations(
            make_episode(arrivals=(10.0, 10.0), service1=(5.0, 5.0)),
            lambda: source_factory(chooser=chooser)(),
            0,
            candidates=(A.IMMEDIATE_HEDGE,),
            parameters=TokenInitialPriceParameters(),
            hedge_delay=2.0,
        )
        row = result.row_for(A.IMMEDIATE_HEDGE)
        self.assertNotEqual(row.baseline_other_actions, row.candidate_other_actions)

    def test_stateful_target_choose_called_once_before_override(self):
        instances = []

        class Stateful:
            def __init__(self):
                self.count = 0
                self.keys = []

            def choose(self, observation, policy_key):
                self.keys.append(policy_key)
                action = A.IMMEDIATE_HEDGE if self.count == 0 else A.NORMAL
                self.count += 1
                return action

        def factory():
            source = Stateful()
            instances.append(source)
            return source

        result = evaluate_token_pathwise_deviations(
            make_episode(arrivals=(10.0, 11.0)),
            factory,
            0,
            candidates=(A.NORMAL,),
            parameters=TokenInitialPriceParameters(),
            hedge_delay=2.0,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual([len(source.keys) for source in instances], [2, 2])
        self.assertEqual(result.baseline_run.decisions[0].requested, A.IMMEDIATE_HEDGE)
        self.assertEqual(result.row_for(A.NORMAL).candidate_requested, A.NORMAL)

    def test_candidate_order_is_canonical_and_factory_state_is_isolated(self):
        instances = []
        result = evaluate_token_pathwise_deviations(
            make_episode(),
            lambda: source_factory(A.NORMAL, sink=instances)(),
            0,
            candidates=(A.IMMEDIATE_HEDGE, A.NORMAL, A.DELAYED_HEDGE),
            hedge_delay=2.0,
            parameters=TokenInitialPriceParameters(),
        )
        self.assertEqual(result.candidate_actions, (A.NORMAL, A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE))
        self.assertEqual(len(instances), 4)
        self.assertTrue(all(len(source.calls) == 1 for source in instances))

    def test_same_crn_fingerprint_and_complete_call_accounting(self):
        episode = make_episode(arrivals=(10.0, 11.0))
        result = evaluate_token_pathwise_deviations(
            episode,
            source_factory(A.NORMAL),
            0,
            parameters=TokenInitialPriceParameters(),
            hedge_delay=2.0,
        )
        self.assertEqual(result.attempted_calls, 4)
        self.assertEqual(result.input_fingerprint, result.baseline_run_fingerprint)
        self.assertTrue(all(row.input_fingerprint == result.input_fingerprint for row in result.rows))

    def test_mid_batch_failure_is_partial_without_retry_or_cost_claim(self):
        calls = []

        def factory():
            calls.append(len(calls))
            if len(calls) == 2:
                raise RuntimeError("injected policy factory failure")
            return source_factory(A.NORMAL)()

        result = evaluate_token_pathwise_deviations(
            make_episode(),
            factory,
            0,
            candidates=(A.NORMAL, A.IMMEDIATE_HEDGE),
            parameters=TokenInitialPriceParameters(),
            hedge_delay=2.0,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.attempted_calls, 2)
        self.assertIsNone(result.failed_cost)
        self.assertEqual(len(calls), 2)

    def test_pathwise_only_output_and_strict_inputs(self):
        result = evaluate_token_pathwise_deviations(
            make_episode(),
            source_factory(A.NORMAL),
            0,
            parameters=TokenExtendedReservationParameters(),
            quote=ExternalQuote(0.0, ADMITTED_RESERVED_WORK),
            hedge_delay=2.0,
        )
        self.assertEqual(result.evidence_label, "finite_token_pathwise_deviation")
        self.assertFalse(result.claims_best_response)
        self.assertFalse(result.claims_regret)
        self.assertFalse(result.claims_nash)
        self.assertFalse(result.claims_mfg)
        for forbidden in ("best_action", "best_response", "regret", "epsilon_nash", "equilibrium"):
            self.assertFalse(hasattr(result, forbidden))
        with self.assertRaises(ValueError):
            evaluate_token_pathwise_deviations(
                make_episode(), source_factory(A.NORMAL), 99,
                parameters=TokenInitialPriceParameters(),
            )
        with self.assertRaises(ValueError):
            evaluate_token_pathwise_deviations(
                make_episode(), source_factory(A.NORMAL), 0,
                candidates=(A.NORMAL, A.NORMAL),
                parameters=TokenInitialPriceParameters(),
            )
        with self.assertRaises(ValueError):
            evaluate_token_pathwise_deviations(
                make_episode(), source_factory(A.NORMAL), 0,
                parameters=TokenInitialPriceParameters(),
                quote=ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
            )


if __name__ == "__main__":
    unittest.main()
