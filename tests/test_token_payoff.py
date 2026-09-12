"""Red tests for ADR-0021 Token payoff records and strict settlement."""

import unittest

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.token_online import ReservationLedger, simulate_episode_online
from mfg_hedge.transient_control import ReservationParameters
from mfg_hedge.token_payoff import (
    ADMITTED_RESERVED_WORK,
    INCREMENTAL_EXECUTED_WORK,
    ExternalQuote,
    TokenExtendedReservationParameters,
    TokenInitialPriceParameters,
    TokenRuntimeADR0009Parameters,
    build_token_cost_parameters,
    score_token_payoff,
)
from tests.token_t2_support import make_episode, source_factory


def run_episode(episode, action=A.NORMAL, *, delay=None, parameters=None):
    return simulate_episode_online(
        episode,
        source_factory(action)(),
        parameters,
        hedge_delay=delay,
    )


class TokenPayoffContractTests(unittest.TestCase):
    def test_hand_computable_n_without_replay(self):
        run = run_episode(
            make_episode(service0=(1.0,), timeline=CommonStateTimeline(100.0, 200.0, 220.0))
        )
        initial = score_token_payoff(
            run, 0, TokenInitialPriceParameters(), ExternalQuote(0.0, INCREMENTAL_EXECUTED_WORK)
        )
        runtime = score_token_payoff(
            run, 0, TokenRuntimeADR0009Parameters(), ExternalQuote(0.0, INCREMENTAL_EXECUTED_WORK)
        )
        extended = score_token_payoff(
            run,
            0,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.0, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual((initial.latency, initial.w_primary, initial.w_replay), (1.0, 1.0, 0.0))
        self.assertEqual((initial.total, runtime.total, extended.total), (1.0, 1.0, 2.0))

    def test_immediate_wins_primary_running_loser(self):
        episode = make_episode(
            service0=(2.0,),
            service1=(1.0,),
            hedge1=(1.0,),
        )
        run = run_episode(episode, A.IMMEDIATE_HEDGE)
        initial = score_token_payoff(
            run,
            0,
            TokenInitialPriceParameters(),
            ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK),
        )
        runtime = score_token_payoff(
            run,
            0,
            TokenRuntimeADR0009Parameters(),
            ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK),
        )
        extended = score_token_payoff(
            run,
            0,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual((initial.latency, initial.settlement_time), (1.0, 14.0))
        self.assertEqual((initial.w_primary, initial.w_hedge, initial.w_lose), (2.0, 1.0, 2.0))
        self.assertEqual((initial.total, runtime.total, extended.total), (1.5, 4.5, 6.5))

    def test_losing_hedge_keeps_running_and_settlement(self):
        episode = make_episode(
            service0=(1.0,),
            service1=(3.0,),
            hedge1=(3.0,),
        )
        run = run_episode(episode, A.IMMEDIATE_HEDGE)
        payoff = score_token_payoff(
            run,
            0,
            TokenRuntimeADR0009Parameters(),
            ExternalQuote(0.0, INCREMENTAL_EXECUTED_WORK),
        )
        self.assertEqual(payoff.winner_attempt_id, 0)
        self.assertEqual(payoff.w_hedge, 3.0)
        self.assertEqual(payoff.w_lose, 3.0)
        self.assertEqual(payoff.settlement_time, 13.0)

    def test_failed_primary_then_replay_retains_executed_failure_work(self):
        episode = make_episode(
            timeline=CommonStateTimeline(0.0, 12.0, 20.0),
            service0=(2.0,),
            replay1=(1.0,),
        )
        run = run_episode(episode)
        payoff = score_token_payoff(
            run,
            0,
            TokenInitialPriceParameters(),
            ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK),
        )
        self.assertEqual(payoff.replay_count, 1)
        self.assertEqual((payoff.w_primary, payoff.w_replay, payoff.w_lose), (1.0, 1.0, 1.0))
        self.assertEqual(payoff.total, 4.5)

    def test_never_started_queued_loser_has_zero_work_and_no_refund(self):
        episode = make_episode(
            arrivals=(9.0, 9.0, 10.0),
            service0=(1.0, 1.0, 1.0),
            service1=(1.0, 5.0, 1.0),
            hedge1=(1.0, 1.0, 4.0),
        )
        run = simulate_episode_online(
            episode,
            source_factory(
                chooser=lambda observation, _key: (
                    A.IMMEDIATE_HEDGE if observation.token_id == 2 else A.NORMAL
                )
            )(),
        )
        payoff = score_token_payoff(
            run,
            2,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual(payoff.w_hedge, 0.0)
        self.assertEqual(payoff.reservation_charge, 1.0)
        self.assertTrue(any(a.attempt_id == 2 and a.executed_work == 0.0 for a in payoff.attempts))

    def test_admitted_delayed_timer_void_keeps_reservation(self):
        run = run_episode(
            make_episode(service0=(1.0,)), A.DELAYED_HEDGE, delay=5.0
        )
        payoff = score_token_payoff(
            run,
            0,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual(payoff.reservation_charge, 1.0)
        self.assertEqual(payoff.timer_status, "voided")
        self.assertEqual(payoff.w_hedge, 0.0)
        self.assertEqual(payoff.price_cost, 0.5)

    def test_refused_d_and_i_score_applied_normal_and_replay(self):
        episode = make_episode(
            arrivals=(10.0, 10.0, 10.0, 10.0, 10.0),
            timeline=CommonStateTimeline(0.0, 11.0, 20.0),
            service0=(1.0, 1.0, 1.0, 1.0, 2.0),
        )
        params = ReservationParameters(mean_requirement=1.0)
        run = simulate_episode_online(
            episode,
            source_factory(A.DELAYED_HEDGE)(),
            params,
            hedge_delay=5.0,
        )
        payoff = score_token_payoff(
            run,
            4,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual(payoff.applied_action, A.NORMAL)
        self.assertEqual(payoff.reservation_charge, 0.0)
        self.assertEqual(payoff.replay_count, 1)

    def test_nonzero_quote_and_three_model_provenance_are_separate(self):
        run = run_episode(make_episode(service0=(1.0,)))
        initial = score_token_payoff(
            run, 0, TokenInitialPriceParameters(), ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK)
        )
        zero_runtime = score_token_payoff(
            run,
            0,
            TokenRuntimeADR0009Parameters(c_inc=0.0, c_waste=0.0),
            ExternalQuote(0.5, INCREMENTAL_EXECUTED_WORK),
        )
        extended = score_token_payoff(
            run,
            0,
            TokenExtendedReservationParameters(),
            ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
        )
        self.assertEqual(initial.total, zero_runtime.total)
        self.assertNotEqual(initial.model_id, zero_runtime.model_id)
        self.assertNotEqual(initial.provenance_fingerprint, zero_runtime.provenance_fingerprint)
        self.assertEqual(initial.price_basis, INCREMENTAL_EXECUTED_WORK)
        self.assertEqual(extended.price_basis, ADMITTED_RESERVED_WORK)

    def test_strict_three_model_parameters_and_withdrawn_id(self):
        with self.assertRaises(ValueError):
            TokenRuntimeADR0009Parameters(c_inc=float("nan"))
        with self.assertRaises(ValueError):
            TokenExtendedReservationParameters(deadline_regular=0.0)
        with self.assertRaises(ValueError):
            build_token_cost_parameters("token_original_runtime_v1")
        with self.assertRaises(ValueError):
            score_token_payoff(
                run_episode(make_episode()),
                0,
                TokenInitialPriceParameters(),
                ExternalQuote(0.5, ADMITTED_RESERVED_WORK),
            )

    def test_incomplete_or_corrupt_run_is_rejected(self):
        run = run_episode(make_episode())
        corrupt = object.__new__(type(run))
        object.__setattr__(corrupt, "episode", run.episode)
        object.__setattr__(corrupt, "simulation", run.simulation)
        object.__setattr__(corrupt, "decisions", run.decisions[:-1])
        with self.assertRaises(ValueError):
            score_token_payoff(
                corrupt,
                0,
                TokenInitialPriceParameters(),
                ExternalQuote(0.0, INCREMENTAL_EXECUTED_WORK),
            )


if __name__ == "__main__":
    unittest.main()
