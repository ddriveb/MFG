"""Red tests for the accepted T3A observable-bin continuation oracle."""

import unittest

from mfg_hedge.attribution_episode import EpisodeProtocol
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction as A
from mfg_hedge.token_continuation import (
    BIN_SCHEMA_V1,
    EVIDENCE_LABEL,
    FROZEN_MAX_SCHEDULER_CALLS,
    MIN_COMPLETE_PANELS_PER_BIN,
    ORACLE_MODE,
    S_V1,
    TokenObservableBinOracleError,
    run_observable_bin_oracle,
)
from mfg_hedge.token_payoff import (
    TokenInitialPriceParameters,
    TokenRuntimeADR0009Parameters,
    TokenExtendedReservationParameters,
)
from tests.token_t2_support import FixedSource, make_episode, source_factory


class TokenContinuationOracleTests(unittest.TestCase):
    def _protocol(self):
        return EpisodeProtocol(CommonStateTimeline(100.0, 200.0, 220.0), 300.0)

    def _episode_factory(self, calls, *, arrivals=(101.0,)):
        protocol = self._protocol()

        def factory(episode_index, namespace, received_protocol):
            calls.append((episode_index, namespace, received_protocol))
            self.assertIsInstance(episode_index, int)
            self.assertIsInstance(namespace, str)
            self.assertIs(received_protocol, protocol)
            return make_episode(
                arrivals=arrivals,
                timeline=protocol.timeline,
                cutoff=protocol.arrival_cutoff,
                base_seed=7100 + episode_index,
            )

        return protocol, factory

    def test_episode_factory_precedes_target_observation_and_never_gets_observation(self):
        calls = []
        protocol, factory = self._episode_factory(calls)
        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=2,
            parameters=(TokenInitialPriceParameters(),),
            hedge_delay=2.0,
        )
        self.assertEqual([call[0] for call in calls], [0, 1])
        self.assertEqual(result.target_selection_rule_id, S_V1.rule_id)
        self.assertTrue(all(record.target_observation is not None for record in result.episodes))

    def test_schema_is_frozen_and_uses_half_open_boundaries(self):
        self.assertEqual(BIN_SCHEMA_V1.schema_id, "bin_schema_v1")
        self.assertEqual(
            BIN_SCHEMA_V1.numeric_boundaries["arrival_time"],
            (0.0, 100.0, 200.0, 220.0, 320.0),
        )
        self.assertEqual(
            BIN_SCHEMA_V1.numeric_boundaries["phase_age"],
            (0.0, 1.5, 3.0, 10.0),
        )
        self.assertEqual(
            BIN_SCHEMA_V1.numeric_boundaries["reservation_balance"],
            (0.0, 1.0, 2.8125),
        )

    def test_s_v1_selects_first_degraded_primary_a_only(self):
        calls = []
        protocol, factory = self._episode_factory(calls, arrivals=(101.0, 102.0))
        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=1,
            parameters=(TokenInitialPriceParameters(),),
            hedge_delay=2.0,
        )
        self.assertEqual(result.episodes[0].target_token_id, 0)

    def test_missing_target_consumes_episode_without_supplement(self):
        calls = []
        protocol, factory = self._episode_factory(calls, arrivals=(10.0,))
        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="validation",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260908,
            protocol=protocol,
            episode_count=2,
            parameters=(TokenInitialPriceParameters(),),
            hedge_delay=2.0,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([record.status for record in result.episodes], ["missing_target"] * 2)
        self.assertEqual(result.attempted_calls, 2)

    def test_same_bin_allows_different_episode_prefixes(self):
        protocol = self._protocol()
        episodes = iter(
            (
                make_episode(arrivals=(101.0,), timeline=protocol.timeline, cutoff=300.0, base_seed=7301),
                make_episode(arrivals=(101.25,), timeline=protocol.timeline, cutoff=300.0, base_seed=7302),
            )
        )

        def factory(_index, _namespace, _protocol):
            return next(episodes)

        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=2,
            parameters=(TokenInitialPriceParameters(),),
            hedge_delay=2.0,
        )
        selected = [record for record in result.episodes if record.target_token_id is not None]
        self.assertEqual(len(selected), 2)
        self.assertNotEqual(
            selected[0].observation_fingerprint,
            selected[1].observation_fingerprint,
        )
        self.assertEqual(selected[0].bin_id, selected[1].bin_id)

    def test_duplicate_episode_trace_fingerprint_fails_closed(self):
        protocol, _ = self._episode_factory([])
        episode = make_episode(
            arrivals=(101.0,),
            timeline=protocol.timeline,
            cutoff=protocol.arrival_cutoff,
            base_seed=7401,
        )

        def factory(_index, _namespace, _protocol):
            return episode

        with self.assertRaises(TokenObservableBinOracleError):
            run_observable_bin_oracle(
                factory,
                source_factory(A.NORMAL),
                split="calibration",
                namespace="token-mfg-restoration:t3a:test",
                macro_seed=20260907,
                protocol=protocol,
                episode_count=2,
                parameters=(TokenInitialPriceParameters(),),
                hedge_delay=2.0,
            )

    def test_three_models_are_separate_and_output_has_no_action_claim(self):
        protocol, factory = self._episode_factory([])
        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=2,
            parameters=(
                TokenInitialPriceParameters(),
                TokenRuntimeADR0009Parameters(),
                TokenExtendedReservationParameters(),
            ),
            hedge_delay=2.0,
        )
        self.assertEqual(result.evidence_label, EVIDENCE_LABEL)
        self.assertFalse(result.claims_best_response)
        self.assertFalse(result.claims_regret)
        self.assertFalse(result.claims_nash)
        self.assertFalse(result.claims_mfg)
        self.assertNotIn("best_action", result.__dataclass_fields__)
        self.assertEqual(
            {estimate.model_id for estimate in result.estimates},
            {
                "token_initial_price_v0",
                "token_runtime_adr0009_v1",
                "token_extended_reservation_v1",
            },
        )

    def test_failed_action_branch_fails_whole_panel_without_retry(self):
        protocol, factory = self._episode_factory([])
        factory_calls = []

        def failing_policy_factory():
            factory_calls.append(len(factory_calls))
            if len(factory_calls) == 5:
                raise RuntimeError("injected T3A branch failure")
            return FixedSource(A.NORMAL)

        result = run_observable_bin_oracle(
            factory,
            failing_policy_factory,
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=1,
            parameters=(TokenInitialPriceParameters(),),
            hedge_delay=2.0,
        )
        self.assertEqual(result.episodes[0].failed_panels, 1)
        self.assertEqual(result.attempted_calls, 5)
        self.assertEqual(len(result.estimates), 3)
        self.assertTrue(all(estimate.effective_n == 0 for estimate in result.estimates))
        self.assertTrue(all(estimate.status == "failed" for estimate in result.estimates))
        self.assertEqual(len(factory_calls), 5)

    def test_insufficient_bin_is_fail_closed_and_budget_is_strict(self):
        protocol, factory = self._episode_factory([])
        result = run_observable_bin_oracle(
            factory,
            source_factory(A.NORMAL),
            split="calibration",
            namespace="token-mfg-restoration:t3a:test",
            macro_seed=20260907,
            protocol=protocol,
            episode_count=1,
            parameters=(TokenInitialPriceParameters(),),
            minimum_complete_panels=MIN_COMPLETE_PANELS_PER_BIN,
            max_scheduler_calls=5,
            hedge_delay=2.0,
        )
        self.assertEqual(result.attempted_calls, 5)
        self.assertTrue(all(estimate.status in {"insufficient", "failed"} for estimate in result.estimates))
        with self.assertRaises(TokenObservableBinOracleError):
            run_observable_bin_oracle(
                factory,
                source_factory(A.NORMAL),
                split="calibration",
                namespace="token-mfg-restoration:t3a:test",
                macro_seed=20260907,
                protocol=protocol,
                episode_count=2,
                parameters=(TokenInitialPriceParameters(),),
                max_scheduler_calls=4,
                hedge_delay=2.0,
            )
        self.assertEqual(FROZEN_MAX_SCHEDULER_CALLS, 512)
        self.assertEqual(ORACLE_MODE, "implementation_validation_only")


if __name__ == "__main__":
    unittest.main()
