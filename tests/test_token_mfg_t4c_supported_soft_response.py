import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TokenMfgT4cTests(unittest.TestCase):
    def test_initial_and_unknown_bin_fallback_is_exact_niin(self):
        from mfg_hedge.common_state import Phase
        from mfg_hedge.domain import TokenClass
        from mfg_hedge.token_mfg_t4c_supported_soft_response import T4CPopulationPolicy
        from mfg_hedge.token_t3a_execution import policy_factory

        t4c = T4CPopulationPolicy()
        niin = policy_factory()
        cases = (
            (10.0, TokenClass.REGULAR),
            (10.0, TokenClass.URGENT),
            (60.0, TokenClass.REGULAR),
            (60.0, TokenClass.URGENT),
        )
        for phase_age, token_class in cases:
            observation = SimpleNamespace(
                phase=Phase.DEGRADED,
                primary_replica=0,
                token_class=token_class,
                phase_age=phase_age,
            )
            with self.subTest(phase_age=phase_age, token_class=token_class):
                self.assertEqual(
                    t4c.base_action(observation),
                    niin.choose(observation, "fixed-policy-key"),
                )

        for phase, replica in ((Phase.HEALTHY, 0), (Phase.DEGRADED, 1)):
            observation = SimpleNamespace(
                phase=phase,
                primary_replica=replica,
                token_class=TokenClass.URGENT,
                phase_age=60.0,
            )
            self.assertEqual(t4c.base_action(observation).value, "N")

    def test_t4c_success_status_uses_supported_soft_fixed_point(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            T4C_SUPPORTED_FIXED_POINT_STATUS,
        )

        self.assertEqual(
            T4C_SUPPORTED_FIXED_POINT_STATUS,
            "supported_soft_fixed_point",
        )

    def test_t4c_round_metrics_reuse_existing_attribution_metric_fields(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            aggregate_round_physics_metrics,
        )

        row = {
            "generated_tokens": 2,
            "latency": {
                "overall": {
                    "sample_count": 2,
                    "latency_sum": 6.0,
                    "mean_latency": 3.0,
                    "latency_p50": 2.5,
                    "latency_p95": 5.0,
                    "latency_p99": 5.5,
                    "cvar95_latency": 5.5,
                    "deadline_miss_count": 1,
                    "deadline_miss_probability": 0.5,
                    "excess_latency_sum": 1.0,
                    "mean_excess_latency": 0.5,
                }
            },
            "replay": {"replayed_tokens": 1, "rate": 0.5},
            "hedge": {
                "requested": 2,
                "launched": 1,
                "winners": 1,
                "cancelled_queued": 0,
                "completed_loser": 0,
                "failed_running": 0,
                "invalidated_queued": 0,
                "executor_suppressed": 1,
            },
            "work": {
                "nominal_primary_work": 4.0,
                "executed_primary_work": 3.0,
                "executed_hedge_work": 1.0,
                "executed_replay_work": 2.0,
                "total_executed_work": 6.0,
                "wasted_work": 1.0,
            },
            "storm": {
                "hedge_launch_count": 1,
                "post_cutoff_hedge_launch_count": 0,
                "post_cutoff_hedge_launch_work": 0.0,
                "observation_duration": 10.0,
                "peak_hedge_launch_rate": 1.0,
                "mean_hedge_launch_rate": 0.1,
                "peak_to_mean_hedge_launch_ratio": 10.0,
                "overloaded_bin_count": 2,
                "sustained_overload_duration": 2.0,
            },
        }

        metrics = aggregate_round_physics_metrics((row,))
        self.assertEqual(metrics["panel_count"], 1)
        self.assertEqual(metrics["latency"]["overall"]["sample_count"], 2)
        self.assertEqual(metrics["latency"]["overall"]["latency_sum"], 6.0)
        self.assertEqual(metrics["replay"]["replayed_tokens"], 1)
        self.assertEqual(metrics["hedge"]["launched"], 1)
        self.assertEqual(metrics["work"]["total_executed_work"], 6.0)
        self.assertEqual(metrics["work"]["wasted_work"], 1.0)
        self.assertEqual(metrics["storm"]["hedge_launch_count"], 1)

    def test_t4c_simulation_exception_is_physical_failed_and_accounts_calls(self):
        from mfg_hedge.config import load_config
        from mfg_hedge.token_mfg_t4c_supported_soft_response import run_t4c

        config = load_config(Path(__file__).parents[1] / "configs" / "v1_minimal.json")
        with patch(
            "mfg_hedge.token_online.simulate_episode_online",
            side_effect=RuntimeError("synthetic physical failure"),
        ):
            result = run_t4c(
                config,
                ("R|D|early|low|high",),
                episode_count=1,
                minimum_complete_panels=1,
                max_rounds=1,
            )

        self.assertEqual(result["status"], "physical_failed")
        self.assertEqual(result["iterations"][0]["status"], "physical_failed")
        failure = result["physical_failure"]
        self.assertEqual(failure["round"], 0)
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertEqual(failure["exception_message"], "synthetic physical failure")
        self.assertEqual(failure["attempted_calls"], 1)
        self.assertEqual(result["execution_profile"], "exploratory")
        self.assertEqual(result["call_limit_per_round"], 5)
        self.assertEqual(result["call_limit_total"], 5)

    def test_identity_library_is_compact_deterministic_and_immutable(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            T4CIdentityLibrary,
            build_identity_library,
            identity_library_fingerprint,
            validate_identity_library,
        )

        library = build_identity_library("test:t4c", 7, episode_count=3)
        self.assertIsInstance(library, T4CIdentityLibrary)
        self.assertEqual(len(library.identities), 3)
        self.assertTrue(all(not hasattr(identity, "workload") for identity in library.identities))
        self.assertEqual(validate_identity_library(library, library.fingerprint), library)
        self.assertEqual(library.fingerprint, identity_library_fingerprint(library.identities))
        with self.assertRaises(FrozenInstanceError):
            library.identities = ()

    def test_identity_reconstruction_is_repeatable_and_not_retained(self):
        from mfg_hedge.attribution_episode import (
            ATTRIBUTION_V1_PROTOCOL,
            EpisodeIdentity,
        )
        from mfg_hedge.config import load_config
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            reconstruct_episode,
            reconstructed_trace_fingerprint,
        )

        config = load_config(Path(__file__).parents[1] / "configs" / "v1_minimal.json")
        identity = EpisodeIdentity("test:t4c", 7, 0)
        first = reconstruct_episode(config, identity, ATTRIBUTION_V1_PROTOCOL)
        second = reconstruct_episode(config, identity, ATTRIBUTION_V1_PROTOCOL)
        self.assertEqual(reconstructed_trace_fingerprint(first), reconstructed_trace_fingerprint(second))
        self.assertIsNot(first, second)

    def test_support_classification_is_fail_closed_for_positive_occupancy(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            classify_bin_support,
            classify_support_rows,
        )

        self.assertEqual(classify_bin_support(0, 0), "inactive")
        self.assertEqual(classify_bin_support(21, 21), "unresolved")
        self.assertEqual(classify_bin_support(32, 32), "active")
        result = classify_support_rows(
            {"inactive": (0, 0), "rare": (21, 21), "ready": (40, 35)}
        )
        self.assertEqual(result["inactive"], "inactive")
        self.assertEqual(result["rare"], "unresolved")
        self.assertEqual(result["ready"], "active")
        self.assertEqual(result["gate_status"], "statistics_insufficient")

    def test_policy_update_changes_active_bins_and_preserves_inactive_bins(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import T4CPolicyState

        state = T4CPolicyState.from_mapping({"inactive": (0.0, 1.0, 0.0)})
        updated = state.updated(
            {"active": (0.0, 0.0, 1.0)},
            active_bins=("active",),
            fallback_by_bin={"active": (1.0, 0.0, 0.0)},
            eta=0.2,
        )
        self.assertEqual(updated.probabilities_for_bin("active", (1.0, 0.0, 0.0)), (0.8, 0.0, 0.2))
        self.assertEqual(updated.probabilities_for_bin("inactive", (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))
        with self.assertRaises(FrozenInstanceError):
            updated.probabilities_by_bin = ()

    def test_support_weighted_and_maximum_active_residuals_are_both_reported(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import support_residuals

        result = support_residuals(
            {"a": (1.0, 0.0, 0.0), "b": (1.0, 0.0, 0.0)},
            {"a": (0.8, 0.2, 0.0), "b": (0.5, 0.0, 0.5)},
            {"a": 0.75, "b": 0.25},
            active_bins=("a", "b"),
        )
        self.assertAlmostEqual(result["weighted_policy_l1"], 0.55)
        self.assertAlmostEqual(result["max_active_policy_l1"], 1.0)
        self.assertEqual(result["active_bins"], ["a", "b"])

    def test_inactive_does_not_block_but_unresolved_does(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import support_gate

        self.assertEqual(
            support_gate({"a": (0, 0), "b": (40, 32)})["status"],
            "ready",
        )
        self.assertEqual(
            support_gate({"a": (0, 0), "b": (21, 21)})["status"],
            "statistics_insufficient",
        )

    def test_t4c_call_budget_is_exact(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            T4C_CALL_LIMIT_PER_ROUND,
            T4C_CALL_LIMIT_TOTAL,
            T4C_EPISODE_SLOTS,
            T4C_MAX_ROUNDS,
        )

        self.assertEqual(T4C_EPISODE_SLOTS, 4096)
        self.assertEqual(T4C_CALL_LIMIT_PER_ROUND, 20480)
        self.assertEqual(T4C_CALL_LIMIT_TOTAL, 409600)
        self.assertEqual(T4C_CALL_LIMIT_TOTAL, T4C_MAX_ROUNDS * T4C_CALL_LIMIT_PER_ROUND)

    def test_t4c_artifact_is_transactional_and_non_overwriting(self):
        from mfg_hedge.token_mfg_t4c_supported_soft_response import write_t4c_artifact

        with tempfile.TemporaryDirectory() as temp:
            write_t4c_artifact(temp, "t4c-test", {"status": "statistics_insufficient"})
            with self.assertRaises(FileExistsError):
                write_t4c_artifact(temp, "t4c-test", {"status": "other"})
            self.assertTrue((Path(temp) / "t4c-test" / "summary.json").is_file())

    def test_bounded_physical_smoke_reconstructs_identity_and_accounts_five_calls(self):
        from mfg_hedge.attribution_episode import ATTRIBUTION_V1_PROTOCOL, EpisodeIdentity
        from mfg_hedge.config import load_config
        from mfg_hedge.token_continuation import BIN_SCHEMA_V1
        from mfg_hedge.token_online import simulate_episode_online
        from mfg_hedge.token_t3a_execution import T3AAnchorSelector
        from mfg_hedge.transient_control import ReservationParameters
        from mfg_hedge.token_mfg_t4c_supported_soft_response import (
            T4C_NAMESPACE,
            T4C_MACRO_SEED,
            T4CPopulationPolicy,
            reconstruct_episode,
            run_t4c,
        )

        config = load_config(Path(__file__).parents[1] / "configs" / "v1_minimal.json")
        class Capture:
            def __init__(self, anchor_id):
                self.source = T4CPopulationPolicy().new_episode()
                self.selector = T3AAnchorSelector(anchor_id)
                self.observation = None

            def choose(self, observation, policy_key):
                if self.observation is None and self.selector.select(observation):
                    self.observation = observation
                return self.source.choose(observation, policy_key)

        target_index = None
        bin_id = None
        for index in range(32):
            episode = reconstruct_episode(
                config,
                EpisodeIdentity(T4C_NAMESPACE, T4C_MACRO_SEED, index),
                ATTRIBUTION_V1_PROTOCOL,
            )
            capture = Capture(index % 4)
            simulate_episode_online(
                episode,
                capture,
                ReservationParameters(window_width=25.0, budget_rate=0.45, scale=0.25, mean_requirement=1.0),
                degraded_slowdown=2.0,
                hedge_delay=2.0,
            )
            if capture.observation is not None:
                target_index = index
                bin_id = BIN_SCHEMA_V1.bin(capture.observation).bin_id
                break
        self.assertIsNotNone(target_index)
        self.assertIsNotNone(bin_id)
        result = run_t4c(
            config,
            (bin_id,),
            episode_count=target_index + 1,
            minimum_complete_panels=1,
            max_rounds=2,
        )
        self.assertGreaterEqual(result["attempted_calls"], 5, msg=repr(result))
        self.assertEqual(result["attempted_calls"], result["reserved_calls"])
        self.assertEqual(len(result["identity_library_rows"]), target_index + 1)
        self.assertGreaterEqual(len(result["iterations"]), 1)
        fingerprints = {
            iteration["reconstruction_fingerprint"]
            for iteration in result["iterations"]
        }
        self.assertEqual(len(fingerprints), 1)


if __name__ == "__main__":
    unittest.main()
