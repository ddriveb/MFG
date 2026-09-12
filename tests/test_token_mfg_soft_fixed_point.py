import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mfg_hedge.domain import ProtectionAction


class TokenMfgSoftFixedPointTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self):
        from mfg_hedge.token_mfg_soft_fixed_point import soft_best_response

        probabilities = soft_best_response(
            {"b": {
                ProtectionAction.NORMAL: 10000.0,
                ProtectionAction.DELAYED_HEDGE: 10001.0,
                ProtectionAction.IMMEDIATE_HEDGE: 10002.0,
            }},
            beta=1.0,
        )["b"]
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(math.fsum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])

    def test_soft_function_mixture_damps_exactly(self):
        from mfg_hedge.token_mfg_soft_fixed_point import SoftPolicyState

        state = SoftPolicyState.initial()
        soft_i = {"b": (0.0, 0.0, 1.0)}
        state = state.updated(soft_i, eta=0.2)
        self.assertEqual(state.probabilities_for("b"), (0.8, 0.0, 0.2))
        state = state.updated(soft_i, eta=0.2)
        self.assertEqual(state.probabilities_for("b"), (0.64, 0.0, 0.36))

    def test_fresh_policy_state_and_unknown_bin_niin_fallback(self):
        from mfg_hedge.token_mfg_soft_fixed_point import SoftPopulationPolicy, SoftPolicyState

        state = SoftPolicyState.initial().updated({"b": (0.0, 0.0, 1.0)}, eta=0.2)
        first = SoftPopulationPolicy(state, seed=3)
        second = first.new_episode()
        self.assertIsNot(first, second)
        self.assertEqual(first.state, second.state)
        with self.assertRaises(FrozenInstanceError):
            state.base_weight = 0.5
        self.assertEqual(
            state.probabilities_for("unknown", base_action=ProtectionAction.IMMEDIATE_HEDGE),
            (0.0, 0.0, 1.0),
        )

    def test_paired_gap_and_se_use_within_panel_differences(self):
        from mfg_hedge.token_mfg_soft_fixed_point import paired_action_gap

        panels = {
            ("b", ProtectionAction.NORMAL): [10.0, 12.0, 14.0],
            ("b", ProtectionAction.DELAYED_HEDGE): [11.0, 13.0, 15.0],
            ("b", ProtectionAction.IMMEDIATE_HEDGE): [20.0, 21.0, 22.0],
        }
        result = paired_action_gap(panels, "b")
        self.assertEqual(result["best_action"], ProtectionAction.NORMAL)
        self.assertEqual(result["runner_up"], ProtectionAction.DELAYED_HEDGE)
        self.assertEqual(result["gap"], 1.0)
        self.assertEqual(result["paired_standard_error"], 0.0)

    def test_library_fingerprint_is_immutable_and_reuse_is_explicit(self):
        from mfg_hedge.token_mfg_soft_fixed_point import (
            episode_library_fingerprint,
            validate_episode_library,
        )

        library = ("trace-a", "trace-b")
        fingerprint = episode_library_fingerprint(library)
        self.assertEqual(validate_episode_library(library, fingerprint), library)
        with self.assertRaises(Exception):
            validate_episode_library(("trace-b", "trace-a"), fingerprint)

    def test_residual_status_does_not_call_hard_br_repeat_a_cycle(self):
        from mfg_hedge.token_mfg_soft_fixed_point import soft_stop_status

        self.assertEqual(soft_stop_status(0.005, 0.004, comparable_rounds=2), "soft_fixed_point")
        self.assertIsNone(soft_stop_status(0.011, 0.004, comparable_rounds=2))

    def test_budget_and_artifact_are_exact_and_non_overwriting(self):
        from mfg_hedge.token_mfg_soft_fixed_point import (
            T4BCallLedger,
            write_soft_fixed_point_artifact,
        )

        ledger = T4BCallLedger(5)
        ledger.reserve(5)
        ledger.start(5)
        self.assertEqual((ledger.reserved, ledger.attempted), (5, 5))
        with tempfile.TemporaryDirectory() as temp:
            write_soft_fixed_point_artifact(temp, "soft-test", {"status": "not_converged_20"})
            with self.assertRaises(FileExistsError):
                write_soft_fixed_point_artifact(temp, "soft-test", {"status": "other"})
            self.assertTrue((Path(temp) / "soft-test" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
