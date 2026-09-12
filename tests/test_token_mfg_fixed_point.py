import json
import tempfile
import unittest
from pathlib import Path

from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.token_online import TokenObservation


class TokenMfgFixedPointTests(unittest.TestCase):
    def test_function_mixture_damps_exactly_and_starts_from_niin(self):
        from mfg_hedge.token_mfg_fixed_point import (
            DampedPolicyState,
            MixedPopulationPolicy,
        )

        state = DampedPolicyState.initial()
        self.assertEqual(state.probabilities_for("bin-a"), (1.0, 0.0, 0.0))
        state = state.updated({"bin-a": ProtectionAction.IMMEDIATE_HEDGE}, eta=0.2)
        self.assertEqual(state.probabilities_for("bin-a"), (0.8, 0.0, 0.2))
        state = state.updated({"bin-a": ProtectionAction.IMMEDIATE_HEDGE}, eta=0.2)
        self.assertEqual(state.probabilities_for("bin-a"), (0.64, 0.0, 0.36))

        policy = MixedPopulationPolicy(state, seed=20260913)
        self.assertIsNot(policy.new_episode(), policy)
        self.assertEqual(policy.action_for_bin("unknown"), ProtectionAction.NORMAL)
        self.assertEqual(
            state.probabilities_for("unknown", base_action=ProtectionAction.IMMEDIATE_HEDGE),
            (0.0, 0.0, 1.0),
        )

    def test_policy_source_is_fresh_and_causal(self):
        from mfg_hedge.token_mfg_fixed_point import MixedPopulationPolicy

        state = MixedPopulationPolicy.initial(seed=7)
        first = state.new_episode()
        second = state.new_episode()
        self.assertIsNot(first, second)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.seed, second.seed)

    def test_br_is_deterministic_and_ties_are_n_d_i(self):
        from mfg_hedge.token_mfg_fixed_point import build_best_response

        rows = {
            ("b", ProtectionAction.NORMAL): 1.0,
            ("b", ProtectionAction.DELAYED_HEDGE): 1.0,
            ("b", ProtectionAction.IMMEDIATE_HEDGE): 1.0,
            ("c", ProtectionAction.NORMAL): 3.0,
            ("c", ProtectionAction.DELAYED_HEDGE): 2.0,
            ("c", ProtectionAction.IMMEDIATE_HEDGE): 1.0,
        }
        self.assertEqual(
            build_best_response(rows, ("b", "c")),
            {"b": ProtectionAction.NORMAL, "c": ProtectionAction.IMMEDIATE_HEDGE},
        )

    def test_q_floor_fails_closed(self):
        from mfg_hedge.token_mfg_fixed_point import (
            summarize_q_rows,
            FixedPointDiagnosticError,
        )

        with self.assertRaises(FixedPointDiagnosticError):
            summarize_q_rows(
                {("b", action): [1.0] for action in ProtectionAction},
                ("b",),
                minimum_complete_panels=2,
            )

    def test_status_helpers_preserve_cycle_and_nonconvergence(self):
        from mfg_hedge.token_mfg_fixed_point import classify_br_history

        self.assertEqual(classify_br_history(["a", "b", "a"]), ("cycle", 2))
        self.assertEqual(classify_br_history(["a", "b", "c"]), (None, None))

    def test_exact_call_budget_and_artifact_no_overwrite(self):
        from mfg_hedge.token_mfg_fixed_point import (
            T4CallLedger,
            write_fixed_point_artifact,
        )

        ledger = T4CallLedger(5)
        ledger.reserve(3)
        ledger.start(3)
        self.assertEqual(ledger.attempted, 3)
        with self.assertRaises(Exception):
            ledger.reserve(3)

        with tempfile.TemporaryDirectory() as temp:
            payload = {"status": "not_converged_10", "claims_mfg": False}
            write_fixed_point_artifact(temp, "t4-test", payload)
            with self.assertRaises(FileExistsError):
                write_fixed_point_artifact(temp, "t4-test", payload)
            self.assertEqual(
                json.loads((Path(temp) / "t4-test" / "summary.json").read_text()),
                payload,
            )


if __name__ == "__main__":
    unittest.main()
