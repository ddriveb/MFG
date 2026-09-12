"""Official-gate classification for predeclared load diagnostics."""

from dataclasses import replace
import unittest

from mfg_hedge.domain import CommonState, ProtectionAction, TokenClass
from mfg_hedge.mfg_solver import SolverStatus
from mfg_hedge.stress import official_gate_diagnostic
from tests.test_paired_metrics import fixture_solution


class OfficialGateDiagnosticTests(unittest.TestCase):
    def test_valid_solution_passes(self) -> None:
        result = official_gate_diagnostic(fixture_solution())
        self.assertTrue(result["pass"])
        self.assertEqual(result["failures"], [])

    def test_infeasible_degraded_state_fails(self) -> None:
        solution = fixture_solution()
        degraded = solution.solutions[CommonState.DEGRADED]
        changed = replace(
            solution,
            solutions={
                **solution.solutions,
                CommonState.DEGRADED: replace(
                    degraded,
                    status=SolverStatus.INFEASIBLE,
                    reason="policy_minimum_exceeds_target",
                ),
            },
        )
        result = official_gate_diagnostic(changed)
        self.assertFalse(result["pass"])
        self.assertIn("D:status=infeasible", result["failures"])

    def test_forced_failed_policy_must_be_exactly_normal(self) -> None:
        solution = fixture_solution()
        failed = solution.solutions[CommonState.FAILED]
        bad_probabilities = {
            cls: {ProtectionAction.NORMAL: 0.9, ProtectionAction.DELAYED_HEDGE: 0.1}
            for cls in (TokenClass.REGULAR, TokenClass.URGENT)
        }
        changed = replace(
            solution,
            solutions={
                **solution.solutions,
                CommonState.FAILED: replace(failed, probabilities=bad_probabilities),
            },
        )
        result = official_gate_diagnostic(changed)
        self.assertFalse(result["pass"])
        self.assertIn("F:policy_not_exact_all_normal", result["failures"])


if __name__ == "__main__":
    unittest.main()
