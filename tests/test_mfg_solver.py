"""Capacity-feasible MFG solver tests (ADR-0007 contract)."""

from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import mfg_hedge.mfg_solver as solver_module
from mfg_hedge.calibration import CalibrationCell, CalibrationTable
from mfg_hedge.config import load_config
from mfg_hedge.domain import ActionStats, CommonState, ProtectionAction, TokenClass
from mfg_hedge.mfg_solver import (
    MFGSolution,
    SolverParameters,
    SolverStatus,
    softmax_stable,
    solve_mfg,
    validate_capacity_acceptance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N, D, I = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
H, DD, F = CommonState.HEALTHY, CommonState.DEGRADED, CommonState.FAILED
R, U = TokenClass.REGULAR, TokenClass.URGENT
GRID = (0.5, 0.7, 0.9, 1.1, 1.3)


def stats_of(
    latency,
    replay_probability,
    hedge_work,
    replay_work=0.01,
    wasted_work=0.0,
):
    return ActionStats(
        mean_latency=latency,
        replay_probability=replay_probability,
        deadline_miss_probability=0.0,
        expected_hedge_work=hedge_work,
        expected_replay_work=replay_work,
        expected_wasted_work=wasted_work,
    )


def cell(state, rho, token_class, action, stats):
    return CalibrationCell(
        state=state,
        target_rho=rho,
        achieved_rho=rho,
        rho_error=0.0,
        token_class=token_class,
        action=action,
        background_arrival_rate=1.0,
        sample_count=400,
        rho_measurement_episode_count=4,
        rho_background_cohort_count=400,
        probe_successful_episode_count=400,
        probe_generation_attempt_count=400,
        probe_skipped_episode_count=0,
        stats=stats,
    )


def make_table(stats_fn, grid=GRID) -> CalibrationTable:
    cells = []
    for state in (H, DD):
        for rho in grid:
            for cls in (R, U):
                for action in (N, D, I):
                    cells.append(cell(state, rho, cls, action, stats_fn(state, rho, cls, action)))
    for rho in grid:
        for cls in (R, U):
            cells.append(cell(F, rho, cls, N, stats_fn(F, rho, cls, N)))
    return CalibrationTable(cells=tuple(cells))


def config_at(load: float):
    return replace(load_config(PROJECT_ROOT / "configs" / "v1_minimal.json"), healthy_offered_load=load)


def benign_stats(state, rho, cls, action):
    # N is strictly best on latency; hedge demand dies without a price.
    if action is N:
        return stats_of(1.0, 0.05, 0.0, replay_work=0.0)
    if action is D:
        return stats_of(3.0, 0.05, 0.2, replay_work=0.0)
    return stats_of(4.0, 0.05, 0.5, replay_work=0.0)


def hedge_attractive_stats(state, rho, cls, action):
    # Hedging cuts latency sharply; only the capacity price restrains it.
    if action is N:
        return stats_of(2.0, 0.10, 0.0, replay_work=0.05)
    if action is D:
        return stats_of(1.2, 0.02, 0.3, replay_work=0.01)
    return stats_of(0.8, 0.005, 0.6, replay_work=0.0)


class SoftmaxTests(unittest.TestCase):
    def test_hand_computed_two_action_softmax(self) -> None:
        out = softmax_stable({"a": 0.0, "b": 1.0})
        self.assertAlmostEqual(out["a"], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(out["b"], math.e / (1.0 + math.e))

    def test_hand_computed_three_action_softmax(self) -> None:
        out = softmax_stable({N: 0.0, D: -1.0, I: -2.0})
        total = 1.0 + math.e ** -1 + math.e ** -2
        self.assertAlmostEqual(out[N], 1.0 / total)
        self.assertAlmostEqual(out[D], math.e ** -1 / total)
        self.assertAlmostEqual(out[I], math.e ** -2 / total)

    def test_extreme_logits_do_not_overflow(self) -> None:
        out = softmax_stable({"a": -10000.0, "b": 0.0})
        self.assertAlmostEqual(out["b"], 1.0)
        self.assertAlmostEqual(out["a"], 0.0)


class ParameterValidationTests(unittest.TestCase):
    def test_illegal_values_are_rejected(self) -> None:
        base = dict(
            inverse_temperature=5.0,
            policy_damping=0.2,
            price_step=0.05,
            price_damping=0.2,
        )
        with self.assertRaises(ValueError):
            SolverParameters(**{**base, "inverse_temperature": float("nan")})
        with self.assertRaises(ValueError):
            SolverParameters(**{**base, "policy_damping": True})
        with self.assertRaises(ValueError):
            SolverParameters(**{**base, "inner_max_iterations": 0})
        with self.assertRaises(ValueError):
            SolverParameters(**{**base, "tolerance": -1.0})
        for field in ("incremental_work_cost", "wasted_work_cost"):
            for bad in (-0.1, True, float("nan"), float("inf")):
                with self.assertRaises(ValueError, msg=f"{field}={bad!r}"):
                    SolverParameters(**{**base, field: bad})


class PersistentRuntimeCostTests(unittest.TestCase):
    @staticmethod
    def cost_sensitive_stats(state, rho, cls, action):
        if action is N:
            return stats_of(1.0, 0.0, 0.0, replay_work=0.0, wasted_work=0.0)
        if action is D:
            return stats_of(0.7, 0.0, 0.2, replay_work=0.0, wasted_work=0.1)
        return stats_of(0.6, 0.0, 0.4, replay_work=0.0, wasted_work=0.3)

    def test_positive_costs_reduce_price_zero_hedge_probability(self) -> None:
        config = config_at(0.5)
        table = make_table(self.cost_sensitive_stats)
        legacy = solve_mfg(
            config,
            table,
            SolverParameters(5.0, 0.2, 0.05, 0.2),
        ).solutions[H]
        priced = solve_mfg(
            config,
            table,
            SolverParameters(
                5.0,
                0.2,
                0.05,
                0.2,
                incremental_work_cost=1.0,
                wasted_work_cost=1.0,
            ),
        ).solutions[H]
        self.assertEqual(legacy.price, 0.0)
        self.assertEqual(priced.price, 0.0)
        for cls in (R, U):
            legacy_hedge = legacy.probabilities[cls][D] + legacy.probabilities[cls][I]
            priced_hedge = priced.probabilities[cls][D] + priced.probabilities[cls][I]
            self.assertLess(priced_hedge, legacy_hedge)

    def test_waste_cost_changes_choice_with_equal_latency_and_incremental_work(self) -> None:
        stats = {
            R: {
                N: stats_of(1.0, 0.0, 0.0, replay_work=0.0, wasted_work=0.0),
                D: stats_of(1.0, 0.0, 0.2, replay_work=0.0, wasted_work=0.0),
                I: stats_of(1.0, 0.0, 0.2, replay_work=0.0, wasted_work=0.5),
            }
        }
        params = SolverParameters(
            5.0,
            0.2,
            0.05,
            0.2,
            incremental_work_cost=0.0,
            wasted_work_cost=1.0,
        )
        response = solver_module._soft_best_response(params, stats, 0.0, (D, I), R)
        self.assertGreater(response[D], response[I])


class PriceZeroSlackTests(unittest.TestCase):
    def test_price_zero_accepted_when_capacity_has_slack(self) -> None:
        solution = solve_mfg(config_at(0.6), make_table(benign_stats))
        h = solution.solutions[H]
        self.assertEqual(h.status, SolverStatus.CONVERGED)
        self.assertEqual(h.price, 0.0)
        self.assertGreater(h.capacity_slack, 0.0)
        self.assertEqual(h.complementarity_residual, 0.0)
        self.assertEqual(h.reason, "price_zero_slack")


class PositivePriceRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solution = solve_mfg(config_at(0.6), make_table(hedge_attractive_stats))
        self.h = self.solution.solutions[H]
        self.d = self.solution.solutions[DD]

    def test_positive_price_boundary_root(self) -> None:
        for solution in (self.h, self.d):
            self.assertEqual(solution.status, SolverStatus.CONVERGED)
            self.assertGreater(solution.price, 0.0)
            self.assertLessEqual(abs(solution.capacity_gap), 1e-6)
            self.assertLessEqual(solution.complementarity_residual, 1e-6)
            self.assertLessEqual(
                solution.hedge_work_rate, solution.hedge_budget_rate + 1e-6
            )
            self.assertFalse(solution.grid_clamped)

    def test_positive_price_with_slack_fails_complementarity(self) -> None:
        violations = validate_capacity_acceptance(
            price=1.0,
            capacity_gap=-0.1,
            policy_residual=0.0,
            load_residual=0.0,
            capacity_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            residual_tolerance=1e-6,
        )
        self.assertIn("complementarity", violations)

    def test_bracket_records_sign_crossing(self) -> None:
        evaluations = self.h.price_evaluations
        self.assertGreaterEqual(len(evaluations), 3)
        self.assertAlmostEqual(evaluations[0][0], 0.0)
        self.assertGreater(evaluations[0][1], 0.0)
        signs = [gap > 0 for _, gap in evaluations]
        self.assertIn(True, signs)
        self.assertIn(False, signs)


class BoundaryPinnedTests(unittest.TestCase):
    def test_binding_branch_runs_the_cold_inner_solve_only_at_price_zero(self) -> None:
        config = config_at(0.6)
        table = make_table(hedge_attractive_stats)
        params = SolverParameters.from_config(config)
        with patch.object(
            solver_module,
            "_inner_solve",
            wraps=solver_module._inner_solve,
        ) as inner_solve:
            solution = solver_module._solve_hd(config, table, params, H)

        self.assertEqual(solution.status, SolverStatus.CONVERGED)
        self.assertEqual(inner_solve.call_count, 1)
        self.assertAlmostEqual(solution.rho, config.target_max_utilization)

    def test_binding_price_evaluations_are_monotone_at_fixed_boundary_stats(self) -> None:
        solution = solve_mfg(config_at(0.6), make_table(hedge_attractive_stats))
        evaluations = solution.solutions[H].price_evaluations
        self.assertGreater(evaluations[0][1], 0.0)
        gaps_by_price = [gap for _, gap in sorted(evaluations)]
        self.assertTrue(
            all(
                right <= left
                for left, right in zip(gaps_by_price, gaps_by_price[1:])
            ),
            evaluations,
        )

    def test_failed_price_zero_solve_never_enters_binding_branch(self) -> None:
        config = config_at(0.6)
        params = replace(
            SolverParameters.from_config(config),
            inner_max_iterations=1,
        )
        with patch.object(solver_module, "_boundary_result") as boundary_result:
            solution = solver_module._solve_hd(
                config,
                make_table(hedge_attractive_stats),
                params,
                H,
            )
        self.assertEqual(solution.status, SolverStatus.NONCONVERGED)
        self.assertEqual(solution.reason, "inner_iteration_cap")
        boundary_result.assert_not_called()

    def test_load_05_degraded_state_has_a_finite_boundary_root(self) -> None:
        solution = solve_mfg(config_at(0.5), make_table(hedge_attractive_stats))
        degraded = solution.solutions[DD]
        self.assertEqual(degraded.status, SolverStatus.CONVERGED)
        self.assertGreater(degraded.price, 0.0)
        self.assertAlmostEqual(degraded.rho, 0.9)

    def test_inconsistent_binding_branch_is_diagnostic(self) -> None:
        config = config_at(0.6)
        params = SolverParameters.from_config(config)
        primary = solver_module.arrival_rate_of(config) * config.healthy_service_mean
        overloaded_base = solver_module._InnerResult(
            probabilities={
                cls: {N: 1.0, D: 0.0, I: 0.0} for cls in (R, U)
            },
            rho=1.0,
            stats={},
            primary=primary,
            hedge=0.7,
            replay=0.0,
            total=primary + 0.7,
            iterations=1,
            converged=True,
        )
        with patch.object(
            solver_module,
            "_inner_solve",
            return_value=overloaded_base,
        ):
            solution = solver_module._solve_hd(
                config,
                make_table(benign_stats),
                params,
                H,
            )
        self.assertEqual(solution.status, SolverStatus.NONCONVERGED)
        self.assertEqual(solution.reason, "binding_branch_inconsistent")


class FailureModeTests(unittest.TestCase):
    def test_policy_minimum_certifies_boundary_infeasibility(self) -> None:
        # Primary alone fits, but every action's Hedge+Replay work makes the
        # boundary impossible. ADR-0008 can certify this before bracketing.
        def stats_fn(state, rho, cls, action):
            if action is N:
                return stats_of(2.0, 0.30, 0.0, replay_work=0.40)
            if action is D:
                return stats_of(1.5, 0.10, 0.15, replay_work=0.20)
            return stats_of(1.2, 0.05, 0.30, replay_work=0.10)

        solution = solve_mfg(config_at(0.6), make_table(stats_fn))
        d = solution.solutions[DD]
        self.assertEqual(d.status, SolverStatus.INFEASIBLE)
        self.assertEqual(d.reason, "boundary_minimum_exceeds_capacity")

    def test_primary_only_overload_certifies_infeasible(self) -> None:
        solution = solve_mfg(config_at(0.95), make_table(benign_stats))
        self.assertEqual(solution.solutions[H].status, SolverStatus.INFEASIBLE)
        self.assertEqual(
            solution.solutions[H].reason,
            "boundary_minimum_exceeds_capacity",
        )
        self.assertEqual(solution.solutions[DD].status, SolverStatus.INFEASIBLE)
        # At load 0.95 the F Normal-only rho is off the calibration grid;
        # clamping is disqualifying even for the forced-Normal transient.
        self.assertEqual(solution.solutions[F].status, SolverStatus.NONCONVERGED)
        self.assertTrue(solution.solutions[F].grid_clamped)

    def test_grid_clamp_stops_the_state_immediately(self) -> None:
        table = make_table(benign_stats, grid=(0.5,))
        solution = solve_mfg(config_at(0.9), table)
        h = solution.solutions[H]
        self.assertEqual(h.status, SolverStatus.NONCONVERGED)
        self.assertTrue(h.grid_clamped)
        self.assertEqual(h.reason, "grid_clamped")
        self.assertLessEqual(h.outer_evaluations, 1)

    def test_finite_bracket_cap_is_nonconverged(self) -> None:
        config = config_at(0.6)
        params = replace(
            SolverParameters.from_config(config),
            bracket_max_doublings=1,
        )
        solution = solve_mfg(config, make_table(hedge_attractive_stats), params)
        self.assertEqual(solution.solutions[H].status, SolverStatus.NONCONVERGED)
        self.assertEqual(solution.solutions[H].reason, "bracket_failed")


class ForcedNormalTests(unittest.TestCase):
    def test_f_state_is_forced_normal_with_structural_violation(self) -> None:
        solution = solve_mfg(config_at(0.6), make_table(benign_stats))
        f = solution.solutions[F]
        self.assertEqual(f.status, SolverStatus.FORCED_NORMAL)
        self.assertEqual(f.price, 0.0)
        self.assertEqual(f.hedge_budget_rate, 0.0)
        self.assertTrue(f.structural_capacity_violation)
        for cls in (R, U):
            self.assertEqual(f.probabilities[cls][N], 1.0)
            self.assertEqual(f.probabilities[cls][D], 0.0)
            self.assertEqual(f.probabilities[cls][I], 0.0)


class SelfConsistencyTests(unittest.TestCase):
    def test_final_residuals_are_undamped_and_self_consistent(self) -> None:
        table = make_table(hedge_attractive_stats)
        solution = solve_mfg(config_at(0.6), table)
        h = solution.solutions[H]
        self.assertEqual(h.status, SolverStatus.CONVERGED)
        # Recompute the undamped best response at the final (rho, price).
        penalties = {R: 1.0, U: 5.0}
        for cls in (R, U):
            costs = {
                action: h.final_stats[cls][action].mean_latency
                + penalties[cls] * h.final_stats[cls][action].replay_probability
                + h.price
                * (
                    h.final_stats[cls][action].expected_hedge_work
                    + h.final_stats[cls][action].expected_replay_work
                )
                for action in (N, D, I)
            }
            best = softmax_stable(
                {action: -5.0 * costs[action] for action in (N, D, I)}
            )
            for action in (N, D, I):
                self.assertAlmostEqual(
                    best[action], h.probabilities[cls][action], places=5
                )
        self.assertAlmostEqual(
            h.load_residual,
            abs(h.total_work_rate / 2.0 - h.rho),
        )

    def test_deterministic_identical_runs(self) -> None:
        table = make_table(hedge_attractive_stats)
        first = solve_mfg(config_at(0.6), table)
        second = solve_mfg(config_at(0.6), table)
        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())


class PenaltyAsymmetryTests(unittest.TestCase):
    def test_higher_replay_penalty_protects_urgent_at_least_as_much(self) -> None:
        def stats_fn(state, rho, cls, action):
            if action is N:
                return stats_of(1.0, 0.10, 0.0, replay_work=0.0)
            if action is D:
                return stats_of(0.9, 0.05, 0.2, replay_work=0.0)
            return stats_of(0.8, 0.01, 0.5, replay_work=0.0)

        solution = solve_mfg(config_at(0.6), make_table(stats_fn))
        probs = solution.solutions[H].probabilities
        regular_hedge = probs[R][D] + probs[R][I]
        urgent_hedge = probs[U][D] + probs[U][I]
        self.assertGreaterEqual(urgent_hedge, regular_hedge)


class DegenerateLoadTests(unittest.TestCase):
    def test_load_07_d_is_primary_overload_infeasible(self) -> None:
        solution = solve_mfg(config_at(0.7), make_table(benign_stats))
        self.assertEqual(solution.solutions[DD].status, SolverStatus.INFEASIBLE)
        self.assertEqual(
            solution.solutions[DD].reason,
            "boundary_minimum_exceeds_capacity",
        )
        self.assertEqual(solution.solutions[H].status, SolverStatus.CONVERGED)
        # F at load 0.7 has rho 1.4, off the calibration grid: clamped.
        self.assertEqual(solution.solutions[F].status, SolverStatus.NONCONVERGED)
        self.assertTrue(solution.solutions[F].grid_clamped)


if __name__ == "__main__":
    unittest.main()
