"""Quota allocator tests: two-phase deficit, budgets, windows, identities."""

from dataclasses import replace
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.config import load_config
from mfg_hedge.domain import ActionStats, CommonState, ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.mfg_solver import (
    MFGSolution,
    SolverParameters,
    SolverStatus,
    StatePolicySolution,
)
from mfg_hedge.quota import project_actions
from mfg_hedge.workload import TokenSpec, WorkloadTrace, generate_workload_with_hedge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N, D, I = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
H, DD, F = CommonState.HEALTHY, CommonState.DEGRADED, CommonState.FAILED
R, U = TokenClass.REGULAR, TokenClass.URGENT
TIMELINE = CommonStateTimeline(100.0, 200.0, 220.0)
G = {N: 0.0, D: 0.2, I: 0.5}


def stats_g(action, replay=0.01):
    return ActionStats(
        mean_latency=1.0,
        replay_probability=replay,
        deadline_miss_probability=0.0,
        expected_hedge_work=G[action],
        expected_replay_work=0.0,
    )


def make_state_solution(state, probabilities, hedge_budget_rate, status=None):
    final_stats = {
        cls: {action: stats_g(action) for action in (N, D, I)} for cls in (R, U)
    }
    if status is None:
        status = (
            SolverStatus.FORCED_NORMAL if state is F else SolverStatus.CONVERGED
        )
    return StatePolicySolution(
        state=state,
        status=status,
        reason="ok",
        grid_clamped=False,
        probabilities=probabilities,
        price=0.0,
        rho=0.7,
        primary_work_rate=1.2,
        hedge_work_rate=0.0,
        replay_work_rate=0.0,
        total_work_rate=1.2,
        target_work_rate=1.35,
        capacity_gap=-0.15,
        capacity_violation=0.0,
        capacity_slack=0.15,
        complementarity_residual=0.0,
        hedge_budget_rate=hedge_budget_rate,
        policy_residual=0.0,
        price_residual=0.0,
        load_residual=0.0,
        iterations=1,
        outer_evaluations=1,
        price_evaluations=((0.0, -0.15),),
        structural_capacity_violation=False,
        final_stats=final_stats,
    )


def make_solution(h_probs, d_probs=None, f_probs=None, h_budget=1.0, d_budget=1.0, f_budget=0.0):
    if d_probs is None:
        d_probs = h_probs
    if f_probs is None:
        f_probs = {cls: {N: 1.0, D: 0.0, I: 0.0} for cls in (R, U)}
    return MFGSolution(
        solutions={
            H: make_state_solution(H, h_probs, h_budget),
            DD: make_state_solution(DD, d_probs, d_budget),
            F: make_state_solution(F, f_probs, f_budget),
        },
        parameters=SolverParameters(
            inverse_temperature=5.0,
            policy_damping=0.2,
            price_step=0.05,
            price_damping=0.2,
        ),
    )


def make_trace(arrivals, classes=None) -> WorkloadTrace:
    count = len(arrivals)
    if classes is None:
        classes = [R] * count
    tokens = tuple(
        TokenSpec(token_id=i, arrival_time=a, token_class=classes[i])
        for i, a in enumerate(arrivals)
    )
    stream = tuple(1.0 for _ in range(count))
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.0,
        tokens=tokens,
        service_times=(stream, stream),
        replay_service_times=(stream, stream),
        hedge_service_times=(stream, stream),
    )


class DeficitMechanicsTests(unittest.TestCase):
    def test_requested_counts_within_one_of_target(self) -> None:
        probs = {R: {N: 0.5, D: 0.3, I: 0.2}, U: {N: 0.5, D: 0.3, I: 0.2}}
        solution = make_solution(probs)
        trace = make_trace([1.0 + 0.1 * i for i in range(10)])
        result = project_actions(trace, solution, TIMELINE)
        requested = [a.requested_action for a in result.assignments]
        for action, target in ((N, 5.0), (D, 3.0), (I, 2.0)):
            count = requested.count(action)
            self.assertLess(abs(count - target), 1.0, msg=f"{action}: {count} vs {target}")

    def test_tie_break_order_is_normal_delayed_immediate(self) -> None:
        third = {N: 1.0 / 3.0, D: 1.0 / 3.0, I: 1.0 / 3.0}
        solution = make_solution({R: dict(third), U: dict(third)}, h_budget=100.0)
        trace = make_trace([1.0, 1.1, 1.2])
        result = project_actions(trace, solution, TIMELINE)
        self.assertEqual(
            [a.requested_action for a in result.assignments], [N, D, I]
        )

    def test_tiny_window_with_single_token(self) -> None:
        solution = make_solution({R: {N: 0.0, D: 1.0, I: 0.0}, U: {N: 0.0, D: 1.0, I: 0.0}}, h_budget=100.0)
        trace = make_trace([1.0])
        result = project_actions(trace, solution, TIMELINE)
        self.assertEqual(result.assignments[0].applied_action, D)

    def test_deficit_not_redebted_after_suppression(self) -> None:
        # D always requested; budget funds only the first; deficits must keep
        # the exact deficit trajectory regardless of suppression.
        solution = make_solution(
            {R: {N: 0.0, D: 1.0, I: 0.0}, U: {N: 0.0, D: 1.0, I: 0.0}},
            h_budget=0.01,  # window budget 0.25: one D (cost 0.2), then suppression
        )
        trace = make_trace([1.0 + 0.1 * i for i in range(4)])
        result = project_actions(trace, solution, TIMELINE)
        applied = [a.applied_action for a in result.assignments]
        self.assertEqual(applied, [D, N, N, N])
        self.assertEqual(
            sum(1 for a in result.assignments if a.quota_suppressed), 3
        )


class BudgetAndWindowTests(unittest.TestCase):
    def test_sufficient_budget_applies_requested(self) -> None:
        solution = make_solution(
            {R: {N: 0.0, D: 1.0, I: 0.0}, U: {N: 0.0, D: 1.0, I: 0.0}},
            h_budget=100.0,
        )
        trace = make_trace([1.0, 1.1])
        result = project_actions(trace, solution, TIMELINE)
        self.assertEqual([a.applied_action for a in result.assignments], [D, D])
        self.assertEqual(result.global_quota_suppressed, 0)

    def test_f_state_is_normal_without_quota_suppression(self) -> None:
        solution = make_solution(
            {R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}},
        )
        trace = make_trace([205.0, 210.0])
        result = project_actions(trace, solution, TIMELINE)
        for assignment in result.assignments:
            self.assertEqual(assignment.requested_action, N)
            self.assertEqual(assignment.applied_action, N)
            self.assertFalse(assignment.quota_suppressed)
        self.assertEqual(result.global_quota_suppressed, 0)

    def test_urgent_share_is_never_borrowed_by_regular(self) -> None:
        # weights 0.8/0.2, equal costs 0.2: shares 80%/20% of window budget 1.0
        # (budget rate 0.04 x duration 25).
        probs = {R: {N: 0.0, D: 1.0, I: 0.0}, U: {N: 0.0, D: 1.0, I: 0.0}}
        solution = make_solution(probs, h_budget=0.04)
        arrivals = [1.0 + 0.1 * i for i in range(9)] + [24.0]
        classes = [R] * 9 + [U]
        result = project_actions(make_trace(arrivals, classes), solution, TIMELINE)
        regular_applied = [
            a.applied_action for a in result.assignments if a.token_class is R
        ]
        urgent_applied = [
            a.applied_action for a in result.assignments if a.token_class is U
        ]
        # Regular share 0.8 funds floor(0.8/0.2) = 4 delayed hedges.
        self.assertEqual(regular_applied.count(D), 4)
        self.assertEqual(urgent_applied, [D])

    def test_state_boundary_forces_window_cut(self) -> None:
        solution = make_solution({R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}})
        trace = make_trace([99.9, 100.0])
        result = project_actions(trace, solution, TIMELINE)
        first, second = result.assignments
        self.assertEqual((first.window_start, first.window_end), (75.0, 100.0))
        self.assertEqual((second.window_start, second.window_end), (100.0, 125.0))
        self.assertEqual(first.state, H)
        self.assertEqual(second.state, DD)

    def test_recovery_window_starts_at_220_not_225(self) -> None:
        solution = make_solution({R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}})
        trace = make_trace([220.0, 244.9, 245.0])
        result = project_actions(trace, solution, TIMELINE)
        first, second, third = result.assignments
        self.assertEqual((first.window_start, first.window_end), (220.0, 245.0))
        self.assertEqual((second.window_start, second.window_end), (220.0, 245.0))
        self.assertEqual((third.window_start, third.window_end), (245.0, 270.0))
        self.assertEqual(first.phase, Phase.RECOVERED)
        self.assertEqual(first.state, H)

    def test_window_counters_reset_at_state_boundary(self) -> None:
        probs = {R: {N: 0.0, D: 1.0, I: 0.0}, U: {N: 0.0, D: 1.0, I: 0.0}}
        # Budget rate 0.01 -> window budget 0.25: one D per window, then
        # suppression.
        solution = make_solution(probs, h_budget=0.01, d_budget=0.01)
        trace = make_trace([99.0, 99.5, 100.0, 100.5])
        result = project_actions(trace, solution, TIMELINE)
        applied = [a.applied_action for a in result.assignments]
        self.assertEqual(applied, [D, N, D, N])


class IdentityAndDeterminismTests(unittest.TestCase):
    def test_counter_identities_hold_per_window_class_and_global(self) -> None:
        probs = {R: {N: 0.4, D: 0.4, I: 0.2}, U: {N: 0.2, D: 0.4, I: 0.4}}
        solution = make_solution(probs, h_budget=1.0, d_budget=0.2)
        arrivals = [10.0 + 0.2 * i for i in range(20)] + [
            110.0 + 0.2 * i for i in range(20)
        ]
        classes = [R if i % 3 else U for i in range(40)]
        result = project_actions(make_trace(arrivals, classes), solution, TIMELINE)
        for window in result.windows:
            for audit in window.classes:
                requested_hedge = (
                    audit.requested_counts[D] + audit.requested_counts[I]
                )
                applied_hedge = audit.applied_counts[D] + audit.applied_counts[I]
                self.assertEqual(
                    requested_hedge, applied_hedge + audit.quota_suppressed
                )
                self.assertLessEqual(
                    audit.planned_expected_hedge_work, audit.class_budget_share + 1e-12
                )
            self.assertLessEqual(
                sum(a.planned_expected_hedge_work for a in window.classes)
                - window.window_budget,
                1e-9,
            )
        requested_hedge = sum(
            result.global_requested_counts[a] for a in (D, I)
        )
        applied_hedge = sum(result.global_applied_counts[a] for a in (D, I))
        self.assertEqual(
            requested_hedge, applied_hedge + result.global_quota_suppressed
        )

    def test_no_future_knowledge(self) -> None:
        probs = {R: {N: 0.5, D: 0.5, I: 0.0}, U: {N: 0.5, D: 0.5, I: 0.0}}
        solution = make_solution(probs)
        first = make_trace([1.0 + i for i in range(10)])
        second = make_trace([1.0 + i for i in range(5)] + [500.0 + i for i in range(5)])
        one = project_actions(first, solution, TIMELINE)
        two = project_actions(second, solution, TIMELINE)
        self.assertEqual(one.assignments[:5], two.assignments[:5])

    def test_identical_runs(self) -> None:
        probs = {R: {N: 0.5, D: 0.3, I: 0.2}, U: {N: 0.2, D: 0.5, I: 0.3}}
        solution = make_solution(probs)
        trace = make_trace([1.0 + 0.1 * i for i in range(30)])
        first = project_actions(trace, solution, TIMELINE)
        second = project_actions(trace, solution, TIMELINE)
        self.assertEqual(first, second)

    def test_engine_hedge_requested_equals_applied_hedge(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        trace = generate_workload_with_hedge(config, 100)
        probs = {
            R: {N: 0.5, D: 0.5, I: 0.0},
            U: {N: 0.0, D: 0.5, I: 0.5},
        }
        solution = make_solution(probs, h_budget=50.0, d_budget=50.0, f_budget=0.0)
        result = project_actions(trace, solution, TIMELINE)
        plan = result.to_action_plan()
        engine_result = simulate_hedge_common_state(
            trace,
            timeline=TIMELINE,
            degraded_slowdown=config.degraded_slowdown,
            actions=plan,
            hedge_delay=1.5,
        )
        applied_hedge = sum(result.global_applied_counts[a] for a in (D, I))
        self.assertEqual(engine_result.hedge_requested, applied_hedge)


class DegenerateLoadTests(unittest.TestCase):
    def test_load_07_d_is_infeasible_and_quota_gate_rejects(self) -> None:
        from mfg_hedge.mfg_solver import solve_mfg
        from tests.test_mfg_solver import benign_stats, make_table

        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        table = make_table(benign_stats)
        solution = solve_mfg(config, table)
        self.assertEqual(
            solution.solutions[DD].status, SolverStatus.INFEASIBLE
        )
        trace = make_trace([110.0 + 0.2 * i for i in range(5)])
        with self.assertRaisesRegex(ValueError, "official|converged"):
            project_actions(trace, solution, TIMELINE)

    def test_load_06_leaves_real_d_headroom(self) -> None:
        from mfg_hedge.mfg_solver import solve_mfg
        from tests.test_mfg_solver import benign_stats, make_table

        config = replace(
            load_config(PROJECT_ROOT / "configs" / "v1_minimal.json"),
            healthy_offered_load=0.6,
        )
        table = make_table(benign_stats)
        solution = solve_mfg(config, table)
        self.assertGreater(solution.solutions[DD].hedge_budget_rate, 0.0)
        self.assertEqual(solution.solutions[DD].status, SolverStatus.CONVERGED)
        # Force a policy that always requests Immediate in D.
        forced = make_solution(
            {R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}},
            d_probs={R: {N: 0.0, D: 0.0, I: 1.0}, U: {N: 0.0, D: 0.0, I: 1.0}},
            d_budget=solution.solutions[DD].hedge_budget_rate,
        )
        trace = make_trace([110.0 + 0.2 * i for i in range(10)])
        result = project_actions(trace, forced, TIMELINE)
        self.assertGreaterEqual(result.global_applied_counts[I], 1)


class OfficialGateTests(unittest.TestCase):
    def test_quota_rejects_nonconverged_or_misstated_solutions(self) -> None:
        good = make_solution({R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}})
        trace = make_trace([1.0, 1.1])
        project_actions(trace, good, TIMELINE)  # baseline works

        bad_h = replace(
            good.solutions[H], status=SolverStatus.NONCONVERGED, reason="inner_iteration_cap"
        )
        broken = replace(good, solutions={**good.solutions, H: bad_h})
        with self.assertRaisesRegex(ValueError, "official|converged"):
            project_actions(trace, broken, TIMELINE)

        bad_f = replace(good.solutions[F], status=SolverStatus.CONVERGED)
        broken_f = replace(good, solutions={**good.solutions, F: bad_f})
        with self.assertRaisesRegex(ValueError, "forced_normal"):
            project_actions(trace, broken_f, TIMELINE)

        bad_complementarity = replace(
            good.solutions[H],
            price=1.0,
            capacity_gap=-0.1,
            capacity_slack=0.1,
            complementarity_residual=0.1,
        )
        broken_complementarity = replace(
            good,
            solutions={**good.solutions, H: bad_complementarity},
        )
        with self.assertRaisesRegex(ValueError, "official|complementarity"):
            project_actions(trace, broken_complementarity, TIMELINE)

        bad_f_probabilities = replace(
            good.solutions[F],
            probabilities={
                R: {N: 0.5, D: 0.5, I: 0.0},
                U: {N: 1.0, D: 0.0, I: 0.0},
            },
        )
        broken_f_probabilities = replace(
            good,
            solutions={**good.solutions, F: bad_f_probabilities},
        )
        with self.assertRaisesRegex(ValueError, "forced_normal"):
            project_actions(trace, broken_f_probabilities, TIMELINE)


if __name__ == "__main__":
    unittest.main()
