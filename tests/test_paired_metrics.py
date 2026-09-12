"""Paired metrics tests, including a fully hand-computed A/B comparison."""

from dataclasses import replace
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.config import load_config
from mfg_hedge.domain import CommonState, ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.mfg_solver import MFGSolution, SolverParameters, SolverStatus, StatePolicySolution
from mfg_hedge.paired import verify_trace_identity
from mfg_hedge.paired_metrics import _storm_metrics, build_arm_summary, build_comparison
from mfg_hedge.quota import project_actions
from mfg_hedge.workload import TokenSpec, WorkloadTrace


N, D, I = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
H, F = CommonState.HEALTHY, CommonState.FAILED
DD = CommonState.DEGRADED
R, U = TokenClass.REGULAR, TokenClass.URGENT
TIMELINE = CommonStateTimeline(10.0, 20.0, 30.0)
SLOWDOWN = 2.0
CAPACITIES = {"H": 2.0, "D": 1.5, "F": 1.0, "R": 2.0}


def manual_trace() -> WorkloadTrace:
    tokens = (
        TokenSpec(0, 12.0, R),
        TokenSpec(1, 12.5, U),
    )
    ones = (1.0, 1.0)
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.0,
        tokens=tokens,
        service_times=((100.0, 1.0), (1.0, 1.0)),
        replay_service_times=(ones, (2.0, 1.0)),
        hedge_service_times=(ones, (1.5, 1.0)),
    )


def stats_g(action):
    work = {N: 0.0, D: 0.2, I: 0.5}[action]
    return ActionStats_like(work)


def ActionStats_like(hedge_work):
    from mfg_hedge.domain import ActionStats

    return ActionStats(
        mean_latency=1.0,
        replay_probability=0.0,
        deadline_miss_probability=0.0,
        expected_hedge_work=hedge_work,
        expected_replay_work=0.0,
    )


def state_solution(state, probs, budget_rate):
    stats = {cls: {a: ActionStats_like(0.5 if a is I else (0.2 if a is D else 0.0)) for a in (N, D, I)} for cls in (R, U)}
    return StatePolicySolution(
        state=state,
        status=SolverStatus.FORCED_NORMAL if state is F else SolverStatus.CONVERGED,
        reason="ok",
        grid_clamped=False,
        probabilities=probs,
        price=0.0,
        rho=0.5,
        primary_work_rate=1.0,
        hedge_work_rate=0.0,
        replay_work_rate=0.0,
        total_work_rate=1.0,
        target_work_rate=2.0,
        capacity_gap=-1.0,
        capacity_violation=0.0,
        capacity_slack=1.0,
        complementarity_residual=0.0,
        hedge_budget_rate=budget_rate,
        policy_residual=0.0,
        price_residual=0.0,
        load_residual=0.0,
        iterations=1,
        outer_evaluations=1,
        price_evaluations=((0.0, -1.0),),
        structural_capacity_violation=False,
        final_stats=stats,
    )


def fixture_solution() -> MFGSolution:
    all_normal = {R: {N: 1.0, D: 0.0, I: 0.0}, U: {N: 1.0, D: 0.0, I: 0.0}}
    d_policy = {R: {N: 0.0, D: 0.0, I: 1.0}, U: {N: 1.0, D: 0.0, I: 0.0}}
    return MFGSolution(
        solutions={
            H: state_solution(H, all_normal, 1.0),
            DD: state_solution(DD, d_policy, 0.05),
            F: state_solution(F, all_normal, 0.0),
        },
        parameters=SolverParameters(
            inverse_temperature=5.0,
            policy_damping=0.2,
            price_step=0.05,
            price_damping=0.2,
        ),
    )


def run_arms():
    trace = manual_trace()
    solution = fixture_solution()
    projection = project_actions(trace, solution, TIMELINE)
    arm_a = simulate_hedge_common_state(trace, timeline=TIMELINE, degraded_slowdown=SLOWDOWN)
    arm_b = simulate_hedge_common_state(
        trace,
        timeline=TIMELINE,
        degraded_slowdown=SLOWDOWN,
        actions=projection.to_action_plan(),
        hedge_delay=1.5,
    )
    return trace, projection, arm_a, arm_b


class HandComputedComparisonTests(unittest.TestCase):
    """Spec scenario 20: manual-trace deltas computed by hand."""

    def setUp(self) -> None:
        self.trace, self.projection, self.arm_a, self.arm_b = run_arms()
        self.summary_a = build_arm_summary(
            self.arm_a, TIMELINE, arm="no_hedge", projection=None,
            bin_width=1.0, capacities=CAPACITIES,
        )
        self.summary_b = build_arm_summary(
            self.arm_b, TIMELINE, arm="mfg_hedge", projection=self.projection,
            bin_width=1.0, capacities=CAPACITIES,
        )
        self.comparison = build_comparison(self.summary_a, self.summary_b)

    def test_arm_a_baseline(self) -> None:
        a = self.summary_a
        self.assertAlmostEqual(a["mean_latency"], 5.5)
        self.assertAlmostEqual(a["replay_rate"], 0.5)
        self.assertAlmostEqual(a["drain_end_time"], 22.0)
        self.assertEqual(a["winners"], {"primary": 1, "hedge": 0, "replay": 1})
        work = a["work"]
        self.assertAlmostEqual(work["nominal_primary_work"], 101.0)
        self.assertAlmostEqual(work["executed_primary_work"], 5.0)
        self.assertAlmostEqual(work["wasted_work"], 4.0)
        self.assertAlmostEqual(work["executed_replay_work"], 2.0)
        self.assertAlmostEqual(work["total_executed_work"], 7.0)
        self.assertAlmostEqual(work["extra_execution_ratio"], 0.0)

    def test_arm_b_hedged(self) -> None:
        b = self.summary_b
        self.assertAlmostEqual(b["mean_latency"], 1.75)
        self.assertAlmostEqual(b["replay_rate"], 0.0)
        self.assertEqual(b["winners"], {"primary": 1, "hedge": 1, "replay": 0})
        work = b["work"]
        self.assertAlmostEqual(work["executed_hedge_work"], 1.5)
        self.assertAlmostEqual(work["executed_primary_work"], 5.0)
        self.assertAlmostEqual(work["wasted_work"], 4.0)
        self.assertAlmostEqual(work["total_executed_work"], 6.5)
        self.assertAlmostEqual(work["extra_execution_ratio"], 1.5 / 101.0)
        self.assertAlmostEqual(work["execution_amplification"], 6.5 / 101.0)
        self.assertEqual(b["hedge"]["launched"], 1)
        self.assertEqual(b["hedge"]["cancelled_queued"], 0)

    def test_comparison_deltas(self) -> None:
        c = self.comparison["metrics"]
        self.assertAlmostEqual(c["mean_latency"]["arm_a"], 5.5)
        self.assertAlmostEqual(c["mean_latency"]["arm_b"], 1.75)
        self.assertAlmostEqual(c["mean_latency"]["delta"], -3.75)
        self.assertAlmostEqual(c["mean_latency"]["relative"], -3.75 / 5.5)
        self.assertAlmostEqual(c["replay_rate"]["delta"], -0.5)
        self.assertAlmostEqual(c["replay_rate"]["relative"], -1.0)
        self.assertIsNone(c["hedge_launches"]["relative"])  # arm A is zero

    def test_migration_matrix(self) -> None:
        matrix_a = self.summary_a["migration_matrix"]
        self.assertEqual(matrix_a["D"]["F"], 1)  # tok0 arrives D, completes F
        self.assertEqual(matrix_a["D"]["D"], 1)  # tok1 arrives D, completes D
        matrix_b = self.summary_b["migration_matrix"]
        self.assertEqual(matrix_b["D"]["D"], 2)  # both complete in D when hedged

    def test_storm_rules(self) -> None:
        storm_a = self.summary_a["storm"]
        self.assertIsNone(storm_a["peak_hedge_launch_rate"])
        self.assertIsNone(storm_a["peak_to_mean_hedge_launch_ratio"])
        storm_b = self.summary_b["storm"]
        self.assertAlmostEqual(storm_b["peak_hedge_launch_rate"], 1.0)
        self.assertAlmostEqual(storm_b["mean_hedge_launch_rate"], 1.0 / 20.0)
        self.assertAlmostEqual(storm_b["peak_to_mean_hedge_launch_ratio"], 20.0)
        self.assertAlmostEqual(storm_b["sustained_overload_duration"], 1.0)

    def test_storm_work_binning_scans_tokens_once(self) -> None:
        class CountingTuple(tuple):
            iterations = 0

            def __iter__(self):
                type(self).iterations += 1
                return super().__iter__()

        counted = CountingTuple(self.arm_a.tokens)
        long_horizon = replace(
            self.arm_a,
            tokens=counted,
            drain_end_time=10_000.5,
        )
        _storm_metrics(long_horizon, TIMELINE, 1.0, CAPACITIES)
        self.assertLessEqual(CountingTuple.iterations, 1)

    def test_planned_vs_realized_window(self) -> None:
        windows = self.summary_b["planned_vs_realized"]
        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual((window["window_start"], window["window_end"]), (10.0, 20.0))
        self.assertAlmostEqual(window["planned_expected_hedge_work"], 0.5)
        self.assertAlmostEqual(window["realized_hedge_work"], 1.5)
        self.assertAlmostEqual(window["realized_excess"], 1.0)
        self.assertTrue(window["realized_violation"])
        self.assertEqual(window["quota_suppressed"], 0)
        self.assertEqual(window["executor_hedge_suppressed"], 0)

    def test_per_class_and_global_action_counts(self) -> None:
        actions = self.summary_b["actions"]
        self.assertEqual(actions["global"]["requested_counts"]["I"], 1)
        self.assertEqual(actions["global"]["applied_counts"]["I"], 1)
        self.assertEqual(actions["global"]["quota_suppressed"], 0)
        per_class = actions["per_class"]
        self.assertEqual(per_class["R"]["requested_counts"]["I"], 1)
        self.assertEqual(per_class["U"]["requested_counts"]["N"], 1)


class TraceIdentityTests(unittest.TestCase):
    def test_identical_traces_pass(self) -> None:
        trace = manual_trace()
        verify_trace_identity(trace, trace)

    def test_perturbed_streams_fail_loudly(self) -> None:
        from dataclasses import replace

        trace = manual_trace()
        corrupted = replace(
            trace, service_times=((100.0, 1.0), (1.5, 1.0))
        )
        with self.assertRaisesRegex(ValueError, "trace identity"):
            verify_trace_identity(trace, corrupted)
        bad_tokens = (TokenSpec(0, 12.0, R), TokenSpec(1, 12.6, U))
        with self.assertRaisesRegex(ValueError, "trace identity"):
            verify_trace_identity(trace, replace(trace, tokens=bad_tokens))


if __name__ == "__main__":
    unittest.main()
