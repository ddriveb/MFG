"""Red tests for the bounded Stage 4 Pi256 solver state machine."""

from dataclasses import replace
import math
import unittest

from mfg_hedge.game_workload import ActionRule, enumerate_action_rules
from mfg_hedge.game_deviations import DeviationScenario
from mfg_hedge.expert_game import shared_trace_fingerprint
from mfg_hedge.shared_backup import SharedWorkDraw
from mfg_hedge.game_solver import (
    BankEvaluation,
    SolverDynamicsStatus,
    SolverPrecisionStatus,
    SolverRuleRow,
    measure_scheduler_timing,
    preflight_stage4,
    solve_symmetric_pi256,
    validate_frozen_candidate,
)
from tests.test_game_deviations import scenario as deviation_scenario


def rule(name):
    return ActionRule(name, tuple(name))


def stub_bank(targets, *, deltas=(0.0, 0.0)):
    bank = enumerate_action_rules()

    def evaluate(current):
        target = targets[current.name]
        rows = tuple(
            SolverRuleRow(
                rule=candidate,
                objective=0.0 if candidate.name == target else 1.0,
                executed_work=0.0 if candidate.name == target else 1.0,
                paired_delete_one_deltas=tuple(deltas),
            )
            for candidate in bank
        )
        return BankEvaluation(
            current_rule=current.name,
            rows=rows,
            environment_fingerprint=f"env:{current.name}",
        )

    return evaluate


class SolverStateMachineTests(unittest.TestCase):
    def test_three_starts_retain_full_rows_and_reach_pure_fixed_points(self):
        evaluator = stub_bank({"NNNN": "NNNN", "NSSN": "NSSN", "XXXX": "XXXX"})
        result = solve_symmetric_pi256(
            expert_count=2,
            starts=(rule("NNNN"), rule("NSSN"), rule("XXXX")),
            max_rounds=32,
            precision_denominator=1.0,
            candidate_evaluator=evaluator,
        )

        self.assertEqual(result.start_count, 3)
        self.assertEqual([start.dynamics_status for start in result.starts], [
            SolverDynamicsStatus.PURE_FIXED_POINT,
            SolverDynamicsStatus.PURE_FIXED_POINT,
            SolverDynamicsStatus.PURE_FIXED_POINT,
        ])
        self.assertEqual([len(start.iterations[0].rule_rows) for start in result.starts],
                         [256, 256, 256])
        self.assertEqual(result.starts[0].iterations[0].environment_fingerprint,
                         "env:NNNN")
        self.assertEqual(result.starts[0].stop_reason, "pure_fixed_point")
        self.assertFalse(result.claims_nash)
        self.assertFalse(result.claims_mfg)

    def test_best_response_tie_breaks_by_work_then_canonical_order(self):
        bank = enumerate_action_rules()

        def evaluator(current):
            rows = tuple(
                SolverRuleRow(
                    rule=candidate,
                    objective=0.0,
                    executed_work=1.0 if candidate.name == "NNNN" else 0.0,
                    paired_delete_one_deltas=(0.0, 0.0),
                )
                for candidate in bank
            )
            return BankEvaluation(
                current_rule=current.name,
                rows=rows,
                environment_fingerprint=f"env:{current.name}",
            )

        result = solve_symmetric_pi256(
            expert_count=2,
            starts=(rule("NNNN"),),
            max_rounds=1,
            precision_denominator=1.0,
            candidate_evaluator=evaluator,
        )

        iteration = result.starts[0].iterations[0]
        self.assertEqual(iteration.selected_rule, "NNND")
        self.assertEqual(iteration.runner_up_rule, "NNNS")

    def test_repeated_rule_is_retained_as_cycle_with_first_repeat_and_period(self):
        evaluator = stub_bank({"NNNN": "NSSN", "NSSN": "NNNN", "XXXX": "XXXX"})
        result = solve_symmetric_pi256(
            expert_count=2,
            starts=(rule("NNNN"), rule("NSSN"), rule("XXXX")),
            precision_denominator=1.0,
            candidate_evaluator=evaluator,
        )

        cycle = result.starts[0]
        self.assertEqual(cycle.dynamics_status, SolverDynamicsStatus.RULE_CYCLE)
        self.assertEqual(cycle.cycle_first_repeat, 0)
        self.assertEqual(cycle.cycle_period, 2)
        self.assertEqual(cycle.visited_rules, ("NNNN", "NSSN"))
        self.assertEqual(cycle.stop_reason, "rule_cycle")
        self.assertEqual(len(cycle.iterations), 2)

    def test_thirty_two_round_nonconvergence_is_not_called_a_fixed_point(self):
        bank = enumerate_action_rules()
        chain = tuple(bank[:33])
        targets = {
            current.name: chain[index + 1].name
            for index, current in enumerate(chain[:-1])
        }
        targets[chain[-1].name] = chain[-1].name
        evaluator = stub_bank(targets)
        result = solve_symmetric_pi256(
            expert_count=2,
            starts=(chain[0], chain[1], chain[2]),
            max_rounds=32,
            precision_denominator=1.0,
            candidate_evaluator=evaluator,
        )

        self.assertEqual(result.starts[0].dynamics_status,
                         SolverDynamicsStatus.NOT_CONVERGED_32)
        self.assertEqual(len(result.starts[0].iterations), 32)
        self.assertEqual(result.starts[0].stop_reason, "not_converged_32")

    def test_paired_jackknife_se_controls_statistics_insufficient(self):
        evaluator = stub_bank(
            {"NNNN": "NNNN", "NSSN": "NSSN", "XXXX": "XXXX"},
            deltas=(0.0, 1.0),
        )
        result = solve_symmetric_pi256(
            expert_count=2,
            starts=(rule("NNNN"), rule("NSSN"), rule("XXXX")),
            precision_denominator=1.0,
            candidate_evaluator=evaluator,
        )

        self.assertEqual(result.starts[0].precision_status,
                         SolverPrecisionStatus.STATISTICS_INSUFFICIENT)
        self.assertAlmostEqual(result.starts[0].iterations[0].selected_gap, 1.0)
        # The winner-vs-runner-up uncertainty is computed from the paired
        # cluster differences.  Identical candidate deltas therefore have SE 0
        # even though each candidate's marginal cluster SE is nonzero.
        self.assertEqual(result.starts[0].iterations[0].selected_jackknife_se, 0.0)
        self.assertGreater(
            result.starts[0].iterations[0].max_candidate_jackknife_se, 0.005
        )
        self.assertEqual(result.starts[0].iterations[0].precision_reason,
                         "paired_jackknife_se_or_gap")

    def test_call_budget_stops_before_a_scheduler_call_overflow(self):
        small = deviation_scenario("budget")
        result = solve_symmetric_pi256(
            fit_scenarios=(small,),
            expert_count=2,
            starts=(rule("NNNN"),),
            max_rounds=2,
            call_budget=10,
            c_b=2.0,
        )

        self.assertEqual(result.call_count, 10)
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.starts[0].dynamics_status,
                         SolverDynamicsStatus.BUDGET_EXHAUSTED)
        self.assertEqual(result.stop_reason, "call_budget_exhausted")

    def test_real_solver_is_deterministic_and_does_not_mutate_fit_trace(self):
        sample = deviation_scenario("deterministic")
        before = shared_trace_fingerprint(sample.trace)
        kwargs = {
            "fit_scenarios": (sample,),
            "expert_count": 2,
            "starts": (rule("NNNN"),),
            "max_rounds": 1,
            "precision_denominator": 1.0,
            "call_budget": 257,
            "c_b": 2.0,
        }
        first = solve_symmetric_pi256(**kwargs)
        second = solve_symmetric_pi256(**kwargs)

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(before, shared_trace_fingerprint(sample.trace))
        self.assertEqual(first.call_count, 257)
        self.assertFalse(first.budget_exhausted)

    def test_call_count_preflight_closes_the_frozen_arithmetic(self):
        plan = preflight_stage4(
            fit_scenario_count=32,
            validation_scenario_count=64,
            expert_count=8,
            starts=3,
            max_rounds=32,
            call_budget=1_000_000,
        )

        self.assertEqual(plan.fit_calls, 789_504)
        self.assertEqual(plan.validation_calls, 131_584)
        self.assertEqual(plan.total_calls, 921_088)
        self.assertTrue(plan.within_budget)
        self.assertEqual(plan.remaining_budget, 78_912)

    def test_timing_preflight_is_finite_and_timestamp_free(self):
        sample = deviation_scenario("timing")
        timing = measure_scheduler_timing(
            sample,
            profile={0: rule("NNNN"), 1: rule("NNNN")},
            c_b=2.0,
        )

        self.assertEqual(timing.scheduler_calls, 1)
        self.assertTrue(math.isfinite(timing.runtime_seconds))
        self.assertGreater(timing.runtime_seconds, 0.0)
        self.assertNotIn("timestamp", repr(timing))

    def test_validation_requires_disjoint_fingerprints_and_runs_all_experts(self):
        fit = deviation_scenario("fit")
        validation_source = deviation_scenario("validation")
        validation_draws = list(validation_source.trace.work)
        first = validation_draws[0]
        validation_draws[0] = SharedWorkDraw(
            attempt0=(first.attempt0[0] + 0.01, *first.attempt0[1:]),
            attempt1=first.attempt1,
            attempt2=first.attempt2,
            attempt3=first.attempt3,
        )
        validation = replace(
            validation_source,
            trace=replace(validation_source.trace, work=tuple(validation_draws)),
            fault_fingerprint="common-fault-validation",
        )
        result = validate_frozen_candidate(
            candidate_rule=rule("NNNN"),
            fit_scenarios=(fit,),
            validation_scenarios=(validation,),
            c_b=2.0,
            call_budget=3_000,
        )

        self.assertEqual(result.row_count, 2 * 256)
        self.assertEqual(result.call_count, 2 * 257)
        self.assertEqual(result.expert_count, 2)
        self.assertTrue(result.all_experts)
        self.assertTrue(result.complete)
        self.assertFalse(result.claims_nash)
        self.assertEqual(result.precision_status,
                         SolverPrecisionStatus.SIMULTANEOUS_BOUND_PENDING)

        with self.assertRaises(ValueError):
            validate_frozen_candidate(
                candidate_rule=rule("NNNN"),
                fit_scenarios=(fit,),
                validation_scenarios=(fit,),
                c_b=2.0,
                call_budget=3_000,
            )


if __name__ == "__main__":
    unittest.main()
