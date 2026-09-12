"""Failure-first tests for the frozen Stage 4 campaign orchestration."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.game_workload import ActionRule
from mfg_hedge.stage4_campaign import (
    _delete_one_objectives,
    build_frozen_stage4_scenarios,
    cluster_jackknife_se,
    compute_simultaneous_regret_bound,
    jackknife_pseudo_values,
    preflight_frozen_stage4,
    ParallelFitResult,
    run_frozen_stage4_campaign,
    run_parallel_fit,
    run_parallel_validation,
)
from mfg_hedge.game_solver import SolverDynamicsStatus, SolverPrecisionStatus, SolverStartResult
from tests.test_game_deviations import scenario
from tests.test_expert_game import cohort_trace, run_episode


def rule(name: str) -> ActionRule:
    return ActionRule(name, tuple(name))


class Stage4CampaignTests(unittest.TestCase):
    def test_delete_one_pool_accepts_one_incomplete_standalone_cluster(self):
        full = cohort_trace()
        kept = [
            (token, draw) for token, draw in zip(full.tokens, full.work)
            if not (token.arrival_time == 4.5 and token.token_class.value == "U")
        ]
        missing_fu = replace(
            full,
            tokens=tuple(replace(token, global_token_id=index, local_token_id=index)
                         for index, (token, _draw) in enumerate(kept)),
            work=tuple(draw for _token, draw in kept),
        )
        runs = (
            run_episode("a0", full, fault="fault-a"),
            run_episode("a1", full, fault="fault-a"),
            run_episode("b0", missing_fu, fault="fault-b"),
            run_episode("b1", missing_fu, fault="fault-b"),
            run_episode("c0", full, fault="fault-c"),
            run_episode("c1", full, fault="fault-c"),
        )

        delete_one = _delete_one_objectives(runs, 0)

        self.assertEqual(tuple(key for key, _value in delete_one),
                         ("fault-a", "fault-b", "fault-c"))
        self.assertTrue(all(value is not None for _key, value in delete_one))

    def test_jackknife_se_and_pseudo_values_follow_declared_formula(self):
        delete_one = (1.0, 2.0, 4.0)
        mean = sum(delete_one) / 3.0
        expected = math.sqrt(
            (2.0 / 3.0) * sum((value - mean) ** 2 for value in delete_one)
        )
        self.assertAlmostEqual(cluster_jackknife_se(delete_one), expected)
        self.assertEqual(
            jackknife_pseudo_values(2.5, delete_one),
            (5.5, 3.5, -0.5),
        )

    def test_missing_delete_one_value_is_fail_closed(self):
        self.assertTrue(math.isinf(cluster_jackknife_se((1.0, None, 2.0))))
        self.assertEqual(jackknife_pseudo_values(2.0, (1.0, None)), (None, None))

    def test_frozen_scenario_builder_has_two_nested_populations_and_disjoint_identity(self):
        fit = build_frozen_stage4_scenarios(
            "shared-backup-game:v1:stage4-fit", 20260905, common_path_count=2,
        )
        validation = build_frozen_stage4_scenarios(
            "shared-backup-game:v1:stage4-validation", 20260906, common_path_count=2,
        )
        self.assertEqual(len(fit), 4)
        self.assertEqual(len(validation), 4)
        self.assertEqual(
            tuple(item.episode_key for item in fit),
            tuple(sorted(item.episode_key for item in fit)),
        )
        self.assertTrue({item.fault_fingerprint for item in fit}.isdisjoint(
            {item.fault_fingerprint for item in validation}
        ))

    def test_frozen_call_plan_is_exact_and_rejects_protocol_changes(self):
        plan = preflight_frozen_stage4()
        self.assertEqual(plan.fit_calls, 789504)
        self.assertEqual(plan.validation_calls, 131584)
        self.assertEqual(plan.total_calls, 921088)
        self.assertTrue(plan.within_budget)
        with self.assertRaises(ValueError):
            preflight_frozen_stage4(fit_common_path_count=15)

    def test_simultaneous_bound_is_deterministic_and_reports_all_rows(self):
        observed = {
            (0, "NNNN"): 0.10,
            (0, "NNND"): 0.20,
            (1, "NNNN"): 0.05,
        }
        pseudo_values = {
            (0, "NNNN"): (0.08, 0.12, 0.09, 0.11),
            (0, "NNND"): (0.18, 0.22, 0.19, 0.21),
            (1, "NNNN"): (0.03, 0.07, 0.04, 0.06),
        }
        first = compute_simultaneous_regret_bound(
            observed, pseudo_values,
            targets={0: 0.5, 1: 0.5}, bootstrap_replicates=128,
        )
        second = compute_simultaneous_regret_bound(
            observed, pseudo_values,
            targets={0: 0.5, 1: 0.5}, bootstrap_replicates=128,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.status, "pass")
        self.assertEqual(set(first.upper_bounds), set(observed))

    def test_parallel_fit_adapter_retains_all_bank_rows(self):
        sample = scenario("fit-orchestration")
        result = run_parallel_fit(
            (sample,), expert_count=2,
            starts=(rule("NNNN"), rule("NSSN"), rule("XXXX")), max_rounds=1,
            max_workers=1, fit_call_budget=771,
        )
        self.assertEqual(result.call_count, 771)
        self.assertEqual(len(result.starts), 3)
        self.assertEqual(len(result.starts[0].iterations[0].rule_rows), 256)
        serialized = result.starts[0].iterations[0].as_dict()
        self.assertIn("selected_jackknife_se", serialized)
        self.assertNotIn("selected_se", serialized)
        self.assertIn(
            "paired_delete_one_deltas", serialized["rule_rows"][0]
        )

    def test_parallel_validation_recomputes_all_experts_and_rules(self):
        fit = scenario("fit-validation")
        validation_source = scenario("validation")
        first = validation_source.trace.work[0]
        validation = replace(
            validation_source,
            trace=replace(
                validation_source.trace,
                work=tuple([
                    replace(
                        first,
                        attempt0=(first.attempt0[0] + 0.01, *first.attempt0[1:]),
                    )
                ] + list(validation_source.trace.work[1:])),
            ),
            fault_fingerprint="common-validation-fault",
        )
        result = run_parallel_validation(
            rule("NNNN"), (fit,), (validation,),
            fit_denominators={0: 1.0, 1: 1.0},
            validation_call_budget=514, expert_count=2, max_workers=1, c_b=2.0,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.call_count, 514)
        self.assertEqual(len(result.rows), 512)
        serialized = result.rows[0].as_dict()
        self.assertIn("paired_delete_one_improvements", serialized)
        self.assertIn("jackknife_pseudo_values", serialized)
        self.assertNotIn("paired_cluster_improvements", serialized)
        self.assertEqual(
            [(row.deviating_expert, row.candidate_rule) for row in result.rows[:3]],
            [(0, "NNNN"), (0, "NNND"), (0, "NNNS")],
        )

    def test_failed_fit_does_not_write_a_half_complete_artifact(self):
        starts = tuple(
            SolverStartResult(
                start_rule=name,
                iterations=(),
                visited_rules=(name,),
                candidate_rule=name,
                dynamics_status=SolverDynamicsStatus.EXECUTION_FAILURE,
                precision_status=SolverPrecisionStatus.STATISTICS_INSUFFICIENT,
                stop_reason="execution_failure",
            )
            for name in ("NNNN", "NSSN", "XXXX")
        )
        failed = ParallelFitResult(
            starts=starts,
            call_count=0,
            call_budget=789504,
            fit_denominators=(),
            complete=False,
            status="execution_failed",
            failure_reason="synthetic failure",
        )
        with tempfile.TemporaryDirectory() as root, patch(
            "mfg_hedge.stage4_campaign.run_parallel_fit", return_value=failed
        ):
            result = run_frozen_stage4_campaign(
                artifacts_root=Path(root),
                run_id="failed-stage4-run",
                max_workers=1,
            )
            self.assertEqual(result.status, "execution_failed")
            self.assertIsNone(result.artifact_run_dir)
            self.assertEqual(tuple(Path(root).iterdir()), ())


if __name__ == "__main__":
    unittest.main()
