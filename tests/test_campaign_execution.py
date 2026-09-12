"""Red tests for the Windows deterministic process batcher."""

import multiprocessing
import unittest

from mfg_hedge.campaign_execution import (
    DeviationTask,
    run_deviation_batch,
    validate_canonical_rule_bank,
)
from mfg_hedge.game_workload import ActionRule, enumerate_action_rules
from tests.test_game_deviations import scenario


def rule(name):
    return ActionRule(name, tuple(name))


class ExplodingRule(ActionRule):
    def request(self, observation):
        raise RuntimeError("intentional worker failure")


class CampaignExecutionTests(unittest.TestCase):
    def test_one_and_two_worker_batches_have_fixed_identical_rows(self):
        sample = scenario("batch")
        tasks = (
            DeviationTask(0, rule("NNNN"), rule("NNNN")),
            DeviationTask(0, rule("SSSS"), rule("NNNN")),
        )
        serial = run_deviation_batch(
            (sample,), tasks, expert_count=2, c_b=2.0,
            max_workers=1, call_budget=10,
        )
        parallel = run_deviation_batch(
            (sample,), tasks, expert_count=2, c_b=2.0,
            max_workers=2, call_budget=10,
        )

        self.assertTrue(serial.complete)
        self.assertTrue(parallel.complete)
        self.assertEqual(serial.rows, parallel.rows)
        self.assertEqual(serial.merge_keys,
                         ((0, "NNNN"), (0, "SSSS")))
        self.assertEqual(serial.reserved_calls, 2)
        self.assertEqual(serial.completed_calls, 2)
        self.assertEqual(serial.failed_calls, 0)
        self.assertEqual(serial.rows, tuple(sorted(serial.rows,
                         key=lambda row: (row.deviating_expert,
                                           row.candidate_rule))))
        self.assertTrue(serial.rss_supported)
        self.assertEqual(serial.start_method, "spawn")
        self.assertEqual(serial.task_granularity,
                         "(deviating_expert, candidate_rule)")
        self.assertTrue(serial.rows[0].scenario_scores[0].invariant_counters)
        self.assertTrue(serial.rows[0].scenario_scores[0].pool_audit_summary)
        self.assertIsNotNone(serial.rows[0].scenario_scores[0].drain_end_time)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_completion_order_does_not_change_parent_merge(self):
        sample = scenario("order")
        tasks = tuple(
            DeviationTask(0, rule(name), rule("NNNN"))
            for name in ("SSSS", "NNNN", "XXXX", "DDDD")
        )
        first = run_deviation_batch(
            (sample,), tasks, expert_count=2, c_b=2.0,
            max_workers=2, call_budget=10,
        )
        second = run_deviation_batch(
            (sample,), tuple(reversed(tasks)), expert_count=2, c_b=2.0,
            max_workers=2, call_budget=10,
        )
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.merge_keys, second.merge_keys)

    def test_budget_reservation_stops_before_an_extra_task(self):
        sample = scenario("budget-batch")
        tasks = tuple(
            DeviationTask(0, rule(name), rule("NNNN"))
            for name in ("NNNN", "SSSS", "XXXX")
        )
        result = run_deviation_batch(
            (sample,), tasks, expert_count=2, c_b=2.0,
            max_workers=2, call_budget=2,
        )

        self.assertFalse(result.complete)
        self.assertEqual(result.reserved_calls, 2)
        self.assertEqual(result.completed_calls, 2)
        self.assertEqual(result.failed_calls, 0)
        self.assertEqual(len(result.rows), 2)
        self.assertIn("budget", result.failure_reason)

    def test_invalid_duplicate_task_keys_fail_before_dispatch(self):
        sample = scenario("duplicate")
        task = DeviationTask(0, rule("NNNN"), rule("NNNN"))
        with self.assertRaises(ValueError):
            run_deviation_batch(
                (sample,), (task, task), expert_count=2,
                c_b=2.0, max_workers=1, call_budget=10,
            )

    def test_canonical_rule_bank_rejects_missing_duplicate_and_reordered_rows(self):
        bank = enumerate_action_rules()
        self.assertEqual(validate_canonical_rule_bank(bank), bank)
        with self.assertRaises(ValueError):
            validate_canonical_rule_bank(bank[:-1])
        with self.assertRaises(ValueError):
            validate_canonical_rule_bank(bank[:-1] + (bank[0],))
        with self.assertRaises(ValueError):
            validate_canonical_rule_bank(tuple(reversed(bank)))

    def test_invalid_parameters_fail_before_dispatch(self):
        sample = scenario("worker-failure")
        with self.assertRaises(ValueError):
            run_deviation_batch(
                (sample,), (DeviationTask(0, rule("NNNN"), rule("NNNN")),),
                expert_count=2, c_b=0.0,
                max_workers=1, call_budget=10,
            )

    def test_worker_exception_is_recorded_once_without_retry_or_candidate(self):
        sample = scenario("worker-exception")
        task = DeviationTask(
            0, ExplodingRule("NNNN", tuple("NNNN")), rule("NNNN")
        )
        result = run_deviation_batch(
            (sample,), (task,), expert_count=2, c_b=2.0,
            max_workers=1, call_budget=10,
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.rows, ())
        self.assertEqual(result.reserved_calls, 1)
        self.assertEqual(result.completed_calls, 0)
        self.assertEqual(result.failed_calls, 1)
        self.assertIn("without retry", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
