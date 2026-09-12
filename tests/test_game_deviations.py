"""Red tests for Stage 3 finite-system unilateral deviations."""

from dataclasses import replace
import math
import time
import unittest

from mfg_hedge.game_workload import ActionRule, enumerate_action_rules
from mfg_hedge.game_deviations import (
    DeviationScenario,
    evaluate_unilateral_deviations,
    evaluate_pi256,
    timing_smoke_pi256,
)
from mfg_hedge.shared_backup import SharedAction
from mfg_hedge.shared_backup import SharedWorkDraw
from mfg_hedge.expert_game import score_population
from tests.test_expert_game import cohort_trace


def rule(name):
    return ActionRule(name, tuple(name))


class RecordingRule:
    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    @property
    def name(self):
        return self.rule.name

    def request(self, observation):
        self.calls += 1
        return self.rule.request(observation)


def scenario(key="s0"):
    trace = cohort_trace(experts=2, work_value=0.1)
    draws = list(trace.work)
    for token in trace.tokens:
        if token.expert_id == 0 and token.arrival_time in (2.0, 2.5):
            draw = draws[token.global_token_id]
            draws[token.global_token_id] = SharedWorkDraw(
                draw.attempt0, draw.attempt1, (5.0, 5.0, 5.0), draw.attempt3,
            )
    return DeviationScenario(
        episode_key=key,
        trace=replace(trace, work=tuple(draws)),
        fault_fingerprint="common-fault-17",
    )


class FiniteDeviationTests(unittest.TestCase):
    def test_candidate_subset_includes_incumbent_and_is_sorted(self):
        incumbent = rule("NNNN")
        result = evaluate_unilateral_deviations(
            (scenario(),),
            profile={0: incumbent, 1: rule("NNNN")},
            deviating_expert=0,
            candidates=(rule("SSSS"), incumbent, rule("DNNN")),
            c_b=2.0,
        )
        self.assertEqual([row.candidate_rule for row in result.rows],
                         ["NNNN", "DNNN", "SSSS"])
        self.assertEqual(result.rows[1].base_objective,
                         result.rows[1].deviated_objective)

    def test_deviation_reruns_shared_physics_and_changes_peer_latency(self):
        opponent = RecordingRule(rule("NNNN"))
        result = evaluate_unilateral_deviations(
            (scenario(),),
            profile={0: rule("NNNN"), 1: opponent},
            deviating_expert=0,
            candidates=(rule("NNNN"), rule("SSSS")),
            c_b=0.5,
        )
        changed = next(row for row in result.rows if row.candidate_rule == "SSSS")

        self.assertGreater(opponent.calls, 0)
        self.assertTrue(changed.complete)
        self.assertNotEqual(changed.peer_cost_delta[1], 0.0)
        self.assertNotEqual(changed.base_trace_fingerprint, "")
        self.assertEqual(changed.base_fault_fingerprints,
                         changed.deviated_fault_fingerprints)
        self.assertEqual(changed.base_trace_fingerprint,
                         changed.deviated_trace_fingerprint)
        self.assertGreaterEqual(changed.social_delta, -math.inf)

    def test_counterfactual_keeps_exogenous_trace_but_not_endogenous_pool(self):
        result = evaluate_unilateral_deviations(
            (scenario(),),
            profile={0: rule("NNNN"), 1: rule("NNNN")},
            deviating_expert=0,
            candidates=(rule("NNNN"), rule("SSSS")),
            c_b=0.5,
        )
        row = next(row for row in result.rows if row.candidate_rule == "SSSS")
        self.assertEqual(row.trace_fingerprints[0], row.deviated_trace_fingerprints[0])
        self.assertNotEqual(row.base_pool_integral, row.deviated_pool_integral)

    def test_rejects_incomplete_scorer_and_missing_incumbent(self):
        with self.assertRaises(ValueError):
            evaluate_unilateral_deviations(
                (scenario(),),
                profile={0: rule("NNNN"), 1: rule("NNNN")},
                deviating_expert=0,
                candidates=(rule("SSSS"),),
            )
        with self.assertRaises(ValueError):
            evaluate_unilateral_deviations(
                (scenario(),),
                profile={0: rule("NNNN")},
                deviating_expert=0,
                candidates=(rule("NNNN"),),
            )

    def test_best_response_and_regret_are_point_estimates_only(self):
        result = evaluate_unilateral_deviations(
            (scenario(),),
            profile={0: rule("NNNN"), 1: rule("NNNN")},
            deviating_expert=0,
            candidates=(rule("NNNN"), rule("SSSS")),
        )
        self.assertIsNotNone(result.best_response)
        self.assertGreaterEqual(result.regret, 0.0)
        self.assertGreater(result.normalization_denominator, 0.0)
        self.assertEqual(result.regret_definition, "Pi256 restricted point estimate")
        self.assertFalse(result.claims_nash)

    def test_official_pi256_bank_is_exact_and_lexical(self):
        result = evaluate_pi256(
            (scenario(),),
            profile={0: rule("NNNN"), 1: rule("NNNN")},
            c_b=2.0,
        )
        self.assertEqual(len(result.rows), 2 * 256)
        self.assertEqual(result.candidate_bank_size, 256)
        self.assertEqual(result.rows[0].deviating_expert, 0)
        self.assertEqual(result.rows[-1].deviating_expert, 1)
        self.assertEqual(result.rows[0].candidate_rule, "NNNN")
        self.assertEqual(result.rows[255].candidate_rule, "XXXX")
        self.assertTrue(all(row.complete for row in result.rows))

    def test_pi256_rejects_missing_duplicate_or_noncanonical_bank(self):
        bank = enumerate_action_rules()
        with self.assertRaises(ValueError):
            evaluate_pi256(
                (scenario(),),
                profile={0: rule("NNNN"), 1: rule("NNNN")},
                candidates=bank[:-1],
            )
        with self.assertRaises(ValueError):
            evaluate_pi256(
                (scenario(),),
                profile={0: rule("NNNN"), 1: rule("NNNN")},
                candidates=bank[:-1] + (bank[0],),
            )

    def test_all_experts_system_max_regret_is_reported(self):
        result = evaluate_pi256(
            (scenario(),),
            profile={0: rule("NNNN"), 1: rule("NNNN")},
        )
        self.assertEqual(result.system_max_individual_regret,
                         max(result.per_expert_regret.values()))
        self.assertEqual(set(result.per_expert_regret), {0, 1})

    def test_repeated_evaluation_is_bytewise_deterministic(self):
        kwargs = dict(
            scenarios=(scenario(),),
            profile={0: rule("NNNN"), 1: rule("NNNN")},
            deviating_expert=0,
            candidates=(rule("NNNN"), rule("SSSS")),
        )
        first = evaluate_unilateral_deviations(**kwargs)
        second = evaluate_unilateral_deviations(**kwargs)
        self.assertEqual(first, second)
        self.assertNotIn("timestamp", repr(first))

    def test_timing_smoke_is_in_memory_and_does_not_select_equilibrium(self):
        started = time.perf_counter()
        smoke = timing_smoke_pi256((scenario(),), profile={
            0: rule("NNNN"), 1: rule("NNNN"),
        })
        elapsed = time.perf_counter() - started
        self.assertEqual(smoke.row_count, 512)
        self.assertTrue(smoke.all_complete)
        self.assertTrue(smoke.deterministic_repeat)
        self.assertTrue(math.isfinite(smoke.runtime_seconds))
        self.assertLess(smoke.runtime_seconds, elapsed + 1.0)
        self.assertIsNone(smoke.selected_equilibrium)


if __name__ == "__main__":
    unittest.main()
