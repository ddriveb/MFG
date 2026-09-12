"""Red tests for Stage 3 Expert and social cost accounting."""

from dataclasses import FrozenInstanceError, replace
import math
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.domain import TokenClass
from mfg_hedge.shared_backup import (
    AttemptStatus,
    SharedBackupTrace,
    SharedTokenSpec,
    SharedWorkDraw,
    simulate_shared_backup,
)
from mfg_hedge.transient_objective import cvar95_fractional_tail
from mfg_hedge.expert_game import (
    ExpertEpisodeRun,
    score_expert,
    score_population,
)


R, U = TokenClass.REGULAR, TokenClass.URGENT


def _draw(value=0.1, *, failed_replay=False):
    if failed_replay:
        return SharedWorkDraw(
            (4.0, 1.0, 4.0), (0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2), (0.2, 0.2, 0.2),
        )
    return SharedWorkDraw(
        (value, value, value), (value, value, value),
        (value, value, value), (value, value, value),
    )


def cohort_trace(*, experts=1, work_value=0.1, extra_h_regular=0,
                 expert_scales=None, cutoff=8.0):
    """Create deterministic one-episode H/D/F/R x Regular/U cohorts."""
    timeline = CommonStateTimeline(2.0, 4.0, 6.0)
    rows = []
    for expert in range(experts):
        scale = (expert_scales or {}).get(expert, work_value)
        base = [
            (0.0, R), (0.5, U), (2.0, R), (2.5, U),
            (4.0, R), (4.5, U), (6.0, R), (6.5, U),
        ]
        base.extend((0.7 + 0.2 * n, R) for n in range(
            extra_h_regular if expert == 0 else 0
        ))
        for local, (arrival, token_class) in enumerate(base):
            destination = 1 + ((expert + local) % 2)
            rows.append((arrival, token_class, expert, local, destination, scale))
    rows.sort(key=lambda row: (row[0], row[2], row[3]))
    tokens = []
    draws = []
    local_ids = [0] * experts
    for global_id, (arrival, token_class, expert, _local, destination, scale) in enumerate(rows):
        local = local_ids[expert]
        local_ids[expert] += 1
        tokens.append(SharedTokenSpec(
            global_id, arrival, token_class, expert, local, destination,
        ))
        draws.append(_draw(scale))
    trace = SharedBackupTrace(
        expert_count=experts,
        timeline=timeline,
        arrival_cutoff=cutoff,
        tokens=tuple(tokens),
        work=tuple(draws),
    )
    return trace


def expert_only_trace(expert_id=0):
    source = cohort_trace(experts=2)
    selected = [
        (token, draw)
        for token, draw in zip(source.tokens, source.work)
        if token.expert_id == expert_id
    ]
    tokens = tuple(
        SharedTokenSpec(index, token.arrival_time, token.token_class,
                        expert_id, index, token.destination_replica)
        for index, (token, _draw) in enumerate(selected)
    )
    return replace(source, tokens=tokens,
                   work=tuple(draw for _token, draw in selected))


def run_episode(key, trace, *, c_b=2.0, action_sources=None, fault="fault-0"):
    return ExpertEpisodeRun(
        episode_key=key,
        trace=trace,
        result=simulate_shared_backup(trace, c_b=c_b,
                                      action_sources=action_sources),
        fault_fingerprint=fault,
    )


class ExpertCostTests(unittest.TestCase):
    def test_hand_calculated_complete_objective(self):
        run = run_episode("e0", cohort_trace())
        score = score_expert((run,), expert_id=0)

        self.assertTrue(score.complete)
        self.assertAlmostEqual(score.total, 0.375)
        self.assertAlmostEqual(score.phase_losses[Phase.HEALTHY.value], 0.1)
        self.assertAlmostEqual(score.phase_losses[Phase.DEGRADED.value], 0.2)
        self.assertAlmostEqual(score.phase_losses[Phase.FAILED.value], 0.1)
        self.assertAlmostEqual(score.phase_losses[Phase.RECOVERED.value], 0.1)
        self.assertAlmostEqual(score.components["phase_loss"], 0.125)
        self.assertAlmostEqual(score.components["fault_tail"], 0.15)
        self.assertAlmostEqual(score.components["executed_work_per_token"], 0.1)
        self.assertAlmostEqual(score.components["wasted_work_per_token"], 0.0)

    def test_pooling_uses_token_numerators_not_episode_ratios(self):
        first = run_episode("one", cohort_trace(work_value=0.1))
        second = run_episode(
            "many", cohort_trace(work_value=0.3, extra_h_regular=4)
        )
        score = score_expert((first, second), expert_id=0)
        first_score = score_expert((first,), expert_id=0)
        second_score = score_expert((second,), expert_id=0)
        h_regular = score.cohorts[Phase.HEALTHY.value][TokenClass.REGULAR.value]
        first_h = first_score.cohorts[Phase.HEALTHY.value][TokenClass.REGULAR.value]
        second_h = second_score.cohorts[Phase.HEALTHY.value][TokenClass.REGULAR.value]

        self.assertEqual(h_regular.count, 6)
        self.assertAlmostEqual(
            h_regular.mean_latency,
            (first_h.total_latency + second_h.total_latency) / h_regular.count,
        )
        self.assertNotAlmostEqual(
            h_regular.mean_latency,
            (first_h.mean_latency + second_h.mean_latency) / 2,
        )
        self.assertEqual(score.episode_count, 2)

    def test_empty_episode_is_retained_and_missing_cohort_is_incomplete(self):
        empty = run_episode("empty-for-zero", expert_only_trace(expert_id=0))
        score = score_expert((empty,), expert_id=1)

        self.assertEqual(score.episode_count, 1)
        self.assertFalse(score.complete)
        self.assertIsNone(score.total)
        self.assertIn((Phase.HEALTHY.value, TokenClass.REGULAR.value),
                      score.missing_cohorts)

    def test_fractional_cvar_is_exact_and_uses_small_sample_max_tail(self):
        self.assertAlmostEqual(cvar95_fractional_tail(tuple(range(1, 21))), 20.0)
        self.assertAlmostEqual(
            cvar95_fractional_tail(tuple(range(1, 22))),
            (21.0 * 20.0 + 20.0) / 21.0,
        )
        score = score_expert((run_episode("e0", cohort_trace()),), expert_id=0)
        self.assertAlmostEqual(score.cvar95[Phase.DEGRADED.value], 0.2)
        self.assertAlmostEqual(score.cvar95[Phase.FAILED.value], 0.1)
        self.assertIn(TokenClass.REGULAR.value,
                      score.class_cvar95[Phase.DEGRADED.value])

    def test_miss_rate_is_separate_from_mean_excess(self):
        trace = cohort_trace()
        # Make only the degraded regular token slow enough to miss its deadline.
        draws = list(trace.work)
        token = next(t for t in trace.tokens
                     if t.token_class is R and t.arrival_time == 2.0)
        draws[token.global_token_id] = _draw(2.0)
        run = run_episode("slow", replace(trace, work=tuple(draws)))
        score = score_expert((run,), expert_id=0)
        cohort = score.cohorts[Phase.DEGRADED.value][TokenClass.REGULAR.value]

        self.assertEqual(cohort.miss_count, 1)
        self.assertAlmostEqual(cohort.miss_rate, 1.0)
        self.assertGreater(cohort.mean_excess, 0.0)
        self.assertNotEqual(cohort.miss_rate, cohort.mean_latency)

    def test_failed_primary_and_replay_work_are_accounted_from_attempts(self):
        trace = cohort_trace()
        draws = list(trace.work)
        token = next(t for t in trace.tokens
                     if t.token_class is R and t.arrival_time == 2.0)
        draws[token.global_token_id] = _draw(failed_replay=True)
        # A failure at the D -> F boundary creates executed failed work and a Replay.
        run = run_episode("replay", replace(trace, work=tuple(draws)), c_b=2.0)
        score = score_expert((run,), expert_id=0)
        attempts = run.result.tokens[token.global_token_id].attempts

        self.assertTrue(any(a.status is AttemptStatus.FAILED_RUNNING for a in attempts))
        self.assertGreater(run.result.wasted_work, 0.0)
        self.assertAlmostEqual(
            score.raw_work_totals["executed"], run.result.total_executed_work
        )
        self.assertAlmostEqual(
            score.raw_work_totals["wasted"], run.result.wasted_work
        )
        self.assertGreater(score.components["wasted_work_per_token"], 0.0)

    def test_social_is_mean_of_experts_and_pooled_is_distinct(self):
        trace = cohort_trace(experts=2, expert_scales={0: 0.1, 1: 5.0})
        population = score_population((run_episode("asym", trace),))

        self.assertAlmostEqual(
            population.social_cost_mean_expert_objective,
            sum(s.total for s in population.expert_scores) / 2,
        )
        self.assertNotEqual(
            population.pooled_across_experts_objective,
            population.social_cost_mean_expert_objective,
        )
        individual_tail_mean = sum(
            s.cvar95[Phase.DEGRADED.value] for s in population.expert_scores
        ) / 2
        self.assertNotEqual(
            individual_tail_mean,
            population.pooled_cvar95[Phase.DEGRADED.value],
        )

    def test_runs_are_immutable_and_duplicate_episode_keys_fail(self):
        run = run_episode("same", cohort_trace())
        with self.assertRaises(ValueError):
            score_expert((run, run), expert_id=0)
        with self.assertRaises(FrozenInstanceError):
            run.result.tokens[0].completion_time = math.inf

    def test_trace_result_mismatch_fails_before_scoring(self):
        first = cohort_trace(work_value=0.1)
        second = cohort_trace(work_value=0.2)
        with self.assertRaises(ValueError):
            ExpertEpisodeRun(
                episode_key="bad", trace=first,
                result=simulate_shared_backup(second, c_b=2.0),
            )


if __name__ == "__main__":
    unittest.main()
