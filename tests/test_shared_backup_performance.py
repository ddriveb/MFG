"""Red tests for exact single-scheduler performance paths."""

from dataclasses import FrozenInstanceError
import math
import unittest
from unittest import mock

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import TokenClass
from mfg_hedge.expert_game import (
    ExpertEpisodeRun,
    score_expert,
    score_tagged_result,
    shared_trace_fingerprint,
)
from mfg_hedge.game_workload import ActionRule
from mfg_hedge.shared_backup import (
    _Engine,
    SharedAction,
    simulate_shared_backup,
    simulate_shared_backup_optimized,
    simulate_shared_backup_tagged,
    prepare_shared_trace,
)
from tests.test_shared_backup import trace, work


R = TokenClass.REGULAR
U = TokenClass.URGENT


class _CapturingRule:
    def __init__(self, rule):
        self.rule = rule
        self.observations = []

    @property
    def name(self):
        return self.rule.name

    def request(self, observation):
        self.observations.append(observation)
        return self.rule.request(observation)


class _CompactCapturingRule:
    """Pi256-shaped source whose full-observation path must stay unused."""

    def __init__(self, rule):
        self.rule = rule
        self.compact_calls = []

    def request(self, observation):
        raise AssertionError("compact Pi256 source received a full observation")

    def request_compact(
        self, *, common_state, phase, phase_age, token_class, primary_replica
    ):
        self.compact_calls.append((
            common_state, phase, phase_age, token_class, primary_replica,
        ))
        if (common_state.value != "D" or phase.value != "D"
                or primary_replica != 0):
            return SharedAction.NORMAL
        late = int(phase_age >= 50.0)
        urgent = int(token_class is U)
        return SharedAction(self.rule.actions[2 * late + urgent])


def mixed_trace():
    rows = [
        (0.0, R, 1, 1),
        (0.0, R, 0, 2),
        (10.0, R, 1, 1),
        (10.0, U, 0, 2),
        (15.0, R, 0, 1),
        (15.0, U, 1, 2),
        (20.0, R, 0, 1),
        (20.0, U, 1, 2),
        (31.0, R, 0, 1),
        (31.0, U, 1, 2),
    ]
    draws = [
        work(a0=(1.0, 1.0, 1.0), a1=(1.0, 1.0, 1.0),
             a2=(1.0, 1.0, 1.0), a3=(1.0, 1.0, 1.0))
        for _ in rows
    ]
    return trace(
        rows,
        draws,
        timeline=CommonStateTimeline(10.0, 20.0, 30.0),
        experts=2,
        cutoff=40.0,
    )


def action_sources():
    return {
        0: ActionRule("XDSN", tuple("XDSN")),
        1: ActionRule("NSDX", tuple("NSDX")),
    }


class SharedBackupPerformanceTests(unittest.TestCase):
    def test_compact_pi256_source_matches_online_full_observation_exactly(self):
        input_trace = mixed_trace()
        full_sources = action_sources()
        compact_sources = {
            expert: _CompactCapturingRule(rule)
            for expert, rule in action_sources().items()
        }
        expected = simulate_shared_backup_tagged(
            prepare_shared_trace(input_trace), c_b=0.5, tagged_expert_id=0,
            action_sources=full_sources,
        )
        actual = simulate_shared_backup_tagged(
            prepare_shared_trace(input_trace), c_b=0.5, tagged_expert_id=0,
            action_sources=compact_sources,
        )

        self.assertEqual(expected, actual)
        for expert, source in compact_sources.items():
            self.assertEqual(
                len(source.compact_calls),
                sum(token.expert_id == expert for token in input_trace.tokens),
            )

    def test_tagged_terminal_check_covers_non_tagged_queue_ledgers(self):
        original = _Engine._audit_queues

        def corrupt_non_tagged_ledger(engine, now):
            original(engine, now)
            if engine.evaluation_mode == "tagged":
                engine.remaining_work[(1, 0)] += 0.25

        with mock.patch.object(_Engine, "_audit_queues", corrupt_non_tagged_ledger):
            with self.assertRaisesRegex(RuntimeError, "terminal queue conservation"):
                simulate_shared_backup_tagged(
                    prepare_shared_trace(mixed_trace()), c_b=0.5,
                    tagged_expert_id=0, action_sources=action_sources(),
                )

    def test_optimized_full_matches_reference_and_observations(self):
        source_reference = {
            expert: _CapturingRule(rule)
            for expert, rule in action_sources().items()
        }
        source_optimized = {
            expert: _CapturingRule(rule)
            for expert, rule in action_sources().items()
        }
        input_trace = mixed_trace()
        reference = simulate_shared_backup(
            input_trace, c_b=0.5, action_sources=source_reference,
        )
        optimized = simulate_shared_backup_optimized(
            input_trace, c_b=0.5, action_sources=source_optimized,
        )

        self.assertEqual(reference, optimized)
        self.assertEqual(reference.tokens, optimized.tokens)
        self.assertEqual(reference.attempts, optimized.attempts)
        self.assertEqual(reference.action_decisions, optimized.action_decisions)
        self.assertEqual(reference.pool_intervals, optimized.pool_intervals)
        self.assertEqual(reference.queue_audits, optimized.queue_audits)
        self.assertEqual(reference.queue_audit_events, optimized.queue_audit_events)
        self.assertEqual(reference.counters, optimized.counters)
        self.assertEqual(reference.drain_end_time, optimized.drain_end_time)
        self.assertEqual(reference.trace_fingerprint, optimized.trace_fingerprint)
        self.assertEqual(reference.fault_fingerprint, optimized.fault_fingerprint)
        self.assertTrue(all(event.conservation_holds for event in optimized.queue_audit_events))
        for expert in range(2):
            self.assertEqual(
                source_reference[expert].observations,
                source_optimized[expert].observations,
            )

    def test_prepared_trace_is_immutable_reusable_and_fingerprint_stable(self):
        input_trace = mixed_trace()
        prepared = prepare_shared_trace(input_trace)

        self.assertEqual(prepared.trace, input_trace)
        self.assertEqual(prepared.trace_fingerprint,
                         shared_trace_fingerprint(input_trace))
        with self.assertRaises(FrozenInstanceError):
            prepared.trace = input_trace

        normal = simulate_shared_backup_optimized(
            prepared, c_b=0.5,
            action_sources={0: ActionRule("NNNN", tuple("NNNN")),
                            1: ActionRule("NNNN", tuple("NNNN"))},
        )
        protected = simulate_shared_backup_optimized(
            prepared, c_b=0.5, action_sources=action_sources(),
        )
        self.assertEqual(prepared.trace_fingerprint,
                         shared_trace_fingerprint(input_trace))
        self.assertNotEqual(normal.action_decisions, protected.action_decisions)

    def test_tagged_output_scores_exactly_like_full_reference_projection(self):
        input_trace = mixed_trace()
        prepared = prepare_shared_trace(input_trace)
        sources = action_sources()
        full = simulate_shared_backup(
            input_trace, c_b=0.5, action_sources=sources,
        )
        tagged = simulate_shared_backup_tagged(
            prepared, c_b=0.5, tagged_expert_id=0,
            action_sources=action_sources(),
        )

        self.assertEqual(tagged.tagged_expert_id, 0)
        self.assertTrue(all(token.expert_id == 0 for token in tagged.tokens))
        self.assertTrue(all(attempt.expert_id == 0 for attempt in tagged.attempts))
        self.assertEqual(
            tagged.tokens,
            tuple(token for token in full.tokens if token.expert_id == 0),
        )
        self.assertEqual(
            tagged.action_decisions,
            tuple(decision for decision in full.action_decisions
                  if decision.expert_id == 0),
        )
        expected = score_expert((ExpertEpisodeRun(
            "tagged-equivalence", input_trace, full,
            fault_fingerprint="fault-equivalence",
        ),), 0)
        actual = score_tagged_result(
            tagged, trace=input_trace, expert_id=0,
            episode_key="tagged-equivalence",
            fault_fingerprint="fault-equivalence",
        )
        self.assertEqual(expected, actual)

    def test_tagged_mode_keeps_other_experts_in_physics(self):
        input_trace = mixed_trace()
        tagged = simulate_shared_backup_tagged(
            prepare_shared_trace(input_trace), c_b=0.5,
            tagged_expert_id=0, action_sources=action_sources(),
        )

        self.assertEqual(tagged.counters.token_arrivals, len(input_trace.tokens))
        self.assertEqual(tagged.counters.completion_winners, len(input_trace.tokens))
        self.assertEqual(tagged.counters.primary_enqueues, len(input_trace.tokens))
        self.assertGreater(tagged.pool_audit_summary.interval_count, 0)
        self.assertEqual(tagged.trace_fingerprint,
                         shared_trace_fingerprint(input_trace))

        full = simulate_shared_backup(
            input_trace, c_b=0.5, action_sources=action_sources(),
        )
        self.assertEqual(
            tagged.pool_audit_summary.interval_count,
            len(full.pool_intervals),
        )
        self.assertTrue(math.isclose(
            tagged.pool_audit_summary.total_executed_work,
            math.fsum(interval.executed_work for interval in full.pool_intervals),
            rel_tol=0.0,
            abs_tol=1e-12,
        ))

    def test_tagged_calls_every_expert_online_and_preserves_queue_audit_projection(self):
        sources = {
            expert: _CapturingRule(ActionRule("NNNN", tuple("NNNN")))
            for expert in range(2)
        }
        tagged = simulate_shared_backup_tagged(
            prepare_shared_trace(mixed_trace()), c_b=0.5,
            tagged_expert_id=0, action_sources=sources,
        )
        expected_calls = {
            expert: sum(token.expert_id == expert for token in mixed_trace().tokens)
            for expert in range(2)
        }
        self.assertEqual(
            {expert: len(source.observations) for expert, source in sources.items()},
            expected_calls,
        )
        self.assertTrue(all(event.expert_id == 0 for event in tagged.queue_audit_events))

    def test_optimized_equivalence_covers_losers_replay_and_timer_paths(self):
        timeline = CommonStateTimeline(10.0, 20.0, 100.0)
        cases = (
            (
                [(10.0, R, 0, 1), (10.1, R, 0, 1)],
                [
                    work(a0=(20.0, 9.0, 9.0), a2=(9.0, 0.25, 9.0)),
                    work(a0=(9.0, 9.0, 9.0), a2=(9.0, 0.25, 9.0),
                         a3=(9.0, 9.0, 2.0)),
                ],
                {0: ActionRule("XDSN", tuple("XDSN"))},
                timeline,
            ),
            (
                [(19.0, R, 0, 1), (20.0, R, 1, 2)],
                [
                    work(a0=(2.0, 9.0, 9.0), a1=(3.0, 3.0, 3.0)),
                    work(a0=(9.0, 9.0, 1.0), a1=(1.0, 1.0, 1.0)),
                ],
                {},
                timeline,
            ),
            (
                [(19.0, R, 0, 1)],
                [work(a0=(0.5, 9.0, 9.0), a1=(1.0, 2.0, 2.0))],
                {0: ActionRule("DDDD", tuple("DDDD"))},
                timeline,
            ),
        )
        for rows, draws, sources, case_timeline in cases:
            input_trace = trace(
                rows, draws, timeline=case_timeline,
                experts=2 if len(rows) > 1 else 1,
            )
            reference = simulate_shared_backup(
                input_trace, c_b=0.5, hedge_delay=1.0,
                action_sources=sources,
            )
            optimized = simulate_shared_backup_optimized(
                input_trace, c_b=0.5, hedge_delay=1.0,
                action_sources=sources,
            )
            self.assertEqual(reference, optimized)


if __name__ == "__main__":
    unittest.main()
