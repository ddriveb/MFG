"""Red/green tests for the isolated Shared-Backup physics slice."""

from dataclasses import replace
import math
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.domain import TokenClass
from mfg_hedge.shared_backup import (
    ActionObservation,
    AttemptStatus,
    SharedAction,
    SharedBackupTrace,
    SharedTokenSpec,
    SharedWorkDraw,
    simulate_shared_backup,
)


N, D, S, X = (SharedAction.NORMAL, SharedAction.DELAYED,
              SharedAction.SINGLE, SharedAction.DUAL)
R, U = TokenClass.REGULAR, TokenClass.URGENT


def work(a0=(1.0, 1.0, 1.0), a1=(1.0, 1.0, 1.0),
         a2=(1.0, 1.0, 1.0), a3=(1.0, 1.0, 1.0)):
    return SharedWorkDraw(tuple(a0), tuple(a1), tuple(a2), tuple(a3))


def trace(rows, draws=None, *, timeline=None, experts=2, cutoff=400.0):
    timeline = timeline or CommonStateTimeline(10.0, 20.0, 30.0)
    draws = list(draws if draws is not None else [work()] * len(rows))
    local = [0] * experts
    tokens = []
    for global_id, row in enumerate(rows):
        if len(row) == 3:
            arrival, token_class, expert = row
            destination = 1 + (global_id % 2)
        else:
            arrival, token_class, expert, destination = row
        tokens.append(SharedTokenSpec(
            global_id, float(arrival), token_class, expert, local[expert],
            destination,
        ))
        local[expert] += 1
    return SharedBackupTrace(
        expert_count=experts, timeline=timeline,
        arrival_cutoff=float(cutoff), tokens=tuple(tokens), work=tuple(draws),
    )


class OnlineRule:
    def __init__(self, fn):
        self.fn = fn
        self.observations = []

    def request(self, observation: ActionObservation):
        self.observations.append(observation)
        return self.fn(observation)


class SharedPoolTests(unittest.TestCase):
    def test_two_experts_share_and_survivor_accelerates(self):
        tl = CommonStateTimeline(1.0, 2.0, 100.0)
        both = trace([(3, R, 0, 1), (3, R, 1, 2)],
                     [work(a0=(9, 1, 9)), work(a0=(9, 9, 1))],
                     timeline=tl)
        result = simulate_shared_backup(both, c_b=0.5)
        self.assertEqual(
            [t.completion_time for t in result.tokens], [5.0, 5.0]
        )
        only_one = trace([(3, R, 0, 1)],
                         [work(a0=(9, 1, 9))], timeline=tl)
        self.assertEqual(simulate_shared_backup(only_one, c_b=0.5)
                         .tokens[0].completion_time, 4.0)

    def test_work_one_and_two_rebalances_at_first_completion(self):
        tl = CommonStateTimeline(1.0, 2.0, 100.0)
        result = simulate_shared_backup(
            trace([(3, R, 0, 1), (3, R, 1, 2)],
                  [work(a0=(9, 1, 9)), work(a0=(9, 9, 2))], timeline=tl),
            c_b=0.5,
        )
        self.assertEqual([t.completion_time for t in result.tokens], [5.0, 6.0])
        self.assertAlmostEqual(sum(i.executed_work for i in result.pool_intervals), 3.0)
        self.assertGreaterEqual(result.counters.stale_completion_events_ignored, 1)

    def test_three_heads_equal_share_and_dual_delays_peer(self):
        tl = CommonStateTimeline(1.0, 2.0, 100.0)
        result = simulate_shared_backup(
            trace([(3, R, 0, 1), (3, R, 1, 1), (3, R, 1, 2)],
                  [work(a0=(9, 1, 9), a1=(9, 1, 9), a2=(9, 1, 9)),
                   work(a0=(9, 1, 9), a1=(9, 1, 9), a2=(9, 1, 9)),
                   work(a0=(9, 9, 1), a1=(9, 9, 1), a2=(9, 9, 1))],
                  timeline=tl), c_b=0.5)
        self.assertEqual([t.completion_time for t in result.tokens], [6.0] * 3)
        self.assertEqual(max(i.active_heads for i in result.pool_intervals), 3)
        self.assertEqual(sum(
            1 for i in result.pool_intervals if i.active_heads == 3
        ), 1)

    def test_idle_and_speed_cap_diagnostics(self):
        tl = CommonStateTimeline(1.0, 2.0, 100.0)
        empty = simulate_shared_backup(
            trace([(0, R, 0)], [work()], timeline=tl), c_b=0.5)
        self.assertTrue(all(i.active_heads == 0 or i.per_head_speed > 0
                            for i in empty.pool_intervals))
        one = simulate_shared_backup(
            trace([(3, R, 0, 1)], [work(a0=(9, 1, 9))], timeline=tl,
                  experts=1), c_b=0.5)
        self.assertEqual(one.tokens[0].completion_time, 5.0)
        uncoupled = simulate_shared_backup(
            trace([(3, R, 0, 1), (3, R, 0, 2)],
                  [work(a0=(9, 1, 9)), work(a0=(9, 9, 1))],
                  timeline=tl, experts=1), c_b=2.0)
        self.assertEqual([t.completion_time for t in uncoupled.tokens], [4.0, 4.0])

    def test_pool_audit_reaches_capacity_bound_when_constrained(self):
        tl = CommonStateTimeline(1.0, 2.0, 100.0)
        result = simulate_shared_backup(
            trace([(3, R, 0, 1), (3, R, 1, 2)],
                  [work(a0=(9, 9, 1)), work(a0=(9, 9, 1))], timeline=tl),
            c_b=0.5,
        )
        constrained = [i for i in result.pool_intervals if i.active_heads]
        self.assertTrue(constrained)
        for interval in constrained:
            self.assertLessEqual(interval.executed_work,
                                 interval.capacity_bound + 1e-12)
            self.assertAlmostEqual(interval.executed_work,
                                   interval.capacity_bound)
            self.assertEqual(len(interval.contributing_attempt_keys),
                             interval.active_heads)


class LifecycleAndOrderingTests(unittest.TestCase):
    def test_running_loser_stays_in_pool_and_queued_loser_has_zero_work(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        source = OnlineRule(lambda obs: X if obs.local_token_id == 1 else N)
        result = simulate_shared_backup(
            trace([(10, R, 0, 1), (10.1, R, 0, 1)],
                  [work(a0=(20, 9, 9), a2=(9, .25, 9)),
                   work(a0=(9, 9, 9), a2=(9, .25, 9), a3=(9, 9, 2))], timeline=tl,
                  experts=1), c_b=0.5, action_sources={0: source})
        loser = next(a for a in result.tokens[1].attempts if a.attempt_id == 0)
        self.assertEqual(loser.status, AttemptStatus.CANCELLED_QUEUED)
        self.assertEqual(loser.executed_work, 0.0)
        self.assertTrue(any(a.status is AttemptStatus.COMPLETED_LOSER
                            for a in result.tokens[1].attempts))
        self.assertTrue(any(i.active_heads == 2 for i in result.pool_intervals))
        self.assertGreater(result.drain_end_time, 11.0)

    def test_fault_first_and_same_time_completion_batch_are_deterministic(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        source = OnlineRule(lambda obs: D)
        result = simulate_shared_backup(
            trace([(19, R, 0, 1)],
                  [work(a0=(.5, 9, 9), a1=(1, 2, 2), a2=(9, 1, 9),
                        a3=(9, 9, 9))], timeline=tl, experts=1),
            c_b=2.0, hedge_delay=1.0, action_sources={0: source})
        token = result.tokens[0]
        self.assertEqual(token.attempts[0].status, AttemptStatus.FAILED_RUNNING)
        self.assertEqual(token.replay_count, 1)
        self.assertFalse(any(a.attempt_id >= 2 for a in token.attempts))
        self.assertEqual(result.counters.hedge_timers_voided, 1)

        tie_source = OnlineRule(lambda obs: X)
        tie = simulate_shared_backup(
            trace([(10, R, 0, 1)],
                  [work(a0=(100, 9, 9), a1=(9, 9, 9),
                        a2=(1, 1, 9), a3=(9, 9, 1))],
                  timeline=tl, experts=1), c_b=2.0,
            action_sources={0: tie_source})
        attempts = {a.attempt_id: a for a in tie.tokens[0].attempts}
        self.assertEqual(tie.tokens[0].winner_attempt_id, 2)
        self.assertEqual(attempts[2].replica_id, 1)
        self.assertEqual(attempts[3].status, AttemptStatus.COMPLETED_LOSER)

    def test_replay_and_f_primary_are_in_shared_pool_and_full_drain(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        result = simulate_shared_backup(
            trace([(19, R, 0, 1), (20, R, 1, 2)],
                  [work(a0=(2, 9, 9), a1=(3, 3, 3)),
                   work(a0=(9, 9, 1), a1=(1, 1, 1))], timeline=tl),
            c_b=0.5)
        self.assertTrue(any(a.attempt_id == 1 for a in result.attempts))
        self.assertTrue(any(a.attempt_id == 0 and a.replica_id in (1, 2)
                            for a in result.attempts))
        self.assertEqual(result.drain_end_time,
                         max(a.terminal_time for a in result.attempts))
        self.assertTrue(any("attempt=1" in key for i in result.pool_intervals
                            for key in i.contributing_attempt_keys))

    def test_same_total_work_has_different_backlog_when_replay_bursts_at_failure(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        spread = simulate_shared_backup(
            trace([(21, R, 0, 1), (22, R, 1, 2)],
                  [work(a0=(9, 2, 9)), work(a0=(9, 9, 2))], timeline=tl),
            c_b=.5)
        burst = simulate_shared_backup(
            trace([(19, R, 0, 1), (19, R, 1, 2)],
                  [work(a0=(2, 1, 9), a1=(2, 2, 2)),
                   work(a0=(2, 9, 1), a1=(2, 2, 2))], timeline=tl),
            c_b=.5)
        self.assertAlmostEqual(
            sum(i.executed_work for i in spread.pool_intervals),
            sum(i.executed_work for i in burst.pool_intervals),
        )
        self.assertNotEqual(
            [t.latency for t in spread.tokens], [t.latency for t in burst.tokens]
        )


class BudgetAndObservationTests(unittest.TestCase):
    def test_budget_charges_and_whole_dual_suppression_without_refund(self):
        tl = CommonStateTimeline(0.0, 100.0, 200.0)
        source = OnlineRule(lambda obs: (
            S if obs.local_token_id < 2 else X
        ))
        result = simulate_shared_backup(
            trace([(1, R, 0), (2, R, 0), (3, R, 0), (4, R, 0)],
                  [work()] * 4, timeline=tl, experts=1),
            c_b=2.0, action_sources={0: source})
        decisions = result.action_decisions
        self.assertEqual([d.applied for d in decisions], [S, S, N, N])
        self.assertEqual([d.charge for d in decisions], [1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(decisions[1].balance_after, .8125)
        self.assertTrue(decisions[2].budget_suppressed)
        self.assertTrue(decisions[3].budget_suppressed)

        empty_x = OnlineRule(lambda obs: X)
        x_result = simulate_shared_backup(
            trace([(1, R, 0)], [work()], timeline=tl, experts=1),
            c_b=2.0, action_sources={0: empty_x})
        self.assertEqual(x_result.action_decisions[0].applied, X)
        self.assertAlmostEqual(x_result.action_decisions[0].balance_after, .8125)

    def test_action_source_is_online_clean_and_local(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        source0 = OnlineRule(lambda obs: S)
        source1 = OnlineRule(lambda obs: N)
        result = simulate_shared_backup(
            trace([(11, R, 0), (12, U, 1), (25, R, 0)],
                  [work(a0=(1, 9, 9))] * 3, timeline=tl),
            c_b=0.5, action_sources={0: source0, 1: source1})
        self.assertEqual(len(source0.observations), 2)
        observation = source0.observations[0]
        self.assertEqual(observation.current_time, 11.0)
        self.assertEqual(observation.token_class, R)
        self.assertEqual(observation.public_active_heads, 0)
        self.assertFalse(hasattr(observation, "remaining_work"))
        self.assertFalse(hasattr(observation, "future_arrivals"))
        self.assertFalse(hasattr(observation, "other_expert_queues"))
        self.assertEqual(result.tokens[0].action, S)
        self.assertEqual(result.tokens[1].action, N)


class ConservationAndValidationTests(unittest.TestCase):
    def test_all_three_queues_conserve_work_eventwise_and_terminally(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        source = OnlineRule(lambda obs: X if obs.local_token_id == 0 else N)
        result = simulate_shared_backup(
            trace([(11, R, 0, 1), (11.1, U, 1, 2), (19, R, 0, 1),
                   (20, R, 1, 2)],
                  [work(a0=(3, 1, 1), a1=(2, 2, 2), a2=(1, 1, 1),
                        a3=(1, 1, 1))] * 4, timeline=tl),
            c_b=.5, action_sources={0: source, 1: source})
        self.assertEqual(len(result.queue_audits), 3 * 2)
        self.assertTrue(all(a.conservation_holds for a in result.queue_audit_events))
        for audit in result.queue_audits:
            self.assertAlmostEqual(
                audit.remaining_work,
                audit.enqueued_work - audit.executed_work
                - audit.cancelled_work - audit.discarded_work,
            )

        for interval in result.pool_intervals:
            self.assertLessEqual(interval.executed_work,
                                 interval.capacity_bound + 1e-10)
            expected = min(interval.active_heads, 2 * .5) * (
                interval.end_time - interval.start_time
            )
            self.assertAlmostEqual(interval.executed_work, expected)

        self.assertTrue(all(
            event.executed_work == 0.0
            for event in result.queue_audit_events
            if event.replica_id == 0 and 20.0 < event.time <= 100.0
        ))
        for token in result.tokens:
            self.assertEqual(sum(a.status is AttemptStatus.COMPLETED_WINNER
                                 for a in token.attempts), 1)
            self.assertLessEqual(token.replay_count, 1)

    def test_invalid_trace_fails_before_action_source_or_event_loop(self):
        class ExplodingSource:
            def request(self, observation):
                raise AssertionError("event loop was entered")

        with self.assertRaises(ValueError):
            SharedTokenSpec(0, math.nan, R, 0, 0, 1)
        with self.assertRaises(ValueError):
            SharedTokenSpec(0, 0.0, R, 0, 0, 3)
        with self.assertRaises(ValueError):
            SharedWorkDraw((1, 1, 1), (1, 1, 1), (1, 1, 1), ())
        with self.assertRaises(ValueError):
            trace([(1, R, 0)], [work(a0=(math.nan, 1, 1))])
        with self.assertRaises(ValueError):
            simulate_shared_backup(
                trace([(1, R, 0)], [work()]), c_b=math.inf,
                action_sources={0: ExplodingSource()})

    def test_cross_expert_action_changes_peer_latency_without_changing_peer_policy(self):
        tl = CommonStateTimeline(10.0, 20.0, 100.0)
        rows = [(11, R, 0, 1), (21, R, 1, 2)]
        draws = [work(a0=(.1, 9, 9), a2=(9, 20, 9)),
                 work(a0=(9, 9, 1))]
        baseline_source = OnlineRule(lambda obs: N)
        changed_source = OnlineRule(lambda obs: S)
        peer_source_a = OnlineRule(lambda obs: N)
        peer_source_b = OnlineRule(lambda obs: N)
        baseline = simulate_shared_backup(
            trace(rows, draws, timeline=tl), c_b=.5,
            action_sources={0: baseline_source, 1: peer_source_a})
        changed = simulate_shared_backup(
            trace(rows, draws, timeline=tl), c_b=.5,
            action_sources={0: changed_source, 1: peer_source_b})
        self.assertGreater(changed.tokens[1].latency, baseline.tokens[1].latency)
        self.assertEqual(
            [d.applied for d in baseline.action_decisions if d.expert_id == 1],
            [d.applied for d in changed.action_decisions if d.expert_id == 1],
        )
        self.assertEqual(
            [d.balance_after for d in baseline.action_decisions if d.expert_id == 1],
            [d.balance_after for d in changed.action_decisions if d.expert_id == 1],
        )


if __name__ == "__main__":
    unittest.main()
