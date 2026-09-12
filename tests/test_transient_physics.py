"""Hand-derived physical-model contracts; no fitted policy or holdout data."""

from dataclasses import replace
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction as Action
from mfg_hedge.hedge_simulation import HedgeAttemptStatus as Status
from mfg_hedge.workload import generate_workload_with_hedge
from tests.transient_physics_support import (
    AuditedEngine,
    audited_run,
    make_trace,
    reserve_requests,
)

N, D, I = Action.NORMAL, Action.DELAYED_HEDGE, Action.IMMEDIATE_HEDGE


def attempt(result, token_id, attempt_id):
    return next(a for a in result.tokens[token_id].attempts if a.attempt_id == attempt_id)


class ReplayPulseTests(unittest.TestCase):
    def test_failure_moves_twenty_replay_work_to_b_at_one_instant(self):
        timeline = CommonStateTimeline(10, 20, 50)
        trace = make_trace((10, 10.25, 11, 11.25, 12),
                           (20, 1, 4, 1, 6), (1, .5, 1, .5, 1),
                           replay_b=(8, 1, 6, 1, 6))
        result, audit = audited_run(trace, timeline)
        before = audit.frame(20, "before:state")
        after = audit.frame(20, "after:state")
        self.assertEqual(before.remaining, (25, 0))
        self.assertEqual(after.remaining, (0, 20))
        self.assertEqual(after.discarded, (25, 0))
        self.assertEqual(after.executed, (5, 1))
        self.assertEqual(after.queued[1], ((0, 1), (2, 1), (4, 1)))
        self.assertEqual([attempt(result, i, 1).enqueue_time for i in (0, 2, 4)],
                         [20, 20, 20])
        self.assertEqual([result.tokens[i].completion_time for i in (0, 2, 4)],
                         [28, 34, 40])
        self.assertEqual(audit.frames[-1].executed, (5, 21))

    def test_same_twenty_work_different_timing_changes_queue_and_latency(self):
        timeline = CommonStateTimeline(10, 20, 50)
        burst = make_trace((20, 20, 20), (1, 1, 1), (8, 6, 6))
        spread = replace(burst, tokens=tuple(
            replace(t, arrival_time=time)
            for t, time in zip(burst.tokens, (20, 28, 34))))
        burst_result, burst_audit = audited_run(burst, timeline)
        spread_result, spread_audit = audited_run(spread, timeline)
        self.assertEqual(burst_audit.frames[-1].executed, (0, 20))
        self.assertEqual(spread_audit.frames[-1].executed, (0, 20))
        self.assertEqual(burst_audit.frame(20, "after:dispatch").remaining, (0, 20))
        self.assertEqual(spread_audit.frame(20, "after:dispatch").remaining, (0, 8))
        self.assertEqual([t.latency for t in burst_result.tokens], [8, 14, 20])
        self.assertEqual([t.latency for t in spread_result.tokens], [8, 6, 6])
        self.assertEqual(burst.service_times, spread.service_times)
        self.assertEqual(burst_result.drain_end_time, spread_result.drain_end_time)


class CancellationFeedbackTests(unittest.TestCase):
    def test_backup_cancels_queued_primary_and_advances_next_a_token(self):
        timeline = CommonStateTimeline(10, 30, 40)
        trace = make_trace((10, 10.25, 11, 11.25, 12),
                           (4, 1, 2, 1, 1), (1, .25, 1, .25, 1),
                           hedge_b=(1, 1, .5, 1, 1))
        normal, normal_audit = audited_run(trace, timeline)
        protected, audit = audited_run(trace, timeline, {2: I})
        cancelled = attempt(protected, 2, 0)
        self.assertEqual(cancelled.status, Status.CANCELLED_QUEUED)
        self.assertIsNone(cancelled.start_time)
        self.assertEqual((cancelled.terminal_time, cancelled.executed_work), (11.5, 0))
        self.assertEqual(audit.frame(11.5, "after:complete").cancelled[0], 2)
        self.assertEqual(attempt(normal, 4, 0).start_time, 22)
        self.assertEqual(attempt(protected, 4, 0).start_time, 18)
        self.assertEqual((normal.tokens[4].completion_time,
                          protected.tokens[4].completion_time), (24, 20))
        self.assertEqual(normal_audit.frames[-1].executed[0], 7)
        self.assertEqual(audit.frames[-1].executed[0], 5)

    def test_running_primary_is_not_preempted_when_backup_wins(self):
        trace = make_trace((10,), (4,), (1,), hedge_b=(.5,))
        result, audit = audited_run(trace, CommonStateTimeline(10, 30, 40), {0: I})
        primary = attempt(result, 0, 0)
        self.assertEqual(result.tokens[0].completion_time, 10.5)
        self.assertEqual(primary.status, Status.COMPLETED_LOSER)
        self.assertEqual((primary.terminal_time, primary.executed_work), (18, 4))
        self.assertEqual(audit.frames[-1].cancelled, (0, 0))
        self.assertEqual(audit.frames[-1].executed, (4, .5))


class AppliedActionFeedbackTests(unittest.TestCase):
    def test_denied_hedge_really_becomes_normal_and_creates_replay(self):
        timeline = CommonStateTimeline(10, 20, 30)
        trace = make_trace((19,), (4,), (1,), replay_b=(3,), hedge_b=(.5,))
        requested = ((0, 19, I),)
        granted = reserve_requests(requested, timeline, cap=1)
        denied = reserve_requests(requested, timeline, cap=.5)
        self.assertEqual(granted[0].requested, denied[0].requested)
        self.assertEqual((granted[0].applied, denied[0].applied), (I, N))
        protected, protected_audit = audited_run(trace, timeline, {0: granted[0].applied})
        suppressed, suppressed_audit = audited_run(trace, timeline, {0: denied[0].applied})
        baseline, _ = audited_run(trace, timeline)
        self.assertEqual(suppressed, baseline)
        self.assertEqual((protected.replay_executions, suppressed.replay_executions), (0, 1))
        self.assertEqual((protected.tokens[0].completion_time,
                          suppressed.tokens[0].completion_time), (19.5, 23))
        self.assertEqual(protected_audit.frames[-1].executed, (.5, .5))
        self.assertEqual(suppressed_audit.frames[-1].executed, (.5, 3))
        self.assertEqual((suppressed.hedge_requested, suppressed.hedge_launches), (0, 0))
        self.assertTrue(denied[0].suppressed)

    def test_budget_charge_is_not_a_pathwise_executed_work_bound(self):
        timeline = CommonStateTimeline(10, 30, 40)
        decisions = reserve_requests(((0, 10, I), (2, 11, D), (4, 12, I)),
                                    timeline, cap=2.8125)
        self.assertEqual([d.applied for d in decisions], [I, D, N])
        self.assertEqual(sum(d.charge for d in decisions), 2)
        self.assertEqual(decisions[-1].remaining, .8125)
        trace = make_trace((10,), (.25,), (1,), hedge_b=(4,))
        result, audit = audited_run(trace, timeline, {0: decisions[0].applied})
        self.assertEqual(attempt(result, 0, 2).executed_work, 4)
        self.assertGreater(audit.frames[-1].executed[1], 2.8125)

    def test_delayed_reservation_is_not_refunded_after_primary_completes(self):
        timeline = CommonStateTimeline(10, 30, 40)
        requests = ((0, 10, D), (2, 12, I))
        decisions = reserve_requests(requests, timeline, cap=1)
        self.assertEqual([d.applied for d in decisions], [D, N])
        trace = make_trace((10, 11, 12), (.25, 1, 1), (1, .25, 1))
        result, _ = audited_run(trace, timeline, {d.token_id: d.applied for d in decisions},
                                hedge_delay=1)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.hedge_timers_voided, 1)
        self.assertEqual(sum(d.charge for d in decisions), 1)

    def test_future_requests_cannot_change_earlier_admission(self):
        timeline = CommonStateTimeline(10, 100, 110)
        prefix = ((0, 10, I), (2, 11, D))
        first = reserve_requests(prefix + ((4, 12, I),), timeline, cap=1)
        second = reserve_requests(prefix + ((4, 80, N),), timeline, cap=1)
        self.assertEqual(first[:2], second[:2])
        # New 25-unit window starts at 35; same-time order remains the input order.
        decisions = reserve_requests(prefix + ((4, 35, I),), timeline, cap=1)
        self.assertEqual(decisions[2].applied, I)


class BoundaryAndAuditTests(unittest.TestCase):
    def test_failure_before_completion_or_timer_preserves_work_ledger(self):
        timeline = CommonStateTimeline(10, 20, 30)
        trace = make_trace((19,), (.5,), (1,), replay_b=(2,))
        result, audit = audited_run(trace, timeline, {0: D}, hedge_delay=1)
        self.assertEqual(attempt(result, 0, 0).status, Status.FAILED_RUNNING)
        self.assertEqual(result.hedge_launches, 0)
        self.assertEqual(result.replay_executions, 1)
        self.assertEqual(audit.frame(20, "after:state").remaining, (0, 2))
        self.assertEqual(audit.frame(20, "after:state").discarded, (0, 0))
        labels = [f.label for f in audit.frames if f.time == 20]
        self.assertLess(labels.index("after:state"), labels.index("after:complete"))
        self.assertLess(labels.index("after:complete"), labels.index("after:timer"))

    def test_missing_replay_admission_is_caught_by_live_state(self):
        class BrokenAdmission(AuditedEngine):
            def _enqueue(self, copy):
                super()._enqueue(copy)
                if copy.attempt_id == 1:
                    self.admitted[copy.replica_id] -= copy.required_work
        trace = make_trace((19,), (4,), (1,), replay_b=(3,))
        with self.assertRaisesRegex(AssertionError, "conservation"):
            audited_run(trace, CommonStateTimeline(10, 20, 30), engine_class=BrokenAdmission)

    def test_missing_cancellation_loss_is_caught(self):
        class BrokenCancellation(AuditedEngine):
            def _record(self, copy, status, terminal_time, executed_work):
                super()._record(copy, status, terminal_time, executed_work)
                if status is Status.CANCELLED_QUEUED:
                    self.cancelled[copy.replica_id] -= copy.required_work
        trace = make_trace((10, 10.25, 11), (4, 1, 2), (1, .25, 1),
                           hedge_b=(1, 1, .5))
        with self.assertRaisesRegex(AssertionError, "conservation"):
            audited_run(trace, CommonStateTimeline(10, 30, 40), {2: I},
                        engine_class=BrokenCancellation)

    def test_missing_failure_discard_is_caught(self):
        class BrokenDiscard(AuditedEngine):
            def _record(self, copy, status, terminal_time, executed_work):
                super()._record(copy, status, terminal_time, executed_work)
                if status is Status.FAILED_RUNNING:
                    self.discarded[copy.replica_id] -= copy.required_work - executed_work
        trace = make_trace((19,), (4,), (1,))
        with self.assertRaisesRegex(AssertionError, "conservation"):
            audited_run(trace, CommonStateTimeline(10, 20, 30), engine_class=BrokenDiscard)

    def test_seeded_mixed_actions_observer_never_changes_engine_result(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "v1_minimal.json")
        timeline = CommonStateTimeline(10, 20, 30)
        for seed in range(8):
            trace = generate_workload_with_hedge(config, 100, base_seed=seed)
            plan = {t.token_id: (N, D, I)[t.token_id % 3] for t in trace.tokens}
            with self.subTest(seed=seed):
                result, audit = audited_run(trace, timeline, plan, hedge_delay=1)
                self.assertGreater(len(audit.frames), 200)
                self.assertEqual(audit.frames[-1].remaining, (0, 0))
                self.assertEqual(result.completed_tokens, 100)


if __name__ == "__main__":
    unittest.main()
