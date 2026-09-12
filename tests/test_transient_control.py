"""Prospective admission and pooled objective: hand fixtures before code."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from mfg_hedge.attribution_episode import (
    EpisodeIdentity, EpisodeProtocol, EpisodeTrace, generate_episode_trace,
    simulate_episode, episode_trace_fingerprint,
)
from mfg_hedge.attribution_metrics import empirical_cvar95
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.transient_control import (
    ReservationParameters, TimeClassRule, plan_transient_actions,
    simulate_transient_episode,
)
from mfg_hedge.transient_objective import (
    cvar95_fractional_tail, build_transient_objective,
)
from mfg_hedge.workload import TokenSpec, WorkloadTrace

N, D, I = A.NORMAL, A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE
TL = CommonStateTimeline(100.0, 200.0, 220.0)
SMALL = CommonStateTimeline(10.0, 20.0, 30.0)


def specs(times, classes=None):
    return tuple(TokenSpec(i, float(t), K.REGULAR if classes is None else classes[i])
                 for i, t in enumerate(times))


def episode(times, *, timeline=TL, cutoff=320.0, work=1.0, index=0,
            classes=None, replay=3.0, hedge=0.5):
    tokens = specs(times, classes)
    stream = (float(work),) * len(tokens)
    trace = WorkloadTrace(1, 0.9, tokens, (stream, stream),
                          ((replay,) * len(tokens),) * 2,
                          ((hedge,) * len(tokens),) * 2)
    return EpisodeTrace(EpisodeIdentity("transient-control:v1:test", 0, index),
                        1, EpisodeProtocol(timeline, cutoff), trace)


class ReservationTests(unittest.TestCase):
    def test_cap_granularity_and_counts_per_window_and_class(self):
        plan = plan_transient_actions(specs([100, 100, 101, 101, 102]), TL,
                                      TimeClassRule(I, I, I, I))
        self.assertEqual([d.applied for d in plan.decisions], [I, N, I, N, N])
        last = plan.decisions[-1]
        self.assertEqual(last.cap, 2.8125)
        self.assertEqual(last.balance_after, 0.8125)
        audit = plan.audit()
        self.assertEqual(audit["global"]["requested_hedges"], 3)
        self.assertEqual(audit["global"]["quota_suppressed"], 1)
        for block in [audit["global"], *audit["by_class"].values(),
                      *[w["counts"] for w in audit["windows"]]]:
            self.assertEqual(block["requested_hedges"],
                             block["applied_hedges"] + block["quota_suppressed"])

    def test_scope_midpoint_and_window_boundaries(self):
        times = [99, 100, 100, 124, 125, 149, 150, 199, 200, 220]
        plan = plan_transient_actions(specs(times), TL, TimeClassRule(D, N, I, N))
        self.assertEqual([d.requested for d in plan.decisions],
                         [N, N, D, N, D, N, I, N, N, N])
        self.assertEqual(plan.decisions[4].window_start, 125)
        self.assertEqual(plan.decisions[6].window_start, 150)
        self.assertIsNone(plan.decisions[8].window_start)

    def test_final_short_window_and_exact_decimal_debits(self):
        tl = CommonStateTimeline(10, 36, 40)
        plan = plan_transient_actions(specs([35]), tl, TimeClassRule(I, I, I, I))
        self.assertEqual(plan.decisions[0].cap, 0.1125)
        self.assertEqual(plan.decisions[0].applied, N)
        params = ReservationParameters(window_width=1, budget_rate=0.3,
                                       scale=1, mean_requirement=0.1)
        plan = plan_transient_actions(specs([10] * 7), tl,
                                      TimeClassRule(I, I, I, I), params)
        self.assertEqual(plan.audit()["global"]["applied_hedges"], 3)
        self.assertEqual(plan.decisions[-1].balance_after, 0)

    def test_pooled_fifo_can_deny_later_urgent(self):
        params = ReservationParameters(budget_rate=0.04, scale=1)
        plan = plan_transient_actions(specs([100, 100, 101], [K.REGULAR]*2+[K.URGENT]),
                                      TL, TimeClassRule(I, I, I, I), params)
        self.assertEqual(plan.decisions[0].applied, I)
        self.assertEqual(plan.decisions[2].applied, N)
        self.assertTrue(plan.decisions[2].quota_suppressed)

    def test_future_metadata_does_not_change_prefix_or_share_a_ledger(self):
        rule = TimeClassRule(I, I, I, I)
        first = plan_transient_actions(specs([100, 101, 102, 103, 104]), TL, rule)
        changed = plan_transient_actions(specs([100, 101, 102, 180, 250]), TL, rule)
        self.assertEqual(first.decisions[:3], changed.decisions[:3])
        self.assertEqual(first, plan_transient_actions(specs([100,101,102,103,104]), TL, rule))

    def test_strict_input_boundaries(self):
        for value in (True, float("nan"), float("inf"), "1", None, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ReservationParameters(scale=value)
        with self.assertRaises(ValueError):
            ReservationParameters(window_width=0)
        with self.assertRaises(ValueError):
            TimeClassRule("I", N, N, N)
        for tokens in ((TokenSpec(1, 100, K.REGULAR),), specs([101, 100]),
                       (TokenSpec(0, float("nan"), K.REGULAR),),
                       (TokenSpec(True, 100, K.REGULAR),)):
            with self.assertRaises(ValueError):
                plan_transient_actions(tokens, TL, TimeClassRule())


class AppliedPhysicsTests(unittest.TestCase):
    def test_suppressed_request_uses_normal_physics_and_replay(self):
        e = episode([19], timeline=SMALL, cutoff=40, work=4)
        rule = TimeClassRule(I, I, I, I)
        allow = simulate_transient_episode(e, rule, ReservationParameters(scale=1))
        deny = simulate_transient_episode(e, rule, ReservationParameters(scale=0))
        self.assertEqual(allow.run.simulation.tokens[0].completion_time, 19.5)
        self.assertEqual(deny.run.simulation.tokens[0].completion_time, 23)
        self.assertEqual(deny.run.simulation.replay_executions, 1)
        self.assertEqual(deny.run, simulate_episode(e, 2.0))
        self.assertEqual(deny.run.simulation.hedge_requested, 0)
        self.assertEqual(deny.plan.audit()["global"]["requested_hedges"], 1)

    def test_voided_delayed_reservation_is_not_refunded(self):
        e = episode([100, 101, 103], work=0.25)
        params = ReservationParameters(budget_rate=0.04, scale=1)
        r = simulate_transient_episode(e, TimeClassRule(D, D, D, D), params, hedge_delay=2)
        self.assertEqual(r.run.simulation.hedge_launches, 0)
        self.assertEqual(r.plan.decisions[2].applied, N)
        self.assertEqual(r.plan.audit()["global"]["reserved_work"], 1)

    def test_decisions_do_not_read_service_draws_and_normal_regresses(self):
        e = episode([100, 101, 102, 103, 125], work=1)
        modified = replace(e, workload=replace(e.workload,
                           service_times=((50.0,)*5,)*2,
                           hedge_service_times=((20.0,)*5,)*2))
        rule = TimeClassRule(I, I, I, I)
        first = simulate_transient_episode(e, rule)
        second = simulate_transient_episode(modified, rule)
        self.assertEqual(first.plan, second.plan)
        self.assertEqual(simulate_transient_episode(e, TimeClassRule()).run,
                         simulate_episode(e, 2))
        with patch("mfg_hedge.transient_control.plan_transient_actions",
                   wraps=plan_transient_actions) as spy:
            simulate_transient_episode(e, rule)
            self.assertEqual(spy.call_args.args[0], e.workload.tokens)
            self.assertTrue(all(isinstance(t, TokenSpec) for t in spy.call_args.args[0]))

    def test_realized_work_can_exceed_reserved_cap(self):
        e = episode([100], work=20, hedge=4)
        r = simulate_transient_episode(e, TimeClassRule(I, I, I, I))
        self.assertEqual(r.plan.decisions[0].charge, 1)
        actual = sum(a.executed_work for a in r.run.simulation.attempts if a.attempt_id == 2)
        self.assertEqual(actual, 4)
        self.assertGreater(actual, r.plan.decisions[0].cap)

    def test_holdout_is_rejected_without_calling_engine(self):
        e = episode([1])
        e = replace(e, identity=EpisodeIdentity("attribution:v1:holdout", 0, 0))
        with patch("mfg_hedge.transient_control.simulate_episode") as engine:
            with self.assertRaisesRegex(ValueError, "namespace"):
                simulate_transient_episode(e, TimeClassRule())
            engine.assert_not_called()


class FractionalTailTests(unittest.TestCase):
    def test_n1_n20_n21_and_variational_definition(self):
        for samples in ([4], list(range(20)), [0]*19+[10,100], [2]*21):
            with self.subTest(n=len(samples)):
                expected = min(eta + 20*sum(max(0, x-eta) for x in samples)/len(samples)
                               for eta in samples)
                self.assertAlmostEqual(cvar95_fractional_tail(samples), expected)
        values = [0]*19+[10,100]
        self.assertAlmostEqual(cvar95_fractional_tail(values), 100.5/1.05)
        self.assertEqual(empirical_cvar95(values), 55)

    def test_tail_rejects_empty_invalid_and_negative_latencies(self):
        for values in ([], [True], [float("nan")], [float("inf")], [-1], ["1"]):
            with self.assertRaises(ValueError):
                cvar95_fractional_tail(values)


class ObjectiveTests(unittest.TestCase):
    def full_run(self, work=4, index=0):
        e = episode([0, 10, 100, 110, 200, 210, 230, 240], work=work,
                    index=index, classes=[K.REGULAR, K.URGENT]*4)
        return simulate_transient_episode(e, TimeClassRule()).run

    def test_complete_hand_score_and_terms(self):
        score = build_transient_objective([self.full_run()])
        self.assertEqual(score["status"], "complete")
        self.assertAlmostEqual(score["components"]["phase_loss"], 8.4)
        self.assertAlmostEqual(score["components"]["fault_tail"], 6)
        self.assertEqual(score["components"]["executed_work"], 4)
        self.assertEqual(score["components"]["wasted_work"], 0)
        self.assertAlmostEqual(score["total"], 18.4)
        self.assertEqual(score["cohorts"]["D"]["regular"]["sample_count"], 1)

    def test_missing_cohort_is_incomplete_and_replay_penalty_is_real(self):
        e = episode([19], timeline=SMALL, cutoff=40, work=4)
        r = simulate_transient_episode(e, TimeClassRule()).run
        score = build_transient_objective([r])
        self.assertEqual(score["status"], "incomplete")
        self.assertIsNone(score["total"])
        self.assertIn("F/urgent", score["missing_cohorts"])
        self.assertEqual(score["cohorts"]["D"]["regular"]["loss"], 6)
        self.assertEqual(score["components"]["executed_work"], 3.5)
        self.assertEqual(score["components"]["wasted_work"], 0.5)

    def test_pool_raw_counts_not_episode_means(self):
        first = self.full_run(work=1)
        e = episode([0, 10], work=4, index=1, classes=[K.REGULAR, K.URGENT])
        second = simulate_transient_episode(e, TimeClassRule()).run
        score = build_transient_objective([first, second])
        self.assertEqual(score["generated_tokens"], 10)
        self.assertAlmostEqual(score["components"]["executed_work"], 16/10)
        self.assertEqual(score["cohorts"]["H"]["regular"]["mean_latency"], 2.5)
        self.assertEqual(score, build_transient_objective([second, first]))

    def test_running_loser_after_cutoff_is_charged(self):
        e = episode([198], work=0.5, hedge=200)
        r = simulate_transient_episode(e, TimeClassRule(I, I, I, I),
                                       ReservationParameters(scale=1)).run
        # A wins at 199; B remains a running loser through drain at 398.
        score = build_transient_objective([r])
        self.assertEqual(score["components"]["executed_work"], 200.5)
        self.assertEqual(score["components"]["wasted_work"], 200)
        self.assertEqual(r.simulation.tokens[0].completion_time, 199)
        self.assertGreater(r.simulation.drain_end_time, e.protocol.arrival_cutoff)

    def test_duplicate_mismatched_or_corrupt_results_rejected(self):
        run = self.full_run()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_transient_objective([run, run])
        with self.assertRaises(ValueError):
            build_transient_objective([])
        other = self.full_run(index=1)
        with self.assertRaisesRegex(ValueError, "contract"):
            build_transient_objective([run, replace(other, hedge_delay=2)])
        with self.assertRaises(RuntimeError):
            build_transient_objective([replace(run, simulation=replace(
                run.simulation, hedge_requested=7))])

    def test_small_development_run_is_repeatable_with_shared_crn(self):
        config = replace(load_config(Path(__file__).resolve().parents[1]/"configs/v1_minimal.json"),
                         healthy_offered_load=0.45)
        protocol = EpisodeProtocol(TL, 320)
        episodes = [generate_episode_trace(config, "transient-control:v1:fit", 10, i, protocol)
                    for i in range(4)]
        rule = TimeClassRule(D, I, D, I)
        def run_all():
            return [simulate_transient_episode(e, rule, hedge_delay=1.5) for e in episodes]
        first, second = run_all(), run_all()
        self.assertEqual(first, second)
        self.assertEqual([episode_trace_fingerprint(r.run.episode) for r in first],
                         [episode_trace_fingerprint(e) for e in episodes])
        score = build_transient_objective([r.run for r in first])
        self.assertEqual(score["status"], "complete")
        self.assertEqual(json.dumps(score, sort_keys=True, allow_nan=False),
                         json.dumps(build_transient_objective([r.run for r in second]),
                                    sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
