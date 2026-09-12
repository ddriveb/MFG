"""Regression tests for defects found by the MFG-readiness review."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import TokenClass
from mfg_hedge.game_deviations import DeviationScenario, _run_profile
from mfg_hedge.game_workload import ActionRule
from mfg_hedge.shared_backup import (
    SharedAction,
    SharedBackupTrace,
    SharedTokenSpec,
    SharedWorkDraw,
    simulate_shared_backup,
)
from mfg_hedge.stage4_campaign import (
    _full_profile_runs,
    compute_simultaneous_regret_bound,
    jackknife_pseudo_values,
    run_parallel_fit,
)
from mfg_hedge import stage4_campaign
from tests.test_game_deviations import scenario as deviation_scenario


def _trace(arrivals=(110.0,), *, destination=2, work=None):
    default = SharedWorkDraw(
        (10.0, 10.0, 10.0), (1.0, 1.0, 1.0),
        (1.0, 8.0, 0.25), (1.0, 1.0, 5.0),
    )
    return SharedBackupTrace(
        1,
        CommonStateTimeline(100.0, 200.0, 220.0),
        360.0,
        tuple(
            SharedTokenSpec(i, t, TokenClass.REGULAR, 0, i, destination)
            for i, t in enumerate(arrivals)
        ),
        tuple(work or default for _ in arrivals),
    )


class _ConstantSource:
    def __init__(self, action):
        self.action = action

    def request(self, observation):
        return self.action


class _EpisodePrivateRule:
    name = "SSSS"

    def __init__(self, registry=None):
        self.calls = 0
        self.registry = registry if registry is not None else []
        self.registry.append(self)

    def new_episode(self):
        return type(self)(self.registry)

    def request(self, observation):
        self.calls += 1
        return SharedAction.SINGLE if self.calls == 1 else SharedAction.NORMAL


class ReadinessFixTests(unittest.TestCase):
    def test_dual_preserves_default_c_attempt_two_crn(self):
        trace = _trace(destination=2)
        single = simulate_shared_backup(
            trace, 2.0, action_sources={0: _ConstantSource(SharedAction.SINGLE)}
        )
        dual = simulate_shared_backup(
            trace, 2.0, action_sources={0: _ConstantSource(SharedAction.DUAL)}
        )
        single_c = next(a for a in single.attempts if a.replica_id == 2)
        dual_c = next(a for a in dual.attempts if a.replica_id == 2)
        self.assertEqual((single_c.attempt_id, single_c.required_work), (2, 0.25))
        self.assertEqual((dual_c.attempt_id, dual_c.required_work), (2, 0.25))
        dual_b = next(a for a in dual.attempts if a.replica_id == 1)
        self.assertEqual((dual_b.attempt_id, dual_b.required_work), (3, 1.0))
        self.assertEqual(single.tokens[0].latency, dual.tokens[0].latency)

    def test_voided_timer_is_not_observed_as_pending(self):
        short = SharedWorkDraw(
            (0.1, 0.1, 0.1), (0.1, 0.1, 0.1),
            (0.1, 0.1, 0.1), (0.1, 0.1, 0.1),
        )

        class Observer:
            def __init__(self):
                self.observations = []

            def request(self, observation):
                self.observations.append(observation)
                return (
                    SharedAction.DELAYED
                    if observation.local_token_id == 0
                    else SharedAction.NORMAL
                )

        observer = Observer()
        result = simulate_shared_backup(
            _trace((110.0, 112.0), work=short), 2.0,
            action_sources={0: observer},
        )
        self.assertTrue(result.tokens[0].timer_voided)
        self.assertEqual(observer.observations[1].timer_history, ((0, "voided"),))

    def test_stateful_rule_is_fresh_for_each_episode(self):
        trace = _trace()
        scenarios = (
            DeviationScenario("episode-a", trace, "fault"),
            DeviationScenario("episode-b", trace, "fault"),
        )
        registry = []
        policy = _EpisodePrivateRule(registry)
        runs = _run_profile(
            scenarios, {0: policy}, c_b=2.0,
            degraded_slowdown=2.0, hedge_delay=1.5,
        )
        self.assertEqual([run.result.tokens[0].action for run in runs], [
            SharedAction.SINGLE, SharedAction.SINGLE,
        ])
        self.assertEqual(len(registry), 3)
        self.assertEqual([instance.calls for instance in registry], [0, 1, 1])

    def test_missing_jackknife_values_fail_closed_without_shape_exception(self):
        missing = jackknife_pseudo_values(0.1, (0.1, None) + (0.1,) * 30)
        self.assertEqual(len(missing), 32)
        self.assertTrue(all(value is None for value in missing))
        bound = compute_simultaneous_regret_bound(
            {(0, "NNNN"): 0.0, (0, "NSNS"): 0.1},
            {(0, "NNNN"): (0.0,) * 32, (0, "NSNS"): missing},
            targets={0: 1.0},
        )
        self.assertEqual(bound.status, "statistics_insufficient")
        self.assertIsNone(bound.simultaneous_upper_bound)
        empty_row_bound = compute_simultaneous_regret_bound(
            {(0, "NNNN"): 0.0, (0, "NSNS"): 0.1},
            {(0, "NNNN"): (0.0,) * 32, (0, "NSNS"): ()},
            targets={0: 1.0},
        )
        self.assertEqual(empty_row_bound.status, "statistics_insufficient")

    def test_one_sided_bound_uses_upper_tail_inversion(self):
        skew = (-31.0,) + (1.0,) * 31
        bound = compute_simultaneous_regret_bound(
            {(0, "NSNS"): 0.0, (0, "NNNN"): 0.0},
            {(0, "NSNS"): skew, (0, "NNNN"): (0.0,) * 32},
            targets={0: 1.5},
        )
        self.assertEqual(bound.status, "fail")
        self.assertAlmostEqual(bound.simultaneous_upper_bound, 2.0)

    def test_partial_baseline_failure_counts_each_attempt(self):
        first = deviation_scenario("partial-a")
        second = deviation_scenario("partial-b")
        calls = []
        observed_calls = []

        reference = stage4_campaign.simulate_shared_backup_optimized

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected baseline failure")
            return reference(*args, **kwargs)

        with patch.object(stage4_campaign, "simulate_shared_backup_optimized",
                          side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected baseline failure"):
                _full_profile_runs(
                    (first, second),
                    {0: ActionRule("NNNN", tuple("NNNN")),
                     1: ActionRule("NNNN", tuple("NNNN"))},
                    c_b=2.0, degraded_slowdown=2.0, hedge_delay=1.5,
                    call_observer=lambda: observed_calls.append(1),
                )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(observed_calls), 2)

    def test_fit_reports_a_call_that_fails_inside_baseline(self):
        sample = deviation_scenario("partial-fit")
        observed_calls = []

        def fail_baseline(scenarios, profile, **kwargs):
            kwargs["call_observer"]()
            observed_calls.append(1)
            raise RuntimeError("injected fit baseline failure")

        with patch.object(stage4_campaign, "_full_profile_runs",
                          side_effect=fail_baseline):
            result = run_parallel_fit(
                (sample,), expert_count=2, max_rounds=1,
                fit_call_budget=10, max_workers=1,
            )
        self.assertEqual(result.status, "execution_failed")
        self.assertEqual(result.call_count, 1)
        self.assertEqual(observed_calls, [1])

    def test_stage4_provenance_is_deterministic_and_complete(self):
        first = stage4_campaign._stage4_provenance()
        second = stage4_campaign._stage4_provenance()
        self.assertEqual(first, second)
        self.assertEqual(len(first["source"]["files"]), 7)
        self.assertTrue(first["source"]["aggregate_sha256"])
        self.assertTrue(first["configuration"]["sha256"])
        self.assertEqual(first["configuration"]["physics"]["c_B"], 0.5)
        self.assertEqual(first["inference"]["bootstrap_seed"], 20260906)
        self.assertEqual(first["inference"]["bootstrap_replicates"], 4096)
        self.assertEqual(first["attempt_binding"]["dual_attempts"],
                         ["default_destination:attempt2", "other:attempt3"])
        self.assertEqual(len(first["provenance_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
