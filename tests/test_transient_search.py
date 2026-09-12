"""Development-only exhaustive rule-search contracts."""

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mfg_hedge.attribution_episode import EpisodeIdentity, EpisodeProtocol, EpisodeTrace
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.transient_control import ReservationParameters
from mfg_hedge.transient_search import (
    DEVELOPMENT_EPISODE_COUNT,
    DEVELOPMENT_MACRO_SEED,
    DEVELOPMENT_NAMESPACE,
    assess_development_safety,
    enumerate_time_class_rules,
    evaluate_rule_bank,
    write_development_search_artifact,
)
from mfg_hedge.workload import TokenSpec, WorkloadTrace


def complete_episode(namespace=DEVELOPMENT_NAMESPACE, index=0):
    timeline = CommonStateTimeline(10, 20, 30)
    times = (0, 1, 10, 11, 20, 21, 31, 32)
    classes = (K.REGULAR, K.URGENT) * 4
    tokens = tuple(TokenSpec(i, t, classes[i]) for i, t in enumerate(times))
    values = (1.0,) * len(tokens)
    trace = WorkloadTrace(1, .9, tokens, (values, values),
                          ((1.0,) * 8,) * 2, ((.5,) * 8,) * 2)
    return EpisodeTrace(EpisodeIdentity(namespace, DEVELOPMENT_MACRO_SEED, index),
                        1, EpisodeProtocol(timeline, 40), trace)


class RuleBankTests(unittest.TestCase):
    def test_exact_lexical_bank(self):
        bank = enumerate_time_class_rules()
        self.assertEqual(len(bank), 81)
        self.assertEqual(len(set(bank)), 81)
        self.assertEqual([a.value for a in bank[0].actions], list("NNNN"))
        self.assertEqual([a.value for a in bank[-1].actions], list("IIII"))
        self.assertEqual([a.value for a in bank[1].actions], list("NNND"))

    def test_zero_cap_makes_all_physics_equal_and_lexical_tie_selects_normal(self):
        result = evaluate_rule_bank(
            [complete_episode()], hedge_delay=1.5,
            reservation=ReservationParameters(scale=0),
        )
        self.assertEqual(result["rule_count"], 81)
        self.assertEqual(result["selected_rule"], "NNNN")
        self.assertEqual(result["baseline_rule"], "NNNN")
        self.assertEqual(len(result["rules"]), 81)
        self.assertTrue(all(row["feasible"] for row in result["rules"]))
        self.assertTrue(all(row["objective"]["total"] == result["rules"][0]["objective"]["total"]
                            for row in result["rules"]))
        self.assertEqual(result, evaluate_rule_bank(
            [complete_episode()], hedge_delay=1.5,
            reservation=ReservationParameters(scale=0)))

    def test_input_order_does_not_change_result_but_bad_cohort_is_rejected(self):
        episodes = [complete_episode(index=0), complete_episode(index=1)]
        kwargs = dict(hedge_delay=1.5, reservation=ReservationParameters(scale=0))
        self.assertEqual(evaluate_rule_bank(episodes, **kwargs),
                         evaluate_rule_bank(list(reversed(episodes)), **kwargs))
        with self.assertRaisesRegex(ValueError, "fit or test"):
            evaluate_rule_bank([complete_episode("attribution:v1:holdout")], **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_rule_bank([episodes[0], episodes[0]], **kwargs)
        changed = replace(episodes[1], protocol=EpisodeProtocol(
            CommonStateTimeline(10, 20, 31), 40))
        with self.assertRaisesRegex(ValueError, "protocol"):
            evaluate_rule_bank([episodes[0], changed], **kwargs)


class SafetyTests(unittest.TestCase):
    def test_exact_thresholds_and_nonfinite_rejection(self):
        baseline = {"H": {"mean": 10., "p95": 20.},
                    "R": {"mean": 5., "p95": 10.}, "work": 100.}
        exact = {"H": {"mean": 10.1, "p95": 20.4},
                 "R": {"mean": 5.05, "p95": 10.2}, "work": 105.}
        status = assess_development_safety(exact, baseline)
        self.assertTrue(status["feasible"])
        self.assertEqual(status["reasons"], [])
        failed = assess_development_safety(
            {**exact, "H": {"mean": 10.1000001, "p95": 20.4}}, baseline)
        self.assertFalse(failed["feasible"])
        self.assertEqual(failed["reasons"], ["H_mean_ratio"])
        with self.assertRaises(ValueError):
            assess_development_safety({**exact, "work": float("nan")}, baseline)


class ArtifactTests(unittest.TestCase):
    def test_transactional_development_artifact_and_no_overwrite(self):
        search = evaluate_rule_bank([complete_episode()], hedge_delay=1.5,
                                    reservation=ReservationParameters(scale=0))
        manifest = {"scenario": "transient_control_81_rule_development_fit",
                    "episode_count": 1, "simulator_version": "test"}
        with tempfile.TemporaryDirectory() as directory:
            path = write_development_search_artifact(directory, "fit-test", manifest, search)
            self.assertEqual(sorted(p.name for p in path.iterdir()),
                             ["manifest.json", "rule_results.json", "summary.json"])
            summary = json.loads((path / "summary.json").read_text("utf-8"))
            self.assertEqual(summary["selected_rule"], "NNNN")
            with self.assertRaises(FileExistsError):
                write_development_search_artifact(directory, "fit-test", manifest, search)

    def test_incomplete_or_duplicate_rule_bank_writes_nothing(self):
        search = evaluate_rule_bank([complete_episode()], hedge_delay=1.5,
                                    reservation=ReservationParameters(scale=0))
        broken = copy.deepcopy(search)
        broken["rules"][-1] = broken["rules"][0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete unique"):
                write_development_search_artifact(directory, "broken", {}, broken)
            self.assertEqual(list(Path(directory).iterdir()), [])


class FrozenProductionBoundaryTests(unittest.TestCase):
    def test_frozen_constants_and_development_count_guard(self):
        self.assertEqual(DEVELOPMENT_NAMESPACE, "transient-control:v1:fit")
        self.assertEqual(DEVELOPMENT_MACRO_SEED, 20260905)
        self.assertEqual(DEVELOPMENT_EPISODE_COUNT, 64)
        with self.assertRaisesRegex(ValueError, "64"):
            evaluate_rule_bank([complete_episode()], hedge_delay=1.5,
                               reservation=ReservationParameters(), require_frozen_fit=True)


if __name__ == "__main__":
    unittest.main()
