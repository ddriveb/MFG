import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.config import load_config
from mfg_hedge.protection_cancellation_experiment import (
    CANCELLATION_ARMS,
    CANCELLATION_BOOTSTRAP_REPLICATES,
    CANCELLATION_CALL_LIMIT,
    CANCELLATION_EPISODES,
    CANCELLATION_MACRO_SEED,
    CANCELLATION_NAMESPACE,
    CancellationExperimentError,
    run_cancellation_ablation,
    run_cancellation_timing_preflight,
    write_cancellation_artifact,
)


PROJECT = Path(__file__).resolve().parents[1]


class CancellationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT / "configs/v1_minimal.json")

    def test_protocol_and_call_budget_are_frozen(self):
        result = run_cancellation_ablation(self.config, episode_count=2)
        self.assertEqual(result["namespace"], CANCELLATION_NAMESPACE)
        self.assertEqual(result["macro_seed"], CANCELLATION_MACRO_SEED)
        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["attempted_calls"], 8)
        self.assertEqual(result["formal_call_limit"], CANCELLATION_CALL_LIMIT)
        self.assertEqual(result["arms"].keys(), {arm.key for arm in CANCELLATION_ARMS})
        self.assertEqual(len({row["trace_fingerprint"] for row in result["episode_rows"]}), 2)

    def test_all_arms_share_each_episode_crn_and_mode_verification(self):
        result = run_cancellation_ablation(self.config, episode_count=1)
        row = result["episode_rows"][0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(len(set(row["arms"][arm.key]["trace_fingerprint"] for arm in CANCELLATION_ARMS)), 1)
        self.assertTrue(row["mechanism_verification"]["crn_equal"])
        self.assertTrue(row["mechanism_verification"]["niin_admission_equal"])
        self.assertEqual(result["failed_episode_count"], 0)

    def test_failed_arm_is_counted_once_and_panel_is_fail_closed(self):
        original = __import__(
            "mfg_hedge.protection_cancellation_experiment",
            fromlist=["simulate_cancellation_arm"],
        ).simulate_cancellation_arm
        calls = {"count": 0}

        def fail_once(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected physical failure")
            return original(*args, **kwargs)

        with patch(
            "mfg_hedge.protection_cancellation_experiment.simulate_cancellation_arm",
            side_effect=fail_once,
        ):
            result = run_cancellation_ablation(self.config, episode_count=1)
        self.assertEqual(calls["count"], 4)
        self.assertEqual(result["attempted_calls"], 4)
        self.assertEqual(result["failed_episode_count"], 1)
        self.assertEqual(result["status"], "physical_failed")
        self.assertEqual(result["paired_statistics"]["status"], "unavailable")
        failed = result["episode_rows"][0]["arms"][CANCELLATION_ARMS[1].key]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exception_type"], "RuntimeError")

    def test_paired_statistics_use_episode_clusters_and_include_did(self):
        result = run_cancellation_ablation(self.config, episode_count=2)
        stats = result["paired_statistics"]
        self.assertEqual(stats["cluster_unit"], "episode_index")
        self.assertEqual(stats["bootstrap"]["replicates"], CANCELLATION_BOOTSTRAP_REPLICATES)
        expected = {
            "niin:preemptive-minus-niin:conservative",
            "laedge:preemptive-minus-laedge:conservative",
            "laedge:conservative-minus-niin:conservative",
            "laedge:preemptive-minus-niin:preemptive",
            "cancellation_x_policy_family_did",
        }
        self.assertEqual(set(stats["comparisons"]), expected)
        for comparison in stats["comparisons"].values():
            for metric in ("overall_mean", "df_cvar95", "total_work", "storm_peak"):
                entry = comparison[metric]
                self.assertIn("point", entry)
                self.assertIn("absolute_difference", entry)
                self.assertIn("relative_change", entry)
                self.assertEqual(len(entry["ci95"]), 2)
                self.assertEqual(entry["sample_count"], 2)

    def test_preflight_does_not_change_formal_episode_or_call_constants(self):
        preflight = run_cancellation_timing_preflight(self.config, episode_count=1)
        self.assertEqual(preflight["status"], "completed")
        self.assertEqual(preflight["scheduler_calls"], 4)
        self.assertEqual(CANCELLATION_EPISODES, 1024)
        self.assertEqual(CANCELLATION_CALL_LIMIT, 4096)

    def test_artifact_is_transactional_and_contains_jsonl_rows(self):
        result = run_cancellation_ablation(self.config, episode_count=1)
        with tempfile.TemporaryDirectory() as root:
            directory = write_cancellation_artifact(root, "cancel-run-1", result, self.config)
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"summary.json", "paired_statistics.json", "episode_rows.jsonl", "manifest.json", "protocol.json"},
            )
            rows = [json.loads(line) for line in (directory / "episode_rows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            with self.assertRaises(FileExistsError):
                write_cancellation_artifact(root, "cancel-run-1", result, self.config)


if __name__ == "__main__":
    unittest.main()
