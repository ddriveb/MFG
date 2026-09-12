import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.config import load_config
from tests.test_selective_hedging import make_episode
from mfg_hedge.selective_hedging_experiment import (
    SELECTIVE_DEVELOPMENT_CALL_BUDGET,
    SELECTIVE_DEVELOPMENT_EPISODES,
    SELECTIVE_DEVELOPMENT_ARM_KEYS,
    SELECTIVE_PLANNED_BUDGET_RATE,
    SELECTIVE_THRESHOLD_GRID,
    build_selective_development_protocol,
    run_selective_development,
    run_selective_development_preflight,
    select_development_threshold,
    write_selective_hedging_artifact,
)


class SelectiveHedgeExperimentContractTests(unittest.TestCase):
    def test_frozen_rate_grid_arm_count_and_call_budget(self):
        self.assertEqual(SELECTIVE_PLANNED_BUDGET_RATE, 0.09304612180547656)
        self.assertEqual(SELECTIVE_THRESHOLD_GRID, (0.15, 0.30, 0.50))
        self.assertEqual(len(SELECTIVE_DEVELOPMENT_ARM_KEYS), 12)
        self.assertEqual(
            SELECTIVE_DEVELOPMENT_CALL_BUDGET,
            SELECTIVE_DEVELOPMENT_EPISODES * 12,
        )

    def test_protocol_records_nominal_mapping_without_claiming_achieved_delta(self):
        protocol = build_selective_development_protocol()
        self.assertEqual(protocol["planned_budget_rate"], SELECTIVE_PLANNED_BUDGET_RATE)
        self.assertEqual(protocol["planned_budget_rate_source"], "budgeted-laedge-v2-nominal-5-percent-mapping")
        self.assertFalse(protocol["planned_rate_means_achieved_delta"])
        self.assertTrue(protocol["underfill_is_valid"])

    def test_selection_is_deterministic_and_uses_delta_zero_baseline(self):
        rows = {
            "selective-laedge:theta_regular=0.15:theta_urgent=0.15": {
                "df_cvar95": {"ci95": [-0.2, -0.01]},
                "df_miss_rate": {"ci95": [-0.01, 0.02]},
                "total_work": {"relative_difference": 0.03},
                "storm_peak": 1.0,
            },
            "selective-laedge:theta_regular=0.3:theta_urgent=0.3": {
                "df_cvar95": {"ci95": [-0.3, -0.05]},
                "df_miss_rate": {"ci95": [-0.2, -0.01]},
                "total_work": {"relative_difference": 0.04},
                "storm_peak": 1.5,
            },
        }
        selected = select_development_threshold(rows)
        self.assertEqual(
            selected["arm_key"],
            "selective-laedge:theta_regular=0.3:theta_urgent=0.3",
        )
        self.assertEqual(selected["status"], "confirmed")

    def test_selection_fails_closed_when_no_candidate_passes(self):
        rows = {
            "selective-laedge:theta_regular=0.15:theta_urgent=0.15": {
                "df_cvar95": {"ci95": [-0.2, 0.1]},
                "df_miss_rate": {"ci95": [-0.01, 0.02]},
                "total_work": {"relative_difference": 0.03},
                "storm_peak": 1.0,
            }
        }
        selected = select_development_threshold(rows)
        self.assertEqual(selected["status"], "non_improving")
        self.assertIsNone(selected["arm_key"])

    def test_artifact_writer_is_transactional_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "summary.json": {"status": "non_improving"},
                "protocol.json": {"schema_id": "selective"},
            }
            written = write_selective_hedging_artifact(root, "development-r1", files)
            self.assertEqual(json.loads((written / "summary.json").read_text()), files["summary.json"])
            with self.assertRaises(FileExistsError):
                write_selective_hedging_artifact(root, "development-r1", files)

    def test_failed_episode_consumes_every_arm_call_without_retry(self):
        config = load_config(Path("configs/v1_minimal.json"))
        failed_arms = {
            arm_key: {
                "status": "failed",
                "exception_type": "SyntheticFailure",
                "exception_message": "synthetic physical failure",
            }
            for arm_key in SELECTIVE_DEVELOPMENT_ARM_KEYS
        }
        episode_result = {
            "status": "physical_failed",
            "trace_fingerprint": "synthetic-trace",
            "arms": failed_arms,
            "failures": list(failed_arms.values()),
        }
        with patch(
            "mfg_hedge.selective_hedging_experiment._run_episode",
            return_value=episode_result,
        ) as run_episode:
            result = run_selective_development(
                config,
                episode_builder=lambda _namespace, _index: make_episode(),
                episode_count=1,
                namespace="replica-routing-baselines:test:development",
                macro_seed=20260916,
            )
        self.assertEqual(run_episode.call_count, 1)
        self.assertEqual(result["scheduler_calls"], len(SELECTIVE_DEVELOPMENT_ARM_KEYS))
        self.assertEqual(result["formal_scheduler_call_budget"], len(SELECTIVE_DEVELOPMENT_ARM_KEYS))
        self.assertEqual(result["status"], "physical_failed")
        self.assertEqual(result["paired_statistics"]["status"], "unavailable")
        self.assertEqual(len(result["failures"]), len(SELECTIVE_DEVELOPMENT_ARM_KEYS))

    def test_development_runner_does_not_relabel_fixed_rate_as_achieved_delta(self):
        protocol = build_selective_development_protocol()
        self.assertFalse(protocol["planned_rate_means_achieved_delta"])
        self.assertTrue(protocol["underfill_is_valid"])
        self.assertEqual(
            protocol["planned_budget_rate_source"],
            "budgeted-laedge-v2-nominal-5-percent-mapping",
        )

    def test_preflight_uses_isolated_identity_and_never_spends_formal_episode_ids(self):
        config = load_config(Path("configs/v1_minimal.json"))
        with patch(
            "mfg_hedge.selective_hedging_experiment._run_episode",
            return_value={"status": "completed", "failures": []},
        ) as run_episode:
            result = run_selective_development_preflight(
                config,
                episode_builder=lambda _namespace, _index: make_episode(),
                episode_count=1,
            )
        self.assertEqual(run_episode.call_count, 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scheduler_calls"], len(SELECTIVE_DEVELOPMENT_ARM_KEYS))
        self.assertIn(":preflight", result["namespace"])
        self.assertNotEqual(result["namespace"], build_selective_development_protocol()["namespace"])

    def test_generated_selective_preflight_preserves_attribution_invariants(self):
        config = load_config(Path("configs/v1_minimal.json"))
        result = run_selective_development_preflight(config, episode_count=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["failures"], [])


if __name__ == "__main__":
    unittest.main()
