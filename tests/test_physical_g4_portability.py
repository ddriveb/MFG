from __future__ import annotations

import json
import importlib
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
ENTRY = V2 / "tools" / "physical_g4_zero_template.py"


class PhysicalG4PortabilityTests(unittest.TestCase):
    def test_portable_g4_qualifier_help_runs_isolated(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", str(V2 / "tools" / "physical_g4_qualify.py"), "--help"],
            cwd=V2.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recorded-planner", result.stdout)
        self.assertIn("--limit-states", result.stdout)
        self.assertIn("--python-include", result.stdout)

    def test_physical_g4_runner_selftest_compiles_and_keeps_strict_fail(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", str(V2 / "tools" / "run_physical_g4.py"), "--selftest"],
            cwd=V2.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertTrue(report["runner_compiles"])
        self.assertTrue(report["incomplete_rank_rejected"])
        self.assertTrue(report["fused_planner_hook_stub_tested"])
        self.assertTrue(report["strict_full_model_failure_retained"])
        self.assertFalse(report["gpu_execution"])

    def test_launcher_rejects_local_tiny_gloo_before_runtime_setup(self) -> None:
        for unsupported_args in (("--local-tiny",), ("--init-file", "unused-init")):
            with self.subTest(args=unsupported_args):
                result = subprocess.run(
                    [sys.executable, "-I", str(V2 / "tools" / "run_physical_g4.py"), *unsupported_args],
                    cwd=V2.parent,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsupported by the fused CUDA ports", result.stderr)

    def test_launcher_rejects_non_four_round_configs_before_runtime_setup(self) -> None:
        config = json.loads((V2 / "configs" / "g4-optimized-paired.json").read_text(encoding="utf-8"))
        config["rounds"] = 8
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "g4-rounds-8.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(V2 / "tools" / "run_physical_g4.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=V2.parent,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("optimized G4 ports support rounds=4 only", result.stderr)
        self.assertNotIn("--output is required", result.stderr)

    def test_external_runtime_is_offline_and_assets_fail_before_torch_setup(self) -> None:
        launcher = runpy.run_path(str(V2 / "tools" / "run_physical_g4.py"))
        environment = {}
        launcher["configure_runtime_environment"](environment)
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
            launcher["configure_runtime_environment"]({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "model"
            model_dir.mkdir()
            schedule = base / "schedule.so"
            schedule.write_bytes(b"stub")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(V2 / "tools" / "run_physical_g4.py"),
                    "--output",
                    "artifacts/physical-g4-no-assets-test",
                    "--model-dir",
                    str(model_dir),
                    "--schedule-library",
                    str(schedule),
                ],
                cwd=V2.parent,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required local Granite asset is missing", result.stderr)
        self.assertNotIn("--output is required", result.stderr)

    def test_launcher_installs_and_restores_optimized_planner_hooks(self) -> None:
        launcher = runpy.run_path(str(V2 / "tools" / "run_physical_g4.py"))
        planner_module = importlib.import_module("mfg_hedge.granite_g4_feedback_planner_v1")
        replay_module = importlib.import_module("mfg_hedge.granite_g4_feedback_replay_v1")
        original_planner = planner_module.DenseSupportPlanner
        original_planner_baseline = planner_module.baseline_quotas
        original_replay_baseline = replay_module.baseline_quotas

        class StubOptimizedPlanner:
            pass

        def stub_optimized_baseline(*args, **kwargs):
            return None

        ports = SimpleNamespace(
            OptimizedPlanner=StubOptimizedPlanner,
            optimized_baseline=stub_optimized_baseline,
        )
        with launcher["install_optimized_planners"](ports):
            self.assertIs(planner_module.DenseSupportPlanner, StubOptimizedPlanner)
            self.assertIs(planner_module.baseline_quotas, stub_optimized_baseline)
            self.assertIs(replay_module.baseline_quotas, stub_optimized_baseline)
        self.assertIs(planner_module.DenseSupportPlanner, original_planner)
        self.assertIs(planner_module.baseline_quotas, original_planner_baseline)
        self.assertIs(replay_module.baseline_quotas, original_replay_baseline)

    def test_zero_count_rows_retain_fixed_template_and_refresh_features(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", str(ENTRY)],
            cwd=V2.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["rows"], 128)
        self.assertEqual(report["variables"], 256)
        self.assertEqual(report["zero_rows"], 26)
        self.assertTrue(report["fixed_support_preserved"])
        self.assertTrue(report["zero_coefficients_and_features_refreshed"])
        self.assertIn("no physical timing claim", report["execution"])


if __name__ == "__main__":
    unittest.main()
