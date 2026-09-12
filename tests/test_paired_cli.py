"""Paired CLI and transactional artifact tests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.__main__ import main
from mfg_hedge.artifacts import write_run_directory
from mfg_hedge.common_state import CommonStateTimeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_paired_comparison.json"
RUNTIME_COST_CONFIG_PATH = PROJECT_ROOT / "configs" / "v2_paired_runtime_cost.json"
EXPECTED_FILES = {
    "manifest.json",
    "no_hedge_summary.json",
    "mfg_hedge_summary.json",
    "comparison.json",
}


def fixture_table():
    from tests.test_mfg_solver import make_table, hedge_attractive_stats

    return make_table(hedge_attractive_stats)


def run_cli(
    artifacts_root: Path,
    run_id: str,
    tokens: int = 300,
    config_path: Path = CONFIG_PATH,
) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with patch(
            "mfg_hedge.__main__.build_calibration_table",
            side_effect=lambda config: fixture_table(),
        ):
            return main(
                [
                    "compare-mfg-hedge-vs-no-hedge",
                    "--config",
                    str(config_path),
                    "--tokens",
                    str(tokens),
                    "--run-id",
                    run_id,
                    "--artifacts-root",
                    str(artifacts_root),
                ]
            )


class PairedCliTests(unittest.TestCase):
    def test_runtime_cost_config_is_auditable_in_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                run_cli(
                    root,
                    "runtime-cost",
                    config_path=RUNTIME_COST_CONFIG_PATH,
                ),
                0,
            )
            comparison = json.loads(
                (root / "runtime-cost" / "comparison.json").read_text()
            )
            self.assertEqual(
                comparison["config"]["resolved_config"]["incremental_work_cost"],
                1.0,
            )
            self.assertEqual(
                comparison["config"]["resolved_config"]["wasted_work_cost"],
                1.0,
            )
            parameters = comparison["solver_parameters"]
            self.assertEqual(parameters["incremental_work_cost"], 1.0)
            self.assertEqual(parameters["wasted_work_cost"], 1.0)

    def test_successful_run_writes_exactly_four_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            run_dir = root / "run-a"
            self.assertEqual({p.name for p in run_dir.iterdir()}, EXPECTED_FILES)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "run-a")
            self.assertEqual(manifest["scenario"], "paired_mfg_vs_no_hedge")
            comparison = json.loads((run_dir / "comparison.json").read_text())
            self.assertIn("trace_identity", comparison)
            self.assertIn("mean_latency", comparison["metrics"])
            arm_b = json.loads((run_dir / "mfg_hedge_summary.json").read_text())
            self.assertGreater(arm_b["hedge"]["launched"], 0)
            arm_a = json.loads((run_dir / "no_hedge_summary.json").read_text())
            self.assertEqual(arm_a["hedge"]["launched"], 0)
            per_window = arm_b["actions"]["per_window"]
            self.assertTrue(per_window)
            for window in per_window:
                if window["phase"] == "F":
                    self.assertEqual(window["applied_counts"]["D"], 0)
                    self.assertEqual(window["applied_counts"]["I"], 0)

    def test_two_runs_byte_identical_except_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            self.assertEqual(run_cli(root, "run-b"), 0)
            for name in ("no_hedge_summary.json", "mfg_hedge_summary.json", "comparison.json"):
                self.assertEqual(
                    (root / "run-a" / name).read_bytes(),
                    (root / "run-b" / name).read_bytes(),
                    msg=name,
                )
            manifest_a = json.loads((root / "run-a" / "manifest.json").read_text())
            manifest_b = json.loads((root / "run-b" / "manifest.json").read_text())
            manifest_a.pop("run_id")
            manifest_b.pop("run_id")
            self.assertEqual(manifest_a, manifest_b)

    def test_existing_run_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            before = (root / "run-a" / "comparison.json").read_bytes()
            self.assertEqual(run_cli(root, "run-a"), 2)
            self.assertEqual((root / "run-a" / "comparison.json").read_bytes(), before)

    def test_run_id_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            self.assertEqual(run_cli(root, "../escape"), 2)
            self.assertFalse((Path(tmp) / "escape").exists())
            self.assertFalse(root.exists() and any(root.iterdir()))

    def test_config_is_read_exactly_once(self) -> None:
        original_open = Path.open
        count = 0

        def counting_open(path, *args, **kwargs):
            nonlocal count
            if Path(path).resolve() == CONFIG_PATH.resolve():
                count += 1
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "open", counting_open):
            self.assertEqual(run_cli(Path(tmp), "single-read"), 0)
        self.assertEqual(count, 1)

    def test_short_trace_fails_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "short", tokens=20), 2)
            self.assertFalse((root / "short").exists())
            self.assertEqual(list(root.iterdir()), [])


class TransactionalWriteTests(unittest.TestCase):
    def test_failed_serialization_leaves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TypeError):
                write_run_directory(
                    root,
                    "run-x",
                    {
                        "manifest.json": {"ok": True},
                        "broken.json": {"bad": object()},
                    },
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_final_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run_directory(root, "run-x", {"a.json": {"v": 1}})
            with self.assertRaises(FileExistsError):
                write_run_directory(root, "run-x", {"a.json": {"v": 2}})

    def test_staging_dir_is_cleaned_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("mfg_hedge.artifacts.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    write_run_directory(root, "run-y", {"a.json": {"v": 1}})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
