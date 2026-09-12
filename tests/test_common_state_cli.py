"""CLI tests for simulate-common-state-no-hedge."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.__main__ import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_common_state.json"


def run_cli(artifacts_root: Path, run_id: str, tokens: int = 1000) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(
            [
                "simulate-common-state-no-hedge",
                "--config",
                str(CONFIG_PATH),
                "--tokens",
                str(tokens),
                "--run-id",
                run_id,
                "--artifacts-root",
                str(artifacts_root),
            ]
        )


def read_summary(artifacts_root: Path, run_id: str) -> dict:
    with (artifacts_root / run_id / "summary.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


class CommonStateCliTests(unittest.TestCase):
    def test_controlled_run_creates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "cs-run"), 0)
            summary = read_summary(root, "cs-run")
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["scenario"], "common_state_no_hedge")
            self.assertEqual(summary["token_count"], 1000)
            self.assertEqual(summary["generated_tokens"], 1000)
            self.assertEqual(
                summary["timeline"],
                {"degraded_start": 100.0, "failed_start": 200.0, "recovered_start": 220.0},
            )
            self.assertEqual(set(summary["phases"]), {"H", "D", "F", "R"})
            for phase in summary["phases"].values():
                self.assertTrue(phase["observed"])
            violations = {
                name: phase["capacity_violation"]
                for name, phase in summary["phases"].items()
            }
            self.assertEqual(violations, {"H": False, "D": True, "F": True, "R": False})
            self.assertGreaterEqual(
                summary["drain_end_time"], summary["last_arrival_time"]
            )
            self.assertLessEqual(summary["latency_p50"], summary["latency_p95"])
            self.assertLessEqual(summary["latency_p95"], summary["latency_p99"])
            faults = summary["faults"]
            self.assertEqual(faults["completed_tokens"], 1000)
            self.assertEqual(faults["hedge_launches"], 0)
            self.assertEqual(faults["replay_executions"], faults["replayed_tokens"])
            work = summary["work"]
            self.assertAlmostEqual(
                work["total_executed_work"],
                work["executed_primary_work"] + work["executed_replay_work"],
            )
            self.assertAlmostEqual(
                work["extra_execution_ratio"],
                work["executed_replay_work"] / work["nominal_primary_work"],
            )
            self.assertAlmostEqual(
                work["execution_amplification"],
                work["total_executed_work"] / work["nominal_primary_work"],
            )
            for value in summary["invariants"].values():
                if isinstance(value, bool):
                    self.assertTrue(value)

    def test_short_trace_fails_before_creating_a_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "cs-short", tokens=50), 2)
            self.assertFalse((root / "cs-short").exists())

    def test_same_seed_runs_are_identical_except_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "cs-a"), 0)
            self.assertEqual(run_cli(root, "cs-b"), 0)
            first = read_summary(root, "cs-a")
            second = read_summary(root, "cs-b")
            first.pop("run_id")
            second.pop("run_id")
            self.assertEqual(first, second)

    def test_existing_run_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "cs-a"), 0)
            before = (root / "cs-a" / "summary.json").read_bytes()
            self.assertEqual(run_cli(root, "cs-a"), 2)
            self.assertEqual((root / "cs-a" / "summary.json").read_bytes(), before)

    def test_run_id_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            self.assertEqual(run_cli(root, "../cs-escape"), 2)
            self.assertFalse((Path(tmp) / "cs-escape").exists())

    def test_configuration_is_read_exactly_once(self) -> None:
        original_open = Path.open
        target = CONFIG_PATH.resolve()
        open_count = 0

        def counting_open(path: Path, *args, **kwargs):
            nonlocal open_count
            if Path(path).resolve() == target:
                open_count += 1
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "open", counting_open):
            self.assertEqual(run_cli(Path(tmp), "cs-single-read"), 0)
        self.assertEqual(open_count, 1)


if __name__ == "__main__":
    unittest.main()
