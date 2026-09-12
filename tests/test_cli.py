import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.__main__ import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_minimal.json"


def run_cli(artifacts_root: Path, run_id: str) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(
            [
                "simulate-healthy-no-hedge",
                "--config",
                str(CONFIG_PATH),
                "--tokens",
                "120",
                "--run-id",
                run_id,
                "--artifacts-root",
                str(artifacts_root),
            ]
        )


def read_summary(artifacts_root: Path, run_id: str) -> dict:
    path = artifacts_root / run_id / "summary.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class SimulateHealthyNoHedgeCliTests(unittest.TestCase):
    def test_cli_creates_unique_run_directory_with_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            summary_path = root / "run-a" / "summary.json"
            self.assertTrue(summary_path.is_file())
            summary = read_summary(root, "run-a")
            self.assertEqual(summary["run_id"], "run-a")
            self.assertEqual(summary["token_count"], 120)

    def test_cli_refuses_to_overwrite_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            first_bytes = (root / "run-a" / "summary.json").read_bytes()
            self.assertNotEqual(run_cli(root, "run-a"), 0)
            self.assertEqual((root / "run-a" / "summary.json").read_bytes(), first_bytes)

    def test_cli_rejects_run_id_that_escapes_artifacts_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            self.assertNotEqual(run_cli(root, "../escape_probe"), 0)
            self.assertFalse((Path(tmp) / "escape_probe").exists())

    def test_same_seed_produces_identical_summary_except_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run_cli(root, "run-a"), 0)
            self.assertEqual(run_cli(root, "run-b"), 0)
            first = read_summary(root, "run-a")
            second = read_summary(root, "run-b")
            first.pop("run_id")
            second.pop("run_id")
            self.assertEqual(first, second)

    def test_cli_opens_configuration_file_exactly_once(self) -> None:
        original_open = Path.open
        config_path = CONFIG_PATH.resolve()
        config_open_count = 0

        def counting_open(path: Path, *args, **kwargs):
            nonlocal config_open_count
            if path.resolve() == config_path:
                config_open_count += 1
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            Path, "open", counting_open
        ):
            self.assertEqual(run_cli(Path(tmp), "single-read"), 0)

        self.assertEqual(config_open_count, 1)


if __name__ == "__main__":
    unittest.main()
