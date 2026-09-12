"""CLI contract tests for a complete transactional campaign."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mfg_hedge.__main__ import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "v1_paired_comparison.json"


def fixture_table():
    from tests.test_mfg_solver import make_table, hedge_attractive_stats

    return make_table(hedge_attractive_stats)


class CampaignCliTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        payload = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        payload["base_seed"] = 101
        payload["seed_count"] = 3
        payload["tokens_per_run"] = 300
        payload["smoke_tokens"] = 100
        path = root / "campaign.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run(self, root: Path, run_id: str) -> tuple[int, int]:
        config = self._config(root)
        builds = 0

        def build_once(_config):
            nonlocal builds
            builds += 1
            return fixture_table()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch("mfg_hedge.__main__.build_calibration_table", side_effect=build_once):
                rc = main(
                    [
                        "campaign-mfg-hedge-vs-no-hedge",
                        "--config",
                        str(config),
                        "--run-id",
                        run_id,
                        "--artifacts-root",
                        str(root / "artifacts"),
                    ]
                )
        return rc, builds

    def test_campaign_writes_every_seed_and_one_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rc, builds = self._run(root, "campaign-a")
            self.assertEqual(rc, 0)
            self.assertEqual(builds, 1)
            run_dir = root / "artifacts" / "campaign-a"
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "manifest.json",
                    "aggregate.json",
                    "seed_101.json",
                    "seed_102.json",
                    "seed_103.json",
                },
            )
            aggregate = json.loads((run_dir / "aggregate.json").read_text())
            self.assertEqual(aggregate["evaluation_seeds"], [101, 102, 103])
            fingerprints = []
            for seed in (101, 102, 103):
                record = json.loads((run_dir / f"seed_{seed}.json").read_text())
                self.assertEqual(record["evaluation_seed"], seed)
                self.assertTrue(all(record["invariants"]["arm_a"].values()))
                self.assertTrue(all(record["invariants"]["arm_b"].values()))
                fingerprints.append(record["trace_identity"]["tokens"])
            self.assertEqual(len(set(fingerprints)), 3)

    def test_existing_campaign_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self._run(root, "campaign-a")[0], 0)
            before = (root / "artifacts" / "campaign-a" / "aggregate.json").read_bytes()
            self.assertEqual(self._run(root, "campaign-a")[0], 2)
            self.assertEqual(
                (root / "artifacts" / "campaign-a" / "aggregate.json").read_bytes(),
                before,
            )


if __name__ == "__main__":
    unittest.main()
