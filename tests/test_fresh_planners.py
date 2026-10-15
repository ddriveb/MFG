from __future__ import annotations

import subprocess
import sys
import unittest
import runpy
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
ENTRY = V2 / "tools" / "fresh_planners.py"


class FreshPlannerCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entry = runpy.run_path(str(ENTRY))

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", str(ENTRY), *args],
            cwd=V2.parent,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_help_is_available_in_isolated_python_mode(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--campaign", result.stdout)

    def test_output_preflight_rejects_paths_outside_v2_artifacts(self) -> None:
        result = self.run_cli("--output", "../fresh-planner-output-outside")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("v2/artifacts", result.stderr)

    def test_each_campaign_regenerates_a_gate_and_matches_saved_topology(self) -> None:
        expected = {"g64": 64, "multiscale": 16, "global_hotspot": 32}
        for campaign, ranks in expected.items():
            with self.subTest(campaign=campaign):
                cases = list(self.entry["generated_cases"]({campaign}, 1))
                self.assertEqual(len(cases), 1)
                self.assertEqual(cases[0]["ranks"], ranks)
                self.assertEqual(len(cases[0]["counts"]), ranks * 16)


if __name__ == "__main__":
    unittest.main()
