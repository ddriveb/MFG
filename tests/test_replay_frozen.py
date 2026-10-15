from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]


class FrozenReplayCliTests(unittest.TestCase):
    def test_replays_all_saved_campaigns_from_the_v2_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "-I", str(V2 / "tools" / "replay_frozen.py"), "--scope", "all"],
            cwd=V2.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["campaigns"]["g64"]["states"], 150)
        self.assertEqual(report["campaigns"]["multiscale"]["states"], 192)
        self.assertEqual(report["campaigns"]["global_hotspot"]["states"], 96)
        self.assertEqual(report["campaigns"]["g4"]["groups"], 150)
        self.assertEqual(report["price_certificates_replayed"], 36)
        self.assertEqual(report["fresh_gates"]["g64"], 150)
        self.assertEqual(report["fresh_gates"]["multiscale"], 192)
        self.assertEqual(report["fresh_gates"]["global_hotspot"], 96)
        focus = report["numeric_summary"]["historical_focal_38"]
        self.assertEqual(focus["sinkhorn_fixed"]["wins"], 38)
        self.assertEqual(focus["sinkhorn_fixed"]["saving_percent"], 12.772851680146045)
        self.assertEqual(focus["lazarus_inspired"]["saving_percent"], 12.510141327440655)
        self.assertEqual(report["numeric_summary"]["g64_150_state"]["paired_path_comparison"]["sinkhorn_fixed"]["paired_wins"], 144)


if __name__ == "__main__":
    unittest.main()
