from dataclasses import replace
import hashlib
from pathlib import Path
import unittest

import mfg_hedge.config as config_module
from mfg_hedge.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        self.assertEqual(config.expert_count, 1)
        self.assertEqual(config.replicas_per_expert, 2)
        self.assertAlmostEqual(config.single_domain_failure_load, 1.4)

    def test_default_config_flags_failure_capacity_risk(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        self.assertEqual(len(config.audit_findings()), 1)

    def test_ratios_must_sum_to_one(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        invalid = replace(config, urgent_token_ratio=0.3)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            invalid.validate()

    def test_config_and_digest_are_derived_from_the_same_bytes(self) -> None:
        path = PROJECT_ROOT / "configs" / "v1_minimal.json"
        expected_bytes = path.read_bytes()
        config, digest = config_module.load_config_with_sha256(path)
        self.assertEqual(config, load_config(path))
        self.assertEqual(digest, hashlib.sha256(expected_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
