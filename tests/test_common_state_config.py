import json
from pathlib import Path
import unittest
from unittest.mock import patch

from mfg_hedge.config import (
    CommonStateExperimentConfig,
    load_common_state_config_with_sha256,
    load_config,
    parse_common_state_config_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_minimal.json"
COMMON_STATE_CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_common_state.json"


def valid_payload() -> dict:
    raw = json.loads(BASE_CONFIG_PATH.read_bytes().decode("utf-8"))
    raw["schema_version"] = 2
    raw["degraded_start"] = 100.0
    raw["failed_start"] = 200.0
    raw["recovered_start"] = 220.0
    return raw


def parse(payload: dict) -> CommonStateExperimentConfig:
    return parse_common_state_config_bytes(json.dumps(payload).encode("utf-8"))


class CommonStateConfigTests(unittest.TestCase):
    def test_schema_two_config_loads(self) -> None:
        config, digest = load_common_state_config_with_sha256(COMMON_STATE_CONFIG_PATH)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.degraded_start, 100.0)
        self.assertEqual(config.failed_start, 200.0)
        self.assertEqual(config.recovered_start, 220.0)
        self.assertEqual(config.experiment_name, "single_expert_mechanism_check")
        self.assertEqual(len(digest), 64)

    def test_missing_and_unknown_keys_are_rejected(self) -> None:
        payload = valid_payload()
        del payload["degraded_start"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse(payload)
        payload = valid_payload()
        payload["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse(payload)

    def test_wrong_schema_version_is_rejected(self) -> None:
        payload = valid_payload()
        payload["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse(payload)

    def test_slice_dimensions_are_fixed(self) -> None:
        for key, value in (
            ("expert_count", 2),
            ("replicas_per_expert", 3),
            ("failure_domain_count", 3),
        ):
            payload = valid_payload()
            payload[key] = value
            with self.assertRaisesRegex(ValueError, key):
                parse(payload)

    def test_invalid_timeline_is_rejected(self) -> None:
        payload = valid_payload()
        payload["failed_start"] = 100.0  # equals degraded_start
        with self.assertRaises(ValueError):
            parse(payload)
        payload = valid_payload()
        payload["recovered_start"] = 199.0
        with self.assertRaises(ValueError):
            parse(payload)

    def test_bool_nan_and_infinity_are_rejected(self) -> None:
        payload = valid_payload()
        payload["degraded_start"] = True
        with self.assertRaises(ValueError):
            parse(payload)
        payload = valid_payload()
        payload["tokens_per_run"] = True
        with self.assertRaises(ValueError):
            parse(payload)
        payload = valid_payload()
        payload["failed_start"] = float("nan")
        with self.assertRaises(ValueError):
            parse(payload)
        payload = valid_payload()
        payload["healthy_offered_load"] = float("inf")
        with self.assertRaises(ValueError):
            parse(payload)

    def test_base_projection_matches_schema_one_config(self) -> None:
        config = parse(valid_payload())
        base = config.base_config()
        reference = load_config(BASE_CONFIG_PATH)
        self.assertEqual(base, reference)

    def test_config_and_digest_come_from_the_same_single_read(self) -> None:
        import hashlib

        expected_bytes = COMMON_STATE_CONFIG_PATH.read_bytes()
        original_open = Path.open
        open_count = 0

        def counting_open(path: Path, *args, **kwargs):
            nonlocal open_count
            if Path(path).resolve() == COMMON_STATE_CONFIG_PATH.resolve():
                open_count += 1
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", counting_open):
            config, digest = load_common_state_config_with_sha256(COMMON_STATE_CONFIG_PATH)
        self.assertEqual(open_count, 1)
        self.assertEqual(digest, hashlib.sha256(expected_bytes).hexdigest())
        self.assertEqual(config.schema_version, 2)


if __name__ == "__main__":
    unittest.main()
