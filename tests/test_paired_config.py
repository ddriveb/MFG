"""Paired-comparison config (schema 3) tests."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from mfg_hedge.config import (
    PairedExperimentConfig,
    load_config,
    load_paired_config_with_sha256,
    parse_paired_config_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "configs" / "v1_minimal.json"
PAIRED_PATH = PROJECT_ROOT / "configs" / "v1_paired_comparison.json"
RUNTIME_COST_PATH = PROJECT_ROOT / "configs" / "v2_paired_runtime_cost.json"


def valid_payload() -> dict:
    raw = json.loads(BASE_PATH.read_bytes().decode("utf-8"))
    raw["schema_version"] = 3
    raw["healthy_offered_load"] = 0.5
    raw["degraded_start"] = 100.0
    raw["failed_start"] = 200.0
    raw["recovered_start"] = 220.0
    raw["control_window"] = 25.0
    raw["storm_bin_width"] = 1.0
    raw["replay_penalty_regular"] = 1.0
    raw["replay_penalty_urgent"] = 5.0
    return raw


def parse(payload: dict) -> PairedExperimentConfig:
    return parse_paired_config_bytes(json.dumps(payload).encode("utf-8"))


class PairedConfigTests(unittest.TestCase):
    def test_paired_config_loads(self) -> None:
        config, digest = load_paired_config_with_sha256(PAIRED_PATH)
        self.assertEqual(config.schema_version, 3)
        self.assertEqual(config.healthy_offered_load, 0.5)
        self.assertEqual(config.control_window, 25.0)
        self.assertEqual(config.storm_bin_width, 1.0)
        self.assertEqual(config.replay_penalty_regular, 1.0)
        self.assertEqual(config.replay_penalty_urgent, 5.0)
        self.assertEqual(
            (config.degraded_start, config.failed_start, config.recovered_start),
            (100.0, 200.0, 220.0),
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(config.incremental_work_cost, 0.0)
        self.assertEqual(config.wasted_work_cost, 0.0)

    def test_schema_four_runtime_cost_config_loads(self) -> None:
        config, digest = load_paired_config_with_sha256(RUNTIME_COST_PATH)
        self.assertEqual(config.schema_version, 4)
        self.assertEqual(config.incremental_work_cost, 1.0)
        self.assertEqual(config.wasted_work_cost, 1.0)
        self.assertEqual(len(digest), 64)

    def test_schema_four_costs_reach_the_paired_solver(self) -> None:
        from tests.test_mfg_solver import hedge_attractive_stats, make_table
        from mfg_hedge.paired import solve_paired_policy

        config, _ = load_paired_config_with_sha256(RUNTIME_COST_PATH)
        solution = solve_paired_policy(config, make_table(hedge_attractive_stats))
        self.assertEqual(solution.parameters.incremental_work_cost, 1.0)
        self.assertEqual(solution.parameters.wasted_work_cost, 1.0)

    def test_schema_four_requires_strictly_positive_costs(self) -> None:
        payload = valid_payload()
        payload["schema_version"] = 4
        payload["incremental_work_cost"] = 1.0
        payload["wasted_work_cost"] = 1.0
        for key in ("incremental_work_cost", "wasted_work_cost"):
            missing = dict(payload)
            del missing[key]
            with self.assertRaisesRegex(ValueError, "Missing"):
                parse(missing)
            for bad in (0.0, -0.1, True, float("nan"), float("inf")):
                invalid = dict(payload)
                invalid[key] = bad
                with self.assertRaises(ValueError, msg=f"{key}={bad!r}"):
                    parse(invalid)

    def test_base_projection_matches_schema_one(self) -> None:
        config = parse(valid_payload())
        base = config.base_config()
        reference = load_config(BASE_PATH)
        self.assertEqual(base.healthy_offered_load, 0.5)
        self.assertEqual(base.service_time_cv, reference.service_time_cv)
        self.assertEqual(base.base_seed, reference.base_seed)

    def test_missing_unknown_and_illegal_keys_fail(self) -> None:
        payload = valid_payload()
        del payload["control_window"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse(payload)
        payload = valid_payload()
        payload["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown"):
            parse(payload)
        payload = valid_payload()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "schema_version"):
            parse(payload)

    def test_bool_nan_infinity_rejected(self) -> None:
        for key, bad in (
            ("control_window", True),
            ("storm_bin_width", float("nan")),
            ("replay_penalty_urgent", float("inf")),
            ("healthy_offered_load", "0.5"),
        ):
            payload = valid_payload()
            payload[key] = bad
            with self.assertRaises(ValueError, msg=f"{key}={bad!r}"):
                parse(payload)

    def test_slice_dimensions_fixed(self) -> None:
        payload = valid_payload()
        payload["replicas_per_expert"] = 3
        with self.assertRaisesRegex(ValueError, "replicas_per_expert"):
            parse(payload)

    def test_config_is_read_exactly_once(self) -> None:
        import hashlib

        expected = PAIRED_PATH.read_bytes()
        original_open = Path.open
        count = 0

        def counting_open(path, *args, **kwargs):
            nonlocal count
            if Path(path).resolve() == PAIRED_PATH.resolve():
                count += 1
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", counting_open):
            config, digest = load_paired_config_with_sha256(PAIRED_PATH)
        self.assertEqual(count, 1)
        self.assertEqual(digest, hashlib.sha256(expected).hexdigest())


if __name__ == "__main__":
    unittest.main()
