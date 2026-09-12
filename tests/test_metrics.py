from dataclasses import asdict
import hashlib
from pathlib import Path
import unittest

import mfg_hedge
from mfg_hedge.config import load_config
from mfg_hedge.metrics import build_summary, percentile
from mfg_hedge.simulation import simulate_healthy_no_hedge
from mfg_hedge.workload import generate_workload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "v1_minimal.json"


class PercentileTests(unittest.TestCase):
    def test_linear_interpolation_on_even_sample(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(data, 50.0), 2.5)
        self.assertAlmostEqual(percentile(data, 25.0), 1.75)

    def test_extreme_percentiles_hit_the_endpoints(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(data, 0.0), 1.0)
        self.assertAlmostEqual(percentile(data, 100.0), 4.0)

    def test_interpolates_between_inner_points(self) -> None:
        data = [float(value) for value in range(101)]
        self.assertAlmostEqual(percentile(data, 99.0), 99.0)
        self.assertAlmostEqual(percentile([10.0, 20.0], 90.0), 19.0)

    def test_unsorted_input_is_sorted(self) -> None:
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 50.0), 2.5)

    def test_single_value_returns_that_value(self) -> None:
        self.assertAlmostEqual(percentile([7.5], 95.0), 7.5)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 50.0)

    def test_out_of_range_percent_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            percentile([1.0], 100.1)
        with self.assertRaises(ValueError):
            percentile([1.0], -0.1)


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.token_count = 300
        trace = generate_workload(self.config, self.token_count)
        result = simulate_healthy_no_hedge(
            trace, replica_count=self.config.replicas_per_expert
        )
        self.config_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
        self.summary = build_summary(
            self.config,
            trace,
            result,
            run_id="unit-test",
            config_sha256=self.config_sha256,
        )

    def test_summary_reports_identity_and_load(self) -> None:
        summary = self.summary
        self.assertEqual(summary["run_id"], "unit-test")
        self.assertEqual(summary["experiment_name"], "single_expert_mechanism_check")
        self.assertEqual(summary["base_seed"], 20260901)
        self.assertEqual(summary["token_count"], self.token_count)
        self.assertAlmostEqual(summary["arrival_rate"], 1.4)

    def test_summary_reports_full_configuration_identity(self) -> None:
        summary = self.summary
        self.assertEqual(summary["config_schema_version"], 1)
        self.assertEqual(summary["resolved_config"], asdict(self.config))
        self.assertEqual(summary["config_sha256"], self.config_sha256)
        self.assertEqual(summary["simulator_version"], mfg_hedge.__version__)
        self.assertEqual(summary["scenario"], "healthy_no_hedge")
        self.assertEqual(summary["dispatcher"], "round_robin")
        self.assertEqual(summary["policy"], "no_hedge")

    def test_summary_reports_latency_and_delay_statistics(self) -> None:
        summary = self.summary
        for key in ("mean_latency", "latency_p50", "latency_p95", "latency_p99", "mean_queue_delay"):
            self.assertGreater(summary[key], 0.0)
        self.assertLessEqual(summary["latency_p50"], summary["latency_p95"])
        self.assertLessEqual(summary["latency_p95"], summary["latency_p99"])
        self.assertLessEqual(summary["mean_queue_delay"], summary["mean_latency"])

    def test_summary_reports_per_replica_load_and_invariants(self) -> None:
        summary = self.summary
        replicas = summary["replicas"]
        self.assertEqual(len(replicas), 2)
        assigned = [entry["assigned"] for entry in replicas]
        self.assertEqual(sum(assigned), self.token_count)
        self.assertLessEqual(max(assigned) - min(assigned), 1)
        for entry in replicas:
            self.assertGreaterEqual(entry["utilization"], 0.0)
            self.assertLessEqual(entry["utilization"], 1.0)
            self.assertGreater(entry["busy_time"], 0.0)
        invariants = summary["invariants"]
        self.assertEqual(invariants["primary_executions"], self.token_count)
        self.assertEqual(invariants["hedge_launches"], 0)
        self.assertEqual(invariants["replay_executions"], 0)

    def test_summary_reports_throughput_over_observation_interval(self) -> None:
        summary = self.summary
        interval = summary["observation_interval"]
        self.assertEqual(interval["start"], 0.0)
        self.assertGreater(interval["end"], 0.0)
        self.assertAlmostEqual(
            summary["throughput"], self.token_count / interval["end"]
        )


if __name__ == "__main__":
    unittest.main()
