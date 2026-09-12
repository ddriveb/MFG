"""Pure aggregation tests for the paired multi-seed campaign."""

import unittest

from mfg_hedge.campaign import (
    build_campaign_aggregate,
    evaluation_seeds,
    paired_statistic,
)


class CampaignAggregationTests(unittest.TestCase):
    def test_evaluation_seeds_are_contiguous_and_complete(self) -> None:
        self.assertEqual(evaluation_seeds(100, 4), (100, 101, 102, 103))
        with self.assertRaises(ValueError):
            evaluation_seeds(100, 0)

    def test_hand_computed_paired_statistic(self) -> None:
        result = paired_statistic((-1.0, -2.0, 0.0))
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], -1.0)
        self.assertAlmostEqual(result["sample_stddev"], 1.0)
        half = 4.302652729911275 / (3.0 ** 0.5)
        self.assertAlmostEqual(result["ci95"][0], -1.0 - half)
        self.assertAlmostEqual(result["ci95"][1], -1.0 + half)
        self.assertEqual(result["sign_counts"], {"negative": 2, "zero": 1, "positive": 0})

    def test_aggregate_uses_seed_level_paired_deltas(self) -> None:
        records = []
        for seed, a, b in ((10, 2.0, 1.0), (11, 4.0, 5.0)):
            records.append(
                {
                    "evaluation_seed": seed,
                    "comparison": {
                        "metrics": {
                            "mean_latency": {
                                "arm_a": a,
                                "arm_b": b,
                                "delta": b - a,
                                "relative": (b - a) / a,
                            }
                        }
                    },
                }
            )
        result = build_campaign_aggregate(records, metric_names=("mean_latency",))
        metric = result["metrics"]["mean_latency"]
        self.assertEqual(result["evaluation_seeds"], [10, 11])
        self.assertEqual(metric["arm_a_values"], [2.0, 4.0])
        self.assertEqual(metric["arm_b_values"], [1.0, 5.0])
        self.assertEqual(metric["delta_values"], [-1.0, 1.0])
        self.assertEqual(metric["delta"]["sign_counts"], {"negative": 1, "zero": 0, "positive": 1})


if __name__ == "__main__":
    unittest.main()
