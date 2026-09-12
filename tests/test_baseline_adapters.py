import unittest
from dataclasses import dataclass

from mfg_hedge.baseline_adapters import (
    BackendCapabilities,
    BaselineAdapterError,
    ComparisonPanel,
    adapt_common_metrics,
    build_default_comparison_plan,
)
from mfg_hedge.replica_routing_baselines import ReplicaLoad


class PlanTests(unittest.TestCase):
    def test_two_panels_have_exact_frozen_membership(self):
        plan = build_default_comparison_plan()
        self.assertEqual(tuple(plan), ("protection_same_topology", "routing_after_placement"))
        self.assertEqual(tuple(plan["protection_same_topology"].arms), (
            "failover_only", "p95_delayed_hedge", "laedge_work_conserving",
            "pi_NIIN_v1", "pi_requested_reservation_price_v1",
        ))
        self.assertEqual(tuple(plan["routing_after_placement"].arms), (
            "uniform_round_robin", "join_shortest_queue",
            "least_unfinished_work", "lplb_token_count_minmax",
        ))
        self.assertIsNone(plan["protection_same_topology"].placement_id)
        self.assertEqual(plan["routing_after_placement"].placement_id, "eplb_style")

    def test_eplb_is_not_a_measured_arm(self):
        for panel in build_default_comparison_plan().values():
            self.assertNotIn("eplb_style", panel.arms)

    def test_plan_construction_does_not_read_sealed_candidate_artifact(self):
        plan = build_default_comparison_plan()
        arm = plan["protection_same_topology"].arms["pi_requested_reservation_price_v1"]
        self.assertEqual(arm.constructor_kind, "sealed_existing_policy")
        self.assertTrue(arm.source_reference.endswith("summary.json"))


class CapabilityTests(unittest.TestCase):
    def test_current_engine_fails_closed_for_laedge_and_lplb(self):
        current = BackendCapabilities(
            online_arrivals=True,
            per_replica_fcfs=True,
            failure_replay=True,
            delayed_timers=True,
        )
        plan = build_default_comparison_plan()
        with self.assertRaisesRegex(BaselineAdapterError, "idle_release"):
            plan["protection_same_topology"].validate_backend(current)
        with self.assertRaisesRegex(BaselineAdapterError, "batch_release"):
            plan["routing_after_placement"].validate_backend(current)

    def test_complete_panel_backends_pass(self):
        protection = BackendCapabilities(
            online_arrivals=True, per_replica_fcfs=True, failure_replay=True,
            delayed_timers=True, idle_release=True, reservation_projection=True,
        )
        routing = BackendCapabilities(
            online_arrivals=True, per_replica_fcfs=True, failure_replay=True,
            primary_router=True, batch_release=True, gpu_capacity=True,
        )
        plan = build_default_comparison_plan()
        plan["protection_same_topology"].validate_backend(protection)
        plan["routing_after_placement"].validate_backend(routing)


class ConstructionTests(unittest.TestCase):
    def test_online_routing_adapters_delegate_to_baseline_kernels(self):
        rows = (
            ReplicaLoad(0, 2, 1.0),
            ReplicaLoad(1, 0, 4.0),
            ReplicaLoad(2, 1, 0.5),
        )
        arms = build_default_comparison_plan()["routing_after_placement"].arms
        self.assertEqual(arms["uniform_round_robin"].choose_primary(4, rows), 1)
        self.assertEqual(arms["join_shortest_queue"].choose_primary(4, rows), 1)
        self.assertEqual(arms["least_unfinished_work"].choose_primary(4, rows), 2)

    def test_lplb_adapter_requires_batch_and_returns_exact_assignment(self):
        arm = build_default_comparison_plan()["routing_after_placement"].arms[
            "lplb_token_count_minmax"
        ]
        with self.assertRaises(BaselineAdapterError):
            arm.choose_primary(0, (ReplicaLoad(0, 0, 0.0),))
        result = arm.assign_batch((0, 0, 1, 1), {0: (0, 1), 1: (1, 2)})
        self.assertEqual(result.maximum_load, 2)

    def test_hedging_adapters_construct_named_kernels(self):
        arms = build_default_comparison_plan()["protection_same_topology"].arms
        self.assertEqual(arms["failover_only"].build_kernel().baseline_id, "failover_only")
        self.assertEqual(
            arms["p95_delayed_hedge"].build_kernel(latency_samples=(1, 2, 3, 4, 100)).baseline_id,
            "p95_delayed_hedge",
        )
        self.assertEqual(
            arms["laedge_work_conserving"].build_kernel(replica_ids=(0, 1)).baseline_id,
            "laedge_work_conserving",
        )


class MetricProjectionTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "latency": {"overall": {"mean": 1.0, "p95": 2.0, "p99": 3.0},
                        "df": {"cvar95": 4.0, "miss_rate": 0.1},
                        "hr": {"p99": 2.5}},
            "replay_rate": 0.02,
            "total_work": 10.0,
            "wasted_work": 0.4,
            "storm_peak": 2.0,
        }

    def test_mapping_and_dataclass_project_identically(self):
        @dataclass
        class Record:
            latency: dict
            replay_rate: float
            total_work: float
            wasted_work: float
            storm_peak: float

        self.assertEqual(adapt_common_metrics(self.row), adapt_common_metrics(Record(**self.row)))

    def test_partial_or_nonfinite_metrics_fail_closed(self):
        partial = dict(self.row)
        partial.pop("storm_peak")
        with self.assertRaises(BaselineAdapterError):
            adapt_common_metrics(partial)
        bad = dict(self.row, replay_rate=float("nan"))
        with self.assertRaises(BaselineAdapterError):
            adapt_common_metrics(bad)


if __name__ == "__main__":
    unittest.main()
