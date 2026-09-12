import unittest
from itertools import product

from mfg_hedge.replica_routing_baselines import (
    BASELINE_CATALOG,
    EPLBPlan,
    FailoverOnly,
    JoinShortestQueueRouter,
    LAEdgeScheduler,
    LeastUnfinishedWorkRouter,
    P95DelayedHedge,
    ReplicaLoad,
    RoundRobinRouter,
    lplb_token_count_minmax,
    rebalance_experts_eplb,
    replicate_experts_eplb,
)


class ProvenanceTests(unittest.TestCase):
    def test_catalog_has_stable_layer_and_source_metadata(self):
        expected = {
            "failover_only", "uniform_round_robin", "join_shortest_queue",
            "least_unfinished_work", "eplb_style", "lplb_token_count_minmax",
            "p95_delayed_hedge", "laedge_work_conserving",
        }
        self.assertEqual(set(BASELINE_CATALOG), expected)
        for key, row in BASELINE_CATALOG.items():
            self.assertEqual(row.baseline_id, key)
            self.assertIn(row.semantic_class, {"placement", "routing", "hedging"})
            self.assertTrue(row.source_url.startswith("https://"))
            self.assertTrue(row.source_revision)
            self.assertIn(row.adaptation_kind, {"native", "independent_stdlib_adaptation"})


class EPLBTests(unittest.TestCase):
    def test_replicates_current_heaviest_per_replica_load_with_stable_ties(self):
        phy2log, ranks, counts = replicate_experts_eplb((9.0, 4.0, 4.0), 6)
        self.assertEqual(phy2log, (0, 1, 2, 0, 0, 1))
        self.assertEqual(ranks, (0, 0, 0, 1, 2, 1))
        self.assertEqual(counts, (3, 2, 1))

    def test_readme_hierarchical_example_matches_upstream_output(self):
        loads = (
            (90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86),
            (20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27),
        )
        plan = rebalance_experts_eplb(
            loads, num_physical_experts=16, num_groups=4,
            num_nodes=2, num_gpus=8,
        )
        self.assertIsInstance(plan, EPLBPlan)
        self.assertEqual(plan.physical_to_logical, (
            (5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1),
            (7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1),
        ))

    def test_invalid_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            rebalance_experts_eplb(((1.0, 2.0),), 3, 1, 1, 2)


class OnlineRoutingTests(unittest.TestCase):
    def setUp(self):
        self.loads = (
            ReplicaLoad(2, queue_depth=1, unfinished_work=5.0),
            ReplicaLoad(4, queue_depth=0, unfinished_work=8.0),
            ReplicaLoad(7, queue_depth=1, unfinished_work=2.0),
        )

    def test_round_robin_is_counter_free_and_stable(self):
        router = RoundRobinRouter()
        self.assertEqual(
            [router.choose(token_id=i, replicas=self.loads) for i in range(5)],
            [2, 4, 7, 2, 4],
        )

    def test_jsq_and_least_work_use_different_observable_loads(self):
        self.assertEqual(JoinShortestQueueRouter().choose(0, self.loads), 4)
        self.assertEqual(LeastUnfinishedWorkRouter().choose(0, self.loads), 7)

    def test_unavailable_replicas_are_never_selected(self):
        rows = (
            ReplicaLoad(0, 0, 0.0, available=False),
            ReplicaLoad(1, 5, 9.0, available=True),
        )
        for router in (RoundRobinRouter(), JoinShortestQueueRouter(), LeastUnfinishedWorkRouter()):
            self.assertEqual(router.choose(0, rows), 1)


class LPLBTests(unittest.TestCase):
    def test_integral_minmax_assignment_with_overlapping_eligibility(self):
        result = lplb_token_count_minmax(
            token_experts=(0, 0, 0, 1, 1, 1),
            eligible_replicas={0: (0, 1), 1: (1, 2)},
        )
        self.assertEqual(result.maximum_load, 2)
        self.assertEqual(result.final_loads, ((0, 2), (1, 2), (2, 2)))
        for expert, replica in zip((0, 0, 0, 1, 1, 1), result.token_to_replica):
            self.assertIn(replica, {0: (0, 1), 1: (1, 2)}[expert])

    def test_initial_load_and_ties_are_deterministic(self):
        kwargs = dict(
            token_experts=(0, 0, 0),
            eligible_replicas={0: (3, 1)},
            initial_loads={1: 1, 3: 0},
        )
        left = lplb_token_count_minmax(**kwargs)
        right = lplb_token_count_minmax(**kwargs)
        self.assertEqual(left, right)
        self.assertEqual(left.maximum_load, 2)
        self.assertEqual(left.final_loads, ((1, 2), (3, 2)))

    def test_missing_eligible_replica_fails_closed(self):
        with self.assertRaises(ValueError):
            lplb_token_count_minmax((0, 1), {0: (0,)})

    def test_small_instances_match_brute_force_optimum(self):
        eligibility = {0: (0, 1), 1: (1, 2)}
        for tokens in ((0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1, 0)):
            result = lplb_token_count_minmax(tokens, eligibility)
            feasible = product(*(eligibility[expert] for expert in tokens))
            optimum = min(
                max(tuple(candidate).count(replica) for replica in (0, 1, 2))
                for candidate in feasible
            )
            self.assertEqual(result.maximum_load, optimum)


class HedgingKernelTests(unittest.TestCase):
    def test_failover_only_never_speculates_and_replays_after_failure(self):
        policy = FailoverOnly()
        self.assertFalse(policy.launch_speculative(primary_complete=False))
        self.assertTrue(policy.launch_replay(primary_failed=True, token_complete=False))
        self.assertFalse(policy.launch_replay(primary_failed=True, token_complete=True))

    def test_p95_threshold_is_frozen_and_timer_requires_incomplete_primary(self):
        policy = P95DelayedHedge.from_latency_samples((1.0, 2.0, 3.0, 4.0, 100.0))
        self.assertAlmostEqual(policy.delay, 80.8)
        self.assertAlmostEqual(policy.timer_at(7.0), 87.8)
        self.assertTrue(policy.launch_at_timer(primary_complete=False, backup_available=True))
        self.assertFalse(policy.launch_at_timer(primary_complete=True, backup_available=True))
        self.assertFalse(policy.launch_at_timer(primary_complete=False, backup_available=False))

    def test_laedge_prioritizes_unserved_then_hedges_oldest_running(self):
        scheduler = LAEdgeScheduler((0, 1, 2))
        first = scheduler.arrive(10)
        self.assertEqual([(x.request_id, x.replica_id, x.kind) for x in first], [
            (10, 0, "primary"), (10, 1, "hedge")
        ])
        self.assertEqual([(x.request_id, x.replica_id) for x in scheduler.arrive(11)], [(11, 2)])
        self.assertEqual(scheduler.arrive(12), ())

        launches = scheduler.complete(10, 0)
        self.assertEqual([(x.request_id, x.replica_id, x.kind) for x in launches], [
            (12, 0, "primary"), (11, 1, "hedge")
        ])
        self.assertEqual(scheduler.unserved_requests, ())
        self.assertEqual(scheduler.idle_replicas, ())

    def test_laedge_second_idle_replica_hedges_after_unserved_work_starts(self):
        scheduler = LAEdgeScheduler((0, 1))
        scheduler.arrive(0)
        scheduler.arrive(1)
        launches = scheduler.complete(0, 0)
        self.assertEqual(
            [(row.request_id, row.replica_id, row.kind) for row in launches],
            [(1, 0, "primary"), (1, 1, "hedge")],
        )


if __name__ == "__main__":
    unittest.main()
