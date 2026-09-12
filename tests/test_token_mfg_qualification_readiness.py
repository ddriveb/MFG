import dataclasses
import unittest

from mfg_hedge.reliability_aware_routing import (
    Topology,
    RoutingToken,
    build_manual_trace,
    generate_routing_trace,
    simulate_routing,
)
from mfg_hedge.population_estimator import run_population_forward
from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
from mfg_hedge.token_population_response import action_buckets_for_token
from mfg_hedge.token_mfg_response_solver import (
    ActionCostEstimate,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    ForwardStateRow,
    MFGCostModel,
    PolicyRow,
    PolicyState,
    confirm_final_policy,
)
from mfg_hedge.token_mfg_qualification import (
    FiniteKGainRow,
    FixedEpisodeLibrary,
    compute_simultaneous_finite_k_gain_bound,
)


FP = "b" * 64


class NormalPolicy:
    def choose(self, observation, context, policy_key):
        return 0

    def predict_token_share(self, observation, context):
        return ((0, 1.0),)


class QualificationReadinessTests(unittest.TestCase):
    def test_topology_is_strict_four_domain_and_qualification_bounded(self):
        self.assertEqual(Topology(8).replica_ids, tuple(range(8)))
        self.assertEqual(Topology(16).replicas_per_domain, 4)
        self.assertEqual(Topology(64).domains.count(0), 16)
        with self.assertRaises(ValueError):
            Topology(12)
        with self.assertRaises(ValueError):
            Topology(16.0)
        with self.assertRaises(ValueError):
            Topology(True)

    def test_scaled_trace_has_real_tokens_faults_draws_and_fixed_release_rate(self):
        left = generate_routing_trace(
            "S0", 0, namespace="topology-red", macro_seed=91, topology=Topology(8),
        )
        right = generate_routing_trace(
            "S0", 0, namespace="topology-red", macro_seed=91, topology=Topology(16),
        )
        self.assertEqual(left.tokens[0].arrival_time, right.tokens[0].arrival_time)
        self.assertGreater(len(right.tokens), len(left.tokens))
        self.assertEqual(len(right.work_draws), len(right.tokens) * 16 * 16)
        self.assertEqual(len(right.replica_ids), 16)
        self.assertEqual(len(right.domains), 16)
        self.assertEqual(right.domains.count(0), 4)

    def test_k8_physics_regresses_and_k16_32_64_are_real_complete_runs(self):
        for k in (8, 16, 32, 64):
            topology = Topology(k)
            trace = build_manual_trace(
                tuple(
                    RoutingToken(i, 0.0, "Regular", 4.0, i % k, 1.0)
                    for i in range(min(k, 16))
                ),
                topology=topology,
            )
            result = simulate_routing(trace, "uniform_rr")
            self.assertTrue(result.complete_drain)
            self.assertEqual(set(result.replica_audit), set(range(k)))
            self.assertEqual(result.invariants["completed_without_winner"], 0)

        k8 = build_manual_trace((RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),))
        self.assertEqual(
            simulate_routing(k8, "uniform_rr"),
            simulate_routing(k8, "uniform_rr"),
        )

    def test_population_and_action_support_follow_trace_topology(self):
        trace = build_manual_trace(
            tuple(RoutingToken(i, 0.0, "Regular", 4.0, i % 16, 0.5) for i in range(4)),
            topology=Topology(16),
        )
        episode = run_population_forward(trace, NormalPolicy)
        self.assertEqual(len(episode.batches[0].nu.replica_buckets), 16)
        buckets = action_buckets_for_token(
            episode.batches[0], episode.batches[0].eta.token_ids[0],
        )
        self.assertTrue(buckets)
        self.assertEqual(len(episode.batches[0].cell_keys[0].replica_state), 16)

    def test_simultaneous_routing_uses_dynamic_replica_state_lengths(self):
        trace = build_manual_trace(
            tuple(RoutingToken(i, 0.0, "Regular", 4.0, i % 32, 0.25) for i in range(8)),
            topology=Topology(32),
        )
        result = simulate_simultaneous_routing(trace, NormalPolicy())
        self.assertEqual(len(result.batch_audits[0].pre_batch_queue_depths), 32)
        self.assertEqual(len(result.batch_audits[0].queue_depths_after_commit), 32)

    def test_final_confirmation_returns_one_policy_environment_q_tuple(self):
        policy = PolicyState((PolicyRow("s0", ("a0", "a1"), (0.5, 0.5)),))

        def snapshot(current, model_id, iteration):
            return ForwardSnapshot(
                model_id, iteration, current.fingerprint, FP, FP, FP, FP, FP, FP,
                (ForwardStateRow(
                    "s0", "Regular", 1.0,
                    (("a0", 0.5), ("a1", 0.5)),
                    (("a0", 0.5), ("a1", 0.5)),
                ),),
                16, True,
            )

        def continuation(current_snapshot, current, model_id, iteration):
            return (
                ActionCostEstimate("s0", "a0", "Regular", 1.0, 0.0, 0.0, 0.0, 0.0, 16, 0.0),
                ActionCostEstimate("s0", "a1", "Regular", 1.0, 0.0, 0.0, 0.0, 0.0, 16, 0.0),
            )

        def deviation(current_snapshot, current, model_id, iteration):
            return FiniteKDeviationDiagnostic(True, 16, 0.0, FP)

        result = confirm_final_policy(
            policy, MFGCostModel.unpriced(), snapshot, continuation, deviation,
            previous_snapshot=snapshot(policy, "unpriced_mfg", 0),
        )
        self.assertEqual(result.status, "confirmed")
        self.assertEqual(result.policy.fingerprint, policy.fingerprint)
        self.assertEqual(result.snapshot.policy_fingerprint, result.policy.fingerprint)
        self.assertEqual(tuple(row.action_id for row in result.q_rows), ("a0", "a1"))
        self.assertEqual(result.population_residual, 0.0)
        self.assertEqual(result.share_residual, 0.0)

    def test_finite_k_gain_is_baseline_minus_deviation_with_simultaneous_upper_bound(self):
        rows = (
            FiniteKGainRow("s0", "a0", 0.5, 0.1, 16),
            FiniteKGainRow("s1", "a1", -0.2, 0.05, 16),
        )
        bound = compute_simultaneous_finite_k_gain_bound(rows, critical_value=2.0)
        self.assertAlmostEqual(bound.max_upper_bound, 0.7)
        self.assertEqual(bound.supported_count, 2)
        self.assertTrue(bound.complete)
        with self.assertRaises(ValueError):
            compute_simultaneous_finite_k_gain_bound((), critical_value=2.0)

    def test_fixed_episode_library_is_canonical_and_exogenous_only(self):
        traces = tuple(
            build_manual_trace(
                (RoutingToken(index, float(index), "Regular", 4.0, index % 8, 1.0),),
                namespace="fixed-library", macro_seed=7, scenario="S0",
                episode_index=index,
            )
            for index in (1, 0)
        )
        library = FixedEpisodeLibrary.from_traces(traces)
        self.assertEqual(
            tuple(trace.episode_index for trace in library.traces), (0, 1),
        )
        self.assertEqual(
            library.fingerprint,
            FixedEpisodeLibrary.from_traces(library.traces).fingerprint,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            library.traces = ()


if __name__ == "__main__":
    unittest.main()
