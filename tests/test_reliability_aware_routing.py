import json
import tempfile
import unittest
import math
from dataclasses import replace
from pathlib import Path

import mfg_hedge.reliability_aware_routing as routing
from mfg_hedge.reliability_aware_routing import (
    POLICY_NAMES,
    SCENARIO_NAMES,
    SMOKE_CALL_BUDGET,
    SMOKE_MACRO_SEED,
    SMOKE_NAMESPACE,
    HealthEvent,
    RoutingToken,
    build_manual_trace,
    estimate_hazard,
    generate_routing_trace,
    run_routing_smoke,
    simulate_routing,
)


class ReliabilityAwareRoutingTests(unittest.TestCase):
    def test_protocol_and_topology_are_frozen(self):
        trace = generate_routing_trace("S0", 0)
        self.assertEqual(trace.replica_ids, tuple(range(8)))
        self.assertEqual(trace.domains, (0, 1, 2, 3, 0, 1, 2, 3))
        self.assertEqual(trace.namespace, SMOKE_NAMESPACE)
        self.assertEqual(trace.macro_seed, SMOKE_MACRO_SEED)
        self.assertEqual(len(SCENARIO_NAMES), 4)
        self.assertEqual(len(POLICY_NAMES), 6)
        self.assertEqual(SMOKE_CALL_BUDGET, 192)

    def test_crn_trace_is_identical_across_policies_and_reproducible(self):
        left = generate_routing_trace("S1", 3)
        right = generate_routing_trace("S1", 3)
        self.assertEqual(left, right)
        results = [simulate_routing(left, policy) for policy in POLICY_NAMES]
        self.assertEqual({row.trace_fingerprint for row in results}, {left.fingerprint})
        self.assertEqual({row.token_fingerprint for row in results}, {left.token_fingerprint})

    def test_failure_is_processed_before_same_time_completion(self):
        trace = build_manual_trace(
            tokens=(RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),),
            events=(HealthEvent(1.0, "failure", "replica", 0),),
        )
        result = simulate_routing(trace, "uniform_rr")
        token = result.token_results[0]
        self.assertEqual(token.replay_count, 1)
        self.assertEqual(token.attempts[0].terminal_status, "failed")
        self.assertEqual(token.attempts[0].lost_work, 1.0)
        self.assertEqual(token.winner_replica_id, 1)

    def test_queued_work_is_displaced_without_replay_or_executed_work(self):
        tokens = tuple(
            RoutingToken(i, 0.0, "Regular", 4.0, i % 4, 1.0) for i in range(9)
        )
        trace = build_manual_trace(
            tokens=tokens,
            events=(HealthEvent(0.5, "failure", "replica", 0),),
        )
        result = simulate_routing(trace, "uniform_rr")
        queued = result.token_results[8]
        self.assertEqual(queued.replay_count, 0)
        self.assertEqual(queued.attempts[0].executed_work, 0.0)
        self.assertEqual(queued.attempts[0].terminal_status, "discarded")
        self.assertEqual(result.invariants["max_live_attempts_per_token"], 1)

    def test_repeated_failures_allow_reactive_replay_but_never_two_live(self):
        trace = build_manual_trace(
            tokens=(RoutingToken(0, 0.0, "Urgent", 2.0, 0, 1.0),),
            events=(
                HealthEvent(1.0, "failure", "replica", 0),
                HealthEvent(2.0, "failure", "replica", 1),
            ),
        )
        result = simulate_routing(trace, "jsq")
        self.assertGreaterEqual(result.token_results[0].replay_count, 1)
        self.assertLessEqual(result.invariants["max_live_attempts_per_token"], 1)
        self.assertTrue(result.complete_drain)

    def test_down_replicas_are_never_started_and_global_wait_is_allowed(self):
        trace = build_manual_trace(
            tokens=(RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),),
            events=tuple(
                [HealthEvent(0.0, "failure", "replica", rid) for rid in range(8)]
                + [HealthEvent(1.0, "recovery", "replica", 3)]
            ),
        )
        result = simulate_routing(trace, "reliability_only")
        self.assertEqual(result.token_results[0].attempts[0].start_time, 1.0)
        self.assertTrue(all(start.replica_id == 3 for start in result.starts))

    def test_gamma_poisson_estimator_uses_only_history_prefix(self):
        events = (
            HealthEvent(-10.0, "failure", "replica", 0),
            HealthEvent(-5.0, "recovery", "replica", 0),
            HealthEvent(5.0, "failure", "replica", 0),
            HealthEvent(50.0, "failure", "replica", 0),
        )
        before = estimate_hazard(events, "replica", 0, 0.0)
        after = estimate_hazard(events, "replica", 0, 10.0)
        self.assertGreater(after, before)
        self.assertEqual(estimate_hazard(events, "replica", 0, 0.0), before)
        with self.assertRaises(ValueError):
            estimate_hazard(events, "replica", 0, float("nan"))

    def test_risk_aware_router_exposes_new_token_service_horizon(self):
        trace = build_manual_trace(
            tokens=(RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),),
            events=(
                HealthEvent(-40.0, "failure", "replica", 0),
                HealthEvent(-30.0, "recovery", "replica", 0),
            ),
        )
        result = simulate_routing(trace, "risk_aware_jsq")
        self.assertEqual(result.starts[0].replica_id, 1)
        observations = routing._risk_observations(
            trace, 0.0, {replica_id: True for replica_id in range(8)},
            {replica_id: [] for replica_id in range(8)}, {}, {},
        )
        for row in observations[:2]:
            self.assertAlmostEqual(
                row.estimated_failure_risk,
                1.0 - math.exp(-row.estimated_hazard),
                places=15,
            )

    def test_fixed_baselines_match_pre_correction_reference(self):
        fixed_baselines = (
            "uniform_rr", "jsq", "loew", "reliability_only",
            "lazarus_algorithm1",
        )
        corrected_observations = routing._risk_observations

        def old_observations(*args, **kwargs):
            rows = corrected_observations(*args, **kwargs)
            return tuple(replace(
                row,
                estimated_failure_risk=1.0 - math.exp(
                    -row.estimated_hazard * max(row.estimated_work, 1e-12)
                ),
            ) for row in rows)

        routing._risk_observations = old_observations
        try:
            reference = {
                (scenario, episode, policy): simulate_routing(
                    generate_routing_trace(scenario, episode), policy
                )
                for scenario in SCENARIO_NAMES
                for episode in range(8)
                for policy in fixed_baselines
            }
        finally:
            routing._risk_observations = corrected_observations
        optimized = {
            (scenario, episode, policy): simulate_routing(
                generate_routing_trace(scenario, episode), policy
            )
            for scenario in SCENARIO_NAMES
            for episode in range(8)
            for policy in fixed_baselines
        }
        self.assertEqual(reference, optimized)

    def test_all_policies_complete_without_hidden_future_service_observation(self):
        trace = generate_routing_trace("S3", 1)
        for policy in POLICY_NAMES:
            result = simulate_routing(trace, policy)
            self.assertTrue(result.complete_drain, policy)
            self.assertEqual(result.invariants["future_work_observations"], 0)
            self.assertEqual(result.invariants["down_starts"], 0)
            self.assertEqual(result.invariants["max_live_attempts_per_token"], 1)
            self.assertIn("urgent_cvar95", result.metrics)
            self.assertIn("regular_deadline_miss_rate", result.metrics)
            self.assertEqual(set(result.replica_audit), set(range(8)))
            self.assertTrue(all("utilization_if_available" in row for row in result.replica_audit.values()))

    def test_domain_failure_invalidates_both_replicas(self):
        trace = build_manual_trace(
            tokens=tuple(RoutingToken(i, 0.0, "Regular", 4.0, i, 1.0) for i in range(8)),
            events=(HealthEvent(0.5, "failure", "domain", 0),),
        )
        result = simulate_routing(trace, "jsq")
        self.assertGreaterEqual(result.metrics["replay_count"], 1.0)
        self.assertEqual(result.invariants["down_starts"], 0)

    def test_smoke_has_exact_budget_and_transactional_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = run_routing_smoke(Path(tmp))
            self.assertEqual(artifact["calls"], 192)
            self.assertEqual(artifact["status"], "mechanism_smoke_complete")
            run_dir = Path(artifact["artifact_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "episode_rows.jsonl").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "protocol.json").exists())
            with (run_dir / "summary.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["calls"], 192)


if __name__ == "__main__":
    unittest.main()
