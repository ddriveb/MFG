import dataclasses
import tempfile
import unittest

from mfg_hedge.qualification_checkpoint import (
    QualificationCheckpointError,
    QualificationCheckpointStore,
)
from mfg_hedge.reliability_aware_routing import RoutingToken, build_manual_trace
from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
from mfg_hedge.topology import Topology
from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationCampaignError,
    QualificationPlan,
    QualificationTimingReport,
    TimingPreflight,
    qualification_source_fingerprint,
)
from mfg_hedge.token_mfg_response_solver import (
    ActionCostEstimate,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    ForwardStateRow,
    MFGCostModel,
    MFGSolverConfig,
    PolicyRow,
    PolicyState,
    confirm_final_policy,
    solve_unpriced_priced_mfg,
)


FP_A = "a" * 64
FP_B = "b" * 64


def _policy():
    return PolicyState((PolicyRow("s0", ("a0", "a1"), (0.5, 0.5)),))


def _snapshot(policy, iteration, calls=32):
    return ForwardSnapshot(
        "unpriced_mfg", iteration, policy.fingerprint, FP_A, FP_A,
        FP_A, FP_A, FP_A, FP_A,
        (ForwardStateRow(
            "s0", "Regular", 1.0,
            (("a0", 0.5), ("a1", 0.5)),
            (("a0", 0.5), ("a1", 0.5)),
        ),),
        32, True, calls,
    )


def _q_rows():
    return (
        ActionCostEstimate("s0", "a0", "Regular", 1.0, 0, 0, 0, 0, 32, 0),
        ActionCostEstimate("s0", "a1", "Regular", 1.0, 0, 0, 0, 0, 32, 0),
    )


class QualificationSafetyTests(unittest.TestCase):
    def test_source_bundle_fingerprint_is_stable_and_complete(self):
        first = qualification_source_fingerprint()
        self.assertEqual(first, qualification_source_fingerprint())
        self.assertEqual(len(first), 64)

    def test_trace_hot_lookup_preserves_canonical_and_legacy_order(self):
        trace = build_manual_trace((RoutingToken(7, 0.0, "Regular", 4.0, 0, 1.25),))
        self.assertEqual(trace.work_for(7, 3, 2), 1.25)
        legacy = dataclasses.replace(trace, work_draws=tuple(reversed(trace.work_draws)))
        self.assertEqual(legacy.work_for(7, 3, 2), 1.25)

    def test_optimized_lookup_is_field_exact_at_every_qualification_k(self):
        class Policy:
            def choose(self, observation, context, policy_key):
                available = tuple(
                    row.replica_id
                    for row in observation.public_state.replica_observations
                    if row.available
                )
                return available[observation.token_id % len(available)]

            def predict_token_share(self, observation, context):
                return ((0, 1.0),)

        for k in (8, 16, 32, 64):
            trace = build_manual_trace(
                tuple(
                    RoutingToken(i, 0.0, "Regular", 4.0, i % k, 1.0)
                    for i in range(4)
                ),
                topology=Topology(k),
            )
            reference = dataclasses.replace(
                trace, work_draws=tuple(reversed(trace.work_draws)),
            )
            self.assertEqual(
                simulate_simultaneous_routing(trace, Policy()),
                simulate_simultaneous_routing(reference, Policy()),
            )

    def test_formal_call_contract_accepts_32_episode_independent_libraries(self):
        policy = _policy()
        config = MFGSolverConfig(
            max_iterations=1, min_complete_samples=8, call_budget=608,
            forward_calls=32, continuation_calls=288, finite_k_calls=288,
        )

        def forward(current, model_id, iteration):
            return _snapshot(current, iteration)

        def continuation(snapshot, current, model_id, iteration):
            return _q_rows()

        def deviation(snapshot, current, model_id, iteration):
            return FiniteKDeviationDiagnostic(True, 32, 0.0, FP_B, 288)

        result = solve_unpriced_priced_mfg(
            forward, continuation, deviation, initial_policy=policy,
            config=config, models=(MFGCostModel.unpriced(),),
        )
        self.assertNotEqual(result.model_results[0].status, "physical_failed")
        self.assertEqual(result.attempted_calls, 608)

        previous = _snapshot(policy, 0)
        confirmed = confirm_final_policy(
            policy, MFGCostModel.unpriced(), forward, continuation, deviation,
            previous_snapshot=previous, config=config,
        )
        self.assertEqual(confirmed.attempted_calls, 608)
        self.assertNotEqual(confirmed.status, "physical_failed")

    def test_start_gate_requires_all_k_scenarios_and_enforces_time(self):
        plan = QualificationPlan()
        self.assertEqual(
            tuple(plan.scenario_schedule.count(name) for name in ("S0", "S1", "S2", "S3")),
            (8, 8, 8, 8),
        )
        rows = tuple(
            TimingPreflight(
                plan.protocol_fingerprint, plan.preflight_namespace,
                plan.preflight_macro_seed, k, scenario, 1, 1, 0,
                1.0, 1.0, 0.0, 1.0, 1, 1, True, FP_A, True,
            )
            for k in plan.k_values
            for scenario in ("S0", "S1", "S2", "S3")
        )
        report = QualificationTimingReport(
            plan.protocol_fingerprint, rows, 24, 0, 1.0, 100.0, True, True,
            tuple(row for row in rows if row.k == 64) * 2,
            8, 10.0, 5.0, 1.6, 1,
        )
        plan.require_formal_start(report)
        with self.assertRaises(QualificationCampaignError):
            plan.require_formal_start(dataclasses.replace(report, worst_k64_call_seconds=31.0))
        with self.assertRaises(QualificationCampaignError):
            plan.require_formal_start(dataclasses.replace(report, projected_eight_worker_seconds=43_201.0))

    def test_checkpoint_reuses_committed_work_and_rejects_unresolved_dispatch(self):
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "run-1", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            self.assertTrue(store.reserve(
                "K8/model/iteration0/forward/episode0",
                input_fingerprint=FP_A, expected_calls=1,
            ))
            store.commit(
                "K8/model/iteration0/forward/episode0",
                input_fingerprint=FP_A, completed_calls=1,
                output={"cost": 2.0},
            )
            resumed = QualificationCheckpointStore(
                root, "run-1", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            self.assertFalse(resumed.reserve(
                "K8/model/iteration0/forward/episode0",
                input_fingerprint=FP_A, expected_calls=1,
            ))
            self.assertEqual(
                resumed.completed_output(
                    "K8/model/iteration0/forward/episode0",
                    input_fingerprint=FP_A,
                )["cost"],
                2.0,
            )
            self.assertTrue(resumed.reserve(
                "K8/model/iteration0/forward/episode1",
                input_fingerprint=FP_B, expected_calls=1,
            ))
            self.assertFalse(resumed.pause_safe)
            with self.assertRaises(QualificationCheckpointError):
                QualificationCheckpointStore(
                    root, "run-1", plan_fingerprint="c" * 64,
                    source_fingerprint=FP_B,
                )

    def test_call_ledger_cannot_settle_one_reservation_twice(self):
        from mfg_hedge.token_mfg_response_solver import CallLedger, MFGSolverError

        ledger = CallLedger(2)
        reservation = ledger.reserve(1)
        ledger.settle(reservation, completed=1)
        with self.assertRaises(MFGSolverError):
            ledger.settle(reservation, completed=1)


if __name__ == "__main__":
    unittest.main()
