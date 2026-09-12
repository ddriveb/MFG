import unittest
import tempfile
import inspect
from unittest import mock

from tests.test_token_mfg_qualification_backend import _library
from mfg_hedge.token_mfg_qualification_backend import (
    ConcreteQualificationBackend,
    RoutingPolicySeed,
    materialize_policy,
)
from mfg_hedge.token_mfg_formal_backend import (
    FormalModelRun,
    FormalQualificationBackend,
    FormalQualificationRunner,
)
from mfg_hedge.reliability_aware_routing import RoutingToken, build_manual_trace
from mfg_hedge.token_mfg_qualification import FixedEpisodeLibrary
from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    QualificationTimingReport,
    TimingPreflight,
)
from mfg_hedge.token_mfg_response_solver import (
    ForwardSnapshot,
    ForwardStateRow,
    PolicyRow,
    PolicyState,
)
from mfg_hedge.topology import Topology


def _passing_preflight(plan):
    fingerprint = "a" * 64
    rows = tuple(
        TimingPreflight(
            plan.protocol_fingerprint, plan.preflight_namespace,
            plan.preflight_macro_seed, k, scenario, 1, 1, 0,
            1.0, 1.0, 0.0, 1.0, 155, 1, True, fingerprint, True,
        )
        for k in plan.k_values
        for scenario in ("S0", "S1", "S2", "S3")
    )
    return QualificationTimingReport(
        plan.protocol_fingerprint, rows, 24, 0, 1.0, 100.0, True, True,
        tuple(row for row in rows if row.k == 64) * 2,
        8, 10.0, 5.0, 1.6, 1,
    )


def _scale_library(k):
    traces = tuple(
        build_manual_trace(
            (
                RoutingToken(0, 1.0, "Regular", 4.0, 0, 1.0),
                RoutingToken(1, 1.0, "Urgent", 4.0, 1, 1.0),
            ),
            namespace=f"r2-scale-k{k}", macro_seed=20261000 + k,
            scenario="S0", episode_index=episode, topology=Topology(k),
        )
        for episode in range(32)
    )
    return FixedEpisodeLibrary.from_traces(traces)


def _synthetic_run(k, start_id, model_id, normalized_ucb):
    states = (
        ForwardStateRow(
            f"k{k}-regular", "Regular", 0.5, (("action", 1.0),),
            (("action", 1.0),),
        ),
        ForwardStateRow(
            f"k{k}-urgent", "Urgent", 0.5, (("action", 1.0),),
            (("action", 1.0),),
        ),
    )
    policy = PolicyState(tuple(
        PolicyRow(state.state_id, ("action",), (1.0,)) for state in states
    ))
    fingerprint = "c" * 64
    snapshot = ForwardSnapshot(
        model_id, 1, policy.fingerprint, fingerprint, fingerprint,
        fingerprint, fingerprint, fingerprint, fingerprint,
        states, 32, True, 32,
    )
    return FormalModelRun(
        k, start_id, model_id, "fixed_point", 1, policy.fingerprint,
        "confirmed", 352, 0.0, "complete", 0.0, 0.0,
        normalized_ucb, policy, snapshot,
    )


class QualificationR2CorrectionTests(unittest.TestCase):
    def test_r2_plan_is_fresh_and_does_not_consume_holdout_identity(self):
        r1 = QualificationPlan()
        r2 = QualificationPlan.r2()
        self.assertNotEqual(r2.protocol_fingerprint, r1.protocol_fingerprint)
        self.assertTrue(r2.forward_namespace.startswith(
            "reliability-aware-token-mfg:v1:r2:qualification:"
        ))
        self.assertNotEqual(r2.forward_macro_seed, r1.forward_macro_seed)
        self.assertNotEqual(r2.continuation_macro_seed, r1.continuation_macro_seed)
        self.assertNotEqual(r2.finite_k_macro_seed, r1.finite_k_macro_seed)
        self.assertEqual(r2.holdout_namespace, r1.holdout_namespace)
        self.assertEqual(r2.holdout_macro_seed, r1.holdout_macro_seed)

    def test_each_episode_contributes_one_target_panel(self):
        backend = ConcreteQualificationBackend(
            _library(),
            mean_field_namespace="reliability-aware-token-mfg:v1:r2:dry-run:continuation",
            mean_field_macro_seed=20260926,
            finite_k_namespace="reliability-aware-token-mfg:v1:r2:dry-run:finite-k",
            finite_k_macro_seed=20260927,
        )
        snapshot = backend._forward(None, "unpriced_mfg", 0)
        self.assertEqual(snapshot.sample_count, 4)
        forward, continuation, finite_k = backend._panel_call_counts()
        self.assertEqual(forward, 4)
        self.assertLessEqual(continuation, 4 * 9)
        self.assertLessEqual(finite_k, 4 * 9)

    def test_dry_run_branch_protocol_is_not_the_r1_protocol(self):
        backend = ConcreteQualificationBackend(
            _library(),
            mean_field_namespace="reliability-aware-token-mfg:v1:r2:dry-run:continuation",
            mean_field_macro_seed=20260926,
            finite_k_namespace="reliability-aware-token-mfg:v1:r2:dry-run:finite-k",
            finite_k_macro_seed=20260927,
        )
        self.assertEqual(
            backend.mean_field_namespace,
            "reliability-aware-token-mfg:v1:r2:dry-run:continuation",
        )
        self.assertEqual(backend.mean_field_macro_seed, 20260926)
        self.assertEqual(
            backend.finite_k_namespace,
            "reliability-aware-token-mfg:v1:r2:dry-run:finite-k",
        )
        self.assertEqual(backend.finite_k_macro_seed, 20260927)

    def test_formal_backend_is_k_scale_two_model_and_panel_bounded(self):
        plan = QualificationPlan.r2()
        library = _library(
            32,
            namespace="reliability-aware-token-mfg:v1:r2:backend-fixture",
            macro_seed=20260928,
        )
        backend = FormalQualificationBackend(library, plan=plan)
        self.assertFalse(backend.bounded)
        self.assertEqual(backend.min_complete_panels, 8)
        snapshot = backend._forward(None, "unpriced_mfg", 0)
        counts = backend.validate_panel_counts()
        self.assertEqual(counts[0], 32)
        self.assertLessEqual(counts[1], 288)
        self.assertLessEqual(counts[2], 288)
        for start_id in plan.starts:
            seed = backend.initial_seed(start_id)
            self.assertEqual(seed.kind, start_id)
        phase = backend.phase(
            start_id="uniform",
            model_id="priced_mfg",
            iteration=0,
            stage="forward_population",
            expected_calls=32,
            input_payload={"policy": "uniform"},
        )
        self.assertTrue(phase.key.startswith("k=8/start=uniform/model=priced_mfg"))

    def test_formal_finite_k_boundary_carries_simultaneous_ucb(self):
        plan = QualificationPlan.r2()
        library = _library(
            32,
            namespace="reliability-aware-token-mfg:v1:r2:ucb-fixture",
            macro_seed=20260931,
        )
        backend = FormalQualificationBackend(library, plan=plan)
        snapshot = backend._forward(None, "unpriced_mfg", 0)
        policy = PolicyState.uniform(snapshot.states)
        diagnostic = backend._finite_k(snapshot, policy, "unpriced_mfg", 0)
        self.assertEqual(diagnostic.simultaneous_ucb_status, "complete")
        self.assertIsNotNone(diagnostic.simultaneous_ucb)
        self.assertGreaterEqual(diagnostic.simultaneous_ucb, 0.0)
        self.assertIsNotNone(diagnostic.normalized_simultaneous_ucb)

    def test_formal_runner_executes_both_models_on_nonformal_fixture(self):
        plan = QualificationPlan.r2()
        library = _library(
            32,
            namespace="reliability-aware-token-mfg:v1:r2:runner-fixture",
            macro_seed=20260930,
        )
        for model_id in ("unpriced_mfg", "priced_mfg"):
            backend = FormalQualificationBackend(library, plan=plan)
            runner = FormalQualificationRunner(
                plan=plan,
                preflight=_passing_preflight(plan),
                checkpoint_root=tempfile.mkdtemp(),
                run_id=f"fixture-{model_id}",
                worker_count=1,
            )
            result = runner.run_model(
                backend,
                start_id="uniform",
                model_id=model_id,
                initial_policy=backend.initial_seed("uniform"),
            )
            self.assertEqual(result.model_id, model_id)
            self.assertEqual(result.iterations, plan.max_iterations)
            self.assertEqual(result.confirmation_status, "not_converged")
            self.assertGreater(result.scheduler_calls, 0)

    def test_formal_runner_requires_preflight_and_has_no_caller_policy_graph(self):
        plan = QualificationPlan.r2()
        parameters = inspect.signature(FormalQualificationRunner).parameters
        self.assertIn("preflight", parameters)
        run_parameters = inspect.signature(
            FormalQualificationRunner.run_qualification
        ).parameters
        self.assertNotIn("initial_policies", run_parameters)
        with self.assertRaises(TypeError):
            FormalQualificationRunner(
                plan=plan, checkpoint_root=tempfile.mkdtemp(),
                run_id="missing-preflight", worker_count=1,
            )

    def test_internal_start_and_scale_projection_are_checkpointable(self):
        plan = QualificationPlan.r2()
        library = _library(
            32, namespace="r2-projection-fixture", macro_seed=20260932,
        )
        backend = FormalQualificationBackend(library, plan=plan)
        start = backend.initial_seed("simultaneous_loew_projection")
        self.assertIsInstance(start, RoutingPolicySeed)
        snapshot = backend._forward(start, "unpriced_mfg", 0)
        policy = materialize_policy(snapshot)
        scale = backend.scale_seed(snapshot, policy)
        self.assertEqual(scale.kind, "scale_marginal_projection")
        self.assertEqual(
            backend.policy_distance(snapshot, policy, snapshot, policy), 0.0,
        )

    def test_global_call_ceiling_is_checked_before_reservation(self):
        plan = QualificationPlan.r2()
        library = _library(
            32, namespace="r2-budget-fixture", macro_seed=20260933,
        )
        backend = FormalQualificationBackend(library, plan=plan)
        with tempfile.TemporaryDirectory() as root:
            runner = FormalQualificationRunner(
                plan=plan, preflight=_passing_preflight(plan),
                checkpoint_root=root, run_id="budget-fixture", worker_count=1,
            )
            runner.checkpoint.reserve(
                "prior-budget", input_fingerprint="b" * 64,
                expected_calls=plan.qualification_calls,
            )
            with self.assertRaisesRegex(ValueError, "call ceiling"):
                runner._phase(
                    backend, start_id="uniform", model_id="unpriced_mfg",
                    iteration=0, stage="forward_population", expected_calls=1,
                    payload={"stage": "forward_population"},
                )

    def test_scale_handoff_is_internal_and_all_final_gates_run(self):
        plan = QualificationPlan.r2()
        libraries = {k: _scale_library(k) for k in plan.k_values}
        with tempfile.TemporaryDirectory() as root:
            runner = FormalQualificationRunner(
                plan=plan, preflight=_passing_preflight(plan),
                checkpoint_root=root, run_id="scale-flow", worker_count=1,
            )
            seen = []

            def result_for(backend, *, start_id, model_id, initial_policy):
                k = backend.library.traces[0].topology.K
                seen.append((k, start_id, model_id, initial_policy))
                ucb = {8: 0.018, 16: 0.016, 32: 0.014, 64: 0.012}[k]
                return _synthetic_run(k, start_id, model_id, ucb)

            with mock.patch.object(runner, "run_model", side_effect=result_for):
                status, rows = runner.run_qualification(libraries=libraries)
        self.assertEqual(status, "qualification_pass")
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(
            isinstance(seed, RoutingPolicySeed)
            for _, _, _, seed in seen
        ))
        self.assertEqual(
            tuple(k for k, start, model, _ in seen if model == "unpriced_mfg"),
            (8, 8, 8, 16, 32, 64),
        )


if __name__ == "__main__":
    unittest.main()
