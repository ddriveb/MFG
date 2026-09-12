import dataclasses
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from mfg_hedge.reliability_aware_routing import RoutingTrace
from mfg_hedge.token_mfg_qualification_campaign import (
    HOLDOUT_CALLS,
    PRE_FLIGHT_MAX_CALLS,
    QUALIFICATION_CALLS,
    TOTAL_CALL_LIMIT,
    QualificationCampaignError,
    QualificationPlan,
    build_qualification_trace_library,
    run_timing_preflight,
)


FP_A = "a" * 64
FP_B = "b" * 64


def _process_order_probe_worker(payload):
    """Top-level spawn-safe worker used by deterministic executor tests."""
    time.sleep(float(payload.get("delay", 0.0)))
    return {
        "completed_calls": int(payload["completed_calls"]),
        "output": {"key": payload["key"], "value": payload["value"]},
    }


def _failing_probe_worker(payload):
    raise RuntimeError(str(payload.get("message", "probe failure")))


def _real_physics_probe_worker(payload):
    from mfg_hedge.simultaneous_token_routing import simulate_simultaneous_routing
    from mfg_hedge.token_mfg_qualification_campaign import _UniformRealPolicy

    trace = payload["trace"]
    result = simulate_simultaneous_routing(
        trace, _UniformRealPolicy(), policy_name="bounded_executor_probe",
    )
    if not result.physical.complete_drain:
        raise RuntimeError("bounded real probe did not drain")
    return {
        "completed_calls": 1,
        "output": {
            "trace_fingerprint": result.physical.trace_fingerprint,
            "complete_drain": result.physical.complete_drain,
        },
    }


def _probe_fp(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class TokenMFGQualificationCampaignTests(unittest.TestCase):
    def test_frozen_plan_has_exact_scale_and_budget_contract(self):
        plan = QualificationPlan()
        self.assertEqual(plan.qualification_calls, QUALIFICATION_CALLS)
        self.assertEqual(plan.holdout_calls, HOLDOUT_CALLS)
        self.assertEqual(plan.total_call_limit, TOTAL_CALL_LIMIT)
        self.assertEqual(plan.total_call_limit, 59168)
        self.assertEqual(plan.preflight_max_calls, PRE_FLIGHT_MAX_CALLS)
        self.assertEqual(plan.k_values, (8, 16, 32, 64))
        self.assertEqual(
            plan.starts,
            ("uniform", "simultaneous_loew_projection",
             "corrected_risk_aware_projection"),
        )
        self.assertEqual(plan.models, ("unpriced_mfg", "priced_mfg"))

    def test_real_trace_library_is_not_a_synthetic_provider(self):
        plan = QualificationPlan()
        library = build_qualification_trace_library(plan, k=8, count=1)
        self.assertEqual(len(library.traces), 1)
        self.assertIsInstance(library.traces[0], RoutingTrace)
        self.assertEqual(library.traces[0].namespace, plan.forward_namespace)
        self.assertEqual(library.traces[0].macro_seed, plan.forward_macro_seed)
        self.assertEqual(library.traces[0].topology.K, 8)
        self.assertTrue(library.fingerprint)

    def test_preflight_is_real_and_cannot_consume_formal_identity_budget(self):
        plan = QualificationPlan()
        result = run_timing_preflight(plan, k=8, episodes=1)
        self.assertLessEqual(result.scheduler_calls, plan.preflight_max_calls)
        self.assertNotEqual(result.namespace, plan.forward_namespace)
        self.assertNotEqual(result.macro_seed, plan.forward_macro_seed)
        self.assertEqual(result.formal_calls_consumed, 0)
        self.assertTrue(result.trace_fingerprint)
        self.assertGreaterEqual(result.wall_seconds, 0.0)

    def test_plan_rejects_protocol_mutation_and_oversized_preflight(self):
        plan = QualificationPlan()
        with self.assertRaises(ValueError):
            dataclasses.replace(plan, max_iterations=7)
        with self.assertRaises(ValueError):
            run_timing_preflight(plan, k=8, episodes=33)

    def test_formal_runner_must_be_gated_by_preflight_and_real_backend(self):
        plan = QualificationPlan()
        with self.assertRaises(QualificationCampaignError):
            plan.require_formal_start(None)

    def test_canonical_executor_is_spawn_safe_and_merge_order_is_not_completion_order(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationWorkUnit,
            execute_canonical_work_units,
        )

        units = tuple(
            QualificationWorkUnit(
                key=f"probe/{index}", kind="forward_population",
                input_fingerprint=_probe_fp({"index": index}),
                expected_calls=1,
                payload={
                    "key": f"probe/{index}", "value": index,
                    "completed_calls": 1, "delay": (3 - index) * 0.01,
                },
            )
            for index in range(4)
        )
        with self.subTest("serial and eight worker bytes match"):
            with self._checkpoint("serial") as serial_store, self._checkpoint("parallel") as parallel_store:
                serial = execute_canonical_work_units(
                    units, checkpoint=serial_store,
                    worker=_process_order_probe_worker, max_workers=1,
                )
                parallel = execute_canonical_work_units(
                    units, checkpoint=parallel_store,
                    worker=_process_order_probe_worker, max_workers=8,
                )
                self.assertEqual(serial.to_bytes(), parallel.to_bytes())
                self.assertEqual(
                    tuple(row[0] for row in serial.outputs),
                    tuple(f"probe/{index}" for index in range(4)),
                )

    def test_checkpoint_resume_reuses_committed_output_without_scheduler_calls(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationWorkUnit,
            execute_canonical_work_units,
        )

        unit = QualificationWorkUnit(
            key="probe/committed", kind="forward_population",
            input_fingerprint=_probe_fp("committed"), expected_calls=3,
            payload={"key": "probe/committed", "value": 7, "completed_calls": 3},
        )
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "resume", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            first = execute_canonical_work_units(
                (unit,), checkpoint=store,
                worker=_process_order_probe_worker, max_workers=1,
            )
            resumed_store = QualificationCheckpointStore(
                root, "resume", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            resumed = execute_canonical_work_units(
                (unit,), checkpoint=resumed_store,
                worker=_failing_probe_worker, max_workers=1,
            )
            self.assertEqual(first.to_bytes(), resumed.to_bytes())
            self.assertEqual(resumed.dispatched_calls, 0)
            self.assertEqual(resumed.reused_calls, 1)

    def test_unknown_reserved_unit_fails_closed_and_is_never_retried(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationCampaignError,
            QualificationWorkUnit,
            execute_canonical_work_units,
        )

        unit = QualificationWorkUnit(
            key="probe/unknown", kind="finite_k_deviation",
            input_fingerprint=_probe_fp("unknown"), expected_calls=1,
            payload={"key": "probe/unknown", "value": 0, "completed_calls": 1},
        )
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "unknown", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            self.assertTrue(store.reserve(
                unit.key, input_fingerprint=unit.input_fingerprint,
                expected_calls=unit.expected_calls,
            ))
            with self.assertRaises(QualificationCampaignError):
                execute_canonical_work_units(
                    (unit,), checkpoint=store,
                    worker=_process_order_probe_worker, max_workers=1,
                )

    def test_worker_failure_commits_nothing_and_does_not_retry(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationCampaignError,
            QualificationWorkUnit,
            execute_canonical_work_units,
        )

        unit = QualificationWorkUnit(
            key="probe/failure", kind="final_policy_confirmation",
            input_fingerprint=_probe_fp("failure"), expected_calls=1,
            payload={"message": "injected"},
        )
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "failure", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            with self.assertRaises(QualificationCampaignError):
                execute_canonical_work_units(
                    (unit,), checkpoint=store,
                    worker=_failing_probe_worker, max_workers=1,
                )
            with self.assertRaises(QualificationCampaignError):
                execute_canonical_work_units(
                    (unit,), checkpoint=store,
                    worker=_process_order_probe_worker, max_workers=1,
                )

    def test_input_fingerprint_mutation_and_duplicate_checkpoint_are_rejected(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationWorkUnit,
            execute_canonical_work_units,
        )

        first = QualificationWorkUnit(
            key="probe/fingerprint", kind="forward_population",
            input_fingerprint=_probe_fp("first"), expected_calls=1,
            payload={"key": "probe/fingerprint", "value": 1, "completed_calls": 1},
        )
        changed = dataclasses.replace(
            first,
            input_fingerprint=_probe_fp("changed"),
            payload={"key": "probe/fingerprint", "value": 2, "completed_calls": 1},
        )
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "fingerprint", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            execute_canonical_work_units(
                (first,), checkpoint=store,
                worker=_process_order_probe_worker, max_workers=1,
            )
            with self.assertRaises(QualificationCampaignError):
                execute_canonical_work_units(
                    (changed,), checkpoint=store,
                    worker=_process_order_probe_worker, max_workers=1,
                )

    def test_executor_bounded_fixture_runs_real_routing_not_a_synthetic_provider(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        from mfg_hedge.token_mfg_qualification_campaign import (
            QualificationWorkUnit,
            build_qualification_trace_library,
            execute_canonical_work_units,
        )

        trace = build_qualification_trace_library(
            QualificationPlan(), k=8, count=1,
            namespace="bounded-executor-probe", macro_seed=20260911,
        ).traces[0]
        unit = QualificationWorkUnit(
            key="probe/real-routing", kind="forward_population",
            input_fingerprint=_probe_fp(trace.fingerprint), expected_calls=1,
            payload={"trace": trace},
        )
        with tempfile.TemporaryDirectory() as root:
            store = QualificationCheckpointStore(
                root, "real-routing", plan_fingerprint=FP_A, source_fingerprint=FP_B,
            )
            result = execute_canonical_work_units(
                (unit,), checkpoint=store,
                worker=_real_physics_probe_worker, max_workers=2,
            )
        self.assertEqual(result.outputs[0][1]["trace_fingerprint"], trace.fingerprint)
        self.assertTrue(result.outputs[0][1]["complete_drain"])

    def _checkpoint(self, run_id):
        return _CheckpointContext(run_id)


class _CheckpointContext:
    def __init__(self, run_id):
        self.run_id = run_id
        self._temporary = tempfile.TemporaryDirectory()

    def __enter__(self):
        from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
        self.store = QualificationCheckpointStore(
            self._temporary.name, self.run_id,
            plan_fingerprint=FP_A, source_fingerprint=FP_B,
        )
        return self.store

    def __exit__(self, exc_type, exc, tb):
        self._temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
