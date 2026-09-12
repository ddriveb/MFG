import hashlib
import json
import tempfile
import unittest

from mfg_hedge.qualification_checkpoint import QualificationCheckpointStore
from mfg_hedge.reliability_aware_routing import (
    RoutingToken,
    build_manual_trace,
)
from mfg_hedge.token_mfg_qualification_backend import (
    ConcreteQualificationBackend,
    BoundedBackendResult,
)
from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    qualification_source_fingerprint,
)
from mfg_hedge.token_mfg_qualification import FixedEpisodeLibrary


def _fp(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _library(
    count=4,
    *,
    namespace="bounded-real-backend",
    macro_seed=20260920,
):
    def tokens_for_episode(episode):
        regular_id, urgent_id = (0, 1) if episode % 2 == 0 else (1, 0)
        regular = RoutingToken(
            token_id=regular_id,
            arrival_time=1.0,
            token_class="Regular",
            deadline=4.0,
            ingress_rank=0,
            service_work=1.0,
        )
        urgent = RoutingToken(
            token_id=urgent_id,
            arrival_time=1.0,
            token_class="Urgent",
            deadline=4.0,
            ingress_rank=0,
            service_work=1.0,
        )
        return (regular, urgent) if episode % 2 == 0 else (urgent, regular)

    traces = tuple(
        build_manual_trace(
            tokens_for_episode(episode),
            namespace=namespace,
            macro_seed=macro_seed,
            scenario="S0",
            episode_index=episode,
        )
        for episode in range(count)
    )
    return FixedEpisodeLibrary.from_traces(traces)


class ConcreteQualificationBackendTests(unittest.TestCase):
    def test_backend_is_real_and_runs_all_five_layers_on_bounded_k8(self):
        backend = ConcreteQualificationBackend(_library())
        self.assertTrue(backend.real_physics)
        self.assertEqual(backend.source_fingerprint, qualification_source_fingerprint())

        result = backend.run_bounded_k8()

        self.assertIsInstance(result, BoundedBackendResult)
        self.assertEqual(result.topology_k, 8)
        self.assertEqual(result.trace_library_fingerprint, backend.library.fingerprint)
        self.assertEqual(
            result.layers,
            (
                "forward_population",
                "mean_field_continuation",
                "finite_k_deviation",
                "response_iteration",
                "final_policy_confirmation",
            ),
        )
        self.assertGreater(result.forward_calls, 0)
        self.assertGreater(result.continuation_calls, 0)
        self.assertGreater(result.finite_k_calls, 0)
        self.assertEqual(result.trace_fingerprint, backend.library.fingerprint)
        self.assertEqual(result.formal_calls, 0)
        self.assertTrue(result.real_physics)

    def test_bounded_backend_reuses_fixed_trace_and_checkpoint_without_dispatch(self):
        backend = ConcreteQualificationBackend(_library())
        first = backend.run_bounded_k8()
        with tempfile.TemporaryDirectory() as root:
            plan = QualificationPlan()
            store = QualificationCheckpointStore(
                root,
                "bounded-backend",
                plan_fingerprint=plan.protocol_fingerprint,
                source_fingerprint=qualification_source_fingerprint(),
            )
            payload = first.to_checkpoint_output()
            key = "bounded-k8/real-backend"
            input_fp = _fp({"trace_library": backend.library.fingerprint, "k": 8})
            self.assertTrue(store.reserve(
                key, input_fingerprint=input_fp, expected_calls=first.total_calls,
            ))
            store.commit(
                key,
                input_fingerprint=input_fp,
                completed_calls=first.total_calls,
                output=payload,
            )
            resumed = backend.resume_bounded_from_checkpoint(store, key, input_fp)

        self.assertEqual(resumed.to_bytes(), first.to_bytes())
        self.assertEqual(resumed.dispatched_calls, 0)
        self.assertEqual(resumed.reused_calls, 1)
        self.assertEqual(resumed.formal_calls, 0)


if __name__ == "__main__":
    unittest.main()
