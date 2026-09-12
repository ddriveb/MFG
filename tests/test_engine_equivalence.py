"""Engine bit-exactness goldens and stateless-policy contract tests (ticket 15).

The golden fingerprints in this file were captured from the pre-fix engine and
must remain identical after each performance fix (F1/F2/F3). Run
``python -m tests.test_engine_equivalence --capture`` to print the golden
values; embed them in _GOLDENS below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import unittest

from mfg_hedge.population_estimator import (
    BIN_SCHEMA_V1,
    run_population_forward,
)
from mfg_hedge.reliability_aware_routing import (
    HealthEvent,
    RoutingToken,
    build_manual_trace,
)
from mfg_hedge.simultaneous_token_routing import (
    PopulationDecisionContext,
    TokenDecisionObservation,
    simulate_simultaneous_routing,
)
from mfg_hedge.token_mfg_qualification_backend import (
    ConcreteQualificationBackend,
    FixedEpisodeLibrary,
    RoutingPolicySeed,
    _PolicyStateAdapter,
    materialize_policy,
)
from mfg_hedge.token_population_response import (
    action_buckets_for_token,
    evaluate_finite_k_deviation,
)
from mfg_hedge.topology import Topology


def _canonical(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, float):
        return value
    return value


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _result_digest(result) -> str:
    return _digest({
        "physical": result.physical,
        "batches": result.batch_audits,
    })


class RecordingPolicy:
    """Deterministic LOEW-style policy that logs its call sequence."""

    def __init__(self):
        self.calls = []

    def choose(self, observation, context, policy_key):
        available = tuple(
            row for row in observation.public_state.replica_observations if row.available
        )
        if not available:
            self.calls.append((policy_key, observation.public_state.fingerprint, None))
            return None
        selected = min(
            available,
            key=lambda row: (row.estimated_work, row.queue_depth, row.replica_id),
        ).replica_id
        self.calls.append((policy_key, observation.public_state.fingerprint, selected))
        return selected

    def call_sequence_digest(self):
        return _digest(tuple(self.calls))


class QueueZeroPolicy:
    """Token 0 -> replica 0; everyone else JSQ over available Replicas."""

    def choose(self, observation, context, policy_key):
        if observation.token_id == 0:
            return 0
        depths = observation.public_state.queue_depths
        available = tuple(
            row.replica_id
            for row in observation.public_state.replica_observations
            if row.available
        )
        if not available:
            return None
        return min(available, key=lambda rid: (depths[rid], rid))

    def predict_token_share(self, observation, context):
        return ((self.choose(observation, context, "prediction"), 1.0),)


def k8_equivalence_trace():
    return build_manual_trace(
        (
            RoutingToken(0, 0.0, "Regular", 40.0, 0, 5.0),
            RoutingToken(1, 0.0, "Urgent", 30.0, 1, 3.0),
            RoutingToken(2, 1.5, "Regular", 40.0, 2, 4.0),
            RoutingToken(3, 2.0, "Regular", 40.0, 3, 2.0),
            RoutingToken(4, 2.0, "Urgent", 30.0, 4, 6.0),
            RoutingToken(5, 6.0, "Regular", 40.0, 5, 3.0),
            RoutingToken(6, 9.0, "Urgent", 30.0, 6, 2.0),
            RoutingToken(7, 12.0, "Regular", 40.0, 7, 4.0),
        ),
        (
            HealthEvent(0.0, "failure", "replica", 4),
            HealthEvent(3.0, "failure", "replica", 1),
            HealthEvent(4.5, "recovery", "replica", 4),
            HealthEvent(7.0, "failure", "domain", 2),
            HealthEvent(10.0, "recovery", "domain", 2),
        ),
        namespace="engine-equivalence",
        macro_seed=1501,
        scenario="S0",
    )


def k64_equivalence_trace():
    tokens = tuple(
        RoutingToken(index, float(index) * 0.5, "Urgent" if index % 3 == 0 else "Regular", 60.0, index, float(2 + index % 5))
        for index in range(16)
    )
    return build_manual_trace(
        tokens,
        (
            HealthEvent(0.0, "failure", "replica", 9),
            HealthEvent(2.0, "failure", "replica", 17),
            HealthEvent(3.0, "failure", "domain", 1),
            HealthEvent(5.0, "recovery", "replica", 9),
            HealthEvent(8.0, "failure", "domain", 3),
            HealthEvent(11.0, "recovery", "domain", 1),
            HealthEvent(14.0, "recovery", "domain", 3),
        ),
        namespace="engine-equivalence",
        macro_seed=6401,
        scenario="S0",
        topology=Topology(64),
    )


def _target_cost(result, target_token_id):
    token = next(
        row for row in result.physical.token_results if row.token_id == target_token_id
    )
    completed = [row for row in token.attempts if row.terminal_status == "completed"]
    return completed[-1].terminal_time if completed else 0.0


def _deviation_digest(trace, policy_factory):
    episode = run_population_forward(trace, policy_factory)
    batch = episode.batches[0]
    target = batch.eta.token_ids[0]
    buckets = action_buckets_for_token(batch, target)
    result = evaluate_finite_k_deviation(
        trace,
        episode,
        policy_factory,
        target,
        buckets,
        _target_cost,
        cost_model_id="latency_v1",
    )
    return _digest({
        "trace": trace.fingerprint,
        "rows": result.rows,
        "attempted": result.attempted_calls,
        "complete": result.complete,
    })


def capture_goldens():
    goldens = {}
    for name, trace in (("k8_sim", k8_equivalence_trace()), ("k64_sim", k64_equivalence_trace())):
        policy = RecordingPolicy()
        result = simulate_simultaneous_routing(trace, policy, policy_name="golden")
        goldens[name] = _result_digest(result)
        goldens[name + "_calls"] = policy.call_sequence_digest()
        goldens[name + "_metrics"] = _digest(dict(result.physical.metrics))
    goldens["k8_deviation"] = _deviation_digest(
        k8_equivalence_trace(), QueueZeroPolicy
    )
    goldens["k64_deviation"] = _deviation_digest(
        k64_equivalence_trace(), QueueZeroPolicy
    )
    return goldens


_GOLDENS = {
    "k8_deviation": "1d592a177e52ccafb0cec155888a9a09bf0f696a9186adb223426651d3155354",
    "k8_sim": "c0baabae513a4926b1eff2ecbefe053b2742074568532ba2d6e556655ff2b890",
    "k8_sim_calls": "4aed6237e4642ae43e9101874373124f79285ab1ff8cb9068bfcb3b79c04b3e2",
    "k8_sim_metrics": "4926c779dd5716c9e228b71a6efbacd4d1dd0fc80a24fb334dd08e28194c8334",
    "k64_deviation": "6cc378070251ed80b367435a444dfe21501f24090dbe3968aad0bc1fb1aa75b0",
    "k64_sim": "56f4870d580f65885cb9b289b99b9400b4cb4adb0d869cf99144ea26bac63ec1",
    "k64_sim_calls": "9635910925d4f57be373652cc46775e1cda2a1c821879e15869ee8eeb2a5812a",
    "k64_sim_metrics": "6ba788a03eb99ecb7ddda3e3dba383eae464646f348870d9f36f0513b2a62165",
}


class SimulationGoldenTests(unittest.TestCase):
    def test_k8_simulation_bit_exact(self):
        policy = RecordingPolicy()
        result = simulate_simultaneous_routing(
            k8_equivalence_trace(), policy, policy_name="golden",
        )
        self.assertEqual(_GOLDENS["k8_sim"], _result_digest(result))
        self.assertEqual(_GOLDENS["k8_sim_calls"], policy.call_sequence_digest())
        self.assertEqual(_GOLDENS["k8_sim_metrics"], _digest(dict(result.physical.metrics)))

    def test_k64_simulation_bit_exact(self):
        policy = RecordingPolicy()
        result = simulate_simultaneous_routing(
            k64_equivalence_trace(), policy, policy_name="golden",
        )
        self.assertEqual(_GOLDENS["k64_sim"], _result_digest(result))
        self.assertEqual(_GOLDENS["k64_sim_calls"], policy.call_sequence_digest())
        self.assertEqual(_GOLDENS["k64_sim_metrics"], _digest(dict(result.physical.metrics)))


class DeviationGoldenTests(unittest.TestCase):
    def test_k8_deviation_rows_bit_exact(self):
        self.assertEqual(
            _GOLDENS["k8_deviation"], _deviation_digest(k8_equivalence_trace(), QueueZeroPolicy),
        )

    def test_k64_deviation_rows_bit_exact(self):
        self.assertEqual(
            _GOLDENS["k64_deviation"], _deviation_digest(k64_equivalence_trace(), QueueZeroPolicy),
        )


class SnapshotForkTests(unittest.TestCase):
    """F2: a run forked from any batch boundary must equal the full replay
    bit-exactly when the policy is pure (RecordingPolicy is pure LOEW)."""

    def _fork_reproduces_full_run(self, trace):
        snapshots = {}
        full = simulate_simultaneous_routing(
            trace, RecordingPolicy(), policy_name="golden",
            snapshot_recorder=lambda bid, snap: snapshots.setdefault(bid, snap),
        )
        self.assertGreaterEqual(len(snapshots), 2)
        expected = _result_digest(full)
        for batch_id, snapshot in snapshots.items():
            forked = simulate_simultaneous_routing(
                trace, RecordingPolicy(), policy_name="golden",
                fork_snapshot=snapshot,
            )
            self.assertEqual(
                expected, _result_digest(forked),
                f"fork at batch {batch_id} diverged from the full replay",
            )

    def test_k8_snapshot_fork_reproduces_full_run(self):
        self._fork_reproduces_full_run(k8_equivalence_trace())

    def test_k64_snapshot_fork_reproduces_full_run(self):
        self._fork_reproduces_full_run(k64_equivalence_trace())


class BaselineCacheTests(unittest.TestCase):
    """F1: a cached baseline must return bit-identical deviation rows."""

    def test_cache_hit_returns_identical_rows(self):
        from mfg_hedge.token_population_response import _BASELINE_CACHE
        _BASELINE_CACHE.clear()
        trace = k8_equivalence_trace()
        factory = _adapter_factory = lambda: _PolicyStateAdapter(
            RoutingPolicySeed(kind="uniform"), BIN_SCHEMA_V1
        )
        episode = run_population_forward(trace, factory)
        batch = episode.batches[0]
        target = batch.eta.token_ids[0]
        buckets = action_buckets_for_token(batch, target)

        def run():
            return evaluate_finite_k_deviation(
                trace, episode, factory, target, buckets, _target_cost,
                cost_model_id="latency_v1",
            )

        first = run()
        self.assertEqual(1, len(_BASELINE_CACHE))
        second = run()
        self.assertEqual(_digest(first.rows), _digest(second.rows))
        self.assertEqual(first.attempted_calls, second.attempted_calls)


class StatelessPolicyContractTests(unittest.TestCase):
    """Ticket-15 contract: snapshot-path policies must be pure functions of
    (observation, context, key); two fresh instances must agree call-for-call
    and must not mutate their instance state."""

    def _observation_sequence(self, trace, seed_kind):
        seed = RoutingPolicySeed(kind=seed_kind) if seed_kind != "scale" else RoutingPolicySeed(
            kind="scale_marginal_projection",
            marginals=(
                ("Regular", (("a", 0.5), ("b", 0.5))),
                ("Urgent", (("a", 0.25), ("b", 0.75))),
            ),
        )
        adapter = _PolicyStateAdapter(seed, BIN_SCHEMA_V1)
        observations = []

        class Recorder:
            def choose(self, observation, context, policy_key):
                observations.append((observation, context, policy_key))
                return adapter.choose(observation, context, policy_key)

        simulate_simultaneous_routing(trace, Recorder(), policy_name="contract")
        return observations

    def _assert_pure(self, trace, seed_kind):
        observations = self._observation_sequence(trace, seed_kind)
        self.assertTrue(observations)

        def fresh():
            if seed_kind == "scale":
                seed = RoutingPolicySeed(
                    kind="scale_marginal_projection",
                    marginals=(
                        ("Regular", (("a", 0.5), ("b", 0.5))),
                        ("Urgent", (("a", 0.25), ("b", 0.75))),
                    ),
                )
            else:
                seed = RoutingPolicySeed(kind=seed_kind)
            return _PolicyStateAdapter(seed, BIN_SCHEMA_V1)

        first = fresh()
        state_before = dict(vars(first))
        outputs_one = [first.choose(obs, ctx, key) for obs, ctx, key in observations]
        self.assertEqual(state_before, dict(vars(first)))
        second = fresh()
        outputs_two = [second.choose(obs, ctx, key) for obs, ctx, key in observations]
        self.assertEqual(outputs_one, outputs_two)

    def test_seed_uniform_is_pure(self):
        self._assert_pure(k8_equivalence_trace(), "uniform")

    def test_seed_loew_projection_is_pure(self):
        self._assert_pure(k8_equivalence_trace(), "simultaneous_loew_projection")

    def test_seed_risk_aware_projection_is_pure(self):
        self._assert_pure(k8_equivalence_trace(), "corrected_risk_aware_projection")

    def test_seed_scale_marginal_projection_is_pure(self):
        self._assert_pure(k8_equivalence_trace(), "scale")

    def test_policy_state_adapter_is_pure(self):
        trace = k8_equivalence_trace()
        library = FixedEpisodeLibrary.from_traces((trace,))
        backend = ConcreteQualificationBackend(
            library,
            schema=BIN_SCHEMA_V1,
            mean_field_namespace="contract-mf",
            mean_field_macro_seed=1,
            finite_k_namespace="contract-fk",
            finite_k_macro_seed=2,
            bounded=False,
            min_complete_panels=1,
        )
        snapshot = backend._forward(RoutingPolicySeed(kind="uniform"), "unpriced_mfg", 0)
        policy = materialize_policy(snapshot)
        observations = self._observation_sequence(trace, "uniform")
        adapter = _PolicyStateAdapter(policy, BIN_SCHEMA_V1)
        before = dict(vars(adapter))
        outputs = [adapter.choose(obs, ctx, key) for obs, ctx, key in observations]
        self.assertEqual(before, dict(vars(adapter)))
        fresh = _PolicyStateAdapter(materialize_policy(snapshot), BIN_SCHEMA_V1)
        self.assertEqual(outputs, [fresh.choose(obs, ctx, key) for obs, ctx, key in observations])
        self.assertTrue(getattr(_PolicyStateAdapter, "__engine_stateless__", False))


if __name__ == "__main__":
    if "--capture" in sys.argv:
        print(json.dumps(capture_goldens(), indent=1, sort_keys=True))
    else:
        unittest.main(argv=[sys.argv[0]])
