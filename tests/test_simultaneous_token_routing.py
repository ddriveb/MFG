import unittest

from mfg_hedge.reliability_aware_routing import (
    RoutingToken,
    build_manual_trace,
    simulate_routing,
)
from mfg_hedge.simultaneous_token_routing import (
    SimultaneousRoutingError,
    simulate_simultaneous_routing,
)


class RecordingPolicy:
    def __init__(self, replica_id=0):
        self.replica_id = replica_id
        self.calls = []

    def choose(self, observation, context, policy_key):
        self.calls.append((
            observation.public_state.time,
            observation.token_id,
            observation.public_state.fingerprint,
            observation.public_state.queue_depths,
            context.cohort_token_count,
            context.published_realized_share,
            getattr(context, "current_batch_actions", None),
        ))
        return self.replica_id


class FailoverPolicy(RecordingPolicy):
    def choose(self, observation, context, policy_key):
        self.calls.append((
            observation.public_state.time,
            observation.token_id,
            observation.public_state.fingerprint,
            observation.public_state.queue_depths,
            context.cohort_token_count,
            context.published_realized_share,
            getattr(context, "current_batch_actions", None),
        ))
        if observation.token_id == 0 and observation.retry_count == 0:
            return 0
        return 1


def two_token_trace():
    return build_manual_trace((
        RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),
        RoutingToken(1, 0.0, "Regular", 4.0, 1, 1.0),
    ))


class SimultaneousTokenRoutingTests(unittest.TestCase):
    def test_same_batch_uses_one_pre_batch_public_state(self):
        policy = RecordingPolicy()
        result = simulate_simultaneous_routing(two_token_trace(), policy)

        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(policy.calls[0][2], policy.calls[1][2])
        self.assertEqual(policy.calls[0][3], policy.calls[1][3])
        self.assertEqual(policy.calls[0][4], 2)
        self.assertEqual(result.batch_audits[0].token_ids, (0, 1))
        self.assertEqual(result.batch_audits[0].pre_batch_queue_depths[0], 0)

    def test_policy_cannot_read_same_batch_realized_actions(self):
        policy = RecordingPolicy()
        simulate_simultaneous_routing(two_token_trace(), policy)

        self.assertEqual(policy.calls[0][5], ())
        self.assertIsNone(policy.calls[0][6])
        self.assertEqual(policy.calls[0][5], policy.calls[1][5])

    def test_assignments_commit_atomically_after_all_decisions(self):
        policy = RecordingPolicy(replica_id=0)
        result = simulate_simultaneous_routing(two_token_trace(), policy)

        audit = result.batch_audits[0]
        self.assertEqual(audit.assigned_replica_ids, (0, 0))
        self.assertEqual(audit.queue_depths_after_commit[0], 2)
        self.assertEqual(audit.queue_depths_after_commit[1], 0)
        self.assertEqual(len(result.physical.starts), 2)
        self.assertEqual(result.physical.starts[0].token_id, 0)
        self.assertEqual(result.physical.starts[0].start_time, 0.0)
        self.assertEqual(result.physical.starts[1].token_id, 1)
        self.assertEqual(result.physical.starts[1].start_time, 1.0)

    def test_replay_and_arrival_form_one_causal_decision_cohort(self):
        from mfg_hedge.reliability_aware_routing import HealthEvent

        trace = build_manual_trace((
            RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),
            RoutingToken(1, 1.0, "Regular", 4.0, 1, 1.0),
        ), (HealthEvent(1.0, "failure", "replica", 0),))
        policy = FailoverPolicy()
        result = simulate_simultaneous_routing(trace, policy)

        t1 = next(audit for audit in result.batch_audits if audit.time == 1.0)
        self.assertEqual(t1.token_ids, (0, 1))
        calls_at_t1 = [call for call in policy.calls if call[0] == 1.0]
        self.assertEqual({call[2] for call in calls_at_t1}, {t1.public_state_fingerprint})

    def test_k8_single_copy_physics_matches_historical_engine(self):
        trace = build_manual_trace((
            RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),
        ))
        policy = RecordingPolicy(replica_id=0)
        simultaneous = simulate_simultaneous_routing(trace, policy).physical
        historical = simulate_routing(trace, "uniform_rr")

        self.assertEqual(simultaneous.trace_fingerprint, historical.trace_fingerprint)
        self.assertEqual(simultaneous.token_results, historical.token_results)
        self.assertEqual(simultaneous.starts, historical.starts)
        self.assertEqual(dict(simultaneous.metrics), dict(historical.metrics))
        self.assertEqual(dict(simultaneous.invariants), dict(historical.invariants))

    def test_repeated_run_is_deterministic(self):
        left = simulate_simultaneous_routing(two_token_trace(), RecordingPolicy())
        right = simulate_simultaneous_routing(two_token_trace(), RecordingPolicy())
        self.assertEqual(left, right)

    def test_invalid_policy_action_fails_before_commit(self):
        class InvalidPolicy:
            def choose(self, observation, context, policy_key):
                return 0.0

        with self.assertRaises(SimultaneousRoutingError):
            simulate_simultaneous_routing(two_token_trace(), InvalidPolicy())


if __name__ == "__main__":
    unittest.main()
