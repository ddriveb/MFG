import dataclasses
import hashlib
import unittest

from mfg_hedge.population_estimator import run_population_forward
from mfg_hedge.reliability_aware_routing import (
    HealthEvent,
    RoutingToken,
    build_manual_trace,
)
from mfg_hedge.simultaneous_token_routing import (
    PopulationDecisionContext,
    TokenDecisionObservation,
)
from mfg_hedge.token_population_response import (
    FINITE_K_LABEL,
    MEAN_FIELD_LABEL,
    FiniteKDeviationCase,
    MeanFieldEnvironment,
    PopulationResponseError,
    evaluate_finite_k_deviation,
    evaluate_finite_k_deviation_panel,
    evaluate_mean_field_continuation,
    action_buckets_for_token,
    select_replica_for_action_bucket,
)


class QueueAwarePolicy:
    def choose(
        self,
        observation: TokenDecisionObservation,
        context: PopulationDecisionContext,
        policy_key: str,
    ) -> int | None:
        if observation.token_id == 0:
            return 0
        depths = observation.public_state.queue_depths
        return 0 if depths[0] <= depths[1] else 1

    def predict_token_share(self, observation, context):
        return ((self.choose(observation, context, "prediction"), 1.0),)


def routing_trace(*, namespace="population-response", macro_seed=70, episode_index=0):
    trace = build_manual_trace(
        (
            RoutingToken(0, 0.0, "Regular", 4.0, 0, 5.0),
            RoutingToken(1, 1.0, "Urgent", 2.0, 1, 1.0),
            RoutingToken(2, 3.0, "Regular", 4.0, 2, 1.0),
        ),
        # Keep replica 4/5 out of the target buckets so that the test can
        # identify a stable bucket without naming a concrete action.
        (
            HealthEvent(0.0, "failure", "replica", 4),
            HealthEvent(0.0, "failure", "replica", 5),
        ),
        namespace=namespace,
        macro_seed=macro_seed,
        scenario="population-response",
    )
    return dataclasses.replace(trace, episode_index=episode_index)


class TokenPopulationResponseTests(unittest.TestCase):
    def setUp(self):
        self.trace = routing_trace()
        self.episode = run_population_forward(self.trace, QueueAwarePolicy)
        self.target_batch = self.episode.batches[0]
        self.target_token_id = self.target_batch.eta.token_ids[0]
        self.buckets = action_buckets_for_token(
            self.target_batch, self.target_token_id,
        )

    def test_mean_field_keeps_environment_exogenous_and_pairs_action_buckets(self):
        environments = []
        for episode_index in range(16):
            trace = routing_trace(
                namespace=f"population-response-{episode_index}",
                episode_index=episode_index,
            )
            trace = dataclasses.replace(
                trace,
                fingerprint=hashlib.sha256(
                    f"{trace.fingerprint}:{episode_index}".encode("ascii")
                ).hexdigest(),
            )
            episode = run_population_forward(trace, QueueAwarePolicy)
            environments.append(MeanFieldEnvironment(
                episode, episode.batches[0].batch_id, 0,
                selection_rule="first_token_at_anchor_v1",
            ))
        environment = environments[0]
        before = environment.path_fingerprint
        seen = []

        def cost(context):
            seen.append((context.crn_key, context.environment.path_fingerprint))
            return float(
                context.selected_replica_id or 0
            ) + context.environment.target_batch.time

        result = evaluate_mean_field_continuation(
            tuple(environments), self.buckets[:2], cost,
            cost_model_id="latency_v1",
        )

        self.assertEqual(result.output_label, MEAN_FIELD_LABEL)
        self.assertEqual(result.attempted_calls, 48)  # 16*(baseline + 2 Q arms)
        self.assertEqual(len(result.rows), 32)
        self.assertEqual(environment.path_fingerprint, before)
        self.assertEqual(
            {row[1] for row in seen},
            {env.path_fingerprint for env in environments},
        )
        self.assertFalse(result.claims_best_action)
        self.assertFalse(result.claims_mfg)
        self.assertEqual(len(result.action_summaries), 2)

    def test_finite_k_reacts_online_and_returns_only_pathwise_difference(self):
        candidate = next(
            bucket for bucket in self.buckets
            if select_replica_for_action_bucket(
                self.target_batch, self.target_token_id, bucket,
                action_key="candidate-test",
            ) == 1
        )
        observed_assignments = []

        def cost(result, target_token_id):
            observed_assignments.append(
                tuple(
                    (audit.batch_id, audit.assigned_replica_ids)
                    for audit in result.batch_audits
                )
            )
            target = next(
                row for row in result.physical.token_results
                if row.token_id == target_token_id
            )
            return float(target.completion_time)

        result = evaluate_finite_k_deviation(
            self.trace,
            self.episode,
            QueueAwarePolicy,
            self.target_token_id,
            (candidate,),
            cost,
            cost_model_id="latency_v1",
        )

        self.assertEqual(result.output_label, FINITE_K_LABEL)
        self.assertEqual(result.attempted_calls, 2)
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0].pathwise_difference,
                         result.rows[0].candidate_cost - result.rows[0].baseline_cost)
        self.assertNotEqual(observed_assignments[0], observed_assignments[1])
        # The later Token is re-routed after the tagged action changes queues.
        self.assertEqual(observed_assignments[0][1][1], (1,))
        self.assertEqual(observed_assignments[1][1][1], (0,))
        self.assertFalse(result.claims_regret)
        self.assertFalse(result.claims_nash)

    def test_same_action_bucket_reproduces_baseline(self):
        baseline_bucket = next(
            bucket for bucket in self.buckets
            if select_replica_for_action_bucket(
                self.target_batch, self.target_token_id, bucket,
                action_key="same-action",
            ) == 0
        )

        def cost(result, target_token_id):
            return float(next(
                row for row in result.physical.token_results
                if row.token_id == target_token_id
            ).completion_time)

        result = evaluate_finite_k_deviation(
            self.trace, self.episode, QueueAwarePolicy,
            self.target_token_id, (baseline_bucket,), cost,
            cost_model_id="latency_v1",
        )
        row = result.rows[0]
        self.assertEqual(row.baseline_cost, row.candidate_cost)
        self.assertEqual(row.pathwise_difference, 0.0)
        self.assertEqual(row.baseline_result_fingerprint,
                         row.candidate_result_fingerprint)

    def test_missing_target_and_unsupported_bucket_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluate_mean_field_continuation(
                (MeanFieldEnvironment(self.episode, 1, 0),),
                self.buckets[:1], lambda context: 1.0,
                cost_model_id="latency_v1",
            )

    def test_bucket_selection_is_id_free_and_stable(self):
        bucket = self.buckets[0]
        left = select_replica_for_action_bucket(
            self.target_batch, self.target_token_id, bucket,
            action_key="stable-action-key",
        )
        right = select_replica_for_action_bucket(
            self.target_batch, self.target_token_id, bucket,
            action_key="stable-action-key",
        )
        self.assertEqual(left, right)
        self.assertNotIn("replica_id", bucket.__dataclass_fields__)
        with self.assertRaises(PopulationResponseError):
            select_replica_for_action_bucket(
                self.target_batch, self.target_token_id, bucket,
                action_key="",
            )

    def test_finite_k_requires_fresh_policy_and_counts_failed_baseline(self):
        created = []

        def factory():
            policy = QueueAwarePolicy()
            created.append(policy)
            return policy

        def failing_cost(result, target_token_id):
            raise RuntimeError("synthetic scorer failure")

        with self.assertRaises(PopulationResponseError) as caught:
            evaluate_finite_k_deviation(
                self.trace, self.episode, factory, self.target_token_id,
                (self.buckets[0],), failing_cost,
                cost_model_id="latency_v1",
            )
        self.assertEqual(caught.exception.attempted_calls, 1)
        self.assertEqual(caught.exception.original_exception_type, "RuntimeError")
        self.assertEqual(len(created), 1)

    def test_finite_k_rejects_duplicate_buckets_and_trace_mismatch(self):
        with self.assertRaises(PopulationResponseError):
            evaluate_finite_k_deviation(
                self.trace, self.episode, QueueAwarePolicy, self.target_token_id,
                (self.buckets[0], self.buckets[0]),
                lambda result, token_id: 1.0,
                cost_model_id="latency_v1",
            )
        mismatched = routing_trace(namespace="different-trace")
        with self.assertRaises(PopulationResponseError):
            evaluate_finite_k_deviation(
                mismatched, self.episode, QueueAwarePolicy, self.target_token_id,
                (self.buckets[0],),
                lambda result, token_id: 1.0,
                cost_model_id="latency_v1",
            )

    def test_finite_k_panel_enforces_sixteen_cases_and_canonical_rows(self):
        cases = []
        for episode_index in range(16):
            trace = routing_trace(
                namespace=f"population-response-panel-{episode_index}",
                episode_index=episode_index,
            )
            episode = run_population_forward(trace, QueueAwarePolicy)
            cases.append(FiniteKDeviationCase(
                trace, episode, episode.batches[0].batch_id, 0,
                (self.buckets[0],),
            ))

        def cost(result, target_token_id):
            return float(next(
                row for row in result.physical.token_results
                if row.token_id == target_token_id
            ).completion_time)

        result = evaluate_finite_k_deviation_panel(
            tuple(reversed(cases)), QueueAwarePolicy, cost,
            cost_model_id="latency_v1",
        )
        self.assertEqual(result.output_label, FINITE_K_LABEL)
        self.assertEqual(result.attempted_calls, 32)
        self.assertEqual(result.per_episode_calls, (2,) * 16)
        self.assertEqual(
            result.episode_keys,
            tuple(sorted(case.episode.identity.key for case in cases)),
        )
        self.assertEqual(
            tuple(row.episode_key for row in result.rows),
            result.episode_keys,
        )
        self.assertFalse(result.claims_regret)
        self.assertFalse(result.claims_nash)

        unsupported = dataclasses.replace(self.buckets[0], health="DOWN")
        with self.assertRaises(ValueError):
            evaluate_finite_k_deviation(
                self.trace, self.episode, QueueAwarePolicy, self.target_token_id,
                (unsupported,), lambda result, token_id: 1.0,
                cost_model_id="latency_v1",
            )


if __name__ == "__main__":
    unittest.main()
