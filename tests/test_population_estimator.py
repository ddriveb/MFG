import dataclasses
import hashlib
import unittest

from mfg_hedge.reliability_aware_routing import (
    HealthEvent,
    RoutingToken,
    build_manual_trace,
)
from mfg_hedge.population_estimator import (
    BIN_SCHEMA_V1,
    PopulationEstimatorError,
    calibrate_population_shares,
    run_population_forward,
)


class SharePolicy:
    def __init__(self, choices=(0, 1)):
        self.choices = tuple(choices)
        self.calls = []

    def choose(self, observation, context, policy_key):
        self.calls.append((
            observation.public_state.time,
            observation.token_id,
            context.active_token_count,
            context.active_token_class_counts,
            policy_key,
        ))
        return self.choices[observation.token_id % len(self.choices)]

    def predict_share(self, context):
        return ((0, 0.5), (1, 0.5))

    def predict_token_share(self, observation, context):
        return ((self.choices[observation.token_id % len(self.choices)], 1.0),)


def two_batch_trace(*, namespace="population-test", macro_seed=11, episode_index=0,
                    future_failure=False):
    events = (HealthEvent(10.0, "failure", "replica", 0),) if future_failure else ()
    trace = build_manual_trace(
        (
            RoutingToken(0, 0.0, "Regular", 4.0, 0, 0.5),
            RoutingToken(1, 0.0, "Urgent", 2.0, 1, 0.5),
            RoutingToken(2, 2.0, "Regular", 4.0, 2, 0.5),
        ),
        events,
        namespace=namespace,
        macro_seed=macro_seed,
        scenario="population-test",
    )
    return dataclasses.replace(
        trace,
        episode_index=episode_index,
        fingerprint=hashlib.sha256(
            f"{trace.fingerprint}:{episode_index}".encode("ascii")
        ).hexdigest(),
    )


def same_cell_trace(*, count, namespace="population-weighting", macro_seed=50,
                    episode_index=0):
    trace = build_manual_trace(
        tuple(
            RoutingToken(token_id, 0.0, "Regular", 4.0, 0, 0.5)
            for token_id in range(count)
        ),
        (),
        namespace=namespace,
        macro_seed=macro_seed,
        scenario="population-weighting",
    )
    return dataclasses.replace(
        trace,
        episode_index=episode_index,
        fingerprint=hashlib.sha256(
            f"{trace.fingerprint}:{episode_index}".encode("ascii")
        ).hexdigest(),
    )


class PopulationEstimatorTests(unittest.TestCase):
    def test_bin_schema_is_immutable_and_half_open(self):
        self.assertEqual(BIN_SCHEMA_V1.bucket_age(0.0), "age_0_1")
        self.assertEqual(BIN_SCHEMA_V1.bucket_age(1.0), "age_1_2")
        self.assertEqual(BIN_SCHEMA_V1.bucket_queue_depth(0), "queue_0_1")
        self.assertEqual(BIN_SCHEMA_V1.bucket_queue_depth(1), "queue_1_2")
        self.assertEqual(BIN_SCHEMA_V1.bucket_running_age(None), "idle")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            BIN_SCHEMA_V1.schema_id = "tampered"
        with self.assertRaises(PopulationEstimatorError):
            BIN_SCHEMA_V1.bucket_age(float("nan"))

    def test_forward_record_keeps_mu_nu_eta_and_x_separate(self):
        policy = SharePolicy()
        episode = run_population_forward(
            two_batch_trace(), lambda: policy,
        )

        self.assertEqual(len(episode.batches), 2)
        first = episode.batches[0]
        self.assertEqual(first.mu.active_token_count, 2)
        self.assertEqual(first.mu.class_counts, (("Regular", 1), ("Urgent", 1)))
        self.assertEqual(first.eta.token_ids, (0, 1))
        self.assertEqual(first.x.predicted, ((0, 0.5), (1, 0.5)))
        self.assertEqual(first.realized_share, ((0, 0.5), (1, 0.5)))
        self.assertNotEqual(first.mu, first.eta)
        self.assertEqual(len(first.nu.replica_buckets), 8)
        self.assertEqual(len(first.cell_keys), 2)

    def test_policy_prediction_is_captured_before_commit_and_other_batches_refresh(self):
        policy = SharePolicy()
        episode = run_population_forward(two_batch_trace(), lambda: policy)

        self.assertEqual(len(policy.calls), 3)
        self.assertEqual(episode.batches[0].x.predicted, ((0, 0.5), (1, 0.5)))
        self.assertEqual(episode.batches[1].x.predicted, ((0, 1.0),))
        self.assertNotEqual(
            episode.batches[0].public_state_fingerprint,
            episode.batches[1].public_state_fingerprint,
        )

    def test_active_population_excludes_future_arrivals(self):
        policy = SharePolicy()
        run_population_forward(two_batch_trace(), lambda: policy)

        t0_calls = [row for row in policy.calls if row[0] == 0.0]
        self.assertEqual({row[2] for row in t0_calls}, {2})
        self.assertEqual({row[3] for row in t0_calls}, {(("Regular", 1), ("Urgent", 1))})

    def test_future_fault_does_not_change_current_common_prefix(self):
        left = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        right = run_population_forward(
            two_batch_trace(future_failure=True), lambda: SharePolicy(),
        )
        self.assertEqual(
            left.batches[0].common_noise_prefix_fingerprint,
            right.batches[0].common_noise_prefix_fingerprint,
        )
        self.assertEqual(left.batches[0].public_state_fingerprint,
                         right.batches[0].public_state_fingerprint)

    def test_same_trace_and_policy_are_bytewise_deterministic(self):
        left = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        right = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        self.assertEqual(left, right)

    def test_calibration_uses_episode_as_unit_and_reports_residual(self):
        episodes = tuple(
            run_population_forward(
                two_batch_trace(
                    namespace="population-calibration", macro_seed=20,
                    episode_index=seed,
                ),
                lambda: SharePolicy(),
            )
            for seed in (0, 1)
        )
        required = tuple(episodes[0].batches[0].cell_keys)
        calibration = calibrate_population_shares(
            episodes, required_cells=required, min_episode_samples=2,
        )

        self.assertTrue(calibration.complete)
        self.assertEqual(calibration.episode_count, 2)
        self.assertEqual(len(calibration.cells), 2)
        self.assertEqual(calibration.cells[0].episode_count, 2)
        self.assertEqual(calibration.cells[0].residual_l1, 0.0)
        self.assertGreater(calibration.cells[0].occupancy_mass, 0.0)

    def test_missing_or_insufficient_cells_fail_closed(self):
        episode = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        missing = (episode.batches[0].cell_keys[0], ("not-a-cell",))
        with self.assertRaises(PopulationEstimatorError):
            calibrate_population_shares(
                (episode,), required_cells=missing, min_episode_samples=1,
            )
        with self.assertRaises(PopulationEstimatorError):
            calibrate_population_shares(
                (episode,), required_cells=(episode.batches[0].cell_keys[0],),
                min_episode_samples=2,
            )

    def test_duplicate_episode_identity_and_library_mismatch_fail_closed(self):
        first = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        duplicate = run_population_forward(two_batch_trace(), lambda: SharePolicy())
        with self.assertRaises(PopulationEstimatorError):
            calibrate_population_shares((first, duplicate), min_episode_samples=1)

        other = run_population_forward(
            two_batch_trace(namespace="other-library"), lambda: SharePolicy(),
        )
        with self.assertRaises(PopulationEstimatorError):
            calibrate_population_shares((first, other), min_episode_samples=1)

    def test_trace_input_is_not_mutated_and_invalid_policy_fails_before_physics(self):
        trace = two_batch_trace()
        original = trace
        with self.assertRaises(PopulationEstimatorError):
            run_population_forward(trace, lambda: object())
        self.assertEqual(trace, original)

    def test_forward_library_creates_one_fresh_policy_per_episode(self):
        class StatefulPolicy(SharePolicy):
            def choose(self, observation, context, policy_key):
                choice = super().choose(observation, context, policy_key)
                if len(self.calls) > 1:
                    return 1
                return choice

        created = []

        def factory():
            policy = StatefulPolicy()
            created.append(policy)
            return policy

        from mfg_hedge.population_estimator import run_population_forward_library

        episodes = run_population_forward_library(
            (
                two_batch_trace(namespace="population-library", macro_seed=30, episode_index=0),
                two_batch_trace(namespace="population-library", macro_seed=30, episode_index=1),
            ),
            factory,
        )
        self.assertEqual(len(created), 2)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(created[0].calls[0][1], created[1].calls[0][1])

    def test_mu_is_a_true_active_token_empirical_measure(self):
        episode = run_population_forward(two_batch_trace(), lambda: SharePolicy())

        active = episode.batches[0].mu.active_tokens
        self.assertEqual(tuple(row.token_id for row in active), (0, 1))
        self.assertEqual(
            tuple(row.token_class for row in active),
            ("Regular", "Urgent"),
        )
        self.assertEqual(tuple(row.status for row in active), ("decision", "decision"))
        self.assertEqual(episode.batches[0].mu.active_token_count, 2)

    def test_mu_keeps_running_and_new_decision_states_separate(self):
        trace = build_manual_trace(
            (
                RoutingToken(0, 0.0, "Regular", 4.0, 0, 5.0),
                RoutingToken(1, 1.0, "Urgent", 2.0, 1, 0.5),
            ),
            (),
            namespace="population-active-state",
            macro_seed=41,
            scenario="population-active-state",
        )
        episode = run_population_forward(trace, lambda: SharePolicy())
        active = episode.batches[1].mu.active_tokens

        self.assertEqual(tuple(row.token_id for row in active), (0, 1))
        self.assertEqual(tuple(row.status for row in active), ("running", "decision"))
        self.assertEqual((active[0].attempt_id, active[0].replica_id), (0, 0))
        self.assertIsNone(active[1].attempt_id)

    def test_mixed_class_cells_keep_per_token_conditional_predictions(self):
        class ClassConditionalPolicy(SharePolicy):
            def predict_token_share(self, observation, context):
                if observation.token_class == "Regular":
                    return ((0, 1.0),)
                return ((1, 1.0),)

        episodes = tuple(
            run_population_forward(
                two_batch_trace(
                    namespace="population-mixed-cell", macro_seed=40,
                    episode_index=episode_index,
                ),
                lambda: ClassConditionalPolicy(),
            )
            for episode_index in (0, 1)
        )
        first = episodes[0].batches[0]
        required = tuple(first.cell_keys)
        calibration = calibrate_population_shares(
            episodes, required_cells=required, min_episode_samples=2,
        )

        by_class = {
            estimate.key.token_bucket.token_class: estimate
            for estimate in calibration.cells
        }
        self.assertEqual(by_class["Regular"].predicted_share, ((0, 1.0),))
        self.assertEqual(by_class["Urgent"].predicted_share, ((1, 1.0),))
        self.assertEqual(by_class["Regular"].realized_share, ((0, 1.0),))
        self.assertEqual(by_class["Urgent"].realized_share, ((1, 1.0),))

        rows = first.token_rows
        self.assertEqual(tuple(row.token_id for row in rows), (0, 1))
        self.assertEqual(rows[0].predicted_share, ((0, 1.0),))
        self.assertEqual(rows[1].predicted_share, ((1, 1.0),))

    def test_calibration_gives_each_episode_one_weight(self):
        class TokenSkewPolicy(SharePolicy):
            def predict_token_share(self, observation, context):
                replica_id = 0 if observation.token_id == 0 else 1
                return ((replica_id, 1.0),)

        episodes = (
            run_population_forward(
                same_cell_trace(count=1, episode_index=0),
                lambda: TokenSkewPolicy(),
            ),
            run_population_forward(
                same_cell_trace(count=3, episode_index=1),
                lambda: TokenSkewPolicy(),
            ),
        )
        cell = episodes[0].batches[0].cell_keys[0]
        calibration = calibrate_population_shares(
            episodes, required_cells=(cell,), min_episode_samples=2,
        )

        estimate = calibration.cells[0]
        self.assertEqual(estimate.episode_count, 2)
        self.assertAlmostEqual(dict(estimate.predicted_share)[0], 2.0 / 3.0)
        self.assertAlmostEqual(dict(estimate.predicted_share)[1], 1.0 / 3.0)
        self.assertAlmostEqual(dict(estimate.realized_share)[0], 5.0 / 6.0)
        self.assertAlmostEqual(dict(estimate.realized_share)[1], 1.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
