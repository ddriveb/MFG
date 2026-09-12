import math
from pathlib import Path
import unittest

from mfg_hedge.config import load_config
from mfg_hedge.domain import TokenClass
from mfg_hedge.simulation import simulate_healthy_no_hedge
from mfg_hedge.workload import (
    WorkloadTrace,
    generate_workload,
    generate_workload_with_replay,
    lognormal_parameters,
    validate_replay_stream_shape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_default_config():
    return load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")


class LognormalParameterTests(unittest.TestCase):
    def test_parameters_follow_spec_formula(self) -> None:
        mean, cv = 1.0, 0.5
        mu, sigma = lognormal_parameters(mean, cv)
        sigma2 = math.log(1.0 + cv**2)
        self.assertAlmostEqual(sigma, math.sqrt(sigma2))
        self.assertAlmostEqual(mu, math.log(mean) - sigma2 / 2.0)

    def test_arithmetic_mean_is_preserved(self) -> None:
        mean, cv = 2.0, 0.75
        mu, sigma = lognormal_parameters(mean, cv)
        implied_mean = math.exp(mu + sigma**2 / 2.0)
        self.assertAlmostEqual(implied_mean, mean)


class WorkloadGenerationTests(unittest.TestCase):
    def test_same_config_and_seed_reproduce_identical_trace(self) -> None:
        config = load_default_config()
        first = generate_workload(config, 200)
        second = generate_workload(config, 200)
        self.assertEqual(first, second)

    def test_different_seed_changes_trace(self) -> None:
        config = load_default_config()
        base = generate_workload(config, 200)
        other = generate_workload(config, 200, base_seed=config.base_seed + 1)
        self.assertNotEqual(base, other)

    def test_arrival_times_are_non_decreasing(self) -> None:
        trace = generate_workload(load_default_config(), 500)
        arrivals = [token.arrival_time for token in trace.tokens]
        self.assertEqual(arrivals, sorted(arrivals))

    def test_service_times_are_positive(self) -> None:
        trace = generate_workload(load_default_config(), 500)
        for stream in trace.service_times:
            self.assertEqual(len(stream), 500)
            for value in stream:
                self.assertGreater(value, 0.0)

    def test_token_classes_come_from_domain_vocabulary(self) -> None:
        trace = generate_workload(load_default_config(), 500)
        classes = {token.token_class for token in trace.tokens}
        self.assertLessEqual(classes, {TokenClass.REGULAR, TokenClass.URGENT})
        self.assertEqual(len(classes), 2)

    def test_arrival_rate_matches_healthy_offered_load_times_capacity(self) -> None:
        config = load_default_config()
        trace = generate_workload(config, 10)
        expected = config.healthy_offered_load * (
            config.replicas_per_expert / config.healthy_service_mean
        )
        self.assertAlmostEqual(trace.arrival_rate, 1.4)
        self.assertAlmostEqual(trace.arrival_rate, expected)

    def test_per_replica_service_streams_are_independent(self) -> None:
        trace = generate_workload(load_default_config(), 100)
        self.assertEqual(len(trace.service_times), 2)
        self.assertNotEqual(trace.service_times[0], trace.service_times[1])


class ReplayStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config()
        self.token_count = 200

    def test_existing_constructor_without_replay_streams_stays_valid(self) -> None:
        trace = generate_workload(self.config, 10)
        self.assertIsNone(trace.replay_service_times)

    def test_attempt_zero_is_identical_to_the_plain_generator(self) -> None:
        plain = generate_workload(self.config, self.token_count)
        augmented = generate_workload_with_replay(self.config, self.token_count)
        self.assertEqual(plain.tokens, augmented.tokens)
        self.assertEqual(plain.service_times, augmented.service_times)
        self.assertEqual(plain.arrival_rate, augmented.arrival_rate)
        self.assertEqual(plain.base_seed, augmented.base_seed)

    def test_attempt_one_streams_are_deterministic(self) -> None:
        first = generate_workload_with_replay(self.config, self.token_count)
        second = generate_workload_with_replay(self.config, self.token_count)
        self.assertEqual(first.replay_service_times, second.replay_service_times)

    def test_attempt_one_is_independent_of_attempt_zero(self) -> None:
        trace = generate_workload_with_replay(self.config, self.token_count)
        self.assertIsNotNone(trace.replay_service_times)
        for replica in range(self.config.replicas_per_expert):
            self.assertNotEqual(
                trace.service_times[replica], trace.replay_service_times[replica]
            )

    def test_different_seed_changes_attempt_one(self) -> None:
        base = generate_workload_with_replay(self.config, self.token_count)
        other = generate_workload_with_replay(
            self.config, self.token_count, base_seed=self.config.base_seed + 1
        )
        self.assertNotEqual(base.replay_service_times, other.replay_service_times)

    def test_healthy_simulator_ignores_replay_streams(self) -> None:
        plain = generate_workload(self.config, self.token_count)
        augmented = generate_workload_with_replay(self.config, self.token_count)
        plain_result = simulate_healthy_no_hedge(plain, self.config.replicas_per_expert)
        augmented_result = simulate_healthy_no_hedge(
            augmented, self.config.replicas_per_expert
        )
        self.assertEqual(plain_result, augmented_result)

    def test_shape_validation_accepts_valid_streams(self) -> None:
        trace = generate_workload_with_replay(self.config, self.token_count)
        validate_replay_stream_shape(trace, self.config.replicas_per_expert)

    def test_shape_validation_rejects_missing_streams(self) -> None:
        trace = generate_workload(self.config, self.token_count)
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_replay_stream_shape(trace, self.config.replicas_per_expert)

    def test_shape_validation_rejects_wrong_replica_count(self) -> None:
        trace = generate_workload_with_replay(self.config, self.token_count)
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_replay_stream_shape(trace, self.config.replicas_per_expert + 1)

    def test_shape_validation_rejects_undersized_streams(self) -> None:
        trace = generate_workload_with_replay(self.config, self.token_count)
        broken = WorkloadTrace(
            base_seed=trace.base_seed,
            arrival_rate=trace.arrival_rate,
            tokens=trace.tokens,
            service_times=trace.service_times,
            replay_service_times=(trace.replay_service_times[0][:5], trace.replay_service_times[1]),
        )
        with self.assertRaisesRegex(ValueError, "replay"):
            validate_replay_stream_shape(broken, self.config.replicas_per_expert)
    def test_replica_count_must_be_a_true_positive_int(self) -> None:
        trace = generate_workload_with_replay(self.config, self.token_count)
        for bad in (2.0, True, "2", None, 0, -3):
            with self.assertRaises(ValueError, msg=f"replica_count={bad!r}"):
                validate_replay_stream_shape(trace, bad)


if __name__ == "__main__":
    unittest.main()
