"""Focused contract tests for the accepted ADR-0023 execution slice."""

from pathlib import Path
import unittest

from mfg_hedge.attribution_episode import ATTRIBUTION_V1_PROTOCOL
from mfg_hedge.common_state import Phase
from mfg_hedge.domain import ProtectionAction as A, TokenClass
from mfg_hedge.token_online import TokenObservation
from mfg_hedge.token_t3a_execution import (
    ANCHOR_WINDOWS,
    COMBINED_CALL_LIMIT,
    OCCUPANCY_EPISODES,
    T3AAnchorSelector,
    T3AExecutionError,
    _run_validation,
    build_source_bundle,
    build_provenance,
    run_occupancy,
    validation_episode_count,
)
from mfg_hedge.config import load_config
from tests.token_t2_support import make_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def observation(age: float, *, primary: int = 0) -> TokenObservation:
    return TokenObservation(
        token_id=0,
        token_class=TokenClass.REGULAR,
        arrival_time=100.0 + age,
        phase=Phase.DEGRADED,
        phase_age=age,
        primary_replica=primary,
        queue_snapshot=((), ()),
        running_attempts=(),
        observed_history=("H", "D"),
        reservation_window=(100.0, 125.0),
        reservation_cap=2.8125,
        reservation_balance=2.8125,
        public_price=0.0,
    )


class AnchorSelectorTests(unittest.TestCase):
    def test_four_half_open_anchor_windows_are_frozen(self):
        self.assertEqual(
            ANCHOR_WINDOWS,
            ((0.0, 1.5), (1.5, 3.0), (3.0, 10.0), (10.0, None)),
        )
        self.assertTrue(T3AAnchorSelector(0).select(observation(0.0)))
        self.assertFalse(T3AAnchorSelector(0).select(observation(1.5)))
        self.assertTrue(T3AAnchorSelector(1).select(observation(1.5)))
        self.assertFalse(T3AAnchorSelector(3).select(observation(9.999)))
        self.assertTrue(T3AAnchorSelector(3).select(observation(10.0)))
        self.assertFalse(T3AAnchorSelector(0).select(observation(0.5, primary=1)))

    def test_anchor_id_is_strict(self):
        for value in (-1, 4, True, 0.0, "0"):
            with self.subTest(value=value), self.assertRaises(T3AExecutionError):
                T3AAnchorSelector(value)


class AllocationTests(unittest.TestCase):
    def test_allocation_uses_only_retained_occupancy_and_is_bounded(self):
        self.assertEqual(validation_episode_count({"a": 16, "b": 40}), 2048)
        self.assertEqual(validation_episode_count({"a": 32}), 1024)
        self.assertEqual(validation_episode_count({"a": 512}), 64)
        self.assertEqual(OCCUPANCY_EPISODES, 512)
        self.assertEqual(COMBINED_CALL_LIMIT, 10_752)

    def test_empty_or_subfloor_screen_fails_closed(self):
        for counts in ({}, {"a": 15}):
            with self.subTest(counts=counts), self.assertRaises(T3AExecutionError):
                validation_episode_count(counts)


class ProvenanceTests(unittest.TestCase):
    def test_source_bundle_is_complete_and_deterministic(self):
        first = build_source_bundle(PROJECT_ROOT)
        second = build_source_bundle(PROJECT_ROOT)
        self.assertEqual(first, second)
        paths = {entry[0] for entry in first.entries}
        self.assertIn("src/mfg_hedge/token_t3a_execution.py", paths)
        self.assertIn("src/mfg_hedge/token_continuation.py", paths)
        self.assertIn("configs/v1_minimal.json", paths)
        self.assertEqual(first.schema_id, "token_t3a_source_bundle_v1")
        self.assertEqual(len(first.fingerprint), 64)

    def test_protocol_clock_is_the_frozen_real_timeline(self):
        timeline = ATTRIBUTION_V1_PROTOCOL.timeline
        self.assertEqual(
            (timeline.degraded_start, timeline.failed_start, timeline.recovered_start),
            (100.0, 200.0, 220.0),
        )
        self.assertEqual(ATTRIBUTION_V1_PROTOCOL.arrival_cutoff, 320.0)


class PhysicalExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        cls.bundle = build_source_bundle(PROJECT_ROOT)

    def _builder(self, index):
        return make_episode(
            arrivals=(100.5, 101.0, 220.5),
            timeline=ATTRIBUTION_V1_PROTOCOL.timeline,
            cutoff=ATTRIBUTION_V1_PROTOCOL.arrival_cutoff,
            base_seed=8800 + index,
        )

    def test_one_episode_uses_one_selection_plus_four_shared_physical_runs(self):
        occupancy = run_occupancy(
            self.config,
            self.bundle,
            episode_count=1,
            episode_builder=self._builder,
        )
        bin_id = occupancy.episodes[0].bin_id
        self.assertIsNotNone(bin_id)
        result = _run_validation(
            self.config,
            self.bundle,
            retained_bins=(bin_id,),
            episode_count=1,
            minimum_complete_panels=1,
            episode_builder=self._builder,
        )
        self.assertEqual(result.attempted_calls, 5)
        self.assertEqual(len(result.estimates), 9)
        self.assertTrue(all(row.effective_n == 1 for row in result.estimates))
        self.assertTrue(all(row.status == "completed" for row in result.estimates))
        self.assertFalse(result.claims_best_response)
        self.assertFalse(result.claims_regret)
        self.assertFalse(result.claims_nash)
        self.assertFalse(result.claims_mfg)

    def test_validation_output_is_deterministic_and_models_stay_separate(self):
        occupancy = run_occupancy(
            self.config,
            self.bundle,
            episode_count=1,
            episode_builder=self._builder,
        )
        bin_id = occupancy.episodes[0].bin_id
        kwargs = dict(
            retained_bins=(bin_id,),
            episode_count=1,
            minimum_complete_panels=1,
            episode_builder=self._builder,
        )
        first = _run_validation(self.config, self.bundle, **kwargs)
        second = _run_validation(self.config, self.bundle, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            {row.model_id for row in first.estimates},
            {
                "token_initial_price_v0",
                "token_runtime_adr0009_v1",
                "token_extended_reservation_v1",
            },
        )
        self.assertEqual(first.provenance, build_provenance(self.config, self.bundle))


if __name__ == "__main__":
    unittest.main()
