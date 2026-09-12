"""Calibration tests: table, interpolation, bisection, aggregation, CRN, real cells."""

import json
from pathlib import Path
import unittest

from mfg_hedge.calibration import (
    CalibrationCell,
    CalibrationTable,
    ProbeSample,
    aggregate_probe_samples,
    background_work_rate,
    build_calibration_background,
    build_calibration_table,
    calibrate_cohort,
    find_background_rate,
    prepare_probe_trace,
    validate_contrast,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import ActionStats, CommonState, ProtectionAction, TokenClass
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.workload import TokenSpec, WorkloadTrace, generate_workload_with_hedge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N, D, I = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
H, F = CommonState.HEALTHY, CommonState.FAILED
DD = CommonState.DEGRADED
R, U = TokenClass.REGULAR, TokenClass.URGENT


def make_stats(
    latency=1.0,
    replay_probability=0.0,
    hedge=0.0,
    replay=0.0,
    wasted=0.0,
):
    return ActionStats(
        mean_latency=latency,
        replay_probability=replay_probability,
        deadline_miss_probability=0.0,
        expected_hedge_work=hedge,
        expected_replay_work=replay,
        expected_wasted_work=wasted,
    )


def make_cell(
    state=H,
    target_rho=0.5,
    token_class=R,
    action=N,
    sample_count=400,
    stats=None,
    achieved_rho=None,
    rate=1.0,
    cohort=400,
    rho_episodes=4,
    probe_episodes=400,
    probe_attempts=400,
    probe_skipped=0,
):
    return CalibrationCell(
        state=state,
        target_rho=target_rho,
        achieved_rho=target_rho if achieved_rho is None else achieved_rho,
        rho_error=0.0,
        token_class=token_class,
        action=action,
        background_arrival_rate=rate,
        sample_count=sample_count,
        rho_measurement_episode_count=rho_episodes,
        rho_background_cohort_count=cohort,
        probe_successful_episode_count=probe_episodes,
        probe_generation_attempt_count=probe_attempts,
        probe_skipped_episode_count=probe_skipped,
        stats=make_stats() if stats is None else stats,
    )


class CalibrationCellValidationTests(unittest.TestCase):
    def test_illegal_numeric_fields_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), True, "0.5"):
            with self.assertRaises(ValueError, msg=f"target_rho={bad!r}"):
                make_cell(target_rho=bad)
        with self.assertRaises(ValueError):
            make_cell(sample_count=-1)
        with self.assertRaises(ValueError):
            make_cell(sample_count=2.5)
        with self.assertRaises(ValueError):
            make_cell(sample_count=True)

    def test_wrong_enum_types_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_cell(state="H")
        with self.assertRaises(ValueError):
            make_cell(action="N")


class CalibrationTableTests(unittest.TestCase):
    def test_duplicate_cell_keys_are_rejected(self) -> None:
        cell = make_cell()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CalibrationTable(cells=(cell, cell))

    def test_lookup_requires_existing_coordinates(self) -> None:
        table = CalibrationTable(cells=(make_cell(),))
        with self.assertRaisesRegex(ValueError, "missing|no cell"):
            table.estimate(F, 0.5, R, N)

    def test_linear_interpolation_between_bracketing_points(self) -> None:
        low = make_cell(target_rho=0.5, stats=make_stats(latency=2.0, wasted=1.0))
        high = make_cell(target_rho=0.7, stats=make_stats(latency=4.0, wasted=3.0))
        table = CalibrationTable(cells=(low, high))
        estimate = table.estimate(H, 0.6, R, N)
        self.assertFalse(estimate.clamped)
        self.assertAlmostEqual(estimate.stats.mean_latency, 3.0)
        self.assertAlmostEqual(estimate.stats.expected_wasted_work, 2.0)

    def test_out_of_grid_lookup_clamps_with_flag(self) -> None:
        low = make_cell(target_rho=0.5, stats=make_stats(latency=2.0))
        high = make_cell(target_rho=0.7, stats=make_stats(latency=4.0))
        table = CalibrationTable(cells=(low, high))
        self.assertTrue(table.estimate(H, 0.4, R, N).clamped)
        self.assertAlmostEqual(table.estimate(H, 0.4, R, N).stats.mean_latency, 2.0)
        self.assertTrue(table.estimate(H, 0.9, R, N).clamped)
        self.assertAlmostEqual(table.estimate(H, 0.9, R, N).stats.mean_latency, 4.0)
        exact = table.estimate(H, 0.7, R, N)
        self.assertFalse(exact.clamped)
        self.assertAlmostEqual(exact.stats.mean_latency, 4.0)

    def test_serialization_is_deterministic_and_timestamp_free(self) -> None:
        cells = (
            make_cell(state=DD, target_rho=0.7, action=D),
            make_cell(target_rho=0.5),
            make_cell(target_rho=0.7),
            make_cell(token_class=U),
        )
        first = CalibrationTable(cells=cells).to_json_bytes()
        second = CalibrationTable(cells=cells).to_json_bytes()
        self.assertEqual(first, second)
        reordered = CalibrationTable(cells=tuple(reversed(cells))).to_json_bytes()
        self.assertEqual(first, reordered)


class BisectionTests(unittest.TestCase):
    def test_linear_measure_converges_to_target(self) -> None:
        result = find_background_rate(lambda rate: rate / 2.0, 0.7)
        # The first midpoint within tolerance wins; exact root not required.
        self.assertAlmostEqual(result.background_arrival_rate, 1.4, delta=0.05)
        self.assertAlmostEqual(result.achieved_rho, 0.7, delta=0.02)
        self.assertLessEqual(result.rho_error, 0.02)
        self.assertEqual(result.target_rho, 0.7)
        self.assertLessEqual(result.iterations, 40)

    def test_unbracketed_target_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "bracket"):
            find_background_rate(lambda rate: min(rate / 2.0, 0.3), 0.5)

    def test_never_within_tolerance_is_not_disguised(self) -> None:
        def wobbling(rate: float) -> float:
            base = 0.2 + 0.1 * rate
            if 1.0 < rate < 2.0:
                base += 0.08
            return base

        with self.assertRaisesRegex(ValueError, "tolerance|converge"):
            find_background_rate(wobbling, 0.35)

    def test_deterministic_repetition(self) -> None:
        first = find_background_rate(lambda rate: rate / 1.5, 0.9)
        second = find_background_rate(lambda rate: rate / 1.5, 0.9)
        self.assertEqual(first, second)


class AggregationTests(unittest.TestCase):
    def test_probe_sample_means_with_zero_fill(self) -> None:
        samples = [
            ProbeSample(
                latency=2.0,
                replayed=True,
                hedge_work=0.0,
                replay_work=1.5,
                wasted_work=0.5,
            ),
            ProbeSample(
                latency=4.0,
                replayed=False,
                hedge_work=3.0,
                replay_work=0.0,
                wasted_work=2.5,
            ),
        ]
        stats = aggregate_probe_samples(samples)
        self.assertAlmostEqual(stats.mean_latency, 3.0)
        self.assertAlmostEqual(stats.replay_probability, 0.5)
        self.assertAlmostEqual(stats.expected_hedge_work, 1.5)
        self.assertAlmostEqual(stats.expected_replay_work, 0.75)
        self.assertAlmostEqual(stats.expected_wasted_work, 1.5)
        self.assertEqual(stats.deadline_miss_probability, 0.0)

    def test_serialization_declares_action_stats_schema_two(self) -> None:
        payload = json.loads(
            CalibrationTable(cells=(make_cell(stats=make_stats(wasted=0.25)),))
            .to_json_bytes()
        )
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["cells"][0]["stats"]["expected_wasted_work"], 0.25)

    def test_empty_samples_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_probe_samples([])


class ContrastTests(unittest.TestCase):
    def test_identical_d_cells_fail_fast(self) -> None:
        stats = make_stats(replay_probability=0.1, replay=0.2)
        cells = tuple(make_cell(state=DD, action=a, stats=stats) for a in (N, D, I))
        with self.assertRaisesRegex(ValueError, "contrast"):
            validate_contrast(CalibrationTable(cells=cells))

    def test_differing_d_cell_passes(self) -> None:
        cells = (
            make_cell(state=DD, action=N, stats=make_stats(replay_probability=0.1)),
            make_cell(state=DD, action=D, stats=make_stats(replay_probability=0.05)),
            make_cell(state=DD, action=I, stats=make_stats(replay_probability=0.1)),
        )
        validate_contrast(CalibrationTable(cells=cells))


class RealDegradedCellTests(unittest.TestCase):
    """One genuine 400-sample D cell through the real engine path."""

    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")

    def calibrate(self):
        return calibrate_cohort(self.config, DD, 0.7, R, samples_per_cell=400)

    def test_real_d_cell_samples_rho_and_contrast(self) -> None:
        cells = self.calibrate()
        self.assertEqual([cell.action for cell in cells], [N, D, I])
        for cell in cells:
            self.assertEqual(cell.sample_count, 400)
            self.assertEqual(cell.target_rho, 0.7)
            self.assertLessEqual(cell.rho_error, 0.02)
            self.assertGreater(cell.background_arrival_rate, 0.0)
            self.assertGreaterEqual(cell.rho_background_cohort_count, 400)
            self.assertEqual(cell.probe_successful_episode_count, 400)
            self.assertEqual(
                cell.probe_generation_attempt_count,
                cell.probe_successful_episode_count + cell.probe_skipped_episode_count,
            )
            self.assertGreaterEqual(cell.stats.expected_wasted_work, 0.0)
        self.assertTrue(
            any(cell.stats.expected_wasted_work > 0.0 for cell in cells),
            msg="real D calibration must expose executed non-winner work",
        )
        by_action = {cell.action: cell for cell in cells}
        normal = by_action[N].stats
        for action in (D, I):
            hedge_stats = by_action[action].stats
            differs = (
                hedge_stats.replay_probability != normal.replay_probability
                or hedge_stats.expected_replay_work != normal.expected_replay_work
            )
            self.assertTrue(differs, msg=f"no contrast for action {action}")

    def test_real_d_cell_is_byte_deterministic(self) -> None:
        first = CalibrationTable(cells=self.calibrate()).to_json_bytes()
        second = CalibrationTable(cells=self.calibrate()).to_json_bytes()
        self.assertEqual(first, second)

    def test_evaluation_namespace_is_untouched(self) -> None:
        before = generate_workload_with_hedge(self.config, 50)
        self.calibrate()
        after = generate_workload_with_hedge(self.config, 50)
        self.assertEqual(before, after)


class RhoFormulaTests(unittest.TestCase):
    def test_background_rho_counts_executed_work_by_arrival_state_per_window(self) -> None:
        # D window [10, 20): tok0 (R0, executed 2.0), tok1 (R1, 1.0),
        # tok2 (R0, fails at 20 with 1.0 executed, Replay executes 2.0).
        # tok3 arrives in F [20, 22) and runs in R; its work belongs to F.
        tokens = tuple(
            TokenSpec(token_id=i, arrival_time=a, token_class=TokenClass.REGULAR)
            for i, a in enumerate((12.0, 14.0, 18.0, 21.0))
        )
        filler = (9.0,) * 4
        trace = WorkloadTrace(
            base_seed=1,
            arrival_rate=1.0,
            tokens=tokens,
            service_times=((2.0, 9.0, 100.0, 9.0), (9.0, 1.0, 9.0, 1.0)),
            replay_service_times=(filler, (9.0, 9.0, 2.0, 9.0)),
            hedge_service_times=(filler, filler),
        )
        timeline = CommonStateTimeline(10.0, 20.0, 22.0)
        result = simulate_hedge_common_state(
            trace, timeline=timeline, degraded_slowdown=2.0
        )
        d_rate = background_work_rate(result, 10.0, 20.0)
        self.assertAlmostEqual(d_rate, (2.0 + 1.0 + 1.0 + 2.0) / 10.0)
        self.assertAlmostEqual(d_rate / 1.5, 0.4)
        f_rate = background_work_rate(result, 20.0, 22.0)
        self.assertAlmostEqual(f_rate, 1.0 / 2.0)
        with self.assertRaises(ValueError):
            background_work_rate(result, 20.0, 20.0)


class ProductionEntryTests(unittest.TestCase):
    def test_samples_below_400_are_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        with self.assertRaisesRegex(ValueError, "400"):
            build_calibration_table(config, samples_per_cell=399)


class ProbeIdentityTests(unittest.TestCase):
    """Designated-probe CRN: background identity survives probe designation."""

    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        self.timeline = CommonStateTimeline(100.0, 200.0, 220.0)

    def test_non_probe_tokens_and_draws_are_identical_to_pure_background(self) -> None:
        label = "calibration:D:background:sample:0"
        background = build_calibration_background(label, 1.0, self.config, horizon=220.0)
        trace, probe_id = prepare_probe_trace(
            self.config, DD, self.timeline, 1.0, label, "calibration:probe:0", U
        )
        self.assertEqual(len(trace.tokens), len(background.tokens))
        self.assertEqual(
            [t.token_id for t in trace.tokens],
            [t.token_id for t in background.tokens],
        )
        for index, (mine, base) in enumerate(zip(trace.tokens, background.tokens)):
            self.assertEqual(mine.arrival_time, base.arrival_time)
            if index == probe_id:
                self.assertEqual(mine.token_class, U)
            else:
                self.assertEqual(mine.token_class, base.token_class)
        for streams_mine, streams_base in zip(
            (trace.service_times, trace.replay_service_times, trace.hedge_service_times),
            (background.service_times, background.replay_service_times, background.hedge_service_times),
        ):
            self.assertEqual(len(streams_mine), len(streams_base))
            for mine, base in zip(streams_mine, streams_base):
                self.assertEqual(len(mine), len(base))
                for token_id, (got, want) in enumerate(zip(mine, base)):
                    if token_id == probe_id:
                        continue
                    self.assertEqual(got, want)

    def test_same_sample_variants_share_one_trace_and_probe_draws_differ_per_sample(
        self,
    ) -> None:
        first = prepare_probe_trace(
            self.config, DD, self.timeline, 1.0,
            "calibration:D:background:sample:3", "calibration:probe:3", R,
        )
        second = prepare_probe_trace(
            self.config, DD, self.timeline, 1.0,
            "calibration:D:background:sample:3", "calibration:probe:3", R,
        )
        third = prepare_probe_trace(
            self.config, DD, self.timeline, 1.0,
            "calibration:D:background:sample:4", "calibration:probe:4", R,
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        probe_a = first[1]
        probe_b = third[1]
        draws_a = first[0].service_times[0][probe_a]
        draws_b = third[0].service_times[0][probe_b]
        self.assertNotEqual(draws_a, draws_b)


class ProductionTableContractTests(unittest.TestCase):
    GRID = (0.5, 0.7, 0.9, 1.1, 1.3)

    def complete_cells(self):
        cells = []
        for state in (H, DD):
            for rho in self.GRID:
                for cls in (R, U):
                    for action in (N, D, I):
                        cells.append(
                            make_cell(state=state, target_rho=rho, token_class=cls, action=action)
                        )
        for rho in self.GRID:
            for cls in (R, U):
                cells.append(make_cell(state=F, target_rho=rho, token_class=cls, action=N))
        return tuple(cells)

    def test_complete_table_has_exactly_seventy_cells(self) -> None:
        cells = self.complete_cells()
        self.assertEqual(len(cells), 70)
        table = CalibrationTable(
            cells=cells, expected_grid=self.GRID, require_complete=True
        )
        self.assertEqual(len(table.cells), 70)

    def test_missing_cell_is_rejected(self) -> None:
        cells = self.complete_cells()[:-1]
        with self.assertRaisesRegex(ValueError, "complete|missing"):
            CalibrationTable(cells=cells, expected_grid=self.GRID, require_complete=True)

    def test_extra_f_hedge_cell_is_rejected(self) -> None:
        cells = self.complete_cells() + (
            make_cell(state=F, target_rho=0.5, token_class=R, action=I),
        )
        with self.assertRaisesRegex(ValueError, "F|Failed|hedge|complete"):
            CalibrationTable(cells=cells, expected_grid=self.GRID, require_complete=True)

    def test_production_sample_floor_and_rho_error_consistency(self) -> None:
        cells = list(self.complete_cells())
        cells[0] = make_cell(sample_count=399)
        with self.assertRaisesRegex(ValueError, "400"):
            CalibrationTable(cells=tuple(cells), expected_grid=self.GRID, require_complete=True)
        cells = list(self.complete_cells())
        cells[0] = CalibrationCell(
            state=H, target_rho=0.5, achieved_rho=0.6, rho_error=0.0,
            token_class=R, action=N, background_arrival_rate=1.0,
            sample_count=400, rho_measurement_episode_count=4,
            rho_background_cohort_count=400,
            probe_successful_episode_count=400,
            probe_generation_attempt_count=400,
            probe_skipped_episode_count=0, stats=make_stats(),
        )
        with self.assertRaisesRegex(ValueError, "rho_error"):
            CalibrationTable(cells=tuple(cells), expected_grid=self.GRID, require_complete=True)

    def test_provenance_is_deterministic_and_complete(self) -> None:
        provenance = {
            "calibration_schema_version": 1,
            "seed_namespace": "calibration",
            "rho_grid": list(self.GRID),
            "timeline": {"degraded_start": 10.0, "failed_start": 20.0, "recovered_start": 22.0},
            "rate_bounds": [0.05, 3.0],
            "rho_tolerance": 0.02,
            "bisection_max_iterations": 40,
            "samples_per_cell": 400,
            "min_background_cohort": 400,
            "config": {"healthy_service_mean": 1.0, "service_time_cv": 0.5},
            "simulator_version": "test",
        }
        cells = self.complete_cells()
        first = CalibrationTable(
            cells=cells, expected_grid=self.GRID, require_complete=True,
            provenance=provenance,
        ).to_json_bytes()
        second = CalibrationTable(
            cells=cells, expected_grid=self.GRID, require_complete=True,
            provenance=provenance,
        ).to_json_bytes()
        self.assertEqual(first, second)
        payload = json.loads(first.decode("utf-8"))
        for key in (
            "calibration_schema_version", "seed_namespace", "rho_grid",
            "timeline", "rate_bounds", "rho_tolerance",
            "bisection_max_iterations", "samples_per_cell",
        ):
            self.assertIn(key, payload["provenance"])
        self.assertNotIn("timestamp", payload["provenance"])


class BisectionContractTests(unittest.TestCase):
    def test_evaluations_log_endpoints_and_midpoints(self) -> None:
        result = find_background_rate(
            lambda rate: rate / 2.0, 0.7, context="state=D:rho=0.7:class=R"
        )
        self.assertGreaterEqual(len(result.evaluations), 3)
        self.assertAlmostEqual(result.evaluations[0][0], 0.05)
        self.assertAlmostEqual(result.evaluations[1][0], 3.0)

    def test_failure_message_carries_context_and_best_candidate(self) -> None:
        def wobbling(rate: float) -> float:
            base = 0.2 + 0.1 * rate
            if 1.0 < rate < 2.0:
                base += 0.08
            return base

        with self.assertRaisesRegex(ValueError, "state=D:rho=0.35:class=U"):
            find_background_rate(wobbling, 0.35, context="state=D:rho=0.35:class=U")


class KnownFailureRegressionTests(unittest.TestCase):
    """Previously failing F cells must now calibrate within tolerance."""

    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")

    def test_failed_state_rho_05_urgent(self) -> None:
        cells = calibrate_cohort(self.config, F, 0.5, U, samples_per_cell=400)
        self.assertEqual([cell.action for cell in cells], [N])
        cell = cells[0]
        self.assertEqual(cell.sample_count, 400)
        self.assertLessEqual(cell.rho_error, 0.02)
        self.assertGreaterEqual(cell.rho_background_cohort_count, 400)
        self.assertEqual(cell.probe_successful_episode_count, 400)

    def test_failed_state_rho_07_regular(self) -> None:
        cells = calibrate_cohort(self.config, F, 0.7, R, samples_per_cell=400)
        cell = cells[0]
        self.assertEqual(cell.sample_count, 400)
        self.assertLessEqual(cell.rho_error, 0.02)
        self.assertGreaterEqual(cell.rho_background_cohort_count, 400)


class TimelineAlignmentTests(unittest.TestCase):
    def test_default_calibration_timeline_matches_paired_evaluation(self) -> None:
        from mfg_hedge.calibration import CALIBRATION_TIMELINE

        self.assertEqual(
            (
                CALIBRATION_TIMELINE.degraded_start,
                CALIBRATION_TIMELINE.failed_start,
                CALIBRATION_TIMELINE.recovered_start,
            ),
            (100.0, 200.0, 220.0),
        )


class HorizonGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")

    def test_high_rate_covers_every_state_window(self) -> None:
        trace = build_calibration_background(
            "calibration:test:horizon-high", 3.0, self.config, horizon=220.0
        )
        arrivals = [t.arrival_time for t in trace.tokens]
        self.assertTrue(all(a < 220.0 for a in arrivals))
        self.assertGreater(len(arrivals), 600)
        in_f = sum(1 for a in arrivals if 200.0 <= a < 220.0)
        in_d = sum(1 for a in arrivals if 100.0 <= a < 200.0)
        in_h = sum(1 for a in arrivals if a < 100.0)
        self.assertGreater(in_h, 200)
        self.assertGreater(in_d, 200)
        self.assertGreater(in_f, 30)

    def test_low_rate_still_reaches_failed_window(self) -> None:
        trace = build_calibration_background(
            "calibration:test:horizon-low", 0.1, self.config, horizon=220.0
        )
        arrivals = [t.arrival_time for t in trace.tokens]
        self.assertTrue(arrivals)
        self.assertGreater(arrivals[-1], 200.0)
        self.assertLess(arrivals[-1], 220.0)

    def test_same_unit_prefix_across_rates(self) -> None:
        slow = build_calibration_background(
            "calibration:test:prefix", 1.0, self.config, horizon=220.0
        )
        fast = build_calibration_background(
            "calibration:test:prefix", 2.0, self.config, horizon=220.0
        )
        # Same label: identical unit draws; rates only rescale arrival times.
        self.assertLess(len(slow.tokens), len(fast.tokens))
        width = len(slow.tokens)
        for slow_streams, fast_streams in zip(
            (slow.service_times, slow.replay_service_times, slow.hedge_service_times),
            (fast.service_times, fast.replay_service_times, fast.hedge_service_times),
        ):
            for slow_stream, fast_stream in zip(slow_streams, fast_streams):
                self.assertEqual(fast_stream[:width], slow_stream)
        self.assertEqual(
            [t.arrival_time for t in slow.tokens],
            [t.arrival_time * 2.0 for t in fast.tokens[:width]],
        )

    def test_illegal_rate_and_runaway_guards(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError, msg=f"rate={bad!r}"):
                build_calibration_background("calibration:test:bad", bad, self.config, horizon=220.0)
        empty = build_calibration_background(
            "calibration:test:empty", 1e-9, self.config, horizon=220.0
        )
        self.assertEqual(len(empty.tokens), 0)
        prepared = prepare_probe_trace(
            self.config, DD, CommonStateTimeline(100.0, 200.0, 220.0),
            1e-9, "calibration:test:empty", "calibration:probe:x", R,
        )
        self.assertIsNone(prepared)


class ProbeQuartileTests(unittest.TestCase):
    """400 designated probes must spread over the window, not cluster mid-window."""

    def test_probe_positions_cover_all_quartiles(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "v1_minimal.json")
        timeline = CommonStateTimeline(100.0, 200.0, 220.0)
        positions = []
        for sample in range(400):
            prepared = prepare_probe_trace(
                config,
                DD,
                timeline,
                1.0,
                f"calibration:D:background:sample:{sample}",
                f"calibration:probe:{sample}",
                R,
            )
            if prepared is None:
                continue
            trace, probe_id = prepared
            arrival = trace.tokens[probe_id].arrival_time
            positions.append((arrival - 100.0) / 100.0)
        self.assertGreaterEqual(len(positions), 390)
        quartiles = [0, 0, 0, 0]
        for position in positions:
            quartiles[min(int(position * 4), 3)] += 1
        for index, count in enumerate(quartiles):
            self.assertGreaterEqual(
                count, 40, msg=f"quartile {index} count {count}"
            )
        mean_position = sum(positions) / len(positions)
        self.assertGreaterEqual(mean_position, 0.4)
        self.assertLessEqual(mean_position, 0.6)


class BisectionEndpointTests(unittest.TestCase):
    def test_endpoint_within_tolerance_is_accepted(self) -> None:
        result = find_background_rate(lambda rate: 0.35 if rate < 1.0 else 2.0, 0.35)
        self.assertAlmostEqual(result.background_arrival_rate, 0.05)
        self.assertLessEqual(result.rho_error, 0.02)
        self.assertGreaterEqual(len(result.evaluations), 1)


if __name__ == "__main__":
    unittest.main()
