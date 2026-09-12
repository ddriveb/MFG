"""Red/green contract tests for the accepted Stage 4b precision pilot."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

from mfg_hedge.precision_pilot import (
    PILOT_COMMON_PATHS,
    PILOT_MACRO_SEED,
    PILOT_NAMESPACE,
    PILOT_POPULATION_REPLICATES,
    PILOT_PROFILES,
    PILOT_RESAMPLING_SEED,
    B_OUTER,
    B_WITHIN,
    PilotAttemptPayload,
    PilotConfig,
    PilotScenarioResult,
    PilotTokenPayload,
    PilotScenario,
    _profile_sources,
    _result_from_tagged,
    build_pilot_library,
    corrected_nested_variance,
    jackknife_se,
    pilot_call_count,
    profile_names,
    select_prefix_cell,
    score_pooled_payloads,
    write_pilot_artifacts,
)
from mfg_hedge.expert_game import score_tagged_result
from mfg_hedge.shared_backup import prepare_shared_trace, simulate_shared_backup_tagged


def _attempt(status="completed_winner", executed=1.0):
    return PilotAttemptPayload(
        attempt_id=0,
        replica_id=0,
        status=status,
        required_work=1.0,
        executed_work=executed,
    )


def _token(global_id, *, phase="D", token_class="R", latency=1.0,
           replay_count=0, action="N"):
    return PilotTokenPayload(
        global_token_id=global_id,
        local_token_id=global_id,
        phase=phase,
        token_class=token_class,
        arrival_time=float(global_id),
        latency=latency,
        replay_count=replay_count,
        requested_action=action,
        applied_action=action,
        attempts=(_attempt(),),
    )


def _payload(profile, common, population, offset=0):
    tokens = tuple(
        _token(offset + index, phase=phase, token_class=cls,
               latency=latency)
        for index, (phase, cls, latency) in enumerate((
            ("H", "R", 1.0), ("H", "U", 1.0),
            ("D", "R", 2.0), ("D", "U", 2.0),
            ("F", "R", 3.0), ("F", "U", 3.0),
            ("R", "R", 1.0), ("R", "U", 1.0),
        ))
    )
    return PilotScenarioResult(
        profile=profile,
        common_path_id=common,
        population_id=population,
        episode_key=f"pilot:{common}:{population}:{profile}",
        trace_fingerprint=f"trace:{common}:{population}",
        fault_fingerprint=f"fault:{common}",
        tokens=tokens,
        counters=(("token_arrivals", len(tokens)),),
        pool_total_executed=float(len(tokens)),
        pool_total_capacity=float(len(tokens)),
        drain_end_time=10.0,
    )


class PrecisionPilotContractTests(unittest.TestCase):
    def test_frozen_namespace_seed_profiles_and_dimensions(self):
        config = PilotConfig()
        self.assertEqual(config.namespace, PILOT_NAMESPACE)
        self.assertEqual(config.macro_seed, PILOT_MACRO_SEED)
        self.assertEqual(config.resampling_seed, PILOT_RESAMPLING_SEED)
        self.assertEqual(config.max_common_paths, PILOT_COMMON_PATHS)
        self.assertEqual(config.max_population_replicates,
                         PILOT_POPULATION_REPLICATES)
        self.assertEqual(config.epsilon_nash, 0.20)
        self.assertEqual(config.pilot_width_target, 0.20)
        self.assertEqual(profile_names(), tuple(name for name, *_ in PILOT_PROFILES))
        self.assertNotIn("shared-backup-game:v1:stage4-fit", config.namespace)
        with self.assertRaises(FrozenInstanceError):
            config.macro_seed = 1

    def test_maximum_library_has_one_canonical_rectangular_prefix(self):
        library = build_pilot_library()
        self.assertEqual(len(library.scenarios), 64 * 8)
        self.assertEqual(library.generation_count, 1)
        cell = select_prefix_cell(library, 16, 2)
        self.assertEqual(len(cell), 32)
        self.assertEqual(
            [(row.common_path_id, row.population_id) for row in cell],
            [(common, population) for common in range(16) for population in range(2)],
        )
        self.assertIs(cell[0], library.scenarios[0])

    def test_profile_set_and_exact_call_arithmetic(self):
        self.assertEqual(
            tuple(PILOT_PROFILES),
            (
                ("NSNS/NSNS", "NSNS", "NSNS"),
                ("NSNS/NSSS", "NSSS", "NSNS"),
                ("NSNS/NSND", "NSND", "NSNS"),
                ("NSNS/NNDX", "NNDX", "NSNS"),
            ),
        )
        self.assertEqual(pilot_call_count(64, 8), 2048)
        self.assertEqual(pilot_call_count(16, 2), 128)

    def test_pooled_score_is_not_an_average_of_path_scores(self):
        rows = []
        for common in range(2):
            for population in range(2):
                rows.append(_payload("NSNS/NSNS", common, population,
                                     offset=(common * 2 + population) * 8))
        score = score_pooled_payloads(rows)
        self.assertTrue(score.complete)
        self.assertEqual(score.generated_token_count, 32)
        self.assertEqual(score.phase_class_counts["D"]["R"], 4)
        self.assertEqual(score.effective_tail_mass["F"], 0.4)

    def test_normative_gain_is_positive_when_deviation_lowers_objective(self):
        baseline = _payload("NSNS/NSNS", 0, 0)
        lower_latency = tuple(replace(token, latency=token.latency * 0.5)
                             for token in baseline.tokens)
        deviation = replace(
            baseline, profile="NSNS/NSSS", episode_key="pilot:0:0:dev",
            tokens=lower_latency,
        )
        base_score = score_pooled_payloads((baseline,))
        dev_score = score_pooled_payloads((deviation,))
        gain = base_score.total - dev_score.total
        self.assertGreater(gain, 0.0)

    def test_weighted_multiplicity_preserves_exact_pooled_tail_and_work(self):
        first = _payload("NSNS/NSNS", 0, 0)
        second = _payload("NSNS/NSNS", 0, 1, offset=100)
        weighted = score_pooled_payloads(
            (first, second), multiplicities={(0, 0): 2, (0, 1): 1}
        )
        expanded = score_pooled_payloads((first, second, replace(
            first, population_id=2, episode_key="pilot:0:2:NSNS/NSNS",
        )))
        self.assertEqual(weighted.generated_token_count, expanded.generated_token_count)
        self.assertAlmostEqual(weighted.raw_work_totals["executed"],
                               expanded.raw_work_totals["executed"])

    def test_multiplicity_shape_must_match_unique_scenario_rows(self):
        row = _payload("NSNS/NSNS", 0, 0)
        with self.assertRaises(ValueError):
            score_pooled_payloads((row,), multiplicities={(0, 1): 1})
        with self.assertRaises(ValueError):
            score_pooled_payloads((row, row))

    def test_failed_executed_attempt_is_retained_as_waste(self):
        row = _payload("NSNS/NSNS", 0, 0)
        token = replace(
            row.tokens[0],
            attempts=(
                PilotAttemptPayload(0, 0, "failed_running", 1.0, 0.5),
                PilotAttemptPayload(1, 1, "completed_winner", 1.0, 1.0),
            ),
        )
        scored = score_pooled_payloads((replace(row, tokens=(token,) + row.tokens[1:]),))
        self.assertGreater(scored.raw_work_totals["wasted"], 0.0)

    def test_nested_correction_clamps_negative_between_component(self):
        result = corrected_nested_variance(
            outer_means=(1.0, 1.0, 1.0),
            within_variances=(100.0, 100.0, 100.0),
            b_within=2,
        )
        self.assertEqual(result.v_between_raw, 0.0)
        self.assertEqual(result.v_between, 0.0)
        self.assertEqual(result.v_total, result.v_within)

    def test_nested_correction_rejects_unpaired_outer_shapes(self):
        with self.assertRaises(ValueError):
            corrected_nested_variance((1.0,), (1.0,), b_within=256)
        with self.assertRaises(ValueError):
            corrected_nested_variance((1.0, 2.0), (1.0,), b_within=256)

    def test_tail_mass_is_fractional_for_every_phase_and_class(self):
        score = score_pooled_payloads((_payload("NSNS/NSNS", 0, 0),))
        self.assertEqual(score.effective_tail_mass["D/R"], 0.05)
        self.assertEqual(score.effective_tail_mass["F/U"], 0.05)

    def test_result_payload_rejects_unknown_profile_and_duplicate_attempt(self):
        row = _payload("NSNS/NSNS", 0, 0)
        with self.assertRaises(ValueError):
            replace(row, profile="NSNS/XXXX")
        with self.assertRaises(ValueError):
            duplicate = replace(
                row.tokens[0],
                attempts=(row.tokens[0].attempts[0], row.tokens[0].attempts[0]),
            )
            replace(row, tokens=(duplicate,) + row.tokens[1:])

    def test_frozen_bootstrap_constants_and_profile_order_are_canonical(self):
        self.assertEqual((B_OUTER, B_WITHIN), (4096, 256))
        self.assertEqual(profile_names(), tuple(name for name, *_ in PILOT_PROFILES))
        self.assertEqual(profile_names()[0], "NSNS/NSNS")

    def test_incomplete_pooled_score_is_not_silently_zero_filled(self):
        row = _payload("NSNS/NSNS", 0, 0)
        incomplete = score_pooled_payloads((replace(row, tokens=(row.tokens[0],)),))
        self.assertFalse(incomplete.complete)
        self.assertIsNone(incomplete.total)
        self.assertNotEqual(incomplete.missing_cohorts, ())

    def test_pooled_payload_matches_stage3_tagged_score(self):
        from tests.test_expert_game import cohort_trace

        trace = cohort_trace()
        scenario = PilotScenario(
            0, 0, "scorer-check:0:0", prepare_shared_trace(trace), "fault:0"
        )
        tagged = simulate_shared_backup_tagged(
            scenario.prepared_trace,
            0.5,
            degraded_slowdown=2.0,
            hedge_delay=1.5,
            action_sources={0: _profile_sources("NSNS/NSNS")[0]},
            tagged_expert_id=0,
        )
        payload = _result_from_tagged("NSNS/NSNS", scenario, tagged)
        stage3 = score_tagged_result(
            tagged,
            trace=scenario.trace,
            expert_id=0,
            episode_key=scenario.episode_key,
            fault_fingerprint=scenario.fault_fingerprint,
        )
        pooled = score_pooled_payloads((payload,))
        self.assertTrue(pooled.complete)
        self.assertEqual(stage3.total, pooled.total)
        for key in ("executed", "wasted", "winner_executed", "attempt_count"):
            self.assertEqual(stage3.raw_work_totals[key], pooled.raw_work_totals[key])

    def test_jackknife_and_nested_correction_match_hand_calculation(self):
        self.assertAlmostEqual(jackknife_se((1.0, 3.0, 2.0)),
                               ((2.0 / 3.0) * 2.0) ** 0.5)
        result = corrected_nested_variance(
            outer_means=(2.0, 4.0),
            within_variances=(1.0, 3.0),
            b_within=2,
        )
        self.assertAlmostEqual(result.v_between_raw, 2.0)
        self.assertAlmostEqual(result.v_within, 2.0)
        self.assertAlmostEqual(result.v_between, 1.0)
        self.assertAlmostEqual(result.v_total, 3.0)

    def test_nested_resampling_requires_independent_repeated_positions_and_paired_indices(self):
        result = corrected_nested_variance(
            outer_means=(1.0, 1.0), within_variances=(4.0, 4.0), b_within=256,
            repeated_path_positions=((3, 3),), paired_rule_indices=True,
        )
        self.assertEqual(result.repeated_path_positions, ((3, 3),))
        self.assertTrue(result.independent_inner_by_position)
        self.assertTrue(result.paired_rule_indices)

    def test_invalid_payload_fails_closed_before_scoring(self):
        good = _payload("NSNS/NSNS", 0, 0)
        incomplete = PilotScenarioResult(
            profile=good.profile,
            common_path_id=good.common_path_id,
            population_id=good.population_id,
            episode_key=good.episode_key + ":incomplete",
            trace_fingerprint=good.trace_fingerprint,
            fault_fingerprint=good.fault_fingerprint,
            tokens=(_token(0, phase="D", token_class="R"),),
            counters=good.counters,
            pool_total_executed=good.pool_total_executed,
            pool_total_capacity=good.pool_total_capacity,
            drain_end_time=good.drain_end_time,
        )
        score = score_pooled_payloads((incomplete,))
        self.assertFalse(score.complete)
        self.assertIsNone(score.total)
        self.assertIn(("H", "R"), score.missing_cohorts)

    def test_artifacts_are_transactional_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = {"schema": "stage4b-pilot", "seed": 20260907}
            pilot = {"status": "precision_plan_infeasible"}
            report = {"cells": []}
            run_dir = write_pilot_artifacts(root, "stage4b-test", manifest,
                                            pilot, report)
            self.assertTrue((Path(run_dir) / "manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                write_pilot_artifacts(root, "stage4b-test", manifest,
                                      pilot, report)
            payload = json.loads((Path(run_dir) / "pilot.json").read_text())
            self.assertEqual(payload["status"], "precision_plan_infeasible")

    def test_result_claim_boundary_is_not_candidate_nash_or_mfg(self):
        from mfg_hedge.precision_pilot import PilotDisposition
        self.assertEqual(PilotDisposition.PRECISION_PLAN_INFEASIBLE.value,
                         "precision_plan_infeasible")
        self.assertFalse(PilotDisposition.PRECISION_PLAN_INFEASIBLE.claims_nash)
        self.assertFalse(PilotDisposition.PRECISION_PLAN_INFEASIBLE.claims_mfg)


if __name__ == "__main__":
    unittest.main()
