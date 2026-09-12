"""Hand-computed metrics tests for the Common State + No Hedge summary."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.common_state_metrics import build_common_state_summary
from mfg_hedge.common_state_simulation import simulate_common_state_no_hedge
from mfg_hedge.config import (
    CommonStateExperimentConfig,
    load_common_state_config_with_sha256,
)
from mfg_hedge.domain import TokenClass
from mfg_hedge.workload import (
    TokenSpec,
    WorkloadTrace,
    generate_workload_with_replay,
)


TIMELINE = CommonStateTimeline(
    degraded_start=10.0, failed_start=20.0, recovered_start=30.0
)
SLOWDOWN = 2.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config() -> CommonStateExperimentConfig:
    return CommonStateExperimentConfig(
        schema_version=2,
        experiment_name="single_expert_mechanism_check",
        expert_count=1,
        replicas_per_expert=2,
        failure_domain_count=2,
        regular_token_ratio=0.8,
        urgent_token_ratio=0.2,
        healthy_service_mean=1.0,
        service_time_cv=0.5,
        degraded_slowdown=2.0,
        hedge_delay_quantile=0.9,
        target_max_utilization=0.9,
        healthy_offered_load=0.7,
        softmax_inverse_temperature=5.0,
        policy_damping=0.2,
        price_step=0.05,
        price_damping=0.2,
        tokens_per_run=100000,
        base_seed=20260901,
        seed_count=20,
        smoke_tokens=1000,
        degraded_start=10.0,
        failed_start=20.0,
        recovered_start=30.0,
    )


def build_trace(
    arrivals, service_r0, service_r1, replay_r0, replay_r1
) -> WorkloadTrace:
    tokens = tuple(
        TokenSpec(token_id=i, arrival_time=a, token_class=TokenClass.REGULAR)
        for i, a in enumerate(arrivals)
    )
    return WorkloadTrace(
        base_seed=1,
        arrival_rate=1.4,
        tokens=tokens,
        service_times=(service_r0, service_r1),
        replay_service_times=(replay_r0, replay_r1),
    )


def summarize(trace: WorkloadTrace, timeline: CommonStateTimeline = TIMELINE) -> dict:
    result = simulate_common_state_no_hedge(
        trace, timeline=timeline, degraded_slowdown=SLOWDOWN
    )
    return build_common_state_summary(
        make_config(), trace, result, run_id="unit-test", config_sha256="0" * 64
    )


# Trace A: one completion per phase H/D/F/R with a running failure and Replay.
TRACE_A = build_trace(
    (2.0, 9.0, 18.0, 35.0),
    (4.0, 9.0, 2.0, 9.0),
    (9.0, 2.0, 9.0, 2.0),
    (9.0, 9.0, 9.0, 9.0),
    (9.0, 9.0, 0.75, 9.0),
)

# Trace B: queued invalidation; exercises cumulative queue delay.
TRACE_B = build_trace(
    (18.0, 18.5, 19.0),
    (8.0, 9.0, 3.0),
    (9.0, 0.5, 9.0),
    (9.0, 9.0, 9.0),
    (1.0, 9.0, 2.0),
)

# Trace C: single early completion; D/F/R never observed.
TRACE_C = build_trace((2.0,), (4.0,), (9.0,), (9.0,), (9.0,))


class IdentityAndAlgebraTests(unittest.TestCase):
    def test_identity_block(self) -> None:
        summary = summarize(TRACE_A)
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["run_id"], "unit-test")
        self.assertEqual(summary["scenario"], "common_state_no_hedge")
        self.assertEqual(summary["dispatcher"], "round_robin_failure_aware")
        self.assertEqual(summary["policy"], "no_hedge")
        self.assertEqual(summary["config_schema_version"], 2)
        self.assertEqual(summary["resolved_config"]["degraded_start"], 10.0)
        self.assertEqual(summary["config_sha256"], "0" * 64)
        self.assertEqual(summary["generated_tokens"], 4)
        self.assertEqual(summary["token_count"], 4)
        self.assertAlmostEqual(summary["arrival_rate"], 1.4)
        self.assertEqual(
            summary["timeline"],
            {"degraded_start": 10.0, "failed_start": 20.0, "recovered_start": 30.0},
        )

    def test_work_algebra_is_exact(self) -> None:
        work = summarize(TRACE_A)["work"]
        self.assertAlmostEqual(work["nominal_primary_work"], 10.0)
        self.assertAlmostEqual(work["wasted_work"], 1.0)
        self.assertAlmostEqual(work["executed_primary_work"], 9.0)
        self.assertAlmostEqual(work["executed_replay_work"], 0.75)
        self.assertAlmostEqual(work["total_executed_work"], 9.75)
        self.assertAlmostEqual(work["extra_execution_ratio"], 0.075)
        self.assertAlmostEqual(work["execution_amplification"], 0.975)

    def test_overall_latency_and_cumulative_queue_delay(self) -> None:
        summary = summarize(TRACE_B)
        self.assertAlmostEqual(summary["mean_latency"], 2.5)
        # tok2 waits 19->20 invalidated, then 20->21 for its Replay.
        self.assertAlmostEqual(summary["mean_queue_delay"], 2.0 / 3.0)
        self.assertLessEqual(summary["latency_p50"], summary["latency_p95"])
        self.assertLessEqual(summary["latency_p95"], summary["latency_p99"])

    def test_drain_and_throughput_fields(self) -> None:
        summary = summarize(TRACE_A)
        self.assertEqual(summary["last_arrival_time"], 35.0)
        self.assertEqual(summary["drain_end_time"], 37.0)
        self.assertEqual(summary["drain_duration"], 2.0)
        self.assertAlmostEqual(summary["throughput"], 4.0 / 37.0)
        self.assertEqual(
            summary["observation_interval"],
            {"start": 0.0, "end": 37.0, "includes_drain": True},
        )
        self.assertEqual(
            summary["throughput_basis"], "end_to_end_including_drain"
        )


class PhaseMetricsTests(unittest.TestCase):
    def test_per_phase_samples_and_verdicts(self) -> None:
        phases = summarize(TRACE_A)["phases"]
        expectations = {
            "H": (1, 4.0, False, 0.0, 10.0),
            "D": (1, 2.0, True, 10.0, 20.0),
            "F": (1, 2.75, True, 20.0, 30.0),
            "R": (1, 2.0, False, 30.0, 37.0),
        }
        for phase, (count, p50, violation, start, end) in expectations.items():
            entry = phases[phase]
            self.assertTrue(entry["observed"], msg=phase)
            self.assertEqual(entry["sample_count"], count, msg=phase)
            self.assertAlmostEqual(entry["latency_p50"], p50, msg=phase)
            self.assertEqual(entry["capacity_violation"], violation, msg=phase)
            self.assertAlmostEqual(entry["observation_start"], start, msg=phase)
            self.assertAlmostEqual(entry["observation_end"], end, msg=phase)

    def test_per_replica_ratios_per_phase(self) -> None:
        phases = summarize(TRACE_A)["phases"]
        for phase in ("H", "R"):
            ratios = phases[phase]["replica_ratios"]
            self.assertAlmostEqual(ratios[0]["ratio"], 0.7)
            self.assertAlmostEqual(ratios[1]["ratio"], 0.7)
        degraded = phases["D"]["replica_ratios"]
        self.assertAlmostEqual(degraded[0]["ratio"], 1.4)
        self.assertAlmostEqual(degraded[1]["ratio"], 0.7)
        failed = phases["F"]["replica_ratios"]
        self.assertIsNone(failed[0]["ratio"])
        self.assertAlmostEqual(failed[1]["ratio"], 1.4)
        # Aggregate ratios exist but are descriptive only.
        self.assertAlmostEqual(phases["D"]["aggregate_ratio"], 1.4 / 1.5)

    def test_unobserved_phases_are_null_not_false(self) -> None:
        phases = summarize(TRACE_C)["phases"]
        self.assertTrue(phases["H"]["observed"])
        self.assertFalse(phases["H"]["capacity_violation"])
        for phase in ("D", "F", "R"):
            entry = phases[phase]
            self.assertFalse(entry["observed"], msg=phase)
            self.assertIsNone(entry["capacity_violation"], msg=phase)
            self.assertIsNone(entry["observation_start"], msg=phase)
            self.assertIsNone(entry["observation_end"], msg=phase)
            self.assertEqual(entry["sample_count"], 0, msg=phase)
            self.assertIsNone(entry["latency_p50"], msg=phase)
            self.assertIsNone(entry["latency_p99"], msg=phase)

    def test_observed_phase_with_zero_samples_has_null_percentiles(self) -> None:
        phases = summarize(TRACE_B)["phases"]
        self.assertTrue(phases["H"]["observed"])
        self.assertEqual(phases["H"]["sample_count"], 0)
        self.assertIsNone(phases["H"]["latency_p50"])
        self.assertFalse(phases["H"]["capacity_violation"])
        self.assertEqual(phases["D"]["sample_count"], 1)
        self.assertEqual(phases["F"]["sample_count"], 2)


class FaultAndReplicaAccountingTests(unittest.TestCase):
    def test_fault_counters_and_replay_rate(self) -> None:
        faults = summarize(TRACE_B)["faults"]
        self.assertEqual(faults["completed_tokens"], 3)
        self.assertEqual(faults["failed_running_primary_executions"], 1)
        self.assertEqual(faults["invalidated_queued_primary_executions"], 1)
        self.assertEqual(faults["failed_primary_executions"], 2)
        self.assertEqual(faults["replayed_tokens"], 2)
        self.assertEqual(faults["replay_executions"], 2)
        self.assertAlmostEqual(faults["replay_rate"], 2.0 / 3.0)
        self.assertEqual(faults["hedge_launches"], 0)
        self.assertEqual(faults["queue_length_at_failed_start"], [0, 2])
        # Both Replays drain by t=23, before recovery at t=30.
        self.assertEqual(faults["queue_length_at_recovered_start"], [0, 0])

    def test_per_replica_busy_wall_time_and_normalized_work_differ_in_d(self) -> None:
        replicas = summarize(TRACE_A)["replicas"]
        r0, r1 = replicas
        self.assertEqual(r0["assigned"], 2)
        self.assertEqual(r0["replay_executions"], 0)
        self.assertAlmostEqual(r0["busy_wall_time"], 6.0)
        self.assertAlmostEqual(r0["normalized_work"], 5.0)
        self.assertAlmostEqual(r0["utilization"], 6.0 / 37.0)
        self.assertEqual(r1["assigned"], 2)
        self.assertEqual(r1["replay_executions"], 1)
        self.assertAlmostEqual(r1["busy_wall_time"], 4.75)
        self.assertAlmostEqual(r1["normalized_work"], 4.75)

    def test_no_fault_trace_reports_zero_fault_metrics(self) -> None:
        summary = summarize(TRACE_C)
        faults = summary["faults"]
        self.assertEqual(faults["replayed_tokens"], 0)
        self.assertAlmostEqual(faults["replay_rate"], 0.0)
        work = summary["work"]
        self.assertAlmostEqual(work["wasted_work"], 0.0)
        self.assertAlmostEqual(work["executed_replay_work"], 0.0)
        self.assertAlmostEqual(work["extra_execution_ratio"], 0.0)
        self.assertAlmostEqual(work["execution_amplification"], 1.0)

    def test_all_algebraic_invariants_hold(self) -> None:
        for trace in (TRACE_A, TRACE_B, TRACE_C):
            summary = summarize(trace)
            invariants = summary["invariants"]
            for key, value in invariants.items():
                if key != "hedge_launches":
                    self.assertTrue(value, msg=f"{key} in {invariants}")
            self.assertEqual(invariants["hedge_launches"], 0)


class HedgeInvariantTests(unittest.TestCase):
    def _config_trace_result(self):
        config = make_config()
        result = simulate_common_state_no_hedge(
            TRACE_A, timeline=TIMELINE, degraded_slowdown=SLOWDOWN
        )
        return config, TRACE_A, result

    def test_zero_hedge_launches_passes(self) -> None:
        config, trace, result = self._config_trace_result()
        summary = build_common_state_summary(
            config, trace, result, run_id="unit-test", config_sha256="0" * 64
        )
        self.assertEqual(summary["faults"]["hedge_launches"], 0)
        self.assertEqual(summary["invariants"]["hedge_launches"], 0)

    def test_nonzero_hedge_launches_are_rejected(self) -> None:
        config, trace, result = self._config_trace_result()
        for bad in (1, 7, -1):
            anomalous = replace(result, hedge_launches=bad)
            with self.assertRaisesRegex(RuntimeError, "hedge_launches"):
                build_common_state_summary(
                    config, trace, anomalous, run_id="unit-test", config_sha256="0" * 64
                )


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
EXISTING_SCHEMA2 = sorted(ARTIFACT_DIR.glob("*common-state-no-hedge-t1000-*/summary.json"))


class ControlledSummaryStabilityTests(unittest.TestCase):
    @unittest.skipUnless(EXISTING_SCHEMA2, "no existing schema-2 artifact to compare")
    def test_fresh_summary_matches_existing_artifact_except_run_id(self) -> None:
        artifact = json.loads(EXISTING_SCHEMA2[0].read_text(encoding="utf-8"))
        config, config_sha256 = load_common_state_config_with_sha256(
            PROJECT_ROOT / "configs" / "v1_common_state.json"
        )
        trace = generate_workload_with_replay(
            config.base_config(), artifact["token_count"]
        )
        result = simulate_common_state_no_hedge(
            trace,
            timeline=config.timeline(),
            degraded_slowdown=config.degraded_slowdown,
            replica_count=config.replicas_per_expert,
        )
        fresh = build_common_state_summary(
            config, trace, result, run_id="probe", config_sha256=config_sha256
        )
        # Per the Healthy stability contract: byte-identity holds under the
        # same simulator_version; across versions, provenance metadata such as
        # run_id and simulator_version may differ while scientific content
        # must stay identical.
        artifact.pop("run_id")
        fresh.pop("run_id")
        artifact.pop("simulator_version")
        fresh.pop("simulator_version")
        self.assertEqual(fresh, artifact)


if __name__ == "__main__":
    unittest.main()
