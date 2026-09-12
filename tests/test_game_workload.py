"""Red/green tests for Stage 2 random noise, nested traces and observations."""

from dataclasses import replace
import math
import unittest

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.domain import CommonState, TokenClass
from mfg_hedge.shared_backup import (
    ActionObservation,
    SharedAction,
    SharedBackupTrace,
    SharedTokenSpec,
    SharedWorkDraw,
    simulate_shared_backup,
)
from mfg_hedge.game_workload import (
    CommonFaultIdentity,
    CommonFaultPath,
    build_common_fault_path,
    build_nested_population,
    build_population_replicates,
    enumerate_action_rules,
    public_fault_observation,
)


N, D, S, X = (SharedAction.NORMAL, SharedAction.DELAYED,
              SharedAction.SINGLE, SharedAction.DUAL)
R, U = TokenClass.REGULAR, TokenClass.URGENT


def draw(a0=(1.0, 1.0, 1.0), a1=(1.0, 1.0, 1.0),
         a2=(1.0, 1.0, 1.0), a3=(1.0, 1.0, 1.0)):
    return SharedWorkDraw(tuple(a0), tuple(a1), tuple(a2), tuple(a3))


def fixed_trace(rows, *, timeline=None, experts=1, cutoff=360.0):
    timeline = timeline or CommonStateTimeline(10.0, 20.0, 220.0)
    local = [0] * experts
    tokens = []
    for global_id, row in enumerate(rows):
        arrival, token_class, expert, destination = row
        tokens.append(SharedTokenSpec(
            global_id, float(arrival), token_class, expert, local[expert], destination,
        ))
        local[expert] += 1
    return SharedBackupTrace(
        expert_count=experts,
        timeline=timeline,
        arrival_cutoff=cutoff,
        tokens=tuple(tokens),
        work=tuple(draw() for _ in tokens),
    )


class RecordingRule:
    def __init__(self, fn):
        self.fn = fn
        self.observations = []

    def request(self, observation):
        self.observations.append(observation)
        return self.fn(observation)


class CommonFaultTests(unittest.TestCase):
    def test_stochastic_fault_key_ranges_and_common_identity(self):
        first = build_common_fault_path("stage2:test", 123, 7)
        same = build_common_fault_path("stage2:test", 123, 7, expert_count=16,
                                       population_id=99)
        changed_episode = build_common_fault_path("stage2:test", 123, 8)
        self.assertEqual(first, same)
        self.assertNotEqual(first.fingerprint, changed_episode.fingerprint)
        self.assertGreaterEqual(first.degraded_duration, 75.0)
        self.assertLessEqual(first.degraded_duration, 125.0)
        self.assertGreaterEqual(first.failed_duration, 10.0)
        self.assertLessEqual(first.failed_duration, 30.0)
        self.assertEqual(first.timeline.degraded_start, 100.0)
        self.assertEqual(
            first.timeline.failed_start, 100.0 + first.degraded_duration
        )
        self.assertEqual(
            first.timeline.recovered_start,
            100.0 + first.degraded_duration + first.failed_duration,
        )

    def test_fault_boundaries_are_half_open_and_inputs_fail_before_simulation(self):
        path = build_common_fault_path(
            "stage2:test", 1, 1, degraded_duration=75.0, failed_duration=10.0,
        )
        self.assertIs(public_fault_observation(path, 99.999, 0.0).health,
                      CommonState.HEALTHY)
        self.assertIs(public_fault_observation(path, 100.0, 0.0).health,
                      CommonState.DEGRADED)
        self.assertIs(public_fault_observation(path, 175.0, 0.0).health,
                      CommonState.FAILED)
        self.assertIs(public_fault_observation(path, 185.0, 0.0).phase,
                      Phase.RECOVERED)
        invalid = (
            {"degraded_duration": math.nan},
            {"degraded_duration": math.inf},
            {"failed_duration": 0.0},
            {"degraded_duration": True},
            {"failed_duration": "10"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_common_fault_path("stage2:test", 1, 1, **kwargs)

    def test_public_fault_observation_exposes_only_observed_history(self):
        path = build_common_fault_path(
            "stage2:test", 1, 1, degraded_duration=100.0, failed_duration=20.0,
        )
        early = public_fault_observation(path, 149.999, 0.5)
        late = public_fault_observation(path, 150.0, 0.5)
        self.assertEqual(early.phase, Phase.DEGRADED)
        self.assertLess(early.phase_age, 50.0)
        self.assertGreaterEqual(late.phase_age, 50.0)
        for forbidden in (
            "degraded_duration", "failed_duration", "failed_start",
            "recovered_start", "timeline", "trace", "future_events",
        ):
            self.assertFalse(hasattr(early, forbidden), forbidden)
        self.assertEqual(early.transitions, (("D", 100.0),))
        self.assertEqual(late.transitions, (("D", 100.0),))


class NestedTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fault = build_common_fault_path("stage2:nested", 20260905, 3)

    def test_nested_n8_n16_preserves_first_eight_local_streams(self):
        n8 = build_nested_population(self.fault, "pop-a", 8)
        n16 = build_nested_population(self.fault, "pop-a", 16)
        self.assertEqual(n8.experts, n16.experts[:8])
        self.assertEqual(n8.common_fault_fingerprint, n16.common_fault_fingerprint)
        self.assertEqual(n8.to_shared_trace().expert_count, 8)
        self.assertEqual(n16.to_shared_trace().expert_count, 16)
        for population in (n8, n16):
            ids = [token.global_token_id for token in population.to_shared_trace().tokens]
            self.assertEqual(ids, list(range(len(ids))))
            self.assertEqual(
                list(population.to_shared_trace().tokens),
                sorted(population.to_shared_trace().tokens,
                       key=lambda token: (token.arrival_time, token.expert_id,
                                          token.local_token_id)),
            )

    def test_population_id_changes_only_idiosyncratic_streams(self):
        first = build_nested_population(self.fault, "pop-a", 8)
        second = build_nested_population(self.fault, "pop-b", 8)
        self.assertEqual(first.fault_path, second.fault_path)
        self.assertEqual(first.common_fault_fingerprint, second.common_fault_fingerprint)
        self.assertNotEqual(first.local_fingerprint, second.local_fingerprint)
        self.assertNotEqual(first.experts[0], second.experts[0])
        self.assertEqual(first.audit_identity["fault_fingerprint"],
                         second.audit_identity["fault_fingerprint"])
        self.assertNotEqual(first.audit_identity["population_id"],
                            second.audit_identity["population_id"])

    def test_nested_rebuild_is_byte_identical_and_global_ids_do_not_drive_draws(self):
        first = build_nested_population(self.fault, "pop-a", 8)
        again = build_nested_population(self.fault, "pop-a", 8)
        self.assertEqual(first, again)
        self.assertEqual(first.trace_fingerprint, again.trace_fingerprint)
        self.assertEqual(
            first.local_stream_key(0, 0, 2, 3),
            ("stage2:nested", 20260905, 3, "pop-a", 0, 0, 2, 3),
        )
        for expert in first.experts:
            for token, work in zip(expert.tokens, expert.work):
                self.assertTrue(0.0 <= token.arrival_time < 360.0)
                for attempt in (work.attempt0, work.attempt1, work.attempt2, work.attempt3):
                    self.assertTrue(all(math.isfinite(value) and value > 0 for value in attempt))
        self.assertEqual(
            tuple((token.local_token_id, token.destination_replica)
                  for token in first.experts[0].tokens),
            tuple((token.local_token_id, token.destination_replica)
                  for token in again.experts[0].tokens),
        )

    def test_multiple_populations_reuse_one_common_path(self):
        populations = build_population_replicates(self.fault, ("p0", "p1"), 8)
        self.assertEqual(len(populations), 2)
        self.assertEqual(populations[0].fault_path, populations[1].fault_path)
        self.assertNotEqual(populations[0].local_fingerprint,
                            populations[1].local_fingerprint)
        self.assertNotIn("timestamp", populations[0].audit_identity)


class RuleAndCausalityTests(unittest.TestCase):
    def test_all_256_rules_are_unique_stable_and_include_nnnn_nssn(self):
        rules = enumerate_action_rules()
        self.assertEqual(len(rules), 256)
        names = [rule.name for rule in rules]
        self.assertEqual(len(set(names)), 256)
        self.assertEqual(names[0], "NNNN")
        self.assertEqual(names[1], "NNND")
        self.assertIn("NSSN", names)
        self.assertEqual(names, [rule.name for rule in enumerate_action_rules()])

    def test_online_rule_uses_only_current_position_and_primary_a(self):
        rule = next(rule for rule in enumerate_action_rules() if rule.name == "DSXN")
        timeline = CommonStateTimeline(100.0, 200.0, 220.0)
        source = RecordingRule(rule.request)
        trace = fixed_trace(
            [(110.0, R, 0, 1), (111.0, U, 0, 1),
             (160.0, R, 0, 1), (161.0, U, 0, 1),
             (210.0, R, 0, 1), (230.0, R, 0, 1)],
            timeline=timeline,
        )
        result = simulate_shared_backup(trace, 2.0, action_sources={0: source})
        self.assertEqual(
            [d.requested for d in result.action_decisions], [D, S, X, N, N, N]
        )
        self.assertEqual(
            [d.applied for d in result.action_decisions], [D, S, X, N, N, N]
        )
        self.assertEqual(result.action_decisions[2].charge, 2.0)
        self.assertEqual(result.action_decisions[2].applied, X)
        self.assertEqual(source.observations[0].public_fault.phase_age, 10.0)
        self.assertEqual(source.observations[2].public_fault.phase_age, 60.0)
        self.assertEqual(source.observations[4].common_state, CommonState.FAILED)
        self.assertEqual(source.observations[5].phase, Phase.RECOVERED)

    def test_malicious_source_cannot_read_future_or_true_work(self):
        probes = []

        class Malicious:
            def request(self, observation: ActionObservation):
                probes.append({
                    "fields": tuple(sorted(observation.__dataclass_fields__)),
                    "public_fields": tuple(sorted(observation.public_fault.__dataclass_fields__)),
                    "values": tuple(
                        hasattr(observation, field)
                        for field in ("failed_start", "recovered_start", "timeline",
                                      "trace", "service_requirement", "remaining_work",
                                      "future_event", "future_arrivals")
                    ),
                })
                return N

        result = simulate_shared_backup(
            fixed_trace([(110.0, R, 0, 1)], timeline=CommonStateTimeline(100, 200, 220)),
            0.5, action_sources={0: Malicious()},
        )
        self.assertEqual(result.completed_tokens, 1)
        self.assertEqual(probes[0]["values"], (False,) * 8)
        for forbidden in ("timeline", "trace", "service_requirement", "remaining_work",
                          "future_event", "future_arrivals"):
            self.assertNotIn(forbidden, probes[0]["fields"])
            self.assertNotIn(forbidden, probes[0]["public_fields"])

    def test_future_fault_duration_does_not_change_prior_observation_action_or_budget(self):
        short = build_common_fault_path(
            "stage2:prefix", 4, 1, degraded_duration=75.0, failed_duration=10.0,
        )
        long = build_common_fault_path(
            "stage2:prefix", 4, 2, degraded_duration=125.0, failed_duration=30.0,
        )
        long = replace(long, identity=short.identity)
        rows = [(110.0, R, 0, 1), (140.0, U, 0, 1)]
        base = fixed_trace(rows, timeline=short.timeline)
        alternate = replace(base, timeline=long.timeline)
        rule = next(rule for rule in enumerate_action_rules() if rule.name == "DDDD")
        source_a, source_b = RecordingRule(rule.request), RecordingRule(rule.request)
        first = simulate_shared_backup(base, 2.0, action_sources={0: source_a})
        second = simulate_shared_backup(alternate, 2.0, action_sources={0: source_b})
        self.assertEqual(first.action_decisions, second.action_decisions)
        self.assertEqual(source_a.observations, source_b.observations)
        self.assertEqual(
            [(token.timer_scheduled, token.timer_fired, token.timer_voided)
             for token in first.tokens],
            [(token.timer_scheduled, token.timer_fired, token.timer_voided)
             for token in second.tokens],
        )

    def test_early_failed_window_does_not_refund_or_prorate(self):
        fault = build_common_fault_path(
            "stage2:budget", 4, 1, degraded_duration=75.0, failed_duration=10.0,
        )
        source = RecordingRule(lambda observation: S)
        result = simulate_shared_backup(
            fixed_trace([(110.0, R, 0, 1), (111.0, R, 0, 1)], timeline=fault.timeline),
            2.0, action_sources={0: source},
        )
        decisions = result.action_decisions
        self.assertEqual([row.applied for row in decisions], [S, S])
        self.assertEqual(decisions[0].cap, 2.8125)
        self.assertEqual(decisions[-1].balance_after, .8125)


class Stage2IntegrationTests(unittest.TestCase):
    def test_stochastic_nested_population_runs_online_and_drains_after_cutoff(self):
        fault = build_common_fault_path("stage2:integration", 5, 1)
        population = build_nested_population(fault, "p0", 2)
        result = simulate_shared_backup(
            population.to_shared_trace(), 0.5, action_sources={0: RecordingRule(lambda _: N),
                                                                1: RecordingRule(lambda _: N)},
        )
        self.assertTrue(result.tokens)
        self.assertTrue(all(token.arrival_time < 360.0 for token in result.tokens))
        self.assertGreaterEqual(result.drain_end_time, 360.0)
        self.assertEqual(result.completed_tokens, len(population.to_shared_trace().tokens))
        self.assertTrue(all(a.conservation_holds for a in result.queue_audit_events))
        for interval in result.pool_intervals:
            self.assertLessEqual(interval.executed_work, interval.capacity_bound + 1e-9)

    def test_scheduled_diagnostic_stays_stage1_compatible_and_input_is_immutable(self):
        timeline = CommonStateTimeline(100.0, 200.0, 220.0)
        trace = fixed_trace([(101.0, R, 0, 1)], timeline=timeline)
        before = trace
        result = simulate_shared_backup(trace, 2.0)
        self.assertEqual(result.tokens[0].completion_time, 103.0)
        self.assertEqual(trace, before)

    def test_changing_one_population_idio_trace_changes_peer_latency_not_common_path(self):
        fault = build_common_fault_path("stage2:peer", 8, 2,
                                        degraded_duration=100.0, failed_duration=20.0)
        first, second = build_population_replicates(fault, ("p0", "p1"), 2)
        first_result = simulate_shared_backup(first.to_shared_trace(), .5)
        second_result = simulate_shared_backup(second.to_shared_trace(), .5)
        self.assertEqual(first.fault_path, second.fault_path)
        self.assertNotEqual(first.local_fingerprint, second.local_fingerprint)
        self.assertEqual(first_result.counters.token_arrivals,
                         len(first.to_shared_trace().tokens))
        self.assertEqual(second_result.counters.token_arrivals,
                         len(second.to_shared_trace().tokens))


if __name__ == "__main__":
    unittest.main()
