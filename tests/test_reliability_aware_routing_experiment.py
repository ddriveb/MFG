import unittest

from mfg_hedge.reliability_aware_routing_experiment import (
    BOOTSTRAP_REPLICATES,
    MAX_SCHEDULER_CALLS,
    ROUND1_CALLS,
    ROUND3_CALLS,
    CandidateConfig,
    build_round1_plan,
    compute_paired_max_t,
    select_development_candidate,
    validate_complete_panel,
)


class ReliabilityAwareRoutingExperimentTests(unittest.TestCase):
    def test_frozen_call_arithmetic_and_candidate_grid(self):
        plan = build_round1_plan()
        self.assertEqual(len(plan), 4 * 256 * 14)
        self.assertEqual(ROUND1_CALLS, 14336)
        self.assertEqual(ROUND3_CALLS, 24576)
        self.assertEqual(BOOTSTRAP_REPLICATES, 4096)
        self.assertLess(24 + ROUND1_CALLS + 14336 + ROUND3_CALLS, MAX_SCHEDULER_CALLS)
        candidates = {task.arm for task in plan if task.is_candidate}
        self.assertEqual(len(candidates), 9)

    def test_candidate_selection_applies_s0_guardrail_and_tie_break(self):
        configs = (
            CandidateConfig(25, 2.0, 3.0),
            CandidateConfig(50, 4.0, 6.0),
        )
        rows = []
        for scenario in ("S0", "S1", "S2", "S3"):
            rows.append({
                "arm": "loew", "scenario": scenario, "episode": 0,
                "metrics": {"overall_mean": 1.0, "overall_cvar95": 10.0,
                             "urgent_deadline_miss_rate": 0.2, "lost_work": 3.0},
                "complete": True,
            })
        for config, s0_mean, s0_cvar, s23_cvar, urgent, lost in (
            (configs[0], 1.00, 10.00, 8.0, 0.20, 3.0),
            (configs[1], 1.01, 10.10, 7.0, 0.25, 4.0),
        ):
            for scenario in ("S0", "S1", "S2", "S3"):
                cvar = s0_cvar if scenario == "S0" else s23_cvar
                mean = s0_mean if scenario == "S0" else 1.0
                rows.append({
                    "arm": config.arm_name,
                    "candidate_config": config,
                    "scenario": scenario,
                    "episode": 0,
                    "metrics": {
                        "overall_mean": mean,
                        "overall_cvar95": cvar,
                        "urgent_deadline_miss_rate": urgent,
                        "lost_work": lost,
                    },
                    "complete": True,
                })
        selected = select_development_candidate(rows, configs, baseline_loew_arm="loew")
        self.assertEqual(selected.status, "development_candidate")
        self.assertEqual(selected.config, configs[1])

    def test_incomplete_panel_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_complete_panel((
                {"scenario": "S0", "episode": 0, "arm": "jsq", "complete": True},
            ), required_arms=("jsq", "loew"), episode_count=1, scenario_count=1)

    def test_max_t_is_paired_deterministic_and_reports_simultaneous_fields(self):
        rows = []
        for scenario in ("S0", "S1", "S2", "S3"):
            for episode in range(8):
                base = float(episode + (1 if scenario == "S3" else 0))
                rows.extend((
                    {"scenario": scenario, "episode": episode, "arm": "candidate", "metrics": {"metric": base}},
                    {"scenario": scenario, "episode": episode, "arm": "jsq", "metrics": {"metric": base + 1.0}},
                    {"scenario": scenario, "episode": episode, "arm": "loew", "metrics": {"metric": base + 2.0}},
                ))
        left = compute_paired_max_t(
            rows, candidate_arm="candidate", comparison_arms=("jsq", "loew"),
            bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=20260912,
        )
        right = compute_paired_max_t(
            rows, candidate_arm="candidate", comparison_arms=("jsq", "loew"),
            bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=20260912,
        )
        self.assertEqual(left, right)
        self.assertEqual(left["bootstrap_replicates"], 4096)
        self.assertIn("simultaneous_95_ci", left["contrasts"]["jsq"])
        self.assertLess(left["contrasts"]["jsq"]["point"], 0.0)


if __name__ == "__main__":
    unittest.main()
