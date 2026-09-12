import unittest

from mfg_hedge.token_mfg_qualification_backend import RoutingPolicySeed
from mfg_hedge.token_mfg_qualification_campaign import QualificationPlan
from mfg_hedge.token_mfg_streaming import StreamingEpisodeSpec
from mfg_hedge.token_mfg_streaming_formal import (
    MAX_CALLS_PER_EPISODE,
    streaming_iteration_episode_worker,
)


class StreamingFormalQualificationTests(unittest.TestCase):
    def test_one_episode_returns_compact_sufficient_statistics(self):
        plan = QualificationPlan.r2()
        spec = StreamingEpisodeSpec(
            8, plan.preflight_namespace, plan.preflight_macro_seed, "S0", 901,
        )
        result = streaming_iteration_episode_worker({
            "episode_spec": spec,
            "policy": RoutingPolicySeed("uniform"),
            "model_id": "unpriced_mfg",
            "iteration": 0,
            "mean_field_namespace": plan.continuation_namespace,
            "mean_field_macro_seed": plan.continuation_macro_seed,
            "finite_k_namespace": plan.finite_k_namespace,
            "finite_k_macro_seed": plan.finite_k_macro_seed,
        })
        self.assertLessEqual(result["completed_calls"], MAX_CALLS_PER_EPISODE)
        self.assertNotIn("episodes", result["output"])
        self.assertNotIn("trace", result["output"])
        self.assertIn("compact_tokens", result["output"])
        self.assertIn("continuation_rows", result["output"])
        self.assertIn("finite_rows", result["output"])

    def test_singleton_continuation_is_legal_inside_worker(self):
        plan = QualificationPlan.r2()
        spec = StreamingEpisodeSpec(
            8, plan.preflight_namespace, plan.preflight_macro_seed, "S1", 902,
        )
        result = streaming_iteration_episode_worker({
            "episode_spec": spec,
            "policy": RoutingPolicySeed("uniform"),
            "model_id": "priced_mfg",
            "iteration": 0,
            "mean_field_namespace": plan.continuation_namespace,
            "mean_field_macro_seed": plan.continuation_macro_seed,
            "finite_k_namespace": plan.finite_k_namespace,
            "finite_k_macro_seed": plan.finite_k_macro_seed,
        })
        self.assertGreater(result["completed_calls"], 1)


if __name__ == "__main__":
    unittest.main()
