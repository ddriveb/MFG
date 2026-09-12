import json
from pathlib import Path
import pickle
import tempfile
import unittest

from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    R2_DRY_RUN_MACRO_SEED,
    R2_DRY_RUN_NAMESPACE,
)
from mfg_hedge.token_mfg_qualification import FixedEpisodeLibrary
from mfg_hedge.token_mfg_qualification_backend import ConcreteQualificationBackend
from mfg_hedge.token_mfg_qualification_state_machine import _unpack
from mfg_hedge.token_mfg_streaming import (
    StreamingEpisodeSpec,
    build_streaming_panel,
    execute_streaming_episode_panel,
    streaming_formal_episode_worker,
)


class QualificationStreamingTests(unittest.TestCase):
    @staticmethod
    def _branch(plan):
        return {
            "mean_field_namespace": plan.continuation_namespace,
            "mean_field_macro_seed": plan.continuation_macro_seed,
            "finite_k_namespace": plan.finite_k_namespace,
            "finite_k_macro_seed": plan.finite_k_macro_seed,
            "bounded": False,
            "min_complete_panels": 8,
        }

    def test_panel_contains_identities_not_materialized_traces(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=64, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        self.assertEqual(len(panel.specs), 32)
        self.assertTrue(all(isinstance(spec, StreamingEpisodeSpec) for spec in panel.specs))
        self.assertFalse(any(hasattr(spec, "trace") for spec in panel.specs))
        self.assertLess(len(json.dumps(panel.to_dict(), sort_keys=True)), 20_000)
        self.assertEqual(
            tuple((spec.scenario, spec.episode_index) for spec in panel.specs),
            tuple(
                (scenario, episode)
                for scenario in ("S0", "S1", "S2", "S3")
                for episode in range(
                    {"S0": 0, "S1": 8, "S2": 16, "S3": 24}[scenario],
                    {"S0": 8, "S1": 16, "S2": 24, "S3": 32}[scenario],
                )
            ),
        )

    def test_work_keys_separate_k_start_model_and_iteration(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=8, count=1, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        with tempfile.TemporaryDirectory() as root:
            common = {
                "panel": panel,
                "checkpoint_root": root,
                "run_id": "key-separation",
                "plan_fingerprint": plan.protocol_fingerprint,
                "source_fingerprint": "f" * 64,
                "worker_count": 1,
                "phase": "forward_population",
                "model_id": "unpriced_mfg",
                "iteration": 0,
                "branch_payload": self._branch(plan),
            }
            first = execute_streaming_episode_panel(**common, start_id="uniform")
            second = execute_streaming_episode_panel(**common, start_id="loew")
            self.assertEqual(first.dispatched_calls, 1)
            self.assertEqual(second.dispatched_calls, 1)
            checkpoint = json.loads(
                (Path(root) / "key-separation" / "checkpoint.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(len(checkpoint["records"]), 2)

    def test_one_episode_worker_is_spawn_safe_and_bounded(self):
        plan = QualificationPlan.r2()
        spec = build_streaming_panel(
            plan, k=8, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        ).specs[0]
        result = streaming_formal_episode_worker({
            "stage": "forward_population",
            "episode_spec": spec,
            "model_id": "unpriced_mfg",
            "iteration": 0,
            "mean_field_namespace": plan.continuation_namespace,
            "mean_field_macro_seed": plan.continuation_macro_seed,
            "finite_k_namespace": plan.finite_k_namespace,
            "finite_k_macro_seed": plan.finite_k_macro_seed,
            "bounded": False,
            "min_complete_panels": 8,
        })
        self.assertEqual(result["completed_calls"], 1)
        self.assertNotIn("library", result["output"])
        self.assertIn("snapshot", result["output"])

    def test_streamed_panel_checkpoint_is_compact_and_resume_reuses_calls(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=8, count=2, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        with tempfile.TemporaryDirectory() as root:
            first = execute_streaming_episode_panel(
                panel,
                checkpoint_root=root,
                run_id="streamed-panel",
                plan_fingerprint=plan.protocol_fingerprint,
                source_fingerprint="a" * 64,
                worker_count=1,
                phase="forward_population",
                model_id="unpriced_mfg",
                iteration=0,
                branch_payload={
                    "mean_field_namespace": plan.continuation_namespace,
                    "mean_field_macro_seed": plan.continuation_macro_seed,
                    "finite_k_namespace": plan.finite_k_namespace,
                    "finite_k_macro_seed": plan.finite_k_macro_seed,
                    "bounded": False,
                    "min_complete_panels": 8,
                },
            )
            second = execute_streaming_episode_panel(
                panel,
                checkpoint_root=root,
                run_id="streamed-panel",
                plan_fingerprint=plan.protocol_fingerprint,
                source_fingerprint="a" * 64,
                worker_count=8,
                phase="forward_population",
                model_id="unpriced_mfg",
                iteration=0,
                branch_payload={
                    "mean_field_namespace": plan.continuation_namespace,
                    "mean_field_macro_seed": plan.continuation_macro_seed,
                    "finite_k_namespace": plan.finite_k_namespace,
                    "finite_k_macro_seed": plan.finite_k_macro_seed,
                    "bounded": False,
                    "min_complete_panels": 8,
                },
            )
            self.assertEqual(first.to_bytes(), second.to_bytes())
            self.assertEqual(second.reused_calls, 2)
            self.assertEqual(second.dispatched_calls, 0)
            checkpoint = Path(root) / "streamed-panel" / "checkpoint.json"
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            records = state["records"]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(
                set(row["output"]) == {"__content_addressed_blob__"}
                for row in records.values()
            ))
            self.assertEqual(len(list((checkpoint.parent / "blobs").iterdir())), 2)

    def test_one_and_eight_workers_have_identical_science_bytes(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=8, count=2, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        with tempfile.TemporaryDirectory() as root:
            kwargs = {
                "panel": panel,
                "checkpoint_root": root,
                "plan_fingerprint": plan.protocol_fingerprint,
                "source_fingerprint": "b" * 64,
                "phase": "forward_population",
                "model_id": "unpriced_mfg",
                "iteration": 0,
                "branch_payload": self._branch(plan),
            }
            serial = execute_streaming_episode_panel(
                **kwargs, run_id="serial", worker_count=1,
            )
            parallel = execute_streaming_episode_panel(
                **kwargs, run_id="parallel", worker_count=8,
            )
            self.assertEqual(serial.to_bytes(), parallel.to_bytes())
            self.assertEqual(serial.completed_calls, parallel.completed_calls)

    def test_streamed_episode_physics_matches_bounded_full_panel(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=8, count=2, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        traces = tuple(spec.generate_trace() for spec in panel.specs)
        full_backend = ConcreteQualificationBackend(
            FixedEpisodeLibrary.from_traces(traces),
            bounded=False,
            min_complete_panels=2,
        )
        full_snapshot = full_backend._forward(None, "unpriced_mfg", 0)
        full_episodes = full_backend._episodes_by_environment[
            full_snapshot.environment_fingerprint
        ]
        with tempfile.TemporaryDirectory() as root:
            streamed = execute_streaming_episode_panel(
                panel,
                checkpoint_root=root,
                run_id="field-equality",
                plan_fingerprint=plan.protocol_fingerprint,
                source_fingerprint="e" * 64,
                worker_count=1,
                phase="forward_population",
                model_id="unpriced_mfg",
                iteration=0,
                branch_payload=self._branch(plan),
            )
        streamed_episodes = tuple(
            _unpack(output["episodes"])[0]
            for _, output in streamed.outputs
        )
        self.assertEqual(len(streamed_episodes), len(full_episodes))
        self.assertEqual(
            [pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL) for row in streamed_episodes],
            [pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL) for row in full_episodes],
        )

    def test_checkpoint_fingerprint_change_is_rejected_without_dispatch(self):
        plan = QualificationPlan.r2()
        panel = build_streaming_panel(
            plan, k=8, count=1, namespace=R2_DRY_RUN_NAMESPACE,
            macro_seed=R2_DRY_RUN_MACRO_SEED,
        )
        with tempfile.TemporaryDirectory() as root:
            kwargs = {
                "panel": panel,
                "checkpoint_root": root,
                "run_id": "fingerprint-guard",
                "plan_fingerprint": plan.protocol_fingerprint,
                "source_fingerprint": "c" * 64,
                "worker_count": 1,
                "phase": "forward_population",
                "model_id": "unpriced_mfg",
                "iteration": 0,
                "branch_payload": self._branch(plan),
            }
            first = execute_streaming_episode_panel(**kwargs)
            self.assertEqual(first.dispatched_calls, 1)
            with self.assertRaises(Exception):
                execute_streaming_episode_panel(
                    **{**kwargs, "source_fingerprint": "d" * 64}
                )


if __name__ == "__main__":
    unittest.main()
