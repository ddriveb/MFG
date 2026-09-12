import tempfile
import unittest
import json
from pathlib import Path

from tests.test_token_mfg_qualification_backend import _library
from mfg_hedge.token_mfg_qualification_backend import ConcreteQualificationBackend
from mfg_hedge.token_mfg_qualification_state_machine import (
    DynamicQualificationStateMachine,
)


class DynamicQualificationStateMachineTests(unittest.TestCase):
    def test_one_and_eight_workers_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as serial_root, tempfile.TemporaryDirectory() as parallel_root:
            serial = DynamicQualificationStateMachine(
                ConcreteQualificationBackend(_library()),
                checkpoint_root=serial_root,
                run_id="bounded-serial",
                worker_count=1,
            ).run_bounded()
            parallel = DynamicQualificationStateMachine(
                ConcreteQualificationBackend(_library()),
                checkpoint_root=parallel_root,
                run_id="bounded-parallel",
                worker_count=8,
            ).run_bounded()

        self.assertEqual(serial.to_bytes(), parallel.to_bytes())
        self.assertEqual(serial.status, "complete")
        self.assertEqual(serial.dispatched_calls, parallel.dispatched_calls)
        self.assertGreater(serial.committed_boundaries, 0)

    def test_interrupt_after_forward_resumes_without_repeating_committed_calls(self):
        with tempfile.TemporaryDirectory() as root:
            first = DynamicQualificationStateMachine(
                ConcreteQualificationBackend(_library()),
                checkpoint_root=root,
                run_id="bounded-resume",
                worker_count=1,
            ).run_bounded(stop_after_phase="forward_population")
            checkpoint_state = json.loads(
                Path(root, "bounded-resume", "checkpoint.json").read_text(
                    encoding="utf-8",
                )
            )
            forward_record = checkpoint_state["records"][
                "bounded/k8/model=unpriced_mfg/iteration=0/phase=forward_population"
            ]
            self.assertEqual(forward_record["status"], "complete")
            self.assertIn("snapshot", forward_record["output"])
            self.assertIn("episodes", forward_record["output"])
            self.assertIn("policy_seed", forward_record["output"])
            resumed = DynamicQualificationStateMachine(
                ConcreteQualificationBackend(_library()),
                checkpoint_root=root,
                run_id="bounded-resume",
                worker_count=8,
            ).run_bounded()
            completed_state = json.loads(
                Path(root, "bounded-resume", "checkpoint.json").read_text(
                    encoding="utf-8",
                )
            )
            continuation_record = completed_state["records"][
                "bounded/k8/model=unpriced_mfg/iteration=0/phase=mean_field_continuation"
            ]
            finite_record = completed_state["records"][
                "bounded/k8/model=unpriced_mfg/iteration=0/phase=finite_k_deviation"
            ]
            confirmation_record = completed_state["records"][
                "bounded/k8/model=unpriced_mfg/iteration=0/phase=final_policy_confirmation"
            ]

        self.assertEqual(first.status, "paused")
        self.assertGreater(first.dispatched_calls, 0)
        self.assertEqual(resumed.status, "complete")
        self.assertGreater(resumed.reused_calls, 0)
        self.assertLess(resumed.dispatched_calls, resumed.scheduler_calls)
        self.assertGreater(resumed.committed_boundaries, first.committed_boundaries)
        self.assertIn("episode_keys", continuation_record["output"])
        self.assertNotIn("policy_after", continuation_record["output"])
        self.assertIn("policy_after", finite_record["output"])
        self.assertEqual(
            finite_record["output"]["episode_keys"],
            continuation_record["output"]["episode_keys"],
        )
        self.assertEqual(
            confirmation_record["output"]["policy"],
            finite_record["output"]["policy_after"],
        )


if __name__ == "__main__":
    unittest.main()
