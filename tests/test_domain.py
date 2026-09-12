import unittest

from mfg_hedge.domain import ActionStats, CommonState, ProtectionAction, TokenClass


class DomainTests(unittest.TestCase):
    def test_enums_have_stable_wire_values(self) -> None:
        self.assertEqual(CommonState.DEGRADED.value, "D")
        self.assertEqual(TokenClass.URGENT.value, "U")
        self.assertEqual(ProtectionAction.IMMEDIATE_HEDGE.value, "I")

    def test_extra_work_includes_hedge_and_replay(self) -> None:
        stats = ActionStats(
            mean_latency=1.0,
            replay_probability=0.1,
            deadline_miss_probability=0.05,
            expected_hedge_work=0.25,
            expected_replay_work=0.1,
            expected_wasted_work=0.2,
        )
        self.assertAlmostEqual(stats.expected_extra_work, 0.35)
        self.assertAlmostEqual(stats.expected_wasted_work, 0.2)

    def test_invalid_wasted_work_is_rejected(self) -> None:
        for bad in (-0.1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"expected_wasted_work={bad!r}"):
                ActionStats(
                    mean_latency=1.0,
                    replay_probability=0.0,
                    deadline_miss_probability=0.0,
                    expected_hedge_work=0.0,
                    expected_replay_work=0.0,
                    expected_wasted_work=bad,
                )

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 1"):
            ActionStats(
                mean_latency=1.0,
                replay_probability=1.1,
                deadline_miss_probability=0.0,
                expected_hedge_work=0.0,
                expected_replay_work=0.0,
            )


if __name__ == "__main__":
    unittest.main()
