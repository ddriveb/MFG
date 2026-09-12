import unittest

from mfg_hedge.reliability_aware_routing import (
    RoutingToken,
    audit_lazarus_homogeneous_balance,
    lazarus_batch_assignment,
)


class LazarusFairnessTests(unittest.TestCase):
    def test_small_batch_remainder_rotates_by_batch_identity(self):
        first = (
            RoutingToken(0, 0.0, "Regular", 4.0, 0, 1.0),
            RoutingToken(1, 0.0, "Regular", 4.0, 4, 1.0),
        )
        second = (
            RoutingToken(2, 1.0, "Regular", 4.0, 2, 1.0),
            RoutingToken(3, 1.0, "Regular", 4.0, 6, 1.0),
        )
        old_shape = lazarus_batch_assignment(first, range(8), batch_identity=0)
        rotated = lazarus_batch_assignment(second, range(8), batch_identity=2)
        self.assertEqual(old_shape, {0: 0, 1: 1})
        self.assertEqual(rotated, {2: 2, 3: 3})

    def test_locality_first_is_preserved_inside_rotated_capacity(self):
        tokens = (
            RoutingToken(10, 0.0, "Regular", 4.0, 5, 1.0),
            RoutingToken(11, 0.0, "Regular", 4.0, 7, 1.0),
        )
        assignment = lazarus_batch_assignment(tokens, range(8), batch_identity=5)
        self.assertEqual(assignment[10], 5)
        self.assertEqual(assignment[11], 6)

    def test_homogeneous_audit_has_explicit_one_token_gap_bound(self):
        audit = audit_lazarus_homogeneous_balance()
        self.assertEqual(audit["batch_count"], 512)
        self.assertEqual(audit["max_count_gap_bound"], 1)
        self.assertLessEqual(audit["max_count_gap"], 1)
        self.assertEqual(max(audit["counts"].values()) - min(audit["counts"].values()), 0)


if __name__ == "__main__":
    unittest.main()
