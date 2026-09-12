import json
from pathlib import Path
import shutil
import tempfile
import unittest

from mfg_hedge.common_state import Phase
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.token_online import TokenObservation
from mfg_hedge.token_best_response import (
    CandidateTable,
    CandidateTableError,
    ConditionalTokenCandidatePolicy,
    load_t3a_runtime_candidate,
)


PARENT_VALIDATION = Path("artifacts/token-t3a-oracle-20260907-r1-validation")
PARENT_OCCUPANCY = Path("artifacts/token-t3a-oracle-20260907-r1-occupancy")
BIN = "D|R|0|[100,200)|[0,1.5)|[1,3)|[2.8125,+inf)"
PARENT = "a" * 64


def _observation(*, queue_load=0, balance=3.0):
    return TokenObservation(
        token_id=4,
        token_class=TokenClass.REGULAR,
        arrival_time=120.0,
        phase=Phase.DEGRADED,
        phase_age=0.5,
        primary_replica=0,
        queue_snapshot=(tuple((100 + i, 0) for i in range(queue_load)), ()),
        running_attempts=(),
        observed_history=(),
        reservation_window=(100.0, 125.0),
        reservation_cap=2.8125,
        reservation_balance=balance,
        public_price=0.0,
    )


def _rows(*, tie=False):
    means = {"N": 1.0, "D": 1.0 if tie else 0.5, "I": 1.0}
    return [
        {
            "bin_id": BIN,
            "requested_action": action,
            "mean": value,
            "standard_error": 0.1,
            "effective_n": 32,
        }
        for action, value in means.items()
    ]


class CandidateTableTests(unittest.TestCase):
    def test_argmin_and_ndI_tie_break(self):
        table = CandidateTable.from_rows(
            _rows(), retained_bins=(BIN,), parent_fingerprint=PARENT
        )
        self.assertEqual(table.action_for_bin(BIN), ProtectionAction.DELAYED_HEDGE)
        tied = CandidateTable.from_rows(
            _rows(tie=True), retained_bins=(BIN,), parent_fingerprint=PARENT
        )
        self.assertEqual(tied.action_for_bin(BIN), ProtectionAction.NORMAL)

    def test_unknown_bin_falls_back_to_normal_and_policy_is_fresh(self):
        table = CandidateTable.from_rows(
            _rows(), retained_bins=(BIN,), parent_fingerprint=PARENT
        )
        policy = ConditionalTokenCandidatePolicy(table)
        episode_policy = policy.new_episode()
        self.assertIsNot(policy, episode_policy)
        self.assertEqual(
            episode_policy.choose(_observation(queue_load=1), "token:4"),
            ProtectionAction.DELAYED_HEDGE,
        )
        unknown = _observation(queue_load=20)
        self.assertEqual(
            episode_policy.choose(unknown, "token:4"), ProtectionAction.NORMAL
        )

    def test_candidate_table_rejects_duplicate_nonfinite_and_missing_bins(self):
        with self.assertRaises(CandidateTableError):
            CandidateTable.from_rows(
                _rows() + [_rows()[0]],
                retained_bins=(BIN,),
                parent_fingerprint=PARENT,
            )
        bad = _rows()
        bad[0] = dict(bad[0], mean=float("nan"))
        with self.assertRaises(CandidateTableError):
            CandidateTable.from_rows(
                bad, retained_bins=(BIN,), parent_fingerprint=PARENT
            )
        with self.assertRaises(CandidateTableError):
            CandidateTable.from_rows(
                _rows(), retained_bins=(BIN, "missing"), parent_fingerprint=PARENT
            )

    def test_loader_rejects_parent_fingerprint_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            validation = root / "validation"
            occupancy = root / "occupancy"
            shutil.copytree(PARENT_VALIDATION, validation)
            shutil.copytree(PARENT_OCCUPANCY, occupancy)
            manifest_path = validation / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["validation_fingerprint"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(CandidateTableError):
                load_t3a_runtime_candidate(validation, occupancy)

    def test_loader_keeps_all_runtime_rows_and_parent_identity(self):
        table = load_t3a_runtime_candidate(PARENT_VALIDATION, PARENT_OCCUPANCY)
        self.assertEqual(len(table.retained_bins), 9)
        self.assertEqual(len(table.rows), 27)
        self.assertEqual(len(table.parent_fingerprint), 64)
        self.assertTrue(
            set(table.action_for_bin(bin_id) for bin_id in table.retained_bins)
            <= set(ProtectionAction)
        )


if __name__ == "__main__":
    unittest.main()
