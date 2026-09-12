"""T3B candidate construction from the sealed T3A runtime oracle.

This module is deliberately a finite conditional-candidate layer.  It does
not claim an expected best response, regret, Nash equilibrium, or MFG result.
The parent T3A files are treated as sealed evidence and are validated for
internal consistency before their runtime rows are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .domain import ProtectionAction
from .token_continuation import BIN_SCHEMA_V1
from .token_online import TokenObservation


RUNTIME_MODEL_ID = "token_runtime_adr0009_v1"
CANDIDATE_LABEL = "conditional_best_response_candidate"
T3A_EVIDENCE_LABEL = "observable_bin_conditional_cost_oracle"
_ACTION_ORDER = {
    ProtectionAction.NORMAL: 0,
    ProtectionAction.DELAYED_HEDGE: 1,
    ProtectionAction.IMMEDIATE_HEDGE: 2,
}
_EXPECTED_ACTIONS = tuple(_ACTION_ORDER)
_EXPECTED_OCCUPANCY_RUN = "token-t3a-oracle-20260907-r1-occupancy"
_EXPECTED_VALIDATION_RUN = "token-t3a-oracle-20260907-r1-validation"


class CandidateTableError(ValueError):
    """A malformed parent artifact or candidate table."""


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CandidateTableError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateTableError(f"cannot read JSON artifact {path}: {error}") from error


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateTableError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise CandidateTableError(f"{name} must be finite")
    return result


def _false_claims(payload: Mapping[str, object], where: str) -> None:
    for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg"):
        if payload.get(name) is not False:
            raise CandidateTableError(f"{where}.{name} must be false")


def _source_bundle_fingerprint(bundle: Mapping[str, object]) -> str:
    if not isinstance(bundle, Mapping):
        raise CandidateTableError("source_bundle must be an object")
    required = {"schema_id", "roots", "entries", "fingerprint"}
    if set(bundle) != required:
        raise CandidateTableError("source_bundle schema is incomplete")
    payload = {
        "schema_id": bundle["schema_id"],
        "roots": bundle["roots"],
        "entries": bundle["entries"],
    }
    expected = _digest(payload)
    if bundle["fingerprint"] != expected:
        raise CandidateTableError("source_bundle fingerprint is inconsistent")
    return expected


@dataclass(frozen=True)
class CandidateEstimate:
    bin_id: str
    requested_action: ProtectionAction
    mean: float
    standard_error: float
    effective_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.bin_id, str) or not self.bin_id:
            raise CandidateTableError("candidate estimate bin_id must be non-empty")
        if not isinstance(self.requested_action, ProtectionAction):
            raise CandidateTableError("candidate estimate action is invalid")
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))
        object.__setattr__(
            self, "standard_error", _finite(self.standard_error, "standard_error", nonnegative=True)
        )
        if type(self.effective_n) is not int or self.effective_n <= 0:
            raise CandidateTableError("effective_n must be a positive int")

    def to_dict(self) -> dict[str, object]:
        return {
            "bin_id": self.bin_id,
            "requested_action": self.requested_action.value,
            "mean": self.mean,
            "standard_error": self.standard_error,
            "effective_n": self.effective_n,
        }


@dataclass(frozen=True)
class CandidateTable:
    """Immutable runtime-model estimates and their deterministic action map."""

    retained_bins: tuple[str, ...]
    rows: tuple[CandidateEstimate, ...]
    selected_actions: tuple[tuple[str, ProtectionAction], ...]
    parent_fingerprint: str
    model_id: str = RUNTIME_MODEL_ID
    label: str = CANDIDATE_LABEL

    def __post_init__(self) -> None:
        if self.model_id != RUNTIME_MODEL_ID or self.label != CANDIDATE_LABEL:
            raise CandidateTableError("candidate table provenance is not T3B runtime")
        if not self.retained_bins or tuple(sorted(set(self.retained_bins))) != self.retained_bins:
            raise CandidateTableError("retained_bins must be sorted and unique")
        if not isinstance(self.parent_fingerprint, str) or len(self.parent_fingerprint) != 64:
            raise CandidateTableError("parent_fingerprint must be a SHA-256 hex string")
        expected = {(row.bin_id, row.requested_action) for row in self.rows}
        required = {(bin_id, action) for bin_id in self.retained_bins for action in _EXPECTED_ACTIONS}
        if expected != required or len(expected) != len(self.rows):
            raise CandidateTableError("candidate rows must cover each retained bin and N/D/I exactly once")
        choices = dict(self.selected_actions)
        if tuple(sorted(choices)) != self.retained_bins or any(
            not isinstance(action, ProtectionAction) for action in choices.values()
        ):
            raise CandidateTableError("selected_actions must cover retained bins")

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        retained_bins: Iterable[str],
        parent_fingerprint: str,
        model_id: str = RUNTIME_MODEL_ID,
    ) -> "CandidateTable":
        bins = tuple(retained_bins)
        if not bins or tuple(sorted(set(bins))) != bins:
            raise CandidateTableError("retained_bins must be sorted and unique")
        parsed: list[CandidateEstimate] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise CandidateTableError("candidate row must be an object")
            try:
                action = ProtectionAction(raw["requested_action"])
                parsed.append(
                    CandidateEstimate(
                        str(raw["bin_id"]),
                        action,
                        raw["mean"],
                        raw["standard_error"],
                        raw["effective_n"],
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise CandidateTableError(f"invalid candidate row: {error}") from error
        if len({(row.bin_id, row.requested_action) for row in parsed}) != len(parsed):
            raise CandidateTableError("candidate rows contain duplicate bin/action keys")
        by_bin: dict[str, list[CandidateEstimate]] = {bin_id: [] for bin_id in bins}
        for row in parsed:
            if row.bin_id not in by_bin:
                raise CandidateTableError("candidate row refers to an unretained bin")
            by_bin[row.bin_id].append(row)
        selected: list[tuple[str, ProtectionAction]] = []
        for bin_id in bins:
            rows_for_bin = by_bin[bin_id]
            if {row.requested_action for row in rows_for_bin} != set(_EXPECTED_ACTIONS):
                raise CandidateTableError(f"candidate bin {bin_id!r} lacks a complete N/D/I panel")
            winner = min(
                rows_for_bin,
                key=lambda row: (row.mean, _ACTION_ORDER[row.requested_action]),
            )
            selected.append((bin_id, winner.requested_action))
        return cls(bins, tuple(parsed), tuple(selected), parent_fingerprint, model_id=model_id)

    def action_for_bin(self, bin_id: str) -> ProtectionAction:
        if not isinstance(bin_id, str):
            raise CandidateTableError("bin_id must be a string")
        return dict(self.selected_actions).get(bin_id, ProtectionAction.NORMAL)

    def action_for_observation(self, observation: TokenObservation) -> ProtectionAction:
        if not isinstance(observation, TokenObservation):
            raise CandidateTableError("candidate policy requires TokenObservation")
        try:
            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception:
            return ProtectionAction.NORMAL
        return self.action_for_bin(bin_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "model_id": self.model_id,
            "bin_schema_id": BIN_SCHEMA_V1.schema_id,
            "bin_schema_fingerprint": BIN_SCHEMA_V1.fingerprint,
            "retained_bins": list(self.retained_bins),
            "rows": [row.to_dict() for row in self.rows],
            "selected_actions": [
                {"bin_id": bin_id, "requested_action": action.value}
                for bin_id, action in self.selected_actions
            ],
            "parent_fingerprint": self.parent_fingerprint,
        }


class ConditionalTokenCandidatePolicy:
    """Causal candidate policy driven only by the current observation."""

    def __init__(self, table: CandidateTable):
        if not isinstance(table, CandidateTable):
            raise CandidateTableError("table must be CandidateTable")
        self.table = table

    def new_episode(self) -> "ConditionalTokenCandidatePolicy":
        return ConditionalTokenCandidatePolicy(self.table)

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise CandidateTableError("policy_key must be a non-empty string")
        return self.table.action_for_observation(observation)


def _validate_parent_artifacts(validation_dir: Path, occupancy_dir: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    validation_manifest = _read_json(validation_dir / "manifest.json")
    occupancy_manifest = _read_json(occupancy_dir / "manifest.json")
    occupancy = _read_json(occupancy_dir / "occupancy.json")
    validation_summary = _read_json(validation_dir / "summary.json")
    estimates_payload = _read_json(validation_dir / "estimates.json")
    episodes_payload = _read_json(validation_dir / "episodes.json")
    for name, payload in (
        ("validation manifest", validation_manifest),
        ("occupancy manifest", occupancy_manifest),
        ("occupancy", occupancy),
        ("validation summary", validation_summary),
        ("estimates", estimates_payload),
        ("episodes", episodes_payload),
    ):
        if not isinstance(payload, Mapping):
            raise CandidateTableError(f"{name} must be an object")
    if validation_manifest.get("schema_id") != "token_t3a_validation_manifest_v1":
        raise CandidateTableError("unexpected T3A validation manifest schema")
    if occupancy_manifest.get("schema_id") != "token_t3a_occupancy_manifest_v1":
        raise CandidateTableError("unexpected T3A occupancy manifest schema")
    if validation_manifest.get("run_id") != _EXPECTED_VALIDATION_RUN or occupancy_manifest.get("run_id") != _EXPECTED_OCCUPANCY_RUN:
        raise CandidateTableError("unexpected T3A parent run id")
    if validation_manifest.get("occupancy_run_id") != _EXPECTED_OCCUPANCY_RUN:
        raise CandidateTableError("validation does not name the sealed occupancy run")
    _false_claims(validation_summary, "validation summary")
    if validation_summary.get("evidence_label") != T3A_EVIDENCE_LABEL or validation_summary.get("status") != "completed":
        raise CandidateTableError("T3A validation is not completed oracle evidence")
    if occupancy.get("status") != "completed" or occupancy.get("evidence_label") != "t3a_occupancy_calibration":
        raise CandidateTableError("T3A occupancy is not completed")
    _false_claims(occupancy, "occupancy")
    if _digest(occupancy) != occupancy_manifest.get("occupancy_fingerprint"):
        raise CandidateTableError("occupancy fingerprint mismatch")
    if validation_manifest.get("occupancy_fingerprint") != occupancy_manifest.get("occupancy_fingerprint"):
        raise CandidateTableError("validation/occupancy fingerprint mismatch")
    validation_rows = estimates_payload.get("rows")
    episode_rows = episodes_payload.get("rows")
    if not isinstance(validation_rows, list) or not isinstance(episode_rows, list):
        raise CandidateTableError("T3A row payloads must be arrays")
    provenance = validation_manifest.get("provenance")
    occupancy_provenance = occupancy_manifest.get("provenance")
    if provenance != occupancy_provenance or not isinstance(provenance, Mapping):
        raise CandidateTableError("T3A provenance differs between parent artifacts")
    source_bundle = validation_manifest.get("source_bundle")
    if not isinstance(source_bundle, Mapping) or _source_bundle_fingerprint(source_bundle) != provenance.get("source_bundle_fingerprint"):
        raise CandidateTableError("T3A source bundle provenance is inconsistent")
    if validation_manifest.get("source_bundle") != occupancy_manifest.get("source_bundle"):
        raise CandidateTableError("T3A source bundles differ")
    if provenance.get("bin_schema_fingerprint") != BIN_SCHEMA_V1.fingerprint:
        raise CandidateTableError("parent bin schema is incompatible with runtime schema")
    if len(validation_rows) != 81 or len(episode_rows) != int(validation_summary.get("episode_count", -1)):
        raise CandidateTableError("T3A parent row counts are not frozen complete counts")
    validation_payload = {
        "evidence_label": T3A_EVIDENCE_LABEL,
        "status": validation_summary.get("status"),
        "namespace": "token-mfg-restoration:t3a:oracle-validation:v1",
        "macro_seed": 20260910,
        "protocol_id": "oracle-validation_v1",
        "episode_count": validation_summary.get("episode_count"),
        "attempted_calls": validation_summary.get("attempted_calls"),
        "panel_floor": 32,
        "retained_bins": occupancy.get("retained_bins"),
        "provenance": provenance,
        "occupancy_fingerprint": occupancy_manifest.get("occupancy_fingerprint"),
        "episodes": episode_rows,
        "estimates": validation_rows,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    if _digest(validation_payload) != validation_manifest.get("validation_fingerprint"):
        raise CandidateTableError("validation fingerprint mismatch")
    retained = occupancy.get("retained_bins")
    if not isinstance(retained, list) or len(retained) != 9 or tuple(sorted(set(retained))) != tuple(retained):
        raise CandidateTableError("T3A retained-bin support is not the frozen nine-bin set")
    return dict(validation_manifest), dict(occupancy_manifest), [dict(row) for row in validation_rows]


def load_t3a_runtime_candidate(
    validation_dir: str | Path,
    occupancy_dir: str | Path,
) -> CandidateTable:
    """Load only the sealed T3A runtime rows needed for candidate construction."""

    validation_manifest, occupancy_manifest, rows = _validate_parent_artifacts(
        Path(validation_dir), Path(occupancy_dir)
    )
    runtime_rows = [
        row
        for row in rows
        if row.get("model_id") == RUNTIME_MODEL_ID
    ]
    retained = tuple(
        sorted(
            set(
                str(row["bin_id"])
                for row in rows
                if isinstance(row.get("bin_id"), str)
            )
        )
    )
    if len(runtime_rows) != 27 or len(retained) != 9:
        raise CandidateTableError("runtime parent rows do not cover 9 bins x N/D/I")
    for row in rows:
        if row.get("status") != "completed":
            raise CandidateTableError("T3A parent contains a non-completed estimate row")
        if row.get("quote_value") != 0.0 or not isinstance(row.get("quote_basis"), str):
            raise CandidateTableError("T3A quote provenance is incomplete")
        if row.get("model_id") not in {
            "token_initial_price_v0",
            RUNTIME_MODEL_ID,
            "token_extended_reservation_v1",
        }:
            raise CandidateTableError("T3A contains an unknown cost model")
        _finite(row.get("mean"), "parent mean")
        _finite(row.get("standard_error"), "parent standard_error", nonnegative=True)
        if type(row.get("effective_n")) is not int or row["effective_n"] < 32:
            raise CandidateTableError("T3A parent effective_n is below the frozen floor")
    model_fingerprints = {
        row["model_id"]: row.get("model_fingerprint") for row in rows
    }
    if len(model_fingerprints) != 3 or any(
        not isinstance(value, str) or len(value) != 64 for value in model_fingerprints.values()
    ):
        raise CandidateTableError("T3A model fingerprints are incomplete")
    runtime_fingerprint = validation_manifest["provenance"]["source_bundle_fingerprint"]
    # The parent validation fingerprint, not the current source, is the table identity.
    return CandidateTable.from_rows(
        runtime_rows,
        retained_bins=retained,
        parent_fingerprint=validation_manifest["validation_fingerprint"],
    )


__all__ = [
    "CANDIDATE_LABEL",
    "CandidateEstimate",
    "CandidateTable",
    "CandidateTableError",
    "ConditionalTokenCandidatePolicy",
    "RUNTIME_MODEL_ID",
    "load_t3a_runtime_candidate",
]
