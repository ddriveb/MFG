"""Crash-aware, deterministic work journal for long qualification runs.

Committed units are reusable only under identical plan/source/input
fingerprints.  A unit that was reserved but not atomically committed is an
explicit unresolved attempt and fails closed; the journal never silently
retries it or discards already completed work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCHEMA = "qualification_checkpoint_v2"


class QualificationCheckpointError(RuntimeError):
    pass


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_fp(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QualificationCheckpointError(f"{name} must be a SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise QualificationCheckpointError(
            f"{name} must be a SHA-256 fingerprint"
        ) from error
    return value


class QualificationCheckpointStore:
    """Parent-owned atomic snapshot journal with canonical work identities."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        *,
        plan_fingerprint: str,
        source_fingerprint: str,
    ) -> None:
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            raise QualificationCheckpointError("checkpoint run_id is unsafe")
        self.plan_fingerprint = _check_fp(plan_fingerprint, "plan fingerprint")
        self.source_fingerprint = _check_fp(source_fingerprint, "source fingerprint")
        base = Path(root).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.run_dir = (base / run_id).resolve()
        if self.run_dir.parent != base:
            raise QualificationCheckpointError("checkpoint path escapes its root")
        self.run_dir.mkdir(exist_ok=True)
        self.path = self.run_dir / "checkpoint.json"
        self.blob_dir = self.run_dir / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        if self.path.exists():
            self._state = self._read()
            if self._state["plan_fingerprint"] != self.plan_fingerprint:
                raise QualificationCheckpointError("checkpoint plan fingerprint mismatch")
            if self._state["source_fingerprint"] != self.source_fingerprint:
                raise QualificationCheckpointError("checkpoint source fingerprint mismatch")
        else:
            self._state = {
                "schema": _SCHEMA,
                "plan_fingerprint": self.plan_fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "records": {},
            }
            self._write()

    def _read(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as error:
            raise QualificationCheckpointError("checkpoint cannot be decoded") from error
        if state.get("schema") != _SCHEMA or not isinstance(state.get("records"), dict):
            raise QualificationCheckpointError("checkpoint schema is invalid")
        return state

    def _write(self) -> None:
        payload = json.dumps(
            self._state, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ) + "\n"
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def put_blob(self, payload: bytes) -> str:
        """Store an immutable content-addressed payload beside the journal."""

        if not isinstance(payload, bytes):
            raise QualificationCheckpointError("blob payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        path = self.blob_dir / digest
        if path.exists():
            if path.read_bytes() != payload:
                raise QualificationCheckpointError("content-addressed blob collision")
            return digest
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return digest

    def read_blob(self, digest: str) -> bytes:
        if not isinstance(digest, str) or len(digest) != 64:
            raise QualificationCheckpointError("blob fingerprint is invalid")
        try:
            int(digest, 16)
        except ValueError as error:
            raise QualificationCheckpointError("blob fingerprint is invalid") from error
        path = self.blob_dir / digest
        if not path.is_file():
            raise QualificationCheckpointError("content-addressed blob is missing")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise QualificationCheckpointError("content-addressed blob fingerprint mismatch")
        return payload

    def put_json_blob(self, value: Mapping[str, Any]) -> str:
        try:
            payload = json.dumps(
                dict(value), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise QualificationCheckpointError("blob output is not canonical JSON") from error
        return self.put_blob(payload)

    def read_json_blob(self, digest: str) -> Mapping[str, Any]:
        try:
            value = json.loads(self.read_blob(digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QualificationCheckpointError("content-addressed JSON blob is invalid") from error
        if not isinstance(value, dict):
            raise QualificationCheckpointError("content-addressed JSON blob must be an object")
        return value

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._state)

    @property
    def unresolved_keys(self) -> tuple[str, ...]:
        return tuple(sorted(
            key for key, row in self._state["records"].items()
            if row["status"] == "reserved"
        ))

    @property
    def pause_safe(self) -> bool:
        return not self.unresolved_keys

    @property
    def reserved_calls(self) -> int:
        return sum(row["reserved_calls"] for row in self._state["records"].values())

    @property
    def completed_calls(self) -> int:
        return sum(
            row["completed_calls"] for row in self._state["records"].values()
            if row["status"] == "complete"
        )

    def reserve(
        self,
        work_key: str,
        *,
        input_fingerprint: str,
        expected_calls: int,
    ) -> bool:
        if not isinstance(work_key, str) or not work_key:
            raise QualificationCheckpointError("work key must be non-empty")
        input_fingerprint = _check_fp(input_fingerprint, "input fingerprint")
        if type(expected_calls) is not int or expected_calls < 1:
            raise QualificationCheckpointError("expected_calls must be a positive true int")
        existing = self._state["records"].get(work_key)
        if existing is not None:
            if existing["input_fingerprint"] != input_fingerprint:
                raise QualificationCheckpointError("work input fingerprint mismatch")
            if existing["expected_calls"] != expected_calls:
                raise QualificationCheckpointError("work call reservation mismatch")
            if existing["status"] == "complete":
                return False
            raise QualificationCheckpointError(
                "work has an unresolved prior dispatch; automatic retry is forbidden"
            )
        self._state["records"][work_key] = {
            "status": "reserved",
            "plan_fingerprint": self.plan_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "input_fingerprint": input_fingerprint,
            "expected_calls": expected_calls,
            "reserved_calls": expected_calls,
            "completed_calls": 0,
            "output_fingerprint": None,
            "output": None,
        }
        self._write()
        return True

    def commit(
        self,
        work_key: str,
        *,
        input_fingerprint: str,
        completed_calls: int,
        output: Mapping[str, Any],
    ) -> None:
        input_fingerprint = _check_fp(input_fingerprint, "input fingerprint")
        if type(completed_calls) is not int or completed_calls < 0:
            raise QualificationCheckpointError("completed_calls must be a true int")
        row = self._state["records"].get(work_key)
        if row is None or row["status"] != "reserved":
            raise QualificationCheckpointError("work was not reserved or is already complete")
        if (
            row.get("plan_fingerprint") != self.plan_fingerprint
            or row.get("source_fingerprint") != self.source_fingerprint
        ):
            raise QualificationCheckpointError("work provenance fingerprint mismatch")
        if row["input_fingerprint"] != input_fingerprint:
            raise QualificationCheckpointError("work input fingerprint mismatch")
        if completed_calls > row["expected_calls"]:
            raise QualificationCheckpointError("work exceeded its call reservation")
        canonical_output = json.loads(json.dumps(
            dict(output), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
        row.update({
            "status": "complete",
            "completed_calls": completed_calls,
            "output_fingerprint": _fingerprint(canonical_output),
            "output": canonical_output,
        })
        self._write()

    def completed_output(
        self, work_key: str, *, input_fingerprint: str,
    ) -> Mapping[str, Any] | None:
        input_fingerprint = _check_fp(input_fingerprint, "input fingerprint")
        row = self._state["records"].get(work_key)
        if row is None:
            return None
        if (
            row.get("plan_fingerprint") != self.plan_fingerprint
            or row.get("source_fingerprint") != self.source_fingerprint
        ):
            raise QualificationCheckpointError("work provenance fingerprint mismatch")
        if row["input_fingerprint"] != input_fingerprint:
            raise QualificationCheckpointError("work input fingerprint mismatch")
        if row["status"] != "complete":
            raise QualificationCheckpointError("work has not completed")
        if _fingerprint(row["output"]) != row["output_fingerprint"]:
            raise QualificationCheckpointError("checkpoint output fingerprint mismatch")
        return json.loads(json.dumps(row["output"], sort_keys=True))

    def completed_call_count(
        self, work_key: str, *, input_fingerprint: str,
    ) -> int:
        """Return persisted completed calls without dispatching the unit again."""

        input_fingerprint = _check_fp(input_fingerprint, "input fingerprint")
        row = self._state["records"].get(work_key)
        if row is None or row["status"] != "complete":
            raise QualificationCheckpointError("work has not completed")
        if (
            row.get("plan_fingerprint") != self.plan_fingerprint
            or row.get("source_fingerprint") != self.source_fingerprint
        ):
            raise QualificationCheckpointError("work provenance fingerprint mismatch")
        if row["input_fingerprint"] != input_fingerprint:
            raise QualificationCheckpointError("work input fingerprint mismatch")
        return int(row["completed_calls"])


__all__ = ["QualificationCheckpointError", "QualificationCheckpointStore"]
