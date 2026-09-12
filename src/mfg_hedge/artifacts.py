"""Artifact writing that never overwrites a previous run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _validate_run_id(run_id: str) -> None:
    """Require ``run_id`` to be a single safe directory name."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            f"run_id must be a single directory name matching "
            f"{_RUN_ID_PATTERN.pattern!r}, got {run_id!r}"
        )
    if run_id.endswith("."):
        raise ValueError(f"run_id must not end with a dot, got {run_id!r}")
    windows_stem = run_id.partition(".")[0].upper()
    if windows_stem in _WINDOWS_RESERVED_STEMS:
        raise ValueError(
            f"run_id uses reserved Windows device name {windows_stem!r}: {run_id!r}"
        )


def write_summary_json(
    artifacts_root: str | Path,
    run_id: str,
    summary: Mapping[str, Any],
) -> Path:
    """Write ``summary.json`` under a fresh ``<artifacts_root>/<run_id>/``.

    Raises ``ValueError`` if ``run_id`` is not a single safe directory name or
    the resolved run directory would land outside ``artifacts_root``. Raises
    ``FileExistsError`` if the run directory already exists, so a previous run
    is never overwritten. The payload is serialized and written to a temporary
    file, then atomically moved into place; if anything fails after the run
    directory is created, the partial directory is removed again.
    """
    _validate_run_id(run_id)
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError(
            f"run directory {run_dir} escapes artifacts root {root}"
        )
    run_dir.mkdir()
    try:
        payload = (
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary_path = run_dir / "summary.json.tmp"
        temporary_path.write_text(payload, encoding="utf-8")
        final_path = run_dir / "summary.json"
        os.replace(temporary_path, final_path)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return final_path


def write_run_directory(
    artifacts_root: str | Path,
    run_id: str,
    files: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Transactionally commit a multi-file run directory.

    Every payload is serialized and written into a staging sibling directory;
    only when all files are complete is the staging directory moved to the
    final run directory in one rename. Any failure removes the staging
    directory entirely, and an existing run directory is never overwritten.
    """
    _validate_run_id(run_id)
    if not files:
        raise ValueError("files must not be empty")
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise ValueError(f"run directory {run_dir} escapes artifacts root {root}")
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    staging = root / f".{run_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for name, payload in files.items():
            if not name.endswith(".json") or "/" in name or "\\" in name:
                raise ValueError(f"invalid artifact file name: {name!r}")
            text = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            temporary = staging / f"{name}.tmp"
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, staging / name)
        os.replace(staging, run_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_dir
