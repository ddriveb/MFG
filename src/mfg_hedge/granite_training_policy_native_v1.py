"""In-process C ABI for the frozen Granite replica quota planners.

Compile granite_training_policy_native_v1.cpp once into a DLL or shared
library and pass that path to NativeTrainingPlanner. This module never
builds code, writes per-call files, starts a subprocess, or falls back to
Python planner implementations.
"""
from __future__ import annotations

import ctypes
import math
import operator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence


class NativeTrainingPlannerError(RuntimeError):
    """The native planner library is unavailable or rejected a request."""


POLICY_IDS = {
    "structured": 0,
    "lplb": 1,
    "sinkhorn_relaxed": 2,
}
_MAX_EXACT_ROUTE_TOTAL = (1 << 53) - 1
_DIAGNOSTIC_NAMES = (
    "policy_cpu_us",
    "native_call_cpu_us",
    "row_residual_routes",
    "column_residual_routes",
    "kkt_residual_routes",
    "continuous_fhard_ms",
    "integer_fhard_ms",
    "f_tau_ms",
    "fw_gap_ms",
    "sinkhorn_epsilon",
    "sinkhorn_rho",
    "sinkhorn_theta",
    "sinkhorn_objective",
    "sinkhorn_update_residual",
    "sinkhorn_kkt_score_range",
    "sinkhorn_uniform_column_deviation_routes",
    "sinkhorn_probability_response_deviation",
)


@dataclass(frozen=True)
class NativeTrainingPlan:
    policy: str
    quota_left: list[int]
    shares_left: list[float]
    status: str
    iterations: int
    diagnostics: dict[str, float | int | str | None]


@lru_cache(maxsize=4)
def _load_library(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        raise NativeTrainingPlannerError(f"native training planner library does not exist: {path}")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise NativeTrainingPlannerError(f"could not load native training planner library: {exc}") from exc
    try:
        function = library.granite_training_policy_plan_v1
    except AttributeError as exc:
        raise NativeTrainingPlannerError("native library is missing granite_training_policy_plan_v1") from exc

    i64p = ctypes.POINTER(ctypes.c_int64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f64p = ctypes.POINTER(ctypes.c_double)
    charp = ctypes.POINTER(ctypes.c_char)
    size = ctypes.c_size_t
    function.argtypes = [
        i64p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        size,
        f64p,
        i64p,
        i32p,
        size,
        f64p,
        size,
        charp,
        size,
        charp,
        size,
    ]
    function.restype = ctypes.c_int
    return library, function


def _checked_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise NativeTrainingPlannerError(f"{name} must be an integer, not bool")
    try:
        checked = operator.index(value)
    except TypeError as exc:
        raise NativeTrainingPlannerError(f"{name} must be an integer") from exc
    return int(checked)


class NativeTrainingPlanner:
    """Callable wrapper around one frozen native planner policy."""

    is_native = True

    def __init__(self, library_path: str | Path, policy: str = "structured"):
        if policy not in POLICY_IDS:
            raise NativeTrainingPlannerError(
                f"unknown policy {policy!r}; expected one of {', '.join(POLICY_IDS)}"
            )
        candidate = Path(library_path).expanduser()
        if not candidate.is_file():
            raise NativeTrainingPlannerError(f"native training planner library does not exist: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise NativeTrainingPlannerError(f"native planner library path is unavailable: {library_path}") from exc
        self.library_path = resolved
        self.policy = policy
        self._library, self._function = _load_library(str(resolved))

    def plan(self, counts: Sequence[int], gpus: int) -> NativeTrainingPlan:
        gpu_count = _checked_int(gpus, "gpus")
        if not (gpu_count == 2 or 4 <= gpu_count <= 512):
            raise NativeTrainingPlannerError("gpus must be 2 or in 4..512")
        try:
            raw_counts = tuple(counts)
        except TypeError as exc:
            raise NativeTrainingPlannerError("counts must be a finite sequence of integers") from exc
        if not raw_counts or len(raw_counts) > 4096:
            raise NativeTrainingPlannerError("counts must contain 1..4096 Expert entries")

        checked_counts: list[int] = []
        total = 0
        for expert, raw in enumerate(raw_counts):
            count = _checked_int(raw, f"counts[{expert}]")
            if count < 0:
                raise NativeTrainingPlannerError(f"counts[{expert}] must be nonnegative")
            if count > (1 << 63) - 1:
                raise NativeTrainingPlannerError(f"counts[{expert}] exceeds int64")
            total += count
            if total > _MAX_EXACT_ROUTE_TOTAL:
                raise NativeTrainingPlannerError("route total exceeds exact binary64 integer range")
            checked_counts.append(count)

        n = len(checked_counts)
        native_counts = (ctypes.c_int64 * n)(*checked_counts)
        shares = (ctypes.c_double * n)()
        quotas = (ctypes.c_int64 * n)()
        metadata = (ctypes.c_int32 * 4)()
        diagnostics = (ctypes.c_double * len(_DIAGNOSTIC_NAMES))()
        policy_status = ctypes.create_string_buffer(64)
        error = ctypes.create_string_buffer(512)
        rc = self._function(
            native_counts,
            n,
            gpu_count,
            POLICY_IDS[self.policy],
            n,
            shares,
            quotas,
            metadata,
            len(metadata),
            diagnostics,
            len(diagnostics),
            policy_status,
            len(policy_status),
            error,
            len(error),
        )
        if rc != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise NativeTrainingPlannerError(
                message or f"native planner returned error code {rc}"
            )

        quota_left = [int(value) for value in quotas]
        shares_left = [float(value) for value in shares]
        if any(q < 0 or q > count for q, count in zip(quota_left, checked_counts)):
            raise NativeTrainingPlannerError("native planner returned quota outside Expert demand")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in shares_left):
            raise NativeTrainingPlannerError("native planner returned an invalid left share")
        values = {}
        for index, name in enumerate(_DIAGNOSTIC_NAMES):
            value = float(diagnostics[index])
            values[name] = value if math.isfinite(value) else None
        raw_kkt_residual = values.get("kkt_residual_routes")
        kkt_residual_measured = self.policy == "structured"
        values["native_kkt_residual_routes"] = raw_kkt_residual
        values["kkt_residual_measured"] = kkt_residual_measured
        values["kkt_residual_routes"] = raw_kkt_residual if kkt_residual_measured else None
        if not kkt_residual_measured:
            values["kkt_residual_applicability_note"] = (
                "The frozen shared result field is not populated by this policy; its raw native value is preserved "
                "as native_kkt_residual_routes and must not be read as a convergence certificate."
            )
        values.update(
            iterations=int(metadata[0]),
            integerization_boundary_flags=int(metadata[1]),
            active_tie_guard_count=int(metadata[2]),
            active_component_count=int(metadata[3]),
            native_policy_status=policy_status.value.decode("utf-8", errors="replace"),
        )
        return NativeTrainingPlan(
            policy=self.policy,
            quota_left=quota_left,
            shares_left=shares_left,
            status=values["native_policy_status"],
            iterations=int(metadata[0]),
            diagnostics=values,
        )


__all__ = [
    "NativeTrainingPlan",
    "NativeTrainingPlanner",
    "NativeTrainingPlannerError",
    "POLICY_IDS",
]
