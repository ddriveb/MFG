"""Compile a frozen integer-quota plan into exact SimAI matrix replay input.

This adapter represents one isolated MoE stage. It does not include planner
runtime or claim physical multi-GPU execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


PHASES = (
    "forward_dispatch",
    "forward_combine",
    "backward_combine_grad",
    "backward_dispatch_grad",
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _sequence(value: Any, name: str) -> Sequence:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("source inputs must be finite JSON-compatible data") from exc
    return hashlib.sha256(payload).hexdigest()


def compile_saved_plan_workload(
    plan: Mapping[str, Any],
    rank_measurements: Sequence[Mapping[str, Any]],
    *,
    bytes_per_route: int = 4096,
) -> dict[str, Any]:
    """Compile one saved plan and its per-rank records for ``encode_replay``.

    The returned payload matrices are dense square sender/receiver matrices.
    Their entries are exact integer bytes; source-local routes are excluded.
    Measurements are sorted by rank before profiles and their provenance hash
    are produced, so callers may supply the records in any order.
    """
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be a mapping")
    gpus = _integer(plan.get("gpus"), "plan.gpus", minimum=1)
    if gpus > 512:
        raise ValueError("GPU count exceeds the replay adapter limit of 512")
    experts = _integer(plan.get("experts"), "plan.experts", minimum=1)
    bytes_per_route = _integer(bytes_per_route, "bytes_per_route", minimum=1)
    state_id = plan.get("state_id")
    arm = plan.get("arm")
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("plan.state_id must be a nonempty string")
    if not isinstance(arm, str) or not arm:
        raise ValueError("plan.arm must be a nonempty string")
    seed = _integer(plan.get("seed"), "plan.seed")

    rows = _sequence(plan.get("rows"), "plan.rows")
    quotas = _sequence(plan.get("quotas"), "plan.quotas")
    if len(rows) != len(quotas):
        raise ValueError("plan rows and quota rows must have equal length")

    # Rebuild present (source, expert) batch rows. Ticket59 stores sparse rows;
    # absent zero-demand pairs are valid and do not imply a missing replica.
    seen_rows: set[tuple[int, int]] = set()
    expert_masks: dict[int, tuple[int, ...]] = {}
    rank_routes = [0] * gpus
    expert_rank_routes: dict[int, list[int]] = {}
    matrices = {phase: [[0] * gpus for _ in range(gpus)] for phase in PHASES}
    total_routes = 0
    local_routes = 0
    nonlocal_routes = 0

    for row_index, (row, quota_row) in enumerate(zip(rows, quotas)):
        if not isinstance(row, Mapping):
            raise ValueError(f"plan.rows[{row_index}] must be a mapping")
        source = _integer(row.get("source_gpu"), f"rows[{row_index}].source_gpu")
        expert = _integer(row.get("expert_id"), f"rows[{row_index}].expert_id")
        demand = _integer(row.get("demand"), f"rows[{row_index}].demand", minimum=1)
        if source >= gpus or expert >= experts:
            raise ValueError(f"rows[{row_index}] source or expert is out of range")
        key = (source, expert)
        if key in seen_rows:
            raise ValueError(f"duplicate source-expert row {key}")
        seen_rows.add(key)

        allowed = _sequence(row.get("allowed_gpus"), f"rows[{row_index}].allowed_gpus")
        if not 1 <= len(allowed) <= 4:
            raise ValueError(f"rows[{row_index}].allowed_gpus must contain 1 to 4 ranks")
        mask = tuple(_integer(gpu, f"rows[{row_index}].allowed_gpus entry") for gpu in allowed)
        if len(set(mask)) != len(mask) or any(gpu >= gpus for gpu in mask):
            raise ValueError(f"rows[{row_index}].allowed_gpus contains duplicate or invalid ranks")
        old_mask = expert_masks.setdefault(expert, mask)
        if old_mask != mask:
            raise ValueError(f"Expert {expert} has inconsistent legal replica masks")

        quota_row = _sequence(quota_row, f"plan.quotas[{row_index}]")
        if len(quota_row) != len(mask):
            raise ValueError(f"plan.quotas[{row_index}] does not match allowed_gpus")
        checked_quotas = [
            _integer(value, f"plan.quotas[{row_index}] entry")
            for value in quota_row
        ]
        if sum(checked_quotas) != demand:
            raise ValueError(f"plan.quotas[{row_index}] does not conserve its demand")

        for destination, count in zip(mask, checked_quotas):
            rank_routes[destination] += count
            expert_rank_routes.setdefault(expert, [0] * gpus)[destination] += count
            total_routes += count
            if source == destination:
                local_routes += count
                continue
            nonlocal_routes += count
            size = count * bytes_per_route
            matrices["forward_dispatch"][source][destination] += size
            matrices["backward_combine_grad"][source][destination] += size
            matrices["forward_combine"][destination][source] += size
            matrices["backward_dispatch_grad"][destination][source] += size

    if not rows:
        raise ValueError("plan must contain at least one sparse source-expert row")

    score = plan.get("score")
    if score is not None:
        if not isinstance(score, Mapping):
            raise ValueError("plan.score must be a mapping")
        saved_rank_counts = score.get("rank_counts")
        if saved_rank_counts is not None:
            saved_rank_counts = _sequence(saved_rank_counts, "plan.score.rank_counts")
            normalized_counts = [
                _integer(value, "plan.score.rank_counts entry") for value in saved_rank_counts
            ]
            if normalized_counts != rank_routes:
                raise ValueError("plan score rank totals do not match integer quotas")

    measurements = _sequence(rank_measurements, "rank_measurements")
    if len(measurements) != gpus:
        raise ValueError("one rank measurement is required for each GPU")
    by_rank: dict[int, Mapping[str, Any]] = {}
    plan_ids: set[int] = set()
    missing_plan_id = False
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            raise ValueError(f"rank_measurements[{index}] must be a mapping")
        if measurement.get("state_id") != state_id or measurement.get("arm") != arm:
            raise ValueError("rank measurement state or arm does not match the saved plan")
        measured_gpus = _integer(measurement.get("gpus"), f"rank_measurements[{index}].gpus", minimum=1)
        if measured_gpus != gpus:
            raise ValueError("rank measurement GPU count does not match the saved plan")
        rank = _integer(measurement.get("rank"), f"rank_measurements[{index}].rank")
        if rank >= gpus or rank in by_rank:
            raise ValueError("rank measurements must contain each GPU exactly once")
        by_rank[rank] = measurement
        if "plan_id" in measurement:
            plan_ids.add(_integer(measurement["plan_id"], f"rank_measurements[{index}].plan_id"))
        else:
            missing_plan_id = True
        routes = _integer(measurement.get("routes"), f"rank_measurements[{index}].routes")
        active = _integer(measurement.get("active"), f"rank_measurements[{index}].active")
        small = _integer(measurement.get("small"), f"rank_measurements[{index}].small")
        if routes != rank_routes[rank]:
            raise ValueError(f"rank {rank} route total does not match integer quotas")
        expected_active = sum(counts[rank] > 0 for counts in expert_rank_routes.values())
        # The saved rank-measurement format counts positive expert batches <128 as small.
        expected_small = sum(0 < counts[rank] < 128 for counts in expert_rank_routes.values())
        if active != expected_active or small != expected_small:
            raise ValueError(f"rank {rank} batch layout does not match integer quotas")
        _finite_nonnegative(measurement.get("FW_ms"), f"rank_measurements[{index}].FW_ms")
        _finite_nonnegative(measurement.get("BW_ms"), f"rank_measurements[{index}].BW_ms")
        for optional_time in ("predicted_FW_ms", "predicted_BW_ms"):
            if optional_time in measurement:
                _finite_nonnegative(measurement[optional_time], f"rank_measurements[{index}].{optional_time}")

    if set(by_rank) != set(range(gpus)):
        raise ValueError("rank measurements are missing one or more GPU profiles")
    if len(plan_ids) > 1:
        raise ValueError("rank measurements refer to multiple saved plans")
    if plan_ids and missing_plan_id:
        raise ValueError("rank measurements must all carry the same plan_id")

    ordered_measurements = [by_rank[rank] for rank in range(gpus)]
    profiles = []
    for rank, measurement in enumerate(ordered_measurements):
        profiles.append({
            "gpu": rank,
            "local_profile_forward_ms": _finite_nonnegative(
                measurement["FW_ms"], f"rank {rank} FW_ms"
            ),
            "local_profile_backward_ms": _finite_nonnegative(
                measurement["BW_ms"], f"rank {rank} BW_ms"
            ),
            "FW_ms": float(measurement["FW_ms"]),
            "BW_ms": float(measurement["BW_ms"]),
            "case_id": measurement.get("case_id"),
            "routes": rank_routes[rank],
            "active": sum(counts[rank] > 0 for counts in expert_rank_routes.values()),
            "small": sum(0 < counts[rank] < 128 for counts in expert_rank_routes.values()),
            **({"predicted_FW_ms": float(measurement["predicted_FW_ms"])} if "predicted_FW_ms" in measurement else {}),
            **({"predicted_BW_ms": float(measurement["predicted_BW_ms"])} if "predicted_BW_ms" in measurement else {}),
        })

    plan_hash = _canonical_sha256(plan)
    measurements_hash = _canonical_sha256(ordered_measurements)
    nonlocal_bytes = nonlocal_routes * bytes_per_route
    workload = {
        "gpu_count": gpus,
        "payload_bytes_by_sender_receiver": matrices,
        "existing_local_compute_profiles": profiles,
        "workload_id": f"{state_id}-{arm}",
        "policy_id": arm,
        "step_id": str(seed),
        "state_id": state_id,
        "arm": arm,
        "seed": seed,
        "experts": experts,
        "total_routes": total_routes,
        "local_routes": local_routes,
        "nonlocal_routes": nonlocal_routes,
        "bytes_per_route": bytes_per_route,
        "phase_nonlocal_routes": {phase: nonlocal_routes for phase in PHASES},
        "nonlocal_bytes_by_phase": {phase: nonlocal_bytes for phase in PHASES},
        "nonlocal_bytes_total": nonlocal_bytes * len(PHASES),
        "plan_id": next(iter(plan_ids)) if plan_ids else None,
        "source_plan_sha256": plan_hash,
        "source_rank_measurements_sha256": measurements_hash,
        "source_hashes": {
            "plan_sha256": plan_hash,
            "rank_measurements_sorted_by_rank_sha256": measurements_hash,
        },
        "stage_only": True,
        "planning_excluded": True,
        "shape_prototype": True,
    }
    return workload

