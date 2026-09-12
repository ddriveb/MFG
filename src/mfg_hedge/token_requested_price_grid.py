"""Zero-call requested-Reservation-price diagnostic on ADR-0027 panels."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_run_directory


EVIDENCE_LABEL = "token_requested_reservation_price_fixed_grid"
COST_MODEL_ID = "token_runtime_requested_reservation_price_v1"
PARENT_RUN_ID = "token-refined-fixed-grid-20260908-r1"
RUN_ID = "token-requested-reservation-price-grid-20260908-r1"
BETAS = (1.0, 2.0, 4.0)
PRICES = (0.0, 0.5, 1.0, 2.0, 4.0)
ETA_REFERENCE = 0.2


def _softmax(costs: Mapping[str, float], beta: float) -> dict[str, float]:
    logits = {action: -beta * costs[action] for action in ("N", "D", "I")}
    peak = max(logits.values())
    weights = {action: math.exp(value - peak) for action, value in logits.items()}
    total = math.fsum(weights.values())
    return {action: weights[action] / total for action in ("N", "D", "I")}


def _validate_grid(grid: Mapping[str, object]) -> None:
    if not isinstance(grid, Mapping):
        raise ValueError("physical grid must be a mapping")
    supported = grid.get("supported_bins")
    stats = grid.get("bin_stats")
    if not isinstance(supported, list) or not supported or not isinstance(stats, Mapping):
        raise ValueError("physical grid has no supported bin statistics")
    if any(bin_id not in stats for bin_id in supported):
        raise ValueError("physical grid is missing supported-bin statistics")


def summarize_requested_price_grid(
    physical_grid: Mapping[str, object],
    *,
    betas: Sequence[float] = BETAS,
    prices: Sequence[float] = PRICES,
) -> dict[str, object]:
    _validate_grid(physical_grid)
    supported = tuple(physical_grid["supported_bins"])
    source_stats = physical_grid["bin_stats"]
    panel_total = sum(int(source_stats[bin_id]["panel_count"]) for bin_id in supported)
    if panel_total <= 0:
        raise ValueError("supported panel total must be positive")
    bin_stats = {}
    for bin_id in supported:
        source = source_stats[bin_id]
        actions = {}
        for action in ("N", "D", "I"):
            row = source["actions"][action]
            actions[action] = {
                "unpriced_cost": float(row["unpriced_cost"]),
                "requested_reservation_units": 0.0 if action == "N" else 1.0,
                "applied_hedge_fraction": float(row["applied_hedge_fraction"]),
            }
        bin_stats[bin_id] = {
            "panel_count": int(source["panel_count"]),
            "actions": actions,
            "baseline_probabilities": dict(source["baseline_probabilities"]),
            "baseline_unpriced_cost": float(source["baseline_unpriced_cost"]),
        }

    cells = []
    for beta_value in betas:
        if isinstance(beta_value, bool) or not isinstance(beta_value, (int, float)):
            raise ValueError("beta must be a finite positive real")
        beta = float(beta_value)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a finite positive real")
        for price_value in prices:
            if isinstance(price_value, bool) or not isinstance(price_value, (int, float)):
                raise ValueError("price must be a finite non-negative real")
            price = float(price_value)
            if not math.isfinite(price) or price < 0.0:
                raise ValueError("price must be a finite non-negative real")
            aggregate_probabilities = {action: 0.0 for action in ("N", "D", "I")}
            priced_expected = 0.0
            unpriced_expected = 0.0
            applied_probability = 0.0
            niin_cost = 0.0
            response_l1 = 0.0
            priced_argmin = 0.0
            for bin_id in supported:
                stats = bin_stats[bin_id]
                weight = stats["panel_count"] / panel_total
                costs = {
                    action: stats["actions"][action]["unpriced_cost"]
                    + price * stats["actions"][action]["requested_reservation_units"]
                    for action in ("N", "D", "I")
                }
                response = _softmax(costs, beta)
                for action in ("N", "D", "I"):
                    probability = response[action]
                    aggregate_probabilities[action] += weight * probability
                    priced_expected += weight * probability * costs[action]
                    unpriced_expected += (
                        weight * probability * stats["actions"][action]["unpriced_cost"]
                    )
                    applied_probability += (
                        weight
                        * probability
                        * stats["actions"][action]["applied_hedge_fraction"]
                    )
                    response_l1 += weight * abs(
                        probability - stats["baseline_probabilities"][action]
                    )
                niin_cost += weight * stats["baseline_unpriced_cost"]
                priced_argmin += weight * min(costs.values())
            requested_probability = (
                aggregate_probabilities["D"] + aggregate_probabilities["I"]
            )
            cells.append({
                "beta": beta,
                "requested_reservation_price": price,
                "action_probabilities": aggregate_probabilities,
                "requested_hedge_probability": requested_probability,
                "expected_requested_reservation_units": requested_probability,
                "one_token_counterfactual_applied_hedge_probability": applied_probability,
                "priced_expected_cost": priced_expected,
                "unpriced_expected_cost": unpriced_expected,
                "priced_argmin_cost": priced_argmin,
                "softmax_cost_above_argmin": priced_expected - priced_argmin,
                "niin_unpriced_cost": niin_cost,
                "undamped_response_l1_from_niin": response_l1,
                "hypothetical_damped_update_l1_eta0p2": ETA_REFERENCE * response_l1,
            })
    return {
        "supported_bins": list(supported),
        "unsupported_bins": dict(physical_grid.get("unsupported_bins", {})),
        "supported_panel_count": panel_total,
        "physical_panel_count": int(physical_grid.get("physical_panel_count", panel_total)),
        "supported_coverage_fraction": float(physical_grid.get("supported_coverage_fraction", 1.0)),
        "bin_stats": bin_stats,
        "cells": cells,
    }


def result_envelope(grid: Mapping[str, object], *, parent_sha256: str) -> dict[str, object]:
    if not isinstance(parent_sha256, str) or len(parent_sha256) != 64:
        raise ValueError("parent_sha256 must be a SHA-256 hex digest")
    return {
        "evidence_label": EVIDENCE_LABEL,
        "execution_profile": "retrospective_conditional_diagnostic",
        "status": "completed",
        "cost_model_id": COST_MODEL_ID,
        "cost_formula": "C_runtime_adr0009(p_exec=0)+p_req*Q_requested",
        "requested_unit": "1 iff eligible pre-projection request is D or I",
        "parent_run_id": PARENT_RUN_ID,
        "parent_summary_sha256": parent_sha256,
        "scheduler_calls": 0,
        "grid": dict(grid),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_requested_price_grid(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str = RUN_ID,
) -> Path:
    from .token_t3a_execution import build_source_bundle

    project = Path(project_root).resolve()
    parent = project / "artifacts" / PARENT_RUN_ID / "summary.json"
    parent_bytes = parent.read_bytes()
    parent_payload = json.loads(parent_bytes)
    grid = summarize_requested_price_grid(parent_payload["grid"])
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    summary = result_envelope(grid, parent_sha256=parent_sha256)
    source = Path(__file__).resolve()
    manifest = {
        "schema_id": "token_requested_reservation_price_grid_manifest_v1",
        "run_id": run_id,
        "evidence_label": EVIDENCE_LABEL,
        "cost_model_id": COST_MODEL_ID,
        "parent_run_id": PARENT_RUN_ID,
        "parent_summary_sha256": parent_sha256,
        "source_bundle": asdict(build_source_bundle(project)),
        "additional_source_sha256": {source.name: _sha256(source)},
        "betas": list(BETAS),
        "prices": list(PRICES),
        "scheduler_calls": 0,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    return write_run_directory(
        artifacts_root,
        run_id,
        {"manifest.json": manifest, "summary.json": summary},
    )


__all__ = [
    "execute_requested_price_grid",
    "result_envelope",
    "summarize_requested_price_grid",
]
