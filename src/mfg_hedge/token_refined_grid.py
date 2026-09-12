"""Refined observation and fixed price-temperature Token diagnostic.

One physical N/D/I panel bank is reused for every descriptive grid cell.  The
module does not update a population policy and never claims BR, Nash, or MFG.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_run_directory
from .common_state import Phase
from .domain import ProtectionAction
from .token_online import TokenObservation, simulate_episode_online


EVIDENCE_LABEL = "token_refined_observation_fixed_price_temperature_grid"
SCHEMA_ID = "token_observation_refined_v1"
SELECTOR_ID = "random_threshold_four_degraded_strata_v1"
NAMESPACE = "token-mfg-restoration:t4e:refined-fixed-grid:v1"
MACRO_SEED = 20260916
EPISODE_COUNT = 1024
PANEL_FLOOR = 8
CALL_LIMIT = 5120
BETAS = (1.0, 2.0, 4.0)
PRICES = (0.0, 0.5, 1.0)
ETA_REFERENCE = 0.2
RUN_ID = "token-refined-fixed-grid-20260908-r1"
POPULATION_POLICY_ID = "pi_NIIN_v1"

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)
_ACTION_ORDER = {action.value: index for index, action in enumerate(_ACTIONS)}
_AGE_STRATA = ((0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bucket(value: float, bounds: tuple[float, ...]) -> str:
    for left, right in zip(bounds, bounds[1:]):
        if left <= value < right:
            return f"[{left:g},{right:g})"
    if value >= bounds[-1]:
        return f"[{bounds[-1]:g},+inf)"
    raise ValueError("observable value is below the first boundary")


def _replica_live_load(observation: TokenObservation, replica_id: int) -> int:
    queued = len(observation.queue_snapshot[replica_id])
    running = sum(item[2] == replica_id for item in observation.running_attempts)
    return queued + running


def refined_bin_id(observation: TokenObservation) -> str:
    if not isinstance(observation, TokenObservation):
        raise ValueError("observation must be TokenObservation")
    if observation.phase is not Phase.DEGRADED or observation.primary_replica != 0:
        raise ValueError("refined pilot bins only degraded Primary-A decisions")
    primary = _replica_live_load(observation, observation.primary_replica)
    backup = _replica_live_load(observation, 1 - observation.primary_replica)
    return "|".join((
        "D",
        observation.token_class.value,
        f"age={_bucket(observation.phase_age, (0.0, 25.0, 50.0, 75.0))}",
        f"primary={_bucket(float(primary), (0.0, 3.0))}",
        f"backup={_bucket(float(backup), (0.0, 3.0))}",
        f"balance={_bucket(observation.reservation_balance, (0.0, 1.0, 2.8125))}",
    ))


@dataclass(frozen=True)
class RandomThresholdSelector:
    stratum: int
    threshold: float

    def __post_init__(self) -> None:
        if type(self.stratum) is not int or not 0 <= self.stratum < 4:
            raise ValueError("stratum must be an int in 0..3")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise ValueError("threshold must be a real number")
        lower, upper = _AGE_STRATA[self.stratum]
        if not math.isfinite(float(self.threshold)) or not lower <= self.threshold < upper:
            raise ValueError("threshold must lie inside its stratum")

    @classmethod
    def for_episode(cls, episode_index: int) -> "RandomThresholdSelector":
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError("episode_index must be a non-negative int")
        stratum = episode_index % 4
        lower, upper = _AGE_STRATA[stratum]
        raw = hashlib.sha256(
            f"{NAMESPACE}:{MACRO_SEED}:{episode_index}:target-threshold".encode()
        ).digest()
        draw = int.from_bytes(raw[:8], "big") / float(1 << 64)
        return cls(stratum, lower + (upper - lower) * draw)

    def select(self, observation: TokenObservation) -> bool:
        if not isinstance(observation, TokenObservation):
            raise ValueError("selector requires TokenObservation")
        lower, upper = _AGE_STRATA[self.stratum]
        return (
            observation.phase is Phase.DEGRADED
            and observation.primary_replica == 0
            and lower <= observation.phase_age < upper
            and observation.phase_age >= self.threshold
        )


class _CaptureSource:
    def __init__(self, source: object, selector: RandomThresholdSelector):
        self.source = source
        self.selector = selector
        self.target_id = None
        self.observation = None

    def choose(self, observation: TokenObservation, policy_key: str) -> ProtectionAction:
        if self.target_id is None and self.selector.select(observation):
            self.target_id = observation.token_id
            self.observation = observation
        return self.source.choose(observation, policy_key)


def _softmax(costs: Mapping[str, float], beta: float) -> dict[str, float]:
    logits = {action: -beta * costs[action] for action in ("N", "D", "I")}
    peak = max(logits.values())
    weights = {action: math.exp(value - peak) for action, value in logits.items()}
    total = math.fsum(weights.values())
    return {action: weights[action] / total for action in ("N", "D", "I")}


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def summarize_fixed_grid(
    panel_rows: Sequence[Mapping[str, object]],
    *,
    betas: Sequence[float] = BETAS,
    prices: Sequence[float] = PRICES,
    panel_floor: int = PANEL_FLOOR,
) -> dict[str, object]:
    if type(panel_floor) is not int or panel_floor <= 0:
        raise ValueError("panel_floor must be positive")
    rows = tuple(panel_rows)
    grouped: dict[tuple[int, str], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        episode = row.get("episode_index")
        bin_id = row.get("bin_id")
        action = row.get("action")
        applied_action = row.get("applied_action")
        if (
            type(episode) is not int
            or not isinstance(bin_id, str)
            or action not in _ACTION_ORDER
            or applied_action not in _ACTION_ORDER
        ):
            raise ValueError("invalid panel row identity")
        key = (episode, bin_id)
        if action in grouped.setdefault(key, {}):
            raise ValueError("duplicate panel action row")
        grouped[key][str(action)] = row
    if any(set(actions) != {"N", "D", "I"} for actions in grouped.values()):
        raise ValueError("each physical panel must contain N/D/I")

    by_bin: dict[str, list[dict[str, Mapping[str, object]]]] = {}
    for (_, bin_id), actions in grouped.items():
        by_bin.setdefault(bin_id, []).append(actions)
    supported = tuple(sorted(bin_id for bin_id, values in by_bin.items() if len(values) >= panel_floor))
    unsupported = {
        bin_id: len(values) for bin_id, values in sorted(by_bin.items()) if len(values) < panel_floor
    }
    supported_panels = sum(len(by_bin[bin_id]) for bin_id in supported)
    total_panels = len(grouped)
    bin_stats = {}
    for bin_id in supported:
        panels = by_bin[bin_id]
        action_stats = {}
        for action in ("N", "D", "I"):
            action_stats[action] = {
                "unpriced_cost": _mean([float(panel[action]["unpriced_cost"]) for panel in panels]),
                "incremental_work": _mean([float(panel[action]["incremental_work"]) for panel in panels]),
                "applied_hedge_fraction": _mean([
                    float(panel[action]["applied_action"] != "N") for panel in panels
                ]),
            }
        baseline_counts = {action: 0 for action in ("N", "D", "I")}
        for panel in panels:
            baseline_counts[str(panel["N"]["baseline_action"])] += 1
        bin_stats[bin_id] = {
            "panel_count": len(panels),
            "actions": action_stats,
            "baseline_probabilities": {
                action: baseline_counts[action] / len(panels) for action in ("N", "D", "I")
            },
            "baseline_unpriced_cost": _mean([
                float(panel["N"]["baseline_unpriced_cost"]) for panel in panels
            ]),
        }

    cells = []
    for beta in betas:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        for price in prices:
            if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)) or price < 0:
                raise ValueError("price must be finite and non-negative")
            action_probabilities = {action: 0.0 for action in ("N", "D", "I")}
            applied_hedge_probability = 0.0
            priced_expected = 0.0
            unpriced_expected = 0.0
            incremental_expected = 0.0
            priced_argmin = 0.0
            response_l1 = 0.0
            niin_expected = 0.0
            for bin_id in supported:
                stats = bin_stats[bin_id]
                weight = stats["panel_count"] / supported_panels
                costs = {
                    action: stats["actions"][action]["unpriced_cost"]
                    + float(price) * stats["actions"][action]["incremental_work"]
                    for action in ("N", "D", "I")
                }
                probabilities = _softmax(costs, float(beta))
                for action in ("N", "D", "I"):
                    action_probabilities[action] += weight * probabilities[action]
                    applied_hedge_probability += (
                        weight
                        * probabilities[action]
                        * stats["actions"][action]["applied_hedge_fraction"]
                    )
                    priced_expected += weight * probabilities[action] * costs[action]
                    unpriced_expected += weight * probabilities[action] * stats["actions"][action]["unpriced_cost"]
                    incremental_expected += weight * probabilities[action] * stats["actions"][action]["incremental_work"]
                    response_l1 += weight * abs(
                        probabilities[action] - stats["baseline_probabilities"][action]
                    )
                priced_argmin += weight * min(costs.values())
                niin_expected += weight * stats["baseline_unpriced_cost"]
            cells.append({
                "beta": float(beta),
                "price": float(price),
                "action_probabilities": action_probabilities,
                "requested_hedge_probability": action_probabilities["D"] + action_probabilities["I"],
                "expected_applied_hedge_probability": applied_hedge_probability,
                "priced_expected_cost": priced_expected,
                "unpriced_expected_cost": unpriced_expected,
                "expected_incremental_work": incremental_expected,
                "priced_argmin_cost": priced_argmin,
                "softmax_cost_above_argmin": priced_expected - priced_argmin,
                "niin_unpriced_cost": niin_expected,
                "undamped_response_l1_from_niin": response_l1,
                "damped_update_l1_eta0p2": ETA_REFERENCE * response_l1,
            })
    return {
        "physical_panel_count": total_panels,
        "supported_panel_count": supported_panels,
        "supported_coverage_fraction": supported_panels / total_panels if total_panels else 0.0,
        "supported_bins": list(supported),
        "unsupported_bins": unsupported,
        "bin_stats": bin_stats,
        "cells": cells,
    }


def result_envelope(
    panel_rows: Sequence[Mapping[str, object]],
    grid_cells: Sequence[Mapping[str, object]],
    *,
    attempted_calls: int,
) -> dict[str, object]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "execution_profile": "exploratory",
        "status": "completed",
        "population_policy_id": POPULATION_POLICY_ID,
        "schema_id": SCHEMA_ID,
        "selector_id": SELECTOR_ID,
        "episode_count": EPISODE_COUNT,
        "panel_floor": PANEL_FLOOR,
        "betas": list(BETAS),
        "prices": list(PRICES),
        "attempted_calls": attempted_calls,
        "call_limit": CALL_LIMIT,
        "physical_panel_row_count": len(panel_rows),
        "grid_cell_count": len(grid_cells),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def run_refined_grid(config: object, *, episode_count: int = EPISODE_COUNT) -> dict[str, object]:
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL, EpisodeIdentity, generate_episode_trace
    from .token_deviations import evaluate_token_pathwise_deviations
    from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
    from .token_t3a_execution import policy_factory, validate_frozen_config
    from .transient_control import ReservationParameters

    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= EPISODE_COUNT:
        raise ValueError("episode_count is outside 1..1024")
    reservation = ReservationParameters(
        window_width=25.0, budget_rate=0.45, scale=0.25, mean_requirement=1.0
    )
    parameters = TokenRuntimeADR0009Parameters()
    quote = ExternalQuote(0.0, "incremental_executed_work")
    attempted = 0
    panel_rows = []
    episode_rows = []
    identity_keys = []
    for index in range(episode_count):
        identity = EpisodeIdentity(NAMESPACE, MACRO_SEED, index)
        identity_keys.append(identity.key)
        episode = generate_episode_trace(
            config, NAMESPACE, MACRO_SEED, index, ATTRIBUTION_V1_PROTOCOL
        )
        selector = RandomThresholdSelector.for_episode(index)
        capture = _CaptureSource(policy_factory(), selector)
        attempted += 1
        try:
            simulate_episode_online(
                episode,
                capture,
                reservation,
                degraded_slowdown=2.0,
                hedge_delay=2.0,
            )
        except Exception as error:
            episode_rows.append({
                "episode_index": index,
                "status": "selection_failed",
                "reason": f"{type(error).__name__}: {error}",
                "attempted_calls": 1,
            })
            continue
        if capture.target_id is None or capture.observation is None:
            episode_rows.append({
                "episode_index": index,
                "status": "missing_target",
                "threshold": selector.threshold,
                "stratum": selector.stratum,
                "attempted_calls": 1,
            })
            continue
        bin_id = refined_bin_id(capture.observation)
        result = evaluate_token_pathwise_deviations(
            episode,
            policy_factory,
            capture.target_id,
            candidates=_ACTIONS,
            parameters=parameters,
            quote=quote,
            reservation=reservation,
            degraded_slowdown=2.0,
            hedge_delay=2.0,
        )
        attempted += result.attempted_calls
        if result.status != "completed" or len(result.rows) != 3:
            episode_rows.append({
                "episode_index": index,
                "status": "panel_failed",
                "bin_id": bin_id,
                "reason": result.failure_reason,
                "attempted_calls": 1 + result.attempted_calls,
            })
            continue
        baseline = result.baseline_payoff
        for row in result.rows:
            payoff = row.candidate_payoff
            panel_rows.append({
                "episode_index": index,
                "target_token_id": capture.target_id,
                "bin_id": bin_id,
                "action": row.candidate_requested.value,
                "applied_action": payoff.applied_action.value,
                "unpriced_cost": payoff.total,
                "incremental_work": payoff.w_hedge + payoff.w_replay,
                "baseline_action": baseline.requested_action.value,
                "baseline_unpriced_cost": baseline.total,
            })
        episode_rows.append({
            "episode_index": index,
            "status": "completed",
            "target_token_id": capture.target_id,
            "bin_id": bin_id,
            "threshold": selector.threshold,
            "stratum": selector.stratum,
            "attempted_calls": 1 + result.attempted_calls,
        })
        if attempted > episode_count * 5:
            raise RuntimeError("refined grid exceeded its physical call budget")
    grid = summarize_fixed_grid(panel_rows)
    envelope = result_envelope(panel_rows, grid["cells"], attempted_calls=attempted)
    envelope.update({
        "episode_count": episode_count,
        "call_limit": episode_count * 5,
        "namespace": NAMESPACE,
        "macro_seed": MACRO_SEED,
        "identity_library_fingerprint": _digest(identity_keys),
        "schema": {
            "phase_age": [0.0, 25.0, 50.0, 75.0],
            "primary_live_load": [0.0, 3.0],
            "backup_live_load": [0.0, 3.0],
            "reservation_balance": [0.0, 1.0, 2.8125],
        },
        "target_selection": {
            "strata": [list(row) for row in _AGE_STRATA],
            "threshold_key": "sha256(namespace:macro_seed:episode_index:target-threshold)",
        },
        "episode_rows": episode_rows,
        "panel_rows": panel_rows,
        "grid": grid,
        "missing_target_count": sum(row["status"] == "missing_target" for row in episode_rows),
        "failed_episode_count": sum(row["status"].endswith("failed") for row in episode_rows),
    })
    return envelope


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_refined_grid(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str = RUN_ID,
) -> Path:
    from .config import load_config
    from .token_t3a_execution import build_source_bundle, environment_fingerprint

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    result = run_refined_grid(config)
    source = Path(__file__).resolve()
    manifest = {
        "schema_id": "token_refined_fixed_grid_manifest_v1",
        "run_id": run_id,
        "evidence_label": EVIDENCE_LABEL,
        "execution_profile": "exploratory",
        "source_bundle": asdict(build_source_bundle(project)),
        "additional_source_sha256": {source.name: _file_sha256(source)},
        "environment_fingerprint": environment_fingerprint(config),
        "identity_library_fingerprint": result["identity_library_fingerprint"],
        "namespace": NAMESPACE,
        "macro_seed": MACRO_SEED,
        "episode_count": EPISODE_COUNT,
        "panel_floor": PANEL_FLOOR,
        "call_limit": CALL_LIMIT,
        "betas": list(BETAS),
        "prices": list(PRICES),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }
    files = {
        "manifest.json": manifest,
        "summary.json": result,
        "panel_rows.json": {"rows": result["panel_rows"]},
        "grid_cells.json": {"rows": result["grid"]["cells"]},
    }
    return write_run_directory(artifacts_root, run_id, files)


__all__ = [
    "BETAS",
    "CALL_LIMIT",
    "EPISODE_COUNT",
    "PANEL_FLOOR",
    "PRICES",
    "RandomThresholdSelector",
    "execute_refined_grid",
    "refined_bin_id",
    "result_envelope",
    "run_refined_grid",
    "summarize_fixed_grid",
]
