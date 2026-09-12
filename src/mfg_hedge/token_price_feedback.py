"""Bounded round-level price feedback over the refined Token observation.

This is an exploratory finite-population controller.  It deliberately exposes
no best-response, regret, Nash, or MFG claim.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import write_run_directory
from .common_state import Phase
from .domain import ProtectionAction
from .token_mfg_t4c_supported_soft_response import (
    T4CPolicyState,
    aggregate_round_physics_metrics,
    build_identity_library,
    reconstruct_episode,
)
from .token_refined_grid import RandomThresholdSelector, refined_bin_id


EVIDENCE_LABEL = "token_refined_round_price_feedback_exploratory"
NAMESPACE = "token-mfg-restoration:t4d:refined-price-feedback:v1"
MACRO_SEED = 20260917
EPISODE_COUNT = 256
MAX_ROUNDS = 4
PANEL_FLOOR = 4
ALPHA = 0.25
BETA = 4.0
ETA = 0.2
CALL_LIMIT = EPISODE_COUNT * MAX_ROUNDS * 5
RUN_ID = "token-t4d-price-feedback-exploratory-20260908-r1"
REQUESTED_RUN_ID = "token-t4d-requested-price-feedback-exploratory-20260908-r1"

_ACTIONS = (
    ProtectionAction.NORMAL,
    ProtectionAction.DELAYED_HEDGE,
    ProtectionAction.IMMEDIATE_HEDGE,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real")
    clean = float(value)
    if not math.isfinite(clean) or (positive and clean <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return clean


def update_public_price(
    price: float,
    *,
    requested_work: float,
    capacity: float,
    alpha: float = ALPHA,
) -> dict[str, float]:
    price = _finite(price, "price")
    requested = _finite(requested_work, "requested_work")
    capacity = _finite(capacity, "capacity", positive=True)
    alpha = _finite(alpha, "alpha", positive=True)
    if price < 0.0 or requested < 0.0:
        raise ValueError("price and requested_work must be non-negative")
    pressure = (requested - capacity) / capacity
    next_price = max(0.0, price + alpha * pressure)
    return {
        "demand_pressure": pressure,
        "next_price": next_price,
        "price_residual": abs(next_price - price),
    }


def requested_price_cost(
    unpriced_cost: float,
    requested_action: ProtectionAction,
    price: float,
) -> float:
    base = _finite(unpriced_cost, "unpriced_cost")
    quote = _finite(price, "price")
    if quote < 0.0:
        raise ValueError("price must be non-negative")
    if not isinstance(requested_action, ProtectionAction):
        raise ValueError("requested_action must be ProtectionAction")
    return base + (0.0 if requested_action is ProtectionAction.NORMAL else quote)


def _one_hot(action: ProtectionAction) -> tuple[float, float, float]:
    return tuple(1.0 if candidate is action else 0.0 for candidate in _ACTIONS)


def _soft_response(costs: Mapping[ProtectionAction, float], beta: float) -> tuple[float, float, float]:
    logits = tuple(-beta * float(costs[action]) for action in _ACTIONS)
    peak = max(logits)
    weights = tuple(math.exp(value - peak) for value in logits)
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


class RefinedPopulationPolicy:
    """Causal refined-bin policy with an exact NIIN fallback and coupled draws."""

    def __init__(
        self,
        state: T4CPolicyState | None = None,
        *,
        seed: int = MACRO_SEED,
    ) -> None:
        self.state = state if state is not None else T4CPolicyState.initial()
        if not isinstance(self.state, T4CPolicyState):
            raise ValueError("state must be T4CPolicyState")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative int")
        self.seed = seed

    def new_episode(self) -> "RefinedPopulationPolicy":
        return RefinedPopulationPolicy(self.state, seed=self.seed)

    @staticmethod
    def base_action(observation: object) -> ProtectionAction:
        phase = getattr(getattr(observation, "phase", None), "value", None)
        primary = getattr(observation, "primary_replica", None)
        token_class = getattr(getattr(observation, "token_class", None), "value", None)
        age = getattr(observation, "phase_age", None)
        late = isinstance(age, (int, float)) and not isinstance(age, bool) and age >= 50.0
        urgent = token_class == "U"
        if phase == "D" and primary == 0 and late != urgent:
            return ProtectionAction.IMMEDIATE_HEDGE
        return ProtectionAction.NORMAL

    def probabilities_for_observation(self, observation: object) -> tuple[float, float, float]:
        fallback = _one_hot(self.base_action(observation))
        try:
            bin_id = refined_bin_id(observation)
        except Exception:
            return fallback
        return self.state.probabilities_for_bin(bin_id, fallback)

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if not isinstance(policy_key, str) or not policy_key:
            raise ValueError("policy_key must be a non-empty string")
        probabilities = self.probabilities_for_observation(observation)
        raw = hashlib.sha256(f"{self.seed}:{policy_key}".encode()).digest()
        draw = int.from_bytes(raw[:8], "big") / float(1 << 64)
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, probabilities):
            cumulative += probability
            if draw < cumulative or action is _ACTIONS[-1]:
                return action
        raise AssertionError("unreachable action draw")


class _CaptureSource:
    def __init__(self, source: RefinedPopulationPolicy, selector: RandomThresholdSelector):
        self.source = source
        self.selector = selector
        self.target_id = None
        self.observation = None

    def choose(self, observation: object, policy_key: str) -> ProtectionAction:
        if self.target_id is None and self.selector.select(observation):
            self.target_id = observation.token_id
            self.observation = observation
        return self.source.choose(observation, policy_key)


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(abs(a - b) for a, b in zip(left, right))


def _run_round(
    config: object,
    library: object,
    policy: RefinedPopulationPolicy,
    *,
    iteration: int,
    public_price: float,
    panel_floor: int,
    beta: float,
    eta: float,
    price_basis: str,
) -> tuple[dict[str, object], RefinedPopulationPolicy]:
    from .attribution_episode import ATTRIBUTION_V1_PROTOCOL
    from .attribution_metrics import build_episode_metrics
    from .token_deviations import evaluate_token_pathwise_deviations
    from .token_online import simulate_episode_online
    from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
    from .transient_control import ReservationParameters

    reservation = ReservationParameters(
        window_width=25.0, budget_rate=0.45, scale=0.25, mean_requirement=1.0
    )
    quote = ExternalQuote(public_price, "incremental_executed_work")
    parameters = TokenRuntimeADR0009Parameters()
    samples: dict[tuple[str, ProtectionAction], list[float]] = {}
    unpriced_samples: dict[tuple[str, ProtectionAction], list[float]] = {}
    occupancy: dict[str, int] = {}
    fallback_by_bin: dict[str, tuple[float, float, float]] = {}
    physics_rows = []
    attempted = 0
    missing_target = 0
    requested_work = 0.0
    admitted_work = 0.0
    suppressed_work = 0.0
    eligible_arrivals = 0
    total_requested = 0
    total_applied = 0
    total_suppressed = 0
    capacity_per_episode = (
        (ATTRIBUTION_V1_PROTOCOL.timeline.failed_start - ATTRIBUTION_V1_PROTOCOL.timeline.degraded_start)
        * reservation.budget_rate
        * reservation.scale
    )

    for index, identity in enumerate(library.identities):
        episode = reconstruct_episode(config, identity, ATTRIBUTION_V1_PROTOCOL)
        capture = _CaptureSource(policy.new_episode(), RandomThresholdSelector.for_episode(index))
        baseline = simulate_episode_online(
            episode,
            capture,
            reservation,
            degraded_slowdown=2.0,
            hedge_delay=2.0,
            public_price=public_price,
        )
        attempted += 1
        physics_rows.append(build_episode_metrics(baseline.simulation, degraded_slowdown=2.0))
        audit = baseline.audit()
        total_requested += int(audit["requested_hedges"])
        total_applied += int(audit["applied_hedges"])
        total_suppressed += int(audit["reservation_suppressed"])
        for decision in baseline.decisions:
            eligible = decision.phase is Phase.DEGRADED and decision.primary_replica == 0
            if eligible:
                eligible_arrivals += 1
                if decision.requested is not ProtectionAction.NORMAL:
                    requested_work += reservation.mean_requirement
                admitted_work += decision.charge
                if decision.reservation_suppressed:
                    suppressed_work += reservation.mean_requirement
        if capture.target_id is None or capture.observation is None:
            missing_target += 1
            continue
        bin_id = refined_bin_id(capture.observation)
        occupancy[bin_id] = occupancy.get(bin_id, 0) + 1
        fallback_by_bin.setdefault(bin_id, _one_hot(policy.base_action(capture.observation)))
        deviations = evaluate_token_pathwise_deviations(
            episode,
            policy.new_episode,
            capture.target_id,
            candidates=_ACTIONS,
            parameters=parameters,
            quote=quote,
            reservation=reservation,
            degraded_slowdown=2.0,
            hedge_delay=2.0,
        )
        attempted += deviations.attempted_calls
        if deviations.status != "completed" or len(deviations.rows) != 3:
            raise RuntimeError(deviations.failure_reason or "incomplete price panel")
        for row in deviations.rows:
            payoff = row.candidate_payoff
            key = (bin_id, row.candidate_requested)
            unpriced = payoff.total - payoff.price_cost
            priced = (
                payoff.total
                if price_basis == "incremental_executed_work"
                else requested_price_cost(unpriced, row.candidate_requested, public_price)
            )
            samples.setdefault(key, []).append(priced)
            unpriced_samples.setdefault(key, []).append(unpriced)

    supported = tuple(sorted(bin_id for bin_id, count in occupancy.items() if count >= panel_floor))
    unsupported = {bin_id: count for bin_id, count in sorted(occupancy.items()) if count < panel_floor}
    if not supported:
        raise RuntimeError("price round has no supported refined bin")
    means = {
        bin_id: {
            action: math.fsum(samples[(bin_id, action)]) / len(samples[(bin_id, action)])
            for action in _ACTIONS
        }
        for bin_id in supported
    }
    unpriced_means = {
        bin_id: {
            action.value: math.fsum(unpriced_samples[(bin_id, action)]) / len(unpriced_samples[(bin_id, action)])
            for action in _ACTIONS
        }
        for bin_id in supported
    }
    soft = {bin_id: _soft_response(means[bin_id], beta) for bin_id in supported}
    current = {
        bin_id: policy.state.probabilities_for_bin(bin_id, fallback_by_bin[bin_id])
        for bin_id in supported
    }
    next_state = policy.state.updated(
        soft,
        active_bins=supported,
        fallback_by_bin=fallback_by_bin,
        eta=eta,
    )
    next_probs = {
        bin_id: next_state.probabilities_for_bin(bin_id, fallback_by_bin[bin_id])
        for bin_id in supported
    }
    support_total = sum(occupancy[bin_id] for bin_id in supported)
    response_l1 = math.fsum(
        occupancy[bin_id] / support_total * _l1(current[bin_id], soft[bin_id])
        for bin_id in supported
    )
    update_l1 = math.fsum(
        occupancy[bin_id] / support_total * _l1(current[bin_id], next_probs[bin_id])
        for bin_id in supported
    )
    reservation_capacity = capacity_per_episode * len(library.identities)
    price_row = update_public_price(
        public_price,
        requested_work=requested_work,
        capacity=reservation_capacity,
        alpha=ALPHA,
    )
    if abs(requested_work - admitted_work - suppressed_work) > 1e-9:
        raise RuntimeError("requested/admitted/suppressed work identity failed")
    return ({
        "iteration": iteration,
        "public_price": public_price,
        **price_row,
        "requested_reservation_work": requested_work,
        "admitted_reservation_work": admitted_work,
        "suppressed_reservation_work": suppressed_work,
        "reservation_capacity": reservation_capacity,
        "eligible_arrivals": eligible_arrivals,
        "requested_hedges_all_phases": total_requested,
        "applied_hedges_all_phases": total_applied,
        "reservation_suppressed_all_phases": total_suppressed,
        "missing_target_count": missing_target,
        "supported_bins": list(supported),
        "unsupported_bins": unsupported,
        "supported_panel_count": support_total,
        "priced_q_means": {
            bin_id: {action.value: means[bin_id][action] for action in _ACTIONS}
            for bin_id in supported
        },
        "unpriced_q_means": unpriced_means,
        "soft_response": {bin_id: list(soft[bin_id]) for bin_id in supported},
        "policy_before": policy.state.to_dict(),
        "policy_after": next_state.to_dict(),
        "undamped_response_l1": response_l1,
        "damped_update_l1_eta0p2": update_l1,
        "physics_metrics": aggregate_round_physics_metrics(physics_rows),
        "attempted_calls": attempted,
        "status": "completed",
    }, RefinedPopulationPolicy(next_state, seed=policy.seed))


def run_price_feedback(
    config: object,
    *,
    episode_count: int = EPISODE_COUNT,
    max_rounds: int = MAX_ROUNDS,
    panel_floor: int = PANEL_FLOOR,
    price_basis: str = "incremental_executed_work",
) -> dict[str, object]:
    from .token_t3a_execution import validate_frozen_config

    validate_frozen_config(config)
    if type(episode_count) is not int or not 0 < episode_count <= EPISODE_COUNT:
        raise ValueError("episode_count is outside 1..256")
    if type(max_rounds) is not int or not 0 < max_rounds <= MAX_ROUNDS:
        raise ValueError("max_rounds is outside 1..4")
    if type(panel_floor) is not int or panel_floor <= 0:
        raise ValueError("panel_floor must be a positive int")
    if price_basis not in {"incremental_executed_work", "requested_reserved_work"}:
        raise ValueError("unsupported price_basis")
    library = build_identity_library(NAMESPACE, MACRO_SEED, episode_count=episode_count)
    policy = RefinedPopulationPolicy(seed=MACRO_SEED)
    price = 0.0
    iterations = []
    for iteration in range(max_rounds):
        row, policy = _run_round(
            config,
            library,
            policy,
            iteration=iteration,
            public_price=price,
            panel_floor=panel_floor,
            beta=BETA,
            eta=ETA,
            price_basis=price_basis,
        )
        iterations.append(row)
        price = float(row["next_price"])
    attempted = sum(int(row["attempted_calls"]) for row in iterations)
    call_limit = episode_count * max_rounds * 5
    if attempted > call_limit:
        raise RuntimeError("price feedback exceeded its physical call budget")
    return {
        "evidence_label": (
            EVIDENCE_LABEL
            if price_basis == "incremental_executed_work"
            else "token_refined_round_requested_reservation_price_feedback_exploratory"
        ),
        "execution_profile": "exploratory",
        "status": "completed_exploratory",
        "stop_reason": "bounded four-round diagnostic completed",
        "population_policy_id": "refined_soft_response_with_exact_niin_fallback_v1",
        "environment_id": "theta_token_load0p7_v1",
        "namespace": NAMESPACE,
        "macro_seed": MACRO_SEED,
        "episode_count": episode_count,
        "max_rounds": max_rounds,
        "panel_floor": panel_floor,
        "alpha": ALPHA,
        "beta": BETA,
        "eta": ETA,
        "price_basis": price_basis,
        "cost_model_id": (
            "token_runtime_adr0009_v1"
            if price_basis == "incremental_executed_work"
            else "token_runtime_requested_reservation_price_v1"
        ),
        "identity_library_fingerprint": library.fingerprint,
        "iterations": iterations,
        "attempted_calls": attempted,
        "call_limit": call_limit,
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
    }


def run_requested_price_feedback(
    config: object,
    *,
    episode_count: int = EPISODE_COUNT,
    max_rounds: int = MAX_ROUNDS,
    panel_floor: int = PANEL_FLOOR,
) -> dict[str, object]:
    return run_price_feedback(
        config,
        episode_count=episode_count,
        max_rounds=max_rounds,
        panel_floor=panel_floor,
        price_basis="requested_reserved_work",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_price_feedback(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str = RUN_ID,
) -> Path:
    from .config import load_config
    from .token_t3a_execution import build_source_bundle, environment_fingerprint

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    summary = run_price_feedback(config)
    source = Path(__file__).resolve()
    manifest = {
        "schema_id": "token_refined_round_price_feedback_manifest_v1",
        "run_id": run_id,
        "evidence_label": EVIDENCE_LABEL,
        "execution_profile": "exploratory",
        "source_bundle": asdict(build_source_bundle(project)),
        "additional_source_sha256": {source.name: _sha256(source)},
        "environment_fingerprint": environment_fingerprint(config),
        "identity_library_fingerprint": summary["identity_library_fingerprint"],
        "episode_count": EPISODE_COUNT,
        "max_rounds": MAX_ROUNDS,
        "call_limit": CALL_LIMIT,
        "alpha": ALPHA,
        "beta": BETA,
        "eta": ETA,
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


def execute_requested_price_feedback(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str = REQUESTED_RUN_ID,
) -> Path:
    from .config import load_config
    from .token_t3a_execution import build_source_bundle, environment_fingerprint

    project = Path(project_root).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    summary = run_requested_price_feedback(config)
    comparison_path = (
        project
        / "artifacts"
        / RUN_ID
        / "summary.json"
    )
    comparison_bytes = comparison_path.read_bytes()
    comparison = json.loads(comparison_bytes)
    comparison_rows = []
    for requested_row, execution_row in zip(summary["iterations"], comparison["iterations"]):
        requested_metrics = requested_row["physics_metrics"]
        execution_metrics = execution_row["physics_metrics"]
        comparison_rows.append({
            "iteration": requested_row["iteration"],
            "requested_price": requested_row["public_price"],
            "execution_price": execution_row["public_price"],
            "requested_work_delta": (
                requested_row["requested_reservation_work"]
                - execution_row["requested_reservation_work"]
            ),
            "admitted_work_delta": (
                requested_row["admitted_reservation_work"]
                - execution_row["admitted_reservation_work"]
            ),
            "mean_latency_delta": (
                requested_metrics["latency"]["overall"]["mean_latency"]
                - execution_metrics["latency"]["overall"]["mean_latency"]
            ),
            "p99_latency_delta": (
                requested_metrics["latency"]["overall"]["latency_p99"]
                - execution_metrics["latency"]["overall"]["latency_p99"]
            ),
            "replayed_tokens_delta": (
                requested_metrics["replay"]["replayed_tokens"]
                - execution_metrics["replay"]["replayed_tokens"]
            ),
            "hedge_launches_delta": (
                requested_metrics["hedge"]["launched"]
                - execution_metrics["hedge"]["launched"]
            ),
        })
    summary["execution_price_parent_run_id"] = RUN_ID
    summary["execution_price_parent_sha256"] = hashlib.sha256(comparison_bytes).hexdigest()
    summary["execution_price_comparison"] = comparison_rows
    source = Path(__file__).resolve()
    manifest = {
        "schema_id": "token_refined_round_requested_price_feedback_manifest_v1",
        "run_id": run_id,
        "evidence_label": summary["evidence_label"],
        "cost_model_id": summary["cost_model_id"],
        "execution_profile": "exploratory",
        "source_bundle": asdict(build_source_bundle(project)),
        "additional_source_sha256": {source.name: _sha256(source)},
        "environment_fingerprint": environment_fingerprint(config),
        "identity_library_fingerprint": summary["identity_library_fingerprint"],
        "comparison_parent_run_id": RUN_ID,
        "comparison_parent_sha256": summary["execution_price_parent_sha256"],
        "episode_count": EPISODE_COUNT,
        "max_rounds": MAX_ROUNDS,
        "call_limit": CALL_LIMIT,
        "alpha": ALPHA,
        "beta": BETA,
        "eta": ETA,
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
    "RefinedPopulationPolicy",
    "execute_price_feedback",
    "execute_requested_price_feedback",
    "requested_price_cost",
    "run_price_feedback",
    "run_requested_price_feedback",
    "update_public_price",
]
