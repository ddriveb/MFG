"""Bounded load-stress diagnostics; never promoted to official A/B results."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from .calibration import build_calibration_table
from .config import PairedExperimentConfig
from .domain import CommonState, ProtectionAction, TokenClass
from .mfg_solver import MFGSolution, SolverParameters, SolverStatus, solve_mfg


def official_gate_diagnostic(solution: MFGSolution) -> dict:
    failures = []
    for state in (CommonState.HEALTHY, CommonState.DEGRADED):
        item = solution.solutions[state]
        if item.status is not SolverStatus.CONVERGED:
            failures.append(f"{state.value}:status={item.status.value}")
        if item.grid_clamped:
            failures.append(f"{state.value}:grid_clamped")
    failed = solution.solutions[CommonState.FAILED]
    if failed.status is not SolverStatus.FORCED_NORMAL:
        failures.append(f"F:status={failed.status.value}")
    if failed.grid_clamped:
        failures.append("F:grid_clamped")
    exact_normal = all(
        probabilities.get(ProtectionAction.NORMAL) == 1.0
        and all(
            probabilities.get(action, 0.0) == 0.0
            for action in (
                ProtectionAction.DELAYED_HEDGE,
                ProtectionAction.IMMEDIATE_HEDGE,
            )
        )
        for token_class, probabilities in failed.probabilities.items()
        if token_class in (TokenClass.REGULAR, TokenClass.URGENT)
    ) and set(failed.probabilities) == {TokenClass.REGULAR, TokenClass.URGENT}
    if not exact_normal:
        failures.append("F:policy_not_exact_all_normal")
    return {
        "pass": not failures,
        "failures": failures,
        "paired_simulation_admissible": not failures,
    }


def _solver_parameters(config: PairedExperimentConfig) -> SolverParameters:
    base = config.base_config()
    return SolverParameters(
        inverse_temperature=base.softmax_inverse_temperature,
        policy_damping=base.policy_damping,
        price_step=base.price_step,
        price_damping=base.price_damping,
        replay_penalty_regular=config.replay_penalty_regular,
        replay_penalty_urgent=config.replay_penalty_urgent,
        incremental_work_cost=config.incremental_work_cost,
        wasted_work_cost=config.wasted_work_cost,
    )


def run_load_stress_diagnostics(
    config: PairedExperimentConfig, loads: Sequence[float]
) -> tuple[dict, ...]:
    numbers = tuple(float(load) for load in loads)
    if not numbers or any(not math.isfinite(load) or load <= 0.0 for load in numbers):
        raise ValueError("stress loads must be non-empty, finite, and positive")
    if len(set(numbers)) != len(numbers):
        raise ValueError("stress loads must be unique")
    records = []
    for load in numbers:
        resolved = replace(config, healthy_offered_load=load)
        resolved.validate()
        table = build_calibration_table(resolved.base_config())
        solution = solve_mfg(
            resolved.base_config(), table, _solver_parameters(resolved)
        )
        gate = official_gate_diagnostic(solution)
        records.append(
            {
                "healthy_offered_load": load,
                "classification": "stress_diagnostic_only",
                "official_gate": gate,
                "paired_simulation_run": False,
                "paired_simulation_reason": (
                    "not_run_gate_failed"
                    if not gate["pass"]
                    else "not_required_by_bounded_solver_diagnostic"
                ),
                "resolved_config": {
                    name: getattr(resolved, name)
                    for name in type(resolved).__dataclass_fields__
                },
                "calibration_provenance": dict(table.provenance),
                "mfg_solution": solution.to_mapping()["states"],
            }
        )
    return tuple(records)


def load_artifact_name(load: float) -> str:
    text = format(float(load), ".12g").replace("-", "m").replace(".", "p")
    return f"load_{text}.json"
