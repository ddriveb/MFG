"""Profile one frozen-shape 32-scenario worker task in-process."""

from __future__ import annotations

import cProfile
from pathlib import Path
import pstats
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mfg_hedge.campaign_execution as execution
from mfg_hedge.game_deviations import DeviationScenario
from mfg_hedge.game_workload import (
    ActionRule,
    build_common_fault_path,
    build_nested_population,
)


def main() -> None:
    scenarios = []
    for common_episode in range(16):
        fault = build_common_fault_path(
            "shared-backup-game:v1:stage4-fit", 20260905, common_episode
        )
        for population_id in (0, 1):
            population = build_nested_population(fault, population_id, 8)
            scenarios.append(DeviationScenario(
                f"common-{common_episode:02d}-population-{population_id}",
                population,
                population.common_fault_fingerprint,
            ))
    prepared = execution._prepare_scenarios(scenarios)
    execution._worker_init(prepared, execution._WorkerParameters(8, 0.5, 2.0, 1.5))
    rule = ActionRule("NNNN", tuple("NNNN"))
    task = execution.DeviationTask(0, rule, rule)
    profile = cProfile.Profile()
    profile.enable()
    result = execution._run_deviation_task(task)
    profile.disable()
    print(
        f"scenarios={len(prepared)} row_scores={len(result.row.scenario_scores)}"
    )
    pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()
