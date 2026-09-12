"""Diagnostic-only Stage 4 process scaling probe.

This script uses the frozen 32-scenario fit task shape and never writes a
campaign artifact or selects a rule.  It intentionally imports private worker
primitives so startup and steady-state time can be measured separately without
changing the production runner merely for profiling.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mfg_hedge.campaign_execution as execution
from mfg_hedge.game_deviations import DeviationScenario
from mfg_hedge.game_workload import (
    ActionRule,
    build_common_fault_path,
    build_nested_population,
)


NAMESPACE = "shared-backup-game:v1:stage4-fit"
MACRO_SEED = 20260905
EXPERT_COUNT = 8
SCENARIO_COUNT = 32
TASK_COUNT = 8


def _ready() -> tuple[int, int | None]:
    return os.getpid(), execution._rss_bytes()


def _timed_task(task: execution.DeviationTask):
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    result = execution._run_deviation_task(task)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    serialized_bytes = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result, wall, cpu, serialized_bytes


def _build_scenarios():
    scenarios = []
    for common_episode in range(16):
        fault = build_common_fault_path(NAMESPACE, MACRO_SEED, common_episode)
        for population_id in (0, 1):
            population = build_nested_population(fault, population_id, EXPERT_COUNT)
            scenarios.append(DeviationScenario(
                f"common-{common_episode:02d}-population-{population_id}",
                population,
                population.common_fault_fingerprint,
            ))
    return tuple(scenarios)


def _measure(worker_count, prepared, tasks):
    context = multiprocessing.get_context("spawn")
    parameters = execution._WorkerParameters(EXPERT_COUNT, 0.5, 2.0, 1.5)
    parent_cpu_start = time.process_time()
    startup_start = time.perf_counter()
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=execution._worker_init,
        initargs=(prepared, parameters),
    )
    ready_futures = [executor.submit(_ready) for _ in range(worker_count)]
    ready_rows = [future.result() for future in ready_futures]
    startup_seconds = time.perf_counter() - startup_start

    steady_start = time.perf_counter()
    futures = [executor.submit(_timed_task, task) for task in tasks]
    results = [future.result() for future in futures]
    steady_seconds = time.perf_counter() - steady_start

    shutdown_start = time.perf_counter()
    executor.shutdown(wait=True, cancel_futures=True)
    shutdown_seconds = time.perf_counter() - shutdown_start
    parent_cpu_seconds = time.process_time() - parent_cpu_start

    task_wall = [row[1] for row in results]
    task_cpu = [row[2] for row in results]
    result_bytes = [row[3] for row in results]
    calls = len(tasks) * len(prepared)
    return {
        "workers": worker_count,
        "worker_pids": sorted({row[0] for row in ready_rows}),
        "startup_seconds": startup_seconds,
        "steady_seconds": steady_seconds,
        "shutdown_seconds": shutdown_seconds,
        "parent_cpu_seconds": parent_cpu_seconds,
        "calls": calls,
        "steady_calls_per_second": calls / steady_seconds,
        "task_wall_median": statistics.median(task_wall),
        "task_wall_max": max(task_wall),
        "task_cpu_median": statistics.median(task_cpu),
        "task_cpu_max": max(task_cpu),
        "sum_task_wall": sum(task_wall),
        "sum_task_cpu": sum(task_cpu),
        "parallel_wall_efficiency": sum(task_wall) / (worker_count * steady_seconds),
        "cpu_to_task_wall_ratio": sum(task_cpu) / sum(task_wall),
        "result_pickle_bytes_total": sum(result_bytes),
        "result_pickle_bytes_per_task": result_bytes,
        "worker_ready_rss_bytes": [row[1] for row in ready_rows],
    }


def main():
    build_start = time.perf_counter()
    scenarios = _build_scenarios()
    build_seconds = time.perf_counter() - build_start
    prepare_start = time.perf_counter()
    prepared = execution._prepare_scenarios(scenarios)
    prepare_seconds = time.perf_counter() - prepare_start
    rule = ActionRule("NNNN", tuple("NNNN"))
    tasks = tuple(
        execution.DeviationTask(expert, rule, rule)
        for expert in range(TASK_COUNT)
    )
    report = {
        "identity": {
            "namespace": NAMESPACE,
            "macro_seed": MACRO_SEED,
            "expert_count": EXPERT_COUNT,
            "scenario_count": len(prepared),
            "task_count": len(tasks),
            "calls_per_measurement": len(prepared) * len(tasks),
            "token_count_min": min(
                scenario.prepared_trace.expected_token_count for scenario in prepared
            ),
            "token_count_max": max(
                scenario.prepared_trace.expected_token_count for scenario in prepared
            ),
        },
        "input": {
            "build_seconds": build_seconds,
            "prepare_seconds": prepare_seconds,
            "initializer_pickle_bytes": len(pickle.dumps(
                prepared, protocol=pickle.HIGHEST_PROTOCOL
            )),
            "task_pickle_bytes": len(pickle.dumps(
                tasks[0], protocol=pickle.HIGHEST_PROTOCOL
            )),
        },
        "measurements": [],
    }
    for workers in (1, 2, 4, 8):
        report["measurements"].append(_measure(workers, prepared, tasks))
        print(json.dumps(report["measurements"][-1], sort_keys=True), flush=True)
    print("FINAL_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
