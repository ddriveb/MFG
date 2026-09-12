"""Multi-seed paired campaign execution and seed-level statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean, stdev
from typing import Mapping, Sequence

from .calibration import CalibrationTable
from .common_state import CommonState
from .config import PairedExperimentConfig
from .mfg_solver import MFGSolution, state_capacity
from .paired import (
    run_paired_comparison,
    solve_paired_policy,
    trace_identity_fingerprint,
)
from .paired_metrics import build_arm_summary, build_comparison


CAMPAIGN_METRICS = (
    "mean_latency",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "latency_p99_arrival_H",
    "latency_p99_arrival_D",
    "latency_p99_arrival_F",
    "latency_p99_arrival_R",
    "mean_queue_delay",
    "replay_rate",
    "total_executed_work",
    "execution_amplification",
    "wasted_work",
    "drain_duration",
    "sustained_overload_duration",
)

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
_T95 = (
    math.nan,
    12.706204736432095,
    4.302652729911275,
    3.182446305284263,
    2.7764451051977987,
    2.570581835636314,
    2.446911848791681,
    2.3646242510102993,
    2.3060041350333704,
    2.2621571627409915,
    2.2281388519649385,
    2.200985160082949,
    2.178812829667227,
    2.1603686564610127,
    2.1447866879169273,
    2.131449545559323,
    2.1199052992210112,
    2.1098155778331806,
    2.10092204024096,
    2.093024054408263,
    2.0859634472658364,
    2.079613844727662,
    2.0738730679040147,
    2.0686576104190406,
    2.0638985616280205,
    2.0595385527532946,
    2.055529438642871,
    2.0518305164802833,
    2.048407141795244,
    2.045229642132703,
    2.0422724563012373,
)


def evaluation_seeds(base_seed: int, seed_count: int) -> tuple[int, ...]:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an int")
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive int")
    return tuple(base_seed + offset for offset in range(seed_count))


def _t95(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    return _T95[df] if df < len(_T95) else 1.959963984540054


def paired_statistic(values: Sequence[float]) -> dict:
    numbers = tuple(float(value) for value in values)
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("paired statistic values must be non-empty and finite")
    mean = fmean(numbers)
    if len(numbers) == 1:
        sample_stddev = None
        ci95 = [mean, mean]
    else:
        sample_stddev = stdev(numbers)
        half_width = _t95(len(numbers) - 1) * sample_stddev / math.sqrt(len(numbers))
        ci95 = [mean - half_width, mean + half_width]
    return {
        "n": len(numbers),
        "mean": mean,
        "sample_stddev": sample_stddev,
        "ci95": ci95,
        "sign_counts": {
            "negative": sum(value < 0.0 for value in numbers),
            "zero": sum(value == 0.0 for value in numbers),
            "positive": sum(value > 0.0 for value in numbers),
        },
    }


def build_campaign_aggregate(
    records: Sequence[Mapping], *, metric_names: Sequence[str] = CAMPAIGN_METRICS
) -> dict:
    if not records:
        raise ValueError("campaign requires at least one seed record")
    seeds = [int(record["evaluation_seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("campaign evaluation seeds must be unique")
    metrics = {}
    for name in metric_names:
        cells = [record["comparison"]["metrics"][name] for record in records]
        arm_a = [float(cell["arm_a"]) for cell in cells]
        arm_b = [float(cell["arm_b"]) for cell in cells]
        delta = [float(cell["delta"]) for cell in cells]
        relative_values = [cell["relative"] for cell in cells]
        relative = (
            paired_statistic(relative_values)
            if all(isinstance(value, (int, float)) and math.isfinite(value) for value in relative_values)
            else None
        )
        metrics[name] = {
            "direction": "lower_is_better",
            "arm_a_values": arm_a,
            "arm_b_values": arm_b,
            "delta_values": delta,
            "arm_a": paired_statistic(arm_a),
            "arm_b": paired_statistic(arm_b),
            "delta": paired_statistic(delta),
            "relative": relative,
        }
    return {
        "seed_count": len(records),
        "evaluation_seeds": seeds,
        "delta_definition": "arm_b_mfg_hedge_minus_arm_a_no_hedge",
        "confidence_interval": "two-sided_95pct_student_t_over_seed_level_paired_values",
        "metrics": metrics,
    }


@dataclass(frozen=True)
class CampaignResult:
    records: tuple[dict, ...]
    aggregate: dict
    solution: MFGSolution


def run_multiseed_campaign(
    config: PairedExperimentConfig,
    table: CalibrationTable,
) -> CampaignResult:
    """Run every declared evaluation seed with one frozen calibrated policy."""
    solution = solve_paired_policy(config, table)
    timeline = config.timeline()
    capacities = {
        "H": state_capacity(config.base_config(), CommonState.HEALTHY),
        "D": state_capacity(config.base_config(), CommonState.DEGRADED),
        "F": state_capacity(config.base_config(), CommonState.FAILED),
        "R": state_capacity(config.base_config(), CommonState.HEALTHY),
    }
    records = []
    for seed in evaluation_seeds(config.base_seed, config.seed_count):
        run = run_paired_comparison(
            config,
            table,
            config.tokens_per_run,
            evaluation_seed=seed,
            solution=solution,
        )
        summary_a = build_arm_summary(
            run.arm_a,
            timeline,
            arm="no_hedge",
            projection=None,
            bin_width=config.storm_bin_width,
            capacities=capacities,
        )
        summary_b = build_arm_summary(
            run.arm_b,
            timeline,
            arm="mfg_hedge",
            projection=run.projection,
            bin_width=config.storm_bin_width,
            capacities=capacities,
        )
        for arm_name, summary in (("A", summary_a), ("B", summary_b)):
            broken = [name for name, ok in summary["invariants"].items() if ok is not True]
            if broken:
                raise ValueError(f"invariants violated in arm {arm_name}, seed {seed}: {broken}")
        comparison = build_comparison(summary_a, summary_b)
        records.append(
            {
                "evaluation_seed": seed,
                "token_count": config.tokens_per_run,
                "trace_identity": trace_identity_fingerprint(run.trace),
                "tau0": run.tau0,
                "invariants": {
                    "arm_a": summary_a["invariants"],
                    "arm_b": summary_b["invariants"],
                },
                "arm_a_no_hedge": summary_a,
                "arm_b_mfg_hedge": summary_b,
                "comparison": comparison,
            }
        )
    frozen = tuple(records)
    return CampaignResult(
        records=frozen,
        aggregate=build_campaign_aggregate(frozen),
        solution=solution,
    )
