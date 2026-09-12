"""Representative-token / tagged-probe ActionStats calibration.

Spec: `.scratch/mfg-hedge-paired-comparison/spec.md` section 11; ADR-0006.

rho coordinate: for a target state z, the background is all-Normal traffic and

    background_work_rate(z) =
        sum over background Tokens whose arrival state is z of
            (attempt-0 executed_work + attempt-1 executed_work)
        / duration of the state-z arrival window
    achieved_rho(z) = background_work_rate(z) / capacity(z)

i.e. it includes Primary work consumed before a failure and later lost, and
the Replay work of those Tokens; it excludes attempt-2 work and every tagged
probe's attempts; the denominator is the arrival window, never
drain_end_time; attribution is by arrival state; H/D/F use their own
capacities; R is not calibrated separately (it reuses H). Queueing cost
closes only through this aggregate rho (finite-action mean-field
approximation; the background action composition is not modeled).

Probes are designated, never inserted: an existing in-window background Token
keeps its id, arrival, and dispatch parity, has its class replaced by the
probe class, and its attempt draws replaced by stable per-sample probe draws
(`calibration:...:probe:<sample>`). Rho measurement runs probe-free
background episodes, so probe work never enters rho.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import random
from statistics import NormalDist, fmean
from typing import Any, Callable, Mapping, Sequence

from .common_state import CommonStateTimeline
from .config import ExperimentConfig
from .domain import ActionStats, CommonState, ProtectionAction, TokenClass
from .hedge_simulation import simulate_hedge_common_state
from .workload import TokenSpec, WorkloadTrace, lognormal_parameters


RHO_GRID_DEFAULT = (0.5, 0.7, 0.9, 1.1, 1.3)
RATE_BOUNDS = (0.05, 3.0)
BISECTION_MAX_ITERATIONS = 40
RHO_TOLERANCE = 0.02
MIN_SAMPLES_PER_CELL = 400
MIN_BACKGROUND_COHORT = 400
MAX_RHO_EPISODES = 100_000
MAX_PROBE_EPISODE_SKIPS = 64
MAX_EPISODE_TOKENS = 1_000_000
CALIBRATION_TIMELINE = CommonStateTimeline(
    degraded_start=100.0, failed_start=200.0, recovered_start=220.0
)


def _seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _state_window(timeline: CommonStateTimeline, state: CommonState) -> tuple[float, float]:
    if state is CommonState.HEALTHY:
        return 0.0, timeline.degraded_start
    if state is CommonState.DEGRADED:
        return timeline.degraded_start, timeline.failed_start
    if state is CommonState.FAILED:
        return timeline.failed_start, timeline.recovered_start
    raise ValueError(f"state must be H, D, or F, got {state!r}")


def _state_capacity(config: ExperimentConfig, state: CommonState) -> float:
    per_replica = 1.0 / config.healthy_service_mean
    if state is CommonState.HEALTHY:
        return config.replicas_per_expert * per_replica
    if state is CommonState.DEGRADED:
        return per_replica + per_replica / config.degraded_slowdown
    if state is CommonState.FAILED:
        return (config.replicas_per_expert - 1) * per_replica
    raise ValueError(f"state must be H, D, or F, got {state!r}")


def background_work_rate(
    result, window_start: float, window_end: float
) -> float:
    """Work rate of background Tokens whose arrival falls in the window.

    Sums attempt-0 and attempt-1 executed work (Primary work consumed before
    a failure and lost is included, Replay work is included, attempt-2 Hedge
    work is excluded), divided by the window duration. Attribution is by
    arrival time, never completion phase; the denominator is never
    drain_end_time.
    """
    start = _require_finite(window_start, "window_start")
    end = _require_finite(window_end, "window_end")
    if not 0.0 <= start < end:
        raise ValueError(
            f"require 0 <= window_start < window_end, got [{start!r}, {end!r})"
        )
    work = 0.0
    for token in result.tokens:
        if start <= token.arrival_time < end:
            for attempt in token.attempts:
                if attempt.attempt_id in (0, 1):
                    work += attempt.executed_work
    return work / (end - start)


@dataclass(frozen=True)
class ProbeSample:
    latency: float
    replayed: bool
    hedge_work: float
    replay_work: float
    wasted_work: float = 0.0

    def __post_init__(self) -> None:
        for name in ("latency", "hedge_work", "replay_work", "wasted_work"):
            value = _require_finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if not isinstance(self.replayed, bool):
            raise ValueError(f"replayed must be a bool, got {self.replayed!r}")


def aggregate_probe_samples(samples: Sequence[ProbeSample]) -> ActionStats:
    """Mean ActionStats over probe samples; missing attempts contribute 0."""
    if not samples:
        raise ValueError("aggregate_probe_samples requires at least one sample")
    count = len(samples)
    return ActionStats(
        mean_latency=fmean(sample.latency for sample in samples),
        replay_probability=sum(1 for sample in samples if sample.replayed) / count,
        deadline_miss_probability=0.0,
        expected_hedge_work=fmean(sample.hedge_work for sample in samples),
        expected_replay_work=fmean(sample.replay_work for sample in samples),
        expected_wasted_work=fmean(sample.wasted_work for sample in samples),
    )


@dataclass(frozen=True)
class BisectionResult:
    target_rho: float
    achieved_rho: float
    rho_error: float
    background_arrival_rate: float
    iterations: int
    evaluations: tuple[tuple[float, float], ...] = ()


def find_background_rate(
    measure: Callable[[float], float], target_rho: float, *, context: str = ""
) -> BisectionResult:
    """Deterministic bisection on the background arrival rate.

    Fixed rules: bounds [0.05, 3.0], at most 40 iterations, midpoint halves
    the bracket, the first midpoint within tolerance 0.02 wins, ties keep the
    earlier candidate, and exceeding the cap without meeting tolerance is an
    error naming the context and the best candidate — never a disguised
    best-effort success.
    """
    target = _require_finite(target_rho, "target_rho")
    if target <= 0.0:
        raise ValueError(f"target_rho must be positive, got {target_rho!r}")
    low, high = RATE_BOUNDS
    evaluations: list[tuple[float, float]] = []
    rho_low = measure(low)
    evaluations.append((low, rho_low))
    if abs(rho_low - target) <= RHO_TOLERANCE:
        return BisectionResult(
            target_rho=target,
            achieved_rho=rho_low,
            rho_error=abs(rho_low - target),
            background_arrival_rate=low,
            iterations=0,
            evaluations=tuple(evaluations),
        )
    rho_high = measure(high)
    evaluations.append((high, rho_high))
    if abs(rho_high - target) <= RHO_TOLERANCE:
        return BisectionResult(
            target_rho=target,
            achieved_rho=rho_high,
            rho_error=abs(rho_high - target),
            background_arrival_rate=high,
            iterations=0,
            evaluations=tuple(evaluations),
        )
    if not (rho_low <= target <= rho_high):
        raise ValueError(
            f"rho target not bracketed ({context or 'no context'}): target "
            f"{target} outside measured [{rho_low}, {rho_high}] at rate "
            f"bounds {RATE_BOUNDS}"
        )
    best_rate, best_rho, best_error = 0.0, math.nan, math.inf
    for iteration in range(1, BISECTION_MAX_ITERATIONS + 1):
        mid = (low + high) / 2.0
        rho_mid = measure(mid)
        evaluations.append((mid, rho_mid))
        error = abs(rho_mid - target)
        if error < best_error:
            best_rate, best_rho, best_error = mid, rho_mid, error
        if error <= RHO_TOLERANCE:
            return BisectionResult(
                target_rho=target,
                achieved_rho=rho_mid,
                rho_error=error,
                background_arrival_rate=mid,
                iterations=iteration,
                evaluations=tuple(evaluations),
            )
        if rho_mid < target:
            low = mid
        else:
            high = mid
    raise ValueError(
        f"rho bisection failed to meet tolerance {RHO_TOLERANCE} within "
        f"{BISECTION_MAX_ITERATIONS} iterations ({context or 'no context'}): "
        f"target={target}, best achieved={best_rho} error={best_error} "
        f"rate={best_rate}"
    )


@dataclass(frozen=True)
class CalibrationCell:
    state: CommonState
    target_rho: float
    achieved_rho: float
    rho_error: float
    token_class: TokenClass
    action: ProtectionAction
    background_arrival_rate: float
    sample_count: int
    rho_measurement_episode_count: int
    rho_background_cohort_count: int
    probe_successful_episode_count: int
    probe_generation_attempt_count: int
    probe_skipped_episode_count: int
    stats: ActionStats

    def __post_init__(self) -> None:
        if not isinstance(self.state, CommonState):
            raise ValueError(f"state must be a CommonState, got {self.state!r}")
        if not isinstance(self.token_class, TokenClass):
            raise ValueError(
                f"token_class must be a TokenClass, got {self.token_class!r}"
            )
        if not isinstance(self.action, ProtectionAction):
            raise ValueError(
                f"action must be a ProtectionAction, got {self.action!r}"
            )
        target = _require_finite(self.target_rho, "target_rho")
        if target <= 0.0:
            raise ValueError(f"target_rho must be positive, got {target!r}")
        if _require_finite(self.achieved_rho, "achieved_rho") < 0.0:
            raise ValueError("achieved_rho must be non-negative")
        if _require_finite(self.rho_error, "rho_error") < 0.0:
            raise ValueError("rho_error must be non-negative")
        if _require_finite(self.background_arrival_rate, "background_arrival_rate") <= 0.0:
            raise ValueError("background_arrival_rate must be positive")
        for name in (
            "sample_count",
            "rho_measurement_episode_count",
            "rho_background_cohort_count",
            "probe_successful_episode_count",
            "probe_generation_attempt_count",
            "probe_skipped_episode_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if not isinstance(self.stats, ActionStats):
            raise ValueError(f"stats must be an ActionStats, got {self.stats!r}")


@dataclass(frozen=True)
class CalibrationEstimate:
    stats: ActionStats
    clamped: bool


_STATE_ORDER = {CommonState.HEALTHY: 0, CommonState.DEGRADED: 1, CommonState.FAILED: 2}
_CLASS_ORDER = {TokenClass.REGULAR: 0, TokenClass.URGENT: 1}
_ACTION_ORDER = {
    ProtectionAction.NORMAL: 0,
    ProtectionAction.DELAYED_HEDGE: 1,
    ProtectionAction.IMMEDIATE_HEDGE: 2,
}
_HEDGE_ACTIONS = (ProtectionAction.DELAYED_HEDGE, ProtectionAction.IMMEDIATE_HEDGE)


@dataclass(frozen=True)
class CalibrationTable:
    cells: tuple[CalibrationCell, ...]
    expected_grid: tuple[float, ...] | None = None
    require_complete: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = [
            (cell.state, cell.target_rho, cell.token_class, cell.action)
            for cell in self.cells
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate calibration cell key")
        if self.expected_grid is not None:
            grid = tuple(float(point) for point in self.expected_grid)
            if not grid or any(
                not math.isfinite(point) or point <= 0.0 for point in grid
            ):
                raise ValueError(f"expected_grid must be finite positive: {grid!r}")
            object.__setattr__(self, "expected_grid", grid)
        if self.require_complete:
            self._validate_production_completeness()

    def _validate_production_completeness(self) -> None:
        if self.expected_grid is None:
            raise ValueError("production tables require expected_grid")
        expected = set()
        for state in (CommonState.HEALTHY, CommonState.DEGRADED):
            for rho in self.expected_grid:
                for token_class in (TokenClass.REGULAR, TokenClass.URGENT):
                    for action in (
                        ProtectionAction.NORMAL,
                        ProtectionAction.DELAYED_HEDGE,
                        ProtectionAction.IMMEDIATE_HEDGE,
                    ):
                        expected.add((state, rho, token_class, action))
        for rho in self.expected_grid:
            for token_class in (TokenClass.REGULAR, TokenClass.URGENT):
                expected.add((CommonState.FAILED, rho, token_class, ProtectionAction.NORMAL))
        actual = {
            (cell.state, cell.target_rho, cell.token_class, cell.action)
            for cell in self.cells
        }
        missing = expected - actual
        if missing:
            raise ValueError(
                f"production table incomplete; missing cells: {sorted(missing, key=repr)[:4]}..."
            )
        extra = actual - expected
        if extra:
            raise ValueError(
                f"production table carries unexpected cells (e.g. F Hedge): "
                f"{sorted(extra, key=repr)[:4]}"
            )
        for cell in self.cells:
            if cell.sample_count < MIN_SAMPLES_PER_CELL:
                raise ValueError(
                    f"production cell {cell.state.value}/{cell.target_rho}/"
                    f"{cell.token_class.value}/{cell.action.value} has "
                    f"sample_count {cell.sample_count} < {MIN_SAMPLES_PER_CELL}"
                )
            if cell.rho_background_cohort_count < MIN_BACKGROUND_COHORT:
                raise ValueError(
                    f"production cell {cell.state.value}/{cell.target_rho} has "
                    f"rho cohort {cell.rho_background_cohort_count} < "
                    f"{MIN_BACKGROUND_COHORT}"
                )
            if cell.probe_successful_episode_count != cell.sample_count:
                raise ValueError(
                    f"production cell {cell.state.value}/{cell.target_rho} has "
                    f"{cell.probe_successful_episode_count} probe episodes for "
                    f"{cell.sample_count} samples"
                )
            if (
                cell.probe_generation_attempt_count
                != cell.probe_successful_episode_count + cell.probe_skipped_episode_count
            ):
                raise ValueError(
                    "probe_generation_attempt_count must equal successful + "
                    f"skipped, got {cell.probe_generation_attempt_count} != "
                    f"{cell.probe_successful_episode_count} + "
                    f"{cell.probe_skipped_episode_count}"
                )
            consistent_error = abs(cell.achieved_rho - cell.target_rho)
            if cell.rho_error != consistent_error:
                raise ValueError(
                    f"cell {cell.state.value}/{cell.target_rho}/"
                    f"{cell.token_class.value}/{cell.action.value} rho_error "
                    f"{cell.rho_error} != abs(achieved-target) {consistent_error}"
                )
            if cell.rho_error > RHO_TOLERANCE:
                raise ValueError(
                    f"cell {cell.state.value}/{cell.target_rho} exceeds rho "
                    f"tolerance {RHO_TOLERANCE}"
                )

    def _group(
        self, state: CommonState, token_class: TokenClass, action: ProtectionAction
    ) -> list[CalibrationCell]:
        group = [
            cell
            for cell in self.cells
            if cell.state is state
            and cell.token_class is token_class
            and cell.action is action
        ]
        if not group:
            raise ValueError(
                f"missing cells for state={state}, class={token_class}, "
                f"action={action}"
            )
        return sorted(group, key=lambda cell: cell.target_rho)

    def estimate(
        self,
        state: CommonState,
        rho: float,
        token_class: TokenClass,
        action: ProtectionAction,
    ) -> CalibrationEstimate:
        """Linear interpolation between bracketing grid points; clamped outside."""
        rho = _require_finite(rho, "rho")
        group = self._group(state, token_class, action)
        if rho <= group[0].target_rho:
            return CalibrationEstimate(stats=group[0].stats, clamped=rho < group[0].target_rho)
        if rho >= group[-1].target_rho:
            return CalibrationEstimate(stats=group[-1].stats, clamped=rho > group[-1].target_rho)
        for low, high in zip(group, group[1:]):
            if low.target_rho <= rho <= high.target_rho:
                if rho == low.target_rho:
                    return CalibrationEstimate(stats=low.stats, clamped=False)
                if rho == high.target_rho:
                    return CalibrationEstimate(stats=high.stats, clamped=False)
                fraction = (rho - low.target_rho) / (high.target_rho - low.target_rho)
                fields = {}
                for name in (
                    "mean_latency",
                    "replay_probability",
                    "deadline_miss_probability",
                    "expected_hedge_work",
                    "expected_replay_work",
                    "expected_wasted_work",
                ):
                    low_value = getattr(low.stats, name)
                    fields[name] = low_value + fraction * (
                        getattr(high.stats, name) - low_value
                    )
                return CalibrationEstimate(stats=ActionStats(**fields), clamped=False)
        raise AssertionError("unreachable: bracketing points must exist")

    def to_json_bytes(self) -> bytes:
        """Deterministic serialization: fixed ordering, provenance, no timestamps."""
        ordered = sorted(
            self.cells,
            key=lambda cell: (
                _STATE_ORDER[cell.state],
                cell.target_rho,
                _CLASS_ORDER[cell.token_class],
                _ACTION_ORDER[cell.action],
            ),
        )
        payload = {
            "schema_version": 2,
            "provenance": dict(self.provenance),
            "cells": [
                {
                    "state": cell.state.value,
                    "target_rho": cell.target_rho,
                    "achieved_rho": cell.achieved_rho,
                    "rho_error": cell.rho_error,
                    "token_class": cell.token_class.value,
                    "action": cell.action.value,
                    "background_arrival_rate": cell.background_arrival_rate,
                    "sample_count": cell.sample_count,
                    "rho_measurement_episode_count": cell.rho_measurement_episode_count,
                    "rho_background_cohort_count": cell.rho_background_cohort_count,
                    "probe_successful_episode_count": cell.probe_successful_episode_count,
                    "probe_generation_attempt_count": cell.probe_generation_attempt_count,
                    "probe_skipped_episode_count": cell.probe_skipped_episode_count,
                    "stats": {
                        "mean_latency": cell.stats.mean_latency,
                        "replay_probability": cell.stats.replay_probability,
                        "deadline_miss_probability": cell.stats.deadline_miss_probability,
                        "expected_hedge_work": cell.stats.expected_hedge_work,
                        "expected_replay_work": cell.stats.expected_replay_work,
                        "expected_wasted_work": cell.stats.expected_wasted_work,
                    },
                }
                for cell in ordered
            ],
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )


def validate_contrast(table: CalibrationTable) -> None:
    """Fail fast unless some D cell separates Normal from a Hedge action."""
    degraded = [cell for cell in table.cells if cell.state is CommonState.DEGRADED]
    groups: dict[tuple[float, TokenClass], dict[ProtectionAction, CalibrationCell]] = {}
    for cell in degraded:
        groups.setdefault((cell.target_rho, cell.token_class), {})[cell.action] = cell
    for group in groups.values():
        normal = group.get(ProtectionAction.NORMAL)
        if normal is None:
            continue
        for action in _HEDGE_ACTIONS:
            other = group.get(action)
            if other is None:
                continue
            if (
                other.stats.replay_probability != normal.stats.replay_probability
                or other.stats.expected_replay_work != normal.stats.expected_replay_work
            ):
                return
    raise ValueError(
        "calibration contrast check failed: no D-state cell separates Normal "
        "from a Hedge action in replay_probability or expected_replay_work"
    )


def build_calibration_background(
    label: str, rate: float, config: ExperimentConfig, horizon: float
) -> WorkloadTrace:
    """Deterministic calibration episode over a fixed time horizon.

    Unit-rate exponential draws from the label are scaled by ``rate``;
    Tokens are generated while the arrival stays below ``horizon``, so every
    state window is populated regardless of rate. The same label yields the
    same unit prefix and service streams at every rate.
    """
    rate = _require_finite(rate, "rate")
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate!r}")
    horizon = _require_finite(horizon, "horizon")
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    arrival_rng = random.Random(_seed(f"{label}:arrival-unit"))
    arrivals: list[float] = []
    now = 0.0
    while True:
        now += arrival_rng.expovariate(1.0)
        arrival = now / rate
        if arrival >= horizon:
            break
        arrivals.append(arrival)
        if len(arrivals) > MAX_EPISODE_TOKENS:
            raise ValueError(
                f"episode {label!r} exceeded {MAX_EPISODE_TOKENS} Tokens"
            )
    token_count = len(arrivals)
    class_rng = random.Random(_seed(f"{label}:token-class"))
    tokens = tuple(
        TokenSpec(
            token_id=index,
            arrival_time=arrivals[index],
            token_class=(
                TokenClass.REGULAR
                if class_rng.random() < config.regular_token_ratio
                else TokenClass.URGENT
            ),
        )
        for index in range(token_count)
    )
    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )

    def draws(attempt: int) -> tuple[tuple[float, ...], ...]:
        streams = []
        for replica in range(config.replicas_per_expert):
            rng = random.Random(_seed(f"{label}:service:{replica}:{attempt}"))
            streams.append(tuple(rng.lognormvariate(mu, sigma) for _ in range(token_count)))
        return tuple(streams)

    return WorkloadTrace(
        base_seed=_seed(label),
        arrival_rate=rate,
        tokens=tokens,
        service_times=draws(0),
        replay_service_times=draws(1),
        hedge_service_times=draws(2),
    )


def _probe_draws(probe_label: str, config: ExperimentConfig) -> tuple[tuple[float, float], ...]:
    """Per-attempt (0, 1, 2) draws for both Replicas from the probe label."""
    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )
    rng = random.Random(_seed(probe_label))
    return tuple(
        tuple(rng.lognormvariate(mu, sigma) for _ in range(config.replicas_per_expert))
        for _ in range(3)
    )


def prepare_probe_trace(
    config: ExperimentConfig,
    state: CommonState,
    timeline: CommonStateTimeline,
    rate: float,
    episode_label: str,
    probe_sample_label: str,
    probe_class: TokenClass,
) -> tuple[WorkloadTrace, int] | None:
    """Designate an in-window background Token as the probe.

    Returns None when the episode has no Token arriving in the state window
    (the caller retries with the next deterministic label). No renumbering,
    no redraw: only the probe's class and its three attempt draws differ from
    the pure background.
    """
    window_start, window_end = _state_window(timeline, state)
    background = build_calibration_background(
        episode_label, rate, config, horizon=timeline.recovered_start
    )
    in_window = [
        spec.token_id
        for spec in background.tokens
        if window_start <= spec.arrival_time < window_end
    ]
    if not in_window:
        return None
    pick_rng = random.Random(_seed(f"{probe_sample_label}:pick"))
    probe_id = in_window[pick_rng.randrange(len(in_window))]
    draws = _probe_draws(probe_sample_label, config)
    tokens = tuple(
        TokenSpec(
            token_id=spec.token_id,
            arrival_time=spec.arrival_time,
            token_class=(
                probe_class if spec.token_id == probe_id else spec.token_class
            ),
        )
        for spec in background.tokens
    )

    def patched(streams: tuple[tuple[float, ...], ...], attempt: int):
        out = []
        for replica in range(config.replicas_per_expert):
            stream = list(streams[replica])
            stream[probe_id] = draws[attempt][replica]
            out.append(tuple(stream))
        return tuple(out)

    trace = replace(
        background,
        tokens=tokens,
        service_times=patched(background.service_times, 0),
        replay_service_times=patched(background.replay_service_times, 1),
        hedge_service_times=patched(background.hedge_service_times, 2),
    )
    return trace, probe_id


def _measure_background_rho(
    config: ExperimentConfig,
    state: CommonState,
    timeline: CommonStateTimeline,
    rate: float,
    label: str,
) -> tuple[float, int, int]:
    """Pooled background work rate; extends episodes until cohort >= 400.

    Probe-free: no probe slot exists in these traces, so probe work is
    excluded by construction. Returns (achieved_rho, cohort, episodes).
    """
    window_start, window_end = _state_window(timeline, state)
    duration = window_end - window_start
    capacity = _state_capacity(config, state)
    total_work = 0.0
    cohort = 0
    episodes = 0
    while cohort < MIN_BACKGROUND_COHORT:
        if episodes >= MAX_RHO_EPISODES:
            raise ValueError(
                f"rho measurement for {label} did not gather "
                f"{MIN_BACKGROUND_COHORT} cohort samples within "
                f"{MAX_RHO_EPISODES} episodes"
            )
        trace = build_calibration_background(
            f"{label}:rho:{episodes}", rate, config, horizon=timeline.recovered_start
        )
        in_window = sum(
            1
            for spec in trace.tokens
            if window_start <= spec.arrival_time < window_end
        )
        if in_window:
            result = simulate_hedge_common_state(
                trace, timeline=timeline, degraded_slowdown=config.degraded_slowdown
            )
            total_work += background_work_rate(result, window_start, window_end) * duration
            cohort += in_window
        episodes += 1
    achieved = (total_work / (episodes * duration)) / capacity
    return achieved, cohort, episodes


def calibrate_cohort(
    config: ExperimentConfig,
    state: CommonState,
    target_rho: float,
    token_class: TokenClass,
    *,
    timeline: CommonStateTimeline = CALIBRATION_TIMELINE,
    samples_per_cell: int = MIN_SAMPLES_PER_CELL,
    rate_cache: dict | None = None,
) -> tuple[CalibrationCell, ...]:
    """Calibrate one (state, rho, class) cohort for all feasible actions.

    Each sample uses its own deterministic background episode (skipping, with
    fixed retry labels, episodes lacking an in-window Token); N/D/I variants
    of a sample share the identical trace and differ only in the probe's
    action. The rho search is cached per (state, target_rho) so classes share
    the identical background rate.
    """
    if not isinstance(state, CommonState) or state not in (
        CommonState.HEALTHY,
        CommonState.DEGRADED,
        CommonState.FAILED,
    ):
        raise ValueError(f"state must be H, D, or F, got {state!r}")
    if not isinstance(token_class, TokenClass):
        raise ValueError(f"token_class must be a TokenClass, got {token_class!r}")
    if (
        isinstance(samples_per_cell, bool)
        or not isinstance(samples_per_cell, int)
        or samples_per_cell <= 0
    ):
        raise ValueError(
            f"samples_per_cell must be a positive int, got {samples_per_cell!r}"
        )

    cache = {} if rate_cache is None else rate_cache
    cache_key = (state, float(target_rho))
    if cache_key not in cache:
        rho_label = f"calibration:{state.value}"

        def measure(rate: float) -> float:
            achieved, cohort, episodes = _measure_background_rho(
                config, state, timeline, rate, rho_label
            )
            measure.cohort = cohort
            measure.episodes = episodes
            return achieved

        bisection = find_background_rate(
            measure,
            target_rho,
            context=(
                f"state={state.value}:rho={target_rho}:class={token_class.value}"
            ),
        )
        cache[cache_key] = (bisection, measure.cohort, measure.episodes)
    bisection, cohort, rho_episodes = cache[cache_key]
    rate = bisection.background_arrival_rate

    mu, sigma = lognormal_parameters(
        config.healthy_service_mean, config.service_time_cv
    )
    tau0 = math.exp(mu + sigma * NormalDist().inv_cdf(config.hedge_delay_quantile))

    actions = (
        (ProtectionAction.NORMAL,)
        if state is CommonState.FAILED
        else (
            ProtectionAction.NORMAL,
            ProtectionAction.DELAYED_HEDGE,
            ProtectionAction.IMMEDIATE_HEDGE,
        )
    )
    samples: dict[ProtectionAction, list[ProbeSample]] = {
        action: [] for action in actions
    }
    episodes_used = 0
    episodes_skipped = 0
    for sample in range(samples_per_cell):
        attempt = 0
        while True:
            if attempt > MAX_PROBE_EPISODE_SKIPS:
                raise ValueError(
                    f"no in-window Token for sample {sample} of "
                    f"{state.value}/{target_rho} after {attempt} skips"
                )
            episode_label = f"calibration:{state.value}:background:sample:{sample}"
            if attempt:
                episode_label = f"{episode_label}:skip:{attempt}"
            prepared = prepare_probe_trace(
                config,
                state,
                timeline,
                rate,
                episode_label,
                f"calibration:{state.value}:{target_rho}:{token_class.value}:probe:{sample}",
                token_class,
            )
            if prepared is not None:
                break
            episodes_skipped += 1
            attempt += 1
        episodes_used += 1
        trace, probe_id = prepared
        for action in actions:
            plan = {} if action is ProtectionAction.NORMAL else {probe_id: action}
            result = simulate_hedge_common_state(
                trace,
                timeline=timeline,
                degraded_slowdown=config.degraded_slowdown,
                actions=plan,
                hedge_delay=tau0,
            )
            token = result.tokens[probe_id]
            samples[action].append(
                ProbeSample(
                    latency=token.latency,
                    replayed=any(a.attempt_id == 1 for a in token.attempts),
                    hedge_work=sum(
                        a.executed_work for a in token.attempts if a.attempt_id == 2
                    ),
                    replay_work=sum(
                        a.executed_work for a in token.attempts if a.attempt_id == 1
                    ),
                    wasted_work=sum(
                        a.executed_work
                        for a in token.attempts
                        if a.attempt_id != token.winner_attempt_id
                    ),
                )
            )
    return tuple(
        CalibrationCell(
            state=state,
            target_rho=target_rho,
            achieved_rho=bisection.achieved_rho,
            rho_error=bisection.rho_error,
            token_class=token_class,
            action=action,
            background_arrival_rate=rate,
            sample_count=len(samples[action]),
            rho_measurement_episode_count=rho_episodes,
            rho_background_cohort_count=cohort,
            probe_successful_episode_count=episodes_used,
            probe_generation_attempt_count=episodes_used + episodes_skipped,
            probe_skipped_episode_count=episodes_skipped,
            stats=aggregate_probe_samples(samples[action]),
        )
        for action in actions
    )


def build_calibration_table(
    config: ExperimentConfig,
    *,
    grid: Sequence[float] = RHO_GRID_DEFAULT,
    timeline: CommonStateTimeline = CALIBRATION_TIMELINE,
    samples_per_cell: int = MIN_SAMPLES_PER_CELL,
) -> CalibrationTable:
    """Production entry: full 70-cell table, >= 400 samples per cell."""
    if (
        isinstance(samples_per_cell, bool)
        or not isinstance(samples_per_cell, int)
        or samples_per_cell < MIN_SAMPLES_PER_CELL
    ):
        raise ValueError(
            f"production calibration requires at least {MIN_SAMPLES_PER_CELL} "
            f"samples per cell, got {samples_per_cell!r}"
        )
    from . import __version__

    rate_cache: dict = {}
    cells: list[CalibrationCell] = []
    for state in (CommonState.HEALTHY, CommonState.DEGRADED, CommonState.FAILED):
        for target_rho in grid:
            for token_class in (TokenClass.REGULAR, TokenClass.URGENT):
                cells.extend(
                    calibrate_cohort(
                        config,
                        state,
                        target_rho,
                        token_class,
                        timeline=timeline,
                        samples_per_cell=samples_per_cell,
                        rate_cache=rate_cache,
                    )
                )
    provenance = {
        "calibration_schema_version": 2,
        "seed_namespace": "calibration",
        "seed_derivation": "sha256-label-v1",
        "rho_grid": [float(point) for point in grid],
        "timeline": {
            "degraded_start": timeline.degraded_start,
            "failed_start": timeline.failed_start,
            "recovered_start": timeline.recovered_start,
        },
        "rate_bounds": [float(RATE_BOUNDS[0]), float(RATE_BOUNDS[1])],
        "rho_tolerance": RHO_TOLERANCE,
        "bisection_max_iterations": BISECTION_MAX_ITERATIONS,
        "samples_per_cell": samples_per_cell,
        "min_background_cohort": MIN_BACKGROUND_COHORT,
        "episode_horizon": "recovered_start",
        "config": {
            "healthy_service_mean": config.healthy_service_mean,
            "service_time_cv": config.service_time_cv,
            "degraded_slowdown": config.degraded_slowdown,
            "regular_token_ratio": config.regular_token_ratio,
            "urgent_token_ratio": config.urgent_token_ratio,
            "hedge_delay_quantile": config.hedge_delay_quantile,
            "healthy_offered_load": config.healthy_offered_load,
            "base_seed": config.base_seed,
        },
        "simulator_version": __version__,
    }
    table = CalibrationTable(
        cells=tuple(cells),
        expected_grid=tuple(float(point) for point in grid),
        require_complete=True,
        provenance=provenance,
    )
    validate_contrast(table)
    return table
