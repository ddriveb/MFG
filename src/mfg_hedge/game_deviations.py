"""Finite-system unilateral deviations for the isolated Shared-Backup game.

Every candidate in this module is a fresh N-Expert simulation from the same
immutable scenario inputs.  Only the deviating Expert's policy object is
replaced.  Shared queue state, speeds, timers, budgets, completions and
opponent observations are consequently endogenous to each rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

from .expert_game import (
    ExpertEpisodeRun,
    ExpertScore,
    PopulationScore,
    fault_timeline_fingerprint,
    score_population,
    shared_trace_fingerprint,
)
from .game_workload import ActionRule, NestedPopulationTrace, enumerate_action_rules
from .shared_backup import SharedAction, SharedBackupTrace, simulate_shared_backup


class GameRuleLike(Protocol):
    """Minimal online policy contract accepted by a finite rerun."""

    name: str

    def request(self, observation: object) -> SharedAction:
        ...


def _freeze(mapping: Mapping[object, object]) -> MappingProxyType:
    return MappingProxyType(dict(mapping))


def _policy_name(policy: object) -> str:
    name = getattr(policy, "name", None)
    if type(name) is not str or len(name) != 4:
        raise ValueError("every GameRule must expose a four-code name")
    if any(code not in "NDSX" for code in name):
        raise ValueError("GameRule names must contain only N, D, S, or X")
    request = getattr(policy, "request", None)
    if not callable(request):
        raise ValueError("every GameRule must expose an online request method")
    return name


def _rule_sort_key(name: str) -> tuple[int, int, int, int]:
    order = {"N": 0, "D": 1, "S": 2, "X": 3}
    return tuple(order[code] for code in name)  # type: ignore[return-value]


@dataclass(frozen=True)
class DeviationScenario:
    """Immutable common-fault and local-arrival input for one episode."""

    episode_key: str
    trace: SharedBackupTrace | NestedPopulationTrace
    fault_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.episode_key) is not str or not self.episode_key:
            raise ValueError("episode_key must be a non-empty string")
        source = self.trace
        if isinstance(source, NestedPopulationTrace):
            if self.fault_fingerprint is None:
                object.__setattr__(self, "fault_fingerprint", source.common_fault_fingerprint)
            source = source.trace
            object.__setattr__(self, "trace", source)
        if not isinstance(source, SharedBackupTrace):
            raise ValueError("trace must be SharedBackupTrace or NestedPopulationTrace")
        fingerprint = self.fault_fingerprint
        if fingerprint is None:
            fingerprint = fault_timeline_fingerprint(
                self.trace.timeline, self.trace.arrival_cutoff
            )
        if type(fingerprint) is not str or not fingerprint:
            raise ValueError("fault_fingerprint must be a non-empty string")
        object.__setattr__(self, "fault_fingerprint", fingerprint)


class _OnlineRuleSource:
    def __init__(self, policy: GameRuleLike):
        self._policy = policy

    def request(self, observation: object) -> SharedAction:
        action = self._policy.request(observation)
        try:
            normalized = action if isinstance(action, SharedAction) else SharedAction(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("GameRule returned an invalid SharedAction") from exc
        return normalized


def _episode_policy(policy: GameRuleLike) -> GameRuleLike:
    """Return an episode-private policy instance when the policy requests it.

    ``ActionRule`` and other immutable/stateless policies remain backwards
    compatible and are reused.  Stateful policies must expose the explicit
    ``new_episode()`` factory; silently sharing a mutable policy would make
    scenario order part of a counterfactual result.
    """

    factory = getattr(policy, "new_episode", None)
    if factory is None:
        return policy
    if not callable(factory):
        raise ValueError("GameRule.new_episode must be callable")
    episode_policy = factory()
    if episode_policy is policy:
        raise ValueError("GameRule.new_episode must return fresh episode state")
    _policy_name(episode_policy)
    return episode_policy


@dataclass(frozen=True)
class DeviationRow:
    """One immutable incumbent/candidate comparison and its externalities."""

    deviating_expert: int
    incumbent_rule: str
    candidate_rule: str
    base_objective: float
    deviated_objective: float
    improvement: float
    regret_contribution: float
    base_executed_work: float
    deviated_executed_work: float
    work_per_token: float
    waste_per_token: float
    replay_rate: float
    degraded_cvar95: float
    failed_cvar95: float
    social_before: float
    social_after: float
    social_delta: float
    peer_cost_delta: Mapping[int, float]
    trace_fingerprints: tuple[str, ...]
    deviated_trace_fingerprints: tuple[str, ...]
    base_fault_fingerprints: tuple[str, ...]
    deviated_fault_fingerprints: tuple[str, ...]
    base_pool_integral: float
    deviated_pool_integral: float
    base_invariant_counters: tuple[str, ...]
    deviated_invariant_counters: tuple[str, ...]
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "peer_cost_delta", _freeze(self.peer_cost_delta))
        if not self.complete:
            raise ValueError("incomplete deviation rows are not admissible")
        for name in (
            "base_objective", "deviated_objective", "improvement",
            "regret_contribution", "base_executed_work", "deviated_executed_work",
            "work_per_token", "waste_per_token", "replay_rate",
            "degraded_cvar95", "failed_cvar95", "social_before", "social_after",
            "social_delta", "base_pool_integral", "deviated_pool_integral",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 and name not in (
                "improvement", "social_delta"
            ):
                raise ValueError(f"deviation field {name} is not finite")

    @property
    def base_trace_fingerprint(self) -> str:
        return self.trace_fingerprints[0]

    @property
    def deviated_trace_fingerprint(self) -> str:
        return self.deviated_trace_fingerprints[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "deviating_expert": self.deviating_expert,
            "incumbent_rule": self.incumbent_rule,
            "candidate_rule": self.candidate_rule,
            "base_objective": self.base_objective,
            "deviated_objective": self.deviated_objective,
            "improvement": self.improvement,
            "regret_contribution": self.regret_contribution,
            "base_executed_work": self.base_executed_work,
            "deviated_executed_work": self.deviated_executed_work,
            "work_per_token": self.work_per_token,
            "waste_per_token": self.waste_per_token,
            "replay_rate": self.replay_rate,
            "degraded_cvar95": self.degraded_cvar95,
            "failed_cvar95": self.failed_cvar95,
            "social_before": self.social_before,
            "social_after": self.social_after,
            "social_delta": self.social_delta,
            "peer_cost_delta": dict(self.peer_cost_delta),
            "trace_fingerprints": list(self.trace_fingerprints),
            "deviated_trace_fingerprints": list(self.deviated_trace_fingerprints),
            "base_fault_fingerprints": list(self.base_fault_fingerprints),
            "deviated_fault_fingerprints": list(self.deviated_fault_fingerprints),
            "base_pool_integral": self.base_pool_integral,
            "deviated_pool_integral": self.deviated_pool_integral,
            "base_invariant_counters": list(self.base_invariant_counters),
            "deviated_invariant_counters": list(self.deviated_invariant_counters),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class DeviationEvaluation:
    """Deterministic candidate table and restricted point-estimate regret."""

    rows: tuple[DeviationRow, ...]
    base_population: PopulationScore
    candidate_bank_size: int
    deviating_expert: int | None
    best_response: str | None
    best_responses: Mapping[int, str]
    regret: float
    normalized_regret: float
    normalization_denominator: float
    regret_definition: str
    claims_nash: bool
    per_expert_regret: Mapping[int, float]
    per_expert_normalized_regret: Mapping[int, float]
    system_max_individual_regret: float

    @property
    def best_response_rule(self) -> str | None:
        return self.best_response

    @property
    def restricted_regret(self) -> float:
        return self.regret

    def __post_init__(self) -> None:
        object.__setattr__(self, "best_responses", _freeze(self.best_responses))
        object.__setattr__(self, "per_expert_regret", _freeze(self.per_expert_regret))
        object.__setattr__(
            self, "per_expert_normalized_regret",
            _freeze(self.per_expert_normalized_regret),
        )
        if self.regret < 0.0 or self.normalized_regret < 0.0:
            raise ValueError("restricted regret must be non-negative")
        if not math.isfinite(self.normalization_denominator) or self.normalization_denominator <= 0.0:
            raise ValueError("NNNN normalization denominator must be positive and finite")

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "candidate_bank_size": self.candidate_bank_size,
            "deviating_expert": self.deviating_expert,
            "best_response": self.best_response,
            "best_responses": dict(self.best_responses),
            "regret": self.regret,
            "normalized_regret": self.normalized_regret,
            "normalization_denominator": self.normalization_denominator,
            "regret_definition": self.regret_definition,
            "claims_nash": self.claims_nash,
            "per_expert_regret": dict(self.per_expert_regret),
            "per_expert_normalized_regret": dict(self.per_expert_normalized_regret),
            "system_max_individual_regret": self.system_max_individual_regret,
        }


@dataclass(frozen=True)
class TimingSmokeResult:
    """In-memory development timing check; it deliberately selects nothing."""

    runtime_seconds: float
    row_count: int
    all_complete: bool
    deterministic_repeat: bool
    selected_equilibrium: None = None


def _validate_scenarios(scenarios: Iterable[DeviationScenario]) -> tuple[DeviationScenario, ...]:
    materialized = tuple(scenarios)
    if not materialized or any(not isinstance(s, DeviationScenario) for s in materialized):
        raise ValueError("deviation evaluation requires nonempty scenarios")
    if len({s.episode_key for s in materialized}) != len(materialized):
        raise ValueError("duplicate scenario episode_key")
    counts = {s.trace.expert_count for s in materialized}
    if len(counts) != 1:
        raise ValueError("scenario Expert counts do not match")
    return tuple(sorted(materialized, key=lambda s: s.episode_key))


def _validate_profile(profile: Mapping[int, GameRuleLike], expert_count: int) -> dict[int, GameRuleLike]:
    if not isinstance(profile, Mapping):
        raise ValueError("profile must map every Expert ID to an online GameRule")
    expected = set(range(expert_count))
    if set(profile) != expected:
        raise ValueError("profile must contain exactly every Expert ID")
    result = dict(profile)
    for policy in result.values():
        _policy_name(policy)
    return result


def _rule(name: str) -> ActionRule:
    return ActionRule(name, tuple(name))


def _run_profile(
    scenarios: tuple[DeviationScenario, ...],
    profile: Mapping[int, GameRuleLike],
    *,
    c_b: float,
    degraded_slowdown: float,
    hedge_delay: float,
    call_observer: Callable[[], None] | None = None,
) -> tuple[ExpertEpisodeRun, ...]:
    runs = []
    for scenario in scenarios:
        sources = {
            expert_id: _OnlineRuleSource(_episode_policy(policy))
            for expert_id, policy in profile.items()
        }
        if call_observer is not None:
            call_observer()
        result = simulate_shared_backup(
            scenario.trace,
            c_b=c_b,
            degraded_slowdown=degraded_slowdown,
            hedge_delay=hedge_delay,
            action_sources=sources,
        )
        runs.append(ExpertEpisodeRun(
            episode_key=scenario.episode_key,
            trace=scenario.trace,
            result=result,
            fault_fingerprint=scenario.fault_fingerprint,
        ))
    return tuple(runs)


def _require_complete(population: PopulationScore) -> None:
    if not population.complete or population.social_cost_mean_expert_objective is None:
        missing = {
            score.expert_id: score.missing_cohorts
            for score in population.expert_scores if not score.complete
        }
        raise ValueError(f"deviation scoring requires complete Expert cohorts: {missing}")


def _pool_integral(runs: tuple[ExpertEpisodeRun, ...]) -> float:
    return math.fsum(
        interval.active_heads * (interval.end_time - interval.start_time)
        for run in runs for interval in run.result.pool_intervals
    )


def _counter_fingerprints(runs: tuple[ExpertEpisodeRun, ...]) -> tuple[str, ...]:
    return tuple(
        repr(run.result.counters)
        for run in runs
    )


def _make_row(
    scenarios: tuple[DeviationScenario, ...],
    base_runs: tuple[ExpertEpisodeRun, ...],
    base_population: PopulationScore,
    candidate_runs: tuple[ExpertEpisodeRun, ...],
    candidate_population: PopulationScore,
    deviating_expert: int,
    incumbent_name: str,
    candidate_name: str,
) -> DeviationRow:
    _require_complete(base_population)
    _require_complete(candidate_population)
    base_score = base_population.expert_scores[deviating_expert]
    candidate_score = candidate_population.expert_scores[deviating_expert]
    base_objective = base_score.total
    candidate_objective = candidate_score.total
    assert base_objective is not None and candidate_objective is not None
    peer_delta = {
        expert_id: candidate_population.expert_scores[expert_id].total
        - base_population.expert_scores[expert_id].total
        for expert_id in range(len(base_population.expert_scores))
        if expert_id != deviating_expert
    }
    generated = candidate_score.generated_token_count
    replay_rate = math.fsum(
        cohort.replay_count
        for cells in candidate_score.cohorts.values()
        for cohort in cells.values()
    ) / generated
    return DeviationRow(
        deviating_expert=deviating_expert,
        incumbent_rule=incumbent_name,
        candidate_rule=candidate_name,
        base_objective=base_objective,
        deviated_objective=candidate_objective,
        improvement=base_objective - candidate_objective,
        regret_contribution=max(0.0, base_objective - candidate_objective),
        base_executed_work=base_score.raw_work_totals["executed"],
        deviated_executed_work=candidate_score.raw_work_totals["executed"],
        work_per_token=candidate_score.components["work_per_token"],
        waste_per_token=candidate_score.components["waste_per_token"],
        replay_rate=replay_rate,
        degraded_cvar95=candidate_score.cvar95["D"],
        failed_cvar95=candidate_score.cvar95["F"],
        social_before=base_population.social_cost_mean_expert_objective,
        social_after=candidate_population.social_cost_mean_expert_objective,
        social_delta=(candidate_population.social_cost_mean_expert_objective
                      - base_population.social_cost_mean_expert_objective),
        peer_cost_delta=peer_delta,
        trace_fingerprints=tuple(shared_trace_fingerprint(s.trace) for s in scenarios),
        deviated_trace_fingerprints=tuple(shared_trace_fingerprint(s.trace) for s in scenarios),
        base_fault_fingerprints=tuple(s.fault_fingerprint for s in scenarios),
        deviated_fault_fingerprints=tuple(s.fault_fingerprint for s in scenarios),
        base_pool_integral=_pool_integral(base_runs),
        deviated_pool_integral=_pool_integral(candidate_runs),
        base_invariant_counters=_counter_fingerprints(base_runs),
        deviated_invariant_counters=_counter_fingerprints(candidate_runs),
        complete=True,
    )


def _normalization_denominator(
    scenarios: tuple[DeviationScenario, ...],
    expert_id: int,
    expert_count: int,
    *,
    c_b: float,
    degraded_slowdown: float,
    hedge_delay: float,
) -> float:
    normal_profile = {expert: _rule("NNNN") for expert in range(expert_count)}
    normal_runs = _run_profile(
        scenarios, normal_profile, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    normal_population = score_population(normal_runs)
    _require_complete(normal_population)
    value = normal_population.expert_scores[expert_id].total
    assert value is not None
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("same-trace NNNN normalization denominator must be positive")
    return value


def evaluate_unilateral_deviations(
    scenarios: Iterable[DeviationScenario],
    *,
    profile: Mapping[int, GameRuleLike],
    deviating_expert: int,
    candidates: Iterable[GameRuleLike],
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
) -> DeviationEvaluation:
    """Evaluate a candidate subset with a complete fresh N-Expert rerun."""

    materialized = _validate_scenarios(scenarios)
    expert_count = materialized[0].trace.expert_count
    profile_map = _validate_profile(profile, expert_count)
    if type(deviating_expert) is not int or deviating_expert not in profile_map:
        raise ValueError("deviating_expert is outside the profile")
    candidate_map = {}
    for candidate in tuple(candidates):
        name = _policy_name(candidate)
        if name in candidate_map:
            raise ValueError("candidate rules must have unique lexical names")
        candidate_map[name] = candidate
    if not candidate_map:
        raise ValueError("candidate subset must be nonempty")
    incumbent_name = _policy_name(profile_map[deviating_expert])
    if incumbent_name not in candidate_map:
        raise ValueError("candidate subset must include the incumbent rule")
    ordered_candidates = tuple(
        candidate_map[name]
        for name in sorted(candidate_map, key=_rule_sort_key)
    )

    base_runs = _run_profile(
        materialized, profile_map, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    base_population = score_population(base_runs)
    _require_complete(base_population)
    rows = []
    for candidate in ordered_candidates:
        candidate_profile = dict(profile_map)
        candidate_profile[deviating_expert] = candidate
        candidate_runs = _run_profile(
            materialized, candidate_profile, c_b=c_b,
            degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
        )
        candidate_population = score_population(candidate_runs)
        rows.append(_make_row(
            materialized, base_runs, base_population, candidate_runs,
            candidate_population, deviating_expert, incumbent_name,
            _policy_name(candidate),
        ))
    rows = tuple(sorted(rows, key=lambda row: _rule_sort_key(row.candidate_rule)))
    best = min(rows, key=lambda row: (
        row.deviated_objective, row.deviated_executed_work,
        _rule_sort_key(row.candidate_rule),
    ))
    regret = max(0.0, base_population.expert_scores[deviating_expert].total
                 - best.deviated_objective)
    denominator = _normalization_denominator(
        materialized, deviating_expert, expert_count, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    return DeviationEvaluation(
        rows=rows,
        base_population=base_population,
        candidate_bank_size=len(rows),
        deviating_expert=deviating_expert,
        best_response=best.candidate_rule,
        best_responses={deviating_expert: best.candidate_rule},
        regret=regret,
        normalized_regret=regret / denominator,
        normalization_denominator=denominator,
        regret_definition="Pi256 restricted point estimate",
        claims_nash=False,
        per_expert_regret={deviating_expert: regret},
        per_expert_normalized_regret={deviating_expert: regret / denominator},
        system_max_individual_regret=regret,
    )


def _validate_full_bank(candidates: Iterable[GameRuleLike] | None) -> tuple[GameRuleLike, ...]:
    expected = enumerate_action_rules()
    if candidates is None:
        return expected
    materialized = tuple(candidates)
    if len(materialized) != len(expected):
        raise ValueError("Pi256 candidate bank must contain exactly 256 rules")
    names = tuple(_policy_name(candidate) for candidate in materialized)
    expected_names = tuple(rule.name for rule in expected)
    if len(set(names)) != 256 or set(names) != set(expected_names):
        raise ValueError("Pi256 candidate bank must be complete and unique")
    if names != expected_names:
        raise ValueError("Pi256 candidate bank must use canonical lexical order")
    return materialized


def evaluate_pi256(
    scenarios: Iterable[DeviationScenario],
    *,
    profile: Mapping[int, GameRuleLike],
    candidates: Iterable[GameRuleLike] | None = None,
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
) -> DeviationEvaluation:
    """Run the official all-Expert, complete canonical Pi256 bank."""

    materialized = _validate_scenarios(scenarios)
    bank = _validate_full_bank(candidates)
    all_rows = []
    evaluations = []
    for expert_id in range(materialized[0].trace.expert_count):
        evaluation = evaluate_unilateral_deviations(
            materialized, profile=profile, deviating_expert=expert_id,
            candidates=bank, c_b=c_b, degraded_slowdown=degraded_slowdown,
            hedge_delay=hedge_delay,
        )
        evaluations.append(evaluation)
        all_rows.extend(evaluation.rows)
    rows = tuple(sorted(
        all_rows,
        key=lambda row: (row.deviating_expert, _rule_sort_key(row.candidate_rule)),
    ))
    regrets = {evaluation.deviating_expert: evaluation.regret for evaluation in evaluations}
    best_responses = {
        evaluation.deviating_expert: evaluation.best_response
        for evaluation in evaluations
    }
    max_regret = max(regrets.values())
    denominators = {
        evaluation.deviating_expert: evaluation.normalization_denominator
        for evaluation in evaluations
    }
    normalized_regrets = {
        expert_id: regrets[expert_id] / denominators[expert_id]
        for expert_id in regrets
    }
    denominator = denominators[min(denominators)]
    return DeviationEvaluation(
        rows=rows,
        base_population=evaluations[0].base_population,
        candidate_bank_size=256,
        deviating_expert=None,
        best_response=None,
        best_responses=best_responses,
        regret=max_regret,
        normalized_regret=max(normalized_regrets.values()),
        normalization_denominator=denominator,
        regret_definition="Pi256 restricted point estimate",
        claims_nash=False,
        per_expert_regret=regrets,
        per_expert_normalized_regret=normalized_regrets,
        system_max_individual_regret=max_regret,
    )


def timing_smoke_pi256(
    scenarios: Iterable[DeviationScenario],
    *,
    profile: Mapping[int, GameRuleLike],
    c_b: float = 0.5,
    degraded_slowdown: float = 2.0,
    hedge_delay: float = 1.5,
) -> TimingSmokeResult:
    """Run and repeat the tiny official bank entirely in memory."""

    materialized = _validate_scenarios(scenarios)
    started = time.perf_counter()
    first = evaluate_pi256(
        materialized, profile=profile, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    second = evaluate_pi256(
        materialized, profile=profile, c_b=c_b,
        degraded_slowdown=degraded_slowdown, hedge_delay=hedge_delay,
    )
    elapsed = time.perf_counter() - started
    return TimingSmokeResult(
        runtime_seconds=elapsed,
        row_count=len(first.rows),
        all_complete=all(row.complete for row in first.rows),
        deterministic_repeat=first == second,
    )


# Descriptive aliases keep the lower-level and official entry points easy to
# discover without creating a second implementation or changing semantics.
evaluate_finite_deviations = evaluate_unilateral_deviations
evaluate_all_pi256 = evaluate_pi256
evaluate_restricted_best_response = evaluate_unilateral_deviations


__all__ = [
    "DeviationEvaluation",
    "DeviationRow",
    "DeviationScenario",
    "GameRuleLike",
    "TimingSmokeResult",
    "evaluate_pi256",
    "evaluate_all_pi256",
    "evaluate_finite_deviations",
    "evaluate_restricted_best_response",
    "evaluate_unilateral_deviations",
    "timing_smoke_pi256",
]
