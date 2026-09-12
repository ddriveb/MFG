"""Exact finite-system development search over ADR-0013's 81 fixed rules."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .artifacts import write_run_directory
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_macro_seed_episodes,
)
from .common_state import Phase, phase_at
from .config import ExperimentConfig
from .domain import ProtectionAction as Action
from .metrics import percentile
from .paired import compute_tau0
from .transient_control import (
    ReservationParameters,
    TimeClassRule,
    simulate_transient_episode,
)
from .transient_objective import build_transient_objective


DEVELOPMENT_NAMESPACE = "transient-control:v1:fit"
DEVELOPMENT_MACRO_SEED = 20260905
DEVELOPMENT_EPISODE_COUNT = 64
_ACTIONS = (Action.NORMAL, Action.DELAYED_HEDGE, Action.IMMEDIATE_HEDGE)


def enumerate_time_class_rules() -> tuple[TimeClassRule, ...]:
    """Return the complete frozen bank in lexical N < D < I order."""
    return tuple(TimeClassRule(*actions) for actions in itertools.product(_ACTIONS, repeat=4))


def _finite_positive(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def assess_development_safety(candidate: Mapping, baseline: Mapping) -> dict:
    """Apply the frozen prospective filters using unrounded ratios."""
    ratios = {}
    for phase in ("H", "R"):
        for metric in ("mean", "p95"):
            denominator = _finite_positive(baseline[phase][metric], f"baseline {phase} {metric}")
            numerator = _finite_positive(candidate[phase][metric], f"candidate {phase} {metric}")
            ratios[f"{phase}_{metric}_ratio"] = numerator / denominator
    ratios["total_work_ratio"] = _finite_positive(candidate["work"], "candidate work") / \
        _finite_positive(baseline["work"], "baseline work")
    limits = {"H_mean_ratio": 1.01, "H_p95_ratio": 1.02,
              "R_mean_ratio": 1.01, "R_p95_ratio": 1.02,
              "total_work_ratio": 1.05}
    reasons = [name for name in limits if ratios[name] > limits[name]]
    return {"feasible": not reasons, "reasons": reasons, "ratios": ratios,
            "limits": limits}


def _resource_summary(runs) -> dict:
    timeline = runs[0].episode.protocol.timeline
    tokens = tuple(token for run in runs for token in run.simulation.tokens)
    result = {}
    for phase in (Phase.HEALTHY, Phase.RECOVERED):
        values = [token.latency for token in tokens
                  if phase_at(timeline, token.arrival_time) is phase]
        if not values:
            raise ValueError(f"missing {phase.value} arrivals for safety metric")
        result[phase.value] = {"sample_count": len(values),
                               "mean": math.fsum(values) / len(values),
                               "p95": percentile(values, 95)}
    result["work"] = math.fsum(
        attempt.executed_work for run in runs for attempt in run.simulation.attempts)
    if not math.isfinite(result["work"]) or result["work"] <= 0:
        raise ValueError("total executed work must be finite and positive")
    return result


def _admission_summary(results) -> dict:
    audits = [result.plan.audit()["global"] for result in results]
    names = ("token_count", "requested_hedges", "applied_hedges",
             "quota_suppressed", "reserved_work")
    aggregate = {name: math.fsum(a[name] for a in audits) if name == "reserved_work"
                 else sum(a[name] for a in audits) for name in names}
    aggregate["requested"] = {action.value: sum(a["requested"][action.value] for a in audits)
                               for action in _ACTIONS}
    aggregate["applied"] = {action.value: sum(a["applied"][action.value] for a in audits)
                             for action in _ACTIONS}
    if aggregate["requested_hedges"] != (
            aggregate["applied_hedges"] + aggregate["quota_suppressed"]):
        raise RuntimeError("aggregate reservation identity failed")
    return aggregate


def _validate_episode_bank(episodes: Iterable[EpisodeTrace], require_frozen_fit: bool):
    rows = tuple(episodes)
    if not rows or any(not isinstance(episode, EpisodeTrace) for episode in rows):
        raise ValueError("episode bank must be nonempty and contain EpisodeTrace values")
    keys = [episode.identity.key for episode in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode identity in rule search")
    rows = tuple(sorted(rows, key=lambda e: e.identity.episode_index))
    first = rows[0]
    if first.identity.namespace not in (DEVELOPMENT_NAMESPACE, "transient-control:v1:test"):
        raise ValueError("rule search accepts fit or test namespace only")
    for episode in rows:
        if episode.identity.namespace != first.identity.namespace:
            raise ValueError("all episodes must share one namespace")
        if episode.identity.macro_seed != first.identity.macro_seed:
            raise ValueError("all episodes must share one macro_seed")
        if episode.protocol != first.protocol:
            raise ValueError("all episodes must share one protocol")
    if require_frozen_fit:
        if len(rows) != DEVELOPMENT_EPISODE_COUNT:
            raise ValueError(f"frozen fit requires exactly {DEVELOPMENT_EPISODE_COUNT} episodes")
        if (first.identity.namespace != DEVELOPMENT_NAMESPACE
                or first.identity.macro_seed != DEVELOPMENT_MACRO_SEED
                or first.protocol != ATTRIBUTION_V1_PROTOCOL
                or [e.identity.episode_index for e in rows]
                != list(range(DEVELOPMENT_EPISODE_COUNT))):
            raise ValueError("episode bank does not match frozen ADR-0013 fit identity")
        if any(not math.isclose(e.workload.arrival_rate, .9, rel_tol=0, abs_tol=1e-12)
               for e in rows):
            raise ValueError("frozen fit requires arrival_rate 0.9 (offered load 0.45)")
    return rows


def _rule_name(rule: TimeClassRule) -> str:
    return "".join(action.value for action in rule.actions)


def evaluate_rule_bank(
    episodes: Iterable[EpisodeTrace], *, hedge_delay: float,
    reservation: ReservationParameters | None = None,
    require_frozen_fit: bool = False,
) -> dict:
    """Evaluate complete rules on identical exogenous episodes and select once."""
    bank = _validate_episode_bank(episodes, require_frozen_fit)
    delay = _finite_positive(hedge_delay, "hedge_delay")
    reservation = ReservationParameters() if reservation is None else reservation
    if not isinstance(reservation, ReservationParameters):
        raise ValueError("reservation must be ReservationParameters")
    rules = enumerate_time_class_rules()
    fingerprints = {e.identity.key: episode_trace_fingerprint(e) for e in bank}
    rows = []
    baseline_resources = None
    for lexical_index, rule in enumerate(rules):
        observed = tuple(simulate_transient_episode(
            episode, rule, reservation, degraded_slowdown=2, hedge_delay=delay)
                         for episode in bank)
        if any(result.run.episode != episode for result, episode in zip(observed, bank)):
            raise RuntimeError("rule evaluation changed or reordered the exogenous trace")
        runs = tuple(result.run for result in observed)
        objective = build_transient_objective(runs)
        resources = _resource_summary(runs)
        if lexical_index == 0:
            baseline_resources = resources
        safety = assess_development_safety(resources, baseline_resources)
        complete = objective["status"] == "complete" and objective["total"] is not None
        row = {"rule": _rule_name(rule), "lexical_index": lexical_index,
               "actions": {"early_regular": rule.early_regular.value,
                           "early_urgent": rule.early_urgent.value,
                           "late_regular": rule.late_regular.value,
                           "late_urgent": rule.late_urgent.value},
               "objective": objective, "resources": resources,
               "safety": safety, "objective_complete": complete,
               "feasible": complete and safety["feasible"],
               "admission": _admission_summary(observed)}
        rows.append(row)
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("no complete development-feasible rule, including NNNN")
    selected = min(feasible, key=lambda row: (
        row["objective"]["total"], row["resources"]["work"], row["lexical_index"]))
    baseline = rows[0]
    comparison = {
        "objective_delta": selected["objective"]["total"] - baseline["objective"]["total"],
        "objective_ratio": selected["objective"]["total"] / baseline["objective"]["total"],
        "total_work_ratio": selected["resources"]["work"] / baseline["resources"]["work"],
        "D_cvar95_ratio": (selected["objective"]["fault_tails"]["D"]["cvar95_fractional_tail"]
                           / baseline["objective"]["fault_tails"]["D"]["cvar95_fractional_tail"]),
        "F_cvar95_ratio": (selected["objective"]["fault_tails"]["F"]["cvar95_fractional_tail"]
                           / baseline["objective"]["fault_tails"]["F"]["cvar95_fractional_tail"]),
    }
    return {"schema_version": 1, "scenario": "transient_control_81_rule_development_fit",
            "claim_boundary": "sample_best_feasible_rule_in_frozen_81_rule_class",
            "namespace": bank[0].identity.namespace,
            "macro_seed": bank[0].identity.macro_seed,
            "episode_count": len(bank), "rule_count": len(rows),
            "baseline_rule": "NNNN", "selected_rule": selected["rule"],
            "feasible_rule_count": len(feasible),
            "selection_order": ["objective_total", "total_executed_work", "lexical_N_D_I"],
            "reservation": {name: getattr(reservation, name)
                            for name in reservation.__dataclass_fields__},
            "hedge_delay": delay, "trace_fingerprints": fingerprints,
            "selected_vs_baseline": comparison, "rules": rows}


def run_frozen_development_search(config: ExperimentConfig) -> dict:
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be ExperimentConfig")
    if not math.isclose(config.healthy_offered_load, .45, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ADR-0013 frozen fit requires healthy_offered_load 0.45")
    episodes = generate_macro_seed_episodes(
        config, DEVELOPMENT_NAMESPACE, DEVELOPMENT_MACRO_SEED,
        DEVELOPMENT_EPISODE_COUNT, ATTRIBUTION_V1_PROTOCOL)
    return evaluate_rule_bank(episodes, hedge_delay=compute_tau0(config),
                              reservation=ReservationParameters(), require_frozen_fit=True)


def rule_bank_sha256() -> str:
    payload = json.dumps([_rule_name(rule) for rule in enumerate_time_class_rules()],
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_development_search_artifact(
    artifacts_root: str | Path, run_id: str, manifest: Mapping, search: Mapping,
) -> Path:
    rows = search.get("rules", ())
    expected_names = [_rule_name(rule) for rule in enumerate_time_class_rules()]
    observed_names = [row.get("rule") for row in rows if isinstance(row, Mapping)]
    if (search.get("rule_count") != 81 or len(rows) != 81
            or observed_names != expected_names or len(set(observed_names)) != 81):
        raise ValueError("development artifact requires the complete unique 81-rule bank in lexical order")
    selected = next(row for row in search["rules"] if row["rule"] == search["selected_rule"])
    baseline = next(row for row in search["rules"] if row["rule"] == "NNNN")
    manifest_payload = dict(manifest)
    manifest_payload["run_id"] = run_id
    summary = {key: search[key] for key in (
        "schema_version", "scenario", "claim_boundary", "namespace", "macro_seed",
        "episode_count", "rule_count", "baseline_rule", "selected_rule",
        "feasible_rule_count", "selection_order", "reservation", "hedge_delay",
        "selected_vs_baseline")}
    summary["selected"] = selected
    summary["baseline"] = baseline
    return write_run_directory(artifacts_root, run_id, {
        "manifest.json": manifest_payload,
        "rule_results.json": {"trace_fingerprints": search["trace_fingerprints"],
                              "rules": search["rules"]},
        "summary.json": summary,
    })


def frozen_manifest(config: ExperimentConfig, config_sha256: str, simulator_version: str) -> dict:
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ValueError("config_sha256 must be a 64-character digest")
    return {"scenario": "transient_control_81_rule_development_fit",
            "development_only": True, "qualification_or_holdout": False,
            "simulator_version": simulator_version,
            "config_sha256": config_sha256,
            "resolved_config": {field.name: getattr(config, field.name) for field in fields(config)},
            "namespace": DEVELOPMENT_NAMESPACE, "macro_seed": DEVELOPMENT_MACRO_SEED,
            "episode_indices": list(range(DEVELOPMENT_EPISODE_COUNT)),
            "episode_count": DEVELOPMENT_EPISODE_COUNT,
            "protocol": {"degraded_start": 100., "failed_start": 200.,
                         "recovered_start": 220., "arrival_cutoff": 320.},
            "rule_bank_sha256": rule_bank_sha256(),
            "rule_order": "early_regular,early_urgent,late_regular,late_urgent; N<D<I",
            "safety_limits": {"H_mean_ratio": 1.01, "H_p95_ratio": 1.02,
                              "R_mean_ratio": 1.01, "R_p95_ratio": 1.02,
                              "total_work_ratio": 1.05}}
