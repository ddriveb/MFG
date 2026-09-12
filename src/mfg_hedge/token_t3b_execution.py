"""T3B conditional candidate holdout qualification.

The execution is intentionally a tagged finite-token diagnostic.  It runs the
complete online physics for every branch, but only the selected Token is
reported and the candidate is never installed as a population policy.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable, Mapping

from .artifacts import write_run_directory
from .attribution_episode import (
    ATTRIBUTION_V1_PROTOCOL,
    EpisodeTrace,
    episode_trace_fingerprint,
    generate_episode_trace,
)
from .config import ExperimentConfig, load_config
from .domain import ProtectionAction
from .token_best_response import (
    CANDIDATE_LABEL,
    CandidateTable,
    CandidateTableError,
    load_t3a_runtime_candidate,
)
from .token_continuation import BIN_SCHEMA_V1, observation_fingerprint
from .token_deviations import evaluate_token_pathwise_deviations
from .token_payoff import ExternalQuote, TokenRuntimeADR0009Parameters
from .token_t3a_execution import (
    ENVIRONMENT_ID,
    POPULATION_POLICY_ID,
    TARGET_SELECTION_ID,
    T3AAnchorSelector,
    _RESERVATION,
    _select_target,
    build_provenance,
    build_source_bundle,
    environment_fingerprint,
    policy_factory,
    population_policy_fingerprint,
    validate_frozen_config,
)


T3B_LABEL = "conditional_best_response_candidate_with_holdout_diagnostics"
QUALIFICATION_NAMESPACE = "token-mfg-restoration:t3b:qualification:v1"
QUALIFICATION_MACRO_SEED = 20260911
QUALIFICATION_EPISODES = 2048
QUALIFICATION_CALL_LIMIT = 10_240
PANEL_FLOOR = 32
DEGRADED_SLOWDOWN = 2.0
HEDGE_DELAY = 2.0
_ACTIONS = tuple(ProtectionAction)
_RUNTIME = TokenRuntimeADR0009Parameters()
_RUNTIME_QUOTE = ExternalQuote(0.0, "incremental_executed_work")
_Z95 = NormalDist().inv_cdf(0.95)


class T3BExecutionError(RuntimeError):
    """Fail-closed T3B protocol violation."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def qualification_call_budget(episode_count: int) -> int:
    if type(episode_count) is not int or episode_count < 0:
        raise T3BExecutionError("episode_count must be a non-negative int")
    return episode_count * 5


def _stats(values: Iterable[float]) -> dict[str, object]:
    data = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in data):
        raise T3BExecutionError("statistics received a non-finite value")
    n = len(data)
    if n == 0:
        return {"mean": None, "standard_error": None, "effective_n": 0}
    mean = math.fsum(data) / n
    if n == 1:
        se = 0.0
    else:
        se = math.sqrt(math.fsum((value - mean) ** 2 for value in data) / (n * (n - 1)))
    return {"mean": mean, "standard_error": se, "effective_n": n}


def paired_mean_se(values: Iterable[float]) -> dict[str, object]:
    """Return mean/SE for paired episode-level differences."""

    return _stats(values)


def _action_key(action: ProtectionAction) -> str:
    return action.value


def summarize_bin(
    bin_id: str,
    *,
    construction_action: ProtectionAction,
    costs: Mapping[ProtectionAction, Iterable[float]],
    panel_floor: int = PANEL_FLOOR,
    missing_target_count: int = 0,
    failed_panel_count: int = 0,
) -> dict[str, object]:
    if not isinstance(construction_action, ProtectionAction):
        raise T3BExecutionError("construction_action must be ProtectionAction")
    if type(panel_floor) is not int or panel_floor <= 0:
        raise T3BExecutionError("panel_floor must be a positive int")
    action_stats = {action: _stats(costs.get(action, ())) for action in _ACTIONS}
    available = [
        action
        for action in _ACTIONS
        if action_stats[action]["effective_n"] >= panel_floor
    ]
    sufficient = all(action in available for action in _ACTIONS)
    descriptive = None
    if sufficient:
        descriptive = min(
            _ACTIONS,
            key=lambda action: (action_stats[action]["mean"], _ACTIONS.index(action)),
        )
    paired: dict[str, object] = {}
    candidate_values = tuple(costs.get(construction_action, ()))
    for alternative in _ACTIONS:
        if alternative is construction_action:
            continue
        alt_values = tuple(costs.get(alternative, ()))
        if len(alt_values) != len(candidate_values):
            raise T3BExecutionError("paired candidate/alternative sample lengths differ")
        difference = paired_mean_se(
            candidate - alternative for candidate, alternative in zip(candidate_values, alt_values)
        )
        difference["upper95_diagnostic"] = (
            None
            if difference["mean"] is None
            else difference["mean"] + _Z95 * difference["standard_error"]
        )
        paired[_action_key(alternative)] = difference
    candidate_stats = action_stats[construction_action]
    candidate_upper = (
        None
        if not sufficient
        else candidate_stats["mean"] + _Z95 * candidate_stats["standard_error"]
    )
    return {
        "bin_id": bin_id,
        "construction_candidate_action": construction_action.value,
        "holdout": {
            _action_key(action): action_stats[action] for action in _ACTIONS
        },
        "holdout_descriptive_minimum": None if descriptive is None else descriptive.value,
        "candidate_is_holdout_descriptive_minimum": (
            descriptive is not None and descriptive is construction_action
        ),
        "paired_deltas_candidate_minus_alternative": paired,
        "candidate_max_upper95": candidate_upper,
        "missing_target_count": missing_target_count,
        "failed_panel_count": failed_panel_count,
        "insufficient": not sufficient,
    }


def _trace_fingerprint(episode: EpisodeTrace) -> str:
    return _digest(episode_trace_fingerprint(episode))


def _episode_factory(config: ExperimentConfig) -> Callable[[int], EpisodeTrace]:
    def create(index: int) -> EpisodeTrace:
        return generate_episode_trace(
            config,
            QUALIFICATION_NAMESPACE,
            QUALIFICATION_MACRO_SEED,
            index,
            ATTRIBUTION_V1_PROTOCOL,
        )

    return create


def _action_costs(physical: object) -> dict[ProtectionAction, float]:
    rows = getattr(physical, "rows", ())
    by_action = {row.candidate_requested: row for row in rows}
    if set(by_action) != set(_ACTIONS):
        raise T3BExecutionError("physical panel does not contain exactly N/D/I")
    costs = {action: float(by_action[action].candidate_payoff.total_cost) for action in _ACTIONS}
    if any(not math.isfinite(value) for value in costs.values()):
        raise T3BExecutionError("physical panel returned a non-finite runtime cost")
    return costs


def run_qualification(
    config: ExperimentConfig,
    candidate: CandidateTable,
    *,
    episode_count: int = QUALIFICATION_EPISODES,
    episode_builder: Callable[[int], EpisodeTrace] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Run the independent holdout library and return an in-memory result."""

    validate_frozen_config(config)
    if not isinstance(candidate, CandidateTable):
        raise T3BExecutionError("candidate must be CandidateTable")
    if type(episode_count) is not int or not 0 < episode_count <= QUALIFICATION_EPISODES:
        raise T3BExecutionError("episode_count must be in 1..2048")
    if qualification_call_budget(episode_count) > QUALIFICATION_CALL_LIMIT:
        raise T3BExecutionError("qualification call budget exceeded")
    build = episode_builder or _episode_factory(config)
    seen: set[str] = set()
    costs: dict[str, dict[ProtectionAction, list[float]]] = {
        bin_id: {action: [] for action in _ACTIONS}
        for bin_id in candidate.retained_bins
    }
    missing_by_bin = {bin_id: 0 for bin_id in candidate.retained_bins}
    failed_by_bin = {bin_id: 0 for bin_id in candidate.retained_bins}
    audits: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    attempted_calls = 0
    for index in range(episode_count):
        episode = build(index)
        if not isinstance(episode, EpisodeTrace):
            raise T3BExecutionError("episode_builder returned a non-EpisodeTrace")
        episode_fp = _trace_fingerprint(episode)
        if episode_fp in seen:
            raise T3BExecutionError("duplicate qualification episode fingerprint")
        seen.add(episode_fp)
        attempted_calls += 1
        selector = T3AAnchorSelector(index % 4)
        try:
            target_id, observation = _select_target(episode, selector)
        except Exception as error:
            audits.append(
                {
                    "episode_index": index,
                    "anchor_id": index % 4,
                    "status": "selection_failed",
                    "bin_id": None,
                    "target_token_id": None,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": None,
                    "attempted_calls": 1,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if target_id is None or observation is None:
            audits.append(
                {
                    "episode_index": index,
                    "anchor_id": index % 4,
                    "status": "missing_target",
                    "bin_id": None,
                    "target_token_id": None,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": None,
                    "attempted_calls": 1,
                    "reason": None,
                }
            )
            continue
        selected_fp = observation_fingerprint(observation)
        try:
            bin_id = BIN_SCHEMA_V1.bin(observation).bin_id
        except Exception as error:
            audits.append(
                {
                    "episode_index": index,
                    "anchor_id": index % 4,
                    "status": "missing_bin",
                    "bin_id": None,
                    "target_token_id": target_id,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": selected_fp,
                    "attempted_calls": 1,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        try:
            physical = evaluate_token_pathwise_deviations(
                episode,
                policy_factory,
                target_id,
                candidates=_ACTIONS,
                parameters=_RUNTIME,
                quote=_RUNTIME_QUOTE,
                reservation=_RESERVATION,
                degraded_slowdown=DEGRADED_SLOWDOWN,
                hedge_delay=HEDGE_DELAY,
            )
            attempted_calls += int(physical.attempted_calls)
        except Exception as error:
            failed_by_bin[bin_id] = failed_by_bin.get(bin_id, 0) + 1
            audits.append(
                {
                    "episode_index": index,
                    "anchor_id": index % 4,
                    "status": "physical_failed",
                    "bin_id": bin_id,
                    "target_token_id": target_id,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": selected_fp,
                    "attempted_calls": 1,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if physical.status != "completed" or len(physical.rows) != 3:
            failed_by_bin[bin_id] = failed_by_bin.get(bin_id, 0) + 1
            audits.append(
                {
                    "episode_index": index,
                    "anchor_id": index % 4,
                    "status": "physical_failed",
                    "bin_id": bin_id,
                    "target_token_id": target_id,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": selected_fp,
                    "attempted_calls": 1 + physical.attempted_calls,
                    "reason": physical.failure_reason or "paired physical panel failed",
                }
            )
            continue
        prefix = physical.rows[0].baseline_prefix_fingerprint
        if any(
            row.baseline_target_observation_fingerprint != selected_fp
            or row.candidate_target_observation_fingerprint != selected_fp
            or row.baseline_prefix_fingerprint != prefix
            or row.candidate_prefix_fingerprint != prefix
            for row in physical.rows
        ):
            raise T3BExecutionError("same target prefix/CRN was not preserved across N/D/I")
        panel_costs = _action_costs(physical)
        if bin_id in costs:
            for action, value in panel_costs.items():
                costs[bin_id][action].append(value)
            construction_action = candidate.action_for_bin(bin_id)
            rows.append(
                {
                    "episode_index": index,
                    "bin_id": bin_id,
                    "target_token_id": target_id,
                    "episode_fingerprint": episode_fp,
                    "observation_fingerprint": selected_fp,
                    "construction_candidate_action": construction_action.value,
                    "costs": {action.value: panel_costs[action] for action in _ACTIONS},
                    "input_fingerprint": physical.input_fingerprint,
                    "baseline_prefix_fingerprint": prefix,
                    "candidate_prefix_fingerprints": {
                        row.candidate_requested.value: row.candidate_prefix_fingerprint
                        for row in physical.rows
                    },
                    "paired_deltas_candidate_minus_alternative": {
                        action.value: panel_costs[construction_action] - panel_costs[action]
                        for action in _ACTIONS
                        if action is not construction_action
                    },
                }
            )
            status = "completed"
        else:
            status = "out_of_scope_bin"
        audits.append(
            {
                "episode_index": index,
                "anchor_id": index % 4,
                "status": status,
                "bin_id": bin_id,
                "target_token_id": target_id,
                "episode_fingerprint": episode_fp,
                "observation_fingerprint": selected_fp,
                "attempted_calls": 1 + physical.attempted_calls,
                "reason": None,
            }
        )
        if progress is not None and ((index + 1) % 64 == 0 or index + 1 == episode_count):
            progress(index + 1, episode_count)
    if attempted_calls > qualification_call_budget(episode_count) or attempted_calls > QUALIFICATION_CALL_LIMIT:
        raise T3BExecutionError("actual qualification calls exceeded frozen budget")
    summaries = []
    for bin_id in candidate.retained_bins:
        summaries.append(
            summarize_bin(
                bin_id,
                construction_action=candidate.action_for_bin(bin_id),
                costs=costs[bin_id],
                panel_floor=PANEL_FLOOR,
                missing_target_count=missing_by_bin[bin_id],
                failed_panel_count=failed_by_bin[bin_id],
            )
        )
    all_sufficient = all(not row["insufficient"] for row in summaries)
    return {
        "label": T3B_LABEL,
        "candidate_label": CANDIDATE_LABEL,
        "status": "completed" if all_sufficient else "insufficient",
        "episode_count": episode_count,
        "attempted_calls": attempted_calls,
        "max_scheduler_calls": qualification_call_budget(episode_count),
        "retained_bin_count": len(candidate.retained_bins),
        "completed_panels": len(rows),
        "missing_target_count": sum(row["status"] == "missing_target" for row in audits),
        "failed_panel_count": sum(row["status"] == "physical_failed" for row in audits),
        "out_of_scope_panel_count": sum(row["status"] == "out_of_scope_bin" for row in audits),
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "simultaneous_bound": False,
        "epsilon_nash": None,
        "bins": summaries,
        "qualification_rows": rows,
        "episode_audit": audits,
    }


def _parent_payload(validation_dir: Path, occupancy_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        validation = json.loads((validation_dir / "manifest.json").read_text(encoding="utf-8"))
        occupancy = json.loads((occupancy_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise T3BExecutionError(f"cannot read T3A parent manifests: {error}") from error
    if not isinstance(validation, dict) or not isinstance(occupancy, dict):
        raise T3BExecutionError("T3A parent manifests must be objects")
    return validation, occupancy


def write_qualification_artifact(
    artifacts_root: str | Path,
    run_id: str,
    result: Mapping[str, object],
    candidate: CandidateTable,
    *,
    source_bundle: Mapping[str, object],
    parent_validation_manifest: Mapping[str, object],
    parent_occupancy_manifest: Mapping[str, object],
    config: ExperimentConfig,
) -> Path:
    if result.get("label") != T3B_LABEL or any(
        result.get(name) is not False
        for name in ("claims_best_response", "claims_regret", "claims_nash", "claims_mfg")
    ):
        raise T3BExecutionError("qualification result claim boundary is invalid")
    actual_calls = result.get("attempted_calls")
    if type(actual_calls) is not int or actual_calls > QUALIFICATION_CALL_LIMIT:
        raise T3BExecutionError("qualification call accounting is invalid")
    provenance = {
        "environment_id": ENVIRONMENT_ID,
        "environment_fingerprint": environment_fingerprint(config),
        "population_policy_id": POPULATION_POLICY_ID,
        "population_policy_fingerprint": population_policy_fingerprint(),
        "target_selection_id": TARGET_SELECTION_ID,
        "bin_schema_id": BIN_SCHEMA_V1.schema_id,
        "bin_schema_fingerprint": BIN_SCHEMA_V1.fingerprint,
        "qualification_namespace": QUALIFICATION_NAMESPACE,
        "qualification_macro_seed": QUALIFICATION_MACRO_SEED,
        "parent_source_bundle_fingerprint": parent_validation_manifest["source_bundle"]["fingerprint"],
        "parent_occupancy_fingerprint": parent_occupancy_manifest["occupancy_fingerprint"],
        "parent_validation_fingerprint": parent_validation_manifest["validation_fingerprint"],
        "current_source_bundle_fingerprint": source_bundle["fingerprint"],
        "runtime_model_id": _RUNTIME.model_id,
        "runtime_quote": {"value": 0.0, "basis": _RUNTIME_QUOTE.basis},
        "reservation": asdict(_RESERVATION),
        "degraded_slowdown": DEGRADED_SLOWDOWN,
        "hedge_delay": HEDGE_DELAY,
        "candidate_parent_fingerprint": candidate.parent_fingerprint,
        "deterministic_order": "episode_index_then_retained_bin_canonical",
        "max_scheduler_calls": QUALIFICATION_CALL_LIMIT,
        "actual_scheduler_calls": actual_calls,
    }
    manifest = {
        "schema_id": "token_t3b_qualification_manifest_v1",
        "run_id": run_id,
        "label": T3B_LABEL,
        "candidate_label": CANDIDATE_LABEL,
        "provenance": provenance,
        "source_bundle": dict(source_bundle),
        "parent_validation_run_id": parent_validation_manifest["run_id"],
        "parent_validation_fingerprint": parent_validation_manifest["validation_fingerprint"],
        "parent_occupancy_run_id": parent_occupancy_manifest["run_id"],
        "parent_occupancy_fingerprint": parent_occupancy_manifest["occupancy_fingerprint"],
        "claims_best_response": False,
        "claims_regret": False,
        "claims_nash": False,
        "claims_mfg": False,
        "no_timestamp": True,
    }
    candidate_payload = candidate.to_dict()
    candidate_payload.update(
        {
            "evidence_label": CANDIDATE_LABEL,
            "claims_best_response": False,
            "claims_regret": False,
            "claims_nash": False,
            "claims_mfg": False,
        }
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"qualification_rows", "episode_audit"}
    }
    summary["provenance"] = provenance
    return write_run_directory(
        artifacts_root,
        run_id,
        {
            "manifest.json": manifest,
            "candidate_policy.json": candidate_payload,
            "qualification_summary.json": summary,
            "qualification_rows.json": {"rows": result["qualification_rows"]},
            "episode_audit.json": {"rows": result["episode_audit"]},
        },
    )


def execute_t3b_qualification(
    project_root: str | Path,
    artifacts_root: str | Path,
    run_id: str,
    *,
    validation_dir: str | Path,
    occupancy_dir: str | Path,
    episode_count: int = QUALIFICATION_EPISODES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    project = Path(project_root).resolve()
    validation = Path(validation_dir).resolve()
    occupancy = Path(occupancy_dir).resolve()
    config = load_config(project / "configs" / "v1_minimal.json")
    bundle = build_source_bundle(project)
    candidate = load_t3a_runtime_candidate(validation, occupancy)
    result = run_qualification(
        config, candidate, episode_count=episode_count, progress=progress
    )
    parent_validation, parent_occupancy = _parent_payload(validation, occupancy)
    bundle_payload = asdict(bundle)
    directory = write_qualification_artifact(
        artifacts_root,
        run_id,
        result,
        candidate,
        source_bundle=bundle_payload,
        parent_validation_manifest=parent_validation,
        parent_occupancy_manifest=parent_occupancy,
        config=config,
    )
    return directory, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded T3B Token candidate qualification")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--occupancy-dir", required=True)
    args = parser.parse_args(argv)
    try:
        directory, result = execute_t3b_qualification(
            args.project_root,
            args.artifacts_root,
            args.run_id,
            validation_dir=args.validation_dir,
            occupancy_dir=args.occupancy_dir,
            progress=lambda done, total: print(f"qualification progress: {done}/{total}", flush=True),
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"artifact: {directory}")
    print(f"status: {result['status']}")
    print(f"calls: {result['attempted_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PANEL_FLOOR",
    "QUALIFICATION_CALL_LIMIT",
    "QUALIFICATION_EPISODES",
    "QUALIFICATION_MACRO_SEED",
    "QUALIFICATION_NAMESPACE",
    "T3BExecutionError",
    "T3B_LABEL",
    "execute_t3b_qualification",
    "paired_mean_se",
    "qualification_call_budget",
    "run_qualification",
    "summarize_bin",
    "write_qualification_artifact",
]
