"""Run the pinned, full-model Granite MoE training comparison.

The formal path expects one torchrun process per GPU.  A separate one-process
loopback mode performs a local full-checkpoint parity and optimizer smoke; it
does not claim distributed performance.  Policy planning and route scheduling
are injected native libraries with no Python fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = ROOT / "configs/granite_full_training_v1_20261001_a.json"
POLICY_SOURCE = ROOT / "src/mfg_hedge/granite_training_policy_native_v1.cpp"
SCHEDULE_SOURCE = ROOT / "src/mfg_hedge/granite_cloud_schedule_native_v1.cpp"
FROZEN_SOLVER_SOURCE = ROOT / "src/mfg_hedge/granite_scaling_sinkhorn_relaxed_v2.cpp"
BACKEND_SOURCE = ROOT / "src/mfg_hedge/granite_training_replica_v1.py"
POLICY_MODULE_SOURCE = ROOT / "src/mfg_hedge/granite_training_policy_native_v1.py"
SCHEDULE_MODULE_SOURCE = ROOT / "src/mfg_hedge/granite_cloud_schedule_native_v1.py"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GraniteFullTrainingError(RuntimeError):
    """A configuration, provenance, smoke, or training-run failure."""


POLICY_IDS = ("structured", "lplb", "sinkhorn_relaxed")
# Preserve the legacy preflight by default; isolated experiment entry points may
# temporarily select a wider full-checkpoint smoke without changing old configs.
DISTRIBUTED_PREFLIGHT_POLICY_IDS = ("structured",)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraniteFullTrainingError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GraniteFullTrainingError(f"JSON root must be an object: {path}")
    return value


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = _read_json(config_path)
    _validate_config(config, config_path)
    return config, config_path


def _validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    required = {
        "schema", "protocol_version", "checkpoint_id", "checkpoint_revision",
        "checkpoint_manifest", "transformers_abi", "world_size", "expert_count",
        "top_k", "policies", "load_policy_orders", "input", "workload",
        "training", "runtime", "timing",
    }
    missing = sorted(required - set(config))
    if missing:
        raise GraniteFullTrainingError(f"config is missing keys: {', '.join(missing)}")
    if config["schema"] != "granite_full_training_config_v1":
        raise GraniteFullTrainingError("unsupported Granite full-training config schema")
    if config["protocol_version"] != 1:
        raise GraniteFullTrainingError("only protocol version 1 is supported")
    if config["world_size"] != 6 or config["expert_count"] != 32 or config["top_k"] != 8:
        raise GraniteFullTrainingError("formal config is frozen to world-size 6, 32 Experts, top-k 8")
    if not isinstance(config["checkpoint_id"], str) or not config["checkpoint_id"]:
        raise GraniteFullTrainingError("checkpoint_id must be nonempty")
    revision = config["checkpoint_revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise GraniteFullTrainingError("checkpoint_revision must be a pinned 40-character commit")
    if Path(str(config["checkpoint_manifest"])).name != str(config["checkpoint_manifest"]):
        raise GraniteFullTrainingError("checkpoint_manifest must be a file name beside the config")
    if not isinstance(config["transformers_abi"], str) or not config["transformers_abi"]:
        raise GraniteFullTrainingError("transformers_abi must be nonempty")

    policies = config["policies"]
    if not isinstance(policies, list) or tuple(row.get("id") for row in policies if isinstance(row, dict)) != POLICY_IDS:
        raise GraniteFullTrainingError(f"policies must be ordered exactly as {POLICY_IDS}")
    if any(not isinstance(row.get("label"), str) or not row["label"].strip() for row in policies):
        raise GraniteFullTrainingError("every policy requires an explicit display label")
    expected_ids = set(POLICY_IDS)
    orders = config["load_policy_orders"]
    if not isinstance(orders, list) or len(orders) != 2:
        raise GraniteFullTrainingError("load_policy_orders must contain two frozen load orders")
    for order in orders:
        if not isinstance(order, list) or len(order) != 3 or set(order) != expected_ids:
            raise GraniteFullTrainingError("each load order must contain every policy exactly once")

    inp = config["input"]
    workload = config["workload"]
    training = config["training"]
    runtime = config["runtime"]
    timing = config["timing"]
    if inp.get("source_file") != "README.md" or inp.get("add_special_tokens") is not False:
        raise GraniteFullTrainingError("input source and tokenizer settings differ from the pinned protocol")
    for name, value in (("readme_sha256", inp.get("readme_sha256")),):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GraniteFullTrainingError(f"input.{name} must be a SHA-256 hex digest")
    if workload.get("sequence_length") != 1024:
        raise GraniteFullTrainingError("formal sequence length is frozen at 1024")
    if workload.get("local_batch_sizes") != [1, 4]:
        raise GraniteFullTrainingError("formal per-rank batch sizes are frozen at 1 and 4")
    if workload.get("warmup_steps") != 3 or workload.get("measured_steps") != 10:
        raise GraniteFullTrainingError("each formal cell requires three warmups and ten measured steps")
    if workload.get("expected_formal_cells") != 6 or workload.get("expected_global_optimizer_steps") != 78:
        raise GraniteFullTrainingError("formal matrix is frozen to six 13-step cells and 78 optimizer steps")
    if inp.get("window_stride_tokens") != 997:
        raise GraniteFullTrainingError("input window stride is frozen at 997 Tokens")
    if not isinstance(workload.get("smoke_sequence_length"), int) or not 1 <= workload["smoke_sequence_length"] <= 1024:
        raise GraniteFullTrainingError("smoke_sequence_length must be in 1..1024")
    if training.get("dtype") != "float32" or training.get("gradient_accumulation_steps") != 1:
        raise GraniteFullTrainingError("training is frozen to FP32 and gradient accumulation 1")
    if training.get("foreach") is not False or training.get("deterministic_algorithms") is not True:
        raise GraniteFullTrainingError("AdamW foreach and deterministic algorithm settings must remain frozen")
    precision = config.get("precision")
    if not isinstance(precision, dict) or precision.get("rtol") != 1e-5 or precision.get("atol") != 1e-5:
        raise GraniteFullTrainingError("full-model parity tolerances are frozen to rtol=atol=1e-5")
    if timing.get("run_budget_seconds") != 2700:
        raise GraniteFullTrainingError("formal run budget is frozen at 2700 seconds")
    if runtime.get("minimum_free_memory_bytes", 0) <= 0:
        raise GraniteFullTrainingError("runtime.minimum_free_memory_bytes must be positive")
    if runtime.get("compute_capability") != [7, 0]:
        raise GraniteFullTrainingError("formal preflight requires SM70")

    manifest_path = config_path.parent / str(config["checkpoint_manifest"])
    manifest = _read_json(manifest_path)
    if manifest.get("checkpoint_id") != config["checkpoint_id"]:
        raise GraniteFullTrainingError("checkpoint manifest ID differs from the config")
    if manifest.get("revision") != revision:
        raise GraniteFullTrainingError("checkpoint manifest revision differs from the config")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != 10:
        raise GraniteFullTrainingError("pinned checkpoint manifest must contain exactly ten files")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise GraniteFullTrainingError("checkpoint manifest contains a malformed file row")
        name = Path(row["name"])
        if name.is_absolute() or ".." in name.parts or row["name"] in names:
            raise GraniteFullTrainingError("checkpoint manifest contains an unsafe or duplicate path")
        names.add(row["name"])
        if not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
            raise GraniteFullTrainingError(f"invalid byte count in checkpoint manifest for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise GraniteFullTrainingError(f"invalid SHA-256 in checkpoint manifest for {name}")
    readme = next((row for row in rows if row["name"] == "README.md"), None)
    if readme is None or readme["sha256"] != inp["readme_sha256"]:
        raise GraniteFullTrainingError("README digest differs between input and model manifests")


def _sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_model_files(model_dir: str | Path, config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve(strict=True)
    manifest = _read_json(config_path.parent / str(config["checkpoint_manifest"]))
    verified: dict[str, Any] = {}
    for row in manifest["files"]:
        name = Path(row["name"])
        path = root / name
        if not path.is_file():
            raise GraniteFullTrainingError(f"pinned checkpoint file is missing: {name}")
        observed = _sha256_file(path)
        expected = {"bytes": int(row["bytes"]), "sha256": row["sha256"]}
        if observed != expected:
            raise GraniteFullTrainingError(f"pinned checkpoint file does not match its manifest: {name}")
        verified[str(name)] = observed
    return {"model_dir": str(root), "checkpoint_id": manifest["checkpoint_id"],
            "revision": manifest["revision"], "files": verified}


def make_windows(
    token_ids: Sequence[int], *, sequence_length: int, batch_size: int, rank: int,
    world_size: int, step_index: int, stride: int,
) -> list[list[int]]:
    """Select deterministic, rank-specific windows from the periodic README corpus."""
    if sequence_length <= 0 or batch_size <= 0 or not 0 <= rank < world_size:
        raise GraniteFullTrainingError("invalid input window dimensions or source rank")
    starts_available = len(token_ids) - sequence_length + 1
    if starts_available < world_size * batch_size:
        raise GraniteFullTrainingError("pinned README is too short for distinct source windows")
    starts = [
        ((step_index * world_size * batch_size + rank * batch_size + sample) * stride)
        % starts_available
        for sample in range(batch_size)
    ]
    if len(set(starts)) != batch_size:
        raise GraniteFullTrainingError("input-window formula produced duplicate windows in a rank batch")
    return [list(token_ids[start:start + sequence_length]) for start in starts]


def token_batch_digest(rows: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for token in row:
            digest.update(int(token).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _new_run_directory(output_root: str | Path, run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in (".", ".."):
        raise GraniteFullTrainingError("run-id must be one safe directory name")
    base = Path(output_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    destination = base / run_id
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise GraniteFullTrainingError(f"run directory already exists; choose a fresh run-id: {destination}") from exc
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class RecordingPlanner:
    """A transparent decorator that records results from the injected C ABI."""

    is_native = True

    def __init__(self, native_planner: Any):
        if not getattr(native_planner, "is_native", False):
            raise GraniteFullTrainingError("the training runner requires NativeTrainingPlanner; Python fallback is disabled")
        self.native_planner = native_planner
        self.policy = native_planner.policy
        self.calls: list[dict[str, Any]] = []

    @property
    def supports_source_counts(self) -> bool:
        return callable(getattr(self.native_planner, "plan_with_source_counts", None))

    def plan(self, counts: Sequence[int], gpus: int):
        result = self.native_planner.plan(counts, gpus)
        return self._record_result(counts, gpus, result)

    def plan_with_source_counts(
        self,
        counts: Sequence[int],
        gpus: int,
        rank_counts: Sequence[Sequence[int]],
    ):
        """Use an explicitly source-aware planner when one is installed.

        The ordinary native planners still receive exactly the same call as
        ``plan``. This is an opt-in seam for policies that can use the Gate
        histograms already gathered by the backend, without a new collective.
        """
        method = getattr(self.native_planner, "plan_with_source_counts", None)
        if callable(method):
            result = method(counts, gpus, rank_counts)
        else:
            result = self.native_planner.plan(counts, gpus)
        return self._record_result(counts, gpus, result)

    def _record_result(self, counts: Sequence[int], gpus: int, result: Any):
        counts_row = [int(value) for value in counts]
        quota_left = [int(value) for value in result.quota_left]
        if len(counts_row) != 32 or len(quota_left) != 32:
            raise GraniteFullTrainingError("native planner must return 32-Expert rows")
        if any(q < 0 or q > count for count, q in zip(counts_row, quota_left)):
            raise GraniteFullTrainingError("native quota is outside its observed Gate count")
        self.calls.append({
            "counts_by_expert": counts_row,
            "quota_left": quota_left,
            "shares_left": [float(value) for value in result.shares_left],
            "status": str(result.status),
            "iterations": int(result.iterations),
            "diagnostics": _json_safe(result.diagnostics),
            "gpu_counts_by_expert": _quota_gpu_matrix(counts_row, quota_left, gpus),
        })
        return result

    def drain(self) -> list[dict[str, Any]]:
        value, self.calls = self.calls, []
        return value


def _quota_gpu_matrix(counts: Sequence[int], quota_left: Sequence[int], world_size: int) -> list[list[int]]:
    matrix = [[0] * len(counts) for _ in range(world_size)]
    for expert, (count, left) in enumerate(zip(counts, quota_left)):
        matrix[expert % world_size][expert] = left
        matrix[(expert + 1) % world_size][expert] = int(count) - left
    return matrix


class GateRouteCollector:
    """Record the actual per-layer top-k Gate histogram without changing its graph."""

    def __init__(self, modules: Iterable[tuple[str, Any]], expert_count: int = 32):
        self.expert_count = expert_count
        self.values: dict[str, list[int]] = {}
        self.handles = []
        for name, module in modules:
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise GraniteFullTrainingError(f"routed MoE output at {name} does not expose router logits")
            logits = output[1]
            if not hasattr(logits, "topk"):
                raise GraniteFullTrainingError(f"routed MoE router logits at {name} are not a tensor")
            top_k = int(_module.router.top_k)
            chosen = logits.detach().topk(top_k, dim=-1).indices.reshape(-1)
            counts = chosen.bincount(minlength=self.expert_count)
            row = [int(value) for value in counts.cpu().tolist()]
            if len(row) != self.expert_count:
                raise GraniteFullTrainingError(f"Gate histogram at {name} has the wrong Expert count")
            if name in self.values:
                raise GraniteFullTrainingError(f"MoE layer {name} ran more than once in one full-model step")
            self.values[name] = row
        return capture

    def drain(self, expected_layers: Sequence[str]) -> dict[str, list[int]]:
        value = self.values
        self.values = {}
        if set(value) != set(expected_layers):
            missing = sorted(set(expected_layers) - set(value))
            extra = sorted(set(value) - set(expected_layers))
            raise GraniteFullTrainingError(f"full-model Gate coverage differs; missing={missing}, extra={extra}")
        return {name: value[name] for name in expected_layers}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _cuda_sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distributed_payloads(dist: Any, payload: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, payload)
    if any(not isinstance(row, dict) for row in gathered):
        raise GraniteFullTrainingError("rank metrics gather returned malformed payload")
    return gathered


def _consensus_error(dist: Any, error: str | None) -> str | None:
    if not dist.is_initialized():
        return error
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, error)
    return next((str(value) for value in gathered if value), None)


def _rank_file(run_dir: Path, rank: int) -> Path:
    return run_dir / f"rank-{rank:02d}.jsonl"


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise GraniteFullTrainingError("cannot summarize an empty timing sample")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _validate_step_gradients(torch: Any, handle: Any) -> dict[str, Any]:
    checked = 0
    nonzero = 0
    for parameter in handle.optimizer_parameters:
        grad = parameter.grad
        if grad is None:
            raise GraniteFullTrainingError("optimizer-owned parameter has no gradient after synchronization")
        if not bool(torch.isfinite(grad).all().item()):
            raise GraniteFullTrainingError("non-finite trainable parameter gradient")
        checked += 1
        if bool(torch.count_nonzero(grad).item()):
            nonzero += 1
    if checked == 0 or nonzero == 0:
        raise GraniteFullTrainingError("full-model backward produced no nonzero owned gradients")
    return {"gradient_parameters_checked": checked, "nonzero_gradient_parameters": nonzero}


def _record_step(
    *, torch: Any, dist: Any, model: Any, handle: Any, planner: RecordingPlanner,
    collector: GateRouteCollector, optimizer: Any, input_ids: Any, labels: Any,
    device: Any, expected_layers: Sequence[str], rank: int, world_size: int,
    local_batch_size: int, sequence_length: int, phase: str, phase_step: int,
    load_index: int, policy_id: str, policy_label: str, run_id: str,
    run_dir: Path, input_hash: str, seed: int, local_step_index: int = 0,
    check_grads: bool = True, records_file: str = "records.jsonl",
    initial_state_hash: str | None = None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    handle.reset_timing_diagnostics()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _cuda_sync(torch, device)
    step_start = time.perf_counter()

    _cuda_sync(torch, device)
    phase_start = time.perf_counter()
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    _cuda_sync(torch, device)
    forward_ms = (time.perf_counter() - phase_start) * 1000.0
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss).all().item()):
        raise GraniteFullTrainingError("full causal-LM forward produced a missing or non-finite loss")
    loss_value = float(loss.detach().item())

    _cuda_sync(torch, device)
    phase_start = time.perf_counter()
    loss.backward()
    _cuda_sync(torch, device)
    backward_ms = (time.perf_counter() - phase_start) * 1000.0
    del output

    _cuda_sync(torch, device)
    phase_start = time.perf_counter()
    handle.sync_gradients()
    _cuda_sync(torch, device)
    gradient_sync_ms = (time.perf_counter() - phase_start) * 1000.0
    if check_grads:
        gradient_check = None
        gradient_error = None
        try:
            gradient_check = _validate_step_gradients(torch, handle)
        except Exception as exc:
            gradient_error = f"{type(exc).__name__}: {exc}"
        gradient_error = _consensus_error(dist, gradient_error)
        if gradient_error:
            raise GraniteFullTrainingError(f"full-model gradient validation failed: {gradient_error}")
        assert gradient_check is not None
    else:
        gradient_check = {"gradient_parameters_checked": None, "nonzero_gradient_parameters": None}

    _cuda_sync(torch, device)
    phase_start = time.perf_counter()
    optimizer.step()
    _cuda_sync(torch, device)
    optimizer_ms = (time.perf_counter() - phase_start) * 1000.0
    _cuda_sync(torch, device)
    direct_step_ms = (time.perf_counter() - step_start) * 1000.0

    identity_started = time.perf_counter()
    identity_ok = bool(handle.check_replica_identity(optimizer))
    identity_check_ms = (time.perf_counter() - identity_started) * 1000.0
    identity_error = _consensus_error(
        dist, None if identity_ok else "Expert parameter or AdamW replica identity check failed"
    )
    if identity_error:
        raise GraniteFullTrainingError(identity_error)

    gate_counts = collector.drain(expected_layers)
    backend = handle.read_timing_diagnostics()
    plans = planner.drain()
    if len(plans) != len(expected_layers) and rank == 0:
        raise GraniteFullTrainingError(
            f"native planner ran {len(plans)} times for {len(expected_layers)} MoE layers"
        )
    expected_routes = local_batch_size * sequence_length * int(model.config.num_experts_per_tok) * len(expected_layers)
    if int(backend["routes"]) != expected_routes:
        raise GraniteFullTrainingError(
            f"backend conserved {backend['routes']} routes, expected {expected_routes} from actual Gate rows"
        )
    for name, row in gate_counts.items():
        if sum(row) != local_batch_size * sequence_length * int(model.config.num_experts_per_tok):
            raise GraniteFullTrainingError(f"Gate route count at {name} does not equal rows × top-k")

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = 0
    payload = {
        "rank": rank,
        "loss": loss_value,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "gradient_sync_ms": gradient_sync_ms,
        "optimizer_ms": optimizer_ms,
        "direct_step_ms": direct_step_ms,
        "input_ids_sha256": input_hash,
        "initial_state_sample_sha256": initial_state_hash,
        "gate_counts_by_layer": gate_counts,
        "planner_calls": plans,
        "backend_diagnostics": _json_safe(backend),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "replica_identity": identity_ok,
        "replica_identity_check_ms": identity_check_ms,
        **gradient_check,
    }
    physical_source_count = int(dist.get_world_size()) if dist.is_initialized() else 1
    rows = _distributed_payloads(dist, payload, physical_source_count) if dist.is_initialized() else [payload]
    if sorted(int(row["rank"]) for row in rows) != list(range(physical_source_count)):
        raise GraniteFullTrainingError("training step did not receive one source payload per rank")

    global_record = None
    if rank == 0:
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        layer_names = list(expected_layers)
        global_gate_counts = {
            layer: [sum(int(row["gate_counts_by_layer"][layer][expert]) for row in rows) for expert in range(32)]
            for layer in layer_names
        }
        planner_by_layer = {}
        if len(rows[0]["planner_calls"]) != len(layer_names):
            raise GraniteFullTrainingError("rank 0 has an incomplete native planner diagnostic set")
        for layer, plan in zip(layer_names, rows[0]["planner_calls"]):
            if plan["counts_by_expert"] != global_gate_counts[layer]:
                raise GraniteFullTrainingError(f"native planner counts disagree with observed Gate counts at {layer}")
            if sum(map(sum, plan["gpu_counts_by_expert"])) != sum(plan["counts_by_expert"]):
                raise GraniteFullTrainingError(f"planned physical-GPU route counts do not conserve routes at {layer}")
            planner_by_layer[layer] = plan
        if sum(map(sum, global_gate_counts.values())) != local_batch_size * sequence_length * int(model.config.num_experts_per_tok) * len(layer_names) * physical_source_count:
            raise GraniteFullTrainingError("all-rank Gate histograms do not conserve the complete source batch")
        max_wall_ms = max(float(row["direct_step_ms"]) for row in rows)
        global_tokens = physical_source_count * local_batch_size * sequence_length
        planner_ms = sum(float(row["backend_diagnostics"].get("planner_ms", 0.0)) for row in rows)
        quota_legal = all(
            len(plan["counts_by_expert"]) == 32
            and len(plan["quota_left"]) == 32
            and all(0 <= quota <= count for quota, count in zip(plan["quota_left"], plan["counts_by_expert"]))
            for plan in planner_by_layer.values()
        )
        gradients_finite = all(
            row["gradient_parameters_checked"] is not None
            and int(row["gradient_parameters_checked"]) > 0
            and int(row["nonzero_gradient_parameters"]) > 0
            for row in rows
        )
        global_record = {
            "schema": "granite_full_training_step_v1",
            "run_id": run_id,
            "load_index": load_index,
            "tokens_per_rank": local_batch_size * sequence_length,
            "global_input_tokens": global_tokens,
            "sequence_length": sequence_length,
            "local_batch_size": local_batch_size,
            "policy": policy_id,
            "policy_label": policy_label,
            "phase": phase,
            "phase_step": phase_step,
            "step_index": int(load_index * (3 + 10) + local_step_index),
            "measured": phase == "measured",
            "seed": seed,
            "loss_mean_across_ranks": statistics.fmean(float(row["loss"]) for row in rows),
            "loss": statistics.fmean(float(row["loss"]) for row in rows),
            "loss_by_rank": [float(row["loss"]) for row in rows],
            "direct_step_wall_ms_max_rank": max_wall_ms,
            "full_step_ms": max_wall_ms,
            "planner_ms": planner_ms,
            "gradients_finite": gradients_finite,
            "replica_identity": all(bool(row["replica_identity"]) for row in rows),
            "quota_legal": quota_legal,
            "global_input_tokens_per_second": global_tokens / (max_wall_ms / 1000.0),
            "phase_wall_ms_max_rank": {
                phase_name: max(float(row[f"{phase_name}_ms"]) for row in rows)
                for phase_name in ("forward", "backward", "gradient_sync", "optimizer")
            },
            "phase_wall_ms_by_rank": [
                {name: float(row[f"{name}_ms"]) for name in ("forward", "backward", "gradient_sync", "optimizer")}
                for row in rows
            ],
            "input_ids_sha256_by_rank": [row["input_ids_sha256"] for row in rows],
            "initial_state_sample_sha256_by_rank": [row["initial_state_sample_sha256"] for row in rows],
            "gate_routes_by_layer_expert": global_gate_counts,
            "gpu_routes_by_layer_expert": {
                layer: planner_by_layer[layer]["gpu_counts_by_expert"] for layer in layer_names
            },
            "native_planner_by_layer": planner_by_layer,
            "backend_host_diagnostics_by_rank": [row["backend_diagnostics"] for row in rows],
            "peak_cuda_memory_allocated_bytes_by_rank": [row["peak_cuda_memory_allocated_bytes"] for row in rows],
            "peak_cuda_memory_reserved_bytes_by_rank": [row["peak_cuda_memory_reserved_bytes"] for row in rows],
            "gradient_parameters_checked_by_rank": [row["gradient_parameters_checked"] for row in rows],
            "nonzero_gradient_parameters_by_rank": [row["nonzero_gradient_parameters"] for row in rows],
            "timing_scope": "direct CUDA-synchronized wall clock from before full-model forward through backward, explicit gradient sync, AdamW update, and final device synchronization; maximum rank; nested planner/schedule clocks are not added",
        }
        _write_jsonl(run_dir / records_file, global_record)
    rank_row = {
            "schema": "granite_full_training_rank_step_v1",
            "run_id": run_id,
            "rank": rank,
            "load_index": load_index,
            "local_batch_size": local_batch_size,
            "policy": policy_id,
            "phase": phase,
            "phase_step": phase_step,
            **{key: payload[key] for key in (
                "loss", "forward_ms", "backward_ms", "gradient_sync_ms", "optimizer_ms",
                "direct_step_ms", "input_ids_sha256", "gate_counts_by_layer",
                "initial_state_sample_sha256",
                "planner_calls", "backend_diagnostics", "peak_cuda_memory_allocated_bytes",
                "peak_cuda_memory_reserved_bytes", "gradient_parameters_checked",
                "nonzero_gradient_parameters", "replica_identity",
                "replica_identity_check_ms",
            )},
        }
    rank_path = _rank_file(run_dir, rank) if records_file == "records.jsonl" else run_dir / f"preflight-rank-{rank:02d}.jsonl"
    _write_jsonl(rank_path, rank_row)
    return global_record or {"rank": rank, "phase": phase, "phase_step": phase_step}


def _tokenize_readme(model_dir: Path, config: Mapping[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    text = (model_dir / str(config["input"]["source_file"])).read_text(encoding="utf-8")
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    tokens = [int(value) for value in encoded["input_ids"][0].tolist()]
    expected = int(config["input"]["readme_token_count"])
    if len(tokens) != expected:
        raise GraniteFullTrainingError(f"README tokenizer produced {len(tokens)} Tokens; manifest expects {expected}")
    return tokenizer, tokens


def _load_model(model_dir: Path, config: Mapping[str, Any], device: Any):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=torch.float32)
    model.train()
    model.config.use_cache = False
    observed_experts = int(getattr(model.config, "num_local_experts", -1))
    observed_top_k = int(getattr(model.config, "num_experts_per_tok", -1))
    if observed_experts != 32 or observed_top_k != 8:
        raise GraniteFullTrainingError(
            f"checkpoint MoE config changed: num_local_experts={observed_experts}, top_k={observed_top_k}"
        )
    moe_names = [
        name for name, module in model.named_modules()
        if name and type(module).__name__ == "GraniteMoeMoE"
    ]
    if not moe_names:
        raise GraniteFullTrainingError("checkpoint has no Transformers GraniteMoeMoE modules")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise GraniteFullTrainingError("checkpoint model contains non-FP32 parameters")
    return model, tuple(moe_names)


def _load_tiny_transformers_model(torch: Any, seed: int, device: Any):
    """Construct a tiny but real Transformers Granite MoE with the formal ABI."""
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    torch.manual_seed(seed)
    model_config = GraniteMoeConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_local_experts=32,
        num_experts_per_tok=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        use_cache=False,
    )
    model = GraniteMoeForCausalLM(model_config).to(device=device, dtype=torch.float32)
    model.train()
    names = tuple(
        name for name, module in model.named_modules()
        if name and type(module).__name__ == "GraniteMoeMoE"
    )
    if len(names) != 2:
        raise GraniteFullTrainingError(f"tiny Transformers Granite control expected two MoE layers, found {len(names)}")
    return model, names


def _initial_state_sample_hash(model: Any) -> str:
    """Hash fixed samples from every initial parameter without copying a checkpoint."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            samples = flat
        else:
            indices = sorted({0, int(flat.numel() // 2), int(flat.numel() - 1)})
            samples = flat.index_select(0, torch.tensor(indices, dtype=torch.long, device=flat.device))
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(samples.to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_model_load_provenance(config: Mapping[str, Any], config_path: Path,
                                 model_dir: Path, policy_lib: Path, schedule_lib: Path) -> dict[str, Any]:
    source_rows = {}
    for path in (config_path, config_path.parent / str(config["checkpoint_manifest"]),
                 BACKEND_SOURCE, POLICY_MODULE_SOURCE, SCHEDULE_MODULE_SOURCE, POLICY_SOURCE,
                 SCHEDULE_SOURCE, FROZEN_SOLVER_SOURCE):
        if not path.is_file():
            raise GraniteFullTrainingError(f"source/provenance file is missing: {path}")
        source_rows[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = _sha256_file(path)
    return {
        "config": source_rows[str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path)],
        "source_files": source_rows,
        "native_policy_library": {"path": str(policy_lib), **_sha256_file(policy_lib)},
        "native_schedule_library": {"path": str(schedule_lib), **_sha256_file(schedule_lib)},
        "model_dir": str(model_dir),
    }


def _load_token_batch(torch: Any, rows: Sequence[Sequence[int]], device: Any):
    ids = torch.tensor(rows, dtype=torch.long, device=device)
    return ids, ids.clone()


def _capture_parity_baseline(model: Any, input_ids: Any, labels: Any, dist: Any,
                             rank: int, world_size: int):
    """Run the checkpoint's ordinary all-Expert path and snapshot strict controls."""
    import torch

    moe_items = [
        (name, module) for name, module in model.named_modules()
        if name and type(module).__name__ == "GraniteMoeMoE"
    ]
    gate_collector = GateRouteCollector(moe_items)
    moe_outputs: dict[str, Any] = {}
    output_handles = []
    for name, module in moe_items:
        def capture_moe(_module, _inputs, output, layer_name=name):
            if not isinstance(output, tuple) or len(output) < 1:
                raise GraniteFullTrainingError(f"reference MoE output is malformed at {layer_name}")
            moe_outputs[layer_name] = output[0].detach().clone()
        output_handles.append(module.register_forward_hook(capture_moe))
    embedding = model.get_input_embeddings()
    embedding_outputs: list[Any] = []

    def retain_embedding(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise GraniteFullTrainingError("input embedding did not return a Tensor")
        output.retain_grad()
        embedding_outputs.append(output)

    embedding_handle = embedding.register_forward_hook(retain_embedding)
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels, use_cache=False)
    _cuda_sync(torch, input_ids.device)
    logits_full = output.logits.detach().clone()
    loss_value = float(output.loss.detach().item())
    if not math.isfinite(loss_value) or not bool(torch.isfinite(logits_full).all().item()):
        raise GraniteFullTrainingError("unmodified Granite reference produced non-finite output")
    output.loss.backward()
    _cuda_sync(torch, input_ids.device)
    if len(embedding_outputs) != 1 or embedding_outputs[0].grad is None:
        raise GraniteFullTrainingError("could not capture the reference hidden-state gradient")
    hidden_gradient = embedding_outputs[0].grad.detach().clone()
    gate_counts = gate_collector.drain(tuple(name for name, _ in moe_items))
    if set(moe_outputs) != {name for name, _ in moe_items}:
        raise GraniteFullTrainingError("reference full-model pass did not capture every MoE layer output")

    router_parameters = []
    expert_specifications = []
    expert_sample_parts = []
    for name, module in moe_items:
        router_parameters.append(module.router.layer.weight)
        if module.router.layer.weight.grad is None:
            raise GraniteFullTrainingError(f"reference Router gradient is missing at {name}")
        for table_name, table in (("w1", module.input_linear.weight), ("w2", module.output_linear.weight)):
            gradient = table.grad
            if gradient is None:
                raise GraniteFullTrainingError(f"reference Expert gradient is missing at {name}.{table_name}")
            for expert in range(32):
                flat = gradient[expert].reshape(-1)
                indexes = sorted({0, int(flat.numel() // 2), int(flat.numel() - 1), (expert * 104729 + len(expert_specifications)) % int(flat.numel())})
                expert_specifications.append((name, expert, table_name, indexes))
                expert_sample_parts.append(
                    flat.index_select(0, torch.tensor(indexes, dtype=torch.long, device=flat.device)).detach().clone()
                )
    router_parts = [parameter.grad.detach().reshape(-1).clone() for parameter in router_parameters]
    router_flat = torch.cat(router_parts)
    if dist.is_initialized():
        dist.all_reduce(router_flat, op=dist.ReduceOp.SUM)
        router_flat.div_(world_size)
        expert_flat = torch.cat(expert_sample_parts)
        dist.all_reduce(expert_flat, op=dist.ReduceOp.SUM)
        expert_flat.div_(world_size)
    else:
        expert_flat = torch.cat(expert_sample_parts)
    if not bool(torch.isfinite(router_flat).all().item()) or not bool(torch.isfinite(expert_flat).all().item()):
        raise GraniteFullTrainingError("reference Router/Expert sample gradients are non-finite")
    expected_router = {}
    cursor = 0
    for (name, module), parameter in zip(moe_items, router_parameters):
        size = int(parameter.numel())
        expected_router[name] = router_flat[cursor:cursor + size].view_as(parameter).clone()
        cursor += size
    expected_expert = {}
    cursor = 0
    for name, expert, table_name, indexes in expert_specifications:
        width = len(indexes)
        expected_expert[(name, expert, table_name)] = (indexes, expert_flat[cursor:cursor + width].clone())
        cursor += width
    embedding_handle.remove()
    gate_collector.close()
    for hook in output_handles:
        hook.remove()
    model.zero_grad(set_to_none=True)
    del output, embedding_outputs, router_flat, expert_flat
    return {
        "loss": loss_value,
        "logits": logits_full,
        "moe_outputs": moe_outputs,
        "hidden_gradient": hidden_gradient,
        "gate_counts": gate_counts,
        "expected_router_gradients": expected_router,
        "expected_expert_gradient_samples": expected_expert,
        "moe_names": tuple(name for name, _ in moe_items),
    }


def _assert_close(torch: Any, actual: Any, expected: Any, *, name: str, rtol: float, atol: float) -> float:
    if actual.shape != expected.shape:
        raise GraniteFullTrainingError(f"parity shape mismatch for {name}: {tuple(actual.shape)} != {tuple(expected.shape)}")
    if not bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)):
        difference = float((actual - expected).abs().max().item())
        raise GraniteFullTrainingError(f"strict parity failed for {name}: max_abs_error={difference:.9g}, rtol={rtol}, atol={atol}")
    return float((actual - expected).abs().max().item()) if actual.numel() else 0.0


def _run_parity_training_smoke(*, torch: Any, dist: Any, model: Any, model_names: Sequence[str],
                               input_ids: Any, labels: Any, planner: RecordingPlanner,
                               schedule_library: Path, world_size: int, rank: int,
                               config: Mapping[str, Any], device: Any, run_dir: Path,
                               run_id: str, mode: str, full_checkpoint: bool) -> dict[str, Any]:
    from .granite_training_replica_v1 import install_training_routing

    train_config = config["training"]
    initial_state_hash = _initial_state_sample_hash(model)
    seed = int(train_config["seed"]) + rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parity = _capture_parity_baseline(model, input_ids, labels, dist, rank, world_size)
    routed_collector = None
    handle = None
    optimizer = None
    embedding_handle = None
    try:
        handle = install_training_routing(
            model, planner, schedule_library, world_size, rank,
            loopback=(mode == "tiny-loopback-smoke"),
        )
        routed_collector = GateRouteCollector(
            ((name, model.get_submodule(name)) for name in handle.routed_layer_names)
        )
        embedding_outputs: list[Any] = []

        def retain_embedding(_module, _inputs, output):
            output.retain_grad()
            embedding_outputs.append(output)

        embedding_handle = model.get_input_embeddings().register_forward_hook(retain_embedding)
        optimizer = _create_optimizer(torch, handle, train_config)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        optimizer.zero_grad(set_to_none=True)
        handle.reset_timing_diagnostics()
        routed_moe_outputs: dict[str, Any] = {}
        routed_output_hooks = []
        for name in handle.routed_layer_names:
            module = model.get_submodule(name)
            def capture_routed_moe(_module, _inputs, output, layer_name=name):
                routed_moe_outputs[layer_name] = output[0].detach().clone()
            routed_output_hooks.append(module.register_forward_hook(capture_routed_moe))
        routed_output = model(input_ids=input_ids, labels=labels, use_cache=False)
        _cuda_sync(torch, device)
        if routed_output.loss is None or not bool(torch.isfinite(routed_output.loss).all().item()):
            raise GraniteFullTrainingError("routed full-model smoke returned a non-finite loss")
        logits_error = _assert_close(
            torch, routed_output.logits, parity["logits"], name="full causal-LM logits",
            rtol=float(config["precision"]["rtol"]), atol=float(config["precision"]["atol"]),
        )
        if set(routed_moe_outputs) != set(handle.routed_layer_names):
            raise GraniteFullTrainingError("routed full-model pass did not capture every MoE layer output")
        moe_errors = [
            _assert_close(
                torch, routed_moe_outputs[name], parity["moe_outputs"][name],
                name=f"MoE output {name}", rtol=float(config["precision"]["rtol"]),
                atol=float(config["precision"]["atol"]),
            )
            for name in handle.routed_layer_names
        ]
        for hook in routed_output_hooks:
            hook.remove()
        routed_loss_value = float(routed_output.loss.detach().item())
        loss_error = _assert_close(
            torch,
            torch.tensor(routed_loss_value, dtype=torch.float32, device=device),
            torch.tensor(parity["loss"], dtype=torch.float32, device=device),
            name="causal-LM loss", rtol=float(config["precision"]["rtol"]),
            atol=float(config["precision"]["atol"]),
        )
        routed_gate_counts = routed_collector.drain(handle.routed_layer_names)
        if routed_gate_counts != parity["gate_counts"]:
            raise GraniteFullTrainingError("routed Gate selected different Experts from the ordinary Granite model")
        routed_output.loss.backward()
        _cuda_sync(torch, device)
        if len(embedding_outputs) != 1 or embedding_outputs[0].grad is None:
            raise GraniteFullTrainingError("routed pass did not produce a hidden-state gradient")
        hidden_error = _assert_close(
            torch, embedding_outputs[0].grad, parity["hidden_gradient"],
            name="embedding hidden-state gradient", rtol=float(config["precision"]["rtol"]),
            atol=float(config["precision"]["atol"]),
        )
        del routed_output
        handle.sync_gradients()
        gradient_check = _validate_step_gradients(torch, handle)
        router_errors = []
        for name in handle.routed_layer_names:
            routed_moe = model.get_submodule(name)
            router_errors.append(_assert_close(
                torch, routed_moe.router.layer.weight.grad,
                parity["expected_router_gradients"][name], name=f"Router gradient {name}",
                rtol=float(config["precision"]["rtol"]), atol=float(config["precision"]["atol"]),
            ))
        expert_errors = []
        for (name, expert, table_name), (indexes, expected) in parity["expected_expert_gradient_samples"].items():
            if expert not in handle.resident_experts:
                continue
            routed_moe = model.get_submodule(name)
            parameter = routed_moe.local_expert_w1[str(expert)] if table_name == "w1" else routed_moe.local_expert_w2[str(expert)]
            index_tensor = torch.tensor(indexes, dtype=torch.long, device=parameter.device)
            actual = parameter.grad.reshape(-1).index_select(0, index_tensor)
            expert_errors.append(_assert_close(
                torch, actual, expected, name=f"Expert gradient sample {name}/{expert}/{table_name}",
                rtol=float(config["precision"]["rtol"]), atol=float(config["precision"]["atol"]),
            ))
        if dist.is_initialized():
            gathered_gates = [None] * world_size
            dist.all_gather_object(gathered_gates, routed_gate_counts)
        else:
            gathered_gates = [routed_gate_counts]
        planner_calls = planner.drain()
        if rank == 0 and len(planner_calls) != len(handle.routed_layer_names):
            raise GraniteFullTrainingError("full-model smoke did not invoke native planner once per MoE layer")
        if rank == 0:
            for layer_index, (layer_name, plan) in enumerate(zip(handle.routed_layer_names, planner_calls)):
                observed = [
                    sum(int(rank_rows[layer_name][expert]) for rank_rows in gathered_gates)
                    for expert in range(32)
                ]
                if observed != plan["counts_by_expert"]:
                    raise GraniteFullTrainingError(f"native smoke quota input differs from Gate histogram at {layer_name}")
                if any(not 0 <= quota <= count for count, quota in zip(observed, plan["quota_left"])):
                    raise GraniteFullTrainingError(f"native smoke produced an illegal Expert quota at {layer_name}")
        optimizer.step()
        _cuda_sync(torch, device)
        if handle.loopback:
            identity = {"passed": True, "scope": "single-process shared-view loopback; no replica collective"}
            identity_check_ms = 0.0
        else:
            identity_started = time.perf_counter()
            identity_ok = handle.check_replica_identity(optimizer)
            identity_check_ms = (time.perf_counter() - identity_started) * 1000.0
            identity = {"passed": bool(identity_ok), "scope": "all adjacent Expert replicas and AdamW states"}
        if not identity["passed"]:
            raise GraniteFullTrainingError("Expert replica identity check failed after full-model update")
        validation = {
            "schema": "granite_full_training_validation_v1",
            "mode": mode,
            "status": "passed",
            "full_checkpoint_parity_gate": "passed" if full_checkpoint else "not_applicable_synthetic_tiny",
            "tiny_parity_gate": "passed" if not full_checkpoint else None,
            "tolerance": {"rtol": float(config["precision"]["rtol"]), "atol": float(config["precision"]["atol"])},
            "rank": rank,
            "reference_loss": parity["loss"],
            "routed_loss": routed_loss_value,
            "loss_max_abs_error": loss_error,
            "full_logits_max_abs_error": logits_error,
            "moe_layer_output_max_abs_error": max(moe_errors, default=0.0),
            "embedding_hidden_gradient_max_abs_error": hidden_error,
            "router_gradient_max_abs_error": max(router_errors, default=0.0),
            "expert_gradient_sample_max_abs_error": max(expert_errors, default=0.0),
            "gradient_check": gradient_check,
            "native_route_diagnostics": {
                "planner_calls": len(planner_calls),
                "planner_statuses": [item["status"] for item in planner_calls],
                "planner_iterations": [item["iterations"] for item in planner_calls],
                "gpu_routes_by_expert": [item["gpu_counts_by_expert"] for item in planner_calls],
            },
            "replica_identity": identity,
            "replica_identity_check_ms": identity_check_ms,
            "input_ids_sha256": token_batch_digest(input_ids.detach().cpu().tolist()),
            "initial_state_sample_sha256": initial_state_hash,
            "source_gate_counts_by_layer": gathered_gates,
            "parity_scope": "full sequence logits; all routed MoE layer outputs; full input embedding output gradient; full Router gradients; four fixed sampled values per Expert weight table/layer, not a full Expert-gradient tensor comparison",
            "optimizer": {"name": "AdamW", "learning_rate": float(train_config["learning_rate"]),
                          "weight_decay": float(train_config["weight_decay"]), "foreach": False, "steps": 1},
        }
        if dist.is_initialized():
            dist.barrier()
        return {"status": "passed", "replica_identity": identity,
                "full_checkpoint_parity_gate": "passed" if full_checkpoint else "not_applicable_synthetic_tiny",
                "tiny_parity_gate": "passed" if not full_checkpoint else None,
                "reference_loss": parity["loss"],
                "routed_loss": routed_loss_value, "validation": validation}
    finally:
        if embedding_handle is not None:
            embedding_handle.remove()
        if routed_collector is not None:
            routed_collector.close()
        if handle is not None:
            handle.close()
        del parity


def _create_optimizer(torch: Any, handle: Any, train_config: Mapping[str, Any]):
    return torch.optim.AdamW(
        handle.optimizer_parameters,
        lr=float(train_config["learning_rate"]),
        betas=tuple(float(value) for value in train_config["betas"]),
        eps=float(train_config["epsilon"]),
        weight_decay=float(train_config["weight_decay"]),
        foreach=bool(train_config["foreach"]),
    )


def _run_capacity_smoke(*, config: Mapping[str, Any], model_dir: Path,
                        policy_library: Path, schedule_library: Path, run_dir: Path,
                        run_id: str, rank: int, world_size: int, dist: Any,
                        torch: Any, device: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    """Exercise the exact 4096-Token/rank workload before any formal cell."""
    from .granite_training_policy_native_v1 import NativeTrainingPlanner
    from .granite_training_replica_v1 import install_training_routing

    torch.manual_seed(int(config["training"]["seed"]) + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["training"]["seed"]) + rank)
    model, names = _load_model(model_dir, config, device)
    initial_hash = _initial_state_sample_hash(model)
    if dist.is_initialized():
        state_hashes: list[Any] = [None] * world_size
        dist.all_gather_object(state_hashes, initial_hash)
        if len(set(state_hashes)) != 1:
            raise GraniteFullTrainingError("capacity-smoke checkpoint sample differs across ranks")
    native = NativeTrainingPlanner(policy_library, policy="structured")
    planner = RecordingPlanner(native)
    handle = None
    collector = None
    optimizer = None
    try:
        handle = install_training_routing(model, planner, schedule_library, world_size, rank)
        collector = GateRouteCollector(
            ((name, model.get_submodule(name)) for name in handle.routed_layer_names)
        )
        optimizer = _create_optimizer(torch, handle, config["training"])
        local_batch_size = 4
        sequence_length = int(config["workload"]["sequence_length"])
        rows = make_windows(
            token_ids, sequence_length=sequence_length, batch_size=local_batch_size,
            rank=rank, world_size=world_size, step_index=0,
            stride=int(config["input"]["window_stride_tokens"]),
        )
        input_ids, labels = _load_token_batch(torch, rows, device)
        record = _record_step(
            torch=torch, dist=dist, model=model, handle=handle, planner=planner,
            collector=collector, optimizer=optimizer, input_ids=input_ids, labels=labels,
            device=device, expected_layers=handle.routed_layer_names, rank=rank,
            world_size=world_size, local_batch_size=local_batch_size,
            sequence_length=sequence_length, phase="capacity-smoke", phase_step=0,
            load_index=-1, policy_id="structured", policy_label="capacity gate only",
            run_id=run_id, run_dir=run_dir, input_hash=token_batch_digest(rows),
            seed=int(config["training"]["seed"]) + rank,
            initial_state_hash=initial_hash, records_file="preflight-steps.jsonl",
        )
        del input_ids, labels
        result = {
            "schema": "granite_full_training_capacity_smoke_v1",
            "status": "passed",
            "policy": "structured",
            "local_batch_size": local_batch_size,
            "sequence_length": sequence_length,
            "tokens_per_rank": local_batch_size * sequence_length,
            "global_input_tokens": world_size * local_batch_size * sequence_length,
            "world_size": world_size,
            "moe_layers": list(names),
            "peak_cuda_memory_allocated_bytes_by_rank": record.get("peak_cuda_memory_allocated_bytes_by_rank"),
            "peak_cuda_memory_reserved_bytes_by_rank": record.get("peak_cuda_memory_reserved_bytes_by_rank"),
            "direct_step_ms_max_rank": record.get("full_step_ms"),
            "gradient_parameters_checked_by_rank": record.get("gradient_parameters_checked_by_rank"),
            "replica_identity": record.get("replica_identity"),
            "scope": "one non-measured complete full-model FP32 forward/backward/sync/AdamW step at the largest formal local batch; no batch reduction or fallback",
        }
        if rank == 0:
            _write_json(run_dir / "capacity-smoke.json", result)
        return result
    finally:
        if collector is not None:
            collector.close()
        if handle is not None:
            handle.close()
        del optimizer, planner, native, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _environment(torch: Any, dist: Any, config: Mapping[str, Any], *, formal: bool, rank: int):
    import transformers

    if transformers.__version__ != config["transformers_abi"]:
        raise GraniteFullTrainingError(
            f"Transformers ABI mismatch: found {transformers.__version__}; expected {config['transformers_abi']}"
        )
    torch.set_num_threads(int(config["runtime"]["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(bool(config["training"]["deterministic_algorithms"]))
    if not torch.cuda.is_available():
        if formal:
            raise GraniteFullTrainingError("CUDA is required for the six-GPU full-checkpoint runner")
        env = {
            "rank": rank, "local_rank": 0, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "nccl_version": None, "device_name": "CPU tiny Transformers control",
            "compute_capability": None, "device_uuid": "", "pci_bus_id": "",
            "free_memory_bytes_before_model": None, "total_memory_bytes": None,
            "cpu_threads": torch.get_num_threads(), "tf32_matmul": False,
            "tf32_cudnn": False, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "formal_distributed": False, "world_size": 1,
        }
        return torch.device("cpu"), env
    if formal:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() != 6:
            raise GraniteFullTrainingError("formal mode requires torchrun with an initialized six-rank process group")
        if str(dist.get_backend()).lower() != "nccl":
            raise GraniteFullTrainingError("formal mode requires the NCCL process-group backend")
        if torch.cuda.device_count() != 6:
            raise GraniteFullTrainingError(f"formal mode requires six visible GPUs, found {torch.cuda.device_count()}")
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != rank or not 0 <= local_rank < 6:
            raise GraniteFullTrainingError("torchrun LOCAL_RANK does not match global rank on this single-node host")
        torch.cuda.set_device(local_rank)
    else:
        if torch.cuda.device_count() < 1:
            raise GraniteFullTrainingError("tiny loopback smoke requires one CUDA device")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank if 0 <= local_rank < torch.cuda.device_count() else 0)
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    cc = [int(props.major), int(props.minor)]
    if formal:
        if cc != config["runtime"]["compute_capability"]:
            raise GraniteFullTrainingError(f"formal GPU is SM{cc[0]}{cc[1]}; required SM70")
        if config["runtime"]["device_name_contains"].lower() not in props.name.lower():
            raise GraniteFullTrainingError(f"formal GPU {props.name!r} does not match required V100 profile")
        if int(free_bytes) < int(config["runtime"]["minimum_free_memory_bytes"]):
            raise GraniteFullTrainingError(
                f"GPU has {free_bytes} free bytes; at least {config['runtime']['minimum_free_memory_bytes']} are required"
            )
        torch_version = torch.__version__
        if torch_version not in config["runtime"]["formal_torch_versions"]:
            raise GraniteFullTrainingError(f"formal torch build is not pinned: {torch_version}")
    env = {
        "rank": rank,
        "local_rank": int(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version() if formal else None,
        "device_name": props.name,
        "compute_capability": cc,
        "device_uuid": str(getattr(props, "uuid", "")),
        "pci_bus_id": str(getattr(props, "pci_bus_id", "")),
        "free_memory_bytes_before_model": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
        "cpu_threads": torch.get_num_threads(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "formal_distributed": formal,
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else 1,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return device, env


def _run_formal_collectives(torch: Any, dist: Any, rank: int):
    value = torch.tensor([rank], dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if int(value.item()) != 15:
        raise GraniteFullTrainingError(f"six-rank NCCL reduction returned {value.item()}, expected 15")
    gathered = [None] * 6
    dist.all_gather_object(gathered, {
        "rank": rank, "local_rank": int(os.environ["LOCAL_RANK"]),
        "uuid": str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", "")),
    })
    if sorted(int(row["rank"]) for row in gathered) != list(range(6)):
        raise GraniteFullTrainingError("NCCL collectives did not include exactly six ranks")
    if sorted(int(row["local_rank"]) for row in gathered) != list(range(6)):
        raise GraniteFullTrainingError("torchrun ranks do not map one-to-one to six local GPUs")
    uuids = [row["uuid"] for row in gathered if row["uuid"]]
    if uuids and len(uuids) != len(set(uuids)):
        raise GraniteFullTrainingError("two source ranks report the same GPU UUID")
    return gathered


def _broadcast_run_dir(torch: Any, dist: Any, output_root: str | Path, run_id: str, rank: int):
    value = [None]
    if rank == 0:
        try:
            value[0] = {"path": str(_new_run_directory(output_root, run_id)), "error": None}
        except Exception as exc:
            value[0] = {"path": None, "error": f"{type(exc).__name__}: {exc}"}
    if dist.is_initialized():
        dist.broadcast_object_list(value, src=0)
    if value[0]["error"]:
        raise GraniteFullTrainingError(value[0]["error"])
    run_dir = Path(value[0]["path"])
    if rank != 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    return run_dir


def _write_initial_manifest(run_dir: Path, run_id: str, mode: str, config_path: Path,
                            config: Mapping[str, Any], args: argparse.Namespace):
    _write_json(run_dir / "manifest.json", {
        "schema": "granite_full_training_run_manifest_v1",
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_id": config["checkpoint_id"],
        "checkpoint_revision": config["checkpoint_revision"],
        "world_size": config["world_size"],
        "formal_benchmark_started": False,
        "cloud_executed_by_preparation": False,
        "native_policy_library_argument": str(Path(args.native_policy_library).expanduser().resolve()) if args.native_policy_library else None,
        "native_schedule_library_argument": str(Path(args.native_schedule_library).expanduser().resolve()) if args.native_schedule_library else None,
    })


def _require_native_libraries(policy_path: str | Path | None, schedule_path: str | Path | None):
    if not policy_path or not schedule_path:
        raise GraniteFullTrainingError("both --native-policy-library and --native-schedule-library are required")
    policy = Path(policy_path).expanduser().resolve(strict=True)
    schedule = Path(schedule_path).expanduser().resolve(strict=True)
    if not policy.is_file() or not schedule.is_file():
        raise GraniteFullTrainingError("native policy and schedule paths must be files")
    from .granite_training_policy_native_v1 import NativeTrainingPlanner
    planner = NativeTrainingPlanner(policy, policy="structured")
    if not planner.is_native:
        raise GraniteFullTrainingError("native planner did not identify itself as native")
    import ctypes
    try:
        library = ctypes.CDLL(str(schedule))
        getattr(library, "granite_cloud_build_schedule_v1")
    except (OSError, AttributeError) as exc:
        raise GraniteFullTrainingError(f"native route-schedule ABI is unavailable: {exc}") from exc
    return policy, schedule


def _run_cell(*, config: Mapping[str, Any], config_path: Path, model_dir: Path,
              policy_library: Path, schedule_library: Path, run_dir: Path,
              run_id: str, load_index: int, policy_id: str, local_batch_size: int,
              rank: int, world_size: int, dist: Any, torch: Any, device: Any,
              token_ids: Sequence[int]) -> dict[str, Any]:
    from .granite_training_policy_native_v1 import NativeTrainingPlanner
    from .granite_training_replica_v1 import install_training_routing

    policy_spec = next(row for row in config["policies"] if row["id"] == policy_id)
    train_config = config["training"]
    workload = config["workload"]
    step_count = int(workload["warmup_steps"]) + int(workload["measured_steps"])
    base_seed = int(train_config["seed"]) + rank
    torch.manual_seed(base_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(base_seed)
    model, _original_names = _load_model(model_dir, config, device)
    initial_state_hash = _initial_state_sample_hash(model)
    if dist.is_initialized():
        state_hashes: list[Any] = [None] * world_size
        dist.all_gather_object(state_hashes, initial_state_hash)
        if len(set(state_hashes)) != 1:
            raise GraniteFullTrainingError("checkpoint initial state sample differs across source ranks")
    native = NativeTrainingPlanner(policy_library, policy=policy_id)
    planner = RecordingPlanner(native)
    handle = None
    collector = None
    optimizer = None
    losses: list[float] = []
    total_times: list[float] = []
    try:
        handle = install_training_routing(model, planner, schedule_library, world_size, rank)
        collector = GateRouteCollector(
            ((name, model.get_submodule(name)) for name in handle.routed_layer_names)
        )
        optimizer = _create_optimizer(torch, handle, train_config)
        window_hashes = [
            token_batch_digest(make_windows(
                token_ids, sequence_length=int(workload["sequence_length"]),
                batch_size=local_batch_size, rank=rank, world_size=world_size,
                step_index=step, stride=int(config["input"]["window_stride_tokens"]),
            ))
            for step in range(step_count)
        ]
        if rank == 0:
            _write_jsonl(run_dir / "cell-starts.jsonl", {
                "schema": "granite_full_training_cell_start_v1", "run_id": run_id,
                "load_index": load_index, "policy": policy_id, "local_batch_size": local_batch_size,
                "world_size": world_size, "moe_layers": list(handle.routed_layer_names),
                "initial_state_sample_sha256": initial_state_hash,
                "source_window_sha256_by_rank_step": window_hashes,
                "optimizer_parameters": len(handle.optimizer_parameters),
                "dense_router_parameters": len(handle.dense_router_parameters),
                "local_expert_parameters": len(handle.local_expert_parameters),
                "optimizer": {"name": "AdamW", "foreach": False,
                              "learning_rate": float(train_config["learning_rate"]),
                              "betas": list(train_config["betas"]),
                              "epsilon": float(train_config["epsilon"]),
                              "weight_decay": float(train_config["weight_decay"])},
            })
        for step in range(step_count):
            if time.monotonic() - _RUN_STARTED_MONOTONIC > float(config["timing"]["run_budget_seconds"]):
                raise GraniteFullTrainingError("2700-second total run budget expired before the next optimizer step")
            rows = make_windows(
                token_ids, sequence_length=int(workload["sequence_length"]),
                batch_size=local_batch_size, rank=rank, world_size=world_size,
                step_index=step, stride=int(config["input"]["window_stride_tokens"]),
            )
            batch_hash = token_batch_digest(rows)
            input_ids, labels = _load_token_batch(torch, rows, device)
            seed = int(train_config["seed"]) + load_index * 100000 + local_batch_size * 10000 + step * 97 + rank
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            phase = "warmup" if step < int(workload["warmup_steps"]) else "measured"
            phase_step = step if phase == "warmup" else step - int(workload["warmup_steps"])
            record = _record_step(
                torch=torch, dist=dist, model=model, handle=handle, planner=planner,
                collector=collector, optimizer=optimizer, input_ids=input_ids, labels=labels,
                device=device, expected_layers=handle.routed_layer_names, rank=rank,
                world_size=world_size, local_batch_size=local_batch_size,
                sequence_length=int(workload["sequence_length"]), phase=phase,
                phase_step=phase_step, load_index=load_index, policy_id=policy_id,
                policy_label=policy_spec["label"], run_id=run_id, run_dir=run_dir,
                input_hash=batch_hash, seed=seed, local_step_index=step,
                initial_state_hash=initial_state_hash,
            )
            del input_ids, labels
            if rank == 0 and phase == "measured":
                losses.append(float(record["loss_mean_across_ranks"]))
                total_times.append(float(record["direct_step_wall_ms_max_rank"]))
        return {
            "load_index": load_index,
            "policy": policy_id,
            "policy_label": policy_spec["label"],
            "local_batch_size": local_batch_size,
            "tokens_per_rank": local_batch_size * int(workload["sequence_length"]),
            "global_tokens_per_step": world_size * local_batch_size * int(workload["sequence_length"]),
            "measured_steps": int(workload["measured_steps"]),
            "loss_median": statistics.median(losses) if rank == 0 else None,
            "loss_first_measured": losses[0] if rank == 0 else None,
            "loss_last_measured": losses[-1] if rank == 0 else None,
            "direct_step_ms_median": statistics.median(total_times) if rank == 0 else None,
            "direct_step_ms_p95": _percentile(total_times, 0.95) if rank == 0 else None,
            "tokens_per_second_median": statistics.median(
                (world_size * local_batch_size * int(workload["sequence_length"]) * 1000.0) / value
                for value in total_times
            ) if rank == 0 else None,
            "replica_identity_checks": "passed after every optimizer step",
        }
    finally:
        if collector is not None:
            collector.close()
        if handle is not None:
            handle.close()
        del optimizer, planner, native, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


_RUN_STARTED_MONOTONIC = 0.0


def _validate_formal_completion(rank: int, cell_summaries: Sequence[Mapping[str, Any]],
                               expected_cells: int) -> None:
    """Only rank 0 owns the summaries that are gathered into summary.json."""
    if rank == 0 and len(cell_summaries) != expected_cells:
        raise GraniteFullTrainingError(
            f"completed {len(cell_summaries)} formal cells, expected {expected_cells}"
        )


def _run(args: argparse.Namespace) -> int:
    global _RUN_STARTED_MONOTONIC
    config, config_path = load_config(args.config)
    mode = "validate-only" if args.validate_only else (
        "preflight" if args.preflight else "tiny-loopback-smoke" if args.tiny_loopback_smoke else "formal"
    )
    if args.validate_only:
        print(json.dumps({"status": "valid", "config": str(config_path), "schema": config["schema"]}, ensure_ascii=False))
        return 0
    tiny_mode = bool(args.tiny_loopback_smoke)
    if not args.model_dir and not tiny_mode:
        raise GraniteFullTrainingError("--model-dir is required for preflight and formal execution")
    model_dir = Path(args.model_dir).expanduser().resolve(strict=True) if args.model_dir else None
    policy_lib, schedule_lib = _require_native_libraries(args.native_policy_library, args.native_schedule_library)
    if not args.output_root or not args.run_id:
        raise GraniteFullTrainingError("--output-root and --run-id are required for execution modes")
    if args.preflight and tiny_mode:
        raise GraniteFullTrainingError("--preflight and --tiny-loopback-smoke are mutually exclusive")

    workspace_config = config["training"]
    expected_cublas = str(workspace_config["cublas_workspace_config"])
    prior_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior_cublas is not None and prior_cublas != expected_cublas:
        raise GraniteFullTrainingError(
            f"CUBLAS_WORKSPACE_CONFIG is {prior_cublas!r}; deterministic protocol requires {expected_cublas!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_cublas
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import torch.distributed as dist

    torch.set_num_threads(int(config["runtime"]["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(bool(workspace_config["deterministic_algorithms"]))
    if not tiny_mode:
        if not dist.is_available():
            raise GraniteFullTrainingError("torch.distributed is unavailable")
        if not dist.is_initialized():
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                raise GraniteFullTrainingError("formal and preflight modes must be launched with torchrun --nproc-per-node=6")
            timeout = __import__("datetime").timedelta(seconds=int(config["runtime"]["process_group_timeout_seconds"]))
            dist.init_process_group(backend="nccl", timeout=timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 6
        if dist.is_initialized():
            raise GraniteFullTrainingError("tiny loopback mode must run as one process without torch.distributed")
    formal_hardware = not tiny_mode
    device, env = _environment(torch, dist, config, formal=formal_hardware, rank=rank)
    run_dir = _broadcast_run_dir(torch, dist, args.output_root, args.run_id, rank) if not tiny_mode else _new_run_directory(args.output_root, args.run_id)
    if rank == 0:
        _write_initial_manifest(run_dir, args.run_id, mode, config_path, config, args)
    if dist.is_initialized():
        dist.barrier()
    _RUN_STARTED_MONOTONIC = time.monotonic()
    formal_started = False
    try:
        if not tiny_mode and world_size != 6:
            raise GraniteFullTrainingError(f"formal runner requires six ranks, got {world_size}")
        collective_rows = _run_formal_collectives(torch, dist, rank) if formal_hardware else []

        provenance = None
        provenance_error = None
        model_files = None
        if rank == 0:
            try:
                if model_dir is None:
                    provenance = {
                        "model_source": "tiny Transformers GraniteMoeConfig fixture",
                        "checkpoint_id": None,
                        "revision": None,
                        "native_policy_library": {"path": str(policy_lib), **_sha256_file(policy_lib)},
                        "native_schedule_library": {"path": str(schedule_lib), **_sha256_file(schedule_lib)},
                    }
                else:
                    provenance = _build_model_load_provenance(config, config_path, model_dir, policy_lib, schedule_lib)
                    provenance["model_files"] = verify_model_files(model_dir, config, config_path)
                    model_files = provenance["model_files"]
            except Exception as exc:
                provenance_error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            packet = [{"provenance": provenance, "model_files": model_files, "error": provenance_error}]
            dist.broadcast_object_list(packet, src=0)
            provenance, model_files, provenance_error = packet[0]["provenance"], packet[0]["model_files"], packet[0]["error"]
        if provenance_error:
            raise GraniteFullTrainingError(provenance_error)

        if model_dir is None:
            token_ids = [((index * 31) + 7) % 37 for index in range(512)]
            readme_token_count = len(token_ids)
        else:
            _tokenizer, token_ids = _tokenize_readme(model_dir, config)
            readme_token_count = len(token_ids)
        if dist.is_initialized():
            token_signature = hashlib.sha256(",".join(map(str, token_ids)).encode("ascii")).hexdigest()
            signatures: list[Any] = [None] * world_size
            dist.all_gather_object(signatures, (len(token_ids), token_signature))
            if len(set(signatures)) != 1:
                raise GraniteFullTrainingError("ranks loaded different README Token sequences")
        env_rows = [env]
        if dist.is_initialized():
            env_rows = [None] * world_size
            dist.all_gather_object(env_rows, env)

        from .granite_training_policy_native_v1 import NativeTrainingPlanner
        smoke_policy_ids = POLICY_IDS if tiny_mode else DISTRIBUTED_PREFLIGHT_POLICY_IDS
        smoke_validations = []
        for policy_id in smoke_policy_ids:
            smoke_seed = int(config["training"]["seed"])
            if device.type == "cuda":
                torch.cuda.manual_seed_all(smoke_seed + rank)
            torch.manual_seed(smoke_seed + rank)
            if model_dir is None:
                smoke_model, smoke_names = _load_tiny_transformers_model(torch, smoke_seed + rank, device)
            else:
                smoke_model, smoke_names = _load_model(model_dir, config, device)
            smoke_seq = int(config["workload"]["smoke_sequence_length"])
            if model_dir is None:
                smoke_rows = make_windows(
                    token_ids, sequence_length=smoke_seq, batch_size=1,
                    rank=rank, world_size=world_size, step_index=0, stride=17,
                )
            else:
                smoke_rows = make_windows(
                    token_ids, sequence_length=smoke_seq, batch_size=1,
                    rank=rank, world_size=world_size, step_index=0,
                    stride=int(config["input"]["window_stride_tokens"]),
                )
            smoke_ids, smoke_labels = _load_token_batch(torch, smoke_rows, device)
            planner_native = NativeTrainingPlanner(policy_lib, policy=policy_id)
            planner = RecordingPlanner(planner_native)
            validation = _run_parity_training_smoke(
                torch=torch, dist=dist, model=smoke_model, model_names=smoke_names,
                input_ids=smoke_ids, labels=smoke_labels, planner=planner,
                schedule_library=schedule_lib, world_size=world_size, rank=rank,
                config=config, device=device, run_dir=run_dir, run_id=args.run_id,
                mode="tiny-loopback-smoke" if tiny_mode else "preflight",
                full_checkpoint=model_dir is not None,
            )
            validation["policy"] = policy_id
            validation["model_source"] = "synthetic_tiny_transformers_granite" if model_dir is None else str(model_dir)
            smoke_validations.append(validation)
            del smoke_model, smoke_ids, smoke_labels, planner, planner_native
            if dist.is_initialized():
                dist.barrier()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        capacity = None
        if formal_hardware:
            if model_dir is None:
                raise GraniteFullTrainingError("formal capacity smoke requires the pinned checkpoint")
            capacity = _run_capacity_smoke(
                config=config, model_dir=model_dir, policy_library=policy_lib,
                schedule_library=schedule_lib, run_dir=run_dir, run_id=args.run_id,
                rank=rank, world_size=world_size, dist=dist, torch=torch,
                device=device, token_ids=token_ids,
            )
            if dist.is_initialized():
                dist.barrier()

        validation_rows: list[Any] = smoke_validations
        if dist.is_initialized():
            validation_rows = [None] * world_size
            dist.all_gather_object(validation_rows, smoke_validations)
        if rank == 0:
            _write_json(run_dir / "validation.json", {
                "schema": "granite_full_training_validation_v1",
                "status": "passed",
                "full_checkpoint_parity_gate": "passed" if model_dir is not None else "not_applicable_synthetic_tiny",
                "per_rank_policy_smokes": validation_rows,
                "capacity_smoke": capacity,
                "strict_tolerance": config["precision"],
                "optimizer_update_scope": "full-model routed forward/backward/AdamW update completed; pre-update parity checks include full logits and each MoE layer output, full embedding hidden gradient, full Router gradients, and sampled Expert gradients",
            })
            _write_json(run_dir / "preflight.json", {
                "schema": "granite_full_training_preflight_v1",
                "status": "passed",
                "mode": mode,
                "world_size": world_size if formal_hardware else 6,
                "environment_by_rank": env_rows,
                "collective_probe": collective_rows,
                "model_files": model_files,
                "readme_token_count": readme_token_count,
                "model_and_library_provenance": provenance,
                "full_checkpoint_parity_gate": "passed" if model_dir is not None else "not_applicable_synthetic_tiny",
                "capacity_smoke": capacity,
                "formal_benchmark_started": False,
            })
        if dist.is_initialized():
            dist.barrier()

        if args.preflight or tiny_mode:
            if rank == 0:
                _write_json(run_dir / "summary.json", {
                    "schema": "granite_full_training_summary_v1",
                    "status": "preflight_passed" if args.preflight else "loopback_smoke_passed",
                    "mode": mode,
                    "formal_cells_started": False,
                    "validation": smoke_validations if tiny_mode else smoke_validations[0],
                    "capacity_smoke": capacity,
                    "cloud_executed": False,
                })
                _finalize_manifest(run_dir, "preflight_passed" if args.preflight else "loopback_smoke_passed", formal_started=False)
            return 0

        formal_started = True
        if rank == 0:
            _write_json(run_dir / "manifest.json", {
                **_read_json(run_dir / "manifest.json"),
                "status": "formal_training_running",
                "formal_benchmark_started": True,
                "provenance": provenance,
                "preflight_status": "passed",
                "validation_status": "passed",
            })
        if dist.is_initialized():
            dist.barrier()

        cell_summaries = []
        global_error = None
        local_batch_sizes = [int(value) for value in config["workload"]["local_batch_sizes"]]
        for load_index, local_batch_size in enumerate(local_batch_sizes):
            order = config["load_policy_orders"][load_index]
            for policy_id in order:
                cell_error = None
                cell_summary = None
                try:
                    cell_summary = _run_cell(
                        config=config, config_path=config_path, model_dir=model_dir,
                        policy_library=policy_lib, schedule_library=schedule_lib,
                        run_dir=run_dir, run_id=args.run_id, load_index=load_index,
                        policy_id=policy_id, local_batch_size=local_batch_size,
                        rank=rank, world_size=world_size, dist=dist, torch=torch,
                        device=device, token_ids=token_ids,
                    )
                except Exception as exc:
                    cell_error = f"{type(exc).__name__}: {exc}"
                if dist.is_initialized():
                    errors: list[Any] = [None] * world_size
                    dist.all_gather_object(errors, cell_error)
                    cell_error = next((item for item in errors if item), None)
                if cell_error:
                    global_error = cell_error
                    if rank == 0:
                        _write_json(run_dir / "failure.json", {
                            "schema": "granite_full_training_failure_v1", "status": "failed",
                            "load_index": load_index, "policy": policy_id,
                            "local_batch_size": local_batch_size, "error": cell_error,
                            "elapsed_seconds": time.monotonic() - _RUN_STARTED_MONOTONIC,
                        })
                    break
                if rank == 0 and cell_summary is not None:
                    cell_summaries.append(cell_summary)
                if dist.is_initialized():
                    dist.barrier()
            if global_error:
                break
        if global_error:
            if rank == 0:
                _write_json(run_dir / "summary.json", {
                    "schema": "granite_full_training_summary_v1", "status": "failed",
                    "completed_cells": cell_summaries, "error": global_error,
                    "cloud_executed": False,
                })
                _finalize_manifest(run_dir, "failed", formal_started=True)
            return 1

        measured_cells = [
            {"policy": policy_id, "tokens_per_rank": batch_size * int(config["workload"]["sequence_length"]),
             "measured_steps": int(config["workload"]["measured_steps"])}
            for batch_size in local_batch_sizes for policy_id in POLICY_IDS
        ]
        _validate_formal_completion(
            rank, cell_summaries, int(config["workload"]["expected_formal_cells"])
        )
        if rank == 0:
            _write_json(run_dir / "verification.json", {
                "schema": "granite_full_training_verification_v1",
                "status": "passed",
                "world_size": world_size,
                "checkpoint_revision": config["checkpoint_revision"],
                "full_checkpoint_parity_gate": "passed",
                "measured_steps_by_cell": measured_cells,
                "total_global_optimizer_steps": int(config["workload"]["expected_global_optimizer_steps"]),
                "completed_formal_cells": len(cell_summaries),
                "capacity_smoke_passed": bool(capacity and capacity["status"] == "passed"),
                "replica_identity_check_cadence": "after every optimizer step, including parameter and AdamW state",
            })
            _write_json(run_dir / "summary.json", {
                "schema": "granite_full_training_summary_v1", "status": "completed",
                "cells": cell_summaries,
                "total_cells": len(cell_summaries),
                "expected_global_optimizer_steps": int(config["workload"]["expected_global_optimizer_steps"]),
                "elapsed_seconds": time.monotonic() - _RUN_STARTED_MONOTONIC,
                "cloud_executed": False,
                "quality_claim": "none; short README-window training comparison only",
            })
            _finalize_manifest(run_dir, "completed", formal_started=True)
        return 0
    except Exception as exc:
        if rank == 0:
            _write_json(run_dir / "failure.json", {
                "schema": "granite_full_training_failure_v1", "status": "failed",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - _RUN_STARTED_MONOTONIC,
            })
            if (run_dir / "manifest.json").exists():
                _finalize_manifest(run_dir, "failed", formal_started=formal_started)
        return 1
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _finalize_manifest(run_dir: Path, status: str, *, formal_started: bool):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    manifest["status"] = status
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["formal_benchmark_started"] = bool(formal_started)
    files = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp"):
            files[path.name] = _sha256_file(path)
    manifest["files"] = files
    _write_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--native-policy-library")
    parser.add_argument("--native-schedule-library")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--tiny-loopback-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"granite full-training runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "GraniteFullTrainingError", "GateRouteCollector", "RecordingPlanner",
    "load_config", "make_windows", "main", "token_batch_digest", "verify_model_files",
]
