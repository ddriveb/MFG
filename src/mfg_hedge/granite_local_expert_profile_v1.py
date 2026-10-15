"""Isolated real-GPU Expert component replay for Ticket06.

Only the local routed Expert MLP is measured. Gate, quota planning, collectives,
attention, LM loss and a complete model step are outside this component.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib, json, math, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .granite_training_numeric_kernels_v1 import TILE_ROWS, fixed_routed_expert_mlp

FROZEN_PROTOCOL_SHA256 = "a8213eb0ce45769d1b88e06bc1893c5492a8fe55fe573d088adaee604948b5eb"
PROFILE_INPUT_SCHEMA = "granite-local-expert-profile-input-v1"
CPU_GENERATION_THREADS = 1
PROFILE_SEEDS = tuple(range(742000, 742008))
PROFILE_ARMS = ("joint_0", "joint_16", "chain_v1", "lazarus_local_first", "tempo_paper_inspired_no_lp", "sinkhorn_relaxed")
_PROFILE_RELATIVE = Path("artifacts/communication-social-v2-high-bandwidth-20261006-a/protocol.json")
_MODEL_MANIFEST_RELATIVE = Path("configs/granite_cloud_model_manifest_20261001_a.json")
_MODEL_RELATIVE = Path("artifacts/granite-moe-local-stage-20260929-a/model")
_W1_KEY = "model.layers.23.block_sparse_moe.input_linear.weight"
_W2_KEY = "model.layers.23.block_sparse_moe.output_linear.weight"
_PINNED_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
_EXPECTED_NODES = {"single_node_g6": (0,0,0,0,0,0), "three_nodes_2_gpu_each": (0,0,1,1,2,2)}

class ProfileInputError(ValueError):
    """A plan, quota, checkpoint, or measurement contract is invalid."""

@dataclass(frozen=True)
class ProfileRow:
    source_gpu: int
    expert_id: int
    demand: int
    allowed_gpus: tuple[int, ...]
    quotas_by_allowed_gpu: tuple[int, ...]

@dataclass(frozen=True)
class ProfileCase:
    topology_id: str
    gpu_nodes: tuple[int, ...]
    seed: int
    arm: str
    rows: tuple[ProfileRow, ...]
    source_rows_sha256: str
    quota_sha256: str

@dataclass(frozen=True)
class ProfileInput:
    profiles: tuple[ProfileCase, ...]
    sha256: str
    protocol_sha256: str

@dataclass(frozen=True)
class AggregateRoutePlan:
    gpu_expert_route_ids: tuple[tuple[tuple[int, ...], ...], ...]
    route_identities: tuple[tuple[int, int, int], ...]
    expert_route_counts: tuple[int, ...]
    total_routes: int

    def rank_batch_metrics(self, rank: int) -> dict[str, Any]:
        if type(rank) is not int or not 0 <= rank < 6:
            raise ProfileInputError("rank is outside the six-GPU profile")
        counts = tuple(len(x) for x in self.gpu_expert_route_ids[rank])
        residents = ring_resident_experts(rank)
        if any(n and e not in residents for e,n in enumerate(counts)):
            raise ProfileInputError("destination received a nonresident Expert")
        tiles = tuple((n+TILE_ROWS-1)//TILE_ROWS if n else 0 for n in counts)
        pad = tuple(t*TILE_ROWS-n if t else 0 for n,t in zip(counts,tiles))
        return {"rank":rank,"resident_experts":list(residents),"expert_batch_counts":list(counts),
                "nonempty_expert_batches":sum(n>0 for n in counts),"route_count":sum(counts),
                "max_expert_batch_routes":max(counts,default=0),"input_projection_tiles":sum(tiles),
                "output_projection_tiles":sum(tiles),"input_projection_padding_rows":sum(pad),
                "output_projection_padding_rows":sum(pad),"padded_rows_by_expert_input":list(pad),
                "padded_rows_by_expert_output":list(pad)}

@dataclass(frozen=True)
class SyntheticRouteData:
    hidden: torch.Tensor
    upstream: torch.Tensor
    route_identities: tuple[tuple[int,int,int], ...]
    generator_scheme: str
    generator_seeds_sha256: str

@dataclass(frozen=True)
class Layer23Weights:
    w1: torch.Tensor
    w2: torch.Tensor
    provenance: Mapping[str, Any]

@dataclass(frozen=True)
class PhaseTiming:
    cuda_event_ms: float
    synchronized_wall_ms: float

@dataclass(frozen=True)
class RankBatch:
    route_ids: torch.Tensor
    inputs: torch.Tensor
    expert_ids: torch.Tensor
    upstream: torch.Tensor
    metrics: Mapping[str, Any]

class RoutedExpertMLP(nn.Module):
    """Cloud-like pure local routed MLP, using the existing fixed-tile kernel."""
    def __init__(self, resident_experts, w1_by_expert, w2_by_expert, *, use_checkpoint=True):
        super().__init__()
        residents=tuple(sorted(resident_experts))
        if not residents or len(set(residents))!=len(residents) or any(type(e) is not int or not 0<=e<32 for e in residents):
            raise ProfileInputError("resident Expert ids must be nonempty, unique and in [0,32)")
        if set(w1_by_expert)!=set(residents) or set(w2_by_expert)!=set(residents):
            raise ProfileInputError("packed weights must exactly match resident Experts")
        self.handle=SimpleNamespace(resident_experts=residents)
        self.local_expert_w1=nn.ParameterDict(); self.local_expert_w2=nn.ParameterDict()
        for e in residents:
            w1=torch.as_tensor(w1_by_expert[e]); w2=torch.as_tensor(w2_by_expert[e])
            if w1.ndim!=2 or w2.ndim!=2 or w1.shape[1]!=w2.shape[0] or w1.shape[0]!=2*w2.shape[1]:
                raise ProfileInputError("W1/W2 packed projection shapes are incompatible")
            if w1.dtype not in (torch.float32,torch.float64) or w2.dtype!=w1.dtype:
                raise ProfileInputError("resident Expert parameters must be FP32")
            self.local_expert_w1[str(e)]=nn.Parameter(w1.detach().clone())
            self.local_expert_w2[str(e)]=nn.Parameter(w2.detach().clone())
        self.activation=F.silu
        self.use_checkpoint=bool(use_checkpoint)

    def _expert_mlp(self, inputs, expert_ids):
        # Preserve cloud component data movement: nonzero, index_select, tiled
        # W1, SwiGLU, tiled W2, index_copy and the resident-Expert guard.
        return fixed_routed_expert_mlp(self,inputs,expert_ids)

    def forward(self, inputs, expert_ids):
        if inputs.ndim!=2 or expert_ids.ndim!=1 or inputs.shape[0]!=expert_ids.numel():
            raise ProfileInputError("inputs and Expert ids must have shapes [routes,H] and [routes]")
        if inputs.dtype!=next(self.parameters()).dtype or expert_ids.dtype!=torch.int64:
            raise ProfileInputError("routed Expert inputs/ids must be FP32/INT64")
        if self.training and torch.is_grad_enabled() and inputs.numel() and self.use_checkpoint:
            return checkpoint(self._expert_mlp,inputs,expert_ids,use_reentrant=False)
        return self._expert_mlp(inputs,expert_ids)

class AdamWReplayState:
    """Fixed resident parameter group with preallocated gradients/moments and reset."""
    def __init__(self, model, *, lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        self.parameters=tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters: raise ProfileInputError("AdamW needs resident parameters")
        self.optimizer=torch.optim.AdamW(self.parameters,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,foreach=False)
        self._initial={p:p.detach().clone() for p in self.parameters}
        self._grad={p:torch.zeros_like(p) for p in self.parameters}
        for p in self.parameters: p.grad=self._grad[p]
        # One zero-gradient step allocates AdamW state; reset below restores t=0.
        self.optimizer.step(); self.reset()

    def reset(self):
        with torch.no_grad():
            for p in self.parameters:
                p.copy_(self._initial[p]); self._grad[p].zero_(); p.grad=self._grad[p]
                state=self.optimizer.state[p]
                for key in ("exp_avg","exp_avg_sq","max_exp_avg_sq","step"):
                    value=state.get(key)
                    if isinstance(value,torch.Tensor): value.zero_()

    def assert_preallocated(self):
        for p in self.parameters:
            if p.grad is None: raise ProfileInputError("resident gradient buffer became None")
            state=self.optimizer.state.get(p,{})
            if not all(k in state for k in ("step","exp_avg","exp_avg_sq")):
                raise ProfileInputError("AdamW state is missing for a resident parameter")

class SynchronizedEventTimer:
    """Synchronized CUDA phase timing, retaining separate device and wall clocks."""
    def __init__(self, *, device=None, event_factory=None, clock_ns=time.perf_counter_ns, synchronize=None, stream=None):
        self.device=torch.device(device) if device is not None else None
        if event_factory is None:
            if self.device is None or not torch.cuda.is_available(): raise ValueError("CUDA event timer needs an available device")
            event_factory=lambda:torch.cuda.Event(enable_timing=True)
        if synchronize is None:
            if self.device is None: raise ValueError("timer needs a synchronization callback")
            synchronize=lambda:torch.cuda.synchronize(self.device)
        if stream is None and self.device is not None and torch.cuda.is_available(): stream=torch.cuda.current_stream(self.device)
        self._clock=clock_ns; self._sync=synchronize; self._stream=stream
        self._start=event_factory(); self._stop=event_factory()
    def measure(self, operation):
        self._sync(); start_ns=self._clock(); self._start.record(self._stream)
        result=operation(); self._stop.record(self._stream); self._stop.synchronize()
        wall=(self._clock()-start_ns)/1_000_000.0; event=float(self._start.elapsed_time(self._stop))
        if not math.isfinite(wall) or not math.isfinite(event) or wall<0 or event<0: raise RuntimeError("invalid phase duration")
        return result,PhaseTiming(event,wall)

def ring_resident_experts(rank: int, world_size: int = 6) -> tuple[int,...]:
    if type(world_size) is not int or world_size!=6: raise ProfileInputError("frozen profile requires six virtual GPUs")
    if type(rank) is not int or not 0<=rank<6: raise ProfileInputError("rank outside [0,6)")
    return tuple(e for e in range(32) if rank in (e%6,(e+1)%6))

def _parse_row(value):
    if isinstance(value,ProfileRow): row=value
    elif isinstance(value,Mapping):
        try:
            row=ProfileRow(value["source_gpu"],value["expert_id"],value["demand"],
                           tuple(value["allowed_gpus"]),tuple(value["quotas_by_allowed_gpu"]))
        except (KeyError,TypeError) as exc: raise ProfileInputError("source row missing a required field") from exc
    else: raise ProfileInputError("source rows must be objects or ProfileRow values")
    for name,v in (("source_gpu",row.source_gpu),("expert_id",row.expert_id),("demand",row.demand)):
        if type(v) is not int: raise ProfileInputError(f"{name} must be an integer")
    if not 0<=row.source_gpu<6 or not 0<=row.expert_id<32 or row.demand<0:
        raise ProfileInputError("source row dimensions or demand are out of range")
    if row.allowed_gpus!=(row.expert_id%6,(row.expert_id+1)%6):
        raise ProfileInputError("row must use the frozen ordered adjacent-ring placement")
    if len(row.quotas_by_allowed_gpu)!=2 or any(type(v) is not int or v<0 for v in row.quotas_by_allowed_gpu):
        raise ProfileInputError("each row requires two non-negative integer quotas")
    if sum(row.quotas_by_allowed_gpu)!=row.demand:
        raise ProfileInputError("integer quotas must conserve each source-row demand")
    return row

def compute_source_rows_sha256(rows):
    records=[[r.source_gpu,r.expert_id,r.demand,list(r.allowed_gpus)] for r in (_parse_row(x) for x in rows)]
    return hashlib.sha256(json.dumps(records,separators=(",",":"),ensure_ascii=True).encode("utf-8")).hexdigest()

def compute_quota_sha256(rows):
    records=[[r.source_gpu,r.expert_id,r.demand,list(r.allowed_gpus),list(r.quotas_by_allowed_gpu)]
             for r in (_parse_row(x) for x in rows)]
    return hashlib.sha256(json.dumps(records,separators=(",",":"),ensure_ascii=True).encode("utf-8")).hexdigest()

def aggregate_route_quotas(rows, *, gpu_count=6, expert_count=32, require_full_source_rows=True):
    if gpu_count!=6 or expert_count!=32: raise ProfileInputError("frozen profile requires G=6 and E=32")
    parsed=tuple(_parse_row(x) for x in rows)
    keys=tuple((r.source_gpu,r.expert_id) for r in parsed)
    if len(set(keys))!=len(keys): raise ProfileInputError("source-Expert rows must be unique")
    if keys!=tuple(sorted(keys)): raise ProfileInputError("source rows must be source-major and Expert-major")
    if require_full_source_rows and set(keys)!={(s,e) for s in range(6) for e in range(32)}:
        raise ProfileInputError("all 192 original source-Expert rows, including zeros, are required")
    if require_full_source_rows:
        per_source=[sum(r.demand for r in parsed if r.source_gpu==s) for s in range(6)]
        if any(n!=8192 for n in per_source): raise ProfileInputError("each source must preserve 8192 Gate routes")
    buckets=[[[] for _ in range(32)] for _ in range(6)]
    identities=[]; expert_counts=[0]*32
    for row in parsed:
        start=len(identities)
        identities.extend((row.source_gpu,row.expert_id,i) for i in range(row.demand))
        expert_counts[row.expert_id]+=row.demand
        used=0
        for gpu,n in zip(row.allowed_gpus,row.quotas_by_allowed_gpu):
            buckets[gpu][row.expert_id].extend(range(start+used,start+used+n)); used+=n
        if used!=row.demand: raise ProfileInputError("quota aggregation lost an original route")
    plan=AggregateRoutePlan(tuple(tuple(tuple(ids) for ids in gpu) for gpu in buckets),
                            tuple(identities),tuple(expert_counts),len(identities))
    if sum(plan.expert_route_counts)!=plan.total_routes: raise ProfileInputError("global Expert totals do not conserve routes")
    for rank in range(6): plan.rank_batch_metrics(rank)
    return plan

def _stream_seed(seed,source_gpu,expert_id,stream_kind):
    if type(seed) is not int or seed<0: raise ProfileInputError("synthetic seed must be a non-negative integer")
    raw=f"granite-local-expert-profile-v1\0{seed}\0{source_gpu}\0{expert_id}\0{stream_kind}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")&((1<<63)-1)

def _make_synthetic_route_data_single_thread(rows, *, seed, hidden_size=1024):
    if type(hidden_size) is not int or hidden_size<=0: raise ProfileInputError("hidden_size must be positive")
    parsed=tuple(_parse_row(x) for x in rows)
    identities=[]; hidden_parts=[]; upstream_parts=[]; seed_records=[]
    for row in parsed:
        identities.extend((row.source_gpu,row.expert_id,i) for i in range(row.demand))
        if not row.demand: continue
        for kind,target in (("hidden",hidden_parts),("upstream",upstream_parts)):
            value_seed=_stream_seed(seed,row.source_gpu,row.expert_id,kind)
            gen=torch.Generator(device="cpu"); gen.manual_seed(value_seed)
            target.append(torch.randn((row.demand,hidden_size),dtype=torch.float32,generator=gen))
            seed_records.append([row.source_gpu,row.expert_id,kind,value_seed])
    hidden=torch.cat(hidden_parts).contiguous() if hidden_parts else torch.empty((0,hidden_size),dtype=torch.float32)
    upstream=torch.cat(upstream_parts).contiguous() if upstream_parts else torch.empty((0,hidden_size),dtype=torch.float32)
    seed_hash=hashlib.sha256(json.dumps(seed_records,separators=(",",":")).encode("utf-8")).hexdigest()
    return SyntheticRouteData(hidden,upstream,tuple(identities),"seed-source-expert-stream-v1",seed_hash)

def make_synthetic_route_data(rows, *, seed, hidden_size=1024):
    previous_threads=torch.get_num_threads()
    if previous_threads!=CPU_GENERATION_THREADS: torch.set_num_threads(CPU_GENERATION_THREADS)
    try: return _make_synthetic_route_data_single_thread(rows,seed=seed,hidden_size=hidden_size)
    finally:
        if previous_threads!=CPU_GENERATION_THREADS: torch.set_num_threads(previous_threads)
def load_frozen_profile_protocol():
    root=Path(__file__).resolve().parents[2]; path=root/"artifacts/communication-social-v2-high-bandwidth-20261006-a/protocol.json"
    raw=path.read_bytes(); actual=hashlib.sha256(raw).hexdigest()
    if actual!=FROZEN_PROTOCOL_SHA256: raise ProfileInputError(f"frozen Ticket06 protocol hash mismatch: {actual}")
    protocol=json.loads(raw)
    prof=protocol.get("local_gpu_profile",{})
    if prof.get("physical_gpus")!=1 or prof.get("virtual_ranks")!=6: raise ProfileInputError("frozen device/rank profile changed")
    return protocol

def _generated_rows(seed,protocol):
    from .moe_communication_social_v2_runner import _make_case_rows
    case={"case_id":"single_node_g6","gpu_count":6,"expert_count":32,"seed":seed,
          "tokens_per_gpu":protocol["workload"]["tokens_per_gpu"],"top_k":protocol["workload"]["top_k"],
          "kind":"synthetic top-k route-count workload","repeat_selected_set_for_each_token":False}
    return tuple((r.source_gpu,r.expert_id,r.demand,tuple(r.allowed_gpus)) for r in _make_case_rows(case))

def validate_profile_input(payload, *, protocol=None):
    protocol=load_frozen_profile_protocol() if protocol is None else protocol
    if not isinstance(payload,Mapping) or payload.get("schema")!=PROFILE_INPUT_SCHEMA:
        raise ProfileInputError(f"profile input schema must be {PROFILE_INPUT_SCHEMA!r}")
    if payload.get("protocol_sha256")!=FROZEN_PROTOCOL_SHA256: raise ProfileInputError("profile input is not bound to frozen protocol")
    frozen=protocol["local_gpu_profile"]
    if tuple(frozen["seeds"])!=PROFILE_SEEDS or tuple(frozen["arms"])!=PROFILE_ARMS:
        raise ProfileInputError("frozen local GPU seed/arm matrix changed")
    topo={x["case_id"]:tuple(x["gpu_nodes"]) for x in protocol["scope"]["topologies"]}
    if topo!=_EXPECTED_NODES: raise ProfileInputError("frozen topology map changed")
    expected={(t,s,a) for t in topo for s in PROFILE_SEEDS for a in PROFILE_ARMS}
    raw=payload.get("profiles")
    if not isinstance(raw,list) or len(raw)!=96: raise ProfileInputError("profile input must contain exactly 96 cells")
    seen=set(); profiles=[]; rows_by_seed={}; expert_counts_by_seed={}
    for item in raw:
        if not isinstance(item,Mapping): raise ProfileInputError("each profile cell must be an object")
        topology=item.get("topology_id"); seed=item.get("seed"); arm=item.get("arm")
        if topology not in topo or type(seed) is not int or seed not in PROFILE_SEEDS or arm not in PROFILE_ARMS:
            raise ProfileInputError("profile contains an off-protocol topology, seed, or arm")
        key=(topology,seed,arm)
        if key in seen: raise ProfileInputError("duplicate profile matrix cell")
        seen.add(key)
        if tuple(item.get("gpu_nodes",()))!=topo[topology]: raise ProfileInputError("GPU layout differs from frozen topology")
        raw_rows=item.get("rows")
        if not isinstance(raw_rows,list): raise ProfileInputError("rows must be a JSON array")
        rows=tuple(_parse_row(x) for x in raw_rows)
        if len(rows)!=192 or tuple((r.source_gpu,r.expert_id) for r in rows)!=tuple((s,e) for s in range(6) for e in range(32)):
            raise ProfileInputError("all 192 original rows, including zeros, must stay in source-major order")
        per_source=[0]*6; per_expert=[0]*32
        for row in rows: per_source[row.source_gpu]+=row.demand; per_expert[row.expert_id]+=row.demand
        if any(n!=protocol["workload"]["tokens_per_gpu"]*protocol["workload"]["top_k"] for n in per_source):
            raise ProfileInputError("each source must conserve exactly 8192 Gate routes")
        identity=tuple((r.source_gpu,r.expert_id,r.demand,r.allowed_gpus) for r in rows)
        row_hash=compute_source_rows_sha256(rows); quota_hash=compute_quota_sha256(rows)
        if item.get("source_rows_sha256")!=row_hash or item.get("quota_sha256")!=quota_hash:
            raise ProfileInputError("source-row identity or quota hash mismatch")
        prior=rows_by_seed.setdefault(seed,identity)
        if prior!=identity: raise ProfileInputError("paired topology/arms must preserve original source rows")
        previous=expert_counts_by_seed.setdefault(seed,tuple(per_expert))
        if previous!=tuple(per_expert): raise ProfileInputError("paired arms changed global Expert work")
        profiles.append(ProfileCase(topology,topo[topology],seed,arm,rows,row_hash,quota_hash))
    if seen!=expected: raise ProfileInputError("profile matrix is missing frozen cells")
    for seed in PROFILE_SEEDS:
        if rows_by_seed[seed]!=_generated_rows(seed,protocol):
            raise ProfileInputError(f"source rows do not reproduce the frozen Gate seed {seed}")
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")
    ordered=tuple(sorted(profiles,key=lambda x:(x.topology_id,x.seed,PROFILE_ARMS.index(x.arm))))
    return ProfileInput(ordered,hashlib.sha256(canonical).hexdigest(),FROZEN_PROTOCOL_SHA256)

def load_profile_input(path):
    raw=Path(path).read_bytes()
    try: payload=json.loads(raw)
    except json.JSONDecodeError as exc: raise ProfileInputError("profile input is not valid JSON") from exc
    parsed=validate_profile_input(payload)
    return replace(parsed,sha256=hashlib.sha256(raw).hexdigest())

def _sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def load_layer23_weights(*, verify_full_file_hashes=True):
    """Read only the pinned checkpoint's two packed layer23 Expert tensors."""
    root=Path(__file__).resolve().parents[2]
    manifest_path=root/_MODEL_MANIFEST_RELATIVE; model_dir=root/_MODEL_RELATIVE
    manifest_bytes=manifest_path.read_bytes(); manifest=json.loads(manifest_bytes)
    if manifest.get("revision")!=_PINNED_REVISION: raise ProfileInputError("Granite checkpoint revision differs from frozen pin")
    files={x["name"]:x for x in manifest.get("files",[])}
    model_path=model_dir/"model.safetensors"; config_path=model_dir/"config.json"
    for name,path in (("model.safetensors",model_path),("config.json",config_path)):
        entry=files.get(name)
        if entry is None or not path.is_file() or path.stat().st_size!=int(entry["bytes"]):
            raise ProfileInputError(f"pinned checkpoint file missing or wrong size: {name}")
        if verify_full_file_hashes and _sha256_file(path)!=entry["sha256"]:
            raise ProfileInputError(f"pinned checkpoint hash mismatch: {name}")
    if not verify_full_file_hashes and _sha256_file(config_path)!=files["config.json"]["sha256"]:
        raise ProfileInputError("pinned config hash mismatch")
    try: from safetensors import safe_open
    except ImportError as exc: raise ProfileInputError("existing safetensors runtime is unavailable") from exc
    with safe_open(str(model_path),framework="pt",device="cpu") as safe:
        keys=set(safe.keys())
        if _W1_KEY not in keys or _W2_KEY not in keys: raise ProfileInputError("layer23 Expert weights are missing")
        raw_w1=safe.get_tensor(_W1_KEY); raw_w2=safe.get_tensor(_W2_KEY)
    if tuple(raw_w1.shape)!=(32,1024,1024) or tuple(raw_w2.shape)!=(32,1024,512):
        raise ProfileInputError("layer23 packed tensor shapes differ from H=1024/I=512")
    if raw_w1.dtype!=torch.bfloat16 or raw_w2.dtype!=torch.bfloat16:
        raise ProfileInputError("pinned checkpoint Expert tensors must be BF16 before FP32 conversion")
    provenance={"checkpoint_id":manifest.get("checkpoint_id"),"revision":manifest["revision"],
        "model_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
        "model_file_sha256":files["model.safetensors"]["sha256"],"config_sha256":files["config.json"]["sha256"],
        "input_weight_key":_W1_KEY,"output_weight_key":_W2_KEY,"input_weight_shape":list(raw_w1.shape),
        "output_weight_shape":list(raw_w2.shape),"checkpoint_dtype":"BF16","profile_dtype":"float32",
        "layer":23,"verified_full_file_hashes":bool(verify_full_file_hashes)}
    return Layer23Weights(raw_w1.to(torch.float32).contiguous(),raw_w2.to(torch.float32).contiguous(),provenance)

def build_rank_batch(plan,route_data,rank,*,device):
    metrics=plan.rank_batch_metrics(rank); residents=metrics["resident_experts"]
    selected=[rid for e in residents for rid in plan.gpu_expert_route_ids[rank][e]]
    expert_values=[e for e in residents for _ in plan.gpu_expert_route_ids[rank][e]]
    route_ids=torch.tensor(selected,dtype=torch.int64)
    expert_ids=torch.tensor(expert_values,dtype=torch.int64,device=device)
    inputs=route_data.hidden.index_select(0,route_ids).to(device=device,dtype=torch.float32).detach().requires_grad_(True)
    upstream=route_data.upstream.index_select(0,route_ids).to(device=device,dtype=torch.float32)
    return RankBatch(route_ids,inputs,expert_ids,upstream,metrics)

def make_rank_model(weights,rank,*,device):
    resident=ring_resident_experts(rank)
    return RoutedExpertMLP(resident,
        {e:weights.w1[e].to(device=device,dtype=torch.float32) for e in resident},
        {e:weights.w2[e].to(device=device,dtype=torch.float32) for e in resident},
        use_checkpoint=True).to(device=device)

@contextmanager
def _tf32_disabled():
    matmul=torch.backends.cuda.matmul.allow_tf32; cudnn=torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    try: yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul; torch.backends.cudnn.allow_tf32=cudnn

def _all_finite(tensor):
    return bool(torch.isfinite(tensor).all().item())

def profile_one_repeat(model,replay,batch,*,timer,device):
    """One repeat. Weight/moment reset is done by the caller before timing."""
    device=torch.device(device); replay.assert_preallocated(); batch.inputs.grad=None
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    allocated_before=int(torch.cuda.memory_allocated(device))
    with _tf32_disabled():
        output,forward=timer.measure(lambda:model(batch.inputs,batch.expert_ids))
        _,backward=timer.measure(lambda:output.backward(batch.upstream))
        if not _all_finite(output) or not _all_finite(batch.inputs.grad):
            raise RuntimeError("synthetic Expert output/input gradient is non-finite")
        for p in replay.parameters:
            if p.grad is None or not _all_finite(p.grad): raise RuntimeError("resident parameter gradient missing or non-finite")
        _,update=timer.measure(replay.optimizer.step)
    torch.cuda.synchronize(device)
    peak_alloc=int(torch.cuda.max_memory_allocated(device)); peak_reserved=int(torch.cuda.max_memory_reserved(device))
    return {"forward_cuda_event_ms":forward.cuda_event_ms,"forward_synchronized_wall_ms":forward.synchronized_wall_ms,
        "backward_cuda_event_ms":backward.cuda_event_ms,"backward_synchronized_wall_ms":backward.synchronized_wall_ms,
        "adamw_cuda_event_ms":update.cuda_event_ms,"adamw_synchronized_wall_ms":update.synchronized_wall_ms,
        "rank_forward_backward_cuda_event_ms":forward.cuda_event_ms+backward.cuda_event_ms,
        "rank_local_cuda_event_ms":forward.cuda_event_ms+backward.cuda_event_ms+update.cuda_event_ms,
        "rank_local_phase_wall_ms_sum":forward.synchronized_wall_ms+backward.synchronized_wall_ms+update.synchronized_wall_ms,
        "memory_allocated_before_bytes":allocated_before,"peak_memory_allocated_bytes":peak_alloc,
        "peak_memory_allocated_delta_bytes":max(0,peak_alloc-allocated_before),"peak_memory_reserved_bytes":peak_reserved,
        "forward_finite":True,"input_gradient_finite":True,"parameter_gradients_finite":True,
        "adamw_state_finite":all(_all_finite(replay.optimizer.state[p][k]) for p in replay.parameters for k in ("exp_avg","exp_avg_sq")),
        "route_count":int(batch.inputs.shape[0]),"nonempty_expert_batches":batch.metrics["nonempty_expert_batches"],
        "input_projection_tiles":batch.metrics["input_projection_tiles"],"output_projection_tiles":batch.metrics["output_projection_tiles"],
        "input_projection_padding_rows":batch.metrics["input_projection_padding_rows"],
        "output_projection_padding_rows":batch.metrics["output_projection_padding_rows"]}

def run_rank_repeats(profile,weights,*,input_manifest_sha256,device="cuda:0",route_data=None):
    if route_data is None: route_data=make_synthetic_route_data(profile.rows,seed=profile.seed)
    plan=aggregate_route_quotas(profile.rows); device=torch.device(device); records=[]
    for rank in range(6):
        model=make_rank_model(weights,rank,device=device)
        replay=AdamWReplayState(model,lr=1e-5,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
        batch=build_rank_batch(plan,route_data,rank,device=device); timer=SynchronizedEventTimer(device=device)
        for repeat in range(4):
            replay.reset(); torch.cuda.synchronize(device)
            timing=profile_one_repeat(model,replay,batch,timer=timer,device=device)
            records.append({"topology_id":profile.topology_id,"seed":profile.seed,"arm":profile.arm,"rank":rank,
                "repeat_index":repeat,"warmup":repeat==0,"measured":repeat!=0,
                "source_rows_sha256":profile.source_rows_sha256,"quota_sha256":profile.quota_sha256,
                "input_manifest_sha256":input_manifest_sha256,"synthetic_route_generator":route_data.generator_scheme,"synthetic_cpu_threads_for_generation":CPU_GENERATION_THREADS,
                "synthetic_route_generator_seeds_sha256":route_data.generator_seeds_sha256,
                "route_identity_sha256":hashlib.sha256(json.dumps(route_data.route_identities,separators=(",",":")).encode("ascii")).hexdigest(),
                "resident_experts":list(ring_resident_experts(rank)),
                "physical_execution":"one RTX 5090 D v2; six virtual ranks run sequentially",
                "unmeasured":["forward_all_to_all","backward_all_to_all","gradient_synchronization","Gate",
                    "attention","LM_loss","full_model_training","overlap"],**timing})
        del timer,batch,replay,model
        torch.cuda.empty_cache()
    return records

def critical_rank_proxy(records):
    """Sum phases within each rank/repeat, then take max; never sum phase maxima."""
    grouped={}
    for row in records:
        key=(row.get("topology_id"),row.get("seed"),row.get("arm"),row.get("repeat_index"))
        grouped.setdefault(key,[]).append(row)
    result=[]
    for key,rows in sorted(grouped.items(),key=lambda x:str(x[0])):
        if len(rows)!=6 or {r.get("rank") for r in rows}!=set(range(6)):
            raise ProfileInputError("critical-rank proxy requires six virtual ranks per repeat")
        fb=[(float(r["forward_cuda_event_ms"])+float(r["backward_cuda_event_ms"]),int(r["rank"])) for r in rows]
        full=[(float(r["forward_cuda_event_ms"])+float(r["backward_cuda_event_ms"])+float(r["adamw_cuda_event_ms"]),int(r["rank"])) for r in rows]
        fbt,fbr=max(fb,key=lambda x:(x[0],-x[1])); ft,fr=max(full,key=lambda x:(x[0],-x[1]))
        result.append({"topology_id":key[0],"seed":key[1],"arm":key[2],"repeat_index":key[3],
            "max_same_rank_forward_backward_cuda_event_ms":fbt,"forward_backward_rank":fbr,
            "max_same_rank_full_local_cuda_event_ms":ft,"full_local_rank":fr,
            "interpretation":"one-GPU sequential virtual-rank proxy; no communication or overlap"})
    return tuple(result)

def cuda_device_preflight(profile_input,*,device="cuda:0",safety_fraction=0.80):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ProfileInputError("frozen local profile requires exactly one available physical CUDA GPU")
    device=torch.device(device)
    if device.type!="cuda" or device.index not in (None,0): raise ProfileInputError("profile must use cuda:0")
    if torch.__version__!="2.11.0+cu128" or torch.version.cuda!="12.8":
        raise ProfileInputError("runtime differs from frozen Torch 2.11.0+cu128 profile")
    name=torch.cuda.get_device_name(0)
    if "RTX 5090 D v2" not in name: raise ProfileInputError(f"unexpected physical GPU: {name}")
    if not 0<safety_fraction<1: raise ProfileInputError("safety_fraction must be in (0,1)")
    max_routes=0; max_padded=0; max_cell=None
    for profile in profile_input.profiles:
        plan=aggregate_route_quotas(profile.rows)
        for rank in range(6):
            m=plan.rank_batch_metrics(rank)
            if m["route_count"]>max_routes:
                max_routes=m["route_count"]; max_cell={"topology_id":profile.topology_id,"seed":profile.seed,"arm":profile.arm,"rank":rank}
            max_padded=max(max_padded,sum(m["padded_rows_by_expert_input"]))
    residents=max(len(ring_resident_experts(r)) for r in range(6))
    params_per_expert=1024*1024+1024*512
    # Parameter, gradient, two AdamW moments and working activations; initial
    # reset weights are GPU-resident too, so include a fifth parameter copy.
    resident_bytes=residents*params_per_expert*4*5
    route_bytes=max_routes*1024*4*4
    activation_bytes=max_padded*(2*512+1024)*4*4
    estimate=resident_bytes+route_bytes+activation_bytes
    free,total=torch.cuda.mem_get_info(device)
    if estimate>int(free*safety_fraction):
        raise ProfileInputError(f"quota-derived memory estimate {estimate} exceeds the {safety_fraction:.0%} free-memory gate")
    return {"device_name":name,"torch_version":torch.__version__,"cuda_version":torch.version.cuda,
        "device_total_bytes":int(total),"device_free_bytes_at_preflight":int(free),
        "max_virtual_rank_routes":max_routes,"max_virtual_rank_padded_rows":max_padded,
        "max_virtual_rank_cell":max_cell,"resident_experts_max":residents,
        "estimated_peak_bytes":estimate,"safety_fraction_of_free_memory":safety_fraction,
        "fits_memory_gate":True,"tf32":False,"physical_gpu_count":1,"virtual_rank_count":6}

def run_frozen_cuda_profile_matrix(profile_input,weights,*,preflight,output_jsonl,root_reviewed=False,device="cuda:0"):
    """Run 96 frozen cells; caller must set root_reviewed only after code review."""
    if root_reviewed is not True: raise ProfileInputError("root code review is required before the formal CUDA matrix")
    if preflight.get("fits_memory_gate") is not True or preflight.get("physical_gpu_count")!=1:
        raise ProfileInputError("formal CUDA matrix requires a successful one-GPU quota memory preflight")
    if weights.provenance.get("revision")!=_PINNED_REVISION or weights.provenance.get("layer")!=23:
        raise ProfileInputError("formal CUDA matrix requires pinned Granite layer23 weights")
    path=Path(output_jsonl)
    if path.exists(): raise ProfileInputError("raw profile output path already exists; use a new run directory")
    path.parent.mkdir(parents=True,exist_ok=True)
    cases_by_seed={s:[p for p in profile_input.profiles if p.seed==s] for s in PROFILE_SEEDS}
    rollups=[]
    with path.open("x",encoding="utf-8",newline="\n") as output:
        for seed in PROFILE_SEEDS:
            seed_cases=cases_by_seed[seed]
            route_data=make_synthetic_route_data(seed_cases[0].rows,seed=seed,hidden_size=1024)
            for profile in seed_cases:
                records=run_rank_repeats(profile,weights,input_manifest_sha256=profile_input.sha256,device=device,route_data=route_data)
                for row in records:
                    output.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
                rollups.extend(critical_rank_proxy(records))
            del route_data
    rollup_path=path.with_name(path.stem+"_critical_rank_proxy.json")
    rollup_path.write_text(json.dumps({"interpretation":"max after same-rank phase addition; six virtual ranks ran sequentially on one GPU",
        "unmeasured":["forward_all_to_all","backward_all_to_all","gradient_synchronization","attention","LM_loss","full_model_training","overlap"],
        "rollups":rollups},indent=2)+"\n",encoding="utf-8")
    return {"input_manifest_sha256":profile_input.sha256,"raw_jsonl_path":str(path),
        "critical_rank_proxy_path":str(rollup_path),"profile_cells":len(profile_input.profiles),
        "raw_rows":96*6*4,"warmup_rows":96*6,"measured_rows":96*6*3,
        "critical_rank_proxy_count":len(rollups),"synthetic_cpu_threads_for_generation":CPU_GENERATION_THREADS,"physical_execution":"one physical GPU; six virtual ranks sequentially",
        "sequential_six_rank_wall_is_not_a_distributed_step":True}

def _main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,help="frozen 96-cell profile input JSON")
    parser.add_argument("--output",required=True,help="new JSONL path for rank/repeat records")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--execute-frozen-matrix",action="store_true")
    parser.add_argument("--root-reviewed",action="store_true",help="explicit root code-review gate for formal GPU matrix")
    args=parser.parse_args()
    if args.preflight_only==args.execute_frozen_matrix: parser.error("choose exactly one execution mode")
    profile_input=load_profile_input(args.input); weights=load_layer23_weights()
    preflight=cuda_device_preflight(profile_input); preflight["input_manifest_sha256"]=profile_input.sha256
    preflight["checkpoint_provenance"]=dict(weights.provenance)
    Path(args.output).with_suffix(".preflight.json").write_text(json.dumps(preflight,indent=2)+"\n",encoding="utf-8")
    if args.execute_frozen_matrix:
        result=run_frozen_cuda_profile_matrix(profile_input,weights,preflight=preflight,output_jsonl=args.output,root_reviewed=args.root_reviewed)
        print(json.dumps(result,indent=2))
    else: print(json.dumps(preflight,indent=2))
    return 0

if __name__=="__main__":  # pragma: no cover
    raise SystemExit(_main())
