"""Isolated G4/E32 fixed-support transfer adapter; no solver corrections.

Torch reference CUDA path, not the G64 fused latency implementation.
All 128 source/Expert rows remain present, including zero demands.
"""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

from .moe_communication_social_v2 import CommunicationConfig, SparseBlock, SparseSocialProblem
from .granite_communication_social_v2_g4_training_policy_v1 import canonical_ring_rows


def default_model_config():
    # Explicit inherited proxy coefficients, not measured A100 calibration.
    return CommunicationConfig((0,0,0,0), mu_routes_per_ms=2513.238,
        dispatch_bytes_per_route=4096, return_bytes_per_route=4096,
        intra_bandwidth_bytes_per_ms=900_000_000., nic_bandwidth_gbps=200.)


def dense_reference_problem(counts, config):
    if len(counts) != 128 or any(type(v) is not int or v < 0 for v in counts):
        raise ValueError('128 nonnegative integer demands required')
    unit = SparseSocialProblem.from_rows(canonical_ring_rows(), config)
    blocks = tuple(SparseBlock(b.name, b.resource_names,
        tuple(tuple((q, c * counts[unit.variable_rows[v]]) for q,c in col)
              for v,col in enumerate(b.columns))) for b in unit.blocks)
    return SparseSocialProblem(tuple(replace(r,demand=n) for r,n in zip(unit.rows,counts)),
        config, unit.row_offsets, unit.variable_rows, unit.variable_gpus, blocks)


def validate_counts(counts, shape):
    import torch
    if counts.dtype != torch.int64 or tuple(counts.shape) != shape:
        raise ValueError('counts require matching int64 shape')
    if bool(((counts < 0) | (counts > 2**20)).any()):
        raise ValueError('counts outside supported exact-count range')


def baseline_quotas(counts, method):
    """Independent Expert columns, same fixed-placement formulas as G64 adapters."""
    import torch
    validate_counts(counts,(4,32))
    s=torch.arange(4,device=counts.device)[:,None]
    e=torch.arange(32,device=counts.device)[None,:]%4
    other=(e+1)%4
    total=counts.sum(0)
    if method == 'sinkhorn':
        # All four ranks share one node: both legal replicas have exp(-1/3) cost.
        a=counts.float(); m=torch.full_like(a,math.exp(-1/3.))
        v0=torch.ones_like(total,dtype=torch.float32);v1=v0.clone();target=total.float()*.5
        for _ in range(50):
            u=a/(m*v0+m*v1).clamp_min(1e-8)
            v0=target/(m*u).sum(0).clamp_min(1e-8)
            v1=target/(m*u).sum(0).clamp_min(1e-8)
        p=torch.where(total[None,:]>0,m*v0/(m*v0+m*v1).clamp_min(1e-8),.5)
        x=a*p;floor=x.floor();frac=x-floor
        q0=floor.long()+((frac>.5)|((frac==.5)&(e<other))).long()
    elif method == 'lazarus':
        t0=total//2+((total%2!=0)&(e[0]<other[0])).long();t1=total-t0
        own0=torch.where(s==e,torch.minimum(counts,t0),0)
        own1=torch.where(s==other,torch.minimum(counts,t1),0)
        rem=counts-own0-own1;prefix=rem.cumsum(0)
        denom=rem.sum(0).clamp_min(1);capacity=t0-own0.sum(0)
        q0=own0+prefix*capacity//denom-(prefix-rem)*capacity//denom
    elif method == 'uniform':
        q0=counts//2+((counts%2!=0)&(e<other)).long()
    else: raise ValueError('unknown method')
    q0=torch.minimum(counts,q0.clamp_min(0))
    return torch.stack((q0,counts-q0),-1)


class DenseSupportPlanner:
    def __init__(self,model,config,*,device):
        import torch
        from .moe_gpu_sparse_planner_v2 import SparseGpuPlanner
        from .moe_gpu_cached_warmstart_v1 import CachedGpuWarmStart
        from .moe_communication_social_v2_g256_sparse_warmstart_v1 import SparseSocialTorchGraph
        if config.gpu_nodes != (0,0,0,0): raise ValueError('single-node G4 required')
        unit=SparseSocialProblem.from_rows(canonical_ring_rows(),config)
        # Build the existing refresh buffers without its positive-only public API.
        c=CachedGpuWarmStart.__new__(CachedGpuWarmStart)
        c.torch=torch;c.runtime=SparseGpuPlanner(unit,device=device,use_graphs=False,fused=False)
        c.graph=SparseSocialTorchGraph.from_problem(unit,device=device);c.model=model
        g=c.graph
        c.block_ids=torch.cat([torch.full((b-a,),i,dtype=torch.long,device=device)
            for i,(a,b) in enumerate(zip(g.block_incidence_offsets,g.block_incidence_offsets[1:]))])
        c.mass_slots=g.incidence_variable*5+c.block_ids
        c.row_width=g.row_slot_mask.sum(-1).double()
        c.resource_total=torch.zeros_like(c.runtime.dual);c.variable_total=torch.zeros_like(c.runtime.p)
        c.variable_mass=torch.zeros((g.variable_count,5),dtype=torch.float64,device=device)
        c.row_mass=torch.zeros(g.row_count,dtype=torch.float64,device=device)
        self.cache=c;self.graph=g;self.model=model.to(device=device,dtype=torch.float64).eval()
        self.model.prepare(g)

    def refresh(self,counts):
        validate_counts(counts,(128,))
        self.cache.runtime.set_demands(counts)
        return self.cache._refresh_features()

    def quotas(self,counts):
        import torch
        with torch.no_grad():
            self.refresh(counts)
            self.shares=self.model.allocation_only(self.graph)
            return self.cache.runtime.round_shares(self.shares)

    def quality(self):
        import torch
        with torch.no_grad():
            shares,dual=self.model(self.graph)
            upper,lower=self.cache.runtime.certificate(shares,torch.cat(dual))
            return {'upper_ms':float(upper),'lower_ms':float(lower),'gap_ms':float(upper-lower)}


def load_checkpoint(path,device,expected_sha256):
    import torch
    from .moe_load_feedback_gnn_v1 import build_load_feedback_gnn
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    c=torch.load(path,map_location=device,weights_only=True)
    model=build_load_feedback_gnn(width=c['width'],price_width=c['price_width'],
        feedback_rounds=c['feedback_rounds'],pressure_features=c['pressure_features'])
    model.load_state_dict(c['state_dict'],strict=True)
    return model.to(device=device,dtype=torch.float64).eval()
