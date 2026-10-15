"""Paired independent-step controls for the A100 G4 feedback experiment."""
from contextlib import contextmanager
import hashlib
import time

import torch
from torch import nn

from .granite_g4_feedback_planner_v1 import baseline_quotas
from .granite_communication_social_v2_g4_training_policy_v1 import SourceQuotaTrainingPlan


class ReplayRouter(nn.Module):
    """Freeze discrete selections, retain derivatives of selected logit softmax."""
    def __init__(self, original):
        super().__init__()
        self.original=original;self.top_k=original.top_k;self.num_experts=original.num_experts
        self.selected=None;self.recording=True

    def forward(self,hidden):
        logits=self.original.layer(hidden).float()
        if self.recording:
            self.selected=logits.topk(self.top_k,dim=1).indices.detach().clone()
        if self.selected is None or self.selected.shape != (hidden.shape[0],self.top_k):
            raise ValueError('frozen Gate shape missing or changed')
        weights=logits.gather(1,self.selected).softmax(1).type_as(hidden)
        ids=self.selected.flatten();order=ids.sort(0).indices
        sizes=torch.bincount(ids,minlength=self.num_experts).tolist()
        return order,order//self.top_k,weights.flatten()[order],sizes,logits


def install_gate_replay(model):
    routers=[]
    for m in model.modules():
        if type(m).__name__=='GraniteMoeMoE':
            m.router=ReplayRouter(m.router);routers.append(m.router)
    if not routers:raise ValueError('no Granite routers found')
    return routers


class PairedPolicy:
    is_native=True
    supports_source_counts=True
    def __init__(self,planner):
        self.planner=planner;self.policy='uniform';self.records=[]

    def plan(self,*args):raise ValueError('exact source counts required')

    def plan_with_source_counts(self,counts,gpus,rank_counts):
        if gpus!=4 or len(rank_counts)!=4 or any(len(r)!=32 for r in rank_counts):
            raise ValueError('G4/E32 required')
        if tuple(counts)!=tuple(sum(r[e] for r in rank_counts) for e in range(32)):
            raise ValueError('source/global counts disagree')
        device=self.planner.graph.row_features.device
        tensor=torch.tensor(rank_counts,device=device,dtype=torch.long)
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        if self.policy=='gnn': q=self.planner.quotas(tensor.flatten()).reshape(4,32,2)
        else:q=baseline_quotas(tensor,self.policy)
        if device.type=='cuda':torch.cuda.synchronize(device)
        resident_ms=(time.perf_counter()-start)*1000
        matrix=tuple(tuple(tuple(pair) for pair in row) for row in q.cpu().tolist())
        if any(sum(matrix[s][e])!=rank_counts[s][e] for s in range(4) for e in range(32)):
            raise ArithmeticError('quota conservation failed')
        left=tuple(sum(matrix[s][e][0] for s in range(4)) for e in range(32))
        self.records.append({'policy':self.policy,'resident_planner_ms':resident_ms,
            'source_counts':rank_counts,'source_quotas':matrix})
        return SourceQuotaTrainingPlan(self.policy,tuple(counts),tuple(map(tuple,rank_counts)),
            matrix,matrix,left,tuple(left[e]/counts[e] if counts[e] else 0 for e in range(32)),
            'direct',0,{'resident_planner_ms':resident_ms,'refinement':'none'})


def test_schedule(expert_ids,token_ids,route_slots,layout,quotas):
    from .granite_cloud_executor_v1 import build_execution_schedule
    return build_execution_schedule(tuple(expert_ids.tolist()),tuple(token_ids.tolist()),
        tuple(route_slots.tolist()),layout,quotas)


@contextmanager
def whole_expert_scope():
    from . import granite_communication_social_v2_training_replica_packed_v1 as packed
    old=packed.fixed_tile_linear;packed.fixed_tile_linear=torch.nn.functional.linear
    try:yield
    finally:packed.fixed_tile_linear=old


def snapshot(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}


def initialize_optimizer(model,optimizer):
    # Preallocate Adam state once, then restore model and reset it before each arm.
    saved=snapshot(model)
    for p in model.parameters():p.grad=torch.zeros_like(p)
    optimizer.step();model.load_state_dict(saved);optimizer.zero_grad(set_to_none=True)
    reset_optimizer(optimizer)


def reset_optimizer(optimizer):
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value,torch.Tensor):value.zero_()


def compare_tensors(actual,expected,*,rtol=1e-5,atol=1e-7):
    """No sampled gradients or tolerance escalation."""
    if set(actual)!=set(expected):raise AssertionError('parameter keys differ')
    maximum=0.
    for name,x in actual.items():
        y=expected[name].to(x.device)
        if not bool(torch.isfinite(x).all()):raise AssertionError('nonfinite '+name)
        torch.testing.assert_close(x,y,rtol=rtol,atol=atol,msg=name)
        maximum=max(maximum,float((x-y).abs().max()))
    return maximum


def gate_digest(routers):
    h=hashlib.sha256()
    for r in routers:h.update(r.selected.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_records(all_records,*,samples,layers):
    """Reject incomplete or mismatched paired logs before emitting completion."""
    import math
    if len(all_records)!=4:raise ValueError('four rank records required')
    for rank,rows in enumerate(all_records):
        if len(rows)!=samples*4:raise ValueError('incomplete arm/sample records')
        for sample in range(samples):
            group=[r for r in rows if r['sample']==sample]
            arms=['gnn','sinkhorn','lazarus'];k=sample%3
            if [r['arm'] for r in group]!=['uniform']+arms[k:]+arms[:k]:raise ValueError('arm order differs')
            if len({r['gate_sha256'] for r in group})!=1:raise ValueError('Gate replay differs')
            for r in group:
                if r['rank']!=rank or not r['passed'] or r['error'] is not None:raise ValueError('failed numerical record')
                if not math.isfinite(r['full_step_ms']) or r['full_step_ms']<=0:raise ValueError('invalid timing')
                if len(r['planner'])!=(layers if rank==0 else 0):raise ValueError('missing layer plans')
            if rank==0:
                for i in range(layers):
                    common=group[0]['planner'][i]['source_counts']
                    for r in group:
                        plan=r['planner'][i]
                        if plan['source_counts']!=common:raise ValueError('source counts differ')
                        q=plan['source_quotas']
                        if len(q)!=4 or any(len(row)!=32 for row in q):raise ValueError('quota shape')
                        for s in range(4):
                            for e in range(32):
                                if len(q[s][e])!=2 or any(type(v) is not int or v<0 for v in q[s][e]) or sum(q[s][e])!=common[s][e]:
                                    raise ValueError('quota conservation')
    return {'complete_ranks':4,'independent_samples':samples,'layers':layers,'paired_audit_passed':True}
