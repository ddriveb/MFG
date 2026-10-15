"""Dedicated G4 source-aware neural warm start and 36 native corrections."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping
from .moe_communication_social_v2 import CommunicationConfig, RouteRow
from .moe_communication_social_v2_g4_fast_runtime_v1 import G4FastWarmStartRuntime

WORLD_SIZE=4
EXPERT_COUNT=32
POLICY_IDS=('social_v2_native36',)

def canonical_ring_rows():
    return tuple(RouteRow(s,e,1,(e%4,(e+1)%4)) for s in range(4) for e in range(32))

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True)
class SourceQuotaTrainingPlan:
    policy: str
    counts_by_expert: tuple
    source_counts_by_expert: tuple
    source_quota_matrix: tuple
    base_source_quota_matrix: tuple
    quota_left: tuple
    shares_left: tuple
    status: str
    iterations: int
    diagnostics: Mapping

class CommunicationSocialV2TrainingPolicy:
    is_native=True
    supports_source_counts=True
    def __init__(self,policy,*,config,config_path,fast_runtime_library,resource_refiner_library,legacy_adapter_library):
        import torch
        from .moe_communication_social_v2_g4_warmstart_v1 import FixedBudgetWarmStartNet
        del resource_refiner_library,legacy_adapter_library
        if policy not in POLICY_IDS or config['world_size']!=4 or config['social_v2']['steps']!=36:
            raise ValueError('only dedicated G4 native36 is supported')
        social=config['social_v2']
        if social.get('checkpoint_prepared') is not True: raise ValueError('prepare a dedicated G4 network first')
        path=Path(config_path).resolve().parent/social['checkpoint_file']
        if _digest(path)!=social['checkpoint_sha256']: raise ValueError('G4 checkpoint hash mismatch')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        meta=checkpoint['metadata']
        if meta.get('topology_id')!='single_node_g4' or meta.get('feature_count')!=16 or meta.get('variable_count')!=32:
            raise ValueError('checkpoint dimensions are not G4; G6 transfer is forbidden')
        self.model=FixedBudgetWarmStartNet('single_node_g4').double().cpu()
        self.model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.model.eval()
        self.mean=checkpoint['feature_mean']; self.std=checkpoint['feature_std']
        if len(self.mean)!=16 or len(self.std)!=16 or any(v<=0 for v in self.std):
            raise ValueError('invalid G4 train-only feature normalization')
        if torch.get_num_threads()!=1: raise ValueError('CPU planner requires one Torch thread')
        raw=dict(social['communication_model']); raw['gpu_nodes']=tuple(raw['gpu_nodes'])
        self.rows=canonical_ring_rows()
        self.runtime=G4FastWarmStartRuntime(self.rows,CommunicationConfig(**raw),library_path=fast_runtime_library)
        self.policy=policy
        self.provenance={'policy_id':policy,'topology_id':'single_node_g4','checkpoint_sha256':social['checkpoint_sha256'],
                         'correction_budget':36,'checkpoint_training':meta,'native_implementation_required':True,
                         'batch_refinement':False}
    def plan(self,counts,gpus):
        raise ValueError('source counts are mandatory')
    def plan_with_source_counts(self,counts,gpus,rank_counts):
        counts=tuple(counts); sources=tuple(tuple(r) for r in rank_counts)
        if gpus!=4 or len(counts)!=32 or len(sources)!=4 or any(len(r)!=32 for r in sources):
            raise ValueError('expected G4/E32 source counts')
        if any(type(v) is not int or v<0 for r in sources for v in r): raise ValueError('invalid source counts')
        if any(type(v) is not int or v<0 for v in counts): raise ValueError('invalid global counts')
        if counts!=tuple(sum(sources[s][e] for s in range(4)) for e in range(32)):
            raise ValueError('source/global Gate count mismatch')
        out=self.runtime.plan(tuple(v for r in sources for v in r),self.model,self.mean,self.std,steps=36)
        matrix=tuple(tuple(out.quotas_by_row[s*32+e] for e in range(32)) for s in range(4))
        if any(sum(matrix[s][e])!=sources[s][e] or any(type(v) is not int or v<0 for v in matrix[s][e])
               for s in range(4) for e in range(32)): raise ValueError('quota legality failure')
        left=tuple(sum(matrix[s][e][0] for s in range(4)) for e in range(32))
        diagnostics={'status':'fixed_budget','native_policy_id':self.policy,'native_policy_status':'fixed_budget',
                     'correction_budget':36,'actual_iterations':36,'iterations':36,'warmstart_config_steps':36,
                     'model_ms':out.model_ms,'native_correction_ms':out.native_ms,
                     'normalization_ms':out.normalization_ms,'operator_norm_bound':out.operator_norm_bound,
                     'source_quota_realization':'exact_source_row_broadcast','refinement':'none'}
        return SourceQuotaTrainingPlan(self.policy,counts,sources,matrix,matrix,left,
            tuple(left[e]/counts[e] if counts[e] else 0.0 for e in range(32)),'fixed_budget',36,diagnostics)
    def close(self):
        if self.runtime is not None: self.runtime.close(); self.runtime=None
