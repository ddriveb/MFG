"""Explicit fixed-shape FP32 Expert kernels for batch-partition parity controls.

Tile size is frozen at 128. This kernel changes GEMM shapes, not the MLP
mathematics, and must be used in both reference and routed comparison paths.
It does not relax any numerical assertion. The original backend is untouched.
"""
import torch
from torch.nn import functional as F
TILE_ROWS=128

def fixed_tile_linear(inputs, weight):
    if inputs.ndim != 2 or weight.ndim != 2:
        raise ValueError('fixed tile linear requires rank-two tensors')
    if inputs.shape[0] == 0:
        return F.linear(inputs,weight)
    pieces=[]
    for start in range(0,inputs.shape[0],TILE_ROWS):
        block=inputs[start:start+TILE_ROWS]
        count=block.shape[0]
        padded=F.pad(block,(0,0,0,TILE_ROWS-count))
        pieces.append(F.linear(padded,weight)[:count])
    return torch.cat(pieces,dim=0)

def fixed_parallel_experts_forward(self,inputs,expert_size):
    blocks=inputs.split(expert_size,dim=0)
    return torch.cat([fixed_tile_linear(block,self.weight[e]) for e,block in enumerate(blocks)],dim=0)

def fixed_routed_expert_mlp(self,inputs,expert_ids):
    result=inputs*0
    for expert in self.handle.resident_experts:
        indexes=torch.nonzero(expert_ids==expert,as_tuple=False).flatten()
        if not indexes.numel():
            continue
        projected=fixed_tile_linear(inputs.index_select(0,indexes),self.local_expert_w1[str(expert)])
        gate,value=projected.chunk(2,dim=-1)
        outputs=fixed_tile_linear(self.activation(gate)*value,self.local_expert_w2[str(expert)])
        result=result.index_copy(0,indexes,outputs)
    if expert_ids.numel() and bool(torch.any(~torch.isin(expert_ids,torch.as_tensor(self.handle.resident_experts,device=expert_ids.device,dtype=expert_ids.dtype)))):
        raise RuntimeError('fixed kernel received a nonresident Expert')
    return result

def install_fixed_expert_kernel_control():
    """Explicit process-local installation for a separately recorded experiment."""
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeParallelExperts
    from .granite_training_replica_v1 import _RoutedGraniteMoE
    GraniteMoeParallelExperts.forward=fixed_parallel_experts_forward
    _RoutedGraniteMoE._expert_mlp=fixed_routed_expert_mlp

def install_attention_checkpointing(model):
    """Checkpoint attention only; never replay Gate/planner/Expert collectives.

    This is an explicit memory-policy revision applied equally to all algorithms.
    Its backward recomputation belongs to measured complete-step time.
    """
    from torch.utils.checkpoint import checkpoint
    installed=[]
    for name,module in model.named_modules():
        if type(module).__name__!='GraniteMoeAttention':
            continue
        if getattr(module,'_mfg_attention_checkpointed',False):
            installed.append(name)
            continue
        original=module.forward
        def wrapped(*args,_original=original,_module=module,**kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        module.forward=wrapped
        module._mfg_attention_checkpointed=True
        installed.append(name)
    if not installed:
        raise RuntimeError('No GraniteMoeAttention modules found for attention checkpointing')
    return tuple(installed)
