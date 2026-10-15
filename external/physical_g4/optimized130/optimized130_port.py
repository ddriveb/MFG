"""G4/E32 integration of existing fused allocation/feature/baseline kernels."""
import os
import torch
import triton.language as tl
from triton.language.extra.cuda import libdevice
from mfg_hedge.granite_g4_feedback_planner_v1 import DenseSupportPlanner,baseline_quotas
import allocation130_port as allocation_module
import mfg_hedge.moe_gnn_allocation_features_v1 as feature_module
from sinkhorn130_port import build_dispatcher as build_sink
from lazarus130_port import build_dispatcher as build_laz

# Triton3.1 resolves these nested-jit symbols through module globals.
allocation_module.tl=tl
allocation_module.libdevice=libdevice
feature_module.tl=tl
feature_module.libdevice=libdevice
ACTIVE=None

class OptimizedPlanner(DenseSupportPlanner):
    def __init__(self,model,config,*,device):
        super().__init__(model,config,device=device)
        global ACTIVE
        ACTIVE=self
        self.prepared=False
        if int(os.environ.get('RANK','0'))!=0:return
        self.count_buffer=torch.ones(128,device=device,dtype=torch.long)
        self.fused=allocation_module.build_fused_allocation(model,self.graph,feedback_rounds=4)
        self.fused_refresh=feature_module.build_allocation_feature_refresh(self.cache)
        sources=[s for s in range(4) for e in range(32)]
        experts=[e for s in range(4) for e in range(32)]
        self.sink=build_sink(sources,experts,ranks=4,ranks_per_node=4,experts_count=32,device=device)
        self.laz=build_laz(sources,experts,ranks=4,experts_count=32,device=device)
        def gnn():
            self.fused_refresh(self.count_buffer)
            return self.cache.runtime.round_shares(self.fused())
        self.functions={'gnn':gnn,'sinkhorn':lambda:self.sink(self.count_buffer),'lazarus':lambda:self.laz(self.count_buffer)}
        self.graphs={};self.outputs={}
        for name,fn in self.functions.items():
            side=torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side),torch.no_grad():
                for _ in range(3):fn()
            torch.cuda.current_stream(device).wait_stream(side);torch.cuda.synchronize(device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph),torch.no_grad():out=fn()
            self.graphs[name]=graph;self.outputs[name]=out
            print('PREPARED_FUSED_GRAPH '+name,flush=True)
        self.prepared=True

    def dispatch(self,counts,arm):
        if not self.prepared:raise RuntimeError('planner runs on rank0 only')
        if counts.device!=self.count_buffer.device or counts.dtype!=torch.int64 or counts.numel()!=128:
            raise ValueError('matching G4/E32 resident int64 counts required')
        self.count_buffer.copy_(counts.reshape(128))
        self.graphs[arm].replay()
        return self.outputs[arm]

    def quotas(self,counts):
        return self.dispatch(counts,'gnn')

def optimized_baseline(counts,method):
    if method=='uniform':return baseline_quotas(counts,method)
    return ACTIVE.dispatch(counts,method).reshape(4,32,2)
