"""Lazarus-inspired fixed-R2 quota kernel; not official Lazarus source.

Inputs are already validated, nonnegative resident int64 counts. Total demand
per Expert must be <= 2**31-1; int64 prefix products remain below 2**62.
No Token packing, collective, adaptive placement, or optimality certificate.
"""


def build_dispatcher(row_sources, row_experts, *, ranks=256, device='cuda:0'):
    if type(ranks) is not int or not 2 <= ranks <= 1024:
        raise ValueError('ranks must be an integer in [2,1024]')
    sources, experts = tuple(row_sources), tuple(row_experts)
    if not sources or len(sources) != len(experts):
        raise ValueError('nonempty equal source/Expert metadata required')
    if any(type(v) is not int or not 0 <= v < ranks for v in sources+experts):
        raise ValueError('source/Expert outside ring')
    if len(set(zip(sources, experts))) != len(sources):
        raise ValueError('duplicate source/Expert row')
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(counts, index, output, G: tl.constexpr, B: tl.constexpr):
        e = tl.program_id(0)
        s = tl.arange(0, B)
        idx = tl.load(index + e*G+s, s<G, other=-1)
        n = tl.load(counts+tl.maximum(idx, 0), (s<G)&(idx>=0), other=0)
        total = tl.sum(n, 0)
        other = (e+1)%G
        target0 = total//2 + ((total%2!=0)&(e<other)).to(tl.int64)
        target1 = total-target0
        local0 = tl.minimum(tl.sum(tl.where(s==e,n,0),0), target0)
        local1 = tl.minimum(tl.sum(tl.where(s==other,n,0),0), target1)
        own0 = tl.where(s==e,local0,0)
        own1 = tl.where(s==other,local1,0)
        remaining = n-own0-own1
        prefix = tl.cumsum(remaining,0)
        denominator = tl.maximum(total-local0-local1,1)
        capacity0 = target0-local0
        q0 = own0 + prefix*capacity0//denominator - (prefix-remaining)*capacity0//denominator
        tl.store(output+idx*2,q0,(s<G)&(idx>=0))
        tl.store(output+idx*2+1,n-q0,(s<G)&(idx>=0))

    class Dispatcher:
        def __init__(self):
            self.device=torch.device(device)
            if self.device.type!='cuda':raise ValueError('CUDA required')
            index=[-1]*(ranks*ranks)
            for i,(s,e) in enumerate(zip(sources,experts)):index[e*ranks+s]=i
            self.index=torch.tensor(index,dtype=torch.int64,device=self.device)
            self.quotas=torch.empty((len(sources),2),dtype=torch.int64,device=self.device)

        def __call__(self, counts):
            if counts.device!=self.quotas.device or counts.dtype!=torch.int64 or counts.shape!=(len(sources),) or not counts.is_contiguous():
                raise ValueError('counts must be contiguous resident int64 with prepared row shape')
            kernel[(ranks,)](counts,self.index,self.quotas,ranks,triton.next_power_of_2(ranks),num_warps=4)
            return self.quotas

    return Dispatcher()
