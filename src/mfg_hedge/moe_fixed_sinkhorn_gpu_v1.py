"""Opt-in fixed-R2 adaptation of TAOT's balanced Sinkhorn recurrence.

Not official guest-placement TAOT. Current resident int64 counts must be
nonnegative and per-Expert totals <=2**24. Returns borrowed integer quotas;
continuous slot0 shares are exposed for audits. See ADR0207 for objectives.
"""


def build_dispatcher(row_sources,row_experts,*,ranks=64,ranks_per_node=8,device='cuda:0'):
    if type(ranks) is not int or not 2<=ranks<=64:
        raise ValueError('ranks must be in [2,64]')
    if type(ranks_per_node) is not int or ranks_per_node<=0:
        raise ValueError('positive ranks_per_node required')
    sources,experts=tuple(row_sources),tuple(row_experts)
    if not sources or len(sources)!=len(experts) or len(set(zip(sources,experts)))!=len(sources):
        raise ValueError('unique nonempty source/Expert rows required')
    if any(type(v) is not int or not 0<=v<ranks for v in sources+experts):
        raise ValueError('metadata outside rank range')
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def solve(counts,index,quotas,shares,G:tl.constexpr,NODE:tl.constexpr,B:tl.constexpr):
        e=tl.program_id(0);s=tl.arange(0,B);other=(e+1)%G
        idx=tl.load(index+e*G+s,s<G,other=-1);valid=(s<G)&(idx>=0)
        n=tl.load(counts+tl.maximum(idx,0),valid,other=0);a=n.to(tl.float32)
        m0=tl.exp(tl.where(s//NODE==e//NODE,-1./3.,-1.))
        m1=tl.exp(tl.where(s//NODE==other//NODE,-1./3.,-1.))
        target=tl.sum(a,0)*.5;v0=1.;v1=1.
        for _ in tl.static_range(50):
            u=a/tl.maximum(m0*v0+m1*v1,1e-8)
            v0=target/tl.maximum(tl.sum(m0*u,0),1e-8)
            v1=target/tl.maximum(tl.sum(m1*u,0),1e-8)
        p=tl.where(target>0,m0*v0/tl.maximum(m0*v0+m1*v1,1e-8),.5)
        x=a*p;floor=tl.floor(x);frac=x-floor
        q0=floor.to(tl.int64)+((frac>.5)|((frac==.5)&(e<other))).to(tl.int64)
        q0=tl.minimum(n,tl.maximum(q0,0))
        tl.store(quotas+2*idx,q0,valid);tl.store(quotas+2*idx+1,n-q0,valid);tl.store(shares+idx,p,valid)

    class Dispatcher:
        def __init__(self):
            self.device=torch.device(device)
            if self.device.type!='cuda':raise ValueError('CUDA required')
            index=[-1]*(ranks*ranks)
            for i,(s,e) in enumerate(zip(sources,experts)):index[e*ranks+s]=i
            self.index=torch.tensor(index,device=self.device,dtype=torch.int64)
            self.quotas=torch.empty((len(sources),2),device=self.device,dtype=torch.int64)
            self.shares=torch.empty(len(sources),device=self.device,dtype=torch.float32)
        def __call__(self,counts):
            if counts.device!=self.quotas.device or counts.dtype!=torch.int64 or counts.shape!=(len(sources),) or not counts.is_contiguous():
                raise ValueError('contiguous resident int64 counts must match prepared rows')
            solve[(ranks,)](counts,self.index,self.quotas,self.shares,ranks,ranks_per_node,triton.next_power_of_2(ranks),num_warps=4)
            return self.quotas
    return Dispatcher()
