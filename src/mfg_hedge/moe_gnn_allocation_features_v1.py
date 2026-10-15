"""Live fixed-R2 feature refresh for allocation inference; no solver state."""


def build_allocation_feature_refresh(cache):
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    g=cache.graph;r=cache.runtime
    if g.device.type!='cuda' or any(b-a!=2 for a,b in zip(g.problem.row_offsets,g.problem.row_offsets[1:])):
        raise ValueError('CUDA fixed R2 required')
    groups=[[] for _ in range(g.resource_features.shape[0])]
    for i,j in enumerate(g.incidence_resource.cpu().tolist()):groups[j].append(i)
    degree=triton.next_power_of_2(max(map(len,groups)))
    index=torch.tensor([v+[-1]*(degree-len(v)) for v in groups],device=g.device,dtype=torch.int64)
    mass=torch.zeros((g.variable_count,5),device=g.device,dtype=torch.float64)
    mass.view(-1).scatter_add_(0,cache.mass_slots,r.unit)
    rowmass=mass.sum(-1).reshape(-1,2).mean(-1).contiguous();varmass=mass.sum(-1).contiguous()
    total=torch.empty(g.resource_features.shape[0],device=g.device,dtype=torch.float64)
    stats=torch.empty(3,device=g.device,dtype=torch.float64)
    offsets=torch.tensor(g.block_resource_offsets,device=g.device,dtype=torch.int64)

    @triton.jit
    def totals(C,IX,IR,U,T,D:tl.constexpr):
        res=tl.program_id(0);j=tl.arange(0,D);inc=tl.load(IX+res*D+j)
        row=tl.load(IR+tl.maximum(inc,0),inc>=0,other=0)
        n=tl.load(C+row).to(tl.float64);unit=tl.load(U+tl.maximum(inc,0),inc>=0,other=0)
        tl.store(T+res,tl.sum(n*unit,0))

    @triton.jit
    def global_stats(C,T,O,S,F,NORM,N:tl.constexpr,CN:tl.constexpr,BN:tl.constexpr,START:tl.constexpr):
        i=tl.arange(0,CN);n=tl.load(C+i,i<N,other=0).to(tl.float64)
        normal=tl.full((),START,tl.float64)
        j=tl.arange(0,BN)
        for k in tl.static_range(5):
            a=tl.load(O+k);b=tl.load(O+k+1);v=tl.load(T+a+j,a+j<b,other=0)
            normal=normal+tl.max(v,0)*.5
        normal=tl.where(normal>0,normal,1.)
        tl.store(S,normal);tl.store(S+1,tl.maximum(libdevice.log1p(tl.max(n,0)),1.));tl.store(S+2,tl.maximum(tl.sum(n,0)/N,1.))
        tl.store(F,normal);tl.store(NORM,normal)

    @triton.jit
    def refresh(C,RC,S,T,M,RM,VM,U,IR,IV,ROWS,RF,EF,RES,CO,RW,VW,NR:tl.constexpr,NV:tl.constexpr,NS:tl.constexpr,NI:tl.constexpr):
        i=tl.program_id(0)*128+tl.arange(0,128);norm=tl.load(S)
        n=tl.load(C+i,i<NR,other=1);nf=n.to(tl.float64)
        tl.store(RC+i,n,i<NR)
        tl.store(RF+i*7,libdevice.log1p(nf)/tl.load(S+1),i<NR)
        tl.store(RF+i*7+1,nf/tl.load(S+2),i<NR)
        rm=tl.load(RM+i,i<NR,other=0);tl.store(RF+i*7+6,rm*nf/norm,i<NR)
        en=tl.load(C+i//2,i<NV,other=1).to(tl.float64)
        for k in tl.static_range(5):
            unitmass=tl.load(M+i*5+k,i<NV,other=0);tl.store(EF+i*12+7+k,unitmass*en/norm,i<NV)
        t=tl.load(T+i,i<NS,other=0);tl.store(RES+i*7+6,t/norm,i<NS)
        row=tl.load(ROWS+i,i<NI,other=0);count=tl.load(C+row).to(tl.float64)
        unit=tl.load(U+i,i<NI,other=0);co=unit*count
        res=tl.load(IR+i,i<NI,other=0);v=tl.load(IV+i,i<NI,other=0)
        rt=tl.load(T+res);vt=tl.load(VM+v)*count
        tl.store(CO+i,co,i<NI);tl.store(RW+i,tl.where(rt>0,co/tl.maximum(rt,1e-300),0.),i<NI)
        tl.store(VW+i,tl.where(vt>0,co/tl.maximum(vt,1e-300),0.),i<NI)

    def update(counts):
        if counts.device!=g.device or counts.dtype!=torch.int64 or counts.shape!=(g.row_count,) or not counts.is_contiguous():
            raise ValueError('matching contiguous resident int64 positive counts required')
        totals[(total.numel(),)](counts,index,r.incidence_rows,r.unit,total,degree,enable_fp_fusion=False)
        global_stats[(1,)](counts,total,offsets,stats,g.f_reference,r.normalization,g.row_count,triton.next_power_of_2(g.row_count),triton.next_power_of_2(max(g.block_widths)),r.problem.fixed_startup_ms,enable_fp_fusion=False)
        refresh[(triton.cdiv(max(g.row_count,g.variable_count,total.numel(),r.unit.numel()),128),)](counts,r.counts,stats,total,mass,rowmass,varmass,r.unit,g.incidence_resource,g.incidence_variable,r.incidence_rows,g.row_features,g.edge_features,g.resource_features,g.incidence_coefficients,g.resource_to_variable_weight,g.variable_to_resource_weight,g.row_count,g.variable_count,total.numel(),r.unit.numel(),enable_fp_fusion=False)
        return g
    return update
