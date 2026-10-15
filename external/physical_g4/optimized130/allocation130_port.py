"""Optional FP64 width16 GNN allocation inference; borrowed output buffers."""


def allocation_only(model,g):
    import torch
    edge=model.allocation['route'](torch.cat((g.row_features[g.row_index],g.edge_features),-1))
    relative=model.relative_load(g.resource_features[:,6])
    messages=torch.zeros((g.resource_features.shape[0],16),device=g.device,dtype=torch.float64).index_add(0,g.incidence_resource,edge[g.incidence_variable]*g.resource_to_variable_weight[:,None])
    resource=model.allocation['resource'](torch.cat((g.resource_features,relative[:,None],messages),-1))
    returned=torch.zeros_like(edge).index_add(0,g.incidence_variable,resource[g.incidence_resource]*g.variable_to_resource_weight[:,None])
    sibling=returned.reshape(-1,2,16).mean(1)[g.row_index]
    return model.allocation['readout'](torch.cat((edge,returned,sibling),-1)).reshape(-1,2).softmax(-1).reshape(-1)


def build_fused_allocation(model,g,*,feedback_rounds=0):
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    if type(feedback_rounds) is not int or feedback_rounds<0:
        raise ValueError('feedback_rounds must be a nonnegative integer')
    if g.device.type!='cuda' or model.width!=16 or next(model.parameters()).dtype!=torch.float64:
        raise ValueError('CUDA FP64 width16 required')
    if any(b-a!=2 for a,b in zip(g.problem.row_offsets,g.problem.row_offsets[1:])):
        raise ValueError('fixed R2 required')
    if any(v.device!=g.device or v.dtype!=torch.float64 or not v.is_contiguous() for v in model.allocation.parameters()):
        raise ValueError('contiguous FP64 allocation weights must match graph device')
    if [tuple(model.allocation[k][0].weight.shape) for k in ('route','resource','readout')]!=[(16,19),(16,24),(16,48)] or tuple(model.allocation['readout'][2].weight.shape)!=(1,16):
        raise ValueError('unexpected allocation architecture')
    if feedback_rounds and (not all(k in model.allocation for k in ('feedback_resource','feedback_readout')) or
            tuple(model.allocation['feedback_resource'][0].weight.shape)!=(16,19) or
            tuple(model.allocation['feedback_readout'][0].weight.shape)!=(16,49) or
            tuple(model.allocation['feedback_readout'][2].weight.shape)!=(1,16)):
        raise ValueError('width16 no-pressure feedback architecture required')
    def padded(ids,n):
        groups=[[] for _ in range(n)]
        for i,j in enumerate(ids.cpu().tolist()):groups[j].append(i)
        width=triton.next_power_of_2(max(1,max(map(len,groups))))
        return torch.tensor([v+[-1]*(width-len(v)) for v in groups],device=g.device,dtype=torch.int64),width
    ri,rd=padded(g.incidence_resource,g.resource_features.shape[0]);vi,vd=padded(g.incidence_variable,g.variable_count)
    bounds=[(a,b) for a,b in zip(g.block_resource_offsets,g.block_resource_offsets[1:]) for _ in range(a,b)]
    bounds=torch.tensor(bounds,device=g.device,dtype=torch.int64)
    edge=torch.empty((g.variable_count,16),device=g.device,dtype=torch.float64)
    res=torch.empty((g.resource_features.shape[0],16),device=g.device,dtype=torch.float64);returned=torch.empty_like(edge)
    p=torch.empty(g.variable_count,device=g.device,dtype=torch.float64)
    logits=torch.empty_like(p)
    loads=torch.empty(res.shape[0],device=g.device,dtype=torch.float64)

    @triton.jit
    def route(RF,EF,W,B,Y,N:tl.constexpr):
        e=tl.program_id(0)*8+tl.arange(0,8);h=tl.arange(0,16)
        value=tl.full((8,16),0,tl.float64)+tl.load(B+h)[None,:]
        for k in tl.static_range(19):
            if k<7:x=tl.load(RF+(e//2)*7+k,e<N,other=0)
            else:x=tl.load(EF+e*12+k-7,e<N,other=0)
            value=value+x[:,None]*tl.load(W+h*19+k)[None,:]
        value=value/(1+libdevice.exp(-value));tl.store(Y+e[:,None]*16+h[None,:],value,e[:,None]<N)

    @triton.jit
    def resource(RI,IV,MW,E,RF,BO,W,B,Y,D:tl.constexpr,BS:tl.constexpr):
        r=tl.program_id(0);j=tl.arange(0,D);h=tl.arange(0,32)
        ix=tl.load(RI+r*D+j);v=tl.load(IV+tl.maximum(ix,0),ix>=0,other=0)
        w=tl.load(MW+tl.maximum(ix,0),ix>=0,other=0)
        ev=tl.load(E+v[:,None]*16+h[None,:],(ix[:,None]>=0)&(h[None,:]<16),other=0)
        msg=tl.sum(ev*w[:,None],0)
        a=tl.load(BO+r*2);b=tl.load(BO+r*2+1);z=a+tl.arange(0,BS)
        maximum=tl.maximum(tl.max(tl.load(RF+z*7+6,z<b,other=0),0),1e-12)
        gather_index=tl.arange(0,32)
        shifted=tl.sum(tl.where(gather_index[None,:]==tl.maximum(h-8,0)[:,None],msg[None,:],0.),1)
        x=tl.where(h<7,tl.load(RF+r*7+h,h<7,other=0),tl.where(h==7,tl.load(RF+r*7+6)/maximum,shifted))
        o=tl.arange(0,16);weight=tl.load(W+o[:,None]*24+h[None,:],h[None,:]<24,other=0)
        value=tl.sum(weight*x[None,:],1)+tl.load(B+o)
        value=value/(1+libdevice.exp(-value));tl.store(Y+r*16+o,value)

    @triton.jit
    def back(VI,IR,MW,R,Y,N:tl.constexpr,D:tl.constexpr):
        e=tl.program_id(0)*4+tl.arange(0,4);j=tl.arange(0,D);h=tl.arange(0,16)
        ix=tl.load(VI+e[:,None]*D+j[None,:],e[:,None]<N,other=-1)
        r=tl.load(IR+tl.maximum(ix,0),ix>=0,other=0);w=tl.load(MW+tl.maximum(ix,0),ix>=0,other=0)
        rv=tl.load(R+r[:,:,None]*16+h[None,None,:],ix[:,:,None]>=0,other=0)
        value=tl.sum(rv*w[:,:,None],1);tl.store(Y+e[:,None]*16+h[None,:],value,e[:,None]<N)

    @triton.jit
    def readout(E,R,W,B,V,C,P,L,N:tl.constexpr,FEEDBACK:tl.constexpr):
        row=tl.program_id(0)*4+tl.arange(0,4);slot=tl.arange(0,2);h=tl.arange(0,16);e=row[:,None]*2+slot[None,:]
        value=tl.full((4,2,16),0,tl.float64)+tl.load(B+h)[None,None,:]
        for k in tl.static_range(49 if FEEDBACK else 48):
            if k<16:x=tl.load(E+e*16+k,row[:,None]<N,other=0)
            elif k<32:x=tl.load(R+e*16+k-16,row[:,None]<N,other=0)
            elif k<48:
                x0=tl.load(R+row*32+k-32,row<N,other=0);x1=tl.load(R+row*32+16+k-32,row<N,other=0)
                x=(x0+x1)[:,None]*.5+tl.full((4,2),0,tl.float64)
            else:x=tl.load(P+e,row[:,None]<N,other=0)
            value=value+x[:,:,None]*tl.load(W+h*(49 if FEEDBACK else 48)+k)[None,None,:]
        value=value/(1+libdevice.exp(-value));logit=tl.sum(value*tl.load(V+h)[None,None,:],2)+tl.load(C)
        if FEEDBACK:logit=logit+tl.load(L+e,row[:,None]<N,other=0)
        tl.store(L+e,logit,row[:,None]<N)
        maximum=tl.max(logit,1);z=libdevice.exp(logit-maximum[:,None]);z=z/tl.sum(z,1)[:,None]
        tl.store(P+e,z,row[:,None]<N)

    @triton.jit
    def physical_load(RI,IV,CO,P,Y,D:tl.constexpr):
        r=tl.program_id(0);j=tl.arange(0,D)
        ix=tl.load(RI+r*D+j);valid=ix>=0
        v=tl.load(IV+tl.maximum(ix,0),valid,other=0)
        c=tl.load(CO+tl.maximum(ix,0),valid,other=0)
        shares=tl.load(P+v,valid,other=0)
        tl.store(Y+r,tl.sum(c*shares,0))

    @triton.jit
    def feedback_resource(R,LD,RF,FR,BO,W,B,BS:tl.constexpr):
        r=tl.program_id(0);h=tl.arange(0,32);o=tl.arange(0,16)
        a=tl.load(BO+r*2);b=tl.load(BO+r*2+1);z=a+tl.arange(0,BS)
        maximum=tl.maximum(tl.max(tl.load(LD+z,z<b,other=0),0),1e-12)
        load=tl.load(LD+r);normalized=load/tl.load(FR)
        prior=tl.load(R+r*16+h,h<16,other=0)
        x=tl.where(h<16,prior,tl.where(h==16,normalized,tl.where(h==17,load/maximum,normalized-tl.load(RF+r*7+6)*.5)))
        weight=tl.load(W+o[:,None]*19+h[None,:],h[None,:]<19,other=0)
        value=tl.sum(weight*x[None,:],1)+tl.load(B+o)
        value=value/(1+libdevice.exp(-value));tl.store(R+r*16+o,value)

    def forward():
        a=model.allocation
        route[(triton.cdiv(g.variable_count,8),)](g.row_features,g.edge_features,a['route'][0].weight,a['route'][0].bias,edge,g.variable_count,enable_fp_fusion=False)
        resource[(res.shape[0],)](ri,g.incidence_variable,g.resource_to_variable_weight,edge,g.resource_features,bounds,a['resource'][0].weight,a['resource'][0].bias,res,rd,triton.next_power_of_2(max(g.block_widths)),num_warps=8,enable_fp_fusion=False)
        back[(triton.cdiv(g.variable_count,4),)](vi,g.incidence_resource,g.variable_to_resource_weight,res,returned,g.variable_count,vd,enable_fp_fusion=False)
        readout[(triton.cdiv(g.row_count,4),)](edge,returned,a['readout'][0].weight,a['readout'][0].bias,a['readout'][2].weight,a['readout'][2].bias,p,logits,g.row_count,False,enable_fp_fusion=False)
        for _ in range(feedback_rounds):
            physical_load[(res.shape[0],)](ri,g.incidence_variable,g.incidence_coefficients,p,loads,rd,enable_fp_fusion=False)
            feedback_resource[(res.shape[0],)](res,loads,g.resource_features,g.f_reference,bounds,a['feedback_resource'][0].weight,a['feedback_resource'][0].bias,triton.next_power_of_2(max(g.block_widths)),enable_fp_fusion=False)
            back[(triton.cdiv(g.variable_count,4),)](vi,g.incidence_resource,g.variable_to_resource_weight,res,returned,g.variable_count,vd,enable_fp_fusion=False)
            readout[(triton.cdiv(g.row_count,4),)](edge,returned,a['feedback_readout'][0].weight,a['feedback_readout'][0].bias,a['feedback_readout'][2].weight,a['feedback_readout'][2].bias,p,logits,g.row_count,True,enable_fp_fusion=False)
        return p
    forward.resource_context=res
    return forward
