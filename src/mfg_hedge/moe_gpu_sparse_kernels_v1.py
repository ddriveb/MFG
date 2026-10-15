"""Installed Triton kernels, isolated from ordinary package imports."""
import torch, triton
import triton.language as tl

@triton.jit
def _spmv(Offsets,Indices,Coefficients,X,Y,B:tl.constexpr):
    row=tl.program_id(0);start=tl.load(Offsets+row);end=tl.load(Offsets+row+1)
    ix=start+tl.arange(0,B);valid=ix<end
    col=tl.load(Indices+ix,valid,0);c=tl.load(Coefficients+ix,valid,0.)
    v=tl.load(X+col,valid,0.)
    tl.store(Y+row,tl.sum(c*v,0))

@triton.jit
def _project(V,O,Y,B:tl.constexpr):
    row=tl.program_id(0);start=tl.load(O+row);end=tl.load(O+row+1)
    i=tl.arange(0,B);valid=i<end-start
    x=tl.load(V+start+i,valid,float('-inf'))
    ordered=tl.sort(x,descending=True);cum=tl.cumsum(tl.where(i<end-start,ordered,0.))
    candidate=(cum-1.)/(i+1)
    threshold=tl.max(tl.where((i<end-start)&(ordered>candidate),candidate,float('-inf')),0)
    projected=tl.where(valid,tl.maximum(x-threshold,0.),0.)
    total=tl.sum(projected,0);projected=projected/tl.maximum(total,1e-300)
    tl.store(Y+start+i,projected,valid)

def spmv(offsets,indices,coefficients,x,rows,width):
    y=torch.empty(rows,device=x.device,dtype=torch.float64)
    _spmv[(rows,)](offsets,indices,coefficients,x,y,triton.next_power_of_2(max(1,width)),num_warps=4,enable_fp_fusion=False)
    return y

def project_rows(values,offsets,rows,width):
    out=torch.empty_like(values)
    _project[(rows,)](values,offsets,out,triton.next_power_of_2(width),num_warps=4,enable_fp_fusion=False)
    return out

def project_blocks(values,offsets,device_offsets):
    out=torch.empty_like(values)
    _project[(len(offsets)-1,)](values,device_offsets,out,triton.next_power_of_2(max(b-a for a,b in zip(offsets,offsets[1:]))),num_warps=4,enable_fp_fusion=False)
    return out

@triton.jit
def _choose(Order,Valid,Feasible,Total,Secondary,Current,CurrentSecondary,Running,Choice,Checked,N:tl.constexpr,Budget:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);inside=i<N;index=tl.load(Order+i,inside,0)
    valid=tl.load(Valid+index,inside,0)&inside
    running=tl.load(Running)
    checked=tl.minimum(tl.sum(valid.to(tl.int32),0),Budget)
    tl.store(Checked,tl.where(running,checked,0))
    eligible=valid&(i<Budget)&tl.load(Feasible+index,inside,0)&running
    value=tl.load(Total+index,inside,float('inf'));sec=tl.load(Secondary+index,inside,float('inf'))
    # Round-to-nearest-even via nearbyint, not a precision-changing tolerance.
    rounded=tl.extra.cuda.libdevice.nearbyint(value*1e10)/1e10
    current=tl.extra.cuda.libdevice.nearbyint(tl.load(Current)*1e10)/1e10
    currentsec=tl.load(CurrentSecondary)
    eligible=eligible&((rounded<current)|((rounded==current)&(sec<currentsec-1e-10)))
    best=tl.min(tl.where(eligible,rounded,float('inf')),0)
    tied=eligible&(rounded==best)
    position=tl.min(tl.where(tied,i,2147483647),0)
    bestsec=tl.sum(tl.where(i==position,sec,0.),0)
    nextpos=tl.min(tl.where(tied&(i>position)&(sec<bestsec-1e-10),i,2147483647),0)
    while nextpos<2147483647:
        position=nextpos
        bestsec=tl.sum(tl.where(i==position,sec,0.),0)
        nextpos=tl.min(tl.where(tied&(i>position)&(sec<bestsec-1e-10),i,2147483647),0)
    selected=tl.sum(tl.where(i==position,index,0),0)
    tl.store(Choice,tl.where(position<2147483647,selected,-1))

def choose_candidate(order,valid,feasible,total,secondary,current,current_secondary,running,budget):
    choice=torch.empty((),device=total.device,dtype=torch.int64);checked=torch.empty_like(choice)
    _choose[(1,)](order,valid,feasible,total,secondary,current,current_secondary,running,choice,checked,order.numel(),budget,triton.next_power_of_2(order.numel()),num_warps=8,enable_fp_fusion=False)
    return choice,checked

@triton.jit
def _candidate_sums(F,B,U,V,FU,FV,BU,BV,SF,SB,N:tl.constexpr,G:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);inside=i<N
    u=tl.load(U+i,inside,0);v=tl.load(V+i,inside,0)
    fu=tl.load(FU+i,inside,0.);fv=tl.load(FV+i,inside,0.)
    bu=tl.load(BU+i,inside,0.);bv=tl.load(BV+i,inside,0.)
    fs=tl.full((BLOCK,),0.,tl.float64);fc=fs;bs=fs;bc=fs
    for j in range(G):
        xf=tl.where(u==j,fu,tl.where(v==j,fv,tl.load(F+j)))
        xb=tl.where(u==j,bu,tl.where(v==j,bv,tl.load(B+j)))
        nf=fs+xf;nb=bs+xb
        fc=fc+tl.where(tl.abs(fs)>=tl.abs(xf),(fs-nf)+xf,(xf-nf)+fs)
        bc=bc+tl.where(tl.abs(bs)>=tl.abs(xb),(bs-nb)+xb,(xb-nb)+bs)
        fs=nf;bs=nb
    tl.store(SF+i,fs+fc,inside);tl.store(SB+i,bs+bc,inside)

@triton.jit
def _current_secondary(F,B,Y,G:tl.constexpr):
    fs=tl.full((),0.,tl.float64);fc=fs;bs=fs;bc=fs
    for j in range(G):
        xf=tl.load(F+j);xb=tl.load(B+j);nf=fs+xf;nb=bs+xb
        fc=fc+tl.where(tl.abs(fs)>=tl.abs(xf),(fs-nf)+xf,(xf-nf)+fs)
        bc=bc+tl.where(tl.abs(bs)>=tl.abs(xb),(bs-nb)+xb,(xb-nb)+bs)
        fs=nf;bs=nb
    tl.store(Y,(fs+fc)+(bs+bc))

def candidate_sums(f,b,u,v,fu,fv,bu,bv):
    sf=torch.empty_like(fu);sb=torch.empty_like(bu)
    _candidate_sums[(triton.cdiv(u.numel(),128),)](f,b,u,v,fu,fv,bu,bv,sf,sb,u.numel(),f.numel(),128,num_warps=4,enable_fp_fusion=False)
    return sf,sb

def current_secondary(f,b):
    y=torch.empty((),device=f.device,dtype=torch.float64)
    _current_secondary[(1,)](f,b,y,f.numel(),num_warps=4,enable_fp_fusion=False)
    return y

@triton.jit
def _pair_bounds(Pairs,F,B,Comm,Untouched,Offsets,Bounds,G:tl.constexpr,N:tl.constexpr,BG:tl.constexpr,BR:tl.constexpr):
    pair=tl.program_id(0);u=tl.load(Pairs+pair*3+1);v=tl.load(Pairs+pair*3+2)
    gi=tl.arange(0,BG);ok=(gi<G)&(gi!=u)&(gi!=v)
    f=tl.load(F+gi,gi<G,float('-inf'));b=tl.load(B+gi,gi<G,float('-inf'))
    tl.store(Bounds+pair*6,tl.max(tl.where(ok,f,float('-inf')),0))
    tl.store(Bounds+pair*6+1,tl.max(tl.where(ok,b,float('-inf')),0))
    ri=tl.arange(0,BR);use=tl.load(Untouched+pair*N+ri,ri<N,0)&(ri<N)
    load=tl.load(Comm+ri,ri<N,float('-inf'))
    for block in tl.static_range(4):
        start=tl.load(Offsets+block);end=tl.load(Offsets+block+1)
        maximum=tl.max(tl.where(use&(ri>=start)&(ri<end),load,float('-inf')),0)
        tl.store(Bounds+pair*6+2+block,maximum)

@triton.jit
def _amounts(Pairs,Counts,Ranks,CurveMap,Lower,Upper,Amounts,Buckets,Valid,G:tl.constexpr,L:tl.constexpr,Shared:tl.constexpr,HasShared:tl.constexpr):
    pair=tl.program_id(0);e=tl.load(Pairs+pair*3);u=tl.load(Pairs+pair*3+1);v=tl.load(Pairs+pair*3+2)
    available=tl.load(Counts+e*G+u);to=tl.load(Counts+e*G+v);ru=tl.load(Ranks+u);rv=tl.load(Ranks+v);cm=tl.load(CurveMap+e)
    lower=tl.load(Lower+cm*L+tl.minimum(available,L-1));upper=tl.load(Upper+cm*L+tl.minimum(to,L-1))
    sl=tl.load(Lower+Shared*L+tl.minimum(ru,L-1));su=tl.load(Upper+Shared*L+tl.minimum(rv,L-1))
    i=tl.arange(0,16)
    amount=tl.where(i==0,available,tl.where(i==1,1,tl.where(i==2,(available+1)//4,tl.where(i==3,available//2,tl.where(i==4,(3*available+3)//4,tl.where(i==5,lower,tl.where(i==6,upper,tl.where(i==7,sl,su))))))))
    valid=(i<9)&(amount>=1)&(amount<=available)&((i<7)|HasShared)
    # Deduplicate against prior valid original choices, then compact bucket IDs.
    duplicate=tl.sum(((amount[:,None]==amount[None,:])&(i[None,:]<i[:,None])&valid[None,:]).to(tl.int32),1)>0
    valid=valid&~duplicate
    bucket=tl.cumsum(valid.to(tl.int32))-1
    tl.store(Amounts+pair*9+i,amount,i<9);tl.store(Buckets+pair*9+i,bucket,i<9);tl.store(Valid+pair*9+i,valid,i<9)

@triton.jit
def _table_cost(F,B,curve,count,L:tl.constexpr):
    inside=(count>=0)&(count<L)
    f=tl.load(F+curve*L+count,inside,float('nan'));b=tl.load(B+curve*L+count,inside,float('nan'))
    return f,b,inside&(f==f)&(b==b)

@triton.jit
def _preview_candidates(Pairs,Amounts,InputValid,Q,Counts,Ranks,F,B,Comm,CurveMap,FTable,BTable,SV,RowValid,RI,RV,RBlock,Diff,Bounds,Total,Valid,Feasible,Moved,Delta,FU,FV,BU,BV,ComputeOnly,G:tl.constexpr,L:tl.constexpr,Shared:tl.constexpr,H:tl.constexpr,U:tl.constexpr,BH:tl.constexpr,BUC:tl.constexpr,AlphaD:tl.constexpr,AlphaR:tl.constexpr):
    candidate=tl.program_id(0);pair=candidate//9;amount=tl.load(Amounts+candidate)
    e=tl.load(Pairs+pair*3);u=tl.load(Pairs+pair*3+1);v=tl.load(Pairs+pair*3+2);cm=tl.load(CurveMap+e)
    eu=tl.load(Counts+e*G+u);ev=tl.load(Counts+e*G+v);ru=tl.load(Ranks+u);rv=tl.load(Ranks+v)
    ouf,oub,_=_table_cost(FTable,BTable,cm,eu,L);ovf,ovb,_=_table_cost(FTable,BTable,cm,ev,L)
    suf,sub,_=_table_cost(FTable,BTable,Shared,ru,L);svf,svb,_=_table_cost(FTable,BTable,Shared,rv,L)
    nuf,nub,ok1=_table_cost(FTable,BTable,cm,eu-amount,L);nvf,nvb,ok2=_table_cost(FTable,BTable,cm,ev+amount,L)
    nsuf,nsub,ok3=_table_cost(FTable,BTable,Shared,ru-amount,L);nsvf,nsvb,ok4=_table_cost(FTable,BTable,Shared,rv+amount,L)
    fu=tl.load(F+u)+((nuf+nsuf)-(ouf+suf));fv=tl.load(F+v)+((nvf+nsvf)-(ovf+svf))
    bu=tl.load(B+u)+((nub+nsub)-(oub+sub));bv=tl.load(B+v)+((nvb+nsvb)-(ovb+svb))
    maxf=tl.maximum(tl.maximum(fu,fv),tl.load(Bounds+pair*6));maxb=tl.maximum(tl.maximum(bu,bv),tl.load(Bounds+pair*6+1))
    h=tl.arange(0,BH);rowok=tl.load(RowValid+pair*H+h,h<H,0)&(h<H)
    source=tl.load(SV+pair*H+h,h<H,0);donor=tl.load(Q+source,rowok,0)
    prefix=tl.cumsum(donor)-donor;move=tl.minimum(donor,tl.maximum(amount-prefix,0))
    feasible=tl.sum(move,0)==amount
    ix=tl.arange(0,BUC);resource=tl.load(RI+pair*U+ix,ix<U,0);resourceok=tl.load(RV+pair*U+ix,ix<U,0)&(ix<U)
    d=tl.load(Diff+pair*H*U+h[:,None]*U+ix[None,:],(h[:,None]<H)&(ix[None,:]<U),0.)
    delta=tl.sum(move[:,None]*d,0)
    patched=tl.load(Comm+resource,resourceok,0.)+delta
    blockid=tl.load(RBlock+pair*U+ix,ix<U,0)
    m0=tl.maximum(tl.load(Bounds+pair*6+2),tl.max(tl.where(resourceok&(blockid==0),patched,float('-inf')),0))
    m1=tl.maximum(tl.load(Bounds+pair*6+3),tl.max(tl.where(resourceok&(blockid==1),patched,float('-inf')),0))
    m2=tl.maximum(tl.load(Bounds+pair*6+4),tl.max(tl.where(resourceok&(blockid==2),patched,float('-inf')),0))
    m3=tl.maximum(tl.load(Bounds+pair*6+5),tl.max(tl.where(resourceok&(blockid==3),patched,float('-inf')),0))
    total=maxf+maxb+2*((AlphaD+m0+m1)+(AlphaR+m2+m3))
    valid=tl.load(InputValid+candidate)&ok1&ok2&ok3&ok4
    tl.store(Total+candidate,total);tl.store(ComputeOnly+candidate,maxf+maxb);tl.store(Valid+candidate,valid);tl.store(Feasible+candidate,feasible)
    tl.store(FU+candidate,fu);tl.store(FV+candidate,fv);tl.store(BU+candidate,bu);tl.store(BV+candidate,bv)
    tl.store(Moved+candidate*H+h,move,h<H);tl.store(Delta+candidate*U+ix,delta,ix<U)

@triton.jit
def _apply_candidate(Choice,Checked,Amounts,Pairs,SV,DV,RowValid,RI,RV,Moved,Delta,FU,FV,BU,BV,Q,Counts,Ranks,F,B,Comm,Running,HistChoice,HistChecked,HistAmount,STEP:tl.constexpr,G:tl.constexpr,H:tl.constexpr,U:tl.constexpr,BLOCK:tl.constexpr):
    chosen=tl.load(Choice);active=tl.load(Running)&(chosen>=0);safe=tl.maximum(chosen,0);pair=safe//9
    ix=tl.arange(0,BLOCK);rowok=tl.load(RowValid+pair*H+ix,ix<H,0)&(ix<H)&active
    src=tl.load(SV+pair*H+ix,ix<H,0);dst=tl.load(DV+pair*H+ix,ix<H,0);move=tl.load(Moved+safe*H+ix,ix<H,0)
    oldsrc=tl.load(Q+src,rowok,0);olddst=tl.load(Q+dst,rowok,0)
    tl.store(Q+src,oldsrc-move,rowok);tl.store(Q+dst,olddst+move,rowok)
    resok=tl.load(RV+pair*U+ix,ix<U,0)&(ix<U)&active
    res=tl.load(RI+pair*U+ix,ix<U,0);delta=tl.load(Delta+safe*U+ix,ix<U,0.)
    load=tl.load(Comm+res,resok,0.);tl.store(Comm+res,load+delta,resok)
    e=tl.load(Pairs+pair*3);u=tl.load(Pairs+pair*3+1);v=tl.load(Pairs+pair*3+2);n=tl.load(Amounts+safe)
    eu=tl.load(Counts+e*G+u);ev=tl.load(Counts+e*G+v);ru=tl.load(Ranks+u);rv=tl.load(Ranks+v)
    tl.store(Counts+e*G+u,eu-n,active);tl.store(Counts+e*G+v,ev+n,active);tl.store(Ranks+u,ru-n,active);tl.store(Ranks+v,rv+n,active)
    tl.store(F+u,tl.load(FU+safe),active);tl.store(F+v,tl.load(FV+safe),active);tl.store(B+u,tl.load(BU+safe),active);tl.store(B+v,tl.load(BV+safe),active)
    tl.store(HistChoice+STEP,tl.where(active,chosen,-1));tl.store(HistChecked+STEP,tl.load(Checked));tl.store(HistAmount+STEP,tl.where(active,n,0));tl.store(Running,active)

def parallel_preview(runtime):
    r=runtime;device=r.device;C=r.candidate_count
    bounds=torch.empty((r.P,6),device=device,dtype=torch.float64)
    amounts=torch.empty(C,device=device,dtype=torch.int64);buckets=torch.empty_like(amounts);inputvalid=torch.empty(C,device=device,dtype=torch.bool)
    _pair_bounds[(r.P,)](r.pairs,r.fw,r.bw,r.comm,r.untouched,r.device_comm_offsets,bounds,r.G,r.N,triton.next_power_of_2(r.G),triton.next_power_of_2(r.N),num_warps=4,enable_fp_fusion=False)
    _amounts[(r.P,)](r.pairs,r.ecn,r.rcn,r.curve_map,r.lower,r.upper,amounts,buckets,inputvalid,r.G,r.L,r.shared_map,r.shared_curve is not None,num_warps=4,enable_fp_fusion=False)
    total=torch.empty(C,device=device,dtype=torch.float64);valid=torch.empty_like(inputvalid);feasible=torch.empty_like(inputvalid)
    moved=torch.empty((C,r.H),device=device,dtype=torch.int64);delta=torch.empty((C,r.U),device=device,dtype=torch.float64)
    fu=torch.empty_like(total);fv=torch.empty_like(total);bu=torch.empty_like(total);bv=torch.empty_like(total);compute=torch.empty_like(total)
    _preview_candidates[(C,)](r.pairs,amounts,inputvalid,r.q,r.ecn,r.rcn,r.fw,r.bw,r.comm,r.curve_map,r.Ftable,r.Btable,r.sv,r.row_valid,r.ri,r.rv,r.rblock,r.diff,bounds,total,valid,feasible,moved,delta,fu,fv,bu,bv,compute,r.G,r.L,r.shared_map,r.H,r.U,triton.next_power_of_2(r.H),triton.next_power_of_2(r.U),r.problem.config.alpha_dispatch_ms,r.problem.config.alpha_return_ms,num_warps=4,enable_fp_fusion=False)
    return amounts,buckets,valid,feasible,total,moved,delta,(fu,fv,bu,bv),compute

def apply_candidate(r,choice,checked,n,moved,delta,parts,step):
    _apply_candidate[(1,)](choice,checked,n,r.pairs,r.sv,r.dv,r.row_valid,r.ri,r.rv,moved,delta,*parts,r.q,r.ecn,r.rcn,r.fw,r.bw,r.comm,r.running,r.hist_choice,r.hist_checked,r.hist_amount,step,r.G,r.H,r.U,triton.next_power_of_2(max(r.H,r.U)),num_warps=4,enable_fp_fusion=False)

@triton.jit
def _primal_step(P,D,RO,AO,AI,AC,Gamma,Normalization,Out,Gradient,W:tl.constexpr,T:tl.constexpr):
    row=tl.program_id(0);start=tl.load(RO+row);end=tl.load(RO+row+1)
    slots=tl.arange(0,W);v=start+slots;valid=v<end
    a=tl.load(AO+v,valid,0);b=tl.load(AO+v+1,valid,0)
    j=tl.arange(0,T);inc=a[:,None]+j[None,:];use=valid[:,None]&(inc<b[:,None])
    resource=tl.load(AI+inc,use,0);coeff=tl.load(AC+inc,use,0.);dual=tl.load(D+resource,use,0.)
    grad=tl.sum(coeff*dual,1);tl.store(Gradient+v,grad,valid)
    old=tl.load(P+v,valid,0.);g=tl.load(Gamma)/tl.load(Normalization)
    x=tl.where(valid,old-g*grad,float('-inf'));ordered=tl.sort(x,descending=True)
    cumulative=tl.cumsum(tl.where(valid,ordered,0.));thresholds=(cumulative-1.)/(slots+1)
    threshold=tl.max(tl.where(valid&(ordered>thresholds),thresholds,float('-inf')),0)
    y=tl.where(valid,tl.maximum(x-threshold,0.),0.);y=y/tl.maximum(tl.sum(y,0),1e-300)
    tl.store(Out+v,y,valid)

@triton.jit
def _dual_step(D,Load,Offsets,Gamma,Normalization,Out,B:tl.constexpr):
    block=tl.program_id(0);start=tl.load(Offsets+block);end=tl.load(Offsets+block+1)
    i=tl.arange(0,B);valid=start+i<end;g=tl.load(Gamma)/tl.load(Normalization)
    old=tl.load(D+start+i,valid,0.);load=tl.load(Load+start+i,valid,0.)
    x=tl.where(valid,old+g*load,float('-inf'));ordered=tl.sort(x,descending=True)
    cumulative=tl.cumsum(tl.where(valid,ordered,0.));thresholds=(cumulative-1.)/(i+1)
    threshold=tl.max(tl.where(valid&(ordered>thresholds),thresholds,float('-inf')),0)
    y=tl.where(valid,tl.maximum(x-threshold,0.),0.);y=y/tl.maximum(tl.sum(y,0),1e-300)
    tl.store(Out+start+i,y,valid)

@triton.jit
def _condition(P,D,PT,DT,PN,DN,Grad,GT,Load,LT,Partial,M:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0);i=block*BLOCK+tl.arange(0,BLOCK);primal=i<M;dual=(i>=M)&(i<M+R);j=i-M
    p=tl.load(P+i,primal,0.);pt=tl.load(PT+i,primal,0.);pn=tl.load(PN+i,primal,0.)
    d=tl.load(D+j,dual,0.);dt=tl.load(DT+j,dual,0.);dn=tl.load(DN+j,dual,0.)
    g=tl.load(Grad+i,primal,0.);gt=tl.load(GT+i,primal,0.);l=tl.load(Load+j,dual,0.);lt=tl.load(LT+j,dual,0.)
    inner=(gt-g)*(pt-pn)-(lt-l)*(dt-dn)
    first=(pt-p)*(pt-p)+(dt-d)*(dt-d);second=(pt-pn)*(pt-pn)+(dt-dn)*(dt-dn)
    tl.store(Partial+block*3,tl.sum(inner,0));tl.store(Partial+block*3+1,tl.sum(first,0));tl.store(Partial+block*3+2,tl.sum(second,0))

@triton.jit
def _accept(Partial,Gamma,Normalization,Maximum,Weight,Accepted,Rejected,Limit,Control,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);use=i<N
    inner=tl.sum(tl.load(Partial+i*3,use,0.),0);first=tl.sum(tl.load(Partial+i*3+1,use,0.),0);second=tl.sum(tl.load(Partial+i*3+2,use,0.),0)
    gamma=tl.load(Gamma);delta=gamma/tl.load(Normalization)*inner-.5*(first+second)
    active=tl.load(Accepted)<tl.load(Limit);accept=(delta<=0)&active
    tl.store(Control,tl.where(accept,gamma,0.));tl.store(Control+1,accept.to(tl.float64));tl.store(Control+2,delta)
    tl.store(Weight,tl.load(Weight)+tl.where(accept,gamma,0.));tl.store(Accepted,tl.load(Accepted)+accept.to(tl.int64))
    tl.store(Rejected,tl.load(Rejected)+(active&~accept).to(tl.int64))
    guess=tl.where(accept,tl.minimum(gamma*1.2,tl.load(Maximum)),gamma*.8);tl.store(Gamma,tl.where(active,guess,gamma))

@triton.jit
def _commit(P,D,PT,DT,PN,DN,PS,DS,LastP,LastD,Control,M:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);primal=i<M;dual=(i>=M)&(i<M+R);j=i-M
    weight=tl.load(Control);accept=tl.load(Control+1)>0.
    pt=tl.load(PT+i,primal,0.);pn=tl.load(PN+i,primal,0.);dt=tl.load(DT+j,dual,0.);dn=tl.load(DN+j,dual,0.)
    tl.store(PS+i,tl.load(PS+i,primal,0.)+weight*pt,primal);tl.store(DS+j,tl.load(DS+j,dual,0.)+weight*dt,dual)
    tl.store(P+i,pn,primal&accept);tl.store(D+j,dn,dual&accept);tl.store(LastP+i,pt,primal&accept);tl.store(LastD+j,dt,dual&accept)

@triton.jit
def _simplex_active(x,valid):
    previous=tl.sum(valid.to(tl.int32),0)
    threshold=(tl.sum(tl.where(valid,x,0.),0)-1.)/previous
    active=valid&(x>threshold)
    count=tl.sum(active.to(tl.int32),0)
    while count<previous:
        threshold=(tl.sum(tl.where(active,x,0.),0)-1.)/tl.maximum(count,1)
        previous=count
        active=active&(x>threshold)
        count=tl.sum(active.to(tl.int32),0)
    y=tl.where(valid,tl.maximum(x-threshold,0.),0.)
    return y/tl.maximum(tl.sum(y,0),1e-300)

@triton.jit
def _project_active(V,O,Y,B:tl.constexpr):
    block=tl.program_id(0);start=tl.load(O+block);end=tl.load(O+block+1)
    i=tl.arange(0,B);valid=start+i<end
    x=tl.load(V+start+i,valid,0.)
    tl.store(Y+start+i,_simplex_active(x,valid),valid)

def project_blocks_active(values,offsets,device_offsets):
    out=torch.empty_like(values)
    _project_active[(len(offsets)-1,)](values,device_offsets,out,triton.next_power_of_2(max(b-a for a,b in zip(offsets,offsets[1:]))),num_warps=4,enable_fp_fusion=False)
    return out

@triton.jit
def _dual_step_active(D,Load,Offsets,Gamma,Normalization,Out,B:tl.constexpr):
    block=tl.program_id(0);start=tl.load(Offsets+block);end=tl.load(Offsets+block+1)
    i=tl.arange(0,B);valid=start+i<end;g=tl.load(Gamma)/tl.load(Normalization)
    old=tl.load(D+start+i,valid,0.);load=tl.load(Load+start+i,valid,0.)
    y=_simplex_active(old+g*load,valid)
    tl.store(Out+start+i,y,valid)

def adaptive_proposal(r):
    m=r.variable_count;n=r.resource_count;kwargs={'num_warps':4,'enable_fp_fusion':False}
    dual_kernel=_dual_step_active if r.projection=='active' else _dual_step
    def primal(dual,out,grad):
        _primal_step[(len(r.problem.rows),)](r.p,dual,r.ro,r.ar,r.ac,r.acoeff,r.gamma,r.normalization,out,grad,triton.next_power_of_2(r.width),triton.next_power_of_2(max(1,r.awidth)),**kwargs)
    def loads(p,out):
        _spmv[(n,)](r.lr,r.lc,r.lcoeff,p,out,triton.next_power_of_2(max(1,r.lwidth)),**kwargs)
    def dual(load,out):
        dual_kernel[(5,)](r.dual,load,r.dual_offsets,r.gamma,r.normalization,out,triton.next_power_of_2(max(b-a for a,b in zip(r.block_offsets,r.block_offsets[1:]))),**kwargs)
    primal(r.dual,r.pt,r.grad);loads(r.p,r.load);dual(r.load,r.dt)
    primal(r.dt,r.pn,r.gt);loads(r.pt,r.lt);dual(r.lt,r.dn)
    chunks=triton.cdiv(m+n,256)
    _condition[(chunks,)](r.p,r.dual,r.pt,r.dt,r.pn,r.dn,r.grad,r.gt,r.load,r.lt,r.partial,m,n,256,**kwargs)
    _accept[(1,)](r.partial,r.gamma,r.normalization,r.max_gamma,r.weight,r.accepted,r.rejected,r.limit,r.control,chunks,triton.next_power_of_2(chunks),**kwargs)
    _commit[(chunks,)](r.p,r.dual,r.pt,r.dt,r.pn,r.dn,r.p_sum,r.d_sum,r.trial_p,r.trial_d,r.control,m,n,256,**kwargs)

@triton.jit
def _initial_rank_costs(Counts,Ranks,Resident,ResidentValid,CurveMap,FTable,BTable,F,B,G:tl.constexpr,W:tl.constexpr,L:tl.constexpr,Shared:tl.constexpr):
    rank=tl.program_id(0);count=tl.load(Ranks+rank)
    f,b,_=_table_cost(FTable,BTable,Shared,count,L)
    for j in range(W):
        valid=tl.load(ResidentValid+rank*W+j);e=tl.load(Resident+rank*W+j)
        n=tl.load(Counts+e*G+rank);curve=tl.load(CurveMap+e)
        ef,eb,_=_table_cost(FTable,BTable,curve,n,L)
        f=f+tl.where(valid,ef,0.);b=b+tl.where(valid,eb,0.)
    tl.store(F+rank,f);tl.store(B+rank,b)

def initialize_rank_costs(r):
    _initial_rank_costs[(r.G,)](r.ecn,r.rcn,r.resident_experts,r.resident_valid,r.curve_map,r.Ftable,r.Btable,r.fw,r.bw,r.G,r.resident_width,r.L,r.shared_map,num_warps=4,enable_fp_fusion=False)
