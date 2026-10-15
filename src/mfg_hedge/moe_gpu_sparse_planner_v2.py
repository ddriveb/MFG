"""Opt-in FP64 sparse device model; no dense resource-by-variable matrix.

Torch and Triton are imported only when a runtime is explicitly constructed.
Adaptive trials stay on device inside captured chunks. Gap checks between
chunks still synchronize the host: this is not a one-kernel exact solver.
"""
from dataclasses import dataclass
import math, time
from .moe_communication_social_v2 import BLOCK_NAMES

@dataclass(frozen=True)
class SparseGpuPlan:
    shares: object
    dual: object
    quotas: object
    status: str
    iterations: int
    proposals: int
    upper_bound_ms: float
    lower_bound_ms: float
    gap_ms: float
    complete_C_ms: float
    graph_capture_ms: float

class SparseGpuPlanner:
    def __init__(self, problem, *, device='cuda:0', use_graphs=True, fused=True, projection='sort'):
        import torch
        self.torch=torch;self.device=torch.device(device);self.problem=problem
        if projection not in ('sort','active'):raise ValueError('projection must be sort or active')
        self.projection=projection
        self.use_graphs=bool(use_graphs and self.device.type=='cuda');self.fused=bool(fused and self.device.type=='cuda')
        if self.fused:
            from . import moe_gpu_sparse_kernels_v1 as kernels
            self.kernels=kernels
        self.graphs={};self.graph_capture_ms=0.
        def tensor(x,integer=False):return torch.tensor(x,device=self.device,dtype=torch.int64 if integer else torch.float64)
        self.tensor=tensor
        self.block_offsets=[0]
        resources=[];variables=[];units=[]
        for block in problem.blocks:
            base=self.block_offsets[-1]
            for j,column in enumerate(block.columns):
                row=problem.variable_rows[j];d=problem.rows[row].demand
                for resource,coefficient in column:
                    resources.append(base+resource);variables.append(j);units.append(coefficient/d if d else 0.)
            self.block_offsets.append(base+len(block.resource_names))
        # Zero-demand rows require rebuilding unit topology from demand-one rows.
        if any(row.demand==0 for row in problem.rows):
            from .moe_communication_social_v2 import SparseSocialProblem, RouteRow
            one=SparseSocialProblem.from_rows(tuple(RouteRow(a.source_gpu,a.expert_id,1,a.allowed_gpus) for a in problem.rows),problem.config)
            resources=[];variables=[];units=[]
            for b,block in enumerate(one.blocks):
                for j,column in enumerate(block.columns):
                    for resource,c in column:resources.append(self.block_offsets[b]+resource);variables.append(j);units.append(c)
        self.resource_count=self.block_offsets[-1];self.variable_count=problem.variable_count
        self.ir=tensor(resources,True);self.iv=tensor(variables,True);self.unit=tensor(units)
        self.variable_rows=tensor(problem.variable_rows,True);self.incidence_rows=self.variable_rows[self.iv]
        self.width=max(len(a.allowed_gpus) for a in problem.rows)
        if self.width>4:raise ValueError('sparse device row projection supports at most four replicas')
        slots=[];valid=[];groups={}
        for i,a in enumerate(problem.rows):
            ix=list(range(problem.row_offsets[i],problem.row_offsets[i+1]));groups.setdefault(len(ix),[]).append(ix)
            slots.append(ix+[0]*(self.width-len(ix)));valid.append([True]*len(ix)+[False]*(self.width-len(ix)))
        self.slots=tensor(slots,True);self.valid=torch.tensor(valid,device=self.device,dtype=torch.bool)
        self.groups={w:tensor(ix,True) for w,ix in groups.items()}
        self.ro=tensor(problem.row_offsets,True)
        self.dual_offsets=tensor(self.block_offsets,True)
        self.counts=tensor([a.demand for a in problem.rows],True)
        self.uniform=tensor(problem.uniform_shares())
        self.p=self.uniform.clone();self.dual=torch.zeros(self.resource_count,device=self.device,dtype=torch.float64)
        self.p_sum=torch.zeros_like(self.p);self.d_sum=torch.zeros_like(self.dual)
        self.trial_p=self.p.clone();self.trial_d=self.dual.clone()
        self.accepted=tensor(0,True);self.rejected=tensor(0,True);self.weight=tensor(0.)
        self.gamma=tensor(0.);self.max_gamma=tensor(0.);self.normalization=tensor(1.)
        self.limit=tensor(10000,True)
        self.pt=torch.empty_like(self.p);self.pn=torch.empty_like(self.p);self.grad=torch.empty_like(self.p);self.gt=torch.empty_like(self.p)
        self.dt=torch.empty_like(self.dual);self.dn=torch.empty_like(self.dual);self.load=torch.empty_like(self.dual);self.lt=torch.empty_like(self.dual)
        self.partial=torch.empty(((self.variable_count+self.resource_count+255)//256,3),device=self.device,dtype=torch.float64)
        self.control=tensor([0.,0.,0.])
        self.coefficient=torch.empty_like(self.unit)
        self._prepare_csr(resources,variables,units)
        self.set_demands([a.demand for a in problem.rows])

    def _prepare_csr(self, resources, variables, units):
        def build(keys,other,size):
            order=sorted(range(len(keys)),key=lambda i:(keys[i],i));offset=[0];pos=0
            for key in range(size):
                while pos<len(order) and keys[order[pos]]==key:pos+=1
                offset.append(pos)
            return self.tensor(offset,True),self.tensor([other[i] for i in order],True),self.tensor(order,True),max((offset[i+1]-offset[i] for i in range(size)),default=1)
        self.lr,self.lc,self.lo,self.lwidth=build(resources,variables,self.resource_count)
        self.ar,self.ac,self.ao,self.awidth=build(variables,resources,self.variable_count)

    def synchronize(self):
        if self.device.type=='cuda':self.torch.cuda.synchronize(self.device)

    def set_demands(self, demands):
        t=self.torch
        if isinstance(demands,t.Tensor):
            if demands.shape!=self.counts.shape or demands.dtype!=t.int64 or demands.device!=self.device:raise ValueError('demands require resident int64 row vector')
            if bool((demands<0).any().item()):raise ValueError('demands must be nonnegative')
            self.counts.copy_(demands)
        else:
            if len(demands)!=len(self.problem.rows) or any(type(n)is not int or n<0 for n in demands):raise ValueError('demands must be nonnegative row integers')
            self.counts.copy_(self.tensor(demands,True))
        self.coefficient.copy_(self.unit*self.counts[self.incidence_rows])
        if not hasattr(self,'lcoeff'):
            self.lcoeff=self.coefficient[self.lo];self.acoeff=self.coefficient[self.ao]
        else:
            self.lcoeff.copy_(self.coefficient[self.lo]);self.acoeff.copy_(self.coefficient[self.ao])
        loads=self.apply(self.uniform)
        normal=self.problem.fixed_startup_ms+sum(loads[a:b].max() for a,b in zip(self.block_offsets,self.block_offsets[1:]))
        self.normalization.copy_(t.where(normal>0,normal,self.tensor(1.)))
        col=t.zeros_like(self.p).scatter_add_(0,self.iv,self.coefficient)
        row=t.zeros_like(self.dual).scatter_add_(0,self.ir,self.coefficient)
        norm=(col.max()*row.max()).sqrt()/self.normalization
        self.initial_gamma=t.where(norm>0,.5/norm,self.tensor(0.));self.max_gamma.copy_(self.initial_gamma*1e4)

    def apply(self,p):
        if self.fused:return self.kernels.spmv(self.lr,self.lc,self.lcoeff,p,self.resource_count,self.lwidth)
        return self.torch.zeros_like(self.dual).scatter_add_(0,self.ir,self.coefficient*p[self.iv])

    def adjoint(self,d):
        if self.fused:return self.kernels.spmv(self.ar,self.ac,self.acoeff,d,self.variable_count,self.awidth)
        return self.torch.zeros_like(self.p).scatter_add_(0,self.iv,self.coefficient*d[self.ir])

    def project_primal(self,v):
        if self.fused:return self.kernels.project_rows(v,self.ro,len(self.problem.rows),self.width)
        from .moe_gpu_three_planners_v1 import simplex_projection
        out=self.torch.empty_like(v)
        for w,ix in self.groups.items():
            a=v[ix]
            if w==2:
                x=((a[:,0]-a[:,1]+1)*.5).clamp(0,1);a=self.torch.stack((x,1-x),-1)
            else:a=simplex_projection(a)
            out[ix.reshape(-1)]=a.reshape(-1)
        return out

    def project_dual(self,v):
        if self.fused:
            project=self.kernels.project_blocks_active if self.projection=='active' else self.kernels.project_blocks
            return project(v,self.block_offsets,self.dual_offsets)
        from .moe_gpu_three_planners_v1 import simplex_projection
        return self.torch.cat([simplex_projection(v[a:b]) for a,b in zip(self.block_offsets,self.block_offsets[1:])])

    def initialize(self, initial=None):
        t=self.torch
        if initial is None:
            self.p.copy_(self.uniform);loads=self.apply(self.p)
            for a,b in zip(self.block_offsets,self.block_offsets[1:]):
                active=(loads[a:b]==loads[a:b].max()).double();self.dual[a:b].copy_(active/active.sum())
        else:
            p,d=initial
            if p.shape!=self.p.shape or d.shape!=self.dual.shape or p.dtype!=t.float64 or d.dtype!=t.float64 or p.device!=self.device or d.device!=self.device:raise ValueError('initial vectors must be matching resident FP64 arrays')
            if not bool((t.isfinite(p).all()&t.isfinite(d).all()).item()):raise ValueError('initial vectors must be finite')
            self.p.copy_(self.project_primal(p));self.dual.copy_(self.project_dual(d))
        self.p_sum.zero_();self.d_sum.zero_();self.accepted.zero_();self.rejected.zero_();self.weight.zero_();self.gamma.copy_(self.initial_gamma)

    def certificate(self,p,d):
        loads=self.apply(p);gradient=self.adjoint(d)
        upper=self.problem.fixed_startup_ms+sum(loads[a:b].max() for a,b in zip(self.block_offsets,self.block_offsets[1:]))
        lower=self.problem.fixed_startup_ms+sum(gradient[ix].min(-1).values.sum() for ix in self.groups.values())
        return upper,lower

    def round_shares(self,p):
        t=self.torch;shares=t.where(self.valid,p[self.slots],0.)
        exact=shares.clamp_min(0)*self.counts[:,None];floors=exact.floor().long();rest=self.counts-floors.sum(-1)
        order=t.argsort(t.where(self.valid,exact-floors,-t.inf),dim=-1,descending=True,stable=True)
        add=t.zeros_like(floors).scatter(-1,order,(t.arange(self.width,device=self.device)[None,:]<rest[:,None]).long())
        return floors+add

    def _proposal(self):
        if self.fused:
            self.kernels.adaptive_proposal(self);return
        t=self.torch;g=self.gamma/self.normalization
        grad=self.adjoint(self.dual);load=self.apply(self.p)
        pt=self.project_primal(self.p-g*grad);dt=self.project_dual(self.dual+g*load)
        gt=self.adjoint(dt);lt=self.apply(pt)
        pn=self.project_primal(self.p-g*gt);dn=self.project_dual(self.dual+g*lt)
        delta=g*((gt-grad)*(pt-pn)).sum()-g*((lt-load)*(dt-dn)).sum()-.5*((pt-self.p).square().sum()+(dt-self.dual).square().sum()+(pt-pn).square().sum()+(dt-dn).square().sum())
        accept=(delta<=0.)&(self.accepted<self.limit)
        self.p_sum.add_(t.where(accept,self.gamma*pt,0.));self.d_sum.add_(t.where(accept,self.gamma*dt,0.))
        self.weight.add_(t.where(accept,self.gamma,0.));self.accepted.add_(accept.long());self.rejected.add_((~accept).long())
        self.p.copy_(t.where(accept,pn,self.p));self.dual.copy_(t.where(accept,dn,self.dual))
        self.trial_p.copy_(t.where(accept,pt,self.trial_p));self.trial_d.copy_(t.where(accept,dt,self.trial_d))
        self.gamma.copy_(t.where(accept,t.minimum(self.gamma*1.2,self.max_gamma),self.gamma*.8))

    def _chunk(self,count):
        for _ in range(count):self._proposal()

    def prepare_graph(self,count=50):
        if not self.use_graphs:return
        if count in self.graphs:return
        t=self.torch;begin=time.perf_counter()
        states=[self.p,self.dual,self.p_sum,self.d_sum,self.trial_p,self.trial_d,self.accepted,self.rejected,self.weight,self.gamma]
        saved=[x.clone() for x in states]
        stream=t.cuda.Stream(device=self.device);stream.wait_stream(t.cuda.current_stream(self.device))
        with t.cuda.stream(stream),t.no_grad():self._chunk(count)
        t.cuda.current_stream(self.device).wait_stream(stream)
        graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph),t.no_grad():self._chunk(count)
        for a,b in zip(states,saved):a.copy_(b)
        self.synchronize();self.graphs[count]=graph;self.graph_capture_ms+=(time.perf_counter()-begin)*1000

    def solve_high_precision(self,demands,*,max_iterations=10000,absolute_gap_ms=.001,relative_gap=0.,chunk_proposals=50,initial=None):
        if type(max_iterations)is not int or max_iterations<=0:raise ValueError('max_iterations must be positive')
        if type(chunk_proposals)is not int or chunk_proposals<=0:raise ValueError('chunk_proposals must be positive')
        if any(not math.isfinite(x) or x<0 for x in (absolute_gap_ms,relative_gap)):raise ValueError('invalid gap threshold')
        self.synchronize();start=time.perf_counter()
        with self.torch.no_grad():
            self.set_demands(demands);self.initialize(initial);self.limit.fill_(max_iterations)
            self.prepare_graph(chunk_proposals)
            best_p=self.p.clone();best_d=self.dual.clone();upper,lower=self.certificate(best_p,best_d)
            status='iteration_cap';proposals=0
            while True:
                u,l=upper.item(),lower.item();gap=u-l
                if not math.isfinite(gap) or gap < -1e-10*max(1.,abs(u),abs(l)):status='certificate_error';break
                if gap<=absolute_gap_ms+relative_gap*max(abs(u),abs(l)):status='converged';break
                if int(self.accepted.item())>=max_iterations or proposals>=max_iterations*4:break
                if self.use_graphs:self.graphs[chunk_proposals].replay()
                else:self._chunk(chunk_proposals)
                proposals+=chunk_proposals
                for p,d in ((self.p,self.dual),(self.trial_p,self.trial_d),(self.p_sum/self.weight.clamp_min(1e-300),self.d_sum/self.weight.clamp_min(1e-300))):
                    cu,cl=self.certificate(p,d)
                    best_p=self.torch.where(cu<upper,p,best_p);best_d=self.torch.where(cl>lower,d,best_d)
                    upper=self.torch.minimum(upper,cu);lower=self.torch.maximum(lower,cl)
            quotas=self.round_shares(best_p);self.synchronize()
            return SparseGpuPlan(best_p,best_d,quotas,status,int(self.accepted.item()),proposals,float(upper),float(lower),float(upper-lower),(time.perf_counter()-start)*1000,self.graph_capture_ms)
