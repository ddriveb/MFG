from pathlib import Path
import argparse,sys,os,json,time,math,random,hashlib,statistics as st,dataclasses

V2=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description="Offline FP64 V2 training; no packages or assets are downloaded.")
parser.add_argument("--output",help="new output directory under v2/artifacts/")
parser.add_argument("--initial-checkpoint",default=str(V2/"models/checkpoints/g64_r4/best.pt"))
parser.add_argument("--device",default="cuda:0",help="Torch training device, normally cuda:0")
parser.add_argument("--python-include",help="optional external Python include path for Triton")
parser.add_argument("--triton-cache",help="optional external Triton cache directory")
parser.add_argument("--allocation-steps",type=int,default=4000)
parser.add_argument("--price-steps",type=int,default=2000)
parser.add_argument("--eval-every",type=int,default=250)
parser.add_argument("--train-per-scale",type=int,default=256)
parser.add_argument("--dev-per-scale",type=int,default=64)
parser.add_argument("--batch-per-scale",type=int,default=2)
parser.add_argument("--seed",type=int,default=132053)
args=parser.parse_args()
R=V2
if args.allocation_steps<=0 or args.price_steps<=0 or args.eval_every<=0:
    raise ValueError("training and evaluation step counts must be positive")
if args.batch_per_scale<=0 or args.train_per_scale<args.batch_per_scale or args.dev_per_scale<=0:
    raise ValueError("train/dev sizes must support the selected per-scale batch")
if args.python_include:
    include=Path(args.python_include).expanduser().resolve()
    if not include.is_dir(): raise FileNotFoundError(include)
    os.environ["CPATH"]=str(include)
if args.triton_cache:
    cache=Path(args.triton_cache).expanduser().resolve(); cache.mkdir(parents=True,exist_ok=True)
    os.environ["TRITON_CACHE_DIR"]=str(cache)
sys.path.insert(0,str(R/"src"));sys.path.insert(0,str(R/"tools"))
if args.output:
    O=Path(args.output).expanduser()
    if not O.is_absolute(): O=R/O
else:
    O=R/"artifacts"/("training-multiscale-"+time.strftime("%Y%m%d-%H%M%S"))
O=O.resolve()
if not O.is_relative_to((R/"artifacts").resolve()): raise ValueError("training output must stay under v2/artifacts/")
O.mkdir(parents=True,exist_ok=False)
if not args.triton_cache: os.environ["TRITON_CACHE_DIR"]=str(O/"triton-cache")
import torch
from controlled_gate import generate
from mfg_hedge.moe_load_feedback_gnn_v1 import build_load_feedback_gnn
from mfg_hedge.moe_gpu_cached_warmstart_v1 import CachedGpuWarmStart
from mfg_hedge.moe_gpu_separated_training_v1 import separated_physical_losses
from mfg_hedge.moe_communication_social_v2_g256_sparse_warmstart_v1 import physical_certificate
from mfg_hedge.moe_legal_variable_gate_v1 import fixed_candidate_support,FAMILIES
from mfg_hedge.moe_communication_social_v2 import CommunicationConfig,RouteRow,SparseSocialProblem
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# The CLI created this unique output directory.
def write(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
initial=Path(args.initial_checkpoint).expanduser().resolve()
if not initial.is_file() or not initial.is_relative_to((V2/'models/checkpoints').resolve()): raise ValueError('initial checkpoint must be an existing file under v2/models/checkpoints/')
protocol=dict(scales=[16,32,64],families=FAMILIES,train_per_scale=args.train_per_scale,dev_per_scale=args.dev_per_scale,train_support_seeds=list(range(1323001,1323005)),dev_support_seeds=[1324001,1324002],
              holdout_support_seed=1325001,train_gate_first=1330001,dev_gate_first=1340001,scale_seed_stride=10000,holdout_seeds=list(range(1350001,1350017)),
              initialization_seed=args.seed,initial_checkpoint_sha256=sha(initial),initial_checkpoint_path=str(initial.relative_to(V2)),allocation_steps=args.allocation_steps,price_steps=args.price_steps,batch=args.batch_per_scale*3,eval_every=args.eval_every,
              selection='allocation meanU/F_unif then maxU/F_unif; price meanrelativegap, devonly',reference_labels=False,feedback_rounds=4,width=16,price_width=64,
              source_hashes={str(p.relative_to(R)):sha(p) for p in [Path(__file__),Path(__file__).with_name('controlled_gate.py'),R/'docs/adr/0222-paper-evidence-alignment-and-weak-scaling.md',
                       R/'src/mfg_hedge/moe_load_feedback_gnn_v1.py',R/'src/mfg_hedge/moe_gpu_separated_training_v1.py']})
write('protocol.json',protocol)
net=build_load_feedback_gnn(feedback_rounds=4).to(args.device);net.load_state_dict(torch.load(initial,weights_only=True)['state_dict'])
dynamic=('row_features','edge_features','resource_features','resource_to_variable_weight','variable_to_resource_weight','incidence_coefficients','f_reference','step_scale')
cases={'train':[],'dev':[]};counts_archive=[];started=time.perf_counter()
for scale_index,P in enumerate(protocol['scales']):
    physical=json.loads((R/'models/checkpoints/g64_r4/protocol.json').read_text())['physical'];physical['gpu_nodes']=[s//8 for s in range(P)]
    config=CommunicationConfig(**{**physical,'gpu_nodes':tuple(physical['gpu_nodes'])})
    for split,first,n,seeds in [('train',1330001,protocol['train_per_scale'],protocol['train_support_seeds']),('dev',1340001,protocol['dev_per_scale'],protocol['dev_support_seeds'])]:
        caches={}
        for i in range(n):
            seed=first+scale_index*10000+i;family=FAMILIES[i%4];support_seed=seeds[(i//4)%len(seeds)]
            support=fixed_candidate_support(gpu_count=P,expert_count=P,candidates=16,seed=support_seed)
            counts,_,strength=generate(support,seed,family,P)
            problem=SparseSocialProblem.from_rows(tuple(RouteRow(s,e,counts[s*16+j],(e,(e+1)%P)) for s,es in enumerate(support) for j,e in enumerate(es)),config)
            if support_seed not in caches:caches[support_seed]=CachedGpuWarmStart(problem,net,projection='active',device=args.device)
            cache=caches[support_seed]
            with torch.no_grad():
                cache.update_features(torch.tensor(counts,device=args.device,dtype=torch.int64));g=dataclasses.replace(cache.graph,problem=problem,**{k:getattr(cache.graph,k).clone() for k in dynamic})
            item=dict(P=P,seed=seed,family=family,support_seed=support_seed,graph=g)
            cases[split].append(item);counts_archive.append(dict(P=P,split=split,seed=seed,family=family,support_seed=support_seed,counts=counts))
        print('graphs',P,split,n,flush=True)
write('train_dev_counts.json',counts_archive)
def evaluate(items,price=False):
    rows=[]
    with torch.no_grad():
        for item in items:
            g=item['graph'];net.prepare(g)
            if price:p=item['fixed_p'];d=net.prices_from_features(item['features'])
            else:p,d=net(g)
            u,l,gap=physical_certificate(g,p,d);f=g.f_reference.item()
            rows.append(dict(P=item['P'],seed=item['seed'],family=item['family'],upper_ms=u.item(),lower_ms=l.item(),gap_ms=gap.item(),relative_upper=u.item()/f,relative_gap=gap.item()/f,uniform_upper_ms=f))
    def one(rr):return dict(n=len(rr),U=st.mean(v['upper_ms'] for v in rr),L=st.mean(v['lower_ms'] for v in rr),gap=st.mean(v['gap_ms'] for v in rr),relative_U=st.mean(v['relative_upper'] for v in rr),
                             relative_gap=st.mean(v['relative_gap'] for v in rr),max_relative_U=max(v['relative_upper'] for v in rr),max_gap=max(v['gap_ms'] for v in rr),gap_le_0_5=sum(v['gap_ms']<=.5 for v in rr))
    return dict(all=one(rows),by_scale={P:one([v for v in rows if v['P']==P]) for P in protocol['scales']}),rows
def record(stage,step,price=False):
    s,rows=evaluate(cases['dev'],price);rec=dict(stage=stage,step=step,elapsed_s=time.perf_counter()-started,summary=s)
    with (O/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    write('progress.json',rec);print(json.dumps(rec),flush=True);return s,rows
rng=random.Random(args.seed);buckets={P:[v for v in cases['train'] if v['P']==P] for P in protocol['scales']}
for p in net.price.parameters():p.requires_grad_(False)
opt=torch.optim.Adam(net.allocation.parameters(),lr=.002);best=None;best_state=None
for step in range(args.allocation_steps+1):
    if step:
        ratio=step/args.allocation_steps;eps=.1*(.005/.1)**ratio
        for group in opt.param_groups:group['lr']=.0001+.0019*(1+math.cos(math.pi*ratio))/2
        opt.zero_grad(set_to_none=True)
        for P in protocol['scales']:
            for item in rng.sample(buckets[P],args.batch_per_scale):
                g=item['graph'];net.prepare(g);p,d=net(g);loss=separated_physical_losses(g,p,d,smoothing_ms=eps)['primal']/(args.batch_per_scale*len(protocol['scales']))
                assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.allocation.parameters(),5.);assert torch.isfinite(norm);opt.step()
    if step%args.eval_every==0:
        s,rows=record('allocation',step);key=(s['all']['relative_U'],s['all']['max_relative_U'])
        if best is None or key<best:
            best=key;best_state={k:v.detach().clone() for k,v in net.allocation.state_dict().items()};torch.save(dict(allocation=best_state,step=step,dev=s),O/'allocation-best.pt')
net.allocation.load_state_dict(best_state)
for p in net.allocation.parameters():p.requires_grad_(False)
for p in net.price.parameters():p.requires_grad_(True)
with torch.no_grad():
    for item in cases['train']+cases['dev']:
        net.prepare(item['graph']);p,features=net.encode(item['graph']);item['fixed_p']=p.clone();item['features']=features.clone()
    features=torch.cat([v['features'] for v in cases['train']]);mean=features.mean(0);std=features.std(0,unbiased=False).clamp_min(.001)
    net.price[0].weight.div_(std[None,:]);net.price[0].bias.sub_(net.price[0].weight@mean);del features
allocation_frozen={k:v.clone() for k,v in net.allocation.state_dict().items()};opt=torch.optim.Adam(net.price.parameters(),lr=.002);best=None;best_state=None
for step in range(args.price_steps+1):
    if step:
        ratio=step/args.price_steps;eps=.1*(.005/.1)**ratio
        for group in opt.param_groups:group['lr']=.0001+.0019*(1+math.cos(math.pi*ratio))/2
        opt.zero_grad(set_to_none=True)
        for P in protocol['scales']:
            for item in rng.sample(buckets[P],args.batch_per_scale):
                g=item['graph'];net.prepare(g);d=net.prices_from_features(item['features']);loss=separated_physical_losses(g,item['fixed_p'],d,smoothing_ms=eps)['dual']/(args.batch_per_scale*len(protocol['scales']))
                assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(net.price.parameters(),5.);assert torch.isfinite(norm);opt.step()
    if step%args.eval_every==0:
        s,rows=record('price',step,True);key=(s['all']['relative_gap'],s['all']['max_gap'])
        if best is None or key<best:
            best=key;best_state={k:v.detach().clone() for k,v in net.price.state_dict().items()};torch.save(dict(price=best_state,step=step,dev=s),O/'price-best.pt')
net.price.load_state_dict(best_state);assert all(torch.equal(v,allocation_frozen[k]) for k,v in net.allocation.state_dict().items())
s,rows=evaluate(cases['dev']);write('selected_dev.json',dict(summary=s,rows=rows))
torch.save(dict(state_dict=net.state_dict(),width=16,price_width=64,feedback_rounds=4,pressure_features=False,dev=s,scales=protocol['scales']),O/'best.pt')
assert all(sha(R/p)==h for p,h in protocol['source_hashes'].items())
write('training_complete.json',dict(elapsed_s=time.perf_counter()-started,checkpoint_sha256=sha(O/'best.pt'),allocation_frozen_during_price=True,holdouts_unopened=True,source_hashes_unchanged=True,run_kind=('full' if (args.allocation_steps,args.price_steps,args.train_per_scale,args.dev_per_scale,args.batch_per_scale,args.eval_every,args.seed)==(4000,2000,256,64,2,250,132053) and initial==(V2/'models/checkpoints/g64_r4/best.pt').resolve() else 'smoke' if (args.allocation_steps<4000 or args.price_steps<2000 or args.train_per_scale<256 or args.dev_per_scale<64 or args.batch_per_scale<2) else 'custom'),
                                   device=torch.cuda.get_device_name(torch.device(args.device)),allocation_parameters=sum(p.numel() for p in net.allocation.parameters()),price_parameters=sum(p.numel() for p in net.price.parameters())))
print('TRAINING_COMPLETE',flush=True)
