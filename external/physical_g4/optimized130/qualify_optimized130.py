"""Exact quota equivalence on saved real Gate states and zero/tie cases."""
import json,time,statistics,hashlib
from pathlib import Path
import torch
from mfg_hedge.granite_g4_feedback_planner_v1 import DenseSupportPlanner,baseline_quotas,load_checkpoint,default_model_config
from optimized130_port import OptimizedPlanner

root=Path('/root/mfg-g4-v100-20261013-a');out=root/'optimized130'
torch.set_num_threads(1);torch.cuda.set_device(0)
cfg=json.loads((root/'timing100130/config.json').read_text())
model=load_checkpoint(Path(cfg['checkpoint']),torch.device('cuda:0'),cfg['checkpoint_sha256'])
ref=DenseSupportPlanner(model,default_model_config(),device=torch.device('cuda:0'))
fast_model=load_checkpoint(Path(cfg['checkpoint']),torch.device('cuda:0'),cfg['checkpoint_sha256'])
fast=OptimizedPlanner(fast_model,default_model_config(),device=torch.device('cuda:0'))
cases=[];seen=set()
for line in (root/'results/deadline-three-arm-20261014-b/raw-rank-0.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row['arm']!='gnn':continue
    for layer,p in enumerate(row['planner']):
        values=tuple(v for r in p['source_counts'] for v in r)
        if values not in seen:cases.append(('saved/'+str(row['sample'])+'/'+str(layer),values));seen.add(values)
cases.extend([('zero',[0]*128),('one',[1]*128),('odd',[3]*128),('sparse',[i%7 if i%4==0 else 0 for i in range(128)])])
for seed in range(10):
    gen=torch.Generator().manual_seed(131000+seed)
    cases.append(('random/'+str(seed),torch.randint(0,8193,(128,),generator=gen).tolist()))
records=[];max_share=0.
with torch.no_grad():
    for i,(name,values) in enumerate(cases):
        c=torch.tensor(values,device='cuda',dtype=torch.long)
        target={'gnn':ref.quotas(c).clone()}
        rp=ref.shares.clone()
        target.update({arm:baseline_quotas(c.reshape(4,32),arm).reshape(128,2) for arm in ('sinkhorn','lazarus')})
        for arm in ('gnn','sinkhorn','lazarus'):
            actual=fast.dispatch(c,arm)
            if not torch.equal(actual,target[arm]):
                bad=dict(case=name,arm=arm,diff=int((actual!=target[arm]).sum()),actual=actual.tolist(),expected=target[arm].tolist())
                (out/'quota-failure.json').write_text(json.dumps(bad,indent=2));raise AssertionError((name,arm,bad['diff']))
            assert torch.equal(actual.sum(-1),c) and bool((actual>=0).all())
        # Shared model weights but independent graph state; compare raw predictions as well.
        delta=float((fast.fused()-rp).abs().max());max_share=max(max_share,delta)
        if delta>1e-10:raise AssertionError(('share difference',name,delta))
        records.append(dict(case=name,quota_equal=True,max_share_abs=delta))
        if (i+1)%100==0:print('QUALIFIED',i+1,flush=True)
timing={}
c=torch.tensor(cases[0][1],device='cuda',dtype=torch.long)
for arm in ('gnn','sinkhorn','lazarus'):
    values=[]
    for i in range(120):
        torch.cuda.synchronize();start=time.perf_counter();fast.dispatch(c,arm);torch.cuda.synchronize()
        if i>=20:values.append((time.perf_counter()-start)*1000)
    timing[arm]=dict(mean_ms=statistics.mean(values),samples=len(values))
result=dict(passed=True,cases=len(cases),quota_comparisons=len(cases)*3,max_share_abs=max_share,records=records,
    timing=timing,checkpoint_sha256=cfg['checkpoint_sha256'],gpu=torch.cuda.get_device_name(),profile='fused_features_gnn4_sinkhorn50_lazarus_cuda_graph',strict_fullmodel_numerics_passed=False)
(out/'qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
