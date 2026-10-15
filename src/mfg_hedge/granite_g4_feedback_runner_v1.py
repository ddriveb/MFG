"""Four-rank paired full-step experiment, or explicit local tiny Gloo check.

Every arm is the first AdamW step from a common restored state. Completion is
written only after collective full-gradient, logits and updated-state checks.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-dir');p.add_argument('--schedule-library')
    p.add_argument('--local-tiny',action='store_true')
    p.add_argument('--init-file',help='local-tiny file-store rendezvous only')
    a=p.parse_args(argv)
    cfgpath=Path(a.config).resolve();cfg=json.loads(cfgpath.read_text())
    if cfg.get('schema')!='g4_feedback_paired_v1':raise ValueError('config schema')
    if cfg['rounds'] not in (4,8):raise ValueError('four/eight rounds required')
    if cfg['gradient_rtol']!=1e-5 or cfg['gradient_atol']!=1e-7:
        raise ValueError('frozen strict tolerance required; no automatic relaxation')
    if not 1<=cfg['measured_steps']<=100:raise ValueError('explicit measured budget 1..100')
    if not a.local_tiny and (not a.model_dir or not a.schedule_library):
        raise ValueError('full run requires local model and compiled native schedule')
    import torch
    import torch.distributed as dist
    from .granite_g4_feedback_planner_v1 import load_checkpoint,default_model_config,DenseSupportPlanner
    from .granite_g4_feedback_replay_v1 import (PairedPolicy,install_gate_replay,test_schedule,
        whole_expert_scope,snapshot,initialize_optimizer,reset_optimizer,compare_tensors,gate_digest,validate_records)
    from .granite_communication_social_v2_training_replica_packed_v1 import install_packed_training_routing
    from .granite_full_training_runner_v1 import _load_tiny_transformers_model,_load_model
    torch.set_num_threads(1)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('exactly four ranks required')
    device=torch.device('cpu' if a.local_tiny else f"cuda:{int(os.environ['LOCAL_RANK'])}")
    if device.type=='cuda':
        if torch.cuda.device_count()<4:raise ValueError('four visible GPUs required')
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    kwargs={}
    if a.init_file:
        if not a.local_tiny:raise ValueError('file rendezvous is local-test only')
        kwargs=dict(init_method=Path(a.init_file).resolve().as_uri(),rank=rank,world_size=world)
    dist.init_process_group('gloo' if a.local_tiny else 'nccl',timeout=timedelta(minutes=10),**kwargs)
    output=Path(a.output).resolve()
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def consensus(error):
        errors=[None]*4;dist.all_gather_object(errors,error)
        if any(errors):raise RuntimeError('paired verification failed: '+str(errors))
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    try:
        ckpt=(cfgpath.parent/cfg['checkpoint']).resolve()
        gnn=load_checkpoint(ckpt,device,cfg['checkpoint_sha256'])
        metadata=torch.load(ckpt,map_location='cpu',weights_only=True)
        if metadata['feedback_rounds']!=cfg['rounds']:raise ValueError('checkpoint depth mismatch')
        del metadata
        dense=DenseSupportPlanner(gnn,default_model_config(),device=device)
        policy=PairedPolicy(dense)
        torch.manual_seed(cfg['seed'])
        if a.local_tiny:model,names=_load_tiny_transformers_model(torch,cfg['seed'],device)
        else:model,names=_load_model(Path(a.model_dir),{},device)
        routers=install_gate_replay(model)
        schedule=test_schedule if a.local_tiny else Path(a.schedule_library).resolve()
        handle=install_packed_training_routing(model,policy,schedule,4,rank)
        initial=snapshot(model)
        opt=torch.optim.AdamW(handle.optimizer_parameters,lr=cfg['learning_rate'],weight_decay=.01)
        initialize_optimizer(model,opt)
        vocab=model.config.vocab_size;length=8 if a.local_tiny else cfg['tokens_per_rank']
        warmup=0 if a.local_tiny else cfg['warmup_steps']
        measured=2 if a.local_tiny else cfg['measured_steps']
        records=[]
        write(f'provenance-rank-{rank}.json',dict(config=cfg,torch=torch.__version__,
            device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            mode='local_tiny_gloo' if a.local_tiny else 'physical_g4',
            reference='common unpadded routed executor; not original-transformers parity',
            sampling='synthetic token IDs; independent first optimizer steps',
            timing='full forward/backward/gradient sync/AdamW including planner staging and collectives',
            source_checkpoint=str(ckpt),completed=False))
        with whole_expert_scope():
            for sample in range(warmup+measured):
                torch.manual_seed(cfg['seed']+sample*4+rank)
                ids=torch.randint(0,vocab,(1,length),device=device)
                # Record discrete Gate choices on the same initial model, outside timing.
                model.load_state_dict(initial);policy.policy='uniform';policy.records=[]
                for r in routers:r.recording=True
                with torch.no_grad():model(input_ids=ids,labels=ids,use_cache=False)
                for r in routers:r.recording=False
                digest=gate_digest(routers)
                arms=['gnn','sinkhorn','lazarus'];k=sample%3;arms=arms[k:]+arms[:k]
                reference=None
                for arm in ['uniform']+arms:
                    model.load_state_dict(initial);reset_optimizer(opt)
                    policy.policy=arm;policy.records=[];handle.reset_timing_diagnostics()
                    torch.manual_seed(cfg['seed']+sample)
                    dist.barrier();sync();start=time.perf_counter()
                    result=model(input_ids=ids,labels=ids,use_cache=False)
                    result.loss.backward();handle.sync_gradients();opt.step();sync()
                    elapsed=(time.perf_counter()-start)*1000
                    # Full parameter checks are outside timing; no sampled gradients.
                    actual={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
                    error=None;maxerr=None
                    try:
                        if gate_digest(routers)!=digest:raise AssertionError('Gate choices changed')
                        if any(v is None for v in actual.values()):raise AssertionError('missing synchronized gradient')
                        values=dict(actual, __logits=result.logits.detach(),__loss=result.loss.detach())
                        if reference is None:
                            compare_tensors(values,values)
                            reference={n:v.detach().cpu().clone() for n,v in values.items()}
                            reference_state=snapshot(model)
                        else:
                            maxerr=compare_tensors(values,reference,rtol=cfg['gradient_rtol'],atol=cfg['gradient_atol'])
                            compare_tensors(model.state_dict(),reference_state,
                                rtol=cfg['gradient_rtol'],atol=cfg['gradient_atol'])
                    except Exception as exc:error=str(exc)
                    row=dict(sample=sample,arm=arm,warmup=sample<warmup,rank=rank,
                        gate_sha256=digest,full_step_ms=elapsed,max_abs_difference=maxerr,
                        passed=error is None,error=error,planner=policy.records,
                        backend=handle.read_timing_diagnostics())
                    if rank==0:
                        # Common certificate per input; deliberately outside the step timer.
                        for plan in row['planner']:
                            with torch.no_grad():
                                c=torch.tensor(plan['source_counts'],device=device,dtype=torch.long).flatten()
                                dense.quotas(c)
                                shares,dual=dense.model(dense.graph)
                                q=torch.tensor(plan['source_quotas'],device=device,dtype=torch.float64).reshape(-1,2)
                                executable=torch.where(c[:,None]>0,q/c[:,None].clamp_min(1),.5).flatten()
                                upper,lower=dense.cache.runtime.certificate(executable,torch.cat(dual))
                                plan.update(integer_model_ms=float(upper),common_lower_ms=float(lower),
                                            integer_gap_ms=float(upper-lower))
                    records.append(row)
                    with (output/f'raw-rank-{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    consensus(error)
                    if not handle.check_replica_identity(optimizer=opt):raise AssertionError('replica state differs')
        all_records=[None]*4;dist.all_gather_object(all_records,records)
        if rank==0:
            audit=validate_records(all_records,samples=warmup+measured,layers=len(names))
            summary={}
            for arm in ('gnn','sinkhorn','lazarus'):
                times=[max(next(x['full_step_ms'] for x in rr if x['sample']==sample and x['arm']==arm)
                           for rr in all_records) for sample in range(warmup,warmup+measured)]
                summary[arm]={'mean_full_step_ms':sum(times)/len(times),'samples':len(times)}
            write('completed.json',dict(completed=True,local_tiny=a.local_tiny,summary=summary,audit=audit,
                paired_numerics_passed=True,physical_run=not a.local_tiny,
                accuracy_training_convergence_claim=False))
    except Exception as exc:
        write(f'failure-rank-{rank}.json',{'error':repr(exc),'completed':False})
        raise
    finally:dist.destroy_process_group()
