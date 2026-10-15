"""User-authorized deadline-bounded timing; strict numerical qualification FAILED."""
import inspect, json, math, time, os, hashlib
from pathlib import Path
from datetime import datetime

def validate_timing(all_records, *, samples, layers):
    if len(all_records)!=4: raise ValueError('four ranks required')
    for rank, rows in enumerate(all_records):
        if len(rows)!=samples*3: raise ValueError('incomplete matched groups')
        for sample in range(samples):
            group=[r for r in rows if r['sample']==sample]
            arms=['gnn','sinkhorn','lazarus']; k=sample%3
            if [r['arm'] for r in group]!=arms[k:]+arms[:k]: raise ValueError('arm order')
            if len({r['gate_sha256'] for r in group})!=1: raise ValueError('Gate mismatch')
            for r in group:
                if r['rank']!=rank or not r['integrity_passed'] or r['error']: raise ValueError('integrity failed')
                if not math.isfinite(r['full_step_ms']) or r['full_step_ms']<=0: raise ValueError('timing invalid')
                if len(r['planner'])!=(layers if rank==0 else 0): raise ValueError('missing plans')
            if rank==0:
                for layer in range(layers):
                    counts=group[0]['planner'][layer]['source_counts']
                    for r in group:
                        plan=r['planner'][layer]
                        if plan['source_counts']!=counts: raise ValueError('different counts')
                        q=plan['source_quotas']
                        if len(q)!=4 or any(len(row)!=32 for row in q): raise ValueError('quota shape')
                        for s in range(4):
                            for e in range(32):
                                if len(q[s][e])!=2 or any(type(v)is not int or v<0 for v in q[s][e]) or sum(q[s][e])!=counts[s][e]: raise ValueError('quota conservation')
    return dict(complete_ranks=4,independent_samples=samples,layers=layers,integrity_passed=True,strict_numerics_passed=False)

def build_source(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace("if not 1<=cfg['measured_steps']<=100:raise ValueError('explicit measured budget 1..100')", "if not 1<=cfg['measured_steps']<=10000:raise ValueError('deadline bounded budget 1..10000')")
    replace("        with whole_expert_scope():", "        completed_samples=0\n        torch.use_deterministic_algorithms(True)\n        with whole_expert_scope():")
    replace("            for sample in range(warmup+measured):", "            for sample in range(warmup+measured):\n                stop=[time.time()>=DEADLINE if rank==0 else None]\n                dist.broadcast_object_list(stop,src=0)\n                if stop[0]:break")
    replace("                for arm in ['uniform']+arms:","                for arm in arms:")
    start=source.index('                        values=dict(actual, __logits=')
    end=source.index('                    except Exception as exc:error=str(exc)',start)
    source=source[:start]+'''                        with torch.no_grad():
                            finite=[torch.isfinite(v).all() for v in actual.values()]
                            finite.extend(torch.isfinite(p).all() for p in model.parameters())
                            finite.extend([torch.isfinite(result.logits).all(),torch.isfinite(result.loss).all()])
                            if not bool(torch.stack(finite).all()):raise AssertionError('nonfinite tensors')
'''+source[end:]
    replace('                        passed=error is None,error=error,planner=policy.records,', '                        integrity_passed=error is None,strict_numerics_passed=False,error=error,planner=policy.records,')
    replace("                    if not handle.check_replica_identity(optimizer=opt):raise AssertionError('replica state differs')", "                completed_samples=sample+1\n                if rank==0:\n                    progress=dict(completed_groups=completed_samples,measured_groups=max(0,completed_samples-warmup),target=measured,strict_numerics_passed=False,updated_at=time.time())\n                    write('progress.json',progress)\n                    print(json.dumps(progress),flush=True)")
    replace("            audit=validate_records(all_records,samples=warmup+measured,layers=len(names))", "            audit=validate_timing(all_records,samples=completed_samples,layers=len(names))")
    replace("for sample in range(warmup,warmup+measured)]", "for sample in range(warmup,completed_samples)]")
    replace("sum(times)/len(times)", "sum(times)/len(times) if times else None")
    replace("write('completed.json',dict(completed=True", "write('timing_completed.json',dict(completed=True,target_reached=completed_samples==warmup+measured")
    replace("paired_numerics_passed=True,physical_run=not a.local_tiny,", "paired_numerics_passed=False,strict_numerics_passed=False,execution_profile='deterministic_whole_expert_fp32',physical_run=not a.local_tiny,")
    return source

if __name__=='__main__':
    from mfg_hedge import granite_g4_feedback_runner_v1 as runner
    from optimized130_port import OptimizedPlanner,optimized_baseline
    import mfg_hedge.granite_g4_feedback_planner_v1 as planner_module
    import mfg_hedge.granite_g4_feedback_replay_v1 as replay_module
    planner_module.DenseSupportPlanner=OptimizedPlanner
    replay_module.baseline_quotas=optimized_baseline
    source=build_source(inspect.getsource(runner))
    assert source.count('dense.quotas(c)')==1
    source=source.replace('dense.quotas(c)','dense.refresh(c)')
    source=source.replace("execution_profile='deterministic_whole_expert_fp32'","execution_profile='fused_planners_cuda_graph_whole_expert_fp32'")
    source=source.replace("completed_groups=completed_samples,","planner_profile='fused_cuda_graph',completed_groups=completed_samples,")
    compile(source,'deadline-timing130','exec')
    if '--selftest' in __import__('sys').argv:
        try:validate_timing([],samples=1,layers=24)
        except ValueError:pass
        else:raise AssertionError('incomplete ranks accepted')
        assert 'paired_numerics_passed=False' in source and "for arm in arms:" in source
        print('source transform and incomplete-rank rejection passed')
    else:
        out=Path('/root/mfg-g4-v100-20261013-a/optimized130')
        (out/('runner-sha-rank-'+os.environ['RANK']+'.txt')).write_text(hashlib.sha256(source.encode()).hexdigest())
        ns={'__name__':'mfg_hedge.deadline_timing130','__package__':'mfg_hedge','validate_timing':validate_timing,
            'DEADLINE':datetime.fromisoformat('2026-10-14T02:02:00+08:00').timestamp()}
        exec(compile(source,'deadline-timing130','exec'),ns)
        ns['main']()
