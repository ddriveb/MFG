"""Read-only Fit diagnosis; bounded replays of existing fitting traces only."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import statistics as st

from mfg_hedge.common_state import phase_at
from mfg_hedge.campaign_execution import _BatchRuleSource
from mfg_hedge.expert_game import TaggedExpertEpisodeRun, score_expert, shared_trace_fingerprint
from mfg_hedge.game_workload import enumerate_action_rules
from mfg_hedge.shared_backup import prepare_shared_trace, simulate_shared_backup_tagged
from mfg_hedge.stage4_campaign import build_frozen_stage4_scenarios

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/stage4-pi256-fit-validation-20260906-r2'
OUTPUT = ROOT / 'artifacts/stage4-fit-variance-diagnosis-20260906'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def se(values):
    n = len(values)
    avg = st.mean(values)
    return math.sqrt((n-1)/n * math.fsum((v-avg)**2 for v in values))

def covariance(a,b):
    return math.fsum((x-st.mean(a))*(y-st.mean(b)) for x,y in zip(a,b))

def components(score):
    result = {'phase_'+p: v/4 for p,v in score.phase_losses.items()}
    result.update({'tail_'+p: score.cvar95[p]/2 for p in ('D','F')})
    result.update({k: score.components[k] for k in
                   ('executed_work_per_token','wasted_work_per_token')})
    assert math.isclose(sum(result.values()),score.total,abs_tol=1e-10)
    return result

def influence(vals,cluster_ids):
    avg = st.mean(vals)
    denom = sum((v-avg)**2 for v in vals)
    return sorted([{'common_id':cluster_ids[i], 'delete_one_delta':v,
                    'variance_share':(v-avg)**2/denom if denom else 0}
                   for i,v in enumerate(vals)],key=lambda r:-r['variance_share'])

def tail_details(runs):
    answer = {}
    for phase in ('D','F'):
        values=[]
        for n,run in enumerate(runs):
            for t in run.result.tokens:
                if phase_at(run.trace.timeline,t.arrival_time).value==phase:
                    values.append((t.latency,n//2,n%2,t.local_token_id))
        values.sort(reverse=True)
        mass=len(values)/20
        left=mass; members=[]
        for latency,common,pop,local in values:
            if left <= 1e-12: break
            weight=min(1,left); left-=weight
            members.append(dict(latency=latency,common_id=common,population=pop,
                                local_token_id=local,weight=weight))
        weighted=sum(m['latency']*m['weight'] for m in members)
        shares=defaultdict(float)
        for m in members: shares[m['common_id']]+=m['weight']/mass
        answer[phase]=dict(count=len(values),tail_mass=mass,cvar=weighted/mass,
                           cluster_mass_shares=dict(shares),members=members)
    return answer

def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    source_hashes={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()}
    fit=json.loads((SOURCE/'fit.json').read_text())
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    # Only the fitting namespace is generated; no validation workload/evaluation.
    scenarios=build_frozen_stage4_scenarios('shared-backup-game:v1:stage4-fit',
                                         20260905,common_path_count=16)
    assert [shared_trace_fingerprint(s.trace) for s in scenarios] == manifest['fit_trace_fingerprints']
    sorted_faults=sorted({s.fault_fingerprint for s in scenarios})
    id_by_fault={s.fault_fingerprint:i//2 for i,s in enumerate(scenarios)}
    cluster_ids=[id_by_fault[k] for k in sorted_faults]
    iterations=[]; checked=0
    for start in fit['starts']:
        for it in start['iterations']:
            for row in it['rule_rows']:
                assert len(row['paired_delete_one_deltas'])==16
                assert math.isclose(se(row['paired_delete_one_deltas']),row['paired_jackknife_se'],abs_tol=1e-12)
                checked+=1
            current=next(r for r in it['rule_rows'] if r['rule']==it['current_rule'])
            assert all(v==0 for v in current['paired_delete_one_deltas'])
            worst=max(it['rule_rows'],key=lambda r:r['paired_jackknife_se'])
            ranked=sorted(it['rule_rows'],key=lambda r:r['objective'])
            gaps=it['selected_runner_up_delete_one_deltas']
            assert math.isclose(se(gaps),it['selected_jackknife_se'],abs_tol=1e-12)
            # The baseline cancels, so a bank can be re-ranked after each deletion.
            omitted_winners=[min(it['rule_rows'],key=lambda r:r['paired_delete_one_deltas'][j])['rule']
                             for j in range(16)]
            iterations.append(dict(start=start['start_rule'],current=it['current_rule'],
                selected=it['selected_rule'],runner_up=it['runner_up_rule'],
                gap=it['selected_gap'],gap_se=it['selected_jackknife_se'],
                max_rule=worst['rule'],max_se=worst['paired_jackknife_se'],
                target=it['precision_target'],
                below_target=sum(r['paired_jackknife_se']<=it['precision_target'] for r in ranked),
                gap_influence=influence(gaps,cluster_ids),
                max_influence=influence(worst['paired_delete_one_deltas'],cluster_ids),
                omitted_winners=dict(zip(cluster_ids,omitted_winners)),top10=ranked[:10]))
    assert checked==1536
    print('Verified all1536 SE values and all32 original Fit fingerprints',flush=True)
    rules={r.name:r for r in enumerate_action_rules()}
    prepared=[prepare_shared_trace(s.trace) for s in scenarios]
    pairs=[('NSNS','NSNS'),('NSNS','NSSS'),('NSNS','NSND'),('NSNS','NNDX'),
           ('NNNN','NNNN'),('NNNN','NSNS')]
    profiles={}; raw={}; calls=0
    for incumbent,candidate in pairs:
        runs=[]
        for i,s in enumerate(scenarios):
            policy={e:_BatchRuleSource(rules[incumbent]) for e in range(8)}
            policy[0]=_BatchRuleSource(rules[candidate])
            result=simulate_shared_backup_tagged(prepared[i],.5,tagged_expert_id=0,
                                                hedge_delay=1.5,action_sources=policy)
            calls+=1
            assert calls<=256
            runs.append(TaggedExpertEpisodeRun(s.episode_key,s.trace,result,s.fault_fingerprint))
        score=score_expert(runs,0)
        target_it=next(it for start in fit['starts'] for it in start['iterations'] if it['current_rule']==incumbent)
        source_row=next(row for row in target_it['rule_rows'] if row['rule']==candidate)
        assert math.isclose(score.total,source_row['objective'],abs_tol=1e-10)
        loo=[]
        for fault in sorted_faults:
            retained=[run for run in runs if run.fault_fingerprint!=fault]
            value=score_expert(retained,0)
            loo.append(dict(total=value.total,components=components(value)))
        profiles[incumbent+'/'+candidate]=dict(total=score.total,components=components(score),
                    score=score.as_dict(),delete_one=loo,tails=tail_details(runs))
        raw[incumbent+'/'+candidate]=[]
        for n,run in enumerate(runs):
            raw[incumbent+'/'+candidate].append(dict(common_id=n//2,population=n%2,
                failed_start=run.trace.timeline.failed_start,
                recovered_start=run.trace.timeline.recovered_start,
                tokens=[asdict(t) for t in run.result.tokens],
                actions=[asdict(a) for a in run.result.action_decisions],
                pool=asdict(run.result.pool_audit_summary)))
        print(f'Replayed {incumbent}/{candidate}, J={score.total:.9f}, diagnostic calls={calls}',flush=True)
    contrasts={}
    for incumbent,candidate in pairs:
        a=profiles[incumbent+'/'+candidate]; b=profiles[incumbent+'/'+incumbent]
        source_it=next(it for start in fit['starts'] for it in start['iterations'] if it['current_rule']==incumbent)
        source_row=next(row for row in source_it['rule_rows'] if row['rule']==candidate)
        differences=[x['total']-y['total'] for x,y in zip(a['delete_one'],b['delete_one'])]
        assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(differences,source_row['paired_delete_one_deltas']))
        comp={k:[x['components'][k]-y['components'][k] for x,y in zip(a['delete_one'],b['delete_one'])]
              for k in a['components']}
        var=covariance(differences,differences)
        contrasts[incumbent+'/'+candidate]=dict(delta=a['total']-b['total'],se=se(differences),
            component_deltas={k:a['components'][k]-b['components'][k] for k in comp},
            component_se={k:se(v) for k,v in comp.items()},
            variance_covariance_shares={k:covariance(v,differences)/var if var else 0 for k,v in comp.items()},
            loo_components=comp,influence=influence(differences,cluster_ids),
            unpaired_se=math.hypot(se([x['total'] for x in a['delete_one']]),se([x['total'] for x in b['delete_one']])))
    # Raw cohort/tail counts reveal the effective sample and concentration.
    counts=[]
    for n,s in enumerate(scenarios):
        cells=Counter((phase_at(s.trace.timeline,t.arrival_time).value,t.token_class.value)
                      for t in s.trace.tokens if t.expert_id==0)
        counts.append(dict(common_id=n//2,population=n%2,fault=s.fault_fingerprint,
            failed_start=s.trace.timeline.failed_start,recovered_start=s.trace.timeline.recovered_start,
            cohorts={p+'_'+k:v for (p,k),v in cells.items()}))
    assert source_hashes == {p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()}
    summary=dict(source_hashes=source_hashes,checked_rows=checked,cluster_order=cluster_ids,
                 diagnostic_calls=calls,formal_fit_calls_added=0,validation_calls=0,
                 iterations=iterations,profiles=profiles,contrasts=contrasts,counts=counts,
                 claim='Retrospective fitting-data diagnosis only; no campaign result changed')
    OUTPUT.mkdir()
    (OUTPUT/'diagnosis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUTPUT/'tagged-replay-rows.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    print('Complete: original hashes unchanged; diagnostic calls',calls,flush=True)

if __name__=='__main__': main()
