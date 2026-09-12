"""Bounded synthetic reproductions of review findings; no production edits."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import math
import random

from mfg_hedge.common_state import CommonStateTimeline, Phase
from mfg_hedge.domain import TokenClass
from mfg_hedge.game_deviations import DeviationScenario, _run_profile
from mfg_hedge.game_workload import ActionRule
from mfg_hedge.shared_backup import (
    SharedAction as A, SharedBackupTrace, SharedTokenSpec, SharedWorkDraw,
    simulate_shared_backup, simulate_shared_backup_optimized,
)
from mfg_hedge.stage4_campaign import (
    compute_simultaneous_regret_bound, jackknife_pseudo_values, run_parallel_fit,
)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/mfg-readiness-review-20260906'

def make_trace(arrivals=(110.,), destination=2, work=None):
    default=SharedWorkDraw((10.,10.,10.),(1.,1.,1.),(1.,8.,.25),(1.,1.,5.))
    return SharedBackupTrace(1,CommonStateTimeline(100,200,220),360.,
        tuple(SharedTokenSpec(i,t,TokenClass.REGULAR,0,i,destination)
              for i,t in enumerate(arrivals)),
        tuple(work or default for _ in arrivals))

class Constant:
    def __init__(self,action): self.action=action
    def request(self,observation): return self.action

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    files=list((ROOT/'src/mfg_hedge').glob('*.py'))+list((ROOT/'configs').glob('*.json'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    findings={};calls=0
    trace=make_trace()
    single=simulate_shared_backup(trace,2.,action_sources={0:Constant(A.SINGLE)});calls+=1
    dual=simulate_shared_backup(trace,2.,action_sources={0:Constant(A.DUAL)});calls+=1
    s_copy=next(a for a in single.attempts if a.replica_id==2)
    x_copy=next(a for a in dual.attempts if a.replica_id==2)
    assert (s_copy.attempt_id,s_copy.required_work)==(2,.25)
    assert (x_copy.attempt_id,x_copy.required_work)==(3,5.)
    assert single.tokens[0].latency==.25 and dual.tokens[0].latency==5.
    findings['dual_crn_mapping']={'single_C':[s_copy.attempt_id,s_copy.required_work],
        'dual_C':[x_copy.attempt_id,x_copy.required_work],
        'single_latency':single.tokens[0].latency,'dual_latency':dual.tokens[0].latency,
        'capacity':2.,'destination':2}

    class Stateful:
        name='SSSS'
        def __init__(self): self.decisions=0
        def request(self,observation):
            self.decisions+=1
            return A.SINGLE if self.decisions==1 else A.NORMAL
    policy=Stateful()
    scenarios=(DeviationScenario('review-a',trace,'review-fault'),
               DeviationScenario('review-b',trace,'review-fault'))
    runs=_run_profile(scenarios,{0:policy},c_b=2.,degraded_slowdown=2.,hedge_delay=1.5)
    calls+=2
    actions=[r.result.tokens[0].action.value for r in runs]
    assert actions==['S','N']
    findings['policy_state_leak']={'identical_episode_actions':actions,
                                   'policy_total_calls':policy.decisions}

    class TimerObserver:
        def __init__(self): self.observations=[]
        def request(self,observation):
            self.observations.append(observation)
            return A.DELAYED if observation.local_token_id==0 else A.NORMAL
    observer=TimerObserver()
    short=SharedWorkDraw((.1,.1,.1),(.1,.1,.1),(.1,.1,.1),(.1,.1,.1))
    result=simulate_shared_backup(make_trace((110.,112.),work=short),2.,
                                  action_sources={0:observer});calls+=1
    assert result.tokens[0].timer_voided and not result.tokens[0].timer_fired
    assert observer.observations[1].timer_history==((0,'pending'),)
    findings['void_timer_observation']={'at':112.,'history':observer.observations[1].timer_history,
                                        'actual_timer_voided':result.tokens[0].timer_voided}

    missing=jackknife_pseudo_values(.1,(.1,None)+(.1,)*30)
    assert missing==()
    try:
        compute_simultaneous_regret_bound({(0,'NNNN'):0.,(0,'NSNS'):.1},
            {(0,'NNNN'):(0.,)*32,(0,'NSNS'):missing},targets={0:1.})
    except ValueError as exc:
        findings['missing_inference_row']={'raised':str(exc),
            'expected_disposition':'statistics_insufficient'}
    else: raise AssertionError('Expected current empty-row composition failure')

    # For an upper limit on theta, the pivot is (theta-hat_theta)/SE.
    # Bootstrap that pivot using (pseudo_mean-bootstrap_mean)/SE.
    skew=(-31.,)+(1.,)*31
    observed={(0,'NSNS'):0.,(0,'NNNN'):0.}
    rows={(0,'NSNS'):skew,(0,'NNNN'):(0.,)*32}
    bound=compute_simultaneous_regret_bound(observed,rows,targets={0:1.5})
    rng=random.Random(20260906); upper_pivots=[]
    for _ in range(4096):
        resampled=[skew[rng.randrange(32)] for _ in range(32)]
        # sample SE of pseudo mean equals1 in this fixture
        upper_pivots.append(max(0.,-sum(resampled)/32))
    upper_pivots.sort(); correct_q=upper_pivots[math.ceil(.95*4096)-1]
    assert bound.status=='pass' and bound.simultaneous_upper_bound==1.
    assert correct_q==2.
    findings['one_sided_bootstrap_direction']={'pseudo_values':skew,
        'implemented_upper':bound.simultaneous_upper_bound,'implemented_status':bound.status,
        'upper_pivot_quantile':correct_q,'target':1.5,
        'note':'Synthetic skew-direction check, not a coverage theorem'}

    # Baseline partial-failure accounting; the second attempted call fails.
    baseline_attempts=[]
    def fail_second(*args,**kwargs):
        baseline_attempts.append(1)
        if len(baseline_attempts)==2: raise RuntimeError('injected baseline failure')
        return simulate_shared_backup_optimized(*args,**kwargs)
    with patch('mfg_hedge.stage4_campaign.simulate_shared_backup_optimized',side_effect=fail_second):
        fit=run_parallel_fit(scenarios,expert_count=1,max_rounds=1,max_workers=1)
    calls+=2
    assert fit.status=='execution_failed' and fit.call_count==0
    assert len(baseline_attempts)==2
    findings['baseline_failure_accounting']={'scheduler_invocations':2,
          'successful_calls':1,'reported_calls':fit.call_count,'status':fit.status}
    assert calls<=32
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    OUT.mkdir()
    (OUT/'probes.json').write_text(json.dumps({'findings':findings,
        'scheduler_invocations':calls,'formal_fit_calls':0,'validation_calls':0,
        'source_and_config_hashes':hashes,'hashes_unchanged':True},indent=2),encoding='utf-8')
    for name,value in findings.items(): print(name,json.dumps(value),flush=True)
    print('All six issues reproduced; scheduler invocations',calls,'including1 injected failure')

if __name__=='__main__': main()
