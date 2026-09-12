from mfg_hedge.token_mfg_qualification_campaign import QualificationPlan
from mfg_hedge.token_mfg_qualification_backend import RoutingPolicySeed
from mfg_hedge.token_mfg_streaming import StreamingEpisodeSpec
from mfg_hedge.token_mfg_streaming_formal import streaming_iteration_episode_worker, _aggregate_iteration_outputs
p=QualificationPlan.r2()
outs=[]
for i,s in enumerate(p.scenario_schedule[:2]):
 spec=StreamingEpisodeSpec(8,p.preflight_namespace,p.preflight_macro_seed,s,100+i)
 payload={'episode_spec':spec,'policy':RoutingPolicySeed('uniform'),'model_id':'unpriced_mfg','iteration':0,'mean_field_namespace':p.continuation_namespace,'mean_field_macro_seed':p.continuation_macro_seed,'finite_k_namespace':p.finite_k_namespace,'finite_k_macro_seed':p.finite_k_macro_seed}
 r=streaming_iteration_episode_worker(payload)
 outs.append((str(i),r['output']))
a=_aggregate_iteration_outputs(tuple(outs),model_id='unpriced_mfg',iteration=0,policy=RoutingPolicySeed('uniform'),panel_fingerprint='a'*64,minimum_samples=2)
print({'states':len(a.snapshot.states),'q':len(a.estimates),'finite':a.diagnostic.simultaneous_ucb_status,'calls':sum(streaming_iteration_episode_worker({'episode_spec':StreamingEpisodeSpec(8,p.preflight_namespace,p.preflight_macro_seed,p.scenario_schedule[0],200+i),'policy':RoutingPolicySeed('uniform'),'model_id':'unpriced_mfg','iteration':0,'mean_field_namespace':p.continuation_namespace,'mean_field_macro_seed':p.continuation_macro_seed,'finite_k_namespace':p.finite_k_namespace,'finite_k_macro_seed':p.finite_k_macro_seed})['completed_calls'] for i in range(1))})
