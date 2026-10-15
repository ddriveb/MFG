import math,random

RATIOS=(.4,.6,.8)

def build_support(P,seed):
    if P not in (32,64):raise ValueError('This route-mass study uses32/64Experts')
    hot=tuple(sorted(random.Random(seed).sample(range(P),8)))
    cold=[e for e in range(P) if e not in hot]
    support=tuple(tuple(sorted(hot+tuple(random.Random(seed*10000+s).sample(cold,8)))) for s in range(P))
    return support,hot

def generate(support,hot,seed,rho,P):
    if rho not in RATIOS:raise ValueError('Use the frozen40/60/80% route-mass levels')
    if len(support)!=P or len(hot)!=8 or len(set(hot))!=8:raise ValueError('Invalid globalhot layout')
    hot=set(hot);target=round(rho*512);counts=[];patterns=[]
    for s,candidates in enumerate(support):
        if len(candidates)!=16 or len(set(candidates))!=16 or not hot<=set(candidates):raise ValueError('Each source requires8hot+8coldcandidates')
        rng=random.Random(seed*10000+s)
        weights=[math.exp(random.Random(seed*1000+e+500).gauss(0,.2)+rng.gauss(0,.15)) for e in candidates]
        indices=list(range(16));rng.shuffle(indices);pp=[indices[:8],indices[8:]]
        q,remainder=divmod(target-8,62);levels=[q+int(i<remainder) for i in range(62)];rng.shuffle(levels)
        hot_indices=[j for j,e in enumerate(candidates) if e in hot];cold_indices=[j for j,e in enumerate(candidates) if e not in hot]
        for number in levels:
            def choose(pool,k):
                return sorted(pool,key=lambda j:math.log(max(rng.random(),1e-300))/weights[j],reverse=True)[:k]
            pp.append(choose(hot_indices,number)+choose(cold_indices,8-number))
        cc=[16*sum(j in group for group in pp) for j in range(16)]
        assert len(pp)==64 and all(len(set(v))==8 for v in pp)
        assert sum(cc)==8192 and min(cc)>0 and max(cc)<=1024
        assert sum(n for e,n in zip(candidates,cc) if e in hot)==target*16
        counts.extend(cc);patterns.append([[candidates[j] for j in group] for group in pp])
    return counts,patterns,target/512
