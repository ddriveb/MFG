import random, math

def generate(support,seed,family,P):
    strength=random.Random(seed).uniform(1,2.5)
    hot=set(sorted(range(P),key=lambda e:random.Random(seed*1000+e).random())[:P//4])
    counts=[];patterns=[]
    for s,candidates in enumerate(support):
        rng=random.Random(seed*10000+s);source_hot=set(rng.sample(candidates,4));weights=[]
        for e in candidates:
            z=random.Random(seed*1000+e+500).gauss(0,.2)+rng.gauss(0,.15)
            if family=='global_hotspot' and e in hot:z+=strength
            if family=='node_local' and e//8==s//8:z+=strength
            if family=='source_skew' and e in source_hot:z+=strength
            weights.append(math.exp(z))
        indices=list(range(16));rng.shuffle(indices);pp=[indices[:8],indices[8:]];cc=[16]*16
        for _ in range(62):
            keys=[math.log(max(rng.random(),1e-300))/w for w in weights];chosen=sorted(range(16),key=keys.__getitem__,reverse=True)[:8]
            for j in chosen:cc[j]+=16
            pp.append(chosen)
        assert len(pp)==64 and all(len(set(v))==8 for v in pp)
        assert [16*sum(j in p for p in pp) for j in range(16)]==cc and sum(cc)==8192 and min(cc)>0
        counts.extend(cc);patterns.append([[candidates[j] for j in p] for p in pp])
    return counts,patterns,strength
