from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];EXT=ROOT/'supplementary_analyses';sys.path.insert(0,str(ROOT/'scripts'))
from generate_decision_architecture import load_cfg,build_direct_metric_tables,score_core
variants=['demand_high','demand_low','fiscal_adverse','fiscal_favorable']
rng=np.random.default_rng(42);W=rng.dirichlet(np.ones(4),size=200_000)
all_scores=[];ranks=[];wins=[]
for v in variants:
    vr=EXT/'_generated'/'exogenous_stress'/v
    core,_,_=build_direct_metric_tables(vr);ps,_,_=score_core(core,load_cfg(vr))
    eq=ps.groupby(['scenario','geo'],as_index=False)['score'].mean().rename(columns={'score':'equal_weight_score'});eq=eq[eq.scenario!='SC0'].copy();eq['stress_variant']=v;all_scores.append(eq)
    for geo,g in eq.groupby('geo'):
        gg=g.sort_values('equal_weight_score',ascending=False).copy();gg['rank']=range(1,len(gg)+1);ranks.append(gg)
        pg=ps[(ps.geo==geo)&(ps.scenario!='SC0')].pivot(index='scenario',columns='pillar',values='score')[['Transportation','Environmental','Social','Economic']]
        winner=np.argmax(W@pg.to_numpy().T,axis=1)
        for i,sc in enumerate(pg.index):wins.append({'stress_variant':v,'geo':geo,'scenario':sc,'win_probability':float(np.mean(winner==i))})
scores=pd.concat(all_scores,ignore_index=True);ranksdf=pd.concat(ranks,ignore_index=True);winsdf=pd.DataFrame(wins)
scores.to_csv(EXT/'exogenous_stress_equal_weight_scores.csv',index=False);ranksdf.to_csv(EXT/'exogenous_stress_rankings.csv',index=False);winsdf.to_csv(EXT/'exogenous_stress_weight_robustness.csv',index=False)
lead_eq=ranksdf[ranksdf['rank']==1][['stress_variant','geo','scenario','equal_weight_score']].rename(columns={'scenario':'equal_weight_leader'})
lead_w=winsdf.loc[winsdf.groupby(['stress_variant','geo']).win_probability.idxmax()][['stress_variant','geo','scenario','win_probability']].rename(columns={'scenario':'weight_robust_leader'})
lead=lead_eq.merge(lead_w,on=['stress_variant','geo']);lead.to_csv(EXT/'exogenous_stress_leadership_summary.csv',index=False)
acc=ranksdf.groupby(['geo','scenario']).agg(best_rank=('rank','min'),worst_rank=('rank','max'),mean_rank=('rank','mean'),n_first=('rank',lambda s:int((s==1).sum()))).reset_index();acc.to_csv(EXT/'exogenous_stress_rank_acceptability.csv',index=False)
spec={'demand_high':'City and district trip-demand paths ramp from 0% deviation in 2024 to +10% in 2030.','demand_low':'City and district trip-demand paths ramp from 0% deviation in 2024 to -10% in 2030.','fiscal_adverse':'Municipal budget, GDP and household income ramp to -20%, while transport fuel, electricity and OPEX inputs ramp to +20%, by 2030.','fiscal_favorable':'Municipal budget, GDP and household income ramp to +20%, while transport fuel, electricity and OPEX inputs ramp to -20%, by 2030.'}
(EXT/'exogenous_stress_specification.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print(lead.to_string(index=False))
