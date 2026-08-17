from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from ensemble_engine import init_worker, run_uncertainty_task

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'supplementary_analyses'
SCENARIOS=[f'SC{i}' for i in range(12)]
HEADLINE={'CO₂ emissions','Congestion index','Electricity use','Final energy','Health indicator','Modal share: car','Modal share: public transport','NOₓ emissions','PM₂.₅ emissions','Time loss (car)','PCE-weighted VKT'}


def make_draws():
    p=pd.read_csv(EXT/'walk_forward_predictions.csv')
    p=p[(p.specification=='deployed') & (p.year!=2020)].copy()
    eps=1e-6
    obs=np.clip(p.truth.to_numpy(float),eps,1-eps); pred=np.clip(p.prediction.to_numpy(float),eps,1-eps)
    p['residual_logit']=np.log(obs/(1-obs))-np.log(pred/(1-pred))
    draws=[]
    for year,g in p.groupby('year',sort=True):
        vec={a:float(v) for a,v in zip(g.agent,g.residual_logit)}
        draws.append({'label':f'empirical_{int(year)}','shifts':vec})
        draws.append({'label':f'sign_reversed_{int(year)}','shifts':{a:-v for a,v in vec.items()}})
    return draws



def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    if not (EXT/'ml_model_cache.joblib').exists(): subprocess.run([sys.executable,str(EXT/'prepare_ml_cache.py')],cwd=ROOT,check=True)
    if not (EXT/'walk_forward_predictions.csv').exists(): subprocess.run([sys.executable,str(EXT/'run_walk_forward_validation.py')],cwd=ROOT,check=True)
    draws=make_draws(); (EXT/'uncertainty_block_draws.json').write_text(json.dumps(draws,indent=2),encoding='utf-8')
    outdir=EXT/'_generated'/'uncertainty_blocks';outdir.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for i,rec in enumerate(draws):
        for sc in SCENARIOS:
            out=outdir/f'block_{i:02d}_{sc}.csv'
            if args.force or not out.exists(): tasks.append((i,sc,rec['shifts'],out))
    print(f'Residual-block all-scenario stress: {len(tasks)} tasks, {args.workers} workers',flush=True)
    fail=[]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        futs=[ex.submit(run_uncertainty_task,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            rc,i,sc,err=f.result()
            if rc: fail.append((i,sc,err))
            if j%12==0 or rc: print(f'{j}/{len(futs)} completed; failures={len(fail)}',flush=True)
    if fail: raise RuntimeError(f'Block failures: {fail[:3]}')
    D=pd.concat([pd.read_csv(f) for f in sorted(outdir.glob('block_*_SC*.csv'))],ignore_index=True)
    label_map={i:r['label'] for i,r in enumerate(draws)};D['block_label']=D.draw.map(label_map)
    D=D.sort_values(['draw','scenario','geo','metric'],kind='stable').reset_index(drop=True)
    D.to_csv(EXT/'uncertainty_block_all_draws_SC0_SC11.csv',index=False)
    B=D[D.scenario=='SC0'].rename(columns={'mean_2024_2030':'base_mean','value_2030':'base_2030'})[['draw','geo','metric','base_mean','base_2030']]
    E=D.merge(B,on=['draw','geo','metric'],how='left')
    E['delta_mean_pct']=np.where(np.abs(E.base_mean)>1e-12,100*(E.mean_2024_2030-E.base_mean)/np.abs(E.base_mean),E.mean_2024_2030-E.base_mean)
    E['delta_2030_pct']=np.where(np.abs(E.base_2030)>1e-12,100*(E.value_2030-E.base_2030)/np.abs(E.base_2030),E.value_2030-E.base_2030)
    E=E.sort_values(['draw','scenario','geo','metric'],kind='stable').reset_index(drop=True)
    E.to_csv(EXT/'uncertainty_block_relative_effects_SC0_SC11.csv',index=False)
    rows=[]
    for (sc,geo,m),g in E[E.scenario!='SC0'].groupby(['scenario','geo','metric'],sort=True):
        for effect in ['delta_mean_pct','delta_2030_pct']:
            v=g[effect].dropna();med=float(v.median());frac=float((v>0).mean() if med>0 else (v<0).mean() if med<0 else (v==0).mean())
            p05=float(v.quantile(.05));p25=float(v.quantile(.25));p75=float(v.quantile(.75));p95=float(v.quantile(.95))
            rows.append({'scenario':sc,'geo':geo,'metric':m,'effect':effect,'n':len(v),'median':med,'p05':p05,'p25':p25,'p75':p75,'p95':p95,'min':float(v.min()),'max':float(v.max()),'sign_fraction':frac,'all_same_direction':bool(frac==1),'span90':p95-p05})
    env=pd.DataFrame(rows);env.to_csv(EXT/'uncertainty_block_envelopes_SC1_SC11.csv',index=False)
    h=env[(env.effect=='delta_mean_pct') & env.metric.isin(HEADLINE)]
    stab=(h.groupby(['scenario','geo'],sort=True).agg(metrics=('metric','size'),stable=('all_same_direction','sum'),min_sign=('sign_fraction','min'),median_span=('span90','median'),max_span=('span90','max')).reset_index())
    stab.to_csv(EXT/'block_headline_stability_all_scenarios.csv',index=False)
    print('DONE residual-block uncertainty SC0-SC11')

if __name__=='__main__':main()
