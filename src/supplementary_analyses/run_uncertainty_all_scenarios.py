from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from ensemble_engine import init_worker, run_uncertainty_task

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'supplementary_analyses'
SCENARIOS = [f'SC{i}' for i in range(12)]
HEADLINE = {
    'CO₂ emissions','Congestion index','Electricity use','Final energy','Health indicator','Modal share: car',
    'Modal share: public transport','NOₓ emissions','PM₂.₅ emissions','Time loss (car)','PCE-weighted VKT'
}



def summarize(outdir: Path) -> None:
    files = sorted(outdir.glob('draw_*_SC*.csv'))
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data = data.sort_values(['draw','scenario','geo','metric'], kind='stable').reset_index(drop=True)
    data.to_csv(EXT/'uncertainty_all_draws_SC0_SC11.csv', index=False)
    base = data[data.scenario=='SC0'].rename(columns={'mean_2024_2030':'base_mean','value_2030':'base_2030'})[
        ['draw','geo','metric','base_mean','base_2030']]
    eff = data.merge(base, on=['draw','geo','metric'], how='left')
    eff['delta_mean_pct'] = np.where(np.abs(eff.base_mean)>1e-12,
        100*(eff.mean_2024_2030-eff.base_mean)/np.abs(eff.base_mean), eff.mean_2024_2030-eff.base_mean)
    eff['delta_2030_pct'] = np.where(np.abs(eff.base_2030)>1e-12,
        100*(eff.value_2030-eff.base_2030)/np.abs(eff.base_2030), eff.value_2030-eff.base_2030)
    eff = eff.sort_values(['draw','scenario','geo','metric'], kind='stable').reset_index(drop=True)
    eff.to_csv(EXT/'uncertainty_relative_effects_SC0_SC11.csv', index=False)
    rows=[]
    for (sc,geo,metric),g in eff[eff.scenario!='SC0'].groupby(['scenario','geo','metric'], sort=True):
        for effect in ['delta_mean_pct','delta_2030_pct']:
            v=g[effect].dropna()
            med=float(v.median())
            frac=float((v>0).mean() if med>0 else (v<0).mean() if med<0 else (v==0).mean())
            p05=float(v.quantile(.05)); p25=float(v.quantile(.25)); p75=float(v.quantile(.75)); p95=float(v.quantile(.95))
            rows.append({'scenario':sc,'geo':geo,'metric':metric,'effect':effect,'n':len(v),'median':med,
                         'p05':p05,'p25':p25,'p75':p75,'p95':p95,'min':float(v.min()),'max':float(v.max()),
                         'sign_fraction':frac,'all_same_direction':bool(frac==1.0),'span90':p95-p05})
    envdf=pd.DataFrame(rows)
    envdf.to_csv(EXT/'uncertainty_envelopes_SC1_SC11.csv', index=False)
    h=envdf[(envdf.effect=='delta_mean_pct') & envdf.metric.isin(HEADLINE)]
    stab=(h.groupby(['scenario','geo'],sort=True)
          .agg(metrics=('metric','size'),stable=('all_same_direction','sum'),min_sign=('sign_fraction','min'),median_span=('span90','median'),max_span=('span90','max'))
          .reset_index())
    stab.to_csv(EXT/'gaussian_headline_stability_all_scenarios.csv', index=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--draws',type=int,default=32)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if not (EXT/'ml_model_cache.joblib').exists():
        subprocess.run([sys.executable,str(EXT/'prepare_ml_cache.py')],cwd=ROOT,check=True)
    if not (EXT/'ml_logit_residual_scales.csv').exists():
        if not (EXT/'walk_forward_predictions.csv').exists():
            subprocess.run([sys.executable,str(EXT/'run_walk_forward_validation.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(EXT/'derive_ml_residual_scales.py')],cwd=ROOT,check=True)
    scales=pd.read_csv(EXT/'ml_logit_residual_scales.csv')
    sigma=dict(zip(scales.agent,scales.rmse))
    rng=np.random.default_rng(20260721)
    draws=[{agent:float(rng.normal(0,sd)) for agent,sd in sigma.items()} for _ in range(args.draws)]
    (EXT/'uncertainty_draw_shifts.json').write_text(json.dumps(draws,indent=2),encoding='utf-8')
    outdir=EXT/'_generated'/'uncertainty_gaussian'; outdir.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for i,shifts in enumerate(draws):
        for sc in SCENARIOS:
            out=outdir/f'draw_{i:03d}_{sc}.csv'
            if args.force or not out.exists(): tasks.append((i,sc,shifts,out))
    print(f'Gaussian all-scenario uncertainty: {len(tasks)} tasks, {args.workers} workers', flush=True)
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        futs=[ex.submit(run_uncertainty_task,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            rc,i,sc,err=f.result()
            if rc: failures.append((i,sc,err))
            if j%24==0 or rc: print(f'{j}/{len(futs)} completed; failures={len(failures)}',flush=True)
    if failures: raise RuntimeError(f'Uncertainty failures: {failures[:3]}')
    summarize(outdir)
    print('DONE Gaussian uncertainty SC0-SC11')

if __name__=='__main__': main()
