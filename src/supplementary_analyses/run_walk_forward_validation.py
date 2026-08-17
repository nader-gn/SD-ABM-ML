from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config'))
import system_core


def run_one(config_path:Path, out_dir:Path):
    cfg=system_core.load_config_from_yaml(config_path)
    # This table is the exact leakage-safe, reconstructed historical feature-state dataset
    # used for the pooled non-COVID validation summary. All rows are annual and end in 2023.
    data=pd.read_csv(ROOT/'supplementary_analyses'/'historical_feature_dataset.csv')
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    system_core.run_walk_forward_one_step(cfg,data,out_dir,year_col='YEAR_GRG',min_train_years=6)


def collect(folder:Path, spec:str):
    rec=[]
    for p in sorted(folder.glob('wf_*.json')):
        j=json.load(open(p,encoding='utf-8'))
        agent=p.stem.replace('wf_','')
        for y,t,pred in zip(j['year'],j['y_true'],j['y_pred']):
            rec.append({'specification':spec,'agent':agent,'year':int(y),'truth':float(t),'prediction':float(pred)})
    return rec


def pooled(pred:pd.DataFrame):
    import numpy as np
    rows=[]
    for (spec,geo),g in pred.assign(geo=lambda x:x.agent.str.contains('_r12_').map({True:'Region 12',False:'Tehran'})).groupby(['specification','geo']):
        for subset,gg in [('all',g),('main_non_covid',g[g.year!=2020]),('covid_stress',g[g.year==2020])]:
            y=gg.truth.to_numpy(float); ph=gg.prediction.to_numpy(float)
            if len(y)==0: continue
            err=y-ph
            mae=float(np.mean(np.abs(err))*100)
            rmse=float(np.sqrt(np.mean(err**2))*100)
            den=float(np.sum((y-y.mean())**2))
            r2=float(1-np.sum(err**2)/den) if den>0 else float('nan')
            rows.append({'specification':spec,'geo':geo,'subset':subset,'n':len(y),'R2':r2,'MAE_pp':mae,'RMSE_pp':rmse})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    full=ROOT/'supplementary_analyses'/'walk_forward_full'
    red=ROOT/'supplementary_analyses'/'walk_forward_reduced'
    run_one(ROOT/'config'/'BASE_CONFIG.yaml',full)
    run_one(ROOT/'supplementary_analyses'/'config_reduced_features.yaml',red)
    rec=collect(full,'deployed')+collect(red,'reduced_feature')
    df=pd.DataFrame(rec)
    df.to_csv(ROOT/'supplementary_analyses'/'walk_forward_predictions.csv',index=False)
    pooled(df).to_csv(ROOT/'supplementary_analyses'/'walk_forward_pooled_summary.csv',index=False)
    # per mode non-COVID
    import numpy as np
    rr=[]
    for (spec,agent),g in df[df.year!=2020].groupby(['specification','agent']):
        y=g.truth.to_numpy(float); ph=g.prediction.to_numpy(float); err=y-ph; den=np.sum((y-y.mean())**2)
        rr.append({'specification':spec,'agent':agent,'geo':'Region 12' if '_r12_' in agent else 'Tehran','n':len(y),
                   'R2':float(1-np.sum(err**2)/den) if den>0 else float('nan'),'MAE_pp':float(np.mean(np.abs(err))*100),'RMSE_pp':float(np.sqrt(np.mean(err**2))*100)})
    pd.DataFrame(rr).to_csv(ROOT/'supplementary_analyses'/'walk_forward_per_mode_non_covid.csv',index=False)
    print(pooled(df).to_string(index=False))
if __name__=='__main__': main()
