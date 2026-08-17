from __future__ import annotations
import argparse, copy, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import joblib
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'supplementary_analyses'
sys.path.insert(0,str(ROOT/'config'));sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from scenario_meta import SCENARIO_TO_OVERLAY
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe

SCENARIOS=[f'SC{i}' for i in range(12)]
VARIANTS=['demand_high','demand_low','fiscal_adverse','fiscal_favorable']

def install_cache(runner):
    cache=joblib.load(EXT/'ml_model_cache.joblib')
    for name,rec in cache['models'].items():
        agent=runner.agents[name]
        mlb=next(b for b in agent.behaviors if isinstance(b,system_core.MLBehavior))
        mlb.model=rec['model'];mlb._n_features_fit=rec['n_features_fit'];mlb.selected_features=list(rec['selected_features'])
        mlb._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])];mlb._target_transform=rec['target_transform'];mlb._is_fitted=True;mlb._warned_not_fitted=False
        agent.config.dependencies=list(rec['selected_features'])
    runner._normalize_agent_dependencies();runner.solver=system_core.SimpleSDSolver(runner.agents,delayed_selector=runner.delayed_selector)

def run_one(variant,scenario):
    rec=yaml.safe_load((EXT/'exogenous_stress_definitions'/f'{variant}.yaml').read_text())
    stress=rec['exogenous_forecast']
    cfg=system_core.load_config_from_yaml(ROOT/'config'/'BASE_CONFIG.yaml');cfg.data_file=str((ROOT/'config'/'DATA_clean.csv').resolve())
    cfg.exogenous_forecast.update(copy.deepcopy(stress))
    if scenario!='SC0':
        ov=yaml.safe_load((ROOT/'scenarios'/SCENARIO_TO_OVERLAY[scenario]).read_text())
        cfg.exogenous_forecast.update(copy.deepcopy(ov.get('exogenous_forecast',{})))
    runner=system_core.HybridSimulationRunner(cfg);install_cache(runner)
    raw=pd.read_csv(ROOT/'config'/'DATA_clean.csv')
    d=pd.DataFrame(runner.run_simulation(raw,skip_offline_train=True)['timeseries'])
    return sync_dataframe(apply_historical_cold_start_anchors(d))

def _task(arg):
    variant,sc,out=arg
    try:
        run_one(variant,sc).to_csv(out,index=False);return True,variant,sc,''
    except Exception as e:
        return False,variant,sc,repr(e)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--force',action='store_true');args=ap.parse_args()
    if not (EXT/'ml_model_cache.joblib').exists():
        import subprocess;subprocess.run([sys.executable,str(EXT/'prepare_ml_cache.py')],cwd=ROOT,check=True)
    base=EXT/'_generated'/'exogenous_stress';tasks=[]
    for v in VARIANTS:
        vr=base/v;(vr/'outputs').mkdir(parents=True,exist_ok=True);(vr/'config').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'config'/'decision_architecture.yaml',vr/'config'/'decision_architecture.yaml')
        shutil.copy2(EXT/'exogenous_stress_definitions'/f'{v}.yaml',vr/'stress_definition.yaml')
        for sc in SCENARIOS:
            out=vr/'outputs'/f'simulation_data_{sc}.csv'
            if args.force or not out.exists():tasks.append((v,sc,out))
    print(f'Exogenous stress: {len(tasks)} runs, {args.workers} workers',flush=True)
    bad=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(_task,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            ok,v,sc,err=f.result();
            if not ok: bad.append((v,sc,err))
            if j%12==0 or not ok:print(j,'/',len(futs),'fail',len(bad),flush=True)
    if bad:raise RuntimeError(bad[:3])
    print('DONE exogenous stress raw runs')

if __name__=='__main__':main()
