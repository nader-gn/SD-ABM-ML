from __future__ import annotations
import sys
from pathlib import Path
import joblib, pandas as pd, yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config')); sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from scenario_meta import SCENARIO_TO_OVERLAY
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe
from extract_manuscript_results import build_rows, load_region12_budget_share

METRICS=['Modal share: public transport','Time loss (car)','PCE-weighted VKT','CO₂ emissions','PM₂.₅ emissions']


def make_runner(scenario:str, multiplier:float|None):
    cfg=system_core.load_config_from_yaml(ROOT/'config/BASE_CONFIG.yaml'); cfg.data_file=str((ROOT/'config/DATA_clean.csv').resolve())
    if scenario!='SC0':
        ov=yaml.safe_load((ROOT/'scenarios'/SCENARIO_TO_OVERLAY[scenario]).read_text()); cfg.exogenous_forecast.update(ov.get('exogenous_forecast',{}))
    base=float(cfg.agents['coef_k_speed'].initial_value)
    if multiplier is not None: cfg.agents['coef_k_speed'].initial_value=base*float(multiplier)
    runner=system_core.HybridSimulationRunner(cfg); cache=joblib.load(ROOT/'supplementary_analyses/ml_model_cache.joblib')
    for name,rec in cache['models'].items():
        ag=runner.agents[name]; mlb=next(b for b in ag.behaviors if isinstance(b,system_core.MLBehavior))
        mlb.model=rec['model']; mlb._n_features_fit=rec['n_features_fit']; mlb.selected_features=list(rec['selected_features']); mlb._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])]; mlb._target_transform=rec['target_transform']; mlb._is_fitted=True; ag.config.dependencies=list(rec['selected_features'])
    runner._normalize_agent_dependencies(); runner.solver=system_core.SimpleSDSolver(runner.agents,delayed_selector=runner.delayed_selector)
    return runner,base


def run(scenario,multiplier):
    runner,base=make_runner(scenario,multiplier); raw=pd.read_csv(ROOT/'config/DATA_clean.csv')
    d=pd.DataFrame(runner.run_simulation(raw,skip_offline_train=True)['timeseries']); d=sync_dataframe(apply_historical_cold_start_anchors(d))
    k=pd.DataFrame(build_rows(d,load_region12_budget_share(ROOT))); return k[(k.year>=2024)&(k.year<=2030)],base


def main():
    k0,base=run('SC0',None); b={(g,k):v.value.mean() for (g,k),v in k0.groupby(['Geo','kpi'])}
    rows=[]
    for mult in [0.5,0.75,1.0,1.25,1.5]:
        k,_=run('SC5',mult)
        for geo in ['Tehran','Region 12']:
            for metric in METRICS:
                g=k[(k.Geo==geo)&(k.kpi==metric)]
                mean=float(g.value.mean()); end=float(g.loc[g.year==2030,'value'].iloc[0]); base_mean=float(b[(geo,metric)])
                rows.append({'coef_k_speed_multiplier':mult,'coef_k_speed':base*mult,'geo':geo,'metric':metric,'mean_2024_2030':mean,'value_2030':end,'delta_vs_SC0_pct':100*(mean-base_mean)/abs(base_mean)})
    out=pd.DataFrame(rows); out.to_csv(ROOT/'supplementary_analyses/sc5_speed_coefficient_sensitivity.csv',index=False)
    print(out[(out.geo=='Tehran')&(out.metric=='Modal share: public transport')].to_string(index=False))

if __name__=='__main__': main()
