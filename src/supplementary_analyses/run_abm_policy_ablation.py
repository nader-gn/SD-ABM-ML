from __future__ import annotations
import copy,sys
from pathlib import Path
import joblib,pandas as pd,yaml,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config'));sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from scenario_meta import SCENARIO_TO_OVERLAY
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe
from extract_manuscript_results import build_rows,load_region12_budget_share
SC=['SC0','SC2','SC5','SC8','SC10']
COEFS=['coef_k_fee','coef_k_park','coef_k_commute','coef_k_fare','coef_k_speed','coef_k_fuel']

def install(r):
 c=joblib.load(ROOT/'supplementary_analyses'/'ml_model_cache.joblib')
 for n,rec in c['models'].items():
  a=r.agents[n];m=next(b for b in a.behaviors if isinstance(b,system_core.MLBehavior));m.model=rec['model'];m._n_features_fit=rec['n_features_fit'];m.selected_features=list(rec['selected_features']);m._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])];m._target_transform=rec['target_transform'];m._is_fitted=True;m._warned_not_fitted=False;a.config.dependencies=list(rec['selected_features'])
 r._normalize_agent_dependencies();r.solver=system_core.SimpleSDSolver(r.agents,delayed_selector=r.delayed_selector)

def run(sc,ablate):
 cfg=system_core.load_config_from_yaml(ROOT/'config'/'BASE_CONFIG.yaml');cfg.data_file=str((ROOT/'config'/'DATA_clean.csv').resolve())
 if ablate:
  for c in COEFS: cfg.agents[c].initial_value=0.0; cfg.exogenous_forecast[c]={y:0.0 for y in range(2024,2031)}
 if sc!='SC0':
  ov=yaml.safe_load(open(ROOT/'scenarios'/SCENARIO_TO_OVERLAY[sc]));cfg.exogenous_forecast.update(copy.deepcopy(ov.get('exogenous_forecast',{})))
 r=system_core.HybridSimulationRunner(cfg);install(r);raw=pd.read_csv(ROOT/'config'/'DATA_clean.csv');d=pd.DataFrame(r.run_simulation(raw,skip_offline_train=True)['timeseries']);return sync_dataframe(apply_historical_cold_start_anchors(d))

def main():
 rows=[];share=load_region12_budget_share(ROOT)
 for ab in [False,True]:
  label='policy_mediation_active' if not ab else 'policy_response_coefficients_zero'
  for sc in SC:
   d=run(sc,ab);k=pd.DataFrame(build_rows(d,share));k=k[k.year.between(2024,2030)]
   for geo in ['Tehran','Region 12']:
    for metric in ['Modal share: public transport','Modal share: car','Time loss (car)','PCE-weighted VKT']:
     g=k[(k.Geo==geo)&(k.kpi==metric)];
     if not g.empty: rows.append({'configuration':label,'scenario':sc,'geo':geo,'metric':metric,'mean_2024_2030':g.value.mean(),'value_2030':g.loc[g.year==2030,'value'].iloc[0]})
 out=pd.DataFrame(rows);base=out[out.scenario=='SC0'][['configuration','geo','metric','mean_2024_2030','value_2030']].rename(columns={'mean_2024_2030':'base_mean','value_2030':'base_2030'})
 out=out.merge(base,on=['configuration','geo','metric']);out['delta_mean_pct']=np.where(out.base_mean.abs()>1e-12,100*(out.mean_2024_2030/out.base_mean-1),np.nan);out['delta_2030_pct']=np.where(out.base_2030.abs()>1e-12,100*(out.value_2030/out.base_2030-1),np.nan)
 out.to_csv(ROOT/'supplementary_analyses'/'abm_policy_ablation.csv',index=False)
 comp=out[out.scenario!='SC0'].pivot_table(index=['scenario','geo','metric'],columns='configuration',values='delta_mean_pct').reset_index();comp['mediation_contribution_pp']=comp['policy_mediation_active']-comp['policy_response_coefficients_zero'];comp.to_csv(ROOT/'supplementary_analyses'/'abm_policy_ablation_comparison.csv',index=False)
 print(comp.to_string(index=False))
if __name__=='__main__':main()
