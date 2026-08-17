from __future__ import annotations
import copy,sys
from pathlib import Path
import joblib,pandas as pd,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config'));sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from scenario_meta import SCENARIO_TO_OVERLAY
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe


def install(r):
 c=joblib.load(ROOT/'supplementary_analyses'/'ml_model_cache.joblib')
 for n,rec in c['models'].items():
  a=r.agents[n];m=next(b for b in a.behaviors if isinstance(b,system_core.MLBehavior));m.model=rec['model'];m._n_features_fit=rec['n_features_fit'];m.selected_features=list(rec['selected_features']);m._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])];m._target_transform=rec['target_transform'];m._is_fitted=True;m._warned_not_fitted=False;a.config.dependencies=list(rec['selected_features'])
 r._normalize_agent_dependencies();r.solver=system_core.SimpleSDSolver(r.agents,delayed_selector=r.delayed_selector)

def run(sc):
 cfg=system_core.load_config_from_yaml(ROOT/'config'/'BASE_CONFIG.yaml');cfg.data_file=str((ROOT/'config'/'DATA_clean.csv').resolve())
 if sc!='SC0':
  ov=yaml.safe_load(open(ROOT/'scenarios'/SCENARIO_TO_OVERLAY[sc]));cfg.exogenous_forecast.update(copy.deepcopy(ov.get('exogenous_forecast',{})))
 r=system_core.HybridSimulationRunner(cfg);install(r);raw=pd.read_csv(ROOT/'config'/'DATA_clean.csv');d=pd.DataFrame(r.run_simulation(raw,skip_offline_train=True)['timeseries']);return sync_dataframe(apply_historical_cold_start_anchors(d))

def main():
 rows=[]
 for i in range(12):
  sc=f'SC{i}';new=run(sc);old=pd.read_csv(ROOT/'outputs'/f'simulation_data_{sc}.csv')
  common=[c for c in old.columns if c in new.columns]
  a=old[common].apply(pd.to_numeric,errors='coerce').to_numpy(float);b=new[common].apply(pd.to_numeric,errors='coerce').to_numpy(float)
  mask=np.isfinite(a)&np.isfinite(b);diff=np.abs(a-b);scale=np.maximum(np.abs(a),1.0);rel=np.where(mask,diff/scale,np.nan)
  maxabs=float(np.nanmax(np.where(mask,diff,np.nan)));maxrel=float(np.nanmax(rel));
  loc=np.unravel_index(np.nanargmax(np.where(mask,diff,np.nan)),diff.shape);col=common[loc[1]];year=float(old.iloc[loc[0]].get('YEAR_GRG',np.nan))
  rows.append({'scenario':sc,'rows':len(old),'common_columns':len(common),'max_absolute_difference':maxabs,'max_relative_difference':maxrel,'max_abs_column':col,'max_abs_year':year,'passes_relative_1e-10':bool(maxrel<1e-10)})
 pd.DataFrame(rows).to_csv(ROOT/'supplementary_analyses'/'central_output_replay.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
