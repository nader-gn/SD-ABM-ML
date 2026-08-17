"""Persistent-worker engine for supplementary full-loop ensembles.

Each worker imports the simulator and loads the fitted ML cache once. This avoids
spawning a fresh interpreter and reloading the model cache for every draw/scenario.
The full-loop task implementation is centralized here so every ensemble run uses the same execution path.
"""
from __future__ import annotations
import copy, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config'));sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from scenario_meta import SCENARIO_TO_OVERLAY
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe
from extract_manuscript_results import build_rows, load_region12_budget_share

KEEP_KPIS={
 'Modal share: public transport','Modal share: car','PCE-weighted VKT','Congestion index','Time loss (car)',
 'CO₂ emissions','NOₓ emissions','PM₂.₅ emissions','Health indicator','Final energy','Electricity use','PT OPEX',
 'Net recurrent public burden','Energy cost','PT affordability ratio','Noise exposure index'}
STOCKS=['inflation_index_effective','cost_index_rhc_fare','cost_index_bus_fare','cost_index_metro_fare',
 'len_bik','len_brt','len_hwy','len_met','population_city','private_cars_total','taxis_total','metro_cars_total','brts_total',
 'buses_total','motorcycles_total','transport_financial_balance_IRR']
_CACHE=None;_RAW=None;_BUDGET=None;_BASE_CFG=None;_OVERLAYS=None

def init_worker():
 global _CACHE,_RAW,_BUDGET,_BASE_CFG,_OVERLAYS
 _CACHE=joblib.load(ROOT/'supplementary_analyses'/'ml_model_cache.joblib')
 _RAW=pd.read_csv(ROOT/'config'/'DATA_clean.csv')
 _BUDGET=load_region12_budget_share(ROOT)
 _BASE_CFG=system_core.load_config_from_yaml(ROOT/'config'/'BASE_CONFIG.yaml');_BASE_CFG.data_file=str((ROOT/'config'/'DATA_clean.csv').resolve())
 _OVERLAYS={}
 for i in range(1,12):
  sc=f'SC{i}';_OVERLAYS[sc]=yaml.safe_load((ROOT/'scenarios'/SCENARIO_TO_OVERLAY[sc]).read_text()).get('exogenous_forecast',{})

def _runner(sc,shifts):
 cfg=copy.deepcopy(_BASE_CFG)
 if sc!='SC0':cfg.exogenous_forecast.update(copy.deepcopy(_OVERLAYS[sc]))
 for name,shift in shifts.items():cfg.agents[name].hyperparameters['prediction_target_scale_shift']=float(shift)
 r=system_core.HybridSimulationRunner(cfg)
 for name,rec in _CACHE['models'].items():
  a=r.agents[name];m=next(b for b in a.behaviors if isinstance(b,system_core.MLBehavior))
  m.model=rec['model'];m._n_features_fit=rec['n_features_fit'];m.selected_features=list(rec['selected_features']);m._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])];m._target_transform=rec['target_transform'];m._is_fitted=True;m._warned_not_fitted=False;a.config.dependencies=list(rec['selected_features'])
 r._normalize_agent_dependencies();r.solver=system_core.SimpleSDSolver(r.agents,delayed_selector=r.delayed_selector)
 return r

def run_uncertainty_task(task):
 draw,sc,shifts,out=task
 try:
  r=_runner(sc,shifts);df=pd.DataFrame(r.run_simulation(_RAW,skip_offline_train=True)['timeseries']);df=sync_dataframe(apply_historical_cold_start_anchors(df))
  rows=pd.DataFrame(build_rows(df,_BUDGET));rows=rows[(rows.year>=2024)&(rows.year<=2030)&rows.kpi.isin(KEEP_KPIS)]
  recs=[]
  for (geo,kpi),g in rows.groupby(['Geo','kpi']):
   recs.append({'draw':draw,'scenario':sc,'geo':geo,'metric':kpi,'mean_2024_2030':float(g.value.mean()),'value_2030':float(g.loc[g.year==2030,'value'].iloc[0])})
  dp=df[(df.YEAR_GRG>=2024)&(df.YEAR_GRG<=2030)]
  for stock in STOCKS:
   if stock in dp.columns:recs.append({'draw':draw,'scenario':sc,'geo':'Tehran','metric':'STOCK::'+stock,'mean_2024_2030':float(dp[stock].mean()),'value_2030':float(dp.loc[dp.YEAR_GRG.round().astype(int)==2030,stock].iloc[0])})
  out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(recs).to_csv(out,index=False)
  return 0,draw,sc,''
 except Exception as e:return 1,draw,sc,repr(e)
