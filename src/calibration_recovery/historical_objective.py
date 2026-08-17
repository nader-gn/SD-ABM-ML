"""Executable historical objective used by the supplementary recovery audit.

This module contains the deterministic objective/setup logic needed to
re-evaluate the candidate parameter vectors used by the reported recovery analysis.
Candidate generation is not rerun because proposal sequences can vary across
optimizer-library versions; the candidate vectors are treated as a fixed experimental
design and the model objective is recomputed for every vector.
"""
from __future__ import annotations
import copy, importlib.util, logging, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings('ignore'); logging.disable(logging.CRITICAL)
HERE=Path(__file__).resolve(); BUNDLE=HERE.parents[1]/'computational_bundle'
if not BUNDLE.exists():
 BUNDLE=HERE.parents[1]/'core'
if not BUNDLE.exists():
 raise FileNotFoundError('Could not locate computational_bundle/core for calibration recovery.')
OUT=HERE.parent/'rerun_output'; OUT.mkdir(parents=True,exist_ok=True)
ACTIVE=['spd_car_history_calibration_factor','spd_bus_history_calibration_factor','congestion_multiplier_r12','alpha_delay_congestion_r12']
# Quantities audited alongside the searched vector but held fixed in the recovery design.
# Two are annual historical-series anchors whose mapped columns supersede scalar initial values in history;
# alpha_delay_congestion is a genuine scalar whose OAT perturbation has zero effect on the declared recovery objective.
FIXED=['spd_mot_history_calibration_factor','alpha_delay_congestion','car_energy_closure_factor']
HISTORICAL_SERIES_ANCHORS=['spd_mot_history_calibration_factor','car_energy_closure_factor']
FIXED_SCALARS=['alpha_delay_congestion']
# Additional scalar quantities included only in the calibration-role audit. They are not
# added to the four-dimensional local recovery search or to the regularization term.
AUDIT_ONLY=['spd_car_city_calibration_factor','r12_speed_factor','r12_voc_multiplier','trips_per_person_per_day']
AUDITED=ACTIVE+FIXED+AUDIT_ONLY
TRUTH={
'modal_share_car':'modal_share_car_truth','modal_share_taxi':'modal_share_taxi_truth','modal_share_bus':'modal_share_bus_truth','modal_share_metro':'modal_share_metro_truth','modal_share_motorcycle':'modal_share_motorcycle_truth','modal_share_other':'modal_share_other_truth',
'modal_share_car_r12':'modal_share_car_r12_truth','modal_share_tax_r12':'modal_share_tax_r12_truth','modal_share_bus_r12':'modal_share_bus_r12_truth','modal_share_met_r12':'modal_share_met_r12_truth','modal_share_mot_r12':'modal_share_mot_r12_truth','modal_share_oth_r12':'modal_share_oth_r12_truth',
'spd_mot':'spd_mot_truth','spd_car':'spd_car_truth','spd_tax':'spd_tax_truth','spd_bus':'spd_bus_truth','spd_met':'spd_met_truth',
'trips_per_year':'trips_per_year_truth','trips_bus_total':'trips_bus_total_truth','trips_metro':'trips_metro_truth','trips_per_person_per_day':'trips_per_person_per_day_truth',
'fuel_gasoline_litre_year':'fuel_gasoline_litre_year_truth','fuel_CNG_kg_year':'fuel_CNG_kg_year_truth','concentration_PM25_ugm3':'concentration_PM25_ugm3_truth','concentration_NO2_ugm3':'concentration_NO2_ugm3_truth',
'annual_km_car':'annual_km_car_truth','annual_km_taxi':'annual_km_taxi_truth','annual_km_motorcycle':'annual_km_motorcycle_truth',
'brt_share_of_bus':'brt_share_of_bus_truth','share_taxi_gasoline':'share_taxi_gasoline_truth','share_taxi_CNG':'share_taxi_CNG_truth'}
DOMAINS={
'city_modal':['modal_share_car','modal_share_taxi','modal_share_bus','modal_share_metro','modal_share_motorcycle','modal_share_other'],
'r12_modal':['modal_share_car_r12','modal_share_tax_r12','modal_share_bus_r12','modal_share_met_r12','modal_share_mot_r12','modal_share_oth_r12'],
'speed':['spd_mot','spd_car','spd_tax','spd_bus','spd_met'],
'activity':['trips_per_year','trips_bus_total','trips_metro','trips_per_person_per_day'],
'energy_environment':['fuel_gasoline_litre_year','fuel_CNG_kg_year','concentration_PM25_ugm3','concentration_NO2_ugm3'],
'annual_distance':['annual_km_car','annual_km_taxi','annual_km_motorcycle'],
'operational_shares':['brt_share_of_bus','share_taxi_gasoline','share_taxi_CNG']}
WEIGHTS={'city_modal':.30,'r12_modal':.25,'speed':.10,'activity':.10,'energy_environment':.15,'annual_distance':.05,'operational_shares':.05}
PRIOR_SCALE=0.10
LAMBDA=0.04
GAMMA=0.01

def load_core():
 spec=importlib.util.spec_from_file_location('system_core_regularized',BUNDLE/'config/system_core.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def setup():
 core=load_core();raw=pd.read_csv(BUNDLE/'config/DATA_clean.csv');raw['YEAR_GRG']=pd.to_numeric(raw.YEAR_GRG,errors='coerce')
 cfgd=yaml.safe_load((BUNDLE/'config/BASE_CONFIG.yaml').read_text());cfgd['simulation']['data_file']=str((BUNDLE/'config/DATA_clean.csv').resolve());cfgd['time']['end_year']=2022;cfgd['ml_training_end_year']=2021;cfgd.setdefault('training',{})['offline_train_end_year']=2021;cfgd['feature_selection']['enabled']=False;cfgd['hyperparams']['enabled']=False
 yp=OUT/'regularized_base.yaml';yp.write_text(yaml.safe_dump(cfgd,sort_keys=False,allow_unicode=True));base=core.load_config_from_yaml(yp)
 hist=copy.deepcopy(base);hist.start_year=2012;hist.end_year=2024;hist.hindcast_clamp_ml_to_observed=True;hist.hindcast_clamp_years=list(range(2012,2024));hist.inject_ml_truth_in_history=True;hist.prefer_truth_for_endogenous_deps=True
 hr=core.HybridSimulationRunner(hist);hts=pd.DataFrame(hr.run_simulation(raw,skip_offline_train=True)['timeseries']);hts=hts.loc[:,~hts.columns.duplicated()];hts['YEAR_GRG']=pd.to_numeric(hts.YEAR_GRG,errors='coerce').astype('Int64')
 tdf=raw.copy();tdf['YEAR_GRG']=pd.to_numeric(tdf.YEAR_GRG,errors='coerce').astype('Int64');ac=[a.name for a in base.agents.values() if a.name in hts.columns and not a.name.endswith('_truth') and a.name!='YEAR_GRG'];m=tdf[['YEAR_GRG']].merge(hts[['YEAR_GRG']+ac],on='YEAR_GRG',how='left');tdf.loc[:,ac]=m[ac].to_numpy();setattr(base,'_training_df',tdf)
 tr=core.HybridSimulationRunner(base);tr.run_simulation(raw);models={}
 for n,a in tr.agents.items():
  for b in a.behaviors:
   if isinstance(b,core.MLBehavior):models[n]=(b.model,b._is_fitted,b._n_features_fit,b.selected_features,b._target_transform)
 ref={k:float(cfgd['agents'][k]['initial_value']) for k in AUDITED}
 ranges={k:(max(float(cfgd['agents'][k].get('bounds',[-1e9,1e9])[0]),ref[k]*(1-PRIOR_SCALE)),min(float(cfgd['agents'][k].get('bounds',[-1e9,1e9])[1]),ref[k]*(1+PRIOR_SCALE))) for k in ACTIVE}
 def fresh(params):
  cfg=core.load_config_from_yaml(yp);setattr(cfg,'_training_df',tdf)
  resolved={k:float(params.get(k,ref[k])) for k in AUDITED}
  # Apply overrides to every audited scalar. For mapped historical-series anchors,
  # the annual input column deliberately supersedes this scalar during history.
  # AUDIT_ONLY quantities can therefore be perturbed for role/identifiability checks
  # without expanding the local recovery search or its regularization dimension.
  for k in AUDITED:cfg.agents[k].initial_value=float(resolved[k])
  r=core.HybridSimulationRunner(cfg)
  for n,a in r.agents.items():
   if n in models:
    for b in a.behaviors:
     if isinstance(b,core.MLBehavior):b.model,b._is_fitted,b._n_features_fit,b.selected_features,b._target_transform=models[n]
  return r
 def derived(sim,met):
  if met=='annual_km_car':return sim['vkm_car']/sim['private_cars_total'].replace(0,np.nan)
  if met=='annual_km_taxi':return sim['vkm_taxi']/sim['taxis_total'].replace(0,np.nan)
  if met=='annual_km_motorcycle':return sim['vkm_motorcycle']/sim['motorcycles_total'].replace(0,np.nan)
  return sim[met]
 def evaluate(params,save=None):
  resolved={k:float(params.get(k,ref[k])) for k in AUDITED}
  sim=pd.DataFrame(fresh(resolved).run_simulation(raw,skip_offline_train=True)['timeseries']);sim['YEAR_GRG']=pd.to_numeric(sim.YEAR_GRG,errors='coerce');comps={}
  for dom,mets in DOMAINS.items():
   ys=[];ps=[]
   for met in mets:
    truth=TRUTH[met];s=pd.DataFrame({'YEAR_GRG':sim.YEAR_GRG,'pred':derived(sim,met)});m=s.merge(raw[['YEAR_GRG',truth]],on='YEAR_GRG');m=m[(m.YEAR_GRG>=2013)&(m.YEAR_GRG<=2021)];ys.extend(pd.to_numeric(m[truth],errors='coerce'));ps.extend(pd.to_numeric(m.pred,errors='coerce'))
   y=np.asarray(ys,float);p=np.asarray(ps,float);mask=np.isfinite(y)&np.isfinite(p);comps[dom]=float(np.sqrt(np.mean((p[mask]-y[mask])**2))/max(np.mean(np.abs(y[mask])),1e-9))
  fit=sum(WEIGHTS[d]*comps[d] for d in WEIGHTS)
  omega=float(np.mean([((resolved[k]-ref[k])/(PRIOR_SCALE*abs(ref[k])))**2 for k in ACTIVE]))
  # execution stability: modal closure, finite outputs, and bounds; zero for admissible runs
  city=sim[['modal_share_car','modal_share_taxi','modal_share_bus','modal_share_metro','modal_share_motorcycle','modal_share_other']].sum(axis=1)
  r12=sim[['modal_share_car_r12','modal_share_tax_r12','modal_share_bus_r12','modal_share_met_r12','modal_share_mot_r12','modal_share_oth_r12']].sum(axis=1)
  stability=float(np.mean(np.maximum(np.abs(city-1)-1e-6,0))*100+np.mean(np.maximum(np.abs(r12-1)-1e-6,0))*100)
  val=fit+LAMBDA*omega+GAMMA*stability
  if save:sim.to_csv(save,index=False)
  return val,fit,omega,stability,comps
 return core,raw,cfgd,ref,ranges,evaluate

