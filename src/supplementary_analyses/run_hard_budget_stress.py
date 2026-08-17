from __future__ import annotations
import copy, sys
from pathlib import Path
import joblib, pandas as pd, numpy as np, yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'config')); sys.path.insert(0,str(ROOT/'scripts'))
import system_core
from run_all_scenarios import apply_historical_cold_start_anchors
from recompute_time_loss_sync import sync_dataframe
from extract_manuscript_results import build_rows, load_region12_budget_share

YEARS=list(range(2024,2031))
SERVICE_VARS=['spd_bus_freeflow','spd_met','len_bus','len_brt','len_met','bus_purchases_vehicles_year','brt_purchases_vehicles_year','metro_car_purchases_vehicles_year']


def install_cache(runner):
    cache=joblib.load(ROOT/'supplementary_analyses'/'ml_model_cache.joblib')
    for name,rec in cache['models'].items():
        agent=runner.agents[name]; mlb=next(b for b in agent.behaviors if isinstance(b,system_core.MLBehavior))
        mlb.model=rec['model']; mlb._n_features_fit=rec['n_features_fit']; mlb.selected_features=list(rec['selected_features'])
        mlb._active_dep_specs_for_online=[tuple(x) for x in rec.get('active_dep_specs',[])]
        mlb._target_transform=rec['target_transform']; mlb._is_fitted=True; mlb._warned_not_fitted=False
        agent.config.dependencies=list(rec['selected_features'])
    runner._normalize_agent_dependencies(); runner.solver=system_core.SimpleSDSolver(runner.agents,delayed_selector=runner.delayed_selector)


def run_with_overlay(overlay):
    cfg=system_core.load_config_from_yaml(ROOT/'config'/'BASE_CONFIG.yaml'); cfg.data_file=str((ROOT/'config'/'DATA_clean.csv').resolve())
    cfg.exogenous_forecast.update(copy.deepcopy(overlay))
    runner=system_core.HybridSimulationRunner(cfg); install_cache(runner)
    raw=pd.read_csv(ROOT/'config'/'DATA_clean.csv')
    res=runner.run_simulation(raw,skip_offline_train=True)
    return sync_dataframe(apply_historical_cold_start_anchors(pd.DataFrame(res['timeseries'])))


def base_path(base_yaml,var):
    if var in base_yaml['exogenous_forecast']:
        d=base_yaml['exogenous_forecast'][var]; return {y:float(d.get(y,d.get(str(y)))) for y in YEARS}
    init=float(base_yaml['agents'][var].get('initial_value',0.0)); return {y:init for y in YEARS}


def main():
    base_yaml=yaml.safe_load(open(ROOT/'config'/'BASE_CONFIG.yaml'))
    sc10_yaml=yaml.safe_load(open(ROOT/'scenarios'/'SC10_pt_first_clean_package.yaml'))
    std=copy.deepcopy(sc10_yaml['exogenous_forecast'])
    bases={v:base_path(base_yaml,v) for v in SERVICE_VARS}
    ratios={y:0.0 for y in YEARS}
    adjusted=copy.deepcopy(std)
    last=None
    for it in range(10):
        factors={2024:1.0}
        for y in YEARS[1:]: factors[y]=min(1.0,1.0/max(float(ratios[y-1]),1e-12)) if ratios[y-1]>1 else 1.0
        for v in SERVICE_VARS:
            adjusted[v]={}
            for y in YEARS:
                sv=float(std[v][y]); bv=float(bases[v][y]); adjusted[v][y]=bv+factors[y]*(sv-bv)
        d=run_with_overlay(adjusted)
        proj=d[d.YEAR_GRG.round().astype(int).isin(YEARS)].copy(); proj['year']=proj.YEAR_GRG.round().astype(int)
        new={int(r.year):float(r.pt_cost_to_effective_budget_ratio) for _,r in proj.iterrows()}
        if last is not None and max(abs(new[y]-last[y]) for y in YEARS)<1e-10: break
        last=new; ratios=new
    # final run with converged prior-year factors
    factors={2024:1.0}
    for y in YEARS[1:]: factors[y]=min(1.0,1.0/max(float(ratios[y-1]),1e-12)) if ratios[y-1]>1 else 1.0
    for v in SERVICE_VARS:
        adjusted[v]={y:float(bases[v][y])+factors[y]*(float(std[v][y])-float(bases[v][y])) for y in YEARS}
    hard=run_with_overlay(adjusted)
    hard.to_csv(ROOT/'supplementary_analyses'/'SC10_hard_budget_simulation.csv',index=False)
    final_yaml={'name':'SC10 hard-budget realization stress','description':'All incremental SC10 public-transport service, network, and fleet paths are attenuated from year t by min(1,1/r_{t-1}), where r is the prior-year PT-cost/effective-budget ratio. Fare, pollutant-control and electrification channels are left unchanged.','exogenous_forecast':adjusted,'realization_factor':factors}
    with open(ROOT/'supplementary_analyses'/'SC10_hard_budget_stress.yaml','w') as f: yaml.safe_dump(final_yaml,f,sort_keys=False)
    # trajectories
    std_df=pd.read_csv(ROOT/'outputs'/'simulation_data_SC10.csv'); sc0_df=pd.read_csv(ROOT/'outputs'/'simulation_data_SC0.csv')
    cols=['YEAR_GRG','pt_cost_to_effective_budget_ratio']+SERVICE_VARS+['modal_share_PT','time_loss_car_hours_year']
    h=hard[hard.YEAR_GRG.round().astype(int).isin(YEARS)][cols].copy(); s=std_df[std_df.YEAR_GRG.round().astype(int).isin(YEARS)][cols].copy()
    tr=s.merge(h,on='YEAR_GRG',suffixes=('_standard','_hard'))
    tr['realization_factor']=tr.YEAR_GRG.round().astype(int).map(factors)
    tr.to_csv(ROOT/'supplementary_analyses'/'SC10_hard_budget_trajectory.csv',index=False)
    # KPI comparison
    share=load_region12_budget_share(ROOT)
    kh=pd.DataFrame(build_rows(hard,share)); ks=pd.DataFrame(build_rows(std_df,share)); kb=pd.DataFrame(build_rows(sc0_df,share))
    out=[]
    common=sorted(set(kh.kpi)&set(ks.kpi)&set(kb.kpi))
    for geo in ['Tehran','Region12','Region 12']:
        gh=kh[kh.Geo==geo]; gs=ks[ks.Geo==geo]; gb=kb[kb.Geo==geo]
        if gh.empty: continue
        gout='Region 12' if geo in ['Region12','Region 12'] else geo
        for k in common:
            ah=gh[(gh.kpi==k)&gh.year.between(2024,2030)]; ast=gs[(gs.kpi==k)&gs.year.between(2024,2030)]; ab=gb[(gb.kpi==k)&gb.year.between(2024,2030)]
            if ah.empty or ast.empty or ab.empty: continue
            hm,sm,bm=ah.value.mean(),ast.value.mean(),ab.value.mean(); he=ah.loc[ah.year==2030,'value'].iloc[0]; se=ast.loc[ast.year==2030,'value'].iloc[0]; be=ab.loc[ab.year==2030,'value'].iloc[0]
            pct=lambda a,b:100*(a/b-1) if abs(b)>1e-12 else np.nan
            out.append({'geo':gout,'kpi':k,'hard_mean':hm,'hard_2030':he,'standard_mean':sm,'sc0_mean':bm,'standard_2030':se,'sc0_2030':be,'hard_vs_standard_mean_pct':pct(hm,sm),'standard_vs_sc0_mean_pct':pct(sm,bm),'hard_vs_sc0_mean_pct':pct(hm,bm),'hard_vs_standard_2030_pct':pct(he,se),'standard_vs_sc0_2030_pct':pct(se,be),'hard_vs_sc0_2030_pct':pct(he,be)})
    pd.DataFrame(out).to_csv(ROOT/'supplementary_analyses'/'SC10_hard_budget_comparison.csv',index=False)
    print(tr[['YEAR_GRG','realization_factor','pt_cost_to_effective_budget_ratio_standard','pt_cost_to_effective_budget_ratio_hard','spd_bus_freeflow_standard','spd_bus_freeflow_hard','spd_met_standard','spd_met_hard','modal_share_PT_standard','modal_share_PT_hard']].to_string(index=False))

if __name__=='__main__': main()
