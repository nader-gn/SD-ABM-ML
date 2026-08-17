#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

def csv_compare(gen:Path, ref:Path, atol=1e-10, rtol=1e-10, sort_cols=None, columns=None):
    if not gen.exists(): return False,{"reason":"missing_generated"}
    a=pd.read_csv(gen); b=pd.read_csv(ref)
    if columns is not None:
        missing=[c for c in columns if c not in a.columns or c not in b.columns]
        if missing:return False,{"reason":f"missing_columns:{missing}"}
        a=a[columns].copy();b=b[columns].copy()
    else:
        common=[c for c in b.columns if c in a.columns]
        # generated table must at least reproduce all reference columns unless explicitly scoped
        if len(common)!=len(b.columns):
            return False,{"reason":f"column_mismatch generated={list(a.columns)} reference={list(b.columns)}"}
        a=a[common];b=b[common]
    if sort_cols:
        a=a.sort_values(sort_cols,kind='stable').reset_index(drop=True);b=b.sort_values(sort_cols,kind='stable').reset_index(drop=True)
    if a.shape!=b.shape:return False,{"reason":f"shape {a.shape} != {b.shape}"}
    max_abs=0.0;max_rel=0.0
    for c in b.columns:
        an=pd.to_numeric(a[c],errors='coerce');bn=pd.to_numeric(b[c],errors='coerce')
        numeric=(pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_numeric_dtype(b[c]) or (an.notna()|bn.notna()).sum() >= max(1,int(.8*len(b))))
        if numeric:
            av=an.to_numpy(float);bv=bn.to_numpy(float)
            nan_eq=np.isnan(av)&np.isnan(bv); mask=~nan_eq
            if np.any(np.isnan(av[mask]) != np.isnan(bv[mask])):return False,{"reason":f"nan_mismatch:{c}"}
            finite=mask & np.isfinite(av)&np.isfinite(bv)
            if finite.any():
                d=np.abs(av[finite]-bv[finite]); denom=np.maximum(np.abs(bv[finite]),1.0)
                max_abs=max(max_abs,float(d.max(initial=0)));max_rel=max(max_rel,float((d/denom).max(initial=0)))
                if not np.allclose(av[finite],bv[finite],atol=atol,rtol=rtol):
                    return False,{"reason":f"numeric_mismatch:{c}","max_abs":max_abs,"max_rel":max_rel}
        else:
            aa=a[c].fillna('<NA>').astype(str).tolist();bb=b[c].fillna('<NA>').astype(str).tolist()
            if aa!=bb:return False,{"reason":f"text_mismatch:{c}"}
    return True,{"max_abs":max_abs,"max_rel":max_rel,"rows":len(a),"cols":len(a.columns)}

def json_compare(gen,ref):
    if not gen.exists():return False,{"reason":"missing_generated"}
    a=json.loads(gen.read_text());b=json.loads(ref.read_text())
    # canonical JSON comparison; float values are deterministic in these draw-definition files.
    ok=a==b
    return ok,{"reason":"ok" if ok else "json_mismatch"}


def format_file_check(path: Path, kind: str):
    if not path.exists():
        return False, {"reason": "missing_generated"}
    data = path.read_bytes()
    if kind == "svg":
        head = data[:4096].decode("utf-8", errors="ignore").lower()
        ok = "<svg" in head and len(data) > 1000
    else:
        ok = len(data) > 0
    return ok, {"bytes": len(data)}


def exact_file_compare(gen: Path, saved: Path):
    if not gen.exists():
        return False, {"reason": "missing_generated"}
    if not saved.exists():
        return False, {"reason": "missing_saved_snapshot"}
    a = gen.read_bytes()
    b = saved.read_bytes()
    return a == b, {
        "generated_bytes": len(a),
        "saved_bytes": len(b),
        "reason": "exact_match" if a == b else "byte_mismatch",
    }

def add(rows,group,name,kind,gen,ref,critical=True,**kw):
    if kind=='csv':ok,detail=csv_compare(gen,ref,**kw)
    elif kind=='json':ok,detail=json_compare(gen,ref)
    elif kind in {'svg','file'}: ok,detail=format_file_check(gen,kind)
    elif kind=='exact': ok,detail=exact_file_compare(gen,ref)
    else:raise ValueError(kind)
    rows.append({"group":group,"check":name,"kind":kind,"critical":critical,"status":"PASS" if ok else "FAIL","generated":str(gen),"reference":str(ref),"detail":json.dumps(detail,ensure_ascii=False,sort_keys=True)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    repo=Path(args.repo).resolve(); work=repo/'reproduced'/'workspace'/'computational_bundle';cal=repo/'reproduced'/'workspace'/'calibration_recovery'/'rerun_output';ref=repo/'reference';ver=repo/'reproduced'/'verification';ver.mkdir(parents=True,exist_ok=True)
    rows=[]
    # 1) Central scenario outputs are the strongest reference-output test.
    for i in range(12):
        f=f'simulation_data_SC{i}.csv';add(rows,'core','central_'+f,'csv',work/'outputs'/f,ref/'core'/'outputs'/f,sort_cols=['YEAR_GRG'],atol=1e-10,rtol=1e-10)
    # Reported machine-readable outputs.
    for generated_name, reference_name, sort in [
      ('kpi_timeseries_selected_long_2024_2030.csv','kpi_timeseries_selected_long_2024_2030.csv',['Scenario','kpi','Geo','year']),
      ('kpi_selected_mean_2024_2030.csv','kpi_selected_mean_2024_2030.csv',['scenario','geo','kpi']),
      ('kpi_selected_2030.csv','kpi_selected_2030.csv',['scenario','geo','kpi']),
      ('Figure_11_equal_weight_scores.csv','Figure_11_equal_weight_scores.csv',['scenario','geo']),
      ('Figure_11_rank_acceptability.csv','Figure_11_rank_acceptability.csv',['scenario','geo']),
      ('Figure_10_outcome_lens_scores.csv','Figure_10_outcome_lens_scores.csv',['scenario','geo','pillar']),
      ('Figure_12_implementation_scores.csv','Figure_12_implementation_scores.csv',['scenario','geo'])]:
        add(rows,'core',reference_name,'csv',work/'outputs'/generated_name,ref/'core'/'outputs'/reference_name,sort_cols=sort,atol=1e-10,rtol=1e-10)
    for generated_name, reference_name, sort in [
      ('Figure_04_modal_share_series.csv','Figure_04_modal_share_series.csv',['area','mode','series','scenario','year']),
      ('Figure_09_auc_input.csv','Figure_09_auc_input.csv',['scenario','geo','kpi'])]:
        add(rows,'core',reference_name,'csv',work/'figure_inputs'/generated_name,ref/'core'/'figure_inputs'/reference_name,sort_cols=sort,atol=1e-10,rtol=1e-10)
    # Core publication figures: SVG-only output validation.
    for number in range(4, 13):
        name = f"Figure {number}.svg"
        add(rows, 'figures', name, 'svg', work / 'figures' / name, Path(), critical=True)
    # Core verification evidence used in paper-level checks.
    for f,sort in [
      ('Table_05_validation_pooled.csv',['split','geo']),('Table_05_validation_per_mode.csv',['split','geo','mode']),
      ('agent_truth_validation_holdout_2022_2023.csv',['agent']),('ml_agent_truth_validation_holdout_2022_2023.csv',['agent']),
      ('historical_reconstruction_2013_2023.csv',['metric']),('logic_sanity_checks.csv',['check']),
      ('config_lint_checks.csv',['check']),('logic_audit_checks.csv',['check'])]:
        gp=work/'verification'/f;rp=ref/'core'/'verification'/f
        if rp.exists():add(rows,'core_validation',f,'csv',gp,rp,sort_cols=sort,atol=1e-10,rtol=1e-10)

    ext=work/'supplementary_analyses'; rr=ref/'supplementary'
    # Rolling-origin validation.
    for f,sort in [('walk_forward_predictions.csv',['specification','agent','year']),('walk_forward_pooled_summary.csv',['specification','geo','subset']),('walk_forward_per_mode_non_covid.csv',['specification','agent']),('ml_logit_residual_scales.csv',['agent'])]:
        add(rows,'supplementary_validation',f,'csv',ext/f,rr/'validation'/f,sort_cols=sort,atol=1e-12,rtol=1e-12)
    # Full all-scenario uncertainty.
    for f,sort in [
      ('uncertainty_all_draws_SC0_SC11.csv',['draw','scenario','geo','metric']),
      ('uncertainty_relative_effects_SC0_SC11.csv',['draw','scenario','geo','metric']),
      ('uncertainty_envelopes_SC1_SC11.csv',['scenario','geo','metric','effect']),
      ('gaussian_headline_stability_all_scenarios.csv',['scenario','geo']),
      ('uncertainty_block_all_draws_SC0_SC11.csv',['draw','scenario','geo','metric']),
      ('uncertainty_block_relative_effects_SC0_SC11.csv',['draw','scenario','geo','metric']),
      ('uncertainty_block_envelopes_SC1_SC11.csv',['scenario','geo','metric','effect']),
      ('block_headline_stability_all_scenarios.csv',['scenario','geo'])]:
        add(rows,'supplementary_uncertainty',f,'csv',ext/f,rr/'uncertainty'/f,sort_cols=sort,atol=2e-10,rtol=2e-10)
    for f in ['uncertainty_draw_shifts.json','uncertainty_block_draws.json']:
        add(rows,'supplementary_uncertainty',f,'json',ext/f,rr/'uncertainty'/f)
    # Mechanism / feasibility stress tests.
    for f,sort in [
      ('abm_policy_ablation.csv',['configuration','scenario','geo','metric']),('abm_policy_ablation_comparison.csv',['scenario','geo','metric']),
      ('sc5_speed_coefficient_sensitivity.csv',None),('SC5_diesel_accounting_sensitivity.csv',None),
      ('sc6_r12_emission_factor_scaling_sensitivity.csv',None),('SC6_R12_PM25_decomposition.csv',None)]:
        add(rows,'supplementary_mechanisms',f,'csv',ext/f,rr/'mechanisms'/f,sort_cols=sort,atol=2e-10,rtol=2e-10)
    for f,sort in [('SC10_hard_budget_simulation.csv',['YEAR_GRG']),('SC10_hard_budget_trajectory.csv',['YEAR_GRG']),('SC10_hard_budget_comparison.csv',['geo','kpi'])]:
        add(rows,'supplementary_hard_budget',f,'csv',ext/f,rr/'hard_budget'/f,sort_cols=sort,atol=2e-10,rtol=2e-10)
    # Exogenous stress scoring.
    for f,sort in [('exogenous_stress_equal_weight_scores.csv',['stress_variant','geo','scenario']),('exogenous_stress_rankings.csv',['stress_variant','geo','rank']),('exogenous_stress_weight_robustness.csv',['stress_variant','geo','scenario']),('exogenous_stress_leadership_summary.csv',['stress_variant','geo']),('exogenous_stress_rank_acceptability.csv',['geo','scenario'])]:
        add(rows,'supplementary_exogenous',f,'csv',ext/f,rr/'exogenous_stress'/f,sort_cols=sort,atol=2e-10,rtol=2e-10)
    add(rows,'supplementary_exogenous','exogenous_stress_specification.json','json',ext/'exogenous_stress_specification.json',rr/'exogenous_stress'/'exogenous_stress_specification.json')
    # Independent central replay itself.
    add(rows,'supplementary','central_output_replay.csv','csv',ext/'central_output_replay.csv',rr/'central_output_replay.csv',sort_cols=['scenario'],atol=1e-10,rtol=1e-10)

    # Calibration recovery: compare the reported recovery evidence.
    calref=rr/'calibration'
    add(rows,'calibration','identifiability_screen.csv','csv',cal/'identifiability_screen.csv',calref/'identifiability_screen.csv',sort_cols=['parameter'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration','lambda_sensitivity.csv','csv',cal/'lambda_sensitivity.csv',calref/'lambda_sensitivity.csv',sort_cols=['lambda'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration','gamma_sensitivity.csv','csv',cal/'gamma_sensitivity.csv',calref/'gamma_sensitivity.csv',sort_cols=['gamma'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration','recovered_parameter_vector_identity.csv','csv',cal/'recovered_parameter_vector_identity.csv',calref/'recovered_parameter_vector_identity.csv',sort_cols=['parameter'],atol=2e-10,rtol=2e-10)
    # Compare the fixed-specification historical NRMSE column used by the recovery objective.
    add(rows,'calibration','objective_metric_nrmse_2013_2021.csv','csv',cal/'historical_metric_nrmse_2013_2021.csv',calref/'objective_metric_nrmse_2013_2021.csv',sort_cols=['metric'],columns=['metric','nrmse_frozen'],atol=2e-10,rtol=2e-10)
    for f in sorted(calref.glob('oat_*.csv')):
        add(rows,'calibration',f.name,'csv',cal/f.name,f,sort_cols=['pct_from_frozen'],atol=2e-10,rtol=2e-10)
    # Candidate parameter vectors are preserved; objective values are recomputed through the historical objective.
    add(rows,'calibration','optuna_tpe_complete_trials.csv','csv',cal/'optuna_tpe_complete_trials.csv',calref/'optuna_tpe_complete_trials.csv',sort_cols=['number'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration','optuna_random_independent_candidates.csv','csv',cal/'optuna_random_independent_candidates.csv',calref/'optuna_random_independent_candidates.csv',sort_cols=['study_file','number'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration_fit','historical_calibration_fit_metrics_2013_2021.csv','csv',cal/'historical_calibration_fit_metrics_2013_2021.csv',calref/'historical_calibration_fit_metrics_2013_2021.csv',sort_cols=['domain','metric'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration_fit','historical_calibration_domain_fit_2013_2021.csv','csv',cal/'historical_calibration_domain_fit_2013_2021.csv',calref/'historical_calibration_domain_fit_2013_2021.csv',sort_cols=['domain'],atol=2e-10,rtol=2e-10)
    add(rows,'calibration_fit','historical_calibration_fit_summary.json','json',cal/'historical_calibration_fit_summary.json',calref/'historical_calibration_fit_summary.json')
    add(rows,'ml_propagation_validation','ml_propagation_validation_metrics.csv','csv',cal/'ml_propagation_validation_metrics.csv',calref/'ml_propagation_validation_metrics.csv',sort_cols=['group','agent'],atol=2e-10,rtol=2e-10)
    add(rows,'ml_propagation_validation','ml_propagation_validation_group_summary.csv','csv',cal/'ml_propagation_validation_group_summary.csv',calref/'ml_propagation_validation_group_summary.csv',sort_cols=['group'],atol=2e-10,rtol=2e-10)
    add(rows,'ml_propagation_validation','ml_modal_layer_validation_pooled.csv','csv',cal/'ml_modal_layer_validation_pooled.csv',calref/'ml_modal_layer_validation_pooled.csv',sort_cols=['split','geography'],atol=2e-10,rtol=2e-10)
    add(rows,'ml_propagation_validation','ml_propagation_validation_scope.csv','csv',cal/'ml_propagation_validation_scope.csv',calref/'ml_propagation_validation_scope.csv',sort_cols=['scope'],atol=2e-10,rtol=2e-10)
    add(rows,'ml_propagation_validation','ml_propagation_conditional_anchors.csv','csv',cal/'ml_propagation_conditional_anchors.csv',calref/'ml_propagation_conditional_anchors.csv',sort_cols=['agent'],atol=2e-10,rtol=2e-10)
    add(rows,'ml_propagation_validation','ml_propagation_validation_summary.json','json',cal/'ml_propagation_validation_summary.json',calref/'ml_propagation_validation_summary.json')

    # Publication-output snapshot: the saved GitHub artifacts must be reproduced path-for-path and byte-for-byte.
    paper_out = repo / 'reproduced' / 'paper_outputs'
    saved_paper_out = repo / 'paper_outputs'
    generated_files = sorted(p.relative_to(paper_out).as_posix() for p in paper_out.rglob('*') if p.is_file())
    saved_files = sorted(p.relative_to(saved_paper_out).as_posix() for p in saved_paper_out.rglob('*') if p.is_file())
    same_set = generated_files == saved_files
    rows.append({
        "group": "paper_snapshot",
        "check": "file_set",
        "kind": "exact",
        "critical": True,
        "status": "PASS" if same_set else "FAIL",
        "generated": str(paper_out),
        "reference": str(saved_paper_out),
        "detail": json.dumps({
            "generated_files": len(generated_files),
            "saved_files": len(saved_files),
            "only_generated": sorted(set(generated_files) - set(saved_files)),
            "only_saved": sorted(set(saved_files) - set(generated_files)),
        }, ensure_ascii=False, sort_keys=True),
    })
    for rel in saved_files:
        add(rows, 'paper_snapshot', rel, 'exact', paper_out / rel, saved_paper_out / rel, critical=True)

    df=pd.DataFrame(rows);df.to_csv(ver/'reference_comparison.csv',index=False)
    crit=df[df.critical.astype(bool)];fail=crit[crit.status!='PASS']
    # Extract core numerical maxima from rows.
    lines=['# Reproduction report','',f'- Checks passed: **{int((df.status=="PASS").sum())}/{len(df)}**.',f'- Critical checks failed: **{len(fail)}**.','- Reference outputs live outside the generated workspace and are never refreshed by the reproduction run.','- Numerical reference comparisons use tight floating-point tolerances; the saved paper-facing snapshot is also required to match the regenerated `reproduced/paper_outputs/` tree byte-for-byte.','']
    by=df.groupby(['group','status']).size().unstack(fill_value=0)
    lines+=['## By group','',by.to_markdown(),'']
    if len(fail):lines+=['## Critical failures','',fail[['group','check','detail']].to_markdown(index=False),'']
    else:lines+=['## Verdict','','**PASS — the generated computational artifacts reproduce the stored paper reference set under the declared tolerances.**','']
    lines+=['## Covered publication artifacts','','The computational reproduction suite covers manuscript Figures 4–12, Supplementary Figure S8, their exported source-data tables, central SC0–SC11 outputs, validation tables, decision outputs, uncertainty analyses, mechanism tests, feasibility stresses, and calibration-recovery evidence. The saved `paper_outputs/` snapshot is checked exactly against the freshly regenerated export.','']
    (ver/'REPRODUCTION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(df[['group','check','status']].to_string(index=False))
    print('\n',ver/'REPRODUCTION_REPORT.md')
    if len(fail): sys.exit(2)

if __name__=='__main__': main()
