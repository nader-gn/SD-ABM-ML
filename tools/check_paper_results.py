#!/usr/bin/env python3
"""Check reported numerical claims against regenerated outputs."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

HEADLINE = [
    "CO₂ emissions", "Congestion index", "Electricity use", "Final energy", "Health indicator",
    "Modal share: car", "Modal share: public transport", "NOₓ emissions", "PM₂.₅ emissions",
    "Time loss (car)", "PCE-weighted VKT",
]


def main(repo: Path) -> None:
    repo = repo.resolve()
    core = repo / "reproduced" / "workspace" / "computational_bundle"
    ext = core / "supplementary_analyses"
    cal = repo / "reproduced" / "workspace" / "calibration_recovery" / "rerun_output"
    ver = repo / "reproduced" / "verification"
    ver.mkdir(parents=True, exist_ok=True)
    rows = []

    def check(name, actual, expected, tol, source):
        actual = float(actual)
        ok = math.isfinite(actual) and abs(actual - expected) <= tol
        rows.append({"check": name, "actual": actual, "expected_manuscript": expected, "absolute_tolerance": tol, "status": "PASS" if ok else "FAIL", "source": source})

    def effect(table, scenario, geo, kpi, value_col):
        a = table[(table.scenario == scenario) & (table.geo == geo) & (table.kpi == kpi)][value_col].iloc[0]
        b = table[(table.scenario == "SC0") & (table.geo == geo) & (table.kpi == kpi)][value_col].iloc[0]
        return 100.0 * (a - b) / abs(b)

    def exact_check(name, actual, expected, source):
        rows.append({
            "check": name, "actual": str(actual), "expected_manuscript": str(expected),
            "absolute_tolerance": "exact", "status": "PASS" if actual == expected else "FAIL", "source": source,
        })

    # Reported scenario registry (Table 3).
    registry = pd.read_csv(core / "scenarios" / "scenario_registry.csv", keep_default_na=False)
    exact_check("Scenario registry codes", registry.scenario.tolist(), [f"SC{i}" for i in range(12)], "scenario_registry.csv")
    expected_package_composition = {
        "SC8": "SC1 full + SC3 light + SC4 full + SC6 medium",
        "SC9": "SC2 full + SC3 light + SC4 medium + SC5 medium + SC6 light",
        "SC10": "SC4 full + SC5 full + SC6 strong + SC7 medium + SC3 light",
        "SC11": "SC1 medium + SC2 medium + SC3 medium + SC5 full + SC6 medium + SC7 medium",
    }
    for sc, content in expected_package_composition.items():
        q = registry[registry.scenario == sc].iloc[0]
        exact_check(f"Table3 {sc} package composition", q.main_intervention_content, content, "scenario_registry.csv")
    expected_public_labels = {
        "SC0":"Baseline", "SC1":"Demand smoothing", "SC2":"Access charging",
        "SC3":"Parking and curb management", "SC4":"Fare support",
        "SC5":"Public-transport service improvement", "SC6":"Local pollutant cleanup",
        "SC7":"Clean-fleet transition", "SC8":"Balanced package",
        "SC9":"Access-led package", "SC10":"PT-first clean package", "SC11":"Broad package",
    }
    for sc, expected_label in expected_public_labels.items():
        actual_label = registry.loc[registry.scenario.eq(sc), "short_label"].iloc[0]
        exact_check(f"Table3 {sc} public label", actual_label, expected_label, "scenario_registry.csv")
    # Every non-baseline registry entry must point to an existing executable overlay.
    missing_overlays = sorted([name for name in registry.loc[registry.scenario.ne("SC0"), "overlay_file"] if not (core / "scenarios" / name).exists()])
    exact_check("Scenario registry overlays resolve", missing_overlays, [], "scenario_registry.csv")

    # Decision architecture (Table 4) must match the retained 4 x 3 KPI representatives.
    da = yaml.safe_load((core / "config" / "decision_architecture.yaml").read_text(encoding="utf-8"))
    expected_kpis = {
        "Transportation": ["Time loss (car)", "Modal share: public transport", "PCE-weighted VKT"],
        "Environmental": ["CO₂ emissions", "NOₓ emissions", "PM₂.₅ emissions"],
        "Social": ["Health indicator", "PT affordability ratio", "Noise exposure index"],
        "Economic": ["PT OPEX", "Net recurrent public burden", "Energy cost"],
    }
    for pillar, expected in expected_kpis.items():
        actual = []
        for family in da["core_outcome"]["pillars"][pillar]["subfamilies"].values():
            actual.extend(item["metric"] for item in family)
        exact_check(f"Table4 {pillar} KPI representatives", actual, expected, "decision_architecture.yaml")

    exact_check("Decision architecture uses implementation_screen key", "implementation_screen" in da, True, "decision_architecture.yaml")
    exact_check("Decision architecture uses implementation_screen key", "implementation_screen" in da, True, "decision_architecture.yaml")

    # Fixed ML specification (Methods §3.2.4 and Supplementary Table S5).
    ml_cfg = yaml.safe_load((core / "config" / "BASE_CONFIG.yaml").read_text(encoding="utf-8"))
    agents_cfg = ml_cfg["agents"]
    expected_r12_ml = {
        "modal_share_mot_r12_ml": (800, 0.03, 4, 0.90, 0.7),
        "modal_share_car_r12_ml": (600, 0.05, 3, 0.90, 0.8),
        "modal_share_tax_r12_ml": (600, 0.05, 3, 0.90, 0.8),
        "modal_share_bus_r12_ml": (600, 0.05, 3, 0.90, 0.8),
        "modal_share_met_r12_ml": (1200, 0.03, 4, 0.90, 0.6),
        "modal_share_oth_r12_ml": (300, 0.05, 3, 0.90, 0.8),
    }
    rates = []
    for agent_name, expected in expected_r12_ml.items():
        hp = agents_cfg[agent_name]["hyperparameters"]
        actual = (int(hp["n_estimators"]), float(hp["learning_rate"]), int(hp["max_depth"]), float(hp["subsample"]), float(hp["update_rate"]))
        exact_check(f"TableS5 {agent_name} fixed settings", actual, expected, "BASE_CONFIG.yaml")
        rates.append(float(hp["update_rate"]))
    exact_check("Methods R12 runtime update-rate range", (min(rates), max(rates)), (0.6, 0.8), "BASE_CONFIG.yaml")
    exact_check("Runtime feature selection disabled", bool(ml_cfg.get("feature_selection", {}).get("enabled", False)), False, "BASE_CONFIG.yaml")
    exact_check("Runtime HPO disabled", bool(ml_cfg.get("hyperparams", {}).get("enabled", False)), False, "BASE_CONFIG.yaml")

    # Historical calibration claims reported in Methods §3.3.1.
    popcal = pd.read_csv(cal / "population_calibration_replay.csv").iloc[0]
    check("Population migration calibration replay", popcal.replayed_migration_rate, 0.00555739, 5e-9, "population_calibration_replay.csv")
    check("Population migration configured coefficient", popcal.deployed_migration_rate, 0.00556, 5e-12, "population_calibration_replay.csv")
    anchors = pd.read_csv(cal / "historical_anchor_checks.csv")
    exact_check("Historical parameterization anchor checks all pass", bool(anchors["pass"].astype(bool).all()), True, "historical_anchor_checks.csv")
    mapped = pd.read_csv(cal / "mapped_historical_input_audit.csv")
    exact_check("Mapped historical input-agent count", len(mapped), 55, "mapped_historical_input_audit.csv")
    exact_check("Mapped historical input columns all present", bool(mapped["column_exists"].astype(bool).all()), True, "mapped_historical_input_audit.csv")
    stock_cov = pd.read_csv(cal / "stock_calibration_coverage.csv")
    exact_check("All stock agents classified in calibration architecture", len(stock_cov), 16, "stock_calibration_coverage.csv")
    exact_check("Stock routes with direct replay/alignment evidence", int(stock_cov.verification_status.eq("REPLAYED").sum()), 13, "stock_calibration_coverage.csv")
    fare_replay = pd.read_csv(cal / "fare_projection_base_replays.csv")
    exact_check("Bus/metro projection-base replay checks all pass", bool(fare_replay["pass"].astype(bool).all()), True, "fare_projection_base_replays.csv")
    check("Bus projection fare-base replay", fare_replay.loc[fare_replay["mode"].eq("bus"), "replayed_projection_base_IRR_trip"].iloc[0], 1155.7298086979542, 1e-9, "fare_projection_base_replays.csv")
    check("Metro projection fare-base replay", fare_replay.loc[fare_replay["mode"].eq("metro"), "replayed_projection_base_IRR_trip"].iloc[0], 1098.363504572412, 1e-9, "fare_projection_base_replays.csv")
    role_audit = pd.read_csv(cal / "calibration_scalar_role_audit.csv")
    car_role = role_audit.loc[role_audit.parameter.eq("spd_car_city_calibration_factor")].iloc[0]
    voc_role = role_audit.loc[role_audit.parameter.eq("r12_voc_multiplier")].iloc[0]
    trip_role = role_audit.loc[role_audit.parameter.eq("trips_per_person_per_day")].iloc[0]
    check("City car-speed redundant multiplier OAT profile gap", float(str(car_role.role_check_note).split("=")[-1]), 0.0, 1e-10, "calibration_scalar_role_audit.csv")
    check("R12 v/c diagnostic multiplier recovery-objective sensitivity", voc_role.max_abs_fit_loss_change_over_plusminus10_percent, 0.0, 1e-12, "calibration_scalar_role_audit.csv")
    exact_check("Trip-rate anchor remains historically sensitive", bool(trip_role.historically_sensitive_in_recovery_objective), True, "calibration_scalar_role_audit.csv")

    # Table 5 / abstract historical validation.
    val = pd.read_csv(core / "verification" / "Table_05_validation_pooled.csv")
    published_validation = {
        ("all_hindcast_2012_2023", "Tehran"): (0.994, 0.64, 1.23),
        ("all_hindcast_2012_2023", "Region12"): (0.947, 1.15, 1.61),
        ("holdout_train_2012_2021", "Tehran"): (0.981, 1.12, 2.16),
        ("holdout_train_2012_2021", "Region12"): (0.929, 1.33, 1.86),
        ("holdout_test_2022_2023", "Tehran"): (0.984, 1.57, 2.32),
        ("holdout_test_2022_2023", "Region12"): (0.694, 2.76, 3.93),
    }
    for (split, geo), (r2, mae, rmse) in published_validation.items():
        q = val[(val.split == split) & (val.geo == geo)].iloc[0]
        check(f"Table5 {split} {geo} R2", q.R2, r2, 0.0006, "Table_05_validation_pooled.csv")
        check(f"Table5 {split} {geo} MAE_pp", q.MAE_pp, mae, 0.006, "Table_05_validation_pooled.csv")
        check(f"Table5 {split} {geo} RMSE_pp", q.RMSE_pp, rmse, 0.006, "Table_05_validation_pooled.csv")

    # Longer rolling-origin validation reported in the validation narrative.
    wf = pd.read_csv(ext / "walk_forward_pooled_summary.csv")
    published_wf = {
        ("deployed", "Tehran"): (0.980, 1.69, 2.65),
        ("deployed", "Region 12"): (0.804, 2.17, 3.10),
        ("reduced_feature", "Tehran"): (0.991, 1.17, 1.82),
        ("reduced_feature", "Region 12"): (0.825, 1.95, 2.93),
    }
    for (spec, geo), (r2, mae, rmse) in published_wf.items():
        q = wf[(wf.specification == spec) & (wf.geo == geo) & (wf.subset == "main_non_covid")].iloc[0]
        check(f"Rolling-origin {spec} {geo} R2", q.R2, r2, 0.0006, "walk_forward_pooled_summary.csv")
        check(f"Rolling-origin {spec} {geo} MAE_pp", q.MAE_pp, mae, 0.006, "walk_forward_pooled_summary.csv")
        check(f"Rolling-origin {spec} {geo} RMSE_pp", q.RMSE_pp, rmse, 0.006, "walk_forward_pooled_summary.csv")

    # SC0 baseline values and derived per-person values used in Results.
    sc0 = pd.read_csv(core / "outputs" / "simulation_data_SC0.csv")
    q0 = sc0[sc0.YEAR_GRG.between(2024, 2030)]
    checks = [
        ("SC0 Tehran mean population millions", q0.population_city.mean()/1e6, 10.23, 0.006),
        ("SC0 Tehran mean trips billion/year", q0.trips_per_year.mean()/1e9, 7.65, 0.006),
        ("SC0 Tehran mean PCE-km billion/year", q0.vkm_pce_total.mean()/1e9, 71.51, 0.006),
        ("SC0 Tehran mean congestion", q0.congestion_index.mean(), 0.40, 0.006),
        ("SC0 Tehran mean car time loss million h", q0.time_loss_car_hours_year.mean()/1e6, 911.36, 0.006),
        ("SC0 Tehran trips/person/day", q0.trips_per_year.mean()/q0.population_city.mean()/365.0, 2.05, 0.006),
        ("SC0 Tehran PCE-km/person/day", q0.vkm_pce_total.mean()/q0.population_city.mean()/365.0, 19.15, 0.006),
        ("SC0 R12 mean population thousand", q0.pop_r12.mean()/1e3, 247.79, 0.006),
        ("SC0 R12 mean trips million/year", q0.trp_r12.mean()/1e6, 309.47, 0.006),
        ("SC0 R12 mean congestion", q0.congestion_index_r12.mean(), 1.22, 0.006),
        ("SC0 R12 trips/person/day", q0.trp_r12.mean()/q0.pop_r12.mean()/365.0, 3.42, 0.006),
        ("SC0 R12 PCE-km/person/day", q0.vkm_pce_total_r12.mean()/q0.pop_r12.mean()/365.0, 11.05, 0.006),
    ]
    for name, actual, expected, tol in checks:
        check(name, actual, expected, tol, "simulation_data_SC0.csv")

    mean = pd.read_csv(core / "outputs" / "kpi_selected_mean_2024_2030.csv")
    end = pd.read_csv(core / "outputs" / "kpi_selected_2030.csv")

    # Scenario claims reported in Results.
    sc1 = pd.read_csv(core / "outputs" / "simulation_data_SC1.csv")
    q1 = sc1[sc1.YEAR_GRG.between(2024, 2030)]
    check("SC1 Tehran mean trips effect %", 100*(q1.trips_per_year.mean()-q0.trips_per_year.mean())/q0.trips_per_year.mean(), -1.99, 0.006, "simulation_data_SC0/SC1.csv")
    check("SC1 Tehran mean car time-loss effect %", effect(mean,"SC1","Tehran","Time loss (car)","mean_2024_2030"), -5.44, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC1 Tehran 2030 car time-loss effect %", effect(end,"SC1","Tehran","Time loss (car)","value_2030"), -9.65, 0.006, "kpi_selected_2030.csv")
    check("SC1 R12 mean car time-loss effect %", effect(mean,"SC1","Region12","Time loss (car)","mean_2024_2030"), -4.25, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC1 R12 mean CO2 effect %", effect(mean,"SC1","Region12","CO₂ emissions","mean_2024_2030"), -0.21, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC1 R12 mean PM2.5 effect %", effect(mean,"SC1","Region12","PM₂.₅ emissions","mean_2024_2030"), -0.14, 0.006, "kpi_selected_mean_2024_2030.csv")
    sc0_pt_teh = mean[(mean.scenario=="SC0")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    sc1_pt_teh = mean[(mean.scenario=="SC1")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    check("SC1 Tehran PT change percentage points", 100*(sc1_pt_teh-sc0_pt_teh), 0.23, 0.006, "kpi_selected_mean_2024_2030.csv")

    check("SC2 R12 mean car share effect %", effect(mean,"SC2","Region12","Modal share: car","mean_2024_2030"), -2.46, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC2 R12 2030 car share effect %", effect(end,"SC2","Region12","Modal share: car","value_2030"), -5.41, 0.006, "kpi_selected_2030.csv")
    check("SC2 R12 mean PM2.5 effect %", effect(mean,"SC2","Region12","PM₂.₅ emissions","mean_2024_2030"), 0.46, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC3 R12 mean car share effect %", effect(mean,"SC3","Region12","Modal share: car","mean_2024_2030"), -2.11, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC3 R12 mean car time-loss effect %", effect(mean,"SC3","Region12","Time loss (car)","mean_2024_2030"), -2.35, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC3 R12 mean PM2.5 effect %", effect(mean,"SC3","Region12","PM₂.₅ emissions","mean_2024_2030"), 0.24, 0.006, "kpi_selected_mean_2024_2030.csv")
    sc0_car_r12_m = mean[(mean.scenario=="SC0")&(mean.geo=="Region12")&(mean.kpi=="Modal share: car")].mean_2024_2030.iloc[0]
    sc2_car_r12_m = mean[(mean.scenario=="SC2")&(mean.geo=="Region12")&(mean.kpi=="Modal share: car")].mean_2024_2030.iloc[0]
    sc3_car_r12_m = mean[(mean.scenario=="SC3")&(mean.geo=="Region12")&(mean.kpi=="Modal share: car")].mean_2024_2030.iloc[0]
    sc0_car_r12_e = end[(end.scenario=="SC0")&(end.geo=="Region12")&(end.kpi=="Modal share: car")].value_2030.iloc[0]
    sc2_car_r12_e = end[(end.scenario=="SC2")&(end.geo=="Region12")&(end.kpi=="Modal share: car")].value_2030.iloc[0]
    check("SC2 R12 mean car-share change percentage points", 100*(sc2_car_r12_m-sc0_car_r12_m), -0.52, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC2 R12 2030 car-share change percentage points", 100*(sc2_car_r12_e-sc0_car_r12_e), -1.14, 0.006, "kpi_selected_2030.csv")
    check("SC3 R12 mean car-share change percentage points", 100*(sc3_car_r12_m-sc0_car_r12_m), -0.45, 0.006, "kpi_selected_mean_2024_2030.csv")

    bpt = mean[(mean.scenario=="SC0")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]*100
    s5pt = mean[(mean.scenario=="SC5")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]*100
    check("SC5 Tehran baseline PT share %", bpt, 12.224, 0.0015, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran PT share %", s5pt, 16.294, 0.0015, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran PT change percentage points", s5pt-bpt, 4.070, 0.0015, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran PT relative effect %", effect(mean,"SC5","Tehran","Modal share: public transport","mean_2024_2030"), 33.30, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran diesel effect %", effect(mean,"SC5","Tehran","Energy: diesel","mean_2024_2030"), 61.19, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran final energy effect %", effect(mean,"SC5","Tehran","Final energy","mean_2024_2030"), -4.95, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC5 Tehran CO2 effect %", effect(mean,"SC5","Tehran","CO₂ emissions","mean_2024_2030"), -4.50, 0.006, "kpi_selected_mean_2024_2030.csv")

    speed = pd.read_csv(ext / "sc5_speed_coefficient_sensitivity.csv")
    s = speed[(speed.geo=="Tehran")&(speed.metric=="Modal share: public transport")].delta_vs_SC0_pct
    check("SC5 speed sensitivity minimum PT effect %", s.min(), 31.58, 0.006, "sc5_speed_coefficient_sensitivity.csv")
    check("SC5 speed sensitivity maximum PT effect %", s.max(), 35.03, 0.006, "sc5_speed_coefficient_sensitivity.csv")

    check("SC6 R12 mean PM2.5 effect %", effect(mean,"SC6","Region12","PM₂.₅ emissions","mean_2024_2030"), -20.978, 0.001, "kpi_selected_mean_2024_2030.csv")
    check("SC6 R12 2030 PM2.5 effect %", effect(end,"SC6","Region12","PM₂.₅ emissions","value_2030"), -35.316, 0.001, "kpi_selected_2030.csv")
    decomp = pd.read_csv(ext / "SC6_R12_PM25_decomposition.csv")
    motorcycle_share = decomp.loc[decomp.component.eq("motorcycle"),"share_of_absolute_reduction_pct"].iloc[0]
    check("SC6 R12 motorcycle share of PM2.5 reduction %", motorcycle_share, 91.8, 0.06, "SC6_R12_PM25_decomposition.csv")
    ef = pd.read_csv(ext / "sc6_r12_emission_factor_scaling_sensitivity.csv")
    check("SC6 EF sensitivity least mean reduction magnitude %", -ef.delta_mean_pct.max(), 18.76, 0.006, "sc6_r12_emission_factor_scaling_sensitivity.csv")
    check("SC6 EF sensitivity greatest mean reduction magnitude %", -ef.delta_mean_pct.min(), 21.85, 0.006, "sc6_r12_emission_factor_scaling_sensitivity.csv")

    check("SC10 Tehran mean PT effect %", effect(mean,"SC10","Tehran","Modal share: public transport","mean_2024_2030"), 44.53, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC10 R12 mean PT effect %", effect(mean,"SC10","Region12","Modal share: public transport","mean_2024_2030"), 12.23, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC10 R12 mean PM2.5 effect %", effect(mean,"SC10","Region12","PM₂.₅ emissions","mean_2024_2030"), -19.58, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC10 R12 mean NOx effect %", effect(mean,"SC10","Region12","NOₓ emissions","mean_2024_2030"), -11.10, 0.006, "kpi_selected_mean_2024_2030.csv")
    sc8_pt_teh = mean[(mean.scenario=="SC8")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    sc9_car_r12_m = mean[(mean.scenario=="SC9")&(mean.geo=="Region12")&(mean.kpi=="Modal share: car")].mean_2024_2030.iloc[0]
    sc9_car_r12_e = end[(end.scenario=="SC9")&(end.geo=="Region12")&(end.kpi=="Modal share: car")].value_2030.iloc[0]
    sc10_pt_teh = mean[(mean.scenario=="SC10")&(mean.geo=="Tehran")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    sc0_pt_r12 = mean[(mean.scenario=="SC0")&(mean.geo=="Region12")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    sc10_pt_r12 = mean[(mean.scenario=="SC10")&(mean.geo=="Region12")&(mean.kpi=="Modal share: public transport")].mean_2024_2030.iloc[0]
    sc9_car_r12_e0 = end[(end.scenario=="SC0")&(end.geo=="Region12")&(end.kpi=="Modal share: car")].value_2030.iloc[0]
    check("SC8 Tehran PT change percentage points", 100*(sc8_pt_teh-sc0_pt_teh), 4.25, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC9 R12 mean car-share change percentage points", 100*(sc9_car_r12_m-sc0_car_r12_m), -1.39, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC9 R12 2030 car-share change percentage points", 100*(sc9_car_r12_e-sc9_car_r12_e0), -2.17, 0.006, "kpi_selected_2030.csv")
    check("SC10 Tehran PT change percentage points", 100*(sc10_pt_teh-sc0_pt_teh), 5.44, 0.006, "kpi_selected_mean_2024_2030.csv")
    check("SC10 R12 PT change percentage points", 100*(sc10_pt_r12-sc0_pt_r12), 4.24, 0.006, "kpi_selected_mean_2024_2030.csv")

    # ABM mediation evidence.
    abl = pd.read_csv(ext / "abm_policy_ablation_comparison.csv")
    def ab(sc,geo,metric,col): return abl[(abl.scenario==sc)&(abl.geo==geo)&(abl.metric==metric)][col].iloc[0]
    check("ABM ablation SC2 R12 car active %", ab("SC2","Region 12","Modal share: car","policy_mediation_active"), -2.46, 0.006, "abm_policy_ablation_comparison.csv")
    check("ABM ablation SC2 R12 car zero coefficients %", ab("SC2","Region 12","Modal share: car","policy_response_coefficients_zero"), -0.015, 0.001, "abm_policy_ablation_comparison.csv")
    check("ABM ablation SC10 Tehran PT active %", ab("SC10","Tehran","Modal share: public transport","policy_mediation_active"), 44.53, 0.006, "abm_policy_ablation_comparison.csv")
    check("ABM ablation SC10 Tehran PT zero coefficients %", ab("SC10","Tehran","Modal share: public transport","policy_response_coefficients_zero"), 20.51, 0.006, "abm_policy_ablation_comparison.csv")
    check("ABM ablation SC10 R12 PT active %", ab("SC10","Region 12","Modal share: public transport","policy_mediation_active"), 12.23, 0.006, "abm_policy_ablation_comparison.csv")
    check("ABM ablation SC10 R12 PT zero coefficients %", ab("SC10","Region 12","Modal share: public transport","policy_response_coefficients_zero"), -1.23, 0.006, "abm_policy_ablation_comparison.csv")

    # Qualitative ordering statements used in the Abstract and decision-synthesis narrative.
    atomic = [f"SC{i}" for i in range(1, 8)]
    atomic_pt_tehran = {sc: effect(mean, sc, "Tehran", "Modal share: public transport", "mean_2024_2030") for sc in atomic}
    exact_check("Strongest atomic Tehran PT-share lever", max(atomic_pt_tehran, key=atomic_pt_tehran.get), "SC5", "kpi_selected_mean_2024_2030.csv")
    atomic_pm_r12 = {sc: effect(mean, sc, "Region12", "PM₂.₅ emissions", "mean_2024_2030") for sc in atomic}
    exact_check("Largest atomic R12 PM2.5 reduction", min(atomic_pm_r12, key=atomic_pm_r12.get), "SC6", "kpi_selected_mean_2024_2030.csv")
    check("SC7 Tehran PT-share change approximately zero %", effect(mean,"SC7","Tehran","Modal share: public transport","mean_2024_2030"), 0.0, 1e-12, "kpi_selected_mean_2024_2030.csv")
    check("SC7 R12 PT-share change approximately zero %", effect(mean,"SC7","Region12","Modal share: public transport","mean_2024_2030"), 0.0, 1e-12, "kpi_selected_mean_2024_2030.csv")
    check("SC7 Tehran congestion change negligible %", effect(mean,"SC7","Tehran","Congestion index","mean_2024_2030"), 0.0, 0.01, "kpi_selected_mean_2024_2030.csv")
    check("SC7 R12 congestion change negligible %", effect(mean,"SC7","Region12","Congestion index","mean_2024_2030"), 0.0, 0.01, "kpi_selected_mean_2024_2030.csv")

    lens = pd.read_csv(core / "outputs" / "Figure_10_outcome_lens_scores.csv")
    expected_lens_leaders = {
        ("Tehran", "Transportation"): "SC11",
        ("Tehran", "Environmental"): "SC8",
        ("Tehran", "Social"): "SC8",
        ("Tehran", "Economic"): "SC1",
        ("Region12", "Transportation"): "SC10",
        ("Region12", "Environmental"): "SC10",
        ("Region12", "Social"): "SC8",
    }
    for (geo, pillar), expected in expected_lens_leaders.items():
        q = lens[(lens.geo == geo) & (lens.pillar == pillar)].sort_values("score", ascending=False)
        exact_check(f"Figure10 {geo} {pillar} leader", q.iloc[0].scenario, expected, "Figure_10_outcome_lens_scores.csv")

    # Decision robustness.
    eq = pd.read_csv(core / "outputs" / "Figure_11_equal_weight_scores.csv")
    acc = pd.read_csv(core / "outputs" / "Figure_11_rank_acceptability.csv")
    def eqv(geo,sc): return eq[(eq.geo==geo)&(eq.scenario==sc)].equal_weight_score.iloc[0]
    def win(geo,sc): return 100*acc[(acc.geo==geo)&(acc.scenario==sc)].win_probability.iloc[0]
    check("Tehran SC8 equal-weight score", eqv("Tehran","SC8"), 55.50, 0.006, "Figure_11_equal_weight_scores.csv")
    check("Tehran SC8 sampled top-rank share %", win("Tehran","SC8"), 93.39, 0.006, "Figure_11_rank_acceptability.csv")
    check("R12 SC8 equal-weight score", eqv("Region12","SC8"), 52.35, 0.006, "Figure_11_equal_weight_scores.csv")
    check("R12 SC10 equal-weight score", eqv("Region12","SC10"), 45.09, 0.006, "Figure_11_equal_weight_scores.csv")
    check("R12 SC8 sampled top-rank share %", win("Region12","SC8"), 59.96, 0.006, "Figure_11_rank_acceptability.csv")
    check("R12 SC10 sampled top-rank share %", win("Region12","SC10"), 29.54, 0.006, "Figure_11_rank_acceptability.csv")

    # Executable ontology and graph-audit claims.
    role = pd.read_csv(core / "verification" / "agent_role_summary.csv")
    graph = pd.read_csv(core / "verification" / "execution_graph_audit.csv")
    def role_count(label):
        return int(role.loc[role.role.eq(label), "count"].iloc[0])
    def graph_value(metric):
        return int(graph.loc[graph.metric.eq(metric), "value"].iloc[0])
    check("Total configured entity count", graph_value("total_agents"), 706, 0, "execution_graph_audit.csv")
    check("Structural input count", graph_value("input_agents"), 237, 0, "execution_graph_audit.csv")
    check("Structural expression count", graph_value("expression_agents"), 441, 0, "execution_graph_audit.csv")
    check("Structural stock count", graph_value("stock_agents"), 16, 0, "execution_graph_audit.csv")
    check("Structural ML count", graph_value("ml_agents"), 12, 0, "execution_graph_audit.csv")
    check("Functional input role count", role_count("Input/exogenous or initialized state"), 237, 0, "agent_role_summary.csv")
    check("Functional SD role count", role_count("SD stock-flow / aggregate consequence / feedback"), 400, 0, "agent_role_summary.csv")
    check("Functional ABM role count", role_count("ABM behavioral mediation / competition / exposure"), 57, 0, "agent_role_summary.csv")
    check("Functional ML role count", role_count("ML modal-prior surrogate"), 12, 0, "agent_role_summary.csv")
    check("Execution-group count", graph_value("execution_group_count"), 706, 0, "execution_graph_audit.csv")
    conn = pd.read_csv(core / "verification" / "agent_connectivity_summary.csv")
    def conn_value(metric): return float(conn.loc[conn.metric.eq(metric), "value"].iloc[0])
    check("Unused configured node count", conn_value("unused_configured_nodes"), 0, 0, "agent_connectivity_summary.csv")
    check("Engine-isolated support-node count", conn_value("engine_isolated_nodes"), 1, 0, "agent_connectivity_summary.csv")
    check("Multi-node execution-group count", graph_value("multi_node_execution_groups"), 0, 0, "execution_graph_audit.csv")
    check("Execution-graph self-loop count", graph_value("self_loops"), 0, 0, "execution_graph_audit.csv")
    check("Delayed-node count", graph_value("delayed_nodes"), 44, 0, "execution_graph_audit.csv")

    # Central replay.
    replay = pd.read_csv(ext / "central_output_replay.csv")
    check("Central replay scenario count", len(replay), 12, 0, "central_output_replay.csv")
    check("Central replay minimum common columns", replay.common_columns.min(), 799, 0, "central_output_replay.csv")
    check("Central replay maximum relative difference", replay.max_relative_difference.max(), 0.0, 1e-10, "central_output_replay.csv")

    # Full-portfolio uncertainty counts, direction stability and convergence.
    g = pd.read_csv(ext / "uncertainty_relative_effects_SC0_SC11.csv")
    gb = pd.read_csv(ext / "uncertainty_block_relative_effects_SC0_SC11.csv")
    check("Gaussian configuration-draw count", g[["draw","scenario"]].drop_duplicates().shape[0], 384, 0, "uncertainty_relative_effects_SC0_SC11.csv")
    check("Residual-block configuration-draw count", gb[["draw","scenario"]].drop_duplicates().shape[0], 96, 0, "uncertainty_block_relative_effects_SC0_SC11.csv")
    policy = g[(g.scenario!="SC0") & g.metric.isin(HEADLINE)]
    stable = 0; total = 0; exceptions = []
    diffs = []
    for (sc,geo,metric), z in policy.groupby(["scenario","geo","metric"]):
        values = z.delta_mean_pct.dropna().values
        med = np.median(values)
        frac = np.mean(np.sign(values)==np.sign(med)) if med != 0 else 1.0
        total += 1
        if frac == 1.0: stable += 1
        else: exceptions.append((sc,geo,metric))
        z = z.sort_values("draw")
        diffs.append(abs(np.median(z[z.draw<24].delta_mean_pct) - np.median(z[z.draw<32].delta_mean_pct)))
    check("Headline policy-scenario-geography combinations", total, 242, 0, "uncertainty_relative_effects_SC0_SC11.csv")
    check("Fully direction-stable Gaussian combinations", stable, 240, 0, "uncertainty_relative_effects_SC0_SC11.csv")
    expected_exceptions = {("SC2","Tehran","PM₂.₅ emissions"),("SC3","Tehran","PM₂.₅ emissions")}
    rows.append({"check":"Gaussian sign-ambiguity exceptions exactly SC2/SC3 Tehran PM2.5","actual":str(sorted(exceptions)),"expected_manuscript":str(sorted(expected_exceptions)),"absolute_tolerance":"exact","status":"PASS" if set(exceptions)==expected_exceptions else "FAIL","source":"uncertainty_relative_effects_SC0_SC11.csv"})
    check("24-to-32 median absolute headline shift", np.median(diffs), 0.008, 0.0006, "uncertainty_relative_effects_SC0_SC11.csv")
    check("24-to-32 maximum absolute headline shift", np.max(diffs), 1.014, 0.0006, "uncertainty_relative_effects_SC0_SC11.csv")

    # Hard-budget implementation stress.
    traj = pd.read_csv(ext / "SC10_hard_budget_trajectory.csv")
    check("Hard-budget realization 2029", traj.loc[traj.YEAR_GRG.eq(2029),"realization_factor"].iloc[0], 0.987, 0.0006, "SC10_hard_budget_trajectory.csv")
    check("Hard-budget realization 2030", traj.loc[traj.YEAR_GRG.eq(2030),"realization_factor"].iloc[0], 0.921, 0.0006, "SC10_hard_budget_trajectory.csv")
    comp = pd.read_csv(ext / "SC10_hard_budget_comparison.csv")
    # These columns are effects of hard vs standard central SC10 at 2030.
    pt = comp[(comp.geo=="Tehran")&(comp.kpi=="Modal share: public transport")].iloc[0]
    tl = comp[(comp.geo=="Tehran")&(comp.kpi=="Time loss (car)")].iloc[0]
    check("Hard-budget 2030 PT effect difference vs standard %", pt.hard_vs_standard_2030_pct, -0.80, 0.02, "SC10_hard_budget_comparison.csv")
    check("Hard-budget 2030 car time-loss difference vs standard %", tl.hard_vs_standard_2030_pct, 0.09, 0.02, "SC10_hard_budget_comparison.csv")

    # Exogenous demand/fiscal stress: manuscript claims SC8 remains both equal-weight
    # and weight-robust leader in both geographies under every declared stress path.
    stress = pd.read_csv(ext / "exogenous_stress_leadership_summary.csv")
    stress_expected_rows = 8  # four stress variants x two geographies
    check("Exogenous-stress leadership row count", len(stress), stress_expected_rows, 0, "exogenous_stress_leadership_summary.csv")
    rows.append({
        "check": "SC8 equal-weight leader in every exogenous stress/geography",
        "actual": str(sorted(stress.equal_weight_leader.unique().tolist())),
        "expected_manuscript": "['SC8']",
        "absolute_tolerance": "exact",
        "status": "PASS" if (stress.equal_weight_leader == "SC8").all() else "FAIL",
        "source": "exogenous_stress_leadership_summary.csv",
    })
    rows.append({
        "check": "SC8 weight-robust leader in every exogenous stress/geography",
        "actual": str(sorted(stress.weight_robust_leader.unique().tolist())),
        "expected_manuscript": "['SC8']",
        "absolute_tolerance": "exact",
        "status": "PASS" if (stress.weight_robust_leader == "SC8").all() else "FAIL",
        "source": "exogenous_stress_leadership_summary.csv",
    })

    # Reported terminology in regenerated SVG figures.
    fig_dir = core / "figures"
    for number in [5, 7]:
        svg = (fig_dir / f"Figure {number}.svg").read_text(encoding="utf-8")
        exact_check(f"Figure {number} uses PCE-weighted VKT label", "PCE-weighted VKT" in svg, True, f"Figure {number}.svg")
        exact_check(f"Figure {number} shows the VKT label for both geographies", svg.count("PCE-weighted VKT") >= 2, True, f"Figure {number}.svg")
        exact_check(f"Figure {number} uses Health indicator label", "Health indicator" in svg, True, f"Figure {number}.svg")
        exact_check(f"Figure {number} shows the health indicator for both geographies", svg.count("Health indicator") >= 2, True, f"Figure {number}.svg")
    svg9 = (fig_dir / "Figure 9.svg").read_text(encoding="utf-8")
    exact_check("Figure 9 uses compact PCE-VKT key label", "PCE-VKT" in svg9, True, "Figure 9.svg")
    exact_check("Figure 9 uses Health burden key label", "Health burden" in svg9, True, "Figure 9.svg")
    exact_check("Figure 9 uses preference-orientation wording", "preference orientation is applied in Figure 10" in svg9, True, "Figure 9.svg")
    exact_check("Figure 9 omits circular-heatmap wording", "circular heatmap" not in svg9.lower(), True, "Figure 9.svg")
    svg10 = (fig_dir / "Figure 10.svg").read_text(encoding="utf-8")
    exact_check("Figure 10 uses normalized domain-score wording", "Normalized domain score" in svg10, True, "Figure 10.svg")
    exact_check("Figure 10 omits outcome-lens wording", "outcome-lens" not in svg10.lower(), True, "Figure 10.svg")
    svg12 = (fig_dir / "Figure 12.svg").read_text(encoding="utf-8")
    exact_check("Figure 12 uses implementation screen wording", "Implementation screen" in svg12, True, "Figure 12.svg")
    exact_check("Figure 12 uses more-favorable wording", "higher = more favorable" in svg12, True, "Figure 12.svg")
    exact_check("Figure 12 uses implementation-screen metrics wording", "Implementation-screen metrics" in svg12, True, "Figure 12.svg")

    out = pd.DataFrame(rows)
    out.to_csv(ver / "paper_result_checks.csv", index=False)
    failures = out[out.status != "PASS"]
    report = [
        "# Reported numerical consistency report", "",
        f"- Checks passed: **{int((out.status=='PASS').sum())}/{len(out)}**.",
        f"- Checks failed: **{len(failures)}**.",
        "- Expected values are the rounded values reported in the manuscript; tolerances reflect only the manuscript rounding precision.",
        "- This check is independent of the reference-output comparison and is intended to catch a paper-versus-code mismatch.", "",
    ]
    if len(failures):
        report += ["## Failures", "", failures.to_markdown(index=False), ""]
    else:
        report += ["## Verdict", "", "**PASS — the reproduction agrees with the reported numerical claims covered by this checker.**", ""]
    (ver / "PAPER_RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(out[["check","status"]].to_string(index=False))
    if len(failures):
        sys.exit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    main(Path(parser.parse_args().repo))
