"""Summarize historical calibration/reconstruction fit for the declared target series.

The table is intentionally separated from terminal validation. It compares the
reference historical run used by the local recovery diagnostic with observed
2013-2021 target series. Directly anchored quantities can have zero error by
construction; their role is labeled explicitly so they are not misread as
out-of-sample predictive performance.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "rerun_output"
BUNDLE = HERE.parent / "computational_bundle"
if not BUNDLE.exists():
    BUNDLE = HERE.parent / "core"

DOMAINS = {
    "city_modal": ["modal_share_car", "modal_share_taxi", "modal_share_bus", "modal_share_metro", "modal_share_motorcycle", "modal_share_other"],
    "r12_modal": ["modal_share_car_r12", "modal_share_tax_r12", "modal_share_bus_r12", "modal_share_met_r12", "modal_share_mot_r12", "modal_share_oth_r12"],
    "speed": ["spd_mot", "spd_car", "spd_tax", "spd_bus", "spd_met"],
    "activity": ["trips_per_year", "trips_bus_total", "trips_metro", "trips_per_person_per_day"],
    "energy_environment": ["fuel_gasoline_litre_year", "fuel_CNG_kg_year", "concentration_PM25_ugm3", "concentration_NO2_ugm3"],
    "annual_distance": ["annual_km_car", "annual_km_taxi", "annual_km_motorcycle"],
    "operational_shares": ["brt_share_of_bus", "share_taxi_gasoline", "share_taxi_CNG"],
}
WEIGHTS = {"city_modal": .30, "r12_modal": .25, "speed": .10, "activity": .10,
           "energy_environment": .15, "annual_distance": .05, "operational_shares": .05}
DOMAIN_LABELS = {
    "city_modal": "Tehran modal shares",
    "r12_modal": "R12 modal shares",
    "speed": "Service speeds",
    "activity": "Travel activity",
    "energy_environment": "Energy and environment",
    "annual_distance": "Annual vehicle distance",
    "operational_shares": "Operational shares",
}
ROLE_OVERRIDES = {
    "spd_met": "historical-series anchor",
    "brt_share_of_bus": "historical-series anchor",
    "trips_per_person_per_day": "empirical scalar anchor",
    "share_taxi_gasoline": "fixed structural fuel-split expression",
    "share_taxi_CNG": "fixed structural fuel-split expression",
    "annual_km_car": "derived historical reconstruction",
    "annual_km_taxi": "derived historical reconstruction",
    "annual_km_motorcycle": "derived historical reconstruction",
}
DERIVED_UNITS = {"annual_km_car": "km/vehicle/year", "annual_km_taxi": "km/vehicle/year", "annual_km_motorcycle": "km/vehicle/year"}


def prediction(sim: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "annual_km_car":
        return sim["vkm_car"] / sim["private_cars_total"].replace(0, np.nan)
    if metric == "annual_km_taxi":
        return sim["vkm_taxi"] / sim["taxis_total"].replace(0, np.nan)
    if metric == "annual_km_motorcycle":
        return sim["vkm_motorcycle"] / sim["motorcycles_total"].replace(0, np.nan)
    return sim[metric]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim_path = OUT / "reference_timeseries.csv"
    if not sim_path.exists():
        raise FileNotFoundError("reference_timeseries.csv is required; run reproduce_full_calibration_evidence.py first")
    raw = pd.read_csv(BUNDLE / "config" / "DATA_clean.csv")
    sim = pd.read_csv(sim_path)
    registry = pd.read_csv(HERE / "historical_metric_registry.csv")
    cfg = yaml.safe_load((BUNDLE / "config" / "BASE_CONFIG.yaml").read_text())
    raw["YEAR_GRG"] = pd.to_numeric(raw["YEAR_GRG"], errors="coerce")
    sim["YEAR_GRG"] = pd.to_numeric(sim["YEAR_GRG"], errors="coerce")
    truth_map = dict(zip(registry.metric.astype(str), registry.truth_column.astype(str)))
    metric_domain = {m: d for d, metrics in DOMAINS.items() for m in metrics}

    metric_rows = []
    domain_vectors: dict[str, tuple[list[float], list[float]]] = {d: ([], []) for d in DOMAINS}
    for metric, truth_col in truth_map.items():
        if truth_col not in raw.columns:
            continue
        try:
            pred = prediction(sim, metric)
        except KeyError:
            continue
        m = pd.DataFrame({"YEAR_GRG": sim["YEAR_GRG"], "sim": pred}).merge(
            raw[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner"
        )
        m = m[m["YEAR_GRG"].between(2013, 2021)].copy()
        y = pd.to_numeric(m[truth_col], errors="coerce").to_numpy(float)
        p = pd.to_numeric(m["sim"], errors="coerce").to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(p)
        y, p = y[mask], p[mask]
        if not len(y):
            continue
        err = p - y
        denom = max(float(np.mean(np.abs(y))), 1e-9)
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        nz = np.abs(y) > 1e-12
        mape = float(np.mean(np.abs(err[nz] / y[nz])) * 100.0) if nz.any() else np.nan
        domain = metric_domain[metric]
        domain_vectors[domain][0].extend(y.tolist())
        domain_vectors[domain][1].extend(p.tolist())
        agent = cfg.get("agents", {}).get(metric, {})
        role = ROLE_OVERRIDES.get(metric, "endogenous historical reconstruction")
        unit = DERIVED_UNITS.get(metric, str(agent.get("units", "")))
        metric_rows.append({
            "domain": domain,
            "domain_label": DOMAIN_LABELS[domain],
            "metric": metric,
            "role": role,
            "unit": unit,
            "n": int(len(y)),
            "observed_mean": float(np.mean(y)),
            "simulated_mean": float(np.mean(p)),
            "mae": mae,
            "rmse": rmse,
            "nrmse_percent": 100.0 * rmse / denom,
            "mape_percent": mape,
            "relative_bias_percent": 100.0 * bias / denom,
        })

    metric_df = pd.DataFrame(metric_rows)
    domain_rows = []
    for domain, (ys, ps) in domain_vectors.items():
        y = np.asarray(ys, float); p = np.asarray(ps, float); err = p - y
        denom = max(float(np.mean(np.abs(y))), 1e-9)
        rmse = float(np.sqrt(np.mean(err**2)))
        loss = rmse / denom
        domain_rows.append({
            "domain": domain,
            "domain_label": DOMAIN_LABELS[domain],
            "target_series": len(DOMAINS[domain]),
            "metric_year_observations": int(len(y)),
            "weight": WEIGHTS[domain],
            "domain_nrmse": loss,
            "domain_nrmse_percent": 100.0 * loss,
            "weighted_objective_contribution": WEIGHTS[domain] * loss,
        })
    domain_df = pd.DataFrame(domain_rows)
    fit_loss = float(domain_df["weighted_objective_contribution"].sum())
    domain_df.loc[len(domain_df)] = {
        "domain": "overall",
        "domain_label": "Weighted fit loss",
        "target_series": int(sum(len(x) for x in DOMAINS.values())),
        "metric_year_observations": int(domain_df[domain_df.domain != "overall"]["metric_year_observations"].sum()),
        "weight": 1.0,
        "domain_nrmse": np.nan,
        "domain_nrmse_percent": np.nan,
        "weighted_objective_contribution": fit_loss,
    }

    metric_df.to_csv(OUT / "historical_calibration_fit_metrics_2013_2021.csv", index=False)
    domain_df.to_csv(OUT / "historical_calibration_domain_fit_2013_2021.csv", index=False)

    exact = {"spd_met", "brt_share_of_bus", "share_taxi_gasoline", "share_taxi_CNG"}
    nonexact = metric_df[~metric_df.metric.isin(exact)].copy()
    summary = {
        "calibration_years": [2013, 2021],
        "number_of_target_series": int(len(metric_df)),
        "exact_replay_or_structural_anchor_series": sorted(exact),
        "number_of_nonexact_series": int(len(nonexact)),
        "nonexact_nrmse_le_15_percent": int((nonexact.nrmse_percent <= 15).sum()),
        "nonexact_nrmse_le_20_percent": int((nonexact.nrmse_percent <= 20).sum()),
        "maximum_nonexact_nrmse_percent": float(nonexact.nrmse_percent.max()),
        "maximum_nonexact_nrmse_metric": str(nonexact.loc[nonexact.nrmse_percent.idxmax(), "metric"]),
        "weighted_fit_loss": fit_loss,
        "interpretation": "In-sample historical calibration/reconstruction diagnostics; not terminal validation. Exact historical anchors are labeled and should not be interpreted as predictive accuracy.",
    }
    (OUT / "historical_calibration_fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE historical calibration fit summary")


if __name__ == "__main__":
    main()
