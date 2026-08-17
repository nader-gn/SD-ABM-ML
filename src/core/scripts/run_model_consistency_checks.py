
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SCENARIOS = [f"SC{i}" for i in range(12)]
HIST_YEARS = list(range(2013, 2024))

DIRECT_METRICS = [
    ("concentration_PM25_ugm3", "concentration_PM25_ugm3_truth"),
    ("concentration_NO2_ugm3", "concentration_NO2_ugm3_truth"),
    ("fuel_gasoline_litre_year", "fuel_gasoline_litre_year_truth"),
    ("fuel_CNG_kg_year", "fuel_CNG_kg_year_truth"),
    ("share_taxi_gasoline", "share_taxi_gasoline_truth"),
    ("share_taxi_CNG", "share_taxi_CNG_truth"),
    ("fc_motorcycle_gasoline_litre_km", "fc_motorcycle_gasoline_litre_km_truth"),
    ("fc_car_gasoline_litre_km", "fc_car_gasoline_litre_km_truth"),
    ("fc_taxi_gasoline_litre_km", "fc_taxi_gasoline_litre_km_truth"),
    ("spd_mot", "spd_mot_truth"),
    ("spd_car", "spd_car_truth"),
    ("spd_tax", "spd_tax_truth"),
    ("spd_bus", "spd_bus_truth"),
    ("spd_met", "spd_met_truth"),
    ("travel_time_motorcycle_hours", "travel_time_motorcycle_hours_truth"),
    ("travel_time_car_hours", "travel_time_car_hours_truth"),
    ("travel_time_taxi_hours", "travel_time_taxi_hours_truth"),
    ("travel_time_bus_hours", "travel_time_bus_hours_truth"),
    ("travel_time_metro_hours", "travel_time_metro_hours_truth"),
    ("trips_per_year", "trips_per_year_truth"),
    ("trips_bus_total", "trips_bus_total_truth"),
    ("trips_metro", "trips_metro_truth"),
    ("trips_per_person_per_day", "trips_per_person_per_day_truth"),
    ("modal_share_car", "modal_share_car_truth"),
    ("modal_share_taxi", "modal_share_taxi_truth"),
    ("modal_share_bus", "modal_share_bus_truth"),
    ("modal_share_metro", "modal_share_metro_truth"),
    ("modal_share_motorcycle", "modal_share_motorcycle_truth"),
    ("modal_share_other", "modal_share_other_truth"),
    ("modal_share_other_wo_bik", "modal_share_other_wo_bik_truth"),
    ("modal_share_car_r12", "modal_share_car_r12_truth"),
    ("modal_share_tax_r12", "modal_share_tax_r12_truth"),
    ("modal_share_bus_r12", "modal_share_bus_r12_truth"),
    ("modal_share_met_r12", "modal_share_met_r12_truth"),
    ("modal_share_mot_r12", "modal_share_mot_r12_truth"),
    ("modal_share_oth_r12", "modal_share_oth_r12_truth"),
    ("brt_share_of_bus", "brt_share_of_bus_truth"),
]
DERIVED_METRICS = [
    ("annual_km_car", "annual_km_car_truth"),
    ("annual_km_taxi", "annual_km_taxi_truth"),
    ("annual_km_motorcycle", "annual_km_motorcycle_truth"),
]

def metric_stats(sim, obs):
    joined = pd.DataFrame({"sim": pd.to_numeric(sim, errors="coerce"), "obs": pd.to_numeric(obs, errors="coerce")}).dropna()
    if len(joined) < 2:
        return None
    err = joined["sim"] - joined["obs"]
    obs_nonzero = joined["obs"].replace(0, np.nan)
    mape = np.nanmean(np.abs(err / obs_nonzero)) * 100.0
    return {
        "n": int(len(joined)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(np.mean(err)),
        "mape_percent": float(mape),
        "obs_mean": float(joined["obs"].mean()),
        "sim_mean": float(joined["sim"].mean()),
        "last_obs": float(joined["obs"].iloc[-1]),
        "last_sim": float(joined["sim"].iloc[-1]),
    }

def compare_history(root: Path):
    raw = pd.read_csv(root / "config" / "DATA_clean.csv")
    sim = pd.read_csv(root / "outputs" / "simulation_data_SC0.csv")
    raw["YEAR_GRG"] = pd.to_numeric(raw["YEAR_GRG"], errors="coerce")
    sim["YEAR_GRG"] = pd.to_numeric(sim["YEAR_GRG"], errors="coerce")
    raw = raw[raw["YEAR_GRG"].isin(HIST_YEARS)].copy()
    sim = sim[sim["YEAR_GRG"].isin(HIST_YEARS)].copy()
    rows = []
    warnings = []
    for sim_col, truth_col in DIRECT_METRICS:
        if sim_col not in sim.columns or truth_col not in raw.columns:
            continue
        joined = sim[["YEAR_GRG", sim_col]].merge(raw[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
        stats = metric_stats(joined[sim_col], joined[truth_col])
        if stats is None:
            continue
        rows.append({"metric": sim_col, "truth_col": truth_col, "kind": "direct", **stats})
        if stats["mape_percent"] > 25:
            warnings.append(f"{sim_col}: MAPE {stats['mape_percent']:.1f}%")
    if {"vkm_car","private_cars_total","vkm_taxi","taxis_total","vkm_motorcycle","motorcycles_total"}.issubset(sim.columns):
        derived = sim[["YEAR_GRG","vkm_car","private_cars_total","vkm_taxi","taxis_total","vkm_motorcycle","motorcycles_total"]].copy()
        derived["annual_km_car"] = derived["vkm_car"] / derived["private_cars_total"].replace(0, np.nan)
        derived["annual_km_taxi"] = derived["vkm_taxi"] / derived["taxis_total"].replace(0, np.nan)
        derived["annual_km_motorcycle"] = derived["vkm_motorcycle"] / derived["motorcycles_total"].replace(0, np.nan)
        for sim_col, truth_col in DERIVED_METRICS:
            if truth_col not in raw.columns:
                continue
            joined = derived[["YEAR_GRG", sim_col]].merge(raw[["YEAR_GRG", truth_col]], on="YEAR_GRG", how="inner")
            stats = metric_stats(joined[sim_col], joined[truth_col])
            if stats is None:
                continue
            rows.append({"metric": sim_col, "truth_col": truth_col, "kind": "derived", **stats})
            if stats["mape_percent"] > 25:
                warnings.append(f"{sim_col}: MAPE {stats['mape_percent']:.1f}%")
    out = pd.DataFrame(rows).sort_values(["kind", "mape_percent", "mae"], ascending=[True, True, True]).reset_index(drop=True)
    return out, warnings

def sanity_checks(root: Path):
    rows = []
    for code in SCENARIOS:
        path = root / "outputs" / f"simulation_data_{code}.csv"
        if not path.exists():
            rows.append({"scenario": code, "check": "file_exists", "status": "fail", "detail": "missing output file"})
            continue
        df = pd.read_csv(path)
        years = pd.to_numeric(df.get("YEAR_GRG"), errors="coerce")
        ok_span = len(df) == 19 and years.min() == 2012 and years.max() in (2029, 2030)
        rows.append({"scenario": code, "check": "year_span", "status": "ok" if ok_span else "fail", "detail": f"min={years.min()} max={years.max()} rows={len(df)}"})
        for c in [x for x in ["trips_per_year","fuel_gasoline_litre_year","fuel_CNG_kg_year","concentration_PM25_ugm3","concentration_NO2_ugm3","private_cars_total","taxis_total","buses_total","brts_total","metro_cars_total"] if x in df.columns]:
            mn = float(pd.to_numeric(df[c], errors="coerce").min())
            rows.append({"scenario": code, "check": f"nonnegative::{c}", "status": "ok" if mn >= -1e-9 else "fail", "detail": f"min={mn}"})
        for check_name, cols, tol in [
            ("city_modal_sum", ["modal_share_car","modal_share_taxi","modal_share_bus","modal_share_metro","modal_share_motorcycle","modal_share_other"], 1e-4),
            ("r12_modal_sum", ["modal_share_car_r12","modal_share_tax_r12","modal_share_bus_r12","modal_share_met_r12","modal_share_mot_r12","modal_share_oth_r12"], 1e-4),
            ("taxi_fuel_sum", ["share_taxi_CNG","share_taxi_EV","share_taxi_gasoline"], 1e-6),
        ]:
            if set(cols).issubset(df.columns):
                dev = float(np.nanmax(np.abs(df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) - 1.0)))
                rows.append({"scenario": code, "check": check_name, "status": "ok" if dev <= tol else "fail", "detail": f"max_abs_dev={dev}"})
        for c in [x for x in ["share_taxi_CNG","share_taxi_EV","share_taxi_gasoline","modal_share_car","modal_share_taxi","modal_share_bus","modal_share_metro","modal_share_motorcycle","modal_share_other","modal_share_car_r12","modal_share_tax_r12","modal_share_bus_r12","modal_share_met_r12","modal_share_mot_r12","modal_share_oth_r12","brt_share_of_bus"] if x in df.columns]:
            s = pd.to_numeric(df[c], errors="coerce")
            ok = bool(((s >= -1e-9) & (s <= 1+1e-9)).all())
            rows.append({"scenario": code, "check": f"bounded01::{c}", "status": "ok" if ok else "fail", "detail": f"min={s.min()} max={s.max()}"})
        for c, lo in [("occ_car",1.0),("occ_mot",1.0),("occ_tax",1.0),("occ_bus",5.0),("occ_met",10.0)]:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                rows.append({"scenario": code, "check": f"plausible::{c}", "status": "ok" if float(s.min()) >= lo else "fail", "detail": f"min={s.min()} threshold={lo}"})
        for c in [x for x in ["spd_mot","spd_car","spd_tax","spd_bus","spd_met","travel_time_motorcycle_hours","travel_time_car_hours","travel_time_taxi_hours","travel_time_bus_hours","travel_time_metro_hours"] if x in df.columns]:
            s = pd.to_numeric(df[c], errors="coerce")
            rows.append({"scenario": code, "check": f"positive::{c}", "status": "ok" if float(s.min()) > 0 else "fail", "detail": f"min={s.min()}"})
    return pd.DataFrame(rows)

def main(root: Path):
    root = root.resolve()
    ver = root / "verification"
    ver.mkdir(exist_ok=True)
    hist, warnings = compare_history(root)
    hist.to_csv(ver / "historical_reconstruction_2013_2023.csv", index=False)
    sanity = sanity_checks(root)
    sanity.to_csv(ver / "logic_sanity_checks.csv", index=False)
    fail = sanity[sanity["status"] != "ok"].copy()
    worst = hist.sort_values(["mape_percent","mae"], ascending=[False, False]).head(8) if not hist.empty else pd.DataFrame()
    lines = [
        "# Model-consistency summary",
        "",
        "- This report adds historical-reconstruction and logic/sanity checks on top of the existing workflow verification.",
        f"- Historical comparison rows produced: **{len(hist)}** for 2013-2023.",
        f"- Logic/sanity checks passing: **{int((sanity['status']=='ok').sum())}/{len(sanity)}**.",
        "",
        "## Best historical fits (lowest MAPE among direct checks)",
        hist[hist["kind"]=="direct"].sort_values(["mape_percent","mae"]).head(12).to_markdown(index=False) if not hist.empty else "_No rows._",
        "",
        "## Weakest residual historical fits",
        worst.to_markdown(index=False) if not worst.empty else "_No rows._",
    ]
    if warnings:
        lines += ["", "## Residual fit warnings", "\n".join(f"- {w}" for w in warnings[:20])]
    if fail.empty:
        lines += ["", "No logic/sanity failures were detected by the added model-consistency checks."]
    else:
        lines += ["", "## Remaining logic/sanity failures", fail.to_markdown(index=False)]
    (ver / "calibrated_bundle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("DONE model consistency checks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    args = ap.parse_args(); main(Path(args.root))
