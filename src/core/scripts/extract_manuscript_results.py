"""Build the manuscript KPI tables and the figure source table from simulation outputs.

Outputs written under outputs/:
  - kpi_timeseries_selected_long_2024_2030.csv
  - kpi_selected_mean_2024_2030.csv
  - kpi_selected_2030.csv
  - timeseries_all.csv

Usage:
  python scripts/extract_manuscript_results.py --root .
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from kpi_defs import (
    ALL_KPIS,
    KPI_CLIMATE_COST,
    KPI_CO2,
    KPI_CONGESTION,
    KPI_ELECTRICITY_USE,
    KPI_ENERGY_CNG,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_GASOLINE,
    KPI_FINAL_ENERGY,
    KPI_HEALTH_COST,
    KPI_MODAL_SHARE_CAR,
    KPI_MODAL_SHARE_PUBLIC,
    KPI_MUNICIPAL_BUDGET,
    KPI_NON_PUBLIC_TRIPS,
    KPI_NOX,
    KPI_PM25,
    KPI_POPULATION,
    KPI_PT_TRIPS,
    KPI_TIME_LOSS_CAR,
    KPI_TRIPS_TOTAL_EFFECTIVE,
    KPI_VKM_PCE,
    UNIT_MAP,
)

from scenario_meta import SCENARIO_CODES, SCENARIO_IDENTITY

SCENARIOS = SCENARIO_CODES


def load_region12_budget_share(root: Path) -> float:
    cfg = yaml.safe_load((root / "config" / "analysis_overrides.yaml").read_text(encoding="utf-8"))
    share = float(cfg["region12_municipal_budget_share_of_tehran"])
    if not (0 < share < 1):
        raise ValueError(f"Invalid Region 12 municipal budget share: {share}")
    return share


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def tailpipe_vkm(vkm: pd.Series, ev_share: pd.Series) -> pd.Series:
    return vkm * (1 - ev_share)


def gasoline_energy_city(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0)
    gasoline_l = (
        car_vkm_eff * df["fc_car_gasoline_litre_km"] * np.maximum(1 - df["share_car_EV"] - df["share_car_CNG"], 0)
        + df["vkm_taxi"] * df["fc_taxi_gasoline_litre_km"] * np.maximum(1 - df["share_taxi_EV"] - df["share_taxi_CNG"], 0)
        + df["vkm_motorcycle"] * df["fc_motorcycle_gasoline_litre_km"] * np.maximum(1 - df["share_motorcycle_EV"], 0)
    )
    return gasoline_l * df["LHV_gasoline_MJ_litre"]


def diesel_energy_city(df: pd.DataFrame) -> pd.Series:
    diesel_l = df["vkm_bus"] * df["fc_bus_diesel_litre_km"] * np.maximum(1 - df["share_bus_EV"], 0)
    return diesel_l * df["LHV_diesel_MJ_litre"]


def cng_energy_city(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0)
    cng_kg = (
        car_vkm_eff * df["share_car_CNG"] * df["fc_car_CNG_kg_km"]
        + df["vkm_taxi"] * df["share_taxi_CNG"] * df["fc_taxi_CNG_kg_km"]
    )
    return cng_kg * df["LHV_CNG_MJ_kg"]


def gasoline_energy_r12(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0)
    gasoline_l = (
        car_vkm_eff * df["fc_car_gasoline_litre_km"] * np.maximum(1 - df["share_car_EV"] - df["share_car_CNG"], 0)
        + df["vkm_taxi_r12"] * df["fc_taxi_gasoline_litre_km"] * np.maximum(1 - df["share_taxi_EV"] - df["share_taxi_CNG"], 0)
        + df["vkm_motorcycle_r12"] * df["fc_motorcycle_gasoline_litre_km"] * np.maximum(1 - df["share_motorcycle_EV"], 0)
    )
    return gasoline_l * df["LHV_gasoline_MJ_litre"]


def diesel_energy_r12(df: pd.DataFrame) -> pd.Series:
    diesel_l = df["vkm_bus_r12"] * df["fc_bus_diesel_litre_km"] * np.maximum(1 - df["share_bus_EV"], 0)
    return diesel_l * df["LHV_diesel_MJ_litre"]


def cng_energy_r12(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0)
    cng_kg = (
        car_vkm_eff * df["share_car_CNG"] * df["fc_car_CNG_kg_km"]
        + df["vkm_taxi_r12"] * df["share_taxi_CNG"] * df["fc_taxi_CNG_kg_km"]
    )
    return cng_kg * df["LHV_CNG_MJ_kg"]


def nox_city(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_NOx_car_g_km"] * car_tailpipe
        + df["EF_NOx_bus_g_km"] * tailpipe_vkm(df["vkm_bus"], df["share_bus_EV"])
        + df["EF_NOx_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi"], df["share_taxi_EV"])
        + df["EF_NOx_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle"], df["share_motorcycle_EV"])
        + df["EF_grid_NOx_g_kWh"] * df["electricity_transport_kWh_year"]
    )
    return total_g / 1e6


def pm25_city(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_PM25_car_g_km"] * car_tailpipe
        + df["EF_PM25_bus_g_km"] * tailpipe_vkm(df["vkm_bus"], df["share_bus_EV"])
        + df["EF_PM25_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi"], df["share_taxi_EV"])
        + df["EF_PM25_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle"], df["share_motorcycle_EV"])
        + df["EF_grid_PM25_g_kWh"] * df["electricity_transport_kWh_year"]
    )
    return total_g / 1e6


def nox_r12(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_NOx_car_g_km"] * car_tailpipe
        + df["EF_NOx_bus_g_km"] * tailpipe_vkm(df["vkm_bus_r12"], df["share_bus_EV"])
        + df["EF_NOx_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi_r12"], df["share_taxi_EV"])
        + df["EF_NOx_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle_r12"], df["share_motorcycle_EV"])
        + df["EF_grid_NOx_g_kWh"] * df["electricity_kWh_year_r12"]
    )
    return total_g / 1e6


def pm25_r12(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_PM25_car_g_km"] * car_tailpipe
        + df["EF_PM25_bus_g_km"] * tailpipe_vkm(df["vkm_bus_r12"], df["share_bus_EV"])
        + df["EF_PM25_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi_r12"], df["share_taxi_EV"])
        + df["EF_PM25_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle_r12"], df["share_motorcycle_EV"])
        + df["EF_grid_PM25_g_kWh"] * df["electricity_kWh_year_r12"]
    )
    return total_g / 1e6


def health_cost_r12(df: pd.DataFrame) -> pd.Series:
    """Direct Region 12 health-cost reconstruction from region-specific emissions,
    concentration-response parameters, and district population.

    The bundle does not expose a fully separate district air-dispersion model, so
    Region 12 NO2 uses the shipped district concentration proxy and PM2.5 uses the
    same concentration mapping form as the city but with district emissions. This
    is substantially more direct and defensible than downscaling the city health indicator
    by motorized-activity share.
    """
    no2_excess_r12 = np.maximum(df["concentration_NO2_ugm3_r12_proxy"] - df["NO2_threshold_ugm3"], 0)
    rr_no2_r12 = np.exp(df["CR_NO2_mortality"] * no2_excess_r12)
    af_no2_r12 = (rr_no2_r12 - 1) / rr_no2_r12

    pm25_conc_r12 = df["PM25_background_ugm3"] + df["k_PM25_concentration_effective_ugm3_per_t"] * df["PM25_t_year_r12"]
    pm25_excess_r12 = np.maximum(pm25_conc_r12 - df["PM25_threshold_ugm3"], 0)
    rr_pm25_r12 = np.exp(df["CR_PM25_mortality_adult"] * pm25_excess_r12)
    af_pm25_r12 = (rr_pm25_r12 - 1) / rr_pm25_r12

    deaths_r12 = df["pop_r12"] * df["mortality_adult_rate"] * (af_no2_r12 + af_pm25_r12)
    admissions_r12 = df["pop_r12"] * df["respiratory_admission_rate_base"] * (np.exp(df["CR_PM25_respiratory_admission"] * pm25_excess_r12) - 1)
    return deaths_r12 * df["VSL_IRR"] + admissions_r12 * df["cost_hospital_day_IRR_indexed"] * 5


def build_rows(df: pd.DataFrame, region12_budget_share: float):
    df = df.copy()
    years = df["YEAR_GRG"].round().astype(int)
    share_pop = safe_ratio(df["pop_r12"], df["population_city"])

    city_public_share = df["modal_share_bus"] + df["modal_share_metro"]
    r12_public_share = df["modal_share_bus_r12"] + df["modal_share_met_r12"]

    city_total_effective = df["trips_per_year"]
    r12_total_effective = df["trp_r12_effective"]

    city_pt_trips = city_total_effective * city_public_share
    r12_pt_trips = r12_total_effective * r12_public_share

    city_energy_gasoline = gasoline_energy_city(df)
    city_energy_diesel = diesel_energy_city(df)
    city_energy_cng = cng_energy_city(df)

    r12_energy_gasoline = gasoline_energy_r12(df)
    r12_energy_diesel = diesel_energy_r12(df)
    r12_energy_cng = cng_energy_r12(df)

    city_nox = nox_city(df)
    city_pm25 = pm25_city(df)
    r12_nox = nox_r12(df)
    r12_pm25 = pm25_r12(df)
    r12_health_cost = health_cost_r12(df)

    items = [
        ("Tehran", KPI_MODAL_SHARE_PUBLIC, city_public_share),
        ("Tehran", KPI_MODAL_SHARE_CAR, df["modal_share_car"]),
        ("Tehran", KPI_VKM_PCE, df["vkm_pce_total"]),
        ("Tehran", KPI_CONGESTION, df["congestion_index"]),
        ("Tehran", KPI_TIME_LOSS_CAR, df["time_loss_car_hours_year"]),
        ("Tehran", KPI_FINAL_ENERGY, df["final_energy_MJ_year"]),
        ("Tehran", KPI_ELECTRICITY_USE, df["electricity_transport_kWh_year"]),
        ("Tehran", KPI_CO2, df["CO2_t_year"]),
        ("Tehran", KPI_PM25, city_pm25),
        ("Tehran", KPI_NOX, city_nox),
        ("Tehran", KPI_HEALTH_COST, df["cost_health_IRR_year"]),
        ("Tehran", KPI_CLIMATE_COST, df["cost_climate_IRR_year"]),
        ("Tehran", KPI_PT_TRIPS, city_pt_trips),
        ("Tehran", KPI_NON_PUBLIC_TRIPS, city_total_effective - city_pt_trips),
        ("Tehran", KPI_ENERGY_GASOLINE, city_energy_gasoline),
        ("Tehran", KPI_ENERGY_DIESEL, city_energy_diesel),
        ("Tehran", KPI_ENERGY_CNG, city_energy_cng),
        ("Tehran", KPI_TRIPS_TOTAL_EFFECTIVE, city_total_effective),
        ("Tehran", KPI_POPULATION, df["population_city"]),
        ("Tehran", KPI_MUNICIPAL_BUDGET, df["municipal_budget_IRR_year"]),
        ("Region 12", KPI_MODAL_SHARE_PUBLIC, r12_public_share),
        ("Region 12", KPI_MODAL_SHARE_CAR, df["modal_share_car_r12"]),
        ("Region 12", KPI_VKM_PCE, df["vkm_pce_total_r12"]),
        ("Region 12", KPI_CONGESTION, df["congestion_index_r12"]),
        ("Region 12", KPI_TIME_LOSS_CAR, df["time_loss_car_hours_year_r12"]),
        ("Region 12", KPI_FINAL_ENERGY, df["final_energy_MJ_year_r12"]),
        ("Region 12", KPI_ELECTRICITY_USE, df["electricity_kWh_year_r12"]),
        ("Region 12", KPI_CO2, df["CO2_t_year_r12"]),
        ("Region 12", KPI_PM25, r12_pm25),
        ("Region 12", KPI_NOX, r12_nox),
        ("Region 12", KPI_HEALTH_COST, r12_health_cost),
        ("Region 12", KPI_CLIMATE_COST, df["cost_climate_IRR_year_r12"]),
        ("Region 12", KPI_PT_TRIPS, r12_pt_trips),
        ("Region 12", KPI_NON_PUBLIC_TRIPS, r12_total_effective - r12_pt_trips),
        ("Region 12", KPI_ENERGY_GASOLINE, r12_energy_gasoline),
        ("Region 12", KPI_ENERGY_DIESEL, r12_energy_diesel),
        ("Region 12", KPI_ENERGY_CNG, r12_energy_cng),
        ("Region 12", KPI_TRIPS_TOTAL_EFFECTIVE, r12_total_effective),
        ("Region 12", KPI_POPULATION, df["pop_r12"]),
        ("Region 12", KPI_MUNICIPAL_BUDGET, df["municipal_budget_IRR_year"] * region12_budget_share),
    ]
    rows = []
    for geo, kpi, vals in items:
        for year, val in zip(years, vals):
            rows.append({"Geo": geo, "year": int(year), "kpi": kpi, "unit": UNIT_MAP[kpi], "value": float(val)})
    return rows


def main(root: Path) -> None:
    root = root.resolve()
    meta = pd.read_csv(root / "tables" / "kpi_list_main_secondary.csv")
    region12_budget_share = load_region12_budget_share(root)
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for scen in SCENARIOS:
        df = pd.read_csv(out_dir / f"simulation_data_{SCENARIO_IDENTITY[scen]}.csv")
        for row in build_rows(df, region12_budget_share):
            row["Scenario"] = scen
            rows.append(row)
    all_long = pd.DataFrame(rows)
    all_long = all_long[all_long["year"].between(2024, 2030)].copy()
    all_long = all_long[["Scenario", "year", "Geo", "kpi", "unit", "value"]]
    all_long = all_long.sort_values(["Scenario", "kpi", "Geo", "year"]).reset_index(drop=True)
    all_long.to_csv(out_dir / "timeseries_all.csv", index=False)

    long = all_long.merge(meta, on=["kpi", "unit"], how="left")
    missing_meta = sorted(long.loc[long["group"].isna(), "kpi"].dropna().unique())
    if missing_meta:
        raise ValueError(f"Missing KPI metadata rows for: {missing_meta}")
    long = long[["Scenario", "year", "Geo", "group", "kpi", "unit", "value"]]
    long = long.rename(columns={"Scenario": "scenario", "Geo": "geo"})
    long["geo"] = long["geo"].replace({"Region 12": "Region12"})
    long["Scenario"] = long["scenario"]
    long["Geo"] = long["geo"].replace({"Region12": "Region 12"})
    long.to_csv(out_dir / "kpi_timeseries_selected_long_2024_2030.csv", index=False)

    mean_df = long.groupby(["scenario", "geo", "group", "kpi", "unit"], observed=True)["value"].mean().reset_index(name="mean_2024_2030")
    mean_df.to_csv(out_dir / "kpi_selected_mean_2024_2030.csv", index=False)

    v2030 = long[long["year"] == 2030][["scenario", "geo", "group", "kpi", "unit", "value"]].rename(columns={"value": "value_2030"})
    v2030.to_csv(out_dir / "kpi_selected_2030.csv", index=False)
    print("Wrote KPI tables and timeseries_all.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Bundle root (default: current directory)")
    args = ap.parse_args()
    main(Path(args.root))
