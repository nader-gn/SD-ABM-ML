"""Run comprehensive logic checks over workflow outputs and figure inputs."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from scenario_meta import SCENARIO_CODES, SCENARIO_IDENTITY
from figure_kpi_source_utils import SELECTED24_KPIS, refresh_metric_globals
from metric_registry import core_metric_order, implementation_metric_order, selected24_metric_order, canonical_labels
from kpi_defs import (
    ALL_KPIS,
    KPI_MODAL_SHARE_CAR,
    KPI_MODAL_SHARE_PUBLIC,
    KPI_NON_PUBLIC_TRIPS,
    KPI_PT_TRIPS,
    KPI_TRIPS_TOTAL_EFFECTIVE,
    KPI_FINAL_ENERGY,
    KPI_HEALTH_COST,
    KPI_MUNICIPAL_BUDGET,
    KPI_ENERGY_GASOLINE,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_CNG,
    KPI_ELECTRICITY_USE,
    KPI_NOX,
    KPI_PM25,
    MODAL_SHARE_MODE_COLUMNS,
    MODAL_SHARE_MODE_TRUTH_COLUMNS,
)

NONNEGATIVE_KPIS = set(ALL_KPIS)
FORBIDDEN_MISLABELS = [
    "Modal share: private",
    "Modal share: public",
    "Time loss (private)",
    "Private transport trips",
]


def _status(ok: bool) -> str:
    return "ok" if ok else "fail"


def add_check(rows: list[dict], check: str, ok: bool, detail: str, value: float | None = None):
    rows.append({"check": check, "status": _status(ok), "detail": detail, "value": value})


def load_region12_budget_share(root: Path) -> float:
    cfg = yaml.safe_load((root / "config" / "analysis_overrides.yaml").read_text(encoding="utf-8"))
    share = float(cfg["region12_municipal_budget_share_of_tehran"])
    if not (0 < share < 1):
        raise ValueError(f"Invalid Region 12 budget share: {share}")
    return share


def safe_series_lookup(ts: pd.DataFrame, scen: str, year: int, geo: str, kpi: str) -> float:
    return float(ts[(ts["Scenario"] == scen) & (ts["year"] == year) & (ts["Geo"] == geo) & (ts["kpi"] == kpi)]["value"].iloc[0])


def tailpipe_vkm(vkm: pd.Series, ev_share: pd.Series) -> pd.Series:
    return vkm * (1 - ev_share)


def direct_energy_gas_city(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0)
    liters = (
        car_vkm_eff * df["fc_car_gasoline_litre_km"] * np.maximum(1 - df["share_car_EV"] - df["share_car_CNG"], 0)
        + df["vkm_taxi"] * df["fc_taxi_gasoline_litre_km"] * np.maximum(1 - df["share_taxi_EV"] - df["share_taxi_CNG"], 0)
        + df["vkm_motorcycle"] * df["fc_motorcycle_gasoline_litre_km"] * np.maximum(1 - df["share_motorcycle_EV"], 0)
    )
    return liters * df["LHV_gasoline_MJ_litre"]


def direct_energy_diesel_city(df: pd.DataFrame) -> pd.Series:
    liters = df["vkm_bus"] * df["fc_bus_diesel_litre_km"] * np.maximum(1 - df["share_bus_EV"], 0)
    return liters * df["LHV_diesel_MJ_litre"]


def direct_energy_cng_city(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0)
    kg = car_vkm_eff * df["share_car_CNG"] * df["fc_car_CNG_kg_km"] + df["vkm_taxi"] * df["share_taxi_CNG"] * df["fc_taxi_CNG_kg_km"]
    return kg * df["LHV_CNG_MJ_kg"]


def direct_energy_gas_r12(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0)
    liters = (
        car_vkm_eff * df["fc_car_gasoline_litre_km"] * np.maximum(1 - df["share_car_EV"] - df["share_car_CNG"], 0)
        + df["vkm_taxi_r12"] * df["fc_taxi_gasoline_litre_km"] * np.maximum(1 - df["share_taxi_EV"] - df["share_taxi_CNG"], 0)
        + df["vkm_motorcycle_r12"] * df["fc_motorcycle_gasoline_litre_km"] * np.maximum(1 - df["share_motorcycle_EV"], 0)
    )
    return liters * df["LHV_gasoline_MJ_litre"]


def direct_energy_diesel_r12(df: pd.DataFrame) -> pd.Series:
    liters = df["vkm_bus_r12"] * df["fc_bus_diesel_litre_km"] * np.maximum(1 - df["share_bus_EV"], 0)
    return liters * df["LHV_diesel_MJ_litre"]


def direct_energy_cng_r12(df: pd.DataFrame) -> pd.Series:
    car_vkm_eff = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0)
    kg = car_vkm_eff * df["share_car_CNG"] * df["fc_car_CNG_kg_km"] + df["vkm_taxi_r12"] * df["share_taxi_CNG"] * df["fc_taxi_CNG_kg_km"]
    return kg * df["LHV_CNG_MJ_kg"]


def direct_nox_city(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_NOx_car_g_km"] * car_tailpipe
        + df["EF_NOx_bus_g_km"] * tailpipe_vkm(df["vkm_bus"], df["share_bus_EV"])
        + df["EF_NOx_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi"], df["share_taxi_EV"])
        + df["EF_NOx_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle"], df["share_motorcycle_EV"])
        + df["EF_grid_NOx_g_kWh"] * df["electricity_transport_kWh_year"]
    )
    return total_g / 1e6


def direct_pm25_city(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_PM25_car_g_km"] * car_tailpipe
        + df["EF_PM25_bus_g_km"] * tailpipe_vkm(df["vkm_bus"], df["share_bus_EV"])
        + df["EF_PM25_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi"], df["share_taxi_EV"])
        + df["EF_PM25_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle"], df["share_motorcycle_EV"])
        + df["EF_grid_PM25_g_kWh"] * df["electricity_transport_kWh_year"]
    )
    return total_g / 1e6


def direct_nox_r12(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_NOx_car_g_km"] * car_tailpipe
        + df["EF_NOx_bus_g_km"] * tailpipe_vkm(df["vkm_bus_r12"], df["share_bus_EV"])
        + df["EF_NOx_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi_r12"], df["share_taxi_EV"])
        + df["EF_NOx_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle_r12"], df["share_motorcycle_EV"])
        + df["EF_grid_NOx_g_kWh"] * df["electricity_kWh_year_r12"]
    )
    return total_g / 1e6


def direct_pm25_r12(df: pd.DataFrame) -> pd.Series:
    car_tailpipe = df["vkm_car_r12"] * df.get("car_energy_closure_factor", 1.0) * (1 - df["share_car_EV"])
    total_g = (
        df["EF_PM25_car_g_km"] * car_tailpipe
        + df["EF_PM25_bus_g_km"] * tailpipe_vkm(df["vkm_bus_r12"], df["share_bus_EV"])
        + df["EF_PM25_taxi_g_km"] * tailpipe_vkm(df["vkm_taxi_r12"], df["share_taxi_EV"])
        + df["EF_PM25_motorcycle_g_km"] * tailpipe_vkm(df["vkm_motorcycle_r12"], df["share_motorcycle_EV"])
        + df["EF_grid_PM25_g_kWh"] * df["electricity_kWh_year_r12"]
    )
    return total_g / 1e6


def direct_health_cost_city(df: pd.DataFrame) -> pd.Series:
    af_no2 = (df["RR_NO2"] - 1) / df["RR_NO2"]
    af_pm25 = (df["RR_PM25"] - 1) / df["RR_PM25"]
    deaths = df["population_city"] * df["mortality_adult_rate"] * (af_no2 + af_pm25)
    admissions = df["population_city"] * df["respiratory_admission_rate_base"] * (np.exp(df["CR_PM25_respiratory_admission"] * df["concentration_PM25_excess_ugm3"]) - 1)
    return deaths * df["VSL_IRR"] + admissions * df["cost_hospital_day_IRR_indexed"] * 5


def direct_health_cost_r12(df: pd.DataFrame) -> pd.Series:
    no2_excess_r12 = np.maximum(df["concentration_NO2_ugm3_r12_proxy"] - df["NO2_threshold_ugm3"], 0)
    rr_no2_r12 = np.exp(df["CR_NO2_mortality"] * no2_excess_r12)
    af_no2_r12 = (rr_no2_r12 - 1) / rr_no2_r12
    pm25_conc_r12 = df["PM25_background_ugm3"] + df["k_PM25_concentration_effective_ugm3_per_t"] * df["PM25_t_year_r12"]
    pm25_excess_r12 = np.maximum(pm25_conc_r12 - df["PM25_threshold_ugm3"], 0)
    rr_pm25_r12 = np.exp(df["CR_PM25_mortality_adult"] * pm25_excess_r12)
    af_pm25_r12 = (rr_pm25_r12 - 1) / rr_pm25_r12
    deaths = df["pop_r12"] * df["mortality_adult_rate"] * (af_no2_r12 + af_pm25_r12)
    admissions = df["pop_r12"] * df["respiratory_admission_rate_base"] * (np.exp(df["CR_PM25_respiratory_admission"] * pm25_excess_r12) - 1)
    return deaths * df["VSL_IRR"] + admissions * df["cost_hospital_day_IRR_indexed"] * 5


def check_timeseries_structure(root: Path, rows: list[dict]):
    ts = pd.read_csv(root / "outputs" / "timeseries_all.csv")
    expected_rows = 12 * 7 * 2 * len(ALL_KPIS)
    add_check(rows, "timeseries_row_count", len(ts) == expected_rows, f"expected {expected_rows}, found {len(ts)}", float(len(ts)))
    key_cols = ["Scenario", "year", "Geo", "kpi"]
    dupes = int(ts.duplicated(key_cols).sum())
    add_check(rows, "timeseries_unique_keys", dupes == 0, f"duplicate keyed rows: {dupes}", float(dupes))
    expected_scenarios = set(SCENARIO_CODES)
    expected_years = set(range(2024, 2031))
    expected_geos = {"Tehran", "Region 12"}
    expected_kpis = set(ALL_KPIS)
    add_check(rows, "timeseries_scenarios_complete", set(ts["Scenario"]) == expected_scenarios, f"scenarios={sorted(set(ts['Scenario']))}")
    add_check(rows, "timeseries_years_complete", set(ts["year"]) == expected_years, f"years={sorted(set(ts['year']))}")
    add_check(rows, "timeseries_geos_complete", set(ts["Geo"]) == expected_geos, f"geos={sorted(set(ts['Geo']))}")
    add_check(rows, "timeseries_kpis_complete", set(ts["kpi"]) == expected_kpis, f"kpi_count={ts['kpi'].nunique()}")


def check_modal_partitions(root: Path, rows: list[dict]):
    max_diff = 0.0
    min_share = np.inf
    max_share = -np.inf
    for scen in range(12):
        df = pd.read_csv(root / "outputs" / f"simulation_data_SC{scen}.csv")
        for area in ["Tehran", "Region12"]:
            cols = MODAL_SHARE_MODE_COLUMNS[area]
            sub = df[list(cols.values())].copy()
            total = sub.sum(axis=1)
            max_diff = max(max_diff, float((total - 1.0).abs().max()))
            min_share = min(min_share, float(sub.min().min()))
            max_share = max(max_share, float(sub.max().max()))
    add_check(rows, "modal_share_partition_sum", np.isclose(max_diff, 0.0, atol=1e-9), f"max abs(sum-1)={max_diff:.3e}", max_diff)
    add_check(rows, "modal_share_bounds", min_share >= -1e-12 and max_share <= 1 + 1e-12, f"min={min_share:.6f}, max={max_share:.6f}")


def check_kpi_identities(root: Path, rows: list[dict]):
    ts = pd.read_csv(root / "outputs" / "timeseries_all.csv")
    max_pt_share_diff = 0.0
    max_car_share_diff = 0.0
    max_nonpublic_diff = 0.0
    max_total_diff = 0.0
    max_pt_trip_diff = 0.0
    max_energy_comp_diff = 0.0
    max_nox_diff = 0.0
    max_pm25_diff = 0.0
    max_health_cost_city_diff = 0.0
    max_health_cost_r12_diff = 0.0
    max_budget_r12_alloc_diff = 0.0
    min_nonpublic = np.inf
    energy_balance_city = 0.0
    energy_balance_r12 = 0.0
    region12_budget_share = load_region12_budget_share(root)
    for scen in range(12):
        df = pd.read_csv(root / "outputs" / f"simulation_data_SC{scen}.csv")
        years = df["YEAR_GRG"].round().astype(int)
        df = df.loc[years.between(2024, 2030)].copy().reset_index(drop=True)
        years = years[years.between(2024, 2030)].reset_index(drop=True)
        scen_label = SCENARIO_IDENTITY[f"SC{scen}"]
        city_pub = df["modal_share_bus"] + df["modal_share_metro"]
        r12_pub = df["modal_share_bus_r12"] + df["modal_share_met_r12"]
        city_total = df["trips_per_year"]
        r12_total = df["trp_r12_effective"]
        city_pt = city_total * city_pub
        r12_pt = r12_total * r12_pub
        city_g = direct_energy_gas_city(df)
        city_d = direct_energy_diesel_city(df)
        city_c = direct_energy_cng_city(df)
        r12_g = direct_energy_gas_r12(df)
        r12_d = direct_energy_diesel_r12(df)
        r12_c = direct_energy_cng_r12(df)
        city_n = direct_nox_city(df)
        city_p = direct_pm25_city(df)
        r12_n = direct_nox_r12(df)
        r12_p = direct_pm25_r12(df)
        city_h = direct_health_cost_city(df)
        r12_h = direct_health_cost_r12(df)
        r12_budget_alloc = df["municipal_budget_IRR_year"] * region12_budget_share
        energy_balance_city = max(energy_balance_city, float((df["final_energy_MJ_year"] - (city_g + city_d + city_c + df["electricity_transport_kWh_year"] * 3.6)).abs().max()))
        energy_balance_r12 = max(energy_balance_r12, float((df["final_energy_MJ_year_r12"] - (r12_g + r12_d + r12_c + df["electricity_kWh_year_r12"] * 3.6)).abs().max()))
        for year_idx, year in enumerate(years):
            lookup_city = {(k): safe_series_lookup(ts, scen_label, int(year), "Tehran", k) for k in [KPI_MODAL_SHARE_PUBLIC, KPI_MODAL_SHARE_CAR, KPI_TRIPS_TOTAL_EFFECTIVE, KPI_PT_TRIPS, KPI_NON_PUBLIC_TRIPS, KPI_ENERGY_GASOLINE, KPI_ENERGY_DIESEL, KPI_ENERGY_CNG, KPI_NOX, KPI_PM25, KPI_HEALTH_COST]}
            lookup_r12 = {(k): safe_series_lookup(ts, scen_label, int(year), "Region 12", k) for k in [KPI_MODAL_SHARE_PUBLIC, KPI_MODAL_SHARE_CAR, KPI_TRIPS_TOTAL_EFFECTIVE, KPI_PT_TRIPS, KPI_NON_PUBLIC_TRIPS, KPI_ENERGY_GASOLINE, KPI_ENERGY_DIESEL, KPI_ENERGY_CNG, KPI_NOX, KPI_PM25, KPI_HEALTH_COST, KPI_MUNICIPAL_BUDGET]}
            max_pt_share_diff = max(max_pt_share_diff, abs(lookup_city[KPI_MODAL_SHARE_PUBLIC] - float(city_pub.iloc[year_idx])), abs(lookup_r12[KPI_MODAL_SHARE_PUBLIC] - float(r12_pub.iloc[year_idx])))
            max_car_share_diff = max(max_car_share_diff, abs(lookup_city[KPI_MODAL_SHARE_CAR] - float(df.iloc[year_idx]["modal_share_car"])), abs(lookup_r12[KPI_MODAL_SHARE_CAR] - float(df.iloc[year_idx]["modal_share_car_r12"])))
            max_total_diff = max(max_total_diff, abs(lookup_city[KPI_TRIPS_TOTAL_EFFECTIVE] - float(city_total.iloc[year_idx])), abs(lookup_r12[KPI_TRIPS_TOTAL_EFFECTIVE] - float(r12_total.iloc[year_idx])))
            max_pt_trip_diff = max(max_pt_trip_diff, abs(lookup_city[KPI_PT_TRIPS] - float(city_pt.iloc[year_idx])), abs(lookup_r12[KPI_PT_TRIPS] - float(r12_pt.iloc[year_idx])))
            max_nonpublic_diff = max(max_nonpublic_diff, abs(lookup_city[KPI_NON_PUBLIC_TRIPS] - float((city_total - city_pt).iloc[year_idx])), abs(lookup_r12[KPI_NON_PUBLIC_TRIPS] - float((r12_total - r12_pt).iloc[year_idx])))
            max_energy_comp_diff = max(max_energy_comp_diff,
                abs(lookup_city[KPI_ENERGY_GASOLINE] - float(city_g.iloc[year_idx])),
                abs(lookup_city[KPI_ENERGY_DIESEL] - float(city_d.iloc[year_idx])),
                abs(lookup_city[KPI_ENERGY_CNG] - float(city_c.iloc[year_idx])),
                abs(lookup_r12[KPI_ENERGY_GASOLINE] - float(r12_g.iloc[year_idx])),
                abs(lookup_r12[KPI_ENERGY_DIESEL] - float(r12_d.iloc[year_idx])),
                abs(lookup_r12[KPI_ENERGY_CNG] - float(r12_c.iloc[year_idx])),
            )
            max_nox_diff = max(max_nox_diff, abs(lookup_city[KPI_NOX] - float(city_n.iloc[year_idx])), abs(lookup_r12[KPI_NOX] - float(r12_n.iloc[year_idx])))
            max_pm25_diff = max(max_pm25_diff, abs(lookup_city[KPI_PM25] - float(city_p.iloc[year_idx])), abs(lookup_r12[KPI_PM25] - float(r12_p.iloc[year_idx])))
            max_health_cost_city_diff = max(max_health_cost_city_diff, abs(lookup_city[KPI_HEALTH_COST] - float(city_h.iloc[year_idx])))
            max_health_cost_r12_diff = max(max_health_cost_r12_diff, abs(lookup_r12[KPI_HEALTH_COST] - float(r12_h.iloc[year_idx])))
            max_budget_r12_alloc_diff = max(max_budget_r12_alloc_diff, abs(lookup_r12[KPI_MUNICIPAL_BUDGET] - float(r12_budget_alloc.iloc[year_idx])))
            min_nonpublic = min(min_nonpublic, lookup_city[KPI_NON_PUBLIC_TRIPS], lookup_r12[KPI_NON_PUBLIC_TRIPS])
    add_check(rows, "kpi_public_share_identity", np.isclose(max_pt_share_diff, 0.0, atol=1e-9), f"max diff={max_pt_share_diff:.3e}", max_pt_share_diff)
    add_check(rows, "kpi_car_share_identity", np.isclose(max_car_share_diff, 0.0, atol=1e-9), f"max diff={max_car_share_diff:.3e}", max_car_share_diff)
    add_check(rows, "kpi_total_trips_identity", np.isclose(max_total_diff, 0.0, atol=1e-6), f"max diff={max_total_diff:.3e}", max_total_diff)
    add_check(rows, "kpi_public_trips_identity", np.isclose(max_pt_trip_diff, 0.0, atol=1e-6), f"max diff={max_pt_trip_diff:.3e}", max_pt_trip_diff)
    add_check(rows, "kpi_nonpublic_trips_identity", np.isclose(max_nonpublic_diff, 0.0, atol=1e-6), f"max diff={max_nonpublic_diff:.3e}", max_nonpublic_diff)
    add_check(rows, "kpi_energy_components_identity", np.isclose(max_energy_comp_diff, 0.0, atol=1e-4), f"max diff={max_energy_comp_diff:.3e}", max_energy_comp_diff)
    add_check(rows, "kpi_nox_identity", np.isclose(max_nox_diff, 0.0, atol=1e-9), f"max diff={max_nox_diff:.3e}", max_nox_diff)
    add_check(rows, "kpi_pm25_identity", np.isclose(max_pm25_diff, 0.0, atol=1e-9), f"max diff={max_pm25_diff:.3e}", max_pm25_diff)
    add_check(rows, "kpi_health_cost_city_identity", np.isclose(max_health_cost_city_diff, 0.0, atol=10.0), f"max diff={max_health_cost_city_diff:.3e}", max_health_cost_city_diff)
    add_check(rows, "kpi_health_cost_region12_identity", np.isclose(max_health_cost_r12_diff, 0.0, atol=0.2), f"max diff={max_health_cost_r12_diff:.3e}", max_health_cost_r12_diff)
    add_check(rows, "kpi_region12_budget_empirical_proxy_share", np.isclose(max_budget_r12_alloc_diff, 0.0, atol=0.1), f"max diff to configured 0.79% Tehran-budget share={max_budget_r12_alloc_diff:.3e}", max_budget_r12_alloc_diff)
    add_check(rows, "energy_balance_city", np.isclose(energy_balance_city, 0.0, atol=2e-4), f"max residual={energy_balance_city:.3e}", energy_balance_city)
    add_check(rows, "energy_balance_region12", np.isclose(energy_balance_r12, 0.0, atol=1e-4), f"max residual={energy_balance_r12:.3e}", energy_balance_r12)
    add_check(rows, "kpi_nonpublic_trips_nonnegative", min_nonpublic >= -1e-6, f"minimum={min_nonpublic:.3f}", min_nonpublic)


def check_selected_tables(root: Path, rows: list[dict]):
    long_df = pd.read_csv(root / "outputs" / "kpi_timeseries_selected_long_2024_2030.csv")
    mean_df = pd.read_csv(root / "outputs" / "kpi_selected_mean_2024_2030.csv")
    end_df = pd.read_csv(root / "outputs" / "kpi_selected_2030.csv")
    base = pd.read_csv(root / "outputs" / "timeseries_all.csv").rename(columns={"Scenario": "scenario", "Geo": "geo"})
    base["geo"] = base["geo"].replace({"Region 12": "Region12"})
    add_check(rows, "selected_long_matches_timeseries", len(long_df) == len(base), f"selected_long_rows={len(long_df)}, base_rows={len(base)}")
    merged = long_df.merge(base, on=["scenario", "year", "geo", "kpi", "unit"], suffixes=("_sel", "_base"), how="outer", indicator=True)
    ok_keys = bool((merged["_merge"] == "both").all())
    max_diff = float((merged["value_sel"] - merged["value_base"]).abs().max()) if ok_keys else np.nan
    add_check(rows, "selected_long_values_match", ok_keys and np.isclose(max_diff, 0.0, atol=1e-12), f"max diff={max_diff}", max_diff)
    mean_recalc = long_df.groupby(["scenario", "geo", "group", "kpi", "unit"], observed=True)["value"].mean().reset_index(name="mean_2024_2030_recalc")
    mean_merge = mean_df.merge(mean_recalc, on=["scenario", "geo", "group", "kpi", "unit"], how="outer", indicator=True)
    mean_ok = bool((mean_merge["_merge"] == "both").all())
    mean_diff = float((mean_merge["mean_2024_2030"] - mean_merge["mean_2024_2030_recalc"]).abs().max()) if mean_ok else np.nan
    mean_scale = float(mean_merge[["mean_2024_2030", "mean_2024_2030_recalc"]].abs().to_numpy().max()) if mean_ok else np.nan
    mean_rel = (mean_diff / mean_scale) if mean_ok and mean_scale else 0.0
    add_check(rows, "selected_mean_matches_long", mean_ok and mean_rel <= 1e-12, f"max diff={mean_diff}, max rel diff={mean_rel}", mean_diff)
    end_recalc = long_df[long_df["year"] == 2030].rename(columns={"value": "value_2030_recalc"})[["scenario", "geo", "group", "kpi", "unit", "value_2030_recalc"]]
    end_merge = end_df.merge(end_recalc, on=["scenario", "geo", "group", "kpi", "unit"], how="outer", indicator=True)
    end_ok = bool((end_merge["_merge"] == "both").all())
    end_diff = float((end_merge["value_2030"] - end_merge["value_2030_recalc"]).abs().max()) if end_ok else np.nan
    add_check(rows, "selected_2030_matches_long", end_ok and np.isclose(end_diff, 0.0, atol=1e-12), f"max diff={end_diff}", end_diff)


def check_auc_input(root: Path, rows: list[dict]):
    refresh_metric_globals(root)
    auc = pd.read_csv(root / "figure_inputs" / "Figure_09_auc_input.csv")
    expected_rows = len(SCENARIO_CODES) * 2 * len(SELECTED24_KPIS)
    add_check(rows, "auc_row_count", len(auc) == expected_rows, f"expected {expected_rows}, found {len(auc)}", float(len(auc)))
    bad_kpis = sorted(set(auc["kpi_display"]) & set(FORBIDDEN_MISLABELS))
    add_check(rows, "auc_no_misleading_display_labels", len(bad_kpis) == 0, f"forbidden labels={bad_kpis}")
    missing = auc["auc"].isna().sum()
    add_check(rows, "auc_no_missing_values", missing == 0, f"missing auc values={missing}", float(missing))



def check_config_metric_labels(root: Path, rows: list[dict]):
    """Verify generated tables and figure inputs use config-declared metric labels."""
    core_expected = set(core_metric_order(root))
    impl_expected = set(implementation_metric_order(root))
    selected_expected = selected24_metric_order(root)
    labels = canonical_labels(root)

    core = pd.read_csv(root / "outputs" / "core_outcome_metric_timeseries.csv")
    impl = pd.read_csv(root / "outputs" / "implementation_feasibility_metric_timeseries.csv")
    implementation = pd.read_csv(root / "outputs" / "Figure_12_implementation_scores.csv")
    auc = pd.read_csv(root / "figure_inputs" / "Figure_09_auc_input.csv")

    core_missing = sorted(core_expected - set(core["metric"]))
    impl_missing = sorted(impl_expected - set(impl["metric"]))
    implementation_missing = sorted(impl_expected - set(implementation["metric"]))
    auc_order = list(dict.fromkeys(auc.sort_values(["geo", "scenario", "kpi"])["kpi"].tolist()))
    auc_missing = sorted(set(selected_expected) - set(auc["kpi"]))

    add_check(rows, "config_core_metric_labels_materialized", not core_missing, f"missing={core_missing}", len(core_missing))
    add_check(rows, "config_implementation_metric_labels_materialized", not impl_missing, f"missing={impl_missing}", len(impl_missing))
    add_check(rows, "config_implementation_metric_labels_materialized", not implementation_missing, f"missing={implementation_missing}", len(implementation_missing))
    add_check(rows, "config_auc_metric_labels_materialized", not auc_missing, f"missing={auc_missing}", len(auc_missing))
    add_check(rows, "config_opex_canonical_labels_present", labels.get("pt_opex") in set(core["metric"]) and labels.get("net_recurrent_public_burden") in set(core["metric"]), f"pt_opex={labels.get('pt_opex')}, net_recurrent={labels.get('net_recurrent_public_burden')}")

def check_modal_share_plot_input(root: Path, rows: list[dict]):
    plot = pd.read_csv(root / "figure_inputs" / "Figure_04_modal_share_series.csv")
    sim_max_diff = 0.0
    obs_max_diff = 0.0
    for scen in range(12):
        df = pd.read_csv(root / "outputs" / f"simulation_data_SC{scen}.csv")
        years = df["YEAR_GRG"].round().astype(int)
        for area, cols in MODAL_SHARE_MODE_COLUMNS.items():
            for mode, col in cols.items():
                plot_vals = plot[(plot["area"] == area) & (plot["scenario"] == f"s{scen}") & (plot["series"] == "sim") & (plot["mode"] == mode)].sort_values("year")["value"].reset_index(drop=True)
                raw_vals = df.loc[years.between(2012, 2030), col].reset_index(drop=True)
                sim_max_diff = max(sim_max_diff, float((plot_vals - raw_vals).abs().max()))
    sc0 = pd.read_csv(root / "outputs" / "simulation_data_SC0.csv")
    years0 = sc0["YEAR_GRG"].round().astype(int)
    for area, cols in MODAL_SHARE_MODE_TRUTH_COLUMNS.items():
        for mode, col in cols.items():
            plot_vals = plot[(plot["area"] == area) & (plot["scenario"] == "obs") & (plot["series"] == "obs") & (plot["mode"] == mode)].sort_values("year")["value"].reset_index(drop=True)
            raw_vals = sc0.loc[years0.between(2012, 2023), col].dropna().reset_index(drop=True)
            obs_max_diff = max(obs_max_diff, float((plot_vals - raw_vals).abs().max()))
    add_check(rows, "modal_share_plot_input_sim_sync", np.isclose(sim_max_diff, 0.0, atol=1e-12), f"max diff={sim_max_diff:.3e}", sim_max_diff)
    add_check(rows, "modal_share_plot_input_obs_sync", np.isclose(obs_max_diff, 0.0, atol=1e-12), f"max diff={obs_max_diff:.3e}", obs_max_diff)


def check_nonnegative_outputs(root: Path, rows: list[dict]):
    ts = pd.read_csv(root / "outputs" / "timeseries_all.csv")
    sub = ts[ts["kpi"].isin(NONNEGATIVE_KPIS)]
    min_val = float(sub["value"].min())
    add_check(rows, "nonnegative_kpis_nonnegative", min_val >= -1e-9, f"minimum across nonnegative KPIs={min_val:.6f}", min_val)


def check_no_active_mislabels(root: Path, rows: list[dict]):
    found = []
    ts = pd.read_csv(root / "outputs" / "timeseries_all.csv")
    auc = pd.read_csv(root / "figure_inputs" / "Figure_09_auc_input.csv")
    active_labels = set(ts["kpi"].astype(str)) | set(auc["kpi_display"].astype(str))
    for label in FORBIDDEN_MISLABELS:
        if label in active_labels:
            found.append(label)
    add_check(rows, "no_active_misleading_labels", len(found) == 0, f"found={found}")


def write_summary(root: Path, checks: pd.DataFrame):
    passed = int(checks["status"].eq("ok").sum())
    total = len(checks)
    lines = [
        "# Logic audit summary",
        f"- Checks passing: **{passed}/{total}**.",
    ]
    failures = checks[checks["status"] != "ok"]
    if failures.empty:
        lines.append("- No logic inconsistencies were detected in workflow outputs or figure inputs.")
    else:
        lines.append("\n## Failing checks")
        lines.append(failures[["check", "detail", "value"]].to_markdown(index=False))
    (root / "verification" / "logic_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(root: Path) -> None:
    root = root.resolve()
    (root / "verification").mkdir(exist_ok=True)
    rows: list[dict] = []
    check_timeseries_structure(root, rows)
    check_modal_partitions(root, rows)
    check_kpi_identities(root, rows)
    check_selected_tables(root, rows)
    check_auc_input(root, rows)
    check_config_metric_labels(root, rows)
    check_modal_share_plot_input(root, rows)
    check_nonnegative_outputs(root, rows)
    check_no_active_mislabels(root, rows)
    out = pd.DataFrame(rows)
    out.to_csv(root / "verification" / "logic_audit_checks.csv", index=False)
    write_summary(root, out)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    args = ap.parse_args(); main(Path(args.root))
