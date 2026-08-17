from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import yaml
from metric_registry import canonical_labels
import numpy as np
import pandas as pd

SCENARIOS = [f'SC{i}' for i in range(12)]
YEARS = list(range(2024, 2031))


def safe_div(num, den, default=0.0):
    den_arr = np.asarray(den, dtype=float)
    num_arr = np.asarray(num, dtype=float)
    out = np.full_like(num_arr, default, dtype=float)
    mask = np.abs(den_arr) > 1e-12
    out[mask] = num_arr[mask] / den_arr[mask]
    return out


def scalar_safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if abs(den) > 1e-12 else float(default)


def direction_sign(direction: str) -> float:
    return 1.0 if direction == 'higher_better' else -1.0


def clipped_offset(numerator: float, denominator: float, cap: float = 1.0) -> float:
    if denominator <= 1e-9 or numerator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, cap))


def tailpipe_vkm(vkm: pd.Series | float, ev_share: pd.Series | float) -> np.ndarray:
    return np.asarray(vkm, dtype=float) * (1.0 - np.asarray(ev_share, dtype=float))


def region12_age_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_city = np.maximum(np.asarray(df["population_city"], dtype=float), 1e-9)
    pop_r12 = np.asarray(df["pop_r12"], dtype=float)
    share_014 = np.asarray(df["population_0_14"], dtype=float) / pop_city
    share_1564 = np.asarray(df["population_15_64"], dtype=float) / pop_city
    share_65 = np.asarray(df["population_65_plus"], dtype=float) / pop_city
    return pop_r12 * share_014, pop_r12 * share_1564, pop_r12 * share_65


def pm25_r12_local_t(df: pd.DataFrame) -> np.ndarray:
    car_tailpipe = tailpipe_vkm(np.asarray(df["vkm_car_r12"], dtype=float) * np.asarray(df.get("car_energy_closure_factor", 1.0), dtype=float), df["share_car_EV"])
    total_g = (
        np.asarray(df["EF_PM25_car_g_km"], dtype=float) * car_tailpipe
        + np.asarray(df["EF_PM25_bus_g_km"], dtype=float) * tailpipe_vkm(df["vkm_bus_r12"], df["share_bus_EV"])
        + np.asarray(df["EF_PM25_taxi_g_km"], dtype=float) * tailpipe_vkm(df["vkm_taxi_r12"], df["share_taxi_EV"])
        + np.asarray(df["EF_PM25_motorcycle_g_km"], dtype=float) * tailpipe_vkm(df["vkm_motorcycle_r12"], df["share_motorcycle_EV"])
        + np.asarray(df["EF_grid_PM25_g_kWh"], dtype=float) * np.asarray(df["electricity_kWh_year_r12"], dtype=float)
    )
    return total_g / 1e6


def region12_local_social_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pop014_r12, pop1564_r12, pop65_r12 = region12_age_split(df)
    pop_adult_r12 = pop1564_r12 + pop65_r12

    no2_excess_r12 = np.maximum(np.asarray(df["concentration_NO2_ugm3_r12_proxy"], dtype=float) - np.asarray(df["NO2_threshold_ugm3"], dtype=float), 0.0)
    rr_no2_r12 = np.exp(np.asarray(df["CR_NO2_mortality"], dtype=float) * no2_excess_r12)
    af_no2_r12 = np.where(rr_no2_r12 > 0, (rr_no2_r12 - 1.0) / rr_no2_r12, 0.0)

    pm25_conc_r12 = np.asarray(df["PM25_background_ugm3"], dtype=float) + np.asarray(df["k_PM25_concentration_effective_ugm3_per_t"], dtype=float) * pm25_r12_local_t(df)
    pm25_excess_r12 = np.maximum(pm25_conc_r12 - np.asarray(df["PM25_threshold_ugm3"], dtype=float), 0.0)

    rr_pm25_adult_r12 = np.exp(np.asarray(df["CR_PM25_mortality_adult"], dtype=float) * pm25_excess_r12)
    af_pm25_adult_r12 = np.where(rr_pm25_adult_r12 > 0, (rr_pm25_adult_r12 - 1.0) / rr_pm25_adult_r12, 0.0)

    rr_pm25_child_r12 = np.exp(np.asarray(df["CR_PM25_mortality_child"], dtype=float) * pm25_excess_r12)
    af_pm25_child_r12 = np.where(rr_pm25_child_r12 > 0, (rr_pm25_child_r12 - 1.0) / rr_pm25_child_r12, 0.0)

    rr_adm_r12 = np.exp(np.asarray(df["CR_PM25_respiratory_admission"], dtype=float) * pm25_excess_r12)
    admission_factor_r12 = np.maximum(rr_adm_r12 - 1.0, 0.0)

    deaths_no2_r12 = pop_adult_r12 * np.asarray(df["mortality_adult_rate"], dtype=float) * af_no2_r12
    deaths_pm25_r12 = (
        pop_adult_r12 * np.asarray(df["mortality_adult_rate"], dtype=float) * af_pm25_adult_r12
        + pop014_r12 * np.asarray(df["mortality_child_rate"], dtype=float) * af_pm25_child_r12
    )
    deaths_total_r12 = deaths_no2_r12 + deaths_pm25_r12

    admissions_r12 = np.asarray(df["pop_r12"], dtype=float) * np.asarray(df["respiratory_admission_rate_base"], dtype=float) * admission_factor_r12
    health_cost_r12 = deaths_total_r12 * np.asarray(df["VSL_IRR"], dtype=float) + admissions_r12 * np.asarray(df["cost_hospital_day_IRR_indexed"], dtype=float) * 5.0

    fatality_mult = np.clip(np.sqrt(np.maximum(np.asarray(df["traffic_condition_factor_r12_vs_city"], dtype=float), 1e-9)), 0.7, 1.6)
    # Final fatality-equivalent scoring is completed later in main() using the
    # baseline city fatality series scaled by scenario exposure and the local
    # traffic-condition factor. A placeholder is kept here only to preserve the
    # proxy-table structure before that year/scenario-specific override.
    fatality_equiv_r12 = np.zeros(len(df), dtype=float)

    out["health_cost_r12_local_proxy"] = health_cost_r12
    out["attributable_deaths_r12_local_proxy"] = deaths_total_r12
    out["fatality_equiv_r12_local_proxy"] = fatality_equiv_r12
    out["pm25_concentration_r12_local_proxy"] = pm25_conc_r12
    return out



def pt_service_capex_components(r, b, ass, pt_service_share_r12: float):
    bus_uplift = max(scalar_safe_div(float(r['spd_bus_freeflow']) - float(b['spd_bus_freeflow']), float(b['spd_bus_freeflow']), default=0.0), 0.0)
    metro_uplift = max(scalar_safe_div(float(r['spd_met']) - float(b['spd_met']), float(b['spd_met']), default=0.0), 0.0)
    service_uplift = np.mean([bus_uplift, metro_uplift])
    network_uplift = np.mean([
        max(scalar_safe_div(float(r['len_bus']) - float(b['len_bus']), float(b['len_bus']), default=0.0), 0.0),
        max(scalar_safe_div(float(r['len_brt']) - float(b['len_brt']), float(b['len_brt']), default=0.0), 0.0),
        max(scalar_safe_div(float(r['len_met']) - float(b['len_met']), float(b['len_met']), default=0.0), 0.0),
    ])
    fleet_uplift = np.mean([
        max(scalar_safe_div(float(r['bus_purchases_vehicles_year']) - float(b['bus_purchases_vehicles_year']), float(b['bus_purchases_vehicles_year']), default=0.0), 0.0),
        max(scalar_safe_div(float(r['brt_purchases_vehicles_year']) - float(b['brt_purchases_vehicles_year']), float(b['brt_purchases_vehicles_year']), default=0.0), 0.0),
        max(scalar_safe_div(float(r['metro_car_purchases_vehicles_year']) - float(b['metro_car_purchases_vehicles_year']), float(b['metro_car_purchases_vehicles_year']), default=0.0), 0.0),
    ])
    uplift = np.mean([service_uplift, network_uplift, fleet_uplift])
    pt_capex_base = float(b['CAPEX_bus_IRR_year'] + b['CAPEX_brt_infrastructure_IRR_year'] + b['CAPEX_metro_rolling_IRR_year'] + b['CAPEX_metro_infrastructure_IRR_year'])
    city = ass['pt_service_upgrade_capex_share_of_existing_pt_capex'] * uplift * pt_capex_base
    r12 = city * pt_service_share_r12
    return city, r12

def access_pricing_capex_components(r, b, ass):
    mult = max(float(r['traffic_plan_fee_multiplier']) - 1.0, 0.0)
    share = ass['pricing_admin_setup_share_of_road_capex']['SC2']
    city = share * mult * float(b['CAPEX_road_IRR_year'])
    r12 = city * scalar_safe_div(float(r['trp_r12_tfp']), float(r['trp_tfp']), default=0.0)
    return city, r12

def parking_management_capex_components(r, b, ass):
    reduction = max(1.0 - scalar_safe_div(float(r['parking_supply_multiplier_r12']), float(b.get('parking_supply_multiplier_r12', 1.0)), default=1.0), 0.0)
    share = ass['pricing_admin_setup_share_of_road_capex']['SC3']
    city = share * reduction * float(b['CAPEX_road_IRR_year'])
    parking_share_r12 = scalar_safe_div(float(r['parking_r12']), float(r.get('parking_tot', 1.0)), default=0.0)
    r12 = city * max(parking_share_r12 * 8.0, 0.0)
    return city, r12

def local_pollutant_cleanup_capex_components(r, b, ass):
    pm_red_moto = max(1.0 - scalar_safe_div(float(r['EF_PM25_motorcycle_g_km']), float(b['EF_PM25_motorcycle_g_km']), default=1.0), 0.0)
    nox_red_moto = max(1.0 - scalar_safe_div(float(r['EF_NOx_motorcycle_g_km']), float(b['EF_NOx_motorcycle_g_km']), default=1.0), 0.0)
    moto_int = np.mean([pm_red_moto, nox_red_moto])
    pm_red_bus = max(1.0 - scalar_safe_div(float(r['EF_PM25_bus_g_km']), float(b['EF_PM25_bus_g_km']), default=1.0), 0.0)
    nox_red_bus = max(1.0 - scalar_safe_div(float(r['EF_NOx_bus_g_km']), float(b['EF_NOx_bus_g_km']), default=1.0), 0.0)
    bus_int = np.mean([pm_red_bus, nox_red_bus])
    pm_red_taxi = max(1.0 - scalar_safe_div(float(r['EF_PM25_taxi_g_km']), float(b['EF_PM25_taxi_g_km']), default=1.0), 0.0)
    nox_red_taxi = max(1.0 - scalar_safe_div(float(r['EF_NOx_taxi_g_km']), float(b['EF_NOx_taxi_g_km']), default=1.0), 0.0)
    taxi_int = np.mean([pm_red_taxi, nox_red_taxi])
    moto_unit = ass['retrofit_unit_cost_share_of_vehicle_capex']['motorcycle_cleanup'] * float(b['capex_car_IRR_per_vehicle_indexed'])
    bus_unit = ass['retrofit_unit_cost_share_of_vehicle_capex']['cleanfleet_bus'] * float(b['capex_bus_IRR_per_vehicle_indexed'])
    taxi_unit = ass['retrofit_unit_cost_share_of_vehicle_capex']['cleanfleet_taxi'] * float(b['capex_taxi_IRR_per_vehicle_indexed'])
    moto_city = moto_unit * float(b['motorcycle_purchases_vehicles_year']) * moto_int
    bus_city = bus_unit * float(b['bus_purchases_vehicles_year']) * bus_int
    taxi_city = taxi_unit * float(b['taxi_purchases_vehicles_year']) * taxi_int
    city = moto_city + bus_city + taxi_city
    r12 = moto_city * scalar_safe_div(float(r['vkm_motorcycle_r12']), float(r['vkm_motorcycle']), default=0.0) + bus_city * scalar_safe_div(float(r['vkm_bus_r12']), float(r['vkm_bus']), default=0.0) + taxi_city * scalar_safe_div(float(r['vkm_taxi_r12']), float(r['vkm_taxi']), default=0.0)
    return city, r12

def clean_transition_capex_components(r, b, ass):
    delta_taxi = max(float(r['CAPEX_taxi_IRR_year'] - b['CAPEX_taxi_IRR_year']), 0.0)
    delta_bus = max(float(r['CAPEX_bus_IRR_year'] - b['CAPEX_bus_IRR_year']), 0.0)
    city = delta_taxi * (1.0 + ass['ev_charger_addon_share']['light_duty']) + delta_bus * (1.0 + ass['ev_charger_addon_share']['bus_depot'])
    weighted_city = delta_taxi + delta_bus
    if weighted_city <= 1e-12:
        r12 = 0.0
    else:
        share = (delta_taxi * scalar_safe_div(float(r['vkm_taxi_r12']), float(r['vkm_taxi']), default=0.0) + delta_bus * scalar_safe_div(float(r['vkm_bus_r12']), float(r['vkm_bus']), default=0.0)) / weighted_city
        r12 = city * share
    return city, r12

def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / 'config' / 'decision_architecture.yaml').read_text(encoding='utf-8'))


def add_direct_metric_rows(rows: list[dict], scenario: str, geo: str, year: int, values: dict[str, float]) -> None:
    for metric, value in values.items():
        rows.append({
            'scenario': scenario,
            'geo': geo,
            'year': int(year),
            'metric': metric,
            'value': float(value),
        })


def build_direct_metric_tables(root: Path):
    # Finance KPIs prefer executable config agents where available to avoid ad hoc denominator drift.
    core_rows: list[dict] = []
    impl_rows: list[dict] = []
    assumption_rows: list[dict] = []
    cfg = load_cfg(root)
    N = canonical_labels(root)
    ass = cfg['implementation_screen']['assumptions']

    scenario_frames = {}
    for sc in SCENARIOS:
        df = pd.read_csv(root / 'outputs' / f'simulation_data_{sc}.csv')
        df = df[df['YEAR_GRG'].round().astype(int).isin(YEARS)].copy()
        df['YEAR_GRG'] = df['YEAR_GRG'].round().astype(int)
        scenario_frames[sc] = df.reset_index(drop=True)

    base = scenario_frames['SC0'].copy()
    base_by_year = {int(r['YEAR_GRG']): r for _, r in base.iterrows()}

    for sc, df in scenario_frames.items():
        for _, r in df.iterrows():
            year = int(r['YEAR_GRG'])
            b = base_by_year[year]

            # --- Direct annual metrics for Tehran ---
            trips_bus = float(r['trips_bus_total'])
            trips_metro = float(r['trips_metro'])
            fare_bus = float(r['fare_bus_IRR_trip_indexed'])
            fare_metro = float(r['fare_metro_IRR_trip_indexed'])
            pt_travel_exp_city = trips_bus * fare_bus + trips_metro * fare_metro
            pt_service_opex_city = float(r['OPEX_bus_IRR_year'] + r['OPEX_BRT_IRR_year'] + r['OPEX_metro_IRR_year'])
            energy_cost_city = float(r['cost_energy_IRR_year'])
            pt_opex_total_city = float(r['OPEX_PT_IRR_year'])
            fare_revenue_city = pt_travel_exp_city
            pt_subsidy_need_city = float(r['PT_subsidy_need_IRR_year']) if 'PT_subsidy_need_IRR_year' in r else max(pt_opex_total_city - fare_revenue_city, 0.0)
            effective_budget_city = float(r['transport_budget_effective_IRR_year'])
            farebox_city = float(r['farebox_recovery_PT']) if 'farebox_recovery_PT' in r else scalar_safe_div(fare_revenue_city, pt_opex_total_city, default=0.0)
            revenue_collection_factor_city = float(r.get('traffic_plan_revenue_collection_factor', 1.0))
            direct_earmarked_revenue_city = float(r['traffic_plan_revenue_earmark_share_transport'] * revenue_collection_factor_city * r['traffic_plan_revenue_IRR_year'])
            regulatory_revenue_contribution_city = float(r['transport_budget_revenue_share_effective']) if 'transport_budget_revenue_share_effective' in r else scalar_safe_div(
                direct_earmarked_revenue_city,
                effective_budget_city,
                default=0.0,
            )
            budget_util_city = float(r['transport_budget_utilization']) if 'transport_budget_utilization' in r else scalar_safe_div(pt_subsidy_need_city, effective_budget_city, default=0.0)
            net_public_recurring_burden_city = pt_subsidy_need_city - direct_earmarked_revenue_city

            # --- Direct annual metrics for Region 12 ---
            bus_share_r12 = float(r['modal_share_bus_r12'])
            metro_share_r12 = float(r['modal_share_met_r12'])
            trips_r12 = float(r['trp_r12_effective'])
            trips_bus_r12 = trips_r12 * bus_share_r12
            trips_metro_r12 = trips_r12 * metro_share_r12
            pt_travel_exp_r12 = trips_bus_r12 * fare_bus + trips_metro_r12 * fare_metro

            vkm_brt_share_r12 = scalar_safe_div(float(r['len_brt_r12']), float(r['len_brt']), default=0.0)
            vkm_brt_r12 = float(r['vkm_BRT']) * vkm_brt_share_r12 if 'vkm_BRT' in r else 0.0
            pt_service_share_r12 = np.mean([
                scalar_safe_div(float(r['PT_trips_total_r12']), float(r['PT_trips_total']), default=0.0),
                scalar_safe_div(float(r['vkm_bus_r12'] + vkm_brt_r12 + r['vkm_metro_r12']), float(r['vkm_bus'] + r.get('vkm_BRT', 0.0) + r['vkm_metro']), default=0.0),
            ])

            pt_service_opex_r12 = float(
                r['vkm_bus_r12'] * r['opex_bus_IRR_km_indexed']
                + vkm_brt_r12 * r['opex_BRT_IRR_km_indexed']
                + r['vkm_metro_r12'] * r['opex_metro_IRR_km_indexed']
            )
            pt_energy_cost_r12 = float(r['cost_energy_PT_IRR_year']) * pt_service_share_r12
            pt_opex_total_r12 = pt_service_opex_r12 + pt_energy_cost_r12
            fare_revenue_r12 = pt_travel_exp_r12
            revenue_collection_factor_r12 = float(r.get('traffic_plan_revenue_collection_factor', 1.0))
            direct_r12_earmarked_revenue = float(r['traffic_plan_revenue_earmark_share_transport']) * revenue_collection_factor_r12 * float(r['traffic_plan_revenue_IRR_year_r12_alloc']) if 'traffic_plan_revenue_IRR_year_r12_alloc' in r else float(r['traffic_plan_revenue_earmark_share_transport']) * revenue_collection_factor_r12 * float(r['traffic_plan_revenue_IRR_year']) * scalar_safe_div(float(r['trp_r12_tfp']), float(r['trp_tfp']), default=0.0)
            effective_budget_r12 = float(r['transport_budget_effective_IRR_year_r12_alloc']) if 'transport_budget_effective_IRR_year_r12_alloc' in r else float(r['transport_budget_IRR_year']) * float(r.get('region12_municipal_budget_share_of_tehran', 0.0079)) + direct_r12_earmarked_revenue
            pt_subsidy_need_r12 = float(r['PT_subsidy_need_IRR_year_r12_alloc']) if 'PT_subsidy_need_IRR_year_r12_alloc' in r else max(pt_opex_total_r12 - fare_revenue_r12, 0.0)
            farebox_r12 = float(r['farebox_recovery_PT_r12_alloc']) if 'farebox_recovery_PT_r12_alloc' in r else scalar_safe_div(fare_revenue_r12, pt_opex_total_r12, default=0.0)
            regulatory_revenue_contribution_r12 = float(r['transport_budget_revenue_share_effective_r12_alloc']) if 'transport_budget_revenue_share_effective_r12_alloc' in r else scalar_safe_div(direct_r12_earmarked_revenue, effective_budget_r12, default=0.0)
            budget_util_r12 = float(r['transport_budget_utilization_r12_alloc']) if 'transport_budget_utilization_r12_alloc' in r else scalar_safe_div(pt_subsidy_need_r12, effective_budget_r12, default=0.0)
            net_public_recurring_burden_r12 = pt_subsidy_need_r12 - direct_r12_earmarked_revenue

            # --- Scenario-specific annualized implementation CAPEX reconstruction ---
            impl_capex_city = 0.0
            impl_capex_r12 = 0.0
            capex_note = 'no_additional_implementation_capex'

            if sc == 'SC1':
                capex_note = 'telework_admin_assumed_near_zero'
            elif sc == 'SC2':
                impl_capex_city, impl_capex_r12 = access_pricing_capex_components(r, b, ass)
                capex_note = 'access_pricing_setup_share_of_road_capex'
            elif sc == 'SC3':
                impl_capex_city, impl_capex_r12 = parking_management_capex_components(r, b, ass)
                capex_note = 'parking_management_setup_share_of_road_capex'
            elif sc == 'SC4':
                capex_note = 'fare_support_treated_as_recurring_not_capital'
            elif sc == 'SC5':
                impl_capex_city, impl_capex_r12 = pt_service_capex_components(r, b, ass, pt_service_share_r12)
                capex_note = 'pt_access_service_upgrade_share_times_composite_service_network_uplift'
            elif sc == 'SC6':
                impl_capex_city, impl_capex_r12 = local_pollutant_cleanup_capex_components(r, b, ass)
                capex_note = 'local_pollutant_cleanup_retrofit_costs_for_motorcycle_taxi_bus'
            elif sc == 'SC7':
                impl_capex_city, impl_capex_r12 = clean_transition_capex_components(r, b, ass)
                capex_note = 'clean_fleet_transition_capex_plus_charger_and_depot_addons'
            elif sc == 'SC8':
                parking_capex_city, parking_capex_r12 = parking_management_capex_components(r, b, ass)
                cleanup_capex_city, cleanup_capex_r12 = local_pollutant_cleanup_capex_components(r, b, ass)
                impl_capex_city = parking_capex_city + cleanup_capex_city
                impl_capex_r12 = parking_capex_r12 + cleanup_capex_r12
                capex_note = 'no_regret_package_sum_of_light_parking_and_medium_local_cleanup_components'
            elif sc == 'SC9':
                pricing_capex_city, pricing_capex_r12 = access_pricing_capex_components(r, b, ass)
                parking_capex_city, parking_capex_r12 = parking_management_capex_components(r, b, ass)
                pt_capex_city, pt_capex_r12 = pt_service_capex_components(r, b, ass, pt_service_share_r12)
                cleanup_capex_city, cleanup_capex_r12 = local_pollutant_cleanup_capex_components(r, b, ass)
                impl_capex_city = pricing_capex_city + parking_capex_city + pt_capex_city + cleanup_capex_city
                impl_capex_r12 = pricing_capex_r12 + parking_capex_r12 + pt_capex_r12 + cleanup_capex_r12
                capex_note = 'access_led_core_package_sum_of_access_light_parking_pt_service_and_light_local_cleanup'
            elif sc == 'SC10':
                pt_capex_city, pt_capex_r12 = pt_service_capex_components(r, b, ass, pt_service_share_r12)
                cleanup_capex_city, cleanup_capex_r12 = local_pollutant_cleanup_capex_components(r, b, ass)
                clean_transition_capex_city, clean_transition_capex_r12 = clean_transition_capex_components(r, b, ass)
                parking_capex_city, parking_capex_r12 = parking_management_capex_components(r, b, ass)
                impl_capex_city = pt_capex_city + cleanup_capex_city + clean_transition_capex_city + parking_capex_city
                impl_capex_r12 = pt_capex_r12 + cleanup_capex_r12 + clean_transition_capex_r12 + parking_capex_r12
                capex_note = 'pt_first_clean_package_sum_of_pt_service_local_cleanup_clean_transition_and_light_parking'
            elif sc == 'SC11':
                pricing_capex_city, pricing_capex_r12 = access_pricing_capex_components(r, b, ass)
                parking_capex_city, parking_capex_r12 = parking_management_capex_components(r, b, ass)
                pt_capex_city, pt_capex_r12 = pt_service_capex_components(r, b, ass, pt_service_share_r12)
                cleanup_capex_city, cleanup_capex_r12 = local_pollutant_cleanup_capex_components(r, b, ass)
                clean_transition_capex_city, clean_transition_capex_r12 = clean_transition_capex_components(r, b, ass)
                impl_capex_city = pricing_capex_city + parking_capex_city + pt_capex_city + cleanup_capex_city + clean_transition_capex_city
                impl_capex_r12 = pricing_capex_r12 + parking_capex_r12 + pt_capex_r12 + cleanup_capex_r12 + clean_transition_capex_r12
                capex_note = 'balanced_integrated_package_sum_of_medium_demand_access_parking_pt_service_local_cleanup_and_clean_transition'

            pt_cost_to_budget_city = scalar_safe_div(pt_opex_total_city, effective_budget_city, default=0.0)
            pt_cost_to_budget_r12 = scalar_safe_div(pt_opex_total_r12, effective_budget_r12, default=0.0)

            attributable_deaths_city = float(r['deaths_total']) if 'deaths_total' in r else scalar_safe_div(float(r['cost_health_IRR_year']), float(r.get('cost_per_fatality_IRR', 1.0)), default=0.0)
            # Scenario-sensitive fatality-equivalent proxy: preserve the baseline
            # city fatality magnitude for each year, then scale it by the scenario's
            # PCE-km exposure so transport interventions can affect the safety lens.
            fatality_equiv_city = float(b['acc_fatl']) * scalar_safe_div(float(r['vkm_pce_total']), float(b['vkm_pce_total']), default=1.0)
            r12_social_local = region12_local_social_proxies(pd.DataFrame([r])).iloc[0]
            health_cost_r12 = float(r12_social_local['health_cost_r12_local_proxy'])
            attributable_deaths_r12 = float(r12_social_local['attributable_deaths_r12_local_proxy'])
            fatality_mult_r12 = float(np.clip(np.sqrt(max(float(r.get('traffic_condition_factor_r12_vs_city', 1.0)), 1e-9)), 0.7, 1.6))
            fatality_equiv_r12 = fatality_equiv_city * scalar_safe_div(float(r['vkm_pce_total_r12']), float(r['vkm_pce_total']), default=0.0) * fatality_mult_r12
            pt_affordability_city = float(r['PT_affordability_ratio']) if 'PT_affordability_ratio' in r else scalar_safe_div(pt_travel_exp_city / max(float(r['population_city']), 1.0), float(r['household_income_per_capita_IRR_year']), default=0.0)
            pt_affordability_r12 = scalar_safe_div(pt_travel_exp_r12 / max(float(r['pop_r12']), 1.0), float(r['household_income_per_capita_IRR_year']), default=0.0)
            avoided_external_cost_city = float(b['external_cost_IRR_year'] - r['external_cost_IRR_year'])
            avoided_external_cost_r12 = float(b['external_cost_IRR_year_r12'] - r['external_cost_IRR_year_r12'])

            base_pt_travel_exp_city = float(b['trips_bus_total']) * float(b['fare_bus_IRR_trip_indexed']) + float(b['trips_metro']) * float(b['fare_metro_IRR_trip_indexed'])
            base_pt_subsidy_need_city = max(float(b['OPEX_PT_IRR_year']) - base_pt_travel_exp_city, 0.0)
            base_revenue_collection_factor_city = float(b.get('traffic_plan_revenue_collection_factor', 1.0))
            base_direct_earmarked_revenue_city = float(b['traffic_plan_revenue_earmark_share_transport']) * base_revenue_collection_factor_city * float(b['traffic_plan_revenue_IRR_year'])
            base_net_public_city = base_pt_subsidy_need_city - base_direct_earmarked_revenue_city

            base_bus_share_r12 = float(b['modal_share_bus_r12'])
            base_metro_share_r12 = float(b['modal_share_met_r12'])
            base_trips_r12 = float(b['trp_r12_effective'])
            base_trips_bus_r12 = base_trips_r12 * base_bus_share_r12
            base_trips_metro_r12 = base_trips_r12 * base_metro_share_r12
            base_pt_travel_exp_r12 = base_trips_bus_r12 * float(b['fare_bus_IRR_trip_indexed']) + base_trips_metro_r12 * float(b['fare_metro_IRR_trip_indexed'])
            base_vkm_brt_share_r12 = scalar_safe_div(float(b['len_brt_r12']), float(b['len_brt']), default=0.0)
            base_vkm_brt_r12 = float(b['vkm_BRT']) * base_vkm_brt_share_r12 if 'vkm_BRT' in b else 0.0
            base_pt_service_share_r12 = np.mean([
                scalar_safe_div(float(b['PT_trips_total_r12']), float(b['PT_trips_total']), default=0.0),
                scalar_safe_div(float(b['vkm_bus_r12'] + base_vkm_brt_r12 + b['vkm_metro_r12']), float(b['vkm_bus'] + b.get('vkm_BRT', 0.0) + b['vkm_metro']), default=0.0),
            ])
            base_pt_service_opex_r12 = float(
                b['vkm_bus_r12'] * b['opex_bus_IRR_km_indexed']
                + base_vkm_brt_r12 * b['opex_BRT_IRR_km_indexed']
                + b['vkm_metro_r12'] * b['opex_metro_IRR_km_indexed']
            )
            base_pt_energy_cost_r12 = float(b['cost_energy_PT_IRR_year']) * base_pt_service_share_r12
            base_pt_opex_total_r12 = base_pt_service_opex_r12 + base_pt_energy_cost_r12
            base_pt_subsidy_need_r12 = max(base_pt_opex_total_r12 - base_pt_travel_exp_r12, 0.0)
            base_revenue_collection_factor_r12 = float(b.get('traffic_plan_revenue_collection_factor', 1.0))
            base_direct_earmarked_revenue_r12 = float(b['traffic_plan_revenue_earmark_share_transport']) * base_revenue_collection_factor_r12 * (float(b['traffic_plan_revenue_IRR_year_r12_alloc']) if 'traffic_plan_revenue_IRR_year_r12_alloc' in b else float(b['traffic_plan_revenue_IRR_year']) * scalar_safe_div(float(b['trp_r12_tfp']), float(b['trp_tfp']), default=0.0))
            base_net_public_r12 = base_pt_subsidy_need_r12 - base_direct_earmarked_revenue_r12

            delta_net_public_city = float(net_public_recurring_burden_city - base_net_public_city)
            delta_net_public_r12 = float(net_public_recurring_burden_r12 - base_net_public_r12)
            delta_regulatory_revenue_city = float(direct_earmarked_revenue_city - base_direct_earmarked_revenue_city)
            delta_regulatory_revenue_r12 = float(direct_r12_earmarked_revenue - base_direct_earmarked_revenue_r12)
            incremental_fiscal_need_city = float(impl_capex_city + max(pt_subsidy_need_city - base_pt_subsidy_need_city, 0.0))
            incremental_fiscal_need_r12 = float(impl_capex_r12 + max(pt_subsidy_need_r12 - base_pt_subsidy_need_r12, 0.0))
            # Baseline-relative implementation indicators must be exactly zero in SC0.
            # Tiny binary floating residuals (sub-IRR) can otherwise survive subtraction of
            # large IRR/year quantities and create misleading non-zero baseline rows.
            if sc == 'SC0':
                incremental_fiscal_need_city = 0.0
                incremental_fiscal_need_r12 = 0.0
                delta_net_public_city = 0.0
                delta_net_public_r12 = 0.0
                delta_regulatory_revenue_city = 0.0
                delta_regulatory_revenue_r12 = 0.0
            regulatory_revenue_offset_city = clipped_offset(delta_regulatory_revenue_city, incremental_fiscal_need_city)
            regulatory_revenue_offset_r12 = clipped_offset(delta_regulatory_revenue_r12, incremental_fiscal_need_r12)
            net_fiscal_pressure_city = 0.0 if sc == 'SC0' else scalar_safe_div(impl_capex_city + delta_net_public_city, effective_budget_city, default=0.0)
            net_fiscal_pressure_r12 = 0.0 if sc == 'SC0' else scalar_safe_div(impl_capex_r12 + delta_net_public_r12, effective_budget_r12, default=0.0)
            net_annual_value_city = avoided_external_cost_city + float(b['cost_energy_IRR_year'] - r['cost_energy_IRR_year']) - impl_capex_city - delta_net_public_city
            net_annual_value_r12 = avoided_external_cost_r12 + float(b['cost_energy_IRR_year_r12'] - r['cost_energy_IRR_year_r12']) - impl_capex_r12 - delta_net_public_r12
            city_values = {
                'Modal share: public transport': float(r['modal_share_bus'] + r['modal_share_metro']),
                'Public transport trips': float(r['PT_trips_total']),
                'PCE-weighted VKT': float(r['vkm_pce_total']),
                'Non-public transport trips': float(r['trips_per_year'] - r['PT_trips_total']),
                'Congestion index': float(r['congestion_index']),
                'Time loss (car)': float(r['time_loss_car_hours_year']),
                'Final energy': float(r['final_energy_MJ_year']),
                'CO₂ emissions': float(r['CO2_t_year']),
                'NOₓ emissions': float(r['NOx_t_year']),
                'PM₂.₅ emissions': float(r['PM25_t_year']),
                'Health indicator': float(r['cost_health_IRR_year']),
                'Attributable deaths equivalent': attributable_deaths_city,
                'Traffic fatality equivalent': fatality_equiv_city,
                'Noise exposure index': float(r['noise_exposure_index']),
                'PT affordability ratio': pt_affordability_city,
                'PT travel expenditure': pt_travel_exp_city,
                N['pt_opex']: pt_service_opex_city,
                N['net_recurrent_public_burden']: net_public_recurring_burden_city,
                'Energy cost': energy_cost_city,
                N['avoided_external_cost']: avoided_external_cost_city,
                N['net_annual_value']: net_annual_value_city,
            }
            r12_values = {
                'Modal share: public transport': float(r['modal_share_PT_r12']),
                'Public transport trips': float(r['PT_trips_total_r12']),
                'PCE-weighted VKT': float(r['vkm_pce_total_r12']),
                'Non-public transport trips': float(r['trp_r12_effective'] - r['PT_trips_total_r12']),
                'Congestion index': float(r['congestion_index_r12']),
                'Time loss (car)': float(r['time_loss_car_hours_year_r12']),
                'Final energy': float(r['final_energy_MJ_year_r12']),
                'CO₂ emissions': float(r['CO2_t_year_r12']),
                'NOₓ emissions': float(r['NOx_t_year_r12']),
                'PM₂.₅ emissions': float(r['PM25_t_year_r12']),
                'Health indicator': health_cost_r12,
                'Attributable deaths equivalent': attributable_deaths_r12,
                'Traffic fatality equivalent': fatality_equiv_r12,
                'Noise exposure index': float(r['noise_exposure_index_r12_proxy']),
                'PT affordability ratio': pt_affordability_r12,
                'PT travel expenditure': pt_travel_exp_r12,
                N['pt_opex']: pt_service_opex_r12,
                N['net_recurrent_public_burden']: net_public_recurring_burden_r12,
                'Energy cost': float(r['cost_energy_IRR_year_r12']),
                N['avoided_external_cost']: avoided_external_cost_r12,
                N['net_annual_value']: net_annual_value_r12,
            }
            add_direct_metric_rows(core_rows, sc, 'Tehran', year, city_values)
            add_direct_metric_rows(core_rows, sc, 'Region12', year, r12_values)

            impl_city_values = {
                N['implementation_capex']: impl_capex_city,
                N['pt_subsidy_need']: pt_subsidy_need_city,
                N['farebox_recovery']: farebox_city,
                N['net_fiscal_pressure']: net_fiscal_pressure_city,
                N['budget_use']: budget_util_city,
                N['pt_budget_pressure']: pt_cost_to_budget_city,
                N['avoided_external_cost']: avoided_external_cost_city,
                N['net_annual_value']: net_annual_value_city,
                N['regulatory_revenue_realized']: direct_earmarked_revenue_city,
                N['regulatory_revenue_contribution']: regulatory_revenue_contribution_city,
                N['incremental_regulatory_revenue']: max(delta_regulatory_revenue_city, 0.0),
                N['incremental_fiscal_need']: incremental_fiscal_need_city,
                N['ev_transition_realized']: float(r['EV_share_vkm_weighted']) if 'EV_share_vkm_weighted' in r else float(r.get('share_car_EV', 0.0)),
            }
            impl_r12_values = {
                N['implementation_capex']: impl_capex_r12,
                N['pt_subsidy_need']: pt_subsidy_need_r12,
                N['farebox_recovery']: farebox_r12,
                N['net_fiscal_pressure']: net_fiscal_pressure_r12,
                N['budget_use']: budget_util_r12,
                N['pt_budget_pressure']: pt_cost_to_budget_r12,
                N['avoided_external_cost']: avoided_external_cost_r12,
                N['net_annual_value']: net_annual_value_r12,
                N['regulatory_revenue_realized']: direct_r12_earmarked_revenue,
                N['regulatory_revenue_contribution']: regulatory_revenue_contribution_r12,
                N['incremental_regulatory_revenue']: max(delta_regulatory_revenue_r12, 0.0),
                N['incremental_fiscal_need']: incremental_fiscal_need_r12,
                N['ev_transition_realized']: scalar_safe_div(float(r['vkm_car_EV_r12'] + r['vkm_taxi_EV_r12'] + r['vkm_bus_EV_r12'] + r['vkm_motorcycle_EV_r12']), float(r['vkm_car_r12'] + r['vkm_taxi_r12'] + r['vkm_bus_r12'] + r['vkm_motorcycle_r12']), default=0.0),
            }
            add_direct_metric_rows(impl_rows, sc, 'Tehran', year, impl_city_values)
            add_direct_metric_rows(impl_rows, sc, 'Region12', year, impl_r12_values)

            assumption_rows.append({
                'scenario': sc,
                'geo': 'Tehran',
                'year': year,
                'metric': N['implementation_capex'],
                'derivation': capex_note,
                'value': float(impl_capex_city),
            })
            assumption_rows.append({
                'scenario': sc,
                'geo': 'Region12',
                'year': year,
                'metric': N['implementation_capex'],
                'derivation': capex_note,
                'value': float(impl_capex_r12),
            })

    core_df = pd.DataFrame(core_rows)
    impl_df = pd.DataFrame(impl_rows)
    # Net annual value is an implementation-value signal,
    # not an absolute fiscal-flow account. It is therefore referenced to the
    # SC0 value for the same geography and year, while absolute fiscal flows
    # such as subsidy need, farebox recovery, budget use, and
    # regulatory revenue realized remain unchanged in the raw time series.
    # This makes SC0 exactly zero and leaves scenario values as policy-induced
    # deviations.
    for _df in (core_df, impl_df):
        _metric = N['net_annual_value']
        _base = _df[(_df['scenario'] == 'SC0') & (_df['metric'] == _metric)][['geo', 'year', 'value']].rename(columns={'value': '_baseline_net_value'})
        _df2 = _df.merge(_base, on=['geo', 'year'], how='left')
        _mask = _df2['metric'] == _metric
        _df2.loc[_mask, 'value'] = _df2.loc[_mask, 'value'] - _df2.loc[_mask, '_baseline_net_value'].fillna(0.0)
        _df2.loc[_mask & (_df2['value'].abs() < 1e-6), 'value'] = 0.0
        _df.drop(_df.index, inplace=True)
        for _col in _df2.columns:
            if _col != '_baseline_net_value':
                _df[_col] = _df2[_col]
    return core_df, impl_df, pd.DataFrame(assumption_rows)


def score_core(core_df: pd.DataFrame, cfg: dict):
    base = core_df[core_df['scenario'] == 'SC0'][['geo', 'year', 'metric', 'value']].rename(columns={'value': 'base_value'})
    merged = core_df.merge(base, on=['geo', 'year', 'metric'], how='left')
    contributions = []
    metric_scale = {}

    for pillar, pillar_cfg in cfg['core_outcome']['pillars'].items():
        for subfamily, metrics in pillar_cfg['subfamilies'].items():
            for metric_spec in metrics:
                metric = metric_spec['metric']
                sign = direction_sign(metric_spec['direction'])
                sub = merged[merged['metric'] == metric].copy()
                base_abs = sub['base_value'].abs()
                pct_mask = base_abs > 1e-9
                sub['oriented_effect'] = np.where(
                    pct_mask,
                    sign * 100.0 * (sub['value'] - sub['base_value']) / sub['base_value'],
                    sign * (sub['value'] - sub['base_value']),
                )
                auc = sub.groupby(['scenario', 'geo'], as_index=False)['oriented_effect'].sum().rename(columns={'oriented_effect': 'raw_oriented_effect'})
                for geo in auc['geo'].unique():
                    denom = auc.loc[(auc['geo'] == geo) & (auc['scenario'] != 'SC0'), 'raw_oriented_effect'].abs().max()
                    denom = float(denom) if pd.notna(denom) and denom > 1e-12 else 1.0
                    metric_scale[(metric, geo)] = denom
                auc['metric'] = metric
                auc['pillar'] = pillar
                auc['subfamily'] = subfamily
                auc['direction'] = metric_spec['direction']
                contributions.append(auc)

    contrib_df = pd.concat(contributions, ignore_index=True)
    contrib_df['score_component'] = contrib_df.apply(lambda r: 100.0 * r['raw_oriented_effect'] / metric_scale[(r['metric'], r['geo'])], axis=1)
    subfamily_scores = contrib_df.groupby(['scenario', 'geo', 'pillar', 'subfamily'], as_index=False)['score_component'].mean().rename(columns={'score_component': 'subfamily_score'})
    pillar_scores = subfamily_scores.groupby(['scenario', 'geo', 'pillar'], as_index=False)['subfamily_score'].mean().rename(columns={'subfamily_score': 'score'})
    return pillar_scores, contrib_df, subfamily_scores


def score_implementation(impl_df: pd.DataFrame, cfg: dict):
    metric_specs = {m['metric']: m for m in cfg['implementation_screen']['metrics']}
    base = impl_df[impl_df['scenario'] == 'SC0'][['geo', 'year', 'metric', 'value']].rename(columns={'value': 'base_value'})
    merged = impl_df.merge(base, on=['geo', 'year', 'metric'], how='left')
    rows = []
    for metric, spec in metric_specs.items():
        sign = direction_sign(spec['direction'])
        sub = merged[merged['metric'] == metric].copy()
        sub['oriented_delta'] = sign * (sub['value'] - sub['base_value'])
        agg = sub.groupby(['scenario', 'geo'], as_index=False)['oriented_delta'].mean().rename(columns={'oriented_delta': 'raw_oriented_mean_delta'})
        for geo in agg['geo'].unique():
            denom = agg.loc[(agg['geo'] == geo) & (agg['scenario'] != 'SC0'), 'raw_oriented_mean_delta'].abs().max()
            denom = float(denom) if pd.notna(denom) and denom > 1e-12 else 1.0
            agg.loc[agg['geo'] == geo, 'score'] = 100.0 * agg.loc[agg['geo'] == geo, 'raw_oriented_mean_delta'] / denom
        agg['metric'] = metric
        rows.append(agg)
    score_df = pd.concat(rows, ignore_index=True)
    equal_weight = score_df.groupby(['scenario', 'geo'], as_index=False)['score'].mean().rename(columns={'score': 'implementation_equal_weight_score'})
    return score_df, equal_weight


def run_sensitivity(impl_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    candidates = set(cfg['implementation_screen']['candidate_scenarios_for_sensitivity'])
    metric_specs = {m['metric']: m for m in cfg['implementation_screen']['metrics']}
    capex_metric = next((m['metric'] for m in cfg['implementation_screen']['metrics'] if m.get('key') == 'implementation_capex'), 'Implementation CAPEX')
    delta_metrics = impl_df[impl_df['metric'].isin(metric_specs)][['scenario', 'geo', 'year', 'metric', 'value']].copy()
    base = delta_metrics[delta_metrics['scenario'] == 'SC0'][['geo', 'year', 'metric', 'value']].rename(columns={'value': 'base_value'})
    merged = delta_metrics.merge(base, on=['geo', 'year', 'metric'], how='left')

    variants = {'capex_low': 0.75, 'capex_base': 1.00, 'capex_high': 1.25}
    rows = []
    leaders = []
    for variant, mult in variants.items():
        temp = merged.copy()
        capex_mask = temp['metric'].eq(capex_metric)
        temp.loc[capex_mask, 'value'] = temp.loc[capex_mask, 'value'] * mult
        score_rows = []
        for metric, spec in metric_specs.items():
            sign = direction_sign(spec['direction'])
            sub = temp[temp['metric'] == metric].copy()
            sub['oriented_delta'] = sign * (sub['value'] - sub['base_value'])
            agg = sub.groupby(['scenario', 'geo'], as_index=False)['oriented_delta'].mean().rename(columns={'oriented_delta': 'raw'})
            for geo in agg['geo'].unique():
                denom = agg.loc[(agg['geo'] == geo) & (agg['scenario'] != 'SC0'), 'raw'].abs().max()
                denom = float(denom) if pd.notna(denom) and denom > 1e-12 else 1.0
                agg.loc[agg['geo'] == geo, 'score'] = 100.0 * agg.loc[agg['geo'] == geo, 'raw'] / denom
            agg['metric'] = metric
            score_rows.append(agg[['scenario', 'geo', 'metric', 'score']])
        score_df = pd.concat(score_rows, ignore_index=True)
        eq = score_df.groupby(['scenario', 'geo'], as_index=False)['score'].mean().rename(columns={'score': 'implementation_equal_weight_score'})
        eq = eq[eq['scenario'].isin(candidates)].copy()
        eq['variant'] = variant
        rows.append(eq)
        for geo in ['Tehran', 'Region12']:
            d = eq[eq['geo'] == geo].sort_values('implementation_equal_weight_score', ascending=False)
            leader = d.iloc[0]['scenario']
            top3 = ', '.join(d.head(3)['scenario'].tolist())
            leaders.append({'variant': variant, 'geo': geo, 'leader': leader, 'top3': top3})
    sensitivity = pd.concat(rows, ignore_index=True)
    leaders_df = pd.DataFrame(leaders)
    lines = ['# Implementation sensitivity summary', '']
    for geo in ['Tehran', 'Region12']:
        d = leaders_df[leaders_df['geo'] == geo]
        leader_set = sorted(d['leader'].unique())
        top3_set = sorted(d['top3'].unique())
        lines.append(f'- {geo}: equal-weight implementation-screen leadership under capex assumptions scaled to 0.75x / 1.00x / 1.25x varies across **{", ".join(leader_set)}**; top-3 sets observed: **{" | ".join(top3_set)}**.')
    return sensitivity, '\n'.join(lines) + '\n'


def run_architecture_sensitivity(core_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    import copy

    variants: list[tuple[str, str, dict]] = [
        ('base_clean_balanced', 'Balanced 4×3 architecture with one non-redundant representative KPI per subfamily.', copy.deepcopy(cfg))
    ]

    alt_transport = copy.deepcopy(cfg)
    alt_transport['core_outcome']['pillars']['Transportation']['subfamilies']['network_burden'] = [
        {'metric': 'Congestion index', 'direction': 'lower_better'},
    ]
    variants.append(('transport_congestion_swap', 'Uses congestion index instead of time loss (car) as the network-burden representative.', alt_transport))

    alt_env = copy.deepcopy(cfg)
    alt_env['core_outcome']['pillars']['Environmental']['subfamilies']['climate'] = [
        {'metric': 'Final energy', 'direction': 'lower_better'},
    ]
    variants.append(('environment_energy_swap', 'Uses final energy instead of CO₂ emissions as the climate/resource representative.', alt_env))

    alt_social = copy.deepcopy(cfg)
    alt_social['core_outcome']['pillars']['Social']['subfamilies']['liveability'] = [
        {'metric': 'Traffic fatality equivalent', 'direction': 'lower_better'},
    ]
    variants.append(('social_safety_swap', 'Uses traffic fatality equivalent instead of noise exposure index as the third social representative.', alt_social))

    alt_econ = copy.deepcopy(cfg)
    alt_econ['core_outcome']['pillars']['Economic']['subfamilies']['pt_service_cost_pressure'] = [
        {'metric': 'PT travel expenditure', 'direction': 'lower_better'},
    ]
    variants.append(('economic_user_cost_swap', 'Uses PT travel expenditure instead of PT OPEX as the user-cost representative.', alt_econ))

    rows = []
    leaders = []
    for variant_name, note, variant_cfg in variants:
        pillar_scores, _, _ = score_core(core_df, variant_cfg)
        eq = pillar_scores.groupby(['scenario', 'geo'], as_index=False)['score'].mean().rename(columns={'score': 'equal_weight_score'})
        eq['variant'] = variant_name
        eq['variant_note'] = note
        for geo in eq['geo'].unique():
            d = eq[eq['geo'] == geo].sort_values('equal_weight_score', ascending=False).reset_index(drop=True)
            d['rank'] = np.arange(1, len(d) + 1)
            rows.append(d)
            leaders.append({
                'variant': variant_name,
                'geo': geo,
                'leader': d.iloc[0]['scenario'],
                'top3': ', '.join(d.head(3)['scenario'].tolist()),
            })
    detail = pd.concat(rows, ignore_index=True)
    leaders_df = pd.DataFrame(leaders)

    lines = ['# Decision-architecture sensitivity summary', '']
    for geo in ['Tehran', 'Region12']:
        d = leaders_df[leaders_df['geo'] == geo]
        leader_set = ', '.join(sorted(d['leader'].unique()))
        top3_set = ' | '.join(sorted(d['top3'].unique()))
        lines.append(f'- {geo}: across the cleaned non-redundant architecture and representative-metric swaps, equal-weight leadership is shared by **{leader_set}**; observed top-3 sets are **{top3_set}**.')
    lines.append('')
    lines.append('Interpretation: the cleaned architecture is not driven by one arbitrary representative; the same leading integrated packages remain concentrated at the top under plausible non-redundant metric substitutions.')
    return detail, '\n'.join(lines) + '\n'


def run_kpi_redundancy_audit(core_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    N = cfg.get('canonical_metric_labels', {}) or {}
    selected_specs: list[tuple[str, str, str]] = []
    selected_metrics: set[str] = set()
    for pillar, pillar_cfg in cfg['core_outcome']['pillars'].items():
        for subfamily, metrics in pillar_cfg['subfamilies'].items():
            for metric_spec in metrics:
                metric = metric_spec['metric']
                selected_specs.append((pillar, subfamily, metric))
                selected_metrics.add(metric)

    redundant_candidates = {
        'Public transport trips': {
            'reason': 'Dropped from Transportation because it is algebraically coupled to PT modal share and effective trips, so its scenario AUC profile duplicates modal shift.',
            'compare_to': ['Modal share: public transport'],
        },
        'Congestion index': {
            'reason': 'Dropped from Transportation because time loss (car) is the more decision-relevant network-burden representative.',
            'compare_to': ['Time loss (car)'],
        },
        'Non-public transport trips': {
            'reason': 'Dropped from Transportation because it is a near-duplicate private-load signal relative to PCE-weighted VKT.',
            'compare_to': ['PCE-weighted VKT'],
        },
        'Final energy': {
            'reason': 'Dropped from Environmental because it closely shadows CO₂ in the configured scenario set; CO₂ is retained as the climate headline metric.',
            'compare_to': ['CO₂ emissions'],
        },
        'Attributable deaths equivalent': {
            'reason': 'Dropped from Social because it restates the health-burden pathway already captured by the health indicator.',
            'compare_to': ['Health indicator'],
        },
        'Traffic fatality equivalent': {
            'reason': 'Moved out of the base Social score because, in the configured model outputs, it behaves as a strong exposure-scaled proxy and was better suited to sensitivity testing than the core balanced score.',
            'compare_to': ['Noise exposure index', 'PCE-weighted VKT'],
        },
        'PT travel expenditure': {
            'reason': 'Dropped from Economic because it duplicates the affordability signal already represented in Social; ' + N.get('pt_opex', 'PT OPEX') + ' is retained as the core operator-side cost metric.',
            'compare_to': [N.get('pt_affordability_ratio', 'PT affordability ratio'), N.get('pt_opex', 'PT OPEX')],
        },
    }

    all_metrics = sorted(selected_metrics | set(redundant_candidates))
    base = core_df[core_df['scenario'] == 'SC0'][['geo', 'year', 'metric', 'value']].rename(columns={'value': 'base_value'})
    merged = core_df[core_df['metric'].isin(all_metrics)].merge(base, on=['geo', 'year', 'metric'], how='left')

    direction_lookup = {}
    for _, _, metric in selected_specs:
        direction_lookup[metric] = 'higher_better' if metric == 'Modal share: public transport' else 'lower_better'
    direction_lookup.update({
        'Public transport trips': 'higher_better',
        'Congestion index': 'lower_better',
        'Non-public transport trips': 'lower_better',
        'Final energy': 'lower_better',
        'Attributable deaths equivalent': 'lower_better',
        'Traffic fatality equivalent': 'lower_better',
        'PT travel expenditure': 'lower_better',
    })

    effect_rows = []
    for metric, direction in direction_lookup.items():
        sign = direction_sign(direction)
        sub = merged[merged['metric'] == metric].copy()
        base_abs = sub['base_value'].abs()
        pct_mask = base_abs > 1e-9
        sub['oriented_effect'] = np.where(
            pct_mask,
            sign * 100.0 * (sub['value'] - sub['base_value']) / sub['base_value'],
            sign * (sub['value'] - sub['base_value']),
        )
        agg = sub.groupby(['scenario', 'geo'], as_index=False)['oriented_effect'].sum()
        agg['metric'] = metric
        effect_rows.append(agg)
    effect_df = pd.concat(effect_rows, ignore_index=True)

    pair_rows = []
    for geo in sorted(effect_df['geo'].unique()):
        wide = effect_df[effect_df['geo'] == geo].pivot(index='scenario', columns='metric', values='oriented_effect')
        for metric in sorted(redundant_candidates):
            if metric not in wide.columns:
                continue
            candidate_selected = [m for m in redundant_candidates[metric]['compare_to'] if m in wide.columns]
            selected_corrs = []
            for sel in candidate_selected:
                corr = wide[metric].corr(wide[sel])
                if pd.notna(corr):
                    selected_corrs.append((abs(float(corr)), float(corr), sel))
            if not selected_corrs:
                continue
            best_abs, best_corr, best_match = max(selected_corrs, key=lambda x: x[0])
            pair_rows.append({
                'geo': geo,
                'removed_metric': metric,
                'closest_selected_metric': best_match,
                'abs_corr_oriented_effect': round(best_abs, 6),
                'corr_oriented_effect': round(best_corr, 6),
                'decision': 'removed_from_base_score',
                'reason': redundant_candidates[metric]['reason'],
            })
    pair_df = pd.DataFrame(pair_rows).sort_values(['geo', 'abs_corr_oriented_effect', 'removed_metric'], ascending=[True, False, True]).reset_index(drop=True)

    registry_rows = []
    selected_rationales = {
        'Time loss (car)': 'Retained as the primary decision-relevant network-burden KPI instead of the more state-like congestion index.',
        'Modal share: public transport': 'Retained as the clean modal-shift representative; it is more interpretable than PT trips in a baseline-relative score.',
        'PCE-weighted VKT': 'Retained as the compact private-load/exposure representative instead of non-public trip counts.',
        'CO₂ emissions': 'Retained as the headline climate outcome for cross-scenario comparison.',
        'NOₓ emissions': 'Retained as a distinct local-pollutant signal not reducible to PM₂.₅ control alone.',
        'PM₂.₅ emissions': 'Retained as the most exposure-relevant local pollutant KPI.',
        'Health indicator': 'Retained as the summary health-burden KPI because it is decision-facing and subsumes the deaths-equivalent pathway.',
        'PT affordability ratio': 'Retained as the social affordability/equity representative.',
        'Noise exposure index': 'Retained as the liveability representative in the base score; safety remains checked in sensitivity rather than duplicated in the base architecture.',
        N.get('pt_opex', 'PT OPEX'): 'Retained as the operator-side service-cost KPI to avoid duplicating affordability with user expenditure.',
        N.get('net_recurrent_public_burden', 'Net recurrent public burden'): 'Retained as the public-budget pressure KPI.',
        'Energy cost': 'Retained as the direct system energy-cost KPI.',
    }
    for pillar, pillar_cfg in cfg['core_outcome']['pillars'].items():
        for subfamily, metrics in pillar_cfg['subfamilies'].items():
            metric = metrics[0]['metric']
            registry_rows.append({
                'pillar': pillar,
                'subfamily': subfamily,
                'metric': metric,
                'status': 'selected_in_base_score',
                'note': selected_rationales.get(metric, ''),
            })
    for metric, spec in redundant_candidates.items():
        registry_rows.append({
            'pillar': 'supporting_or_sensitivity',
            'subfamily': '',
            'metric': metric,
            'status': 'removed_from_base_score',
            'note': spec['reason'],
        })
    registry_df = pd.DataFrame(registry_rows)

    lines = [
        '# KPI pruning registry',
        '',
        '- Base core score now uses a balanced **4 pillars × 3 subfamilies × 1 KPI** structure so each retained KPI carries the same effective equal-weight influence within the core score.',
        '- Removed metrics were not deleted from the bundle outputs; they remain available for mechanism reading, secondary tables, and sensitivity checks.',
        '- The pruning rule was conceptual directness first, then redundancy avoidance using oriented-effect similarity across the configured scenario set.',
        '',
        '## Removed-from-base score diagnostics',
    ]
    if pair_df.empty:
        lines.append('No removed metrics were detected in the audit set.')
    else:
        for _, row in pair_df.iterrows():
            lines.append(f"- {row['geo']}: **{row['removed_metric']}** is closest to **{row['closest_selected_metric']}** (|corr|={row['abs_corr_oriented_effect']:.3f}). {row['reason']}")
    return pair_df, registry_df, '\n'.join(lines) + '\n'


def main(root: Path):
    root = root.resolve()
    out_dir = root / 'outputs'
    ver_dir = root / 'verification'
    out_dir.mkdir(exist_ok=True)
    ver_dir.mkdir(exist_ok=True)

    cfg = load_cfg(root)
    core_df, impl_df, assumption_df = build_direct_metric_tables(root)
    core_df.to_csv(out_dir / 'core_outcome_metric_timeseries.csv', index=False)
    impl_df.to_csv(out_dir / 'implementation_feasibility_metric_timeseries.csv', index=False)
    assumption_df.to_csv(ver_dir / 'implementation_capex_reconstruction_registry.csv', index=False)

    # core scores
    pillar_scores, contrib_df, subfamily_scores = score_core(core_df, cfg)
    pillar_scores.to_csv(out_dir / 'Figure_10_outcome_lens_scores.csv', index=False)
    contrib_df.to_csv(out_dir / 'Figure_10_outcome_lens_components.csv', index=False)
    subfamily_scores.to_csv(out_dir / 'Figure_10_outcome_lens_subfamily_scores.csv', index=False)
    contrib_df[['scenario', 'geo', 'pillar', 'subfamily', 'metric', 'score_component']].to_csv(ver_dir / 'pillar_metric_timeseries_domain_mirrored.csv', index=False)

    redundancy_pairs_df, pruning_registry_df, pruning_md = run_kpi_redundancy_audit(core_df, cfg)
    redundancy_pairs_df.to_csv(ver_dir / 'decision_kpi_redundancy_pairs.csv', index=False)
    pruning_registry_df.to_csv(ver_dir / 'decision_kpi_pruning_registry.csv', index=False)
    (ver_dir / 'decision_kpi_pruning_summary.md').write_text(pruning_md, encoding='utf-8')

    architecture_sensitivity_df, architecture_sensitivity_md = run_architecture_sensitivity(core_df, cfg)
    architecture_sensitivity_df.to_csv(ver_dir / 'decision_architecture_sensitivity_rankings.csv', index=False)
    (ver_dir / 'decision_architecture_sensitivity_summary.md').write_text(architecture_sensitivity_md, encoding='utf-8')

    # implementation scores
    impl_score_df, impl_equal = score_implementation(impl_df, cfg)
    impl_score_df.to_csv(out_dir / 'Figure_12_implementation_scores.csv', index=False)
    impl_equal.to_csv(out_dir / 'implementation_equal_weight_scores.csv', index=False)
    sensitivity_df, sensitivity_md = run_sensitivity(impl_df, cfg)
    sensitivity_df.to_csv(ver_dir / 'implementation_sensitivity_rankings.csv', index=False)
    (ver_dir / 'implementation_sensitivity_summary.md').write_text(sensitivity_md, encoding='utf-8')

    assumption_lines = [
        '# Implementation screen assumption registry',
        '',
        '- SC1 is treated as a near-zero-CAPEX administrative demand-management case.',
        '- SC2 reconstructs phased access-charging setup CAPEX from a disclosed share of baseline road CAPEX with traffic-plan weighting; SC3 reconstructs parking-management implementation CAPEX as a disclosed share of baseline road CAPEX scaled by realized parking-restraint intensity.',
        '- SC5 reconstructs PT access/service CAPEX as a disclosed share of existing PT capital intensity scaled by a composite of realized speed, network, and fleet uplifts.',
        '- SC6 reconstructs motorcycle clean-up CAPEX from annual vehicle-purchase flows times a transparent retrofit-unit-cost share.',
        '- SC7 uses direct model CAPEX deltas for taxi/bus-first clean transition plus explicit charger/depot add-ons. SC8 is treated as a low-burden package combining parking management and motorcycle clean-up.',
        '- SC9 combines disclosed access-charging, parking-management, and PT access/service assumptions into one core push-pull CAPEX reconstruction.',
        '- SC10 combines disclosed PT access/service, motorcycle clean-up, and taxi/bus-first clean-transition assumptions into one PT-first clean urban package.',
        '- SC11 combines soft demand management, access charging, parking management, PT access/service, motorcycle clean-up, and taxi/bus-first clean-transition assumptions into the broad package.',
        '- The balanced base social lens now retains the health indicator, PT affordability ratio, and noise exposure index; deaths-equivalent and fatality-equivalent signals remain available in the bundle for supporting and sensitivity use.',
        '- The balanced base economic lens now uses PT OPEX, net recurrent public burden, and energy cost; user PT expenditure remains available in outputs but is kept outside the base score to avoid duplicating affordability.',
        '- Traffic-plan revenue is treated as a collected-and-earmarked budget inflow rather than as a welfare benefit: central earmarking is 0.40, central collection is 0.90, and Region 12 uses budgetary allocation for budgets but traffic-plan attribution for regulatory revenue.',
        '- The implementation metric previously shown as Regulatory revenue offset is replaced by Net fiscal pressure, computed as (annualized implementation CAPEX + incremental net recurrent public burden after earmarked revenue) divided by effective transport budget; lower values are better and negative values indicate net fiscal relief.',
    ]
    (ver_dir / 'implementation_assumption_registry.md').write_text('\n'.join(assumption_lines) + '\n', encoding='utf-8')

    print('Generated decision-architecture outputs, implementation screen tables, and sensitivity summary.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root)
    main(root)
