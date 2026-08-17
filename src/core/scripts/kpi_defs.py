"""Canonical KPI definitions and display labels for the reproducibility workflow.

This module centralizes KPI names so manuscript tables, figure inputs, figures,
and audit checks cannot silently drift apart.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

KPI_MODAL_SHARE_PUBLIC = "Modal share: public transport"
KPI_MODAL_SHARE_CAR = "Modal share: car"
KPI_VKM_PCE = "PCE-weighted VKT"
KPI_CONGESTION = "Congestion index"
KPI_TIME_LOSS_CAR = "Time loss (car)"
KPI_FINAL_ENERGY = "Final energy"
KPI_ELECTRICITY_USE = "Electricity use"
KPI_CO2 = "CO₂ emissions"
KPI_PM25 = "PM₂.₅ emissions"
KPI_NOX = "NOₓ emissions"
KPI_HEALTH_COST = "Health indicator"
KPI_CLIMATE_COST = "Climate cost"
KPI_PT_TRIPS = "Public transport trips"
KPI_NON_PUBLIC_TRIPS = "Non-public transport trips"
KPI_ENERGY_GASOLINE = "Energy: gasoline"
KPI_ENERGY_DIESEL = "Energy: diesel"
KPI_ENERGY_CNG = "Energy: CNG"
KPI_TRIPS_TOTAL_EFFECTIVE = "Trips total (effective)"
KPI_POPULATION = "Population"
KPI_MUNICIPAL_BUDGET = "Municipal budget"

FIGURE_KPI_ORDER = [
    KPI_CONGESTION,
    KPI_TIME_LOSS_CAR,
    KPI_PT_TRIPS,
    KPI_NON_PUBLIC_TRIPS,
    KPI_VKM_PCE,
    KPI_FINAL_ENERGY,
    KPI_HEALTH_COST,
    KPI_CLIMATE_COST,
    KPI_MODAL_SHARE_CAR,
    KPI_MODAL_SHARE_PUBLIC,
    KPI_CO2,
    KPI_PM25,
    KPI_NOX,
    KPI_ELECTRICITY_USE,
    KPI_ENERGY_GASOLINE,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_CNG,
]

ALL_KPIS = [
    KPI_MODAL_SHARE_PUBLIC,
    KPI_MODAL_SHARE_CAR,
    KPI_VKM_PCE,
    KPI_CONGESTION,
    KPI_TIME_LOSS_CAR,
    KPI_FINAL_ENERGY,
    KPI_ELECTRICITY_USE,
    KPI_CO2,
    KPI_PM25,
    KPI_NOX,
    KPI_HEALTH_COST,
    KPI_CLIMATE_COST,
    KPI_PT_TRIPS,
    KPI_NON_PUBLIC_TRIPS,
    KPI_ENERGY_GASOLINE,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_CNG,
    KPI_TRIPS_TOTAL_EFFECTIVE,
    KPI_POPULATION,
    KPI_MUNICIPAL_BUDGET,
]

UNIT_MAP = {
    KPI_MODAL_SHARE_PUBLIC: "–",
    KPI_MODAL_SHARE_CAR: "–",
    KPI_VKM_PCE: "PCE-km/year",
    KPI_CONGESTION: "–",
    KPI_TIME_LOSS_CAR: "hours/year",
    KPI_FINAL_ENERGY: "MJ/year",
    KPI_ELECTRICITY_USE: "kWh/year",
    KPI_CO2: "t/year",
    KPI_PM25: "t/year",
    KPI_NOX: "t/year",
    KPI_HEALTH_COST: "IRR/year",
    KPI_CLIMATE_COST: "IRR/year",
    KPI_PT_TRIPS: "trips/year",
    KPI_NON_PUBLIC_TRIPS: "trips/year",
    KPI_ENERGY_GASOLINE: "MJ/year",
    KPI_ENERGY_DIESEL: "MJ/year",
    KPI_ENERGY_CNG: "MJ/year",
    KPI_TRIPS_TOTAL_EFFECTIVE: "trips/year",
    KPI_POPULATION: "persons",
    KPI_MUNICIPAL_BUDGET: "IRR/year",
}

DISPLAY_MAP = {
    KPI_ENERGY_GASOLINE: "Energy: gasoline use",
    KPI_ENERGY_DIESEL: "Energy: diesel use",
    KPI_ENERGY_CNG: "Energy: CNG use",
    KPI_ELECTRICITY_USE: "Energy: electricity use",
    KPI_CO2: "Emissions: CO₂",
    KPI_PM25: "Emissions: PM₂.₅",
    KPI_NOX: "Emissions: NOₓ",
}

FIG6_MAIN_KPIS = [
    KPI_CONGESTION,
    KPI_TIME_LOSS_CAR,
    KPI_PT_TRIPS,
    KPI_NON_PUBLIC_TRIPS,
    KPI_VKM_PCE,
    KPI_FINAL_ENERGY,
    KPI_HEALTH_COST,
    KPI_CLIMATE_COST,
]

FIG7_SECONDARY_KPIS = [
    KPI_MODAL_SHARE_CAR,
    KPI_MODAL_SHARE_PUBLIC,
    KPI_CO2,
    KPI_PM25,
    KPI_NOX,
    KPI_ELECTRICITY_USE,
    KPI_ENERGY_GASOLINE,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_CNG,
]

HEATMAP_MAIN_KPIS = FIG6_MAIN_KPIS
HEATMAP_SECONDARY_KPIS = [
    KPI_MODAL_SHARE_CAR,
    KPI_MODAL_SHARE_PUBLIC,
    KPI_CO2,
    KPI_PM25,
    KPI_NOX,
    KPI_ELECTRICITY_USE,
    KPI_ENERGY_GASOLINE,
    KPI_ENERGY_DIESEL,
    KPI_ENERGY_CNG,
]

MODAL_SHARE_MODE_COLUMNS = {
    "Tehran": {
        "car": "modal_share_car",
        "taxi": "modal_share_taxi",
        "bus": "modal_share_bus",
        "metro": "modal_share_metro",
        "motorcycle": "modal_share_motorcycle",
        "other": "modal_share_other",
    },
    "Region12": {
        "car": "modal_share_car_r12",
        "taxi": "modal_share_tax_r12",
        "bus": "modal_share_bus_r12",
        "metro": "modal_share_met_r12",
        "motorcycle": "modal_share_mot_r12",
        "other": "modal_share_oth_r12",
    },
}

MODAL_SHARE_MODE_TRUTH_COLUMNS = {
    "Tehran": {
        "car": "modal_share_car_truth",
        "taxi": "modal_share_taxi_truth",
        "bus": "modal_share_bus_truth",
        "metro": "modal_share_metro_truth",
        "motorcycle": "modal_share_motorcycle_truth",
        "other": "modal_share_other_truth",
    },
    "Region12": {
        "car": "modal_share_car_r12_truth",
        "taxi": "modal_share_tax_r12_truth",
        "bus": "modal_share_bus_r12_truth",
        "metro": "modal_share_met_r12_truth",
        "motorcycle": "modal_share_mot_r12_truth",
        "other": "modal_share_oth_r12_truth",
    },
}
