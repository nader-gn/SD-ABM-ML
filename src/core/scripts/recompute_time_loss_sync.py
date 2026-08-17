from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SYNC_COLUMNS = [
    "travel_time_car_hours",
    "travel_time_car_congested_hours",
    "travel_time_car_r12_base_hours",
    "travel_time_car_r12_hours",
    "time_loss_car_hours_year",
    "time_loss_car_hours_year_r12",
]

def sync_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"trip_length_km","spd_car"}.issubset(out.columns):
        out["travel_time_car_hours"] = out["trip_length_km"] / out["spd_car"].clip(lower=1.0)
    if {"travel_time_car_hours","dly_tot"}.issubset(out.columns):
        out["travel_time_car_congested_hours"] = out["travel_time_car_hours"] * (1.0 + out["dly_tot"])
    if {"trip_length_r12_km","spd_car_r12"}.issubset(out.columns):
        out["travel_time_car_r12_base_hours"] = out["trip_length_r12_km"] / out["spd_car_r12"].clip(lower=1.0)
    if {"travel_time_car_r12_base_hours","dly_r12"}.issubset(out.columns):
        out["travel_time_car_r12_hours"] = out["travel_time_car_r12_base_hours"] * (1.0 + out["dly_r12"])
    if {"trips_car","travel_time_car_hours","dly_tot"}.issubset(out.columns):
        out["time_loss_car_hours_year"] = out["trips_car"] * out["travel_time_car_hours"] * out["dly_tot"]
    if {"trp_r12","modal_share_car_r12","travel_time_car_r12_base_hours","dly_r12"}.issubset(out.columns):
        out["time_loss_car_hours_year_r12"] = (
            out["trp_r12"] * out["modal_share_car_r12"] * out["travel_time_car_r12_base_hours"] * out["dly_r12"]
        )
    return out

def sync_file(path: Path) -> None:
    df = pd.read_csv(path)
    synced = sync_dataframe(df)
    synced.to_csv(path, index=False)

def main(root: Path) -> None:
    out_dir = root / "outputs"
    for path in sorted(out_dir.glob("simulation_data_SC*.csv")):
        sync_file(path)
        print(f"synced {path.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    main(Path(args.root).resolve())
