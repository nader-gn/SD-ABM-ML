"""Regenerate the canonical modal-share tidy plot input from simulation outputs.

The file contains:
- simulated modal shares for SC0..SC11 over the full 2012–2030 window
- observed/hindcast modal-share truth values for 2012–2023 from SC0 truth columns
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import pandas as pd
from kpi_defs import MODAL_SHARE_MODE_COLUMNS, MODAL_SHARE_MODE_TRUTH_COLUMNS

SIM_YEARS = range(2012, 2031)
OBS_YEARS = range(2012, 2024)


def main(root: Path) -> None:
    root = root.resolve()
    out_dir = root / "figure_inputs"
    out_dir.mkdir(exist_ok=True)
    rows: list[dict] = []

    for scen in range(12):
        df = pd.read_csv(root / "outputs" / f"simulation_data_SC{scen}.csv")
        years = df["YEAR_GRG"].round().astype(int)
        for area, mode_map in MODAL_SHARE_MODE_COLUMNS.items():
            for mode, col in mode_map.items():
                sub = df.loc[years.isin(SIM_YEARS), [col]].copy()
                for year, value in zip(years[years.isin(SIM_YEARS)], sub[col]):
                    rows.append({
                        "year": int(year),
                        "area": area,
                        "scenario": f"s{scen}",
                        "mode": mode,
                        "value": float(value),
                        "series": "sim",
                    })

    sc0 = pd.read_csv(root / "outputs" / "simulation_data_SC0.csv")
    years0 = sc0["YEAR_GRG"].round().astype(int)
    for area, mode_map in MODAL_SHARE_MODE_TRUTH_COLUMNS.items():
        for mode, col in mode_map.items():
            mask = years0.isin(OBS_YEARS)
            for year, value in zip(years0[mask], sc0.loc[mask, col]):
                if pd.isna(value):
                    continue
                rows.append({
                    "year": int(year),
                    "area": area,
                    "scenario": "obs",
                    "mode": mode,
                    "value": float(value),
                    "series": "obs",
                })

    out = pd.DataFrame(rows).sort_values(["area", "mode", "series", "scenario", "year"]).reset_index(drop=True)
    out_path = out_dir / "Figure_04_modal_share_series.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    args = ap.parse_args(); main(Path(args.root))
