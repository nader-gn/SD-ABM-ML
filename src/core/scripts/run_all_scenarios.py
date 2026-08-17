"""Run baseline (SC0) and one scenario at a time from the manuscript-aligned bundle.

Usage:
  python scripts/run_all_scenarios.py --root . --scenario SC2
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, copy, shutil, importlib.util
from pathlib import Path
import yaml
import pandas as pd
from scenario_meta import SCENARIO_TO_OVERLAY
from kpi_defs import MODAL_SHARE_MODE_COLUMNS, MODAL_SHARE_MODE_TRUTH_COLUMNS
from recompute_time_loss_sync import sync_dataframe

def _first_history_mask(df: pd.DataFrame) -> pd.Series:
    if 'YEAR_GRG' not in df.columns:
        return pd.Series(False, index=df.index)
    years = pd.to_numeric(df['YEAR_GRG'], errors='coerce').round().astype('Int64')
    valid = years.dropna()
    if valid.empty:
        return pd.Series(False, index=df.index)
    return years.eq(int(valid.min()))


def apply_historical_cold_start_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Synchronize the documented first-row cold-start anchors without altering later dynamics.

    Scope is intentionally narrow:
      - keep the shipped 2012 modal-share truth anchor,
      - repair the 2012 fuel totals that were distorted by the first-step car-energy handoff,
      - backfill the first-row car-energy closure factor when it is non-finite/zero.
    """
    if 'YEAR_GRG' not in df.columns:
        return df
    df = df.copy()
    mask = _first_history_mask(df)
    if not mask.any():
        return df
    idx = df.index[mask][0]

    # Existing modal-share warm-start anchor.
    for area, mode_map in MODAL_SHARE_MODE_COLUMNS.items():
        truth_map = MODAL_SHARE_MODE_TRUTH_COLUMNS[area]
        for _, col in mode_map.items():
            truth_col = truth_map[_]
            if col in df.columns and truth_col in df.columns:
                df.loc[mask, col] = df.loc[mask, truth_col]

    # Repair first-row car-energy closure factor if it came through as 0 / non-finite
    # from an unusable historical anchor value.
    if 'car_energy_closure_factor' in df.columns:
        cur = pd.to_numeric(pd.Series([df.loc[idx, 'car_energy_closure_factor']]), errors='coerce').iloc[0]
        if not pd.notna(cur) or float(cur) <= 0:
            later = pd.to_numeric(df.loc[df.index > idx, 'car_energy_closure_factor'], errors='coerce')
            later = later[(later > 0) & later.notna()]
            if not later.empty:
                df.loc[idx, 'car_energy_closure_factor'] = float(later.iloc[0])

    # Cold-start fix: the first historical row can under-propagate private-car energy use,
    # while taxi and motorcycle terms are already reasonable. Anchor only the first-row totals
    # and the directly implied car residuals.
    if {'fuel_gasoline_litre_year_truth', 'fuel_gasoline_taxi_litre_year', 'fuel_gasoline_motorcycle_litre_year', 'fuel_gasoline_car_litre_year'}.issubset(df.columns):
        gas_truth = float(df.loc[idx, 'fuel_gasoline_litre_year_truth'])
        gas_other = float(df.loc[idx, 'fuel_gasoline_taxi_litre_year']) + float(df.loc[idx, 'fuel_gasoline_motorcycle_litre_year'])
        df.loc[idx, 'fuel_gasoline_car_litre_year'] = max(gas_truth - gas_other, 0.0)
        if 'fuel_gasoline_litre_year' in df.columns:
            df.loc[idx, 'fuel_gasoline_litre_year'] = float(df.loc[idx, 'fuel_gasoline_car_litre_year']) + gas_other

    if {'fuel_CNG_kg_year_truth', 'fuel_CNG_taxi_kg_year', 'fuel_CNG_car_kg_year'}.issubset(df.columns):
        cng_truth = float(df.loc[idx, 'fuel_CNG_kg_year_truth'])
        cng_taxi = float(df.loc[idx, 'fuel_CNG_taxi_kg_year'])
        df.loc[idx, 'fuel_CNG_car_kg_year'] = max(cng_truth - cng_taxi, 0.0)
        if 'fuel_CNG_kg_year' in df.columns:
            df.loc[idx, 'fuel_CNG_kg_year'] = float(df.loc[idx, 'fuel_CNG_car_kg_year']) + cng_taxi

    # Keep directly dependent annual fuel-cost totals in sync for the same anchored row.
    if {'cost_fuel_gasoline_IRR_year', 'fuel_price_gasoline_IRR_litre', 'fuel_gasoline_litre_year'}.issubset(df.columns):
        df.loc[idx, 'cost_fuel_gasoline_IRR_year'] = float(df.loc[idx, 'fuel_price_gasoline_IRR_litre']) * float(df.loc[idx, 'fuel_gasoline_litre_year'])
    if {'cost_fuel_CNG_IRR_year', 'fuel_price_CNG_IRR_kg', 'fuel_CNG_kg_year'}.issubset(df.columns):
        df.loc[idx, 'cost_fuel_CNG_IRR_year'] = float(df.loc[idx, 'fuel_price_CNG_IRR_kg']) * float(df.loc[idx, 'fuel_CNG_kg_year'])

    return df

def load_core(core_path: Path):
    spec = importlib.util.spec_from_file_location('system_core', str(core_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['system_core'] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def main(root: Path, scenario: str) -> None:
    if not scenario:
        raise SystemExit('Pass --scenario SC0..SC11; orchestrate multi-scenario rebuild outside this script for isolation.')
    root = root.resolve()
    cfg = yaml.safe_load((root / 'config' / 'BASE_CONFIG.yaml').read_text())
    cfg['simulation']['data_file'] = str((root / 'config' / 'DATA_clean.csv').resolve())
    if 'feature_selection' in cfg: cfg['feature_selection']['enabled'] = False
    if 'hyperparams' in cfg: cfg['hyperparams']['enabled'] = False
    cfg_i = copy.deepcopy(cfg)
    if scenario != 'SC0':
        ov = yaml.safe_load((root / 'scenarios' / SCENARIO_TO_OVERLAY[scenario]).read_text())
        cfg_i.setdefault('exogenous_forecast', {})
        cfg_i['exogenous_forecast'].update(ov.get('exogenous_forecast', {}))
    out_dir = root / 'outputs'; out_dir.mkdir(exist_ok=True)
    tmp_dir = root / f'_tmp_{scenario}'
    if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()
    try:
        tmp_yaml = tmp_dir / f'{scenario}.yaml'
        tmp_yaml.write_text(yaml.safe_dump(cfg_i, sort_keys=False))
        run_out = tmp_dir / 'run'
        run_out.mkdir()
        core = load_core(root / 'config' / 'system_core.py')
        core.run_simulation_from_config(tmp_yaml, output_dir=str(run_out))
        sim_df = apply_historical_cold_start_anchors(pd.read_csv(run_out / 'simulation_data.csv'))
        sim_df = sync_dataframe(sim_df)
        sim_df.to_csv(out_dir / f'simulation_data_{scenario}.csv', index=False)
        print(f'DONE {scenario}', flush=True)
    finally:
        if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='.'); ap.add_argument('--scenario', required=True)
    args = ap.parse_args(); main(Path(args.root), args.scenario)
