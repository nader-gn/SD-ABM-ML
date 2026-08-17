from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from figure_kpi_source_utils import SELECTED24_KPIS, UNIT_MAP_24, choose_source_rows, dynamic_delta_table, write_csv, refresh_metric_globals
from metric_registry import display_map

DISPLAY_MAP = {
    'CO₂ emissions': 'Emissions: CO₂',
    'NOₓ emissions': 'Emissions: NOₓ',
    'PM₂.₅ emissions': 'Emissions: PM₂.₅',
    'Public transport trips': 'PT trips',
    'Non-public transport trips': 'Non-PT trips',
    'Budget use': 'Budget use',
    'Implementation CAPEX': 'CAPEX',
    'PT subsidy need': 'PT subsidy',
    'Net fiscal pressure': 'Net fiscal pressure',
    'Farebox recovery': 'Farebox recovery',
    'Avoided external cost': 'Avoided ext. cost',
    'Net annual value': 'Net annual value',
    'PT OPEX': 'PT OPEX',
    'Net recurrent public burden': 'Net recurrent burden',
    'PT affordability ratio': 'PT affordability',
    'Noise exposure index': 'Noise exposure',
}
SCENARIOS = [f'SC{i}' for i in range(12)]
GEOS = ['Tehran', 'Region 12']


def load_kpi_timeseries(root: Path):
    df, audit = choose_source_rows(root, SELECTED24_KPIS, UNIT_MAP_24)
    write_csv(audit, root / 'verification' / 'kpi_source_audit_fig10_auc.csv')
    return df


def compute_auc_rows(df: pd.DataFrame):
    rows = []
    mode_rows = []
    for kpi in SELECTED24_KPIS:
        delta_df, mode_audit = dynamic_delta_table(df, kpi)
        mode_rows.append(mode_audit)
        for geo in GEOS:
            geo_mode = mode_audit.loc[mode_audit['Geo'] == geo, 'delta_mode'].iloc[0]
            sub = delta_df[delta_df['Geo'] == geo].copy()
            for scen in SCENARIOS:
                g = sub[sub['Scenario'] == scen].sort_values('year')
                finite = np.isfinite(g['delta_display'].to_numpy())
                if finite.sum() >= 2:
                    auc = float(np.trapezoid(g.loc[finite, 'delta_display'].to_numpy(), g.loc[finite, 'year'].to_numpy()))
                elif finite.sum() == 1:
                    auc = float(g.loc[finite, 'delta_display'].iloc[0])
                else:
                    auc = 0.0
                rows.append({
                    'scenario': str(scen),
                    'kpi': kpi,
                    'auc': auc,
                    'geo': geo.replace('Region 12', 'Region12'),
                    'kpi_display': DISPLAY_MAP.get(kpi, kpi),
                    'delta_mode': geo_mode,
                })
    return pd.DataFrame(rows), pd.concat(mode_rows, ignore_index=True)


def main(root: Path):
    root = root.resolve()
    refresh_metric_globals(root)
    DISPLAY_MAP.clear(); DISPLAY_MAP.update(display_map(root))
    df = load_kpi_timeseries(root)
    out, mode_audit = compute_auc_rows(df)
    out = out.sort_values(['geo', 'scenario', 'kpi']).reset_index(drop=True)
    write_csv(out, root / 'figure_inputs' / 'Figure_09_auc_input.csv')
    write_csv(mode_audit, root / 'verification' / 'kpi_delta_modes_fig10_auc.csv')
    # KPI-level validation summary
    summary = out.groupby(['geo', 'kpi'], as_index=False).agg(
        count_rows=('auc', 'count'),
        finite_auc_rows=('auc', lambda s: int(np.isfinite(s).sum())),
        nonfinite_auc_rows=('auc', lambda s: int((~np.isfinite(s)).sum())),
    )
    write_csv(summary, root / 'verification' / 'auc_circular_validation_summary.csv')
    print(f"Wrote {root / 'figure_inputs' / 'Figure_09_auc_input.csv'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    main(Path(args.root))
