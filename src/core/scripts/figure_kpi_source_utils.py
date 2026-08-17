from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from metric_registry import (
    core_metric_order,
    secondary_metric_order,
    selected24_metric_order,
    unit_map_24,
)

EXPECTED_ROWS_PER_KPI = 168  # 12 scenarios × 7 years × 2 geographies

# Mutable globals kept for backward-compatible imports. They are refreshed from
# config/decision_architecture.yaml at the start of source selection and audits.
DECISION12_KPIS = [
    'Time loss (car)',
    'Modal share: public transport',
    'PCE-weighted VKT',
    'CO₂ emissions',
    'NOₓ emissions',
    'PM₂.₅ emissions',
    'Health indicator',
    'PT affordability ratio',
    'Noise exposure index',
    'PT OPEX',
    'Net recurrent public burden',
    'Energy cost',
]
SECONDARY12_KPIS = [
    'Congestion index',
    'Public transport trips',
    'Non-public transport trips',
    'Budget use',
    'Final energy',
    'Electricity use',
    'Implementation CAPEX',
    'PT subsidy need',
    'Net fiscal pressure',
    'Farebox recovery',
    'Avoided external cost',
    'Net annual value',
]
SELECTED24_KPIS = DECISION12_KPIS + SECONDARY12_KPIS
UNIT_MAP_24 = {
    'Time loss (car)': 'hours/year',
    'Modal share: public transport': '–',
    'PCE-weighted VKT': 'PCE-km/year',
    'CO₂ emissions': 't/year',
    'NOₓ emissions': 't/year',
    'PM₂.₅ emissions': 't/year',
    'Health indicator': 'IRR/year',
    'PT affordability ratio': 'ratio',
    'Noise exposure index': 'index',
    'PT OPEX': 'IRR/year',
    'Net recurrent public burden': 'IRR/year',
    'Energy cost': 'IRR/year',
    'Congestion index': '–',
    'Public transport trips': 'trips/year',
    'Non-public transport trips': 'trips/year',
    'Budget use': 'ratio',
    'Final energy': 'MJ/year',
    'Electricity use': 'kWh/year',
    'Implementation CAPEX': 'IRR',
    'PT subsidy need': 'IRR/year',
    'Net fiscal pressure': 'ratio',
    'Farebox recovery': 'ratio',
    'PT budget pressure': 'ratio',
    'Avoided external cost': 'IRR/year',
    'Net annual value': 'IRR/year',
}


def refresh_metric_globals(root: Path) -> None:
    """Refresh mutable KPI lists from the decision-architecture config."""
    decision = core_metric_order(root)
    secondary = secondary_metric_order(root)
    units = unit_map_24(root)
    DECISION12_KPIS[:] = decision
    SECONDARY12_KPIS[:] = secondary
    SELECTED24_KPIS[:] = decision + secondary
    UNIT_MAP_24.clear()
    UNIT_MAP_24.update(units)

def load_all_sources(root: Path):
    base = pd.read_csv(root / 'outputs' / 'timeseries_all.csv')[['Scenario', 'year', 'Geo', 'kpi', 'value']].copy()
    base['source'] = 'timeseries_all'

    core = pd.read_csv(root / 'outputs' / 'core_outcome_metric_timeseries.csv').rename(
        columns={'scenario': 'Scenario', 'geo': 'Geo', 'metric': 'kpi'}
    )
    core['Geo'] = core['Geo'].replace({'Region12': 'Region 12'})
    core = core[['Scenario', 'year', 'Geo', 'kpi', 'value']].copy()
    core['source'] = 'core_outcome_metric_timeseries'

    imp = pd.read_csv(root / 'outputs' / 'implementation_feasibility_metric_timeseries.csv').rename(
        columns={'scenario': 'Scenario', 'geo': 'Geo', 'metric': 'kpi'}
    )
    imp['Geo'] = imp['Geo'].replace({'Region12': 'Region 12'})
    imp = imp[['Scenario', 'year', 'Geo', 'kpi', 'value']].copy()
    imp['source'] = 'implementation_feasibility_metric_timeseries'
    return {
        'timeseries_all': base,
        'core_outcome_metric_timeseries': core,
        'implementation_feasibility_metric_timeseries': imp,
    }


def source_priority_map():
    pri = {}
    for k in DECISION12_KPIS:
        pri[k] = ['core_outcome_metric_timeseries', 'timeseries_all', 'implementation_feasibility_metric_timeseries']

    # Secondary-set source priorities are position-based, so renaming in the
    # config does not require script edits.
    core_secondary_idx = [0, 1, 2, 4]  # congestion, PT trips, non-PT trips, final energy
    for i in core_secondary_idx:
        if i < len(SECONDARY12_KPIS):
            pri[SECONDARY12_KPIS[i]] = ['core_outcome_metric_timeseries', 'timeseries_all']
    if len(SECONDARY12_KPIS) > 3:
        pri[SECONDARY12_KPIS[3]] = ['implementation_feasibility_metric_timeseries']
    if len(SECONDARY12_KPIS) > 5:
        pri[SECONDARY12_KPIS[5]] = ['timeseries_all', 'core_outcome_metric_timeseries']
    for k in SECONDARY12_KPIS[6:]:
        pri[k] = ['implementation_feasibility_metric_timeseries', 'core_outcome_metric_timeseries']
    return pri


def choose_source_rows(root: Path, kpi_order: list[str] | None = None, unit_map=None):
    refresh_metric_globals(root)
    if kpi_order is None:
        kpi_order = SELECTED24_KPIS
    sources = load_all_sources(root)
    priorities = source_priority_map()

    selected_parts = []
    audit_rows = []

    for kpi in kpi_order:
        source_counts = {name: int((df['kpi'] == kpi).sum()) for name, df in sources.items()}
        chosen_source = None
        chosen_df = None
        for sname in priorities.get(kpi, list(sources.keys())):
            sub = sources[sname][sources[sname]['kpi'] == kpi].copy()
            if len(sub) >= EXPECTED_ROWS_PER_KPI:
                chosen_source = sname
                chosen_df = sub
                break
        if chosen_df is None:
            best_name = max(source_counts, key=source_counts.get)
            chosen_source = best_name
            chosen_df = sources[best_name][sources[best_name]['kpi'] == kpi].copy()

        chosen_df['unit'] = (unit_map or UNIT_MAP_24).get(kpi, '–')
        selected_parts.append(chosen_df[['Scenario', 'year', 'Geo', 'kpi', 'unit', 'value']])
        audit_rows.append({
            'kpi': kpi,
            'chosen_source': chosen_source,
            'chosen_rows': len(chosen_df),
            'complete_coverage': bool(len(chosen_df) >= EXPECTED_ROWS_PER_KPI),
            'timeseries_all_rows': source_counts['timeseries_all'],
            'core_outcome_metric_timeseries_rows': source_counts['core_outcome_metric_timeseries'],
            'implementation_feasibility_metric_timeseries_rows': source_counts['implementation_feasibility_metric_timeseries'],
        })

    merged = pd.concat(selected_parts, ignore_index=True)
    merged['kpi'] = pd.Categorical(merged['kpi'], kpi_order, ordered=True)
    merged = merged.sort_values(['kpi', 'Geo', 'Scenario', 'year']).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return merged, audit


def dynamic_delta_table(df: pd.DataFrame, kpi: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a delta table for one KPI.

    For each geo independently:
    - if SC0 has any zero/near-zero baseline year, use normalized raw delta for the entire geo panel;
    - otherwise use percent delta.

    Returns (delta_rows, mode_audit_rows).
    """
    sub = df[df['kpi'] == kpi].copy()
    base = sub[sub['Scenario'] == 'SC0'][['year', 'Geo', 'value']].rename(columns={'value': 'base'})
    sub = sub.merge(base, on=['year', 'Geo'], how='left')
    out_parts = []
    mode_rows = []
    for geo, g in sub.groupby('Geo', sort=False):
        has_zero_baseline = bool((g['base'].abs() <= 1e-12).any())
        if has_zero_baseline:
            g['delta_metric'] = g['value'] - g['base']
            max_abs = float(np.nanmax(np.abs(g['delta_metric'].to_numpy()))) if len(g) else 0.0
            g['delta_display'] = 0.0 if max_abs <= 0 else (g['delta_metric'] / max_abs) * 100.0
            g['annot'] = g['delta_metric']
            g['annot_mode'] = 'raw'
            mode = 'raw_normalized'
        else:
            g['delta_display'] = np.where(np.abs(g['base']) > 1e-12, (g['value'] - g['base']) / g['base'] * 100.0, np.nan)
            g['annot'] = g['delta_display']
            g['annot_mode'] = 'pct'
            mode = 'percent'
        out_parts.append(g[['Scenario', 'year', 'Geo', 'delta_display', 'annot', 'annot_mode']])
        mode_rows.append({'kpi': kpi, 'Geo': geo, 'delta_mode': mode})
    return pd.concat(out_parts, ignore_index=True), pd.DataFrame(mode_rows)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
