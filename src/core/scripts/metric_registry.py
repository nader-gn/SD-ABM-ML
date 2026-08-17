"""Configuration-defined metric names, units, and figure display labels.

The decision architecture YAML is the single source of truth for the
reported metric names. Generator and figure scripts call this module
instead of carrying private OPEX/fiscal label copies.
"""
from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_CANONICAL_LABELS = {
    'time_loss_car': 'Time loss (car)',
    'modal_share_pt': 'Modal share: public transport',
    'vkm_pce': 'PCE-weighted VKT',
    'co2_emissions': 'CO₂ emissions',
    'nox_emissions': 'NOₓ emissions',
    'pm25_emissions': 'PM₂.₅ emissions',
    'health_cost': 'Health indicator',
    'pt_affordability_ratio': 'PT affordability ratio',
    'noise_exposure_index': 'Noise exposure index',
    'pt_opex': 'PT OPEX',
    'net_recurrent_public_burden': 'Net recurrent public burden',
    'energy_cost': 'Energy cost',
    'congestion_index': 'Congestion index',
    'public_transport_trips': 'Public transport trips',
    'non_public_transport_trips': 'Non-public transport trips',
    'budget_use': 'Budget use',
    'final_energy': 'Final energy',
    'electricity_use': 'Electricity use',
    'implementation_capex': 'Implementation CAPEX',
    'pt_subsidy_need': 'PT subsidy need',
    'farebox_recovery': 'Farebox recovery',
    'net_fiscal_pressure': 'Net fiscal pressure',
    'pt_budget_pressure': 'PT budget pressure',
    'avoided_external_cost': 'Avoided external cost',
    'net_annual_value': 'Net annual value',
    'pt_travel_expenditure': 'PT travel expenditure',
    'attributable_deaths_equivalent': 'Attributable deaths equivalent',
    'traffic_fatality_equivalent': 'Traffic fatality equivalent',
    'regulatory_revenue_realized': 'Regulatory revenue realized',
    'regulatory_revenue_contribution': 'Regulatory revenue contribution',
    'incremental_regulatory_revenue': 'Incremental regulatory revenue',
    'incremental_fiscal_need': 'Incremental fiscal need',
    'ev_transition_realized': 'EV transition realized',
}

DEFAULT_DISPLAY_LABELS = {
    'CO₂ emissions': 'Emissions: CO₂',
    'NOₓ emissions': 'Emissions: NOₓ',
    'PM₂.₅ emissions': 'Emissions: PM₂.₅',
    'Public transport trips': 'PT trips',
    'Non-public transport trips': 'Non-PT trips',
    'Implementation CAPEX': 'CAPEX',
    'PT subsidy need': 'PT subsidy',
    'Avoided external cost': 'Avoided ext. cost',
    'PT OPEX': 'PT OPEX',
    'Net recurrent public burden': 'Net recurrent burden',
    'PT affordability ratio': 'PT affordability',
    'Noise exposure index': 'Noise exposure',
}

DEFAULT_UNIT_MAP_24 = {
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


def load_decision_architecture(root: Path) -> dict:
    root = Path(root).resolve()
    return yaml.safe_load((root / 'config' / 'decision_architecture.yaml').read_text(encoding='utf-8'))


def canonical_labels(root: Path) -> dict[str, str]:
    cfg = load_decision_architecture(root)
    labels = dict(DEFAULT_CANONICAL_LABELS)
    labels.update(cfg.get('canonical_metric_labels', {}) or {})
    return labels


def label(root: Path, key: str) -> str:
    labels = canonical_labels(root)
    if key not in labels:
        raise KeyError(f'Missing canonical_metric_labels entry for {key!r}')
    return labels[key]


def core_metric_order(root: Path) -> list[str]:
    cfg = load_decision_architecture(root)
    metrics: list[str] = []
    for _pillar, pcfg in cfg['core_outcome']['pillars'].items():
        for _subfamily, specs in pcfg['subfamilies'].items():
            for spec in specs:
                metrics.append(spec['metric'])
    return metrics


def secondary_metric_order(root: Path) -> list[str]:
    cfg = load_decision_architecture(root)
    if cfg.get('figure_metric_sets', {}).get('secondary12'):
        return list(cfg['figure_metric_sets']['secondary12'])
    labels = canonical_labels(root)
    return [
        labels['congestion_index'], labels['public_transport_trips'], labels['non_public_transport_trips'],
        labels['budget_use'], labels['final_energy'], labels['electricity_use'], labels['implementation_capex'],
        labels['pt_subsidy_need'], labels['net_fiscal_pressure'], labels['farebox_recovery'],
        labels['avoided_external_cost'], labels['net_annual_value'],
    ]


def selected24_metric_order(root: Path) -> list[str]:
    cfg = load_decision_architecture(root)
    if cfg.get('figure_metric_sets', {}).get('decision12'):
        decision = list(cfg['figure_metric_sets']['decision12'])
    else:
        decision = core_metric_order(root)
    return decision + secondary_metric_order(root)


def unit_map_24(root: Path) -> dict[str, str]:
    cfg = load_decision_architecture(root)
    units = dict(DEFAULT_UNIT_MAP_24)
    units.update(cfg.get('unit_map_24', {}) or {})
    return units


def display_map(root: Path) -> dict[str, str]:
    cfg = load_decision_architecture(root)
    mapping = dict(DEFAULT_DISPLAY_LABELS)
    mapping.update(cfg.get('figure_display_labels', {}) or {})
    for item in cfg.get('implementation_screen', {}).get('metrics', []):
        mapping[item['metric']] = item.get('display', item['metric'])
    return mapping


def implementation_metric_order(root: Path) -> list[str]:
    cfg = load_decision_architecture(root)
    return [item['metric'] for item in cfg['implementation_screen']['metrics']]


def implementation_display_labels(root: Path) -> dict[str, str]:
    cfg = load_decision_architecture(root)
    return {item['metric']: item.get('display', item['metric']) for item in cfg['implementation_screen']['metrics']}
