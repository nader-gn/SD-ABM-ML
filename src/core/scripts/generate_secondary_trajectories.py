from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_utils import scientific_scale_for_values, style_preset, reserve_bottom_for_accessory, place_bottom_legend, set_exact_x_span
from scenario_meta import SCENARIO_CODES as SC_ORDER, SCENARIO_TO_COLOR as PALETTE
from figure_kpi_source_utils import SECONDARY12_KPIS, UNIT_MAP_24, choose_source_rows, write_csv, refresh_metric_globals
from metric_registry import display_map

COMPACT_STYLE = dict(style_preset('half'))
COMPACT_STYLE.update({'title': 10.2, 'region': 12.0, 'axis': 9.2, 'tick': 8.3, 'legend': 8.2, 'small': 8.0})
A4_LANDSCAPE = (10.8, 10.35)  # matched to Figures 7-8 subplot height
SUBPLOT_TARGET_SCALE = 1.10
GRID_ROWS = 6
GRID_COLS = 4
LINE_SUBPLOTS_ADJUST = dict(left=0.072, right=0.995, top=0.920, bottom=0.130, wspace=0.24, hspace=0.52)
ROOT = SCRIPT_DIR.parent
DISPLAY_MAP = {
    'Public transport trips': 'PT trips',
    'Non-public transport trips': 'Non-PT trips',
    'Budget use': 'Budget use',
    'Implementation CAPEX': 'CAPEX',
    'PT subsidy need': 'PT subsidy need',
    'Net fiscal pressure': 'Net fiscal pressure',
    'Farebox recovery': 'Farebox recovery',
    'Avoided external cost': 'Avoided external cost',
    'Net annual value': 'Net annual value',
}


def load_timeseries(root: Path):
    refresh_metric_globals(root)
    DISPLAY_MAP.clear(); DISPLAY_MAP.update(display_map(root))
    df, audit = choose_source_rows(root, SECONDARY12_KPIS, UNIT_MAP_24)
    write_csv(audit, root / 'verification' / 'Figure_06_source_audit.csv')
    return df


def plot_panel(ax, df, kpi, geo, panel, style):
    sub = df[(df['kpi'] == kpi) & (df['Geo'] == geo)].copy()
    years = sorted(sub['year'].unique())
    for scen in [s for s in SC_ORDER if s != 'SC0'] + ['SC0']:
        g = sub[sub['Scenario'] == scen].sort_values('year')
        ls = {'linestyle': '--', 'linewidth': 2.0, 'zorder': 12} if scen == 'SC0' else {'linestyle': '-', 'linewidth': 1.55, 'zorder': 5}
        ax.plot(g['year'], g['value'], color=PALETTE[scen], label=scen, **ls)
    ax.set_title(f"{panel} {DISPLAY_MAP.get(kpi, kpi)}", loc='left', fontsize=style['title'] + 0.5, pad=7)
    ax.grid(False)
    ax.tick_params(labelsize=style['tick'])
    ymin, ymax = ax.get_ylim()
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        ticks = np.linspace(ymin, ymax, 5)
        ax.set_yticks(ticks)
        scale, scale_label = scientific_scale_for_values(ticks, single_digit_ticks=True)
        tick_labels = [f"{t / scale:.2f}" for t in ticks]
        tick_labels[-1] = ''
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis='y', labelsize=style['small'], pad=1.5, direction='out', length=2.2)
        if scale_label and scale_label != '×1e0':
            dx_axes = (2.0 / 25.4) / (ax.figure.get_figwidth() * ax.get_position().width)
            ax.text(-0.095 - dx_axes, 1.005, scale_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=style['small'], clip_on=False)
    set_exact_x_span(ax, years)
    if geo == 'Tehran':
        ax.set_ylabel(UNIT_MAP_24.get(kpi, sub['unit'].iloc[0] if len(sub) else ''), fontsize=style['axis'])


def generate_figure(df: pd.DataFrame, root: Path):
    style = COMPACT_STYLE
    fig, axes = plt.subplots(nrows=GRID_ROWS, ncols=GRID_COLS, figsize=A4_LANDSCAPE, sharex=True)
    axes = np.asarray(axes)
    geos = ['Tehran', 'Region 12']
    years = sorted(df['year'].unique())
    for i, kpi in enumerate(SECONDARY12_KPIS):
        row = i % GRID_ROWS
        pair = i // GRID_ROWS
        for j, geo in enumerate(geos):
            col = pair * 2 + j
            ax = axes[row, col]
            panel = f"({chr(ord('a') + i)}{j+1})"
            plot_panel(ax, df, kpi, geo, panel, style)
            if row == 0:
                ax.annotate(geo, xy=(0.98, 1.0), xycoords='axes fraction', xytext=(0, 35), textcoords='offset points', ha='right', va='bottom', fontsize=style['region'], fontweight='bold', annotation_clip=False)
            if row < GRID_ROWS - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xticks(years)
                ax.set_xticklabels([str(y) for y in years], fontsize=style['tick'], rotation=90, ha='center', va='top')
                ax.tick_params(axis='x', pad=1)
            if col % 2 == 1:
                ax.set_ylabel('')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    hmap = {lab: h for h, lab in zip(handles, labels)}
    labels = SC_ORDER
    handles = [hmap[lab] for lab in labels]
    reserve_bottom_for_accessory(fig, style['small'], style['legend'], gap_mult=4.0, accessory_type='legend', min_bottom=0.135)
    fig.subplots_adjust(**LINE_SUBPLOTS_ADJUST)
    place_bottom_legend(fig, axes.ravel(), handles, labels, xlabel_fontsize=style['small'], legend_fontsize=style['legend'], gap_mult=4.0, ncol=len(labels), handlelength=1.0, columnspacing=0.65)
    out_svg = root / 'figures' / 'Figure 6.svg'
    from figure_utils import save_figure_svg
    save_figure_svg(fig, out_svg, bbox_inches='tight')
    plt.close(fig)
    return out_svg


if __name__ == '__main__':
    print(generate_figure(load_timeseries(ROOT), ROOT))
