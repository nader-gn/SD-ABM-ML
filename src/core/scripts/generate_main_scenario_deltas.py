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
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from figure_utils import save_figure_svg, discrete_diverging_with_neutral, style_preset, reserve_bottom_for_accessory, place_bottom_colorbar
from scenario_meta import SCENARIO_CODES as SC_ORDER
from figure_kpi_source_utils import DECISION12_KPIS, UNIT_MAP_24, choose_source_rows, dynamic_delta_table, write_csv

STYLE = style_preset('half')
COMPACT_STYLE = dict(STYLE)
COMPACT_STYLE.update({'title': 10.0, 'region': 12.0, 'axis': 9.2, 'tick': 8.2, 'legend': 8.0, 'cbar_label': 9.0, 'cbar_tick': 8.0, 'mono': 5.8, 'small': 7.8})
A4_LANDSCAPE = (11.0, 10.55)  # extra height gives SC0-SC11 labels clear vertical separation
SUBPLOT_TARGET_SCALE = 1.10
GRID_ROWS = 6
GRID_COLS = 4
HEATMAP_SUBPLOTS_ADJUST = dict(left=0.072, right=0.995, top=0.920, bottom=0.155, wspace=0.24, hspace=0.34)
ROOT = SCRIPT_DIR.parent
DISPLAY_MAP = {'CO₂ emissions': 'Emissions: CO₂', 'NOₓ emissions': 'Emissions: NOₓ', 'PM₂.₅ emissions': 'Emissions: PM₂.₅'}
BOUNDS = np.array([-50, -40, -15, -4, -1, -0.1, 0, 0.1, 1, 4, 15, 40, 50], dtype=float)
DHCOL_GAP = 0.03
DHCOL_WIDTH = 0.95
GRAY_TXT = '#7a7a7a'


def load_decision12_timeseries(root: Path):
    precomputed = root / 'figure_data' / 'Figure_07_main_scenario_deltas.csv'
    if precomputed.exists():
        return pd.read_csv(precomputed)
    df, audit = choose_source_rows(root, DECISION12_KPIS, UNIT_MAP_24)
    write_csv(audit, root / 'verification' / 'Figure_07_source_audit.csv')
    return df


def _annotation_style(v: float):
    av = abs(float(v))
    if av <= 1:
        return GRAY_TXT, 1.0
    return ('white', 1.0) if av >= 15 else ('black', 1.0)


def _draw_dh_column(ax, row_mean, cmap, norm, fmt, fs, n_years):
    x0 = (n_years - 0.5) + DHCOL_GAP
    dh_center = x0 + DHCOL_WIDTH / 2.0
    for si, mean_v in enumerate(row_mean):
        if np.isnan(mean_v):
            face = 'white'; label = ' nan'; txt_color, alpha = 'black', 1.0
        else:
            face = cmap(norm(mean_v)); label = fmt.format(mean_v); txt_color, alpha = _annotation_style(mean_v)
        ax.add_patch(Rectangle((x0, si - 0.5), DHCOL_WIDTH, 1.0, facecolor=face, edgecolor='none', linewidth=0.0, clip_on=False, zorder=3))
        ax.text(dh_center, si, label, ha='center', va='center', fontsize=fs, fontfamily='DejaVu Sans Mono', color=txt_color, alpha=alpha, clip_on=False, zorder=4)
    return dh_center


def generate_figure(df: pd.DataFrame, root: Path):
    style = COMPACT_STYLE
    geos = ['Tehran', 'Region 12']
    years = sorted(df['year'].unique())
    n_rows = len(SC_ORDER)
    fs = 4.9
    fmt = '{:+04.1f}'
    cmap = discrete_diverging_with_neutral(BOUNDS, base_name='coolwarm', neutral_interval=(-0.1, 0.1), neutral_rgba=(0.94, 0.94, 0.94, 1.0))
    norm = BoundaryNorm(BOUNDS, cmap.N, clip=True)
    fig, axes = plt.subplots(nrows=GRID_ROWS, ncols=GRID_COLS, figsize=A4_LANDSCAPE, sharex=False)
    axes = np.asarray(axes)
    mode_rows = []
    plot_data_rows = []
    precomputed = {'delta_display', 'annot', 'annot_mode', 'kpi'}.issubset(df.columns) and 'value' not in df.columns
    for i, kpi in enumerate(DECISION12_KPIS):
        if precomputed:
            d = df[df['kpi'] == kpi].copy()
            mode_audit = (d[['Geo', 'annot_mode']].drop_duplicates()
                            .assign(kpi=kpi, delta_mode=lambda x: x['annot_mode'].map({'raw': 'raw_normalized', 'pct': 'percent'}))
                            [['kpi', 'Geo', 'delta_mode']])
        else:
            d, mode_audit = dynamic_delta_table(df, kpi)
        mode_rows.append(mode_audit)
        d_export = d.copy()
        d_export['kpi'] = kpi
        plot_data_rows.append(d_export)
        title = DISPLAY_MAP.get(kpi, kpi)
        row = i % GRID_ROWS
        pair = i // GRID_ROWS
        for j, geo in enumerate(geos):
            col = pair * 2 + j
            ax = axes[row, col]
            mat = np.full((n_rows, len(years)), np.nan)
            for si, s in enumerate(SC_ORDER):
                for yi, y in enumerate(years):
                    v = d[(d['Scenario'] == s) & (d['Geo'] == geo) & (d['year'] == y)]['delta_display']
                    if len(v):
                        mat[si, yi] = float(v.values[0])
            row_mean = np.nanmean(mat, axis=1)
            ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=0.28)
            ax.tick_params(which='minor', bottom=False, left=False)
            panel = f"({chr(ord('a') + i)}{1 if j == 0 else 2})"
            ax.set_title(f"{panel} {title}", loc='left', fontsize=style['title'], pad=3)
            if row == 0:
                ax.annotate(geo, xy=(0.98, 1.0), xycoords='axes fraction', xytext=(0, 35), textcoords='offset points', ha='right', va='bottom', fontsize=style['region'], fontweight='bold', annotation_clip=False)
            dh_center = _draw_dh_column(ax, row_mean, cmap, norm, fmt, fs, len(years))
            ax.set_xlim(-0.5, dh_center + DHCOL_WIDTH / 2.0 + 0.04)
            if row == GRID_ROWS - 1:
                xticks = list(range(len(years))) + [dh_center]
                labels = [str(y) for y in years[:-1]] + [f"{years[-1]}\n$\\mathbf{{[\\Delta e]}}$", "ALL\n$\\mathbf{[\\Delta h]}$"]
                ax.set_xticks(xticks)
                ax.set_xticklabels(labels, fontsize=style['tick'] - 0.2, rotation=90, ha='center', va='top')
                ax.tick_params(axis='x', labelbottom=True, length=2.0, width=0.55, pad=1)
            else:
                ax.set_xticks(list(range(len(years))))
                ax.set_xticklabels([])
                ax.tick_params(axis='x', labelbottom=False, length=2.0, width=0.55, pad=1)
            ax.set_yticks(range(n_rows))
            if col in (0, 2):
                ax.set_yticklabels(SC_ORDER, fontsize=(style['tick'] - 1.0))
                ax.tick_params(axis='y', pad=5.0)
                for lab in ax.get_yticklabels():
                    lab.set_horizontalalignment('right')
                    lab.set_x(-0.02)
            else:
                ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_zorder(10)
            for si in range(n_rows):
                for yi in range(len(years)):
                    v = mat[si, yi]
                    if np.isnan(v):
                        continue
                    txt_color, alpha = _annotation_style(v)
                    ax.text(yi, si, fmt.format(v), ha='center', va='center', fontsize=fs, fontfamily='DejaVu Sans Mono', color=txt_color, alpha=alpha, zorder=5)
    write_csv(pd.concat(mode_rows, ignore_index=True), root / 'verification' / 'Figure_07_delta_modes.csv')
    write_csv(pd.concat(plot_data_rows, ignore_index=True), root / 'figure_data' / 'Figure_07_main_scenario_deltas.csv')
    reserve_bottom_for_accessory(fig, style['axis'], style['cbar_label'], gap_mult=4.8, height_mult=0.9, accessory_type='colorbar', min_bottom=0.145)
    fig.subplots_adjust(**HEATMAP_SUBPLOTS_ADJUST)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = place_bottom_colorbar(fig, axes.ravel(), mappable, xlabel_fontsize=style['axis'], label='Δ vs SC0 (%)', ticks=[-40, -15, -4, -1, -0.1, 0, 0.1, 1, 4, 15, 40], label_fontsize=style['cbar_label'], tick_fontsize=style['cbar_tick'], gap_mult=4.8, height_mult=0.9)
    cb.outline.set_linewidth(0.6)
    out_svg = root / 'figures' / 'Figure 7.svg'
    save_figure_svg(fig, out_svg, bbox_inches='tight')
    plt.close(fig)
    return out_svg


if __name__ == '__main__':
    print(generate_figure(load_decision12_timeseries(ROOT), ROOT))
