"""Generate the modal-share hindcast/projection grid from workflow-regenerated tidy plot data."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import generate_historical_modal_validation_input
from figure_utils import save_figure_svg, modal_figure_size, apply_scaled_fixed2_yaxis, style_preset, reserve_bottom_for_accessory, axes_union_bbox, fig_y_from_points

MODES = ['car', 'taxi', 'bus', 'metro', 'motorcycle', 'other']
TITLES = {'car':'Car','taxi':'Taxi','bus':'Bus','metro':'Metro','motorcycle':'Motorcycle','other':'Other'}
AREAS = ['Tehran', 'Region12']
AREA_LABEL = {'Tehran':'Tehran', 'Region12':'Region 12'}
ENVELOPE_SCENS = [f's{i}' for i in range(1,12)]
STYLE = style_preset('threequarter')


def metrics_lookup(root: Path) -> dict[tuple[str,str], tuple[float,float]]:
    df = pd.read_csv(root / 'verification' / 'Table_05_validation_per_mode.csv')
    df = df[df['split'] == 'all_hindcast_2012_2023'].copy()
    out = {}
    for _, r in df.iterrows():
        geo = 'Region12' if str(r['geo']).replace(' ', '') == 'Region12' else 'Tehran'
        out[(geo, r['mode'])] = (float(r['MAE_pp']), float(r['RMSE_pp']))
    return out


def main(root: Path):
    root = root.resolve()
    generate_historical_modal_validation_input.main(root)
    df = pd.read_csv(root / 'figure_inputs' / 'Figure_04_modal_share_series.csv')
    metrics = metrics_lookup(root)
    width, height = modal_figure_size(6)
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(width, height * 0.70), sharex=True)
    years = list(range(2012, 2031))
    for i, mode in enumerate(MODES):
        for j, area in enumerate(AREAS):
            ax = axes[i, j]
            sub = df[(df['mode'] == mode) & (df['area'] == area)]
            sim0 = sub[(sub['scenario'] == 's0') & (sub['series'] == 'sim')].sort_values('year')
            obs = sub[sub['series'] == 'obs'].sort_values('year')
            proj = sub[(sub['series'] == 'sim') & (sub['scenario'].isin(ENVELOPE_SCENS)) & (sub['year'].between(2024,2030))]
            if not proj.empty:
                env = proj.groupby('year')['value'].agg(['min','max']).reset_index()
                ax.fill_between(env['year'], env['min'], env['max'], color='#cc8fb7', alpha=0.48, zorder=1)
                for scen in ENVELOPE_SCENS:
                    g = sub[(sub['series'] == 'sim') & (sub['scenario'] == scen) & (sub['year'].between(2024,2030))].sort_values('year')
                    if not g.empty:
                        ax.plot(g['year'], g['value'], color='#9d4f98', alpha=0.24, linewidth=1.15, zorder=2)
            ax.axvspan(2023.5, 2030.5, color='#eadede', alpha=0.7, zorder=0)
            ax.plot(sim0['year'], sim0['value'], color='black', linewidth=1.95, label='Simulated (s0)', zorder=6)
            if not obs.empty:
                ax.scatter(obs['year'], obs['value'], color='red', s=20, label='Observed (hindcast)', zorder=7)
            panel = f"({chr(ord('a') + i)}{j+1})"
            ax.set_title(f"{panel} {TITLES[mode]}", loc='left', fontsize=STYLE['title'], pad=8)
            mae, rmse = metrics.get((area, mode), (float('nan'), float('nan')))
            ax.text(0.5, 1.03, f"[MAE={mae:.2f}pp, RMSE={rmse:.2f}pp]", transform=ax.transAxes, ha='center', va='bottom', fontsize=STYLE['small'])
            if i == 0:
                ax.text(0.98, 1.03, AREA_LABEL[area], transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
            ax.set_xlim(2012, 2030)
            ax.set_xticks(years)
            ax.tick_params(axis='x', labelsize=STYLE['tick'], rotation=90)
            ax.tick_params(axis='y', labelsize=STYLE['tick'])
            y0, y1 = ax.get_ylim()
            apply_scaled_fixed2_yaxis(ax, [y0, y1], label_fontsize=STYLE['small'], text_y=1.01, single_digit_ticks=True)
    for ax in axes[-1, :]:
        ax.set_xlabel('Year', fontsize=STYLE['axis'])
        ax.tick_params(axis='x', labelsize=STYLE['tick'])
    handles = [
        plt.Line2D([0],[0], color='black', lw=1.95),
        plt.Line2D([0],[0], marker='o', color='red', linestyle='None', markersize=5),
        plt.Rectangle((0,0),1,1,color='#cc8fb7', alpha=0.7),
        plt.Line2D([0],[0], color='#9d4f98', lw=1.4, alpha=0.55),
        plt.Rectangle((0,0),1,1,color='#eadede', alpha=0.7),
    ]
    labels = ['Simulated (s0)', 'Observed (hindcast)', 'Scenario envelope (projection)', 'Scenario paths (projection)', 'Projection window (2024–2030)']
    bottom_reserved = max(
        reserve_bottom_for_accessory(fig, STYLE['axis'], STYLE['legend'], gap_mult=2.0, accessory_type='legend'),
        fig_y_from_points(fig, 4.1 * STYLE['axis'] + 1.55 * STYLE['legend']) + 0.02,
    )
    fig.tight_layout(rect=[0, bottom_reserved, 1, 1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = axes_union_bbox(axes.ravel())
    xlabel_boxes = []
    for ax in axes[-1, :]:
        lab = ax.xaxis.get_label()
        try:
            bb = lab.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            xlabel_boxes.append(bb)
        except Exception:
            pass
    xlabel_bottom = min((bb.y0 for bb in xlabel_boxes), default=bbox.y0)
    gap = fig_y_from_points(fig, 2.0 * STYLE['axis'])
    legend_h = fig_y_from_points(fig, 1.55 * STYLE['legend'])
    legend_y0 = max(0.02, xlabel_bottom - gap - legend_h)
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        loc='lower left',
        bbox_to_anchor=(bbox.x0, legend_y0, bbox.width, legend_h),
        bbox_transform=fig.transFigure,
        mode='expand',
        borderaxespad=0.0,
        frameon=False,
        fontsize=STYLE['legend'],
        handlelength=1.4,
        columnspacing=0.9,
    )
    out_dir = root / 'figures'; out_dir.mkdir(exist_ok=True)
    save_figure_svg(fig, out_dir / 'Figure 4.svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='.')
    args = ap.parse_args(); main(Path(args.root))
