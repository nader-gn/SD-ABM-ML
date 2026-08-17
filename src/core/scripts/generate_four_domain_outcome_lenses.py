from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from figure_utils import style_preset, reserve_bottom_for_accessory, place_bottom_legend, save_figure_svg

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
})

STYLE = style_preset('threequarter')
SC = [f'SC{i}' for i in range(12)]
PILLARS = ['Transportation', 'Environmental', 'Social', 'Economic']
DISPLAY_LABELS = {
    'Transportation': 'Transportation',
    'Environmental': 'Environmental',
    'Social': 'Social',
    'Economic': 'Economic',
}
COLORS = {
    'Transportation': '#4C78A8',
    'Environmental': '#72B463',
    'Social': '#E3B34B',
    'Economic': '#B65C5C',
}


def pivot(df: pd.DataFrame, geo: str) -> np.ndarray:
    mat = np.zeros((len(SC), len(PILLARS)))
    d = df[df['geo'] == geo].copy()
    for i, s in enumerate(SC):
        for j, p in enumerate(PILLARS):
            row = d[(d['scenario'] == s) & (d['pillar'] == p)]
            if row.empty:
                raise ValueError((geo, s, p))
            mat[i, j] = float(row['score'].iloc[0])
    return mat


def draw_panel(ax, mat: np.ndarray, region_label: str, title: str) -> None:
    x = np.arange(len(SC))
    width = 0.18
    offsets = (np.arange(len(PILLARS)) - (len(PILLARS) - 1) / 2) * width
    bar_label_entries = []
    for j, p in enumerate(PILLARS):
        vals = mat[:, j]
        bars = ax.bar(
            x + offsets[j], vals, width=width, color=COLORS[p],
            label=DISPLAY_LABELS[p], edgecolor='white', linewidth=0.65
        )
        for bar, val in zip(bars, vals):
            xc = bar.get_x() + bar.get_width() / 2
            label_pad = 1.05
            inside_y = val - label_pad if val >= 0 else val + label_pad
            inside_va = 'top' if val >= 0 else 'bottom'
            txt = f'{val:+.2f}'
            t = ax.text(xc, inside_y, txt, rotation=90, ha='center', va=inside_va, fontsize=STYLE['cell'], fontfamily='DejaVu Sans Mono')
            bar_label_entries.append((bar, val, t))
    ax.axhline(0, color='black', linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(SC, fontsize=STYLE['tick'])
    ax.set_ylabel('Normalized domain score', fontsize=STYLE['axis'])
    ax.set_title(title, loc='left', fontsize=STYLE['title'])
    ax.text(1.0, 1.04, region_label, transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.35)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=STYLE['tick'])

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bar_zero_px = ax.transData.transform((0, 0))[1]
    for bar, val, t in bar_label_entries:
        bbox = t.get_window_extent(renderer=renderer)
        bar_end_px = ax.transData.transform((0, val))[1]
        bar_len_px = abs(bar_end_px - bar_zero_px)
        txt_len_px = bbox.height
        if txt_len_px > bar_len_px - 2:
            outside_pad = 0.65
            outside_y = val + outside_pad if val >= 0 else val - outside_pad
            outside_va = 'bottom' if val >= 0 else 'top'
            t.set_position((bar.get_x() + bar.get_width() / 2, outside_y))
            t.set_va(outside_va)


def main(root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'outputs' / 'Figure_10_outcome_lens_scores.csv')
    df['pillar'] = df['pillar'].replace({'Social': 'Social', 'Exposure–Health Burden': 'Social'})
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=False)
    draw_panel(axes[0], pivot(df, 'Tehran'), 'Tehran', '(a) Normalized domain score (higher = better within domain)')
    draw_panel(axes[1], pivot(df, 'Region12'), 'Region 12', '(b) Normalized domain score (higher = better within domain)')
    handles, labels = axes[0].get_legend_handles_labels()
    bottom_reserved = reserve_bottom_for_accessory(fig, STYLE['axis'], STYLE['legend'], gap_mult=4.0, accessory_type='legend')
    fig.tight_layout(rect=[0, bottom_reserved, 1, 1])
    fig.canvas.draw()
    union = axes[0].get_position().frozen()
    union = union.union([axes[1].get_position()])
    legend_y = max(0.02, union.y0 - (4.0 * STYLE['axis'] / 72.0) / fig.get_figheight() - (1.55 * STYLE['legend'] / 72.0) / fig.get_figheight())
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        loc='lower center',
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=STYLE['legend'],
        handlelength=1.2,
        columnspacing=1.1,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    save_figure_svg(fig, root / 'figures' / 'Figure 10.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
