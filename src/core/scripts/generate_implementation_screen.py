from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from figure_utils import style_preset, reserve_bottom_for_accessory, place_bottom_colorbar, save_figure_svg
from metric_registry import implementation_metric_order, implementation_display_labels

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
})

STYLE = style_preset('threequarter')
SC = [f'SC{i}' for i in range(12)]
METRIC_ORDER = [
    'Implementation CAPEX',
    'PT subsidy need',
    'Farebox recovery',
    'Net fiscal pressure',
    'Budget use',
    'PT budget pressure',
    'Avoided external cost',
    'Net annual value',
]
DISPLAY = {m: m for m in METRIC_ORDER}


def _labels(root: Path) -> tuple[list[str], dict[str, str]]:
    return implementation_metric_order(root), implementation_display_labels(root)


def _pivot(df: pd.DataFrame, geo: str, metric_order: list[str]) -> pd.DataFrame:
    return (df[df['geo'] == geo].pivot(index='scenario', columns='metric', values='score').reindex(index=SC, columns=metric_order))


def _draw(ax, mat: pd.DataFrame, title: str, region_label: str, show_region: bool, metric_order: list[str], display: dict[str, str]):
    vals = mat.values
    vmax = max(1.0, float(np.nanmax(np.abs(vals))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(vals, aspect='auto', cmap='RdYlGn', norm=norm)
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels([display.get(m, m) for m in metric_order], fontsize=STYLE['tick'], rotation=0)
    ax.set_yticks(range(len(SC)))
    ax.set_yticklabels(SC, fontsize=STYLE['tick'])
    ax.set_title(title, loc='left', fontsize=STYLE['title'])
    if show_region:
        ax.text(1.0, 1.04, region_label, transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SC), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = f'{v:+.2f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=STYLE['cell'] + 1.5, color='black' if abs(v) < 45 else 'white', fontfamily='DejaVu Sans Mono')
    return im


def main(root: Path | None = None):
    if root is None:
        root = Path(__file__).resolve().parents[1]
    metric_order, display = _labels(root)
    df = pd.read_csv(root / 'outputs' / 'Figure_12_implementation_scores.csv')
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.1), gridspec_kw={'height_ratios':[1,1]})
    im = _draw(axes[0], _pivot(df, 'Tehran', metric_order), '(a) Implementation screen (higher = more favorable)', 'Tehran', True, metric_order, display)
    _draw(axes[1], _pivot(df, 'Region12', metric_order), '(b) Implementation screen (higher = more favorable)', 'Region 12', True, metric_order, display)
    axes[1].set_xlabel('Implementation-screen metrics', fontsize=STYLE['axis'])
    bottom_reserved = reserve_bottom_for_accessory(fig, STYLE['axis'], STYLE['cbar_label'], gap_mult=4.0, height_mult=1.0, accessory_type='colorbar')
    fig.tight_layout(rect=[0, bottom_reserved, 1, 1])
    cb = place_bottom_colorbar(fig, axes, im, xlabel_fontsize=STYLE['axis'], label='Normalized implementation-screen value', label_fontsize=STYLE['cbar_label'], tick_fontsize=STYLE['cbar_tick'], gap_mult=4.0, height_mult=1.0)
    cb.outline.set_linewidth(0.6)
    save_figure_svg(fig, root / 'figures' / 'Figure 12.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
