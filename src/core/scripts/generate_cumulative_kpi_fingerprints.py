from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge, Arc, Rectangle

from figure_utils import save_figure_svg, style_preset
from figure_kpi_source_utils import SELECTED24_KPIS, refresh_metric_globals
from scenario_meta import SCENARIO_CODES as ALL_SCENARIOS

STYLE = style_preset("full")
KPI_ORDER = SELECTED24_KPIS
SCENARIOS = [s for s in ALL_SCENARIOS if s != "SC0"]

SHORT = [
    "Car time loss", "PT share", "PCE-VKT",
    "CO₂ emissions", "NOₓ emissions", "PM₂.₅ emissions",
    "Health burden", "PT affordability", "Noise exposure",
    "PT OPEX", "Recurrent burden", "Energy cost",
    "Congestion index", "PT trips", "Non-PT trips",
    "Budget use", "Final energy", "Electricity use",
    "Implementation CAPEX", "PT subsidy need", "Net fiscal pressure",
    "Farebox recovery", "Avoided external cost", "Net annual value",
]
CORE_GROUPS = [
    ("Transportation", 1, 3),
    ("Environmental", 4, 6),
    ("Social", 7, 9),
    ("Economic", 10, 12),
]

# Higher-contrast muted natural diverging palette. The legend uses discrete bins
# so the differences are clear in print and on screen.
BOUNDARIES = np.array([-50, -40, -15, -4, -1, -0.1, 0, 0.1, 1, 4, 15, 40, 50], dtype=float)
TICKS = np.array([-40, -15, -4, -1, -0.1, 0, 0.1, 1, 4, 15, 40], dtype=float)
BIN_COLORS = [
    "#0B4F4A", "#176D66", "#358B82", "#67AAA0", "#A0C9BF",
    "#F3EEE3", "#F3EEE3",
    "#F0D7C9", "#E7AD90", "#D47D5F", "#B9533D", "#8B2E25",
]
CMAP = mpl.colors.ListedColormap(BIN_COLORS)
NORM = mpl.colors.BoundaryNorm(BOUNDARIES, ncolors=CMAP.N, clip=True)

FS_TITLE = STYLE["title"]
FS_PANEL = STYLE["region"] - 0.2
FS_NUM = STYLE["small"] + 0.2
FS_SC = STYLE["small"] + 0.2
FS_KEY_HEAD = STYLE["title"] + 0.1
FS_KEY_GROUP = STYLE["small"] + 0.8
FS_KEY_ITEM = STYLE["small"] + 0.4
FS_CBAR = STYLE["cbar_label"] - 0.1
FS_TICK = STYLE["cbar_tick"] - 0.2
FS_NOTE = STYLE["small"] + 0.1

INNER = 0.78
RING_H = 0.34
SECTOR_GAP = 0.42

def polar_xy(cx: float, cy: float, r: float, ang_deg: float) -> tuple[float, float]:
    a = np.deg2rad(ang_deg)
    return cx + r * np.cos(a), cy + r * np.sin(a)

def mat_for_geo(df: pd.DataFrame, geo: str) -> np.ndarray:
    out = np.zeros((len(SCENARIOS), len(KPI_ORDER)))
    sub = df[df["geo"] == geo]
    for i, sc in enumerate(SCENARIOS):
        for j, kpi in enumerate(KPI_ORDER):
            v = sub[(sub["scenario"] == sc) & (sub["kpi"] == kpi)]["auc"]
            out[i, j] = float(v.iloc[0]) if len(v) else 0.0
    return out

def draw_fan(ax, cx: float, cy: float, mat: np.ndarray, title: str, letter: str) -> float:
    outer = INNER + len(SCENARIOS) * RING_H
    sector = 180.0 / len(KPI_ORDER)

    for i in range(len(SCENARIOS)):
        r0 = INNER + i * RING_H
        for j in range(len(KPI_ORDER)):
            ax.add_patch(Wedge(
                (cx, cy),
                r0 + RING_H * 0.94,
                j * sector + SECTOR_GAP,
                (j + 1) * sector - SECTOR_GAP,
                width=RING_H * 0.90,
                facecolor=CMAP(NORM(mat[i, j])),
                edgecolor="white",
                linewidth=0.40,
            ))

    for i in range(len(SCENARIOS) + 1):
        rr = INNER + i * RING_H
        ax.add_patch(Arc(
            (cx, cy), 2 * rr, 2 * rr,
            theta1=0, theta2=180,
            linewidth=0.28, alpha=0.16, color="#3A3A3A",
        ))

    for idx, lw, alpha in [(3, 0.54, 0.33), (6, 0.54, 0.33), (9, 0.54, 0.33), (12, 0.96, 0.60)]:
        a = idx * sector
        x1, y1 = polar_xy(cx, cy, INNER - 0.03, a)
        x2, y2 = polar_xy(cx, cy, outer + 0.12, a)
        ax.plot([x1, x2], [y1, y2], color="#3F3F3F", linewidth=lw, alpha=alpha)

    for j in range(24):
        a = (j + 0.5) * sector
        x, y = polar_xy(cx, cy, outer + 0.31, a)
        ax.text(x, y, str(j + 1), ha="center", va="center", fontsize=FS_NUM)

    # All scenario labels shown, rotated 90°, aligned to ring centers.
    baseline_y = cy - 0.13
    for idx, sc in enumerate(SCENARIOS):
        rc = INNER + (idx + 0.47) * RING_H
        x = cx + rc
        ax.plot([x, x], [cy - 0.03, baseline_y + 0.05], color="#B9B3AA", linewidth=0.42)
        ax.text(x, baseline_y, sc, ha="center", va="top", fontsize=FS_SC,
                rotation=90, rotation_mode="anchor")

    ax.text(cx, cy + outer + 0.88, title, ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold")
    ax.text(cx - outer - 0.48, cy + outer + 0.84, letter, ha="left", va="bottom",
            fontsize=FS_PANEL, fontweight="bold")
    return outer

def draw_key_and_colorbar(ax) -> None:
    # Compact vertical stacking to eliminate the former blank band.
    key_top = -0.95
    ax.text(-10.0, key_top + 0.44, "KPI key", fontsize=FS_KEY_HEAD, fontweight="bold", ha="left")

    core_x = [-9.9, -7.2, -4.5, -1.8]
    for x, (g, i0, i1) in zip(core_x, CORE_GROUPS):
        ax.text(x, key_top + 0.10, g, fontsize=FS_KEY_GROUP, fontweight="bold", ha="left")
        for rr, idx in enumerate(range(i0, i1 + 1)):
            ax.text(x, key_top - 0.24 - rr * 0.37, f"{idx}. {SHORT[idx-1]}",
                    fontsize=FS_KEY_ITEM, ha="left")

    ax.text(1.30, key_top + 0.10, "Supporting indicators",
            fontsize=FS_KEY_GROUP, fontweight="bold", ha="left")
    for col, start in enumerate([13, 19]):
        x = 1.30 + col * 4.05
        for rr, idx in enumerate(range(start, start + 6)):
            ax.text(x, key_top - 0.24 - rr * 0.37, f"{idx}. {SHORT[idx-1]}",
                    fontsize=FS_KEY_ITEM, ha="left")

    # Discrete segmented legend with visible segment boundaries.
    bar_x0, bar_x1 = -7.0, 7.0
    bar_y0, bar_h = -3.92, 0.26
    seg_n = len(BIN_COLORS)
    seg_w = (bar_x1 - bar_x0) / seg_n
    for i, color in enumerate(BIN_COLORS):
        x0 = bar_x0 + i * seg_w
        ax.add_patch(Rectangle((x0, bar_y0), seg_w, bar_h,
                               facecolor=color, edgecolor="white", linewidth=0.55))
    ax.add_patch(Rectangle((bar_x0, bar_y0), bar_x1 - bar_x0, bar_h,
                           fill=False, edgecolor="#57534E", linewidth=0.66))

    # Map ticks to bin boundaries.
    vmin, vmax = BOUNDARIES[0], BOUNDARIES[-1]
    for t in TICKS:
        # piecewise linear mapping through boundaries
        idx = np.searchsorted(BOUNDARIES, t, side="right") - 1
        idx = max(0, min(idx, len(BOUNDARIES) - 2))
        left, right = BOUNDARIES[idx], BOUNDARIES[idx + 1]
        frac_in_bin = 0.0 if right == left else (t - left) / (right - left)
        frac = (idx + frac_in_bin) / (len(BOUNDARIES) - 1)
        x = bar_x0 + frac * (bar_x1 - bar_x0)
        ax.plot([x, x], [bar_y0, bar_y0 - 0.095], color="#4A4742", linewidth=0.66)
        ax.text(x, bar_y0 - 0.16, f"{t:g}", fontsize=FS_TICK, ha="center", va="top")

    ax.text(0, bar_y0 - 0.52, "Cumulative change relative to SC0 (AUC, %·year)",
            ha="center", va="top", fontsize=FS_CBAR)
    ax.text(0, -4.82,
            "SC0 is omitted because baseline-relative AUCs are zero. Sign indicates direction, not desirability; preference orientation is applied in Figure 10.",
            ha="center", va="top", fontsize=FS_NOTE, color="#625E57")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    refresh_metric_globals(root)

    data_path = root / "figure_inputs" / "Figure_09_auc_input.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fresh Figure 9 input was not generated: {data_path}. "
            "Run the upstream core extraction steps before figure generation."
        )
    df = pd.read_csv(data_path)

    teh = mat_for_geo(df, "Tehran")
    r12 = mat_for_geo(df, "Region12")

    fig = plt.figure(figsize=(10.7, 6.5))
    ax = fig.add_axes([0.018, 0.03, 0.964, 0.94])
    ax.set_aspect("equal")
    ax.axis("off")

    centers = [(-5.25, 2.80), (5.25, 2.80)]
    draw_fan(ax, *centers[0], teh, "Tehran", "a")
    draw_fan(ax, *centers[1], r12, "Region 12", "b")
    draw_key_and_colorbar(ax)

    ax.set_xlim(-10.7, 10.7)
    ax.set_ylim(-5.12, 8.15)

    out_svg = root / "figures" / "Figure 9.svg"
    out_svg.parent.mkdir(exist_ok=True)
    save_figure_svg(fig, out_svg, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
