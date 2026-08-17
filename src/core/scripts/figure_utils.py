from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import io
import re
import xml.etree.ElementTree as ET
import math
from typing import Iterable
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox

SVG_HASHSALT = "tehran-sd-abm-ml-reproducibility"
SVG_METADATA = {"Date": None}

REFERENCE_FIG_WIDTH = 11.0
REFERENCE_ROW_HEIGHT = 12.5 / 8.0
LINE_FIG_EXTRA_HEIGHT = 0.9
HEATMAP_FIG_EXTRA_HEIGHT = 2.1
MODAL_FIG_EXTRA_HEIGHT = 0.9

_STYLE_PRESETS = {
    "full": {
        "title": 11.8,
        "region": 12.8,
        "axis": 10.4,
        "tick": 9.4,
        "legend": 9.1,
        "cbar_label": 9.6,
        "cbar_tick": 8.9,
        "cell": 7.7,
        "mono": 7.5,
        "small": 8.4,
    },
    "threequarter": {
        "title": 11.4,
        "region": 12.4,
        "axis": 10.0,
        "tick": 9.0,
        "legend": 8.8,
        "cbar_label": 9.2,
        "cbar_tick": 8.6,
        "cell": 7.5,
        "mono": 7.3,
        "small": 8.2,
    },
    "half": {
        "title": 11.0,
        "region": 12.0,
        "axis": 9.7,
        "tick": 8.8,
        "legend": 8.6,
        "cbar_label": 9.0,
        "cbar_tick": 8.4,
        "cell": 7.3,
        "mono": 7.1,
        "small": 8.0,
    },
}


def configure_deterministic_matplotlib() -> None:
    mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT


def canonicalize_svg_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    try:
        out = io.StringIO()
        ET.canonicalize(xml_data=txt, out=out, with_comments=False)
        txt = out.getvalue()
    except Exception:
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        txt = re.sub(r">\s+<", "><", txt)
        txt = re.sub(r"\n+", "\n", txt).strip() + "\n"
    return txt


def svg_c14n_md5(path: Path) -> str:
    return hashlib.md5(canonicalize_svg_text(path).encode("utf-8")).hexdigest()


def save_figure_svg(fig, out_svg: Path, *, bbox_inches: str | None = "tight") -> None:
    """Save a publication figure as SVG only and remove embedded generator metadata."""
    configure_deterministic_matplotlib()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches=bbox_inches, metadata=SVG_METADATA)
    txt = out_svg.read_text(encoding="utf-8")
    txt = re.sub(r"<metadata>.*?</metadata>", "", txt, flags=re.S)
    out_svg.write_text(txt, encoding="utf-8")


def scientific_scale_for_values(values, *, threshold_high: float = 1e4, threshold_low: float = 1e-2, single_digit_ticks: bool = True) -> tuple[float, str]:
    vals = [abs(float(v)) for v in values if v is not None]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 1.0, ""
    vmax = max(vals)
    if vmax == 0:
        return 1.0, ""
    if single_digit_ticks:
        if 0.1 <= vmax < 10:
            return 1.0, ""
        exp = int(math.floor(math.log10(vmax)))
        exp = max(exp, 0)
    else:
        if threshold_low <= vmax < threshold_high:
            return 1.0, ""
        exp = int(math.floor(math.log10(vmax)))
    scale = 10.0 ** exp
    return scale, f"×1e{exp}"


def apply_scaled_fixed2_yaxis(ax, values, *, label_fontsize: int = 8, text_x: float | None = None, text_y: float = 1.01, single_digit_ticks: bool = True) -> tuple[float, str]:
    scale, scale_label = scientific_scale_for_values(values, single_digit_ticks=single_digit_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x/scale:.2f}"))
    if scale_label and scale_label != "×1e0":
        fig = ax.figure
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            labels = [lab for lab in ax.get_yticklabels() if lab.get_text()]
            if labels:
                boxes = [lab.get_window_extent(renderer=renderer) for lab in labels]
                x_disp = float(sum(bb.x0 + 0.15 * bb.width for bb in boxes) / len(boxes))
                x_axes = float(ax.transAxes.inverted().transform((x_disp, 0))[0])
            else:
                x_axes = -0.085
        except Exception:
            x_axes = -0.085
        if text_x is not None:
            x_axes = text_x
        ax.text(x_axes, text_y, scale_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=label_fontsize, clip_on=False)
    return scale, scale_label


def discrete_diverging_with_neutral(bounds, *, base_name: str = "coolwarm", neutral_interval: tuple[float, float] = (-1, 1), neutral_rgba=(0.94, 0.94, 0.94, 1.0)):
    n_bins = len(bounds) - 1
    base = mpl.colormaps.get_cmap(base_name).resampled(n_bins)
    colors = [base(i) for i in range(base.N)]
    low, high = neutral_interval
    for i in range(n_bins):
        left = float(bounds[i]); right = float(bounds[i+1])
        if right <= low or left >= high:
            continue
        colors[i] = neutral_rgba
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")
    return cmap


def line_figure_size(nrows: int, ncols: int = 2) -> tuple[float, float]:
    return (REFERENCE_FIG_WIDTH, REFERENCE_ROW_HEIGHT * nrows + LINE_FIG_EXTRA_HEIGHT)


def heatmap_figure_size(nrows: int, ncols: int = 2) -> tuple[float, float]:
    return (REFERENCE_FIG_WIDTH, REFERENCE_ROW_HEIGHT * nrows + HEATMAP_FIG_EXTRA_HEIGHT)


def modal_figure_size(nrows: int, ncols: int = 2) -> tuple[float, float]:
    return (REFERENCE_FIG_WIDTH, REFERENCE_ROW_HEIGHT * nrows + MODAL_FIG_EXTRA_HEIGHT)


def style_preset(kind: str) -> dict:
    if kind not in _STYLE_PRESETS:
        raise KeyError(kind)
    return dict(_STYLE_PRESETS[kind])


def fig_y_from_points(fig, pts: float) -> float:
    return (pts / 72.0) / fig.get_figheight()


def fig_x_from_points(fig, pts: float) -> float:
    return (pts / 72.0) / fig.get_figwidth()


def axes_union_bbox(axes: Iterable) -> Bbox:
    axes = list(axes)
    return Bbox.union([ax.get_position() for ax in axes])


def reserve_bottom_for_accessory(fig, xlabel_fontsize: float, accessory_fontsize: float, *, gap_mult: float = 4.0, height_mult: float = 1.0, accessory_type: str = "colorbar", min_bottom: float = 0.07) -> float:
    gap = fig_y_from_points(fig, gap_mult * xlabel_fontsize)
    height = fig_y_from_points(fig, height_mult * xlabel_fontsize)
    if accessory_type == "legend":
        acc_h = max(height, fig_y_from_points(fig, 1.55 * accessory_fontsize))
    else:
        acc_h = height + fig_y_from_points(fig, 1.55 * accessory_fontsize)
    return max(min_bottom, gap + acc_h + 0.02)


def place_bottom_colorbar(fig, axes, mappable, *, xlabel_fontsize: float, label: str = "", ticks=None, label_fontsize: float = 9.0, tick_fontsize: float = 8.0, gap_mult: float = 4.0, height_mult: float = 1.0, formatter=None):
    fig.canvas.draw()
    bbox = axes_union_bbox(axes)
    gap = fig_y_from_points(fig, gap_mult * xlabel_fontsize)
    height = fig_y_from_points(fig, height_mult * xlabel_fontsize)
    y0 = max(0.02, bbox.y0 - gap - height)
    cax = fig.add_axes([bbox.x0, y0, bbox.width, height])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    if label:
        cb.set_label(label, fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)
    if formatter is not None:
        cb.ax.xaxis.set_major_formatter(formatter)
    return cb


def place_bottom_legend(fig, axes, handles, labels, *, xlabel_fontsize: float, legend_fontsize: float = 9.0, gap_mult: float = 4.0, ncol: int | None = None, handlelength: float = 1.4, columnspacing: float = 1.0):
    fig.canvas.draw()
    bbox = axes_union_bbox(axes)
    gap = fig_y_from_points(fig, gap_mult * xlabel_fontsize)
    legend_h = fig_y_from_points(fig, 1.55 * legend_fontsize)
    y0 = max(0.02, bbox.y0 - gap - legend_h)
    leg = fig.legend(
        handles,
        labels,
        ncol=(ncol or len(labels)),
        loc="lower left",
        bbox_to_anchor=(bbox.x0, y0, bbox.width, legend_h),
        bbox_transform=fig.transFigure,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )
    return leg


def set_exact_x_span(ax, xs) -> None:
    xs = list(xs)
    if not xs:
        return
    ax.set_xlim(min(xs), max(xs))
    ax.margins(x=0)
