from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
from scenario_meta import SCENARIO_TO_COLOR
from figure_utils import style_preset, reserve_bottom_for_accessory, place_bottom_legend, place_bottom_colorbar, save_figure_svg
from metric_registry import canonical_labels

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
})

STYLE = style_preset('full')
PILLARS = ['Transportation', 'Social', 'Environmental', 'Economic']
PILLAR_SHORT = {p: p for p in PILLARS}
TWO_DEC_FMT = FormatStrFormatter('%.2f')


def widen_column_gap(fig, left_axes, right_axes, delta_mm: float = 2.0) -> None:
    delta_fig = (delta_mm / 25.4) / fig.get_figwidth()
    half = delta_fig / 2.0
    for ax in left_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width - half, pos.height])
    for ax in right_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + half, pos.y0, pos.width - half, pos.height])


def load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / 'config' / 'decision_architecture.yaml').read_text(encoding='utf-8'))


def feasibility_screen(root: Path, candidate_codes: list[str], gate_cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(root / 'outputs' / 'implementation_feasibility_metric_timeseries.csv')
    N = canonical_labels(root)
    rows = []
    for geo in ['Tehran', 'Region12']:
        for sc in candidate_codes:
            sub = raw[(raw['scenario'] == sc) & (raw['geo'] == geo)]
            mean_net = float(sub[sub['metric'] == N['net_annual_value']]['value'].mean())
            mean_budget = float(sub[sub['metric'] == N['budget_use']]['value'].mean())
            mean_ptbudget = float(sub[sub['metric'] == N['pt_budget_pressure']]['value'].mean())
            ok = (
                mean_net >= float(gate_cfg['min_mean_net_annual_value']) and
                mean_budget <= float(gate_cfg['max_mean_budget_use']) and
                mean_ptbudget <= float(gate_cfg['max_mean_pt_budget_pressure'])
            )
            rows.append({'scenario': sc, 'geo': geo, 'mean_net_annual_value': mean_net, 'mean_budget_use': mean_budget, 'mean_pt_budget_pressure': mean_ptbudget, 'passes_gate': bool(ok)})
    return pd.DataFrame(rows)


def scenario_palette(scenarios: list[str]) -> dict[str, str]:
    return {sc: SCENARIO_TO_COLOR[sc] for sc in scenarios}


def draw_equal_weight(ax, d: pd.DataFrame, title: str, region_label: str, scenarios: list[str], colors_map: dict[str, str], show_region: bool) -> None:
    d = d.set_index('scenario').loc[scenarios].reset_index().sort_values('equal_weight_score', ascending=True)
    y = np.arange(len(d))
    colors = [colors_map[s] for s in d['scenario']]
    ax.barh(y, d['equal_weight_score'], color=colors, edgecolor='white', linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d['scenario'], fontsize=STYLE['tick'])
    ax.set_xlabel('Equal-weight score', fontsize=STYLE['axis'])
    ax.set_title(title, loc='left', fontsize=STYLE['title'])
    if show_region:
        ax.text(1.0, 1.04, region_label, transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
    ax.axvline(0, color='black', linewidth=1.0)
    ax.grid(True, axis='x', linewidth=0.3, alpha=0.35)
    ax.xaxis.set_major_formatter(TWO_DEC_FMT)
    xmax = max(1.0, float(d['equal_weight_score'].max()))
    xmin = float(d['equal_weight_score'].min())
    if xmin >= 0:
        ax.set_xlim(-0.01 * max(xmax, 1.0), xmax + 0.05 * max(xmax, 1.0))
    else:
        pad = 0.05 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)
    ax.tick_params(axis='x', labelsize=STYLE['tick'])
    for yi, (_, r) in zip(y, d.iterrows()):
        x = r['equal_weight_score']
        ha = 'left' if x >= 0 else 'right'
        dx = 0.35 if x >= 0 else -0.35
        ax.text(x + dx, yi, f"{x:.2f}", va='center', ha=ha, fontsize=STYLE['mono'], fontfamily='DejaVu Sans Mono')


def draw_accept(ax, d: pd.DataFrame, title: str, region_label: str, scenarios: list[str], show_region: bool) -> None:
    d = d.set_index('scenario').loc[scenarios].reset_index().sort_values('mean_rank')
    disp = np.column_stack([
        d['win_probability'].values,
        d['top2_probability'].values,
        d['top3_probability'].values,
        1 - (d['mean_rank'].values - 1) / max(len(scenarios) - 1, 1),
    ])
    ax.imshow(disp, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Win', 'Top-2', 'Top-3', 'Mean rank\n(best→worst)'], fontsize=STYLE['tick'])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d['scenario'], fontsize=STYLE['tick'])
    ax.set_title(title, loc='left', fontsize=STYLE['title'])
    if show_region:
        ax.text(1.0, 1.04, region_label, transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
    for i in range(len(d)):
        vals = d.iloc[i]
        texts = [f"{100*vals['win_probability']:.1f}%", f"{100*vals['top2_probability']:.1f}%", f"{100*vals['top3_probability']:.1f}%", f"{vals['mean_rank']:.2f}"]
        for j, txt in enumerate(texts):
            ax.text(j, i, txt, ha='center', va='center', fontsize=STYLE['mono'], color='black' if disp[i, j] < 0.62 else 'white', fontfamily='DejaVu Sans Mono')
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(d), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_ylabel('Scenario', fontsize=STYLE['axis'])


def draw_winner_map(ax, winners: pd.DataFrame, alphas: np.ndarray, geo: str, title: str, region_label: str, scenarios: list[str], colors_map: dict[str, str], show_region: bool) -> None:
    d = winners[winners.geo == geo].copy()
    mat = np.zeros((len(PILLARS), len(alphas)), dtype=int)
    for i, p in enumerate(PILLARS):
        dp = d[d.focus == p].sort_values('alpha')
        mat[i, :] = [scenarios.index(s) for s in dp['winner']]
    cmap = ListedColormap([colors_map[s] for s in scenarios])
    norm = BoundaryNorm(np.arange(-0.5, len(scenarios) + 0.5, 1), cmap.N)
    ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm, extent=[alphas.min(), alphas.max(), len(PILLARS)-0.5, -0.5])
    ax.set_yticks(range(len(PILLARS)))
    ax.set_yticklabels([PILLAR_SHORT[p] for p in PILLARS], fontsize=STYLE['tick'])
    xticks = np.linspace(0, 1, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x:.2f}' for x in xticks], fontsize=STYLE['tick'])
    ax.set_xlabel('Focused dimension weight', fontsize=STYLE['axis'])
    ax.set_title(title, loc='left', fontsize=STYLE['title'])
    if show_region:
        ax.text(1.0, 1.04, region_label, transform=ax.transAxes, ha='right', va='bottom', fontsize=STYLE['region'], fontweight='bold')
    for i, p in enumerate(PILLARS):
        dp = d[d.focus == p].sort_values('alpha')
        seg_start = float(dp.iloc[0]['alpha'])
        cur = dp.iloc[0]['winner']
        prev = seg_start
        for _, r in dp.iloc[1:].iterrows():
            if r['winner'] != cur:
                if prev - seg_start >= 0.10:
                    ax.text((seg_start + prev) / 2, i, cur, ha='center', va='center', fontsize=STYLE['small'], fontweight='bold', color='black')
                seg_start = float(r['alpha'])
                cur = r['winner']
            prev = float(r['alpha'])
        if prev - seg_start >= 0.10:
            ax.text((seg_start + prev) / 2, i, cur, ha='center', va='center', fontsize=STYLE['small'], fontweight='bold', color='black')
    ax.axvline(0.25, color='black', linestyle='--', linewidth=1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PILLARS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)


def main(root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(root)
    scores = pd.read_csv(root / 'outputs' / 'Figure_10_outcome_lens_scores.csv')
    candidate_codes = [s for s in scores['scenario'].unique() if s != 'SC0']
    scores['pillar'] = scores['pillar'].replace({'Social': 'Social', 'Exposure–Health Burden': 'Social'})
    gate = feasibility_screen(root, candidate_codes, cfg['shortlist_gate'])
    gate.to_csv(root / 'verification' / 'candidate_shortlist_gate.csv', index=False)
    active_by_geo = {geo: candidate_codes[:] for geo in ['Tehran','Region12']}

    equal_rows=[]; accept_rows=[]; winner_rows=[]
    alphas = np.linspace(0,1,51)
    for geo in ['Tehran','Region12']:
        use = active_by_geo[geo]
        d = (scores[(scores.geo == geo) & (scores.scenario.isin(use))].pivot(index='scenario', columns='pillar', values='score').reindex(index=use)[PILLARS])
        eq = d.mean(axis=1)
        for scenario, val in eq.items():
            equal_rows.append({'geo': geo, 'scenario': scenario, 'equal_weight_score': float(val)})
        rng = np.random.default_rng(42)
        W = rng.dirichlet(np.ones(4), size=200_000)
        vals = d.values
        agg = W @ vals.T
        order = np.argsort(-agg, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(W.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, len(use)+1)
        for i, s in enumerate(use):
            r = ranks[:, i]
            accept_rows.append({'geo': geo, 'scenario': s, 'win_probability': float(np.mean(r==1)), 'top2_probability': float(np.mean(r<=min(2,len(use)))), 'top3_probability': float(np.mean(r<=min(3,len(use)))), 'mean_rank': float(np.mean(r)), 'median_rank': float(np.median(r))})
        for focus_idx, focus in enumerate(PILLARS):
            for alpha in alphas:
                other=(1-alpha)/(len(PILLARS)-1)
                w=np.full(len(PILLARS),other); w[focus_idx]=alpha
                agg=vals@w
                order=np.argsort(-agg)
                winner_rows.append({'geo':geo,'focus':focus,'alpha':alpha,'winner':use[order[0]]})
    equal_weight=pd.DataFrame(equal_rows)
    accept=pd.DataFrame(accept_rows)
    winners=pd.DataFrame(winner_rows)
    equal_weight.to_csv(root / 'outputs' / 'Figure_11_equal_weight_scores.csv', index=False)
    accept.to_csv(root / 'outputs' / 'Figure_11_rank_acceptability.csv', index=False)
    winners.to_csv(root / 'outputs' / 'Figure_11_priority_sweep_winners.csv', index=False)

    summary_lines = ['# Weight-robustness summary', '', 'The decision-synthesis analysis evaluates the full policy portfolio (SC1-SC11) under a common decision architecture.']
    for geo in ['Tehran','Region12']:
        d = accept[accept['geo'] == geo].sort_values(['win_probability', 'top2_probability', 'mean_rank'], ascending=[False, False, True])
        leaders = ', '.join([f"{r.scenario} ({100*r.win_probability:.2f}%)" for _, r in d.iterrows()])
        summary_lines.append(f'- {geo}: full-portfolio decision leadership is concentrated in {leaders}.')
    (root / 'outputs' / 'weight_robustness_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

    scenario_colors = scenario_palette(candidate_codes)
    fig = plt.figure(figsize=(11.4, 9.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.05, 1.15], hspace=0.34, wspace=0.24)
    ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0]); ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0]); ax21 = fig.add_subplot(gs[2, 1])

    draw_equal_weight(ax00, equal_weight[equal_weight.geo == 'Tehran'], '(a) Equal-weight score', 'Tehran', active_by_geo['Tehran'], scenario_colors, True)
    draw_equal_weight(ax01, equal_weight[equal_weight.geo == 'Region12'], '(b) Equal-weight score', 'Region 12', active_by_geo['Region12'], scenario_colors, True)
    draw_accept(ax10, accept[accept.geo == 'Tehran'], '(c) Rank acceptability', 'Tehran', active_by_geo['Tehran'], False)
    draw_accept(ax11, accept[accept.geo == 'Region12'], '(d) Rank acceptability', 'Region 12', active_by_geo['Region12'], False)
    draw_winner_map(ax20, winners, alphas, 'Tehran', '(e) Winner under priority sweep', 'Tehran', active_by_geo['Tehran'], scenario_colors, False)
    draw_winner_map(ax21, winners, alphas, 'Region12', '(f) Winner under priority sweep', 'Region 12', active_by_geo['Region12'], scenario_colors, False)
    # Keep the shared scenario-color mapping and legend order.
    legend_codes = candidate_codes[:]
    handles = [mpl.patches.Patch(color=scenario_colors[s], label=s) for s in legend_codes]
    widen_column_gap(fig, [ax00, ax10, ax20], [ax01, ax11, ax21], delta_mm=4.0)
    bottom_reserved = reserve_bottom_for_accessory(fig, STYLE['axis'], STYLE['legend'], gap_mult=4.0, accessory_type='legend')
    fig.tight_layout(rect=[0, bottom_reserved, 1, 1])
    place_bottom_legend(fig, [ax00,ax01,ax10,ax11,ax20,ax21], handles, legend_codes, xlabel_fontsize=STYLE['axis'], legend_fontsize=STYLE['legend'], gap_mult=4.0, ncol=len(legend_codes), handlelength=1.0, columnspacing=0.7)
    save_figure_svg(fig, root / 'figures' / 'Figure 11.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
