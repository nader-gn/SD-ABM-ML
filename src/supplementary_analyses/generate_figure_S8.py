from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'tehran-sd-abm-ml-reproducibility'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'supplementary_analyses'
OUT = ROOT / 'figures_supplementary'
DATA_OUT = ROOT / 'figure_inputs_supplementary'
OUT.mkdir(exist_ok=True)
DATA_OUT.mkdir(exist_ok=True)

SCENS = [f'SC{i}' for i in range(1, 12)]
PAL = {
    'SC0':'black','SC1':'#1f9eb7','SC2':'#E15759','SC3':'#4E79A7','SC4':'#76B7B2',
    'SC5':'#59A14F','SC6':'#EDC948','SC7':'#B07AA1','SC8':'#9C755F','SC9':'#BAB0AC',
    'SC10':'#8E63CE','SC11':'#FF9DA7'
}


G = pd.read_csv(EXT / 'uncertainty_relative_effects_SC0_SC11.csv')
central = pd.read_csv(ROOT / 'outputs' / 'kpi_timeseries_selected_long_2024_2030.csv')

# Financial-balance central series is read directly from the central simulations.
fin = []
for sc in [f'SC{i}' for i in range(12)]:
    d = pd.read_csv(ROOT / 'outputs' / f'simulation_data_{sc}.csv')
    for _, r in d[d.YEAR_GRG.between(2024, 2030)].iterrows():
        fin.append({
            'scenario': sc,
            'year': int(round(r.YEAR_GRG)),
            'value': float(r.transport_financial_balance_IRR),
        })
fin = pd.DataFrame(fin)

plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'font.size':6.5,
    'axes.titlesize':8.8,
    'axes.labelsize':7.9,
    'xtick.labelsize':6.9,
    'ytick.labelsize':6.9,
    'legend.fontsize':6.8,
    'axes.linewidth':.75,
    'axes.spines.top':False,
    'axes.spines.right':False,
    'ps.fonttype':42,
    'svg.fonttype':'none',
})

central['Geo'] = central.Geo.replace({'Region12':'Region 12'})
base = central[central.Scenario=='SC0'][['year','Geo','kpi','value']].rename(columns={'value':'base'})
C = central.merge(base, on=['year','Geo','kpi'], how='left')
C['delta'] = np.where(np.abs(C.base)>1e-12, 100*(C.value-C.base)/np.abs(C.base), C.value-C.base)
fb = fin[fin.scenario=='SC0'][['year','value']].rename(columns={'value':'base'})
F = fin.merge(fb, on='year')
F['delta'] = np.where(np.abs(F.base)>1e-12, 100*(F.value-F.base)/np.abs(F.base), F.value-F.base)


def c2030(sc: str, geo: str, kpi: str) -> float:
    if kpi.startswith('STOCK::'):
        q = F[(F.scenario==sc) & (F.year==2030)]
    else:
        q = C[(C.Scenario==sc) & (C.Geo==geo) & (C.kpi==kpi) & (C.year==2030)]
    return float(q.delta.iloc[0]) if len(q) else np.nan


def panel(ax, kpi: str, title: str) -> None:
    ax.set_title(title, loc='left', pad=2)
    y = np.arange(len(SCENS))[::-1]
    ax.axvline(0, color='black', lw=.75, zorder=0)
    for yi, sc in zip(y, SCENS):
        for geo, off, marker in [('Tehran',+.13,'o'), ('Region 12',-.13,'^')]:
            v = G[(G.scenario==sc) & (G.geo==geo) & (G.metric==kpi)].delta_2030_pct.dropna().values
            if not len(v):
                continue
            p05,p25,med,p75,p95 = np.quantile(v,[.05,.25,.5,.75,.95])
            col = PAL[sc]
            ax.plot([p05,p95],[yi+off,yi+off],color=col,lw=.80,alpha=.92,zorder=2)
            ax.plot([p25,p75],[yi+off,yi+off],color=col,lw=2.7,solid_capstyle='round',zorder=3)
            ax.scatter(med,yi+off,s=12,marker=marker,color=col,edgecolor='white',lw=.25,zorder=4)
            cen = c2030(sc,geo,kpi)
            ax.scatter(cen,yi+off,s=16,marker=marker,facecolor='white',edgecolor=col,lw=.8,zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(SCENS)
    ax.set_xlabel('2030 effect vs matched SC0 (%)')
    ax.grid(True,axis='x',alpha=.18,lw=.25)


fig, axes = plt.subplots(3,3,figsize=(11.69,8.27))
axes = np.asarray(axes)
panel(axes[0,0],'Modal share: public transport','(a) Transportation: public-transport modal share')
panel(axes[0,1],'Time loss (car)','(b) Transportation: car time loss')
panel(axes[0,2],'PCE-weighted VKT','(c) Transportation: PCE-weighted VKT')
panel(axes[1,0],'CO₂ emissions','(d) Environmental: CO₂ emissions')
panel(axes[1,1],'NOₓ emissions','(e) Environmental: NOₓ emissions')
panel(axes[1,2],'PM₂.₅ emissions','(f) Environmental: PM₂.₅ emissions')
panel(axes[2,0],'Health indicator','(g) Social: health indicator')

# Endogenous financial-balance stock.
ax = axes[2,1]
ax.set_title('(h) Endogenous SD stock: transport financial balance',loc='left',pad=2)
y = np.arange(len(SCENS))[::-1]
ax.axvline(0,color='black',lw=.75)
for yi,sc in zip(y,SCENS):
    v = G[(G.scenario==sc)&(G.geo=='Tehran')&(G.metric=='STOCK::transport_financial_balance_IRR')].delta_2030_pct.dropna().values
    if len(v):
        p05,p25,med,p75,p95=np.quantile(v,[.05,.25,.5,.75,.95])
        col=PAL[sc]
        ax.plot([p05,p95],[yi,yi],color=col,lw=.8)
        ax.plot([p25,p75],[yi,yi],color=col,lw=2.7,solid_capstyle='round')
        ax.scatter(med,yi,s=12,color=col,edgecolor='white',lw=.25)
        ax.scatter(c2030(sc,'Tehran','STOCK::transport_financial_balance_IRR'),yi,s=16,facecolor='white',edgecolor=col,lw=.8)
ax.set_yticks(y)
ax.set_yticklabels(SCENS)
ax.set_xlabel('2030 effect vs matched SC0 (%)')
ax.grid(True,axis='x',alpha=.18,lw=.25)

# Which endogenous stocks inherit meaningful serving uncertainty?
ax = axes[2,2]
ax.set_title('(i) Which SD stocks inherit ML uncertainty?',loc='left',pad=2)
stocks = [
    ('STOCK::transport_financial_balance_IRR','Financial balance'),
    ('STOCK::population_city','Population'),
    ('STOCK::private_cars_total','Private cars'),
    ('STOCK::motorcycles_total','Motorcycles'),
    ('STOCK::buses_total','Bus fleet'),
    ('STOCK::len_met','Metro length'),
]
rows=[]
for metric,label in stocks:
    spans=[]
    for sc in SCENS:
        v=G[(G.scenario==sc)&(G.geo=='Tehran')&(G.metric==metric)].delta_2030_pct.dropna().values
        if len(v):
            spans.append(np.quantile(v,.95)-np.quantile(v,.05))
    rows.append((label,np.median(spans),np.max(spans)))
ST=pd.DataFrame(rows,columns=['stock','median_span','max_span'])
y=np.arange(len(ST))[::-1]
for yi,r in zip(y,ST.itertuples()):
    ax.plot([0,r.max_span],[yi,yi],color='#8A8A8A',lw=.85)
    ax.scatter(r.median_span,yi+.07,s=18,color='#1f77b4')
    ax.scatter(r.max_span,yi-.07,s=18,marker='|',color='#E15759')
ax.set_yticks(y)
ax.set_yticklabels(ST.stock)
ax.set_xscale('symlog',linthresh=.02,linscale=1)
ax.set_xlim(0,13)
ax.set_xticks([0,.01,.1,1,10])
ax.set_xticklabels(['0','0.01','0.1','1','10'])
ax.set_xlabel('2030 90% envelope span (%-effect units; symlog)')
ax.grid(True,axis='x',alpha=.18,lw=.25)
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='#1f77b4',lw=0,markersize=4,label='median across SC1–SC11'),
    Line2D([0],[0],marker='|',color='#E15759',lw=0,markersize=5,label='maximum')
],frameon=False,loc='lower right',fontsize=5.45)

fig.legend(handles=[
    Line2D([0],[0],marker='o',color='#555',lw=0,markerfacecolor='#555',markersize=3.8,label='Tehran median'),
    Line2D([0],[0],marker='^',color='#555',lw=0,markerfacecolor='#555',markersize=3.8,label='Region 12 median'),
    Line2D([0],[0],color='#555',lw=2.7,label='IQR; thin line = 5–95%'),
    Line2D([0],[0],marker='o',color='#555',lw=0,markerfacecolor='white',markersize=4,label='Central result')
],ncol=4,loc='lower center',bbox_to_anchor=(.5,.010),frameon=False,columnspacing=1.2,handletextpad=.45)

# Export exact panel data summarized in panels (a)-(h), plus stock-span data in (i).
quant_rows=[]
figure_metrics=[
    'Modal share: public transport','Time loss (car)','PCE-weighted VKT',
    'CO₂ emissions','NOₓ emissions','PM₂.₅ emissions','Health indicator'
]
for kpi in figure_metrics:
    for sc in SCENS:
        for geo in ['Tehran','Region 12']:
            v=G[(G.scenario==sc)&(G.geo==geo)&(G.metric==kpi)].delta_2030_pct.dropna().values
            if not len(v):
                continue
            p05,p25,med,p75,p95=np.quantile(v,[.05,.25,.5,.75,.95])
            quant_rows.append({
                'panel_metric':kpi,'scenario':sc,'geo':geo,
                'p05':p05,'p25':p25,'median':med,'p75':p75,'p95':p95,
                'frozen_central_2030':c2030(sc,geo,kpi)
            })
for sc in SCENS:
    v=G[(G.scenario==sc)&(G.geo=='Tehran')&(G.metric=='STOCK::transport_financial_balance_IRR')].delta_2030_pct.dropna().values
    if len(v):
        p05,p25,med,p75,p95=np.quantile(v,[.05,.25,.5,.75,.95])
        quant_rows.append({
            'panel_metric':'Transport financial balance stock','scenario':sc,'geo':'Tehran',
            'p05':p05,'p25':p25,'median':med,'p75':p75,'p95':p95,
            'frozen_central_2030':c2030(sc,'Tehran','STOCK::transport_financial_balance_IRR')
        })
pd.DataFrame(quant_rows).to_csv(DATA_OUT/'Figure_S8_effect_quantiles.csv',index=False)
ST.to_csv(DATA_OUT/'Figure_S8_stock_uncertainty_spans.csv',index=False)

fig.tight_layout(rect=[.035,.060,.995,.992],h_pad=1.10,w_pad=.85)
out_svg = OUT/'Figure S8.svg'
fig.savefig(out_svg, format='svg', bbox_inches='tight', metadata={'Date': None})
_txt = out_svg.read_text(encoding='utf-8')
out_svg.write_text(re.sub(r'<metadata>.*?</metadata>', '', _txt, flags=re.S), encoding='utf-8')
plt.close(fig)
print('Wrote Figure S8 and source-data tables')
