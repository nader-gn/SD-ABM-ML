from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];EXT=ROOT/'supplementary_analyses'
d0=pd.read_csv(ROOT/'outputs'/'simulation_data_SC0.csv');d5=pd.read_csv(ROOT/'outputs'/'simulation_data_SC5.csv')
m0=float(d0[d0.YEAR_GRG.between(2024,2030)]['energy_diesel_MJ_year'].mean());m5=float(d5[d5.YEAR_GRG.between(2024,2030)]['energy_diesel_MJ_year'].mean())
rows=[]
for share in [0.0,0.1,0.2,0.3,0.4]:
    a=(1-share)*m0;b=(1-share)*m5
    rows.append({'assumed_non_diesel_or_zero_tailpipe_share':share,'SC0_diesel_MJ_year':a,'SC5_diesel_MJ_year':b,'SC5_vs_SC0_pct':100*(b/a-1)})
pd.DataFrame(rows).to_csv(EXT/'SC5_diesel_accounting_sensitivity.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
