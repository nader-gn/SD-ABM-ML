from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];EXT=ROOT/'supplementary_analyses'
def components(df):
    d=df[df.YEAR_GRG.between(2024,2030)].copy()
    car_tail=d['vkm_car_r12']*d.get('car_energy_closure_factor',1.0)*(1-d['share_car_EV'])
    return {
      'car':d['EF_PM25_car_g_km']*car_tail/1e6,
      'bus':d['EF_PM25_bus_g_km']*d['vkm_bus_r12']*(1-d['share_bus_EV'])/1e6,
      'taxi':d['EF_PM25_taxi_g_km']*d['vkm_taxi_r12']*(1-d['share_taxi_EV'])/1e6,
      'motorcycle':d['EF_PM25_motorcycle_g_km']*d['vkm_motorcycle_r12']*(1-d['share_motorcycle_EV'])/1e6,
      'grid':d['EF_grid_PM25_g_kWh']*d['electricity_kWh_year_r12']/1e6,
    }
c0=components(pd.read_csv(ROOT/'outputs'/'simulation_data_SC0.csv'));c6=components(pd.read_csv(ROOT/'outputs'/'simulation_data_SC6.csv'))
changes={k:float(c6[k].mean()-c0[k].mean()) for k in c0};tot=sum(changes.values())
rows=[]
for k in ['car','bus','taxi','motorcycle','grid']:
    a=float(c0[k].mean());b=float(c6[k].mean());delta=b-a
    rows.append({'component':k,'SC0_mean_t_year':a,'SC6_mean_t_year':b,'absolute_change_t_year':delta,'relative_change_pct':100*delta/a,'share_of_absolute_reduction_pct':100*delta/tot})
a=sum(float(c0[k].mean()) for k in c0);b=sum(float(c6[k].mean()) for k in c6);delta=b-a
rows.append({'component':'total','SC0_mean_t_year':a,'SC6_mean_t_year':b,'absolute_change_t_year':delta,'relative_change_pct':100*delta/a,'share_of_absolute_reduction_pct':100.0})
pd.DataFrame(rows).to_csv(EXT/'SC6_R12_PM25_decomposition.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
