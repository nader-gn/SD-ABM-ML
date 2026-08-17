from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def components(df: pd.DataFrame):
    d=df[(df.YEAR_GRG>=2024)&(df.YEAR_GRG<=2030)].copy()
    car_tail=d['vkm_car_r12']*d.get('car_energy_closure_factor',1.0)*(1-d['share_car_EV'])
    comp={
      'car':d['EF_PM25_car_g_km']*car_tail/1e6,
      'bus':d['EF_PM25_bus_g_km']*d['vkm_bus_r12']*(1-d['share_bus_EV'])/1e6,
      'taxi':d['EF_PM25_taxi_g_km']*d['vkm_taxi_r12']*(1-d['share_taxi_EV'])/1e6,
      'motorcycle':d['EF_PM25_motorcycle_g_km']*d['vkm_motorcycle_r12']*(1-d['share_motorcycle_EV'])/1e6,
      'grid':d['EF_grid_PM25_g_kWh']*d['electricity_kWh_year_r12']/1e6,
    }
    return d,comp

def total_at(comp,m):
    return m*(comp['car']+comp['bus']+comp['taxi']+comp['motorcycle'])+comp['grid']

def main():
    d0,c0=components(pd.read_csv(ROOT/'outputs/simulation_data_SC0.csv'))
    d6,c6=components(pd.read_csv(ROOT/'outputs/simulation_data_SC6.csv'))
    rows=[]
    for m in [0.5,0.75,1.0,1.25,1.5]:
      t0=total_at(c0,m);t6=total_at(c6,m)
      rows.append({
       'district_mobile_EF_multiplier':m,
       'SC0_mean_PM25_t_year':float(t0.mean()),
       'SC6_mean_PM25_t_year':float(t6.mean()),
       'delta_mean_pct':float(100*(t6.mean()-t0.mean())/t0.mean()),
       'delta_2030_pct':float(100*(t6.iloc[-1]-t0.iloc[-1])/t0.iloc[-1]),
      })
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'supplementary_analyses/sc6_r12_emission_factor_scaling_sensitivity.csv',index=False)
    print(out.to_string(index=False))
if __name__=='__main__':main()
