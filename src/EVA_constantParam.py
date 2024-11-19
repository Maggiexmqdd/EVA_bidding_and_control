
import numpy as np
import pandas as pd


# Game parameters 
nx = 1      # state dimension
nu = 1      # input dimension
N = 80       # No. of agents
T = 24      
H =6       # prediction horizon

#  %%==== electricity_market_data ====
from data_process import *
df_energy_data = pd.read_csv('../input/rt_hrl_lmps.csv', sep=',')
df_fre_data = pd.read_csv('../input/ancillary_services.csv', sep=',')
pr_e_rt=transfer(df_energy_data,target_cols=['value'] )/1000
pr_fre=transfer(df_fre_data,target_cols=['value'])/1000

# frequency price and mileage:
df = pd.read_csv('../input/reserve_market_results.csv')
df_m = pd.read_csv('../input/reg_market_results.csv')
c_rcap = transfer(df, target_cols=['reg_ccp']).flatten()
c_rper = transfer(df, target_cols=['reg_pcp']).flatten()
mileage = transfer(df_m, target_cols=['regd_hourly']).flatten()
# Calculate final frequency price
c_rper = np.array(c_rper) * np.array(mileage)
pr_fre = (c_rcap + c_rper) / 1000  # Normalize to $/kWh

charging_fee=compute_charging_fee()
price_plot(pr_e_rt , pr_fre , charging_fee,1)