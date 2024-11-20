"""
Electric Vehicle (EV) Aggregator Simulation Script
---------------------------------------------------
This script processes electricity market data, computes relevant price signals, 
and generates EV data for simulations. Key features include:
- Data processing for real-time electricity and frequency regulation markets.
- Calculation of charging fees and plotting of price signals.
- Generation of EV parameters with random preferences for flexibility simulations.

Parameters:
    N (int): Number of agents (EVs).
    T (int): Time steps (hours in a day).
    H (int): Prediction horizon (in hours).
"""
import numpy as np
import pandas as pd
from data_process import *
# Game parameters 
N = 80       # No. of agents
T = 24      
H = 6       # prediction horizon
eta_c, eta_d = 0.9, 0.93  
#  %%==== electricity_market_data ====
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
# %% EV data
np.random.seed(66)
ev_params = generate_ev_data(N, k_value=1,charging_fee=charging_fee, k_v=1)  #1是有不同preference
# np.savez("../output/ev_params.npz", **ev_params)