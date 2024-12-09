import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


def transfer(data, datetime_col='datetime_beginning_ept', group_col='hour', target_cols=None):
    """
    Generalized data transformation function.
    Converts a datetime column to datetime type, extracts the hour,
    and computes hourly averages for specified target columns.
    The resulting data is rearranged by swapping the first 12 hours with the last 12 hours.
    
    Args:
        data (pd.DataFrame): Input data.
        datetime_col (str): Name of the datetime column to process.
        group_col (str): Name of the column to group by (e.g., 'hour').
        target_cols (list): List of columns to compute averages for.
                            If None, computes averages for all columns.

    Returns:
        np.ndarray: Rearranged array of hourly averaged values.
    """
    # Ensure the datetime column is of datetime type
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Extract hour from the datetime column
    data[group_col] = data[datetime_col].dt.hour
    
    # Compute hourly averages for specified columns (or all columns if not specified)
    if target_cols is not None:
        hourly_avg = data.groupby(group_col)[target_cols].mean()
    else:
        hourly_avg = data.groupby(group_col).mean()
    
    # Convert to array and rearrange (swap first 12 hours with last 12 hours)
    arr = hourly_avg.values
    arr[:12], arr[12:] = arr[12:], arr[:12].copy()

    return arr

def compute_charging_fee():
    """
    Calculate the hourly charging fee rates based on predefined off-peak, super off-peak, and peak prices. 
    The charging fee array is rearranged by swapping the first 12 hours with the last 12 hours.

    Returns:
        list: A list of 24 hourly charging fee rates, rearranged as per the specified swapping logic.
    """
    # Define constants
    aaaaa = 1 / 2
    super_offpeak = 0.16505 * aaaaa
    offpeak = 0.19171 * aaaaa
    peak = 0.38372 * aaaaa

    # Define charging fee schedule (24 hours)
    charging_fee = (
        [offpeak] * 9
        + [super_offpeak] * 5
        + [offpeak] * 2
        + [peak] * 5
        + [offpeak] * 3
    )

    # Rearrange the fee schedule by swapping the first 12 hours with the last 12 hours
    temp = charging_fee[:12].copy()  # Copy the first 12 hours
    charging_fee[:12] = charging_fee[12:]  # Move the last 12 hours to the first half
    charging_fee[12:] = temp  # Place the original first 12 hours in the second half

    return charging_fee

def price_plot(pr_e_rt, pr_fre, charging_fee, mode=0):
    """
    Plots energy price, reserve price, and charging fee with dual y-axes.

    Args:
        pr_e_rt (array-like): Energy price data over time.
        pr_fre (array-like): Reserve price data over time.
        charging_fee (array-like): Charging fee data over time.
        mode (int, optional): If set to 1, saves the plot as a PDF file. Defaults to 0.

    Returns:
        None
    """

    # Initialize the figure and axes
    fig, ax1 = plt.subplots(figsize=(5, 3))
    x = range(len(pr_e_rt))  # Time periods

    # Plot energy and reserve prices on the primary y-axis
    ax1.plot(x, pr_e_rt, alpha=0.6, color='#040676', marker='o', label='Energy Price')
    ax1.plot(x, pr_fre, alpha=0.6, color='#F1B656', marker='v', label='Reserve Price')
    ax1.set_xlim(-0.5, 24.5)
    ax1.set_xlabel('Time Period (1h)')
    ax1.set_ylabel('Market Price ($/kWh)')

    # Adjust x-axis ticks and labels with larger intervals (e.g., every 4 hours)
    major_ticks = range(0, 25, 4)  # Major ticks at every 4 hours
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels([str((i + 12) % 24) for i in major_ticks])  # Circular time labels

    # Create a secondary y-axis for charging fees
    ax2 = ax1.twinx()
    ax2.plot(x, charging_fee, alpha=0.6, color='#A4514F', marker='+', label='Charging Fee')
    ax2.set_ylabel('Charging Fee ($/kWh)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),  # Adjusted position
        ncol=3
    )

    # Save the plot if mode is set to 1
    if mode == 1:
        plt.tight_layout()
        plt.savefig(
            "../output/market_price.pdf",
            bbox_inches='tight',
        )

    # Always show the plot for visualization
    plt.show()

# %%
import numpy as np
import math
from scipy.stats import truncnorm

def generate_ev_data(N, k_mode, k_v, charging_fee, mode=1):
    """
    Generate synthetic EV data based on the input parameters.

    Args:
        N (int): Number of EVs.
        k_value (int): Parameter controlling k distribution.
        charging_fee (array-like): Charging fee data.
        k_v (int, optional): Default k value. Defaults to 120.
        mode (int, optional): Mode of operation (0: night, 1: day, 2: mix). Defaults to 1.

    Returns:
        dict: Generated EV parameters.
    """
    ev_id = [i for i in range(1, N + 1)]
    charging_fee = np.array(charging_fee)

    # Initialize parameters
    p_max = np.full(N, 10.0)  # maximal charging rate (kW)
    t_arr,t_dep = np.zeros(N, dtype=int), np.zeros(N, dtype=int)  # arrival and departure time 
    soc_arr,soc_dep = np.zeros(N),np.zeros(N)  # arrival SOC/departure SOC
    B = np.random.uniform(45, 50, size=N).round(2)  # battery capacity (kWh)
    eff_ch,eff_dis = np.full(N, 0.9),np.full(N, 0.93)  # charging/discharging efficiency 
    E_req = np.zeros(N)  # energy required (kWh)
    e = np.ones(N) * 10

    # Generate k values based on charging fee
    k=generate_preference_matrix(N,k_mode,k_v,charging_fee)

    # Generate EV arrival, departure times, and SOC
    generate_ev_times(t_arr, t_dep, soc_arr, soc_dep, N, mode)

    # Calculate energy requirements
    calculate_energy_requirements(E_req, soc_arr, soc_dep, B)

    # Ensure feasibility of initial SOC and times
    ensure_feasibility(t_arr, t_dep, soc_arr, soc_dep, E_req, B, p_max, eff_ch)

    return {
        "N": N,
        "t_arr": t_arr,
        "t_dep": t_dep,
        "soc_arr": soc_arr,
        "soc_dep": soc_dep,
        "Battery_capacity": B,
        "required_power": E_req,
        "maximum_charging": p_max,
        "charging_eff": eff_ch,
        "discharging_eff": eff_dis,
        "k": k,
        "e": e
    }

def generate_preference_matrix(N, k_different,k_v,charging_fee):
    """
    Populate the preference matrix k based on k_value and charging_fee.
    """
    k = np.zeros((N, 24))  # preference matrix
    max_=30
    if k_different == 1:
        k[:N // 4, :] = 0.25 * max_ / charging_fee
        k[N // 4:2 * N // 4, :] = 0.5 * max_ / charging_fee
        k[2 * N // 4:3 * N // 4, :] = 0.75 * max_ / charging_fee
        k[3 * N // 4:, :] = 1 * max_ / charging_fee
    elif k_different == 0:
        k[:,:]=k_v
    return k

def generate_ev_times(t_arr, t_dep, soc_arr, soc_dep, N, mode):
    """
    Generate arrival and departure times for EVs.
    """
    for n in range(N):
        dn, up, mu, sigma = 1, 24, 7, 2.5
        dn2, up2, mu2, sigma2 = 1, 24, 20, 2

        t_arr[n] = int(round(truncnorm.rvs((dn - mu) / sigma, (up - mu) / sigma, loc=mu, scale=sigma)) - 1)
        t_dep[n] = int(round(truncnorm.rvs((dn2 - mu2) / sigma2, (up2 - mu2) / sigma2, loc=mu2, scale=sigma2)) - 1)

        if mode == 0:  # Night
            t_arr[n] = round(truncnorm.rvs((1 - 6) / 2, (12 - 6) / 2, loc=6, scale=2)) - 1
            t_dep[n] = round(truncnorm.rvs((12 - 22) / 2, (24 - 22) / 2, loc=22, scale=2)) - 1
        elif mode == 2:  # Mix
            if n >= N // 2:
                t_arr[n] = round(truncnorm.rvs((1 - 6) / 2, (12 - 6) / 2, loc=6, scale=2)) - 1
                t_dep[n] = round(truncnorm.rvs((12 - 20) / 2, (24 - 20) / 2, loc=20, scale=2)) - 1

        soc_arr[n] = round(np.random.uniform(0.2, 0.4), 2)
        soc_dep[n] = round(np.random.uniform(0.7, 0.9), 2)

def calculate_energy_requirements(E_req, soc_arr, soc_dep, B):
    """
    Calculate energy requirements for all EVs.
    """
    for n in range(len(E_req)):
        E_req[n] = B[n] * (soc_dep[n] - soc_arr[n])

def ensure_feasibility(t_arr, t_dep, soc_arr, soc_dep, E_req, B, p_max, eff_ch):
    """
    Ensure that the EV charging schedules are feasible.
    """
    for n in range(len(E_req)):
        if (t_dep[n] - t_arr[n]) < (E_req[n] / (p_max[n] * eff_ch[n])):
            print(f'{n}th solution is not feasible: SOC dep {soc_dep[n]}, E_req {E_req[n]}')
            soc_dep[n] = p_max[n] * (t_dep[n] - t_arr[n]) * eff_ch[n] / B[n] + soc_arr[n]
            soc_dep[n] = math.floor(soc_dep[n] * 100) / 100
            E_req[n] = B[n] * (soc_dep[n] - soc_arr[n])

# Example usage
if __name__ == "__main__":
    np.random.seed(66)
    N = 20
    charging_fee = [0.15] * 24  # Example charging fee
    ev_params = generate_ev_data(N, k_value=1, charging_fee=charging_fee, mode=1)
    # print(ev_params)