import numpy as np
import math
from scipy.stats import truncnorm

def generate_ev_data(N, k_value, charging_fee, k_v=120, mode=1):
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
    t_arr = np.zeros(N, dtype=int)  # arrival time
    t_dep = np.zeros(N, dtype=int)  # departure time
    soc_arr = np.zeros(N)  # arrival SOC
    soc_dep = np.zeros(N)  # departure SOC
    B = np.random.uniform(45, 50, size=N).round(2)  # battery capacity (kWh)
    eff_ch = np.full(N, 0.9)  # charging efficiency
    eff_dis = np.full(N, 0.93)  # discharging efficiency
    E_req = np.zeros(N)  # energy required (kWh)
    k = np.zeros((N, 24))  # preference matrix
    e = np.ones(N) * 10

    # Generate k values based on charging fee
    generate_preference_matrix(k, N, k_value, charging_fee)

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

def generate_preference_matrix(k, N, k_value, charging_fee):
    """
    Populate the preference matrix k based on k_value and charging_fee.
    """
    if k_value == 1:
        k[:N // 4, :] = 0.3 * 30 / charging_fee
        k[N // 4:2 * N // 4, :] = 20 / charging_fee
        k[2 * N // 4:3 * N // 4, :] = 1 * 30 / charging_fee
        k[3 * N // 4:, :] = 1.5 * 30 / charging_fee

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
