# %% ===== Imports =====
import EVA_constantParam as cpm
import numpy as np

# matrix
# Create dynamics in compact form
def power_dispatch_plp(t_cur):
    """
    Prepare and save parametric linear programming (PLP) data for power dispatch.

    Parameters:
    - self: Object containing system parameters and data.
    - t_cur: Current time step (integer).

    Outputs:
    - Saves PLP parameters as a .mat file for later use.
    """

    # ===== Parameters and Initialization =====
    N = cpm.N                          # Number of agents (EVs)
    eta_c, eta_d= cpm.eta_c, cpm.eta_d   # Charging and discharging efficiencies
    # Load pre-saved trajectories
    bidding_data = np.load("../output/bidding_trajectories.npz")
    P_bid = bidding_data['P_bid_trajectory']
    R_bid = bidding_data['R_bid_trajectory']

    data = np.load("../output/EV_power_trajectories.npz")
    p0 = data['p0_trajectory']

    flex_data = np.load("../output/output_trajectories.npz")
    la=flex_data['la_trajectory']
    deg =flex_data['deg_trajectory']
    du= flex_data['du_trajectory']
    dd= flex_data['dd_trajectory']

    e = np.ones((1, N))                 # Row vector of ones (length N)
    I = np.eye(N)                       # Identity matrix of size NxN
    aaa = np.ones((1, 1))               # Scalar (used for Feq construction)

    # ===== Equality Constraints =====
    # Aeq: Matrix for equality constraints
    Aeq = np.block([
        [e, -e, e * 0],                 # Sum of charging and discharging equals demand
        [I, -I, I]                     # Individual energy balance constraints
    ])
    
    # Feq: Coefficients for parametric equality constraints
    Feq = np.hstack((aaa * R_bid[t_cur], e * 0)).T

    # beq: Right-hand side of equality constraints
    beq = np.hstack((
        aaa * P_bid[t_cur],        # Total power demand at time t_cur
        p0[:, t_cur].reshape(-1, 1).T  # Initial state for each agent
    )).reshape(-1, 1)

    # ===== Inequality Constraints =====
    # A: Matrix for inequality constraints
    A = np.block([
        [-I, I * 0, I * 0],             # Charging power bounds
        [I * 0, -I, I * 0],            # Discharging power bounds
        [I * 0, I * 0, I],             # Upward flexibility bounds
        [I * 0, I * 0, -I]             # Downward flexibility bounds
    ])
    
    # b: Right-hand side of inequality constraints
    b = np.hstack((
        e * 0,                          # Charging lower bound
        e * 0,                          # Discharging lower bound
        du[:, t_cur].reshape(-1, 1).T,  # Upward reserve constraints
        dd[:, t_cur].reshape(-1, 1).T   # Downward reserve constraints
    )).reshape(-1, 1)

    # ===== Terminal Constraints =====
    At = np.array([[1], [-1]])          # Terminal constraint coefficients
    bt = np.array([[1], [1]])           # Terminal bounds

    # ===== Cost Function =====
    # Linear cost coefficients
    c = np.hstack((
        e * 0,                          # No cost for charging
        la[:, t_cur].reshape(-1, 1).T / eta_d,  # Cost for discharging
        la[:, t_cur].reshape(-1, 1).T           # Cost for flexibility
    )).reshape(-1, 1)

    # ===== Save PLP Data =====
    plp = {
        'A': A,
        'b': b,
        'Aeq': Aeq,
        'beq': beq,
        'Feq': Feq,
        'At': At,
        'bt': bt,
        'c': c,
    }
    # savemat(f'./crs/plp_{t_cur}.mat', plp)
    # critical_region_affine(A, b, Aeq, beq, c, F, param_range)
    return plp


power_dispatch_plp(10)
    