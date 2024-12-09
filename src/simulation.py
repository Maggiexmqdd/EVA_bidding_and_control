# %% ===== Imports =====
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import EVA_constantParam as cpm
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

# %%  ===== Proposed Stochastic MPC Bidding Methods =====
from model import EVA_ms
from data_process import *

def run_simulation(kd,k,cpm,flex):
    """
    Run MPC simulation for a given k value and calculate costs.

    Parameters:
        k (float): Parameter for flexibility adjustment.
        cpm (object): Contains system parameters such as N (number of EVs) and T (time steps).
        bound_args (dict): Boundary arguments including initial conditions and constraints.
        bb (module): Module for building the optimization model.
    
    Returns:
        total_cost (float): Total cost for the given k.
        results (dict): Trajectories and results for the simulation.
    """
    print(f"Running simulation for k = {k}")
    
    # Update EV parameters with the given k value
    ev_params = generate_ev_data(cpm.N,k_mode=kd,k_v=k,
                                charging_fee=cpm.charging_fee)
    # Instantiate EVA model with generated EV parameters
    bb = EVA_ms(ev_params)
    bound_args = bb.aggregation_v2g()

    # Initialize variables
    E_ini = [array[0] for array in bound_args['E_min']]  # Initial energy levels
    P_bid_trajectory = []  # To store power bid trajectory
    R_bid_trajectory = []  # To store regulation bid trajectory
    # flexibility
    la_trajectory = np.zeros((cpm.N, cpm.T))
    deg_trajectory = np.zeros((cpm.N, cpm.T))  # Shape: N x T
    du_trajectory = np.zeros((cpm.N, cpm.T))
    dd_trajectory = np.zeros((cpm.N, cpm.T))
    p0_trajectory, e_trajectory=np.zeros((cpm.N, cpm.T)),np.zeros((cpm.N, cpm.T))
    dispatch_cost=0
    # closed-loop Receding horizon optimization
    for t in range(cpm.T):
        # Build the optimization model for the current time step
        flag, P_bid, R_bid, E_ini,la,deg, du, dd,pp0,dispatch_cost = bb.build_model_reg(cpm, bound_args, t, E_ini,flex=flex)
        
        # Check if the optimization failed
        if flag == -1:
            print(f"Warning: MPC optimization failed at step {t}")
            break
        
        # Append results to trajectories
        P_bid_trajectory.append(P_bid)
        R_bid_trajectory.append(R_bid)
        # Append results for current time step to the corresponding columns
        la_trajectory[:, t] = la
        deg_trajectory[:, t] = deg
        du_trajectory[:, t] = du
        dd_trajectory[:, t] = dd
        p0_trajectory[:, t]=pp0
        e_trajectory[:,t]=E_ini
        dispatch_cost+=dispatch_cost
        print(f'====t={t}')
    
    # ==== Cost calculation =====
    T = cpm.T  # Number of time steps
    energy_cost = sum(P_bid_trajectory[t] * cpm.pr_e_rt[t] for t in range(T))
    regulation_income = sum(R_bid_trajectory[t] * cpm.pr_fre[t] for t in range(T))  # Regulation income
    
    flex_procurement_cost = sum(((deg_trajectory[n, t] + dd_trajectory[n, t] + du_trajectory[n, t]) * la_trajectory[n, t]) 
                       for n in range(cpm.N) for t in range(T))  # Penalty cost
    
    total_cost = energy_cost - regulation_income + flex_procurement_cost+dispatch_cost  # Total cost
    
    # Print detailed costs
    print('=========fre_reg model================')
    print(f'总成本={total_cost}($)')
    print(f'电能量市场购电成本={energy_cost}($)')
    print(f'调频市场收益={regulation_income}($)')
    print(f'补贴成本={flex_procurement_cost}($)')
    print(f'dispatch成本={dispatch_cost}($)')
    
    
    # Package cost details
    cost_details = {
        "charging_cost": energy_cost,
        "regulation_income": regulation_income,
        "penalty_cost": flex_procurement_cost,
        "dispatch_cost":dispatch_cost,
        "total_cost": total_cost
    }
    
    # Package results
    results = {
        "P_bid_trajectory": P_bid_trajectory,
        "R_bid_trajectory": R_bid_trajectory,
        "la_trajectory": la_trajectory,
        "deg_trajectory": deg_trajectory,
        "du_trajectory": du_trajectory,
        "dd_trajectory": dd_trajectory,
        "p0_trajectory": p0_trajectory,
        "e_trajectory": e_trajectory
    }
    return cost_details, results

# %% Main script
np.random.seed(66)
cost_details,results = run_simulation(1,200,cpm,1)
# Save results for current k
np.savez(f"../output/bidding_trajectories.npz", 
             P_bid_trajectory=results["P_bid_trajectory"], 
             R_bid_trajectory=results["R_bid_trajectory"])
np.savez(f"../output/output_trajectories.npz", 
             la_trajectory=results["la_trajectory"], 
             deg_trajectory=results["deg_trajectory"], 
             du_trajectory=results["du_trajectory"], 
             dd_trajectory=results["dd_trajectory"])
np.savez(f"../output/EV_power_trajectories.npz", 
             p0_trajectory=results["p0_trajectory"], 
             e_trajectory=results["e_trajectory"])
print(f"=== Results have been saved.")
# %%
k_values = [50,100,150,200,250,300,350,400]  # Define different k values
# 成本和收入存储
charging_costs,penalty_costs = np.zeros(len(k_values)),np.zeros(len(k_values))  # 用 NumPy 数组初始化，大小为 k_values 的长度
dispatch_costs=np.zeros(len(k_values))
regulation_incomes = np.zeros(len(k_values))
total_costs = np.zeros(len(k_values))
all_results = {}  # To store results for each k

for idx,k in enumerate(k_values):
    cost_details,results = run_simulation(0,k,cpm,1)

    charging_costs[idx] = cost_details["charging_cost"]
    penalty_costs[idx] = cost_details["penalty_cost"]
    regulation_incomes[idx] = cost_details["regulation_income"]
    dispatch_costs[idx] = cost_details["dispatch_cost"]
    total_costs[idx] = cost_details["total_cost"]
    
    all_results[k] = results

    # Save results for current k
    np.savez(f"../output/bidding_trajectories_k_{k}.npz", 
             P_bid_trajectory=results["P_bid_trajectory"], 
             R_bid_trajectory=results["R_bid_trajectory"])
    np.savez(f"../output/output_trajectories_k_{k}.npz", 
             la_trajectory=results["la_trajectory"], 
             deg_trajectory=results["deg_trajectory"], 
             du_trajectory=results["du_trajectory"], 
             dd_trajectory=results["dd_trajectory"])
    np.savez(f"../output/EV_power_trajectories_k_{k}.npz", 
             p0_trajectory=results["p0_trajectory"], 
             e_trajectory=results["e_trajectory"])
    print(f"=== Results for k = {k} have been saved.")

# %% === Save overall results and visualize ===
x = np.arange(len(k_values))  # x 轴位置
width = 0.4  # 柱状图宽度
fig, ax = plt.subplots(figsize=(5, 3))
# 绘制柱状图分量
ax.bar(x, -charging_costs, width, label='Energy Cost', color='tab:blue')
ax.bar(x, -penalty_costs, width, bottom=-charging_costs, 
       label='Flexibility Compensation Cost', color='tab:orange')
# ax.bar(x, -dispatch_costs, width,bottom=-charging_costs-penalty_costs, label='Dispatch Cost', color='tab:red')

ax.bar(x, regulation_incomes, width, label='Frequency Income', color='tab:green')

# 总成本线条
ax.plot(x, -total_costs, marker='o', linestyle='--', color='red', label='Total Cost')

# 添加标签和标题
ax.set_xticks(x)
ax.set_xticklabels([k for k in k_values])
# ax.set_ylabel('Cost ($)')
# ax.set_title('Cost Breakdown and Total Cost for Different k Values')
ax.legend()

# 显示网格
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xlabel('k')
ax.set_ylabel('Cost / Revenue ($)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# 展示图表
plt.tight_layout()
plt.savefig(f"../output/plot_k.pdf")
plt.show()


# %% save optimization results 
# Save the bidding trajectories for later use in visualization
save_file_path="../output/bidding_trajectories.npz"
np.savez(save_file_path, P_bid_trajectory=P_bid_trajectory, R_bid_trajectory=R_bid_trajectory)
print("=== 1.bidding results have been saved.")
# Save flexibility procurement data to files for later use
np.savez("../output/output_trajectories.npz", la_trajectory=la_trajectory,deg_trajectory=deg_trajectory, du_trajectory=du_trajectory, dd_trajectory=dd_trajectory)
print("=== 2.flexibility procurement results have been saved.")
np.savez("../output/EV_power_trajectories.npz", p0_trajectory=p0_trajectory, e_trajectory=e_trajectory)


# %% ===  outcome visualize ====
np.savez("../output/bound_args.npz", **bound_args) 
print("EV parameters saved successfully.")

# %% ==== Comparisonn1 =====
ev_params = generate_ev_data(cpm.N,k_mode=0,k_v=k,charging_fee=cpm.charging_fee)
    # Instantiate EVA model with generated EV parameters
bb = EVA_ms(ev_params)
bound_args = bb.aggregation_v2g()
# Method1: fast_charging 
data = np.load("../output/output_trajectories.npz")
la = data['la_trajectory']
p_fast = bb.fast_charging(cpm.pr_e_rt)
bb.res_fast_plot(p_fast,1)
bb.build_model_reg_woflex(cpm,bound_args,la)
bb.bidding_res_plot(0)
# %% 矩阵化
from dispatch_matrices import *
for t in range(cpm.T):
    power_dispatch_plp(t)
#%%

# Define the curve_plot function
def curve_plot(t, num_crs, label_t, color):
    for cr_id in range(num_crs):
        # Load the critical region data from the file
        filename = f'./data/crs_90/cr{t}_{cr_id}.mat'
        cr = loadmat(filename)

        # Extract the value function coefficients and region bounds
        vf_coeff_t = cr['cr']['vf_coeff_t'].item()
        vf_b = cr['cr']['vf_b'].item()

        # Region bounds E * theta <= f
        E, f = cr['cr']['E'].item(), cr['cr']['f'].item()
        ra = f / E
        x = [ra[0], ra[1]]  # Boundaries of theta in this region

        # Calculate the value function at the boundaries
        vf = vf_coeff_t * np.array(x) + vf_b+5.5

        # Plot the critical region, only label for the first region (cr_id == 0)
        if cr_id == 0:
            ax.plot(x, vf , 'x', linewidth=2, color=color, label=label_t)
            ax.plot(x, vf , linewidth=2, alpha=0.7, color=color)
        else:
            ax.plot(x, vf , 'x', linewidth=2, color=color)
            ax.plot(x, vf , linewidth=2, alpha=0.7, color=color)

# %% Create the plot
fig, ax = plt.subplots(figsize=(7, 3.5))

# Plot for different t values with corresponding labels and different colors
curve_plot(4, 16, label_t='t=4', color='b')  # Blue for t=4
curve_plot(5, 24, label_t='t=5', color='g')  # Green for t=5
curve_plot(6, 50, label_t='t=6', color='r')  # Red for t=6
curve_plot(7, 58, label_t='t=7', color='c')  # Cyan for t=7
curve_plot(8, 76, label_t='t=8', color='m')  # Magenta for t=8
curve_plot(9, 85, label_t='t=9', color='y')  # Yellow for t=9
# curve_plot(10, 85, label_t='t=9', color='y')  # Yellow for t=9

# Add legend, grid, and titles
plt.legend()
# plt.grid(True)
# plt.title('Piecewise Linear Value Function for Different t Values')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$Vf(\theta)$')

# Show the plot
# plt.show()
plt.tight_layout()
plt.savefig(
            "/Users/maggie/Library/Mobile Documents/com~apple~CloudDocs/Desktop/01-weekly report/科研/paper_pic/value_fuc.pdf",
            # dpi=600,
            bbox_inches='tight',
            )
# %%
pppp10,user10,jain=bb.power_dispatch_plp(10)
print(jain)






# %% trade-off
pppp10,user10,jain=bb.power_dispatch_by_hand(1, 10,bound_args,0.3534918835281166)
jain_index = calculate_jain_fairness(user10)
print(jain)
print("Jain Fairness Index for users:", jain_index)
# %%


# print(users)
jain_index = calculate_jain_fairness(users)
jain_index2 = calculate_jain_fairness(users2)
print("Jain Fairness Index for users:", jain_index)
print("Jain Fairness Index for users2:", jain_index2)


bb.fig_EV_soc(10, 1)
bb.fig_EV_soc(30, 1)
bb.fig_EV_soc(70, 1)
bb.fig_flex_price()
# %%
users = [10, 10, 10, 10, 10, 10]
jain_index = calculate_jain_fairness(users)
print("Jain Fairness Index for users:", jain_index)

