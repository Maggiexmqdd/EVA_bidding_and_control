# %%
'''
Visualization of Power Bid and Flexibility Trajectories

This script visualizes the optimization results (degradation, upward regulation, and downward regulation) 
for multiple groups across time steps. The data is loaded from pre-saved trajectories stored in "output_trajectories.npz".
The results are displayed as stacked bar charts and saved as PDF files.

Colors used:
- Green: Degradation
- Purple: Upward Regulation
- Teal: Downward Regulation
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import EVA_constantParam as cpm
# %% flexibility
def flexibility_plot(mode=0):
    """
    Plot the flexibility contribution for each group, with degradation, upward regulation, 
    and downward regulation, and optionally save the plots.

    Parameters:
        mode (int): If 1, saves the figure as a PDF. Default is 0 (no saving).
    """
    # Load pre-saved trajectories
    data = np.load("../output/output_trajectories.npz")
    deg_trajectory = data['deg_trajectory']
    du_trajectory = data['du_trajectory']
    dd_trajectory = data['dd_trajectory']
    
    # Define the number of time steps (T) and the number of entities (N)
    T = deg_trajectory.shape[1]
    
    # Define colors for each category
    dev_color = '#93B237'
    mil_color = '#AF9BC9'
    flex_color = '#A9C7C4'

    bar_width = 0.8
    index = np.arange(T)  # Time steps

    # Create plots for each group
    for i in range(4):  # Assuming we divide the data into 4 groups
        plt.figure(figsize=(4, 3))  # Create a new figure
        
        # Plot degradation, upward regulation, and downward regulation
        plt.bar(index, np.mean(deg_trajectory[i * (deg_trajectory.shape[0] // 4):(i + 1) * (deg_trajectory.shape[0] // 4), :], axis=0), bar_width, color=dev_color, alpha=0.8, label=r'$p^{dis}$')
        plt.bar(index, np.mean(du_trajectory[i * (du_trajectory.shape[0] // 4):(i + 1) * (du_trajectory.shape[0] // 4), :], axis=0), bar_width, bottom=np.mean(deg_trajectory[i * (deg_trajectory.shape[0] // 4):(i + 1) * (deg_trajectory.shape[0] // 4), :], axis=0), color=mil_color, alpha=0.8, label=r'$\Delta p^{up}$')
        plt.bar(index, np.mean(dd_trajectory[i * (dd_trajectory.shape[0] // 4):(i + 1) * (dd_trajectory.shape[0] // 4), :], axis=0), bar_width, bottom=np.mean(deg_trajectory[i * (deg_trajectory.shape[0] // 4):(i + 1) * (deg_trajectory.shape[0] // 4), :], axis=0) + np.mean(du_trajectory[i * (du_trajectory.shape[0] // 4):(i + 1) * (du_trajectory.shape[0] // 4), :], axis=0), color=flex_color, alpha=0.8, label=r'$\Delta p^{dn}$')
        
        ax = plt.gca()
        plt.xlabel('Time period (1h)')
        plt.ylabel('Flexibility Contribution (kWh)')
        plt.xlim(0, cpm.T+0.5)
        # Set x-axis labels to show time in hours
        new_labels = list(range(12, 24, 4)) + list(range(0, 12+1, 4))  # Hourly labels with 4-hour interval
        formatted_labels = [f"{hour}:00" for hour in new_labels]  # Format as HH:00
        ax.set_xticks(range(0, T+1, 4))  # Set tick positions with a 4-hour interval
        ax.set_xticklabels(formatted_labels[:len(range(0, T+1, 4))])  # Adjust labels based on T and interval

        plt.legend()
        plt.tight_layout()  # Adjust layout to avoid label clipping

        # Save figure if mode is 1
        if mode == 1:
            plt.savefig(f"../output/group_{i}.pdf", bbox_inches='tight')
        
        plt.show()

# Example call
flexibility_plot(1)

# %% 2 bidding
def bidding_res_plot(mode=0,k=200):
    """
    Plot the bidding results with energy bids and regulation bids (upward and downward).

    Parameters:
        mode (int): If 1, saves the figure as a PDF. Default is 0 (no saving).
    """
    # Load pre-saved trajectories
    if k==1:
        loc=f"../output/bidding_trajectories.npz"
    else:
        loc=f"../output/bidding_trajectories_k_{k}.npz"
    data = np.load(loc)
    P_bid_trajectory = data['P_bid_trajectory']
    R_bid_trajectory = data['R_bid_trajectory']
    data2 = np.load("../output/output_trajectories.npz")
    deg = data2['deg_trajectory']
    du = data2['du_trajectory']
    dd = data2['dd_trajectory']
    data3 = np.load("../output/output_trajectories.npz")
    la = data3['la_trajectory']
    # dd_trajectory = data['dd_trajectory']

    T = len(P_bid_trajectory)  # Number of time steps

    # Convert to numpy arrays for easier manipulation
    P_bid = np.array(P_bid_trajectory)
    R_bid = np.array(R_bid_trajectory)

    plt.figure(figsize=(6, 4))

    

    # Plot downward regulation (P_bid + R_bid) with fill_between
    plt.fill_between(range(T), P_bid, P_bid + R_bid, color='#397FC7', alpha=0.2, label=r'$R^{dn}$')

    # Plot upward regulation (P_bid - R_bid) with fill_between
    plt.fill_between(range(T), P_bid, P_bid - R_bid, color='#040676', alpha=0.2, label=r'$R^{up}$')

    # Plot downward regulation (P_bid + R_bid) as a line with markers
    plt.plot(range(T), P_bid + R_bid, marker='+', color='#397FC7', alpha=0.7, label=r'$P+R$')

    # Plot upward regulation (P_bid - R_bid) as a line with markers
    plt.plot(range(T), P_bid - R_bid, marker='o', color='#040676', alpha=0.7, label=r'$P-R$')


    # First, plot energy bids (bar chart, at the bottom)
    plt.bar(range(T), P_bid, width=0.8, color='#F1B656', alpha=1, label='P')
    # Horizontal axis line at y=0
    ax = plt.gca()
    ax.axhline(0, color='gray', linewidth=1, linestyle='--')

    # Set x-axis labels and ticks
    plt.xlim(0, T +0.5)
    
    # Create hourly labels with a 4-hour interval
    new_labels = list(range(12, 24, 4)) + list(range(0, 12+1, 4))  # Hourly labels with 4-hour interval
    formatted_labels = [f"{hour}:00" for hour in new_labels]  # Format as HH:00
    ax.set_xticks(range(0, T+1, 4))  # Set tick positions with a 4-hour interval
    ax.set_xticklabels(formatted_labels[:len(range(0, T+1, 4))])  # Adjust labels based on T and interval
    
    plt.xlabel('Time period (1h)')
    plt.ylabel('Bids Capacity (kW)')
    plt.legend(fontsize=10)
    ax.legend()
    
    if mode == 1:
        plt.tight_layout()
        plt.savefig(f"../output/bidding_res_{k}.pdf")

    plt.show()
    # ====cost calculation and print=======
    buy_from_grid = sum(P_bid[t] * cpm.pr_e_rt[t] for t in range(T))
    conp = sum(((deg[n,t]+dd[n,t]+du[n,t]) * la[n,t]) for n in range(cpm.N) for t in range(T))
    # ev_charging_fee = 0.1 *quicksum(pi_o[ome]*(P_ch[ome,n,t].x - P_dis[ome,n,t].x )
        #                                           for ome in range(2,4) for n in range(N) for t in range(T))
    fre_income = sum(R_bid[t] * cpm.pr_fre[t] for t in range(T))
    Cost_with_grid = buy_from_grid -fre_income+conp
    print('=========fre_reg model================')
    print(f'总成本={Cost_with_grid}($)')
    print(f'充电成本={buy_from_grid}($)')
    print(f'补贴成本={conp}($)')
        # print(f'充电收益={ev_charging_fee}($)') 
    print(f'调频收益={fre_income}($)')
bidding_res_plot(0,1)
# bidding_res_plot(1,loc='../output/bidding_trajectories2.npz')
# %% 3 soc
def EV_soc_plot(n,mode=0):
    """
    Plot the EV state of charge (SoC) and charging/discharging power for a given EV.

    Parameters:
        n (int): Index of the EV.
        ppp (np.ndarray): Power bid data, shape (N, T).
        E_da (np.ndarray): Actual energy data, shape (N, T).
        E_max (np.ndarray): Maximum energy capacity, shape (N, T).
        E_min (np.ndarray): Minimum energy capacity, shape (N, T).
        B (np.ndarray): Battery capacity, shape (N,).
        mode (int): If 1, saves the figure as a PDF. Default is 0 (no saving).
    """
    # Load pre-saved trajectories
    data = np.load("../output/EV_power_trajectories.npz")
    p0 = data['p0_trajectory']
    e= data['e_trajectory']

    with open("../output/bound_args.pkl", "rb") as f:
        bound_args = pickle.load(f)
    E_min = bound_args['E_min']
    E_max = bound_args['E_max']
    
    data = np.load("../output/ev_params.npz")
    # 直接访问参数
    B = data['Battery_capacity']

    fig, ax1 = plt.subplots(figsize=(4, 3))
    # Plot charging/discharging load
    x=range(cpm.T)
    ax1.bar(x, p0[n, :], width=0.8, color='g', alpha=0.5, label=rf'$p^0$')
    

    ax2 = ax1.twinx()
    ax2.plot(x, e[n, :] / B[n] * 100, marker='o', color='b', alpha=1, label=f'SoC')
    ax2.plot(x, E_max[n, :] / B[n] * 100,color='r', alpha=1, label=rf'$SoC^+$')
    ax2.plot(x, E_min[n, :] / B[n] * 100,color='purple', alpha=1, label=rf'$SoC^-$')

    ax1.set_ylabel('EV (Dis)Charging Load (kW)')
    ax1.set_xlabel('Time period (1h)')
    ax2.set_ylabel('SoC (%)')
    # xy
    plt.xlim(0, cpm.T+0.5)
    ax2.set_ylim(0, 100)
    ax1.legend(loc=2,prop = {'size':8})
    ax2.legend(loc='right',prop = {'size':8})
    # ax2.legend(loc='upper center', , ncol=3)
    new_labels = list(range(12, 24)) + list(range(0, 12+1))
    formatted_labels = [f"{hour}:00" for hour in new_labels]  # Format as HH:00
    ax1.set_xticks(range(0, cpm.T+1))  # 明确设置x轴的刻度位置
    ax1.set_xticklabels(formatted_labels)  # 为这些刻度设置标签
    x_major_locator = plt.MultipleLocator(4)
    ax1.xaxis.set_major_locator(x_major_locator)

    if mode == 1:
        plt.tight_layout()
        plt.savefig(f"../output/soc{n}.pdf")

# EV_soc_plot(15,1)
# EV_soc_plot(30,1)
EV_soc_plot(43,1)
EV_soc_plot(64,1)
# for i in range(60,80):
#     EV_soc_plot(i)
    # EV_soc_plot(74,1)

# %%
def flex_price_plot(cpm):
        # Load pre-saved trajectories
        data = np.load("../output/output_trajectories.npz")
        la = data['la_trajectory']
        x = np.arange(cpm.N)
        y = np.arange(cpm.T)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(111, projection='3d')
        # fig= plt.subplots(figsize=(6, 3))
        # ax = Axes3D(fig,auto_add_to_figure=False)
        # fig.add_axes(ax)
        ax.plot_surface(X, Y, Z.T, cmap='plasma')  # 注意需要转置 Z
        # ax1.legend(loc=2)
        new_labels = list(range(12, 24)) + list(range(0, 12)) 
        ax.set_yticks(y)
        ax.set_yticklabels(new_labels)
        x_major_locator = plt.MultipleLocator(2)
        ax.yaxis.set_major_locator(x_major_locator)
        ax.set_zlabel('flexibility price($/kWh)')
        ax.set_box_aspect(aspect=None, zoom=0.9)
        ax.view_init(elev=10, azim=23)
        ax.zaxis.labelpad=3.8
        ax.set_xlabel('EV users')
        ax.set_ylabel('Time period (1h)')
        plt.tight_layout()
        plt.savefig(
            "../output/flex_price.pdf",
            # dpi=600,
            bbox_inches='tight',
            )
    