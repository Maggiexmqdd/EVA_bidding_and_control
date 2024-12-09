"""
Electric Vehicle Aggregator Model for Bidding and Dispatch in Energy and Regulation Markets
--------------------------------------------------------------------------------------------
This script implements the optimization and modeling for an Electric Vehicle Aggregator (EVA).
Key functionalities:
1. Aggregation of EV flexibility, including charge and discharge ranges.
2. Fast charging cost calculation and visualization.
3. Model formulation for market bidding, incorporating both energy and regulation markets.

The implementation includes:
- EVA's operational constraints (e.g., battery state of charge (SoC), energy bounds).
- Optimization of EVA's revenue by leveraging charging/discharging flexibility of EVs.
- Usage of Gurobi to solve the optimization model.

Dependencies:
- numpy
- matplotlib
- gurobipy
- scipy.stats
"""

# %%
from gurobipy import *
import numpy as np
# import os
# from scipy.stats import truncnorm
import time
# import math
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


class EVA_ms:
    def __init__(self, ev_params) -> None:
        """
        Initialize the EVA model with parameters for EVs.
        
        Args:
        ev_params (dict): Contains EV-specific parameters like arrival/departure times, 
        battery capacities, required energy, etc.
        """
        self.N = ev_params['N']  # Number of EVs
        self.T = 24  # Number of time periods (e.g., 24 hours in a day)
        self.t_arr, self.t_dep = ev_params['t_arr'], ev_params['t_dep']  # Arrival and departure times
        self.soc_arr, self.soc_dep = ev_params['soc_arr'], ev_params['soc_dep']  # Initial and target SoC
        self.B, self.E_req = ev_params['Battery_capacity'], ev_params['required_power']  # Battery capacity and energy required
        self.p_max, self.eff = ev_params['maximum_charging'], ev_params['charging_eff']  # Max charging power and efficiency
        self.E_arr = self.B * self.soc_arr  # Initial energy
        self.E_r = self.B * self.soc_dep  # Target energy
        self.k = ev_params['k']  # Flexibility pricing coefficient
        self.e = ev_params['e']  # Flexibility pricing coefficient
        self.eta_c, self.eta_d = ev_params['charging_eff'], ev_params['discharging_eff']  # Charging/discharging efficiency

    def aggregation_v2g(self):
        """
        Compute the aggregated flexibility ranges for all EVs.
        
        Returns:
            bound_args (dict): Contains the flexibility bounds (E_min, E_max, E_best) 
                               and power ranges (p_low, p_up) for all EVs.
        """
        T = self.T
        num = self.N
        N = num
        eta_c = self.eta_c
        eta_d = self.eta_d
        E_min = np.zeros((num, T+1))
        E_max = np.zeros((num, T+1))
        E_best = np.zeros((num, T))
        p_up = np.zeros((num, T))
        p_low = np.zeros((num, T))
        vmin = np.zeros((num, T+1))
        vmin_ = np.zeros((num, T+1))
        vmax = np.zeros((num, T+1))

        for i in range(N):
            Emin = self.B[i] * 0.1
            Emax = self.B[i] * 0.95
            E_min[i, T] = Emin
            E_max[i, T] = Emax
            for t in range(T):
                if t < self.t_arr[i]:
                    E_min[i, t] = self.E_arr[i]
                    E_max[i, t] = self.E_arr[i]
                    E_best[i, t] = self.E_arr[i]
                    p_up[i, t] = 0
                    p_low[i, t] = 0
                elif t==self.t_arr[i]:
                    E_min[i, t] = self.E_arr[i]
                    E_max[i, t] = self.E_arr[i]
                    E_best[i, t] = self.E_arr[i]
                    p_up[i, t] = self.p_max[i]
                    p_low[i, t] = -self.p_max[i]
                elif t >= self.t_dep[i]:
                    E_min[i, t] = self.E_r[i]
                    E_max[i, t] = Emax
                    E_best[i, t] = self.E_r[i]
                    p_up[i, t] = 0
                    p_low[i, t] = 0
                else:
                    vmin_[i,t] = self.E_arr[i] - self.p_max[i] / self.eta_d[i] * (
                            t - self.t_arr[i])
                    vmin[i, t] = self.E_r[i] - self.eff[i] * self.p_max[i] * (
                        self.t_dep[i] - t)
                    vmax[i,t] = self.E_arr[i] + self.eta_c[i] * self.p_max[i] * (
                             t - self.t_arr[i])
                    E_min[i, t] = max(vmin_[i, t], vmin[i, t], Emin)
                    E_max[i, t] = min(Emax, vmax[i, t])
                    E_best[i, t] = min(self.E_r[i], vmax[i, t])
                    p_up[i, t] = self.p_max[i]
                    p_low[i, t] = -self.p_max[i]

        bound_args = {
            "E_min": E_min,
            "E_max": E_max,
            "E_best": E_best,
            "p_low": p_low,
            "p_up": p_up
        }
        self.E_min = E_min
        self.E_max = E_max
        self.E_best = E_best
        self.p_low = p_low
        self.p_up = p_up
        return bound_args
        # return E_min,E_max,p_low,p_up
        # fig, ax1 = plt.subplots()
    
    def boundary_calc(self):
        """
        Compute the power/energy ranges for all EVs.
        Returns:
            bound_args (dict): Contains the flexibility bounds (E_min, E_max, E_best) 
                               and power ranges (p_low, p_up) for all EVs.
        """
        T = self.T
        num = self.N
        N = num
        eta_c = self.eta_c
        eta_d = self.eta_d
        E_min = np.zeros((num, T))
        E_max = np.zeros((num, T))
        E_best = np.zeros((num, T))
        p_up = np.zeros((num, T))
        p_low = np.zeros((num, T))
        vmin = np.zeros((num, T))
        vmin_ = np.zeros((num, T))
        vmax = np.zeros((num, T))

        for i in range(N):
            Emin = self.B[i] * 0.1
            Emax = self.B[i] * 0.95
            E_r=self.E_r[i]
            for t in range(T):
                if t < self.t_arr[i]:
                    # E_min[i, t] = 0
                    # E_max[i, t] = 0
                    E_min[i, t] = self.E_arr[i]
                    E_max[i, t] = self.E_arr[i]
                    E_best[i, t] = self.E_arr[i]
                    p_up[i, t],p_low[i, t] = 0,0
                elif t==self.t_arr[i]:
                    E_min[i, t] = self.E_arr[i]
                    E_max[i, t] = self.E_arr[i]
                    E_best[i, t] = self.E_arr[i]
                    p_up[i, t],p_low[i, t] = self.p_max[i],-self.p_max[i]
                elif t > self.t_dep[i]:
                    E_min[i, t] = E_r
                    E_max[i, t] = Emax
                    E_best[i, t] = E_r
                    p_up[i, t],p_low[i, t] = 0,0
                else:
                    vmin_[
                        i,
                        t] = self.E_arr[i] - self.p_max[i] / self.eta_d[i] * (
                            t - self.t_arr[i])
                    vmin[i, t] = self.E_r[i] - self.eff[i] * self.p_max[i] * (
                        self.t_dep[i] - t)
                    vmax[i,
                         t] = self.E_arr[i] + self.eta_c[i] * self.p_max[i] * (
                             t - self.t_arr[i])
                    # vmax[i,t] = self.E_arr[i] + self.eta_c[i] * self.p_max[i] * (
                    #          t - self.t_arr[i])

                    E_min[i, t] = max(vmin_[i, t], vmin[i, t], Emin)
                    E_max[i, t] = min(Emax, vmax[i, t])
                    E_best[i, t] = min(self.E_r[i], vmax[i, t])
                    p_up[i, t] = self.p_max[i]
                    p_low[i, t] = -self.p_max[i]

        bound_args = {
            "E_min": E_min,
            "E_max": E_max,
            "E_best": E_best,
            "p_low": p_low,
            "p_up": p_up
        }
        self.E_min = E_min
        self.E_max = E_max
        self.E_best = E_best
        self.p_low = p_low
        self.p_up = p_up
        return bound_args
        # return E_min,E_max,p_low,p_up
        # fig, ax1 = plt.subplots()
    
    def fast_charging(self, pr_e_da):
        """
        Calculate the cost and revenue of fast charging for all EVs.

        Args:
            pr_e_da (array): Day-ahead electricity prices.
            charging_fee (array): Charging fees for the EVs.

        Returns:
            p_act (array): Actual charging power for all EVs.
        """
        T, N = self.T, self.N
        eta = 0.93
        p_act = np.zeros((N, T))
        cost, ch_fee = 0, 0
        for i in range(N):
            t0 = int((self.E_r[i] - self.E_arr[i]) / (self.p_max[i] * eta))
            for tau in range(T):
                if (tau >= self.t_arr[i]) and (tau <= self.t_arr[i] + t0):
                    p_act[i, tau] = self.p_max[i]
            cost += np.sum(p_act[i, t] * pr_e_da[t] for t in range(T))
            # cost2+=np.sum(p[i,t]/1000*pr_e_rt[t] for t in range(T))
            # ch_fee += np.sum(p_act[i, t] * charging_fee[t] for t in range(T))
        print('fast charging cost:', cost)
        # print('fast charging fee:', ch_fee)
        print('total revenue:',-cost)
        return p_act
    def res_fast_plot(self, p_act,mode=0):
        # fig, ax1 = plt.subplots(figsize=(5, 3))
        plt.figure(figsize=(6, 4))
        p1 = np.sum(p_act, axis=0)
        x = range(24)
        plt.bar(x,
                p1,
                width=0.8,
                color='#F1B656',
                alpha=1,
                label=f'Energy bids')

        ax = plt.gca()  #获取坐标轴对象
        # 将横坐标轴设置在 y=0 处
        # ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        
        plt.xlim(0, 24.5)
        new_labels = list(range(12, 24,4)) + list(range(0, 12+1,4)) 
        formatted_labels = [f"{hour}:00" for hour in new_labels]  # Format as HH:00
        ax.set_xticks(range(0, 24+1, 4))  # Set tick positions with a 4-hour interval
        ax.set_xticklabels(formatted_labels[:len(range(0, 24+1, 4))])  # Adjust labels based on T and interval
        # ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel('Time period(1h)')
        plt.ylabel('Bids Capacity(kW)')
        plt.legend(fontsize=10)
        ax.legend()
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        if mode == 1:
            plt.tight_layout()
            plt.savefig( "../output/bidding_res_fast.pdf",
                # dpi=600,
                )
    def build_model_reg(self, cpm, bound_args, tau, EE0,flex=1):
        """
        Build and solve the optimization problem for time step `tau`.

        Parameters:
            cpm: Market price parameter object, containing real-time electricity and frequency regulation prices.
            bound_args: Dictionary containing power bounds and energy constraints.
            tau: Current time step.
            EE0: Initial energy states of the EVs.

        Returns:
            flag: Solver status flag (0 for success, -1 for failure).
            P_bid: Power bid quantity of the EVA at time `tau`.
            R_bid: Regulation bid quantity of the EVA at time `tau`.
            E0_new: Energy states of EVs for the next time step.
        """
        # === Parameter Initialization ===
        B, s = [-1, 1], [-1, 1, -0.5, 0.5]
        pi_o = [0, 0, 0.45, 0.55]  # 概率分布
        S_num, T, N = 4, cpm.T, cpm.N
        p_low, p_up = bound_args['p_low'],bound_args['p_up']
        E_min, E_max, E_best = bound_args['E_min'], bound_args['E_max'], bound_args['E_best']
        horizon = min(cpm.H, T - tau)  # Horizon for the optimization problem
        
        # === Create Optimization Model ===
        m = Model(f'EV_reg_at_{tau}')

        # ===== Variable Definitions =====
        P, R = {}, {}  # EVA bids
        P_ch, P_dis, D, E, E0 = {}, {}, {}, {}, {}  # EV-level variables
        p0,p0_c,p0_d, dd, du = {}, {}, {} , {}, {}      # EV baseline power
        delta_up, delta_dn = {}, {}   # Flexibility adjustments
        lamda, reg = {}, {}  # EV flexibility costs and contributions
        

        for t in range(tau,tau+horizon):
            # EVA参与市场申报的量
            P[t] = m.addVar(lb=-900,ub=900,vtype=GRB.CONTINUOUS,name=f'P_{t}')
            R[t] = m.addVar(lb=0, ub=900, vtype=GRB.CONTINUOUS, name=f'R_{t}')
            for n in range(N):
                p0[n,t]=m.addVar(lb=-10,ub=10,vtype=GRB.CONTINUOUS,name=f'p0_{n,t}')
                p0_c[n,t]=m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p0_c_{n,t}')
                p0_d[n,t]=m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p0_d_{n,t}')
                dd[n,t]=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'D_dn{n,t}')
                du[n,t]=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'D_up{n,t}')
                if flex==1:
                    lamda[n,t] = m.addVar(vtype=GRB.CONTINUOUS,name=f'lambda_{n,t}')
                # # dev[n,t] = m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'dev_{n,t}')
                # deg[n, t] = m.addVar(vtype=GRB.CONTINUOUS,name=f'deg_{n,t}')
                # reg[n, t] = m.addVar(vtype=GRB.CONTINUOUS,name=f'reg_{n,t}')
                for ome in range(S_num):
                    P_ch[ome, n, t] = m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'p_ch_ev_{ome,n,t}')
                    P_dis[ome, n, t] = m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'p_dis_ev_{ome,n,t}')
                    delta_up[ome,n,t]=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'd_up{ome,n,t}')
                    delta_dn[ome,n,t]=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'd_dn{ome,n,t}')
                    # D[ome, n, t] = m.addVar(vtype=GRB.BINARY,name=f'bi_{ome,n,t}')
        for t in range(tau,tau+horizon+1):  
             for n in range(N):
                for ome in range(S_num):       
                    E[ome, n, t] = m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'E_{ome,n,t}')
                    E0[n, t] = m.addVar(lb=0,vtype=GRB.CONTINUOUS,name=f'E0_{n,t}')

        ## ===== objective ======
        energy_grid = quicksum(P[t] * cpm.pr_e_rt[t] for t in range(tau,tau+horizon))
        fre_to_grid = quicksum(R[t] * cpm.pr_fre[t]  for t in range(tau,tau+horizon))
        market_cost=energy_grid-fre_to_grid
        if flex==1:
            com_to_EVs = quicksum(lamda[n,t] * (self.k[n,t] * lamda[n,t]) for n in range(N) for t in range(tau,tau+horizon))
        else:
            com_to_EVs=0
        # here we consider the redipatch cost for === different scenarios typcial===:
        # now the redipatch cost is set all to be 0.05
        ev_dispatch_cost = quicksum(pi_o[ome]*(delta_up[ome,n,t]-delta_dn[ome,n,t])*cpm.pr_e_rt[t]/4
                                for t in range(tau,tau+horizon) for n in range(N) for ome in range(2,4))
        # ev_dispatch_cost=0 # Placeholder for EV dispatch cost; can be updated as needed
        ev_cost = com_to_EVs + ev_dispatch_cost

        m.setObjective(market_cost+ev_cost, sense=GRB.MINIMIZE)
        ## ===== constraints ======
        # market-level
        for t in range(tau,tau+horizon):
            m.addConstr(P[t] == quicksum(p0[n, t] for n in range(N)),name=f'e_balance{t}')
            m.addConstr(R[t] <= quicksum(du[n,t] for n in range(N)),name=f'reg_range_up{t}')
            m.addConstr(R[t] <= quicksum(dd[n,t] for n in range(N)),name=f'reg_range_dn{t}')
        # EV level
        for n in range(N):
            m.addConstr((E0[n, tau]== EE0[n]), name=f'intial of E0_{n}')
            for t in range(tau,tau+horizon):
                #baseline
                m.addConstr(p0[n, t] == p0_c[n, t]-p0_d[n, t])
                m.addConstr(p0_c[n, t]*p0_d[n, t]==0)
                # # ====flex related constraints====
                if flex==1:
                    m.addConstr((self.k[n,t] * lamda[n,t] ==  p0_d[n, t]+dd[n,t]+du[n,t]),name=f'flex_proc{n,t}')
                    # m.addConstr((self.k[n,t] * lamda[n,t] ==  p0_d[n, t]+
                    #              quicksum(pi_o[ome]*(delta_up[ome,n,t]+delta_dn[ome,n,t]) 
                    #                       for ome in range(2,4))),name=f'flex_proc{n,t}')
                # Energy bounds and dynamics for baseline and extreme cases
                m.addConstr((E0[n,t+1] == E0[n,t] + quicksum(pi_o[ome]*(P_ch[ome, n, t] * self.eta_c[n] -P_dis[ome, n, t] / self.eta_d[n]) 
                                                             for ome in range(2,4))),name=f'overall_energy_update_{n,t}')
                m.addConstr((E0[n, t+1] >= E_min[n, t+1]),name=f'energy_lower0_bounds{n,t+1}')
                m.addConstr((E0[n, t+1] <= E_max[n, t+1]),name=f'energy_upper0_bounds{n,t+1}')
                
                m.addConstr((p0[n, t] <= p_up[n, t]))
                m.addConstr((p0[n, t] >= p_low[n, t]))
                # Upward and downward flexibility bounds
                m.addConstr(du[n,t] <=  p0[n, t]- p_low[n, t], name=f'EV_range_up{n,t}') 
                m.addConstr(dd[n,t] <= -p0[n, t]+ p_up[n, t],  name=f'EV_range_dn{n,t}')
                m.addConstr(p0[n, t]-du[n,t] ==  P_ch[0,n,t]-P_dis[0,n,t]) 
                m.addConstr(p0[n, t]+dd[n,t] ==  P_ch[1,n,t]-P_dis[1,n,t])
                # for extreme cases, and 
                for ome in range(2):
                    m.addConstr((E[ome, n, tau]== EE0[n]),name=f'initial_e_{ome,n}')
                    m.addConstr((E[ome, n,t+1] == E[ome, n, t] 
                                 +P_ch[ome, n, t] * self.eta_c[n]- P_dis[ome, n, t]/ self.eta_d[n]) ,name=f'energy_update_{ome,n,t}')
                    # Energy bounds for extreme cases
                    m.addConstr((E[ome, n, t+1] >= E_min[n, t+1]),name=f'energy_lower_bounds{ome,n,t+1}')
                    m.addConstr((E[ome, n, t+1] <= E_max[n, t+1]),name=f'energy_upper_bounds{ome,n,t+1}')

                # for all 
                for ome in range(2,4):
                    # Power balance considering stochastic regulation signal
                    m.addConstr((P[t] - s[ome] * R[t] == quicksum((P_ch[ome, n, t] - P_dis[ome, n, t]) for n in range(N))) )
                    # Power charging/discharging bounds                    
                    # m.addConstr((P_ch[ome, n, t] <= (1 - D[ome, n, t]) * p_up[n, t]))
                    # m.addConstr((P_dis[ome, n, t] <= -D[ome, n, t] * p_low[n, t]))
                # for ome in range(2,4):
                    m.addConstr((P_ch[ome, n, t] - P_dis[ome, n, t]==p0[n, t]-delta_up[ome,n,t]+delta_dn[ome,n,t])) 
                    m.addConstr(P_ch[ome, n, t] * P_dis[ome, n, t]==0)
                    # Bounds on delta adjustments
                    m.addConstr(delta_up[ome,n,t]<=du[n,t])     
                    m.addConstr(delta_dn[ome,n,t]<=dd[n,t])
                    m.addConstr(delta_dn[ome,n,t]*delta_up[ome,n,t]==0)
                

        # ===== solver setting====
        m.setParam('OutputFlag', 0)
        m.setParam
        m.write(f'../output/modelprint/model{tau}.lp')
        m.optimize()
        if m.status != GRB.Status.OPTIMAL:
            print(f"=== Warning ===\nThe model at t={tau} did not reach an optimal solution. Status: {m.status}")
            return -1
        
        flag=0
        # save the optimal value
        P_bid,R_bid=P[tau].x,R[tau].x
        # EV control & state
        E0_new = np.array([E0[n, tau + 1].x for n in range(N)])#this is for updation
        pp0 = np.array([p0_c[n, tau].x-p0_d[n, tau].x for n in range(N)]) 
        # Extract deg_res, du_res, dd_res at time tau
        if flex==1:
            la = np.array([lamda[n, tau].x for n in range(N)])
        else:
            la=np.array([ 0.016 ])
        deg = np.array([p0_d[n, tau].x for n in range(N)])  # Example: degradation
        du = np.array([du[n, tau].x for n in range(N)])    # Example: upward regulation
        dd = np.array([dd[n, tau].x for n in range(N)])    # Example: downward regulation
        dispatch_cost=sum(pi_o[ome]*(delta_up[ome,n,tau].x-delta_dn[ome,n,tau].x)*cpm.pr_e_rt[tau]/4
                                for n in range(N) for ome in range(2,4))
        # Return results including the new variables
        return flag, P_bid, R_bid, E0_new, la, deg, du, dd, pp0,dispatch_cost

    def bidding_res_plot(self, mode=0):
        # fig, ax1 = plt.subplots(figsize=(5, 3))
        plt.figure(figsize=(6, 4))
        p1 = np.sum(self.ppp, axis=0)
        x = range(24)
        width = 0.8
        # Plot downward regulation (P_bid + R_bid) with fill_between
        plt.fill_between(x, p1, p1 + self.Fre, color='#397FC7', alpha=0.2, label=r'Downward regulation range')

        # Plot upward regulation (P_bid - R_bid) with fill_between
        plt.fill_between(x, p1, p1 - self.Fre, color='#040676', alpha=0.2, label='Upward regulation range')
        plt.plot(x,
                p1+self.Fre,
                marker='+',
                color='#397FC7',
                alpha=0.7,
                label=f'downward regulation')
        plt.plot(x,
                 p1-self.Fre,
                 marker='o',
                 color='#040676',
                 alpha=0.7,
                 label=f'upward regulation')
        plt.bar(x,p1,width=width,color='#F1B656',
                alpha=1,label=f'energy bids')
        ax = plt.gca()  #获取坐标轴对象
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        
        plt.xlim(0, 24.5)
        new_labels = list(range(12, 24,4)) + list(range(0, 12+1,4)) 
        formatted_labels = [f"{hour}:00" for hour in new_labels]  # Format as HH:00
        ax.set_xticks(range(0, 24+1, 4))  # Set tick positions with a 4-hour interval
        ax.set_xticklabels(formatted_labels[:len(range(0, 24+1, 4))])  # Adjust labels based on T and interval
    
        # ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel('Time period(1h)')
        plt.ylabel('Bids Capacity(kW)')
        plt.legend(fontsize=10)
        ax.legend()
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        if mode == 1:
            plt.tight_layout()
            plt.savefig(
                "../output/bidding_resB.pdf",
            )


    def build_model_reg_woflex(self, cpm, bound_args,la):
        """
        Build and solve the optimization problem for centralized.

        Parameters:
            cpm: Market price parameter object, containing real-time electricity and frequency regulation prices.
            bound_args: Dictionary containing power bounds and energy constraints.
            tau: Current time step.
            EE0: Initial energy states of the EVs.

        Returns:
            flag: Solver status flag (0 for success, -1 for failure).
            P_bid: Power bid quantity of the EVA at time `tau`.
            R_bid: Regulation bid quantity of the EVA at time `tau`.
            E0_new: Energy states of EVs for the next time step.
        """
        s = [-1, 1, -0.4, 0.6]
        pi_o = [0.1, 0.1, 0.5, 0.5]
        S_num, T, N = 4, self.T, self.N
        p_low, p_up = bound_args['p_low'],bound_args['p_up']
        E_min, E_max, E_best = bound_args['E_min'], bound_args['E_max'], bound_args['E_best']
        
        m = Model('EV_reg_woflex')
        # ====== variable =======
        P, R = {}, {} # EVA bids
        P_ch, P_dis, D, E ,E0= {}, {}, {}, {},{}
        p0, dd, du = {}, {}, {}       # EV dispatch variables
        delta_up, delta_dn = {}, {}   # Flexibility adjustments
        for t in range(T):
            # EVA参与市场申报的量
            P[t] = m.addVar(lb=-1000,ub=1000,vtype=GRB.CONTINUOUS,name=f'P_eva_{t}')
            R[t] = m.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS, name=f'R_{t}')
            for n in range(N):
                p0[n,t]=m.addVar(lb=-10,ub=10,vtype=GRB.CONTINUOUS,name=f'p0_{n,t}')
                dd[n,t]=m.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name=f'dd_{n,t}')
                du[n,t]=m.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name=f'du_{n,t}')
                for ome in range(S_num):
                    P_ch[ome, n, t] = m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'P_ch_ev_{ome,n,t}')
                    P_dis[ome, n, t] = m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'P_dis_ev_{ome,n,t}')
                    D[ome, n, t] = m.addVar(vtype=GRB.BINARY,
                                            name=f'D_{ome,n,t}')
                    E[ome, n, t] = m.addVar(lb=0,ub=50,
                                            vtype=GRB.CONTINUOUS,
                                            name=f'E_{ome,n,t}')
                    E0[n, t] = m.addVar(lb=0,ub=50,vtype=GRB.CONTINUOUS,name=f'E_{n,t}')
                    delta_up[ome,n,t]=m.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name=f'delta_up{ome,n,t}')
                    delta_dn[ome,n,t]=m.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name=f'delta_dn{ome,n,t}')

        ## ===== objective ======
        energy_grid = quicksum(P[t] * cpm.pr_e_rt[t] for t in range(T))
        fre_to_grid = quicksum(R[t] * cpm.pr_fre[t] for t in range(T))
        cost_eva = energy_grid - fre_to_grid
        m.setObjective(cost_eva, sense=GRB.MINIMIZE)
        # ===== constraints ======
        # EVA
        for t in range(T):
            m.addConstr(P[t] == quicksum(p0[n, t] for n in range(N)))
            m.addConstr(R[t] <= quicksum(du[n,t] for n in range(N)))
            m.addConstr(R[t] <= quicksum(dd[n,t] for n in range(N)))
            for n in range(N):
                m.addConstr(du[n,t] <= p0[n, t]-p_low[n, t]) 
                m.addConstr(dd[n,t] <= -p0[n, t]+p_up[n, t]) 
                for ome in range(S_num):
                    m.addConstr((P_ch[ome, n, t] - P_dis[ome, n, t]==p0[n, t]-delta_up[ome,n,t]+delta_dn[ome,n,t])) 
                    m.addConstr(delta_up[ome,n,t]<=du[n,t])     
                    m.addConstr(delta_dn[ome,n,t]<=dd[n,t])
                    m.addConstr(du[n,t]*dd[n,t]==0)
        m.addConstrs((P[t] - s[ome] * R[t] == quicksum(
            (P_ch[ome, n, t] - P_dis[ome, n, t]) for n in range(N)))for t in range(T) for ome in range(S_num))
        for ome in range(S_num):
            m.addConstrs((P_ch[ome, n, t] <= (1 - D[ome, n, t]) * p_up[n, t])for t in range(T) for n in range(N))
            m.addConstrs((P_dis[ome, n, t] <= -D[ome, n, t] * p_low[n, t])for t in range(T) for n in range(N))
        # energy bounds
        m.addConstrs(
            (E0[n,t] == E0[n, t - 1] + quicksum(pi_o[ome]*(P_ch[ome, n, t] * self.eta_c[n] -P_dis[ome, n, t] / self.eta_d[n]) for ome in range(2,4))) for t in range(1, T)
            for n in range(N) )
        m.addConstrs((E0[n, t] >= E_min[n, t]) for t in range(T) for n in range(N) )
        m.addConstrs((E0[n, t] <= E_max[n, t]) for t in range(T) for n in range(N) )
        m.addConstrs(
            (E[ome, n,t] == E0[n, t - 1] + P_ch[ome, n, t] * self.eta_c[n] -P_dis[ome, n, t] / self.eta_d[n]) for t in range(1, T)
            for n in range(N) for ome in range(2))
        m.addConstrs((E[ome, n, t] >= E_min[n, t]) for t in range(T) for n in range(N) for ome in range(2))
        m.addConstrs((E[ome, n, t] <= E_max[n, t]) for t in range(T) for n in range(N) for ome in range(2))
        # ===== solver setting====
        m.update()
        m.optimize()
        if m.status == GRB.Status.INFEASIBLE:
            print("The model is infeasible.")
            m.computeIIS()
            # m.write("model.ilp")
        # 存储优化结果
        self.P_da, self.Fre = np.zeros((T)), np.zeros((T))
        self.E_da = np.zeros((N, T))
        self.flex_nor = np.zeros((N,T))
        self.ppp = np.zeros((N, T))
        for tau in range(T):
            self.P_da[tau] = P[tau].x
            self.Fre[tau] = R[tau].x
            for n in range(N):
                self.ppp[n, tau] = sum(pi_o[ome] * (P_ch[ome, n, tau].x - P_dis[ome, n, tau].x)
                    for ome in range(2, 4))
                self.E_da[n, tau] = E0[n, tau].x
                self.flex_nor[n,tau]=sum(pi_o[ome] * (P_dis[ome, n, tau].x)for ome in range(2, 4))+dd[n,tau].x+du[n,tau].x

        # ====cost calculation and print=======
        buy_from_grid = sum(P[t].x * cpm.pr_e_rt[t] for t in range(T))
        conp = sum((self.flex_nor[n,t] * la[n,t]) for n in range(N) for t in range(T))
        # ev_charging_fee = 0.1 *quicksum(pi_o[ome]*(P_ch[ome,n,t].x - P_dis[ome,n,t].x )
        #                                           for ome in range(2,4) for n in range(N) for t in range(T))
        fre_income = sum(self.Fre[t] * cpm.pr_fre[t] for t in range(T))
        Cost_with_grid = buy_from_grid -fre_income+conp
        print('=========fre_reg model================')
        print(f'总成本={Cost_with_grid}($)')
        print(f'充电成本={buy_from_grid}($)')
        print(f'补贴成本={conp}($)')
        # print(f'充电收益={ev_charging_fee}($)') 
        print(f'调频收益={fre_income}($)')
        # ======save====


    def normal_flex_cal(self):
        return self.flex_nor
    def power_dispatch_by_hand(self,s_d,t_cur,bound_args,F_idx):
        env = Env()
        T = self.T
        N = self.N
        eta_c = 0.9
        eta_d = 0.93
        p_low = bound_args['p_low']
        p_up = bound_args['p_up']
        E_min = bound_args['E_min']
        E_max = bound_args['E_max']
        E_best = bound_args['E_best']

        m2=Model('power_allocation')
        p_ch,p_dis={},{}
        flex,D={},{}
        for n in range(N):
            p_ch[n]=m2.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p_ch{n}')
            p_dis[n]=m2.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p_dis{n}')
            D[n]=m2.addVar(vtype=GRB.BINARY,name=f'D{n}')
            flex[n]=m2.addVar(lb=-10,ub=25,vtype=GRB.CONTINUOUS,name=f'flex{n}')
        # ======objective=======
        obj=quicksum(self.la[n,t_cur]*(p_ch[n]*eta_c-2*p_dis[n]/eta_d) for n in range(N))
        # obj=0
        m2.setObjective(obj, sense=GRB.MINIMIZE)
        # =====constraints======

        m2.addConstr(quicksum((p_ch[n] - p_dis[n]) for n in range(N))==self.P_da[t_cur]-s_d * self.Fre[t_cur])
        # m.addConstr(P_da[t_cur]-s_d*Fre[t]== quicksum((p_ch[n]-p_dis[n])for n in range(N)))
        # m2.addConstrs((p_ch[n]*p_dis[n]==0)for n in range(N))
        # m2.addConstrs((p_ch[n] -p_dis[n] >= p_low[n,t_cur]) for n in range(N))
        # m2.addConstrs((p_ch[n] -p_dis[n] <= p_up[n,t_cur]) for n in range(N))
        m2.addConstrs((p_ch[n] <= (1 - D[n]) * p_up[n,t_cur]) for n in range(N))
        m2.addConstrs((p_dis[n] <= -D[n] * p_low[n, t_cur]) for n in range(N))
        E_t = self.E_da[:, t_cur - 1]  # bidding得到的
        m2.addConstrs(p_ch[n]*eta_c-p_dis[n]/eta_d>=E_min[n,t_cur]-E_t[n] for n in range(N))
        m2.addConstrs(p_ch[n]*eta_c-p_dis[n]/eta_d<=E_max[n,t_cur]-E_t[n]for n in range(N))
        
        m2.addConstrs((flex[n]==self.la[n,t_cur]*(-p_ch[n]*eta_c+2*p_dis[n]/eta_d+E_best[n,t_cur]-E_t[n])) for n in range(N))
        m2.addConstr(((sum(flex[n] for n in range(N))**2)  >= F_idx * N* sum(flex[n]**2 for n in range(N))))
        
        m2.update()
        m2.optimize()
        if m2.status == GRB.Status.INFEASIBLE:
            print("The model is infeasible.")
        elif m2.status == GRB.Status.UNBOUNDED:
            print("The model is unbounded.")
        else:
            print('Status =', m2.status)   #解的状态查询
        p,flex_res=np.zeros(N),np.zeros(N)
        for n in range(N):
            p[n]=p_ch[n].x-p_dis[n].x
            # flex_res[n]=-p_ch[n].x*eta_c+2*p_dis[n].x/eta_d
            flex_res[n]=(-p_ch[n].x*eta_c+2*p_dis[n].x/eta_d+E_best[n,t_cur]-E_t[n])*self.la[n,t_cur]
        jain=sum(flex[n].x for n in range(N))**2/sum(flex[n].x**2 for n in range(N))
        env.dispose()
        return p,flex_res,jain
        
    
    def power_dispatch_run(self,s_d,t_cur,bound_args):
        env = Env()
        T = self.T
        N = self.N
        eta_c = 0.9
        eta_d = 0.93
        p_low = bound_args['p_low']
        p_up = bound_args['p_up']
        E_min = bound_args['E_min']
        E_max = bound_args['E_max']
        E_best = bound_args['E_best']
        E_t = self.E_da[:, t_cur - 1]  # bidding得到的
        m2=Model('power_allocation')
        p_ch,p_dis={},{}
        flex,D={},{}
        deltap,delta_dn={},{}
        for n in range(N):
            p_ch[n]=m2.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p_ch{n}')
            p_dis[n]=m2.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS,name=f'p_dis{n}')
            D[n]=m2.addVar(vtype=GRB.BINARY,name=f'D{n}')
            deltap[n]=m2.addVar(lb=-20,ub=20,vtype=GRB.CONTINUOUS,name=f'd_up{n}')
            # delta_dn[n]=m2.addVar(lb=0,ub=self.dd_res[n,t_cur],vtype=GRB.CONTINUOUS,name=f'd_dn{n}')
            # flex[n]=m2.addVar(lb=-10,ub=25,vtype=GRB.CONTINUOUS,name=f'flex{n}')
        # ======objective=======
        # obj=quicksum(self.la[n,t_cur]*self.k[n]*(p_ch[n]*eta_c-p_dis[n]/eta_d) for n in range(N))
        # obj=0 self.la[n,t_cur]*
        obj=quicksum(self.la[n,t_cur]*(p_dis[n]/eta_d+deltap[n])for n in range(N))
        m2.setObjective(obj, sense=GRB.MINIMIZE)
        # =====constraints======

        m2.addConstr(quicksum((p_ch[n] - p_dis[n]) for n in range(N))==self.P_da[t_cur]-s_d * self.Fre[t_cur])
        # m.addConstr(P_da[t_cur]-s_d*Fre[t]== quicksum((p_ch[n]-p_dis[n])for n in range(N)))
        m2.addConstrs((p_ch[n]*p_dis[n]==0)for n in range(N))
        m2.addConstrs((p_ch[n]-p_dis[n]==self.p0[n,t_cur]-deltap[n])for n in range(N))
        m2.addConstrs(deltap[n]<=self.du_res[n,t_cur] for n in range(N))
        m2.addConstrs(deltap[n]>=-self.dd_res[n,t_cur] for n in range(N))
        # m2.addConstrs(delta_up[n]*delta_dn[n]==0 for n in range(N))
        # m2.addConstrs((p_ch[n] -p_dis[n] >= p_low[n,t_cur]) for n in range(N))
        # m2.addConstrs((p_ch[n] -p_dis[n] <= p_up[n,t_cur]) for n in range(N))
        # ===将bilinear 线性化
        # m2.addConstrs((p_ch[n] <= (1 - D[n]) * p_up[n,t_cur]) for n in range(N))
        # m2.addConstrs((p_dis[n] <= -D[n] * p_low[n, t_cur]) for n in range(N))
        
        # m2.addConstrs(p_ch[n]*eta_c-p_dis[n]/eta_d>=E_min[n,t_cur]-E_t[n] for n in range(N))
        # m2.addConstrs(p_ch[n]*eta_c-p_dis[n]/eta_d<=E_max[n,t_cur]-E_t[n]for n in range(N))
        
        # m2.addConstrs((flex[n]==-p_ch[n]*eta_c+2*p_dis[n]/eta_d+E_best[n,t_cur]-E_t[n]) for n in range(N))
        # m2.addConstr(((sum(flex[n] for n in range(N))**2)  >= F_idx * N* sum(flex[n]**2 for n in range(N))))
        
        m2.update()
        m2.optimize()
        if m2.status == GRB.Status.INFEASIBLE:
            print("The model is infeasible.")
        elif m2.status == GRB.Status.UNBOUNDED:
            print("The model is unbounded.")
        else:
            print('Status =', m2.status)   #解的状态查询
        p,flex_res=np.zeros(N),np.zeros(N)
        for n in range(N):
            p[n]=p_ch[n].x-p_dis[n].x
            # flex_res[n]=-p_ch[n].x*eta_c+2*p_dis[n].x/eta_d
            flex_res[n]=(E_best[n,t_cur]-(E_t[n]+p_ch[n].x*eta_c-2*p_dis[n].x/eta_d))*self.la[n,t_cur]
        jain=sum(flex_res[n]for n in range(N))**2/(N*sum(flex_res[n]**2 for n in range(N)))
        env.dispose()
        return p,flex_res,jain
    
    def power_dispatch_plp(self,t_cur):
        N = self.N
        e = np.ones((1, N))
        I = np.eye(N)
        aaa = np.ones((1, 1))
        eta_c, eta_d = 0.9, 0.93
        # R=6
        # Pe=20
        # p0=np.array([2,3,4,5,6]).reshape(-1,1)
        # print(p0.shape)
        I = np.eye(N)
        Aeq = np.block([[e, -e,e*0],
                    [I,-I,I]])
        Feq=np.hstack((aaa*self.Fre[t_cur],e*0)).T
        beq=np.hstack((aaa*self.P_da[t_cur],self.p0[:,t_cur].reshape(-1,1).T)).reshape(-1,1)

        A = np.block([[-I, I*0,I*0], [I*0,-I,I*0],
                            [I*0,I*0,I], [I*0,I*0,-I]])
        
        b=np.hstack((e*0,e*0,self.du_res[:,t_cur].reshape(-1,1).T,self.dd_res[:,t_cur].reshape(-1,1).T)).reshape(-1,1)
        At = np.array([[1], [-1]])
        bt = np.array([[1], [1]])
        c = np.hstack((e*0, self.la[:,t_cur].reshape(-1,1).T/eta_d,self.la[:,t_cur].reshape(-1,1).T)).reshape(-1, 1)
        plp = {
            'A': A,
            'b': b,
            # 'F': F,
            'Aeq': Aeq,
            'beq': beq,
            'Feq': Feq,
            'At': At,
            'bt': bt,
            # 'Q': Q,
            'c': c,
        }
        savemat(f'./crs/plp_{t_cur}.mat', plp)

    def power_disaptch_model(self, bound_args, t):
        T = self.T
        N = self.N
        eta_c, eta_d = 0.9, 0.93
        # a,c=0.6,0.4
        p_low = bound_args['p_low']
        p_up = bound_args['p_up']
        E_min = bound_args['E_min']
        E_max = bound_args['E_max']
        I = np.eye(N)
        e = np.ones((1, N))
        eng = matlab.engine.start_matlab()
        A = np.block([[I * eta_c, -I * eta_c], [I * -1 / eta_d, I * 1 / eta_d],
                      [I * eta_c, -I * eta_c], [I * -1 / eta_d,
                                                I * 1 / eta_d]])

        E_max_t = E_max[:, t]
        E_min_t = E_min[:, t]
        P_max_t = p_up[:, t]
        P_min_t = p_low[:, t]
        E_t = self.E_da[:, t - 1]  # bidding得到的
        b = np.hstack((E_max_t - E_t, E_t - E_min_t, P_max_t * eta_c,
                       P_min_t * eta_d)).reshape(-1, 1)
        F = b * 0
        print(np.shape(A), np.shape(b), np.shape(F))
        Aeq = np.hstack((e, -e))
        beq = np.array([self.P_da[t]]).reshape(-1, 1)
        Feq = np.array([self.Fre[t]]).reshape(-1, 1)
        print(np.shape(Aeq), np.shape(beq), np.shape(Feq))
        At = np.array([[1], [-1]])
        bt = np.array([[1], [1]])
        print(np.shape(At), np.shape(bt))
        # objective
        Q = e * 0
        c0 = self.k
        c = np.hstack((c0, -c0)).reshape(-1, 1)
        print(f'c={np.shape(c)}')
        mpqp = {
            'A': A,
            'b': b,
            'F': F,
            'Aeq': Aeq,
            'beq': beq,
            'Feq': Feq,
            'At': At,
            'bt': bt,
            'Q': Q,
            'c': c,
        }
        savemat('./plp.mat', mpqp)
        st = time.time()
        num_crs = int(eng.calc_crs())
        et = time.time()
        print('==========')
        print(f'{t} solving for {et-st} seconds.')
        print(f'agent {t} # of CR:', num_crs)


def calculate_jain_fairness(users):
    n = len(users)
    sum_x = sum(users)
    sum_x_squared = sum(map(lambda x: x**2, users))
    jain_index = (sum_x**2) / (n * sum_x_squared)
    return jain_index
