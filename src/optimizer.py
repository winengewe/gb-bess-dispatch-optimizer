import pulp
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class BESSModularOptimizer:
    """
    Linear Programming (LP) dispatch optimizer for a Battery Energy Storage System (BESS)
    operating in the Great Britain wholesale electricity market (48 settlement periods/day).
    """

    def __init__(
        self,
        power_capacity_mw: float = 50.0,
        energy_capacity_mwh: float = 100.0,
        round_trip_efficiency: float = 0.88,
        degradation_cost_per_mwh: float = 12.50,
        min_soc_pct: float = 0.0,
        max_soc_pct: float = 1.0,
        dt_hours: float = 0.5
    ):
        self.p_max = power_capacity_mw
        self.e_max = energy_capacity_mwh
        self.rte = round_trip_efficiency
        self.deg_cost = degradation_cost_per_mwh
        self.min_soc = min_soc_pct * self.e_max
        self.max_soc = max_soc_pct * self.e_max
        self.dt = dt_hours

        # Split round-trip efficiency equally between charging and discharging
        self.eta_ch = np.sqrt(self.rte)
        self.eta_dis = np.sqrt(self.rte)

    def optimize_dispatch(self, prices: np.ndarray, initial_soc_mwh: float = 50.0) -> pd.DataFrame:
        """
        Solves the BESS arbitrage dispatch LP for a given sequence of wholesale prices (£/MWh).

        Returns:
            pd.DataFrame: Settlement-period dispatch schedule including power, SoC, and cash flows.
        """
        T = len(prices)
        model = pulp.LpProblem("BESS_Dispatch_Optimization", pulp.LpMaximize)

        # 1. Decision Variables
        p_ch = [pulp.LpVariable(f"p_ch_{t}", lowBound=0, upBound=self.p_max) for t in range(T)]
        p_dis = [pulp.LpVariable(f"p_dis_{t}", lowBound=0, upBound=self.p_max) for t in range(T)]
        soc = [pulp.LpVariable(f"soc_{t}", lowBound=self.min_soc, upBound=self.max_soc) for t in range(T)]

        # 2. Objective Function: Net Profit = Gross Revenue - Degradation Penalty
        gross_revenue = pulp.lpSum([
            (p_dis[t] - p_ch[t]) * prices[t] * self.dt for t in range(T)
        ])
        degradation_penalty = pulp.lpSum([
            (p_ch[t] + p_dis[t]) * self.deg_cost * self.dt for t in range(T)
        ])
        model += gross_revenue - degradation_penalty

        # 3. Constraints
        for t in range(T):
            # Energy Conservation Equation
            prev_soc = initial_soc_mwh if t == 0 else soc[t - 1]
            model += soc[t] == prev_soc + (p_ch[t] * self.eta_ch - p_dis[t] / self.eta_dis) * self.dt

        # Terminal SoC Boundary Condition (Return battery at 50% capacity at end of horizon)
        model += soc[T - 1] == initial_soc_mwh

        # 4. Solve LP
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        if pulp.LpStatus[model.status] != "Optimal":
            raise RuntimeError(f"Optimization failed with status: {pulp.LpStatus[model.status]}")

        # 5. Parse Results
        ch_vals = np.array([p_ch[t].varValue for t in range(T)])
        dis_vals = np.array([p_dis[t].varValue for t in range(T)])
        soc_vals = np.array([soc[t].varValue for t in range(T)])
        net_power = dis_vals - ch_vals  # Positive = Discharging, Negative = Charging

        gross_rev = (dis_vals * prices - ch_vals * prices) * self.dt
        deg_pen = (ch_vals + dis_vals) * self.deg_cost * self.dt
        net_rev = gross_rev - deg_pen

        return pd.DataFrame({
            "SettlementPeriod": np.arange(1, T + 1),
            "Price_GBP_MWh": prices,
            "Charge_MW": np.round(ch_vals, 2),
            "Discharge_MW": np.round(dis_vals, 2),
            "NetPower_MW": np.round(net_power, 2),
            "SoC_MWh": np.round(soc_vals, 2),
            "GrossRevenue_GBP": np.round(gross_rev, 2),
            "DegradationCost_GBP": np.round(deg_pen, 2),
            "NetRevenue_GBP": np.round(net_rev, 2)
        })

    def run_dual_pass_benchmark(
        self, 
        forecasted_prices: np.ndarray, 
        actual_prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Runs dual-pass benchmarking to calculate Financial Capture Rate:
        - Pass 1 (Model Run): Dispatch planned using forecasted prices, evaluated on actual settlement prices.
        - Pass 2 (Perfect Foresight): Dispatch planned directly on actual settlement prices.
        """
        # Pass 1: Schedule using forecasts
        plan_df = self.optimize_dispatch(forecasted_prices)
        
        # Re-evaluate model's chosen dispatch schedule against actual prices
        ch_vals = plan_df["Charge_MW"].values
        dis_vals = plan_df["Discharge_MW"].values
        
        model_gross_rev = np.sum((dis_vals * actual_prices - ch_vals * actual_prices) * self.dt)
        model_deg_cost = np.sum((ch_vals + dis_vals) * self.deg_cost * self.dt)
        model_net_rev = model_gross_rev - model_deg_cost

        # Pass 2: Perfect Foresight Schedule
        pf_df = self.optimize_dispatch(actual_prices)
        pf_net_rev = pf_df["NetRevenue_GBP"].sum()

        # Metrics & Full Cycles
        capture_rate = (model_net_rev / pf_net_rev * 100.0) if pf_net_rev > 0 else 0.0
        total_throughput_mwh = np.sum(ch_vals + dis_vals) * self.dt
        equivalent_cycles = total_throughput_mwh / (2.0 * self.e_max)

        return {
            "model_dispatch_df": plan_df,
            "perfect_foresight_df": pf_df,
            "gross_revenue_gbp": round(float(model_gross_rev), 2),
            "degradation_cost_gbp": round(float(model_deg_cost), 2),
            "model_net_revenue_gbp": round(float(model_net_rev), 2),
            "perfect_foresight_revenue_gbp": round(float(pf_net_rev), 2),
            "capture_rate_pct": round(float(capture_rate), 1),
            "equivalent_full_cycles": round(float(equivalent_cycles), 2)
        }


if __name__ == "__main__":
    from src.data_ingestion import generate_synthetic_gb_prices

    # Quick end-to-end LP test
    df_sample = generate_synthetic_gb_prices(periods=48)
    actuals = df_sample["MarketIndexPrice"].values
    # Add noise to simulate forecast error
    forecasts = actuals + np.random.normal(0, 6.0, size=48)

    optimizer = BESSModularOptimizer()
    benchmark = optimizer.run_dual_pass_benchmark(forecasts, actuals)

    print("✅ Optimization Engine Test Complete:")
    print(f"Model Net Revenue    : £{benchmark['model_net_revenue_gbp']:.2f}")
    print(f"Perfect Foresight Rev: £{benchmark['perfect_foresight_revenue_gbp']:.2f}")
    print(f"Financial Capture    : {benchmark['capture_rate_pct']}%")
    print(f"Daily Cycles (EFC)   : {benchmark['equivalent_full_cycles']}")
