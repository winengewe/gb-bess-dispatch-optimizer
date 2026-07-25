import math
import pulp
import pandas as pd


class BESSModularOptimizer:
    def __init__(self, config: dict):
        self.power_mw = config["asset"]["power_mw"]
        self.capacity_mwh = config["asset"]["capacity_mwh"]
        self.eta_rt = config["asset"]["round_trip_efficiency"]
        self.eta_one_way = math.sqrt(self.eta_rt)
        self.deg_cost = config["asset"]["degradation_cost_gbp_per_mwh"]
        self.initial_soc = self.capacity_mwh * config["asset"]["initial_soc_pct"]
        self.dt = config["market"]["time_step_hours"]

    def solve_day_ahead_dispatch(
        self, prices: list[float]
    ) -> tuple[pd.DataFrame, float]:
        """
        Solves optimal charge/discharge schedule for N settlement periods.
        """
        T = len(prices)
        prob = pulp.LpProblem("BESS_Arbitrage_Optimization", pulp.LpMaximize)

        # Decision Variables
        p_ch = [
            pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=self.power_mw)
            for t in range(T)
        ]
        p_dis = [
            pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=self.power_mw)
            for t in range(T)
        ]
        soc = [
            pulp.LpVariable(f"SoC_{t}", lowBound=0, upBound=self.capacity_mwh)
            for t in range(T)
        ]

        # Objective: Maximize Arbitrage Revenue - Degradation Penalty
        revenue = pulp.lpSum(
            [(p_dis[t] * prices[t] - p_ch[t] * prices[t]) * self.dt for t in range(T)]
        )
        degradation = pulp.lpSum(
            [self.deg_cost * (p_ch[t] + p_dis[t]) * self.dt for t in range(T)]
        )
        prob += revenue - degradation

        # Constraints
        for t in range(T):
            prev_soc = soc[t - 1] if t > 0 else self.initial_soc
            # Energy conservation constraint
            prob += (
                soc[t]
                == prev_soc
                + (p_ch[t] * self.eta_one_way - p_dis[t] / self.eta_one_way) * self.dt
            )

        # Terminal SoC boundary constraint (end at initial SoC to avoid empty-battery artifacts)
        prob += soc[-1] == self.initial_soc

        # Solve via default CBC solver
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        results = pd.DataFrame(
            {
                "Period": range(1, T + 1),
                "Price_GBP_MWh": prices,
                "Charge_MW": [p_ch[t].varValue for t in range(T)],
                "Discharge_MW": [p_dis[t].varValue for t in range(T)],
                "SoC_MWh": [soc[t].varValue for t in range(T)],
            }
        )

        results["Net_Power_MW"] = results["Discharge_MW"] - results["Charge_MW"]
        results["Gross_Revenue_GBP"] = (
            (results["Discharge_MW"] - results["Charge_MW"])
            * results["Price_GBP_MWh"]
            * self.dt
        )
        results["Degradation_Cost_GBP"] = (
            self.deg_cost * (results["Charge_MW"] + results["Discharge_MW"]) * self.dt
        )
        results["Net_Profit_GBP"] = (
            results["Gross_Revenue_GBP"] - results["Degradation_Cost_GBP"]
        )

        return results, pulp.value(prob.objective)
