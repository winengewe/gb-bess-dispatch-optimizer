import yaml
from src.data_ingestion import generate_synthetic_gb_prices
from src.optimizer import BESSModularOptimizer


def main():
    print("⚡ Launching GB BESS Price Forecasting & Dispatch Optimizer...")

    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Ingest/Generate Day-Ahead Prices
    df_market = generate_synthetic_gb_prices(
        periods=config["market"]["settlement_periods"]
    )
    prices = df_market["MarketIndexPrice"].tolist()

    # 2. Initialize LP Optimizer
    optimizer = BESSModularOptimizer(config)

    # 3. Solve Dispatch Schedule
    dispatch_df, net_obj_value = optimizer.solve_day_ahead_dispatch(prices)

    # 4. Calculate KPIs
    total_gross_rev = dispatch_df["Gross_Revenue_GBP"].sum()
    total_deg_cost = dispatch_df["Degradation_Cost_GBP"].sum()
    total_discharged_mwh = (
        dispatch_df["Discharge_MW"] * config["market"]["time_step_hours"]
    ).sum()
    efc = total_discharged_mwh / config["asset"]["capacity_mwh"]

    # Print Summary Report
    print("\n" + "=" * 50)
    print(" 📊 DAILY DISPATCH OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(
        f" Asset Rating         : {config['asset']['power_mw']} MW / {config['asset']['capacity_mwh']} MWh"
    )
    print(f" Gross Revenue        : £{total_gross_rev:,.2f}")
    print(f" Degradation Penalty  : £{total_deg_cost:,.2f}")
    print(f" Net Daily Profit     : £{net_obj_value:,.2f}")
    print(f" Equivalent Cycles    : {efc:.2f} EFC/day")
    print("=" * 50)

    print("\nSample Dispatch Output (First 6 Settlement Periods):")
    print(
        dispatch_df[
            ["Period", "Price_GBP_MWh", "Net_Power_MW", "SoC_MWh", "Net_Profit_GBP"]
        ]
        .head(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
