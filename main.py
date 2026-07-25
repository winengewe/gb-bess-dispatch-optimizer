import yaml
from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer
from src.visualization import plot_bess_dispatch


def main():
    print("⚡ Launching GB BESS Price Forecasting & Dispatch Optimizer...\n")

    # 1. Load Configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Data Ingestion (Market Data)
    print("📥 Ingesting market data...")
    df_raw = generate_synthetic_gb_prices(
        periods=config["market"]["settlement_periods"]
    )

    # 3. Feature Engineering
    print("🛠️  Engineering time-series features...")
    fe = FeatureEngineer(price_col="MarketIndexPrice", sp_col="SettlementPeriod")
    df_featured = fe.create_features(df_raw, drop_na=False)

    # 4. Train Forecasting Model & Predict Day-Ahead Prices
    print("🤖 Training XGBoost price forecaster...")
    feature_cols = get_feature_columns(df_featured, target_col="MarketIndexPrice")
    X = df_featured[feature_cols]
    y = df_featured["MarketIndexPrice"]

    forecaster = PriceForecaster()
    forecaster.train(X, y)
    predicted_prices = forecaster.predict(X).tolist()

    # 5. Linear Programming Dispatch Optimization
    print("🧮 Solving linear program for optimal asset schedule...")
    optimizer = BESSModularOptimizer(config)
    dispatch_df, net_obj_value = optimizer.solve_day_ahead_dispatch(predicted_prices)

    # 6. Generate & Save Dispatch Chart
    print("📈 Generating dispatch visualization chart...")
    plot_bess_dispatch(
        dispatch_df=dispatch_df,
        capacity_mwh=config["asset"]["capacity_mwh"],
        save_path="docs/dispatch_plot.png",
    )

    # 7. Calculate Key Performance Indicators (KPIs)
    total_gross_rev = dispatch_df["Gross_Revenue_GBP"].sum()
    total_deg_cost = dispatch_df["Degradation_Cost_GBP"].sum()
    total_discharged_mwh = (
        dispatch_df["Discharge_MW"] * config["market"]["time_step_hours"]
    ).sum()
    efc = total_discharged_mwh / config["asset"]["capacity_mwh"]

    # 8. Print Executive Summary Report
    print("\n" + "=" * 55)
    print(" 📊 DAILY DISPATCH OPTIMIZATION SUMMARY REPORT")
    print("=" * 55)
    print(
        f" Asset Rating         : {config['asset']['power_mw']} MW / {config['asset']['capacity_mwh']} MWh"
    )
    print(
        f" Round-Trip Efficiency: {config['asset']['round_trip_efficiency'] * 100:.1f}%"
    )
    print(f" Gross Arbitrage Rev  : £{total_gross_rev:,.2f}")
    print(f" Degradation Penalty  : £{total_deg_cost:,.2f}")
    print(f" Net Daily Revenue    : £{net_obj_value:,.2f}")
    print(f" Equivalent Cycles    : {efc:.2f} EFC/day")
    print("=" * 55)

    print("\nSample Dispatch Output (First 6 Settlement Periods):")
    print(
        dispatch_df[
            ["Period", "Price_GBP_MWh", "Net_Power_MW", "SoC_MWh", "Net_Profit_GBP"]
        ]
        .head(6)
        .to_string(index=False)
    )
    print("\n✅ Pipeline complete! Chart saved to 'docs/dispatch_plot.png'.")


if __name__ == "__main__":
    main()
