import os
import yaml
import pandas as pd

from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer
from src.visualization import plot_bess_dispatch


def main():
    print("⚡ Launching GB BESS Price Forecasting & Dispatch Optimizer...\n")

    # 1. Load Central Configuration
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Ingest Historical Training Data & Day-Ahead Target Data
    print("📥 [1/4] Ingesting market data...")
    target_periods = config["market"]["settlement_periods"]  # 48 periods
    train_periods = 1440  # 30-day historical horizon for ML training

    df_train_raw = generate_synthetic_gb_prices(periods=train_periods, random_seed=42)
    df_target_raw = generate_synthetic_gb_prices(periods=target_periods, random_seed=101)

    # 3. Feature Engineering
    print("🛠️  [2/4] Engineering time-series features...")
    fe = FeatureEngineer()
    df_train_feat = fe.transform(df_train_raw)
    df_target_feat = fe.transform(df_target_raw)

    feature_cols = get_feature_columns()
    X_train, y_train = df_train_feat[feature_cols], df_train_feat["MarketIndexPrice"]
    X_target = df_target_feat[feature_cols]
    actual_prices = df_target_raw["MarketIndexPrice"].values

    # 4. Train XGBoost Forecaster & Predict Day-Ahead Prices
    print("🤖 [3/4] Training XGBoost price forecaster...")
    forecaster = PriceForecaster()
    forecaster.fit(X_train, y_train)
    predicted_prices = forecaster.predict(X_target)

    # 5. Solve Dual-Pass Linear Program Benchmark (Model vs. Perfect Foresight)
    print("🧮 [4/4] Solving linear program & dual-pass benchmark...")
    optimizer = BESSModularOptimizer(
        power_capacity_mw=config["asset"]["power_mw"],
        energy_capacity_mwh=config["asset"]["capacity_mwh"],
        round_trip_efficiency=config["asset"]["round_trip_efficiency"],
        degradation_cost_per_mwh=config["asset"]["degradation_cost_gbp_mwh"],
        dt_hours=config["market"]["time_step_hours"]
    )

    benchmark = optimizer.run_dual_pass_benchmark(
        forecasted_prices=predicted_prices,
        actual_prices=actual_prices
    )
    dispatch_df = benchmark["model_dispatch_df"]

    # 6. Generate & Save High-Resolution Dispatch Chart
    print("📈 Generating publication-grade dispatch visualization...")
    plot_bess_dispatch(
        df_dispatch=dispatch_df,
        save_path="docs/dispatch_plot.png"
    )

    # 7. Print Executive Summary Report
    print("\n" + "=" * 55)
    print(" 📊 DAILY DISPATCH OPTIMIZATION SUMMARY REPORT")
    print("=" * 55)
    print(f" Asset Rating         : {config['asset']['power_mw']} MW / {config['asset']['capacity_mwh']} MWh")
    print(f" Round-Trip Efficiency: {config['asset']['round_trip_efficiency'] * 100:.1f}%")
    print(f" Gross Arbitrage Rev  : £{benchmark['gross_revenue_gbp']:,.2f}")
    print(f" Degradation Penalty  : £{benchmark['degradation_cost_gbp']:,.2f}")
    print(f" Model Net Revenue    : £{benchmark['model_net_revenue_gbp']:,.2f}")
    print(f" Perfect Foresight Rev: £{benchmark['perfect_foresight_revenue_gbp']:,.2f}")
    print(f" Financial Capture    : {benchmark['capture_rate_pct']:.1f}%")
    print(f" Equivalent Cycles    : {benchmark['equivalent_full_cycles']:.2f} EFC/day")
    print("=" * 55)

    # 8. Sample Output Preview
    print("\nSample Dispatch Output (First 6 Settlement Periods):")
    preview_cols = ["SettlementPeriod", "Price_GBP_MWh", "NetPower_MW", "SoC_MWh", "NetRevenue_GBP"]
    available_preview = [c for c in preview_cols if c in dispatch_df.columns]
    print(dispatch_df[available_preview].head(6).to_string(index=False))

    print("\n✅ Pipeline complete! Chart saved to 'docs/dispatch_plot.png'.")


if __name__ == "__main__":
    main()
