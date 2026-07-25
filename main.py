import yaml
from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer


def main():
    print("⚡ Running GB BESS Price Forecasting & Dispatch Pipeline...\n")

    # 1. Load Configurations
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Data Ingestion (Simulating 10 days of 30-min settlement data = 480 periods)
    print("[1/4] Ingesting Market Data...")
    df_raw = generate_synthetic_gb_prices(periods=480)

    # 3. Feature Engineering
    print("[2/4] Engineering Features (Lags, Cyclical Encodings)...")
    fe = FeatureEngineer(price_col="MarketIndexPrice", sp_col="SettlementPeriod")
    df_featured = fe.create_features(df_raw, drop_na=True)

    # 4. Model Training & Forecasting
    print("[3/4] Training XGBoost Forecaster & Generating Price Predictions...")
    feature_cols = get_feature_columns(df_featured, target_col="MarketIndexPrice")
    X = df_featured[feature_cols]
    y = df_featured["MarketIndexPrice"]

    forecaster = PriceForecaster()
    forecaster.train(X.iloc[:-48], y.iloc[:-48])  # Train on past days
    predicted_prices = forecaster.predict(X.tail(48))  # Predict final 24 hours

    # 5. Asset Dispatch Optimization
    print("[4/4] Solving LP Battery Dispatch Schedule via PuLP...")
    optimizer = BESSModularOptimizer(config)
    results_df, net_profit = optimizer.solve_day_ahead_dispatch(predicted_prices)

    # 6. Report Key Performance Indicators (KPIs)
    total_revenue = results_df["Gross_Revenue_GBP"].sum()
    total_degradation = results_df["Degradation_Cost_GBP"].sum()
    total_discharged_mwh = (
        results_df["Discharge_MW"] * config["market"]["time_step_hours"]
    ).sum()
    efc = total_discharged_mwh / config["asset"]["capacity_mwh"]

    print("\n" + "=" * 55)
    print(" 📊 DAILY BESS DISPATCH & FINANCIAL SUMMARY")
    print("=" * 55)
    print(
        f" Asset Specs          : {config['asset']['power_mw']} MW / {config['asset']['capacity_mwh']} MWh"
    )
    print(f" Gross Arbitrage Rev  : £{total_revenue:,.2f}")
    print(f" Cycle Wear Cost      : £{total_degradation:,.2f}")
    print(f" Net Daily Profit     : £{net_profit:,.2f}")
    print(f" Daily Throughput     : {total_discharged_mwh:.1f} MWh ({efc:.2f} EFC/day)")
    print("=" * 55)


if __name__ == "__main__":
    main()
