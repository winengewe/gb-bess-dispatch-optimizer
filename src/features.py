import numpy as np
import pandas as pd
from typing import List


class FeatureEngineer:
    """
    Engineers time-series, cyclical Fourier, and domain-specific features 
    for GB wholesale electricity price forecasting across 48 settlement periods.
    """

    def __init__(self, lag_periods: List[int] = None, rolling_windows: List[int] = None):
        # Lags: 48 (24h ago), 96 (48h ago), 336 (7 days ago)
        self.lag_periods = lag_periods or [48, 96, 336]
        # Rolling windows: 24 (12h), 48 (24h)
        self.rolling_windows = rolling_windows or [24, 48]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms ingested market data into an engineered feature matrix.
        """
        data = df.copy()

        # Ensure correct chronological order
        if "Timestamp" in data.columns:
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
            data = data.sort_values("Timestamp").reset_index(drop=True)

        # 1. Cyclical Settlement Period Encodings (Fourier Transforms for 48 half-hour SPs)
        sp = data["SettlementPeriod"].astype(float)
        data["sp_sin"] = np.sin(2 * np.pi * sp / 48.0)
        data["sp_cos"] = np.cos(2 * np.pi * sp / 48.0)

        # Calendar Day/Weekend Encodings
        if "Timestamp" in data.columns:
            data["day_of_week"] = data["Timestamp"].dt.dayofweek
            data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
        else:
            data["day_of_week"] = 0
            data["is_weekend"] = 0

        # 2. Domain Ratios (Renewable Penetration / System Stress)
        if "WindForecast_MW" in data.columns and "SystemDemand_MW" in data.columns:
            data["wind_demand_ratio"] = data["WindForecast_MW"] / (data["SystemDemand_MW"] + 1e-5)
        else:
            data["wind_demand_ratio"] = 0.0

        # 3. Time-Series Price Lags
        if "MarketIndexPrice" in data.columns:
            for lag in self.lag_periods:
                data[f"price_lag_{lag}"] = data["MarketIndexPrice"].shift(lag)

            # 4. Rolling Price Statistics (Shifted by 1 to prevent target leakage)
            for window in self.rolling_windows:
                shifted_price = data["MarketIndexPrice"].shift(1)
                data[f"price_roll_mean_{window}"] = shifted_price.rolling(window=window, min_periods=1).mean()
                data[f"price_roll_std_{window}"] = shifted_price.rolling(window=window, min_periods=1).std()

        # Forward/Backward fill initial NaNs from lag/rolling shifts
        data = data.bfill().ffill().fillna(0)

        return data


def get_feature_columns() -> List[str]:
    """
    Returns the strict list of feature column names fed into XGBoost.
    """
    return [
        "SettlementPeriod",
        "sp_sin",
        "sp_cos",
        "day_of_week",
        "is_weekend",
        "wind_demand_ratio",
        "price_lag_48",
        "price_lag_96",
        "price_lag_336",
        "price_roll_mean_24",
        "price_roll_std_24",
        "price_roll_mean_48",
        "price_roll_std_48",
    ]


if __name__ == "__main__":
    from src.data_ingestion import generate_synthetic_gb_prices

    # Quick sanity test execution
    raw_df = generate_synthetic_gb_prices(periods=96)
    fe = FeatureEngineer()
    processed_df = fe.transform(raw_df)

    print("✅ Feature Pipeline Execution Successful:")
    print(f"Matrix Shape: {processed_df[get_feature_columns()].shape}")
    print(f"Feature Columns: {get_feature_columns()}")
