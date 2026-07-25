import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Constructs time-series lag, cyclical calendar, and domain features
    for Great Britain 30-minute settlement period price forecasting.
    """

    def __init__(
        self, price_col: str = "MarketIndexPrice", sp_col: str = "SettlementPeriod"
    ):
        self.price_col = price_col
        self.sp_col = sp_col

    def create_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Engineers feature set from raw time-series DataFrame.
        Expected columns: [SettlementPeriod, MarketIndexPrice, (optional) WindForecast, DemandForecast]
        """
        data = df.copy()

        # 1. Cyclical Time Encodings (48 Settlement Periods / Day)
        if self.sp_col in data.columns:
            data["sp_sin"] = np.sin(2 * np.pi * data[self.sp_col] / 48.0)
            data["sp_cos"] = np.cos(2 * np.pi * data[self.sp_col] / 48.0)

        # Day of week cyclical encoding (0-6)
        if "Timestamp" in data.columns or isinstance(data.index, pd.DatetimeIndex):
            dt_series = data["Timestamp"] if "Timestamp" in data.columns else data.index
            day_of_week = dt_series.dt.dayofweek
            data["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
            data["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # 2. Historical Price Lags (48 periods = 24 Hours)
        if self.price_col in data.columns:
            data["price_lag_48"] = data[self.price_col].shift(48)  # Same time yesterday
            data["price_lag_96"] = data[self.price_col].shift(
                96
            )  # Same time 2 days ago
            data["price_lag_336"] = data[self.price_col].shift(
                336
            )  # Same time last week

            # Rolling Price Statistics over past 24 hours (48 periods)
            data["price_roll_mean_24h"] = (
                data[self.price_col].shift(48).rolling(window=48).mean()
            )
            data["price_roll_std_24h"] = (
                data[self.price_col].shift(48).rolling(window=48).std()
            )

        # 3. Domain Features: Grid Load & Wind Generation Interactivity
        if "WindForecast" in data.columns and "DemandForecast" in data.columns:
            # Net Load = National Load - Variable Renewable Wind Output
            data["net_demand"] = data["DemandForecast"] - data["WindForecast"]
            # Wind Penetration Ratio
            data["wind_ratio"] = data["WindForecast"] / (data["DemandForecast"] + 1e-5)

            # Wind/Demand Lags
            data["wind_lag_48"] = data["WindForecast"].shift(48)
            data["demand_lag_48"] = data["DemandForecast"].shift(48)

        if drop_na:
            data = data.dropna().reset_index(drop=True)

        return data


def get_feature_columns(
    df: pd.DataFrame, target_col: str = "MarketIndexPrice"
) -> list[str]:
    """
    Returns list of feature column names excluding target and meta-columns.
    """
    exclude_cols = [target_col, "Timestamp", "SettlementPeriod", "Date"]
    return [col for col in df.columns if col not in exclude_cols]
