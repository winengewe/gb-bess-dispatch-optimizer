import numpy as np
import pandas as pd


def generate_synthetic_gb_prices(periods: int = 48, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic 30-min settlement period GB wholesale prices with morning and evening peaks.
    """
    np.random.seed(seed)
    t = np.arange(periods)

    # Base load profile (£/MWh)
    base = 65.0
    overnight_dip = -25.0 * np.exp(-((t - 6) ** 2) / 8.0)
    morning_peak = 45.0 * np.exp(-((t - 16) ** 2) / 12.0)
    evening_peak = 75.0 * np.exp(-((t - 36) ** 2) / 10.0)
    noise = np.random.normal(0, 4.0, periods)

    prices = base + overnight_dip + morning_peak + evening_peak + noise

    df = pd.DataFrame(
        {
            "SettlementPeriod": t + 1,
            "MarketIndexPrice": np.round(prices, 2),
        }
    )
    return df
