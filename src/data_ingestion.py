import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class ElexonAPIClient:
    """
    Client for ingesting live Great Britain electricity market data 
    from the Elexon Insights API (BMRS).
    """

    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEXON_API_KEY")

    def fetch_market_index_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch half-hourly Market Index Prices (B1770 stream).
        
        Parameters:
            start_date (str): Format 'YYYY-MM-DD'
            end_date (str): Format 'YYYY-MM-DD'
        """
        endpoint = f"{self.BASE_URL}/datasets/B1770"
        params = {
            "publishDateTimeFrom": f"{start_date}T00:00:00Z",
            "publishDateTimeTo": f"{end_date}T23:59:59Z",
            "format": "json"
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            records = data.get("data", [])
            if records:
                df = pd.DataFrame(records)
                df["SettlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date
                df["SettlementPeriod"] = df["settlementPeriod"].astype(int)
                df["MarketIndexPrice"] = df["price"].astype(float)
                return df[["SettlementDate", "SettlementPeriod", "MarketIndexPrice"]].drop_duplicates()
        except Exception as e:
            print(f"⚠️ Elexon API request failed: {e}. Falling back to synthetic generator.")
            
        return pd.DataFrame()

    def fetch_wind_forecast(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Day-Ahead Wind Generation Forecasts (B1440 stream)."""
        endpoint = f"{self.BASE_URL}/datasets/B1440"
        params = {
            "publishDateTimeFrom": f"{start_date}T00:00:00Z",
            "publishDateTimeTo": f"{end_date}T23:59:59Z",
            "format": "json"
        }
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                df["SettlementPeriod"] = df["settlementPeriod"].astype(int)
                df["WindForecast_MW"] = df["quantity"].astype(float)
                return df[["settlementDate", "SettlementPeriod", "WindForecast_MW"]]
        except Exception:
            pass
        return pd.DataFrame()


def generate_synthetic_gb_prices(
    periods: int = 48,
    start_date: str = "2025-01-01",
    base_price: float = 65.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic 30-minute GB wholesale electricity market data.
    Simulates diurnal double-peak demand (08:00 & 18:00 peaks), night price troughs,
    renewable generation suppression, and random volatility spikes across settlement periods (1-48).
    """
    np.random.seed(random_seed)
    
    # 30-minute time grid
    dt_range = pd.date_range(start=start_date, periods=periods, freq="30min")
    settlement_periods = [(t.hour * 2 + t.minute // 30) + 1 for t in dt_range]
    
    # Diurnal shape (morning peak ~SP16 / 08:00, evening peak ~SP36 / 18:00, night trough ~SP8 / 04:00)
    sp = np.array(settlement_periods)
    morning_peak = 25.0 * np.exp(-((sp - 16) ** 2) / 18)
    evening_peak = 45.0 * np.exp(-((sp - 36) ** 2) / 24)
    night_trough = -20.0 * np.exp(-((sp - 8) ** 2) / 12)
    
    # Base curve + Gaussian noise
    noise = np.random.normal(0, 5.0, size=periods)
    price_curve = base_price + morning_peak + evening_peak + night_trough + noise
    
    # Random wholesale price spikes (5% probability)
    spikes = np.random.choice([0, 1], size=periods, p=[0.95, 0.05]) * np.random.uniform(30, 80, size=periods)
    final_prices = price_curve + spikes

    df = pd.DataFrame({
        "Timestamp": dt_range,
        "SettlementDate": dt_range.date,
        "SettlementPeriod": settlement_periods,
        "MarketIndexPrice": np.round(final_prices, 2),
        "WindForecast_MW": np.round(np.random.uniform(4000, 14000, size=periods), 1),
        "SystemDemand_MW": np.round(25000 + morning_peak * 300 + evening_peak * 400 + np.random.normal(0, 500, size=periods), 1)
    })
    
    return df


if __name__ == "__main__":
    # Test synthetic generation
    df_sample = generate_synthetic_gb_prices(periods=48)
    print("✅ Sample Data Ingested Successfully:")
    print(df_sample.head())
