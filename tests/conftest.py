# tests/conftest.py
import pytest
import yaml
import pandas as pd


@pytest.fixture
def sample_config():
    """Provides standard project configuration dictionary."""
    return {
        "asset": {
            "power_mw": 50.0,
            "capacity_mwh": 100.0,
            "round_trip_efficiency": 0.88,
            "degradation_cost_gbp_mwh": 12.50,
            "initial_soc_mwh": 50.0,
        },
        "market": {"time_step_hours": 0.5},
    }


@pytest.fixture
def valid_dispatch_df():
    """Provides a valid 2-period dispatch dataframe."""
    return pd.DataFrame(
        {
            "SettlementPeriod": [1, 2],
            "Price_GBP_MWh": [30.0, 90.0],
            "NetPower_MW": [-50.0, 50.0],
            "SoC_MWh": [73.45, 50.0],
        }
    )
