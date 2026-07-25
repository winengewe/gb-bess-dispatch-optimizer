# tests/test_validation.py
import pytest
import pandas as pd
from src.validation import evaluate_forecaster, validate_dispatch_schedule


def test_evaluate_forecaster_accuracy():
    """Tests ML metric calculations."""
    y_true = [100.0, 200.0]
    y_pred = [110.0, 190.0]

    metrics = evaluate_forecaster(y_true, y_pred)

    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0
    assert "WAPE_pct" in metrics


def test_validate_dispatch_schedule_success(valid_dispatch_df, sample_config):
    """Ensures valid dispatch schedule passes without errors."""
    assert validate_dispatch_schedule(valid_dispatch_df, sample_config) is True


def test_validate_dispatch_schedule_power_overflow(sample_config):
    """Ensures exceeding battery rated power (MW) triggers AssertionError."""
    invalid_df = pd.DataFrame(
        {
            "SettlementPeriod": [1],
            "Price_GBP_MWh": [50.0],
            "NetPower_MW": [100.0],  # ❌ Exceeds 50 MW max power limit
            "SoC_MWh": [50.0],
        }
    )

    with pytest.raises(AssertionError, match="Discharge power bounds violated"):
        validate_dispatch_schedule(invalid_df, sample_config)


def test_validate_dispatch_schedule_soc_overflow(sample_config):
    """Ensures exceeding storage capacity (MWh) triggers AssertionError."""
    invalid_df = pd.DataFrame(
        {
            "SettlementPeriod": [1],
            "Price_GBP_MWh": [50.0],
            "NetPower_MW": [0.0],
            "SoC_MWh": [150.0],  # ❌ Exceeds 100 MWh capacity limit
        }
    )

    with pytest.raises(AssertionError, match="Storage capacity bounds violated"):
        validate_dispatch_schedule(invalid_df, sample_config)
