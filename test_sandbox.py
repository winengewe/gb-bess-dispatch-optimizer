"""
test_sandbox.py - Custom script for interactive testing & edge-case validation
"""

import yaml
import pandas as pd
from src.validation import evaluate_forecaster, validate_dispatch_schedule

# 1. Load Central Configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("🔍 --- TEST 1: Forecaster Metrics ---")
y_actual = [50.0, 65.0, 80.0, 110.0, 45.0]
y_forecast = [52.0, 61.0, 78.0, 105.0, 48.0]
y_naive = [48.0, 50.0, 70.0, 90.0, 40.0]

ml_metrics = evaluate_forecaster(y_actual, y_forecast, y_baseline=y_naive)
print("ML Metrics Output:", ml_metrics)


print("\n🔍 --- TEST 2: Valid Dispatch Invariants ---")
valid_df = pd.DataFrame(
    {
        "SettlementPeriod": [1, 2],
        "Price_GBP_MWh": [40.0, 80.0],
        "NetPower_MW": [-50.0, 50.0],  # -50 MW charge, +50 MW discharge
        "SoC_MWh": [73.45, 50.0],  # Valid energy tracking & ends at 50 MWh
    }
)

# Should output: ✅ All optimization and physical invariant assertions passed.
validate_dispatch_schedule(valid_df, config)


print("\n🔍 --- TEST 3: Edge Case Violation (Capacity Overflow) ---")
invalid_df = pd.DataFrame(
    {
        "SettlementPeriod": [1],
        "Price_GBP_MWh": [50.0],
        "NetPower_MW": [0.0],
        "SoC_MWh": [150.0],  # ❌ Exceeds max 100 MWh capacity!
    }
)

try:
    validate_dispatch_schedule(invalid_df, config)
except AssertionError as e:
    print(f"  ❌ Assertion Caught Successfully: {e}")
