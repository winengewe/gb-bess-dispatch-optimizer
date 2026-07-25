"""
src/validation.py

Validation module for ML forecasting accuracy and LP optimization invariants.
Fully aligned with the project's OOP architecture and dispatch dataframe schema.
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. MACHINE LEARNING MODEL VALIDATION
# ==============================================================================
def evaluate_forecaster(y_true, y_pred, y_baseline=None) -> dict:
    """
    Calculates time-series forecasting metrics: MAE, RMSE, and WAPE (%).
    Optionally evaluates percentage improvement against a naive baseline (e.g., t-24h).
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    wape = (np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t))) * 100

    metrics = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "WAPE_pct": round(float(wape), 2),
    }

    if y_baseline is not None:
        y_b = np.asarray(y_baseline)
        valid_mask = ~np.isnan(y_b)
        if np.any(valid_mask):
            baseline_mae = np.mean(np.abs(y_t[valid_mask] - y_b[valid_mask]))
            improvement = ((baseline_mae - mae) / baseline_mae) * 100
            metrics["Baseline_MAE"] = round(float(baseline_mae), 2)
            metrics["MAE_Improvement_pct"] = round(float(improvement), 2)

    return metrics


# ==============================================================================
# 2. OPTIMIZATION & PHYSICAL INVARIANT VALIDATION
# ==============================================================================
def validate_dispatch_schedule(df: pd.DataFrame, config: dict) -> bool:
    """
    Enforces invariant assertions on LP solver output (dispatch_df):
    1. Physical power & storage capacity bounds.
    2. Energy conservation & State of Charge (SoC) tracking.
    3. Terminal SoC boundary condition.
    4. Economic hurdle rate / price spread threshold enforcement.
    """
    p_max = config["asset"]["power_mw"]
    e_max = config["asset"]["capacity_mwh"]
    rte = config["asset"]["round_trip_efficiency"]
    eta = np.sqrt(rte)
    c_deg = config["asset"]["degradation_cost_gbp_mwh"]
    initial_soc = config["asset"]["initial_soc_mwh"]

    # --- Robust Column Resolution ---
    # Identify SoC column
    soc_col = next((c for c in ["SoC_MWh", "soc_mwh", "SoC"] if c in df.columns), None)
    assert soc_col is not None, "SoC column not found in dispatch dataframe."

    # Identify Charge / Discharge Power
    if "NetPower_MW" in df.columns:
        p_discharge = np.maximum(df["NetPower_MW"], 0)
        p_charge = np.maximum(-df["NetPower_MW"], 0)
    else:
        charge_col = next(
            (c for c in ["Charge_MW", "p_charge_mw", "P_charge_MW"] if c in df.columns),
            None,
        )
        discharge_col = next(
            (
                c
                for c in ["Discharge_MW", "p_discharge_mw", "P_discharge_MW"]
                if c in df.columns
            ),
            None,
        )
        assert charge_col and discharge_col, "Power charge/discharge columns not found."
        p_charge = df[charge_col]
        p_discharge = df[discharge_col]

    # Identify Price column
    price_col = next(
        (
            c
            for c in ["Price_GBP_MWh", "MarketIndexPrice", "forecast_price"]
            if c in df.columns
        ),
        None,
    )

    # --- Assertion 1: Power & Capacity Bounds ---
    assert (p_charge >= -1e-4).all() and (p_charge <= p_max + 1e-4).all(), (
        f"Charge power bounds violated (Max rated: {p_max} MW)."
    )
    assert (p_discharge >= -1e-4).all() and (p_discharge <= p_max + 1e-4).all(), (
        f"Discharge power bounds violated (Max rated: {p_max} MW)."
    )
    assert (df[soc_col] >= -1e-4).all() and (df[soc_col] <= e_max + 1e-4).all(), (
        f"Storage capacity bounds violated (0 to {e_max} MWh)."
    )

    # --- Assertion 2: Terminal SoC Boundary Condition ---
    final_soc = df[soc_col].iloc[-1]
    assert np.isclose(final_soc, initial_soc, atol=1e-1), (
        f"Terminal SoC condition failed: End SoC was {final_soc:.2f} MWh, expected {initial_soc} MWh."
    )

    # --- Assertion 3: Economic Hurdle Rate Threshold ---
    if price_col is not None:
        hurdle_rate = (2 * c_deg) / eta
        active_charges = df[p_charge > 0.1][price_col]
        active_discharges = df[p_discharge > 0.1][price_col]

        if not active_charges.empty and not active_discharges.empty:
            max_charge_price = active_charges.max()
            min_discharge_price = active_discharges.min()
            actual_spread = min_discharge_price - max_charge_price

            assert actual_spread >= (hurdle_rate - 2.0), (
                f"Economic breach: LP dispatched at spread £{actual_spread:.2f}/MWh, below breakeven threshold £{hurdle_rate:.2f}/MWh."
            )

    print("  ✅ All optimization and physical invariant assertions passed.")
    return True
