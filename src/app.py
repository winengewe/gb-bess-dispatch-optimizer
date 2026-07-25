import streamlit as st
import matplotlib.pyplot as plt

from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer
from src.visualization import plot_bess_dispatch

# Page Setup
st.set_page_config(page_title="GB BESS Optimizer", layout="wide", page_icon="⚡")
st.title("⚡ GB BESS Price Forecasting & Dispatch Optimizer")

# Sidebar Parameters
st.sidebar.header("⚙️ Asset & Market Settings")
power_mw = st.sidebar.slider("Power Rating (MW)", 10.0, 100.0, 50.0, 5.0)
capacity_mwh = st.sidebar.slider("Energy Capacity (MWh)", 20.0, 200.0, 100.0, 10.0)
rte = st.sidebar.slider("Round-Trip Efficiency (%)", 70.0, 100.0, 88.0, 1.0) / 100.0
deg_cost = st.sidebar.slider("Degradation Penalty (£/MWh)", 0.0, 30.0, 12.50, 0.50)

config = {
    "asset": {
        "power_mw": power_mw,
        "capacity_mwh": capacity_mwh,
        "round_trip_efficiency": rte,
        "degradation_cost_gbp_per_mwh": deg_cost,
        "initial_soc_pct": 0.50,
    },
    "market": {"time_step_hours": 0.5, "settlement_periods": 48},
}

# Pipeline Execution
with st.spinner("Running ML price forecaster & PuLP solver..."):
    df_raw = generate_synthetic_gb_prices(periods=48)
    actual_prices = df_raw["MarketIndexPrice"].tolist()

    fe = FeatureEngineer(price_col="MarketIndexPrice", sp_col="SettlementPeriod")
    df_featured = fe.create_features(df_raw, drop_na=False)
    feature_cols = get_feature_columns(df_featured, target_col="MarketIndexPrice")

    forecaster = PriceForecaster()
    forecaster.train(df_featured[feature_cols], df_featured["MarketIndexPrice"])
    predicted_prices = forecaster.predict(df_featured[feature_cols]).tolist()

    optimizer = BESSModularOptimizer(config)
    dispatch_df, net_obj_value = optimizer.solve_day_ahead_dispatch(predicted_prices)
    _, perfect_foresight_val = optimizer.solve_day_ahead_dispatch(actual_prices)

# KPI Summary
capture_rate = (
    (net_obj_value / perfect_foresight_val * 100) if perfect_foresight_val > 0 else 0.0
)
efc = (dispatch_df["Discharge_MW"] * 0.5).sum() / capacity_mwh

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Net Revenue", f"£{net_obj_value:,.2f}")
c2.metric("Perfect Foresight Rev", f"£{perfect_foresight_val:,.2f}")
c3.metric("Financial Capture Rate", f"{capture_rate:.1f}%")
c4.metric("Daily Cycles", f"{efc:.2f} EFC")

# Visualizations & Table
st.subheader("📈 Day-Ahead Dispatch Profile & State of Charge")
fig = plot_bess_dispatch(dispatch_df, capacity_mwh)
st.pyplot(fig if fig else plt.gcf(), use_container_width=True)

with st.expander("📋 View Raw Dispatch Schedule"):
    st.dataframe(dispatch_df, use_container_width=True)
