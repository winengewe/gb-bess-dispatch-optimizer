import numpy as np
import pandas as pd
import streamlit as st

from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer
from src.visualization import plot_bess_dispatch

# --- Page Config ---
st.set_page_config(
    page_title="GB BESS Dispatch Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title & Description ---
st.title("⚡ GB BESS Price Forecasting & Dispatch Optimizer")
st.markdown(
    """
    An end-to-end quantitative web application for a **Battery Energy Storage System (BESS)** 
    operating in Great Britain's wholesale electricity market (48 settlement periods/day). 
    
    Adjust asset parameters in the sidebar to simulate live XGBoost price forecasting, PuLP linear 
    programming dispatch, and **Financial Capture Rate %** benchmarking against a Perfect Foresight baseline.
    """
)

st.divider()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ BESS Asset Parameters")

power_mw = st.sidebar.number_input(
    "Power Capacity (MW)", 
    min_value=1.0, 
    max_value=500.0, 
    value=50.0, 
    step=5.0
)

capacity_mwh = st.sidebar.number_input(
    "Energy Capacity (MWh)", 
    min_value=1.0, 
    max_value=1000.0, 
    value=100.0, 
    step=10.0
)

rte_pct = st.sidebar.slider(
    "Round-Trip Efficiency (%)", 
    min_value=50.0, 
    max_value=100.0, 
    value=88.0, 
    step=1.0
) / 100.0

deg_cost = st.sidebar.number_input(
    "Degradation Penalty (£/MWh)", 
    min_value=0.0, 
    max_value=50.0, 
    value=12.50, 
    step=0.5
)

st.sidebar.divider()
st.sidebar.header("🎲 Market Environment")

base_price = st.sidebar.slider(
    "Base Electricity Price (£/MWh)", 
    min_value=20.0, 
    max_value=150.0, 
    value=65.0, 
    step=5.0
)

random_seed = st.sidebar.number_input(
    "Simulation Seed", 
    min_value=1, 
    max_value=999, 
    value=42, 
    step=1
)

# --- Pipeline Execution ---
@st.cache_data(show_spinner=False)
def run_pipeline(
    power_mw: float, 
    capacity_mwh: float, 
    rte_pct: float, 
    deg_cost: float, 
    base_price: float, 
    seed: int
):
    # 1. Ingest historical & current settlement period data
    fe = FeatureEngineer()
    hist_raw = generate_synthetic_gb_prices(periods=1440, base_price=base_price, random_seed=seed)
    hist_features = fe.transform(hist_raw)
    
    current_raw = generate_synthetic_gb_prices(periods=48, base_price=base_price, random_seed=seed + 1)
    current_features = fe.transform(current_raw)
    
    feature_cols = get_feature_columns()
    
    # 2. Train XGBoost Forecaster & Predict Day-Ahead Prices
    forecaster = PriceForecaster()
    forecaster.fit(hist_features[feature_cols], hist_features["MarketIndexPrice"])
    
    y_forecast = forecaster.predict(current_features[feature_cols])
    y_actual = current_features["MarketIndexPrice"].values

    # 3. Formulate & Solve PuLP Linear Program
    optimizer = BESSModularOptimizer(
        power_capacity_mw=power_mw,
        energy_capacity_mwh=capacity_mwh,
        round_trip_efficiency=rte_pct,
        degradation_cost_per_mwh=deg_cost
    )
    
    benchmark = optimizer.run_dual_pass_benchmark(
        forecasted_prices=y_forecast,
        actual_prices=y_actual
    )
    
    return benchmark

with st.spinner("Running XGBoost forecast and PuLP dispatch optimization..."):
    benchmark_results = run_pipeline(
        power_mw, capacity_mwh, rte_pct, deg_cost, base_price, random_seed
    )

df_dispatch = benchmark_results["model_dispatch_df"]

# --- KPI Metric Cards ---
st.subheader("📊 Dispatch Financial Summary")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Gross Revenue", f"£{benchmark_results['gross_revenue_gbp']:,.2f}")
col2.metric("Degradation Penalty", f"£{benchmark_results['degradation_cost_gbp']:,.2f}")
col3.metric("Model Net Revenue", f"£{benchmark_results['model_net_revenue_gbp']:,.2f}")
col4.metric(
    "Capture Rate", 
    f"{benchmark_results['capture_rate_pct']:.1f}%", 
    delta=f"{(benchmark_results['capture_rate_pct'] - 100.0):.1f}% vs Perfect",
    help="Model Revenue achieved relative to Perfect Foresight baseline."
)
col5.metric("Daily Cycles", f"{benchmark_results['equivalent_full_cycles']:.2f} EFC")

st.divider()

# --- Interactive Visualizations ---
st.subheader("📈 24-Hour Settlement Period Dispatch Schedule")
fig, _ = plot_bess_dispatch(df_dispatch)
st.pyplot(fig)

# --- Data Table & CSV Export ---
with st.expander("🔍 View Detailed 48-Period Settlement Schedule"):
    st.dataframe(
        df_dispatch.style.format({
            "Price_GBP_MWh": "£{:.2f}",
            "Charge_MW": "{:.2f} MW",
            "Discharge_MW": "{:.2f} MW",
            "NetPower_MW": "{:.2f} MW",
            "SoC_MWh": "{:.2f} MWh",
            "GrossRevenue_GBP": "£{:.2f}",
            "DegradationCost_GBP": "£{:.2f}",
            "NetRevenue_GBP": "£{:.2f}"
        }),
        use_container_width=True
    )
    
    csv_bytes = df_dispatch.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Dispatch Schedule CSV",
        data=csv_bytes,
        file_name="bess_dispatch_schedule.csv",
        mime="text/csv"
    )
