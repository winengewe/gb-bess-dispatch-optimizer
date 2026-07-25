import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Import Core Production Package Modules
from src.data_ingestion import generate_synthetic_gb_prices
from src.features import FeatureEngineer, get_feature_columns
from src.model import PriceForecaster
from src.optimizer import BESSModularOptimizer
from src.visualization import plot_bess_dispatch

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="GB BESS Dispatch Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ Great Britain BESS Price Forecasting & Dispatch Optimizer")
st.markdown(
    "Interactive quantitative workbench for a **Battery Energy Storage System (BESS)** "
    "operating in GB's 48 half-hourly wholesale settlement market."
)

st.divider()

# ==============================================================================
# 2. LOAD YAML CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ==============================================================================
config_path = os.path.join("config", "config.yaml")

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "asset": {
            "power_mw": 50.0,
            "capacity_mwh": 100.0,
            "round_trip_efficiency": 0.88,
            "degradation_cost_gbp_mwh": 12.50,
        },
        "market": {"settlement_periods": 48, "time_step_hours": 0.5},
        "model": {"random_seed": 101},
        "paths": {"output_plot_path": "docs/dispatch_plot.png"},
    }

cfg_asset = config["asset"]
cfg_market = config["market"]
cfg_model = config["model"]

# ==============================================================================
# 3. SIDEBAR PARAMETER CONTROLS (SYNCED WITH YAML)
# ==============================================================================
st.sidebar.header("⚙️ Battery Specifications")

power_mw = st.sidebar.slider(
    "Rated Power (MW)",
    min_value=10.0,
    max_value=100.0,
    value=float(cfg_asset.get("power_mw", 50.0)),
    step=5.0,
)

capacity_mwh = st.sidebar.slider(
    "Storage Capacity (MWh)",
    min_value=20.0,
    max_value=300.0,
    value=float(cfg_asset.get("capacity_mwh", 100.0)),
    step=10.0,
)

rte = (
    st.sidebar.slider(
        "Round-Trip Efficiency (%)",
        min_value=70.0,
        max_value=98.0,
        value=float(cfg_asset.get("round_trip_efficiency", 0.88) * 100.0),
        step=1.0,
    )
    / 100.0
)

deg_cost = st.sidebar.number_input(
    "Degradation Cost (£/MWh)",
    min_value=0.0,
    max_value=50.0,
    value=float(cfg_asset.get("degradation_cost_gbp_mwh", 12.50)),
    step=0.50,
)

st.sidebar.divider()
st.sidebar.header("🎲 Market Simulation")
market_seed = st.sidebar.number_input(
    "Market Data Seed",
    min_value=1,
    max_value=999,
    value=int(cfg_model.get("random_seed", 101)),
    step=1,
)

# Hurdle Rate Calculation
eta_val = np.sqrt(rte)
hurdle_rate = (2 * deg_cost) / eta_val
st.sidebar.info(
    f"💡 **Breakeven Hurdle Rate:** ~£{hurdle_rate:.2f}/MWh price spread required to trigger dispatch."
)


# ==============================================================================
# 4. PIPELINE EXECUTION ENGINE (CACHED)
# ==============================================================================
@st.cache_data
def run_optimization_pipeline(seed, p_mw, c_mwh, rte_val, deg_val):
    # 1. Ingest Data
    df_train_raw = generate_synthetic_gb_prices(periods=1440, random_seed=42)
    df_target_raw = generate_synthetic_gb_prices(periods=48, random_seed=seed)

    # 2. Features
    fe = FeatureEngineer()
    df_train_feat = fe.transform(df_train_raw)
    df_target_feat = fe.transform(df_target_raw)

    feature_cols = get_feature_columns()
    X_train, y_train = df_train_feat[feature_cols], df_train_feat["MarketIndexPrice"]
    X_target = df_target_feat[feature_cols]
    actual_prices = df_target_raw["MarketIndexPrice"].values

    # 3. Model
    forecaster = PriceForecaster()
    forecaster.fit(X_train, y_train)
    predicted_prices = forecaster.predict(X_target)

    # 4. Optimization
    optimizer = BESSModularOptimizer(
        power_capacity_mw=p_mw,
        energy_capacity_mwh=c_mwh,
        round_trip_efficiency=rte_val,
        degradation_cost_per_mwh=deg_val,
        dt_hours=0.5,
    )

    benchmark = optimizer.run_dual_pass_benchmark(
        forecasted_prices=predicted_prices, actual_prices=actual_prices
    )

    return benchmark, predicted_prices, actual_prices


benchmark, predicted_prices, actual_prices = run_optimization_pipeline(
    market_seed, power_mw, capacity_mwh, rte, deg_cost
)

dispatch_df = benchmark["model_dispatch_df"]

# ==============================================================================
# 5. KPI METRIC CARDS
# ==============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Gross Revenue", f"£{benchmark['gross_revenue_gbp']:,.2f}")
col2.metric("Degradation Penalty", f"-£{benchmark['degradation_cost_gbp']:,.2f}")
col3.metric(
    "Model Net Revenue",
    f"£{benchmark['model_net_revenue_gbp']:,.2f}",
    delta=f"{benchmark['model_net_revenue_gbp'] - benchmark['perfect_foresight_revenue_gbp']:,.2f} vs Benchmark",
)
col4.metric("Financial Capture Rate", f"{benchmark['capture_rate_pct']:.1f}%")
col5.metric("Daily Utilization", f"{benchmark['equivalent_full_cycles']:.2f} EFC")

st.divider()

# ==============================================================================
# 6. VISUAL DISPATCH SCHEDULE PLOT
# ==============================================================================
st.subheader("📊 Optimal Arbitrage Dispatch Schedule")

fig, _ = plot_bess_dispatch(df_dispatch=dispatch_df, save_path=None)
st.pyplot(fig)

# ==============================================================================
# 7. DATA TABLE & EXPORT OPTIONS
# ==============================================================================
st.divider()

col_table, col_summary = st.columns([2, 1])

with col_table:
    st.subheader("📋 Half-Hourly Settlement Schedule")
    st.dataframe(dispatch_df, use_container_width=True, height=300)

with col_summary:
    st.subheader("📥 Export Results")
    st.markdown("Download the optimized schedule for further execution analysis:")

    csv_data = dispatch_df.to_csv(index=False)
    st.download_button(
        label="Download Schedule CSV",
        data=csv_data,
        file_name=f"bess_dispatch_seed{market_seed}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        f"**Config Source:** `{config_path}`\n\n"
        f"**Active Configuration:**\n"
        f"* Asset Rating: {power_mw}MW / {capacity_mwh}MWh\n"
        f"* Round-Trip Efficiency: {rte * 100:.1f}%\n"
        f"* Degradation Penalty: £{deg_cost:.2f}/MWh\n"
        f"* Net Profit: £{benchmark['model_net_revenue_gbp']:,.2f}"
    )
