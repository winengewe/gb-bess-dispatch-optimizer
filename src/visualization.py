import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def plot_bess_dispatch(
    df_dispatch: pd.DataFrame,
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Generates a two-panel dispatch visualization:
    1. Settlement Period Wholesale Prices (£/MWh) alongside Charge & Discharge Power (MW).
    2. Battery State of Charge (SoC in MWh) trajectory over 48 settlement periods.

    Parameters:
        df_dispatch (pd.DataFrame): Output DataFrame from optimizer containing 'SettlementPeriod', 
                                    'Price_GBP_MWh', 'Charge_MW', 'Discharge_MW', and 'SoC_MWh'.
        save_path (str, optional): File path to save output image (e.g., 'docs/dispatch_plot.png').
        show_plot (bool): Whether to display the figure via plt.show().

    Returns:
        Tuple[plt.Figure, np.ndarray]: Matplotlib figure and axes references.
    """
    # Visual style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    sp = df_dispatch["SettlementPeriod"].values
    prices = df_dispatch["Price_GBP_MWh"].values
    charge = df_dispatch["Charge_MW"].values
    discharge = df_dispatch["Discharge_MW"].values
    soc = df_dispatch["SoC_MWh"].values

    # --- Subplot 1: Price Curve & Power Dispatch Bars ---
    # Left Y-Axis: Wholesale Market Price Curve
    color_price = "#1f77b4"  # Royal Blue
    ax1.set_ylabel("Wholesale Price (£/MWh)", color=color_price, fontsize=11, fontweight="bold")
    ax1.plot(sp, prices, color=color_price, linewidth=2.5, label="Wholesale Price (£/MWh)", zorder=3)
    ax1.tick_params(axis="y", labelcolor=color_price)
    ax1.set_title("BESS Optimal Arbitrage Dispatch Schedule (50MW / 100MWh)", fontsize=14, pad=12, fontweight="bold")

    # Right Y-Axis: Charge / Discharge Power Bars
    ax2 = ax1.twinx()
    color_ch = "#d62728"   # Red (Charging / Cost)
    color_dis = "#2ca02c"  # Green (Discharging / Revenue)

    ax2.set_ylabel("Power Dispatch (MW)", fontsize=11, fontweight="bold")
    ax2.bar(sp - 0.15, -charge, width=0.4, color=color_ch, alpha=0.6, label="Charge (MW)", zorder=2)
    ax2.bar(sp + 0.15, discharge, width=0.4, color=color_dis, alpha=0.6, label="Discharge (MW)", zorder=2)
    ax2.set_ylim(-65, 65)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Combine legends across twinned axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    # --- Subplot 2: State of Charge (SoC) Trajectory ---
    color_soc = "#9467bd"  # Purple
    ax3.plot(sp, soc, color=color_soc, linewidth=2.5, label="State of Charge (SoC)")
    ax3.fill_between(sp, 0, soc, color=color_soc, alpha=0.2)
    ax3.set_xlabel("Settlement Period (1 - 48)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("SoC (MWh)", color=color_soc, fontsize=11, fontweight="bold")
    ax3.set_ylim(0, max(soc.max() * 1.15, 100))
    ax3.set_xticks(np.arange(1, 49, 2))
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()

    # Save figure asset if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Dispatch plot successfully saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig, (ax1, ax3)


if __name__ == "__main__":
    from src.data_ingestion import generate_synthetic_gb_prices
    from src.optimizer import BESSModularOptimizer

    # Sanity check run
    df_data = generate_synthetic_gb_prices(periods=48)
    prices = df_data["MarketIndexPrice"].values

    optimizer = BESSModularOptimizer()
    df_dispatch = optimizer.optimize_dispatch(prices)

    fig, _ = plot_bess_dispatch(df_dispatch, save_path="docs/dispatch_plot.png")
