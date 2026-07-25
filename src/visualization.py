import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_bess_dispatch(
    dispatch_df: pd.DataFrame,
    capacity_mwh: float = 100.0,
    save_path: str = "docs/dispatch_plot.png",
) -> None:
    """
    Generates and saves a two-panel chart showing day-ahead prices vs BESS dispatch,
    and the resulting State of Charge (SoC).
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set up a two-panel figure (Prices/Power on top, SoC on bottom)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    periods = dispatch_df["Period"]
    prices = dispatch_df["Price_GBP_MWh"]

    # Invert charge values so they appear below the zero line (representing buying/drawing power)
    charge = -dispatch_df["Charge_MW"]
    discharge = dispatch_df["Discharge_MW"]
    soc = dispatch_df["SoC_MWh"]

    # ==========================================
    # Top Panel: Prices vs. Dispatch Power
    # ==========================================
    ax1_price = ax1.twinx()

    # 1. Plot the Price Curve (Line)
    (line_price,) = ax1_price.plot(
        periods, prices, color="#1f77b4", linewidth=2.5, label="Price (£/MWh)"
    )
    ax1_price.set_ylabel("Price (£/MWh)", color="#1f77b4", fontweight="bold")
    ax1_price.tick_params(axis="y", labelcolor="#1f77b4")

    # 2. Plot the Battery Dispatch (Bars)
    bar_dis = ax1.bar(
        periods, discharge, color="#2ca02c", alpha=0.8, label="Discharge (Sell)"
    )
    bar_chg = ax1.bar(periods, charge, color="#d62728", alpha=0.8, label="Charge (Buy)")
    ax1.set_ylabel("Battery Power (MW)", fontweight="bold")

    # Add a zero baseline for clarity
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Combine legends for the top panel
    lines_1 = [line_price, bar_dis, bar_chg]
    labels_1 = [l.get_label() for l in lines_1]
    ax1.legend(lines_1, labels_1, loc="upper left", frameon=True, facecolor="white")
    ax1.set_title(
        "Optimal BESS Dispatch vs. Day-Ahead Wholesale Prices",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # ==========================================
    # Bottom Panel: State of Charge (SoC)
    # ==========================================
    ax2.fill_between(periods, 0, soc, color="#9467bd", alpha=0.3)
    ax2.plot(periods, soc, color="#9467bd", linewidth=2, label="State of Charge (MWh)")

    ax2.set_ylabel("SoC (MWh)", color="#9467bd", fontweight="bold")
    ax2.set_xlabel("Settlement Period (30-Minute Intervals)", fontweight="bold")

    # Lock the Y-axis to the battery's physical limits (plus 5% visual headroom)
    ax2.set_ylim(0, capacity_mwh * 1.05)
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    ax2.legend(loc="upper left", frameon=True, facecolor="white")
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # Formatting & Saving
    # ==========================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Dispatch visualization successfully saved to: {save_path}")
