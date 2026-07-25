# ⚡ GB BESS Price Forecasting & Dispatch Optimizer

An end-to-end quantitative energy pipeline that ingests live UK power market data from the **Elexon Insights API**, predicts 30-minute day-ahead electricity prices using **XGBoost**, and formulates a **Linear Programming (LP)** model in **PuLP** to optimize revenue dispatch for a 50MW / 100MWh Battery Energy Storage System (BESS).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Domain](https://img.shields.io/badge/Domain-Energy%20Trading%20%26%20Flexibility-green)
![Optimization](https://img.shields.io/badge/Solver-PuLP%20%2F%20CBC-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 💡 Overview

Battery Energy Storage Systems (BESS) are critical for balancing the Great Britain (GB) power grid, which is increasingly dominated by variable wind energy. To operate profitably in wholesale arbitrage markets, an asset manager must:
1. **Forecast** volatile 30-minute settlement period prices ($\Delta t = 0.5\text{ hrs}$).
2. **Optimize** charge/discharge cycles under physical efficiency and capacity constraints.

This repository provides a complete, modular workflow demonstrating how machine learning forecasts translate directly into optimal asset scheduling and financial backtesting.

---

## 🏗 System Architecture

[ Elexon API ] ---> [ Feature Engineering ] ---> [ XGBoost Price Model ]
(B1770 / B1440)     (Lags, Wind, Demand)           (30-min Forecasts)
|
v
[ Streamlit UI ] <--- [ Financial Backtest ] <--- [ PuLP LP Solver ]
(Visualizations)     (Foresight Gap Analysis)      (50MW BESS Schedule)

---

## 🧮 Mathematical Formulation

The optimization engine solves a daily Linear Program over $T = 48$ settlement periods to maximize arbitrage revenue:

$$\max \sum_{t=1}^{T} \left( P_{\text{dis}, t} \cdot \hat{C}_t - P_{\text{ch}, t} \cdot \hat{C}_t \right) \cdot \Delta t$$

**Subject to Asset Constraints:**

1. **State of Charge ($\text{SoC}$) Energy Balance:**
   $$\text{SoC}_t = \text{SoC}_{t-1} + \left( P_{\text{ch}, t} \cdot \eta_{\text{ch}} - \frac{P_{\text{dis}, t}}{\eta_{\text{dis}}} \right) \cdot \Delta t$$
2. **Power & Capacity Limits:**
   $$0 \le P_{\text{ch}, t} \le P_{\text{max}} \quad (50 \text{ MW})$$
   $$0 \le P_{\text{dis}, t} \le P_{\text{max}} \quad (50 \text{ MW})$$
   $$\text{SoC}_{\text{min}} \le \text{SoC}_t \le \text{SoC}_{\text{max}} \quad (0 \le \text{SoC}_t \le 100 \text{ MWh})$$

*Where $\eta_{\text{rt}} = 88\%$ ($\eta_{\text{ch}} = \eta_{\text{dis}} \approx 0.938$) and $\Delta t = 0.5 \text{ hours}$.*

---

## ✨ Key Features

* **Data Ingestion:** Automated fetching and cleaning of Elexon settlement data (Market Index Prices `B1770`, Wind Forecasts `B1440`, and Total National Load).
* **ML Price Engine:** Time-series feature engineering (24h/48h lags, wind-to-demand ratio, hour-of-day encoding) using XGBoost with Walk-Forward Cross-Validation.
* **LP Dispatch Optimizer:** High-performance LP dispatch solver using `PuLP` handling round-trip efficiency losses and battery energy limits.
* **Foresight Gap Analysis:** Compares revenue generated using predicted prices against a "perfect foresight" model (actual outturn prices) to quantify the financial cost of forecast error ($\text{MAE}$).

---

## 📁 Repository Structure

```text
├── data/                  # Sample raw and processed Elexon datasets
├── notebooks/             # Exploratory Data Analysis & Prototype LP
│   ├── 01_elexon_eda.ipynb
│   └── 02_lp_proof_of_concept.ipynb
├── src/                   # Production modular Python package
│   ├── data_ingestion.py  # Elexon API wrapper & resampler
│   ├── features.py        # Lag & calendar feature generator
│   ├── model.py           # XGBoost pipeline & evaluation
│   └── optimizer.py       # PuLP Linear Programming engine
├── app.py                 # (Optional) Interactive Streamlit Dashboard
├── main.py                # End-to-end execution script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

🚀 Getting Started
1. Prerequisites & Installation
Clone the repository and set up a virtual environment:

git clone [https://github.com/your-username/gb-bess-dispatch-optimizer.git](https://github.com/your-username/gb-bess-dispatch-optimizer.git)
cd gb-bess-dispatch-optimizer

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Run the Full PipelineExecute the end-to-end pipeline (Data Download $\rightarrow$ Model Train $\rightarrow$ LP Optimize $\rightarrow$ Backtest):Bash

python main.py --start-date 2025-01-01 --end-date 2025-06-01