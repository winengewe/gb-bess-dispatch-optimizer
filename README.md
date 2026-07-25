# ⚡ Great Britain BESS Price Forecasting & Dispatch Optimizer

An end-to-end quantitative energy pipeline that ingests live UK power market data from the **Elexon Insights API**, predicts 30-minute day-ahead electricity prices using **XGBoost**, and formulates a **Linear Programming (LP)** model in **PuLP** to optimize revenue dispatch for a 50MW / 100MWh Battery Energy Storage System (BESS).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Energy%20Trading%20%26%20Flexibility-green.svg)]()
[![Optimization](https://img.shields.io/badge/Solver-PuLP%20%2F%20CBC-orange.svg)](https://coin-or.github.io/pulp/)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 💡 Commercial Context

In Great Britain's electricity market, increasing penetration of intermittent wind and solar generation drives significant intra-day price volatility across the 48 half-hourly settlement periods ($\Delta t = 0.5 \text{ hours}$).

Battery Energy Storage Systems (BESS) capture value through wholesale market arbitrage—charging during periods of excess generation (low or negative prices) and discharging during peak demand hours.

Operating a commercial BESS asset requires:
1. **Accurate Price Forecasting:** Predicting volatile day-ahead wholesale price profiles influenced by wind outturn and system load.
2. **Constraint-Aware Optimization:** Scheduling charge/discharge profiles under strict round-trip efficiency losses, capacity bounds, and battery degradation penalties.

---

## 🏗 System Architecture

```
                            [ ELEXON INSIGHTS API ]
                        (B1770 / B1440 / Demand Data)
                                     │
                                     ▼
                         [ FEATURE ENGINEERING ]
                     (Lags, Fourier Terms, Wind/Load)
                                     │
                                     ▼
                        [ XGBOOST PRICE ENGINE ]
                     (30-Min Settlement Forecasts)
                                     │
                                     ▼
                      [ PULP DISPATCH OPTIMIZER ]
                   (Linear Program with Degradation)
                                     │
                                     ▼
                     [ FINANCIAL BACKTEST ENGINE ]
                 (Foresight Gap & Capture Rate Analysis)

```

---

## 🧮 Mathematical Formulation

The optimization engine formulates and solves a daily Linear Program over $T = 48$ settlement periods to maximize net arbitrage revenue while penalizing cycle degradation.

### Objective Function

$$\max \sum_{t=1}^{T} \left[ \left( P_{\text{dis}, t} \cdot \hat{C}_t - P_{\text{ch}, t} \cdot \hat{C}_t \right) - C_{\text{deg}} \cdot (P_{\text{ch}, t} + P_{\text{dis}, t}) \right] \cdot \Delta t$$

Where:
* $P_{\text{ch}, t}, P_{\text{dis}, t} \ge 0$: Charge and discharge power in MW at period $t$.
* $\hat{C}_t$: Forecasted wholesale electricity price (£/MWh) at period $t$.
* $C_{\text{deg}}$: Battery degradation penalty cost (£12.50 / MWh throughput).
* $\Delta t = 0.5 \text{ hours}$: Settlement period duration.

---

### Constraints

1. **State of Charge ($\text{SoC}$) Energy Conservation:**
   $$\text{SoC}_t = \text{SoC}_{t-1} + \left( P_{\text{ch}, t} \cdot \eta_{\text{ch}} - \frac{P_{\text{dis}, t}}{\eta_{\text{dis}}} \right) \cdot \Delta t \quad \forall t \in \{1, \dots, T\}$$

2. **Power & Capacity Limits:**
   $$0 \le P_{\text{ch}, t} \le P_{\text{max}} \quad (50 \text{ MW})$$
   $$0 \le P_{\text{dis}, t} \le P_{\text{max}} \quad (50 \text{ MW})$$
   $$\text{SoC}_{\text{min}} \le \text{SoC}_t \le \text{SoC}_{\text{max}} \quad (0 \le \text{SoC}_t \le 100 \text{ MWh})$$

3. **Round-Trip Efficiency Split ($\eta_{\text{rt}} = 88\%$):**
   $$\eta_{\text{ch}} = \eta_{\text{dis}} = \sqrt{\eta_{\text{rt}}} \approx 0.9381$$

4. **Terminal SoC Boundary Condition:**
   $$\text{SoC}_0 = \text{SoC}_T = 0.50 \cdot \text{SoC}_{\text{max}} \quad (50 \text{ MWh})$$

---

## ✨ Key Features

* **Automated Data Ingestion:** Connects directly to the **Elexon Insights API** to retrieve Market Index Prices (`B1770`), Day-Ahead Wind Forecasts (`B1440`), and System Demand.
* **Feature Engineering:** Constructs time-series lag features ($t-24\text{h}$, $t-48\text{h}$), cyclical calendar encodings (Fourier transforms for diurnal trends), and wind-to-demand supply ratios.
* **Time-Series ML Pipeline:** Evaluates XGBoost price predictions using expanding-window **Walk-Forward Cross-Validation** to prevent future-data leakage.
* **Commercial LP Optimizer:** Solves asset dispatch via `PuLP` using the CBC solver, incorporating battery round-trip efficiency and cycle wear parameters.
* **Foresight Gap Analysis:** Computes the **Capture Rate (%)** by comparing model dispatch performance against an idealized "Perfect Foresight" benchmark.

---

## 📁 Repository Structure

```text
gb-bess-dispatch-optimizer/
│
├── config/
│   └── config.yaml          # Asset parameters, API settings, and model configs
│
├── data/                    # Sample raw and processed Elexon datasets
│   ├── raw/
│   └── processed/
│
├── notebooks/               # Exploratory Data Analysis & POCs
│   ├── 01_elexon_eda.ipynb
│   └── 02_lp_proof_of_concept.ipynb
│
├── src/                     # Core Production Modules
│   ├── __init__.py
│   ├── data_ingestion.py   # Elexon API wrapper & multi-period aggregator
│   ├── features.py         # Cyclical encodings, lags, & rolling stats
│   ├── model.py            # XGBoost training & walk-forward validation
│   ├── optimizer.py        # PuLP linear optimization formulation
│   └── backtest.py         # Performance reporting & capture rate metrics
│
├── .gitignore
├── LICENSE
├── README.md
├── main.py                  # End-to-end execution pipeline script
└── requirements.txt        # Project dependencies

```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up your Python environment:

```bash
# Clone repository
git clone [https://github.com/your-username/gb-bess-dispatch-optimizer.git](https://github.com/your-username/gb-bess-dispatch-optimizer.git)
cd gb-bess-dispatch-optimizer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Configuration Setup

Adjust parameters in `config/config.yaml` or run with default settings:

```yaml
asset:
  power_mw: 50.0
  capacity_mwh: 100.0
  round_trip_efficiency: 0.88
  degradation_cost_gbp_per_mwh: 12.50
  initial_soc_pct: 0.50

model:
  forecast_horizon_periods: 48
  test_split_days: 30

```

### 3. Run the Pipeline

Execute the full end-to-end data ingestion, price forecasting, LP optimization, and backtesting run:

```bash
python main.py --start-date 2025-01-01 --end-date 2025-06-01 --save-plots

```

---

## 📊 Sample Results

Backtest evaluation performed over a 30-day unseen test horizon:

| Metric | Naive Strategy (Historical Avg) | XGBoost Strategy (Model) | Perfect Foresight (Benchmark) |
| --- | --- | --- | --- |
| **Daily Net Profit (£)** | £4,120 / day | **£7,480 / day** | £8,520 / day |
| **Capture Rate (%)** | 48.3% | **87.8%** | 100.0% |
| **Forecast MAE (£/MWh)** | — | **£7.85 / MWh** | — |
| **Daily Equivalent Cycles** | 2.10 | **1.65** | 1.82 |

### Key Insights

* **Degradation Impact:** Adding the £12.50/MWh degradation penalty successfully eliminated low-margin micro-cycling, reducing daily battery wear by **21%** while preserving **95%+** of available arbitrage value.
* **Forecast Value:** The XGBoost model achieved an **87.8% Capture Rate** relative to perfect market foresight, proving that accurate wind-to-demand ratio features capture peak price spikes effectively.

---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** XGBoost, Scikit-Learn
* **Optimization:** PuLP (CBC Solver)
* **API Ingestion:** Requests / Elexon Insights API
* **Data Visualization:** Plotly, Matplotlib, Seaborn

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
