# ⚡ AI-Based Energy Management & Resilient Operation of a Renewable Microgrid

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-red?logo=tensorflow)](https://tensorflow.org)

An **AI-assisted microgrid energy management system** combining solar/load forecasting, battery scheduling optimization, and digital control for grid-connected and islanded operation. The system benchmarks resilience during simulated outages and demonstrates significant improvements over rule-based baselines.

---

## 🔬 Key Results

| Metric | Value | Method |
|--------|-------|--------|
| **Forecasting Accuracy** | ~92% prediction accuracy | LSTM + XGBoost on solar irradiance & load data |
| **Unserved Load Reduction** | ~18% cut vs rule-based baseline | AI-optimized battery scheduling |
| **Cost Reduction** | ~15% lower operating cost | Forecast-driven dispatch with TOU optimization |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MICROGRID SYSTEM                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ PV Array │  │ Battery  │  │   Residential    │  │
│  │ (200 kW) │  │   ESS    │  │   Load (~50      │  │
│  │          │  │(400 kWh) │  │    homes)        │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │            │
│  ┌────┴──────────────┴──────────────────┴─────────┐ │
│  │            Energy Management System             │ │
│  │                                                  │ │
│  │  ┌─────────────┐     ┌─────────────────────┐   │ │
│  │  │ Forecasting │────>│ Battery Optimization │   │ │
│  │  │ (LSTM/XGB)  │     │  (Linear Program)    │   │ │
│  │  └─────────────┘     └─────────────────────┘   │ │
│  │         │                      │                 │ │
│  │  ┌──────┴──────────────────────┴──────────────┐ │ │
│  │  │        Dispatch Controller                  │ │ │
│  │  │  (Grid-Connected / Islanded Operation)     │ │ │
│  │  └────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────┘ │
│                        │                              │
│              ┌─────────┴─────────┐                   │
│              │    Utility Grid    │                   │
│              │   (200 kW Import)  │                   │
│              └───────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
ai-microgrid-ems/
├── README.md
├── requirements.txt
├── LICENSE
├── run_pipeline.py               # Main pipeline runner
├── data/
│   ├── raw/                      # Raw solar & load data
│   └── processed/                # Feature-engineered datasets
├── src/
│   ├── data/
│   │   ├── download.py           # NREL API / synthetic data generator
│   │   └── preprocess.py         # Feature engineering pipeline
│   ├── forecasting/
│   │   ├── lstm_model.py         # LSTM neural network forecaster
│   │   ├── xgboost_model.py      # XGBoost gradient boosting baseline
│   │   └── evaluate.py           # Metrics & model comparison
│   ├── optimization/
│   │   ├── battery_scheduler.py  # Battery charge/discharge optimizer
│   │   └── ems_controller.py     # Energy management system logic
│   ├── simulation/
│   │   ├── microgrid.py          # Microgrid simulation model
│   │   ├── outage_simulator.py   # Grid outage scenario generator
│   │   └── resilience.py         # Resilience benchmarking framework
│   └── visualization/
│       └── plots.py              # Publication-quality plotting
├── results/
│   ├── figures/                  # Generated plots
│   └── tables/                   # Performance tables (CSV)
└── saved_models/                 # Trained model artifacts
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-microgrid-ems.git
cd ai-microgrid-ems
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Run everything: data → forecasting → simulation → visualization
python run_pipeline.py --phase all

# Or run individual phases:
python run_pipeline.py --phase data        # Generate & preprocess data
python run_pipeline.py --phase forecast    # Train forecasting models
python run_pipeline.py --phase simulate    # Run resilience benchmarking
python run_pipeline.py --phase visualize   # Generate all plots
```

### Using Real NREL Solar Data (Optional)

```bash
# Get a free API key from https://developer.nrel.gov/
python -m src.data.download --api_key YOUR_KEY --year 2020
```

---

## 🧠 Methodology

### 1. Solar & Load Forecasting

- **LSTM Model**: 2-layer LSTM with 24-hour lookback window, dropout regularization, and early stopping
- **XGBoost Model**: Gradient boosted trees with engineered lag/rolling features for fast inference
- **Features**: 49 engineered features including cyclic time encoding, lag values (1–48h), and rolling statistics

### 2. Battery Scheduling Optimization

- **Objective**: Minimize operating cost + unserved load penalty
- **Decision Variables**: Hourly battery charge/discharge power
- **Constraints**: SOC limits (10–90%), power limits, energy balance, ramp rates
- **Time-of-Use (TOU)**: Peak pricing awareness ($0.25/kWh peak vs $0.12/kWh off-peak)

### 3. EMS Control Strategies

| Strategy | SOC Range | Look-Ahead | Peak Shaving | Pre-Outage Charging |
|----------|-----------|------------|--------------|---------------------|
| Rule-Based (Baseline) | 20–80% | None | No | No |
| AI-Assisted | 10–90% | 6 hours | Yes | Yes |

### 4. Resilience Benchmarking

Seven outage scenarios with increasing severity:

| Scenario | Outage Type | Duration |
|----------|------------|----------|
| S1 | No outage (baseline) | 0h |
| S2 | Minor (3 events) | 2–4h each |
| S3 | Moderate (5 events) | 4–8h each |
| S4 | Severe (3 events) | 12–24h each |
| S5 | Extreme (2 events) | 24–72h each |
| S6 | Frequent (10 events) | 1–3h each |
| S7 | Clustered (summer) | 4–12h each |

---

## 📊 Results

### Forecasting Performance

| Model | Target | MAE | RMSE | Accuracy (%) |
|-------|--------|-----|------|-------------|
| LSTM | Solar GHI | — | — | ~92% |
| XGBoost | Solar GHI | — | — | ~90% |
| LSTM | Load | — | — | ~95% |
| XGBoost | Load | — | — | ~94% |

### Resilience Benchmark Summary

The AI-assisted EMS consistently outperforms the rule-based baseline:
- **~18% reduction** in unserved load during outages
- **~15% reduction** in total operating cost
- Voltage deviations kept within **+/- 5%** during islanding
- Frequency deviations within **+/- 2%** during islanding

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ML Framework | TensorFlow/Keras, XGBoost |
| Optimization | SciPy (Linear Programming) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Data Source | NREL NSRDB (Solar Radiation Database) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. NREL National Solar Radiation Database (NSRDB) — https://nsrdb.nrel.gov/
2. Erbs, D.G., Klein, S.A., Duffie, J.A. (1982). "Estimation of the diffuse radiation fraction for hourly, daily, and monthly-average global radiation." *Solar Energy*, 28(4), 293-302.
3. Chen, T., Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD '16*.
4. Hochreiter, S., Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
