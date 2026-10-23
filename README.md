# ⚡ AI-Based Energy Management & Resilient Operation of a Renewable Microgrid

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-red?logo=tensorflow)](https://tensorflow.org)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization%20(HiGHS)-blue)](https://scipy.org)

An **AI-assisted microgrid energy management system (EMS)** combining solar irradiance and load forecasting, multi-period battery scheduling optimization, and digital control for grid-connected and islanded operation. The system benchmarks resilience against physical baselines across 7 outage scenarios, providing significant reductions in unserved energy and operational cost.

---

## 🔬 Key Results

| Metric | Value | Method |
|--------|-------|--------|
| **Forecasting Accuracy** | ~92% (Solar GHI), ~94% (Load) | LSTM Neural Networks & XGBoost Gradient Boosting |
| **Unserved Load Reduction** | **42.7% average cut** during outages (up to **77.9%**) | Multi-Period Linear Programming (HiGHS LP) |
| **Operating Cost Reduction** | **14.8% average reduction** (up to **28.6%**) | Multi-Period Linear Programming with TOU Arbitrage |
| **Power Quality Bounds** | Voltage ±5%, Frequency ±2% | Controlled inverter droop during islanded operation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       MICROGRID SYSTEM                          │
│                                                                 │
│  ┌──────────┐        ┌──────────┐        ┌──────────────────┐  │
│  │ PV Array │        │ Battery  │        │   Residential    │  │
│  │ (200 kW) │        │   BESS   │        │   Load (~50      │  │
│  │          │        │(400 kWh) │        │    homes)        │  │
│  └────┬─────┘        └────┬─────┘        └────────┬─────────┘  │
│       │                   │                       │            │
│  ┌────┴───────────────────┴───────────────────────┴──────────┐ │
│  │               Energy Management System (EMS)              │ │
│  │                                                           │ │
│  │  ┌──────────────┐     ┌────────────────────────────────┐  │ │
│  │  │ Forecasting  │────>│   Battery Optimization Dispatch │  │ │
│  │  │  (LSTM/XGB)  │     │   - Real-Time Look-Ahead        │  │ │
│  │  │              │     │   - Multi-Period LP (HiGHS)     │  │ │
│  │  └──────────────┘     └────────────────────────────────┘  │ │
│  │         │                             │                   │ │
│  │  ┌──────┴─────────────────────────────┴─────────────────┐ │ │
│  │  │               Digital Dispatch Controller             │ │ │
│  │  │        (Grid-Connected / Islanded Microgrid)         │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                  │                             │
│                        ┌─────────┴─────────┐                   │
│                        │    Utility Grid   │                   │
│                        │   (200 kW Import) │                   │
│                        └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
ai-microgrid-ems/
├── README.md
├── requirements.txt
├── LICENSE
├── run_pipeline.py               # Complete end-to-end pipeline runner
├── data/
│   ├── raw/                      # Raw solar & load profiles
│   └── processed/                # Preprocessed feature-engineered datasets
├── src/
│   ├── data/
│   │   ├── download.py           # NREL NSRDB API & physics-based data generator
│   │   └── preprocess.py         # 49-feature time-series engineering pipeline
│   ├── forecasting/
│   │   ├── lstm_model.py         # 2-layer LSTM recurrent neural network
│   │   ├── xgboost_model.py      # XGBoost gradient boosting forecaster
│   │   └── evaluate.py           # Statistical metrics (MAE, RMSE, MAPE)
│   ├── optimization/
│   │   ├── battery_scheduler.py  # HiGHS Linear Program & look-ahead heuristic
│   │   └── ems_controller.py     # Grid-connected and islanded dispatch logic
│   ├── simulation/
│   │   ├── microgrid.py          # Microgrid electrical balance & power quality
│   │   ├── outage_simulator.py   # Multi-severity outage scenario generator
│   │   └── resilience.py         # Resilience benchmarking framework
│   └── visualization/
│       └── plots.py              # Publication-ready visualization generation
├── results/
│   ├── figures/                  # Generated performance figures
│   └── tables/                   # Quantitative benchmark tables (CSV)
└── saved_models/                 # Exported model weights and configurations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/ShaikhAyan3053/ai-microgrid-ems.git
cd ai-microgrid-ems
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Execute complete pipeline: data → forecasting → simulation → visualization
python run_pipeline.py --phase all

# Or run individual modules:
python run_pipeline.py --phase data        # Generate & preprocess datasets
python run_pipeline.py --phase forecast    # Train & evaluate XGBoost/LSTM forecasters
python run_pipeline.py --phase simulate    # Run multi-scenario resilience benchmark
python run_pipeline.py --phase visualize   # Render performance figures
```

---

## 📊 Data Provenance & Empirical NSRDB Access

- **Committed Benchmark Data**: The baseline datasets in `data/processed/` are **physics-informed synthetic benchmarks** calibrated to Phoenix, AZ latitude (33.45° N). Solar irradiance is computed using clear-sky geometric equations (Ineichen model) combined with stochastic cloud attenuation, and load profiles model aggregate residential smart-meter diurnal curves with weekend variance.
- **Empirical NREL NSRDB Integration**: For live empirical satellite data, the repository includes a built-in client for the **NREL National Solar Radiation Database (NSRDB PSM3 API)**:
  ```bash
  # Download empirical 2020 solar irradiance data using a free NREL API key
  python -m src.data.download --api_key YOUR_NREL_KEY --year 2020
  ```

---

## 🧠 Methodology

### 1. Solar & Load Forecasting
- **LSTM Architecture**: 2-layer recurrent network (64 and 32 hidden units, dropout 0.2) evaluated across a 24-hour sliding temporal context.
- **XGBoost Forecaster**: 500 gradient-boosted regression trees with early stopping and feature subsampling.
- **Feature Engineering**: 49 domain features including solar zenith angle approximations, cyclic trigonometric time encodings, multi-horizon lags ($t-1\text{h}$ to $t-24\text{h}$), and rolling statistics (6h, 12h, 24h rolling mean/std).

### 2. Battery Storage Optimization & Dispatch
The EMS provides two dispatch engines in [`src/optimization/battery_scheduler.py`](src/optimization/battery_scheduler.py):
1. **Multi-Period Linear Program (LP)**: Solved via `scipy.optimize.linprog` using the interior-point/simplex **HiGHS** solver over multi-hour horizons.
   $$\min \sum_{t=1}^{T} \left[ c_{\text{imp}}(t) P_{\text{imp}}(t) - c_{\text{exp}} P_{\text{exp}}(t) + c_{\text{uns}} P_{\text{uns}}(t) + c_{\text{deg}} (P_{\text{ch}}(t) + P_{\text{dis}}(t)) \right]$$
   subject to:
   - Bus power balance: $P_{\text{ch}}(t) - P_{\text{dis}}(t) - P_{\text{imp}}(t) + P_{\text{exp}}(t) - P_{\text{uns}}(t) + P_{\text{curt}}(t) = P_{\text{pv}}(t) - P_{\text{load}}(t)$
   - Storage state transition: $E(t) = (1 - \delta)E(t-1) + \eta_{\text{ch}} P_{\text{ch}}(t) - \frac{1}{\eta_{\text{dis}}} P_{\text{dis}}(t)$
   - Physical constraints: $P_{\text{ch}} \le P_{\text{ch}}^{\max}$, $P_{\text{dis}} \le P_{\text{dis}}^{\max}$, $\text{SOC}_{\min} \le \text{SOC} \le \text{SOC}_{\max}$.
2. **Online Look-Ahead Heuristic**: A fast, deterministic controller for edge execution that evaluates 12-hour look-ahead deficit forecasts to trigger anticipatory grid charging before forecasted outages.

### 3. EMS Control Comparison

| Parameter | Rule-Based Baseline | AI-Assisted EMS |
|-----------|---------------------|-----------------|
| **Operating SOC Limits** | Conservative (20% – 80%) | Dynamic (10% – 90%) |
| **Inverter Headroom** | Derated to 80% (safety factor) | 100% rated power (100 kW) |
| **Look-Ahead Foresight** | None (instantaneous reactive) | 6–12h rolling forecast |
| **Anticipatory Charging** | No (only charges on PV surplus) | Yes (pre-charges from grid before outage) |
| **TOU Peak Shaving** | Reactive | Predictive cost minimization |

---

## 📈 Experimental Results

### Forecasting Accuracy

| Model | Target Variable | MAE | RMSE | Accuracy Score |
|-------|-----------------|-----|------|----------------|
| **XGBoost** | Solar GHI ($W/m^2$) | 28.4 | 46.1 | ~90.2% |
| **LSTM** | Solar GHI ($W/m^2$) | 24.1 | 38.7 | ~92.4% |
| **XGBoost** | Load Demand (kW) | 3.2 | 4.8 | ~94.1% |
| **LSTM** | Load Demand (kW) | 2.9 | 4.2 | ~95.3% |

### Outage Resilience Benchmark

Evaluated across a test set of 1,752 hours (Q4 test partition) under 7 standardized outage scenarios:

| Scenario | Outage Severity | Outage Hours | Rule-Based Unserved | AI (LP) Unserved | Unserved Reduction | Cost Reduction |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **S1** | Grid-connected (baseline) | 0h | 63.4 kWh | 0.0 kWh | **-100.0%** | +8.3% |
| **S2** | Minor (3 short events) | 10h | 1,266.4 kWh | 405.3 kWh | **-68.0%** | +17.7% |
| **S3** | Moderate (5 events, 4–8h) | 33h | 3,863.5 kWh | 2,306.5 kWh | **-40.3%** | +20.1% |
| **S4** | Severe (3 events, 12–24h) | 52h | 5,529.2 kWh | 4,553.8 kWh | **-17.6%** | +12.1% |
| **S5** | Extreme (2 multi-day events) | 62h | 6,807.8 kWh | 6,136.3 kWh | **-9.9%** | +8.5% |
| **S6** | Frequent (10 recurring events)| 22h | 2,626.4 kWh | 580.5 kWh | **-77.9%** | +28.6% |
| **S7** | Clustered summer peak | 0h | 63.4 kWh | 0.0 kWh | **-100.0%** | +8.3% |

> **Key Finding**: Multi-period Linear Programming achieves massive resilience gains (40% to 78% unserved load reduction) across short, moderate, and recurring outages by dynamically scheduling energy reservations ahead of grid disconnection. In extreme multi-day outages (S4–S5), unserved load reductions moderate to 10–18% because battery duration is physically bound by the 400 kWh storage capacity.

---

## ⚠️ Limitations & Future Work

Transparently stating assumptions is essential for credible power systems research:
1. **Data Provenance**: The default benchmark uses physics-based synthetic data calibrated to Phoenix, AZ; validation against empirical utility smart-meter telemetry is an active next step.
2. **Computational Architecture**: The EMS natively employs multi-period Linear Programming (`optimize_lp` via SciPy HiGHS) as the primary dispatch engine for forecast-driven energy management, while maintaining a lightweight look-ahead heuristic as an edge fallback for embedded hardware with low compute envelopes.
3. **Outage Foresight Assumption**: Anticipatory charging assumes a 6–12 hour advisory window prior to outages (modeling severe weather forecasts or utility warning alerts). Ongoing work includes stochastic chance-constrained optimization for unexpected instantaneous line trips.
4. **Battery Degradation Modeling**: Aging is currently captured through a linearized throughput degradation cost ($0.01/kWh) rather than non-linear electro-chemical models (e.g., Rainflow cycle counting or SEI growth).

---

## 🛠️ Technical Stack

- **Language**: Python 3.9+
- **Machine Learning**: TensorFlow / Keras (LSTM), XGBoost
- **Optimization**: SciPy (`scipy.optimize.linprog` with HiGHS solver)
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Integration**: NREL NSRDB PSM3 REST API

---

## 📄 License

This repository is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

1. Sengupta, M., et al. (2018). "The National Solar Radiation Data Base (NSRDB)." *Renewable and Sustainable Energy Reviews*, 89, 51-60.
2. Ineichen, P., & Perez, R. (2002). "A new airmass independent formulation for the Linke turbidity coefficient." *Solar Energy*, 73(3), 151-157.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *ACM SIGKDD*.
4. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
