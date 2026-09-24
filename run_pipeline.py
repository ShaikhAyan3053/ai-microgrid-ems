"""
AI-Based Microgrid EMS - Main Runner
======================================
Orchestrates the complete pipeline:
    1. Data generation/download
    2. Preprocessing
    3. Forecasting model training (LSTM + XGBoost)
    4. Battery optimization & EMS dispatch
    5. Resilience benchmarking
    6. Visualization & results export

Usage:
    python run_pipeline.py --phase all
    python run_pipeline.py --phase data
    python run_pipeline.py --phase forecast
    python run_pipeline.py --phase simulate
    python run_pipeline.py --phase visualize
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_data_pipeline(year: int = 2020):
    """Phase 1: Generate data and preprocess."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Data Pipeline")
    print("=" * 70)

    from src.data.download import download_and_save
    from src.data.preprocess import preprocess_and_save

    download_and_save(use_synthetic=True, year=year)
    data = preprocess_and_save(year=year)
    return data


def run_forecasting(year: int = 2020):
    """Phase 2: Train forecasting models and evaluate."""
    print("\n" + "=" * 70)
    print("  PHASE 2: Forecasting Models")
    print("=" * 70)

    from src.forecasting.xgboost_model import XGBoostForecaster
    from src.forecasting.evaluate import evaluate_model, compare_models

    # Load data
    train_df = pd.read_csv(
        PROCESSED_DATA_DIR / f"train_{year}.csv",
        index_col="timestamp", parse_dates=True,
    )
    val_df = pd.read_csv(
        PROCESSED_DATA_DIR / f"val_{year}.csv",
        index_col="timestamp", parse_dates=True,
    )
    test_df = pd.read_csv(
        PROCESSED_DATA_DIR / f"test_{year}.csv",
        index_col="timestamp", parse_dates=True,
    )

    results = {}

    # --- XGBoost Solar ---
    print("\n--- Training XGBoost Solar Forecaster ---")
    xgb_solar = XGBoostForecaster(n_estimators=500, max_depth=6, learning_rate=0.05)
    xgb_solar.fit(train_df, val_df, target_col="ghi")
    xgb_solar_pred = xgb_solar.predict(test_df)
    xgb_solar_metrics = evaluate_model(
        test_df["ghi"].values, xgb_solar_pred, name="XGBoost Solar"
    )
    results["XGBoost Solar"] = xgb_solar_metrics
    xgb_solar.save("xgb_solar")

    # --- XGBoost Load ---
    print("\n--- Training XGBoost Load Forecaster ---")
    xgb_load = XGBoostForecaster(n_estimators=500, max_depth=6, learning_rate=0.05)
    xgb_load.fit(train_df, val_df, target_col="load_kw")
    xgb_load_pred = xgb_load.predict(test_df)
    xgb_load_metrics = evaluate_model(
        test_df["load_kw"].values, xgb_load_pred, name="XGBoost Load"
    )
    results["XGBoost Load"] = xgb_load_metrics
    xgb_load.save("xgb_load")

    # --- Model Comparison ---
    comparison_df = compare_models(results)

    # Save comparison
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(tables_dir / "forecast_comparison.csv")

    return {
        "results": results,
        "comparison": comparison_df,
        "predictions": {
            "xgb_solar": xgb_solar_pred,
            "xgb_load": xgb_load_pred,
        },
        "test_df": test_df,
    }


def run_simulation(year: int = 2020, forecasts: dict = None):
    """Phase 3-5: Simulation, optimization, and resilience benchmarking."""
    print("\n" + "=" * 70)
    print("  PHASE 3-5: Simulation & Resilience Benchmarking")
    print("=" * 70)

    from src.simulation.resilience import ResilienceBenchmark

    test_df = pd.read_csv(
        PROCESSED_DATA_DIR / f"test_{year}.csv",
        index_col="timestamp", parse_dates=True,
    )

    benchmark = ResilienceBenchmark()
    
    pv_forecast = forecasts["xgb_solar"] if forecasts else None
    load_forecast = forecasts["xgb_load"] if forecasts else None
    
    results = benchmark.run_full_benchmark(
        test_df, 
        pv_forecast=pv_forecast, 
        load_forecast=load_forecast
    )

    return results


def run_visualization(year: int = 2020):
    """Phase 6: Generate all plots."""
    print("\n" + "=" * 70)
    print("  PHASE 6: Visualization")
    print("=" * 70)

    from src.visualization.plots import MicrogridPlotter
    from src.optimization.ems_controller import EMSController

    test_df = pd.read_csv(
        PROCESSED_DATA_DIR / f"test_{year}.csv",
        index_col="timestamp", parse_dates=True,
    )

    plotter = MicrogridPlotter()

    # 1. Solar/Load profile
    print("\n  Generating solar/load profile plot...")
    plotter.plot_solar_load_profile(test_df, days=7)

    # 2. Energy dispatch
    print("  Generating energy dispatch plot...")
    ems = EMSController(strategy="ai_assisted")
    result_df, _ = ems.run(test_df)
    plotter.plot_energy_dispatch(result_df, days=3)

    # 3. Resilience comparison
    tables_dir = RESULTS_DIR / "tables"
    benchmark_path = tables_dir / "strategy_improvements.csv"
    if benchmark_path.exists():
        print("  Generating resilience comparison plot...")
        improvements_df = pd.read_csv(benchmark_path)
        plotter.plot_resilience_comparison(improvements_df)

    # 4. Forecast comparison
    forecast_path = tables_dir / "forecast_comparison.csv"
    if forecast_path.exists():
        print("  Generating model comparison plot...")
        comparison_df = pd.read_csv(forecast_path, index_col=0)
        plotter.plot_model_comparison(comparison_df)

    print(f"\n  All plots saved to: {RESULTS_DIR / 'figures'}")


def run_all(year: int = 2020):
    """Run complete pipeline."""
    print("\n" + "#" * 70)
    print("  AI-Based Microgrid Energy Management System")
    print("  Complete Pipeline Execution")
    print("#" * 70)

    run_data_pipeline(year)
    forecast_results = run_forecasting(year)
    sim_results = run_simulation(year, forecast_results["predictions"])
    run_visualization(year)

    print("\n" + "#" * 70)
    print("  Pipeline Complete!")
    print("#" * 70)
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Figures:          {RESULTS_DIR / 'figures'}")
    print(f"  Tables:           {RESULTS_DIR / 'tables'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Microgrid EMS Pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "data", "forecast", "simulate", "visualize"],
        help="Pipeline phase to run",
    )
    parser.add_argument("--year", type=int, default=2020, help="Data year")
    args = parser.parse_args()

    if args.phase == "all":
        run_all(args.year)
    elif args.phase == "data":
        run_data_pipeline(args.year)
    elif args.phase == "forecast":
        run_forecasting(args.year)
    elif args.phase == "simulate":
        run_simulation(args.year)
    elif args.phase == "visualize":
        run_visualization(args.year)
