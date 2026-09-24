"""
Forecasting Evaluation & Comparison
=====================================
Provides comprehensive evaluation metrics and comparison between LSTM and
XGBoost forecasting models.

Metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R-squared (Coefficient of Determination)
    - Prediction Accuracy (1 - MAPE)

Usage:
    from src.forecasting.evaluate import evaluate_model, compare_models
    metrics = evaluate_model(y_true, y_pred, name="LSTM Solar")
    compare_models(results_dict)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute comprehensive forecasting metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, MAPE, R2, and prediction accuracy.
    """
    # Filter out zero/near-zero actuals for MAPE calculation
    mask = np.abs(y_true) > 1e-3
    y_true_nz = y_true[mask]
    y_pred_nz = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (only on non-zero actuals to avoid division issues)
    if len(y_true_nz) > 0:
        mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100
    else:
        mape = 0.0

    # Prediction accuracy (as mentioned in CV: ~92%)
    accuracy = max(0, 100 - mape)

    # Normalized RMSE
    nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-8) * 100

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
        "nRMSE (%)": round(nrmse, 2),
        "R2": round(r2, 4),
        "Accuracy (%)": round(accuracy, 2),
        "N_samples": len(y_true),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
    verbose: bool = True,
) -> dict:
    """
    Evaluate a single forecasting model and print results.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    name : str
        Model name for display.
    verbose : bool
        Print results table.

    Returns
    -------
    dict
        Evaluation metrics.
    """
    metrics = compute_metrics(y_true, y_pred)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Evaluation: {name}")
        print(f"{'='*50}")
        for key, val in metrics.items():
            print(f"  {key:20s}: {val}")
        print(f"{'='*50}")

    return metrics


def compare_models(
    results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Parameters
    ----------
    results : dict
        Dictionary mapping model name to metrics dict.
    save_path : str, optional
        Path to save comparison table as CSV.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"

    print(f"\n{'='*70}")
    print("  Model Comparison")
    print(f"{'='*70}")
    print(df.to_string())
    print(f"{'='*70}")

    # Highlight best model
    if "Accuracy (%)" in df.columns:
        best_model = df["Accuracy (%)"].idxmax()
        print(f"\n  >> Best model by accuracy: {best_model} "
              f"({df.loc[best_model, 'Accuracy (%)']}%)")

    if save_path:
        df.to_csv(save_path)
        print(f"  >> Saved to {save_path}")

    return df


def evaluate_by_time_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    name: str = "Model",
) -> pd.DataFrame:
    """
    Evaluate model performance broken down by time period.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    timestamps : pd.DatetimeIndex
        Corresponding timestamps.
    name : str
        Model name.

    Returns
    -------
    pd.DataFrame
        Metrics broken down by hour of day.
    """
    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "hour": timestamps.hour,
    })

    hourly_metrics = []
    for hour in range(24):
        mask = df["hour"] == hour
        if mask.sum() < 2:
            continue
        subset = df[mask]
        metrics = compute_metrics(subset["actual"].values, subset["predicted"].values)
        metrics["Hour"] = hour
        hourly_metrics.append(metrics)

    result = pd.DataFrame(hourly_metrics).set_index("Hour")

    print(f"\n  Hourly Performance Breakdown: {name}")
    print(result[["MAE", "RMSE", "Accuracy (%)"]].to_string())

    return result


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.random.uniform(0, 500, 1000)
    y_pred_good = y_true + np.random.normal(0, 20, 1000)
    y_pred_bad = y_true + np.random.normal(0, 80, 1000)

    m1 = evaluate_model(y_true, y_pred_good, name="Good Model")
    m2 = evaluate_model(y_true, y_pred_bad, name="Bad Model")

    compare_models({"Good Model": m1, "Bad Model": m2})
