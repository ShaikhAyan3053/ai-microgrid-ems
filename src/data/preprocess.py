"""
Data Preprocessing & Feature Engineering for Microgrid EMS
==========================================================
Cleans raw solar and load data, handles missing values, creates time-based
features, computes PV power output, and prepares train/test splits for
the forecasting models.

Features Created:
    - Time features: hour, day_of_week, month, day_of_year, is_weekend
    - Cyclic encoding: hour_sin, hour_cos, month_sin, month_cos
    - Solar features: pv_power_kw (from GHI using panel model)
    - Lag features: configurable lag windows
    - Rolling statistics: mean, std over configurable windows
    - Net load: load_kw - pv_power_kw

Usage:
    python -m src.data.preprocess --year 2020
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================================
# PV System Parameters
# ============================================================================

PV_SYSTEM = {
    "capacity_kw": 200.0,       # Installed PV capacity (kW)
    "efficiency": 0.18,          # Panel efficiency (18%)
    "area_m2": 1111.0,           # Total panel area (m²) ≈ capacity / (efficiency * 1kW/m²)
    "temp_coeff": -0.004,        # Power temperature coefficient (%/°C)
    "ref_temp": 25.0,            # Reference temperature (°C)
    "inverter_efficiency": 0.96, # Inverter efficiency
    "soiling_loss": 0.02,        # Soiling/degradation loss (2%)
}


def load_raw_data(year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw solar and load CSV files.

    Parameters
    ----------
    year : int
        Year of data to load.

    Returns
    -------
    tuple
        (solar_df, load_df) DataFrames with datetime index.
    """
    solar_path = RAW_DATA_DIR / f"solar_{year}.csv"
    load_path = RAW_DATA_DIR / f"load_{year}.csv"

    if not solar_path.exists() or not load_path.exists():
        raise FileNotFoundError(
            f"Raw data not found. Run 'python -m src.data.download --use_synthetic' first."
        )

    solar_df = pd.read_csv(solar_path, index_col="timestamp", parse_dates=True)
    load_df = pd.read_csv(load_path, index_col="timestamp", parse_dates=True)

    return solar_df, load_df


def clean_data(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """
    Clean raw data: handle missing values, remove duplicates, clip outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with datetime index.
    name : str
        Name for logging.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    initial_len = len(df)

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Resample to ensure consistent hourly frequency
    df = df.resample("h").mean()

    # Forward-fill small gaps (up to 3 hours), then backward-fill
    df = df.ffill(limit=3).bfill(limit=3)

    # Fill remaining NaN with column median
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Clip negative values for irradiance columns
    irradiance_cols = ["ghi", "dni", "dhi"]
    for col in irradiance_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    final_len = len(df)
    print(f"[CLEAN] {name}: {initial_len} -> {final_len} records "
          f"(NaN remaining: {df.isna().sum().sum()})")

    return df


def compute_pv_power(solar_df: pd.DataFrame) -> pd.Series:
    """
    Compute PV power output from GHI using a simplified single-diode model.

    Accounts for panel efficiency, temperature derating, inverter losses,
    and soiling losses.

    Parameters
    ----------
    solar_df : pd.DataFrame
        Must contain 'ghi' and 'temperature' columns.

    Returns
    -------
    pd.Series
        PV power output in kW.
    """
    ghi = solar_df["ghi"].values  # W/m²
    temp = solar_df.get("temperature", pd.Series(25.0, index=solar_df.index)).values

    # Temperature derating factor
    temp_factor = 1.0 + PV_SYSTEM["temp_coeff"] * (temp - PV_SYSTEM["ref_temp"])
    temp_factor = np.clip(temp_factor, 0.7, 1.1)

    # DC power output
    pv_dc = (
        ghi
        * PV_SYSTEM["area_m2"]
        * PV_SYSTEM["efficiency"]
        * temp_factor
        / 1000.0  # Convert W to kW
    )

    # AC power output (after inverter and soiling losses)
    pv_ac = pv_dc * PV_SYSTEM["inverter_efficiency"] * (1 - PV_SYSTEM["soiling_loss"])

    # Clip to installed capacity
    pv_ac = np.clip(pv_ac, 0, PV_SYSTEM["capacity_kw"])

    return pd.Series(pv_ac, index=solar_df.index, name="pv_power_kw")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features for forecasting models.

    Creates: hour, day_of_week, month, day_of_year, is_weekend,
    plus cyclic (sin/cos) encodings for hour and month.

    Parameters
    ----------
    df : pd.DataFrame
        Data with datetime index.

    Returns
    -------
    pd.DataFrame
        Data with added time features.
    """
    df = df.copy()

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Cyclic encoding (preserves circular nature of time)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list = None,
) -> pd.DataFrame:
    """
    Add lagged values of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Column to create lags for.
    lags : list
        List of lag values in hours (default: [1, 2, 3, 6, 12, 24, 48]).

    Returns
    -------
    pd.DataFrame
        Data with lag features added.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48]

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list = None,
) -> pd.DataFrame:
    """
    Add rolling mean and std features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Column to compute rolling stats for.
    windows : list
        Rolling window sizes in hours (default: [6, 12, 24]).

    Returns
    -------
    pd.DataFrame
        Data with rolling features added.
    """
    if windows is None:
        windows = [6, 12, 24]

    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}h"] = (
            df[target_col].rolling(window=w, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{w}h"] = (
            df[target_col].rolling(window=w, min_periods=1).std().fillna(0)
        )

    return df


def build_microgrid_dataset(
    solar_df: pd.DataFrame, load_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge solar and load data into a single microgrid dataset.

    Computes PV power, net load, and adds all engineered features.

    Parameters
    ----------
    solar_df : pd.DataFrame
        Cleaned solar data.
    load_df : pd.DataFrame
        Cleaned load data.

    Returns
    -------
    pd.DataFrame
        Complete feature-engineered microgrid dataset.
    """
    # Compute PV power output
    pv_power = compute_pv_power(solar_df)
    solar_df = solar_df.copy()
    solar_df["pv_power_kw"] = pv_power

    # Merge solar and load on timestamp
    df = solar_df.join(load_df, how="inner")

    # Compute net load (positive = grid import, negative = export)
    df["net_load_kw"] = df["load_kw"] - df["pv_power_kw"]

    # Add time features
    df = add_time_features(df)

    # Add lag features for key targets
    df = add_lag_features(df, "ghi", lags=[1, 2, 3, 6, 12, 24])
    df = add_lag_features(df, "load_kw", lags=[1, 2, 3, 6, 12, 24])
    df = add_lag_features(df, "pv_power_kw", lags=[1, 2, 3, 6, 12, 24])

    # Add rolling statistics
    df = add_rolling_features(df, "ghi", windows=[6, 12, 24])
    df = add_rolling_features(df, "load_kw", windows=[6, 12, 24])

    # Drop rows with NaN from lagging (first 48 hours)
    initial = len(df)
    df = df.dropna()
    print(f"[FEATURE] Built dataset: {initial} -> {len(df)} records "
          f"({df.shape[1]} features)")

    return df


def create_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> dict:
    """
    Create chronological train/validation/test split.

    Uses time-based splitting (no shuffling) to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with datetime index.
    test_ratio : float
        Fraction of data for testing (default: 0.2 = last ~2.5 months).
    val_ratio : float
        Fraction of data for validation (default: 0.1 = ~1.2 months).

    Returns
    -------
    dict
        Dictionary with 'train', 'val', 'test' DataFrames.
    """
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))

    splits = {
        "train": df.iloc[:val_start],
        "val": df.iloc[val_start:test_start],
        "test": df.iloc[test_start:],
    }

    for name, split in splits.items():
        print(f"[SPLIT] {name}: {len(split)} records "
              f"({split.index[0].date()} to {split.index[-1].date()})")

    return splits


def preprocess_and_save(year: int = 2020) -> dict:
    """
    Run full preprocessing pipeline and save processed data.

    Parameters
    ----------
    year : int
        Year of data to process.

    Returns
    -------
    dict
        Dictionary with 'full', 'train', 'val', 'test' DataFrames.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Preprocessing Microgrid Data - Year {year}")
    print("=" * 60)

    # Load raw data
    solar_df, load_df = load_raw_data(year)

    # Clean
    solar_df = clean_data(solar_df, name="solar")
    load_df = clean_data(load_df, name="load")

    # Build feature-engineered dataset
    df = build_microgrid_dataset(solar_df, load_df)

    # Train/val/test split
    splits = create_train_test_split(df)

    # Save processed data
    df.to_csv(PROCESSED_DATA_DIR / f"microgrid_{year}.csv")
    splits["train"].to_csv(PROCESSED_DATA_DIR / f"train_{year}.csv")
    splits["val"].to_csv(PROCESSED_DATA_DIR / f"val_{year}.csv")
    splits["test"].to_csv(PROCESSED_DATA_DIR / f"test_{year}.csv")

    print(f"\n[INFO] Saved processed data to {PROCESSED_DATA_DIR}")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Features: {list(df.columns)}")

    return {"full": df, **splits}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess microgrid data")
    parser.add_argument("--year", type=int, default=2020, help="Data year (default: 2020)")
    args = parser.parse_args()

    preprocess_and_save(args.year)
