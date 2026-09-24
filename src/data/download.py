"""
NREL Solar & Synthetic Load Data Downloader
============================================
Downloads solar irradiance data from the NREL National Solar Radiation Database
(NSRDB) API and generates realistic synthetic residential load profiles.

Data Sources:
    - Solar: NREL NSRDB (Physical Solar Model v3)
    - Load: Synthetic profiles based on typical residential patterns

Usage:
    python -m src.data.download --api_key YOUR_NREL_KEY --year 2020
    python -m src.data.download --use_synthetic   # No API key needed
"""

import os
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_nrel_solar(
    api_key: str,
    lat: float = 33.45,
    lon: float = -111.95,
    year: int = 2020,
    interval: int = 60,
) -> pd.DataFrame:
    """
    Download solar irradiance data from NREL NSRDB API.

    Parameters
    ----------
    api_key : str
        NREL API key (free from https://developer.nrel.gov/).
    lat : float
        Latitude of the microgrid site (default: Phoenix, AZ — high solar).
    lon : float
        Longitude of the microgrid site.
    year : int
        Year of data to download.
    interval : int
        Time interval in minutes (30 or 60).

    Returns
    -------
    pd.DataFrame
        Hourly solar irradiance data with columns: timestamp, ghi, dni, dhi,
        temperature, wind_speed.
    """
    base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

    params = {
        "api_key": api_key,
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "leap_day": "false",
        "interval": str(interval),
        "utc": "false",
        "email": "user@example.com",
        "attributes": "ghi,dni,dhi,air_temperature,wind_speed,solar_zenith_angle",
    }

    print(f"[INFO] Downloading NREL solar data for ({lat}, {lon}), year={year}...")
    response = requests.get(base_url, params=params, timeout=120)

    if response.status_code != 200:
        raise ConnectionError(
            f"NREL API returned status {response.status_code}: {response.text[:500]}"
        )

    # NSRDB CSV has 2 header rows: metadata row + column names
    lines = response.text.split("\n")
    metadata_line = lines[0]
    header_line = lines[1]

    # Parse the actual data starting from row 2
    from io import StringIO

    data_str = "\n".join(lines[1:])
    df = pd.read_csv(StringIO(data_str))

    # Build timestamp column
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]]
        .astype(str)
        .agg("-".join, axis=1),
        format="%Y-%m-%d-%H-%M",
    )

    # Rename and select relevant columns
    col_map = {
        "GHI": "ghi",
        "DNI": "dni",
        "DHI": "dhi",
        "Temperature": "temperature",
        "Wind Speed": "wind_speed",
        "Solar Zenith Angle": "solar_zenith_angle",
    }
    df = df.rename(columns=col_map)
    keep_cols = ["timestamp"] + list(col_map.values())
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()
    df = df.set_index("timestamp").sort_index()

    print(f"[INFO] Downloaded {len(df)} records ({df.index[0]} to {df.index[-1]})")
    return df


def generate_synthetic_solar(year: int = 2020, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic solar irradiance data (no API key needed).

    Uses a clear-sky GHI model with added cloud variability to produce
    realistic hourly solar profiles for a full year.

    Parameters
    ----------
    year : int
        Year to generate data for.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Hourly synthetic solar data with columns: ghi, dni, dhi, temperature,
        wind_speed.
    """
    np.random.seed(seed)
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n = len(hours)

    # Day of year and hour of day
    doy = hours.dayofyear.values
    hod = hours.hour.values + hours.minute.values / 60.0

    # Solar geometry (simplified)
    lat_rad = np.radians(33.45)  # Phoenix, AZ
    declination = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    decl_rad = np.radians(declination)

    hour_angle = np.radians(15.0 * (hod - 12.0))
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle)
    )
    cos_zenith = np.clip(cos_zenith, 0, 1)

    # Clear-sky GHI (simplified Ineichen model)
    solar_constant = 1361.0  # W/m²
    am = np.where(
        cos_zenith > 0.01, 1.0 / (cos_zenith + 0.50572 * (96.07995 - np.degrees(np.arccos(cos_zenith))) ** (-1.6364)), 0
    )
    clearness = np.exp(-0.185 * am) * 0.7
    ghi_clear = solar_constant * cos_zenith * clearness
    ghi_clear = np.clip(ghi_clear, 0, 1200)

    # Add cloud variability (realistic intermittency)
    # Cloud factor: mostly clear with occasional cloudy periods
    cloud_state = np.ones(n)
    cloud_transition = np.random.random(n)
    in_cloud = False
    for i in range(n):
        if not in_cloud and cloud_transition[i] < 0.05:
            in_cloud = True
        elif in_cloud and cloud_transition[i] < 0.15:
            in_cloud = False
        if in_cloud:
            cloud_state[i] = np.random.uniform(0.2, 0.7)

    ghi = ghi_clear * cloud_state
    ghi = np.clip(ghi, 0, None)

    # DNI and DHI from GHI using Erbs decomposition
    kt = np.where(
        ghi_clear > 10,
        ghi / (solar_constant * np.maximum(cos_zenith, 0.01)),
        0,
    )
    kt = np.clip(kt, 0, 1)

    # Erbs correlation for diffuse fraction
    kd = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )

    dhi = ghi * kd
    dni = np.where(
        cos_zenith > 0.01, (ghi - dhi) / np.maximum(cos_zenith, 0.01), 0
    )
    dni = np.clip(dni, 0, 1200)

    # Temperature: seasonal + diurnal pattern + noise
    temp_seasonal = 25 + 15 * np.sin(np.radians(360 / 365 * (doy - 100)))
    temp_diurnal = 5 * np.sin(np.radians(360 / 24 * (hod - 6)))
    temperature = temp_seasonal + temp_diurnal + np.random.normal(0, 2, n)

    # Wind speed: Weibull distribution
    wind_speed = np.random.weibull(2.0, n) * 4.5

    df = pd.DataFrame(
        {
            "ghi": np.round(ghi, 2),
            "dni": np.round(dni, 2),
            "dhi": np.round(dhi, 2),
            "temperature": np.round(temperature, 2),
            "wind_speed": np.round(wind_speed, 2),
        },
        index=hours,
    )
    df.index.name = "timestamp"

    print(f"[INFO] Generated {len(df)} synthetic solar records for {year}")
    return df


def generate_synthetic_load(year: int = 2020, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic residential microgrid load profiles.

    Models a small community microgrid (~50 homes) with seasonal, weekly, and
    diurnal patterns plus random variability.

    Parameters
    ----------
    year : int
        Year to generate data for.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Hourly load data with column: load_kw.
    """
    np.random.seed(seed + 1)
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    n = len(hours)

    doy = hours.dayofyear.values
    hod = hours.hour.values
    dow = hours.dayofweek.values  # 0=Mon, 6=Sun

    # Base load: ~150 kW average for 50-home microgrid
    base_load = 150.0

    # Seasonal variation (higher in summer due to AC, higher in winter due to heating)
    seasonal = 1.0 + 0.25 * np.cos(np.radians(360 / 365 * (doy - 200)))

    # Diurnal pattern: morning peak (7-9), evening peak (18-21), low overnight
    diurnal = np.ones(n)
    for i in range(n):
        h = hod[i]
        if 0 <= h < 6:
            diurnal[i] = 0.6 + 0.05 * h
        elif 6 <= h < 9:
            diurnal[i] = 0.9 + 0.1 * (h - 6)
        elif 9 <= h < 12:
            diurnal[i] = 1.1
        elif 12 <= h < 14:
            diurnal[i] = 1.15
        elif 14 <= h < 17:
            diurnal[i] = 1.05
        elif 17 <= h < 21:
            diurnal[i] = 1.2 + 0.05 * (h - 17)
        elif 21 <= h < 23:
            diurnal[i] = 1.1 - 0.1 * (h - 21)
        else:
            diurnal[i] = 0.8

    # Weekend factor: ~10% lower on weekends
    weekend_factor = np.where((dow >= 5), 0.9, 1.0)

    # Random variability
    noise = np.random.normal(1.0, 0.08, n)

    load_kw = base_load * seasonal * diurnal * weekend_factor * noise
    load_kw = np.clip(load_kw, 30, 400)  # Physical bounds

    df = pd.DataFrame({"load_kw": np.round(load_kw, 2)}, index=hours)
    df.index.name = "timestamp"

    print(f"[INFO] Generated {len(df)} synthetic load records for {year}")
    return df


def download_and_save(
    api_key: str = None,
    use_synthetic: bool = False,
    year: int = 2020,
    lat: float = 33.45,
    lon: float = -111.95,
) -> dict:
    """
    Main function to download/generate all microgrid data and save to CSV.

    Parameters
    ----------
    api_key : str, optional
        NREL API key. If None, synthetic data is used.
    use_synthetic : bool
        Force synthetic data generation.
    year : int
        Year of data.
    lat : float
        Site latitude.
    lon : float
        Site longitude.

    Returns
    -------
    dict
        Dictionary with 'solar' and 'load' DataFrames.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Solar data
    if api_key and not use_synthetic:
        try:
            solar_df = download_nrel_solar(api_key, lat, lon, year)
        except Exception as e:
            print(f"[WARN] NREL download failed ({e}). Falling back to synthetic data.")
            solar_df = generate_synthetic_solar(year)
    else:
        solar_df = generate_synthetic_solar(year)

    # Load data (always synthetic for now)
    load_df = generate_synthetic_load(year)

    # Save raw data
    solar_path = RAW_DATA_DIR / f"solar_{year}.csv"
    load_path = RAW_DATA_DIR / f"load_{year}.csv"

    solar_df.to_csv(solar_path)
    load_df.to_csv(load_path)

    print(f"[INFO] Saved solar data to {solar_path}")
    print(f"[INFO] Saved load data to {load_path}")

    return {"solar": solar_df, "load": load_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download NREL solar data or generate synthetic microgrid data"
    )
    parser.add_argument("--api_key", type=str, default=None, help="NREL API key")
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data (no API key needed)",
    )
    parser.add_argument("--year", type=int, default=2020, help="Data year (default: 2020)")
    parser.add_argument("--lat", type=float, default=33.45, help="Latitude (default: Phoenix, AZ)")
    parser.add_argument("--lon", type=float, default=-111.95, help="Longitude")

    args = parser.parse_args()

    if not args.api_key and not args.use_synthetic:
        print("[INFO] No API key provided. Using synthetic data generation.")
        args.use_synthetic = True

    download_and_save(
        api_key=args.api_key,
        use_synthetic=args.use_synthetic,
        year=args.year,
        lat=args.lat,
        lon=args.lon,
    )
