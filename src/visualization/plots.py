"""
Publication-Quality Visualization Module
==========================================
Creates research-grade plots for the microgrid EMS project.

Plot Types:
    - Solar/load time series with PV power overlay
    - Forecasting results (actual vs predicted)
    - Battery SOC and dispatch profiles
    - Resilience benchmark comparison charts
    - Energy balance Sankey-style breakdown

Usage:
    from src.visualization.plots import MicrogridPlotter
    plotter = MicrogridPlotter()
    plotter.plot_energy_dispatch(result_df)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, Dict

# Publication-quality settings
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
})

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


class MicrogridPlotter:
    """
    Publication-quality plot generator for microgrid EMS results.

    Parameters
    ----------
    save_dir : Path, optional
        Directory to save figures (default: results/figures/).
    style : str
        Matplotlib style (default: 'seaborn-v0_8-whitegrid').
    """

    COLORS = {
        "pv": "#FFA500",       # Orange
        "load": "#4169E1",     # Royal Blue
        "battery": "#2E8B57",  # Sea Green
        "grid_import": "#DC143C",  # Crimson
        "grid_export": "#9370DB",  # Medium Purple
        "unserved": "#FF4500",     # Orange Red
        "soc": "#20B2AA",      # Light Sea Green
        "rule_based": "#95a5a6",   # Gray
        "ai_assisted": "#2ecc71",  # Green
    }

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_solar_load_profile(
        self,
        df: pd.DataFrame,
        days: int = 7,
        start_idx: int = 0,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot solar generation and load demand profiles.

        Parameters
        ----------
        df : pd.DataFrame
            Data with 'ghi', 'pv_power_kw', 'load_kw' columns.
        days : int
            Number of days to plot.
        start_idx : int
            Starting index.
        save : bool
            Save figure to disk.
        """
        end_idx = start_idx + days * 24
        subset = df.iloc[start_idx:end_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Solar irradiance
        ax1.fill_between(
            range(len(subset)), subset["ghi"], alpha=0.3, color=self.COLORS["pv"]
        )
        ax1.plot(
            range(len(subset)), subset["ghi"],
            color=self.COLORS["pv"], linewidth=1.5, label="GHI (W/m2)"
        )
        ax1.set_ylabel("Solar Irradiance (W/m2)")
        ax1.set_title(f"Microgrid Profile - {days} Days", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")

        # PV power and load
        ax2.fill_between(
            range(len(subset)), subset["pv_power_kw"],
            alpha=0.3, color=self.COLORS["pv"]
        )
        ax2.plot(
            range(len(subset)), subset["pv_power_kw"],
            color=self.COLORS["pv"], linewidth=1.5, label="PV Power (kW)"
        )
        ax2.plot(
            range(len(subset)), subset["load_kw"],
            color=self.COLORS["load"], linewidth=1.5, label="Load (kW)"
        )
        ax2.set_ylabel("Power (kW)")
        ax2.set_xlabel("Hour")
        ax2.legend(loc="upper right")

        plt.tight_layout()

        if save:
            path = self.save_dir / "solar_load_profile.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"[PLOT] Saved: {path}")

        return fig

    def plot_forecast_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        target_name: str = "GHI",
        model_name: str = "LSTM",
        days: int = 7,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot forecasting results: actual vs predicted.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values.
        y_pred : np.ndarray
            Predicted values.
        timestamps : pd.DatetimeIndex
            Time index.
        target_name : str
            Name of target variable.
        model_name : str
            Model name.
        days : int
            Days to plot.
        save : bool
            Save figure.
        """
        n = min(days * 24, len(y_true))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        # Main plot: actual vs predicted
        ax1 = axes[0]
        ax1.plot(range(n), y_true[:n], color=self.COLORS["load"],
                 linewidth=1.5, label="Actual", alpha=0.8)
        ax1.plot(range(n), y_pred[:n], color=self.COLORS["pv"],
                 linewidth=1.5, label=f"{model_name} Forecast", linestyle="--")
        ax1.fill_between(
            range(n), y_true[:n], y_pred[:n],
            alpha=0.15, color="red"
        )
        ax1.set_ylabel(target_name)
        ax1.set_title(
            f"{model_name} {target_name} Forecast vs Actual ({days} Days)",
            fontsize=14, fontweight="bold"
        )
        ax1.legend(loc="upper right")

        # Error plot
        ax2 = axes[1]
        errors = y_true[:n] - y_pred[:n]
        ax2.bar(range(n), errors, color=np.where(errors >= 0, "#2ecc71", "#e74c3c"),
                alpha=0.6, width=1.0)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Error")
        ax2.set_xlabel("Hour")

        plt.tight_layout()

        if save:
            path = self.save_dir / f"forecast_{target_name.lower()}_{model_name.lower()}.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"[PLOT] Saved: {path}")

        return fig

    def plot_energy_dispatch(
        self,
        result_df: pd.DataFrame,
        days: int = 3,
        start_idx: int = 0,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot energy dispatch: PV, load, battery, grid, and SOC.

        Parameters
        ----------
        result_df : pd.DataFrame
            EMS dispatch results.
        days : int
            Days to plot.
        start_idx : int
            Starting index.
        save : bool
            Save figure.
        """
        end_idx = start_idx + days * 24
        subset = result_df.iloc[start_idx:end_idx]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        hours = range(len(subset))

        # Plot 1: Generation and Load
        ax1 = axes[0]
        ax1.fill_between(hours, subset["pv_power_kw"], alpha=0.3, color=self.COLORS["pv"])
        ax1.plot(hours, subset["pv_power_kw"], color=self.COLORS["pv"],
                 linewidth=1.5, label="PV Generation")
        ax1.plot(hours, subset["load_kw"], color=self.COLORS["load"],
                 linewidth=1.5, label="Load Demand")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title("Energy Dispatch Profile", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")

        # Plot 2: Battery and Grid
        ax2 = axes[1]
        ax2.bar(hours, subset["battery_charge"], color=self.COLORS["battery"],
                alpha=0.7, label="Battery Charge", width=0.8)
        ax2.bar(hours, -subset["battery_discharge"], color=self.COLORS["battery"],
                alpha=0.4, label="Battery Discharge", width=0.8)
        ax2.bar(hours, subset["grid_import"], color=self.COLORS["grid_import"],
                alpha=0.5, label="Grid Import", width=0.4)
        if "unserved_load" in subset.columns:
            ax2.bar(hours, -subset["unserved_load"], color=self.COLORS["unserved"],
                    alpha=0.7, label="Unserved Load", width=0.4)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Power (kW)")
        ax2.legend(loc="upper right", ncol=2)

        # Plot 3: SOC
        ax3 = axes[2]
        ax3.fill_between(hours, subset["soc"] * 100, alpha=0.3, color=self.COLORS["soc"])
        ax3.plot(hours, subset["soc"] * 100, color=self.COLORS["soc"],
                 linewidth=2, label="Battery SOC")
        ax3.axhline(y=90, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="SOC Limits")
        ax3.axhline(y=10, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax3.set_ylabel("SOC (%)")
        ax3.set_xlabel("Hour")
        ax3.set_ylim(0, 100)
        ax3.legend(loc="upper right")

        plt.tight_layout()

        if save:
            path = self.save_dir / "energy_dispatch.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"[PLOT] Saved: {path}")

        return fig

    def plot_resilience_comparison(
        self,
        improvements_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot resilience benchmark comparison across scenarios.

        Parameters
        ----------
        improvements_df : pd.DataFrame
            Strategy improvement results.
        save : bool
            Save figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        scenarios = improvements_df["scenario"].values
        x = np.arange(len(scenarios))
        width = 0.35

        # Cost comparison
        ax1 = axes[0]
        ax1.bar(x - width/2, improvements_df["rule_cost"],
                width, label="Rule-Based", color=self.COLORS["rule_based"], alpha=0.8)
        ax1.bar(x + width/2, improvements_df["ai_cost"],
                width, label="AI-Assisted", color=self.COLORS["ai_assisted"], alpha=0.8)
        ax1.set_ylabel("Total Cost ($)")
        ax1.set_title("Operating Cost Comparison", fontsize=13, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=9)
        ax1.legend()

        # Unserved load comparison
        ax2 = axes[1]
        ax2.bar(x - width/2, improvements_df["rule_unserved"],
                width, label="Rule-Based", color=self.COLORS["rule_based"], alpha=0.8)
        ax2.bar(x + width/2, improvements_df["ai_unserved"],
                width, label="AI-Assisted", color=self.COLORS["ai_assisted"], alpha=0.8)
        ax2.set_ylabel("Unserved Load (kWh)")
        ax2.set_title("Unserved Load Comparison", fontsize=13, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=9)
        ax2.legend()

        plt.tight_layout()

        if save:
            path = self.save_dir / "resilience_comparison.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"[PLOT] Saved: {path}")

        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot forecasting model comparison (LSTM vs XGBoost).

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Model comparison metrics.
        save : bool
            Save figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = comparison_df.index.tolist()
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(models)]

        # MAE
        axes[0].bar(models, comparison_df["MAE"], color=colors, alpha=0.8)
        axes[0].set_title("MAE (Lower is Better)", fontweight="bold")
        axes[0].set_ylabel("MAE")

        # RMSE
        axes[1].bar(models, comparison_df["RMSE"], color=colors, alpha=0.8)
        axes[1].set_title("RMSE (Lower is Better)", fontweight="bold")
        axes[1].set_ylabel("RMSE")

        # Accuracy
        axes[2].bar(models, comparison_df["Accuracy (%)"], color=colors, alpha=0.8)
        axes[2].set_title("Prediction Accuracy", fontweight="bold")
        axes[2].set_ylabel("Accuracy (%)")
        axes[2].set_ylim(80, 100)

        plt.suptitle("Forecasting Model Comparison", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save:
            path = self.save_dir / "model_comparison.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"[PLOT] Saved: {path}")

        return fig


if __name__ == "__main__":
    # Quick test with processed data
    data_path = PROJECT_ROOT / "data" / "processed" / "test_2020.csv"
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    plotter = MicrogridPlotter()
    plotter.plot_solar_load_profile(df, days=7)
    print("Test plots generated!")
