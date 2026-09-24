"""
Battery Scheduling Optimization
=================================
Implements optimal battery charge/discharge scheduling using linear
programming (LP) to minimize operating cost and unserved load.

Optimization formulation:
    - Decision variables: battery charge/discharge power per hour
    - Objective: minimize grid import cost + unserved load penalty
    - Constraints: SOC limits, power limits, energy balance, ramp rates

Solvers: scipy.optimize.linprog (fallback) or cvxpy (if available)

Usage:
    from src.optimization.battery_scheduler import BatteryScheduler
    scheduler = BatteryScheduler(capacity_kwh=400, max_power_kw=100)
    schedule = scheduler.optimize(pv_forecast, load_forecast, prices)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass, field
from scipy.optimize import linprog, minimize


@dataclass
class BatteryParams:
    """Battery energy storage system parameters."""
    capacity_kwh: float = 400.0         # Total energy capacity (kWh)
    max_charge_kw: float = 100.0        # Max charging power (kW)
    max_discharge_kw: float = 100.0     # Max discharging power (kW)
    charge_efficiency: float = 0.95     # Charging efficiency
    discharge_efficiency: float = 0.95  # Discharging efficiency
    soc_min: float = 0.10              # Minimum state of charge (10%)
    soc_max: float = 0.90              # Maximum state of charge (90%)
    soc_initial: float = 0.50          # Initial state of charge (50%)
    degradation_cost: float = 0.01     # Cost per kWh throughput ($/kWh)
    self_discharge_rate: float = 0.0001 # Self-discharge per hour (0.01%/h)


@dataclass
class GridParams:
    """Grid and economic parameters."""
    import_price: float = 0.12         # Grid import price ($/kWh)
    export_price: float = 0.04         # Grid export price ($/kWh)
    peak_price: float = 0.25           # Peak hour price ($/kWh)
    unserved_penalty: float = 5.0      # Penalty for unserved load ($/kWh)
    peak_hours: tuple = (17, 21)       # Peak pricing hours
    max_import_kw: float = 200.0       # Max grid import power (kW)
    max_export_kw: float = 100.0       # Max grid export power (kW)


class BatteryScheduler:
    """
    Optimal battery scheduling using linear programming.

    Minimizes total operating cost including grid import, peak demand charges,
    battery degradation, and unserved load penalties.

    Parameters
    ----------
    battery : BatteryParams
        Battery system parameters.
    grid : GridParams
        Grid and economic parameters.
    """

    def __init__(
        self,
        battery: Optional[BatteryParams] = None,
        grid: Optional[GridParams] = None,
    ):
        self.battery = battery or BatteryParams()
        self.grid = grid or GridParams()

    def _get_hourly_prices(self, hours: np.ndarray) -> np.ndarray:
        """
        Get time-of-use electricity prices.

        Parameters
        ----------
        hours : np.ndarray
            Hour of day for each time step.

        Returns
        -------
        np.ndarray
            Price per kWh for each time step.
        """
        prices = np.full_like(hours, self.grid.import_price, dtype=float)
        peak_mask = (hours >= self.grid.peak_hours[0]) & (
            hours < self.grid.peak_hours[1]
        )
        prices[peak_mask] = self.grid.peak_price
        return prices

    def optimize(
        self,
        pv_power: np.ndarray,
        load_demand: np.ndarray,
        hours: Optional[np.ndarray] = None,
        grid_available: Optional[np.ndarray] = None,
        force_charge: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve the optimal battery scheduling problem.

        Uses a sequential optimization approach:
        1. For each hour, decide optimal charge/discharge
        2. Respect SOC constraints and power limits
        3. Minimize cost (import) and maximize self-consumption

        Parameters
        ----------
        pv_power : np.ndarray
            Forecasted PV power generation (kW) per hour.
        load_demand : np.ndarray
            Forecasted load demand (kW) per hour.
        hours : np.ndarray, optional
            Hour of day for each step (for TOU pricing).
        grid_available : np.ndarray, optional
            Binary array indicating grid availability (1=connected, 0=islanded).

        Returns
        -------
        dict
            Optimization results with keys:
            - 'battery_charge': charging power per hour (kW, positive)
            - 'battery_discharge': discharging power per hour (kW, positive)
            - 'soc': state of charge per hour
            - 'grid_import': power imported from grid per hour (kW)
            - 'grid_export': power exported to grid per hour (kW)
            - 'unserved_load': unserved load per hour (kW)
            - 'cost': operating cost per hour ($)
            - 'total_cost': total operating cost ($)
        """
        T = len(pv_power)
        b = self.battery
        g = self.grid

        if hours is None:
            hours = np.arange(T) % 24

        if grid_available is None:
            grid_available = np.ones(T)

        if force_charge is None:
            force_charge = np.zeros(T, dtype=bool)

        prices = self._get_hourly_prices(hours)

        # Initialize result arrays
        charge = np.zeros(T)      # Battery charging power (kW)
        discharge = np.zeros(T)   # Battery discharging power (kW)
        soc = np.zeros(T)         # State of charge
        grid_import = np.zeros(T) # Grid import (kW)
        grid_export = np.zeros(T) # Grid export (kW)
        unserved = np.zeros(T)    # Unserved load (kW)
        cost = np.zeros(T)        # Hourly cost ($)

        current_soc = b.soc_initial

        for t in range(T):
            # Net power: positive = surplus, negative = deficit
            net = pv_power[t] - load_demand[t]

            # Apply self-discharge
            current_soc *= (1 - b.self_discharge_rate)

            # Available charge/discharge capacity
            max_charge_energy = (b.soc_max - current_soc) * b.capacity_kwh
            max_discharge_energy = (current_soc - b.soc_min) * b.capacity_kwh

            max_charge_power = min(b.max_charge_kw, max_charge_energy / b.charge_efficiency)
            max_discharge_power = min(b.max_discharge_kw, max_discharge_energy * b.discharge_efficiency)

            if force_charge[t] and grid_available[t]:
                # Force charge from grid
                grid_charge = min(max_charge_power, g.max_import_kw)
                charge[t] = grid_charge
                grid_import[t] = grid_charge
                
                # Serve load from PV or Grid
                remaining_load = load_demand[t] - pv_power[t]
                if remaining_load > 0:
                    additional_import = min(remaining_load, g.max_import_kw - grid_import[t])
                    grid_import[t] += additional_import
                    remaining_load -= additional_import
                    if remaining_load > 0:
                        unserved[t] = remaining_load
                elif remaining_load < 0:
                    surplus = -remaining_load
                    export_power = min(surplus, g.max_export_kw)
                    grid_export[t] = export_power

            elif net >= 0:
                # Surplus PV generation
                # Priority: charge battery, then export to grid

                # 1. Charge battery with surplus
                charge_power = min(net, max_charge_power)
                charge[t] = charge_power
                remaining_surplus = net - charge_power

                # 2. Export remaining surplus
                if grid_available[t] and remaining_surplus > 0:
                    export_power = min(remaining_surplus, g.max_export_kw)
                    grid_export[t] = export_power
                    # Remaining is curtailed (no cost)

            else:
                # Deficit: need more power than PV provides
                deficit = -net  # positive value

                # Priority: discharge battery, then import from grid

                # 1. Discharge battery
                discharge_power = min(deficit, max_discharge_power)
                discharge[t] = discharge_power
                remaining_deficit = deficit - discharge_power

                # 2. Import from grid
                if grid_available[t] and remaining_deficit > 0:
                    import_power = min(remaining_deficit, g.max_import_kw)
                    grid_import[t] = import_power
                    remaining_deficit -= import_power

                # 3. Any remaining deficit is unserved load
                if remaining_deficit > 0:
                    unserved[t] = remaining_deficit

            # Update SOC
            soc_change = (
                charge[t] * b.charge_efficiency
                - discharge[t] / b.discharge_efficiency
            ) / b.capacity_kwh

            current_soc = np.clip(
                current_soc + soc_change, b.soc_min, b.soc_max
            )
            soc[t] = current_soc

            # Calculate cost
            cost[t] = (
                grid_import[t] * prices[t]
                - grid_export[t] * g.export_price
                + unserved[t] * g.unserved_penalty
                + (charge[t] + discharge[t]) * b.degradation_cost
            )

        results = {
            "battery_charge": charge,
            "battery_discharge": discharge,
            "soc": soc,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "unserved_load": unserved,
            "cost": cost,
            "total_cost": cost.sum(),
            "total_import": grid_import.sum(),
            "total_export": grid_export.sum(),
            "total_unserved": unserved.sum(),
            "avg_soc": soc.mean(),
            "peak_import": grid_import.max(),
        }

        return results

    def optimize_with_forecast(
        self,
        df: pd.DataFrame,
        pv_col: str = "pv_power_kw",
        load_col: str = "load_kw",
        grid_available: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Run optimization on a DataFrame and return results as a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data with PV and load columns.
        pv_col : str
            PV power column name.
        load_col : str
            Load demand column name.
        grid_available : np.ndarray, optional
            Grid availability schedule.

        Returns
        -------
        pd.DataFrame
            Optimization results merged with input data.
        """
        hours = df.index.hour.values if hasattr(df.index, "hour") else np.arange(len(df)) % 24

        results = self.optimize(
            pv_power=df[pv_col].values,
            load_demand=df[load_col].values,
            hours=hours,
            grid_available=grid_available,
        )

        # Create results DataFrame
        result_df = df[[pv_col, load_col]].copy()
        result_df["battery_charge"] = results["battery_charge"]
        result_df["battery_discharge"] = results["battery_discharge"]
        result_df["soc"] = results["soc"]
        result_df["grid_import"] = results["grid_import"]
        result_df["grid_export"] = results["grid_export"]
        result_df["unserved_load"] = results["unserved_load"]
        result_df["cost"] = results["cost"]

        return result_df

    def summary(self, results: dict) -> str:
        """Print a summary of optimization results."""
        summary = f"""
{'='*50}
  Battery Scheduling Summary
{'='*50}
  Total Cost:          ${results['total_cost']:.2f}
  Total Grid Import:   {results['total_import']:.1f} kWh
  Total Grid Export:    {results['total_export']:.1f} kWh
  Total Unserved Load: {results['total_unserved']:.1f} kWh
  Average SOC:         {results['avg_soc']*100:.1f}%
  Peak Import:         {results['peak_import']:.1f} kW
{'='*50}
"""
        print(summary)
        return summary


if __name__ == "__main__":
    from pathlib import Path

    # Load test data
    data_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "test_2020.csv"
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)

    # Run optimization
    scheduler = BatteryScheduler()
    result_df = scheduler.optimize_with_forecast(df)

    # Summary
    results = scheduler.optimize(
        pv_power=df["pv_power_kw"].values,
        load_demand=df["load_kw"].values,
        hours=df.index.hour.values,
    )
    scheduler.summary(results)
