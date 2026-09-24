"""
Microgrid Simulation Model
============================
Simulates a renewable microgrid with PV generation, battery storage,
load demand, and grid connection. Supports both grid-connected and
islanded operation modes.

Components:
    - PV Array (200 kW)
    - Battery ESS (400 kWh / 100 kW)
    - Residential Load (~50 homes)
    - Grid Connection (200 kW import / 100 kW export)

Usage:
    from src.simulation.microgrid import Microgrid
    mg = Microgrid()
    results = mg.simulate(data_df, outage_schedule)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from src.optimization.ems_controller import EMSController, MicrogridConfig
from src.optimization.battery_scheduler import BatteryParams, GridParams


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    timestep_hours: float = 1.0
    voltage_nominal: float = 480.0    # Nominal voltage (V)
    frequency_nominal: float = 60.0   # Nominal frequency (Hz)
    voltage_tolerance: float = 0.05   # +/- 5% voltage tolerance
    frequency_tolerance: float = 0.02 # +/- 2% frequency tolerance


class Microgrid:
    """
    Microgrid simulation model.

    Simulates the energy balance, battery operation, grid interaction,
    and power quality metrics for a renewable microgrid.

    Parameters
    ----------
    config : MicrogridConfig, optional
        Microgrid hardware configuration.
    sim_config : SimulationConfig, optional
        Simulation parameters.
    """

    def __init__(
        self,
        config: Optional[MicrogridConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
    ):
        self.config = config or MicrogridConfig()
        self.sim_config = sim_config or SimulationConfig()

        # Initialize EMS controllers
        self.ems_rule = EMSController(strategy="rule_based", config=self.config)
        self.ems_ai = EMSController(strategy="ai_assisted", config=self.config)

    def simulate(
        self,
        df: pd.DataFrame,
        grid_available: Optional[np.ndarray] = None,
        strategy: str = "ai_assisted",
        pv_forecast: Optional[np.ndarray] = None,
        load_forecast: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run full microgrid simulation.

        Parameters
        ----------
        df : pd.DataFrame
            Microgrid data with 'pv_power_kw' and 'load_kw'.
        grid_available : np.ndarray, optional
            Grid availability (1=connected, 0=islanded).
        strategy : str
            EMS strategy ('rule_based' or 'ai_assisted').
        pv_forecast : np.ndarray, optional
            PV forecast for AI-assisted mode.
        load_forecast : np.ndarray, optional
            Load forecast for AI-assisted mode.

        Returns
        -------
        dict
            Simulation results including dispatch, power quality, and metrics.
        """
        T = len(df)

        if grid_available is None:
            grid_available = np.ones(T)

        # Select EMS
        ems = self.ems_ai if strategy == "ai_assisted" else self.ems_rule

        # Run EMS dispatch
        result_df, dispatch = ems.run(
            df, grid_available, pv_forecast, load_forecast
        )

        # Compute power quality metrics
        voltage, frequency = self._simulate_power_quality(
            dispatch, grid_available
        )

        # Compute resilience metrics
        resilience = self._compute_resilience_metrics(
            dispatch, grid_available, df["load_kw"].values
        )

        # Add power quality to results
        result_df["voltage_pu"] = voltage
        result_df["frequency_hz"] = frequency
        result_df["grid_available"] = grid_available

        return {
            "dispatch_df": result_df,
            "dispatch": dispatch,
            "resilience": resilience,
            "strategy": strategy,
        }

    def _simulate_power_quality(
        self,
        dispatch: dict,
        grid_available: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate voltage and frequency deviations.

        During grid-connected mode: voltage and frequency are stable.
        During islanded mode: deviations depend on power balance.

        Parameters
        ----------
        dispatch : dict
            EMS dispatch results.
        grid_available : np.ndarray
            Grid availability.

        Returns
        -------
        tuple
            (voltage_pu, frequency_hz) arrays.
        """
        T = len(grid_available)
        sc = self.sim_config

        voltage = np.ones(T)       # Per-unit voltage
        frequency = np.full(T, sc.frequency_nominal)

        for t in range(T):
            if grid_available[t]:
                # Grid-connected: tight regulation
                voltage[t] = 1.0 + np.random.normal(0, 0.005)
                frequency[t] = sc.frequency_nominal + np.random.normal(0, 0.01)
            else:
                # Islanded: voltage/frequency depend on power balance
                unserved_fraction = dispatch["unserved_load"][t] / max(
                    dispatch["unserved_load"][t]
                    + dispatch["battery_discharge"][t]
                    + 1e-6,
                    1.0,
                )

                # Voltage droop with unserved load
                voltage_drop = unserved_fraction * 0.08
                voltage[t] = 1.0 - voltage_drop + np.random.normal(0, 0.01)

                # Frequency deviation (proportional to power imbalance)
                freq_deviation = unserved_fraction * 1.5
                frequency[t] = sc.frequency_nominal - freq_deviation + np.random.normal(0, 0.05)

        # Clip to physical limits
        voltage = np.clip(voltage, 0.85, 1.15)
        frequency = np.clip(frequency, 57.0, 63.0)

        return voltage, frequency

    def _compute_resilience_metrics(
        self,
        dispatch: dict,
        grid_available: np.ndarray,
        total_load: np.ndarray,
    ) -> dict:
        """
        Compute resilience metrics during outage periods.

        Parameters
        ----------
        dispatch : dict
            EMS dispatch results.
        grid_available : np.ndarray
            Grid availability.
        total_load : np.ndarray
            Total load demand.

        Returns
        -------
        dict
            Resilience metrics.
        """
        outage_mask = grid_available == 0
        n_outage_hours = outage_mask.sum()

        if n_outage_hours == 0:
            return {
                "outage_hours": 0,
                "load_served_during_outage_pct": 100.0,
                "unserved_during_outage_kwh": 0.0,
                "avg_soc_during_outage": 0.0,
                "min_soc_during_outage": 0.0,
            }

        outage_load = total_load[outage_mask]
        outage_unserved = dispatch["unserved_load"][outage_mask]
        outage_soc = dispatch["soc"][outage_mask]

        total_outage_demand = outage_load.sum()
        total_outage_unserved = outage_unserved.sum()

        load_served_pct = (
            (total_outage_demand - total_outage_unserved) / total_outage_demand * 100
            if total_outage_demand > 0
            else 100.0
        )

        return {
            "outage_hours": int(n_outage_hours),
            "load_served_during_outage_pct": round(load_served_pct, 1),
            "unserved_during_outage_kwh": round(total_outage_unserved, 1),
            "total_outage_demand_kwh": round(total_outage_demand, 1),
            "avg_soc_during_outage": round(outage_soc.mean() * 100, 1),
            "min_soc_during_outage": round(outage_soc.min() * 100, 1),
        }

    def run_comparative_study(
        self,
        df: pd.DataFrame,
        grid_available: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run both strategies and compare results.

        Parameters
        ----------
        df : pd.DataFrame
            Microgrid data.
        grid_available : np.ndarray, optional
            Grid availability schedule.

        Returns
        -------
        dict
            Comparative results.
        """
        if grid_available is None:
            grid_available = np.ones(len(df))

        # Run both strategies
        rule_results = self.simulate(df, grid_available, strategy="rule_based")
        ai_results = self.simulate(df, grid_available, strategy="ai_assisted")

        # Print comparison
        print(f"\n{'='*70}")
        print("  Microgrid Simulation - Comparative Study")
        print(f"{'='*70}")

        for label, res in [("Rule-Based", rule_results), ("AI-Assisted", ai_results)]:
            d = res["dispatch"]
            r = res["resilience"]
            print(f"\n  --- {label} ---")
            print(f"  Total Cost:            ${d['total_cost']:.2f}")
            print(f"  Total Unserved Load:   {d['total_unserved']:.1f} kWh")
            print(f"  Outage Hours:          {r['outage_hours']}")
            print(f"  Load Served (outage):  {r['load_served_during_outage_pct']:.1f}%")

        # Improvements
        if rule_results["dispatch"]["total_unserved"] > 0:
            unserved_improvement = (
                (rule_results["dispatch"]["total_unserved"]
                 - ai_results["dispatch"]["total_unserved"])
                / rule_results["dispatch"]["total_unserved"]
                * 100
            )
        else:
            unserved_improvement = 0

        cost_improvement = (
            (rule_results["dispatch"]["total_cost"]
             - ai_results["dispatch"]["total_cost"])
            / max(rule_results["dispatch"]["total_cost"], 0.01)
            * 100
        )

        print(f"\n  >> AI Improvement:")
        print(f"     Cost Reduction:          {cost_improvement:.1f}%")
        print(f"     Unserved Load Reduction: {unserved_improvement:.1f}%")
        print(f"{'='*70}")

        return {
            "rule_based": rule_results,
            "ai_assisted": ai_results,
            "cost_improvement_pct": cost_improvement,
            "unserved_improvement_pct": unserved_improvement,
        }
