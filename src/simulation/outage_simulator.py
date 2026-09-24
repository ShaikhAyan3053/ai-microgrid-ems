"""
Grid Outage Simulator
=======================
Generates realistic grid outage scenarios for resilience benchmarking.

Outage types:
    1. Scheduled maintenance: predictable, short duration
    2. Weather events: longer, clustered outages
    3. Equipment failure: random, varying duration
    4. Cascading failure: extended outages

Usage:
    from src.simulation.outage_simulator import OutageSimulator
    simulator = OutageSimulator()
    schedule = simulator.generate_outage_schedule(n_hours=8760)
    scenarios = simulator.generate_benchmark_scenarios(n_hours=8760)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OutageScenario:
    """Configuration for an outage scenario."""
    name: str
    description: str
    n_outages: int
    min_duration_hours: int
    max_duration_hours: int
    clustering: float = 0.0  # 0=random, 1=highly clustered
    seasonal_bias: Optional[str] = None  # 'summer', 'winter', or None


class OutageSimulator:
    """
    Grid outage simulator for resilience analysis.

    Generates realistic grid outage schedules with configurable
    frequency, duration, and clustering parameters.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_outage_schedule(
        self,
        n_hours: int,
        n_outages: int = 5,
        min_duration: int = 2,
        max_duration: int = 12,
        avoid_overlap: bool = True,
    ) -> np.ndarray:
        """
        Generate a grid availability schedule with random outages.

        Parameters
        ----------
        n_hours : int
            Total hours in simulation.
        n_outages : int
            Number of outage events.
        min_duration : int
            Minimum outage duration (hours).
        max_duration : int
            Maximum outage duration (hours).
        avoid_overlap : bool
            Prevent overlapping outages.

        Returns
        -------
        np.ndarray
            Grid availability (1=available, 0=outage).
        """
        schedule = np.ones(n_hours)
        outage_events = []

        for _ in range(n_outages):
            duration = self.rng.randint(min_duration, max_duration + 1)

            # Try to find non-overlapping slot
            max_attempts = 100
            for _ in range(max_attempts):
                start = self.rng.randint(0, max(1, n_hours - duration))
                end = start + duration

                if avoid_overlap:
                    # Check for overlap with existing outages
                    overlap = False
                    for s, e in outage_events:
                        if start < e and end > s:
                            overlap = True
                            break
                    if not overlap:
                        break
                else:
                    break

            schedule[start:end] = 0
            outage_events.append((start, end))

        return schedule

    def generate_benchmark_scenarios(
        self, n_hours: int = 8760
    ) -> Dict[str, np.ndarray]:
        """
        Generate a set of benchmark outage scenarios for comprehensive
        resilience evaluation.

        Scenarios:
        1. No outages (baseline)
        2. Minor: 3 short outages (2-4 hours each)
        3. Moderate: 5 medium outages (4-8 hours each)
        4. Severe: 3 long outages (12-24 hours each)
        5. Extreme: 2 multi-day outages (24-72 hours)
        6. Frequent: 10 short random outages
        7. Clustered: outages concentrated in one month

        Parameters
        ----------
        n_hours : int
            Total simulation hours (default: 8760 for 1 year).

        Returns
        -------
        dict
            Dictionary mapping scenario name to grid availability array.
        """
        scenarios = {}

        # Scenario 1: No outages (baseline)
        scenarios["S1_no_outage"] = np.ones(n_hours)

        # Scenario 2: Minor outages
        self.rng = np.random.RandomState(self.seed)
        scenarios["S2_minor"] = self.generate_outage_schedule(
            n_hours, n_outages=3, min_duration=2, max_duration=4
        )

        # Scenario 3: Moderate outages
        self.rng = np.random.RandomState(self.seed + 1)
        scenarios["S3_moderate"] = self.generate_outage_schedule(
            n_hours, n_outages=5, min_duration=4, max_duration=8
        )

        # Scenario 4: Severe outages
        self.rng = np.random.RandomState(self.seed + 2)
        scenarios["S4_severe"] = self.generate_outage_schedule(
            n_hours, n_outages=3, min_duration=12, max_duration=24
        )

        # Scenario 5: Extreme outages
        self.rng = np.random.RandomState(self.seed + 3)
        scenarios["S5_extreme"] = self.generate_outage_schedule(
            n_hours, n_outages=2, min_duration=24, max_duration=72
        )

        # Scenario 6: Frequent short outages
        self.rng = np.random.RandomState(self.seed + 4)
        scenarios["S6_frequent"] = self.generate_outage_schedule(
            n_hours, n_outages=10, min_duration=1, max_duration=3
        )

        # Scenario 7: Clustered outages (summer storms)
        self.rng = np.random.RandomState(self.seed + 5)
        clustered = np.ones(n_hours)
        # Place outages in summer months (hours 4000-6000)
        summer_start = 4000
        summer_end = 6000
        for _ in range(5):
            start = self.rng.randint(summer_start, summer_end - 12)
            duration = self.rng.randint(4, 12)
            clustered[start : start + duration] = 0
        scenarios["S7_clustered"] = clustered

        # Print scenario summary
        print(f"\n{'='*60}")
        print("  Outage Benchmark Scenarios")
        print(f"{'='*60}")
        for name, sched in scenarios.items():
            outage_hours = int((sched == 0).sum())
            pct = outage_hours / n_hours * 100
            print(f"  {name:20s}: {outage_hours:4d} outage hours ({pct:.2f}%)")
        print(f"{'='*60}")

        return scenarios

    def get_outage_events(
        self, schedule: np.ndarray
    ) -> List[Dict]:
        """
        Extract individual outage events from a schedule.

        Parameters
        ----------
        schedule : np.ndarray
            Grid availability schedule.

        Returns
        -------
        list
            List of outage event dicts with start, end, duration.
        """
        events = []
        in_outage = False
        start = 0

        for i, avail in enumerate(schedule):
            if avail == 0 and not in_outage:
                start = i
                in_outage = True
            elif avail == 1 and in_outage:
                events.append({
                    "start_hour": start,
                    "end_hour": i,
                    "duration_hours": i - start,
                })
                in_outage = False

        # Handle outage at end
        if in_outage:
            events.append({
                "start_hour": start,
                "end_hour": len(schedule),
                "duration_hours": len(schedule) - start,
            })

        return events


if __name__ == "__main__":
    simulator = OutageSimulator()
    scenarios = simulator.generate_benchmark_scenarios()

    # Show events for moderate scenario
    print("\nModerate scenario outage events:")
    events = simulator.get_outage_events(scenarios["S3_moderate"])
    for e in events:
        print(f"  Hour {e['start_hour']} - {e['end_hour']} "
              f"({e['duration_hours']}h)")
