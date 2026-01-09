"""
Validation functions for Layer 1 objects.

All validators raise ValueError with clear messages.
"""

import numpy as np
import pandas as pd

from ressmith.objects.domain import ProductionSeries, RateSeries


def assert_monotonic_time(time_index: pd.DatetimeIndex) -> None:
    """Assert that time index is strictly monotonic increasing."""
    if not time_index.is_monotonic_increasing:
        raise ValueError(
            f"Time index must be strictly monotonic increasing. "
            f"Found {len(time_index)} values with duplicates or decreases."
        )
    if time_index.duplicated().any():
        raise ValueError("Time index contains duplicate values")


def assert_positive_rates(rate: np.ndarray, name: str = "rate") -> None:
    """Assert that rates are non-negative."""
    if np.any(rate < 0):
        n_negative = np.sum(rate < 0)
        min_rate = np.min(rate)
        raise ValueError(
            f"{name} contains {n_negative} negative values. "
            f"Minimum value: {min_rate:.6f}"
        )


def assert_units_present(units: dict[str, str], required: list[str]) -> None:
    """Assert that required unit keys are present."""
    missing = [key for key in required if key not in units]
    if missing:
        raise ValueError(
            f"Missing required units: {missing}. "
            f"Found units: {list(units.keys())}"
        )


def assert_alignment(series: ProductionSeries) -> None:
    """Assert that all arrays in ProductionSeries are aligned."""
    n = len(series.time_index)
    arrays = {
        "oil": series.oil,
        "gas": series.gas,
        "water": series.water,
    }
    if series.choke is not None:
        arrays["choke"] = series.choke
    if series.pressure is not None:
        arrays["pressure"] = series.pressure

    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(
                f"{name} array length {len(arr)} does not match "
                f"time_index length {n}"
            )


def assert_rate_series_alignment(series: RateSeries) -> None:
    """Assert that RateSeries arrays are aligned."""
    if len(series.rate) != len(series.time_index):
        raise ValueError(
            f"rate array length {len(series.rate)} does not match "
            f"time_index length {len(series.time_index)}"
        )

