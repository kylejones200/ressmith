"""
Preprocessing primitives for time series data.
"""

import numpy as np
import pandas as pd


def to_daily_time(time_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert time index to daily frequency."""
    return pd.date_range(start=time_index.min(), end=time_index.max(), freq="D")


def resample_to_frequency(
    time_index: pd.DatetimeIndex, frequency: str
) -> pd.DatetimeIndex:
    """
    Resample time index to specified frequency.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Input time index
    frequency : str
        Pandas frequency string (e.g., 'D', 'M', 'W')

    Returns
    -------
    pd.DatetimeIndex
        Resampled time index
    """
    return pd.date_range(start=time_index.min(), end=time_index.max(), freq=frequency)


def trim_to_start(
    time_index: pd.DatetimeIndex, start_date: pd.Timestamp
) -> pd.DatetimeIndex:
    """Trim time index to start from specified date."""
    return time_index[time_index >= start_date]


def compute_cum_from_rate(time_index: pd.DatetimeIndex, rate: np.ndarray) -> np.ndarray:
    """
    Compute cumulative from rate using trapezoidal integration.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index
    rate : np.ndarray
        Rate values

    Returns
    -------
    np.ndarray
        Cumulative values
    """
    if len(time_index) != len(rate):
        raise ValueError("time_index and rate must have same length")

    days = (time_index - time_index[0]).days.values.astype(float)
    dt = np.diff(days)
    cum = np.zeros_like(rate, dtype=float)
    cum[1:] = np.cumsum(0.5 * (rate[:-1] + rate[1:]) * dt)
    return cum


def compute_rate_from_cum(
    time_index: pd.DatetimeIndex, cumulative: np.ndarray
) -> np.ndarray:
    """
    Compute rate from cumulative using finite differences.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index
    cumulative : np.ndarray
        Cumulative values

    Returns
    -------
    np.ndarray
        Rate values
    """
    if len(time_index) != len(cumulative):
        raise ValueError("time_index and cumulative must have same length")

    days = (time_index - time_index[0]).days.values.astype(float)
    dcum = np.diff(cumulative)
    dt = np.diff(days)
    rate = np.zeros_like(cumulative, dtype=float)
    rate[1:] = dcum / dt
    rate[0] = rate[1] if len(rate) > 1 else 0.0
    return rate


def smooth_rate(
    rate: np.ndarray, window: int = 3, method: str = "median", center: bool = False
) -> np.ndarray:
    """
    Smooth rate series.

    Parameters
    ----------
    rate : np.ndarray
        Rate values
    window : int
        Rolling window size (default: 3)
    method : str
        Smoothing method: 'median' or 'mean' (default: 'median')
    center : bool
        Whether to center the window (default: False).
        WARNING: center=True uses future data and should NOT be used for forecasting.

    Returns
    -------
    np.ndarray
        Smoothed rate values
    """
    if method == "median":
        return pd.Series(rate).rolling(window=window, center=center).median().values
    elif method == "mean":
        return pd.Series(rate).rolling(window=window, center=center).mean().values
    else:
        raise ValueError(f"Unknown method: {method}")
