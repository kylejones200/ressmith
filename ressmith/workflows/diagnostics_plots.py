"""Diagnostic plot workflows for RTA analysis.

Provides workflows for generating standard diagnostic plots:
- Log-log plots (rate vs time)
- Square-root time plots (for linear flow identification)
- Flow regime diagnostic plots
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_log_log_data(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Prepare data for log-log diagnostic plot.

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, rate, log_time, log_rate

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> data = prepare_log_log_data(time, rate)
    >>> print(data.head())
    """
    logger.info("Preparing log-log diagnostic plot data")

    # Convert to arrays
    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(rate, pd.Series):
        rate = rate.values

    valid_mask = (time > 0) & (rate > 0)
    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Calculate log values
    log_time = np.log10(time_valid)
    log_rate = np.log10(rate_valid)

    return pd.DataFrame(
        {
            "time": time_valid,
            "rate": rate_valid,
            "log_time": log_time,
            "log_rate": log_rate,
        }
    )


def prepare_sqrt_time_data(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Prepare data for square-root time diagnostic plot (linear flow).

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, rate, sqrt_time

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> data = prepare_sqrt_time_data(time, rate)
    >>> print(data.head())
    """
    logger.info("Preparing square-root time diagnostic plot data")

    # Convert to arrays
    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(rate, pd.Series):
        rate = rate.values

    valid_mask = (time > 0) & (rate > 0)
    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Calculate square root of time
    sqrt_time = np.sqrt(time_valid)

    return pd.DataFrame(
        {
            "time": time_valid,
            "rate": rate_valid,
            "sqrt_time": sqrt_time,
        }
    )


def calculate_flow_regime_slopes(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate slopes for flow regime identification.

    Calculates slopes for different diagnostic plots to identify flow regimes:
    - Log-log slope: identifies power-law behavior
    - Square-root time slope: identifies linear flow (fracture dominated)

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)

    Returns
    -------
    dict
        Dictionary with slopes:
        - log_log_slope: Slope of log(rate) vs log(time)
        - sqrt_time_slope: Slope of rate vs sqrt(time)
        - linear_time_slope: Slope of rate vs time

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> slopes = calculate_flow_regime_slopes(time, rate)
    >>> print(f"Log-log slope: {slopes['log_log_slope']:.2f}")
    """
    logger.info("Calculating flow regime slopes")

    # Convert to arrays
    if isinstance(time, pd.Series):
        time = time.values
    if isinstance(rate, pd.Series):
        rate = rate.values

    valid_mask = (time > 0) & (rate > 0)
    if np.sum(valid_mask) < 3:
        return {
            "log_log_slope": np.nan,
            "sqrt_time_slope": np.nan,
            "linear_time_slope": np.nan,
        }

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Log-log slope
    log_time = np.log10(time_valid)
    log_rate = np.log10(rate_valid)
    log_log_slope = float(np.polyfit(log_time, log_rate, 1)[0])

    # Square-root time slope
    sqrt_time = np.sqrt(time_valid)
    sqrt_time_slope = float(np.polyfit(sqrt_time, rate_valid, 1)[0])

    # Linear time slope
    linear_time_slope = float(np.polyfit(time_valid, rate_valid, 1)[0])

    return {
        "log_log_slope": log_log_slope,
        "sqrt_time_slope": sqrt_time_slope,
        "linear_time_slope": linear_time_slope,
    }


def identify_flow_regime_from_plots(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
) -> str:
    """Identify flow regime from diagnostic plot slopes.

    Uses diagnostic plot slopes to identify flow regime:
    - Linear flow: slope of rate vs sqrt(time) ≈ constant (fracture dominated)
    - Boundary dominated: exponential decline (log-log slope ≈ -1)
    - Bilinear flow: intermediate behavior

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)

    Returns
    -------
    str
        Identified flow regime: 'linear', 'boundary_dominated', 'bilinear', 'transient'

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> regime = identify_flow_regime_from_plots(time, rate)
    >>> print(f"Flow regime: {regime}")
    """
    logger.info("Identifying flow regime from diagnostic plots")

    slopes = calculate_flow_regime_slopes(time, rate)

    log_log_slope = slopes["log_log_slope"]
    sqrt_time_slope = slopes["sqrt_time_slope"]

    # Linear flow: q proportional to 1/sqrt(t), so q vs sqrt(t) has negative slope
    if not np.isnan(sqrt_time_slope) and sqrt_time_slope < -10:
        return "linear"

    # Boundary dominated: exponential decline (log-log slope ≈ -1)
    if not np.isnan(log_log_slope) and -1.5 < log_log_slope < -0.5:
        return "boundary_dominated"

    # Bilinear flow: intermediate slope
    if not np.isnan(log_log_slope) and -0.5 < log_log_slope < -0.2:
        return "bilinear"

    return "transient"


def generate_diagnostic_plot_data(
    time: np.ndarray | pd.Series,
    rate: np.ndarray | pd.Series,
    plot_type: str = "all",
) -> dict[str, Any]:
    """Generate data for diagnostic plots.

    Parameters
    ----------
    time : np.ndarray or pd.Series
        Production time (days)
    rate : np.ndarray or pd.Series
        Production rate (STB/day or MCF/day)
    plot_type : str
        Type of plot data to generate: 'log_log', 'sqrt_time', 'all' (default: 'all')

    Returns
    -------
    dict
        Dictionary with plot data DataFrames:
        - log_log: DataFrame for log-log plot
        - sqrt_time: DataFrame for square-root time plot
        - slopes: Dictionary with calculated slopes
        - flow_regime: Identified flow regime

    Examples
    --------
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> plot_data = generate_diagnostic_plot_data(time, rate)
    >>> print(plot_data['flow_regime'])
    """
    logger.info(f"Generating diagnostic plot data: plot_type={plot_type}")

    result: dict[str, Any] = {}

    if plot_type in ("log_log", "all"):
        result["log_log"] = prepare_log_log_data(time, rate)

    if plot_type in ("sqrt_time", "all"):
        result["sqrt_time"] = prepare_sqrt_time_data(time, rate)

    result["slopes"] = calculate_flow_regime_slopes(time, rate)
    result["flow_regime"] = identify_flow_regime_from_plots(time, rate)

    return result

