"""Pressure normalization workflows for RTA analysis.

Provides workflows for normalizing production data with pressure,
a critical step in Rate Transient Analysis (RTA).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.pressure_normalization import (
    calculate_pseudopressure,
    normalize_cumulative_with_pressure,
    normalize_for_type_curve_matching,
    normalize_rate_with_pressure,
)

logger = logging.getLogger(__name__)


def normalize_production_with_pressure(
    data: pd.DataFrame,
    rate_col: str = "oil",
    pressure_col: str = "pressure",
    initial_pressure: float | None = None,
    method: str = "pressure_ratio",
) -> pd.DataFrame:
    """Normalize production data with pressure.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with rate and pressure columns
    rate_col : str
        Rate column name (default: 'oil')
    pressure_col : str
        Pressure column name (default: 'pressure')
    initial_pressure : float, optional
        Initial reservoir pressure (if None, uses first pressure value)
    method : str
        Normalization method ('pressure_ratio', 'pseudopressure', 'material_balance')

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized rate and original columns

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'oil': [1000, 900, 800],
    ...     'pressure': [4500, 4200, 3900]
    ... })
    >>> normalized = normalize_production_with_pressure(
    ...     data, initial_pressure=5000
    ... )
    >>> print(normalized[['oil', 'normalized_rate']])
    """
    logger.info(f"Normalizing production with pressure: method={method}")

    if rate_col not in data.columns:
        raise ValueError(f"Rate column '{rate_col}' not found in data")
    if pressure_col not in data.columns:
        raise ValueError(f"Pressure column '{pressure_col}' not found in data")

    rate = data[rate_col].values
    pressure = data[pressure_col].values

    if initial_pressure is None:
        initial_pressure = float(pressure[0])
        logger.info(f"Using first pressure value as initial: {initial_pressure:.1f} psi")

    # Normalize rate
    normalized_rate = normalize_rate_with_pressure(
        rate, pressure, initial_pressure, method=method
    )

    # Create result DataFrame
    result = data.copy()
    result["normalized_rate"] = normalized_rate
    result["initial_pressure"] = initial_pressure

    return result


def normalize_for_rta_analysis(
    data: pd.DataFrame,
    time_col: str | None = None,
    rate_col: str = "oil",
    pressure_col: str | None = None,
    cumulative_col: str | None = None,
    initial_pressure: float | None = None,
) -> dict[str, Any]:
    """Normalize production data for RTA type curve matching.

    Parameters
    ----------
    data : pd.DataFrame
        Production data
    time_col : str, optional
        Time column name (if None, uses index)
    rate_col : str
        Rate column name (default: 'oil')
    pressure_col : str, optional
        Pressure column name (if None, no pressure normalization)
    cumulative_col : str, optional
        Cumulative production column name
    initial_pressure : float, optional
        Initial reservoir pressure

    Returns
    -------
    dict
        Dictionary with normalized data arrays

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'time': [1, 10, 30, 60],
    ...     'oil': [1000, 900, 800, 700],
    ...     'pressure': [4500, 4200, 3900, 3600]
    ... })
    >>> normalized = normalize_for_rta_analysis(
    ...     data, time_col='time', pressure_col='pressure', initial_pressure=5000
    ... )
    """
    logger.info("Normalizing data for RTA analysis")

    # Extract time
    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    # Extract rate
    if rate_col not in data.columns:
        raise ValueError(f"Rate column '{rate_col}' not found")
    rate = data[rate_col].values

    # Extract pressure
    pressure = None
    if pressure_col is not None:
        if pressure_col in data.columns:
            pressure = data[pressure_col].values
            if initial_pressure is None:
                initial_pressure = float(pressure[0])
        else:
            logger.warning(f"Pressure column '{pressure_col}' not found, skipping pressure normalization")

    # Extract cumulative
    cumulative = None
    if cumulative_col is not None and cumulative_col in data.columns:
        cumulative = data[cumulative_col].values

    # Normalize
    normalized = normalize_for_type_curve_matching(
        time=time,
        rate=rate,
        pressure=pressure,
        initial_pressure=initial_pressure if initial_pressure else 5000.0,
        cumulative=cumulative,
    )

    return normalized


def calculate_pseudopressure_workflow(
    data: pd.DataFrame,
    pressure_col: str = "pressure",
    temperature: float = 200.0,
    gas_gravity: float = 0.7,
) -> pd.DataFrame:
    """Calculate pseudopressure for gas production data.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with pressure column
    pressure_col : str
        Pressure column name (default: 'pressure')
    temperature : float
        Reservoir temperature (°F)
    gas_gravity : float
        Gas specific gravity (air = 1.0)

    Returns
    -------
    pd.DataFrame
        DataFrame with pseudopressure column added

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'pressure': [4000, 3500, 3000, 2500]
    ... })
    >>> result = calculate_pseudopressure_workflow(data, temperature=200)
    >>> print(result[['pressure', 'pseudopressure']])
    """
    logger.info("Calculating pseudopressure")

    if pressure_col not in data.columns:
        raise ValueError(f"Pressure column '{pressure_col}' not found")

    pressure = data[pressure_col].values
    pseudopressure = calculate_pseudopressure(
        pressure, temperature=temperature, gas_gravity=gas_gravity
    )

    result = data.copy()
    result["pseudopressure"] = pseudopressure

    return result
