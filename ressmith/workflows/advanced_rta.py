"""Advanced RTA workflows: Blasingame, DN, FMB type curves and complex fractures.

Provides workflows for advanced RTA analysis including Blasingame,
Duong-Nguyen (DN), and Flowing Material Balance (FMB) type curves.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.advanced_rta import (
    analyze_complex_fracture_network,
    generate_blasingame_type_curve,
    generate_fmb_type_curve,
)

logger = logging.getLogger(__name__)


def analyze_blasingame(
    data: pd.DataFrame,
    time_col: str | None = None,
    rate_col: str = "oil",
    pressure_col: str | None = None,
    initial_pressure: float = 5000.0,
) -> dict[str, Any]:
    """Analyze production data using Blasingame type curve.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    time_col : str, optional
        Time column name (if None, uses index)
    rate_col : str
        Rate column name (default: 'oil')
    pressure_col : str, optional
        Pressure column name
    initial_pressure : float
        Initial reservoir pressure (psi)

    Returns
    -------
    dict
        Dictionary with Blasingame analysis results:
        - normalized_rate: Normalized rate array
        - normalized_time: Normalized time array
        - flow_regime: Identified flow regime
        - permeability: Estimated permeability
        - drainage_area: Estimated drainage area

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90])
    >>> rate = np.array([1000, 800, 600, 500, 450])
    >>> df = pd.DataFrame({'time': time, 'oil': rate})
    >>> result = analyze_blasingame(df, time_col='time')
    >>> print(f"Flow regime: {result['flow_regime']}")
    """
    logger.info("Analyzing production data using Blasingame type curve")

    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    rate = data[rate_col].values

    pressure = None
    if pressure_col and pressure_col in data.columns:
        pressure = data[pressure_col].values

    result = generate_blasingame_type_curve(time, rate, pressure, initial_pressure)

    logger.info(f"Blasingame analysis completed. Flow regime: {result.flow_regime}")

    return {
        "normalized_rate": result.normalized_rate,
        "normalized_time": result.normalized_time,
        "flow_regime": result.flow_regime,
        "permeability": result.permeability,
        "drainage_area": result.drainage_area,
    }


def analyze_fmb(
    data: pd.DataFrame,
    time_col: str | None = None,
    cumulative_col: str = "cumulative_oil",
    pressure_col: str = "pressure",
    initial_pressure: float = 5000.0,
    formation_volume_factor: float = 1.2,
) -> dict[str, Any]:
    """Analyze production data using Flowing Material Balance (FMB) type curve.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    time_col : str, optional
        Time column name (if None, uses index)
    cumulative_col : str
        Cumulative production column name
    pressure_col : str
        Pressure column name
    initial_pressure : float
        Initial reservoir pressure (psi)
    formation_volume_factor : float
        Oil FVF (RB/STB)

    Returns
    -------
    dict
        Dictionary with FMB analysis results:
        - normalized_pressure: Normalized pressure array
        - normalized_cumulative: Normalized cumulative array
        - estimated_ooip: Estimated OOIP
        - recovery_factor: Recovery factor

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90])
    >>> cumulative = np.array([1000, 10000, 30000, 60000, 90000])
    >>> pressure = np.array([4800, 4600, 4400, 4200, 4000])
    >>> df = pd.DataFrame({'time': time, 'cumulative_oil': cumulative, 'pressure': pressure})
    >>> result = analyze_fmb(df, time_col='time')
    >>> print(f"Estimated OOIP: {result['estimated_ooip']:.0f} STB")
    """
    logger.info("Analyzing production data using FMB type curve")

    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    cumulative = data[cumulative_col].values
    pressure = data[pressure_col].values

    result = generate_fmb_type_curve(
        time, cumulative, pressure, initial_pressure, formation_volume_factor
    )

    logger.info(f"FMB analysis completed. Estimated OOIP: {result.estimated_ooip:.0f} STB")

    return {
        "normalized_pressure": result.normalized_pressure,
        "normalized_cumulative": result.normalized_cumulative,
        "estimated_ooip": result.estimated_ooip,
        "recovery_factor": result.recovery_factor,
    }


def analyze_fracture_network(
    data: pd.DataFrame,
    time_col: str | None = None,
    rate_col: str = "oil",
    number_of_stages: int = 1,
    stage_spacing: float = 300.0,
) -> dict[str, Any]:
    """Analyze complex fracture network production.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    time_col : str, optional
        Time column name (if None, uses index)
    rate_col : str
        Rate column name (default: 'oil')
    number_of_stages : int
        Number of fracture stages
    stage_spacing : float
        Spacing between stages (ft)

    Returns
    -------
    dict
        Dictionary with fracture network analysis results:
        - number_of_stages: Number of stages
        - estimated_srv: Estimated SRV (acre-ft)
        - effective_fracture_half_length: Effective fracture half-length (ft)
        - estimated_decline_rate: Estimated decline rate

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90])
    >>> rate = np.array([1000, 800, 600, 500, 450])
    >>> df = pd.DataFrame({'time': time, 'oil': rate})
    >>> result = analyze_fracture_network(df, time_col='time', number_of_stages=10)
    >>> print(f"Estimated SRV: {result['estimated_srv']:.0f} acre-ft")
    """
    logger.info(f"Analyzing fracture network: {number_of_stages} stages")

    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    rate = data[rate_col].values

    result = analyze_complex_fracture_network(
        time, rate, number_of_stages, stage_spacing
    )

    logger.info(f"Fracture network analysis completed. SRV: {result['estimated_srv']:.0f} acre-ft")

    return result

