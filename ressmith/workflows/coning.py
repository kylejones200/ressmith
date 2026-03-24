"""Water and gas coning analysis workflows.

Provides workflows for analyzing coning, calculating critical rates,
and predicting breakthrough.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.coning import (
    analyze_coning,
    forecast_gor_with_breakthrough,
    forecast_water_cut_with_breakthrough,
    forecast_wor_with_breakthrough,
)

logger = logging.getLogger(__name__)


def analyze_well_coning(
    production_rate: float,
    oil_density: float,
    water_density: float,
    permeability: float,
    reservoir_thickness: float,
    well_completion_interval: float,
    gas_density: float | None = None,
    method: str = "meyer_gardner",
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    porosity: float = 0.15,
) -> dict[str, Any]:
    """Analyze water/gas coning for a well.

    Parameters
    ----------
    production_rate : float
        Actual production rate (STB/day)
    oil_density : float
        Oil density (lb/ft³)
    water_density : float
        Water density (lb/ft³)
    permeability : float
        Permeability (md)
    reservoir_thickness : float
        Total reservoir thickness (ft)
    well_completion_interval : float
        Perforated interval length (ft)
    gas_density : float, optional
        Gas density (lb/ft³, for gas coning analysis)
    method : str
        Calculation method ('meyer_gardner' or 'chierici_ciucci')
    oil_viscosity : float
        Oil viscosity (cp)
    formation_volume_factor : float
        Oil FVF (RB/STB)
    porosity : float
        Porosity (fraction)

    Returns
    -------
    dict
        Dictionary with coning analysis results:
        - critical_rate: Critical rate (STB/day)
        - breakthrough_time: Breakthrough time (days, or None)
        - coning_index: Coning index (dimensionless)
        - method: Method used
        - coning_risk: Risk level ('low', 'moderate', 'high')

    Examples
    --------
    >>> result = analyze_well_coning(
    ...     production_rate=500,
    ...     oil_density=50.0,
    ...     water_density=62.4,
    ...     permeability=100.0,
    ...     reservoir_thickness=50.0,
    ...     well_completion_interval=20.0
    ... )
    >>> print(f"Critical rate: {result['critical_rate']:.1f} STB/day")
    >>> print(f"Coning risk: {result['coning_risk']}")
    """
    logger.info(
        f"Analyzing coning: rate={production_rate:.1f} STB/day, " f"method={method}"
    )

    result = analyze_coning(
        production_rate=production_rate,
        oil_density=oil_density,
        water_density=water_density,
        permeability=permeability,
        reservoir_thickness=reservoir_thickness,
        well_completion_interval=well_completion_interval,
        gas_density=gas_density,
        method=method,
        oil_viscosity=oil_viscosity,
        formation_volume_factor=formation_volume_factor,
        porosity=porosity,
    )

    # Determine coning risk
    if result.coning_index < 1.0:
        coning_risk = "low"
    elif result.coning_index < 2.0:
        coning_risk = "moderate"
    else:
        coning_risk = "high"

    return {
        "critical_rate": result.critical_rate,
        "breakthrough_time": result.breakthrough_time,
        "coning_index": result.coning_index,
        "method": result.method,
        "coning_risk": coning_risk,
    }


def forecast_wor_gor_with_coning(
    time: pd.Series | np.ndarray,
    oil_rate: pd.Series | np.ndarray,
    production_rate: float,
    oil_density: float,
    water_density: float,
    permeability: float,
    reservoir_thickness: float,
    well_completion_interval: float,
    gas_density: float | None = None,
    method: str = "meyer_gardner",
    initial_wor: float = 0.0,
    initial_gor: float = 1000.0,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    porosity: float = 0.15,
    **kwargs: Any,
) -> pd.DataFrame:
    """Forecast WOR and GOR with breakthrough models based on coning analysis.

    Parameters
    ----------
    time : pd.Series or np.ndarray
        Time array (days)
    oil_rate : pd.Series or np.ndarray
        Oil production rate array (STB/day)
    production_rate : float
        Actual production rate (STB/day)
    oil_density : float
        Oil density (lb/ft³)
    water_density : float
        Water density (lb/ft³)
    permeability : float
        Permeability (md)
    reservoir_thickness : float
        Total reservoir thickness (ft)
    well_completion_interval : float
        Perforated interval length (ft)
    gas_density : float, optional
        Gas density (lb/ft³, for gas coning analysis)
    method : str
        Calculation method ('meyer_gardner' or 'chierici_ciucci')
    initial_wor : float
        Initial WOR before breakthrough (bbl water / bbl oil)
    initial_gor : float
        Initial GOR before breakthrough (SCF/STB)
    oil_viscosity : float
        Oil viscosity (cp)
    formation_volume_factor : float
        Oil FVF (RB/STB)
    porosity : float
        Porosity (fraction)
    **kwargs
        Additional parameters for WOR/GOR forecasting

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: time, oil_rate, wor, gor, water_cut, breakthrough_time

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = pd.Series(np.arange(0, 1000, 30))
    >>> oil_rate = pd.Series(np.full(len(time), 500))
    >>> result = forecast_wor_gor_with_coning(
    ...     time, oil_rate,
    ...     production_rate=500,
    ...     oil_density=50.0,
    ...     water_density=62.4,
    ...     permeability=100.0,
    ...     reservoir_thickness=50.0,
    ...     well_completion_interval=20.0
    ... )
    >>> print(result.head())
    """
    import numpy as np

    logger.info("Forecasting WOR/GOR with coning analysis")

    # Convert to numpy arrays if needed
    if isinstance(time, pd.Series):
        time_array = time.values
    else:
        time_array = np.asarray(time)

    if isinstance(oil_rate, pd.Series):
        oil_rate_array = oil_rate.values
    else:
        oil_rate_array = np.asarray(oil_rate)

    # Analyze coning to get breakthrough time
    coning_result = analyze_well_coning(
        production_rate=production_rate,
        oil_density=oil_density,
        water_density=water_density,
        permeability=permeability,
        reservoir_thickness=reservoir_thickness,
        well_completion_interval=well_completion_interval,
        gas_density=gas_density,
        method=method,
        oil_viscosity=oil_viscosity,
        formation_volume_factor=formation_volume_factor,
        porosity=porosity,
    )

    breakthrough_time = coning_result["breakthrough_time"]

    # Forecast WOR
    wor = forecast_wor_with_breakthrough(
        time=time_array,
        oil_rate=oil_rate_array,
        breakthrough_time=breakthrough_time,
        initial_wor=initial_wor,
        post_breakthrough_wor_slope=kwargs.get("post_breakthrough_wor_slope", 0.01),
        max_wor=kwargs.get("max_wor", 10.0),
    )

    # Forecast GOR
    gor = forecast_gor_with_breakthrough(
        time=time_array,
        oil_rate=oil_rate_array,
        breakthrough_time=breakthrough_time,
        initial_gor=initial_gor,
        post_breakthrough_gor_slope=kwargs.get("post_breakthrough_gor_slope", 50.0),
        max_gor=kwargs.get("max_gor", 50000.0),
    )

    # Forecast water cut
    water_cut = forecast_water_cut_with_breakthrough(
        time=time_array,
        oil_rate=oil_rate_array,
        breakthrough_time=breakthrough_time,
        initial_water_cut=kwargs.get("initial_water_cut", 0.0),
        post_breakthrough_water_cut_slope=kwargs.get(
            "post_breakthrough_water_cut_slope", 0.0001
        ),
        max_water_cut=kwargs.get("max_water_cut", 0.95),
    )

    # Calculate water and gas rates
    water_rate = oil_rate_array * wor
    gas_rate = oil_rate_array * gor / 1000.0  # Convert SCF to MCF

    result_df = pd.DataFrame(
        {
            "time": time_array,
            "oil_rate": oil_rate_array,
            "wor": wor,
            "gor": gor,
            "water_cut": water_cut,
            "water_rate": water_rate,
            "gas_rate": gas_rate,
            "breakthrough_time": (
                breakthrough_time if breakthrough_time is not None else np.nan
            ),
        }
    )

    return result_df
