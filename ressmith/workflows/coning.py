"""Water and gas coning analysis workflows.

Provides workflows for analyzing coning, calculating critical rates,
and predicting breakthrough.
"""

import logging
from typing import Any

import pandas as pd

from ressmith.primitives.coning import analyze_coning

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
        f"Analyzing coning: rate={production_rate:.1f} STB/day, "
        f"method={method}"
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

