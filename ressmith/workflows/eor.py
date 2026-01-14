"""Enhanced Oil Recovery (EOR) pattern analysis workflows.

Provides workflows for analyzing waterflood patterns, injection efficiency,
and sweep efficiency.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from ressmith.primitives.eor import (
    analyze_waterflood_pattern,
    calculate_mobility_ratio,
)

logger = logging.getLogger(__name__)


def analyze_waterflood(
    pattern_type: Literal["five_spot", "line_drive", "peripheral"],
    injection_rate: float,
    production_rate: float,
    mobility_ratio: float,
    oil_saturation_initial: float,
    oil_saturation_residual: float,
    pore_volumes_injected: float,
    permeability_variation: float = 0.5,
    voidage_replacement_ratio: float = 1.0,
) -> dict[str, Any]:
    """Analyze waterflood pattern performance.

    Parameters
    ----------
    pattern_type : str
        Pattern type ('five_spot', 'line_drive', 'peripheral')
    injection_rate : float
        Injection rate (STB/day)
    production_rate : float
        Production rate (STB/day)
    mobility_ratio : float
        Mobility ratio (water/oil mobility)
    oil_saturation_initial : float
        Initial oil saturation (fraction)
    oil_saturation_residual : float
        Residual oil saturation (fraction)
    pore_volumes_injected : float
        Pore volumes injected (fraction)
    permeability_variation : float
        Permeability variation (V) (default: 0.5)
    voidage_replacement_ratio : float
        Voidage replacement ratio (default: 1.0)

    Returns
    -------
    dict
        Dictionary with analysis results:
        - pattern_type: Pattern type
        - sweep_efficiency: Areal sweep efficiency
        - displacement_efficiency: Displacement efficiency
        - recovery_efficiency: Overall recovery efficiency
        - injection_efficiency: Injection efficiency
        - injection_rate, production_rate: Rates

    Examples
    --------
    >>> result = analyze_waterflood(
    ...     pattern_type='five_spot',
    ...     injection_rate=1000,
    ...     production_rate=800,
    ...     mobility_ratio=2.0,
    ...     oil_saturation_initial=0.70,
    ...     oil_saturation_residual=0.25,
    ...     pore_volumes_injected=0.5
    ... )
    >>> print(f"Recovery efficiency: {result['recovery_efficiency']:.2%}")
    """
    logger.info(f"Analyzing waterflood pattern: {pattern_type}")

    result = analyze_waterflood_pattern(
        pattern_type=pattern_type,
        injection_rate=injection_rate,
        production_rate=production_rate,
        mobility_ratio=mobility_ratio,
        oil_saturation_initial=oil_saturation_initial,
        oil_saturation_residual=oil_saturation_residual,
        pore_volumes_injected=pore_volumes_injected,
        permeability_variation=permeability_variation,
        voidage_replacement_ratio=voidage_replacement_ratio,
    )

    return {
        "pattern_type": result.pattern_type,
        "injection_rate": result.injection_rate,
        "production_rate": result.production_rate,
        "sweep_efficiency": result.sweep_efficiency,
        "displacement_efficiency": result.displacement_efficiency,
        "recovery_efficiency": result.recovery_efficiency,
        "injection_efficiency": result.injection_efficiency,
    }


def calculate_mobility_ratio_workflow(
    water_viscosity: float,
    oil_viscosity: float,
    water_relative_permeability: float = 1.0,
    oil_relative_permeability: float = 1.0,
) -> float:
    """Calculate mobility ratio for waterflood analysis.

    Parameters
    ----------
    water_viscosity : float
        Water viscosity (cp)
    oil_viscosity : float
        Oil viscosity (cp)
    water_relative_permeability : float
        Water relative permeability (default: 1.0)
    oil_relative_permeability : float
        Oil relative permeability (default: 1.0)

    Returns
    -------
    float
        Mobility ratio (dimensionless)

    Examples
    --------
    >>> M = calculate_mobility_ratio_workflow(water_viscosity=0.5, oil_viscosity=2.0)
    >>> print(f"Mobility ratio: {M:.2f}")
    """
    logger.info("Calculating mobility ratio")

    return calculate_mobility_ratio(
        water_viscosity=water_viscosity,
        oil_viscosity=oil_viscosity,
        water_relative_permeability=water_relative_permeability,
        oil_relative_permeability=oil_relative_permeability,
    )

