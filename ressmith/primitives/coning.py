"""Water and gas coning analysis for conventional reservoirs.

This module provides functions for analyzing water and gas coning,
predicting breakthrough, and calculating critical production rates.

References:
- Meyer, H.I. and Garder, A.O., "Mechanics of Two Immiscible Fluids in Porous Media,"
  JPT, November 1954.
- Chierici, G.L., Ciucci, G.M., and Pizzi, G., "A Systematic Study of Gas and
  Water Coning by Potentiometric Models," JPT, August 1964.
- Chaperon, I., "Theoretical Study of Coning Toward Horizontal and Vertical Wells
  in Anisotropic Formations: Subcritical and Critical Rates," SPE 15377, 1986.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConingResult:
    """Container for coning analysis results.

    Attributes:
        critical_rate: Critical production rate to prevent coning (STB/day)
        breakthrough_time: Estimated time to breakthrough (days)
        coning_index: Coning index (dimensionless)
        method: Method used for calculation
    """

    critical_rate: float
    breakthrough_time: float | None
    coning_index: float
    method: str


def meyer_gardner_critical_rate(
    oil_density: float,
    water_density: float,
    permeability: float,
    reservoir_thickness: float,
    well_completion_interval: float,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
) -> float:
    """Calculate critical rate using Meyer-Gardner method.

    Meyer-Gardner method for water coning in vertical wells.

    Formula:
        qc = 0.001535 * (k * h^2 * (ρw - ρo)) / (μo * Bo)

    Args:
        oil_density: Oil density (lb/ft³)
        water_density: Water density (lb/ft³)
        permeability: Permeability (md)
        reservoir_thickness: Total reservoir thickness (ft)
        well_completion_interval: Perforated interval length (ft)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        Critical production rate (STB/day)

    Reference:
        Meyer, H.I. and Garder, A.O., "Mechanics of Two Immiscible Fluids in
        Porous Media," JPT, November 1954.

    Example:
        >>> qc = meyer_gardner_critical_rate(
        ...     oil_density=50.0,
        ...     water_density=62.4,
        ...     permeability=100.0,
        ...     reservoir_thickness=50.0,
        ...     well_completion_interval=20.0
        ... )
    """
    density_diff = water_density - oil_density
    k_md = permeability

    # Meyer-Gardner formula (field units)
    # qc = 0.001535 * (k * h^2 * Δρ) / (μo * Bo)
    qc = (
        0.001535
        * k_md
        * reservoir_thickness**2
        * density_diff
        / (oil_viscosity * formation_volume_factor)
    )

    # Apply completion interval correction
    completion_factor = well_completion_interval / reservoir_thickness
    qc = qc * completion_factor

    return max(0.0, qc)


def chierici_ciucci_critical_rate(
    oil_density: float,
    water_density: float,
    gas_density: float,
    permeability: float,
    reservoir_thickness: float,
    well_completion_interval: float,
    gas_oil_contact_depth: float | None = None,
    water_oil_contact_depth: float | None = None,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
) -> float:
    """Calculate critical rate using Chierici-Ciucci method.

    Chierici-Ciucci method accounts for both gas and water coning.

    Args:
        oil_density: Oil density (lb/ft³)
        water_density: Water density (lb/ft³)
        gas_density: Gas density (lb/ft³)
        permeability: Permeability (md)
        reservoir_thickness: Total reservoir thickness (ft)
        well_completion_interval: Perforated interval length (ft)
        gas_oil_contact_depth: Depth to GOC from top (ft, optional)
        water_oil_contact_depth: Depth to WOC from top (ft, optional)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        Critical production rate (STB/day)

    Reference:
        Chierici, G.L., Ciucci, G.M., and Pizzi, G., "A Systematic Study of Gas
        and Water Coning by Potentiometric Models," JPT, August 1964.

    Example:
        >>> qc = chierici_ciucci_critical_rate(
        ...     oil_density=50.0,
        ...     water_density=62.4,
        ...     gas_density=10.0,
        ...     permeability=100.0,
        ...     reservoir_thickness=50.0,
        ...     well_completion_interval=20.0
        ... )
    """
    k_md = permeability

    # Determine which interface is closer (gas or water)
    if gas_oil_contact_depth is not None and water_oil_contact_depth is not None:
        completion_mid = reservoir_thickness / 2.0
        gas_distance = abs(gas_oil_contact_depth - completion_mid)
        water_distance = abs(water_oil_contact_depth - completion_mid)

        if gas_distance < water_distance:
            density_diff = oil_density - gas_density
            limiting_distance = gas_distance
        else:
            density_diff = water_density - oil_density
            limiting_distance = water_distance
    else:
        density_diff = water_density - oil_density
        limiting_distance = reservoir_thickness / 2.0

    # Similar to Meyer-Gardner but with distance correction
    qc = (
        0.001535
        * k_md
        * limiting_distance**2
        * density_diff
        / (oil_viscosity * formation_volume_factor)
    )

    # Apply completion interval correction
    completion_factor = well_completion_interval / reservoir_thickness
    qc = qc * completion_factor

    return max(0.0, qc)


def calculate_coning_index(
    production_rate: float,
    critical_rate: float,
) -> float:
    """Calculate coning index.

    Coning index = production_rate / critical_rate
    - < 1.0: No coning expected
    - 1.0-2.0: Moderate coning risk
    - > 2.0: High coning risk

    Args:
        production_rate: Actual production rate (STB/day)
        critical_rate: Critical rate to prevent coning (STB/day)

    Returns:
        Coning index (dimensionless)

    Example:
        >>> index = calculate_coning_index(production_rate=500, critical_rate=400)
        >>> print(f"Coning index: {index:.2f}")
    """
    if critical_rate <= 0:
        return float("inf")

    return production_rate / critical_rate


def estimate_breakthrough_time(
    production_rate: float,
    critical_rate: float,
    reservoir_thickness: float,
    porosity: float = 0.15,
    permeability: float = 100.0,
    oil_viscosity: float = 1.0,
) -> float:
    """Estimate time to water/gas breakthrough.

    Simplified breakthrough time estimation based on coning index.

    Args:
        production_rate: Actual production rate (STB/day)
        critical_rate: Critical rate to prevent coning (STB/day)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        permeability: Permeability (md)
        oil_viscosity: Oil viscosity (cp)

    Returns:
        Estimated breakthrough time (days), or None if no breakthrough expected

    Example:
        >>> t_bt = estimate_breakthrough_time(
        ...     production_rate=500,
        ...     critical_rate=400,
        ...     reservoir_thickness=50.0
        ... )
    """
    coning_index = calculate_coning_index(production_rate, critical_rate)

    if coning_index < 1.0:
        return float("inf")

    excess_rate = production_rate - critical_rate
    if excess_rate <= 0:
        return float("inf")
    C = (reservoir_thickness**2 * porosity * oil_viscosity) / (permeability * 0.1)

    t_bt = C / excess_rate
    return max(1.0, min(18250.0, t_bt))


def analyze_coning(
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
) -> ConingResult:
    """Analyze water/gas coning for a well.

    Args:
        production_rate: Actual production rate (STB/day)
        oil_density: Oil density (lb/ft³)
        water_density: Water density (lb/ft³)
        permeability: Permeability (md)
        reservoir_thickness: Total reservoir thickness (ft)
        well_completion_interval: Perforated interval length (ft)
        gas_density: Gas density (lb/ft³, optional for gas coning)
        method: Calculation method ('meyer_gardner' or 'chierici_ciucci')
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        porosity: Porosity (fraction)

    Returns:
        ConingResult with analysis results

    Example:
        >>> result = analyze_coning(
        ...     production_rate=500,
        ...     oil_density=50.0,
        ...     water_density=62.4,
        ...     permeability=100.0,
        ...     reservoir_thickness=50.0,
        ...     well_completion_interval=20.0
        ... )
        >>> print(f"Critical rate: {result.critical_rate:.1f} STB/day")
        >>> print(f"Coning index: {result.coning_index:.2f}")
    """
    if method == "meyer_gardner":
        critical_rate = meyer_gardner_critical_rate(
            oil_density=oil_density,
            water_density=water_density,
            permeability=permeability,
            reservoir_thickness=reservoir_thickness,
            well_completion_interval=well_completion_interval,
            oil_viscosity=oil_viscosity,
            formation_volume_factor=formation_volume_factor,
        )
    elif method == "chierici_ciucci":
        if gas_density is None:
            gas_density = 10.0
        critical_rate = chierici_ciucci_critical_rate(
            oil_density=oil_density,
            water_density=water_density,
            gas_density=gas_density,
            permeability=permeability,
            reservoir_thickness=reservoir_thickness,
            well_completion_interval=well_completion_interval,
            oil_viscosity=oil_viscosity,
            formation_volume_factor=formation_volume_factor,
        )
    else:
        critical_rate = meyer_gardner_critical_rate(
            oil_density=oil_density,
            water_density=water_density,
            permeability=permeability,
            reservoir_thickness=reservoir_thickness,
            well_completion_interval=well_completion_interval,
            oil_viscosity=oil_viscosity,
            formation_volume_factor=formation_volume_factor,
        )
        method = "meyer_gardner"

    coning_index = calculate_coning_index(production_rate, critical_rate)

    breakthrough_time = estimate_breakthrough_time(
        production_rate=production_rate,
        critical_rate=critical_rate,
        reservoir_thickness=reservoir_thickness,
        porosity=porosity,
        permeability=permeability,
        oil_viscosity=oil_viscosity,
    )

    if breakthrough_time == float("inf"):
        breakthrough_time = None

    return ConingResult(
        critical_rate=critical_rate,
        breakthrough_time=breakthrough_time,
        coning_index=coning_index,
        method=method,
    )

