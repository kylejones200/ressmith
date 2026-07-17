"""Volumetric reservoir calculations (OOIP/OGIP, Darcy rate, skin, recovery).

Ported from petrosmith core/reservoir.py (volumetric and PTA helpers only;
material balance lives in material_balance.py).
"""

from __future__ import annotations

import logging
import math

from ressmith.primitives.constants import PhysicalConstants

logger = logging.getLogger(__name__)


def original_oil_in_place(
    area: float,
    net_pay: float,
    porosity: float,
    oil_saturation: float,
    formation_volume_factor: float,
) -> float:
    """Volumetric OOIP (STB).

    OOIP = 7758 * A * h * φ * So / Bo
    """
    if any(x <= 0 for x in [area, net_pay, formation_volume_factor]):
        raise ValueError("Area, net pay, and FVF must be positive")
    if not (0 < porosity <= 1):
        raise ValueError("Porosity must be between 0 and 1")
    if not (0 <= oil_saturation <= 1):
        raise ValueError("Oil saturation must be between 0 and 1")

    return (
        PhysicalConstants.BBL_PER_ACRE_FT
        * area
        * net_pay
        * porosity
        * oil_saturation
    ) / formation_volume_factor


def original_gas_in_place(
    area: float,
    net_pay: float,
    porosity: float,
    gas_saturation: float,
    formation_volume_factor: float,
) -> float:
    """Volumetric OGIP (scf).

    OGIP = 43560 * A * h * φ * Sg / Bg
    """
    if any(x <= 0 for x in [area, net_pay, formation_volume_factor]):
        raise ValueError("Area, net pay, and FVF must be positive")
    if not (0 < porosity <= 1):
        raise ValueError("Porosity must be between 0 and 1")
    if not (0 <= gas_saturation <= 1):
        raise ValueError("Gas saturation must be between 0 and 1")

    return (
        PhysicalConstants.ACRE_TO_SQ_FT
        * area
        * net_pay
        * porosity
        * gas_saturation
    ) / formation_volume_factor


def recovery_factor(
    initial_pressure: float,
    abandonment_pressure: float,
    drive_mechanism: str,
) -> float:
    """Estimate recovery factor from drive mechanism and pressure depletion."""
    if initial_pressure <= abandonment_pressure:
        raise ValueError("Initial pressure must be greater than abandonment pressure")

    recovery_factors = {
        "solution_gas": 0.20,
        "gas_cap": 0.40,
        "water_drive": 0.50,
        "gravity_drainage": 0.60,
        "combination": 0.45,
    }
    base_rf = recovery_factors.get(drive_mechanism.lower(), 0.30)
    pressure_ratio = (initial_pressure - abandonment_pressure) / initial_pressure
    return min(base_rf * pressure_ratio, 0.70)


def darcy_flow_rate(
    permeability: float,
    thickness: float,
    pressure_drawdown: float,
    viscosity: float,
    formation_volume_factor: float,
    drainage_radius: float,
    wellbore_radius: float,
    skin_factor: float = 0.0,
) -> float:
    """Radial Darcy flow rate (STB/day or Mscf/day depending on FVF units)."""
    if any(
        x <= 0
        for x in [permeability, thickness, viscosity, formation_volume_factor]
    ):
        raise ValueError("Permeability, thickness, viscosity, and FVF must be positive")
    if pressure_drawdown < 0:
        raise ValueError("Pressure drawdown cannot be negative")
    if drainage_radius <= wellbore_radius:
        raise ValueError("Drainage radius must be greater than wellbore radius")

    numerator = (
        PhysicalConstants.DARCY_FLOW_CONSTANT
        * permeability
        * thickness
        * pressure_drawdown
    )
    denominator = viscosity * formation_volume_factor * (
        math.log(drainage_radius / wellbore_radius) + skin_factor
    )
    return numerator / denominator


def permeability_from_buildup(
    thickness: float,
    viscosity: float,
    formation_volume_factor: float,
    flow_rate: float,
    pressure_buildup_slope: float,
) -> float:
    """Permeability (md) from Horner plot slope."""
    if any(
        x <= 0
        for x in [
            thickness,
            viscosity,
            formation_volume_factor,
            flow_rate,
            pressure_buildup_slope,
        ]
    ):
        raise ValueError("All inputs must be positive")

    return (
        PhysicalConstants.LOG_SLOPE_TO_PERM
        * flow_rate
        * viscosity
        * formation_volume_factor
    ) / (thickness * pressure_buildup_slope)


def skin_factor_from_buildup(
    shut_in_pressure: float,
    flowing_pressure_1hr: float,
    horner_slope: float,
    permeability: float,
    porosity: float,
    viscosity: float,
    total_compressibility: float,
    wellbore_radius: float,
    flow_time: float = 1.0,
) -> float:
    """Skin factor from pressure buildup (Matthews-Russell form).

    S = 1.151 * [(ΔP_1hr)/m - log10(k/(φ μ ct rw²)) + 3.23]
    """
    del flow_time  # retained for API compatibility with petrosmith
    if any(
        x <= 0
        for x in [
            horner_slope,
            permeability,
            porosity,
            viscosity,
            total_compressibility,
            wellbore_radius,
        ]
    ):
        raise ValueError("All physical parameters must be positive")

    delta_p_1hr = shut_in_pressure - flowing_pressure_1hr
    log_term = math.log10(
        permeability
        / (porosity * viscosity * total_compressibility * (wellbore_radius**2))
    )
    return PhysicalConstants.SKIN_FACTOR_RATIO * (
        (delta_p_1hr / horner_slope) - log_term + 3.23
    )


def skin_factor_from_pressures(
    actual_pressure: float,
    ideal_pressure: float,
    flow_rate: float,
    permeability: float,
    thickness: float,
    viscosity: float,
    formation_volume_factor: float,
) -> float:
    """Skin from steady-state pressure difference (quick estimate)."""
    if any(
        x <= 0
        for x in [
            flow_rate,
            permeability,
            thickness,
            viscosity,
            formation_volume_factor,
        ]
    ):
        raise ValueError(
            "Flow rate, permeability, thickness, viscosity, and FVF must be positive"
        )

    delta_p_skin = ideal_pressure - actual_pressure
    return (0.00708 * permeability * thickness * delta_p_skin) / (
        flow_rate * viscosity * formation_volume_factor
    )


__all__ = [
    "original_oil_in_place",
    "original_gas_in_place",
    "recovery_factor",
    "darcy_flow_rate",
    "permeability_from_buildup",
    "skin_factor_from_buildup",
    "skin_factor_from_pressures",
]
