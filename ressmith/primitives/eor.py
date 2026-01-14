"""Enhanced Oil Recovery (EOR) pattern analysis.

This module provides functions for analyzing waterflood patterns,
injection efficiency, and sweep efficiency.

References:
- Craig, F.F., "The Reservoir Engineering Aspects of Waterflooding," SPE Monograph, 1971.
- Willhite, G.P., "Waterflooding," SPE Textbook Series, 1986.
- Dykstra, H. and Parsons, R.L., "The Prediction of Oil Recovery by Waterflood,"
  Secondary Recovery of Oil in the United States, API, 1950.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WaterfloodPatternResult:
    """Container for waterflood pattern analysis results.

    Attributes:
        pattern_type: Pattern type ('five_spot', 'line_drive', 'peripheral')
        injection_rate: Injection rate (STB/day)
        production_rate: Production rate (STB/day)
        sweep_efficiency: Areal sweep efficiency (fraction)
        displacement_efficiency: Displacement efficiency (fraction)
        recovery_efficiency: Overall recovery efficiency (fraction)
        injection_efficiency: Injection efficiency (fraction)
    """

    pattern_type: str
    injection_rate: float
    production_rate: float
    sweep_efficiency: float
    displacement_efficiency: float
    recovery_efficiency: float
    injection_efficiency: float


def calculate_areal_sweep_efficiency(
    pattern_type: Literal["five_spot", "line_drive", "peripheral"],
    mobility_ratio: float,
    pore_volumes_injected: float,
) -> float:
    """Calculate areal sweep efficiency for waterflood pattern.

    Args:
        pattern_type: Pattern type ('five_spot', 'line_drive', 'peripheral')
        mobility_ratio: Mobility ratio (water/oil mobility)
        pore_volumes_injected: Pore volumes injected (fraction)

    Returns:
        Areal sweep efficiency (0-1)

    Reference:
        Craig, F.F., "The Reservoir Engineering Aspects of Waterflooding," SPE Monograph, 1971.

    Example:
        >>> Ea = calculate_areal_sweep_efficiency('five_spot', mobility_ratio=2.0, pore_volumes_injected=0.5)
    """
    M = mobility_ratio
    PV_inj = min(pore_volumes_injected, 2.0)  # Cap at 2.0 PV

    if pattern_type == "five_spot":
        # Dykstra-Parsons correlation for five-spot
        # Ea ≈ 0.546 + 0.454 * (1 - exp(-0.214 * M * PV_inj))
        Ea = 0.546 + 0.454 * (1 - np.exp(-0.214 * M * PV_inj))
    elif pattern_type == "line_drive":
        # Line drive correlation
        # Ea ≈ 0.72 + 0.28 * (1 - exp(-0.18 * M * PV_inj))
        Ea = 0.72 + 0.28 * (1 - np.exp(-0.18 * M * PV_inj))
    elif pattern_type == "peripheral":
        # Peripheral injection (usually higher sweep)
        # Ea ≈ 0.85 + 0.15 * (1 - exp(-0.15 * M * PV_inj))
        Ea = 0.85 + 0.15 * (1 - np.exp(-0.15 * M * PV_inj))
    else:
        Ea = 0.546 + 0.454 * (1 - np.exp(-0.214 * M * PV_inj))

    return max(0.0, min(1.0, Ea))


def calculate_displacement_efficiency(
    oil_saturation_initial: float,
    oil_saturation_residual: float,
) -> float:
    """Calculate displacement efficiency (Buckley-Leverett).

    Args:
        oil_saturation_initial: Initial oil saturation (fraction)
        oil_saturation_residual: Residual oil saturation (fraction)

    Returns:
        Displacement efficiency (0-1)

    Reference:
        Buckley, S.E. and Leverett, M.C., "Mechanism of Fluid Displacement in Sands,"
        Trans. AIME, 1942.

    Example:
        >>> Ed = calculate_displacement_efficiency(0.70, 0.25)
    """
    Soi = oil_saturation_initial
    Sor = oil_saturation_residual

    if Soi <= 0:
        return 0.0

    Ed = (Soi - Sor) / Soi
    return max(0.0, min(1.0, Ed))


def calculate_vertical_sweep_efficiency(
    mobility_ratio: float,
    permeability_variation: float,
    pore_volumes_injected: float,
) -> float:
    """Calculate vertical sweep efficiency (Dykstra-Parsons).

    Args:
        mobility_ratio: Mobility ratio (water/oil mobility)
        permeability_variation: Permeability variation (V) from Dykstra-Parsons
        pore_volumes_injected: Pore volumes injected (fraction)

    Returns:
        Vertical sweep efficiency (0-1)

    Reference:
        Dykstra, H. and Parsons, R.L., "The Prediction of Oil Recovery by Waterflood,"
        Secondary Recovery of Oil in the United States, API, 1950.

    Example:
        >>> Ev = calculate_vertical_sweep_efficiency(mobility_ratio=2.0, permeability_variation=0.5, pore_volumes_injected=0.5)
    """
    M = mobility_ratio
    V = permeability_variation
    PV_inj = min(pore_volumes_injected, 2.0)

    # Dykstra-Parsons correlation
    Ev = 1.0 - V * (1 - np.exp(-M * PV_inj))

    return max(0.0, min(1.0, Ev))


def calculate_injection_efficiency(
    injection_rate: float,
    production_rate: float,
    voidage_replacement_ratio: float = 1.0,
) -> float:
    """Calculate injection efficiency.

    Injection efficiency = production_rate / injection_rate

    Args:
        injection_rate: Injection rate (STB/day)
        production_rate: Production rate (STB/day)
        voidage_replacement_ratio: Voidage replacement ratio (default: 1.0)

    Returns:
        Injection efficiency (fraction)

    Example:
        >>> Ei = calculate_injection_efficiency(injection_rate=1000, production_rate=800)
    """
    if injection_rate <= 0:
        return 0.0

    # Injection efficiency = production / injection (accounting for voidage)
    Ei = (production_rate / injection_rate) * voidage_replacement_ratio

    return max(0.0, min(1.0, Ei))


def analyze_waterflood_pattern(
    pattern_type: Literal["five_spot", "line_drive", "peripheral"],
    injection_rate: float,
    production_rate: float,
    mobility_ratio: float,
    oil_saturation_initial: float,
    oil_saturation_residual: float,
    pore_volumes_injected: float,
    permeability_variation: float = 0.5,
    voidage_replacement_ratio: float = 1.0,
) -> WaterfloodPatternResult:
    """Analyze waterflood pattern performance.

    Args:
        pattern_type: Pattern type ('five_spot', 'line_drive', 'peripheral')
        injection_rate: Injection rate (STB/day)
        production_rate: Production rate (STB/day)
        mobility_ratio: Mobility ratio (water/oil mobility)
        oil_saturation_initial: Initial oil saturation (fraction)
        oil_saturation_residual: Residual oil saturation (fraction)
        pore_volumes_injected: Pore volumes injected (fraction)
        permeability_variation: Permeability variation (V) (default: 0.5)
        voidage_replacement_ratio: Voidage replacement ratio (default: 1.0)

    Returns:
        WaterfloodPatternResult with analysis results

    Example:
        >>> result = analyze_waterflood_pattern(
        ...     pattern_type='five_spot',
        ...     injection_rate=1000,
        ...     production_rate=800,
        ...     mobility_ratio=2.0,
        ...     oil_saturation_initial=0.70,
        ...     oil_saturation_residual=0.25,
        ...     pore_volumes_injected=0.5
        ... )
    """
    # Calculate areal sweep efficiency
    Ea = calculate_areal_sweep_efficiency(pattern_type, mobility_ratio, pore_volumes_injected)

    # Calculate displacement efficiency
    Ed = calculate_displacement_efficiency(oil_saturation_initial, oil_saturation_residual)

    # Calculate vertical sweep efficiency
    Ev = calculate_vertical_sweep_efficiency(mobility_ratio, permeability_variation, pore_volumes_injected)

    # Overall recovery efficiency = Ea * Ed * Ev
    Er = Ea * Ed * Ev

    # Injection efficiency
    Ei = calculate_injection_efficiency(injection_rate, production_rate, voidage_replacement_ratio)

    return WaterfloodPatternResult(
        pattern_type=pattern_type,
        injection_rate=injection_rate,
        production_rate=production_rate,
        sweep_efficiency=Ea,
        displacement_efficiency=Ed,
        recovery_efficiency=Er,
        injection_efficiency=Ei,
    )


def calculate_mobility_ratio(
    water_viscosity: float,
    oil_viscosity: float,
    water_relative_permeability: float = 1.0,
    oil_relative_permeability: float = 1.0,
) -> float:
    """Calculate mobility ratio.

    M = (krw / μw) / (kro / μo)

    Args:
        water_viscosity: Water viscosity (cp)
        oil_viscosity: Oil viscosity (cp)
        water_relative_permeability: Water relative permeability (default: 1.0)
        oil_relative_permeability: Oil relative permeability (default: 1.0)

    Returns:
        Mobility ratio (dimensionless)

    Example:
        >>> M = calculate_mobility_ratio(water_viscosity=0.5, oil_viscosity=2.0)
    """
    if oil_viscosity <= 0 or water_viscosity <= 0:
        return 1.0

    M = (water_relative_permeability / water_viscosity) / (
        oil_relative_permeability / oil_viscosity
    )

    return max(0.01, min(100.0, M))  # Reasonable bounds

