"""Well interference analysis for spacing optimization.

This module provides functions for analyzing well-to-well interference,
estimating drainage volumes, and optimizing well spacing.

References:
- Fetkovich, M.J., "Decline Curve Analysis Using Type Curves," JPT, June 1980.
- Wattenbarger, R.A., et al., "Gas Reservoir Engineering," SPE Textbook Series, 1998.
- Lee, J., et al., "Pressure Transient Testing," SPE Textbook Series, 2003.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InterferenceResult:
    """Container for well interference analysis results.

    Attributes:
        distance: Distance between wells (ft)
        interference_factor: Interference factor (dimensionless, 0-1)
        drainage_overlap: Drainage area overlap fraction (0-1)
        estimated_interference_percent: Estimated production interference (%)
    """

    distance: float
    interference_factor: float
    drainage_overlap: float
    estimated_interference_percent: float


def calculate_well_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate distance between two wells using Haversine formula.

    Args:
        lat1: Latitude of first well (degrees)
        lon1: Longitude of first well (degrees)
        lat2: Latitude of second well (degrees)
        lon2: Longitude of second well (degrees)

    Returns:
        Distance between wells (feet)

    Example:
        >>> distance = calculate_well_distance(32.0, -97.0, 32.001, -97.001)
        >>> print(f"Distance: {distance:.0f} ft")
    """
    # Earth radius in feet
    R = 20902231.0  # feet

    # Convert to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_ft = R * c
    return distance_ft


def estimate_drainage_radius(
    permeability: float,
    porosity: float,
    total_compressibility: float,
    production_time: float,
    viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
) -> float:
    """Estimate drainage radius from production time.

    Uses transient flow radius of investigation:
        r_inv ≈ sqrt(k * t / (φ * μ * ct))

    Args:
        permeability: Formation permeability (md)
        porosity: Porosity (fraction)
        total_compressibility: Total compressibility (1/psi)
        production_time: Production time (days)
        viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        Estimated drainage radius (ft)

    Reference:
        Lee, J., et al., "Pressure Transient Testing," SPE Textbook Series, 2003.

    Example:
        >>> r_drain = estimate_drainage_radius(
        ...     permeability=0.1,
        ...     porosity=0.15,
        ...     total_compressibility=1e-5,
        ...     production_time=365
        ... )
    """
    # Convert permeability to darcy
    k_darcy = permeability / 1000.0

    # Radius of investigation (feet)
    # r_inv ≈ sqrt(k * t / (φ * μ * ct))
    r_inv = np.sqrt(
        k_darcy * production_time / (porosity * viscosity * total_compressibility)
    )

    r_inv_ft = np.sqrt(
        (permeability * production_time)
        / (948.0 * porosity * viscosity * total_compressibility)
    )

    return max(10.0, min(10000.0, r_inv_ft))  # Reasonable bounds


def calculate_interference_factor(
    distance: float,
    drainage_radius_1: float,
    drainage_radius_2: float | None = None,
) -> float:
    """Calculate interference factor between two wells.

    Interference factor represents the overlap of drainage volumes.
    Uses simple geometric overlap model.

    Args:
        distance: Distance between wells (ft)
        drainage_radius_1: Drainage radius of first well (ft)
        drainage_radius_2: Drainage radius of second well (ft, optional)

    Returns:
        Interference factor (0-1, where 0 = no interference, 1 = complete overlap)

    Example:
        >>> factor = calculate_interference_factor(
        ...     distance=500,
        ...     drainage_radius_1=600,
        ...     drainage_radius_2=600
        ... )
        >>> print(f"Interference factor: {factor:.2f}")
    """
    if drainage_radius_2 is None:
        drainage_radius_2 = drainage_radius_1

    # If wells are far apart, no interference
    if distance >= (drainage_radius_1 + drainage_radius_2):
        return 0.0

    # If one well is completely inside the other
    if distance <= abs(drainage_radius_1 - drainage_radius_2):
        return 1.0

    # Calculate overlap area of two circles
    # Area of overlap = r1^2 * arccos((d^2 + r1^2 - r2^2) / (2*d*r1))
    #                 + r2^2 * arccos((d^2 + r2^2 - r1^2) / (2*d*r2))
    #                 - 0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    r1 = drainage_radius_1
    r2 = drainage_radius_2
    d = distance

    # Calculate overlap area
    term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * np.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )

    overlap_area = term1 + term2 - term3

    # Area of smaller circle
    min_area = np.pi * min(r1, r2) ** 2

    # Interference factor = overlap / min_area
    if min_area > 0:
        interference = overlap_area / min_area
    else:
        interference = 0.0

    return max(0.0, min(1.0, interference))  # Clamp to [0, 1]


def estimate_production_interference(
    distance: float,
    drainage_radius_1: float,
    drainage_radius_2: float | None = None,
    method: str = "geometric",
) -> float:
    """Estimate production interference percentage.

    Estimates the percentage reduction in production due to well interference.

    Args:
        distance: Distance between wells (ft)
        drainage_radius_1: Drainage radius of first well (ft)
        drainage_radius_2: Drainage radius of second well (ft, optional)
        method: Interference model ('geometric' or 'exponential')

    Returns:
        Estimated production interference (0-100%)

    Example:
        >>> interference = estimate_production_interference(
        ...     distance=500,
        ...     drainage_radius_1=600
        ... )
        >>> print(f"Interference: {interference:.1f}%")
    """
    if method == "geometric":
        interference_factor = calculate_interference_factor(
            distance, drainage_radius_1, drainage_radius_2
        )
        interference_percent = interference_factor * 30.0  # Max 30% interference
    elif method == "exponential":
        # Exponential decay model
        if drainage_radius_2 is None:
            drainage_radius_2 = drainage_radius_1
        avg_radius = (drainage_radius_1 + drainage_radius_2) / 2.0
        if distance > 0:
            interference_percent = 30.0 * np.exp(-distance / (avg_radius * 0.5))
        else:
            interference_percent = 30.0
    else:
        interference_factor = calculate_interference_factor(
            distance, drainage_radius_1, drainage_radius_2
        )
        interference_percent = interference_factor * 30.0

    return max(0.0, min(100.0, interference_percent))


def analyze_well_interference(
    distance: float,
    drainage_radius_1: float,
    drainage_radius_2: float | None = None,
    method: str = "geometric",
) -> InterferenceResult:
    """Analyze interference between two wells.

    Args:
        distance: Distance between wells (ft)
        drainage_radius_1: Drainage radius of first well (ft)
        drainage_radius_2: Drainage radius of second well (ft, optional)
        method: Interference model ('geometric' or 'exponential')

    Returns:
        InterferenceResult with analysis results

    Example:
        >>> result = analyze_well_interference(
        ...     distance=500,
        ...     drainage_radius_1=600,
        ...     drainage_radius_2=600
        ... )
        >>> print(f"Interference: {result.estimated_interference_percent:.1f}%")
    """
    if drainage_radius_2 is None:
        drainage_radius_2 = drainage_radius_1

    interference_factor = calculate_interference_factor(
        distance, drainage_radius_1, drainage_radius_2
    )

    # Calculate overlap as fraction of average drainage area
    avg_radius = (drainage_radius_1 + drainage_radius_2) / 2.0
    avg_area = np.pi * avg_radius**2

    if distance < (drainage_radius_1 + drainage_radius_2):
        overlap_area = (
            np.pi
            * min(drainage_radius_1, drainage_radius_2) ** 2
            * interference_factor
        )
        drainage_overlap = overlap_area / avg_area if avg_area > 0 else 0.0
    else:
        drainage_overlap = 0.0

    estimated_interference_percent = estimate_production_interference(
        distance, drainage_radius_1, drainage_radius_2, method=method
    )

    return InterferenceResult(
        distance=distance,
        interference_factor=interference_factor,
        drainage_overlap=drainage_overlap,
        estimated_interference_percent=estimated_interference_percent,
    )


def optimize_well_spacing(
    drainage_radius: float,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
    target_interference: float = 5.0,
) -> dict[str, Any]:
    """Recommend optimal well spacing based on drainage radius.

    Args:
        drainage_radius: Estimated drainage radius (ft)
        min_spacing: Minimum spacing constraint (ft)
        max_spacing: Maximum spacing constraint (ft)
        target_interference: Target interference percentage (default: 5%)

    Returns:
        Dictionary with recommended spacing and analysis

    Example:
        >>> result = optimize_well_spacing(
        ...     drainage_radius=600,
        ...     target_interference=5.0
        ... )
        >>> print(f"Recommended spacing: {result['recommended_spacing']:.0f} ft")
    """
    # Adjust based on target interference
    base_spacing = 2.0 * drainage_radius

    # Scale based on target interference (5% interference ~ 1.8x radius)
    if target_interference <= 5.0:
        recommended_spacing = base_spacing * 1.0
    elif target_interference <= 10.0:
        recommended_spacing = base_spacing * 0.9
    elif target_interference <= 20.0:
        recommended_spacing = base_spacing * 0.75
    else:
        recommended_spacing = base_spacing * 0.6

    # Apply constraints
    recommended_spacing = max(min_spacing, min(max_spacing, recommended_spacing))

    # Calculate expected interference at recommended spacing
    interference_result = analyze_well_interference(
        distance=recommended_spacing,
        drainage_radius_1=drainage_radius,
        drainage_radius_2=drainage_radius,
    )

    return {
        "recommended_spacing": recommended_spacing,
        "expected_interference_percent": interference_result.estimated_interference_percent,
        "interference_factor": interference_result.interference_factor,
        "drainage_overlap": interference_result.drainage_overlap,
        "analysis": interference_result,
    }


def calculate_eur_based_interference(
    eur_1: float,
    eur_2: float,
    distance: float,
    drainage_radius_1: float,
    drainage_radius_2: float | None = None,
) -> dict[str, float]:
    """Calculate interference based on EUR and spacing.

    Uses EUR to estimate interference impact on recovery.
    Higher EUR wells typically have larger drainage volumes.

    Args:
        eur_1: EUR of first well (STB or MCF)
        eur_2: EUR of second well (STB or MCF)
        distance: Distance between wells (ft)
        drainage_radius_1: Drainage radius of first well (ft)
        drainage_radius_2: Drainage radius of second well (ft, optional)

    Returns:
        Dictionary with interference metrics:
        - eur_interference_factor: EUR reduction factor (0-1)
        - estimated_eur_loss_1: Estimated EUR loss for well 1 (STB or MCF)
        - estimated_eur_loss_2: Estimated EUR loss for well 2 (STB or MCF)
        - total_eur_loss: Total EUR loss (STB or MCF)

    Reference:
        Based on empirical relationships between spacing and EUR interference
        in unconventional plays.

    Example:
        >>> result = calculate_eur_based_interference(
        ...     eur_1=500000,
        ...     eur_2=450000,
        ...     distance=500,
        ...     drainage_radius_1=600
        ... )
    """
    if drainage_radius_2 is None:
        drainage_radius_2 = drainage_radius_1

    # Calculate geometric interference
    interference_factor = calculate_interference_factor(
        distance, drainage_radius_1, drainage_radius_2
    )

    # EUR interference is typically less than geometric overlap
    # Empirical relationship: EUR interference ~ 0.6 * geometric interference
    eur_interference_factor = interference_factor * 0.6

    # Estimate EUR loss proportional to interference
    estimated_eur_loss_1 = eur_1 * eur_interference_factor
    estimated_eur_loss_2 = eur_2 * eur_interference_factor
    total_eur_loss = estimated_eur_loss_1 + estimated_eur_loss_2

    return {
        "eur_interference_factor": float(eur_interference_factor),
        "estimated_eur_loss_1": float(estimated_eur_loss_1),
        "estimated_eur_loss_2": float(estimated_eur_loss_2),
        "total_eur_loss": float(total_eur_loss),
    }


def optimize_spacing_from_eur(
    eur_values: dict[str, float],
    well_locations: dict[str, tuple[float, float]],
    drainage_radii: dict[str, float] | None = None,
    target_eur_loss_percent: float = 5.0,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
) -> dict[str, Any]:
    """Recommend optimal spacing based on EUR interference models.

    Optimizes spacing to minimize EUR loss while maintaining economic viability.

    Args:
        eur_values: Dictionary mapping well_id to EUR (STB or MCF)
        well_locations: Dictionary mapping well_id to (latitude, longitude)
        drainage_radii: Dictionary mapping well_id to drainage radius (ft)
        target_eur_loss_percent: Target maximum EUR loss percentage (default: 5%)
        min_spacing: Minimum spacing constraint (ft)
        max_spacing: Maximum spacing constraint (ft)

    Returns:
        Dictionary with spacing recommendations:
        - recommended_spacing: Recommended spacing (ft)
        - expected_eur_loss_percent: Expected EUR loss at recommended spacing
        - spacing_analysis: Analysis for each well pair

    Example:
        >>> eur_vals = {'well_1': 500000, 'well_2': 450000}
        >>> locations = {'well_1': (32.0, -97.0), 'well_2': (32.001, -97.001)}
        >>> result = optimize_spacing_from_eur(eur_vals, locations)
    """
    if drainage_radii is None:
        # Estimate drainage radius from EUR
        # Rough approximation: r_drain ~ sqrt(EUR / (π * h * φ * So / Bo))
        # Using typical values for unconventional plays
        drainage_radii = {}
        for well_id, eur in eur_values.items():
            # Rough estimate: 1 MBO ~ 10 acres drainage (typical for shale)
            drainage_area_acres = eur / 50000.0  # Rough approximation
            drainage_radius_ft = np.sqrt(drainage_area_acres * 43560.0 / np.pi)
            drainage_radii[well_id] = max(200.0, min(2000.0, drainage_radius_ft))

    well_ids = list(well_locations.keys())
    spacing_analysis = []

    # Calculate average spacing needed to achieve target EUR loss
    avg_eur = np.mean(list(eur_values.values()))
    avg_radius = np.mean(list(drainage_radii.values()))

    # Target interference factor to achieve target EUR loss
    target_interference = target_eur_loss_percent / (100.0 * 0.6)  # Reverse of EUR interference model

    # Estimate spacing from interference model
    # For two equal circles: interference = f(distance / radius)
    # Approximate: spacing ~ 2 * radius * (1 - target_interference)
    recommended_spacing = 2.0 * avg_radius * (1.0 - target_interference * 0.5)
    recommended_spacing = max(min_spacing, min(max_spacing, recommended_spacing))

    # Analyze each well pair
    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:
                lat1, lon1 = well_locations[well_id_1]
                lat2, lon2 = well_locations[well_id_2]
                distance = calculate_well_distance(lat1, lon1, lat2, lon2)

                eur_1 = eur_values.get(well_id_1, avg_eur)
                eur_2 = eur_values.get(well_id_2, avg_eur)
                r1 = drainage_radii.get(well_id_1, avg_radius)
                r2 = drainage_radii.get(well_id_2, avg_radius)

                eur_interference = calculate_eur_based_interference(
                    eur_1, eur_2, distance, r1, r2
                )

                spacing_analysis.append({
                    "well_id_1": well_id_1,
                    "well_id_2": well_id_2,
                    "distance": float(distance),
                    "eur_1": float(eur_1),
                    "eur_2": float(eur_2),
                    "eur_interference_factor": eur_interference["eur_interference_factor"],
                    "total_eur_loss": eur_interference["total_eur_loss"],
                })

    # Calculate expected EUR loss at recommended spacing
    avg_r1 = avg_radius
    avg_r2 = avg_radius
    avg_eur_1 = avg_eur
    avg_eur_2 = avg_eur

    expected_interference = calculate_interference_factor(
        recommended_spacing, avg_r1, avg_r2
    )
    expected_eur_interference = expected_interference * 0.6
    expected_eur_loss = (avg_eur_1 + avg_eur_2) * expected_eur_interference
    expected_eur_loss_percent = (expected_eur_loss / (avg_eur_1 + avg_eur_2)) * 100.0

    return {
        "recommended_spacing": float(recommended_spacing),
        "expected_eur_loss_percent": float(expected_eur_loss_percent),
        "spacing_analysis": spacing_analysis,
    }

