"""Full multi-well interaction modeling.

This module provides comprehensive multi-well interaction analysis including
drainage volumes, interference modeling, and spacing optimization.

References:
- Lee, J., et al., "Pressure Transient Testing," SPE Textbook Series, 2003.
- Fetkovich, M.J., "Decline Curve Analysis Using Type Curves," JPT, June 1980.
- Wattenbarger, R.A., et al., "Gas Reservoir Engineering," SPE Textbook Series, 1998.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ressmith.primitives.interference import (
    calculate_interference_factor,
    calculate_well_distance,
    estimate_drainage_radius,
)

logger = logging.getLogger(__name__)


@dataclass
class DrainageVolume:
    """Container for drainage volume analysis results.

    Attributes:
        well_id: Well identifier
        drainage_radius: Drainage radius (ft)
        drainage_area: Drainage area (acres)
        drainage_volume: Drainage volume (acre-ft)
        estimated_ooip: Estimated OOIP in drainage volume (STB)
    """

    well_id: str
    drainage_radius: float
    drainage_area: float
    drainage_volume: float
    estimated_ooip: float


@dataclass
class MultiWellInteraction:
    """Container for multi-well interaction analysis.

    Attributes:
        well_pairs: List of well pair interactions
        total_overlap: Total drainage overlap (acres)
        average_interference: Average interference factor
        spacing_recommendations: Recommended spacing for optimization
    """

    well_pairs: list[dict[str, Any]]
    total_overlap: float
    average_interference: float
    spacing_recommendations: dict[str, Any]


def calculate_drainage_volume(
    drainage_radius: float,
    reservoir_thickness: float,
    porosity: float = 0.15,
    oil_saturation: float = 0.70,
    formation_volume_factor: float = 1.2,
) -> dict[str, float]:
    """Calculate drainage volume from drainage radius.

    Args:
        drainage_radius: Drainage radius (ft)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_saturation: Oil saturation (fraction)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        Dictionary with drainage area, volume, and estimated OOIP

    Example:
        >>> result = calculate_drainage_volume(
        ...     drainage_radius=600,
        ...     reservoir_thickness=50.0
        ... )
        >>> print(f"Drainage area: {result['drainage_area']:.2f} acres")
    """
    drainage_area_acres = (np.pi * drainage_radius**2) / 43560.0
    drainage_volume_acre_ft = drainage_area_acres * reservoir_thickness
    ooip = (
        drainage_area_acres
        * 43560.0
        * reservoir_thickness
        * porosity
        * oil_saturation
        / formation_volume_factor
    ) / 5.615  # Convert to STB

    return {
        "drainage_area": float(drainage_area_acres),
        "drainage_volume": float(drainage_volume_acre_ft),
        "estimated_ooip": float(ooip),
    }


def calculate_drainage_overlap_matrix(
    well_locations: dict[str, tuple[float, float]],
    drainage_radii: dict[str, float],
) -> dict[tuple[str, str], float]:
    """Calculate drainage overlap matrix for all well pairs.

    Args:
        well_locations: Dictionary mapping well_id to (latitude, longitude)
        drainage_radii: Dictionary mapping well_id to drainage radius (ft)

    Returns:
        Dictionary mapping (well_id_1, well_id_2) to overlap area (acres)

    Example:
        >>> locations = {'well_1': (32.0, -97.0), 'well_2': (32.001, -97.001)}
        >>> radii = {'well_1': 600, 'well_2': 600}
        >>> overlaps = calculate_drainage_overlap_matrix(locations, radii)
    """
    overlaps: dict[tuple[str, str], float] = {}
    well_ids = list(well_locations.keys())

    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:  # Only calculate upper triangle
                lat1, lon1 = well_locations[well_id_1]
                lat2, lon2 = well_locations[well_id_2]

                distance = calculate_well_distance(lat1, lon1, lat2, lon2)
                r1 = drainage_radii.get(well_id_1, 600.0)
                r2 = drainage_radii.get(well_id_2, 600.0)

                interference_factor = calculate_interference_factor(
                    distance, r1, r2
                )

                # Calculate overlap area
                min_area = np.pi * min(r1, r2) ** 2
                overlap_area_ft2 = min_area * interference_factor
                overlap_area_acres = overlap_area_ft2 / 43560.0

                overlaps[(well_id_1, well_id_2)] = float(overlap_area_acres)

    return overlaps


def optimize_multi_well_spacing(
    well_locations: dict[str, tuple[float, float]],
    drainage_radii: dict[str, float],
    target_interference: float = 5.0,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
) -> dict[str, Any]:
    """Optimize spacing for multiple wells.

    Args:
        well_locations: Dictionary mapping well_id to (latitude, longitude)
        drainage_radii: Dictionary mapping well_id to drainage radius (ft)
        target_interference: Target interference percentage (default: 5%)
        min_spacing: Minimum spacing constraint (ft)
        max_spacing: Maximum spacing constraint (ft)

    Returns:
        Dictionary with optimization results and recommendations

    Example:
        >>> locations = {'well_1': (32.0, -97.0), 'well_2': (32.001, -97.001)}
        >>> radii = {'well_1': 600, 'well_2': 600}
        >>> result = optimize_multi_well_spacing(locations, radii)
    """
    # Calculate current overlaps
    overlaps = calculate_drainage_overlap_matrix(well_locations, drainage_radii)

    # Calculate average drainage radius
    avg_radius = np.mean(list(drainage_radii.values()))

    # Recommended spacing based on average radius
    recommended_spacing = 2.0 * avg_radius * (1.0 - target_interference / 100.0)
    recommended_spacing = max(min_spacing, min(max_spacing, recommended_spacing))

    # Calculate total overlap
    total_overlap = sum(overlaps.values())

    # Calculate average interference
    well_ids = list(well_locations.keys())
    interference_sum = 0.0
    n_pairs = 0

    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:
                lat1, lon1 = well_locations[well_id_1]
                lat2, lon2 = well_locations[well_id_2]
                distance = calculate_well_distance(lat1, lon1, lat2, lon2)
                r1 = drainage_radii.get(well_id_1, 600.0)
                r2 = drainage_radii.get(well_id_2, 600.0)

                interference_factor = calculate_interference_factor(
                    distance, r1, r2
                )
                interference_sum += interference_factor
                n_pairs += 1

    average_interference = (
        interference_sum / n_pairs if n_pairs > 0 else 0.0
    ) * 100.0  # Convert to percentage

    return {
        "recommended_spacing": float(recommended_spacing),
        "current_total_overlap": float(total_overlap),
        "average_interference": float(average_interference),
        "overlap_matrix": {
            f"{k[0]}_{k[1]}": v for k, v in overlaps.items()
        },
    }


def analyze_drainage_volumes(
    well_ids: list[str],
    drainage_radii: dict[str, float],
    reservoir_thickness: float,
    porosity: float = 0.15,
    oil_saturation: float = 0.70,
    formation_volume_factor: float = 1.2,
) -> dict[str, DrainageVolume]:
    """Analyze drainage volumes for multiple wells.

    Args:
        well_ids: List of well identifiers
        drainage_radii: Dictionary mapping well_id to drainage radius (ft)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_saturation: Oil saturation (fraction)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        Dictionary mapping well_id to DrainageVolume

    Example:
        >>> volumes = analyze_drainage_volumes(
        ...     well_ids=['well_1', 'well_2'],
        ...     drainage_radii={'well_1': 600, 'well_2': 600},
        ...     reservoir_thickness=50.0
        ... )
    """
    volumes: dict[str, DrainageVolume] = {}

    for well_id in well_ids:
        r_drain = drainage_radii.get(well_id, 600.0)
        volume_info = calculate_drainage_volume(
            r_drain, reservoir_thickness, porosity, oil_saturation, formation_volume_factor
        )

        volumes[well_id] = DrainageVolume(
            well_id=well_id,
            drainage_radius=r_drain,
            drainage_area=volume_info["drainage_area"],
            drainage_volume=volume_info["drainage_volume"],
            estimated_ooip=volume_info["estimated_ooip"],
        )

    return volumes


def model_multi_well_interaction(
    well_locations: dict[str, tuple[float, float]],
    drainage_radii: dict[str, float],
    production_rates: dict[str, float] | None = None,
) -> MultiWellInteraction:
    """Model full multi-well interaction.

    Analyzes interference, overlaps, and interactions between all well pairs.

    Args:
        well_locations: Dictionary mapping well_id to (latitude, longitude)
        drainage_radii: Dictionary mapping well_id to drainage radius (ft)
        production_rates: Optional dictionary mapping well_id to production rate

    Returns:
        MultiWellInteraction with analysis results

    Example:
        >>> locations = {'well_1': (32.0, -97.0), 'well_2': (32.001, -97.001)}
        >>> radii = {'well_1': 600, 'well_2': 600}
        >>> interaction = model_multi_well_interaction(locations, radii)
    """
    well_ids = list(well_locations.keys())
    well_pairs = []

    # Calculate overlaps
    overlaps = calculate_drainage_overlap_matrix(well_locations, drainage_radii)
    total_overlap = sum(overlaps.values())

    interference_sum = 0.0
    n_pairs = 0

    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:
                lat1, lon1 = well_locations[well_id_1]
                lat2, lon2 = well_locations[well_id_2]
                distance = calculate_well_distance(lat1, lon1, lat2, lon2)
                r1 = drainage_radii.get(well_id_1, 600.0)
                r2 = drainage_radii.get(well_id_2, 600.0)

                interference_result = calculate_interference_factor(
                    distance, r1, r2
                )
                interference_sum += interference_result
                n_pairs += 1

                overlap_area = overlaps.get((well_id_1, well_id_2), 0.0)

                pair_info = {
                    "well_id_1": well_id_1,
                    "well_id_2": well_id_2,
                    "distance": float(distance),
                    "interference_factor": float(interference_result),
                    "overlap_area": float(overlap_area),
                }

                if production_rates:
                    interference_percent = interference_result * 30.0
                    pair_info["estimated_interference_percent"] = float(
                        interference_percent
                    )

                well_pairs.append(pair_info)

    average_interference = (
        interference_sum / n_pairs if n_pairs > 0 else 0.0
    )

    # Spacing recommendations
    spacing_result = optimize_multi_well_spacing(
        well_locations, drainage_radii
    )

    return MultiWellInteraction(
        well_pairs=well_pairs,
        total_overlap=float(total_overlap),
        average_interference=float(average_interference),
        spacing_recommendations=spacing_result,
    )

