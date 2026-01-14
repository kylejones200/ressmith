"""Multi-well interaction modeling workflows.

Provides workflows for full multi-well interaction analysis including
drainage volumes, interference modeling, and spacing optimization.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.interference import estimate_drainage_radius
from ressmith.primitives.multi_well import (
    analyze_drainage_volumes,
    calculate_drainage_overlap_matrix,
    model_multi_well_interaction,
    optimize_multi_well_spacing,
)

logger = logging.getLogger(__name__)


def analyze_multi_well_interaction(
    well_locations: pd.DataFrame,
    drainage_radii: dict[str, float] | pd.Series | None = None,
    production_data: dict[str, pd.DataFrame] | None = None,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_saturation: float = 0.70,
    formation_volume_factor: float = 1.2,
) -> dict[str, Any]:
    """Analyze multi-well interaction for a field.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    drainage_radii : dict or Series, optional
        Dictionary/Series mapping well_id to drainage radius (ft)
    production_data : dict, optional
        Dictionary mapping well_id to production DataFrame
    reservoir_thickness : float
        Reservoir thickness (ft)
    porosity : float
        Porosity (fraction)
    oil_saturation : float
        Oil saturation (fraction)
    formation_volume_factor : float
        Oil FVF (RB/STB)

    Returns
    -------
    dict
        Dictionary with analysis results:
        - drainage_volumes: Drainage volume analysis for each well
        - interaction_matrix: Interaction analysis
        - overlap_matrix: Drainage overlap matrix
        - spacing_recommendations: Spacing optimization recommendations

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2', 'well_3'],
    ...     'latitude': [32.0, 32.001, 32.002],
    ...     'longitude': [-97.0, -97.001, -97.002]
    ... })
    >>> radii = {'well_1': 600, 'well_2': 600, 'well_3': 600}
    >>> result = analyze_multi_well_interaction(locations, drainage_radii=radii)
    >>> print(f"Total overlap: {result['interaction']['total_overlap']:.2f} acres")
    """
    logger.info(f"Analyzing multi-well interaction for {len(well_locations)} wells")

    # Convert locations to dictionary
    well_ids = well_locations["well_id"].values
    locations_dict = {
        well_id: (
            well_locations.loc[well_locations["well_id"] == well_id, "latitude"].iloc[
                0
            ],
            well_locations.loc[
                well_locations["well_id"] == well_id, "longitude"
            ].iloc[0],
        )
        for well_id in well_ids
    }

    # Get drainage radii
    if drainage_radii is None:
        if production_data:
            drainage_radii = {}
            for well_id in well_ids:
                if well_id in production_data:
                    drainage_radii[well_id] = 600.0
        else:
            drainage_radii = {well_id: 600.0 for well_id in well_ids}
    elif isinstance(drainage_radii, pd.Series):
        drainage_radii = drainage_radii.to_dict()

    # Get production rates if available
    production_rates = None
    if production_data:
        production_rates = {
            well_id: df["oil"].mean() if "oil" in df.columns else df.iloc[:, 0].mean()
            for well_id, df in production_data.items()
            if well_id in well_ids
        }

    # Analyze drainage volumes
    drainage_volumes = analyze_drainage_volumes(
        list(well_ids),
        drainage_radii,
        reservoir_thickness,
        porosity,
        oil_saturation,
        formation_volume_factor,
    )

    # Model multi-well interaction
    interaction = model_multi_well_interaction(
        locations_dict, drainage_radii, production_rates
    )

    # Calculate overlap matrix
    overlap_matrix = calculate_drainage_overlap_matrix(
        locations_dict, drainage_radii
    )

    # Spacing optimization
    spacing_opt = optimize_multi_well_spacing(locations_dict, drainage_radii)

    logger.info("Multi-well interaction analysis completed")

    return {
        "drainage_volumes": {
            well_id: {
                "drainage_radius": vol.drainage_radius,
                "drainage_area": vol.drainage_area,
                "drainage_volume": vol.drainage_volume,
                "estimated_ooip": vol.estimated_ooip,
            }
            for well_id, vol in drainage_volumes.items()
        },
        "interaction": {
            "well_pairs": interaction.well_pairs,
            "total_overlap": interaction.total_overlap,
            "average_interference": interaction.average_interference,
        },
        "overlap_matrix": {
            f"{k[0]}_{k[1]}": v for k, v in overlap_matrix.items()
        },
        "spacing_recommendations": spacing_opt,
    }


def optimize_field_spacing(
    well_locations: pd.DataFrame,
    drainage_radii: dict[str, float] | pd.Series,
    target_interference: float = 5.0,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
) -> dict[str, Any]:
    """Optimize well spacing for a field.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    drainage_radii : dict or Series
        Dictionary/Series mapping well_id to drainage radius (ft)
    target_interference : float
        Target interference percentage (default: 5%)
    min_spacing : float
        Minimum spacing constraint (ft)
    max_spacing : float
        Maximum spacing constraint (ft)

    Returns
    -------
    dict
        Dictionary with spacing optimization results

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2'],
    ...     'latitude': [32.0, 32.001],
    ...     'longitude': [-97.0, -97.001]
    ... })
    >>> radii = {'well_1': 600, 'well_2': 600}
    >>> result = optimize_field_spacing(locations, radii)
    >>> print(f"Recommended spacing: {result['recommended_spacing']:.0f} ft")
    """
    logger.info("Optimizing field spacing")

    # Convert locations to dictionary
    well_ids = well_locations["well_id"].values
    locations_dict = {
        well_id: (
            well_locations.loc[well_locations["well_id"] == well_id, "latitude"].iloc[
                0
            ],
            well_locations.loc[
                well_locations["well_id"] == well_id, "longitude"
            ].iloc[0],
        )
        for well_id in well_ids
    }

    if isinstance(drainage_radii, pd.Series):
        drainage_radii = drainage_radii.to_dict()

    result = optimize_multi_well_spacing(
        locations_dict,
        drainage_radii,
        target_interference=target_interference,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )

    logger.info(f"Spacing optimization completed. Recommended: {result['recommended_spacing']:.0f} ft")

    return result

