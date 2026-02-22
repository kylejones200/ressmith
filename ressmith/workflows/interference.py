"""Well interference analysis workflows.

Provides workflows for analyzing well-to-well interference,
spacing optimization, and drainage volume estimation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.interference import (
    analyze_well_interference,
    calculate_eur_based_interference,
    calculate_well_distance,
    estimate_drainage_radius,
    optimize_spacing_from_eur,
    optimize_well_spacing,
)

logger = logging.getLogger(__name__)


def calculate_well_distances(
    well_locations: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate distances between all well pairs.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude

    Returns
    -------
    pd.DataFrame
        Distance matrix with well_id pairs and distances (ft)

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2', 'well_3'],
    ...     'latitude': [32.0, 32.001, 32.002],
    ...     'longitude': [-97.0, -97.001, -97.002]
    ... })
    >>> distances = calculate_well_distances(locations)
    >>> print(distances.head())
    """
    logger.info(f"Calculating distances for {len(well_locations)} wells")

    distances = []
    well_ids = well_locations["well_id"].values
    latitudes = well_locations["latitude"].values
    longitudes = well_locations["longitude"].values

    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:  # Only calculate upper triangle
                distance = calculate_well_distance(
                    latitudes[i],
                    longitudes[i],
                    latitudes[j],
                    longitudes[j],
                )
                distances.append(
                    {
                        "well_id_1": well_id_1,
                        "well_id_2": well_id_2,
                        "distance_ft": distance,
                    }
                )

    return pd.DataFrame(distances)


def analyze_interference_matrix(
    well_locations: pd.DataFrame,
    drainage_radii: dict[str, float] | pd.Series | None = None,
    default_drainage_radius: float = 600.0,
) -> pd.DataFrame:
    """Analyze interference between all well pairs.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    drainage_radii : dict or Series, optional
        Dictionary or Series mapping well_id to drainage radius (ft)
    default_drainage_radius : float
        Default drainage radius if not specified (ft)

    Returns
    -------
    pd.DataFrame
        Interference analysis results with columns:
        - well_id_1, well_id_2: Well pair identifiers
        - distance_ft: Distance between wells (ft)
        - drainage_radius_1, drainage_radius_2: Drainage radii (ft)
        - interference_factor: Interference factor (0-1)
        - drainage_overlap: Overlap fraction (0-1)
        - estimated_interference_percent: Estimated interference (%)

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2'],
    ...     'latitude': [32.0, 32.001],
    ...     'longitude': [-97.0, -97.001]
    ... })
    >>> radii = {'well_1': 600, 'well_2': 600}
    >>> interference = analyze_interference_matrix(locations, drainage_radii=radii)
    >>> print(interference)
    """
    logger.info(f"Analyzing interference for {len(well_locations)} wells")

    # Calculate distances
    distances_df = calculate_well_distances(well_locations)

    # Get drainage radii
    if drainage_radii is None:
        drainage_radii = {well_id: default_drainage_radius for well_id in well_locations["well_id"]}
    elif isinstance(drainage_radii, pd.Series):
        drainage_radii = drainage_radii.to_dict()

    # Analyze interference for each pair
    results = []
    for _, row in distances_df.iterrows():
        well_id_1 = row["well_id_1"]
        well_id_2 = row["well_id_2"]
        distance = row["distance_ft"]

        r1 = drainage_radii.get(well_id_1, default_drainage_radius)
        r2 = drainage_radii.get(well_id_2, default_drainage_radius)

        interference_result = analyze_well_interference(
            distance=distance,
            drainage_radius_1=r1,
            drainage_radius_2=r2,
        )

        results.append(
            {
                "well_id_1": well_id_1,
                "well_id_2": well_id_2,
                "distance_ft": distance,
                "drainage_radius_1": r1,
                "drainage_radius_2": r2,
                "interference_factor": interference_result.interference_factor,
                "drainage_overlap": interference_result.drainage_overlap,
                "estimated_interference_percent": interference_result.estimated_interference_percent,
            }
        )

    return pd.DataFrame(results)


def recommend_spacing(
    drainage_radius: float,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
    target_interference: float = 5.0,
) -> dict[str, Any]:
    """Recommend optimal well spacing.

    Parameters
    ----------
    drainage_radius : float
        Estimated drainage radius (ft)
    min_spacing : float
        Minimum spacing constraint (ft)
    max_spacing : float
        Maximum spacing constraint (ft)
    target_interference : float
        Target interference percentage (default: 5%)

    Returns
    -------
    dict
        Dictionary with recommended spacing and analysis

    Examples
    --------
    >>> result = recommend_spacing(
    ...     drainage_radius=600,
    ...     target_interference=5.0
    ... )
    >>> print(f"Recommended spacing: {result['recommended_spacing']:.0f} ft")
    """
    logger.info(
        f"Recommending spacing: r_drain={drainage_radius:.0f} ft, "
        f"target_interference={target_interference:.1f}%"
    )

    return optimize_well_spacing(
        drainage_radius=drainage_radius,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        target_interference=target_interference,
    )


def analyze_interference_with_production_history(
    well_locations: pd.DataFrame,
    production_data: dict[str, pd.DataFrame],
    model_name: str = "arps_hyperbolic",
    phase: str = "oil",
    **kwargs: Any,
) -> pd.DataFrame:
    """Analyze interference using production history to estimate EUR and drainage.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    production_data : dict
        Dictionary mapping well_id to production DataFrame
    model_name : str
        Decline model name for EUR estimation
    phase : str
        Phase to analyze: 'oil', 'gas', 'water'
    **kwargs
        Additional parameters for EUR estimation

    Returns
    -------
    pd.DataFrame
        Interference analysis with EUR-based metrics

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2'],
    ...     'latitude': [32.0, 32.001],
    ...     'longitude': [-97.0, -97.001]
    ... })
    >>> prod_data = {
    ...     'well_1': pd.DataFrame({'oil': [100, 95, 90]}, ...),
    ...     'well_2': pd.DataFrame({'oil': [95, 90, 85]}, ...)
    ... }
    >>> result = analyze_interference_with_production_history(
    ...     locations, prod_data
    ... )
    """
    from ressmith.workflows.analysis import estimate_eur

    logger.info(
        f"Analyzing interference with production history for {len(well_locations)} wells"
    )

    # Estimate EUR for each well
    eur_values = {}
    drainage_radii = {}
    well_ids = well_locations["well_id"].values

    for well_id in well_ids:
        if well_id in production_data:
            try:
                eur_result = estimate_eur(
                    production_data[well_id],
                    model_name=model_name,
                    phase=phase,
                    **kwargs,
                )
                eur_values[well_id] = eur_result["eur"]

                # Estimate drainage radius from EUR
                # Rough approximation for unconventional plays
                drainage_area_acres = eur_result["eur"] / 50000.0
                drainage_radius_ft = np.sqrt(drainage_area_acres * 43560.0 / np.pi)
                drainage_radii[well_id] = max(200.0, min(2000.0, drainage_radius_ft))
            except Exception as e:
                logger.warning(f"EUR estimation failed for {well_id}: {e}")
                eur_values[well_id] = 0.0
                drainage_radii[well_id] = 600.0

    # Convert locations to dictionary
    locations_dict = {
        well_id: (
            well_locations.loc[well_locations["well_id"] == well_id, "latitude"].iloc[0],
            well_locations.loc[well_locations["well_id"] == well_id, "longitude"].iloc[0],
        )
        for well_id in well_ids
    }

    # Analyze interference with EUR
    results = []
    for i, well_id_1 in enumerate(well_ids):
        for j, well_id_2 in enumerate(well_ids):
            if i < j:
                lat1, lon1 = locations_dict[well_id_1]
                lat2, lon2 = locations_dict[well_id_2]
                distance = calculate_well_distance(lat1, lon1, lat2, lon2)

                eur_1 = eur_values.get(well_id_1, 0.0)
                eur_2 = eur_values.get(well_id_2, 0.0)
                r1 = drainage_radii.get(well_id_1, 600.0)
                r2 = drainage_radii.get(well_id_2, 600.0)

                # Calculate EUR-based interference
                eur_interference = calculate_eur_based_interference(
                    eur_1, eur_2, distance, r1, r2
                )

                # Calculate geometric interference
                interference_result = analyze_well_interference(
                    distance=distance,
                    drainage_radius_1=r1,
                    drainage_radius_2=r2,
                )

                results.append({
                    "well_id_1": well_id_1,
                    "well_id_2": well_id_2,
                    "distance_ft": distance,
                    "eur_1": eur_1,
                    "eur_2": eur_2,
                    "drainage_radius_1": r1,
                    "drainage_radius_2": r2,
                    "interference_factor": interference_result.interference_factor,
                    "eur_interference_factor": eur_interference["eur_interference_factor"],
                    "total_eur_loss": eur_interference["total_eur_loss"],
                    "estimated_interference_percent": interference_result.estimated_interference_percent,
                })

    return pd.DataFrame(results)


def recommend_spacing_from_eur(
    well_locations: pd.DataFrame,
    production_data: dict[str, pd.DataFrame],
    model_name: str = "arps_hyperbolic",
    phase: str = "oil",
    target_eur_loss_percent: float = 5.0,
    min_spacing: float = 200.0,
    max_spacing: float = 2000.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Recommend optimal spacing based on EUR interference models.

    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    production_data : dict
        Dictionary mapping well_id to production DataFrame
    model_name : str
        Decline model name for EUR estimation
    phase : str
        Phase to analyze: 'oil', 'gas', 'water'
    target_eur_loss_percent : float
        Target maximum EUR loss percentage (default: 5%)
    min_spacing : float
        Minimum spacing constraint (ft)
    max_spacing : float
        Maximum spacing constraint (ft)
    **kwargs
        Additional parameters for EUR estimation

    Returns
    -------
    dict
        Dictionary with spacing recommendations based on EUR

    Examples
    --------
    >>> locations = pd.DataFrame({
    ...     'well_id': ['well_1', 'well_2'],
    ...     'latitude': [32.0, 32.001],
    ...     'longitude': [-97.0, -97.001]
    ... })
    >>> prod_data = {
    ...     'well_1': pd.DataFrame({'oil': [100, 95, 90]}, ...),
    ...     'well_2': pd.DataFrame({'oil': [95, 90, 85]}, ...)
    ... }
    >>> result = recommend_spacing_from_eur(locations, prod_data)
    """
    from ressmith.workflows.analysis import estimate_eur

    logger.info("Recommending spacing from EUR analysis")

    # Estimate EUR for each well
    eur_values = {}
    drainage_radii = {}
    well_ids = well_locations["well_id"].values

    for well_id in well_ids:
        if well_id in production_data:
            try:
                eur_result = estimate_eur(
                    production_data[well_id],
                    model_name=model_name,
                    phase=phase,
                    **kwargs,
                )
                eur_values[well_id] = eur_result["eur"]

                # Estimate drainage radius from EUR
                drainage_area_acres = eur_result["eur"] / 50000.0
                drainage_radius_ft = np.sqrt(drainage_area_acres * 43560.0 / np.pi)
                drainage_radii[well_id] = max(200.0, min(2000.0, drainage_radius_ft))
            except Exception as e:
                logger.warning(f"EUR estimation failed for {well_id}: {e}")
                eur_values[well_id] = 0.0
                drainage_radii[well_id] = 600.0

    # Convert locations to dictionary
    locations_dict = {
        well_id: (
            well_locations.loc[well_locations["well_id"] == well_id, "latitude"].iloc[0],
            well_locations.loc[well_locations["well_id"] == well_id, "longitude"].iloc[0],
        )
        for well_id in well_ids
    }

    return optimize_spacing_from_eur(
        eur_values=eur_values,
        well_locations=locations_dict,
        drainage_radii=drainage_radii,
        target_eur_loss_percent=target_eur_loss_percent,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )



