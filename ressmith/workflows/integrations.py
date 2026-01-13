"""
Integration workflows with other Smith ecosystem packages.

Provides integration points for plotsmith, anomsmith, and geosmith.
These workflows require optional dependencies and will gracefully degrade
if the packages are not available.
"""

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

try:
    import plotsmith

    HAS_PLOTSMITH = True
except ImportError:
    HAS_PLOTSMITH = False
    plotsmith = None  # type: ignore[assignment, unused-ignore]

try:
    import anomsmith

    HAS_ANOMSMITH = True
except ImportError:
    HAS_ANOMSMITH = False
    anomsmith = None  # type: ignore[assignment, unused-ignore]

try:
    import geosmith

    HAS_GEOSMITH = True
except ImportError:
    HAS_GEOSMITH = False
    geosmith = None  # type: ignore[assignment, unused-ignore]

from ressmith.objects.domain import ForecastResult

logger = logging.getLogger(__name__)


def plot_forecast(
    forecast: ForecastResult,
    historical: pd.Series | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> tuple["Figure", "Axes"] | None:
    """
    Plot forecast using plotsmith (if available).

    Parameters
    ----------
    forecast : ForecastResult
        Forecast result to plot
    historical : pd.Series, optional
        Historical production data to overlay (will be combined with forecast)
    title : str, optional
        Plot title
    **kwargs
        Additional arguments passed to plotsmith.plot_timeseries

    Returns
    -------
    tuple[Figure, Axes] | None
        Matplotlib figure and axes tuple, or None if plotsmith not available

    Examples
    --------
    >>> from ressmith import fit_forecast, plot_forecast
    >>>
    >>> forecast, _ = fit_forecast(data, model_name='arps_hyperbolic', horizon=24)
    >>> fig, ax = plot_forecast(forecast, historical=data['oil'], title='Production Forecast')
    """
    if not HAS_PLOTSMITH or plotsmith is None:
        logger.warning("plotsmith not available. Install with: pip install plotsmith")
        return None

    try:
        if historical is not None:
            plot_data = pd.DataFrame(
                {"Historical": historical, "Forecast": forecast.yhat}
            )
        else:
            plot_data = forecast.yhat

        bands = None
        if forecast.intervals is not None:
            bands = {
                "confidence": (
                    forecast.intervals.iloc[:, 0],
                    forecast.intervals.iloc[:, -1],
                )
            }

        return plotsmith.plot_timeseries(
            data=plot_data,
            bands=bands,
            title=title or "Production Forecast",
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Error plotting with plotsmith: {e}")
        return None


def detect_outliers(
    data: pd.Series | pd.DataFrame,
    method: str = "statistical",
    **kwargs: Any,
) -> pd.Series | pd.DataFrame:
    """
    Detect outliers in production data using anomsmith (if available).

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Production data to analyze
    method : str
        Outlier detection method (default: 'statistical')
    **kwargs
        Additional arguments passed to anomsmith

    Returns
    -------
    pd.Series or pd.DataFrame
        Boolean mask or outlier flags

    Examples
    --------
    >>> from ressmith import detect_outliers
    >>>
    >>> outliers = detect_outliers(data['oil'], method='statistical')
    >>> clean_data = data[~outliers]
    """
    if not HAS_ANOMSMITH or anomsmith is None:
        logger.warning("anomsmith not available. Install with: pip install anomsmith")
        if isinstance(data, pd.Series):
            return pd.Series(False, index=data.index)
        else:
            return pd.DataFrame(False, index=data.index, columns=data.columns)

    try:
        # Try workflows first, then direct function
        if hasattr(anomsmith, "workflows") and hasattr(
            anomsmith.workflows, "detect_outliers"
        ):
            return anomsmith.workflows.detect_outliers(data, method=method, **kwargs)
        elif hasattr(anomsmith, "detect_outliers"):
            return anomsmith.detect_outliers(data, method=method, **kwargs)
        else:
            logger.warning("anomsmith.detect_outliers not found")
            if isinstance(data, pd.Series):
                return pd.Series(False, index=data.index)
            else:
                return pd.DataFrame(False, index=data.index, columns=data.columns)
    except Exception as e:
        logger.warning(f"Error detecting outliers with anomsmith: {e}")
        if isinstance(data, pd.Series):
            return pd.Series(False, index=data.index)
        else:
            return pd.DataFrame(False, index=data.index, columns=data.columns)


def spatial_analysis(
    well_data: pd.DataFrame,
    well_locations: pd.DataFrame | None = None,
    analysis_type: str = "kriging",
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Perform spatial analysis using geosmith (if available).

    Parameters
    ----------
    well_data : pd.DataFrame
        Well production data with well_id column
    well_locations : pd.DataFrame, optional
        DataFrame with columns: well_id, latitude, longitude
    analysis_type : str
        Type of analysis: 'kriging', 'interpolation', etc. (default: 'kriging')
    **kwargs
        Additional arguments passed to geosmith

    Returns
    -------
    dict[str, Any] | None
        Spatial analysis results (or None if geosmith not available)

    Examples
    --------
    >>> from ressmith import spatial_analysis
    >>>
    >>> # Perform kriging for EUR estimation
    >>> results = spatial_analysis(
    ...     well_data,
    ...     well_locations=locations,
    ...     analysis_type='kriging',
    ...     variable='eur'
    ... )
    """
    if not HAS_GEOSMITH or geosmith is None:
        logger.warning("geosmith not available. Install with: pip install geosmith")
        return None

    try:
        if hasattr(geosmith, "workflows") and hasattr(
            geosmith.workflows, analysis_type
        ):
            func = getattr(geosmith.workflows, analysis_type)
            return func(well_data, well_locations=well_locations, **kwargs)
        elif hasattr(geosmith, analysis_type):
            func = getattr(geosmith, analysis_type)
            return func(well_data, well_locations=well_locations, **kwargs)
        else:
            logger.warning(f"geosmith.{analysis_type} not found")
            return None
    except Exception as e:
        logger.warning(f"Error performing spatial analysis with geosmith: {e}")
        return None


def create_well_pointset(
    portfolio_results: pd.DataFrame,
    well_locations: pd.DataFrame,
    variable: str = "eur",
) -> Any | None:
    """
    Create geosmith PointSet from portfolio results and well locations.

    Parameters
    ----------
    portfolio_results : pd.DataFrame
        Results from analyze_portfolio with well_id column
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    variable : str
        Variable to map: 'eur', 'npv', 'irr', etc. (default: 'eur')

    Returns
    -------
    PointSet | None
        geosmith PointSet object, or None if geosmith not available

    Examples
    --------
    >>> from ressmith import analyze_portfolio, create_well_pointset
    >>>
    >>> portfolio = analyze_portfolio(well_data)
    >>> pointset = create_well_pointset(portfolio, locations, variable='eur')
    >>> print(pointset.attributes.head())
    """
    if not HAS_GEOSMITH or geosmith is None:
        logger.warning("geosmith not available. Install with: pip install geosmith")
        return None

    try:
        import numpy as np
        from geosmith.objects import PointSet

        if "well_id" not in portfolio_results.columns:
            raise ValueError("portfolio_results must have 'well_id' column")
        if variable not in portfolio_results.columns:
            raise ValueError(f"Variable '{variable}' not found in portfolio_results")

        if "well_id" not in well_locations.columns:
            raise ValueError("well_locations must have 'well_id' column")
        if (
            "latitude" not in well_locations.columns
            or "longitude" not in well_locations.columns
        ):
            raise ValueError(
                "well_locations must have 'latitude' and 'longitude' columns"
            )

        merged = portfolio_results.merge(well_locations, on="well_id", how="inner")

        if len(merged) == 0:
            logger.warning("No matching wells found between results and locations")
            return None

        coordinates = np.column_stack(
            [merged["longitude"].values, merged["latitude"].values]
        )

        attributes = merged[[variable]].copy()

        pointset = PointSet(coordinates=coordinates, attributes=attributes)

        logger.info(f"Created PointSet with {len(pointset)} wells")
        return pointset
    except Exception as e:
        logger.warning(f"Error creating PointSet: {e}")
        return None


def map_portfolio_spatially(
    portfolio_results: pd.DataFrame,
    well_locations: pd.DataFrame,
    variable: str = "eur",
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Create spatial map of portfolio metrics using geosmith.

    Combines ressmith portfolio analysis with geosmith spatial capabilities.

    Parameters
    ----------
    portfolio_results : pd.DataFrame
        Results from analyze_portfolio
    well_locations : pd.DataFrame
        DataFrame with columns: well_id, latitude, longitude
    variable : str
        Variable to map: 'eur', 'npv', 'irr' (default: 'eur')
    **kwargs
        Additional arguments for geosmith.make_features

    Returns
    -------
    dict[str, Any] | None
        Spatial analysis results with PointSet and features, or None if geosmith unavailable

    Examples
    --------
    >>> from ressmith import analyze_portfolio, map_portfolio_spatially
    >>>
    >>> portfolio = analyze_portfolio(well_data, econ_spec=econ)
    >>> spatial_map = map_portfolio_spatially(
    ...     portfolio,
    ...     locations,
    ...     variable='npv'
    ... )
    >>> print(spatial_map['pointset'].attributes.head())
    """
    if not HAS_GEOSMITH or geosmith is None:
        logger.warning("geosmith not available. Install with: pip install geosmith")
        return None

    try:
        pointset = create_well_pointset(portfolio_results, well_locations, variable)
        if pointset is None:
            return None

        if hasattr(geosmith, "make_features"):
            features = geosmith.make_features(
                pointset, operations=kwargs.get("operations", {})
            )
            return {
                "pointset": pointset,
                "features": features,
                "variable": variable,
            }
        else:
            return {
                "pointset": pointset,
                "variable": variable,
            }
    except Exception as e:
        logger.warning(f"Error mapping portfolio spatially: {e}")
        return None
