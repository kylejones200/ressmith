"""
Integration workflows with other Smith ecosystem packages.

Provides integration points for plotsmith, anomsmith, and geosmith.
These workflows require optional dependencies and will gracefully degrade
if the packages are not available.
"""

import logging
from typing import Any, Optional

import pandas as pd

try:
    import plotsmith
    HAS_PLOTSMITH = True
except ImportError:
    HAS_PLOTSMITH = False

try:
    import anomsmith
    HAS_ANOMSMITH = True
except ImportError:
    HAS_ANOMSMITH = False

try:
    import geosmith
    HAS_GEOSMITH = True
except ImportError:
    HAS_GEOSMITH = False

from ressmith.objects.domain import ForecastResult

logger = logging.getLogger(__name__)


def plot_forecast(
    forecast: ForecastResult,
    historical: Optional[pd.Series] = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot forecast using plotsmith (if available).

    Parameters
    ----------
    forecast : ForecastResult
        Forecast result to plot
    historical : pd.Series, optional
        Historical production data to overlay
    title : str, optional
        Plot title
    **kwargs
        Additional arguments passed to plotsmith

    Returns
    -------
    Any
        Plot object from plotsmith (or None if plotsmith not available)

    Examples
    --------
    >>> from ressmith import fit_forecast, plot_forecast
    >>> 
    >>> forecast, _ = fit_forecast(data, model_name='arps_hyperbolic', horizon=24)
    >>> plot_forecast(forecast, historical=data['oil'], title='Production Forecast')
    """
    if not HAS_PLOTSMITH:
        logger.warning(
            "plotsmith not available. Install with: pip install plotsmith"
        )
        return None

    try:
        # Try to use plotsmith.workflows.plot_timeseries if available
        if hasattr(plotsmith, "workflows"):
            if hasattr(plotsmith.workflows, "plot_timeseries"):
                return plotsmith.workflows.plot_timeseries(
                    forecast.yhat, historical=historical, title=title or "Forecast", **kwargs
                )
        # Fallback: try direct import
        elif hasattr(plotsmith, "plot_timeseries"):
            return plotsmith.plot_timeseries(
                forecast.yhat, historical=historical, title=title or "Forecast", **kwargs
            )
        else:
            logger.warning("plotsmith.plot_timeseries not found")
            return None
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
    if not HAS_ANOMSMITH:
        logger.warning(
            "anomsmith not available. Install with: pip install anomsmith"
        )
        # Return all False (no outliers detected)
        if isinstance(data, pd.Series):
            return pd.Series(False, index=data.index)
        else:
            return pd.DataFrame(False, index=data.index, columns=data.columns)

    try:
        if hasattr(anomsmith, "workflows"):
            if hasattr(anomsmith.workflows, "detect_outliers"):
                return anomsmith.workflows.detect_outliers(
                    data, method=method, **kwargs
                )
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
    well_locations: Optional[pd.DataFrame] = None,
    analysis_type: str = "kriging",
    **kwargs: Any,
) -> Any:
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
    Any
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
    if not HAS_GEOSMITH:
        logger.warning(
            "geosmith not available. Install with: pip install geosmith"
        )
        return None

    try:
        if hasattr(geosmith, "workflows"):
            if hasattr(geosmith.workflows, analysis_type):
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

