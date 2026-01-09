"""
ResSmith: Reservoir decline curve analysis and forecasting library.

A strict 4-layer architecture for decline curve analysis with clean boundaries.
"""

__version__ = "0.0.1"

# Public API: Workflows
from ressmith.workflows import (
    evaluate_economics,
    fit_forecast,
    fit_segmented_forecast,
    forecast_many,
    full_run,
)

# Public API: Base types
from ressmith.primitives.base import BaseDeclineModel, BaseEconModel

# Public API: Core objects
from ressmith.objects import (
    CumSeries,
    DeclineSpec,
    EconResult,
    EconSpec,
    ForecastResult,
    ForecastSpec,
    ProductionSeries,
    RateSeries,
    WellMeta,
)

__all__ = [
    # Workflows
    "fit_forecast",
    "fit_segmented_forecast",
    "forecast_many",
    "evaluate_economics",
    "full_run",
    # Base types
    "BaseDeclineModel",
    "BaseEconModel",
    # Core objects
    "WellMeta",
    "ProductionSeries",
    "RateSeries",
    "CumSeries",
    "DeclineSpec",
    "ForecastSpec",
    "EconSpec",
    "ForecastResult",
    "EconResult",
]
