"""
ResSmith: Reservoir decline curve analysis and forecasting library.

A strict 4-layer architecture for decline curve analysis with clean boundaries.
"""

__version__ = "0.1.0"

# Public API: Workflows
from ressmith.workflows import (
    aggregate_portfolio_forecast,
    analyze_portfolio,
    compare_models,
    detect_outliers,
    ensemble_forecast,
    estimate_eur,
    evaluate_economics,
    evaluate_scenarios,
    fit_forecast,
    fit_segmented_forecast,
    forecast_many,
    forecast_with_yields,
    full_run,
    plot_forecast,
    probabilistic_forecast,
    rank_wells,
    scenario_summary,
    spatial_analysis,
    walk_forward_backtest,
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
    "forecast_with_yields",
    "ensemble_forecast",
    "probabilistic_forecast",
    "evaluate_economics",
    "evaluate_scenarios",
    "scenario_summary",
    "walk_forward_backtest",
    "compare_models",
    "estimate_eur",
    "full_run",
    "plot_forecast",
    "detect_outliers",
    "spatial_analysis",
    "analyze_portfolio",
    "aggregate_portfolio_forecast",
    "rank_wells",
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
