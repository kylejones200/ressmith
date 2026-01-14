"""
ResSmith: Reservoir engineering library.

A comprehensive reservoir engineering toolkit with strict 4-layer architecture
for production analysis, decline curve analysis, forecasting, and economic evaluation.
"""

__version__ = "0.2.1"

# Public API: Workflows
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

# Public API: Base types
from ressmith.primitives.base import BaseDeclineModel, BaseEconModel
from ressmith.workflows import (
    aggregate_portfolio_forecast,
    analyze_blasingame,
    analyze_fmb,
    analyze_fracture_network,
    analyze_multi_well_interaction,
    analyze_portfolio,
    analyze_waterflood,
    analyze_well_coning,
    compare_models,
    create_well_pointset,
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
    generate_diagnostic_plot_data,
    match_type_curve_workflow,
    map_portfolio_spatially,
    optimize_field_spacing,
    plot_forecast,
    probabilistic_forecast,
    rank_wells,
    scenario_summary,
    spatial_analysis,
    walk_forward_backtest,
)

__all__ = [
    # Workflows - Core
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
    "create_well_pointset",
    "map_portfolio_spatially",
    "analyze_portfolio",
    "aggregate_portfolio_forecast",
    "rank_wells",
    # Multi-well interaction
    "analyze_multi_well_interaction",
    "optimize_field_spacing",
    # Coning analysis
    "analyze_well_coning",
    # EOR workflows
    "analyze_waterflood",
    # Type curves & RTA
    "match_type_curve_workflow",
    "generate_diagnostic_plot_data",
    "analyze_blasingame",
    "analyze_fmb",
    "analyze_fracture_network",
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
