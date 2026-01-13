"""
Layer 4: Workflows

User-facing entry points. Can import I/O and plotting libraries.
"""

from ressmith.workflows.analysis import compare_models, estimate_eur
from ressmith.workflows.backtesting import walk_forward_backtest
from ressmith.workflows.batch_processing import BatchManifest, batch_fit
from ressmith.workflows.calendar import CalendarConfig, place_monthly_data
from ressmith.workflows.core import (
    evaluate_economics,
    fit_forecast,
    fit_segmented_forecast,
    forecast_many,
    full_run,
)
from ressmith.workflows.data_utils import (
    load_price_csv,
    load_production_csvs,
    make_panel,
    to_monthly,
)
from ressmith.workflows.downtime import (
    DowntimeResult,
    reconstruct_rate_from_uptime,
    validate_uptime_data,
)
from ressmith.workflows.ensemble import ensemble_forecast, ensemble_forecast_custom
from ressmith.workflows.evaluation import (
    evaluate_forecast,
    mae,
    mape,
    r2_score,
    rmse,
    smape,
)
from ressmith.workflows.forecast_statistical import (
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from ressmith.workflows.history_matching import (
    HistoryMatchResult,
    history_match_material_balance,
)
from ressmith.workflows.integrations import (
    create_well_pointset,
    detect_outliers,
    map_portfolio_spatially,
    plot_forecast,
    spatial_analysis,
)
from ressmith.workflows.io import read_csv_production, write_csv_results
from ressmith.workflows.leakage_check import (
    comprehensive_leakage_check,
    validate_no_future_data,
    validate_training_split,
)
from ressmith.workflows.multiphase import forecast_with_yields
from ressmith.workflows.panel_analysis import (
    analyze_by_company,
    calculate_spatial_features,
    company_fixed_effects_regression,
    eur_with_company_controls,
    prepare_panel_data,
    spatial_eur_analysis,
)
from ressmith.workflows.parameter_resample import (
    approximate_posterior,
    fast_arps_resample,
)
from ressmith.workflows.portfolio import (
    aggregate_portfolio_forecast,
    analyze_portfolio,
    rank_wells,
)
from ressmith.workflows.profiling import (
    print_stats,
    profile,
    profile_context,
)
from ressmith.workflows.reports import (
    generate_field_pdf_report,
    generate_field_summary,
    generate_well_report,
)
from ressmith.workflows.risk import calculate_risk_metrics, portfolio_risk_report
from ressmith.workflows.scenarios import evaluate_scenarios, scenario_summary
from ressmith.workflows.sensitivity import run_sensitivity
from ressmith.workflows.uncertainty import probabilistic_forecast

__all__ = [
    "fit_forecast",
    "fit_segmented_forecast",
    "forecast_many",
    "evaluate_economics",
    "evaluate_scenarios",
    "scenario_summary",
    "walk_forward_backtest",
    "full_run",
    "read_csv_production",
    "write_csv_results",
    "forecast_with_yields",
    "compare_models",
    "estimate_eur",
    "ensemble_forecast",
    "ensemble_forecast_custom",
    "probabilistic_forecast",
    "plot_forecast",
    "detect_outliers",
    "spatial_analysis",
    "create_well_pointset",
    "map_portfolio_spatially",
    "analyze_portfolio",
    "aggregate_portfolio_forecast",
    "rank_wells",
    # Risk analysis
    "calculate_risk_metrics",
    "portfolio_risk_report",
    # Statistical forecasting
    "simple_exponential_smoothing",
    "moving_average_forecast",
    "linear_trend_forecast",
    "holt_winters_forecast",
    "calculate_confidence_intervals",
    # Panel analysis
    "prepare_panel_data",
    "calculate_spatial_features",
    "eur_with_company_controls",
    "company_fixed_effects_regression",
    "spatial_eur_analysis",
    "analyze_by_company",
    # Reports
    "generate_well_report",
    "generate_field_pdf_report",
    "generate_field_summary",
    # Evaluation metrics
    "rmse",
    "mae",
    "smape",
    "mape",
    "r2_score",
    "evaluate_forecast",
    # Data utilities
    "load_production_csvs",
    "to_monthly",
    "make_panel",
    "load_price_csv",
    # Calendar
    "CalendarConfig",
    "place_monthly_data",
    # Downtime
    "DowntimeResult",
    "reconstruct_rate_from_uptime",
    "validate_uptime_data",
    # Sensitivity
    "run_sensitivity",
    # Batch processing
    "BatchManifest",
    "batch_fit",
    # History matching
    "history_match_material_balance",
    "HistoryMatchResult",
    # Parameter resampling
    "fast_arps_resample",
    "approximate_posterior",
    # Profiling
    "profile",
    "profile_context",
    "print_stats",
    # Leakage check
    "validate_no_future_data",
    "validate_training_split",
    "comprehensive_leakage_check",
]
