"""
Layer 4: Workflows

User-facing entry points. Can import I/O and plotting libraries.
"""

from ressmith.objects.domain import HistoryMatchResult
from ressmith.workflows.advanced_rta import (
    analyze_blasingame,
    analyze_dn_type_curve,
    analyze_fmb,
    analyze_fracture_network,
    optimize_multi_stage_fracture,
)
from ressmith.workflows.analysis import compare_models, estimate_eur
from ressmith.workflows.artificial_lift_optimization import (
    compare_artificial_lift_methods,
    optimize_esp_system,
    optimize_gas_lift_system,
    optimize_rod_pump_system,
)
from ressmith.workflows.backtesting import walk_forward_backtest
from ressmith.workflows.batch_processing import BatchManifest, batch_fit
from ressmith.workflows.calendar import CalendarConfig, place_monthly_data
from ressmith.workflows.choke_optimization import (
    analyze_choke_performance,
    optimize_choke_size,
)
from ressmith.workflows.coning import (
    analyze_well_coning,
    forecast_wor_gor_with_coning,
)
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
from ressmith.workflows.diagnostic_plotting import (
    generate_all_diagnostic_plots,
    plot_boundary_dominated_flow,
    plot_diagnostic_with_flow_regime_identification,
    plot_linear_flow_diagnostic,
    plot_log_log_diagnostic,
    plot_sqrt_time_diagnostic,
)
from ressmith.workflows.diagnostics_plots import (
    calculate_flow_regime_slopes,
    generate_diagnostic_plot_data,
    identify_flow_regime_from_plots,
    prepare_boundary_dominated_plot_data,
    prepare_log_log_data,
    prepare_sqrt_time_data,
)
from ressmith.workflows.downtime import (
    DowntimeResult,
    reconstruct_rate_from_uptime,
    validate_uptime_data,
)
from ressmith.workflows.ensemble import ensemble_forecast, ensemble_forecast_custom
from ressmith.workflows.eor import (
    analyze_waterflood,
    calculate_mobility_ratio_workflow,
    predict_waterflood_performance,
)
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
    calculate_history_match_objective,
    history_match_material_balance,
    run_parameter_sensitivity_analysis,
)
from ressmith.workflows.integrations import (
    create_well_pointset,
    detect_outliers,
    map_portfolio_spatially,
    plot_forecast,
    spatial_analysis,
)
from ressmith.workflows.interference import (
    analyze_interference_matrix,
    analyze_interference_with_production_history,
    calculate_well_distances,
    recommend_spacing,
    recommend_spacing_from_eur,
)
from ressmith.workflows.io import read_csv_production, write_csv_results
from ressmith.workflows.leakage_check import (
    comprehensive_leakage_check,
    validate_no_future_data,
    validate_training_split,
)
from ressmith.workflows.multi_well import (
    analyze_multi_well_interaction,
    analyze_well_pattern,
    optimize_field_spacing,
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
from ressmith.workflows.parameter_sensitivity import (
    analyze_parameter_sensitivity,
    calculate_sensitivity_coefficients,
    identify_critical_parameters,
)
from ressmith.workflows.portfolio import (
    aggregate_portfolio_forecast,
    analyze_portfolio,
    rank_wells,
)
from ressmith.workflows.pressure_normalization import (
    calculate_pseudopressure_workflow,
    normalize_for_rta_analysis,
    normalize_production_with_pressure,
)
from ressmith.workflows.production_ops import (
    allocate_production,
    apply_constraints_to_production,
    optimize_production,
)
from ressmith.workflows.profiling import (
    print_stats,
    profile,
    profile_context,
)
from ressmith.workflows.relative_permeability import (
    apply_hysteresis_workflow,
    generate_capillary_pressure_curve,
    generate_oil_water_relative_permeability,
    generate_relative_permeability_curves,
    generate_three_phase_relative_permeability,
)
from ressmith.workflows.reports import (
    generate_field_pdf_report,
    generate_field_summary,
    generate_well_report,
)
from ressmith.workflows.risk import calculate_risk_metrics, portfolio_risk_report
from ressmith.workflows.scenarios import evaluate_scenarios, scenario_summary
from ressmith.workflows.sensitivity import run_sensitivity
from ressmith.workflows.simulator import (
    compare_simulation_to_forecast,
    export_for_simulator,
    export_simulator_input,
    import_simulation_results,
    import_simulator_output,
)
from ressmith.workflows.type_curves import (
    generate_type_curve_library,
    match_type_curve_workflow,
)
from ressmith.workflows.uncertainty import probabilistic_forecast
from ressmith.workflows.well_test_validation import (
    WellTestValidationResult,
    validate_and_analyze_well_test,
    validate_well_test_data,
    validate_well_test_results,
)

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
    # Interference
    "calculate_well_distances",
    "analyze_interference_matrix",
    "analyze_interference_with_production_history",
    "recommend_spacing",
    "recommend_spacing_from_eur",
    # Coning
    "analyze_well_coning",
    "forecast_wor_gor_with_coning",
    # Multi-well patterns
    "analyze_well_pattern",
    # Diagnostic plots
    "prepare_log_log_data",
    "prepare_sqrt_time_data",
    "prepare_boundary_dominated_plot_data",
    "calculate_flow_regime_slopes",
    "identify_flow_regime_from_plots",
    "generate_diagnostic_plot_data",
    "plot_log_log_diagnostic",
    "plot_sqrt_time_diagnostic",
    "plot_linear_flow_diagnostic",
    "plot_boundary_dominated_flow",
    "generate_all_diagnostic_plots",
    "plot_diagnostic_with_flow_regime_identification",
    # Type Curves
    "match_type_curve_workflow",
    "generate_type_curve_library",
    # EOR
    "analyze_waterflood",
    "calculate_mobility_ratio_workflow",
    "predict_waterflood_performance",
    # Pressure normalization
    "normalize_production_with_pressure",
    "normalize_for_rta_analysis",
    "calculate_pseudopressure_workflow",
    # Relative permeability
    "generate_relative_permeability_curves",
    "generate_oil_water_relative_permeability",
    "generate_three_phase_relative_permeability",
    "generate_capillary_pressure_curve",
    "apply_hysteresis_workflow",
    # Choke optimization
    "optimize_choke_size",
    "analyze_choke_performance",
    # Artificial lift optimization
    "optimize_esp_system",
    "optimize_gas_lift_system",
    "optimize_rod_pump_system",
    "compare_artificial_lift_methods",
    # Simulator Integration
    "export_for_simulator",
    "export_simulator_input",
    "import_simulation_results",
    "import_simulator_output",
    "compare_simulation_to_forecast",
    # History Matching
    "history_match_material_balance",
    "calculate_history_match_objective",
    "run_parameter_sensitivity_analysis",
    # Parameter Sensitivity
    "analyze_parameter_sensitivity",
    "calculate_sensitivity_coefficients",
    "identify_critical_parameters",
    # Well Test Validation
    "validate_well_test_data",
    "validate_well_test_results",
    "validate_and_analyze_well_test",
    "WellTestValidationResult",
    # Advanced RTA
    "analyze_dn_type_curve",
    "optimize_multi_stage_fracture",
    # Multi-Well Interaction
    "analyze_multi_well_interaction",
    "optimize_field_spacing",
    # Production Operations
    "allocate_production",
    "optimize_production",
    "apply_constraints_to_production",
    # Advanced RTA
    "analyze_blasingame",
    "analyze_fmb",
    "analyze_fracture_network",
]
