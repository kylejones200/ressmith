"""
Layer 4: Workflows

User-facing entry points. Can import I/O and plotting libraries.
"""

from ressmith.workflows.analysis import compare_models, estimate_eur
from ressmith.workflows.backtesting import walk_forward_backtest
from ressmith.workflows.core import (
    evaluate_economics,
    fit_forecast,
    fit_segmented_forecast,
    forecast_many,
    full_run,
)
from ressmith.workflows.ensemble import ensemble_forecast, ensemble_forecast_custom
from ressmith.workflows.io import read_csv_production, write_csv_results
from ressmith.workflows.multiphase import forecast_with_yields
from ressmith.workflows.scenarios import evaluate_scenarios, scenario_summary
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
]

