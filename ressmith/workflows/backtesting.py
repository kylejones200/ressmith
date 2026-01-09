"""
Backtesting workflows for forecast validation.

Provides workflows for validating forecasts against historical data using
walk-forward and rolling origin methods.
"""

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import ForecastResult, ForecastSpec
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.diagnostics import compute_diagnostics
from ressmith.tasks.core import FitDeclineTask

logger = logging.getLogger(__name__)


def walk_forward_backtest(
    data: pd.DataFrame,
    model_name: str = "arps_hyperbolic",
    model_factory: Optional[Callable[[], BaseDeclineModel]] = None,
    forecast_horizons: list[int] = [12, 24, 36],
    min_train_size: int = 12,
    step_size: int = 1,
    phase: str = "oil",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Walk-forward backtest: refit model at each step and validate forecast.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_name : str
        Model name for model factory (if model_factory not provided)
    model_factory : callable, optional
        Function that returns a BaseDeclineModel instance
    forecast_horizons : list
        List of forecast horizons to test (in periods)
    min_train_size : int
        Minimum training set size (default: 12)
    step_size : int
        Steps between refits (default: 1)
    phase : str
        Phase to forecast: 'oil', 'gas', 'water' (default: 'oil')
    **kwargs
        Additional parameters for model

    Returns
    -------
    pd.DataFrame
        Backtest results with columns: cut_idx, horizon, rmse, mae, mape, r_squared

    Examples
    --------
    >>> from ressmith import walk_forward_backtest
    >>> 
    >>> results = walk_forward_backtest(
    ...     data,
    ...     model_name='arps_hyperbolic',
    ...     forecast_horizons=[12, 24, 36],
    ...     min_train_size=12
    ... )
    >>> print(results.groupby('horizon')['rmse'].mean())
    """
    logger.info(
        f"Starting walk-forward backtest: {len(forecast_horizons)} horizons, "
        f"min_train_size={min_train_size}"
    )

    # Extract phase data
    phase_col = kwargs.get("rate_col", phase)
    if phase_col not in data.columns:
        raise ValueError(f"Phase column '{phase_col}' not found in data")

    rates = data[phase_col].values
    time_index = data.index
    n_periods = len(rates)

    results = []

    # Walk forward through data
    max_cut_idx = n_periods - max(forecast_horizons) - 1
    for cut_idx in range(min_train_size, max_cut_idx, step_size):
        # Test each horizon
        for horizon in forecast_horizons:
            test_end_idx = min(cut_idx + horizon, n_periods)
            if test_end_idx >= n_periods:
                continue

            try:
                # Training data
                train_data = data.iloc[:cut_idx]
                
                # Fit model on training data
                from ressmith.workflows.core import fit_forecast
                forecast, _ = fit_forecast(
                    train_data,
                    model_name=model_name,
                    horizon=horizon,
                    phase=phase,
                    **kwargs
                )

                # Extract actual test values
                test_data = rates[cut_idx:test_end_idx]
                predicted_values = forecast.yhat.values[:len(test_data)]

                # Compute diagnostics
                diag = compute_diagnostics(test_data, predicted_values)

                # Store result
                results.append({
                    "cut_idx": cut_idx,
                    "horizon": horizon,
                    "rmse": diag.rmse,
                    "mae": diag.mae,
                    "mape": diag.mape,
                    "r_squared": diag.r_squared,
                })

            except Exception as e:
                logger.warning(f"Backtest failed at cut_idx={cut_idx}, horizon={horizon}: {e}")
                results.append({
                    "cut_idx": cut_idx,
                    "horizon": horizon,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "r_squared": np.nan,
                })

    return pd.DataFrame(results)

