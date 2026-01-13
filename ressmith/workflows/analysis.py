"""
Model comparison and EUR estimation workflows.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.diagnostics import compute_diagnostics
from ressmith.primitives.reserves import calculate_eur_from_params
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


def compare_models(
    data: pd.DataFrame,
    model_names: list[str],
    horizon: int = 24,
    phase: str = "oil",
    return_forecasts: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Compare multiple models on the same well.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_names : list
        List of model names to compare
    horizon : int
        Forecast horizon for comparison
    phase : str
        Phase to compare: 'oil', 'gas', 'water'
    return_forecasts : bool
        If True, include forecast results in output
    **kwargs
        Additional parameters for models

    Returns
    -------
    pd.DataFrame
        Comparison table with columns: model_name, rmse, mae, mape, r_squared, eur

    Examples
    --------
    >>> from ressmith import compare_models
    >>>
    >>> results = compare_models(
    ...     data,
    ...     model_names=['arps_exponential', 'arps_hyperbolic', 'power_law'],
    ...     horizon=24
    ... )
    >>> print(results.sort_values('rmse'))
    """
    logger.info(f"Comparing {len(model_names)} models")

    comparison_rows = []
    forecasts_dict = {}

    for model_name in model_names:
        try:
            # Fit and forecast
            forecast, params = fit_forecast(
                data, model_name=model_name, horizon=horizon, phase=phase, **kwargs
            )

            phase_col = kwargs.get("rate_col", phase)
            if phase_col not in data.columns:
                continue

            actual_values = data[phase_col].values
            in_sample_forecast, _ = fit_forecast(
                data, model_name=model_name, horizon=len(data), phase=phase, **kwargs
            )
            predicted_values = in_sample_forecast.yhat.values[: len(actual_values)]

            diag = compute_diagnostics(actual_values, predicted_values)

            eur = calculate_eur_from_params(params, model_name)

            # Store result
            row = {
                "model_name": model_name,
                "rmse": diag.rmse,
                "mae": diag.mae,
                "mape": diag.mape,
                "r_squared": diag.r_squared,
                "eur": eur,
            }

            for param_name, param_value in params.items():
                row[f"param_{param_name}"] = param_value

            comparison_rows.append(row)

            if return_forecasts:
                forecasts_dict[model_name] = forecast

        except Exception as e:
            logger.warning(f"Model comparison failed for {model_name}: {e}")
            comparison_rows.append(
                {
                    "model_name": model_name,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "r_squared": np.nan,
                    "eur": None,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(comparison_rows)

    if return_forecasts:
        return results_df, forecasts_dict
    return results_df


def estimate_eur(
    data: pd.DataFrame,
    model_name: str = "arps_hyperbolic",
    phase: str = "oil",
    t_max: float = 360.0,
    econ_limit: float = 10.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Estimate EUR (Estimated Ultimate Recovery) from production data.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_name : str
        Model name for EUR calculation
    phase : str
        Phase to estimate EUR for: 'oil', 'gas', 'water'
    t_max : float
        Maximum time for EUR calculation in months (default: 360 = 30 years)
    econ_limit : float
        Economic limit in production units (default: 10)
    **kwargs
        Additional parameters for model

    Returns
    -------
    dict
        Dictionary with EUR, parameters, and metadata

    Examples
    --------
    >>> from ressmith import estimate_eur
    >>>
    >>> result = estimate_eur(
    ...     data,
    ...     model_name='arps_hyperbolic',
    ...     t_max=360,
    ...     econ_limit=10.0
    ... )
    >>> print(f"EUR: {result['eur']:.2f}")
    >>> print(f"Parameters: {result['params']}")
    """
    logger.info(f"Estimating EUR with model={model_name}, phase={phase}")

    forecast, params = fit_forecast(
        data, model_name=model_name, horizon=24, phase=phase, **kwargs
    )

    eur = calculate_eur_from_params(params, model_name, t_max, econ_limit)

    if eur is None:
        logger.warning(f"EUR calculation failed for {model_name}")
        eur = 0.0

    return {
        "eur": eur,
        "params": params,
        "model_name": model_name,
        "phase": phase,
        "t_max": t_max,
        "econ_limit": econ_limit,
        "metadata": {},
    }
