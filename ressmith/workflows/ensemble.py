"""
Ensemble forecasting workflows.

Provides workflows for combining multiple decline models to improve
forecast accuracy and reliability.
"""

import logging
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd

from ressmith.objects.domain import ForecastResult
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.ensemble import (
    confidence_weighted_ensemble,
    median_ensemble_forecast,
    weighted_ensemble_forecast,
)
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


def ensemble_forecast(
    data: pd.DataFrame,
    model_names: list[str],
    method: Literal["weighted", "median", "confidence"] = "weighted",
    weights: list[float] | None = None,
    horizon: int = 24,
    in_sample_data: pd.Series | None = None,
    **kwargs: Any,
) -> ForecastResult:
    """
    Generate ensemble forecast from multiple models.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_names : list
        List of model names to combine
    method : str
        Ensemble method: 'weighted', 'median', or 'confidence' (default: 'weighted')
    weights : list, optional
        Weights for weighted method (default: equal weights)
    horizon : int
        Forecast horizon in periods (default: 24)
    in_sample_data : pd.Series, optional
        In-sample data for confidence weighting
    **kwargs
        Additional parameters for models

    Returns
    -------
    ForecastResult
        Combined ensemble forecast

    Examples
    --------
    >>> from ressmith import ensemble_forecast
    >>>
    >>> # Combine multiple models
    >>> forecast = ensemble_forecast(
    ...     data,
    ...     model_names=['arps_hyperbolic', 'power_law', 'duong'],
    ...     method='weighted',
    ...     weights=[0.5, 0.3, 0.2],
    ...     horizon=36
    ... )
    >>> print(forecast.yhat.head())
    """
    logger.info(f"Generating ensemble forecast with {len(model_names)} models")

    forecasts = []
    for model_name in model_names:
        try:
            forecast, params = fit_forecast(
                data, model_name=model_name, horizon=horizon, **kwargs
            )
            forecast.metadata["model_name"] = model_name
            forecasts.append(forecast)
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}, skipping")
            continue

    if len(forecasts) == 0:
        raise ValueError("No models produced valid forecasts")

    if len(forecasts) == 1:
        logger.info("Only one model succeeded, returning single forecast")
        return forecasts[0]

    # Combine forecasts based on method
    if method == "weighted":
        result = weighted_ensemble_forecast(forecasts, weights=weights)
    elif method == "median":
        result = median_ensemble_forecast(forecasts)
    elif method == "confidence":
        result = confidence_weighted_ensemble(forecasts, in_sample_data=in_sample_data)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    logger.info(f"Ensemble forecast completed using {method} method")
    return result


def ensemble_forecast_custom(
    data: pd.DataFrame,
    model_factories: list[Callable[[], BaseDeclineModel]],
    method: Literal["weighted", "median", "confidence"] = "weighted",
    weights: list[float] | None = None,
    horizon: int = 24,
    **kwargs: Any,
) -> ForecastResult:
    """
    Generate ensemble forecast from custom model factories.

    Allows combining any BaseDeclineModel instances, not just predefined models.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_factories : list
        List of callables that return BaseDeclineModel instances
    method : str
        Ensemble method (default: 'weighted')
    weights : list, optional
        Weights for weighted method
    horizon : int
        Forecast horizon in periods
    **kwargs
        Additional parameters

    Returns
    -------
    ForecastResult
        Combined ensemble forecast
    """
    logger.info(
        f"Generating custom ensemble forecast with {len(model_factories)} models"
    )

    from ressmith.tasks.core import FitDeclineTask

    forecasts = []
    for i, factory in enumerate(model_factories):
        try:
            model = factory()
            task = FitDeclineTask(model=model, phase=kwargs.get("phase", "oil"))
            fitted_model, forecast_result = task.run(data, horizon=horizon)

            forecast_result.metadata["model_index"] = i
            forecasts.append(forecast_result)
        except Exception as e:
            logger.warning(f"Model {i} failed: {e}, skipping")
            continue

    if len(forecasts) == 0:
        raise ValueError("No models produced valid forecasts")

    if len(forecasts) == 1:
        return forecasts[0]

    # Combine forecasts
    if method == "weighted":
        result = weighted_ensemble_forecast(forecasts, weights=weights)
    elif method == "median":
        result = median_ensemble_forecast(forecasts)
    elif method == "confidence":
        result = confidence_weighted_ensemble(forecasts)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return result
