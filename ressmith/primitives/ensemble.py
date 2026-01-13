"""
Ensemble forecasting: combining multiple decline models.

Provides methods for combining predictions from different models to improve
accuracy and reliability.
"""

import numpy as np
import pandas as pd

from ressmith.objects.domain import ForecastResult


def weighted_ensemble_forecast(
    forecasts: list[ForecastResult],
    weights: list[float] | None = None,
) -> ForecastResult:
    """
    Combine multiple forecasts using weighted average.

    Parameters
    ----------
    forecasts : list
        List of ForecastResult objects
    weights : list, optional
        Weights for each forecast (default: equal weights)

    Returns
    -------
    ForecastResult
        Combined forecast
    """
    if len(forecasts) == 0:
        raise ValueError("At least one forecast required")

    if len(forecasts) == 1:
        return forecasts[0]

    # Normalize weights
    if weights is None:
        weights = [1.0 / len(forecasts)] * len(forecasts)
    else:
        if len(weights) != len(forecasts):
            raise ValueError("Number of weights must match number of forecasts")
        total = sum(weights)
        weights = [w / total for w in weights]

    base_index = forecasts[0].yhat.index
    combined_rates = np.zeros(len(base_index))

    for forecast, weight in zip(forecasts, weights):
        if not forecast.yhat.index.equals(base_index):
            forecast_rates = forecast.yhat.reindex(base_index, method="nearest").values
        else:
            forecast_rates = forecast.yhat.values

        combined_rates += weight * forecast_rates

    combined_series = pd.Series(combined_rates, index=base_index, name="forecast")
    combined_metadata = {
        "ensemble_method": "weighted",
        "n_models": len(forecasts),
        "weights": weights,
    }

    return ForecastResult(
        yhat=combined_series,
        metadata=combined_metadata,
        model_spec=forecasts[0].model_spec,  # Use first model's spec as reference
    )


def median_ensemble_forecast(forecasts: list[ForecastResult]) -> ForecastResult:
    """
    Combine multiple forecasts using median.

    More robust to outliers than weighted average.

    Parameters
    ----------
    forecasts : list
        List of ForecastResult objects

    Returns
    -------
    ForecastResult
        Combined forecast
    """
    if len(forecasts) == 0:
        raise ValueError("At least one forecast required")

    if len(forecasts) == 1:
        return forecasts[0]

    # Align all forecasts to same index
    base_index = forecasts[0].yhat.index
    forecast_matrix = np.zeros((len(forecasts), len(base_index)))

    for i, forecast in enumerate(forecasts):
        if not forecast.yhat.index.equals(base_index):
            forecast_rates = forecast.yhat.reindex(base_index, method="nearest").values
        else:
            forecast_rates = forecast.yhat.values
        forecast_matrix[i, :] = forecast_rates

    # Take median across models
    combined_rates = np.median(forecast_matrix, axis=0)

    combined_series = pd.Series(combined_rates, index=base_index, name="forecast")
    combined_metadata = {
        "ensemble_method": "median",
        "n_models": len(forecasts),
    }

    return ForecastResult(
        yhat=combined_series,
        metadata=combined_metadata,
        model_spec=forecasts[0].model_spec,
    )


def confidence_weighted_ensemble(
    forecasts: list[ForecastResult],
    in_sample_data: pd.Series | None = None,
) -> ForecastResult:
    """
    Combine forecasts using confidence weights based on in-sample fit quality.

    Models with better in-sample fit get higher weights.

    Parameters
    ----------
    forecasts : list
        List of ForecastResult objects
    in_sample_data : pd.Series, optional
        In-sample data for computing fit quality (if None, uses equal weights)

    Returns
    -------
    ForecastResult
        Combined forecast
    """
    if in_sample_data is None or len(in_sample_data) == 0:
        # Fall back to equal weights
        return weighted_ensemble_forecast(forecasts)

    weights = []
    for forecast in forecasts:
        if "r_squared" in forecast.metadata:
            r_squared = forecast.metadata["r_squared"]
            weights.append(max(0.0, r_squared))
        else:
            weights.append(1.0)

    if sum(weights) == 0:
        weights = [1.0 / len(forecasts)] * len(forecasts)

    return weighted_ensemble_forecast(forecasts, weights=weights)
