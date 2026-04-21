"""
Decline curve analysis API - compatibility layer for legacy pydca/decline-curve and PetroSmith.

This module provides the dca API (single_well, forecast) that maps to ResSmith
workflows and ported ML forecasters from the former decline-curve project.

Example:
    >>> from ressmith import dca
    >>> forecast = dca.single_well(series, model='arps', kind='hyperbolic', horizon=12)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd

from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)

# Model name mapping: (model, kind) -> ressmith model_name
_ARPS_MODEL_MAP = {
    ("arps", "exponential"): "arps_exponential",
    ("arps", "harmonic"): "arps_harmonic",
    ("arps", "hyperbolic"): "arps_hyperbolic",
    ("arps", None): "arps_hyperbolic",
}


def _series_to_oil_frame(series: pd.Series) -> pd.DataFrame:
    data = series.to_frame("oil")
    if data.index.name is not None:
        data = data.copy()
        data.index.name = None
    return data


def single_well(
    series: pd.Series,
    model: Literal[
        "arps",
        "arima",
        "timesfm",
        "chronos",
    ] = "arps",
    kind: Literal["exponential", "harmonic", "hyperbolic"] | None = "hyperbolic",
    horizon: int = 12,
    return_params: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, Any]]:
    """
    Analyze a single well with decline curve or ML/statistical forecast.

    Args:
        series: Historical production time series with DatetimeIndex
        model: ``arps`` uses ResSmith fit_forecast; ``arima``, ``timesfm``,
            ``chronos`` use ported forecasters (see optional extras).
        kind: Arps decline type when ``model='arps'``
        horizon: Number of periods to forecast (future only for ``arps``;
            full history + horizon for ARIMA/Chronos/TimesFM-style outputs
            where the underlying forecaster returns a long series).
        return_params: If True, also return fitted parameters (Arps only;
            empty dict for other models).

    Returns:
        Forecast series, or (series, params) if ``return_params=True``.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have DatetimeIndex")
    if len(series) < 3:
        raise ValueError("Need at least 3 data points for decline curve fitting")

    if model == "arps":
        model_name = _ARPS_MODEL_MAP.get(
            (model, kind), _ARPS_MODEL_MAP.get((model, None), "arps_hyperbolic")
        )
        data = _series_to_oil_frame(series)
        forecast_result, params = fit_forecast(
            data, model_name=model_name, horizon=horizon, phase="oil"
        )
        forecast_series = forecast_result.yhat
        if return_params:
            params_dict = {
                "qi": params.get("qi"),
                "di": params.get("di"),
                "b": params.get("b", 0.5),
                "kind": kind or "hyperbolic",
            }
            return forecast_series, params_dict
        return forecast_series

    from ressmith.primitives.ml_forecast.forecaster import Forecaster

    fc = Forecaster(series)
    result = fc.forecast(
        model=model,  # type: ignore[arg-type]
        kind=kind,
        horizon=horizon,
    )
    if return_params:
        return result, {}
    return result


def forecast(
    series: pd.Series,
    model: Literal[
        "arps",
        "timesfm",
        "chronos",
        "arima",
        "deepar",
        "tft",
        "ensemble",
        "exponential_smoothing",
        "moving_average",
        "linear_trend",
        "holt_winters",
    ] = "arps",
    kind: Literal["exponential", "harmonic", "hyperbolic"] | None = "hyperbolic",
    horizon: int = 12,
    verbose: bool = False,
    deepar_model: Any = None,
    tft_model: Any = None,
    production_data: pd.DataFrame | None = None,
    well_id: str | None = None,
    quantiles: list[float] | None = None,
    return_interpretation: bool = False,
    ensemble_models: list[str] | None = None,
    ensemble_weights: Any = None,
    ensemble_method: Literal["weighted", "confidence", "stacking"] = "weighted",
    lstm_model: Any = None,
    **kwargs: Any,
) -> pd.Series | Any:
    """
    Extended forecast API (decline-curve compatible).

    DeepAR and TFT require trained models and multi-well ``production_data``.
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    if model == "deepar":
        if deepar_model is None:
            raise ValueError(
                "deepar_model required for DeepAR. Train DeepARForecaster first."
            )
        if production_data is None or well_id is None:
            raise ValueError("production_data and well_id required for DeepAR")
        forecasts = deepar_model.predict_quantiles(
            well_id=well_id,
            production_data=production_data,
            quantiles=quantiles,
            horizon=horizon,
            n_samples=500,
        )
        phase = list(forecasts.keys())[0]
        return forecasts[phase].get(
            "q50",
            forecasts[phase][list(forecasts[phase].keys())[0]],
        )

    if model == "tft":
        if tft_model is None:
            raise ValueError("tft_model required for TFT. Train TFTForecaster first.")
        if production_data is None or well_id is None:
            raise ValueError("production_data and well_id required for TFT")
        result = tft_model.predict(
            well_id=well_id,
            production_data=production_data,
            horizon=horizon,
            return_interpretation=return_interpretation,
        )
        if return_interpretation:
            forecasts, interpretation = result
            phase = list(forecasts.keys())[0]
            if verbose:
                logger.debug("TFT forecast with interpretation, horizon=%s", horizon)
            return forecasts[phase], interpretation
        phase = list(result.keys())[0]
        if verbose:
            logger.debug("TFT forecast, horizon=%s", horizon)
        return result[phase]

    if model == "ensemble":
        from ressmith.primitives.ml_forecast.ensemble import EnsembleForecaster

        forecaster = EnsembleForecaster(
            models=ensemble_models or ["arps", "arima"],
            weights=ensemble_weights,
            method=ensemble_method,
        )
        return forecaster.forecast(
            series=series,
            horizon=horizon,
            arps_kind=kind or "hyperbolic",
            lstm_model=lstm_model,
            deepar_model=deepar_model,
            production_data=production_data,
            well_id=well_id,
            quantiles=quantiles,
            verbose=verbose,
        )

    from ressmith.primitives.ml_forecast.forecaster import Forecaster

    fc = Forecaster(series)
    result = fc.forecast(
        model=model,  # type: ignore[arg-type]
        kind=kind,
        horizon=horizon,
        **kwargs,
    )
    if verbose:
        logger.debug("Forecast model=%s horizon=%s", model, horizon)
    return result


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """RMSE, MAE, SMAPE on overlapping index."""
    from ressmith.workflows.evaluation import mae, rmse, smape

    common = y_true.index.intersection(y_pred.index)
    yt = y_true.loc[common]
    yp = y_pred.loc[common]
    return {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "smape": smape(yt, yp),
    }
