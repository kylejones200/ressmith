"""Multi-model production forecaster (decline-curve compatible)."""

from typing import Literal, cast

import numpy as np
import pandas as pd

from ressmith.primitives.decline import (
    arps_exponential,
    arps_harmonic,
    arps_hyperbolic,
    fit_arps_exponential,
    fit_arps_harmonic,
    fit_arps_hyperbolic,
)
from ressmith.workflows.evaluation import mae, rmse, smape

from .forecast_arima import forecast_arima
from .forecast_chronos import forecast_chronos
from .forecast_statistical import (
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from .forecast_timesfm import forecast_timesfm


def _forecast_arps_series(
    series: pd.Series,
    kind: Literal["exponential", "harmonic", "hyperbolic"],
    horizon: int,
) -> pd.Series:
    t = np.arange(len(series), dtype=float)
    q = series.to_numpy(dtype=float)
    full_t = np.arange(len(series) + horizon, dtype=float)
    if kind == "exponential":
        p = fit_arps_exponential(t, q)
        yhat = arps_exponential(full_t, p["qi"], p["di"])
    elif kind == "harmonic":
        p = fit_arps_harmonic(t, q)
        yhat = arps_harmonic(full_t, p["qi"], p["di"])
    else:
        p = fit_arps_hyperbolic(t, q)
        yhat = arps_hyperbolic(full_t, p["qi"], p["di"], p["b"])
    freq = series.index.freq or pd.infer_freq(series.index)
    idx = pd.date_range(series.index[0], periods=len(yhat), freq=freq)
    return pd.Series(yhat, index=idx, name=f"arps_{kind}")


class Forecaster:
    """Forecast production time series (Arps, ARIMA, TimesFM, Chronos, statistical)."""

    def __init__(self, series: pd.Series) -> None:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Input must be indexed by datetime")
        s = series.dropna().copy()
        if s.index.freq is None:
            inferred = pd.infer_freq(s.index)
            if inferred is not None:
                s = s.asfreq(inferred)
        self.series = s
        self.last_forecast: pd.Series | None = None

    def forecast(
        self,
        model: Literal[
            "arps",
            "timesfm",
            "chronos",
            "arima",
            "exponential_smoothing",
            "moving_average",
            "linear_trend",
            "holt_winters",
        ],
        kind: Literal["exponential", "harmonic", "hyperbolic"] | None = "hyperbolic",
        horizon: int = 12,
        **kwargs: object,
    ) -> pd.Series:
        if model == "arps":
            arps_kind: Literal["exponential", "harmonic", "hyperbolic"] = (
                "hyperbolic" if kind is None else cast(Literal["exponential", "harmonic", "hyperbolic"], kind)
            )
            forecast = _forecast_arps_series(self.series, arps_kind, horizon)
        elif model == "timesfm":
            forecast = forecast_timesfm(self.series, horizon=horizon)
        elif model == "chronos":
            forecast = forecast_chronos(self.series, horizon=horizon)
        elif model == "arima":
            forecast_part = forecast_arima(self.series, horizon=horizon)
            full_index = pd.date_range(
                self.series.index[0],
                periods=len(self.series) + horizon,
                freq=self.series.index.freq or pd.infer_freq(self.series.index),
            )
            full_forecast = pd.concat([self.series, forecast_part])
            forecast = pd.Series(full_forecast.values, index=full_index, name="arima_forecast")
        elif model == "exponential_smoothing":
            alpha = float(kwargs.get("alpha", 0.3))
            forecast = simple_exponential_smoothing(self.series, alpha=alpha, horizon=horizon)
        elif model == "moving_average":
            window = int(kwargs.get("window", 6))
            forecast = moving_average_forecast(self.series, window=window, horizon=horizon)
        elif model == "linear_trend":
            forecast = linear_trend_forecast(self.series, horizon=horizon)
        elif model == "holt_winters":
            seasonal_periods = kwargs.get("seasonal_periods", None)
            sp = int(seasonal_periods) if seasonal_periods is not None else None
            forecast_result = holt_winters_forecast(
                self.series, horizon=horizon, seasonal_periods=sp
            )
            if forecast_result is None:
                raise ValueError("Holt-Winters forecast failed")
            forecast = forecast_result
        else:
            raise ValueError(f"Unknown model: {model}")

        self.last_forecast = forecast
        return forecast

    def evaluate(self, actual: pd.Series) -> dict[str, float]:
        if self.last_forecast is None:
            raise RuntimeError("Call .forecast() first.")
        common = self.last_forecast.index.intersection(actual.index)
        if len(common) == 0:
            raise ValueError("No overlapping dates to compare.")
        yhat = self.last_forecast.loc[common]
        ytrue = actual.loc[common]
        return {
            "rmse": rmse(ytrue, yhat),
            "mae": mae(ytrue, yhat),
            "smape": smape(ytrue, yhat),
        }
