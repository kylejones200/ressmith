"""ARPS decline curve models.

This module contains the three basic ARPS decline curve models:
- Exponential (b=0)
- Hyperbolic (0 < b < 1)
- Harmonic (b=1)
"""

from datetime import datetime
from typing import Any

import pandas as pd

from ressmith.objects.domain import DeclineSpec, ForecastResult, ForecastSpec, ProductionSeries, RateSeries
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.constraints import clip_parameters, get_default_bounds, validate_parameters
from ressmith.primitives.data_utils import extract_rate_data
from ressmith.primitives.decline import arps_exponential, arps_harmonic, arps_hyperbolic, fit_arps_exponential, fit_arps_harmonic, fit_arps_hyperbolic
from ressmith.utils.errors import ERR_MODEL_NOT_FITTED


class ArpsExponentialModel(BaseDeclineModel):
    """Exponential decline model (b=0)."""

    def __init__(self, **params: Any) -> None:
        """Initialize exponential model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None
        self._t0: float = 0.0

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "ArpsExponentialModel":
        """Fit exponential decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_arps_exponential(t, rate)
        bounds = get_default_bounds("exponential")
        warnings = validate_parameters(params, bounds, "exponential")
        if warnings:
            params = clip_parameters(params, bounds)
        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        if self._start_date is None:
            raise ValueError(ERR_MODEL_NOT_FITTED)
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        t_forecast = (forecast_index - start).days.values.astype(float)

        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        yhat = arps_exponential(t_forecast, qi, di, self._t0)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_exponential",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(yhat=yhat_series, metadata={}, model_spec=model_spec)

    @property
    def tags(self) -> dict[str, Any]:
        """Model tags."""
        return {
            "supports_oil": True,
            "supports_gas": True,
            "supports_water": False,
            "supports_multiphase": False,
            "supports_irregular_time": False,
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }


class ArpsHyperbolicModel(BaseDeclineModel):
    """Hyperbolic decline model (0 < b < 1)."""

    def __init__(self, **params: Any) -> None:
        """Initialize hyperbolic model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None
        self._t0: float = 0.0

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "ArpsHyperbolicModel":
        """Fit hyperbolic decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_arps_hyperbolic(t, rate)
        bounds = get_default_bounds("hyperbolic")
        warnings = validate_parameters(params, bounds, "hyperbolic")
        if warnings:
            params = clip_parameters(params, bounds)
        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        if self._start_date is None:
            raise ValueError(ERR_MODEL_NOT_FITTED)
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        t_forecast = (forecast_index - start).days.values.astype(float)

        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        b = self._fitted_params["b"]
        yhat = arps_hyperbolic(t_forecast, qi, di, b, self._t0)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_hyperbolic",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(yhat=yhat_series, metadata={}, model_spec=model_spec)

    @property
    def tags(self) -> dict[str, Any]:
        """Model tags."""
        return {
            "supports_oil": True,
            "supports_gas": True,
            "supports_water": False,
            "supports_multiphase": False,
            "supports_irregular_time": False,
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }


class ArpsHarmonicModel(BaseDeclineModel):
    """Harmonic decline model (b=1)."""

    def __init__(self, **params: Any) -> None:
        """Initialize harmonic model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None
        self._t0: float = 0.0

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "ArpsHarmonicModel":
        """Fit harmonic decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_arps_harmonic(t, rate)
        bounds = get_default_bounds("harmonic")
        warnings = validate_parameters(params, bounds, "harmonic")
        if warnings:
            params = clip_parameters(params, bounds)
        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        if self._start_date is None:
            raise ValueError(ERR_MODEL_NOT_FITTED)
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        t_forecast = (forecast_index - start).days.values.astype(float)

        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        yhat = arps_harmonic(t_forecast, qi, di, self._t0)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_harmonic",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(yhat=yhat_series, metadata={}, model_spec=model_spec)

    @property
    def tags(self) -> dict[str, Any]:
        """Model tags."""
        return {
            "supports_oil": True,
            "supports_gas": True,
            "supports_water": False,
            "supports_multiphase": False,
            "supports_irregular_time": False,
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }

