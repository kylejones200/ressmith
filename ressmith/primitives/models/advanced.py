"""Advanced decline curve models.

This module contains advanced decline curve models for unconventional reservoirs:
- Duong model
- Power law decline model
- Stretched exponential model
"""

from datetime import datetime
from typing import Any

import pandas as pd

from ressmith.objects.domain import DeclineSpec, ForecastResult, ForecastSpec, ProductionSeries, RateSeries
from ressmith.primitives.advanced_decline import duong_rate, fit_duong, fit_power_law, fit_stretched_exponential, power_law_rate, stretched_exponential_rate
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.constraints import clip_parameters, get_default_bounds, validate_parameters
from ressmith.primitives.data_utils import extract_rate_data
from ressmith.utils.errors import ERR_MODEL_NOT_FITTED


class PowerLawDeclineModel(BaseDeclineModel):
    """Power law decline model."""

    def __init__(self, **params: Any) -> None:
        """Initialize power law model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "PowerLawDeclineModel":
        """Fit power law decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_power_law(t, rate)
        bounds = get_default_bounds("power_law")
        warnings = validate_parameters(params, bounds, "power_law")
        if warnings:
            params = clip_parameters(params, bounds)

        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
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
        n = self._fitted_params["n"]

        yhat = power_law_rate(t_forecast, qi, di, n)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="power_law",
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


class DuongModel(BaseDeclineModel):
    """Duong decline model for unconventional reservoirs."""

    def __init__(self, **params: Any) -> None:
        """Initialize Duong model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "DuongModel":
        """Fit Duong decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_duong(t, rate)
        bounds = get_default_bounds("duong")
        warnings = validate_parameters(params, bounds, "duong")
        if warnings:
            params = clip_parameters(params, bounds)

        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
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
        a = self._fitted_params["a"]
        m = self._fitted_params["m"]

        yhat = duong_rate(t_forecast, qi, a, m)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="duong",
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


class StretchedExponentialModel(BaseDeclineModel):
    """Stretched exponential decline model."""

    def __init__(self, **params: Any) -> None:
        """Initialize stretched exponential model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "StretchedExponentialModel":
        """Fit stretched exponential decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_stretched_exponential(t, rate)
        bounds = get_default_bounds("stretched_exponential")
        warnings = validate_parameters(params, bounds, "stretched_exponential")
        if warnings:
            params = clip_parameters(params, bounds)

        self._fitted_params = params
        self._start_date = time_index[0].to_pydatetime()
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
        tau = self._fitted_params["tau"]
        beta = self._fitted_params["beta"]

        yhat = stretched_exponential_rate(t_forecast, qi, tau, beta)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="stretched_exponential",
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

