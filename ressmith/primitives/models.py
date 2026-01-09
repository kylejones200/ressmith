"""
Decline model classes wrapping primitive functions.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ressmith.objects.domain import (
    DeclineSpec,
    ForecastResult,
    ForecastSpec,
    ProductionSeries,
    RateSeries,
)
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.decline import (
    arps_exponential,
    arps_hyperbolic,
    arps_harmonic,
    fit_arps_exponential,
    fit_arps_hyperbolic,
    fit_arps_harmonic,
)


class ArpsExponentialModel(BaseDeclineModel):
    """Exponential decline model (b=0)."""

    def __init__(self, **params: Any) -> None:
        """Initialize exponential model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None
        self._t0: float = 0.0

    def fit(
        self, data: ProductionSeries | RateSeries, **fit_params: Any
    ) -> "ArpsExponentialModel":
        """Fit exponential decline model."""
        # Extract rate data
        if isinstance(data, ProductionSeries):
            rate = data.oil  # Default to oil phase
            time_index = data.time_index
        else:
            rate = data.rate
            time_index = data.time_index

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        self._fitted_params = fit_arps_exponential(t, rate)
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        # Generate forecast time index
        if self._start_date is None:
            raise ValueError("Model not fitted")
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        # Convert to days from start
        t_forecast = (forecast_index - start).days.values.astype(float)

        # Predict rates
        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        yhat = arps_exponential(t_forecast, qi, di, self._t0)

        # Create result
        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_exponential",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(
            yhat=yhat_series, metadata={}, model_spec=model_spec
        )

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
        self, data: ProductionSeries | RateSeries, **fit_params: Any
    ) -> "ArpsHyperbolicModel":
        """Fit hyperbolic decline model."""
        # Extract rate data
        if isinstance(data, ProductionSeries):
            rate = data.oil  # Default to oil phase
            time_index = data.time_index
        else:
            rate = data.rate
            time_index = data.time_index

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        self._fitted_params = fit_arps_hyperbolic(t, rate)
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        # Generate forecast time index
        if self._start_date is None:
            raise ValueError("Model not fitted")
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        # Convert to days from start
        t_forecast = (forecast_index - start).days.values.astype(float)

        # Predict rates
        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        b = self._fitted_params["b"]
        yhat = arps_hyperbolic(t_forecast, qi, di, b, self._t0)

        # Create result
        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_hyperbolic",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(
            yhat=yhat_series, metadata={}, model_spec=model_spec
        )

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
        self, data: ProductionSeries | RateSeries, **fit_params: Any
    ) -> "ArpsHarmonicModel":
        """Fit harmonic decline model."""
        # Extract rate data
        if isinstance(data, ProductionSeries):
            rate = data.oil  # Default to oil phase
            time_index = data.time_index
        else:
            rate = data.rate
            time_index = data.time_index

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        self._fitted_params = fit_arps_harmonic(t, rate)
        self._start_date = time_index[0].to_pydatetime()
        self._t0 = 0.0
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        # Generate forecast time index
        if self._start_date is None:
            raise ValueError("Model not fitted")
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        # Convert to days from start
        t_forecast = (forecast_index - start).days.values.astype(float)

        # Predict rates
        qi = self._fitted_params["qi"]
        di = self._fitted_params["di"]
        yhat = arps_harmonic(t_forecast, qi, di, self._t0)

        # Create result
        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="arps_harmonic",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(
            yhat=yhat_series, metadata={}, model_spec=model_spec
        )

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


class LinearDeclineModel(BaseDeclineModel):
    """
    Simple linear decline model (empirical approach).

    q(t) = q0 - m * t

    This is a simple alternative to ARPS models for cases where
    a linear decline is observed.
    """

    def __init__(self, **params: Any) -> None:
        """Initialize linear decline model."""
        super().__init__(**params)
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries, **fit_params: Any
    ) -> "LinearDeclineModel":
        """Fit linear decline model using least squares."""
        # Extract rate data
        if isinstance(data, ProductionSeries):
            rate = data.oil  # Default to oil phase
            time_index = data.time_index
        else:
            rate = data.rate
            time_index = data.time_index

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Linear regression: q = q0 - m*t
        # Use numpy polyfit for simplicity
        coeffs = np.polyfit(t, rate, deg=1)
        q0 = coeffs[1]  # Intercept
        m = -coeffs[0]  # Negative slope (decline)

        # Ensure positive parameters
        q0 = max(q0, 0.1)
        m = max(m, 0.0)

        self._fitted_params = {"q0": q0, "m": m}
        self._start_date = time_index[0].to_pydatetime()
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast."""
        self._check_fitted()

        # Generate forecast time index
        if self._start_date is None:
            raise ValueError("Model not fitted")
        start = pd.Timestamp(self._start_date)
        forecast_index = pd.date_range(
            start=start, periods=spec.horizon, freq=spec.frequency
        )

        # Convert to days from start
        t_forecast = (forecast_index - start).days.values.astype(float)

        # Predict rates: q = q0 - m*t
        q0 = self._fitted_params["q0"]
        m = self._fitted_params["m"]
        yhat = np.maximum(q0 - m * t_forecast, 0.0)  # Ensure non-negative

        # Create result
        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="linear_decline",
            parameters=self._fitted_params,
            start_date=self._start_date,
        )

        return ForecastResult(
            yhat=yhat_series, metadata={}, model_spec=model_spec
        )

    @property
    def tags(self) -> dict[str, Any]:
        """Model tags."""
        return {
            "supports_oil": True,
            "supports_gas": True,
            "supports_water": False,
            "supports_multiphase": False,
            "supports_irregular_time": True,  # Linear can handle irregular
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }
