"""Variant decline curve models.

This module contains variant decline curve models:
- Linear decline model
- Hyperbolic-to-exponential switch model
- Fixed terminal decline model
"""

from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ressmith.objects.domain import DeclineSpec, ForecastResult, ForecastSpec, ProductionSeries, RateSeries
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.data_utils import extract_rate_data
from ressmith.primitives.fitting_utils import initial_guess_hyperbolic
from ressmith.primitives.switch import hyperbolic_to_exponential_rate
from ressmith.primitives.variants import fit_fixed_terminal_decline, fixed_terminal_decline_rate
from ressmith.utils.errors import ERR_MODEL_NOT_FITTED


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
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "LinearDeclineModel":
        """Fit linear decline model using least squares."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        # Linear regression: q = q0 - m*t
        coeffs = np.polyfit(t, rate, deg=1)
        q0 = coeffs[1]  # Intercept
        m = -coeffs[0]  # Negative slope (decline)

        q0 = max(q0, 0.1)
        m = max(m, 0.0)

        self._fitted_params = {"q0": q0, "m": m}
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

        # Linear: q = q0 - m*t
        q0 = self._fitted_params["q0"]
        m = self._fitted_params["m"]
        yhat = np.maximum(q0 - m * t_forecast, 0.0)  # Ensure non-negative

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="linear_decline",
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
            "supports_irregular_time": True,  # Linear can handle irregular
            "requires_positive": True,
            "supports_censoring": False,
            "supports_intervals": False,
        }


class HyperbolicToExponentialSwitchModel(BaseDeclineModel):
    """
    Hyperbolic decline switching to exponential at a transition time.

    Rate equation:
    - For t < t_switch: q(t) = qi / (1 + b * di * t)^(1/b)
    - For t >= t_switch: q(t) = q_switch * exp(-di_exp * (t - t_switch))
    """

    def __init__(
        self,
        t_switch: float | None = None,
        di_exp: float | None = None,
        **params: Any,
    ) -> None:
        """Initialize switch model."""
        super().__init__(**params)
        self._t_switch = t_switch
        self._di_exp = di_exp
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "HyperbolicToExponentialSwitchModel":
        """Fit switch model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)
        hyper_guess = initial_guess_hyperbolic(t, rate)

        if self._t_switch is None:
            t_span = t[-1] - t[0] if len(t) > 1 else 1.0
            t_switch = t_span * 0.67
        else:
            t_switch = self._t_switch

        if self._di_exp is None:
            di_exp = hyper_guess["di"]
        else:
            di_exp = self._di_exp

        self._fitted_params = {
            "qi": hyper_guess["qi"],
            "di": hyper_guess["di"],
            "b": hyper_guess["b"],
            "t_switch": t_switch,
            "di_exp": di_exp,
        }
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
        b = self._fitted_params["b"]
        t_switch = self._fitted_params["t_switch"]
        di_exp = self._fitted_params["di_exp"]

        yhat = hyperbolic_to_exponential_rate(t_forecast, qi, di, b, t_switch, di_exp)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="hyperbolic_to_exponential_switch",
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


class FixedTerminalDeclineModel(BaseDeclineModel):
    """
    Fixed terminal decline model.

    Fits initial decline (hyperbolic/exponential/harmonic) and transitions
    to a fixed terminal decline rate to prevent unrealistic long-term forecasts.
    """

    def __init__(
        self,
        kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
        terminal_decline_rate: float = 0.05,
        transition_criteria: Literal["rate", "time"] = "rate",
        transition_value: float | None = None,
        **params: Any,
    ) -> None:
        """
        Initialize fixed terminal decline model.

        Parameters
        ----------
        kind : str
            Initial decline type (default: 'hyperbolic')
        terminal_decline_rate : float
            Annual terminal decline rate (default: 0.05 = 5% per year)
        transition_criteria : str
            When to transition: 'rate' or 'time' (default: 'rate')
        transition_value : float, optional
            Threshold value for transition
        """
        super().__init__(**params)
        self.kind = kind
        self.terminal_decline_rate = terminal_decline_rate
        self.transition_criteria = transition_criteria
        self.transition_value = transition_value
        self._fitted_params: dict[str, float] = {}
        self._start_date: datetime | None = None

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "FixedTerminalDeclineModel":
        """Fit fixed terminal decline model."""
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        params = fit_fixed_terminal_decline(
            t,
            rate,
            kind=self.kind,
            terminal_decline_rate=self.terminal_decline_rate,
            transition_criteria=self.transition_criteria,
            transition_value=self.transition_value,
        )

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
        b = self._fitted_params["b"]
        t_switch = self._fitted_params["t_switch"]
        di_terminal = self._fitted_params["di_terminal"]

        yhat = fixed_terminal_decline_rate(t_forecast, qi, di, b, t_switch, di_terminal)

        yhat_series = pd.Series(yhat, index=forecast_index, name="forecast")
        model_spec = DeclineSpec(
            model_name="fixed_terminal_decline",
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

