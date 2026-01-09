"""
Decline model classes wrapping primitive functions.
"""

from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import (
    DeclineSegment,
    DeclineSpec,
    ForecastResult,
    ForecastSpec,
    ProductionSeries,
    RateSeries,
    SegmentedDeclineResult,
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
from ressmith.primitives.data_utils import extract_rate_data
from ressmith.primitives.advanced_decline import (
    duong_rate,
    fit_duong,
    fit_power_law,
    fit_stretched_exponential,
    power_law_rate,
    stretched_exponential_rate,
)
from ressmith.primitives.variants import (
    fit_fixed_terminal_decline,
    fixed_terminal_decline_rate,
)
from ressmith.primitives.constraints import (
    clip_parameters,
    get_default_bounds,
    validate_parameters,
)
from ressmith.primitives.segmented import (
    check_continuity,
    fit_segment,
    predict_segment,
)
from ressmith.primitives.switch import hyperbolic_to_exponential_rate
from ressmith.primitives.fitting_utils import initial_guess_hyperbolic


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
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        params = fit_arps_exponential(t, rate)
        # Validate and clip parameters
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
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "ArpsHyperbolicModel":
        """Fit hyperbolic decline model."""
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        params = fit_arps_hyperbolic(t, rate)
        # Validate and clip parameters
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
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "ArpsHarmonicModel":
        """Fit harmonic decline model."""
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        # Convert time to days from start
        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
        params = fit_arps_harmonic(t, rate)
        # Validate and clip parameters
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
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "LinearDeclineModel":
        """Fit linear decline model using least squares."""
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

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


class SegmentedDeclineModel(BaseDeclineModel):
    """Segmented decline model with multiple ARPS segments."""

    def __init__(
        self,
        segment_dates: list[tuple[datetime, datetime]],
        kinds: Optional[list[Literal["exponential", "harmonic", "hyperbolic"]]] = None,
        enforce_continuity: bool = True,
        **params: Any,
    ) -> None:
        """
        Initialize segmented decline model.

        Parameters
        ----------
        segment_dates : list
            List of (start_date, end_date) tuples for each segment
        kinds : list, optional
            ARPS decline types for each segment (default: all 'hyperbolic')
        enforce_continuity : bool
            Enforce rate continuity at segment boundaries (default: True)
        **params
            Additional parameters
        """
        super().__init__(**params)
        self.segment_dates = segment_dates
        self.kinds = kinds or ["hyperbolic"] * len(segment_dates)
        self.enforce_continuity = enforce_continuity
        self._fitted_segments: list[DeclineSegment] = []
        self._start_date: datetime | None = None
        self._continuity_errors: list[str] = []

        if len(self.kinds) != len(segment_dates):
            raise ValueError("Number of kinds must match number of segments")

    def fit(
        self, data: ProductionSeries | RateSeries | pd.DataFrame, **fit_params: Any
    ) -> "SegmentedDeclineModel":
        """Fit segmented decline model."""
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        if not isinstance(time_index, pd.DatetimeIndex):
            raise ValueError("Time index must be DatetimeIndex for segmented model")

        self._start_date = time_index[0].to_pydatetime()

        # Validate and sort segment dates
        sorted_segments = sorted(self.segment_dates, key=lambda x: x[0])
        for i in range(len(sorted_segments) - 1):
            if sorted_segments[i][1] > sorted_segments[i + 1][0]:
                raise ValueError(
                    f"Segments overlap: {sorted_segments[i]} and {sorted_segments[i+1]}"
                )

        # Fit each segment
        segments = []
        self._continuity_errors = []

        for i, (start_date, end_date) in enumerate(sorted_segments):
            # Find indices for this segment
            mask = (time_index >= pd.Timestamp(start_date)) & (
                time_index < pd.Timestamp(end_date)
            )
            segment_series = pd.Series(rate, index=time_index)[mask]

            if len(segment_series) < 3:
                self._continuity_errors.append(
                    f"Segment {i} has insufficient data: {len(segment_series)} points"
                )
                continue

            # Convert time to days from start
            t_segment = (segment_series.index - time_index[0]).days.values.astype(
                float
            )
            q_segment = segment_series.values

            try:
                # Fit segment
                params = fit_segment(t_segment, q_segment, self.kinds[i])

                # Create segment object
                start_idx = time_index.get_loc(segment_series.index[0])
                end_idx = time_index.get_loc(segment_series.index[-1]) + 1

                segment = DeclineSegment(
                    kind=self.kinds[i],
                    parameters=params,
                    t_start=float(start_idx),
                    t_end=float(end_idx),
                    start_date=segment_series.index[0].to_pydatetime(),
                    end_date=segment_series.index[-1].to_pydatetime(),
                )
                segments.append(segment)

                # Check continuity with previous segment
                if self.enforce_continuity and len(segments) > 1:
                    prev_segment = segments[-2]
                    curr_segment = segments[-1]
                    t_transition = curr_segment.t_start

                    is_continuous, error_msg = check_continuity(
                        prev_segment, curr_segment, t_transition
                    )
                    if not is_continuous:
                        self._continuity_errors.append(error_msg)

            except Exception as e:
                self._continuity_errors.append(f"Segment {i} fitting failed: {str(e)}")
                continue

        self._fitted_segments = segments
        self._fitted = True
        return self

    def predict(self, spec: ForecastSpec) -> ForecastResult:
        """Generate forecast from segmented model."""
        self._check_fitted()

        if len(self._fitted_segments) == 0:
            raise ValueError("No segments fitted")

        # Generate forecast by concatenating segment forecasts
        forecast_parts = []

        for i, segment in enumerate(self._fitted_segments):
            # Calculate time range for this segment
            if i == 0:
                t_segment = np.arange(segment.t_start, segment.t_end, 1.0)
            else:
                # Continue from where previous segment ended
                t_segment = np.arange(0, segment.t_end - segment.t_start, 1.0)

            # Predict for this segment
            q_segment = predict_segment(t_segment, segment.parameters, segment.kind)

            # Create index for this segment
            if i == 0 and self._start_date:
                start = pd.Timestamp(self._start_date)
                n_periods = len(q_segment)
                segment_dates_idx = pd.date_range(
                    start=start, periods=n_periods, freq=spec.frequency
                )
            else:
                # Continue date index from previous segment
                last_date = forecast_parts[-1].index[-1]
                n_periods = len(q_segment)
                segment_dates_idx = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=n_periods,
                    freq=spec.frequency,
                )

            forecast_part = pd.Series(q_segment, index=segment_dates_idx)
            forecast_parts.append(forecast_part)

        # Concatenate all segments
        if forecast_parts:
            full_forecast = pd.concat(forecast_parts)
            # Limit to requested horizon
            if len(full_forecast) > spec.horizon:
                full_forecast = full_forecast.iloc[: spec.horizon]
        else:
            full_forecast = pd.Series(dtype=float)

        # Create result
        yhat_series = pd.Series(full_forecast.values, index=full_forecast.index, name="forecast")
        model_spec = DeclineSpec(
            model_name="segmented_decline",
            parameters={"segments": len(self._fitted_segments)},
            start_date=self._start_date or datetime.now(),
        )

        return ForecastResult(
            yhat=yhat_series, metadata={}, model_spec=model_spec
        )

    def get_segments(self) -> list[DeclineSegment]:
        """Get fitted segments."""
        self._check_fitted()
        return self._fitted_segments

    def get_continuity_errors(self) -> list[str]:
        """Get continuity errors if any."""
        return self._continuity_errors

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


class HyperbolicToExponentialSwitchModel(BaseDeclineModel):
    """
    Hyperbolic decline switching to exponential at a transition time.

    Rate equation:
    - For t < t_switch: q(t) = qi / (1 + b * di * t)^(1/b)
    - For t >= t_switch: q(t) = q_switch * exp(-di_exp * (t - t_switch))
    """

    def __init__(
        self,
        t_switch: Optional[float] = None,
        di_exp: Optional[float] = None,
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
        # Extract rate data
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
            raise ValueError("Model not fitted")
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
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
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
            raise ValueError("Model not fitted")
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
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
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
            raise ValueError("Model not fitted")
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
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
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
            raise ValueError("Model not fitted")
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
        transition_value: Optional[float] = None,
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
        # Extract rate data
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        t = (time_index - time_index[0]).days.values.astype(float)

        # Fit model
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
            raise ValueError("Model not fitted")
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
