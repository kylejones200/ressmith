"""Segmented decline curve model.

This module contains the segmented decline model that fits multiple ARPS segments
to different time periods of production data.
"""

from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ressmith.objects.domain import DeclineSegment, DeclineSpec, ForecastResult, ForecastSpec, ProductionSeries, RateSeries
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.data_utils import extract_rate_data
from ressmith.primitives.segmented import check_continuity, fit_segment, predict_segment


class SegmentedDeclineModel(BaseDeclineModel):
    """Segmented decline model with multiple ARPS segments."""

    def __init__(
        self,
        segment_dates: list[tuple[datetime, datetime]],
        kinds: list[Literal["exponential", "harmonic", "hyperbolic"]] | None = None,
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
        rate_series, time_index = extract_rate_data(data)
        rate = rate_series.values

        if not isinstance(time_index, pd.DatetimeIndex):
            raise ValueError("Time index must be DatetimeIndex for segmented model")

        self._start_date = time_index[0].to_pydatetime()

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
            mask = (time_index >= pd.Timestamp(start_date)) & (
                time_index < pd.Timestamp(end_date)
            )
            segment_series = pd.Series(rate, index=time_index)[mask]

            if len(segment_series) < 3:
                self._continuity_errors.append(
                    f"Segment {i} has insufficient data: {len(segment_series)} points"
                )
                continue

            t_segment = (segment_series.index - time_index[0]).days.values.astype(float)
            q_segment = segment_series.values

            try:
                params = fit_segment(t_segment, q_segment, self.kinds[i])

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

        forecast_parts = []

        for i, segment in enumerate(self._fitted_segments):
            if i == 0:
                t_segment = np.arange(segment.t_start, segment.t_end, 1.0)
            else:
                t_segment = np.arange(0, segment.t_end - segment.t_start, 1.0)

            q_segment = predict_segment(t_segment, segment.parameters, segment.kind)

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

        yhat_series = pd.Series(
            full_forecast.values, index=full_forecast.index, name="forecast"
        )
        model_spec = DeclineSpec(
            model_name="segmented_decline",
            parameters={"segments": len(self._fitted_segments)},
            start_date=self._start_date or datetime.now(),
        )

        return ForecastResult(yhat=yhat_series, metadata={}, model_spec=model_spec)

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

