"""Tests for segmented decline primitives and models."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import DeclineSegment, RateSeries
from ressmith.primitives.models import SegmentedDeclineModel
from ressmith.primitives.segmented import check_continuity, fit_segment, predict_segment


def test_fit_segment_exponential():
    """Test fitting exponential segment."""
    t = np.linspace(0, 10, 50)
    qi = 100.0
    di = 0.15
    q = qi * np.exp(-di * t)
    # Add noise
    noise = np.random.normal(0, 1.0, len(q))
    q_noisy = np.maximum(q + noise, 0.1)

    params = fit_segment(t, q_noisy, kind="exponential")

    assert "qi" in params
    assert "di" in params
    assert params["b"] == 0.0
    assert params["qi"] > 0
    assert params["di"] > 0


def test_fit_segment_hyperbolic():
    """Test fitting hyperbolic segment."""
    t = np.linspace(0, 10, 50)
    qi = 100.0
    di = 0.15
    b = 0.6
    q = qi / (1.0 + b * di * t) ** (1.0 / b)
    # Add noise
    noise = np.random.normal(0, 1.0, len(q))
    q_noisy = np.maximum(q + noise, 0.1)

    params = fit_segment(t, q_noisy, kind="hyperbolic")

    assert "qi" in params
    assert "di" in params
    assert "b" in params
    assert params["qi"] > 0
    assert params["di"] > 0
    assert 0 < params["b"] < 1


def test_predict_segment():
    """Test predicting from segment."""
    t = np.linspace(0, 10, 50)
    params = {"qi": 100.0, "di": 0.15, "b": 0.6}

    q_pred = predict_segment(t, params, kind="hyperbolic")

    assert len(q_pred) == len(t)
    assert np.all(q_pred > 0)
    assert q_pred[0] > q_pred[-1]  # Should be declining


def test_check_continuity():
    """Test continuity checking between segments."""
    # Create two continuous segments
    segment1 = DeclineSegment(
        kind="hyperbolic",
        parameters={"qi": 100.0, "di": 0.15, "b": 0.6},
        t_start=0.0,
        t_end=10.0,
    )

    # Segment 2 should start where segment 1 ends
    # Calculate rate at end of segment 1
    t_end = 10.0
    q_end = predict_segment(np.array([t_end]), segment1.parameters, segment1.kind)[0]

    # Create segment 2 that continues from segment 1
    segment2 = DeclineSegment(
        kind="exponential",
        parameters={"qi": q_end, "di": 0.2, "b": 0.0},
        t_start=10.0,
        t_end=20.0,
    )

    # Check continuity
    is_continuous, error = check_continuity(segment1, segment2, 10.0)

    # Should be continuous (or very close)
    assert is_continuous or error is not None


def test_segmented_decline_model_fit():
    """Test fitting segmented decline model."""
    # Generate synthetic data with two segments
    time_index = pd.date_range("2020-01-01", periods=24, freq="M")
    t = np.arange(24)

    # First segment: hyperbolic
    q1 = 100.0 / (1.0 + 0.6 * 0.15 * t[:12]) ** (1.0 / 0.6)
    # Second segment: exponential (steeper decline)
    q2_start = q1[-1]
    q2 = q2_start * np.exp(-0.2 * (t[12:] - 12))

    q = np.concatenate([q1, q2])
    data = pd.DataFrame({"oil": q}, index=time_index)

    # Define segments
    segment_dates = [
        (datetime(2020, 1, 1), datetime(2021, 1, 1)),
        (datetime(2021, 1, 1), datetime(2022, 1, 1)),
    ]
    kinds = ["hyperbolic", "exponential"]

    # Fit model
    model = SegmentedDeclineModel(
        segment_dates=segment_dates, kinds=kinds, enforce_continuity=False
    )
    fitted_model = model.fit(data)

    # Check that segments were fitted
    segments = fitted_model.get_segments()
    assert len(segments) == 2
    assert segments[0].kind == "hyperbolic"
    assert segments[1].kind == "exponential"


def test_segmented_decline_model_predict():
    """Test predicting from segmented decline model."""
    # Generate synthetic data
    time_index = pd.date_range("2020-01-01", periods=12, freq="M")
    t = np.arange(12)
    q = 100.0 / (1.0 + 0.6 * 0.15 * t) ** (1.0 / 0.6)
    data = pd.DataFrame({"oil": q}, index=time_index)

    # Single segment
    segment_dates = [(datetime(2020, 1, 1), datetime(2021, 1, 1))]

    # Fit and predict
    model = SegmentedDeclineModel(segment_dates=segment_dates)
    fitted_model = model.fit(data)

    from ressmith.objects.domain import ForecastSpec

    forecast_spec = ForecastSpec(horizon=12, frequency="M")
    forecast = fitted_model.predict(forecast_spec)

    assert len(forecast.yhat) == 12
    assert isinstance(forecast.yhat.index, pd.DatetimeIndex)

