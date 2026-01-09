"""Tests for workflow functions."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import ForecastResult
from ressmith.workflows import fit_forecast


def test_fit_forecast_returns_pandas_outputs():
    """Test that fit_forecast returns pandas outputs with aligned index."""
    # Generate synthetic hyperbolic decline
    time_index = pd.date_range("2020-01-01", periods=24, freq="D")
    t = np.arange(24)
    qi = 100.0
    di = 0.1
    b = 0.6
    # Hyperbolic decline
    q = qi / (1.0 + b * di * t) ** (1.0 / b)
    # Add noise
    noise = np.random.normal(0, 2.0, len(q))
    q_noisy = np.maximum(q + noise, 0.1)

    # Create DataFrame
    data = pd.DataFrame({"oil": q_noisy}, index=time_index)

    # Fit and forecast
    forecast, params = fit_forecast(
        data, model_name="arps_hyperbolic", horizon=12
    )

    # Check outputs
    assert isinstance(forecast, ForecastResult)
    assert isinstance(forecast.yhat, pd.Series)
    assert len(forecast.yhat) == 12
    assert isinstance(forecast.yhat.index, pd.DatetimeIndex)

    # Check params
    assert isinstance(params, dict)
    assert "qi" in params
    assert "di" in params
    assert "b" in params

    # Check forecast is reasonable
    assert forecast.yhat.min() > 0
    assert forecast.yhat.max() < 200  # Should be less than initial rate

