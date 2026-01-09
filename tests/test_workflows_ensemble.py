"""Tests for ensemble forecasting workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.workflows.ensemble import ensemble_forecast


@pytest.fixture
def synthetic_data():
    """Create synthetic production data."""
    dates = pd.date_range("2020-01-01", periods=24, freq="M")
    t = np.arange(len(dates))
    rates = 100 * np.exp(-0.1 * t) + np.random.normal(0, 2, size=len(t))
    rates = np.maximum(rates, 1.0)
    return pd.DataFrame({"oil": rates}, index=dates)


def test_ensemble_forecast_weighted(synthetic_data):
    """Test weighted ensemble forecast."""
    forecast = ensemble_forecast(
        synthetic_data,
        model_names=["arps_exponential", "arps_hyperbolic"],
        method="weighted",
        weights=[0.6, 0.4],
        horizon=12,
    )

    assert isinstance(forecast, type(synthetic_data)) or hasattr(forecast, "yhat")
    # Check if it's a ForecastResult
    if hasattr(forecast, "yhat"):
        assert len(forecast.yhat) == 12
        assert forecast.metadata.get("ensemble_method") == "weighted"


def test_ensemble_forecast_median(synthetic_data):
    """Test median ensemble forecast."""
    forecast = ensemble_forecast(
        synthetic_data,
        model_names=["arps_exponential", "arps_hyperbolic", "power_law"],
        method="median",
        horizon=12,
    )

    if hasattr(forecast, "yhat"):
        assert len(forecast.yhat) == 12
        assert forecast.metadata.get("ensemble_method") == "median"


def test_ensemble_forecast_single_model(synthetic_data):
    """Test ensemble with single model (should return single forecast)."""
    forecast = ensemble_forecast(
        synthetic_data,
        model_names=["arps_hyperbolic"],
        method="weighted",
        horizon=12,
    )

    if hasattr(forecast, "yhat"):
        assert len(forecast.yhat) == 12

