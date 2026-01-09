"""Tests for ensemble forecasting."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import ForecastResult, ForecastSpec
from ressmith.primitives.ensemble import (
    confidence_weighted_ensemble,
    median_ensemble_forecast,
    weighted_ensemble_forecast,
)


@pytest.fixture
def sample_forecasts():
    """Create sample forecasts for testing."""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")

    # Three different forecasts
    forecast1 = ForecastResult(
        yhat=pd.Series(100 * np.exp(-0.1 * np.arange(12)), index=dates, name="forecast1"),
        metadata={"model_name": "model1", "r_squared": 0.9},
    )
    forecast2 = ForecastResult(
        yhat=pd.Series(95 * np.exp(-0.12 * np.arange(12)), index=dates, name="forecast2"),
        metadata={"model_name": "model2", "r_squared": 0.85},
    )
    forecast3 = ForecastResult(
        yhat=pd.Series(105 * np.exp(-0.08 * np.arange(12)), index=dates, name="forecast3"),
        metadata={"model_name": "model3", "r_squared": 0.8},
    )

    return [forecast1, forecast2, forecast3]


def test_weighted_ensemble_forecast(sample_forecasts):
    """Test weighted ensemble forecast."""
    result = weighted_ensemble_forecast(sample_forecasts, weights=[0.5, 0.3, 0.2])

    assert isinstance(result, ForecastResult)
    assert len(result.yhat) == 12
    assert all(result.yhat.values >= 0)
    assert result.metadata["ensemble_method"] == "weighted"
    assert result.metadata["n_models"] == 3


def test_weighted_ensemble_equal_weights(sample_forecasts):
    """Test weighted ensemble with equal weights."""
    result = weighted_ensemble_forecast(sample_forecasts)

    assert isinstance(result, ForecastResult)
    # Should be average of three forecasts
    avg_first = (
        sample_forecasts[0].yhat.iloc[0]
        + sample_forecasts[1].yhat.iloc[0]
        + sample_forecasts[2].yhat.iloc[0]
    ) / 3
    assert abs(result.yhat.iloc[0] - avg_first) < 1.0


def test_median_ensemble_forecast(sample_forecasts):
    """Test median ensemble forecast."""
    result = median_ensemble_forecast(sample_forecasts)

    assert isinstance(result, ForecastResult)
    assert len(result.yhat) == 12
    assert result.metadata["ensemble_method"] == "median"
    assert result.metadata["n_models"] == 3


def test_confidence_weighted_ensemble(sample_forecasts):
    """Test confidence-weighted ensemble."""
    result = confidence_weighted_ensemble(sample_forecasts)

    assert isinstance(result, ForecastResult)
    assert len(result.yhat) == 12
    # Model with highest R² should have more weight
    assert result.metadata["ensemble_method"] == "weighted"


def test_ensemble_single_forecast(sample_forecasts):
    """Test ensemble with single forecast."""
    result = weighted_ensemble_forecast([sample_forecasts[0]])

    assert isinstance(result, ForecastResult)
    assert len(result.yhat) == len(sample_forecasts[0].yhat)
    # Should return the single forecast unchanged
    assert result.yhat.iloc[0] == sample_forecasts[0].yhat.iloc[0]

