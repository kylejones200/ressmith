"""Tests for advanced decline models."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import ForecastSpec, RateSeries
from ressmith.primitives.models import (
    DuongModel,
    PowerLawDeclineModel,
    StretchedExponentialModel,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic decline data."""
    dates = pd.date_range("2020-01-01", periods=24, freq="M")
    # Create declining rates
    t = np.arange(len(dates))
    rates = 100 * np.exp(-0.1 * t) + np.random.normal(0, 2, size=len(t))
    rates = np.maximum(rates, 1.0)  # Ensure positive
    return RateSeries(time_index=dates, rate=rates, units={})


def test_power_law_model_fit(synthetic_data):
    """Test PowerLawDeclineModel fitting."""
    model = PowerLawDeclineModel()
    fitted = model.fit(synthetic_data)

    assert fitted.is_fitted
    assert "qi" in fitted._fitted_params
    assert "di" in fitted._fitted_params
    assert "n" in fitted._fitted_params
    assert fitted._fitted_params["qi"] > 0
    assert fitted._fitted_params["di"] > 0
    assert fitted._fitted_params["n"] > 0


def test_power_law_model_predict(synthetic_data):
    """Test PowerLawDeclineModel prediction."""
    model = PowerLawDeclineModel()
    fitted = model.fit(synthetic_data)

    spec = ForecastSpec(horizon=12, frequency="M")
    forecast = fitted.predict(spec)

    assert len(forecast.yhat) == 12
    assert all(forecast.yhat.values >= 0)  # Non-negative rates
    assert forecast.model_spec.model_name == "power_law"


def test_duong_model_fit(synthetic_data):
    """Test DuongModel fitting."""
    model = DuongModel()
    fitted = model.fit(synthetic_data)

    assert fitted.is_fitted
    assert "qi" in fitted._fitted_params
    assert "a" in fitted._fitted_params
    assert "m" in fitted._fitted_params
    assert fitted._fitted_params["qi"] > 0
    assert fitted._fitted_params["a"] > 0
    assert 0 < fitted._fitted_params["m"] < 2


def test_duong_model_predict(synthetic_data):
    """Test DuongModel prediction."""
    model = DuongModel()
    fitted = model.fit(synthetic_data)

    spec = ForecastSpec(horizon=12, frequency="M")
    forecast = fitted.predict(spec)

    assert len(forecast.yhat) == 12
    assert all(forecast.yhat.values >= 0)
    assert forecast.model_spec.model_name == "duong"


def test_stretched_exponential_model_fit(synthetic_data):
    """Test StretchedExponentialModel fitting."""
    model = StretchedExponentialModel()
    fitted = model.fit(synthetic_data)

    assert fitted.is_fitted
    assert "qi" in fitted._fitted_params
    assert "tau" in fitted._fitted_params
    assert "beta" in fitted._fitted_params
    assert fitted._fitted_params["qi"] > 0
    assert fitted._fitted_params["tau"] > 0
    assert 0 < fitted._fitted_params["beta"] <= 1


def test_stretched_exponential_model_predict(synthetic_data):
    """Test StretchedExponentialModel prediction."""
    model = StretchedExponentialModel()
    fitted = model.fit(synthetic_data)

    spec = ForecastSpec(horizon=12, frequency="M")
    forecast = fitted.predict(spec)

    assert len(forecast.yhat) == 12
    assert all(forecast.yhat.values >= 0)
    assert forecast.model_spec.model_name == "stretched_exponential"


def test_advanced_models_with_dataframe():
    """Test advanced models with DataFrame input."""
    dates = pd.date_range("2020-01-01", periods=24, freq="M")
    rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
    df = pd.DataFrame({"oil": rates}, index=dates)

    # Test PowerLaw
    model = PowerLawDeclineModel()
    fitted = model.fit(df)
    assert fitted.is_fitted

    # Test Duong
    model = DuongModel()
    fitted = model.fit(df)
    assert fitted.is_fitted

    # Test StretchedExponential
    model = StretchedExponentialModel()
    fitted = model.fit(df)
    assert fitted.is_fitted

