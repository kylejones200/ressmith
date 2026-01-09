"""Tests for decline curve variants."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import ForecastSpec, RateSeries
from ressmith.primitives.models import FixedTerminalDeclineModel
from ressmith.primitives.variants import (
    fit_fixed_terminal_decline,
    fixed_terminal_decline_rate,
)


@pytest.fixture
def synthetic_declining_data():
    """Create synthetic declining production data."""
    dates = pd.date_range("2020-01-01", periods=36, freq="M")
    t = np.arange(len(dates))
    # Hyperbolic decline
    qi, di, b = 100.0, 0.3, 0.7
    rates = qi / np.power(1 + b * di * t, 1 / b) + np.random.normal(0, 2, size=len(t))
    rates = np.maximum(rates, 1.0)
    return RateSeries(time_index=dates, rate=rates, units={})


def test_fixed_terminal_decline_rate():
    """Test fixed terminal decline rate calculation."""
    t = np.arange(0, 120, 1.0)
    qi, di, b = 100.0, 0.3, 0.7
    t_switch = 60.0
    di_terminal = 0.05 / 12.0  # 5% per year in monthly

    rates = fixed_terminal_decline_rate(t, qi, di, b, t_switch, di_terminal)

    assert len(rates) == len(t)
    assert all(rates >= 0)
    # Should decline
    assert rates[-1] < rates[0]
    # Rate at switch should match
    rate_at_switch = rates[int(t_switch)]
    assert rate_at_switch > 0


def test_fit_fixed_terminal_decline():
    """Test fitting fixed terminal decline model."""
    t = np.arange(0, 36, 1.0)
    qi, di, b = 100.0, 0.3, 0.7
    # Generate synthetic hyperbolic decline
    q = qi / np.power(1 + b * di * t, 1 / b) + np.random.normal(0, 1, size=len(t))
    q = np.maximum(q, 1.0)

    params = fit_fixed_terminal_decline(
        t, q, kind="hyperbolic", terminal_decline_rate=0.05
    )

    assert "qi" in params
    assert "di" in params
    assert "b" in params
    assert "t_switch" in params
    assert "di_terminal" in params
    assert params["qi"] > 0
    assert params["di"] > 0
    assert params["t_switch"] >= 0


def test_fixed_terminal_decline_model_fit(synthetic_declining_data):
    """Test FixedTerminalDeclineModel fitting."""
    model = FixedTerminalDeclineModel(
        kind="hyperbolic", terminal_decline_rate=0.06, transition_criteria="rate"
    )
    fitted = model.fit(synthetic_declining_data)

    assert fitted.is_fitted
    assert "qi" in fitted._fitted_params
    assert "di" in fitted._fitted_params
    assert "t_switch" in fitted._fitted_params
    assert "di_terminal" in fitted._fitted_params


def test_fixed_terminal_decline_model_predict(synthetic_declining_data):
    """Test FixedTerminalDeclineModel prediction."""
    model = FixedTerminalDeclineModel(
        kind="hyperbolic", terminal_decline_rate=0.05
    )
    fitted = model.fit(synthetic_declining_data)

    spec = ForecastSpec(horizon=120, frequency="M")
    forecast = fitted.predict(spec)

    assert len(forecast.yhat) == 120
    assert all(forecast.yhat.values >= 0)
    # Should show decline
    assert forecast.yhat.iloc[-1] < forecast.yhat.iloc[0]
    assert forecast.model_spec.model_name == "fixed_terminal_decline"


def test_fixed_terminal_decline_transition_time():
    """Test transition at fixed time."""
    t = np.arange(0, 48, 1.0)
    q = 100 * np.exp(-0.1 * t) + np.random.normal(0, 1, size=len(t))
    q = np.maximum(q, 1.0)

    params = fit_fixed_terminal_decline(
        t,
        q,
        kind="exponential",
        terminal_decline_rate=0.05,
        transition_criteria="time",
        transition_value=24.0,  # Transition at 24 months
    )

    assert params["t_switch"] == 24.0

