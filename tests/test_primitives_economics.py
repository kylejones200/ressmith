"""Tests for economics primitives."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.primitives.economics import cashflow_from_forecast, npv


def test_cashflow_from_forecast():
    """Test cashflow computation from forecast."""
    # Create simple forecast
    time_index = pd.date_range("2020-01-01", periods=12, freq="M")
    yhat = pd.Series([100.0] * 12, index=time_index, name="forecast")
    forecast = ForecastResult(yhat=yhat)

    # Create econ spec
    spec = EconSpec(
        price_assumptions={"oil": 50.0},
        opex=10.0,
        capex=1000.0,
        discount_rate=0.1,
    )

    # Compute cashflows
    cashflows = cashflow_from_forecast(forecast, spec)

    # Check structure
    assert isinstance(cashflows, pd.DataFrame)
    assert "revenue" in cashflows.columns
    assert "opex" in cashflows.columns
    assert "capex" in cashflows.columns
    assert "net_cashflow" in cashflows.columns

    # Check values
    assert cashflows.loc[0, "capex"] == -1000.0
    assert cashflows.loc[0, "revenue"] == 100.0 * 50.0
    assert cashflows.loc[0, "opex"] == -10.0


def test_npv():
    """Test NPV computation."""
    # Simple cashflow: invest 1000, get 200 per period for 5 periods
    cashflows = np.array([-1000.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    discount_rate = 0.1

    npv_value = npv(cashflows, discount_rate)

    # Should be negative (not profitable at 10% discount)
    assert npv_value < 0
    # Should be reasonable magnitude
    assert abs(npv_value) < 1000

