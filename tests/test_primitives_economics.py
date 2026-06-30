"""Tests for economics primitives."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.primitives.economics import cashflow_from_forecast, npv


def test_cashflow_from_forecast():
    """Test cashflow computation from forecast."""
    # Create simple forecast
    time_index = pd.date_range("2020-01-01", periods=12, freq="ME")
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
    """NPV uses the canonical effective-annual convention (delegated to decline-curve).

    ``discount_rate`` is ANNUAL and applied at the effective monthly rate
    ``(1+r)**(1/12)-1`` over the monthly cashflow series (period 0 undiscounted).
    """
    # invest 1000 at t0, then 200/month for 5 months
    cashflows = np.array([-1000.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    annual_rate = 0.10

    npv_value = npv(cashflows, annual_rate)

    # Pinned golden under the effective-annual convention.
    assert npv_value == pytest.approx(-23.4843, abs=1e-3)

    # It must equal the explicit effective-annual reference...
    monthly = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    expected = sum(cf / (1.0 + monthly) ** i for i, cf in enumerate(cashflows))
    assert npv_value == pytest.approx(expected, rel=1e-12)

    # ...and must NOT match the old per-period bug (~-241.84), which silently
    # treated the annual rate as a monthly rate.
    old_per_period = sum(cf / (1.0 + annual_rate) ** i for i, cf in enumerate(cashflows))
    assert abs(npv_value - old_per_period) > 100.0


def test_npv_delegates_to_kernel():
    """ressmith.npv is bit-identical to the decline-curve kernel function."""
    from decline_curve.economics import npv_from_cashflow

    cashflows = np.array([-1500.0, 300.0, 280.0, 260.0, 240.0, 220.0, 200.0])
    for r in (0.05, 0.10, 0.15):
        assert npv(cashflows, r) == pytest.approx(
            npv_from_cashflow(cashflows, r), rel=1e-12
        )

