"""Tests for multi-phase forecasting workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.workflows.multiphase import forecast_with_yields


@pytest.fixture
def multiphase_data():
    """Create synthetic multi-phase production data."""
    dates = pd.date_range("2020-01-01", periods=36, freq="M")
    # Oil decline
    t = np.arange(len(dates))
    oil = 100 * np.exp(-0.1 * t) + np.random.normal(0, 2, size=len(t))
    oil = np.maximum(oil, 1.0)

    # Gas from oil (constant GOR ~ 1000 scf/bbl)
    gor = 1000.0
    gas = oil * gor + np.random.normal(0, 100, size=len(oil))
    gas = np.maximum(gas, 0.0)

    # Water from oil (constant water cut ~ 0.1)
    water_cut = 0.1
    water = oil * water_cut + np.random.normal(0, 1, size=len(oil))
    water = np.maximum(water, 0.0)

    return pd.DataFrame(
        {"oil": oil, "gas": gas, "water": water}, index=dates
    )


def test_forecast_with_yields(multiphase_data):
    """Test multi-phase forecasting with yield models."""
    results = forecast_with_yields(
        multiphase_data,
        primary_phase="oil",
        associated_phases=["gas", "water"],
        model_name="arps_exponential",
        horizon=24,
    )

    assert "oil" in results
    assert "gas" in results
    assert "water" in results

    # Check forecasts have correct length
    assert len(results["oil"].yhat) == 24
    assert len(results["gas"].yhat) == 24
    assert len(results["water"].yhat) == 24

    # Check yields are reasonable
    # Gas should be roughly 1000x oil (GOR)
    gas_yield = results["gas"].yhat.iloc[0] / results["oil"].yhat.iloc[0]
    assert 500 < gas_yield < 2000  # Rough check for GOR

    # Water should be roughly 0.1x oil (water cut)
    water_yield = results["water"].yhat.iloc[0] / results["oil"].yhat.iloc[0]
    assert 0.0 < water_yield < 0.5  # Rough check for water cut


def test_forecast_with_yields_specified_models(multiphase_data):
    """Test multi-phase forecasting with specified yield models."""
    results = forecast_with_yields(
        multiphase_data,
        primary_phase="oil",
        associated_phases=["gas"],
        yield_models={"gas": "constant"},
        horizon=12,
    )

    assert "oil" in results
    assert "gas" in results
    assert results["gas"].metadata["yield_model"] == "constant"


def test_forecast_with_yields_missing_phase(multiphase_data):
    """Test multi-phase forecasting with missing phase data."""
    # Remove water column
    data_no_water = multiphase_data.drop(columns=["water"])

    results = forecast_with_yields(
        data_no_water,
        primary_phase="oil",
        associated_phases=["water"],  # Request missing phase
        horizon=12,
    )

    # Should skip missing phase but still return results
    assert "oil" in results
    # Water might be skipped or return zero forecast

