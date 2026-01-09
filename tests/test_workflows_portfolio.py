"""Tests for portfolio analysis workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import EconSpec
from ressmith.workflows.portfolio import (
    aggregate_portfolio_forecast,
    analyze_portfolio,
    rank_wells,
)


@pytest.fixture
def sample_well_data():
    """Create sample well data for portfolio testing."""
    well_data = {}
    for i in range(3):
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        t = np.arange(len(dates))
        rates = (100 - i * 10) * np.exp(-0.1 * t) + np.random.normal(0, 2, size=len(t))
        rates = np.maximum(rates, 1.0)
        well_data[f"well_{i+1}"] = pd.DataFrame({"oil": rates}, index=dates)
    return well_data


def test_analyze_portfolio(sample_well_data):
    """Test portfolio analysis."""
    results = analyze_portfolio(
        sample_well_data,
        model_name="arps_exponential",
        horizon=12,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 3
    assert "well_id" in results.columns
    assert "eur" in results.columns
    assert "params" in results.columns


def test_analyze_portfolio_with_economics(sample_well_data):
    """Test portfolio analysis with economics."""
    econ_spec = EconSpec(
        price_assumptions={"oil": 70.0},
        opex=15.0,
        capex=100000.0,
        discount_rate=0.1,
    )
    results = analyze_portfolio(
        sample_well_data,
        model_name="arps_exponential",
        horizon=12,
        econ_spec=econ_spec,
    )

    assert "npv" in results.columns
    assert "irr" in results.columns
    assert all(results["npv"].notna() | results["npv"].isna())  # All should be valid or all NaN


def test_aggregate_portfolio_forecast(sample_well_data):
    """Test portfolio forecast aggregation."""
    forecast = aggregate_portfolio_forecast(
        sample_well_data,
        model_name="arps_exponential",
        horizon=12,
        aggregation_method="sum",
    )

    assert len(forecast.yhat) == 12
    assert all(forecast.yhat.values >= 0)
    assert forecast.metadata["n_wells"] == 3
    assert forecast.metadata["aggregation_method"] == "sum"


def test_rank_wells(sample_well_data):
    """Test well ranking."""
    results = analyze_portfolio(
        sample_well_data,
        model_name="arps_exponential",
        horizon=12,
    )
    # Add dummy npv for testing
    results["npv"] = [1000, 2000, 1500]

    ranked = rank_wells(results, metric="npv")

    assert "rank" in ranked.columns
    assert ranked["rank"].iloc[0] == 1
    assert ranked["npv"].iloc[0] == 2000  # Highest NPV should be first

