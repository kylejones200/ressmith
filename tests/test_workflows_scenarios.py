"""Tests for scenario analysis workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.domain import EconSpec, ForecastResult, ForecastSpec
from ressmith.workflows.scenarios import evaluate_scenarios, scenario_summary


@pytest.fixture
def sample_forecast():
    """Create a sample forecast for testing."""
    dates = pd.date_range("2020-01-01", periods=24, freq="M")
    rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
    yhat = pd.Series(rates, index=dates, name="forecast")
    return ForecastResult(yhat=yhat, metadata={})


@pytest.fixture
def base_econ_spec():
    """Create a base economics specification."""
    return EconSpec(
        price_assumptions={"oil": 70.0},
        opex=15.0,
        capex=100000.0,
        discount_rate=0.1,
    )


def test_evaluate_scenarios_basic(sample_forecast, base_econ_spec):
    """Test basic scenario evaluation."""
    scenarios = {
        "low": {"prices": {"oil": 50.0}},
        "base": {},
        "high": {"prices": {"oil": 90.0}},
    }

    results = evaluate_scenarios(sample_forecast, base_econ_spec, scenarios)

    assert len(results) == 3
    assert "low" in results
    assert "base" in results
    assert "high" in results

    # High price scenario should have higher NPV
    assert results["high"].npv > results["low"].npv
    assert results["base"].npv > results["low"].npv


def test_evaluate_scenarios_with_opex_override(sample_forecast, base_econ_spec):
    """Test scenario evaluation with opex override."""
    scenarios = {
        "high_opex": {"opex": 20.0},
        "low_opex": {"opex": 10.0},
    }

    results = evaluate_scenarios(sample_forecast, base_econ_spec, scenarios)

    # Lower opex should have higher NPV
    assert results["low_opex"].npv > results["high_opex"].npv


def test_evaluate_scenarios_with_discount_rate(sample_forecast, base_econ_spec):
    """Test scenario evaluation with discount rate override."""
    scenarios = {
        "low_discount": {"discount_rate": 0.05},
        "high_discount": {"discount_rate": 0.15},
    }

    results = evaluate_scenarios(sample_forecast, base_econ_spec, scenarios)

    # Lower discount rate should have higher NPV (all else equal)
    assert results["low_discount"].npv > results["high_discount"].npv


def test_scenario_summary(sample_forecast, base_econ_spec):
    """Test scenario summary table creation."""
    scenarios = {
        "low": {"prices": {"oil": 50.0}},
        "base": {},
        "high": {"prices": {"oil": 90.0}},
    }

    results = evaluate_scenarios(sample_forecast, base_econ_spec, scenarios)
    summary = scenario_summary(results)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3
    assert "scenario" in summary.columns
    assert "npv" in summary.columns
    assert "irr" in summary.columns
    assert "payout_time" in summary.columns
    assert "total_revenue" in summary.columns

    # Check scenarios are in summary
    assert set(summary["scenario"].values) == {"low", "base", "high"}


def test_evaluate_scenarios_multi_phase(sample_forecast, base_econ_spec):
    """Test scenario evaluation with multi-phase price assumptions."""
    base_spec_multi = EconSpec(
        price_assumptions={"oil": 70.0, "gas": 3.0},
        opex=15.0,
        capex=100000.0,
        discount_rate=0.1,
    )

    scenarios = {
        "low": {"prices": {"oil": 50.0, "gas": 2.0}},
        "high": {"prices": {"oil": 90.0, "gas": 4.0}},
    }

    results = evaluate_scenarios(sample_forecast, base_spec_multi, scenarios)

    assert len(results) == 2
    assert results["high"].npv > results["low"].npv

