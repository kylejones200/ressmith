"""Tests for backtesting workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.workflows.backtesting import walk_forward_backtest


@pytest.fixture
def synthetic_historical_data():
    """Create synthetic historical production data."""
    dates = pd.date_range("2020-01-01", periods=48, freq="M")
    # Create declining rates with some noise
    t = np.arange(len(dates))
    rates = 100 * np.exp(-0.1 * t) + np.random.normal(0, 2, size=len(t))
    rates = np.maximum(rates, 1.0)  # Ensure positive
    
    return pd.DataFrame({"oil": rates}, index=dates)


def test_walk_forward_backtest_basic(synthetic_historical_data):
    """Test basic walk-forward backtest."""
    results = walk_forward_backtest(
        synthetic_historical_data,
        model_name="arps_exponential",
        forecast_horizons=[12],
        min_train_size=12,
        step_size=6,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "cut_idx" in results.columns
    assert "horizon" in results.columns
    assert "rmse" in results.columns
    assert "mae" in results.columns
    assert "mape" in results.columns
    assert "r_squared" in results.columns


def test_walk_forward_backtest_multiple_horizons(synthetic_historical_data):
    """Test walk-forward backtest with multiple horizons."""
    results = walk_forward_backtest(
        synthetic_historical_data,
        model_name="arps_hyperbolic",
        forecast_horizons=[6, 12, 24],
        min_train_size=12,
        step_size=12,
    )

    assert len(results) > 0
    # Should have results for each horizon
    horizons_in_results = results["horizon"].unique()
    assert all(h in horizons_in_results for h in [6, 12, 24])


def test_walk_forward_backtest_min_train_size(synthetic_historical_data):
    """Test that min_train_size is respected."""
    results = walk_forward_backtest(
        synthetic_historical_data,
        model_name="arps_exponential",
        forecast_horizons=[12],
        min_train_size=24,
        step_size=6,
    )

    # All cut_idx should be >= min_train_size
    assert all(results["cut_idx"] >= 24)


def test_walk_forward_backtest_errors_handled(synthetic_historical_data):
    """Test that backtest handles errors gracefully."""
    # Create problematic data that might cause errors
    bad_data = synthetic_historical_data.copy()
    bad_data.iloc[10:15, 0] = np.nan  # Add some NaN values

    # Should still return results, possibly with NaN metrics
    results = walk_forward_backtest(
        bad_data,
        model_name="arps_exponential",
        forecast_horizons=[6],
        min_train_size=12,
        step_size=6,
    )

    assert isinstance(results, pd.DataFrame)
    # May have NaN values in metrics for failed backtests
    assert "rmse" in results.columns

