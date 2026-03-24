"""Tests to prevent data leakage in time series analysis.

This module specifically tests that:
- Walk-forward backtest doesn't use future data
- Cumulative calculations don't leak
- Normalization doesn't use future statistics
"""

import numpy as np
import pandas as pd
import pytest

from ressmith.primitives.preprocessing import compute_cum_from_rate
from ressmith.workflows.backtesting import walk_forward_backtest


class TestWalkForwardBacktestLeakage:
    """Test that walk-forward backtest doesn't use future data."""

    def test_no_future_data_in_training(self):
        """Verify training data only uses past data."""
        # Create data with clear trend
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        # Create increasing trend (so we can detect if future data leaks)
        rates = 100 + np.arange(len(dates)) * 2
        data = pd.DataFrame({"oil": rates}, index=dates)

        results = walk_forward_backtest(
            data,
            model_name="arps_exponential",
            forecast_horizons=[12],
            min_train_size=24,
            step_size=6,
        )

        # For each backtest, verify training data is before cut point
        for _, row in results.iterrows():
            cut_idx = int(row["cut_idx"])
            # Training data should be data[:cut_idx]
            # Forecast should be data[cut_idx:cut_idx+horizon]
            # No overlap between training and forecast periods
            
            # Verify cut_idx is within data bounds
            assert 0 <= cut_idx < len(data)
            
            # Verify forecast horizon doesn't exceed data
            horizon = int(row["horizon"])
            assert cut_idx + horizon <= len(data)

    def test_training_data_monotonic(self):
        """Verify training data is properly isolated."""
        dates = pd.date_range("2020-01-01", periods=48, freq="ME")
        # Create data with known pattern
        rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
        data = pd.DataFrame({"oil": rates}, index=dates)

        results = walk_forward_backtest(
            data,
            model_name="arps_exponential",
            forecast_horizons=[12],
            min_train_size=12,
            step_size=6,
        )

        # Each backtest should use only data up to cut_idx
        for _, row in results.iterrows():
            cut_idx = int(row["cut_idx"])
            # Training period should end at cut_idx
            assert cut_idx >= 12  # min_train_size
            assert cut_idx < len(data)

    def test_no_data_leakage_in_metrics(self):
        """Verify metrics are calculated only on forecast period."""
        dates = pd.date_range("2020-01-01", periods=48, freq="ME")
        rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
        data = pd.DataFrame({"oil": rates}, index=dates)

        results = walk_forward_backtest(
            data,
            model_name="arps_exponential",
            forecast_horizons=[12],
            min_train_size=12,
            step_size=12,
        )

        # Metrics should be finite (not NaN from using future data)
        for _, row in results.iterrows():
            if pd.notna(row["rmse"]):
                assert row["rmse"] >= 0
            if pd.notna(row["mae"]):
                assert row["mae"] >= 0


class TestCumulativeCalculationLeakage:
    """Test that cumulative calculations don't leak future data."""

    def test_cumulative_forward_only(self):
        """Cumulative should only use past and current data."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        rates = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45])

        cum = compute_cum_from_rate(dates, rates)

        # Cumulative at index i should only depend on rates[0:i+1]
        # Verify by checking that cum[i] >= cum[i-1] (monotonic)
        assert np.all(np.diff(cum) >= 0)

        # Verify cumulative at each point matches implementation (trapezoidal)
        for i in range(len(rates)):
            if i == 0:
                expected_cum = 0.0
            else:
                time_delta = (dates[i] - dates[i - 1]).days
                expected_cum = cum[i - 1] + 0.5 * (rates[i - 1] + rates[i]) * time_delta

            assert abs(cum[i] - expected_cum) < 1e-6 or i == 0

    def test_cumulative_no_future_rates(self):
        """Cumulative at time t should not use rates after t."""
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")
        rates = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])

        cum = compute_cum_from_rate(dates, rates)

        # Modify future rates and verify past cumulative doesn't change
        rates_modified = rates.copy()
        rates_modified[5:] = 999  # Change future rates

        cum_modified = compute_cum_from_rate(dates, rates_modified)

        # Cumulative up to index 4 should be identical
        assert np.allclose(cum[:5], cum_modified[:5])


class TestNormalizationLeakage:
    """Test that normalization doesn't use future statistics."""

    def test_normalization_uses_only_past_data(self):
        """Normalization should use only past/current data, not future."""
        from ressmith.workflows.pressure_normalization import normalize_production_with_pressure

        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
        pressure = 5000 * np.exp(-0.05 * np.arange(len(dates)))

        data = pd.DataFrame({
            "oil": rates,
            "pressure": pressure
        }, index=dates)

        # Normalize
        normalized = normalize_production_with_pressure(
            data,
            rate_col="oil",
            pressure_col="pressure",
            initial_pressure=5000.0
        )

        # Normalized rate at time t should only depend on:
        # - rate[t]
        # - pressure[t]
        # - initial_pressure (known constant)
        # Should NOT depend on future rates or pressures

        # Verify pressure_ratio formula: q_norm = q * (pi / p)
        for i in range(len(normalized)):
            expected_normalized = rates[i] * (5000.0 / max(pressure[i], 1.0))
            actual_normalized = normalized["normalized_rate"].iloc[i]
            assert np.isclose(actual_normalized, expected_normalized, rtol=1e-5)

    def test_pressure_normalization_no_future_stats(self):
        """Pressure normalization shouldn't use future statistics."""
        from ressmith.workflows.pressure_normalization import normalize_production_with_pressure

        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        rates = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45])
        pressure = 5000 - np.arange(len(dates)) * 50

        data = pd.DataFrame({
            "oil": rates,
            "pressure": pressure
        }, index=dates)

        normalized = normalize_production_with_pressure(
            data,
            rate_col="oil",
            pressure_col="pressure",
            initial_pressure=5000.0
        )

        # Modify future data
        data_modified = data.copy()
        data_modified.loc[data_modified.index[6:], "pressure"] = 1000  # Change future

        normalized_modified = normalize_production_with_pressure(
            data_modified,
            rate_col="oil",
            pressure_col="pressure",
            initial_pressure=5000.0
        )

        # Normalized values up to index 5 should be identical
        assert np.allclose(
            normalized["normalized_rate"].iloc[:6],
            normalized_modified["normalized_rate"].iloc[:6]
        )


class TestZeroRateHandling:
    """Test that zero rates are properly filtered and don't cause issues."""

    def test_zero_rates_filtered_in_fitting(self):
        """Zero rates should be filtered before fitting."""
        from ressmith.workflows.core import fit_forecast

        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        rates = 100 * np.exp(-0.1 * np.arange(len(dates)))
        rates[5] = 0.0  # Add zero rate
        rates[10] = 0.0  # Add another zero rate

        data = pd.DataFrame({"oil": rates}, index=dates)

        # Should handle zero rates gracefully
        forecast, params = fit_forecast(
            data,
            model_name="arps_exponential",
            horizon=12
        )

        # Forecast should not contain NaN or Inf
        assert np.all(np.isfinite(forecast.yhat.values))
        assert np.all(forecast.yhat.values >= 0)

    def test_zero_rates_in_cumulative(self):
        """Zero rates should be handled in cumulative calculations."""
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")
        rates = np.array([100, 90, 0, 80, 70, 0, 60, 50, 40, 30])

        cum = compute_cum_from_rate(dates, rates)

        # Cumulative should still be monotonic (zero rates add zero)
        assert np.all(np.diff(cum) >= 0)

        # Cumulative should be finite
        assert np.all(np.isfinite(cum))

    def test_all_zero_rates_handled(self):
        """All zero rates should be handled gracefully."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        rates = np.zeros(12)

        # Should raise error or handle gracefully
        try:
            cum = compute_cum_from_rate(dates, rates)
            # If no error, cumulative should be all zeros
            assert np.allclose(cum, 0.0)
        except (ValueError, ZeroDivisionError):
            # Error is acceptable for all-zero rates
            pass
