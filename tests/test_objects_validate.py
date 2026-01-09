"""Tests for Layer 1 validators."""

import numpy as np
import pandas as pd
import pytest

from ressmith.objects.validate import (
    assert_alignment,
    assert_monotonic_time,
    assert_positive_rates,
    assert_rate_series_alignment,
)
from ressmith.objects.domain import ProductionSeries, RateSeries


def test_assert_monotonic_time():
    """Test monotonic time validation."""
    # Valid: strictly increasing
    time_index = pd.date_range("2020-01-01", periods=10, freq="D")
    assert_monotonic_time(time_index)  # Should not raise

    # Invalid: duplicates
    time_index_dup = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-01", "2020-01-02"]
    )
    with pytest.raises(ValueError, match="duplicate"):
        assert_monotonic_time(time_index_dup)

    # Invalid: decreasing
    time_index_dec = pd.DatetimeIndex(
        ["2020-01-03", "2020-01-02", "2020-01-01"]
    )
    with pytest.raises(ValueError, match="monotonic"):
        assert_monotonic_time(time_index_dec)


def test_assert_positive_rates():
    """Test positive rates validation."""
    # Valid: all positive
    rates = np.array([1.0, 2.0, 3.0])
    assert_positive_rates(rates)  # Should not raise

    # Invalid: negative values
    rates_neg = np.array([1.0, -0.5, 3.0])
    with pytest.raises(ValueError, match="negative"):
        assert_positive_rates(rates_neg)


def test_assert_rate_series_alignment():
    """Test rate series alignment validation."""
    time_index = pd.date_range("2020-01-01", periods=5, freq="D")
    rate = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    series = RateSeries(time_index=time_index, rate=rate)
    assert_rate_series_alignment(series)  # Should not raise

    # Invalid: misaligned (dataclass __post_init__ will catch this)
    rate_bad = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="length"):
        RateSeries(time_index=time_index, rate=rate_bad)


def test_assert_alignment():
    """Test production series alignment validation."""
    time_index = pd.date_range("2020-01-01", periods=5, freq="D")
    oil = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gas = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    water = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    series = ProductionSeries(
        time_index=time_index, oil=oil, gas=gas, water=water
    )
    assert_alignment(series)  # Should not raise

    # Invalid: misaligned (dataclass __post_init__ will catch this)
    oil_bad = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="length"):
        ProductionSeries(
            time_index=time_index, oil=oil_bad, gas=gas, water=water
        )

