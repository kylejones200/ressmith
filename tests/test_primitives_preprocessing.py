"""Tests for preprocessing primitives."""

import numpy as np
import pandas as pd
import pytest

from ressmith.primitives.preprocessing import (
    compute_cum_from_rate,
    compute_rate_from_cum,
)


def test_compute_cum_from_rate_roundtrip():
    """Test cum from rate and rate from cum round trip."""
    # Create synthetic data
    time_index = pd.date_range("2020-01-01", periods=10, freq="D")
    # Simple constant rate
    rate = np.ones(10) * 100.0  # 100 units/day

    # Compute cumulative
    cum = compute_cum_from_rate(time_index, rate)

    # Should be increasing
    assert np.all(np.diff(cum) >= 0)

    # Compute rate back
    rate_recovered = compute_rate_from_cum(time_index, cum)

    # Should be close (allowing for numerical differences)
    np.testing.assert_allclose(rate[1:], rate_recovered[1:], rtol=1e-3)


def test_compute_cum_from_rate_simple():
    """Test cumulative computation on simple case."""
    time_index = pd.date_range("2020-01-01", periods=3, freq="D")
    rate = np.array([100.0, 100.0, 100.0])

    cum = compute_cum_from_rate(time_index, rate)

    # First value should be 0
    assert cum[0] == 0.0
    # Should be increasing
    assert cum[1] > 0
    assert cum[2] > cum[1]


def test_compute_rate_from_cum_simple():
    """Test rate computation from cumulative."""
    time_index = pd.date_range("2020-01-01", periods=3, freq="D")
    # Linear cumulative
    cum = np.array([0.0, 100.0, 200.0])

    rate = compute_rate_from_cum(time_index, cum)

    # Should recover approximately constant rate
    assert rate[1] > 0
    assert rate[2] > 0

