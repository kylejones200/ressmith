"""Tests for decline primitives."""

import numpy as np
import pandas as pd
import pytest

from ressmith.primitives.decline import (
    arps_exponential,
    arps_hyperbolic,
    arps_harmonic,
    fit_arps_exponential,
    fit_arps_hyperbolic,
    fit_arps_harmonic,
)


def test_arps_exponential_shape():
    """Test exponential decline shape."""
    t = np.linspace(0, 10, 100)
    qi = 100.0
    di = 0.1

    q = arps_exponential(t, qi, di)

    # Should start at qi
    assert np.isclose(q[0], qi)
    # Should be decreasing
    assert np.all(np.diff(q) <= 0)
    # Should be positive
    assert np.all(q > 0)


def test_arps_hyperbolic_shape():
    """Test hyperbolic decline shape."""
    t = np.linspace(0, 10, 100)
    qi = 100.0
    di = 0.1
    b = 0.5

    q = arps_hyperbolic(t, qi, di, b)

    # Should start at qi
    assert np.isclose(q[0], qi)
    # Should be decreasing
    assert np.all(np.diff(q) <= 0)
    # Should be positive
    assert np.all(q > 0)


def test_arps_harmonic_shape():
    """Test harmonic decline shape."""
    t = np.linspace(0, 10, 100)
    qi = 100.0
    di = 0.1

    q = arps_harmonic(t, qi, di)

    # Should start at qi
    assert np.isclose(q[0], qi)
    # Should be decreasing
    assert np.all(np.diff(q) <= 0)
    # Should be positive
    assert np.all(q > 0)


def test_fit_arps_exponential_synthetic():
    """Test exponential fit on synthetic data."""
    # Generate synthetic data
    t = np.linspace(0, 10, 50)
    qi_true = 100.0
    di_true = 0.15
    q_true = arps_exponential(t, qi_true, di_true)
    # Add noise
    noise = np.random.normal(0, 1.0, len(q_true))
    q_noisy = q_true + noise
    q_noisy = np.maximum(q_noisy, 0.1)  # Ensure positive

    # Fit
    params = fit_arps_exponential(t, q_noisy)

    # Check parameters are reasonable
    assert "qi" in params
    assert "di" in params
    assert params["qi"] > 0
    assert params["di"] > 0
    # Should be close to true values (allowing for noise)
    assert abs(params["qi"] - qi_true) < 20
    assert abs(params["di"] - di_true) < 0.1


def test_fit_arps_hyperbolic_synthetic():
    """Test hyperbolic fit on synthetic data."""
    # Generate synthetic data
    t = np.linspace(0, 10, 50)
    qi_true = 100.0
    di_true = 0.15
    b_true = 0.6
    q_true = arps_hyperbolic(t, qi_true, di_true, b_true)
    # Add noise
    noise = np.random.normal(0, 1.0, len(q_true))
    q_noisy = q_true + noise
    q_noisy = np.maximum(q_noisy, 0.1)  # Ensure positive

    # Fit
    params = fit_arps_hyperbolic(t, q_noisy)

    # Check parameters are reasonable
    assert "qi" in params
    assert "di" in params
    assert "b" in params
    assert params["qi"] > 0
    assert params["di"] > 0
    assert 0 < params["b"] < 1

