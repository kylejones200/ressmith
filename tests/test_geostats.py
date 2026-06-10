"""Tests for geostatistics module (variogram and kriging)."""

import numpy as np
import pytest

from petrosmith.geostats import (
    compute_empirical_variogram,
    fit_variogram_wls,
    ordinary_kriging,
    variogram_model,
)


def test_variogram_models():
    """Variogram models return correct shape and non-negative values."""
    h = np.linspace(0.1, 50, 20)  # Avoid h=0 for clean evaluation
    for model in ["spherical", "exponential", "gaussian"]:
        g = variogram_model(h, model, range_=30.0, sill=10.0, nugget=1.0)
        assert g.shape == h.shape
        assert np.all(g >= 0)
        assert np.all(np.isfinite(g))


def test_empirical_variogram():
    """Empirical variogram returns valid lags and gamma."""
    np.random.seed(42)
    x = np.random.rand(30) * 100
    y = np.random.rand(30) * 100
    z = 50 + 0.5 * x + 0.3 * y + np.random.randn(30) * 5
    lags, gamma, counts = compute_empirical_variogram(x, y, z, n_lags=10)
    assert len(lags) > 0
    assert len(gamma) == len(lags)
    assert np.all(counts >= 0)
    assert np.nanmax(gamma) > 0


def test_fit_variogram():
    """Variogram fit returns valid parameters."""
    lags = np.array([5, 15, 25, 35, 45])
    gamma = np.array([2, 6, 8, 9.2, 9.8])
    counts = np.array([10, 20, 25, 18, 12])
    r, s, n = fit_variogram_wls(lags, gamma, counts, "spherical")
    assert r > 0
    assert s > 0
    assert n >= 0


def test_ordinary_kriging():
    """Ordinary kriging produces reasonable predictions."""
    np.random.seed(42)
    x = np.random.rand(20) * 50
    y = np.random.rand(20) * 50
    z = 10 + 0.2 * x + 0.1 * y + np.random.randn(20)
    x_grid = np.array([25.0])
    y_grid = np.array([25.0])
    z_pred, z_var = ordinary_kriging(
        x, y, z, x_grid, y_grid,
        model="spherical", range_=25.0, sill=5.0, nugget=1.0,
    )
    assert not np.isnan(z_pred[0])
    assert not np.isnan(z_var[0])
    assert z_var[0] >= 0
    assert abs(z_pred[0] - np.mean(z)) < 20  # Rough sanity check
