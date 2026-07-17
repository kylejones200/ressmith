import numpy as np

from ressmith.primitives.geostats import (
    compute_empirical_variogram,
    fit_variogram_wls,
    ordinary_kriging,
    variogram_model,
)


def test_variogram_models_are_finite_and_nonnegative():
    h = np.linspace(0.1, 50, 20)
    for model in ("spherical", "exponential", "gaussian"):
        gamma = variogram_model(h, model, range_=30, sill=10, nugget=1)
        assert gamma.shape == h.shape
        assert np.all(np.isfinite(gamma))
        assert np.all(gamma >= 0)


def test_empirical_variogram_returns_populated_bins():
    rng = np.random.default_rng(42)
    x = rng.random(30) * 100
    y = rng.random(30) * 100
    z = 50 + 0.5 * x + 0.3 * y + rng.normal(0, 5, 30)
    lags, gamma, counts = compute_empirical_variogram(x, y, z, n_lags=10)
    assert len(lags) == len(gamma) == len(counts)
    assert len(lags) > 0
    assert np.all(counts > 0)
    assert np.nanmax(gamma) > 0


def test_weighted_variogram_fit_returns_valid_parameters():
    lags = np.array([5, 15, 25, 35, 45])
    gamma = np.array([2, 6, 8, 9.2, 9.8])
    counts = np.array([10, 20, 25, 18, 12])
    range_, sill, nugget = fit_variogram_wls(lags, gamma, counts, "spherical")
    assert range_ > 0
    assert sill > 0
    assert nugget >= 0


def test_ordinary_kriging_produces_finite_prediction_and_variance():
    rng = np.random.default_rng(42)
    x = rng.random(20) * 50
    y = rng.random(20) * 50
    z = 10 + 0.2 * x + 0.1 * y + rng.normal(size=20)
    prediction, variance = ordinary_kriging(
        x,
        y,
        z,
        np.array([25.0]),
        np.array([25.0]),
        model="spherical",
        range_=25,
        sill=5,
        nugget=1,
    )
    assert np.isfinite(prediction[0])
    assert np.isfinite(variance[0])
    assert variance[0] >= 0
    assert abs(prediction[0] - np.mean(z)) < 20
