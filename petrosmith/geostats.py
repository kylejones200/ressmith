"""
Geostatistics module: variogram computation and ordinary kriging.

Implements empirical variogram, spherical/exponential/gaussian models,
and ordinary kriging for 2D spatial interpolation.
"""

from __future__ import annotations

import logging
from typing import Literal, Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

logger = logging.getLogger(__name__)

VariogramModel = Literal["spherical", "exponential", "gaussian", "linear", "power"]


def _power(h: np.ndarray, range_: float, sill: float, nugget: float = 0) -> np.ndarray:
    """Power variogram: gamma(h) = nugget + sill * (h/range)^alpha; alpha default 1.5."""
    h = np.asarray(h)
    out = np.full_like(h, nugget, dtype=float)
    mask = h > 0
    out[mask] = nugget + sill * (h[mask] / max(range_, 1e-6)) ** 1.5
    return out


def _spherical(h: np.ndarray, range_: float, sill: float, nugget: float = 0) -> np.ndarray:
    """Spherical variogram model: gamma(h) = nugget + sill * (1.5*h/a - 0.5*(h/a)^3) for h < a."""
    h = np.asarray(h)
    out = np.full_like(h, nugget + sill, dtype=float)
    mask = (h > 0) & (h < range_)
    hr = h[mask] / range_
    out[mask] = nugget + sill * (1.5 * hr - 0.5 * hr**3)
    out[(h > 0) & (h >= range_)] = nugget + sill
    return out


def _exponential(h: np.ndarray, range_: float, sill: float, nugget: float = 0) -> np.ndarray:
    """Exponential variogram: gamma(h) = nugget + sill * (1 - exp(-3h/a))."""
    h = np.asarray(h)
    out = np.full_like(h, nugget, dtype=float)
    mask = h > 0
    out[mask] = nugget + sill * (1 - np.exp(-3 * h[mask] / range_))
    return out


def _gaussian(h: np.ndarray, range_: float, sill: float, nugget: float = 0) -> np.ndarray:
    """Gaussian variogram: gamma(h) = nugget + sill * (1 - exp(-3*(h/a)^2))."""
    h = np.asarray(h)
    out = np.full_like(h, nugget, dtype=float)
    mask = h > 0
    out[mask] = nugget + sill * (1 - np.exp(-3 * (h[mask] / range_) ** 2))
    return out


def _linear(h: np.ndarray, range_: float, sill: float, nugget: float = 0) -> np.ndarray:
    """Linear variogram: gamma(h) = nugget + sill * h / range_ for h < range_, else sill+nugget."""
    h = np.asarray(h)
    out = np.full_like(h, nugget + sill, dtype=float)
    mask = (h > 0) & (h < range_)
    out[mask] = nugget + sill * h[mask] / range_
    return out


def variogram_model(
    h: np.ndarray, model: VariogramModel, range_: float, sill: float, nugget: float = 0
) -> np.ndarray:
    """Evaluate variogram model at lag distances h."""
    models = {
        "spherical": _spherical,
        "exponential": _exponential,
        "gaussian": _gaussian,
        "linear": _linear,
        "power": _power,
    }
    return models[model](h, range_, sill, nugget)


def compute_empirical_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_lags: int = 15,
    max_lag: float | None = None,
    lag_tolerance: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical (experimental) semivariogram.

    Returns:
        lags: Lag distances (bin centers)
        gamma: Semivariance values
        counts: Number of pairs per bin
    """
    coords = np.column_stack([x, y])
    n = len(z)
    # Pairwise distances and squared differences (both n x n)
    dist = squareform(pdist(coords))
    diff_sq = ((z[:, None] - z[None, :]) ** 2) / 2  # 0.5 * (z_i - z_j)^2
    # Upper triangle only (avoid double counting)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    g = diff_sq[iu]
    if max_lag is None:
        max_lag = np.percentile(d, 95)
    lag_width = max_lag / n_lags
    lags = np.zeros(n_lags)
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags)
    for i in range(n_lags):
        lag_center = (i + 0.5) * lag_width
        lag_low = i * lag_width
        lag_high = (i + 1) * lag_width
        if lag_tolerance != 0.5:
            half_tol = lag_width * lag_tolerance
            lag_low = lag_center - half_tol
            lag_high = lag_center + half_tol
        mask = (d >= lag_low) & (d < lag_high)
        if np.any(mask):
            lags[i] = lag_center
            gamma[i] = np.mean(g[mask])
            counts[i] = np.sum(mask)
        else:
            lags[i] = lag_center
            gamma[i] = np.nan
            counts[i] = 0
    valid = counts > 0
    return lags[valid], gamma[valid], counts[valid]


def fit_variogram_wls(
    lags: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    model: VariogramModel,
) -> Tuple[float, float, float]:
    """
    Fit variogram model using weighted least squares (weight = n_pairs).

    Returns:
        range_, sill, nugget
    """
    from scipy.optimize import minimize

    def residuals(params: np.ndarray) -> float:
        r, s, n = params
        if r <= 0 or s <= 0 or n < 0:
            return 1e10
        pred = variogram_model(lags, model, r, s, n)
        w = np.sqrt(counts)
        return np.sum(w * (gamma - pred) ** 2)

    # Initial guess: range ~ max lag, sill ~ max gamma, nugget ~ 0
    range_init = np.max(lags) * 0.5
    sill_init = np.nanmax(gamma)
    nugget_init = 0.0
    result = minimize(
        residuals,
        [range_init, sill_init, nugget_init],
        bounds=[(1e-6, None), (1e-6, None), (0, None)],
        method="L-BFGS-B",
    )
    if not result.success:
        logger.warning("Variogram fit did not converge: %s", result.message)
    r, s, n = result.x
    return float(r), float(s), float(n)


def ordinary_kriging(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    model: VariogramModel,
    range_: float,
    sill: float,
    nugget: float,
    max_neighbors: int = 25,
    min_neighbors: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ordinary kriging on a 2D grid.

    Returns:
        z_pred: Predicted values on grid
        z_var: Kriging variance on grid
    """
    n_obs = len(z_obs)
    coords_obs = np.column_stack([x_obs, y_obs])
    n_grid = len(x_grid)
    z_pred = np.full(n_grid, np.nan)
    z_var = np.full(n_grid, np.nan)
    for i in range(n_grid):
        pt = np.array([[x_grid[i], y_grid[i]]])
        d_to_obs = cdist(pt, coords_obs)[0]
        n_use = min(max_neighbors, np.sum(d_to_obs < 1e10))
        if n_use < min_neighbors:
            n_use = min_neighbors
        idx = np.argsort(d_to_obs)[:n_use]
        x_n = x_obs[idx]
        y_n = y_obs[idx]
        z_n = z_obs[idx]
        # Gamma matrix between neighbors
        coords_n = np.column_stack([x_n, y_n])
        d_nn = squareform(pdist(coords_n))
        Gamma_nn = variogram_model(d_nn, model, range_, sill, nugget)
        np.fill_diagonal(Gamma_nn, 0)
        # Gamma from target to neighbors
        d_tn = cdist(pt, coords_n)[0]
        gamma_tn = variogram_model(d_tn, model, range_, sill, nugget)
        # Kriging system: [Gamma 1; 1' 0] [w; mu] = [gamma_tn; 1]
        m = len(z_n)
        A = np.ones((m + 1, m + 1))
        A[:m, :m] = Gamma_nn
        A[m, m] = 0
        b = np.ones(m + 1)
        b[:m] = gamma_tn
        try:
            w = np.linalg.solve(A, b)
            z_pred[i] = np.dot(w[:m], z_n)
            z_var[i] = np.dot(w[:m], gamma_tn)
        except np.linalg.LinAlgError:
            z_pred[i] = np.mean(z_n)
            z_var[i] = np.var(z_n)
    return z_pred, z_var
