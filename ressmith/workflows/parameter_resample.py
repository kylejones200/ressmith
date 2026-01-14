"""Fast parameter resampling for Arps with approximate posteriors.

This module implements fast parameter resampling using simple approximate
posteriors (normal or lognormal) around the point estimate with scale tied
to residual error. This provides a fast alternative to full Bayesian inference.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.decline import (
    arps_exponential,
    arps_harmonic,
    arps_hyperbolic,
    fit_arps_exponential,
    fit_arps_harmonic,
    fit_arps_hyperbolic,
)

logger = logging.getLogger(__name__)

try:
    from ressmith.workflows.uncertainty import ForecastDraws, ParameterDistribution
except ImportError:
    # Fallback definitions
    @dataclass
    class ForecastDraws:
        p10: pd.Series
        p50: pd.Series
        p90: pd.Series
        samples: Any | None = None

    @dataclass
    class ParameterDistribution:
        qi: np.ndarray
        di: np.ndarray
        b: np.ndarray | None = None


def fast_arps_resample(
    series: pd.Series,
    kind: str = "hyperbolic",
    n_draws: int = 1000,
    seed: int | None = None,
    method: str = "residual_based",
    horizon: int = 12,
) -> ForecastDraws:
    """
    Fast parameter resampling for Arps models with approximate posteriors.

    Uses simple approximate posteriors (normal or lognormal) around the point
    estimate with scale tied to residual error. This is much faster than
    full Bayesian inference but provides reasonable uncertainty estimates.

    Args:
        series: Historical production time series
        kind: Arps decline type ('exponential', 'harmonic', 'hyperbolic')
        n_draws: Number of parameter samples
        seed: Random seed for reproducibility
        method: Resampling method:
            - 'residual_based': Scale uncertainty from residual error
            - 'fixed_scale': Use fixed scale factors (faster, less accurate)
        horizon: Forecast horizon

    Returns:
        ForecastDraws with uncertainty quantification

    Example:
        >>> import pandas as pd
        >>> from decline_curve.parameter_resample import fast_arps_resample
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> draws = fast_arps_resample(production, kind='hyperbolic', n_draws=1000)
        >>> print(f"P50 forecast: {draws.p50.iloc[-1]:.2f}")
        >>> print(f"P10-P90 range: {draws.p10.iloc[-1]:.2f} - {draws.p90.iloc[-1]:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(len(series))
    q = series.values

    if kind == "exponential":
        params = fit_arps_exponential(t, q)
        qi, di = params["qi"], params["di"]
        b = 0.0
        q_pred = arps_exponential(t, qi, di)
    elif kind == "harmonic":
        params = fit_arps_harmonic(t, q)
        qi, di = params["qi"], params["di"]
        b = 1.0
        q_pred = arps_harmonic(t, qi, di)
    else:  # hyperbolic
        params = fit_arps_hyperbolic(t, q)
        qi, di, b = params["qi"], params["di"], params["b"]
        q_pred = arps_hyperbolic(t, qi, di, b)

    # Calculate residual error
    residuals = q - q_pred
    rmse = np.sqrt(np.mean(residuals**2))
    _ = np.mean(np.abs(residuals))  # noqa: F841

    if method == "residual_based":
        # Scale uncertainty based on residual error
        # Higher residual error -> higher parameter uncertainty
        relative_error = rmse / np.mean(q) if np.mean(q) > 0 else 0.1

        # Parameter uncertainty scales with relative error
        qi_std = qi * relative_error * 0.5  # Conservative scaling
        di_std = di * relative_error * 0.5
        b_std = b * relative_error * 0.3 if kind == "hyperbolic" else 0.0

    elif method == "fixed_scale":
        qi_std = qi * 0.15  # 15% uncertainty
        di_std = di * 0.20  # 20% uncertainty
        b_std = b * 0.10 if kind == "hyperbolic" else 0.0  # 10% uncertainty
    else:
        raise ValueError(f"Unknown method: {method}")

    qi_samples = np.random.lognormal(
        np.log(max(qi, 1e-6)), qi_std / qi, n_draws
    )
    di_samples = np.random.lognormal(
        np.log(max(di, 1e-6)), di_std / di, n_draws
    )

    if kind == "hyperbolic":
        # b-factor: use truncated normal (0 to 2)
        b_samples = np.random.normal(b, b_std, n_draws)
        b_samples = np.clip(b_samples, 0.0, 2.0)
    elif kind == "exponential":
        b_samples = np.zeros(n_draws)
    else:  # harmonic
        b_samples = np.ones(n_draws)

    n_periods = len(series) + horizon
    draws = np.zeros((n_draws, n_periods))

    t_full = np.arange(n_periods)

    for i in range(n_draws):
        try:
            if kind == "exponential":
                forecast = arps_exponential(t_full, qi_samples[i], di_samples[i])
            elif kind == "harmonic":
                forecast = arps_harmonic(t_full, qi_samples[i], di_samples[i])
            else:  # hyperbolic
                forecast = arps_hyperbolic(t_full, qi_samples[i], di_samples[i], b_samples[i])
            draws[i] = forecast
        except Exception as e:
            logger.warning(f"Failed to generate forecast for sample {i}: {e}")
            draws[i] = q_pred

    dates = pd.date_range(
        series.index[0], periods=n_periods, freq=series.index.freq or "MS"
    )

    # Calculate percentiles
    p10 = pd.Series(np.percentile(draws, 10, axis=0), index=dates)
    p50 = pd.Series(np.percentile(draws, 50, axis=0), index=dates)
    p90 = pd.Series(np.percentile(draws, 90, axis=0), index=dates)

    return ForecastDraws(
        p10=p10,
        p50=p50,
        p90=p90,
        samples=draws,
    )


def approximate_posterior(
    qi: float,
    di: float,
    b: float | None,
    residuals: np.ndarray,
    kind: str = "hyperbolic",
) -> ParameterDistribution:
    """
    Create approximate posterior distribution from point estimate and residuals.

    This is a fast approximation that assumes:
    - qi and di follow lognormal distributions (must be positive)
    - b follows truncated normal distribution (0 to 2)
    - Scale is tied to residual error

    Args:
        qi: Initial rate parameter
        di: Decline rate parameter
        b: Decline exponent (None for exponential/harmonic)
        residuals: Residual errors from fit
        kind: Decline type

    Returns:
        ParameterDistribution object
    """
    # Calculate residual error
    rmse = np.sqrt(np.mean(residuals**2))
    relative_error = rmse / np.mean(np.abs(residuals) + 1e-6)

    qi_std = qi * relative_error * 0.5
    di_std = di * relative_error * 0.5
    b_std = b * relative_error * 0.3 if kind == "hyperbolic" else 0.0

    qi_dist = {
        "type": "lognormal",
        "mean": qi,
        "std": qi_std / qi,  # Coefficient of variation
    }

    di_dist = {
        "type": "lognormal",
        "mean": di,
        "std": di_std / di,
    }

    if kind == "hyperbolic":
        b_dist = {
            "type": "normal",
            "mean": b or 0.0,
            "std": b_std,
            "min": 0.0,
            "max": 2.0,
        }
    elif kind == "exponential":
        b_dist = {"type": "normal", "mean": 0.0, "std": 0.0}
    else:  # harmonic
        b_dist = {"type": "normal", "mean": 1.0, "std": 0.0}

    return ParameterDistribution(
        qi=np.array([qi_dist["mean"]]),
        di=np.array([di_dist["mean"]]),
        b=np.array([b_dist["mean"]]) if kind == "hyperbolic" else None,
    )
