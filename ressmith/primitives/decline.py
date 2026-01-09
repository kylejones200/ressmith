"""
Decline curve primitives: ARPS models and fitting functions.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy import optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def arps_exponential(
    t: np.ndarray, qi: float, di: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute exponential decline rates.

    q(t) = qi * exp(-di * (t - t0))

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    di : float
        Decline rate (1/time)
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    return qi * np.exp(-di * (t - t0))


def arps_hyperbolic(
    t: np.ndarray, qi: float, di: float, b: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute hyperbolic decline rates.

    q(t) = qi / (1 + b * di * (t - t0))^(1/b)

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    di : float
        Initial decline rate (1/time)
    b : float
        Hyperbolic exponent (0 < b < 1)
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    return qi / (1.0 + b * di * (t - t0)) ** (1.0 / b)


def arps_harmonic(
    t: np.ndarray, qi: float, di: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute harmonic decline rates (b=1 case of hyperbolic).

    q(t) = qi / (1 + di * (t - t0))

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    di : float
        Initial decline rate (1/time)
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    return qi / (1.0 + di * (t - t0))


def cumulative_exponential(
    t: np.ndarray, qi: float, di: float, t0: float = 0.0
) -> np.ndarray:
    """Compute cumulative production for exponential decline."""
    dt = t - t0
    return (qi / di) * (1.0 - np.exp(-di * dt))


def cumulative_hyperbolic(
    t: np.ndarray, qi: float, di: float, b: float, t0: float = 0.0
) -> np.ndarray:
    """Compute cumulative production for hyperbolic decline."""
    dt = t - t0
    if b == 1.0:
        # Harmonic case
        return (qi / di) * np.log(1.0 + di * dt)
    else:
        # General hyperbolic
        return (qi / (di * (1 - b))) * (
            1.0 - (1.0 + b * di * dt) ** ((b - 1.0) / b)
        )


def cumulative_harmonic(
    t: np.ndarray, qi: float, di: float, t0: float = 0.0
) -> np.ndarray:
    """Compute cumulative production for harmonic decline."""
    dt = t - t0
    return (qi / di) * np.log(1.0 + di * dt)


def _objective_exponential(
    params: tuple[float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for exponential fit."""
    qi, di = params
    q_pred = arps_exponential(t, qi, di)
    return np.sum((q - q_pred) ** 2)


def _objective_hyperbolic(
    params: tuple[float, float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for hyperbolic fit."""
    qi, di, b = params
    q_pred = arps_hyperbolic(t, qi, di, b)
    return np.sum((q - q_pred) ** 2)


def _objective_harmonic(
    params: tuple[float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for harmonic fit."""
    qi, di = params
    q_pred = arps_harmonic(t, qi, di)
    return np.sum((q - q_pred) ** 2)


def fit_arps_exponential(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit exponential decline model.

    Parameters
    ----------
    t : np.ndarray
        Time values
    q : np.ndarray
        Rate values
    use_scipy : bool, optional
        Force use of scipy (default: auto-detect)

    Returns
    -------
    dict
        Fitted parameters: {'qi': float, 'di': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    if use_scipy and HAS_SCIPY:
        # Use scipy optimization
        qi_init = q[0] if len(q) > 0 else 1.0
        di_init = 0.1
        result = optimize.minimize(
            _objective_exponential,
            (qi_init, di_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=((0.01, None), (1e-6, 1.0)),
        )
        if result.success:
            return {"qi": result.x[0], "di": result.x[1]}
        # Fall through to grid search

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 50)
    di_range = np.linspace(1e-4, 0.5, 50)
    best_error = np.inf
    best_params = {"qi": q[0], "di": 0.1}

    for qi in qi_range:
        for di in di_range:
            error = _objective_exponential((qi, di), t, q)
            if error < best_error:
                best_error = error
                best_params = {"qi": qi, "di": di}

    return best_params


def fit_arps_hyperbolic(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit hyperbolic decline model.

    Parameters
    ----------
    t : np.ndarray
        Time values
    q : np.ndarray
        Rate values
    use_scipy : bool, optional
        Force use of scipy (default: auto-detect)

    Returns
    -------
    dict
        Fitted parameters: {'qi': float, 'di': float, 'b': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    if use_scipy and HAS_SCIPY:
        # Use scipy optimization
        qi_init = q[0] if len(q) > 0 else 1.0
        di_init = 0.1
        b_init = 0.5
        result = optimize.minimize(
            _objective_hyperbolic,
            (qi_init, di_init, b_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=((0.01, None), (1e-6, 1.0), (0.01, 0.99)),
        )
        if result.success:
            return {"qi": result.x[0], "di": result.x[1], "b": result.x[2]}
        # Fall through to grid search

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 30)
    di_range = np.linspace(1e-4, 0.5, 30)
    b_range = np.linspace(0.1, 0.9, 20)
    best_error = np.inf
    best_params = {"qi": q[0], "di": 0.1, "b": 0.5}

    for qi in qi_range:
        for di in di_range:
            for b in b_range:
                error = _objective_hyperbolic((qi, di, b), t, q)
                if error < best_error:
                    best_error = error
                    best_params = {"qi": qi, "di": di, "b": b}

    return best_params


def fit_arps_harmonic(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit harmonic decline model.

    Parameters
    ----------
    t : np.ndarray
        Time values
    q : np.ndarray
        Rate values
    use_scipy : bool, optional
        Force use of scipy (default: auto-detect)

    Returns
    -------
    dict
        Fitted parameters: {'qi': float, 'di': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    if use_scipy and HAS_SCIPY:
        # Use scipy optimization
        qi_init = q[0] if len(q) > 0 else 1.0
        di_init = 0.1
        result = optimize.minimize(
            _objective_harmonic,
            (qi_init, di_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=((0.01, None), (1e-6, 1.0)),
        )
        if result.success:
            return {"qi": result.x[0], "di": result.x[1]}
        # Fall through to grid search

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 50)
    di_range = np.linspace(1e-4, 0.5, 50)
    best_error = np.inf
    best_params = {"qi": q[0], "di": 0.1}

    for qi in qi_range:
        for di in di_range:
            error = _objective_harmonic((qi, di), t, q)
            if error < best_error:
                best_error = error
                best_params = {"qi": qi, "di": di}

    return best_params

