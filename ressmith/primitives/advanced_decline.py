"""
Advanced decline curve models: Power Law, Duong, Stretched Exponential.
"""

from typing import Any, Optional

import numpy as np

try:
    from scipy import optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ressmith.primitives.constraints import get_default_bounds, get_scipy_bounds
from ressmith.primitives.fitting_utils import initial_guess_hyperbolic


def power_law_rate(
    t: np.ndarray, qi: float, di: float, n: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute power law decline rates.

    q(t) = qi * (1 + n * di * (t - t0))^(-1/n)

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    di : float
        Decline rate
    n : float
        Power law exponent
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    t_shifted = t - t0
    # Avoid division by zero
    n_safe = max(n, 1e-6)
    return qi * np.power(1 + n_safe * di * np.maximum(t_shifted, 0), -1 / n_safe)


def duong_rate(
    t: np.ndarray, qi: float, a: float, m: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute Duong decline rates.

    q(t) = qi * t^(-m) * exp(a * (t^(1-m) - 1) / (1-m))

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    a : float
        Duong parameter a
    m : float
        Duong parameter m
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    t_shifted = np.maximum(t - t0, 1e-6)  # Avoid log(0)
    m_safe = max(min(m, 0.999), 0.001)  # Keep m away from 1

    # Duong formula
    t_power = np.power(t_shifted, 1 - m_safe)
    exp_term = np.exp(a * (t_power - 1) / (1 - m_safe))
    power_term = np.power(t_shifted, -m_safe)

    return qi * power_term * exp_term


def stretched_exponential_rate(
    t: np.ndarray, qi: float, tau: float, beta: float, t0: float = 0.0
) -> np.ndarray:
    """
    Compute stretched exponential decline rates.

    q(t) = qi * exp(-((t - t0) / tau)^beta)

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    tau : float
        Characteristic time
    beta : float
        Stretch parameter (0 < beta <= 1)
    t0 : float
        Reference time (default: 0.0)

    Returns
    -------
    np.ndarray
        Rate values
    """
    t_shifted = np.maximum(t - t0, 0)
    return qi * np.exp(-np.power(t_shifted / tau, beta))


def _objective_power_law(
    params: tuple[float, float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for power law fitting."""
    qi, di, n = params
    q_pred = power_law_rate(t, qi, di, n)
    return np.sum((q - q_pred) ** 2)


def _objective_duong(
    params: tuple[float, float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for Duong fitting."""
    qi, a, m = params
    q_pred = duong_rate(t, qi, a, m)
    return np.sum((q - q_pred) ** 2)


def _objective_stretched_exponential(
    params: tuple[float, float, float], t: np.ndarray, q: np.ndarray
) -> float:
    """Objective function for stretched exponential fitting."""
    qi, tau, beta = params
    q_pred = stretched_exponential_rate(t, qi, tau, beta)
    return np.sum((q - q_pred) ** 2)


def fit_power_law(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit power law decline model.

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
        Fitted parameters: {'qi': float, 'di': float, 'n': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    # Initial guess (use hyperbolic as starting point)
    hyper_guess = initial_guess_hyperbolic(t, q)
    qi_init = hyper_guess["qi"]
    di_init = hyper_guess["di"]
    n_init = 0.5  # Typical power law exponent

    if use_scipy and HAS_SCIPY:
        bounds = get_default_bounds("power_law")
        scipy_bounds = get_scipy_bounds(bounds, ["qi", "di", "n"])

        result = optimize.minimize(
            _objective_power_law,
            (qi_init, di_init, n_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=scipy_bounds,
        )
        if result.success:
            return {"qi": result.x[0], "di": result.x[1], "n": result.x[2]}

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 20)
    di_range = np.linspace(1e-4, 1.0, 20)
    n_range = np.linspace(0.1, 1.5, 15)
    best_error = np.inf
    best_params = {"qi": qi_init, "di": di_init, "n": n_init}

    for qi in qi_range:
        for di in di_range:
            for n in n_range:
                error = _objective_power_law((qi, di, n), t, q)
                if error < best_error:
                    best_error = error
                    best_params = {"qi": qi, "di": di, "n": n}

    return best_params


def fit_duong(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit Duong decline model.

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
        Fitted parameters: {'qi': float, 'a': float, 'm': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    # Initial guess
    hyper_guess = initial_guess_hyperbolic(t, q)
    qi_init = hyper_guess["qi"]
    a_init = 0.1  # Typical Duong a parameter
    m_init = 0.5  # Typical Duong m parameter

    if use_scipy and HAS_SCIPY:
        bounds = get_default_bounds("duong")
        scipy_bounds = get_scipy_bounds(bounds, ["qi", "a", "m"])

        result = optimize.minimize(
            _objective_duong,
            (qi_init, a_init, m_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=scipy_bounds,
        )
        if result.success:
            return {"qi": result.x[0], "a": result.x[1], "m": result.x[2]}

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 20)
    a_range = np.linspace(1e-4, 2.0, 20)
    m_range = np.linspace(0.1, 1.5, 15)
    best_error = np.inf
    best_params = {"qi": qi_init, "a": a_init, "m": m_init}

    for qi in qi_range:
        for a in a_range:
            for m in m_range:
                error = _objective_duong((qi, a, m), t, q)
                if error < best_error:
                    best_error = error
                    best_params = {"qi": qi, "a": a, "m": m}

    return best_params


def fit_stretched_exponential(
    t: np.ndarray, q: np.ndarray, use_scipy: Optional[bool] = None
) -> dict[str, float]:
    """
    Fit stretched exponential decline model.

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
        Fitted parameters: {'qi': float, 'tau': float, 'beta': float}
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    # Initial guess
    hyper_guess = initial_guess_hyperbolic(t, q)
    qi_init = hyper_guess["qi"]
    tau_init = t[-1] * 0.5  # Characteristic time around mid-point
    beta_init = 0.5  # Typical stretch parameter

    if use_scipy and HAS_SCIPY:
        bounds = get_default_bounds("stretched_exponential")
        scipy_bounds = get_scipy_bounds(bounds, ["qi", "tau", "beta"])

        result = optimize.minimize(
            _objective_stretched_exponential,
            (qi_init, tau_init, beta_init),
            args=(t, q),
            method="L-BFGS-B",
            bounds=scipy_bounds,
        )
        if result.success:
            return {"qi": result.x[0], "tau": result.x[1], "beta": result.x[2]}

    # Grid search fallback
    qi_range = np.linspace(q.max() * 0.5, q.max() * 2.0, 20)
    tau_range = np.linspace(t[1] if len(t) > 1 else 1.0, t[-1] * 2, 20)
    beta_range = np.linspace(0.1, 1.0, 15)
    best_error = np.inf
    best_params = {"qi": qi_init, "tau": tau_init, "beta": beta_init}

    for qi in qi_range:
        for tau in tau_range:
            for beta in beta_range:
                error = _objective_stretched_exponential((qi, tau, beta), t, q)
                if error < best_error:
                    best_error = error
                    best_params = {"qi": qi, "tau": tau, "beta": beta}

    return best_params

