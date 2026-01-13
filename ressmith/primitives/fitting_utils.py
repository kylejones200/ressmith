"""
Fitting utilities with ramp-aware initial guesses.
"""

import numpy as np


def initial_guess_exponential(t: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """
    Generate ramp-aware initial guess for exponential decline.

    Uses max rate in first period (ramp-aware) and estimates di from decline.

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    q : np.ndarray
        Rate array

    Returns
    -------
    dict
        Initial guess: {'qi': float, 'di': float}
    """
    # Filter valid data
    valid_mask = q > 0
    if not np.any(valid_mask):
        return {"qi": 1.0, "di": 0.01}

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    # Ramp-aware qi: max rate in first 30% of data or first 3 months
    first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
    qi = float(np.max(q_valid[:first_period]))

    # Estimate di from decline (exponential: q = qi * exp(-di * t))
    if len(q_valid) >= 2:
        q0 = q_valid[0]
        q_last = q_valid[-1]
        t_span = t_valid[-1] - t_valid[0]
        if t_span > 0 and q_last > 0 and q0 > 0:
            di = -np.log(q_last / q0) / t_span
            di = max(0.001, min(di, 10.0))  # Reasonable bounds
        else:
            di = 0.01
    else:
        di = 0.01

    return {"qi": qi, "di": di}


def initial_guess_hyperbolic(t: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """
    Generate ramp-aware initial guess for hyperbolic decline.

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    q : np.ndarray
        Rate array

    Returns
    -------
    dict
        Initial guess: {'qi': float, 'di': float, 'b': float}
    """
    # Filter valid data
    valid_mask = q > 0
    if not np.any(valid_mask):
        return {"qi": 1.0, "di": 0.01, "b": 0.5}

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    # Ramp-aware qi: max rate in first 30% of data
    first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
    qi = float(np.max(q_valid[:first_period]))

    # Estimate di from decline
    if len(q_valid) >= 2:
        q0 = q_valid[0]
        q_last = q_valid[-1]
        t_span = t_valid[-1] - t_valid[0]
        if t_span > 0 and q_last > 0 and q0 > 0:
            # For hyperbolic: q = qi / (1 + b*di*t)^(1/b)
            # Approximate with harmonic for initial guess
            di = (q0 / q_last - 1) / t_span
            di = max(0.001, min(di, 1.0))
        else:
            di = 0.1
    else:
        di = 0.1

    # Initial b guess: 0.5 (typical for shale)
    b = 0.5

    return {"qi": qi, "di": di, "b": b}


def initial_guess_harmonic(t: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """
    Generate ramp-aware initial guess for harmonic decline.

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    q : np.ndarray
        Rate array

    Returns
    -------
    dict
        Initial guess: {'qi': float, 'di': float}
    """
    # Filter valid data
    valid_mask = q > 0
    if not np.any(valid_mask):
        return {"qi": 1.0, "di": 0.01}

    t_valid = t[valid_mask]
    q_valid = q[valid_mask]

    # Ramp-aware qi
    first_period = min(len(q_valid), max(3, int(len(q_valid) * 0.3)))
    qi = float(np.max(q_valid[:first_period]))

    # Estimate di from harmonic decline: q = qi / (1 + di*t)
    if len(q_valid) >= 2:
        q0 = q_valid[0]
        q_last = q_valid[-1]
        t_span = t_valid[-1] - t_valid[0]
        if t_span > 0 and q_last > 0 and q0 > 0:
            di = (q0 / q_last - 1) / t_span
            di = max(0.001, min(di, 10.0))
        else:
            di = 0.01
    else:
        di = 0.01

    return {"qi": qi, "di": di}
