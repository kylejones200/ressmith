"""
Yield models for secondary phase production.

Yield models compute secondary phase rates from primary phase rates.
Examples: GOR (Gas-Oil Ratio), CGR (Condensate-Gas Ratio), water cut.
"""

from typing import Optional

import numpy as np


def constant_yield_rate(
    primary_rate: np.ndarray, yield_ratio: float
) -> np.ndarray:
    """
    Compute secondary rate with constant yield ratio.

    secondary_rate = primary_rate * yield_ratio

    Parameters
    ----------
    primary_rate : np.ndarray
        Primary phase rate array
    yield_ratio : float
        Constant yield ratio

    Returns
    -------
    np.ndarray
        Secondary phase rate array
    """
    return primary_rate * yield_ratio


def declining_yield_rate(
    t: np.ndarray,
    primary_rate: np.ndarray,
    yield_initial: float,
    decline_rate: float,
) -> np.ndarray:
    """
    Compute secondary rate with declining yield ratio.

    yield_ratio(t) = yield_initial * exp(-decline_rate * t)
    secondary_rate = primary_rate * yield_ratio(t)

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    primary_rate : np.ndarray
        Primary phase rate array
    yield_initial : float
        Initial yield ratio
    decline_rate : float
        Decline rate for yield ratio

    Returns
    -------
    np.ndarray
        Secondary phase rate array
    """
    yield_ratio = yield_initial * np.exp(-decline_rate * t)
    return primary_rate * yield_ratio


def hyperbolic_yield_rate(
    t: np.ndarray,
    primary_rate: np.ndarray,
    yield_initial: float,
    decline_rate: float,
    b: float,
) -> np.ndarray:
    """
    Compute secondary rate with hyperbolic yield decline.

    yield_ratio(t) = yield_initial / (1 + b * decline_rate * t)^(1/b)
    secondary_rate = primary_rate * yield_ratio(t)

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    primary_rate : np.ndarray
        Primary phase rate array
    yield_initial : float
        Initial yield ratio
    decline_rate : float
        Decline rate for yield ratio
    b : float
        b-factor for hyperbolic decline (0 < b < 1)

    Returns
    -------
    np.ndarray
        Secondary phase rate array
    """
    if abs(b) < 1e-6:
        # Exponential case
        yield_ratio = yield_initial * np.exp(-decline_rate * t)
    else:
        # Hyperbolic case
        yield_ratio = yield_initial / np.power(1 + b * decline_rate * t, 1 / b)

    return primary_rate * yield_ratio


def fit_constant_yield(
    t: np.ndarray, primary_rate: np.ndarray, secondary_rate: np.ndarray
) -> dict[str, float]:
    """
    Fit constant yield model.

    Parameters
    ----------
    t : np.ndarray
        Time array
    primary_rate : np.ndarray
        Primary phase rates
    secondary_rate : np.ndarray
        Secondary phase rates

    Returns
    -------
    dict
        Fitted parameters: {'yield_ratio': float}
    """
    # Simple ratio: yield = secondary / primary
    valid_mask = primary_rate > 0
    if not np.any(valid_mask):
        return {"yield_ratio": 0.0}

    ratios = secondary_rate[valid_mask] / primary_rate[valid_mask]
    yield_ratio = float(np.median(ratios))  # Use median to be robust to outliers

    return {"yield_ratio": max(0.0, yield_ratio)}


def fit_declining_yield(
    t: np.ndarray, primary_rate: np.ndarray, secondary_rate: np.ndarray
) -> dict[str, float]:
    """
    Fit declining yield model.

    Parameters
    ----------
    t : np.ndarray
        Time array
    primary_rate : np.ndarray
        Primary phase rates
    secondary_rate : np.ndarray
        Secondary phase rates

    Returns
    -------
    dict
        Fitted parameters: {'yield_initial': float, 'decline_rate': float}
    """
    # Calculate yield ratios
    valid_mask = primary_rate > 0
    if not np.any(valid_mask):
        return {"yield_initial": 0.0, "decline_rate": 0.0}

    ratios = secondary_rate[valid_mask] / primary_rate[valid_mask]
    t_valid = t[valid_mask]

    # Fit exponential decline to yield ratios
    # log(ratio) = log(yield_initial) - decline_rate * t
    if len(ratios) < 2:
        return {"yield_initial": float(ratios[0]) if len(ratios) > 0 else 0.0, "decline_rate": 0.0}

    log_ratios = np.log(np.maximum(ratios, 1e-6))
    # Linear fit: log_ratio = log(yield_initial) - decline_rate * t
    coeffs = np.polyfit(t_valid, log_ratios, deg=1)
    yield_initial = float(np.exp(coeffs[1]))
    decline_rate = float(-coeffs[0])

    return {
        "yield_initial": max(0.0, yield_initial),
        "decline_rate": max(0.0, decline_rate),
    }

