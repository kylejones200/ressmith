"""
Decline curve variants: practical modifications to standard models.

This module provides variants that prevent unrealistic long-term forecasts
by transitioning to fixed terminal decline rates.
"""

from typing import Literal

import numpy as np

from ressmith.primitives.decline import (
    arps_exponential,
    arps_hyperbolic,
    fit_arps_exponential,
    fit_arps_harmonic,
    fit_arps_hyperbolic,
)


def fixed_terminal_decline_rate(
    t: np.ndarray,
    qi: float,
    di: float,
    b: float,
    t_switch: float,
    di_terminal: float,
) -> np.ndarray:
    """
    Compute rates with fixed terminal decline.

    Hyperbolic decline until t_switch, then exponential decline at di_terminal.

    Parameters
    ----------
    t : np.ndarray
        Time values
    qi : float
        Initial rate
    di : float
        Initial decline rate
    b : float
        b-factor for hyperbolic phase
    t_switch : float
        Time to switch to terminal decline
    di_terminal : float
        Terminal decline rate (monthly)

    Returns
    -------
    np.ndarray
        Rate values
    """
    rates = np.zeros_like(t)

    if t_switch > 0:
        t_transition = np.array([t_switch])
        if b > 0:
            q_switch = arps_hyperbolic(t_transition, qi, di, b)[0]
        else:
            q_switch = arps_exponential(t_transition, qi, di)[0]
    else:
        q_switch = qi
        t_switch = 0.0

    for i, t_val in enumerate(t):
        if t_val <= t_switch:
            # Initial hyperbolic phase
            if b > 0:
                rates[i] = arps_hyperbolic(np.array([t_val]), qi, di, b)[0]
            else:
                rates[i] = arps_exponential(np.array([t_val]), qi, di)[0]
        else:
            # Terminal exponential phase
            t_terminal = t_val - t_switch
            rates[i] = q_switch * np.exp(-di_terminal * t_terminal)

    return rates


def fit_fixed_terminal_decline(
    t: np.ndarray,
    q: np.ndarray,
    kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
    terminal_decline_rate: float = 0.05,
    transition_criteria: Literal["rate", "time"] = "rate",
    transition_value: float | None = None,
) -> dict[str, float]:
    """
    Fit decline curve with transition to fixed terminal decline rate.

    Parameters
    ----------
    t : np.ndarray
        Time values
    q : np.ndarray
        Rate values
    kind : str
        Initial decline type ('exponential', 'harmonic', 'hyperbolic')
    terminal_decline_rate : float
        Fixed annual decline rate for terminal phase (default: 0.05 = 5% per year)
    transition_criteria : str
        When to transition: 'rate' or 'time'
    transition_value : float, optional
        Threshold value for transition

    Returns
    -------
    dict
        Fitted parameters including initial_params and terminal parameters
    """
    # Fit initial decline
    if kind == "exponential":
        initial_params = fit_arps_exponential(t, q)
        b = 0.0
    elif kind == "harmonic":
        initial_params = fit_arps_harmonic(t, q)
        b = 1.0
    else:  # hyperbolic
        initial_params = fit_arps_hyperbolic(t, q)
        b = initial_params.get("b", 0.5)

    qi = initial_params["qi"]
    di = initial_params["di"]

    terminal_decline_monthly = terminal_decline_rate / 12.0

    if transition_criteria == "time":
        if transition_value is None:
            transition_time = 60.0  # 5 years default
        else:
            transition_time = transition_value
    else:  # rate-based transition
        if transition_value is None:
            transition_value = terminal_decline_monthly

        # Find when decline rate reaches threshold
        # For hyperbolic: D(t) = di / (1 + b*di*t)
        # Solve for t when D(t) = transition_value
        if b > 0:
            transition_time = (di / transition_value - 1) / (b * di)
            transition_time = max(0, min(transition_time, t[-1] * 2))
        else:
            # Exponential: constant decline, transition immediately or at fixed time
            transition_time = 0.0

    t_transition = np.array([transition_time])
    if b > 0:
        q_transition = arps_hyperbolic(t_transition, qi, di, b)[0]
    else:
        q_transition = arps_exponential(t_transition, qi, di)[0]

    return {
        "qi": qi,
        "di": di,
        "b": b,
        "t_switch": transition_time,
        "q_switch": q_transition,
        "di_terminal": terminal_decline_monthly,
        "terminal_decline_rate_annual": terminal_decline_rate,
        "kind": kind,
    }
