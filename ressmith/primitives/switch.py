"""
Hyperbolic-to-Exponential switch model primitives.
"""

import numpy as np

from ressmith.primitives.decline import arps_exponential, arps_hyperbolic


def hyperbolic_to_exponential_rate(
    t: np.ndarray,
    qi: float,
    di: float,
    b: float,
    t_switch: float,
    di_exp: float,
) -> np.ndarray:
    """
    Compute rate with hyperbolic-to-exponential switch.

    For t < t_switch: q(t) = qi / (1 + b * di * t)^(1/b)
    For t >= t_switch: q(t) = q_switch * exp(-di_exp * (t - t_switch))

    Parameters
    ----------
    t : np.ndarray
        Time array (days from start)
    qi : float
        Initial rate
    di : float
        Initial decline rate (hyperbolic phase)
    b : float
        Hyperbolic exponent
    t_switch : float
        Switch time (days from start)
    di_exp : float
        Exponential decline rate (after switch)

    Returns
    -------
    np.ndarray
        Rate values
    """
    q = np.zeros_like(t)

    # Hyperbolic phase
    mask_hyper = t < t_switch
    if np.any(mask_hyper):
        t_hyper = t[mask_hyper]
        if b == 0.0:
            q[mask_hyper] = arps_exponential(t_hyper, qi, di)
        else:
            q[mask_hyper] = arps_hyperbolic(t_hyper, qi, di, b)

    # Exponential phase
    mask_exp = t >= t_switch
    if np.any(mask_exp):
        # Compute rate at switch point
        if b == 0.0:
            q_switch = arps_exponential(np.array([t_switch]), qi, di)[0]
        else:
            q_switch = arps_hyperbolic(np.array([t_switch]), qi, di, b)[0]

        t_exp = t[mask_exp] - t_switch
        q[mask_exp] = arps_exponential(t_exp, q_switch, di_exp)

    return q

