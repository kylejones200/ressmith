"""
EUR (Estimated Ultimate Recovery) estimation primitives.

Layer 2 primitives for calculating EUR from decline curves.
"""

import numpy as np


def calculate_eur_exponential(
    qi: float,
    di: float,
    t_max: float = 360.0,
    econ_limit: float = 10.0,
) -> float:
    """
    Calculate EUR for exponential decline.

    EUR = qi / di * (1 - exp(-di * t_max))
    Or: EUR = qi / di * (1 - q_final / qi) where q_final = econ_limit

    Parameters
    ----------
    qi : float
        Initial rate
    di : float
        Decline rate (1/time)
    t_max : float
        Maximum time for EUR calculation (default: 360 months = 30 years)
    econ_limit : float
        Economic limit (minimum production rate) (default: 10)

    Returns
    -------
    float
        Estimated Ultimate Recovery
    """
    if di <= 0:
        raise ValueError("Decline rate must be positive")
    if econ_limit >= qi:
        # If econ limit is higher than initial rate, EUR is 0
        return 0.0

    # Time to reach economic limit
    t_econ = -np.log(econ_limit / qi) / di if econ_limit > 0 else t_max
    t_effective = min(t_econ, t_max)

    # EUR = integral of q(t) from 0 to t_effective
    # For exponential: EUR = qi / di * (1 - exp(-di * t))
    eur = (qi / di) * (1 - np.exp(-di * t_effective))

    return float(eur)


def calculate_eur_hyperbolic(
    qi: float,
    di: float,
    b: float,
    t_max: float = 360.0,
    econ_limit: float = 10.0,
) -> float:
    """
    Calculate EUR for hyperbolic decline.

    EUR = qi / (di * (1 - b)) * (1 - (1 + b*di*t_max)^(-(1-b)/b))
    Or integrated to economic limit

    Parameters
    ----------
    qi : float
        Initial rate
    di : float
        Decline rate
    b : float
        b-factor (0 < b < 1)
    t_max : float
        Maximum time for EUR calculation (default: 360 months)
    econ_limit : float
        Economic limit (default: 10)

    Returns
    -------
    float
        Estimated Ultimate Recovery
    """
    if di <= 0:
        raise ValueError("Decline rate must be positive")
    if b <= 0 or b >= 1:
        raise ValueError("b-factor must be between 0 and 1")
    if econ_limit >= qi:
        return 0.0

    # Time to reach economic limit
    # q(t) = qi / (1 + b*di*t)^(1/b)
    # Solve for t when q(t) = econ_limit
    if econ_limit > 0:
        t_econ = ((qi / econ_limit) ** b - 1) / (b * di)
        t_econ = max(0, t_econ)
    else:
        t_econ = t_max

    t_effective = min(t_econ, t_max)

    # EUR = integral of q(t) from 0 to t_effective
    # For hyperbolic: EUR = qi / (di * (1 - b)) * (1 - (1 + b*di*t)^(-(1-b)/b))
    if abs(b - 1.0) < 1e-6:
        # Harmonic case (b=1): EUR = qi / di * ln(1 + di*t)
        eur = (qi / di) * np.log(1 + di * t_effective)
    else:
        # General hyperbolic: EUR = qi / (di * (1 - b)) * (1 - (1 + b*di*t)^(-(1-b)/b))
        term = 1 + b * di * t_effective
        exponent = -(1 - b) / b
        eur = (qi / (di * (1 - b))) * (1 - np.power(term, exponent))

    return float(max(0, eur))


def calculate_eur_harmonic(
    qi: float,
    di: float,
    t_max: float = 360.0,
    econ_limit: float = 10.0,
) -> float:
    """
    Calculate EUR for harmonic decline.

    EUR = qi / di * ln(1 + di * t_max)
    Or integrated to economic limit

    Parameters
    ----------
    qi : float
        Initial rate
    di : float
        Decline rate
    t_max : float
        Maximum time for EUR calculation (default: 360 months)
    econ_limit : float
        Economic limit (default: 10)

    Returns
    -------
    float
        Estimated Ultimate Recovery
    """
    if di <= 0:
        raise ValueError("Decline rate must be positive")
    if econ_limit >= qi:
        return 0.0

    # Time to reach economic limit
    # q(t) = qi / (1 + di*t)
    # Solve for t when q(t) = econ_limit
    if econ_limit > 0:
        t_econ = (qi / econ_limit - 1) / di
        t_econ = max(0, t_econ)
    else:
        t_econ = t_max

    t_effective = min(t_econ, t_max)

    # EUR = integral of q(t) from 0 to t_effective
    # For harmonic: EUR = qi / di * ln(1 + di*t)
    eur = (qi / di) * np.log(1 + di * t_effective)

    return float(max(0, eur))


def calculate_eur_from_params(
    params: dict[str, float],
    model_name: str,
    t_max: float = 360.0,
    econ_limit: float = 10.0,
) -> float | None:
    """
    Calculate EUR from fitted parameters.

    Parameters
    ----------
    params : dict
        Fitted parameters (must include 'qi', 'di', and 'b' if applicable)
    model_name : str
        Model name: 'exponential', 'hyperbolic', 'harmonic'
    t_max : float
        Maximum time for EUR calculation (default: 360 months)
    econ_limit : float
        Economic limit (default: 10)

    Returns
    -------
    float or None
        Estimated Ultimate Recovery, or None if calculation fails
    """
    try:
        qi = params.get("qi", 0.0)
        di = params.get("di", 0.0)

        if qi <= 0 or di <= 0:
            return None

        if model_name == "exponential":
            return calculate_eur_exponential(qi, di, t_max, econ_limit)
        elif model_name == "hyperbolic":
            b = params.get("b", 0.5)
            if b <= 0 or b >= 1:
                # Invalid b-factor, treat as exponential
                return calculate_eur_exponential(qi, di, t_max, econ_limit)
            return calculate_eur_hyperbolic(qi, di, b, t_max, econ_limit)
        elif model_name == "harmonic":
            return calculate_eur_harmonic(qi, di, t_max, econ_limit)
        else:
            # Unknown model, use exponential as default
            return calculate_eur_exponential(qi, di, t_max, econ_limit)

    except Exception:
        return None
