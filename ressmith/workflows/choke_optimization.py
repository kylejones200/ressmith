"""Choke optimization workflows.

Provides workflows for optimizing choke size to achieve target production rates.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ressmith.primitives.vlp import calculate_choke_performance

logger = logging.getLogger(__name__)


def optimize_choke_size(
    upstream_pressure: float,
    downstream_pressure: float,
    target_rate: float,
    gas_liquid_ratio: float = 500.0,
    oil_gravity: float = 30.0,
    gas_gravity: float = 0.65,
    min_choke_size: float = 0.125,
    max_choke_size: float = 2.0,
) -> dict[str, Any]:
    """Optimize choke size to achieve target production rate.

    Parameters
    ----------
    upstream_pressure : float
        Upstream pressure (psi)
    downstream_pressure : float
        Downstream pressure (psi)
    target_rate : float
        Target production rate (STB/day)
    gas_liquid_ratio : float
        Gas-liquid ratio (SCF/STB)
    oil_gravity : float
        Oil API gravity (°API)
    gas_gravity : float
        Gas specific gravity (air=1.0)
    min_choke_size : float
        Minimum choke size (inches)
    max_choke_size : float
        Maximum choke size (inches)

    Returns
    -------
    dict
        Dictionary with optimization results:
        - optimal_choke_size: Optimal choke size (inches)
        - achieved_rate: Achieved rate at optimal choke (STB/day)
        - error: Difference between target and achieved rate

    Examples
    --------
    >>> result = optimize_choke_size(
    ...     upstream_pressure=2000,
    ...     downstream_pressure=500,
    ...     target_rate=1000
    ... )
    >>> print(f"Optimal choke size: {result['optimal_choke_size']:.3f} inches")
    """
    logger.info(
        f"Optimizing choke size: target_rate={target_rate:.0f} STB/day, "
        f"upstream={upstream_pressure:.0f} psi"
    )

    def objective(choke_size: float) -> float:
        """Objective function: minimize difference between target and achieved rate."""
        rate = calculate_choke_performance(
            upstream_pressure=upstream_pressure,
            downstream_pressure=downstream_pressure,
            choke_size=choke_size,
            gas_liquid_ratio=gas_liquid_ratio,
            oil_gravity=oil_gravity,
            gas_gravity=gas_gravity,
        )
        error = abs(rate - target_rate)
        return error

    # Optimize
    result = minimize_scalar(
        objective, bounds=(min_choke_size, max_choke_size), method="bounded"
    )

    optimal_choke_size = result.x
    achieved_rate = calculate_choke_performance(
        upstream_pressure=upstream_pressure,
        downstream_pressure=downstream_pressure,
        choke_size=optimal_choke_size,
        gas_liquid_ratio=gas_liquid_ratio,
        oil_gravity=oil_gravity,
        gas_gravity=gas_gravity,
    )

    return {
        "optimal_choke_size": float(optimal_choke_size),
        "achieved_rate": float(achieved_rate),
        "error": float(abs(achieved_rate - target_rate)),
        "target_rate": float(target_rate),
    }


def analyze_choke_performance(
    upstream_pressure: float,
    downstream_pressure: float,
    choke_sizes: np.ndarray | list[float],
    gas_liquid_ratio: float = 500.0,
    oil_gravity: float = 30.0,
    gas_gravity: float = 0.65,
) -> pd.DataFrame:
    """Analyze choke performance for different choke sizes.

    Parameters
    ----------
    upstream_pressure : float
        Upstream pressure (psi)
    downstream_pressure : float
        Downstream pressure (psi)
    choke_sizes : np.ndarray or list
        Array of choke sizes to analyze (inches)
    gas_liquid_ratio : float
        Gas-liquid ratio (SCF/STB)
    oil_gravity : float
        Oil API gravity (°API)
    gas_gravity : float
        Gas specific gravity (air=1.0)

    Returns
    -------
    pd.DataFrame
        DataFrame with choke sizes and corresponding flow rates

    Examples
    --------
    >>> import numpy as np
    >>> choke_sizes = np.array([0.25, 0.5, 0.75, 1.0, 1.5])
    >>> performance = analyze_choke_performance(2000, 500, choke_sizes)
    >>> print(performance)
    """
    logger.info(f"Analyzing choke performance for {len(choke_sizes)} sizes")

    if isinstance(choke_sizes, list):
        choke_sizes = np.array(choke_sizes)

    rates = []
    for choke_size in choke_sizes:
        rate = calculate_choke_performance(
            upstream_pressure=upstream_pressure,
            downstream_pressure=downstream_pressure,
            choke_size=choke_size,
            gas_liquid_ratio=gas_liquid_ratio,
            oil_gravity=oil_gravity,
            gas_gravity=gas_gravity,
        )
        rates.append(rate)

    return pd.DataFrame(
        {
            "choke_size": choke_sizes,
            "flow_rate": rates,
            "upstream_pressure": upstream_pressure,
            "downstream_pressure": downstream_pressure,
        }
    )
