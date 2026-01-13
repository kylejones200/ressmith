"""Sensitivity analysis for reservoir engineering parameters."""

import logging

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ressmith.primitives.decline import arps_hyperbolic
from ressmith.primitives.economics import npv

logger = logging.getLogger(__name__)


def _compute_sensitivity_case(
    qi: float,
    di: float,
    b: float,
    price: float,
    opex: float,
    discount_rate: float,
    t_max: float,
    econ_limit: float,
    dt: float,
) -> dict[str, float] | None:
    """Compute a single sensitivity case (used for parallel execution)."""
    t = np.arange(0, t_max + dt, dt)
    q = arps_hyperbolic(t, qi, di, b)
    mask = q > econ_limit
    if not np.any(mask):
        return None

    t_valid = t[mask]
    q_valid = q[mask]
    eur = np.trapz(q_valid, t_valid)

    # Calculate economics
    revenue = q_valid * price
    costs = np.full_like(q_valid, opex)
    cashflows = revenue - costs
    npv_value = npv(cashflows, discount_rate / 12.0)  # Convert annual to monthly

    # Calculate payback (first period where cumulative cashflow >= 0)
    cumulative_cf = np.cumsum(cashflows)
    payout_idx = np.where(cumulative_cf >= 0)[0]
    payback_month = float(payout_idx[0]) if len(payout_idx) > 0 else None

    return {
        "qi": qi,
        "di": di,
        "b": b,
        "price": price,
        "EUR": eur,
        "NPV": npv_value,
        "Payback_month": payback_month,
    }


def run_sensitivity(
    param_grid: list[tuple[float, float, float]],
    prices: list[float],
    opex: float,
    discount_rate: float = 0.10,
    t_max: float = 240,
    econ_limit: float = 10.0,
    dt: float = 1.0,
    n_jobs: int = -1,  # -1 uses all available cores
) -> pd.DataFrame:
    """
    Run sensitivity analysis across Arps parameters and prices.

    Args:
        param_grid: List of (qi, di, b) tuples.
        prices: List of oil/gas prices to test.
        opex: Operating cost per unit.
        discount_rate: Annual discount rate.
        t_max: Time horizon in months.
        econ_limit: Minimum economic production rate.
        dt: Time step in months.
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential).

    Returns:
        DataFrame with qi, di, b, price, EUR, NPV, payback.
    """
    # Create list of all parameter combinations
    cases = [
        (qi, di, b, price, opex, discount_rate, t_max, econ_limit, dt)
        for price in prices
        for qi, di, b in param_grid
    ]

    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel execution
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_sensitivity_case)(*case) for case in cases
        )
    else:
        # Sequential execution (fallback)
        results = [_compute_sensitivity_case(*case) for case in cases]

    # Filter out None results
    results = [r for r in results if r is not None]

    return pd.DataFrame(results)
