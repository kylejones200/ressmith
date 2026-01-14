"""Type curve matching for rate transient analysis.

This module provides type curve generation and matching capabilities for RTA.
Type curves are standardized decline curves used to analyze production data.

References:
- Fetkovich, M.J., "Decline Curve Analysis Using Type Curves," JPT, June 1980.
- Wattenbarger, R.A., et al., "Gas Reservoir Engineering," SPE Textbook Series, 1998.
- Palacio, J.C. and Blasingame, T.A., "Decline-Curve Analysis Using Type Curves,"
  SPE 25909, 1993.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ressmith.primitives.decline import arps_hyperbolic

logger = logging.getLogger(__name__)


@dataclass
class TypeCurveMatch:
    """Container for type curve matching results.

    Attributes:
        matched_params: Matched parameters (qi, di, b)
        match_error: Matching error (RMSE)
        correlation: Correlation coefficient
        matched_curve: Matched type curve data
    """

    matched_params: dict[str, float]
    match_error: float
    correlation: float
    matched_curve: np.ndarray


def generate_arps_type_curve(
    qi_normalized: float = 1.0,
    di_normalized: float = 1.0,
    b: float = 0.5,
    time_normalized: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized ARPS type curve.

    Generates normalized type curve for hyperbolic decline.

    Args:
        qi_normalized: Normalized initial rate (default: 1.0)
        di_normalized: Normalized decline rate (default: 1.0)
        b: b-factor (default: 0.5)
        time_normalized: Normalized time array (if None, generates default)

    Returns:
        Tuple of (normalized_time, normalized_rate)

    Reference:
        Fetkovich, M.J., "Decline Curve Analysis Using Type Curves," JPT, June 1980.

    Example:
        >>> t_norm, q_norm = generate_arps_type_curve(b=0.5)
        >>> print(f"Type curve length: {len(t_norm)}")
    """
    if time_normalized is None:
        # Generate normalized time from 0.01 to 100
        time_normalized = np.logspace(-2, 2, 100)

    # Generate normalized rate using ARPS hyperbolic
    rate_normalized = arps_hyperbolic(
        time_normalized, qi_normalized, di_normalized, b
    )

    return time_normalized, rate_normalized


def normalize_production_data(
    time: np.ndarray,
    rate: np.ndarray,
    qi_ref: float | None = None,
    di_ref: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Normalize production data for type curve matching.

    Normalizes time and rate data using reference values.

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        qi_ref: Reference initial rate (if None, uses max rate)
        di_ref: Reference decline rate (if None, estimates from data)

    Returns:
        Tuple of (normalized_time, normalized_rate, normalization_factors)

    Example:
        >>> time = np.array([1, 10, 30, 60, 90])
        >>> rate = np.array([1000, 800, 600, 500, 450])
        >>> t_norm, q_norm, factors = normalize_production_data(time, rate)
    """
    valid_mask = (rate > 0) & (time > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError(format_error(ERR_INSUFFICIENT_DATA, min_points=3, analysis="normalization"))

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Reference values
    if qi_ref is None:
        qi_ref = np.max(rate_valid)

    if di_ref is None:
        if len(rate_valid) >= 3:
            early_rate = rate_valid[:3]
            early_time = time_valid[:3]
            di_ref = -np.mean(np.diff(np.log(early_rate)) / np.diff(early_time))
        else:
            di_ref = 0.001

    # Normalize
    time_normalized = time_valid * di_ref
    rate_normalized = rate_valid / qi_ref

    normalization_factors = {
        "qi_ref": float(qi_ref),
        "di_ref": float(di_ref),
    }

    return time_normalized, rate_normalized, normalization_factors


def match_type_curve(
    time: np.ndarray,
    rate: np.ndarray,
    b_values: np.ndarray | None = None,
    initial_guess: dict[str, float] | None = None,
) -> TypeCurveMatch:
    """Match production data to ARPS type curve.

    Matches production data to normalized ARPS type curves by optimizing
    qi, di, and b parameters.

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        b_values: Array of b-values to test (if None, uses default range)
        initial_guess: Initial parameter guess (if None, estimates from data)

    Returns:
        TypeCurveMatch with matched parameters and error

    Example:
        >>> time = np.array([1, 10, 30, 60, 90, 120])
        >>> rate = np.array([1000, 800, 600, 500, 450, 400])
        >>> match = match_type_curve(time, rate)
        >>> print(f"Matched b: {match.matched_params['b']:.2f}")
    """
    valid_mask = (rate > 0) & (time > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError(format_error(ERR_INSUFFICIENT_DATA, min_points=3, analysis="type curve matching"))

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    if b_values is None:
        b_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Initial guess
    if initial_guess is None:
        qi_guess = np.max(rate_valid)
        if len(rate_valid) >= 3:
            early_rate = rate_valid[:3]
            early_time = time_valid[:3]
            di_guess = -np.mean(np.diff(np.log(early_rate)) / np.diff(early_time))
        else:
            di_guess = 0.001
    else:
        qi_guess = initial_guess.get("qi", np.max(rate_valid))
        di_guess = initial_guess.get("di", 0.001)

    best_match = None
    best_error = float("inf")

    for b in b_values:
        try:
            # Objective function for matching
            def objective(params: np.ndarray) -> float:
                qi = params[0]
                di = params[1]
                # Generate type curve
                q_pred = arps_hyperbolic(time_valid, qi, di, b)
                # Calculate error
                error = np.sum((rate_valid - q_pred) ** 2)
                return error

            # Optimize qi and di for this b-value
            result = minimize(
                objective,
                x0=[qi_guess, di_guess],
                bounds=[(0.1 * qi_guess, 10 * qi_guess), (1e-5, 1.0)],
                method="L-BFGS-B",
            )

            if result.success and result.fun < best_error:
                best_error = result.fun
                qi_matched = result.x[0]
                di_matched = result.x[1]

                # Generate matched curve
                q_matched = arps_hyperbolic(time_valid, qi_matched, di_matched, b)

                # Calculate correlation
                correlation = np.corrcoef(rate_valid, q_matched)[0, 1]

                best_match = TypeCurveMatch(
                    matched_params={
                        "qi": float(qi_matched),
                        "di": float(di_matched),
                        "b": float(b),
                    },
                    match_error=float(np.sqrt(best_error / len(rate_valid))),
                    correlation=float(correlation),
                    matched_curve=q_matched,
                )

        except Exception as e:
            logger.warning(f"Type curve matching failed for b={b}: {e}")
            continue

    if best_match is None:
        raise ValueError("Type curve matching failed for all b-values")

    return best_match


def match_multiple_type_curves(
    time: np.ndarray,
    rate: np.ndarray,
    type_curve_types: list[str] | None = None,
) -> dict[str, TypeCurveMatch]:
    """Match production data to multiple type curve types.

    Matches data to different type curve families (ARPS, Power Law, etc.)

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        type_curve_types: List of type curve types to try (if None, uses ARPS)

    Returns:
        Dictionary mapping type curve type to match results

    Example:
        >>> matches = match_multiple_type_curves(time, rate)
        >>> print(f"ARPS match error: {matches['arps'].match_error:.2f}")
    """
    if type_curve_types is None:
        type_curve_types = ["arps"]

    matches: dict[str, TypeCurveMatch] = {}

    for curve_type in type_curve_types:
        try:
            if curve_type == "arps":
                match = match_type_curve(time, rate)
                matches["arps"] = match
            else:
                logger.warning(f"Unknown type curve type: {curve_type}")
        except Exception as e:
            logger.warning(f"Type curve matching failed for {curve_type}: {e}")

    return matches


def calculate_type_curve_statistics(
    match: TypeCurveMatch,
    time: np.ndarray,
    rate: np.ndarray,
) -> dict[str, float]:
    """Calculate statistics for type curve match.

    Calculates R², RMSE, and other statistics.

    Args:
        match: TypeCurveMatch result
        time: Original time data
        rate: Original rate data

    Returns:
        Dictionary with match statistics
    """
    valid_mask = (rate > 0) & (time > 0)
    rate_valid = rate[valid_mask]

    # Calculate R²
    ss_res = np.sum((rate_valid - match.matched_curve) ** 2)
    ss_tot = np.sum((rate_valid - np.mean(rate_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Calculate RMSE
    rmse = np.sqrt(np.mean((rate_valid - match.matched_curve) ** 2))

    # Calculate MAE
    mae = np.mean(np.abs(rate_valid - match.matched_curve))

    return {
        "r_squared": float(r_squared),
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": match.correlation,
        "match_error": match.match_error,
    }

