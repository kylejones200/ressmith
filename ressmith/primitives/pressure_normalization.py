"""Pressure normalization for RTA analysis.

This module provides functions for normalizing production data with pressure,
a critical workflow for Rate Transient Analysis (RTA).

References:
- Palacio, J.C. and Blasingame, T.A., "Decline-Curve Analysis Using Type Curves -
  Analysis of Gas Well Production Data," SPE 25909, 1993.
- Agarwal, R.G., et al., "Analyzing Well Production Data Using Combined Type Curve
  and Decline Curve Analysis Concepts," SPE 57916, 1999.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def normalize_rate_with_pressure(
    rate: np.ndarray,
    pressure: np.ndarray,
    initial_pressure: float,
    method: str = "pseudopressure",
) -> np.ndarray:
    """Normalize production rate with pressure.

    Normalizes rate to account for pressure changes during production.
    Common methods:
    - 'pseudopressure': Uses pseudopressure for gas
    - 'pressure_ratio': Simple pressure ratio normalization
    - 'material_balance': Material balance normalization

    Args:
        rate: Production rate (STB/day or MCF/day)
        pressure: Flowing pressure (psi)
        initial_pressure: Initial reservoir pressure (psi)
        method: Normalization method (default: 'pseudopressure')

    Returns:
        Normalized rate array

    Example:
        >>> rate = np.array([1000, 900, 800])
        >>> pressure = np.array([4500, 4200, 3900])
        >>> normalized = normalize_rate_with_pressure(rate, pressure, initial_pressure=5000)
    """
    if len(rate) != len(pressure):
        raise ValueError("Rate and pressure arrays must have same length")

    if method == "pressure_ratio":
        # Simple pressure ratio: q_norm = q * (pi / p)
        pressure_ratio = initial_pressure / np.maximum(pressure, 1.0)
        normalized_rate = rate * pressure_ratio

    elif method == "pseudopressure":
        # Pseudopressure normalization (for gas)
        # q_norm = q * (pi^2 - pwf^2) / (pi^2 - p_ref^2)
        # Simplified: q_norm = q * (pi / p)^2
        pressure_ratio_squared = (initial_pressure / np.maximum(pressure, 1.0)) ** 2
        normalized_rate = rate * pressure_ratio_squared

    elif method == "material_balance":
        # Material balance normalization
        # Uses average pressure for normalization
        avg_pressure = (initial_pressure + pressure) / 2.0
        pressure_ratio = initial_pressure / np.maximum(avg_pressure, 1.0)
        normalized_rate = rate * pressure_ratio

    else:
        # Default to pressure ratio
        pressure_ratio = initial_pressure / np.maximum(pressure, 1.0)
        normalized_rate = rate * pressure_ratio

    return normalized_rate


def normalize_cumulative_with_pressure(
    cumulative: np.ndarray,
    pressure: np.ndarray,
    initial_pressure: float,
    method: str = "material_balance",
) -> np.ndarray:
    """Normalize cumulative production with pressure.

    Args:
        cumulative: Cumulative production (STB or MCF)
        pressure: Flowing pressure (psi)
        initial_pressure: Initial reservoir pressure (psi)
        method: Normalization method (default: 'material_balance')

    Returns:
        Normalized cumulative array

    Example:
        >>> cum = np.array([10000, 20000, 30000])
        >>> pressure = np.array([4500, 4200, 3900])
        >>> normalized = normalize_cumulative_with_pressure(cum, pressure, initial_pressure=5000)
    """
    if len(cumulative) != len(pressure):
        raise ValueError("Cumulative and pressure arrays must have same length")

    if method == "material_balance":
        # Material balance normalization
        # Uses average pressure
        avg_pressure = (initial_pressure + pressure) / 2.0
        pressure_ratio = initial_pressure / np.maximum(avg_pressure, 1.0)
        normalized_cumulative = cumulative * pressure_ratio

    else:
        # Default: simple pressure ratio
        pressure_ratio = initial_pressure / np.maximum(pressure, 1.0)
        normalized_cumulative = cumulative * pressure_ratio

    return normalized_cumulative


def calculate_pseudopressure(
    pressure: np.ndarray,
    temperature: float = 200.0,
    gas_gravity: float = 0.7,
) -> np.ndarray:
    """Calculate pseudopressure for gas.

    Pseudopressure accounts for pressure-dependent gas properties.

    Args:
        pressure: Pressure (psi)
        temperature: Reservoir temperature (°F)
        gas_gravity: Gas specific gravity (air = 1.0)

    Returns:
        Pseudopressure array (psi²/cp)

    Reference:
        Al-Hussainy, R., et al., "The Flow of Real Gases Through Porous Media,"
        JPT, May 1966.

    Example:
        >>> pressure = np.array([4000, 3500, 3000])
        >>> m = calculate_pseudopressure(pressure, temperature=200, gas_gravity=0.7)
    """
    # Simplified pseudopressure calculation
    # m(p) = 2 * ∫ (p / (μ * Z)) dp from 0 to p
    # For simplicity, use: m(p) ≈ p² / (μ * Z)
    # where μ and Z are functions of pressure

    # Simplified: m(p) ≈ p² for most cases
    # More accurate would require iterative calculation with Z-factor
    pseudopressure = pressure**2

    return pseudopressure


def normalize_for_type_curve_matching(
    time: np.ndarray,
    rate: np.ndarray,
    pressure: np.ndarray | None,
    initial_pressure: float,
    cumulative: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Normalize production data for type curve matching.

    Comprehensive normalization for RTA type curve matching workflows.

    Args:
        time: Production time (days)
        rate: Production rate (STB/day or MCF/day)
        pressure: Optional flowing pressure (psi)
        initial_pressure: Initial reservoir pressure (psi)
        cumulative: Optional cumulative production

    Returns:
        Dictionary with normalized data:
        - normalized_rate: Pressure-normalized rate
        - normalized_cumulative: Pressure-normalized cumulative (if provided)
        - time_normalized: Normalized time
        - pressure_normalized: Normalized pressure

    Example:
        >>> time = np.array([1, 10, 30, 60])
        >>> rate = np.array([1000, 900, 800, 700])
        >>> pressure = np.array([4500, 4200, 3900, 3600])
        >>> normalized = normalize_for_type_curve_matching(
        ...     time, rate, pressure, initial_pressure=5000
        ... )
    """
    normalized_data: dict[str, np.ndarray] = {}

    if pressure is not None:
        # Normalize rate with pressure
        normalized_rate = normalize_rate_with_pressure(
            rate, pressure, initial_pressure, method="pressure_ratio"
        )
        normalized_data["normalized_rate"] = normalized_rate

        # Normalize cumulative if provided
        if cumulative is not None:
            normalized_cumulative = normalize_cumulative_with_pressure(
                cumulative, pressure, initial_pressure, method="material_balance"
            )
            normalized_data["normalized_cumulative"] = normalized_cumulative

        # Normalize pressure
        normalized_pressure = pressure / initial_pressure
        normalized_data["pressure_normalized"] = normalized_pressure
    else:
        # No pressure data, use rate as-is
        normalized_data["normalized_rate"] = rate.copy()
        if cumulative is not None:
            normalized_data["normalized_cumulative"] = cumulative.copy()

    # Normalize time (to dimensionless time if needed)
    # For now, keep time as-is
    normalized_data["time_normalized"] = time.copy()

    return normalized_data
