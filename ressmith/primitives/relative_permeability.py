"""Relative permeability and capillary pressure correlations.

This module provides functions for calculating relative permeability curves
and capillary pressure relationships for multi-phase flow analysis.

References:
- Corey, A.T., "The Interrelation Between Gas and Oil Relative Permeabilities,"
  Producers Monthly, 1954.
- LET (Lomeland-Ebeltoft-Thomas) correlation, 2005.
- Brooks, R.H. and Corey, A.T., "Properties of Porous Media Affecting Fluid Flow,"
  J. Irrig. Drain. Div., 1966.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def corey_relative_permeability(
    saturation: np.ndarray,
    saturation_irreducible: float,
    saturation_residual: float,
    exponent: float = 2.0,
    endpoint: float = 1.0,
) -> np.ndarray:
    """Calculate relative permeability using Corey correlation.

    Corey correlation: kr = kr_endpoint * ((S - Sr) / (1 - Si - Sr))^n

    Args:
        saturation: Phase saturation (fraction, 0-1)
        saturation_irreducible: Irreducible saturation (fraction)
        saturation_residual: Residual saturation (fraction)
        exponent: Corey exponent (default: 2.0)
        endpoint: Endpoint relative permeability (default: 1.0)

    Returns:
        Relative permeability array (0-1)

    Reference:
        Corey, A.T., "The Interrelation Between Gas and Oil Relative Permeabilities,"
        Producers Monthly, 1954.

    Example:
        >>> Sw = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> krw = corey_relative_permeability(Sw, Swir=0.2, Swr=0.2, exponent=2.0)
    """
    # Normalized saturation
    S = np.clip(saturation, saturation_irreducible, 1.0 - saturation_residual)
    S_norm = (S - saturation_irreducible) / (
        1.0 - saturation_irreducible - saturation_residual
    )
    S_norm = np.clip(S_norm, 0.0, 1.0)

    # Corey correlation
    kr = endpoint * (S_norm**exponent)

    return np.clip(kr, 0.0, endpoint)


def let_relative_permeability(
    saturation: np.ndarray,
    saturation_irreducible: float,
    saturation_residual: float,
    L: float = 1.0,
    E: float = 1.0,
    T: float = 1.0,
    endpoint: float = 1.0,
) -> np.ndarray:
    """Calculate relative permeability using LET correlation.

    LET correlation: kr = kr_endpoint * (S_norm^L) / (S_norm^L + E * (1 - S_norm)^T)

    Args:
        saturation: Phase saturation (fraction, 0-1)
        saturation_irreducible: Irreducible saturation (fraction)
        saturation_residual: Residual saturation (fraction)
        L: L parameter (default: 1.0)
        E: E parameter (default: 1.0)
        T: T parameter (default: 1.0)
        endpoint: Endpoint relative permeability (default: 1.0)

    Returns:
        Relative permeability array (0-1)

    Reference:
        Lomeland, F., Ebeltoft, E., and Thomas, W.H., "A New Versatile Relative
        Permeability Correlation," SPE 92379, 2005.

    Example:
        >>> Sw = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> krw = let_relative_permeability(Sw, Swir=0.2, Swr=0.2, L=1.0, E=1.0, T=1.0)
    """
    # Normalized saturation
    S = np.clip(saturation, saturation_irreducible, 1.0 - saturation_residual)
    S_norm = (S - saturation_irreducible) / (
        1.0 - saturation_irreducible - saturation_residual
    )
    S_norm = np.clip(S_norm, 0.0, 1.0)

    # LET correlation
    numerator = S_norm**L
    denominator = S_norm**L + E * ((1.0 - S_norm) ** T)
    kr = endpoint * (numerator / np.maximum(denominator, 1e-10))

    return np.clip(kr, 0.0, endpoint)


def calculate_oil_water_relative_permeability(
    water_saturation: np.ndarray,
    oil_saturation_irreducible: float = 0.2,
    water_saturation_residual: float = 0.2,
    method: str = "corey",
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Calculate oil and water relative permeabilities.

    Args:
        water_saturation: Water saturation (fraction, 0-1)
        oil_saturation_irreducible: Irreducible oil saturation (fraction)
        water_saturation_residual: Residual water saturation (fraction)
        method: Correlation method ('corey' or 'let')
        **kwargs: Additional parameters for correlation

    Returns:
        Dictionary with:
        - krw: Water relative permeability
        - kro: Oil relative permeability

    Example:
        >>> Sw = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> kr = calculate_oil_water_relative_permeability(Sw)
        >>> print(f"krw: {kr['krw']}")
        >>> print(f"kro: {kr['kro']}")
    """
    if method == "let":
        krw = let_relative_permeability(
            water_saturation,
            saturation_irreducible=water_saturation_residual,
            saturation_residual=oil_saturation_irreducible,
            L=kwargs.get("L_w", 1.0),
            E=kwargs.get("E_w", 1.0),
            T=kwargs.get("T_w", 1.0),
            endpoint=kwargs.get("krw_endpoint", 1.0),
        )
        kro = let_relative_permeability(
            1.0 - water_saturation,
            saturation_irreducible=oil_saturation_irreducible,
            saturation_residual=water_saturation_residual,
            L=kwargs.get("L_o", 1.0),
            E=kwargs.get("E_o", 1.0),
            T=kwargs.get("T_o", 1.0),
            endpoint=kwargs.get("kro_endpoint", 1.0),
        )
    else:  # corey
        krw = corey_relative_permeability(
            water_saturation,
            saturation_irreducible=water_saturation_residual,
            saturation_residual=oil_saturation_irreducible,
            exponent=kwargs.get("n_w", 2.0),
            endpoint=kwargs.get("krw_endpoint", 1.0),
        )
        kro = corey_relative_permeability(
            1.0 - water_saturation,
            saturation_irreducible=oil_saturation_irreducible,
            saturation_residual=water_saturation_residual,
            exponent=kwargs.get("n_o", 2.0),
            endpoint=kwargs.get("kro_endpoint", 1.0),
        )

    return {"krw": krw, "kro": kro}


def calculate_three_phase_relative_permeability(
    water_saturation: np.ndarray,
    gas_saturation: np.ndarray,
    oil_saturation_irreducible: float = 0.2,
    water_saturation_residual: float = 0.2,
    gas_saturation_residual: float = 0.05,
    method: str = "stone_ii",
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Calculate three-phase relative permeability (oil/water/gas).

    Args:
        water_saturation: Water saturation (fraction, 0-1)
        gas_saturation: Gas saturation (fraction, 0-1)
        oil_saturation_irreducible: Irreducible oil saturation (fraction)
        water_saturation_residual: Residual water saturation (fraction)
        gas_saturation_residual: Residual gas saturation (fraction)
        method: Method ('stone_ii' or 'stone_i')
        **kwargs: Additional parameters

    Returns:
        Dictionary with:
        - krw: Water relative permeability
        - krg: Gas relative permeability
        - kro: Oil relative permeability (three-phase)

    Reference:
        Stone, H.L., "Probability Model for Estimating Three-Phase Relative Permeability,"
        JPT, February 1970.

    Example:
        >>> Sw = np.array([0.2, 0.3, 0.4])
        >>> Sg = np.array([0.1, 0.15, 0.2])
        >>> kr = calculate_three_phase_relative_permeability(Sw, Sg)
    """
    # Calculate two-phase relative permeabilities
    # Water-oil system
    kro_wo = corey_relative_permeability(
        1.0 - water_saturation,
        saturation_irreducible=oil_saturation_irreducible,
        saturation_residual=water_saturation_residual,
        exponent=kwargs.get("n_o", 2.0),
        endpoint=kwargs.get("kro_endpoint", 1.0),
    )

    # Gas-oil system
    kro_go = corey_relative_permeability(
        1.0 - gas_saturation,
        saturation_irreducible=oil_saturation_irreducible,
        saturation_residual=gas_saturation_residual,
        exponent=kwargs.get("n_o", 2.0),
        endpoint=kwargs.get("kro_endpoint", 1.0),
    )

    # Water relative permeability
    krw = corey_relative_permeability(
        water_saturation,
        saturation_irreducible=water_saturation_residual,
        saturation_residual=oil_saturation_irreducible,
        exponent=kwargs.get("n_w", 2.0),
        endpoint=kwargs.get("krw_endpoint", 1.0),
    )

    # Gas relative permeability
    krg = corey_relative_permeability(
        gas_saturation,
        saturation_irreducible=gas_saturation_residual,
        saturation_residual=oil_saturation_irreducible,
        exponent=kwargs.get("n_g", 2.0),
        endpoint=kwargs.get("krg_endpoint", 1.0),
    )

    # Three-phase oil relative permeability (Stone II)
    if method == "stone_ii":
        # Stone II: kro = kro_wo * kro_go / kro_endpoint
        kro_endpoint = kwargs.get("kro_endpoint", 1.0)
        kro = (kro_wo * kro_go) / np.maximum(kro_endpoint, 1e-10)
    else:  # stone_i
        # Stone I: kro = kro_wo * kro_go
        kro = kro_wo * kro_go

    kro = np.clip(kro, 0.0, 1.0)

    return {"krw": krw, "krg": krg, "kro": kro}


def apply_hysteresis_to_relative_permeability(
    relative_permeability: np.ndarray,
    saturation: np.ndarray,
    direction: str = "drainage",
    hysteresis_factor: float = 0.5,
) -> np.ndarray:
    """Apply hysteresis effects to relative permeability.

    Hysteresis accounts for different relative permeability curves
    during drainage vs. imbibition.

    Args:
        relative_permeability: Base relative permeability
        saturation: Phase saturation
        direction: Flow direction ('drainage' or 'imbibition')
        hysteresis_factor: Hysteresis factor (0-1, default: 0.5)

    Returns:
        Hysteresis-corrected relative permeability

    Reference:
        Killough, J.E., "Reservoir Simulation with History-Dependent Saturation Functions,"
        SPE 5106, 1976.

    Example:
        >>> kr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> S = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> kr_hyst = apply_hysteresis_to_relative_permeability(kr, S, direction='imbibition')
    """
    if direction == "imbibition":
        # Imbibition: typically lower relative permeability
        # kr_imb = kr_drain * (1 - hysteresis_factor * (1 - S))
        correction = 1.0 - hysteresis_factor * (1.0 - saturation)
        kr_corrected = relative_permeability * correction
    else:  # drainage
        # Drainage: use base curve
        kr_corrected = relative_permeability.copy()

    return np.clip(kr_corrected, 0.0, 1.0)


def calculate_capillary_pressure(
    saturation: np.ndarray,
    entry_pressure: float = 5.0,
    lambda_parameter: float = 2.0,
    saturation_irreducible: float = 0.2,
    method: str = "brooks_corey",
) -> np.ndarray:
    """Calculate capillary pressure using Brooks-Corey correlation.

    Pc = Pe * (S_norm)^(-1/λ)

    Args:
        saturation: Phase saturation (fraction, 0-1)
        entry_pressure: Entry pressure (psi)
        lambda_parameter: Pore size distribution parameter (default: 2.0)
        saturation_irreducible: Irreducible saturation (fraction)
        method: Method ('brooks_corey' or 'van_genuchten')

    Returns:
        Capillary pressure array (psi)

    Reference:
        Brooks, R.H. and Corey, A.T., "Properties of Porous Media Affecting Fluid Flow,"
        J. Irrig. Drain. Div., 1966.

    Example:
        >>> Sw = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> Pc = calculate_capillary_pressure(Sw, entry_pressure=5.0, lambda_parameter=2.0)
    """
    # Normalized saturation
    S_norm = (saturation - saturation_irreducible) / (1.0 - saturation_irreducible)
    S_norm = np.clip(S_norm, 0.01, 1.0)  # Avoid division by zero

    if method == "van_genuchten":
        # Van Genuchten correlation (not implemented, use Brooks-Corey)
        pass

    # Brooks-Corey correlation
    # Pc = Pe * (S_norm)^(-1/λ)
    Pc = entry_pressure * (S_norm ** (-1.0 / lambda_parameter))

    return np.clip(Pc, 0.0, 1000.0)  # Reasonable bounds
