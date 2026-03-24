"""Relative permeability and capillary pressure workflows.

Provides workflows for calculating relative permeability curves
and capillary pressure relationships.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.relative_permeability import (
    apply_hysteresis_to_relative_permeability,
    calculate_capillary_pressure,
    calculate_oil_water_relative_permeability,
    calculate_three_phase_relative_permeability,
    corey_relative_permeability,
    let_relative_permeability,
)

logger = logging.getLogger(__name__)


def generate_relative_permeability_curves(
    saturation: np.ndarray | pd.Series,
    saturation_irreducible: float = 0.2,
    saturation_residual: float = 0.2,
    method: str = "corey",
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate relative permeability curves.

    Parameters
    ----------
    saturation : np.ndarray or pd.Series
        Phase saturation array (0-1)
    saturation_irreducible : float
        Irreducible saturation (fraction)
    saturation_residual : float
        Residual saturation (fraction)
    method : str
        Correlation method ('corey' or 'let')
    **kwargs
        Additional parameters for correlation

    Returns
    -------
    pd.DataFrame
        DataFrame with saturation and relative permeability

    Examples
    --------
    >>> import numpy as np
    >>> Sw = np.linspace(0.2, 0.8, 50)
    >>> curves = generate_relative_permeability_curves(Sw, method='corey')
    >>> print(curves.head())
    """
    logger.info(f"Generating relative permeability curves: method={method}")

    if isinstance(saturation, pd.Series):
        saturation = saturation.values

    if method == "let":
        kr = let_relative_permeability(
            saturation,
            saturation_irreducible=saturation_irreducible,
            saturation_residual=saturation_residual,
            L=kwargs.get("L", 1.0),
            E=kwargs.get("E", 1.0),
            T=kwargs.get("T", 1.0),
            endpoint=kwargs.get("endpoint", 1.0),
        )
    else:  # corey
        kr = corey_relative_permeability(
            saturation,
            saturation_irreducible=saturation_irreducible,
            saturation_residual=saturation_residual,
            exponent=kwargs.get("exponent", 2.0),
            endpoint=kwargs.get("endpoint", 1.0),
        )

    return pd.DataFrame({"saturation": saturation, "relative_permeability": kr})


def generate_oil_water_relative_permeability(
    water_saturation: np.ndarray | pd.Series,
    oil_saturation_irreducible: float = 0.2,
    water_saturation_residual: float = 0.2,
    method: str = "corey",
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate oil-water relative permeability curves.

    Parameters
    ----------
    water_saturation : np.ndarray or pd.Series
        Water saturation array (0-1)
    oil_saturation_irreducible : float
        Irreducible oil saturation (fraction)
    water_saturation_residual : float
        Residual water saturation (fraction)
    method : str
        Correlation method ('corey' or 'let')
    **kwargs
        Additional parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with saturation, krw, and kro

    Examples
    --------
    >>> import numpy as np
    >>> Sw = np.linspace(0.2, 0.8, 50)
    >>> curves = generate_oil_water_relative_permeability(Sw)
    >>> print(curves[['saturation', 'krw', 'kro']].head())
    """
    logger.info("Generating oil-water relative permeability curves")

    if isinstance(water_saturation, pd.Series):
        water_saturation = water_saturation.values

    kr = calculate_oil_water_relative_permeability(
        water_saturation,
        oil_saturation_irreducible=oil_saturation_irreducible,
        water_saturation_residual=water_saturation_residual,
        method=method,
        **kwargs,
    )

    return pd.DataFrame(
        {
            "water_saturation": water_saturation,
            "oil_saturation": 1.0 - water_saturation,
            "krw": kr["krw"],
            "kro": kr["kro"],
        }
    )


def generate_three_phase_relative_permeability(
    water_saturation: np.ndarray | pd.Series,
    gas_saturation: np.ndarray | pd.Series,
    oil_saturation_irreducible: float = 0.2,
    water_saturation_residual: float = 0.2,
    gas_saturation_residual: float = 0.05,
    method: str = "stone_ii",
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate three-phase relative permeability curves.

    Parameters
    ----------
    water_saturation : np.ndarray or pd.Series
        Water saturation array (0-1)
    gas_saturation : np.ndarray or pd.Series
        Gas saturation array (0-1)
    oil_saturation_irreducible : float
        Irreducible oil saturation (fraction)
    water_saturation_residual : float
        Residual water saturation (fraction)
    gas_saturation_residual : float
        Residual gas saturation (fraction)
    method : str
        Method ('stone_ii' or 'stone_i')
    **kwargs
        Additional parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with saturations and relative permeabilities

    Examples
    --------
    >>> import numpy as np
    >>> Sw = np.array([0.2, 0.3, 0.4])
    >>> Sg = np.array([0.1, 0.15, 0.2])
    >>> curves = generate_three_phase_relative_permeability(Sw, Sg)
    """
    logger.info("Generating three-phase relative permeability curves")

    if isinstance(water_saturation, pd.Series):
        water_saturation = water_saturation.values
    if isinstance(gas_saturation, pd.Series):
        gas_saturation = gas_saturation.values

    kr = calculate_three_phase_relative_permeability(
        water_saturation,
        gas_saturation,
        oil_saturation_irreducible=oil_saturation_irreducible,
        water_saturation_residual=water_saturation_residual,
        gas_saturation_residual=gas_saturation_residual,
        method=method,
        **kwargs,
    )

    oil_saturation = 1.0 - water_saturation - gas_saturation

    return pd.DataFrame(
        {
            "water_saturation": water_saturation,
            "gas_saturation": gas_saturation,
            "oil_saturation": oil_saturation,
            "krw": kr["krw"],
            "krg": kr["krg"],
            "kro": kr["kro"],
        }
    )


def generate_capillary_pressure_curve(
    saturation: np.ndarray | pd.Series,
    entry_pressure: float = 5.0,
    lambda_parameter: float = 2.0,
    saturation_irreducible: float = 0.2,
    method: str = "brooks_corey",
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate capillary pressure curve.

    Parameters
    ----------
    saturation : np.ndarray or pd.Series
        Phase saturation array (0-1)
    entry_pressure : float
        Entry pressure (psi)
    lambda_parameter : float
        Pore size distribution parameter
    saturation_irreducible : float
        Irreducible saturation (fraction)
    method : str
        Method ('brooks_corey' or 'van_genuchten')
    **kwargs
        Passed to :func:`ressmith.primitives.relative_permeability.calculate_capillary_pressure`
        (e.g. ``vg_alpha``, ``vg_n``, ``vg_m`` for van Genuchten).

    Returns
    -------
    pd.DataFrame
        DataFrame with saturation and capillary pressure

    Examples
    --------
    >>> import numpy as np
    >>> Sw = np.linspace(0.2, 0.8, 50)
    >>> curve = generate_capillary_pressure_curve(Sw, entry_pressure=5.0)
    >>> print(curve.head())
    """
    logger.info("Generating capillary pressure curve")

    if isinstance(saturation, pd.Series):
        saturation = saturation.values

    Pc = calculate_capillary_pressure(
        saturation,
        entry_pressure=entry_pressure,
        lambda_parameter=lambda_parameter,
        saturation_irreducible=saturation_irreducible,
        method=method,
        **kwargs,
    )

    return pd.DataFrame({"saturation": saturation, "capillary_pressure": Pc})


def apply_hysteresis_workflow(
    relative_permeability: np.ndarray | pd.Series,
    saturation: np.ndarray | pd.Series,
    direction: str = "drainage",
    hysteresis_factor: float = 0.5,
) -> pd.Series:
    """Apply hysteresis effects to relative permeability.

    Parameters
    ----------
    relative_permeability : np.ndarray or pd.Series
        Base relative permeability
    saturation : np.ndarray or pd.Series
        Phase saturation
    direction : str
        Flow direction ('drainage' or 'imbibition')
    hysteresis_factor : float
        Hysteresis factor (0-1)

    Returns
    -------
    pd.Series
        Hysteresis-corrected relative permeability

    Examples
    --------
    >>> import numpy as np
    >>> kr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> S = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    >>> kr_hyst = apply_hysteresis_workflow(kr, S, direction='imbibition')
    """
    logger.info(f"Applying hysteresis: direction={direction}")

    if isinstance(relative_permeability, pd.Series):
        relative_permeability = relative_permeability.values
    if isinstance(saturation, pd.Series):
        saturation = saturation.values

    kr_corrected = apply_hysteresis_to_relative_permeability(
        relative_permeability, saturation, direction, hysteresis_factor
    )

    return pd.Series(kr_corrected, name="relative_permeability_hysteresis")
