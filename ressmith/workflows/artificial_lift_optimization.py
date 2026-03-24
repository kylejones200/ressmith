"""Artificial lift optimization workflows.

Provides workflows for optimizing ESP, gas lift, and rod pump systems.
"""

import logging
from typing import Any

import pandas as pd

from ressmith.primitives.vlp import perform_nodal_analysis

logger = logging.getLogger(__name__)


def optimize_esp_system(
    reservoir_pressure: float,
    productivity_index: float,
    wellhead_pressure: float,
    tubing_depth: float,
    target_rate: float | None = None,
    efficiency: float = 0.6,
) -> dict[str, Any]:
    """Optimize ESP (Electric Submersible Pump) system.

    Parameters
    ----------
    reservoir_pressure : float
        Reservoir pressure (psi)
    productivity_index : float
        Productivity index (STB/day/psi)
    wellhead_pressure : float
        Wellhead pressure (psi)
    tubing_depth : float
        Tubing depth (ft)
    target_rate : float, optional
        Target production rate (STB/day)
    efficiency : float
        Pump efficiency (fraction, default: 0.6)

    Returns
    -------
    dict
        Dictionary with optimization results:
        - operating_rate: Operating production rate (STB/day)
        - operating_pressure: Operating pressure (psi)
        - power_required: Power required (HP)
        - efficiency: Pump efficiency

    Examples
    --------
    >>> result = optimize_esp_system(
    ...     reservoir_pressure=5000,
    ...     productivity_index=1.0,
    ...     wellhead_pressure=500,
    ...     tubing_depth=5000
    ... )
    >>> print(f"Power required: {result['power_required']:.1f} HP")
    """
    logger.info("Optimizing ESP system")

    # Perform nodal analysis
    nodal_result = perform_nodal_analysis(
        reservoir_pressure=reservoir_pressure,
        productivity_index=productivity_index,
        wellhead_pressure=wellhead_pressure,
        tubing_depth=tubing_depth,
    )

    operating_rate = nodal_result.operating_rate
    operating_pressure = nodal_result.operating_pressure

    # Calculate ESP power
    # Power (HP) = (q * Δp * 0.000017) / efficiency
    # where Δp is pressure boost required
    pressure_boost = operating_pressure - wellhead_pressure
    power_hp = (operating_rate * pressure_boost * 0.000017) / efficiency

    return {
        "operating_rate": float(operating_rate),
        "operating_pressure": float(operating_pressure),
        "power_required": float(power_hp),
        "efficiency": float(efficiency),
        "pressure_boost": float(pressure_boost),
    }


def optimize_gas_lift_system(
    reservoir_pressure: float,
    productivity_index: float,
    wellhead_pressure: float,
    tubing_depth: float,
    target_rate: float | None = None,
    gas_injection_depth: float | None = None,
) -> dict[str, Any]:
    """Optimize gas lift system.

    Parameters
    ----------
    reservoir_pressure : float
        Reservoir pressure (psi)
    productivity_index : float
        Productivity index (STB/day/psi)
    wellhead_pressure : float
        Wellhead pressure (psi)
    tubing_depth : float
        Tubing depth (ft)
    target_rate : float, optional
        Target production rate (STB/day)
    gas_injection_depth : float, optional
        Gas injection depth (ft, if None uses tubing_depth/2)

    Returns
    -------
    dict
        Dictionary with optimization results:
        - operating_rate: Operating production rate (STB/day)
        - gas_injection_rate: Required gas injection rate (MCF/day)
        - gas_liquid_ratio: Gas-liquid ratio (MCF/STB)

    Examples
    --------
    >>> result = optimize_gas_lift_system(
    ...     reservoir_pressure=5000,
    ...     productivity_index=1.0,
    ...     wellhead_pressure=500,
    ...     tubing_depth=5000
    ... )
    >>> print(f"Gas injection rate: {result['gas_injection_rate']:.0f} MCF/day")
    """
    logger.info("Optimizing gas lift system")

    if gas_injection_depth is None:
        gas_injection_depth = tubing_depth / 2.0

    # Perform nodal analysis
    nodal_result = perform_nodal_analysis(
        reservoir_pressure=reservoir_pressure,
        productivity_index=productivity_index,
        wellhead_pressure=wellhead_pressure,
        tubing_depth=tubing_depth,
    )

    operating_rate = nodal_result.operating_rate

    # Estimate gas injection rate
    # Simplified: gas rate ≈ 0.5 * oil rate for typical gas lift
    gas_injection_rate = operating_rate * 0.5  # MCF/day
    gas_liquid_ratio = (
        gas_injection_rate / operating_rate if operating_rate > 0 else 0.0
    )

    return {
        "operating_rate": float(operating_rate),
        "gas_injection_rate": float(gas_injection_rate),
        "gas_liquid_ratio": float(gas_liquid_ratio),
        "gas_injection_depth": float(gas_injection_depth),
    }


def optimize_rod_pump_system(
    reservoir_pressure: float,
    productivity_index: float,
    wellhead_pressure: float,
    tubing_depth: float,
    target_rate: float | None = None,
    pump_efficiency: float = 0.7,
) -> dict[str, Any]:
    """Optimize rod pump system.

    Parameters
    ----------
    reservoir_pressure : float
        Reservoir pressure (psi)
    productivity_index : float
        Productivity index (STB/day/psi)
    wellhead_pressure : float
        Wellhead pressure (psi)
    tubing_depth : float
        Tubing depth (ft)
    target_rate : float, optional
        Target production rate (STB/day)
    pump_efficiency : float
        Pump efficiency (fraction, default: 0.7)

    Returns
    -------
    dict
        Dictionary with optimization results:
        - operating_rate: Operating production rate (STB/day)
        - pump_stroke_length: Recommended stroke length (inches)
        - strokes_per_minute: Recommended strokes per minute
        - power_required: Power required (HP)

    Examples
    --------
    >>> result = optimize_rod_pump_system(
    ...     reservoir_pressure=5000,
    ...     productivity_index=1.0,
    ...     wellhead_pressure=500,
    ...     tubing_depth=5000
    ... )
    >>> print(f"Power required: {result['power_required']:.1f} HP")
    """
    logger.info("Optimizing rod pump system")

    # Perform nodal analysis
    nodal_result = perform_nodal_analysis(
        reservoir_pressure=reservoir_pressure,
        productivity_index=productivity_index,
        wellhead_pressure=wellhead_pressure,
        tubing_depth=tubing_depth,
    )

    operating_rate = nodal_result.operating_rate

    # Estimate pump parameters
    # Simplified calculations
    pump_stroke_length = 60.0  # inches (typical)
    strokes_per_minute = operating_rate / (pump_stroke_length * 0.1)  # Simplified
    strokes_per_minute = max(5.0, min(20.0, strokes_per_minute))  # Reasonable bounds

    # Power calculation (simplified)
    pressure_boost = nodal_result.operating_pressure - wellhead_pressure
    power_hp = (operating_rate * pressure_boost * 0.00002) / pump_efficiency

    return {
        "operating_rate": float(operating_rate),
        "pump_stroke_length": float(pump_stroke_length),
        "strokes_per_minute": float(strokes_per_minute),
        "power_required": float(power_hp),
        "pump_efficiency": float(pump_efficiency),
    }


def compare_artificial_lift_methods(
    reservoir_pressure: float,
    productivity_index: float,
    wellhead_pressure: float,
    tubing_depth: float,
) -> pd.DataFrame:
    """Compare different artificial lift methods.

    Parameters
    ----------
    reservoir_pressure : float
        Reservoir pressure (psi)
    productivity_index : float
        Productivity index (STB/day/psi)
    wellhead_pressure : float
        Wellhead pressure (psi)
    tubing_depth : float
        Tubing depth (ft)

    Returns
    -------
    pd.DataFrame
        DataFrame comparing ESP, gas lift, and rod pump

    Examples
    --------
    >>> comparison = compare_artificial_lift_methods(
    ...     reservoir_pressure=5000,
    ...     productivity_index=1.0,
    ...     wellhead_pressure=500,
    ...     tubing_depth=5000
    ... )
    >>> print(comparison)
    """
    logger.info("Comparing artificial lift methods")

    results = []

    # ESP
    esp_result = optimize_esp_system(
        reservoir_pressure, productivity_index, wellhead_pressure, tubing_depth
    )
    results.append(
        {
            "method": "ESP",
            "operating_rate": esp_result["operating_rate"],
            "power_required": esp_result["power_required"],
            "efficiency": esp_result["efficiency"],
        }
    )

    # Gas lift
    gl_result = optimize_gas_lift_system(
        reservoir_pressure, productivity_index, wellhead_pressure, tubing_depth
    )
    results.append(
        {
            "method": "Gas Lift",
            "operating_rate": gl_result["operating_rate"],
            "gas_injection_rate": gl_result["gas_injection_rate"],
            "power_required": gl_result["gas_injection_rate"] * 0.001,  # Simplified
        }
    )

    # Rod pump
    rp_result = optimize_rod_pump_system(
        reservoir_pressure, productivity_index, wellhead_pressure, tubing_depth
    )
    results.append(
        {
            "method": "Rod Pump",
            "operating_rate": rp_result["operating_rate"],
            "power_required": rp_result["power_required"],
            "efficiency": rp_result["pump_efficiency"],
        }
    )

    return pd.DataFrame(results)
