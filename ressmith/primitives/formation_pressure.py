"""Formation pressure analysis for geopressure evaluation.

This module implements formation pressure analysis including:
- Pore pressure prediction methods
- Fracture gradient estimation
- Overburden stress calculations
- Pressure-depth relationships
- Abnormal pressure detection
- Geopressure analysis
- Eaton's method, d-exponent, and other prediction techniques

Based on industry-standard formation pressure evaluation methods.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ressmith.primitives.constants import PhysicalConstants

logger = logging.getLogger(__name__)


@dataclass
class OverburdenData:
    """Overburden stress data."""

    depth: float  # Depth in ft
    density_log: list[float]  # Bulk density from logs (g/cc)
    depth_points: list[float]  # Depth points for density (ft)


@dataclass
class PressurePoint:
    """Single pressure measurement point."""

    depth: float  # Measured depth in ft
    pressure: float  # Pressure in psi
    pressure_type: str  # 'pore', 'fracture', 'overburden'
    source: str  # Measurement source


@dataclass
class LogData:
    """Well log data for pressure analysis."""

    depth: list[float]  # Depth in ft
    resistivity: list[float]  # Resistivity in ohm-m
    sonic_dt: list[float]  # Sonic transit time in μs/ft
    density: list[float]  # Bulk density in g/cc
    neutron_porosity: list[float]  # Neutron porosity (fraction)


# ---------------------------------------------------------------------------
# Overburden stress
# ---------------------------------------------------------------------------


def calculate_overburden(
    depths: list[float],
    densities: list[float],
    water_depth: float = 0.0,
) -> dict:
    """Calculate overburden stress from density log.

    Args:
        depths: Depth points in ft
        densities: Bulk density in g/cc at each depth
        water_depth: Water depth in ft (for offshore)

    Returns:
        dict with overburden pressure profile
    """
    depths_arr = np.array(depths)
    densities_arr = np.array(densities)

    # Initialize arrays
    overburden_pressure = np.zeros_like(depths_arr)
    overburden_gradient = np.zeros_like(depths_arr)

    # Water column (if offshore): seawater gradient ≈ 0.465 psi/ft
    if water_depth > 0:
        water_pressure = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD * water_depth
    else:
        water_pressure = 0.0

    # Integrate density to get overburden
    for i in range(len(depths_arr)):
        if i == 0:
            # First point
            if water_depth > 0:
                overburden_pressure[i] = (
                    water_pressure
                    + 0.433 * densities_arr[i] * (depths_arr[i] - water_depth)
                )
            else:
                overburden_pressure[i] = 0.433 * densities_arr[i] * depths_arr[i]
        else:
            # Trapezoidal integration
            depth_interval = depths_arr[i] - depths_arr[i - 1]
            avg_density = (densities_arr[i] + densities_arr[i - 1]) / 2.0
            overburden_pressure[i] = (
                overburden_pressure[i - 1] + 0.433 * avg_density * depth_interval
            )

        # Calculate gradient
        if depths_arr[i] > 0:
            overburden_gradient[i] = overburden_pressure[i] / depths_arr[i]

    return {
        "depths_ft": depths_arr.tolist(),
        "overburden_pressure_psi": overburden_pressure.tolist(),
        "overburden_gradient_psi_ft": overburden_gradient.tolist(),
        "equivalent_mud_weight_ppg": (
            overburden_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT
        ).tolist(),
    }


def estimate_overburden_from_depth(
    depth: float,
    water_depth: float = 0.0,
    average_density: float = 2.31,
) -> dict:
    """Estimate overburden using average density.

    Args:
        depth: Depth in ft
        water_depth: Water depth in ft
        average_density: Average formation density in g/cc

    Returns:
        dict with overburden estimate
    """
    if water_depth > 0:
        # Offshore
        water_pressure = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD * water_depth
        sediment_pressure = 0.433 * average_density * (depth - water_depth)
        total_overburden = water_pressure + sediment_pressure
    else:
        # Onshore
        total_overburden = 0.433 * average_density * depth

    gradient = total_overburden / depth if depth > 0 else 0
    emw = (
        gradient / PhysicalConstants.HYDROSTATIC_GRADIENT if gradient > 0 else 0
    )

    return {
        "depth_ft": depth,
        "overburden_pressure_psi": round(total_overburden, 0),
        "gradient_psi_ft": round(gradient, 3),
        "equivalent_mud_weight_ppg": round(emw, 2),
    }


# ---------------------------------------------------------------------------
# Pore pressure prediction
# ---------------------------------------------------------------------------


def normal_compaction_trend(
    depth: float,
    surface_porosity: float = 0.40,
    compaction_constant: float = 0.0003,
) -> float:
    """Calculate normal compaction trend porosity.

    Args:
        depth: Depth in ft
        surface_porosity: Surface porosity (fraction)
        compaction_constant: Compaction constant (1/ft)

    Returns:
        Normal trend porosity
    """
    # Athy's law: φ = φ0 * exp(-c * z)
    porosity = surface_porosity * math.exp(-compaction_constant * depth)
    return porosity


def eatons_method(
    observed_parameter: float,
    normal_parameter: float,
    overburden_gradient: float,
    normal_pressure_gradient: float = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
    exponent: float = 3.0,
) -> dict:
    """Eaton's method for pore pressure prediction.

    Args:
        observed_parameter: Observed log parameter (resistivity or sonic)
        normal_parameter: Normal trend parameter at same depth
        overburden_gradient: Overburden gradient in psi/ft
        normal_pressure_gradient: Normal pressure gradient in psi/ft
        exponent: Eaton's exponent (typically 1.2 for sonic, 3.0 for resistivity)

    Returns:
        dict with predicted pore pressure gradient
    """
    if normal_parameter == 0:
        return {"error": "Invalid normal parameter value"}

    # Eaton's equation
    # Ppore = OB - (OB - Pnormal) * (observed/normal)^x

    ratio = observed_parameter / normal_parameter

    pore_pressure_gradient = overburden_gradient - (
        (overburden_gradient - normal_pressure_gradient) * (ratio**exponent)
    )

    emw = pore_pressure_gradient / 0.052

    return {
        "pore_pressure_gradient_psi_ft": round(pore_pressure_gradient, 4),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "parameter_ratio": round(ratio, 3),
        "method": "Eatons Method",
    }


def sonic_method(
    observed_dt: float,
    normal_dt: float,
    overburden_gradient: float,
    normal_pressure_gradient: float = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
) -> dict:
    """Sonic method for pore pressure prediction.

    Args:
        observed_dt: Observed sonic transit time (μs/ft)
        normal_dt: Normal trend sonic time (μs/ft)
        overburden_gradient: Overburden gradient (psi/ft)
        normal_pressure_gradient: Normal pressure gradient (psi/ft)

    Returns:
        dict with predicted pore pressure
    """
    # Using Eaton's method with sonic exponent
    return eatons_method(
        observed_parameter=normal_dt,  # Inverted for sonic
        normal_parameter=observed_dt,
        overburden_gradient=overburden_gradient,
        normal_pressure_gradient=normal_pressure_gradient,
        exponent=3.0,
    )


def resistivity_method(
    observed_resistivity: float,
    normal_resistivity: float,
    overburden_gradient: float,
    normal_pressure_gradient: float = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
) -> dict:
    """Resistivity method for pore pressure prediction.

    Args:
        observed_resistivity: Observed resistivity (ohm-m)
        normal_resistivity: Normal trend resistivity (ohm-m)
        overburden_gradient: Overburden gradient (psi/ft)
        normal_pressure_gradient: Normal pressure gradient (psi/ft)

    Returns:
        dict with predicted pore pressure
    """
    # Using Eaton's method with resistivity exponent
    return eatons_method(
        observed_parameter=observed_resistivity,
        normal_parameter=normal_resistivity,
        overburden_gradient=overburden_gradient,
        normal_pressure_gradient=normal_pressure_gradient,
        exponent=1.2,
    )


def d_exponent_method(
    rop: float,
    rpm: float,
    wob: float,
    bit_diameter: float,
    mud_weight: float,
    normal_d_exp: float = 1.0,
) -> dict:
    """D-exponent method for pore pressure detection.

    Args:
        rop: Rate of penetration (ft/hr)
        rpm: Rotary speed (RPM)
        wob: Weight on bit (1000 lbs)
        bit_diameter: Bit diameter (inches)
        mud_weight: Mud weight (ppg)
        normal_d_exp: Normal trend d-exponent

    Returns:
        dict with d-exponent and pressure indication
    """
    # D-exponent calculation
    # d = log(ROP/60) / log(12*RPM / (1000*WOB/D))

    if rpm == 0 or wob == 0 or bit_diameter == 0:
        return {"error": "Invalid input parameters"}

    try:
        numerator = math.log10(rop / 60)
        denominator = math.log10((12 * rpm) / (1000 * wob / bit_diameter))

        if denominator == 0:
            return {"error": "Invalid calculation - division by zero"}

        d_exp = numerator / denominator

        # Corrected d-exponent (accounting for mud weight)
        d_corrected = d_exp * (mud_weight / 8.33)

        # Determine if overpressure
        if d_corrected < normal_d_exp * 0.7:
            indication = "Significant overpressure - reduce mud weight or investigate"
        elif d_corrected < normal_d_exp * 0.9:
            indication = "Possible overpressure - monitor closely"
        else:
            indication = "Normal pressure trend"

        return {
            "d_exponent": round(d_exp, 3),
            "d_corrected": round(d_corrected, 3),
            "normal_trend": normal_d_exp,
            "indication": indication,
            "method": "D-Exponent",
        }
    except (ValueError, ZeroDivisionError):
        return {"error": "Mathematical error in calculation"}


def equivalent_depth_method(
    observed_parameter: float,
    depths: list[float],
    normal_parameters: list[float],
    normal_gradients: list[float],
) -> dict:
    """Equivalent depth method.

    Args:
        observed_parameter: Observed parameter value
        depths: Depth points for normal trend
        normal_parameters: Normal trend parameter values
        normal_gradients: Normal pressure gradients at each depth

    Returns:
        dict with predicted pore pressure
    """
    # Find equivalent depth where observed parameter matches normal trend
    depths_arr = np.array(depths)
    normal_parameters_arr = np.array(normal_parameters)
    normal_gradients_arr = np.array(normal_gradients)

    # Interpolate to find equivalent depth
    if (
        observed_parameter < normal_parameters_arr[0]
        or observed_parameter > normal_parameters_arr[-1]
    ):
        return {"error": "Observed parameter outside normal trend range"}

    equivalent_depth = np.interp(
        observed_parameter, normal_parameters_arr, depths_arr
    )
    equivalent_gradient = np.interp(
        equivalent_depth, depths_arr, normal_gradients_arr
    )

    emw = equivalent_gradient / 0.052

    return {
        "equivalent_depth_ft": round(equivalent_depth, 0),
        "pore_pressure_gradient_psi_ft": round(equivalent_gradient, 4),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "method": "Equivalent Depth",
    }


# ---------------------------------------------------------------------------
# Fracture gradient
# ---------------------------------------------------------------------------


def matthews_kelly_method(
    depth: float,
    overburden_gradient: float,
    pore_pressure_gradient: float,
    matrix_stress_coefficient: float = 0.33,
) -> dict:
    """Matthews & Kelly method for fracture gradient.

    Args:
        depth: Depth in ft
        overburden_gradient: Overburden gradient (psi/ft)
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        matrix_stress_coefficient: Matrix stress coefficient (Poisson's ratio related)

    Returns:
        dict with fracture gradient
    """
    # F/D = (S - P) * (K / (1-K)) + P
    # where S = overburden, P = pore pressure, K = matrix stress coefficient

    frac_gradient = (
        (overburden_gradient - pore_pressure_gradient)
        * (matrix_stress_coefficient / (1 - matrix_stress_coefficient))
        + pore_pressure_gradient
    )

    frac_pressure = frac_gradient * depth
    emw = frac_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT

    return {
        "fracture_gradient_psi_ft": round(frac_gradient, 4),
        "fracture_pressure_psi": round(frac_pressure, 0),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "method": "Matthews & Kelly",
    }


def eatons_fracture_method(
    depth: float,
    overburden_gradient: float,
    pore_pressure_gradient: float,
    poissons_ratio: float = 0.25,
) -> dict:
    """Eaton's method for fracture gradient.

    Args:
        depth: Depth in ft
        overburden_gradient: Overburden gradient (psi/ft)
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        poissons_ratio: Poisson's ratio

    Returns:
        dict with fracture gradient
    """
    # F = (ν/(1-ν)) * (S - P) + P

    coefficient = poissons_ratio / (1 - poissons_ratio)
    frac_gradient = (
        coefficient * (overburden_gradient - pore_pressure_gradient)
        + pore_pressure_gradient
    )

    frac_pressure = frac_gradient * depth
    emw = frac_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT

    return {
        "fracture_gradient_psi_ft": round(frac_gradient, 4),
        "fracture_pressure_psi": round(frac_pressure, 0),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "poissons_ratio": poissons_ratio,
        "method": "Eatons Fracture Gradient",
    }


def hubert_willis_method(
    depth: float,
    pore_pressure_gradient: float = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
) -> dict:
    """Hubert & Willis method (simplified).

    Args:
        depth: Depth in ft
        pore_pressure_gradient: Pore pressure gradient (psi/ft)

    Returns:
        dict with fracture gradient
    """
    # Simplified empirical relationship
    # Shallow depths: higher gradient
    # Deep depths: approaches overburden

    if depth < 5000:
        frac_gradient = 0.85 + (0.1 * (depth / 5000))
    elif depth < 10000:
        frac_gradient = 0.95 + (0.05 * ((depth - 5000) / 5000))
    else:
        frac_gradient = 1.0

    frac_pressure = frac_gradient * depth
    emw = frac_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT

    return {
        "fracture_gradient_psi_ft": round(frac_gradient, 4),
        "fracture_pressure_psi": round(frac_pressure, 0),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "method": "Hubert & Willis (Empirical)",
    }


def leak_off_test_analysis(
    test_pressure: float,
    depth: float,
    mud_weight: float,
) -> dict:
    """Analyze leak-off test (LOT) results.

    Args:
        test_pressure: Surface test pressure (psi)
        depth: Test depth (ft)
        mud_weight: Mud weight during test (ppg)

    Returns:
        dict with fracture gradient from LOT
    """
    # Total pressure at shoe
    hydrostatic = PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * depth
    total_pressure = hydrostatic + test_pressure

    # Fracture gradient
    frac_gradient = total_pressure / depth
    emw = frac_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT

    return {
        "lot_surface_pressure_psi": test_pressure,
        "hydrostatic_pressure_psi": round(hydrostatic, 0),
        "total_pressure_at_shoe_psi": round(total_pressure, 0),
        "fracture_gradient_psi_ft": round(frac_gradient, 4),
        "equivalent_mud_weight_ppg": round(emw, 2),
        "test_depth_ft": depth,
        "method": "Leak-Off Test",
    }


# ---------------------------------------------------------------------------
# Abnormal pressure detection
# ---------------------------------------------------------------------------


def classify_pressure_regime(
    pore_pressure_gradient: float,
    normal_gradient: float = 0.465,
) -> dict:
    """Classify pressure regime.

    Args:
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        normal_gradient: Normal pressure gradient (psi/ft)

    Returns:
        dict with pressure classification
    """
    ratio = pore_pressure_gradient / normal_gradient

    if ratio < 0.9:
        regime = "Subnormal (Underpressured)"
        description = "Pressure below hydrostatic"
    elif ratio < 1.1:
        regime = "Normal Pressure"
        description = "Hydrostatic pressure"
    elif ratio < 1.3:
        regime = "Slightly Abnormal"
        description = "Mild overpressure"
    elif ratio < 1.6:
        regime = "Moderately Abnormal"
        description = "Significant overpressure"
    else:
        regime = "Highly Abnormal"
        description = "Severe overpressure"

    emw = pore_pressure_gradient / 0.052

    return {
        "regime": regime,
        "description": description,
        "gradient_ratio": round(ratio, 2),
        "pore_pressure_gradient_psi_ft": round(pore_pressure_gradient, 4),
        "equivalent_mud_weight_ppg": round(emw, 2),
    }


def pressure_transition_zone(
    depths: list[float],
    gradients: list[float],
    threshold: float = 0.05,
) -> dict:
    """Identify pressure transition zones.

    Args:
        depths: Depth points (ft)
        gradients: Pressure gradients at each depth (psi/ft)
        threshold: Gradient change threshold for detection

    Returns:
        dict with transition zones
    """
    depths_arr = np.array(depths)
    gradients_arr = np.array(gradients)

    # Calculate gradient changes
    gradient_changes = np.diff(gradients_arr)

    # Find significant transitions
    transitions = []
    for i in range(len(gradient_changes)):
        if abs(gradient_changes[i]) > threshold:
            transitions.append(
                {
                    "depth_ft": depths_arr[i + 1],
                    "gradient_change_psi_ft": round(gradient_changes[i], 4),
                    "type": "Increase" if gradient_changes[i] > 0 else "Decrease",
                }
            )

    return {
        "transition_zones": transitions,
        "num_transitions": len(transitions),
        "max_change": (
            round(max(abs(gradient_changes)), 4) if len(gradient_changes) > 0 else 0
        ),
    }


def drilling_window_analysis(
    depth: float,
    pore_pressure_gradient: float,
    fracture_gradient: float,
    mud_weight: float,
    safety_margin: float = 0.5,
) -> dict:
    """Analyze drilling window.

    Args:
        depth: Depth (ft)
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        fracture_gradient: Fracture gradient (psi/ft)
        mud_weight: Current mud weight (ppg)
        safety_margin: Safety margin (ppg)

    Returns:
        dict with drilling window analysis
    """
    # Convert to EMW
    pore_emw = pore_pressure_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT
    frac_emw = fracture_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT

    # Required MW with safety margin
    required_mw = pore_emw + safety_margin
    max_mw = frac_emw - 0.2  # Leave margin below fracture

    # Drilling window
    window = max_mw - required_mw

    # Check current mud weight
    if mud_weight < required_mw:
        status = "UNDERBALANCED - Increase mud weight"
        risk = "KICK RISK"
    elif mud_weight > max_mw:
        status = "OVERBALANCED - Risk of losses"
        risk = "LOSSES RISK"
    else:
        status = "Within drilling window"
        risk = "ACCEPTABLE"

    # Window classification
    if window < 1.0:
        window_type = "Narrow window - Tight margin"
    elif window < 2.0:
        window_type = "Moderate window"
    else:
        window_type = "Wide window - Good margin"

    return {
        "depth_ft": depth,
        "pore_pressure_emw_ppg": round(pore_emw, 2),
        "fracture_gradient_emw_ppg": round(frac_emw, 2),
        "required_mud_weight_ppg": round(required_mw, 2),
        "maximum_mud_weight_ppg": round(max_mw, 2),
        "drilling_window_ppg": round(window, 2),
        "current_mud_weight_ppg": mud_weight,
        "status": status,
        "risk": risk,
        "window_classification": window_type,
    }


# ---------------------------------------------------------------------------
# Geopressure analysis
# ---------------------------------------------------------------------------


def multi_method_prediction(
    depth: float,
    sonic_dt: Optional[float] = None,
    normal_sonic: Optional[float] = None,
    resistivity: Optional[float] = None,
    normal_resistivity: Optional[float] = None,
    d_exponent: Optional[float] = None,
    normal_d_exp: Optional[float] = None,
    overburden_gradient: float = 1.0,
) -> dict:
    """Predict pore pressure using multiple methods.

    Args:
        depth: Depth (ft)
        sonic_dt: Observed sonic transit time
        normal_sonic: Normal trend sonic
        resistivity: Observed resistivity
        normal_resistivity: Normal trend resistivity
        d_exponent: Observed d-exponent
        normal_d_exp: Normal d-exponent
        overburden_gradient: Overburden gradient (psi/ft)

    Returns:
        dict with predictions from available methods
    """
    predictions = {}

    # Sonic method
    if sonic_dt and normal_sonic:
        sonic_pred = sonic_method(sonic_dt, normal_sonic, overburden_gradient)
        predictions["sonic"] = sonic_pred

    # Resistivity method
    if resistivity and normal_resistivity:
        res_pred = resistivity_method(
            resistivity, normal_resistivity, overburden_gradient
        )
        predictions["resistivity"] = res_pred

    # Average prediction if multiple methods available
    if predictions:
        gradients = [
            p["pore_pressure_gradient_psi_ft"] for p in predictions.values()
        ]
        avg_gradient = np.mean(gradients)
        avg_emw = avg_gradient / 0.052

        return {
            "depth_ft": depth,
            "individual_predictions": predictions,
            "average_gradient_psi_ft": round(avg_gradient, 4),
            "average_emw_ppg": round(avg_emw, 2),
            "methods_used": list(predictions.keys()),
        }
    else:
        return {
            "depth_ft": depth,
            "message": "Insufficient data for prediction",
            "individual_predictions": {},
        }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def complete_pressure_analysis(
    depth: float,
    mud_weight: float,
    overburden_gradient: float,
    sonic_dt: Optional[float] = None,
    normal_sonic: Optional[float] = None,
    water_depth: float = 0.0,
) -> dict:
    """Complete pressure analysis at a given depth.

    Args:
        depth: Depth (ft)
        mud_weight: Current mud weight (ppg)
        overburden_gradient: Overburden gradient (psi/ft)
        sonic_dt: Observed sonic transit time (optional)
        normal_sonic: Normal trend sonic (optional)
        water_depth: Water depth for offshore (ft)

    Returns:
        dict with complete pressure analysis
    """
    # Overburden
    ob_analysis = estimate_overburden_from_depth(depth, water_depth)

    # Pore pressure prediction
    pore_pred = None
    if sonic_dt and normal_sonic:
        pore_pred = sonic_method(sonic_dt, normal_sonic, overburden_gradient)

    # Assume normal pressure if no prediction available
    pore_gradient = (
        pore_pred["pore_pressure_gradient_psi_ft"]
        if pore_pred
        else PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD
    )

    # Fracture gradient
    frac_pred = matthews_kelly_method(depth, overburden_gradient, pore_gradient)

    # Pressure classification
    pressure_class = classify_pressure_regime(pore_gradient)

    # Drilling window
    drilling_window = drilling_window_analysis(
        depth, pore_gradient, frac_pred["fracture_gradient_psi_ft"], mud_weight
    )

    return {
        "depth_ft": depth,
        "overburden": ob_analysis,
        "pore_pressure": (
            pore_pred
            if pore_pred
            else {
                "estimate": "normal",
                "gradient_psi_ft": PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
            }
        ),
        "fracture_gradient": frac_pred,
        "pressure_classification": pressure_class,
        "drilling_window": drilling_window,
    }
