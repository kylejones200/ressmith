"""Subsea and deepwater drilling operations calculations.

This module implements subsea drilling calculations including:
- Riser analysis and design
- BOP stack calculations
- Subsea wellhead loads
- Deepwater mud weight management
- Dual gradient drilling
- Managed pressure drilling (MPD)
- Riser gas handling
- Kick tolerance in deepwater
- Temperature and pressure effects in deep water

Based on deepwater and subsea drilling engineering principles.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ressmith.primitives.constants import PhysicalConstants

logger = logging.getLogger(__name__)


@dataclass
class RiserConfiguration:
    """Riser configuration data."""

    water_depth: float  # Water depth in ft
    riser_id: float  # Riser inner diameter in inches
    riser_od: float  # Riser outer diameter in inches
    riser_weight: float  # Riser weight in lb/ft
    mud_weight: float  # Mud weight in ppg
    seawater_gradient: float  # Seawater gradient in psi/ft (typically 0.465)


@dataclass
class BOPStack:
    """BOP stack specifications."""

    working_pressure: float  # Working pressure rating in psi
    test_pressure: float  # Test pressure in psi
    height: float  # Stack height in ft
    weight: float  # Dry weight in lbs
    buoyancy_factor: float  # Buoyancy factor in seawater


@dataclass
class SubseaWellhead:
    """Subsea wellhead data."""

    conductor_size: float  # Conductor size in inches
    surface_casing_size: float  # Surface casing size in inches
    wellhead_pressure_rating: float  # Pressure rating in psi
    mudline_depth: float  # Mudline depth in ft


# ---------------------------------------------------------------------------
# Riser analysis
# ---------------------------------------------------------------------------


def riser_tension(
    riser_weight_air: float,
    riser_length: float,
    mud_weight: float,
    riser_id: float,
    riser_od: float,
    top_tension_factor: float = 1.3,
) -> dict:
    """Calculate riser tension requirements.

    Args:
        riser_weight_air: Riser weight in air (lb/ft)
        riser_length: Riser length (ft)
        mud_weight: Mud weight (ppg)
        riser_id: Riser ID (inches)
        riser_od: Riser OD (inches)
        top_tension_factor: Safety factor for top tension

    Returns:
        dict with riser tension analysis
    """
    total_weight_air = riser_weight_air * riser_length

    steel_volume = (
        (math.pi / 4)
        * ((riser_od / 12) ** 2 - (riser_id / 12) ** 2)
        * riser_length
    )

    buoyancy = steel_volume * 64.0

    mud_volume = (math.pi / 4) * (riser_id / 12) ** 2 * riser_length
    mud_weight_lbs = (
        mud_volume * mud_weight * 7.48 * PhysicalConstants.WATER_DENSITY_LB_GAL
    )

    effective_weight = total_weight_air - buoyancy + mud_weight_lbs
    top_tension = effective_weight * top_tension_factor

    return {
        "riser_weight_air_lbs": round(total_weight_air, 0),
        "buoyancy_force_lbs": round(buoyancy, 0),
        "mud_weight_lbs": round(mud_weight_lbs, 0),
        "effective_weight_lbs": round(effective_weight, 0),
        "required_top_tension_lbs": round(top_tension, 0),
        "required_top_tension_kips": round(top_tension / 1000, 1),
        "safety_factor": top_tension_factor,
    }


def riser_recoil(
    stored_tension: float,
    riser_weight_per_ft: float,
    water_depth: float,
) -> dict:
    """Calculate riser recoil after disconnect.

    Args:
        stored_tension: Stored tension in riser (lbs)
        riser_weight_per_ft: Effective riser weight (lb/ft)
        water_depth: Water depth (ft)

    Returns:
        dict with recoil analysis
    """
    if riser_weight_per_ft <= 0:
        return {"error": "Invalid riser weight"}

    recoil_height = stored_tension / riser_weight_per_ft

    g = 32.2  # ft/s²
    velocity = math.sqrt(2 * g * recoil_height)

    time_to_surface = water_depth / velocity if velocity > 0 else 0

    return {
        "recoil_height_ft": round(recoil_height, 1),
        "recoil_velocity_fps": round(velocity, 1),
        "time_to_surface_sec": round(time_to_surface, 1),
        "stored_tension_lbs": round(stored_tension, 0),
    }


def riser_pressure_integrity(
    water_depth: float,
    mud_weight: float,
    formation_pressure_gradient: float,
    shoe_depth: float,
) -> dict:
    """Analyze riser pressure integrity.

    Args:
        water_depth: Water depth (ft)
        mud_weight: Mud weight (ppg)
        formation_pressure_gradient: Formation gradient (psi/ft)
        shoe_depth: Casing shoe depth below mudline (ft)

    Returns:
        dict with pressure analysis
    """
    seawater_pressure = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD * water_depth

    shoe_depth_total = water_depth + shoe_depth
    mud_pressure_at_shoe = (
        PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * shoe_depth_total
    )

    formation_pressure = formation_pressure_gradient * shoe_depth_total

    overbalance = mud_pressure_at_shoe - formation_pressure
    riser_margin = mud_pressure_at_shoe - seawater_pressure

    return {
        "water_depth_ft": water_depth,
        "seawater_pressure_psi": round(seawater_pressure, 0),
        "mud_pressure_at_shoe_psi": round(mud_pressure_at_shoe, 0),
        "formation_pressure_psi": round(formation_pressure, 0),
        "overbalance_psi": round(overbalance, 0),
        "riser_margin_psi": round(riser_margin, 0),
    }


# ---------------------------------------------------------------------------
# Deepwater mud management
# ---------------------------------------------------------------------------


def seawater_riser_effect(
    target_mud_weight: float,
    water_depth: float,
    total_depth: float,
) -> dict:
    """Calculate effect of seawater in riser.

    Args:
        target_mud_weight: Target equivalent mud weight at target (ppg)
        water_depth: Water depth (ft)
        total_depth: Total depth including water (ft)

    Returns:
        dict with mud weight requirements
    """
    sw_gradient = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD

    target_pressure = (
        target_mud_weight * PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )

    sw_pressure = sw_gradient * water_depth
    remaining_pressure = target_pressure - sw_pressure

    depth_below_mudline = total_depth - water_depth

    if depth_below_mudline <= 0:
        return {"error": "Invalid depth configuration"}

    required_mw = remaining_pressure / (0.052 * depth_below_mudline)
    mw_difference = required_mw - target_mud_weight

    return {
        "target_emw_ppg": target_mud_weight,
        "required_mud_weight_ppg": round(required_mw, 2),
        "mud_weight_increase_ppg": round(mw_difference, 2),
        "water_depth_ft": water_depth,
        "seawater_contribution_psi": round(sw_pressure, 0),
        "mud_contribution_psi": round(remaining_pressure, 0),
    }


def dual_gradient_drilling(
    water_depth: float,
    shoe_depth: float,
    formation_gradient: float,
    fracture_gradient: float,
    seawater_gradient: float = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD,
) -> dict:
    """Analyze dual gradient drilling system.

    Args:
        water_depth: Water depth (ft)
        shoe_depth: Shoe depth below mudline (ft)
        formation_gradient: Pore pressure gradient (psi/ft)
        fracture_gradient: Fracture gradient (psi/ft)
        seawater_gradient: Seawater gradient (psi/ft)

    Returns:
        dict with dual gradient analysis
    """
    total_depth = water_depth + shoe_depth

    formation_pressure = formation_gradient * total_depth
    fracture_pressure = fracture_gradient * total_depth

    conv_mud_weight = (
        formation_gradient / PhysicalConstants.HYDROSTATIC_GRADIENT + 0.5
    )

    mudline_pressure = seawater_gradient * water_depth

    required_pressure_below = (
        formation_pressure
        + 0.5 * PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )
    dg_mud_weight = (
        required_pressure_below - mudline_pressure
    ) / (PhysicalConstants.HYDROSTATIC_GRADIENT * shoe_depth)

    conv_pressure = (
        formation_gradient * PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )
    conv_window = (fracture_pressure - conv_pressure) / (
        PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )

    dg_pressure = (
        mudline_pressure
        + dg_mud_weight * PhysicalConstants.HYDROSTATIC_GRADIENT * shoe_depth
    )
    dg_window = (fracture_pressure - dg_pressure) / (
        PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )

    return {
        "water_depth_ft": water_depth,
        "shoe_depth_ft": shoe_depth,
        "conventional": {
            "required_mud_weight_ppg": round(conv_mud_weight, 2),
            "pressure_at_shoe_psi": round(conv_pressure, 0),
            "drilling_window_ppg": round(conv_window, 2),
        },
        "dual_gradient": {
            "required_mud_weight_ppg": round(dg_mud_weight, 2),
            "pressure_at_shoe_psi": round(dg_pressure, 0),
            "drilling_window_ppg": round(dg_window, 2),
            "mud_weight_reduction_ppg": round(conv_mud_weight - dg_mud_weight, 2),
        },
        "benefit_ppg": round(dg_window - conv_window, 2),
    }


def temperature_effects(
    surface_temp: float,
    bottom_temp: float,
    water_depth: float,
    total_depth: float,
    mud_weight_surface: float,
) -> dict:
    """Calculate temperature effects on mud weight.

    Args:
        surface_temp: Surface temperature (°F)
        bottom_temp: Bottom hole temperature (°F)
        water_depth: Water depth (ft)
        total_depth: Total depth (ft)
        mud_weight_surface: Mud weight at surface temp (ppg)

    Returns:
        dict with temperature effects
    """
    if total_depth > water_depth:
        temp_gradient = (bottom_temp - surface_temp) / (total_depth - water_depth)
    else:
        temp_gradient = 0

    expansion_coef = 0.00025
    temp_diff = bottom_temp - surface_temp

    mw_reduction = mud_weight_surface * expansion_coef * temp_diff
    mw_bottomhole = mud_weight_surface - mw_reduction

    pressure_loss = (
        PhysicalConstants.HYDROSTATIC_GRADIENT
        * mw_reduction
        * (total_depth - water_depth)
    )

    return {
        "surface_temperature_f": surface_temp,
        "bottomhole_temperature_f": bottom_temp,
        "temperature_gradient_f_ft": round(temp_gradient, 4),
        "mud_weight_surface_ppg": mud_weight_surface,
        "mud_weight_bottomhole_ppg": round(mw_bottomhole, 2),
        "mud_weight_reduction_ppg": round(mw_reduction, 2),
        "pressure_loss_psi": round(pressure_loss, 1),
    }


# ---------------------------------------------------------------------------
# Kick tolerance (deepwater)
# ---------------------------------------------------------------------------


def maximum_kick_tolerance(
    water_depth: float,
    shoe_depth: float,
    fracture_gradient: float,
    mud_weight: float,
    gas_gradient: float = 0.1,
) -> dict:
    """Calculate maximum allowable kick size.

    Args:
        water_depth: Water depth (ft)
        shoe_depth: Shoe depth below mudline (ft)
        fracture_gradient: Fracture gradient at shoe (psi/ft)
        mud_weight: Current mud weight (ppg)
        gas_gradient: Gas gradient (psi/ft)

    Returns:
        dict with kick tolerance analysis
    """
    total_shoe_depth = water_depth + shoe_depth

    fracture_pressure = fracture_gradient * total_shoe_depth

    seawater_pressure = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD * water_depth
    mud_pressure = PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * shoe_depth
    current_pressure = seawater_pressure + mud_pressure

    pressure_margin = fracture_pressure - current_pressure

    gradient_diff = (
        PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight - gas_gradient
    )

    if gradient_diff <= 0:
        return {"error": "Invalid gradient configuration"}

    max_kick_height = pressure_margin / gradient_diff

    hole_diameter = 8.5
    pipe_diameter = 5.0
    annular_capacity = (hole_diameter ** 2 - pipe_diameter ** 2) / 1029.4

    max_kick_volume = max_kick_height * annular_capacity
    pit_gain = max_kick_volume

    if pit_gain < 10:
        classification = "Very limited kick tolerance"
    elif pit_gain < 25:
        classification = "Limited kick tolerance"
    elif pit_gain < 50:
        classification = "Moderate kick tolerance"
    else:
        classification = "Good kick tolerance"

    return {
        "water_depth_ft": water_depth,
        "fracture_pressure_at_shoe_psi": round(fracture_pressure, 0),
        "current_pressure_at_shoe_psi": round(current_pressure, 0),
        "pressure_margin_psi": round(pressure_margin, 0),
        "max_kick_height_ft": round(max_kick_height, 1),
        "max_kick_volume_bbls": round(max_kick_volume, 1),
        "pit_gain_bbls": round(pit_gain, 1),
        "classification": classification,
    }


def kick_margin_analysis(
    water_depth: float,
    shoe_depth: float,
    mud_weight: float,
    pore_pressure_gradient: float,
    fracture_gradient: float,
) -> dict:
    """Complete kick margin analysis.

    Args:
        water_depth: Water depth (ft)
        shoe_depth: Shoe depth below mudline (ft)
        mud_weight: Mud weight (ppg)
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        fracture_gradient: Fracture gradient (psi/ft)

    Returns:
        dict with complete kick margin analysis
    """
    total_depth = water_depth + shoe_depth

    seawater_pressure = PhysicalConstants.NORMAL_PORE_PRESSURE_GRAD * water_depth
    mud_pressure = PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * shoe_depth
    current_hydrostatic = seawater_pressure + mud_pressure

    pore_pressure = pore_pressure_gradient * total_depth
    fracture_pressure = fracture_gradient * total_depth

    overbalance = current_hydrostatic - pore_pressure
    fracture_margin = fracture_pressure - current_hydrostatic

    drilling_window = (fracture_margin + overbalance) / (
        PhysicalConstants.HYDROSTATIC_GRADIENT * total_depth
    )

    if fracture_margin < 200:
        status = "CRITICAL - Very narrow window"
    elif fracture_margin < 500:
        status = "LIMITED - Narrow window"
    else:
        status = "ACCEPTABLE - Adequate margin"

    return {
        "water_depth_ft": water_depth,
        "pore_pressure_psi": round(pore_pressure, 0),
        "current_hydrostatic_psi": round(current_hydrostatic, 0),
        "fracture_pressure_psi": round(fracture_pressure, 0),
        "overbalance_psi": round(overbalance, 0),
        "fracture_margin_psi": round(fracture_margin, 0),
        "drilling_window_ppg": round(drilling_window, 2),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Riser gas handling
# ---------------------------------------------------------------------------


def gas_rise_velocity(
    mud_weight: float,
    mud_viscosity: float,
    gas_bubble_diameter: float = 0.5,
) -> dict:
    """Calculate gas rise velocity in riser.

    Args:
        mud_weight: Mud weight (ppg)
        mud_viscosity: Mud viscosity (cp)
        gas_bubble_diameter: Bubble diameter (inches)

    Returns:
        dict with gas rise velocity
    """
    rho_mud = mud_weight * 7.48 * 0.00194  # slugs/ft³
    rho_gas = 0.00012  # slugs/ft³ (approximate)
    mu = mud_viscosity * 0.000672  # lbf·s/ft²
    d = gas_bubble_diameter / 12  # ft
    g = PhysicalConstants.GRAVITY_CONSTANT  # ft/s²

    r = d / 2

    if mu > 0:
        velocity = (2 / 9) * (rho_mud - rho_gas) * g * r ** 2 / (mu / (rho_mud))
    else:
        velocity = 0

    velocity_ft_hr = velocity * 3600
    velocity_ft_hr = max(100, min(velocity_ft_hr, 1000))

    return {
        "rise_velocity_fps": round(velocity, 2),
        "rise_velocity_ft_hr": round(velocity_ft_hr, 0),
        "bubble_diameter_in": gas_bubble_diameter,
        "mud_weight_ppg": mud_weight,
    }


def gas_unloading_time(
    water_depth: float,
    gas_rise_velocity: float,
    riser_volume: float,
    circulation_rate: float,
) -> dict:
    """Calculate time for gas to exit riser.

    Args:
        water_depth: Water depth (ft)
        gas_rise_velocity: Gas rise velocity (ft/hr)
        riser_volume: Riser volume (bbls)
        circulation_rate: Circulation rate (gpm)

    Returns:
        dict with gas unloading analysis
    """
    if gas_rise_velocity <= 0:
        time_rise_no_circ = 999
    else:
        time_rise_no_circ = water_depth / gas_rise_velocity  # hours

    if circulation_rate <= 0:
        time_circulate = 999
    else:
        time_circulate = riser_volume / (
            circulation_rate * PhysicalConstants.MINUTES_PER_HOUR
        )

    effective_velocity = gas_rise_velocity + (
        circulation_rate
        * 60
        * PhysicalConstants.GALLONS_PER_BBL
        / riser_volume
        * water_depth
    )

    if effective_velocity <= 0:
        time_combined = time_rise_no_circ
    else:
        time_combined = water_depth / effective_velocity

    time_actual = min(time_circulate, time_combined)

    return {
        "water_depth_ft": water_depth,
        "time_to_surface_no_circ_hr": round(time_rise_no_circ, 2),
        "time_to_circulate_riser_hr": round(time_circulate, 2),
        "actual_unloading_time_hr": round(time_actual, 2),
        "actual_unloading_time_min": round(
            time_actual * PhysicalConstants.MINUTES_PER_HOUR, 1
        ),
    }


# ---------------------------------------------------------------------------
# Managed pressure drilling
# ---------------------------------------------------------------------------


def surface_backpressure_required(
    target_bottomhole_pressure: float,
    mud_weight: float,
    tvd: float,
    friction_pressure: float,
) -> dict:
    """Calculate required surface backpressure for MPD.

    Args:
        target_bottomhole_pressure: Target BHP (psi)
        mud_weight: Mud weight (ppg)
        tvd: True vertical depth (ft)
        friction_pressure: Annular friction pressure (psi)

    Returns:
        dict with backpressure requirements
    """
    hydrostatic = PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * tvd

    surface_bp = target_bottomhole_pressure - hydrostatic + friction_pressure

    esd = target_bottomhole_pressure / (
        PhysicalConstants.HYDROSTATIC_GRADIENT * tvd
    )

    return {
        "target_bhp_psi": target_bottomhole_pressure,
        "hydrostatic_pressure_psi": round(hydrostatic, 0),
        "friction_pressure_psi": friction_pressure,
        "required_surface_backpressure_psi": round(surface_bp, 0),
        "equivalent_static_density_ppg": round(esd, 2),
        "mud_weight_ppg": mud_weight,
    }


def mpd_pressure_profile(
    mud_weight: float,
    depths: list[float],
    friction_losses: list[float],
    surface_backpressure: float,
) -> dict:
    """Calculate pressure profile for MPD operations.

    Args:
        mud_weight: Mud weight (ppg)
        depths: Depth points (ft)
        friction_losses: Friction losses at each depth (psi)
        surface_backpressure: Applied surface backpressure (psi)

    Returns:
        dict with pressure profile
    """
    depths_arr = np.array(depths)
    friction_arr = np.array(friction_losses)

    hydrostatic = PhysicalConstants.HYDROSTATIC_GRADIENT * mud_weight * depths_arr

    total_pressure = hydrostatic + surface_backpressure - friction_arr

    ecd = total_pressure / (PhysicalConstants.HYDROSTATIC_GRADIENT * depths_arr)

    return {
        "depths_ft": depths_arr.tolist(),
        "hydrostatic_psi": hydrostatic.tolist(),
        "friction_losses_psi": friction_arr.tolist(),
        "total_pressure_psi": total_pressure.tolist(),
        "ecd_ppg": ecd.tolist(),
        "surface_backpressure_psi": surface_backpressure,
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def subsea_well_design_analysis(
    water_depth: float,
    shoe_depth: float,
    mud_weight: float,
    pore_pressure_gradient: float,
    fracture_gradient: float,
    formation_temp: float,
) -> dict:
    """Complete subsea well design analysis.

    Args:
        water_depth: Water depth (ft)
        shoe_depth: Shoe depth below mudline (ft)
        mud_weight: Mud weight (ppg)
        pore_pressure_gradient: Pore pressure gradient (psi/ft)
        fracture_gradient: Fracture gradient (psi/ft)
        formation_temp: Formation temperature (°F)

    Returns:
        dict with complete subsea analysis
    """
    riser_analysis = riser_pressure_integrity(
        water_depth, mud_weight, pore_pressure_gradient, shoe_depth
    )

    mud_effect = seawater_riser_effect(
        mud_weight, water_depth, water_depth + shoe_depth
    )

    kick_tol = maximum_kick_tolerance(
        water_depth, shoe_depth, fracture_gradient, mud_weight
    )

    kick_margin = kick_margin_analysis(
        water_depth,
        shoe_depth,
        mud_weight,
        pore_pressure_gradient,
        fracture_gradient,
    )

    dual_gradient = dual_gradient_drilling(
        water_depth, shoe_depth, pore_pressure_gradient, fracture_gradient
    )

    temp_effects = temperature_effects(
        40, formation_temp, water_depth, water_depth + shoe_depth, mud_weight
    )

    return {
        "water_depth_ft": water_depth,
        "shoe_depth_ft": shoe_depth,
        "mud_weight_ppg": mud_weight,
        "riser_integrity": riser_analysis,
        "mud_weight_management": mud_effect,
        "kick_tolerance": kick_tol,
        "kick_margins": kick_margin,
        "dual_gradient_analysis": dual_gradient,
        "temperature_effects": temp_effects,
    }
