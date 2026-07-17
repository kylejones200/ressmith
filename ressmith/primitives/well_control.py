"""Well control calculations and kick detection (WELCON model).

This module implements well control calculations including:
- Kick detection and analysis
- Kill procedures (Driller's Method, Wait & Weight, Concurrent)
- Choke pressure calculations
- Gas migration modeling
- Wellbore pressure profiles during kicks
- BOP response and control procedures

Based on industry-standard well control principles.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ressmith.primitives.constants import PhysicalConstants

logger = logging.getLogger(__name__)


@dataclass
class KickData:
    """Container for kick data."""

    pit_gain: float  # Pit gain in bbls
    shut_in_drillpipe_pressure: float  # SIDPP in psi
    shut_in_casing_pressure: float  # SICP in psi
    formation_pressure: float  # Formation pressure in psi
    kick_intensity: float  # Kick intensity in ppg
    kick_height: float  # Height of kick in wellbore (ft)
    kick_type: str  # 'gas', 'oil', or 'water'


@dataclass
class WellGeometry:
    """Wellbore geometry data."""

    measured_depth: float  # MD in ft
    true_vertical_depth: float  # TVD in ft
    hole_diameter: float  # inches
    drillpipe_od: float  # Drillpipe OD in inches
    drillpipe_id: float  # Drillpipe ID in inches
    drillcollar_od: float  # Drill collar OD in inches
    drillcollar_id: float  # Drill collar ID in inches
    drillcollar_length: float  # Drill collar length in ft
    casing_id: float  # Casing ID in inches


@dataclass
class MudProperties:
    """Drilling mud properties."""

    weight: float  # Mud weight in ppg
    plastic_viscosity: float  # PV in cp
    yield_point: float  # YP in lb/100ft²
    funnel_viscosity: float  # Funnel viscosity in sec/qt
    gel_strength_10sec: float  # 10-second gel strength in lb/100ft²
    gel_strength_10min: float  # 10-minute gel strength in lb/100ft²


# ---------------------------------------------------------------------------
# Kick detection
# ---------------------------------------------------------------------------


def detect_kick(
    pit_volume_gain: float = 0.0,
    flow_rate_increase: float = 0.0,
    pump_rate_decrease: bool = False,
    drilling_break: bool = False,
    connection_flow: bool = False,
) -> dict:
    """Detect potential kick based on indicators.

    Args:
        pit_volume_gain: Pit gain in bbls
        flow_rate_increase: Flow rate increase percentage
        pump_rate_decrease: Pump rate decreased but flow increased
        drilling_break: Sudden increase in ROP
        connection_flow: Flow observed during connection

    Returns:
        dict with kick detection results
    """
    indicators: list[str] = []
    severity = 0

    if pit_volume_gain > 5.0:
        indicators.append("significant_pit_gain")
        severity += 3
    elif pit_volume_gain > 2.0:
        indicators.append("moderate_pit_gain")
        severity += 2
    elif pit_volume_gain > 0.5:
        indicators.append("minor_pit_gain")
        severity += 1

    if flow_rate_increase > 10.0:
        indicators.append("flow_rate_increase")
        severity += 2

    if pump_rate_decrease:
        indicators.append("pump_rate_flow_anomaly")
        severity += 2

    if drilling_break:
        indicators.append("drilling_break")
        severity += 1

    if connection_flow:
        indicators.append("connection_flow")
        severity += 3

    if severity >= 5:
        kick_status = "KICK DETECTED - IMMEDIATE ACTION REQUIRED"
        action = "SHUT IN WELL"
    elif severity >= 3:
        kick_status = "POSSIBLE KICK - MONITOR CLOSELY"
        action = "INCREASE MONITORING, PREPARE TO SHUT IN"
    elif severity >= 1:
        kick_status = "ANOMALY DETECTED"
        action = "VERIFY INDICATORS, INCREASE MONITORING"
    else:
        kick_status = "NORMAL"
        action = "CONTINUE OPERATIONS"

    return {
        "status": kick_status,
        "severity": severity,
        "indicators": indicators,
        "recommended_action": action,
    }


def formation_pressure_from_sidpp(
    shut_in_drillpipe_pressure: float,
    mud_weight: float,
    tvd: float,
) -> float:
    """Calculate formation pressure from SIDPP.

    Args:
        shut_in_drillpipe_pressure: SIDPP in psi
        mud_weight: Mud weight in ppg
        tvd: True vertical depth in ft

    Returns:
        Formation pressure in psi
    """
    hydrostatic = 0.052 * mud_weight * tvd
    return hydrostatic + shut_in_drillpipe_pressure


def kick_intensity(formation_pressure: float, tvd: float) -> float:
    """Calculate kick intensity (equivalent mud weight).

    Args:
        formation_pressure: Formation pressure in psi
        tvd: True vertical depth in ft

    Returns:
        Kick intensity in ppg
    """
    if tvd <= 0:
        return 0.0

    return formation_pressure / (0.052 * tvd)


# ---------------------------------------------------------------------------
# Well control procedure helpers
# ---------------------------------------------------------------------------


def _annular_volume(geometry: WellGeometry) -> float:
    """Calculate annular volume in bbls."""
    annular_cap = (
        geometry.hole_diameter ** 2 - geometry.drillpipe_od ** 2
    ) / 1029.4
    return annular_cap * geometry.measured_depth


def _drillpipe_capacity(geometry: WellGeometry) -> float:
    """Calculate drillpipe capacity in bbls."""
    dp_cap = geometry.drillpipe_id ** 2 / 1029.4
    return dp_cap * geometry.measured_depth


def _annular_pressure_loss(
    geometry: WellGeometry,
    mud: MudProperties,
    mud_weight: float,
    flow_rate: float,
) -> float:
    """Calculate annular pressure loss in psi."""
    dh = geometry.hole_diameter - geometry.drillpipe_od

    annular_area = (
        geometry.hole_diameter ** 2 - geometry.drillpipe_od ** 2
    ) / 183.3
    velocity = flow_rate / annular_area

    apl = (
        mud.plastic_viscosity * velocity * geometry.measured_depth
    ) / (300 * dh)

    return apl


def _generate_pressure_schedule(
    icp: float,
    fcp: float,
    strokes: float,
) -> list[tuple[int, float]]:
    """Generate choke pressure schedule at 10-stroke intervals."""
    schedule: list[tuple[int, float]] = []
    num_points = int(strokes / 10) + 1

    for i in range(num_points):
        stroke_number = i * 10
        if stroke_number > strokes:
            stroke_number = strokes

        pressure = icp - (icp - fcp) * (stroke_number / strokes)
        schedule.append((int(stroke_number), round(pressure, 1)))

    return schedule


def _estimate_fracture_pressure(geometry: WellGeometry) -> float:
    """Estimate formation fracture pressure."""
    if geometry.true_vertical_depth < 5000:
        fracture_gradient = 0.8
    elif geometry.true_vertical_depth < 10000:
        fracture_gradient = 0.9
    else:
        fracture_gradient = 1.0

    return fracture_gradient * geometry.true_vertical_depth


def calculate_kill_mud_weight_from_kick(
    kick: KickData,
    safety_margin: float = 0.5,
) -> float:
    """Calculate required kill mud weight from kick intensity.

    Args:
        kick: Kick data
        safety_margin: Safety margin in ppg

    Returns:
        Kill mud weight in ppg
    """
    return kick.kick_intensity + safety_margin


def drillers_method(
    geometry: WellGeometry,
    mud: MudProperties,
    kick: KickData,
    pump_rate: float,
    pump_pressure: float,
) -> dict:
    """Driller's Method kill procedure.

    Two-circulation method:
    1. First circulation: circulate kick out with original mud
    2. Second circulation: circulate kill mud through system

    Args:
        geometry: Wellbore geometry
        mud: Current mud properties
        kick: Kick data
        pump_rate: Pump rate in gpm
        pump_pressure: Initial circulating pressure in psi

    Returns:
        dict with kill procedure parameters
    """
    kmw = calculate_kill_mud_weight_from_kick(kick)

    icp = kick.shut_in_drillpipe_pressure + pump_pressure
    fcp_first = pump_pressure

    apl_original = _annular_pressure_loss(geometry, mud, mud.weight, pump_rate)
    apl_kill = _annular_pressure_loss(geometry, mud, kmw, pump_rate)

    icp_second = pump_pressure + apl_kill - apl_original
    fcp_second = pump_pressure

    annular_volume = _annular_volume(geometry)
    strokes_to_bit = _drillpipe_capacity(geometry) / pump_rate
    time_to_bit = (
        strokes_to_bit / pump_rate * PhysicalConstants.MINUTES_PER_HOUR
    )

    return {
        "method": "Drillers Method",
        "kill_mud_weight": kmw,
        "first_circulation": {
            "icp": icp,
            "fcp": fcp_first,
            "mud_weight": mud.weight,
            "time_minutes": time_to_bit + annular_volume / pump_rate,
        },
        "second_circulation": {
            "icp": icp_second,
            "fcp": fcp_second,
            "mud_weight": kmw,
            "time_minutes": time_to_bit + annular_volume / pump_rate,
        },
        "total_time_minutes": 2 * (time_to_bit + annular_volume / pump_rate),
        "choke_pressure_schedule": _generate_pressure_schedule(
            icp, fcp_first, strokes_to_bit
        ),
    }


def wait_and_weight_method(
    geometry: WellGeometry,
    mud: MudProperties,
    kick: KickData,
    pump_rate: float,
    pump_pressure: float,
) -> dict:
    """Wait and Weight Method kill procedure.

    Single-circulation method where kill mud is pumped immediately after mixing.

    Args:
        geometry: Wellbore geometry
        mud: Current mud properties
        kick: Kick data
        pump_rate: Pump rate in gpm
        pump_pressure: Initial circulating pressure in psi

    Returns:
        dict with kill procedure parameters
    """
    kmw = calculate_kill_mud_weight_from_kick(kick)

    apl_original = _annular_pressure_loss(geometry, mud, mud.weight, pump_rate)
    apl_kill = _annular_pressure_loss(geometry, mud, kmw, pump_rate)

    pressure_increase = (
        (kmw - mud.weight)
        * PhysicalConstants.HYDROSTATIC_GRADIENT
        * geometry.true_vertical_depth
    )
    icp = kick.shut_in_drillpipe_pressure + pump_pressure + pressure_increase

    fcp = pump_pressure + apl_kill - apl_original

    drillpipe_capacity = _drillpipe_capacity(geometry)
    annular_volume = _annular_volume(geometry)
    strokes_to_bit = drillpipe_capacity / pump_rate

    return {
        "method": "Wait and Weight Method",
        "kill_mud_weight": kmw,
        "icp": icp,
        "fcp": fcp,
        "time_minutes": (drillpipe_capacity + annular_volume) / pump_rate,
        "strokes_to_bit": strokes_to_bit,
        "choke_pressure_schedule": _generate_pressure_schedule(
            icp, fcp, strokes_to_bit
        ),
    }


def concurrent_method(
    geometry: WellGeometry,
    mud: MudProperties,
    kick: KickData,
    pump_rate: float,
    pump_pressure: float,
) -> dict:
    """Concurrent Method (Bullheading).

    Used when unable to circulate normally.

    Args:
        geometry: Wellbore geometry
        mud: Current mud properties
        kick: Kick data
        pump_rate: Pump rate in gpm
        pump_pressure: Initial circulating pressure in psi

    Returns:
        dict with kill procedure parameters
    """
    kmw = calculate_kill_mud_weight_from_kick(kick)

    formation_fracture_pressure = _estimate_fracture_pressure(geometry)

    max_pressure = min(
        formation_fracture_pressure * 0.9,
        formation_fracture_pressure - 200,
    )

    required_pressure = kick.formation_pressure + pump_pressure

    return {
        "method": "Concurrent Method (Bullheading)",
        "kill_mud_weight": kmw,
        "required_pressure": required_pressure,
        "max_allowable_pressure": max_pressure,
        "feasible": required_pressure < max_pressure,
        "warning": "Use only when normal circulation is not possible",
    }


# ---------------------------------------------------------------------------
# Gas migration
# ---------------------------------------------------------------------------


def gas_migration_rate(
    gas_gradient: float = 0.1,
    mud_weight: float = 10.0,
    wellbore_diameter: float = 8.5,
) -> float:
    """Calculate gas migration rate.

    Args:
        gas_gradient: Gas gradient in psi/ft
        mud_weight: Mud weight in ppg
        wellbore_diameter: Wellbore diameter in inches

    Returns:
        Migration rate in ft/hr
    """
    mud_gradient = 0.052 * mud_weight
    pressure_differential = mud_gradient - gas_gradient

    migration_rate = 100 + pressure_differential * 100

    return max(0, min(migration_rate, 1000))


def gas_migration_pressure_increase(
    initial_gas_depth: float,
    migration_distance: float,
    gas_gradient: float,
    mud_weight: float,
) -> float:
    """Calculate pressure increase due to gas migration.

    Args:
        initial_gas_depth: Initial depth of gas top (ft)
        migration_distance: Distance gas migrated (ft)
        gas_gradient: Gas gradient in psi/ft
        mud_weight: Mud weight in ppg

    Returns:
        Pressure increase in psi
    """
    mud_gradient = 0.052 * mud_weight
    return (mud_gradient - gas_gradient) * migration_distance


def gas_expansion(
    initial_pressure: float,
    initial_volume: float,
    final_pressure: float,
    temperature: float = 150.0,
    gas_gravity: float = 0.6,
) -> float:
    """Calculate gas expansion using real gas law.

    Args:
        initial_pressure: Initial pressure in psia
        initial_volume: Initial volume in bbls
        final_pressure: Final pressure in psia
        temperature: Temperature in °F
        gas_gravity: Gas specific gravity (air=1.0)

    Returns:
        Final volume in bbls
    """
    Z1 = 1.0
    Z2 = 1.0

    final_volume = initial_volume * (initial_pressure / final_pressure) * (Z2 / Z1)

    return final_volume


# ---------------------------------------------------------------------------
# Choke management
# ---------------------------------------------------------------------------


def choke_pressure(
    drillpipe_pressure: float,
    target_pressure: float,
    current_strokes: int,
    total_strokes: int,
    icp: float,
    fcp: float,
) -> float:
    """Calculate required choke pressure adjustment.

    Args:
        drillpipe_pressure: Current drillpipe pressure in psi
        target_pressure: Target drillpipe pressure from schedule
        current_strokes: Current pump strokes
        total_strokes: Total strokes to complete circulation
        icp: Initial circulating pressure in psi
        fcp: Final circulating pressure in psi

    Returns:
        Required choke adjustment in psi
    """
    expected_pressure = icp - (icp - fcp) * (current_strokes / total_strokes)
    adjustment = drillpipe_pressure - expected_pressure
    return adjustment


def maximum_allowable_annular_pressure(
    casing_pressure_test: float,
    mud_weight: float,
    tvd: float,
    safety_factor: float = 0.9,
) -> float:
    """Calculate maximum allowable annular surface pressure (MAASP).

    Args:
        casing_pressure_test: Casing pressure test value in psi
        mud_weight: Current mud weight in ppg
        tvd: True vertical depth in ft
        safety_factor: Safety factor (typically 0.8-0.9)

    Returns:
        MAASP in psi
    """
    hydrostatic = 0.052 * mud_weight * tvd
    maasp = (casing_pressure_test * safety_factor) - hydrostatic
    return max(0, maasp)


# ---------------------------------------------------------------------------
# Kick simulation
# ---------------------------------------------------------------------------


def _influx_rate(
    geometry: WellGeometry,
    pressure_diff: float,
    permeability: float,
) -> float:
    """Calculate gas influx rate using Darcy's law."""
    h = 10  # Assume 10 ft pay zone
    mu = 0.02  # Gas viscosity (cp)
    re = 1000  # Drainage radius (ft)
    rw = geometry.hole_diameter / (2 * 12)  # Convert to ft

    if re <= rw:
        return 0.0

    q = (2 * math.pi * permeability * h * pressure_diff) / (
        mu * math.log(re / rw)
    )

    q_cuft_min = q / 1440

    return max(0, q_cuft_min)


def simulate_gas_kick(
    geometry: WellGeometry,
    mud: MudProperties,
    formation_pressure: float,
    permeability: float,
    time_step: float = 1.0,
    duration: float = 10.0,
) -> dict:
    """Simulate gas kick influx.

    Args:
        geometry: Wellbore geometry
        mud: Mud properties
        formation_pressure: Formation pressure in psi
        permeability: Formation permeability in md
        time_step: Time step for simulation in minutes
        duration: Total simulation duration in minutes

    Returns:
        dict with simulation results
    """
    bhp = (
        PhysicalConstants.HYDROSTATIC_GRADIENT
        * mud.weight
        * geometry.true_vertical_depth
    )

    pressure_diff = formation_pressure - bhp

    if pressure_diff <= 0:
        return {
            "kick_occurred": False,
            "message": "No kick - mud weight sufficient",
        }

    time_points = np.arange(0, duration, time_step)
    influx_rate = _influx_rate(geometry, pressure_diff, permeability)

    cumulative_influx = []
    pit_gain = []

    for t in time_points:
        current_influx = influx_rate * np.exp(-0.1 * t)
        cum_influx = current_influx * t
        cumulative_influx.append(cum_influx)
        pit_gain.append(cum_influx * 5.615)  # Convert to bbls

    return {
        "kick_occurred": True,
        "time_minutes": time_points.tolist(),
        "influx_rate_cuft_min": influx_rate,
        "cumulative_influx_cuft": cumulative_influx,
        "pit_gain_bbls": pit_gain,
        "formation_pressure": formation_pressure,
        "bottomhole_pressure": bhp,
        "pressure_differential": pressure_diff,
    }


# ---------------------------------------------------------------------------
# Legacy convenience wrappers (former WellControlCalculations)
# ---------------------------------------------------------------------------


def calculate_kill_mud_weight(
    formation_pressure: float,
    tvd: float | None = None,
    *,
    true_vertical_depth: float | None = None,
    safety_margin: float = 0.5,
) -> float:
    """Calculate kill mud weight from formation pressure and TVD."""
    depth = true_vertical_depth if true_vertical_depth is not None else tvd
    if depth is None:
        raise ValueError("Must provide tvd or true_vertical_depth")
    intensity = kick_intensity(formation_pressure, depth)
    return intensity + safety_margin


def calculate_formation_pressure(
    sidpp: float | None = None,
    mud_weight: float | None = None,
    tvd: float | None = None,
    *,
    true_vertical_depth: float | None = None,
) -> float:
    """Calculate formation pressure from SIDPP, mud weight, and TVD."""
    depth = true_vertical_depth if true_vertical_depth is not None else tvd
    if sidpp is None or mud_weight is None or depth is None:
        raise ValueError(
            "Must provide sidpp, mud_weight, and tvd or true_vertical_depth"
        )
    return formation_pressure_from_sidpp(sidpp, mud_weight, depth)


def calculate_kick_volume(pit_gain: float) -> tuple[float, int]:
    """Return kick volume and severity from pit gain.

    Returns:
        (kick_volume_bbls, severity_score)
    """
    detection = detect_kick(pit_volume_gain=pit_gain)
    return pit_gain, detection["severity"]


def calculate_initial_circulating_pressure(
    slow_pump_rate_pressure: float,
    shut_in_drillpipe_pressure: float | None = None,
    *,
    kill_mud_weight: float | None = None,
    original_mud_weight: float | None = None,
) -> float:
    """Initial circulating pressure when opening choke to circulate kick out.

    ICP = SPP + SIDPP, or ICP = SPP * (KMW/OMW) if SIDPP not available.
    """
    if shut_in_drillpipe_pressure is not None:
        return slow_pump_rate_pressure + shut_in_drillpipe_pressure
    if (
        kill_mud_weight is not None
        and original_mud_weight is not None
        and original_mud_weight > 0
    ):
        return slow_pump_rate_pressure * (kill_mud_weight / original_mud_weight)
    return slow_pump_rate_pressure


def calculate_final_circulating_pressure(
    slow_pump_rate_pressure: float,
    kill_mud_weight: float | None = None,
    original_mud_weight: float | None = None,
) -> float:
    """Final circulating pressure when kill mud has filled the wellbore.

    FCP = SPP * (KMW/OMW)
    """
    if (
        kill_mud_weight is not None
        and original_mud_weight is not None
        and original_mud_weight > 0
    ):
        return slow_pump_rate_pressure * (kill_mud_weight / original_mud_weight)
    return slow_pump_rate_pressure


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def detect_and_analyze_kick(
    pit_gain: float,
    sidpp: float,
    sicp: float,
    mud_weight: float,
    tvd: float,
) -> dict:
    """Complete kick detection and analysis.

    Args:
        pit_gain: Pit volume gain in bbls
        sidpp: Shut-in drillpipe pressure in psi
        sicp: Shut-in casing pressure in psi
        mud_weight: Mud weight in ppg
        tvd: True vertical depth in ft

    Returns:
        dict with kick analysis results
    """
    detection = detect_kick(pit_volume_gain=pit_gain)

    fp = formation_pressure_from_sidpp(sidpp, mud_weight, tvd)
    intensity = kick_intensity(fp, tvd)

    pressure_ratio = sicp / max(sidpp, 1.0)
    if pressure_ratio > 2.0:
        kick_type = "gas"
    elif pressure_ratio > 1.3:
        kick_type = "oil"
    else:
        kick_type = "water"

    return {
        "detection": detection,
        "formation_pressure_psi": fp,
        "kick_intensity_ppg": intensity,
        "kick_type": kick_type,
        "pressure_ratio": pressure_ratio,
        "recommended_kmw": intensity + 0.5,
    }
