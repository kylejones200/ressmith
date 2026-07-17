"""Well test analysis (Pressure Transient Analysis - PTA).

This module provides well test analysis capabilities including:
- Permeability estimation from pressure buildup/drawdown
- Skin factor calculation
- Boundary detection
- Well test data interpretation

References:
- Horne, R.N., "Modern Well Test Analysis," 5th Ed., 2019.
- Lee, J., Rollins, J.B., and Spivey, J.P., "Pressure Transient Testing," SPE Textbook Series, 2003.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WellTestResult:
    """Container for well test analysis results.

    Attributes:
        permeability: Estimated permeability (md)
        skin: Skin factor (dimensionless)
        wellbore_storage: Wellbore storage coefficient (bbl/psi)
        boundary_distance: Distance to boundary (ft)
        boundary_type: Type of boundary ('no_flow', 'constant_pressure', 'unknown')
        reservoir_pressure: Estimated reservoir pressure (psi)
    """

    permeability: float
    skin: float
    wellbore_storage: float = 0.0
    boundary_distance: float | None = None
    boundary_type: str = "unknown"
    reservoir_pressure: float | None = None


def analyze_buildup_test(
    time: np.ndarray,
    pressure: np.ndarray,
    production_rate: float,
    production_time: float,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    total_compressibility: float = 1e-5,
    wellbore_radius: float = 0.25,
) -> WellTestResult:
    """Analyze pressure buildup test.

    Uses Horner plot method for buildup analysis.

    Args:
        time: Shut-in time (hours)
        pressure: Pressure during buildup (psi)
        production_rate: Production rate before shut-in (STB/day)
        production_time: Total production time before shut-in (hours)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        total_compressibility: Total compressibility (1/psi)
        wellbore_radius: Wellbore radius (ft)

    Returns:
        WellTestResult with permeability, skin, etc.

    Reference:
        Horne, R.N., "Modern Well Test Analysis," 5th Ed., 2019.

    Example:
        >>> time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
        >>> pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])
        >>> result = analyze_buildup_test(time, pressure, 1000, 720)
    """
    # Filter valid data
    valid_mask = (time > 0) & (pressure > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient data for buildup analysis")

    time_valid = time[valid_mask]
    pressure_valid = pressure[valid_mask]

    # Horner time: (tp + Δt) / Δt
    horner_time = (production_time + time_valid) / time_valid

    # Horner plot: pressure vs log(horner_time)
    log_horner = np.log(horner_time)

    # Fit straight line to middle time region (radial flow)
    # Use middle 50% of data
    mid_start = len(log_horner) // 4
    mid_end = 3 * len(log_horner) // 4

    if mid_end - mid_start < 3:
        mid_start = 0
        mid_end = len(log_horner)

    log_horner_mid = log_horner[mid_start:mid_end]
    pressure_mid = pressure_valid[mid_start:mid_end]

    # Fit line: p = m * log((tp + Δt)/Δt) + b
    coeffs = np.polyfit(log_horner_mid, pressure_mid, 1)
    slope = coeffs[0]  # m (psi/log cycle)
    intercept = coeffs[1]  # b

    # Calculate permeability from slope
    # k = (162.6 * q * μ * Bo) / (m * h)
    if abs(slope) > 0:
        permeability = (
            162.6
            * production_rate
            * oil_viscosity
            * formation_volume_factor
            / (abs(slope) * reservoir_thickness)
        )
    else:
        permeability = 0.0

    # Calculate skin factor
    # S = 1.151 * [(p1hr - pwf) / m - log(k / (φ * μ * ct * rw^2)) - 3.23]
    # Find pressure at 1 hour (or extrapolate)
    if len(time_valid) > 0:
        # Find closest to 1 hour
        idx_1hr = np.argmin(np.abs(time_valid - 1.0))
        p1hr = (
            pressure_valid[idx_1hr]
            if idx_1hr < len(pressure_valid)
            else pressure_valid[0]
        )

        # Initial flowing pressure (first point)
        pwf = pressure_valid[0]

        if abs(slope) > 0:
            log_term = np.log(
                permeability
                / (
                    porosity
                    * oil_viscosity
                    * total_compressibility
                    * wellbore_radius**2
                )
            )
            skin = 1.151 * ((p1hr - pwf) / abs(slope) - log_term - 3.23)
        else:
            skin = 0.0

        # Estimate reservoir pressure (extrapolate to infinite time)
        # p* = intercept (pressure at Horner time = 1, i.e., infinite shut-in time)
        reservoir_pressure = intercept
    else:
        skin = 0.0
        reservoir_pressure = None

    # Detect boundaries (simplified)
    boundary_distance, boundary_type = detect_boundaries(
        time_valid, pressure_valid, permeability, porosity, total_compressibility
    )

    return WellTestResult(
        permeability=max(0.001, min(1000.0, permeability)),
        skin=skin,
        boundary_distance=boundary_distance,
        boundary_type=boundary_type,
        reservoir_pressure=reservoir_pressure,
    )


def analyze_drawdown_test(
    time: np.ndarray,
    pressure: np.ndarray,
    production_rate: float,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    total_compressibility: float = 1e-5,
    wellbore_radius: float = 0.25,
    initial_pressure: float | None = None,
) -> WellTestResult:
    """Analyze pressure drawdown test.

    Uses semilog plot method for drawdown analysis.

    Args:
        time: Production time (hours)
        pressure: Flowing pressure (psi)
        production_rate: Production rate (STB/day)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        total_compressibility: Total compressibility (1/psi)
        wellbore_radius: Wellbore radius (ft)
        initial_pressure: Initial reservoir pressure (psi)

    Returns:
        WellTestResult with permeability, skin, etc.
    """
    # Filter valid data
    valid_mask = (time > 0) & (pressure > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient data for drawdown analysis")

    time_valid = time[valid_mask]
    pressure_valid = pressure[valid_mask]

    # Semilog plot: pressure vs log(time)
    log_time = np.log(time_valid)

    # Fit straight line to middle time region (radial flow)
    mid_start = len(log_time) // 4
    mid_end = 3 * len(log_time) // 4

    if mid_end - mid_start < 3:
        mid_start = 0
        mid_end = len(log_time)

    log_time_mid = log_time[mid_start:mid_end]
    pressure_mid = pressure_valid[mid_start:mid_end]

    # Fit line: p = m * log(t) + b
    coeffs = np.polyfit(log_time_mid, pressure_mid, 1)
    slope = coeffs[0]  # m (psi/log cycle)

    # Calculate permeability from slope
    if abs(slope) > 0:
        permeability = (
            162.6
            * production_rate
            * oil_viscosity
            * formation_volume_factor
            / (abs(slope) * reservoir_thickness)
        )
    else:
        permeability = 0.0

    # Calculate skin factor
    if initial_pressure is not None and abs(slope) > 0:
        # Find pressure at 1 hour
        idx_1hr = np.argmin(np.abs(time_valid - 1.0))
        p1hr = (
            pressure_valid[idx_1hr]
            if idx_1hr < len(pressure_valid)
            else pressure_valid[0]
        )

        log_term = np.log(
            permeability
            / (porosity * oil_viscosity * total_compressibility * wellbore_radius**2)
        )
        skin = 1.151 * ((initial_pressure - p1hr) / abs(slope) - log_term - 3.23)
    else:
        skin = 0.0

    # Detect boundaries
    boundary_distance, boundary_type = detect_boundaries(
        time_valid, pressure_valid, permeability, porosity, total_compressibility
    )

    return WellTestResult(
        permeability=max(0.001, min(1000.0, permeability)),
        skin=skin,
        boundary_distance=boundary_distance,
        boundary_type=boundary_type,
        reservoir_pressure=initial_pressure,
    )


def detect_boundaries(
    time: np.ndarray,
    pressure: np.ndarray,
    permeability: float,
    porosity: float,
    total_compressibility: float,
) -> tuple[float | None, str]:
    """Detect reservoir boundaries from pressure data.

    Detects:
    - No-flow boundaries (faults, pinchouts)
    - Constant pressure boundaries (aquifers, gas caps)

    Args:
        time: Time array (hours)
        pressure: Pressure array (psi)
        permeability: Permeability (md)
        porosity: Porosity (fraction)
        total_compressibility: Total compressibility (1/psi)

    Returns:
        Tuple of (boundary_distance, boundary_type)
    """
    if len(time) < 5:
        return None, "unknown"

    # Look for pressure derivative doubling (no-flow boundary)
    # or pressure stabilization (constant pressure boundary)

    # Calculate pressure derivative
    dp_dt = np.gradient(pressure, time)

    # Check for doubling of derivative (no-flow boundary indicator)
    if len(dp_dt) >= 5:
        early_derivative = np.mean(dp_dt[: len(dp_dt) // 3])
        late_derivative = np.mean(dp_dt[-len(dp_dt) // 3 :])

        if early_derivative > 0 and late_derivative > 0:
            ratio = late_derivative / early_derivative
            if ratio > 1.5:  # Significant increase
                # Estimate distance to boundary
                # r_boundary ≈ sqrt(0.00105 * k * t_boundary / (φ * μ * ct))
                # Use time where derivative doubles
                t_boundary = time[np.argmax(dp_dt > early_derivative * 1.5)]
                k_darcy = permeability / 1000.0
                distance = np.sqrt(
                    0.00105 * k_darcy * t_boundary / (porosity * total_compressibility)
                )
                return max(10.0, min(10000.0, distance)), "no_flow"

    # Check for pressure stabilization (constant pressure boundary)
    if len(pressure) >= 5:
        late_pressure = pressure[-len(pressure) // 3 :]
        pressure_change = np.max(late_pressure) - np.min(late_pressure)
        pressure_range = np.max(pressure) - np.min(pressure)

        if pressure_range > 0 and pressure_change / pressure_range < 0.05:
            # Pressure has stabilized
            return None, "constant_pressure"

    return None, "unknown"


def calculate_productivity_index_from_test(
    test_result: WellTestResult,
    reservoir_thickness: float = 50.0,
    wellbore_radius: float = 0.25,
    drainage_radius: float = 745.0,
) -> float:
    """Calculate productivity index from well test results.

    J = (0.00708 * k * h) / (μ * Bo * (ln(re/rw) + S))

    Args:
        test_result: WellTestResult from buildup or drawdown analysis
        reservoir_thickness: Reservoir thickness (ft)
        wellbore_radius: Wellbore radius (ft)
        drainage_radius: Drainage radius (ft)

    Returns:
        Productivity index (STB/day/psi)
    """
    # This is a simplified calculation
    # Full calculation would require viscosity and FVF
    # For now, return a normalized PI
    if test_result.permeability > 0:
        # Simplified: J proportional to k / (ln(re/rw) + S)
        skin_factor = max(0.0, test_result.skin)  # Positive skin reduces PI
        ln_term = np.log(drainage_radius / wellbore_radius) + skin_factor
        if ln_term > 0:
            J = test_result.permeability * reservoir_thickness / ln_term
            return max(0.1, J)
    return 0.0


def identify_flow_regimes(
    time: np.ndarray,
    pressure: np.ndarray,
) -> dict:
    """Log-log derivative analysis for flow-regime identification.

    Returns dict with flow_regimes, derivative arrays, and log-transformed series.
    """
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    dp = pressure[0] - pressure
    dt = np.diff(time)
    ddp = np.diff(dp)
    derivative = np.abs(ddp / (dt + 1e-12))

    valid_idx = (dp[1:] > 0) & (derivative > 0)
    log_time = np.log10(time[1:][valid_idx])
    log_dp = np.log10(dp[1:][valid_idx])
    log_derivative = np.log10(derivative[valid_idx])

    regimes: list[dict] = []
    if len(log_time) >= 3:
        slopes = np.diff(log_derivative) / (np.diff(log_time) + 1e-12)
        for i, slope in enumerate(slopes):
            if slope > 0.8:
                rtype = "wellbore_storage"
            elif abs(slope) < 0.1:
                rtype = "radial_flow"
            elif slope > 0.3:
                rtype = "boundary_effect"
            else:
                continue
            regimes.append(
                {
                    "type": rtype,
                    "start_time": float(10 ** log_time[i]),
                    "end_time": float(10 ** log_time[i + 1]),
                }
            )

    return {
        "flow_regimes": regimes,
        "derivative": derivative[valid_idx],
        "log_time": log_time,
        "log_pressure_change": log_dp,
        "log_derivative": log_derivative,
    }


def calculate_wellbore_storage(
    time: np.ndarray,
    pressure: np.ndarray,
    rate: float,
    fvf: float = 1.2,
) -> float:
    """Estimate wellbore storage coefficient C (bbl/psi) from early-time unit slope.

    C ≈ q * B / (24 * dp/dt) on the early unit-slope segment.
    """
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if len(time) < 3 or abs(rate) < 1e-6:
        return 0.0

    n_early = max(3, len(time) // 10)
    dt = time[1:n_early] - time[0]
    dp = np.abs(pressure[1:n_early] - pressure[0])
    valid = dt > 0
    if not np.any(valid):
        return 0.0
    slope = np.mean(dp[valid] / dt[valid])
    if slope <= 0:
        return 0.0
    return abs(rate) * fvf / (24.0 * slope)


def analyze_mdr(
    time: np.ndarray,
    pressure: np.ndarray,
) -> dict:
    """Miller-Dyes-Hutchinson (MDR) analysis: pws vs log(Δt)."""
    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    log_dt = np.log10(np.maximum(time, 1e-12))
    mid_start = len(pressure) // 4
    mid_end = 3 * len(pressure) // 4
    x = log_dt[mid_start:mid_end]
    y = pressure[mid_start:mid_end]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return {
        "slope": float(m),
        "intercept": float(c),
        "log_time": log_dt,
        "pressure": pressure,
    }


def analyze_multirate_superposition(
    time: np.ndarray,
    pressure: np.ndarray,
    rate: np.ndarray,
    rate_changes: list[tuple[float, float]],
    thickness: float,
    wellbore_radius: float,
    porosity: float,
    viscosity: float,
    fvf: float,
    compressibility: float,
) -> WellTestResult:
    """Multi-rate superposition analysis for permeability and skin."""
    import math

    time = np.asarray(time, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    rate = np.asarray(rate, dtype=float)

    superposition_time = np.zeros_like(time)
    for i, t in enumerate(time):
        active_rates = [(0.0, float(rate[0]))]
        for tc, qc in rate_changes:
            if tc < t:
                active_rates.append((tc, qc))
        st = 0.0
        for j in range(len(active_rates)):
            tc, qc = active_rates[j]
            if j < len(active_rates) - 1:
                q_delta = active_rates[j + 1][1] - qc
            else:
                q_delta = rate[i] - qc
            if abs(q_delta) > 0.01:
                st += q_delta * math.log10(t - tc + 1e-6)
        superposition_time[i] = st

    p_normalized = pressure / (rate + 1e-6)
    mid_start = len(superposition_time) // 3
    mid_end = 2 * len(superposition_time) // 3
    x = superposition_time[mid_start:mid_end]
    y = p_normalized[mid_start:mid_end]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    q_avg = float(np.mean(rate))
    k = 162.6 * q_avg * fvf * viscosity / (abs(m) * thickness + 1e-12)
    pi = float(pressure[0])
    skin = 1.151 * (
        (c - pi / (q_avg + 1e-12)) / (m + 1e-12)
        - math.log10(k / (porosity * viscosity * compressibility * wellbore_radius**2))
        + 3.23
    )

    return WellTestResult(
        permeability=float(k),
        skin=float(skin),
        wellbore_storage=0.0,
        reservoir_pressure=pi,
    )


def generate_dimensionless_pressure(
    tD: np.ndarray,
    CD: float = 100.0,
    skin: float = 0.0,
) -> np.ndarray:
    """Generate dimensionless pressure response (storage + radial + late)."""
    tD = np.asarray(tD, dtype=float)
    pD = np.zeros_like(tD)
    for i, t in enumerate(tD):
        if t < 1e-3:
            pD[i] = t / CD
        elif t < 100:
            pD[i] = 0.5 * (np.log(t) + 0.80907 + 2 * skin)
        else:
            pD[i] = 0.5 * (np.log(t) + 0.80907 + 2 * skin) + t / 1000
    return pD


def match_well_test_type_curve(
    pressure_data: np.ndarray,
    time_data: np.ndarray,
    rate: float,
    thickness: float,
    porosity: float,
    viscosity: float,
    compressibility: float,
    wellbore_radius: float,
) -> dict | None:
    """Simplified type-curve match returning CD, skin, and k estimate."""
    del rate, thickness  # rate/h used in full PTA scaling; kept for API parity
    pressure_data = np.asarray(pressure_data, dtype=float)
    time_data = np.asarray(time_data, dtype=float)

    best_match = None
    best_error = float("inf")

    for cd_exp in range(-2, 8):
        CD = 10**cd_exp
        for skin in np.linspace(-5, 10, 16):
            tD = time_data * 0.000264 * 100 / (
                porosity * viscosity * compressibility * wellbore_radius**2
            )
            pD_curve = generate_dimensionless_pressure(tD, CD, float(skin))
            dp = pressure_data[0] - pressure_data
            dp_max = np.max(dp)
            pD_max = np.max(pD_curve)
            if pD_max > 0:
                scaled_pD = pD_curve * (dp_max / pD_max)
                error = float(np.mean((dp - scaled_pD) ** 2))
                if error < best_error:
                    best_error = error
                    best_match = {
                        "CD": CD,
                        "skin": float(skin),
                        "k_estimate": 100 * (dp_max / pD_max),
                        "error": error,
                    }
    return best_match
