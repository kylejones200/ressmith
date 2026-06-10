#!/usr/bin/env python3
"""
Example 3: Formation Evaluation - Pressure & Rock Mechanics

This example demonstrates formation evaluation calculations:
- Overburden stress estimation
- Pore pressure prediction (Eaton's method)
- Fracture gradient estimation
- Drilling window analysis
- Wellbore stability analysis

Installation:
    uv sync

Run this example:
    python examples/03_formation_evaluation.py
"""

import logging

from petrosmith.core.formation_pressure import (
    OverburdenStress,
    PorePressurePrediction,
    FractureGradient,
    AbnormalPressureDetection,
)
from petrosmith.core.rock_mechanics import (
    ElasticProperties,
    RockStrength,
    InSituStress,
    WellboreStability,
    RockStrengthProperties,
    StressState,
)

logger = logging.getLogger(__name__)


def run_overburden(depth: float, water_depth: float, average_density: float) -> dict:
    """Run overburden stress calculation and log results. Returns overburden dict."""
    logger.info("STEP 1: Overburden Stress Calculation")
    logger.info("-" * 70)
    overburden = OverburdenStress.estimate_overburden_from_depth(
        depth=depth,
        water_depth=water_depth,
        average_density=average_density,
    )
    logger.info("   Average Formation Density: %s g/cc", average_density)
    logger.info(
        "   → Overburden Pressure: %s psi",
        f"{overburden['overburden_pressure_psi']:,.0f}",
    )
    logger.info("   → Overburden Gradient: %.3f psi/ft", overburden["gradient_psi_ft"])
    logger.info(
        "   → Equivalent Mud Weight: %.2f ppg",
        overburden["equivalent_mud_weight_ppg"],
    )
    logger.info("")
    return overburden


def run_pore_pressure(
    depth: float,
    overburden: dict,
    observed_sonic: float,
    normal_sonic: float,
    normal_pressure_gradient: float,
) -> dict:
    """Run pore pressure prediction (Eaton) and log results. Returns pore_pressure dict."""
    logger.info("STEP 2: Pore Pressure Prediction (Eaton's Method)")
    logger.info("-" * 70)
    pore_pressure = PorePressurePrediction.eatons_method(
        observed_parameter=observed_sonic,
        normal_parameter=normal_sonic,
        overburden_gradient=overburden["gradient_psi_ft"],
        normal_pressure_gradient=normal_pressure_gradient,
        exponent=3.0,
    )
    logger.info("   Observed Sonic: %s µs/ft", observed_sonic)
    logger.info("   Normal Trend Sonic: %s µs/ft", normal_sonic)
    logger.info(
        "   → Pore Pressure: %s psi",
        f"{pore_pressure['pore_pressure_psi']:,.0f}",
    )
    logger.info(
        "   → Pressure Gradient: %.3f psi/ft",
        pore_pressure["pore_pressure_gradient_psi_ft"],
    )
    logger.info(
        "   → Equivalent MW: %.2f ppg",
        pore_pressure["equivalent_mud_weight_ppg"],
    )
    if pore_pressure["pressure_state"] == "overpressured":
        logger.warning("   WARNING: OVERPRESSURED ZONE DETECTED")
    logger.info("")
    return pore_pressure


def run_fracture_gradient(
    depth: float, overburden: dict, pore_pressure: dict
) -> dict:
    """Run fracture gradient (Matthews & Kelly) and log results. Returns fracture dict."""
    logger.info("STEP 3: Fracture Gradient Estimation (Matthews & Kelly)")
    logger.info("-" * 70)
    fracture = FractureGradient.matthews_kelly_method(
        depth=depth,
        overburden_gradient=overburden["gradient_psi_ft"],
        pore_pressure_gradient=pore_pressure["pore_pressure_gradient_psi_ft"],
    )
    logger.info(
        "   → Fracture Pressure: %s psi",
        f"{fracture['fracture_pressure_psi']:,.0f}",
    )
    logger.info(
        "   → Fracture Gradient: %.3f psi/ft",
        fracture["fracture_gradient_psi_ft"],
    )
    logger.info(
        "   → Equivalent MW: %.2f ppg",
        fracture["equivalent_mud_weight_ppg"],
    )
    logger.info("")
    return fracture


def run_drilling_window(
    depth: float,
    pore_pressure: dict,
    fracture: dict,
    current_mud_weight: float,
    safety_margin: float,
) -> dict:
    """Run drilling window analysis and log results. Returns window dict."""
    logger.info("STEP 4: Drilling Window Analysis")
    logger.info("-" * 70)
    window = AbnormalPressureDetection.drilling_window_analysis(
        depth=depth,
        pore_pressure_gradient=pore_pressure["pore_pressure_gradient_psi_ft"],
        fracture_gradient=fracture["fracture_gradient_psi_ft"],
        mud_weight=current_mud_weight,
        safety_margin=safety_margin,
    )
    logger.info("   Current Mud Weight: %s ppg", current_mud_weight)
    logger.info("   Safety Margin: %s ppg", safety_margin)
    logger.info("")
    logger.info(
        "   → Required MW (Pore + Safety): %.2f ppg",
        window["required_mud_weight_ppg"],
    )
    logger.info(
        "   → Maximum MW (Fracture): %.2f ppg",
        window["maximum_mud_weight_ppg"],
    )
    logger.info(
        "   → Drilling Window: %.2f ppg",
        window["drilling_window_ppg"],
    )
    logger.info("   → Status: %s", window["status"])
    if window["status"] != "Within drilling window":
        logger.warning("   WARNING: %s", window["status"])
    logger.info("")
    return window


def run_elastic_properties(
    vp_sonic: float, vs_sonic: float, density: float
) -> tuple[float, float, float, float]:
    """Run elastic properties from sonic and log results. Returns (youngs_modulus, poissons_ratio, vp, vs)."""
    logger.info("STEP 5: Rock Mechanics - Elastic Properties")
    logger.info("-" * 70)
    vp = ElasticProperties.sonic_to_velocity(vp_sonic)
    vs = ElasticProperties.sonic_to_velocity(vs_sonic)
    youngs_modulus = ElasticProperties.calculate_youngs_modulus(vp, vs, density)
    poissons_ratio = ElasticProperties.calculate_poissons_ratio(vp, vs)
    logger.info("   P-wave Sonic: %s µs/ft → Velocity: %s ft/s", vp_sonic, f"{vp:,.0f}")
    logger.info("   S-wave Sonic: %s µs/ft → Velocity: %s ft/s", vs_sonic, f"{vs:,.0f}")
    logger.info("   → Young's Modulus: %s psi", f"{youngs_modulus:,.0f}")
    logger.info("   → Poisson's Ratio: %.3f", poissons_ratio)
    logger.info("")
    return youngs_modulus, poissons_ratio, vp, vs


def run_rock_strength(vp_sonic: float) -> tuple[float, float, float, float]:
    """Run rock strength properties and log results. Returns (ucs, tensile_strength, friction_angle, cohesion)."""
    logger.info("STEP 6: Rock Strength Properties")
    logger.info("-" * 70)
    ucs = RockStrength.ucs_from_sonic(vp_sonic, correlation="sandstone")
    tensile_strength = RockStrength.tensile_strength_from_ucs(ucs)
    friction_angle = RockStrength.estimate_friction_angle("sandstone")
    cohesion = ucs / 4
    logger.info("   → Unconfined Compressive Strength (UCS): %s psi", f"{ucs:,.0f}")
    logger.info("   → Tensile Strength: %s psi", f"{tensile_strength:,.0f}")
    logger.info("   → Friction Angle: %s°", friction_angle)
    logger.info("   → Cohesion: %s psi (estimated)", f"{cohesion:,.0f}")
    logger.info("")
    return ucs, tensile_strength, friction_angle, cohesion


def run_insitu_stress(
    overburden: dict, pore_pressure: dict, poissons_ratio: float
) -> dict:
    """Run in-situ stress estimation and log results. Returns stresses dict."""
    logger.info("STEP 7: In-Situ Stress State")
    logger.info("-" * 70)
    vertical_stress = overburden["overburden_pressure_psi"]
    pore_pressure_psi = pore_pressure["pore_pressure_psi"]
    stresses = InSituStress.estimate_horizontal_stress(
        vertical_stress=vertical_stress,
        pore_pressure=pore_pressure_psi,
        poissons_ratio=poissons_ratio,
        stress_regime="normal",
    )
    logger.info("   Stress Regime: Normal Faulting")
    logger.info(
        "   → Vertical Stress (Sv): %s psi",
        f"{stresses['vertical_stress_psi']:,.0f}",
    )
    logger.info(
        "   → Max Horizontal Stress (SHmax): %s psi",
        f"{stresses['max_horizontal_stress_psi']:,.0f}",
    )
    logger.info(
        "   → Min Horizontal Stress (Shmin): %s psi",
        f"{stresses['min_horizontal_stress_psi']:,.0f}",
    )
    logger.info("")
    return stresses


def run_wellbore_stability(
    overburden: dict,
    pore_pressure: dict,
    stresses: dict,
    rock_props: RockStrengthProperties,
    current_mud_weight: float,
    depth: float,
) -> dict:
    """Run wellbore stability (mud weight window) and log results. Returns stability dict."""
    logger.info("STEP 8: Wellbore Stability (Mud Weight Window)")
    logger.info("-" * 70)
    vertical_stress = overburden["overburden_pressure_psi"]
    pore_pressure_psi = pore_pressure["pore_pressure_psi"]
    stress_state = StressState(
        vertical_stress=vertical_stress,
        max_horizontal_stress=stresses["max_horizontal_stress_psi"],
        min_horizontal_stress=stresses["min_horizontal_stress_psi"],
        pore_pressure=pore_pressure_psi,
        depth=depth,
    )
    stability = WellboreStability.mud_weight_window(
        stress_state,
        rock_props,
        current_mud_weight=current_mud_weight,
    )
    logger.info(
        "   → Minimum MW (Collapse): %.2f ppg",
        stability["minimum_mud_weight_ppg"],
    )
    logger.info(
        "   → Maximum MW (Fracture): %.2f ppg",
        stability["maximum_mud_weight_ppg"],
    )
    logger.info(
        "   → Stability Window: %.2f ppg",
        stability["mud_weight_window_ppg"],
    )
    logger.info("   → Current MW: %s ppg", current_mud_weight)
    logger.info("   → Status: %s", stability["status"])
    if stability["status"] != "Within stability window":
        logger.warning("   WARNING: Wellbore instability risk!")
    logger.info("")
    return stability


def log_formation_summary(
    depth: float,
    overburden: dict,
    pore_pressure: dict,
    fracture: dict,
    window: dict,
    stability: dict,
) -> None:
    """Log formation evaluation summary."""
    logger.info("=" * 70)
    logger.info("FORMATION EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info("   Depth: %s ft", f"{depth:,}")
    logger.info("   Overburden: %.2f ppg", overburden["equivalent_mud_weight_ppg"])
    logger.info("   Pore Pressure: %.2f ppg", pore_pressure["equivalent_mud_weight_ppg"])
    logger.info("   Fracture Gradient: %.2f ppg", fracture["equivalent_mud_weight_ppg"])
    logger.info("   Drilling Window: %.2f ppg", window["drilling_window_ppg"])
    logger.info("   Stability Window: %.2f ppg", stability["mud_weight_window_ppg"])
    logger.info(
        "   Recommended MW: %.2f - %.2f ppg",
        window["required_mud_weight_ppg"],
        window["maximum_mud_weight_ppg"],
    )
    logger.info("=" * 70)


def main() -> None:
    """Orchestrate formation evaluation example."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Example 3: Formation Evaluation - Pressure & Rock Mechanics")
    logger.info("=" * 70)
    logger.info("")

    depth = 10000.0
    water_depth = 0.0
    average_density = 2.5
    observed_sonic = 85
    normal_sonic = 70
    normal_pressure_gradient = 0.465
    current_mud_weight = 11.5
    safety_margin = 0.5
    vp_sonic = 55
    vs_sonic = 95
    density = 2.5

    logger.info("SCENARIO: Formation Evaluation at %s ft (Onshore)", f"{depth:,}")
    logger.info("=" * 70)
    logger.info("")

    overburden = run_overburden(depth, water_depth, average_density)
    pore_pressure = run_pore_pressure(
        depth, overburden, observed_sonic, normal_sonic, normal_pressure_gradient
    )
    fracture = run_fracture_gradient(depth, overburden, pore_pressure)
    window = run_drilling_window(
        depth, pore_pressure, fracture, current_mud_weight, safety_margin
    )
    youngs_modulus, poissons_ratio, vp, vs = run_elastic_properties(
        vp_sonic, vs_sonic, density
    )
    ucs, tensile_strength, friction_angle, cohesion = run_rock_strength(vp_sonic)
    stresses = run_insitu_stress(overburden, pore_pressure, poissons_ratio)

    rock_props = RockStrengthProperties(
        ucs=ucs,
        tensile_strength=tensile_strength,
        cohesion=cohesion,
        friction_angle=friction_angle,
        poissons_ratio=poissons_ratio,
        youngs_modulus=youngs_modulus,
    )
    stability = run_wellbore_stability(
        overburden, pore_pressure, stresses, rock_props, current_mud_weight, depth
    )

    log_formation_summary(depth, overburden, pore_pressure, fracture, window, stability)


if __name__ == "__main__":
    main()
