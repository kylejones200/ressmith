#!/usr/bin/env python3
"""
Example 2: Well Control - Kick Detection and Kill Procedure

This example demonstrates well control calculations:
- Kick detection from pit gain and flow changes
- Formation pressure calculation
- Kill mud weight determination
- Driller's Method kill procedure

Installation:
    uv sync

Run this example:
    python examples/02_well_control_kick_analysis.py
"""

import logging

from petrosmith.core.well_control import (
    KickDetection,
    WellControlProcedures,
    KickData,
    WellGeometry,
    MudProperties,
)

logger = logging.getLogger(__name__)


def run_kick_detection(
    pit_gain: float, flow_increase: float, connection_flow: bool
) -> dict:
    """Run kick detection and log results. Returns detection dict."""
    logger.info("STEP 1: Kick Detection")
    logger.info("-" * 70)
    logger.info("   Pit Volume Gain: %s bbls", pit_gain)
    logger.info("   Flow Rate Increase: %s%%", flow_increase)
    logger.info("   Flow at Connection: %s", "Yes" if connection_flow else "No")
    logger.info("")

    detection = KickDetection.detect_kick(
        pit_volume_gain=pit_gain,
        flow_rate_increase=flow_increase,
        connection_flow=connection_flow,
    )
    logger.info("   STATUS: %s", detection["status"])
    logger.info("   SEVERITY: %s", detection["severity"])
    logger.info("   RECOMMENDED ACTION: %s", detection["recommended_action"])
    logger.info("")

    if detection["severity"] in ["HIGH", "CRITICAL"]:
        logger.warning("   WARNING: SHUT IN WELL IMMEDIATELY!")
        logger.info("")

    return detection


def run_formation_pressure_analysis(
    sidpp: float, sicp: float, current_mud_weight: float, tvd: float
) -> tuple[float, float]:
    """Run formation pressure and kick intensity. Returns (formation_pressure, kick_intensity)."""
    logger.info("STEP 2: Formation Pressure Analysis")
    logger.info("-" * 70)
    logger.info("   Current Mud Weight: %s ppg", current_mud_weight)
    logger.info("   True Vertical Depth: %s ft", f"{tvd:,}")
    logger.info("   Shut-In Drill Pipe Pressure (SIDPP): %s psi", sidpp)
    logger.info("   Shut-In Casing Pressure (SICP): %s psi", sicp)
    logger.info("")

    formation_pressure = KickDetection.calculate_formation_pressure(
        shut_in_drillpipe_pressure=sidpp,
        mud_weight=current_mud_weight,
        tvd=tvd,
    )
    kick_intensity = KickDetection.calculate_kick_intensity(
        formation_pressure=formation_pressure,
        tvd=tvd,
    )
    logger.info("   → Formation Pressure: %s psi", f"{formation_pressure:,.0f}")
    logger.info("   → Kick Intensity: %.2f ppg", kick_intensity)
    logger.info("   → Overbalance Needed: %.2f ppg", kick_intensity - current_mud_weight)
    logger.info("")

    return formation_pressure, kick_intensity


def run_kill_mud_weight_calculation(
    kick_intensity: float, safety_margin: float
) -> float:
    """Compute and log kill mud weight. Returns kill_mud_weight."""
    logger.info("STEP 3: Kill Mud Weight Calculation")
    logger.info("-" * 70)
    kill_mud_weight = kick_intensity + safety_margin
    logger.info("   Kick Intensity: %.2f ppg", kick_intensity)
    logger.info("   Safety Margin: %s ppg", safety_margin)
    logger.info("   → Kill Mud Weight: %.2f ppg", kill_mud_weight)
    logger.info("")

    return kill_mud_weight


def run_drillers_method(
    tvd: float,
    current_mud_weight: float,
    pit_gain: float,
    sidpp: float,
    sicp: float,
    formation_pressure: float,
    kick_intensity: float,
    pump_rate: float,
    pump_pressure: float,
) -> dict:
    """Build well/mud/kick data, run Driller's Method, log plan. Returns kill_plan."""
    logger.info("STEP 4: Driller's Method Kill Procedure")
    logger.info("-" * 70)

    well_geo = WellGeometry(
        measured_depth=10000,
        true_vertical_depth=tvd,
        hole_diameter=8.5,
        drillpipe_od=5.0,
        drillpipe_id=4.276,
        drillcollar_od=6.5,
        drillcollar_id=2.75,
        drillcollar_length=500,
        casing_id=12.615,
    )
    mud_props = MudProperties(
        weight=current_mud_weight,
        plastic_viscosity=25,
        yield_point=15,
        funnel_viscosity=45,
        gel_strength_10sec=8,
        gel_strength_10min=12,
    )
    kick_data = KickData(
        pit_gain=pit_gain,
        shut_in_drillpipe_pressure=sidpp,
        shut_in_casing_pressure=sicp,
        formation_pressure=formation_pressure,
        kick_intensity=kick_intensity,
        kick_height=200,
        kick_type="gas",
    )

    wcp = WellControlProcedures(well_geo, mud_props, kick_data)
    kill_plan = wcp.drillers_method(pump_rate=pump_rate, pump_pressure=pump_pressure)

    logger.info("   Pump Rate: %s bpm", pump_rate)
    logger.info("   Pump Pressure: %s psi", pump_pressure)
    logger.info("")
    logger.info("   FIRST CIRCULATION (Kill the kick):")
    logger.info(
        "     • Initial Circulating Pressure (ICP): %s psi",
        f"{kill_plan['first_circulation']['icp']:,.0f}",
    )
    logger.info(
        "     • Final Circulating Pressure (FCP): %s psi",
        f"{kill_plan['first_circulation']['fcp']:,.0f}",
    )
    logger.info(
        "     • Circulation Time: %.1f minutes",
        kill_plan["first_circulation"]["time_minutes"],
    )
    logger.info("")
    logger.info("   SECOND CIRCULATION (Weight up to kill mud):")
    logger.info("     • Kill Mud Weight: %.2f ppg", kill_plan["kill_mud_weight"])
    logger.info(
        "     • New ICP: %s psi",
        f"{kill_plan['second_circulation']['icp']:,.0f}",
    )
    logger.info(
        "     • New FCP: %s psi",
        f"{kill_plan['second_circulation']['fcp']:,.0f}",
    )
    logger.info(
        "     • Circulation Time: %.1f minutes",
        kill_plan["second_circulation"]["time_minutes"],
    )
    logger.info("")
    logger.info(
        "   TOTAL TIME: %.1f minutes (%.1f hours)",
        kill_plan["total_time_minutes"],
        kill_plan["total_time_minutes"] / 60,
    )
    logger.info("")

    return kill_plan


def log_safety_recommendations() -> None:
    """Log well control safety recommendations."""
    logger.info("SAFETY RECOMMENDATIONS:")
    logger.info("-" * 70)
    logger.info("   1. Maintain constant drillpipe pressure during circulation")
    logger.info("   2. Monitor casing pressure for gas migration")
    logger.info("   3. Do not exceed MAASP (Maximum Allowable Annular Surface Pressure)")
    logger.info("   4. Have BOP ready for secondary well control if needed")
    logger.info("   5. Notify all crew and company management")
    logger.info("")


def main() -> None:
    """Orchestrate well control kick analysis example."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Example 2: Well Control - Kick Detection & Kill Procedure")
    logger.info("=" * 70)
    logger.info("")
    logger.info("SCENARIO: Drilling at 10,000 ft TVD")
    logger.info("=" * 70)
    logger.info("")

    pit_gain = 15.0
    flow_increase = 12.0
    connection_flow = True
    current_mud_weight = 10.5
    tvd = 10000.0
    sidpp = 350.0
    sicp = 700.0
    safety_margin = 0.5
    pump_rate = 8.0
    pump_pressure = 2500.0

    run_kick_detection(pit_gain, flow_increase, connection_flow)
    formation_pressure, kick_intensity = run_formation_pressure_analysis(
        sidpp, sicp, current_mud_weight, tvd
    )
    run_kill_mud_weight_calculation(kick_intensity, safety_margin)
    run_drillers_method(
        tvd=tvd,
        current_mud_weight=current_mud_weight,
        pit_gain=pit_gain,
        sidpp=sidpp,
        sicp=sicp,
        formation_pressure=formation_pressure,
        kick_intensity=kick_intensity,
        pump_rate=pump_rate,
        pump_pressure=pump_pressure,
    )
    log_safety_recommendations()

    logger.info("=" * 70)
    logger.info("Well control procedure calculated successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
