#!/usr/bin/env python3
"""
Example 1: Basic Drilling Calculations

This example demonstrates the most common drilling calculations:
- Hydrostatic pressure
- Equivalent Circulating Density (ECD)
- Annular velocity
- Casing design

Installation:
    uv sync

Run this example:
    python examples/01_basic_drilling_calculations.py
"""

import logging

from petrosmith.api import DrillingAPI
from petrosmith.exceptions import InvalidMudWeightError, InvalidDepthError

logger = logging.getLogger(__name__)


def run_hydrostatic_example(api: DrillingAPI) -> None:
    """Run hydrostatic pressure calculation and log results."""
    logger.info("1. Hydrostatic Pressure Calculation")
    logger.info("-" * 70)
    mud_weight = 10.5  # ppg
    tvd = 10000  # feet
    pressure = api.calculate_hydrostatic_pressure(mud_weight, tvd)
    logger.info("   Mud Weight: %s ppg", mud_weight)
    logger.info("   True Vertical Depth: %s ft", f"{tvd:,}")
    logger.info("   → Hydrostatic Pressure: %s psi", f"{pressure:,.0f}")
    logger.info("")


def run_ecd_example(api: DrillingAPI) -> None:
    """Run ECD calculation and log results."""
    logger.info("2. Equivalent Circulating Density (ECD)")
    logger.info("-" * 70)
    mud_weight = 10.5
    tvd = 10000
    annular_pressure_loss = 450  # psi
    ecd = api.calculate_ecd(mud_weight, annular_pressure_loss, tvd)
    logger.info("   Static Mud Weight: %s ppg", mud_weight)
    logger.info("   Annular Pressure Loss: %s psi", annular_pressure_loss)
    logger.info("   True Vertical Depth: %s ft", f"{tvd:,}")
    logger.info("   → ECD: %.2f ppg", ecd)
    logger.info("   → ECD Increase: %.2f ppg", ecd - mud_weight)
    logger.info("")


def run_casing_design_example(api: DrillingAPI) -> None:
    """Run casing design analysis and log results."""
    logger.info("3. Casing Design Analysis")
    logger.info("-" * 70)
    casing_od = 9.625  # inches
    wall_thickness = 0.545  # inches
    yield_strength = 80000  # psi
    design = api.calculate_casing_design(casing_od, wall_thickness, yield_strength)
    logger.info("   Casing OD: %s in", casing_od)
    logger.info("   Wall Thickness: %s in", wall_thickness)
    logger.info("   Yield Strength: %s psi", f"{yield_strength:,}")
    logger.info("   → Burst Pressure Rating: %s psi", f"{design['burst_pressure']:,.0f}")
    logger.info("   → Collapse Pressure Rating: %s psi", f"{design['collapse_pressure']:,.0f}")
    logger.info("   → Inner Diameter: %.3f in", design["inner_diameter"])
    logger.info("")


def run_pressure_at_depths_example(api: DrillingAPI) -> None:
    """Run pressure-at-depth table and log results."""
    logger.info("4. Pressure at Multiple Depths")
    logger.info("-" * 70)
    mud_weight = 10.5
    depths = [5000, 10000, 15000, 20000]
    logger.info("   Mud Weight: %s ppg", mud_weight)
    logger.info("")
    logger.info("   Depth (ft) | Pressure (psi) | Gradient (psi/ft)")
    logger.info("   " + "-" * 55)
    for depth in depths:
        pressure = api.calculate_hydrostatic_pressure(mud_weight, depth)
        gradient = pressure / depth if depth > 0 else 0
        logger.info(
            "   %s | %s | %s",
            f"{depth:>10,}",
            f"{pressure:>14,.0f}",
            f"{gradient:>16.3f}",
        )
    logger.info("")


def run_validation_example(api: DrillingAPI) -> None:
    """Run input validation (error handling) examples and log results."""
    logger.info("5. Input Validation (Error Handling)")
    logger.info("-" * 70)
    try:
        api.calculate_hydrostatic_pressure(25.0, 10000)
    except InvalidMudWeightError as e:
        logger.error("   Caught error for invalid mud weight: %s", e)
    logger.info("")
    try:
        api.calculate_hydrostatic_pressure(10.5, -1000)
    except InvalidDepthError as e:
        logger.error("   Caught error for negative depth: %s", e)
    logger.info("")


def main() -> None:
    """Orchestrate basic drilling calculation examples."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Example 1: Basic Drilling Calculations")
    logger.info("=" * 70)
    logger.info("")

    api = DrillingAPI()
    run_hydrostatic_example(api)
    run_ecd_example(api)
    run_casing_design_example(api)
    run_pressure_at_depths_example(api)
    run_validation_example(api)

    logger.info("=" * 70)
    logger.info("Example complete! All calculations successful.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
