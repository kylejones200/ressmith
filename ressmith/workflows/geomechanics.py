"""Geomechanics and formation-pressure workflows.

Thin Layer-4 wrappers around rock-mechanics and pore-pressure primitives.
"""

from __future__ import annotations

import logging
from typing import Any

from ressmith.primitives.formation_pressure import complete_pressure_analysis
from ressmith.primitives.rock_mechanics import complete_geomechanical_analysis

logger = logging.getLogger(__name__)


def geomechanical_study(
    depth: float,
    sonic_dt_compressional: float,
    sonic_dt_shear: float,
    bulk_density: float,
    porosity: float,
    pore_pressure: float,
    mud_weight: float,
) -> dict[str, Any]:
    """Complete geomechanical analysis (elastic props, UCS, mud-weight window)."""
    logger.info("Running geomechanical study at %.0f ft", depth)
    return complete_geomechanical_analysis(
        depth=depth,
        sonic_dt_compressional=sonic_dt_compressional,
        sonic_dt_shear=sonic_dt_shear,
        bulk_density=bulk_density,
        porosity=porosity,
        pore_pressure=pore_pressure,
        mud_weight=mud_weight,
    )


def formation_pressure_study(
    depth: float,
    mud_weight: float,
    overburden_gradient: float,
    sonic_dt: float | None = None,
    normal_sonic: float | None = None,
    water_depth: float = 0.0,
) -> dict[str, Any]:
    """Pore-pressure / fracture-gradient / drilling-window analysis."""
    logger.info("Running formation pressure study at %.0f ft", depth)
    return complete_pressure_analysis(
        depth=depth,
        mud_weight=mud_weight,
        overburden_gradient=overburden_gradient,
        sonic_dt=sonic_dt,
        normal_sonic=normal_sonic,
        water_depth=water_depth,
    )
