"""Drilling and well-control workflows.

Thin Layer-4 wrappers around drilling / mud / kick primitives.
"""

from __future__ import annotations

import logging
from typing import Any

from ressmith.primitives.drilling_fluids import (
    FiltrationProperties,
    RheologyData,
    analyze_mud_system,
)
from ressmith.primitives.well_control import detect_and_analyze_kick

logger = logging.getLogger(__name__)


def mud_system_study(
    mud_weight: float,
    reading_600: float,
    reading_300: float,
    gel_10sec: float,
    gel_10min: float,
    api_filtrate: float = 8.0,
    hthp_filtrate: float = 15.0,
    filter_cake_thickness: float = 2.0,
    reading_200: float = 0.0,
    reading_100: float = 0.0,
    reading_6: float = 0.0,
    reading_3: float = 0.0,
    temperature: float = 120.0,
    spurt_loss: float = 1.0,
) -> dict[str, Any]:
    """End-to-end mud system analysis (rheology, gels, filtration)."""
    logger.info("Running mud system study: mw=%.1f ppg", mud_weight)
    rheology = RheologyData(
        reading_600=reading_600,
        reading_300=reading_300,
        reading_200=reading_200 or reading_300 * 0.75,
        reading_100=reading_100 or reading_300 * 0.5,
        reading_6=reading_6 or reading_300 * 0.1,
        reading_3=reading_3 or reading_300 * 0.08,
        temperature=temperature,
    )
    filtration = FiltrationProperties(
        api_filtrate=api_filtrate,
        hthp_filtrate=hthp_filtrate,
        filter_cake_thickness=filter_cake_thickness,
        spurt_loss=spurt_loss,
    )
    return analyze_mud_system(
        mud_weight=mud_weight,
        rheology=rheology,
        gel_10sec=gel_10sec,
        gel_10min=gel_10min,
        filtration=filtration,
    )


def kick_analysis_study(
    pit_gain: float,
    sidpp: float,
    sicp: float,
    mud_weight: float,
    tvd: float,
) -> dict[str, Any]:
    """Kick detection and kill-parameter analysis from shut-in pressures."""
    logger.info(
        "Running kick analysis: pit_gain=%.1f bbl, SIDPP=%.0f psi, TVD=%.0f ft",
        pit_gain,
        sidpp,
        tvd,
    )
    return detect_and_analyze_kick(
        pit_gain=pit_gain,
        sidpp=sidpp,
        sicp=sicp,
        mud_weight=mud_weight,
        tvd=tvd,
    )
