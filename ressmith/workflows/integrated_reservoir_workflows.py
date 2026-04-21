"""Integrated study workflows: interference, coning, and enhanced RTA.

These functions orchestrate existing Layer-4 workflows into one return dict
for scripting and discovery. They do not replace the underlying modules;
use those when you need fine-grained control.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

from ressmith.workflows.advanced_rta import (
    analyze_blasingame,
    analyze_dn_type_curve,
    analyze_fmb,
    analyze_fracture_network,
)
from ressmith.workflows.coning import analyze_well_coning, forecast_wor_gor_with_coning
from ressmith.workflows.diagnostics_plots import generate_diagnostic_plot_data
from ressmith.workflows.interference import (
    analyze_interference_matrix,
    analyze_interference_with_production_history,
    calculate_well_distances,
    recommend_spacing,
)
from ressmith.workflows.pressure_normalization import normalize_for_rta_analysis
from ressmith.workflows.type_curves import match_type_curve_workflow

logger = logging.getLogger(__name__)

# Default physical / screening values for bundled studies (documented API defaults).
_DEFAULT_DRAINAGE_RADIUS_FT = 600.0
_DEFAULT_TARGET_INTERFERENCE_PCT = 5.0
_DEFAULT_MIN_SPACING_FT = 200.0
_DEFAULT_MAX_SPACING_FT = 2000.0
_DEFAULT_RTA_INITIAL_PRESSURE_PSI = 5000.0
_DEFAULT_FRACTURE_STAGE_SPACING_FT = 300.0


def well_interference_study(
    well_locations: pd.DataFrame,
    *,
    drainage_radii: dict[str, float] | pd.Series | None = None,
    default_drainage_radius: float = _DEFAULT_DRAINAGE_RADIUS_FT,
    production_data: dict[str, pd.DataFrame] | None = None,
    model_name: str = "arps_hyperbolic",
    phase: str = "oil",
    include_spacing_recommendation: bool = True,
    target_interference: float = _DEFAULT_TARGET_INTERFERENCE_PCT,
    min_spacing: float = _DEFAULT_MIN_SPACING_FT,
    max_spacing: float = _DEFAULT_MAX_SPACING_FT,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Run a full well-interference study: distances, geometric pairs, optional EUR-based pairs, spacing.

    Parameters
    ----------
    well_locations
        Columns ``well_id``, ``latitude``, ``longitude``.
    drainage_radii
        Per-well drainage radius (ft); defaults to ``default_drainage_radius``.
    default_drainage_radius
        Used when ``drainage_radii`` is missing a well.
    production_data
        If provided, adds ``analyze_interference_with_production_history`` using ``estimate_eur``-style fits.
    model_name, phase
        Passed to production-based interference when ``production_data`` is set.
    include_spacing_recommendation
        If True, calls ``recommend_spacing`` using the median of resolved drainage radii.
    target_interference, min_spacing, max_spacing
        Passed to ``recommend_spacing``.
    **fit_kwargs
        Extra arguments for EUR estimation in the production-based path.

    Returns
    -------
    dict
        ``distances``, ``geometric_interference``, ``production_interference`` (optional),
        ``spacing_recommendation`` (optional), ``metadata``.
    """
    logger.info("Starting well_interference_study for %s wells", len(well_locations))

    distances = calculate_well_distances(well_locations)
    geometric = analyze_interference_matrix(
        well_locations,
        drainage_radii=drainage_radii,
        default_drainage_radius=default_drainage_radius,
    )

    radii_map: dict[str, float]
    if drainage_radii is None:
        radii_map = {
            str(w): float(default_drainage_radius)
            for w in well_locations["well_id"].astype(str).unique()
        }
    elif isinstance(drainage_radii, pd.Series):
        radii_map = {str(k): float(v) for k, v in drainage_radii.items()}
    else:
        radii_map = {str(k): float(v) for k, v in drainage_radii.items()}

    prod_interference: pd.DataFrame | None = None
    if production_data:
        try:
            prod_interference = analyze_interference_with_production_history(
                well_locations,
                production_data,
                model_name=model_name,
                phase=phase,
                **fit_kwargs,
            )
        except Exception as e:
            logger.warning("Production-based interference failed: %s", e)

    spacing: dict[str, Any] | None = None
    if include_spacing_recommendation and radii_map:
        r_med = float(np.median(list(radii_map.values())))
        spacing = recommend_spacing(
            drainage_radius=r_med,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            target_interference=target_interference,
        )

    return {
        "distances": distances,
        "geometric_interference": geometric,
        "production_interference": prod_interference,
        "spacing_recommendation": spacing,
        "metadata": {
            "n_wells": int(len(well_locations)),
            "used_production_history": prod_interference is not None,
        },
    }


def coning_study(
    *,
    production_rate: float,
    oil_density: float,
    water_density: float,
    permeability: float,
    reservoir_thickness: float,
    well_completion_interval: float,
    gas_density: float | None = None,
    method: Literal["meyer_gardner", "chierici_ciucci"] = "meyer_gardner",
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    porosity: float = 0.15,
    include_yield_forecast: bool = False,
    forecast_time: pd.Series | np.ndarray | None = None,
    forecast_oil_rate: pd.Series | np.ndarray | None = None,
    initial_wor: float = 0.0,
    initial_gor: float = 1000.0,
) -> dict[str, Any]:
    """Water/gas coning screening plus optional WOR/GOR forecast along a schedule.

    Parameters
    ----------
    production_rate
        Reference oil rate (STB/day) for critical-rate analysis.
    include_yield_forecast
        If True, requires ``forecast_time`` and ``forecast_oil_rate``; returns a yield table.
    forecast_time, forecast_oil_rate
        Arrays aligned with the coning yield model (see ``forecast_wor_gor_with_coning``).

    Returns
    -------
    dict
        ``coning_analysis`` (from ``analyze_well_coning``), optional ``yield_forecast`` DataFrame,
        and ``metadata``.
    """
    logger.info("Starting coning_study method=%s", method)

    coning_analysis = analyze_well_coning(
        production_rate=production_rate,
        oil_density=oil_density,
        water_density=water_density,
        permeability=permeability,
        reservoir_thickness=reservoir_thickness,
        well_completion_interval=well_completion_interval,
        gas_density=gas_density,
        method=method,
        oil_viscosity=oil_viscosity,
        formation_volume_factor=formation_volume_factor,
        porosity=porosity,
    )

    yield_forecast: pd.DataFrame | None = None
    if include_yield_forecast:
        if forecast_time is None or forecast_oil_rate is None:
            raise ValueError(
                "include_yield_forecast=True requires forecast_time and forecast_oil_rate"
            )
        yield_forecast = forecast_wor_gor_with_coning(
            forecast_time,
            forecast_oil_rate,
            production_rate=production_rate,
            oil_density=oil_density,
            water_density=water_density,
            permeability=permeability,
            reservoir_thickness=reservoir_thickness,
            well_completion_interval=well_completion_interval,
            gas_density=gas_density,
            method=method,
            initial_wor=initial_wor,
            initial_gor=initial_gor,
            oil_viscosity=oil_viscosity,
            formation_volume_factor=formation_volume_factor,
            porosity=porosity,
        )

    return {
        "coning_analysis": coning_analysis,
        "yield_forecast": yield_forecast,
        "metadata": {"method": method, "yield_forecast": yield_forecast is not None},
    }


def enhanced_rta_study(
    data: pd.DataFrame,
    *,
    time_col: str | None = None,
    rate_col: str = "oil",
    pressure_col: str | None = None,
    cumulative_col: str | None = None,
    initial_pressure: float | None = None,
    run_normalization: bool = True,
    run_type_curve: bool = True,
    run_blasingame: bool = True,
    run_dn: bool = True,
    run_fmb: bool = False,
    run_fracture_network: bool = False,
    fracture_number_of_stages: int = 1,
    fracture_stage_spacing: float = _DEFAULT_FRACTURE_STAGE_SPACING_FT,
) -> dict[str, Any]:
    """Enhanced RTA bundle: normalization, type-curve match, Blasingame, DN, optional FMB and fracture SRV.

    Parameters
    ----------
    data
        Production history (DatetimeIndex or ``time_col``); rate column required.
    time_col, rate_col, pressure_col, cumulative_col, initial_pressure
        Passed through to normalization / advanced RTA helpers where applicable.
    run_normalization
        If True, runs ``normalize_for_rta_analysis`` when enough columns exist.
    run_type_curve
        If True, runs ``match_type_curve_workflow``.
    run_blasingame, run_dn, run_fmb, run_fracture_network
        Toggle advanced analyses. FMB runs only if ``cumulative_col`` is present in ``data`` when
        ``run_fmb`` is True.

    Returns
    -------
    dict
        Keys among ``normalization``, ``type_curve_match``, ``blasingame``, ``dn_type_curve``,
        ``fmb``, ``fracture_network``, ``diagnostic_plot_data`` (log-log style data for plotting).
    """
    logger.info(
        "Starting enhanced_rta_study rows=%s rate_col=%s",
        len(data),
        rate_col,
    )

    out: dict[str, Any] = {}

    if run_normalization:
        try:
            out["normalization"] = normalize_for_rta_analysis(
                data,
                time_col=time_col,
                rate_col=rate_col,
                pressure_col=pressure_col,
                cumulative_col=cumulative_col,
                initial_pressure=initial_pressure,
            )
        except Exception as e:
            logger.warning("RTA normalization step failed: %s", e)
            out["normalization"] = None

    if run_type_curve:
        try:
            out["type_curve_match"] = match_type_curve_workflow(
                data, time_col=time_col, rate_col=rate_col
            )
        except Exception as e:
            logger.warning("Type curve match failed: %s", e)
            out["type_curve_match"] = None

    if run_blasingame:
        try:
            out["blasingame"] = analyze_blasingame(
                data,
                time_col=time_col,
                rate_col=rate_col,
                pressure_col=pressure_col,
                initial_pressure=initial_pressure or _DEFAULT_RTA_INITIAL_PRESSURE_PSI,
            )
        except Exception as e:
            logger.warning("Blasingame analysis failed: %s", e)
            out["blasingame"] = None

    if run_dn:
        try:
            out["dn_type_curve"] = analyze_dn_type_curve(
                data,
                time_col=time_col,
                rate_col=rate_col,
                pressure_col=pressure_col,
                initial_pressure=initial_pressure or _DEFAULT_RTA_INITIAL_PRESSURE_PSI,
            )
        except Exception as e:
            logger.warning("DN type curve failed: %s", e)
            out["dn_type_curve"] = None

    if run_fmb:
        if cumulative_col is None or cumulative_col not in data.columns:
            logger.warning(
                "Skipping FMB: cumulative_col %r not in data columns",
                cumulative_col,
            )
            out["fmb"] = None
        else:
            try:
                out["fmb"] = analyze_fmb(
                    data,
                    time_col=time_col,
                    cumulative_col=cumulative_col,
                    pressure_col=pressure_col or "pressure",
                    initial_pressure=initial_pressure or _DEFAULT_RTA_INITIAL_PRESSURE_PSI,
                )
            except Exception as e:
                logger.warning("FMB analysis failed: %s", e)
                out["fmb"] = None

    if run_fracture_network:
        try:
            out["fracture_network"] = analyze_fracture_network(
                data,
                time_col=time_col,
                rate_col=rate_col,
                number_of_stages=fracture_number_of_stages,
                stage_spacing=fracture_stage_spacing,
            )
        except Exception as e:
            logger.warning("Fracture network analysis failed: %s", e)
            out["fracture_network"] = None

    try:
        if time_col is None and isinstance(data.index, pd.DatetimeIndex):
            t = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        elif time_col is not None:
            t = data[time_col].to_numpy()
        else:
            t = np.arange(len(data), dtype=float)
        r = data[rate_col].to_numpy()
        out["diagnostic_plot_data"] = generate_diagnostic_plot_data(
            t, r, plot_type="log_log"
        )
    except Exception as e:
        logger.warning("Diagnostic plot data generation failed: %s", e)
        out["diagnostic_plot_data"] = None

    out["metadata"] = {
        "rows": len(data),
        "rate_col": rate_col,
        "pressure_col": pressure_col,
        "flags": {
            "normalization": run_normalization,
            "type_curve": run_type_curve,
            "blasingame": run_blasingame,
            "dn": run_dn,
            "fmb": run_fmb,
            "fracture_network": run_fracture_network,
        },
    }
    return out
