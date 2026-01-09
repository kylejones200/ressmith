"""
Multi-phase forecasting workflows using yield models.

Provides workflows for forecasting associated phases (gas, water) from
primary phase (oil) using yield models (GOR, CGR, water cut).
"""

import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import ForecastResult, ForecastSpec
from ressmith.primitives.yields import (
    constant_yield_rate,
    declining_yield_rate,
    fit_constant_yield,
    fit_declining_yield,
    hyperbolic_yield_rate,
)
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


def forecast_with_yields(
    data: pd.DataFrame,
    primary_phase: str = "oil",
    associated_phases: list[str] = ["gas", "water"],
    model_name: str = "arps_hyperbolic",
    yield_models: Optional[dict[str, Literal["constant", "declining", "hyperbolic"]]] = None,
    horizon: int = 24,
    **kwargs: Any,
) -> dict[str, ForecastResult]:
    """
    Forecast primary phase and associated phases using yield models.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index and phase columns
    primary_phase : str
        Primary phase to forecast: 'oil' or 'gas' (default: 'oil')
    associated_phases : list
        Associated phases to forecast from primary: ['gas', 'water'] (default)
    model_name : str
        Decline model name for primary phase (default: 'arps_hyperbolic')
    yield_models : dict, optional
        Dictionary mapping phase name to yield model type.
        If None, auto-fits yield models from historical data.
    horizon : int
        Forecast horizon in periods (default: 24)
    **kwargs
        Additional parameters for decline model

    Returns
    -------
    dict
        Dictionary mapping phase names to ForecastResult objects

    Examples
    --------
    >>> from ressmith import forecast_with_yields
    >>> 
    >>> # Forecast oil, gas, and water
    >>> results = forecast_with_yields(
    ...     data,
    ...     primary_phase='oil',
    ...     associated_phases=['gas', 'water'],
    ...     model_name='arps_hyperbolic',
    ...     horizon=36
    ... )
    >>> oil_forecast = results['oil']
    >>> gas_forecast = results['gas']
    """
    logger.info(
        f"Forecasting {primary_phase} with associated phases {associated_phases}"
    )

    # Forecast primary phase
    primary_forecast, primary_params = fit_forecast(
        data, model_name=model_name, horizon=horizon, phase=primary_phase, **kwargs
    )

    results: dict[str, ForecastResult] = {primary_phase: primary_forecast}

    # Prepare time array for yield calculations
    time_index = primary_forecast.yhat.index
    if isinstance(time_index, pd.DatetimeIndex):
        start_date = time_index[0]
        t_forecast = (time_index - start_date).days.values.astype(float)
    else:
        t_forecast = np.arange(len(time_index))

    primary_rates = primary_forecast.yhat.values

    # Forecast associated phases using yield models
    if yield_models is None:
        yield_models = {}

    for phase in associated_phases:
        if phase not in data.columns:
            logger.warning(f"Phase '{phase}' not in data, skipping")
            continue

        # Fit yield model from historical data if not specified
        if phase not in yield_models:
            # Auto-detect yield model type (default to constant)
            yield_model_type = "constant"
        else:
            yield_model_type = yield_models[phase]

        # Fit yield model from historical data
        primary_hist = data[primary_phase].values
        phase_hist = data[phase].values

        # Filter valid data
        valid_mask = (primary_hist > 0) & (phase_hist >= 0)
        if not np.any(valid_mask):
            logger.warning(f"No valid data for {phase} yield model, using zero forecast")
            phase_forecast = pd.Series(
                np.zeros(len(primary_rates)), index=time_index, name=phase
            )
            results[phase] = ForecastResult(
                yhat=phase_forecast,
                metadata={"yield_model": "zero", "yield_params": {}},
            )
            continue

        primary_valid = primary_hist[valid_mask]
        phase_valid = phase_hist[valid_mask]
        t_hist = np.arange(len(primary_valid))

        # Fit yield model
        if yield_model_type == "constant":
            yield_params = fit_constant_yield(t_hist, primary_valid, phase_valid)
            phase_rates = constant_yield_rate(primary_rates, yield_params["yield_ratio"])
        elif yield_model_type == "declining":
            yield_params = fit_declining_yield(t_hist, primary_valid, phase_valid)
            phase_rates = declining_yield_rate(
                t_forecast,
                primary_rates,
                yield_params["yield_initial"],
                yield_params["decline_rate"],
            )
        elif yield_model_type == "hyperbolic":
            # For hyperbolic, use declining as approximation
            yield_params = fit_declining_yield(t_hist, primary_valid, phase_valid)
            # Approximate hyperbolic with declining for now
            phase_rates = declining_yield_rate(
                t_forecast,
                primary_rates,
                yield_params["yield_initial"],
                yield_params["decline_rate"],
            )
            yield_params["model_type"] = "hyperbolic"
        else:
            logger.warning(f"Unknown yield model type '{yield_model_type}', using constant")
            yield_params = fit_constant_yield(t_hist, primary_valid, phase_valid)
            phase_rates = constant_yield_rate(primary_rates, yield_params["yield_ratio"])

        # Create forecast result
        phase_forecast = pd.Series(phase_rates, index=time_index, name=phase)
        results[phase] = ForecastResult(
            yhat=phase_forecast,
            metadata={
                "yield_model": yield_model_type,
                "yield_params": yield_params,
                "primary_phase": primary_phase,
            },
        )

    logger.info(f"Completed multi-phase forecast: {list(results.keys())}")
    return results

