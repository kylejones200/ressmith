"""
Core workflow functions for ResSmith.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Literal, Optional

import pandas as pd

from ressmith.objects.domain import (
    EconResult,
    EconSpec,
    ForecastResult,
    ForecastSpec,
)
from ressmith.primitives.base import BaseDeclineModel
from ressmith.primitives.models import (
    ArpsExponentialModel,
    ArpsHarmonicModel,
    ArpsHyperbolicModel,
    DuongModel,
    HyperbolicToExponentialSwitchModel,
    LinearDeclineModel,
    PowerLawDeclineModel,
    SegmentedDeclineModel,
    StretchedExponentialModel,
)
from ressmith.tasks.core import BatchTask, EconTask, FitDeclineTask, ForecastTask

logger = logging.getLogger(__name__)


def fit_forecast(
    data: pd.DataFrame,
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    **kwargs: Any,
) -> tuple[ForecastResult, dict[str, Any]]:
    """
    Fit a decline model and generate forecast.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_name : str
        Model name: 'arps_exponential', 'arps_hyperbolic', 'arps_harmonic'
    horizon : int
        Forecast horizon in periods
    **kwargs
        Additional parameters for model or task

    Returns
    -------
    tuple
        (forecast_result, params_dict)
    """
    logger.info(f"Starting fit_forecast with model={model_name}, horizon={horizon}")

    # Create model
    model_map = {
        "arps_exponential": ArpsExponentialModel,
        "arps_hyperbolic": ArpsHyperbolicModel,
        "arps_harmonic": ArpsHarmonicModel,
        "linear_decline": LinearDeclineModel,
        "hyperbolic_to_exponential": HyperbolicToExponentialSwitchModel,
        "power_law": PowerLawDeclineModel,
        "duong": DuongModel,
        "stretched_exponential": StretchedExponentialModel,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_map[model_name](**kwargs)

    # Create and run fit task
    task = FitDeclineTask(model=model, phase=kwargs.get("phase", "oil"))
    fitted_model, forecast_result = task.run(data, horizon=horizon)

    # Extract parameters
    params = {}
    if hasattr(fitted_model, "_fitted_params"):
        params = fitted_model._fitted_params.copy()

    logger.info(f"Completed fit_forecast. Params: {params}")
    return forecast_result, params


def forecast_many(
    well_ids: list[str],
    loader: Callable[[str], pd.DataFrame],
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    parallel: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Forecast multiple wells.

    Parameters
    ----------
    well_ids : list
        List of well IDs
    loader : callable
        Function that takes well_id and returns DataFrame
    model_name : str
        Model name
    horizon : int
        Forecast horizon
    parallel : bool
        Enable parallel execution
    **kwargs
        Additional parameters

    Returns
    -------
    pd.DataFrame
        Results table with well_id, fit_params, eur, npv, success, error
    """
    logger.info(f"Starting forecast_many for {len(well_ids)} wells")

    def process_well(well_id: str) -> dict[str, Any]:
        """Process single well."""
        try:
            data = loader(well_id)
            forecast, params = fit_forecast(
                data, model_name=model_name, horizon=horizon, **kwargs
            )
            # Compute EUR (simplified: sum of forecast)
            eur = forecast.yhat.sum() if len(forecast.yhat) > 0 else 0.0
            return {
                "fit_params": params,
                "eur": eur,
                "npv": None,  # Would need econ spec
            }
        except Exception as e:
            logger.error(f"Error processing {well_id}: {e}")
            raise

    # Create and run batch task
    batch_task = BatchTask(process_well, parallel=parallel)
    results = batch_task.run(well_ids)

    logger.info(f"Completed forecast_many. Success: {results['success'].sum()}/{len(results)}")
    return results


def evaluate_economics(
    forecast: ForecastResult, spec: EconSpec
) -> EconResult:
    """
    Evaluate economics for a forecast.

    Parameters
    ----------
    forecast : ForecastResult
        Forecast results
    spec : EconSpec
        Economics specification

    Returns
    -------
    EconResult
        Economics results
    """
    logger.info("Starting evaluate_economics")
    task = EconTask()
    result, summary = task.run(forecast, spec)
    logger.info(f"Completed evaluate_economics. NPV: {result.npv:.2f}")
    return result


def full_run(
    data: pd.DataFrame,
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    econ_spec: Optional[EconSpec] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Full workflow: fit, forecast, and evaluate economics.

    Parameters
    ----------
    data : pd.DataFrame
        Production data
    model_name : str
        Model name
    horizon : int
        Forecast horizon
    econ_spec : EconSpec, optional
        Economics specification
    **kwargs
        Additional parameters

    Returns
    -------
    dict
        Result bundle with forecast, params, econ_result (if spec provided)
    """
    logger.info("Starting full_run")
    # Fit and forecast
    forecast, params = fit_forecast(
        data, model_name=model_name, horizon=horizon, **kwargs
    )

    result = {
        "forecast": forecast,
        "params": params,
    }

    # Evaluate economics if spec provided
    if econ_spec is not None:
        econ_result = evaluate_economics(forecast, econ_spec)
        result["econ_result"] = econ_result

    logger.info("Completed full_run")
    return result


def fit_segmented_forecast(
    data: pd.DataFrame,
    segment_dates: list[tuple[datetime, datetime]],
    kinds: Optional[list[Literal["exponential", "harmonic", "hyperbolic"]]] = None,
    horizon: int = 24,
    enforce_continuity: bool = True,
    **kwargs: Any,
) -> tuple[Any, list[Any], list[str]]:
    """
    Fit a segmented decline model and generate forecast.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    segment_dates : list
        List of (start_date, end_date) tuples for each segment
    kinds : list, optional
        ARPS decline types for each segment (default: all 'hyperbolic')
    horizon : int
        Forecast horizon in periods (default: 24)
    enforce_continuity : bool
        Enforce rate continuity at segment boundaries (default: True)
    **kwargs
        Additional parameters

    Returns
    -------
    tuple
        (forecast_result, segments, continuity_errors)
    """
    logger.info(
        f"Starting fit_segmented_forecast with {len(segment_dates)} segments, "
        f"horizon={horizon}"
    )

    # Create model
    model = SegmentedDeclineModel(
        segment_dates=segment_dates,
        kinds=kinds,
        enforce_continuity=enforce_continuity,
        **kwargs,
    )

    # Fit model
    fitted_model = model.fit(data)

    # Generate forecast
    forecast_spec = ForecastSpec(horizon=horizon, frequency="D")
    forecast_result = fitted_model.predict(forecast_spec)

    # Get segments and errors
    segments = fitted_model.get_segments()
    continuity_errors = fitted_model.get_continuity_errors()

    logger.info(
        f"Completed fit_segmented_forecast. Segments: {len(segments)}, "
        f"Errors: {len(continuity_errors)}"
    )

    return forecast_result, segments, continuity_errors

