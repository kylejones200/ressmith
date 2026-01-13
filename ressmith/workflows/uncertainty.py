"""
Uncertainty quantification workflows.

Provides workflows for probabilistic forecasting and risk analysis.
"""

import logging
from typing import Any

import pandas as pd

from ressmith.objects.domain import ForecastSpec
from ressmith.primitives.uncertainty import monte_carlo_forecast
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


def probabilistic_forecast(
    data: pd.DataFrame,
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    n_samples: int = 1000,
    param_uncertainty: dict[str, tuple[float, str]] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generate probabilistic forecast with uncertainty quantification.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index
    model_name : str
        Decline model name (default: 'arps_hyperbolic')
    horizon : int
        Forecast horizon in periods (default: 24)
    n_samples : int
        Number of Monte Carlo samples (default: 1000)
    param_uncertainty : dict, optional
        Dictionary mapping parameter names to (std, distribution_type) tuples.
        distribution_type can be 'normal' or 'lognormal'.
    seed : int, optional
        Random seed for reproducibility
    **kwargs
        Additional parameters for model

    Returns
    -------
    dict
        Dictionary with:
        - 'forecast': ForecastResult with median (p50) forecast
        - 'p10': 10th percentile forecast Series
        - 'p50': 50th percentile forecast Series
        - 'p90': 90th percentile forecast Series
        - 'samples': array of all forecast samples
        - 'metadata': uncertainty metadata

    Examples
    --------
    >>> from ressmith import probabilistic_forecast
    >>>
    >>> # Generate probabilistic forecast
    >>> result = probabilistic_forecast(
    ...     data,
    ...     model_name='arps_hyperbolic',
    ...     horizon=36,
    ...     n_samples=2000,
    ...     param_uncertainty={
    ...         'qi': (50.0, 'normal'),  # 50 std for qi
    ...         'di': (0.05, 'normal'),  # 0.05 std for di
    ...         'b': (0.1, 'normal')     # 0.1 std for b
    ...     }
    ... )
    >>> print(f"P50 EUR: {result['p50'].sum():.0f}")
    >>> print(f"P10 EUR: {result['p10'].sum():.0f}")
    >>> print(f"P90 EUR: {result['p90'].sum():.0f}")
    """
    logger.info(
        f"Generating probabilistic forecast: model={model_name}, "
        f"horizon={horizon}, n_samples={n_samples}"
    )

    # Fit model
    forecast, params = fit_forecast(
        data, model_name=model_name, horizon=horizon, **kwargs
    )

    from ressmith.primitives.models import (
        ArpsExponentialModel,
        ArpsHarmonicModel,
        ArpsHyperbolicModel,
        DuongModel,
        FixedTerminalDeclineModel,
        HyperbolicToExponentialSwitchModel,
        LinearDeclineModel,
        PowerLawDeclineModel,
        StretchedExponentialModel,
    )
    from ressmith.tasks.core import FitDeclineTask

    model_map = {
        "arps_exponential": ArpsExponentialModel,
        "arps_hyperbolic": ArpsHyperbolicModel,
        "arps_harmonic": ArpsHarmonicModel,
        "linear_decline": LinearDeclineModel,
        "hyperbolic_to_exponential": HyperbolicToExponentialSwitchModel,
        "power_law": PowerLawDeclineModel,
        "duong": DuongModel,
        "stretched_exponential": StretchedExponentialModel,
        "fixed_terminal_decline": FixedTerminalDeclineModel,
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_map[model_name](**kwargs)
    task = FitDeclineTask(model=model, phase=kwargs.get("phase", "oil"))
    fitted_model, _ = task.run(data, horizon=None)

    forecast_spec = ForecastSpec(horizon=horizon, frequency="D")
    result = monte_carlo_forecast(
        fitted_model,
        forecast_spec,
        n_samples=n_samples,
        param_uncertainty=param_uncertainty,
        seed=seed,
    )

    logger.info("Probabilistic forecast completed")
    return result
