"""
Uncertainty quantification for decline curve forecasts.

Provides Monte Carlo simulation and probabilistic forecasting capabilities.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from ressmith.objects.domain import ForecastResult, ForecastSpec
from ressmith.primitives.base import BaseDeclineModel


def monte_carlo_forecast(
    model: BaseDeclineModel,
    spec: ForecastSpec,
    n_samples: int = 1000,
    param_uncertainty: Optional[dict[str, tuple[float, float]]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate probabilistic forecast using Monte Carlo simulation.

    Samples from parameter uncertainty to generate forecast distribution.

    Parameters
    ----------
    model : BaseDeclineModel
        Fitted decline model
    spec : ForecastSpec
        Forecast specification
    n_samples : int
        Number of Monte Carlo samples (default: 1000)
    param_uncertainty : dict, optional
        Dictionary mapping parameter names to (std, distribution_type) tuples.
        If None, uses default uncertainty assumptions.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with:
        - 'forecast': ForecastResult with median forecast
        - 'p10': 10th percentile forecast
        - 'p50': 50th percentile (median) forecast
        - 'p90': 90th percentile forecast
        - 'samples': array of all forecast samples
        - 'metadata': uncertainty metadata
    """
    if not model.is_fitted:
        raise ValueError("Model must be fitted before Monte Carlo simulation")

    if seed is not None:
        np.random.seed(seed)

    # Get base parameters
    if not hasattr(model, "_fitted_params"):
        raise ValueError("Model does not have fitted parameters")

    base_params = model._fitted_params.copy()

    # Default uncertainty if not specified
    if param_uncertainty is None:
        param_uncertainty = {}
        for param_name in base_params.keys():
            # Default: 10% coefficient of variation
            param_value = base_params[param_name]
            std = abs(param_value) * 0.1
            param_uncertainty[param_name] = (std, "normal")

    # Generate parameter samples
    param_samples = []
    for _ in range(n_samples):
        sample_params = base_params.copy()
        for param_name, (std, dist_type) in param_uncertainty.items():
            if param_name not in sample_params:
                continue

            if dist_type == "normal":
                # Normal distribution
                sample_value = np.random.normal(base_params[param_name], std)
            elif dist_type == "lognormal":
                # Log-normal distribution (for positive parameters)
                log_mean = np.log(max(base_params[param_name], 1e-6))
                log_std = std / base_params[param_name] if base_params[param_name] > 0 else 0.1
                sample_value = np.exp(np.random.normal(log_mean, log_std))
            else:
                # Default to normal
                sample_value = np.random.normal(base_params[param_name], std)

            # Clip to reasonable bounds
            if param_name in ["qi", "di", "a", "tau"]:
                sample_value = max(0.0, sample_value)
            elif param_name == "b":
                sample_value = max(0.0, min(1.0, sample_value))
            elif param_name == "beta":
                sample_value = max(0.0, min(1.0, sample_value))

            sample_params[param_name] = sample_value

        param_samples.append(sample_params)

    # Generate forecasts for each parameter sample
    forecast_samples = []
    base_forecast = model.predict(spec)
    forecast_index = base_forecast.yhat.index

    for sample_params in param_samples:
        # Create temporary model with sampled parameters
        temp_model = model.__class__(**{})
        temp_model._fitted_params = sample_params
        temp_model._start_date = model._start_date if hasattr(model, "_start_date") else None
        temp_model._fitted = True

        try:
            sample_forecast = temp_model.predict(spec)
            forecast_samples.append(sample_forecast.yhat.values)
        except Exception:
            # If prediction fails, use base forecast
            forecast_samples.append(base_forecast.yhat.values)

    forecast_samples = np.array(forecast_samples)

    # Compute percentiles
    p10 = np.percentile(forecast_samples, 10, axis=0)
    p50 = np.percentile(forecast_samples, 50, axis=0)
    p90 = np.percentile(forecast_samples, 90, axis=0)

    # Create result dictionary
    result = {
        "forecast": ForecastResult(
            yhat=pd.Series(p50, index=forecast_index, name="forecast_p50"),
            metadata={"uncertainty_method": "monte_carlo", "n_samples": n_samples},
        ),
        "p10": pd.Series(p10, index=forecast_index, name="forecast_p10"),
        "p50": pd.Series(p50, index=forecast_index, name="forecast_p50"),
        "p90": pd.Series(p90, index=forecast_index, name="forecast_p90"),
        "samples": forecast_samples,
        "metadata": {
            "n_samples": n_samples,
            "param_uncertainty": param_uncertainty,
            "seed": seed,
        },
    }

    return result

