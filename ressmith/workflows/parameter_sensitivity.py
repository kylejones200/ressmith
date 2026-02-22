"""Parameter sensitivity analysis workflows.

Provides workflows for analyzing parameter sensitivity in history matching
and model fitting.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.objects.domain import HistoryMatchResult

logger = logging.getLogger(__name__)


def analyze_parameter_sensitivity(
    base_params: dict[str, float],
    objective_function: Any,
    param_ranges: dict[str, tuple[float, float]] | None = None,
    n_points: int = 10,
) -> pd.DataFrame:
    """Analyze sensitivity of objective function to parameter variations.

    Parameters
    ----------
    base_params : dict
        Base parameter values
    objective_function : callable
        Function that takes parameter dict and returns objective value
    param_ranges : dict, optional
        Parameter ranges to test (if None, uses ±20% of base values)
    n_points : int
        Number of points to test per parameter (default: 10)

    Returns
    -------
    pd.DataFrame
        DataFrame with sensitivity analysis results

    Examples
    --------
    >>> def obj_func(params):
    ...     return (params['qi'] - 1000)**2 + (params['di'] - 0.1)**2
    >>> base = {'qi': 1000, 'di': 0.1}
    >>> sensitivity = analyze_parameter_sensitivity(base, obj_func)
    >>> print(sensitivity.head())
    """
    logger.info("Analyzing parameter sensitivity")

    if param_ranges is None:
        param_ranges = {}
        for param_name, param_value in base_params.items():
            if param_value > 0:
                param_ranges[param_name] = (
                    param_value * 0.8,
                    param_value * 1.2,
                )

    results = []

    for param_name, (min_val, max_val) in param_ranges.items():
        test_values = np.linspace(min_val, max_val, n_points)

        for test_value in test_values:
            test_params = base_params.copy()
            test_params[param_name] = test_value

            try:
                objective_value = objective_function(test_params)
            except Exception as e:
                logger.warning(f"Error evaluating objective for {param_name}={test_value}: {e}")
                objective_value = float("inf")

            variation_pct = (
                (test_value / base_params[param_name] - 1) * 100
                if base_params[param_name] > 0
                else 0.0
            )

            results.append(
                {
                    "parameter": param_name,
                    "value": test_value,
                    "variation_pct": variation_pct,
                    "objective_value": objective_value,
                }
            )

    return pd.DataFrame(results)


def calculate_sensitivity_coefficients(
    sensitivity_results: pd.DataFrame,
    base_objective: float,
) -> pd.DataFrame:
    """Calculate sensitivity coefficients from sensitivity analysis.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        Results from sensitivity analysis
    base_objective : float
        Base objective function value

    Returns
    -------
    pd.DataFrame
        DataFrame with sensitivity coefficients

    Examples
    --------
    >>> sensitivity = analyze_parameter_sensitivity(base_params, obj_func)
    >>> coefficients = calculate_sensitivity_coefficients(sensitivity, base_obj=100.0)
    >>> print(coefficients)
    """
    logger.info("Calculating sensitivity coefficients")

    coefficients = []

    for param_name in sensitivity_results["parameter"].unique():
        param_data = sensitivity_results[sensitivity_results["parameter"] == param_name]

        # Calculate sensitivity coefficient: d(objective)/d(parameter)
        if len(param_data) >= 2:
            # Linear regression: objective = a + b * parameter
            values = param_data["value"].values
            objectives = param_data["objective_value"].values

            if len(values) >= 2:
                coeffs = np.polyfit(values, objectives, 1)
                sensitivity_coeff = coeffs[0]  # Slope

                # Normalized sensitivity
                base_value = values[len(values) // 2]
                normalized_sensitivity = (
                    sensitivity_coeff * base_value / base_objective
                    if base_objective > 0
                    else 0.0
                )

                coefficients.append(
                    {
                        "parameter": param_name,
                        "sensitivity_coefficient": float(sensitivity_coeff),
                        "normalized_sensitivity": float(normalized_sensitivity),
                    }
                )

    return pd.DataFrame(coefficients)


def identify_critical_parameters(
    sensitivity_results: pd.DataFrame,
    threshold: float = 0.1,
) -> list[str]:
    """Identify critical parameters from sensitivity analysis.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        Results from sensitivity analysis
    threshold : float
        Threshold for normalized sensitivity (default: 0.1)

    Returns
    -------
    list
        List of critical parameter names

    Examples
    --------
    >>> sensitivity = analyze_parameter_sensitivity(base_params, obj_func)
    >>> coefficients = calculate_sensitivity_coefficients(sensitivity, base_obj=100.0)
    >>> critical = identify_critical_parameters(coefficients, threshold=0.1)
    >>> print(f"Critical parameters: {critical}")
    """
    logger.info(f"Identifying critical parameters: threshold={threshold}")

    if "normalized_sensitivity" not in sensitivity_results.columns:
        # Calculate coefficients first
        base_obj = sensitivity_results["objective_value"].median()
        coefficients = calculate_sensitivity_coefficients(sensitivity_results, base_obj)
    else:
        coefficients = sensitivity_results

    critical = coefficients[
        coefficients["normalized_sensitivity"].abs() > threshold
    ]["parameter"].tolist()

    return critical
