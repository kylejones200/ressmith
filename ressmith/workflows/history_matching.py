"""History matching for material balance and decline curve models.

This module provides systematic parameter optimization to match historical
production and pressure data.

Features:
- Material balance history matching
- Parameter optimization using scipy.optimize
- Uncertainty quantification
- Sensitivity analysis
- Multiple history match scenarios
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from ressmith.objects.domain import HistoryMatchResult
from ressmith.primitives.material_balance import (
    GasReservoirParams,
    SolutionGasDriveParams,
    gas_reservoir_pz_method,
    solution_gas_drive_material_balance,
)

logger = logging.getLogger(__name__)


def history_match_material_balance(
    time: np.ndarray,
    production: np.ndarray,
    pressure: np.ndarray | None = None,
    drive_mechanism: str = "solution_gas",
    initial_params: dict[str, float] | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
    method: str = "differential_evolution",
) -> HistoryMatchResult:
    """History match material balance to production and pressure data.

    Optimizes material balance parameters to match historical data.

    Args:
        time: Time array (days)
        production: Cumulative production (STB)
        pressure: Optional pressure array (psi)
        drive_mechanism: Drive mechanism ('solution_gas', 'water_drive', 'gas_reservoir')
        initial_params: Initial parameter guess
        param_bounds: Parameter bounds for optimization
        weights: Weights for objective function ('production', 'pressure')
        method: Optimization method ('differential_evolution', 'minimize')

    Returns:
        HistoryMatchResult with optimized parameters

    Example:
        >>> time = np.array([30, 60, 90, 120, 150, 180])
        >>> production = np.array([10000, 20000, 30000, 40000, 50000, 60000])
        >>> pressure = np.array([5000, 4800, 4600, 4400, 4200, 4000])
        >>> result = history_match_material_balance(time, production, pressure)
    """
    if len(time) != len(production):
        raise ValueError("time and production must have same length")

    if pressure is not None and len(pressure) != len(time):
        raise ValueError("pressure must have same length as time")

    if weights is None:
        weights = {"production": 1.0, "pressure": 1.0 if pressure is not None else 0.0}

    if param_bounds is None:
        param_bounds = {
            "N": (1e5, 1e7),  # OOIP
            "pi": (1000.0, 10000.0),  # Initial pressure
            "D": (1e-6, 0.1),  # Decline rate
        }

    if initial_params is None:
        initial_params = {
            "N": np.max(production) * 10,
            "pi": pressure[0] if pressure is not None else 5000.0,
            "D": 0.001,
        }

    def objective_function(params_array: np.ndarray) -> float:
        """Objective function for optimization."""
        # Convert array to parameter dict
        param_names = list(param_bounds.keys())
        params_dict = dict(zip(param_names, params_array))

        # Calculate material balance
        try:
            if drive_mechanism == "solution_gas":
                mb_params = SolutionGasDriveParams(
                    N=params_dict.get("N", 1e6),
                    pi=params_dict.get("pi", 5000.0),
                    pb=params_dict.get("pb", 3000.0),
                )

                # Calculate cumulative for each time step
                calculated_production = np.zeros_like(time)
                for i, (t_i, p_i) in enumerate(
                    zip(
                        time,
                        (
                            pressure
                            if pressure is not None
                            else [params_dict.get("pi", 5000.0)] * len(time)
                        ),
                    )
                ):
                    result = solution_gas_drive_material_balance(
                        p_i, calculated_production[i - 1] if i > 0 else 0.0, mb_params
                    )
                    calculated_production[i] = result["Np_calculated"]

            elif drive_mechanism == "gas_reservoir":
                calculated_production = np.zeros_like(time)
                G = params_dict.get("G", 1e9)
                for i, p_i in enumerate(
                    pressure
                    if pressure is not None
                    else [params_dict.get("pi", 5000.0)] * len(time)
                ):
                    result = gas_reservoir_pz_method(
                        p_i,
                        calculated_production[i - 1] if i > 0 else 0.0,
                        GasReservoirParams(G=G, pi=params_dict.get("pi", 5000.0)),
                    )
                    calculated_production[i] = result["G_calculated"] * 0.1

            else:
                N = params_dict.get("N", 1e6)
                D = params_dict.get("D", 0.001)
                calculated_production = N * (1 - np.exp(-D * time))

            # Calculate errors
            production_error = np.sum((calculated_production - production) ** 2)

            pressure_error = 0.0
            if pressure is not None:
                estimated_pressure = params_dict.get("pi", 5000.0) * np.exp(
                    -params_dict.get("D", 0.001) * time
                )
                pressure_error = np.sum((estimated_pressure - pressure) ** 2)

            # Weighted objective
            objective = (
                weights["production"] * production_error
                + weights["pressure"] * pressure_error
            )

            return objective

        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return 1e10

    # Prepare bounds for optimization
    bounds = [param_bounds[name] for name in param_bounds.keys()]

    # Initial guess
    x0 = [
        initial_params.get(name, (bounds[i][0] + bounds[i][1]) / 2)
        for i, name in enumerate(param_bounds.keys())
    ]

    # Optimize
    if method == "differential_evolution":
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
        )
    else:
        result = minimize(
            objective_function,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )

    # Convert result back to parameter dict
    optimized_params = dict(zip(param_bounds.keys(), result.x))

    if drive_mechanism == "solution_gas":
        mb_params = SolutionGasDriveParams(
            N=optimized_params.get("N", 1e6),
            pi=optimized_params.get("pi", 5000.0),
        )
        calculated_production = np.zeros_like(time)
        pressure_array = (
            pressure
            if pressure is not None
            else np.full(len(time), optimized_params.get("pi", 5000.0))
        )
        for i, p_i in enumerate(pressure_array):
            mb_result = solution_gas_drive_material_balance(
                p_i, calculated_production[i - 1] if i > 0 else 0.0, mb_params
            )
            calculated_production[i] = mb_result["Np_calculated"]
    else:
        N = optimized_params.get("N", 1e6)
        D = optimized_params.get("D", 0.001)
        calculated_production = N * (1 - np.exp(-D * time))

    production_rmse = np.sqrt(np.mean((calculated_production - production) ** 2))
    production_mae = np.mean(np.abs(calculated_production - production))

    pressure_rmse = 0.0
    pressure_mae = 0.0
    if pressure is not None:
        estimated_pressure = optimized_params.get("pi", 5000.0) * np.exp(
            -optimized_params.get("D", 0.001) * time
        )
        pressure_rmse = np.sqrt(np.mean((estimated_pressure - pressure) ** 2))
        pressure_mae = np.mean(np.abs(estimated_pressure - pressure))

    return HistoryMatchResult(
        optimized_params=optimized_params,
        objective_value=result.fun,
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
        iterations=result.nit if hasattr(result, "nit") else 0,
        pressure_match={"rmse": pressure_rmse, "mae": pressure_mae},
        production_match={"rmse": production_rmse, "mae": production_mae},
    )


def quantify_parameter_uncertainty(
    history_match_result: HistoryMatchResult,
    time: np.ndarray,
    production: np.ndarray,
    n_samples: int = 1000,
) -> dict[str, dict[str, float]]:
    """Quantify uncertainty in history-matched parameters.

    Uses Monte Carlo sampling around optimized parameters.

    Args:
        history_match_result: History matching result
        time: Time array (days)
        production: Production data (STB)
        n_samples: Number of Monte Carlo samples

    Returns:
        Dictionary with parameter uncertainty statistics (mean, std, p10, p50, p90)
    """
    optimized = history_match_result.optimized_params

    # Sample parameters around optimized values
    # Use ±20% variation
    samples = {}
    for param_name, param_value in optimized.items():
        if param_value > 0:
            std = param_value * 0.2  # 20% standard deviation
            samples[param_name] = np.random.normal(param_value, std, n_samples)
            samples[param_name] = np.maximum(
                samples[param_name], param_value * 0.1
            )  # Lower bound
        else:
            samples[param_name] = np.full(n_samples, param_value)

    # Calculate statistics
    uncertainty = {}
    for param_name in optimized.keys():
        param_samples = samples[param_name]
        uncertainty[param_name] = {
            "mean": float(np.mean(param_samples)),
            "std": float(np.std(param_samples)),
            "p10": float(np.percentile(param_samples, 10)),
            "p50": float(np.percentile(param_samples, 50)),
            "p90": float(np.percentile(param_samples, 90)),
        }

    return uncertainty


def sensitivity_analysis_material_balance(
    time: np.ndarray,
    production: np.ndarray,
    base_params: dict[str, float],
    param_variations: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Perform sensitivity analysis on material balance parameters.

    Args:
        time: Time array (days)
        production: Production data (STB)
        base_params: Base parameter values
        param_variations: Parameter variations to test (if None, uses ±20%)

    Returns:
        DataFrame with sensitivity results
    """
    if param_variations is None:
        param_variations = {}
        for param_name, param_value in base_params.items():
            if param_value > 0:
                param_variations[param_name] = [
                    param_value * 0.8,
                    param_value * 0.9,
                    param_value,
                    param_value * 1.1,
                    param_value * 1.2,
                ]

    results = []

    for param_name, variations in param_variations.items():
        for variation in variations:
            test_params = base_params.copy()
            test_params[param_name] = variation

            # Calculate material balance
            N = test_params.get("N", 1e6)
            D = test_params.get("D", 0.001)
            calculated = N * (1 - np.exp(-D * time))

            # Calculate error
            error = np.sqrt(np.mean((calculated - production) ** 2))

            results.append(
                {
                    "parameter": param_name,
                    "value": variation,
                    "variation_pct": (variation / base_params[param_name] - 1) * 100,
                    "rmse": error,
                }
            )

    return pd.DataFrame(results)


def calculate_history_match_objective(
    observed_data: pd.DataFrame,
    calculated_data: pd.DataFrame,
    weights: dict[str, float] | None = None,
    objective_type: str = "weighted_rmse",
) -> dict[str, float]:
    """Calculate objective function for history matching.

    Calculates various objective functions for history matching workflows.

    Args:
        observed_data: Observed production/pressure data
        calculated_data: Calculated production/pressure data
        weights: Optional weights for different phases
        objective_type: Objective function type ('weighted_rmse', 'weighted_mae', 'nash_sutcliffe')

    Returns:
        Dictionary with objective function value and component metrics

    Example:
        >>> import pandas as pd
        >>> observed = pd.DataFrame({'oil': [100, 95, 90]}, index=pd.date_range('2020-01-01', periods=3))
        >>> calculated = pd.DataFrame({'oil': [98, 94, 89]}, index=pd.date_range('2020-01-01', periods=3))
        >>> objective = calculate_history_match_objective(observed, calculated)
        >>> print(f"Objective value: {objective['objective_value']:.2f}")
    """
    logger.info(f"Calculating history match objective: type={objective_type}")

    if weights is None:
        weights = {col: 1.0 for col in observed_data.columns}

    # Align data
    common_index = observed_data.index.intersection(calculated_data.index)
    if len(common_index) == 0:
        raise ValueError("No common time index between observed and calculated data")

    observed_aligned = observed_data.loc[common_index]
    calculated_aligned = calculated_data.loc[common_index]

    component_metrics = {}
    weighted_errors = []

    for col in observed_aligned.columns:
        if col not in calculated_aligned.columns:
            continue

        obs = observed_aligned[col].values
        calc = calculated_aligned[col].values

        # Calculate metrics
        residuals = obs - calc
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # MAPE
        valid_mask = obs > 0
        if valid_mask.any():
            mape = np.mean(np.abs(residuals[valid_mask] / obs[valid_mask])) * 100
        else:
            mape = float("inf")

        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        component_metrics[col] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r_squared": float(r_squared),
        }

        # Weighted error
        weight = weights.get(col, 1.0)
        if objective_type == "weighted_rmse":
            weighted_errors.append(weight * rmse)
        elif objective_type == "weighted_mae":
            weighted_errors.append(weight * mae)
        elif objective_type == "nash_sutcliffe":
            # Nash-Sutcliffe efficiency
            nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            weighted_errors.append(
                weight * (1 - nse)
            )  # Convert to error (lower is better)
        else:
            weighted_errors.append(weight * rmse)

    objective_value = (
        sum(weighted_errors) / len(weighted_errors) if weighted_errors else float("inf")
    )

    return {
        "objective_value": float(objective_value),
        "objective_type": objective_type,
        "component_metrics": component_metrics,
    }


def run_parameter_sensitivity_analysis(
    history_match_result: HistoryMatchResult,
    time: np.ndarray,
    production: np.ndarray,
    pressure: np.ndarray | None = None,
    param_variations: dict[str, list[float]] | None = None,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Run parameter sensitivity analysis for history matching.

    Analyzes sensitivity of history match quality to parameter variations.

    Args:
        history_match_result: History matching result
        time: Time array (days)
        production: Production data (STB)
        pressure: Optional pressure data (psi)
        param_variations: Parameter variations to test (if None, uses ±20%)
        n_samples: Number of samples per parameter

    Returns:
        DataFrame with sensitivity analysis results

    Example:
        >>> result = history_match_material_balance(time, production, pressure)
        >>> sensitivity = run_parameter_sensitivity_analysis(
        ...     result, time, production, pressure
        ... )
        >>> print(sensitivity.head())
    """
    logger.info("Running parameter sensitivity analysis")

    optimized_params = history_match_result.optimized_params

    if param_variations is None:
        param_variations = {}
        for param_name, param_value in optimized_params.items():
            if param_value > 0:
                variations = np.linspace(
                    param_value * 0.5, param_value * 1.5, n_samples
                )
                param_variations[param_name] = variations.tolist()

    results = []

    for param_name, variations in param_variations.items():
        for variation in variations:
            test_params = optimized_params.copy()
            test_params[param_name] = variation

            # Calculate objective function with test parameters
            # Simplified: use exponential decline model
            N = test_params.get("N", optimized_params.get("N", 1e6))
            D = test_params.get("D", optimized_params.get("D", 0.001))
            calculated_production = N * (1 - np.exp(-D * time))

            # Calculate error
            production_error = np.sqrt(
                np.mean((calculated_production - production) ** 2)
            )

            pressure_error = 0.0
            if pressure is not None:
                pi = test_params.get("pi", optimized_params.get("pi", 5000.0))
                estimated_pressure = pi * np.exp(-D * time)
                pressure_error = np.sqrt(np.mean((estimated_pressure - pressure) ** 2))

            total_error = production_error + pressure_error

            results.append(
                {
                    "parameter": param_name,
                    "value": variation,
                    "variation_pct": (
                        (variation / optimized_params[param_name] - 1) * 100
                        if optimized_params[param_name] > 0
                        else 0.0
                    ),
                    "production_error": production_error,
                    "pressure_error": pressure_error,
                    "total_error": total_error,
                }
            )

    return pd.DataFrame(results)
