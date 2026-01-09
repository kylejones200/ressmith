"""
Parameter constraints and bounds for decline curve models.
"""

from typing import Any, Optional, Tuple

import numpy as np


def get_default_bounds(model_name: str) -> dict[str, Tuple[float, Optional[float]]]:
    """
    Get default parameter bounds for a model.

    Parameters
    ----------
    model_name : str
        Model name: 'exponential', 'hyperbolic', 'harmonic', etc.

    Returns
    -------
    dict
        Dictionary mapping parameter names to (lower, upper) bounds.
        None for upper bound means no upper limit.
    """
    bounds: dict[str, Tuple[float, Optional[float]]] = {}

    if model_name == "exponential":
        bounds = {
            "qi": (0.01, None),  # Must be positive
            "di": (1e-6, 1.0),  # Reasonable decline rate
        }
    elif model_name == "hyperbolic":
        bounds = {
            "qi": (0.01, None),
            "di": (1e-6, 1.0),
            "b": (0.01, 0.99),  # b-factor between 0 and 1 (exclusive)
        }
    elif model_name == "harmonic":
        bounds = {
            "qi": (0.01, None),
            "di": (1e-6, 1.0),
        }
    elif model_name == "linear":
        bounds = {
            "q0": (0.01, None),
            "m": (0.0, None),  # Slope can be any positive value
        }
    elif model_name == "power_law":
        bounds = {
            "qi": (0.01, None),
            "di": (1e-6, 10.0),
            "n": (0.1, 2.0),  # Power law exponent
        }
    elif model_name == "duong":
        bounds = {
            "qi": (0.01, None),
            "a": (1e-6, 10.0),  # Duong parameter a
            "m": (0.1, 2.0),  # Duong parameter m
        }
    elif model_name == "stretched_exponential":
        bounds = {
            "qi": (0.01, None),
            "tau": (1.0, 1000.0),  # Characteristic time
            "beta": (0.1, 1.0),  # Stretch parameter
        }
    else:
        # Default bounds for unknown models
        bounds = {
            "qi": (0.01, None),
            "di": (1e-6, 1.0),
        }

    return bounds


def validate_parameters(
    params: dict[str, float],
    bounds: dict[str, Tuple[float, Optional[float]]],
    model_name: str = "unknown",
) -> list[str]:
    """
    Validate parameters against bounds.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    bounds : dict
        Bounds dictionary mapping parameter names to (lower, upper) tuples
    model_name : str
        Model name for error messages

    Returns
    -------
    list
        List of warning messages (empty if all valid)
    """
    warnings: list[str] = []

    for param_name, value in params.items():
        if param_name not in bounds:
            continue  # Skip parameters without bounds

        lower, upper = bounds[param_name]

        if value < lower:
            warnings.append(
                f"{model_name}: {param_name}={value:.4f} below lower bound {lower:.4f}"
            )
        elif upper is not None and value > upper:
            warnings.append(
                f"{model_name}: {param_name}={value:.4f} above upper bound {upper:.4f}"
            )

    return warnings


def clip_parameters(
    params: dict[str, float],
    bounds: dict[str, Tuple[float, Optional[float]]],
) -> dict[str, float]:
    """
    Clip parameters to bounds.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    bounds : dict
        Bounds dictionary

    Returns
    -------
    dict
        Clipped parameter dictionary
    """
    clipped = params.copy()

    for param_name, value in clipped.items():
        if param_name not in bounds:
            continue

        lower, upper = bounds[param_name]
        clipped[param_name] = max(lower, value)
        if upper is not None:
            clipped[param_name] = min(clipped[param_name], upper)

    return clipped


def get_scipy_bounds(
    bounds: dict[str, Tuple[float, Optional[float]]],
    param_order: list[str],
) -> list[Tuple[float, Optional[float]]]:
    """
    Convert bounds dictionary to scipy format.

    Parameters
    ----------
    bounds : dict
        Bounds dictionary
    param_order : list
        Order of parameters for scipy

    Returns
    -------
    list
        List of (lower, upper) tuples in param_order
    """
    scipy_bounds = []
    for param in param_order:
        if param in bounds:
            scipy_bounds.append(bounds[param])
        else:
            # Default bounds if not specified
            scipy_bounds.append((0.0, None))
    return scipy_bounds

