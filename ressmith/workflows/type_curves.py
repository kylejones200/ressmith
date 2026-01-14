"""Type curve matching workflows for RTA analysis.

Provides workflows for matching production data to type curves,
a standard RTA workflow.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ressmith.primitives.type_curves import (
    calculate_type_curve_statistics,
    generate_arps_type_curve,
    match_multiple_type_curves,
    match_type_curve,
    normalize_production_data,
)
from ressmith.utils.errors import ERR_INSUFFICIENT_DATA, format_error

logger = logging.getLogger(__name__)


def match_type_curve_workflow(
    data: pd.DataFrame,
    time_col: str | None = None,
    rate_col: str = "oil",
    b_values: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Match production data to type curves.

    Parameters
    ----------
    data : pd.DataFrame
        Production data with datetime index or time column
    time_col : str, optional
        Time column name (if None, uses index)
    rate_col : str
        Rate column name (default: 'oil')
    b_values : np.ndarray, optional
        Array of b-values to test (if None, uses default range)
    **kwargs
        Additional parameters

    Returns
    -------
    dict
        Dictionary with match results:
        - matches: Dictionary of type curve matches
        - best_match: Best match (lowest error)
        - statistics: Match statistics

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> time = np.array([1, 10, 30, 60, 90, 120])
    >>> rate = np.array([1000, 800, 600, 500, 450, 400])
    >>> df = pd.DataFrame({'time': time, 'oil': rate})
    >>> result = match_type_curve_workflow(df, time_col='time')
    >>> print(f"Best match b: {result['best_match']['matched_params']['b']:.2f}")
    """
    logger.info("Starting type curve matching workflow")

    if time_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            time = np.array(
                [(data.index[i] - data.index[0]).days for i in range(len(data))]
            )
        else:
            time = np.arange(len(data))
    else:
        time = data[time_col].values

    rate = data[rate_col].values

    matches = match_multiple_type_curves(time, rate)

    if len(matches) == 0:
        raise ValueError(format_error(ERR_INSUFFICIENT_DATA, min_points=5, analysis="type curve matching"))
    best_match_name = min(matches.keys(), key=lambda k: matches[k].match_error)
    best_match = matches[best_match_name]
    statistics = calculate_type_curve_statistics(best_match, time, rate)

    logger.info(f"Type curve matching completed. Best match: {best_match_name}")

    return {
        "matches": {
            name: {
                "matched_params": match.matched_params,
                "match_error": match.match_error,
                "correlation": match.correlation,
            }
            for name, match in matches.items()
        },
        "best_match": {
            "type": best_match_name,
            "matched_params": best_match.matched_params,
            "match_error": best_match.match_error,
            "correlation": best_match.correlation,
        },
        "statistics": statistics,
    }


def generate_type_curve_library(
    b_values: np.ndarray | None = None,
    qi_values: np.ndarray | None = None,
    di_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate library of type curves for analysis.

    Parameters
    ----------
    b_values : np.ndarray, optional
        Array of b-values (if None, uses default range)
    qi_values : np.ndarray, optional
        Array of qi values (if None, uses normalized)
    di_values : np.ndarray, optional
        Array of di values (if None, uses normalized)

    Returns
    -------
    dict
        Dictionary with type curve library

    Examples
    --------
    >>> library = generate_type_curve_library()
    >>> print(f"Type curves generated: {len(library['curves'])}")
    """
    logger.info("Generating type curve library")

    if b_values is None:
        b_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    curves = {}
    for b in b_values:
        t_norm, q_norm = generate_arps_type_curve(b=b)
        curves[f"b_{b:.1f}"] = {
            "b": float(b),
            "time_normalized": t_norm,
            "rate_normalized": q_norm,
        }

    return {
        "curves": curves,
        "b_values": b_values.tolist(),
    }

