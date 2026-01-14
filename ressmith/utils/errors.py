"""Standardized error messages for ResSmith.

This module provides constants and utilities for consistent error messaging
across the codebase.
"""

ERR_MODEL_NOT_FITTED = "Model not fitted. Call fit() before predict()."
ERR_INSUFFICIENT_DATA = "Insufficient valid data for {analysis}. Minimum {min_points} data points required."
ERR_UNSUPPORTED_FORMAT = "Unsupported format: {format}. Supported formats: {supported}."
ERR_NO_COMMON_TIME_INDEX = "No common time index between datasets."
ERR_TIME_INDEX_DUPLICATES = "Time index contains duplicate values."
ERR_TIME_INDEX_NOT_MONOTONIC = "Time index must be strictly monotonic increasing."
ERR_DATA_MISMATCH = "Data length mismatch: {name1} length {len1} does not match {name2} length {len2}."
ERR_NEGATIVE_VALUES = "{name} contains {count} negative values. Minimum value: {min_val:.6f}."
ERR_MISSING_COLUMNS = "Missing required columns: {missing}. Available columns: {available}."
ERR_UNSUPPORTED_DATA_TYPE = "Unsupported data type: {type}. Expected one of: {expected}."


def format_error(template: str, **kwargs: str | int | float) -> str:
    """Format an error message template with keyword arguments.

    Parameters
    ----------
    template : str
        Error message template with {placeholders}
    **kwargs
        Values to fill in placeholders

    Returns
    -------
    str
        Formatted error message

    Examples
    --------
    >>> format_error(ERR_INSUFFICIENT_DATA, min_points=5)
    'Insufficient valid data for analysis. Minimum 5 data points required.'
    """
    return template.format(**kwargs)

