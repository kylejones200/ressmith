"""
I/O helpers for workflows.

This module provides functions for reading/writing production data files.
For loading multiple files, see workflows.data_utils.load_production_csvs.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_production(
    filepath: str | Path,
    time_column: str = "date",
    rate_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Read a single production data CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    time_column : str
        Name of time column (default: 'date')
    rate_columns : dict, optional
        Mapping of phase names to column names (not currently used, reserved for future)

    Returns
    -------
    pd.DataFrame
        Production data with datetime index

    See Also
    --------
    load_production_csvs : Load multiple CSV files and stack them
    """
    logger.info(f"Reading production data from {filepath}")
    df = pd.read_csv(filepath)
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
    else:
        raise ValueError(f"Time column '{time_column}' not found")

    return df


def write_csv_results(
    results: pd.DataFrame | dict,
    filepath: str | Path,
    **kwargs: Any,
) -> None:
    """
    Write results to CSV.

    Parameters
    ----------
    results : pd.DataFrame or dict
        Results to write
    filepath : str or Path
        Output path
    **kwargs
        Additional arguments to pd.DataFrame.to_csv
    """
    logger.info(f"Writing results to {filepath}")
    if isinstance(results, dict):
        if "yhat" in results:
            results = results["yhat"].to_frame()
        else:
            results = pd.DataFrame([results])

    if isinstance(results, pd.DataFrame):
        results.to_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Cannot write type {type(results)} to CSV")
