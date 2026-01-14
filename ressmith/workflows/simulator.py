"""Simulator integration workflows for history matching.

Provides workflows for exporting data for reservoir simulators
and importing simulation results for comparison.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ressmith.utils.errors import ERR_NO_COMMON_TIME_INDEX, ERR_UNSUPPORTED_FORMAT, format_error

logger = logging.getLogger(__name__)


def export_for_simulator(
    production_data: pd.DataFrame,
    output_path: str | Path,
    pressure_data: pd.DataFrame | None = None,
    format: str = "csv",
    well_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Export production and pressure data for reservoir simulator input.

    Exports data in formats compatible with common reservoir simulators.

    Parameters
    ----------
    production_data : pd.DataFrame
        Production data with datetime index
    pressure_data : pd.DataFrame, optional
        Pressure data with datetime index
    output_path : str or Path
        Output file path
    format : str
        Export format ('csv', 'json') (default: 'csv')
    well_id : str, optional
        Well identifier
    **kwargs
        Additional export options

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'oil': [100, 95, 90, 85]}, index=pd.date_range('2020-01-01', periods=4))
    >>> export_for_simulator(df, output_path='well_data.csv')
    """
    logger.info(f"Exporting data for simulator: format={format}")

    output_path = Path(output_path)

    if format == "csv":
        export_df = production_data.copy()
        export_df.index.name = "date"

        if pressure_data is not None:
            for col in pressure_data.columns:
                export_df[f"pressure_{col}"] = pressure_data[col]

        if well_id:
            export_df["well_id"] = well_id

        export_df.to_csv(output_path)

    elif format == "json":
        # Export as JSON
        export_dict = {
            "production": production_data.to_dict(orient="index"),
        }
        if pressure_data is not None:
            export_dict["pressure"] = pressure_data.to_dict(orient="index")

        if well_id:
            export_dict["well_id"] = well_id

        import json

        with open(output_path, "w") as f:
            json.dump(export_dict, f, indent=2, default=str)

    else:
        raise ValueError(format_error(ERR_UNSUPPORTED_FORMAT, format=format, supported="csv, json"))

    logger.info(f"Exported data to {output_path}")


def import_simulation_results(
    file_path: str | Path,
    format: str = "csv",
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Import simulation results for comparison.

    Imports simulation results from common formats.

    Parameters
    ----------
    file_path : str or Path
        Input file path
    format : str
        Import format ('csv', 'json') (default: 'csv')
    **kwargs
        Additional import options

    Returns
    -------
    dict
        Dictionary with DataFrames:
        - production: Production data
        - pressure: Pressure data (if available)
        - metadata: Additional metadata (if available)

    Examples
    --------
    >>> results = import_simulation_results('simulation_results.csv')
    >>> print(results['production'].head())
    """
    logger.info(f"Importing simulation results: format={format}")

    file_path = Path(file_path)

    if format == "csv":
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        production_cols = [col for col in df.columns if "pressure" not in col.lower()]
        pressure_cols = [col for col in df.columns if "pressure" in col.lower()]

        production_data = df[production_cols]
        pressure_data = df[pressure_cols] if pressure_cols else None

        results = {"production": production_data}
        if pressure_data is not None:
            results["pressure"] = pressure_data

    elif format == "json":
        import json

        with open(file_path, "r") as f:
            data = json.load(f)

        production_data = pd.DataFrame(data["production"]).T
        production_data.index = pd.to_datetime(production_data.index)

        results = {"production": production_data}

        if "pressure" in data:
            pressure_data = pd.DataFrame(data["pressure"]).T
            pressure_data.index = pd.to_datetime(pressure_data.index)
            results["pressure"] = pressure_data

        if "metadata" in data:
            results["metadata"] = data["metadata"]

    else:
        raise ValueError(format_error(ERR_UNSUPPORTED_FORMAT, format=format, supported="csv, json"))

    logger.info(f"Imported simulation results from {file_path}")
    return results


def compare_simulation_to_forecast(
    simulation_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    phase: str = "oil",
) -> dict[str, float]:
    """Compare simulation results to forecast.

    Calculates comparison metrics between simulation and forecast.

    Parameters
    ----------
    simulation_data : pd.DataFrame
        Simulation production data
    forecast_data : pd.DataFrame
        Forecast production data
    phase : str
        Phase to compare (default: 'oil')

    Returns
    -------
    dict
        Dictionary with comparison metrics:
        - rmse: Root mean square error
        - mae: Mean absolute error
        - mape: Mean absolute percentage error
        - r_squared: R² coefficient

    Examples
    --------
    >>> sim_data = pd.DataFrame({'oil': [100, 95, 90]}, index=pd.date_range('2020-01-01', periods=3))
    >>> forecast_data = pd.DataFrame({'oil': [98, 94, 89]}, index=pd.date_range('2020-01-01', periods=3))
    >>> metrics = compare_simulation_to_forecast(sim_data, forecast_data)
    >>> print(f"RMSE: {metrics['rmse']:.2f}")
    """
    logger.info("Comparing simulation to forecast")

    # Align data
    common_index = simulation_data.index.intersection(forecast_data.index)

    if len(common_index) == 0:
        raise ValueError(ERR_NO_COMMON_TIME_INDEX)

    sim_values = simulation_data.loc[common_index, phase].values
    forecast_values = forecast_data.loc[common_index, phase].values

    # Calculate metrics
    residuals = sim_values - forecast_values

    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    # MAPE
    valid_mask = sim_values > 0
    if valid_mask.any():
        mape = np.mean(np.abs(residuals[valid_mask] / sim_values[valid_mask])) * 100
    else:
        mape = float("inf")

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sim_values - np.mean(sim_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r_squared": float(r_squared),
    }

