"""Batch processing module for reservoir engineering analysis.

This module provides deterministic batch processing capabilities for
fitting and forecasting across many wells, with support for parallelization
and reproducible results.

Features:
- Deterministic batch runner with fixed seeds
- Joblib parallelism
- Manifest-based input
- Parquet output support
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ressmith.workflows.core import fit_forecast
from ressmith.workflows.io import read_csv_production

logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning(
        "joblib not available. Install with: pip install joblib. "
        "Parallel batch processing will be unavailable."
    )


@dataclass
class BatchManifest:
    """Manifest for batch processing.

    Attributes:
        wells: List of well entries with paths and metadata
    """

    wells: list[dict[str, Any]]

    @classmethod
    def from_file(cls, filepath: str) -> "BatchManifest":
        """Load manifest from YAML or JSON file."""
        import json

        import yaml

        path = Path(filepath)
        with open(path) as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(wells=data.get("wells", []))

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {"wells": self.wells}


def process_single_well(
    well_id: str,
    data_path: str,
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Process a single well.

    Args:
        well_id: Well identifier
        data_path: Path to well data file
        model_name: Decline model name (default: 'arps_hyperbolic')
        horizon: Forecast horizon in periods (default: 24)
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for fit_forecast

    Returns:
        Dictionary with well results
    """
    try:
        # Load data
        if data_path.endswith(".csv"):
            df = read_csv_production(data_path)
        else:
            df = pd.read_parquet(data_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

        forecast, params = fit_forecast(
            df, model_name=model_name, horizon=horizon, **kwargs
        )

        # Calculate EUR (sum of forecast)
        eur = float(forecast.yhat.sum()) if len(forecast.yhat) > 0 else 0.0

        return {
            "well_id": well_id,
            "success": True,
            "params": params,
            "eur": eur,
            "forecast_length": len(forecast.yhat),
        }
    except Exception as e:
        logger.warning(f"Failed to process well {well_id}: {e}")
        return {
            "well_id": well_id,
            "success": False,
            "error": str(e),
        }


def batch_fit(
    manifest: BatchManifest | str,
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    output_dir: str = "batch_output",
    n_jobs: int = -1,
    seed: int = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run batch fitting on multiple wells.

    Args:
        manifest: BatchManifest or path to manifest file
        model_name: Decline model name (default: 'arps_hyperbolic')
        horizon: Forecast horizon in periods (default: 24)
        output_dir: Output directory for results
        n_jobs: Number of parallel jobs (-1 for all cores)
        seed: Base random seed (each well gets seed + well_index)
        **kwargs: Additional parameters for fit_forecast

    Returns:
        DataFrame with one row per well containing fit results
    """
    # Load manifest
    if isinstance(manifest, str):
        manifest = BatchManifest.from_file(manifest)

    logger.info(
        f"Starting batch fit: {len(manifest.wells)} wells",
        extra={"n_wells": len(manifest.wells), "n_jobs": n_jobs},
    )

    # Process wells
    if JOBLIB_AVAILABLE and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_well)(
                well["well_id"],
                well["data_path"],
                model_name=model_name,
                horizon=horizon,
                seed=seed + i if seed else None,
                **kwargs,
            )
            for i, well in enumerate(manifest.wells)
        )
    else:
        results = [
            process_single_well(
                well["well_id"],
                well["data_path"],
                model_name=model_name,
                horizon=horizon,
                seed=seed + i if seed else None,
                **kwargs,
            )
            for i, well in enumerate(manifest.wells)
        ]

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save to Parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "batch_results.parquet"
    df_results.to_parquet(output_file, index=False)

    n_successful = int(df_results["success"].sum()) if "success" in df_results.columns else 0
    logger.info(
        f"Batch fit complete: {n_successful}/{len(df_results)} successful",
        extra={
            "n_successful": n_successful,
            "output_file": str(output_file),
        },
    )

    return df_results
