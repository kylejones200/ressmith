"""Benchmark scripts for measuring throughput and error rates.

This module provides tools for benchmarking reservoir engineering analysis
on known datasets to measure performance and validate results.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ressmith.workflows.analysis import estimate_eur
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Attributes:
        dataset_name: Name of the dataset
        n_wells: Number of wells processed
        n_records: Total number of records
        elapsed_time: Elapsed time in seconds
        throughput_wells_per_sec: Throughput in wells per second
        throughput_records_per_sec: Throughput in records per second
        success_rate: Percentage of wells that succeeded
        error_count: Number of errors encountered
        errors: List of error messages
    """

    dataset_name: str
    n_wells: int
    n_records: int
    elapsed_time: float
    throughput_wells_per_sec: float
    throughput_records_per_sec: float
    success_rate: float
    error_count: int
    errors: list[str]


def benchmark_fit_forecast(
    df: pd.DataFrame,
    well_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil",
    model_name: str = "arps_hyperbolic",
    horizon: int = 12,
    dataset_name: str = "unknown",
) -> BenchmarkResult:
    """
    Benchmark fit_forecast function.

    Args:
        df: Production DataFrame
        well_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        model_name: Model to use
        horizon: Forecast horizon
        dataset_name: Name of dataset for reporting

    Returns:
        BenchmarkResult with performance metrics

    Example:
        >>> from ressmith.workflows.benchmarks import benchmark_fit_forecast
        >>> result = benchmark_fit_forecast(df, dataset_name='Pennsylvania')
        >>> print(f"Throughput: {result.throughput_wells_per_sec:.1f} wells/sec")
    """
    n_wells = df[well_col].nunique()
    n_records = len(df)
    errors = []

    start_time = time.time()

    try:
        successful_wells = 0
        for well_id, well_data in df.groupby(well_col):
            try:
                # Prepare data for fit_forecast
                well_df = well_data.set_index(date_col)
                well_df = well_df[[value_col]].rename(columns={value_col: "oil"})

                _ = fit_forecast(well_df, model_name=model_name, horizon=horizon)
                successful_wells += 1
            except Exception as e:
                errors.append(f"{well_id}: {str(e)}")

        elapsed_time = time.time() - start_time
        success_rate = (successful_wells / n_wells * 100) if n_wells > 0 else 0

    except Exception as e:
        elapsed_time = time.time() - start_time
        errors.append(str(e))
        successful_wells = 0
        success_rate = 0.0

    throughput_wells = n_wells / elapsed_time if elapsed_time > 0 else 0
    throughput_records = n_records / elapsed_time if elapsed_time > 0 else 0

    return BenchmarkResult(
        dataset_name=dataset_name,
        n_wells=n_wells,
        n_records=n_records,
        elapsed_time=elapsed_time,
        throughput_wells_per_sec=throughput_wells,
        throughput_records_per_sec=throughput_records,
        success_rate=success_rate,
        error_count=len(errors),
        errors=errors,
    )


def benchmark_eur_calculation(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil",
    dataset_name: str = "unknown",
) -> BenchmarkResult:
    """
    Benchmark EUR calculation.

    Args:
        df: Production DataFrame
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value
        dataset_name: Name of dataset for reporting

    Returns:
        BenchmarkResult with performance metrics
    """
    n_wells = df[well_id_col].nunique()
    n_records = len(df)
    errors = []

    start_time = time.time()

    try:
        successful_wells = 0
        for well_id, well_data in df.groupby(well_id_col):
            try:
                # Prepare data for estimate_eur
                well_df = well_data.set_index(date_col)
                well_df = well_df[[value_col]].rename(columns={value_col: "oil"})

                _ = estimate_eur(well_df)
                successful_wells += 1
            except Exception as e:
                errors.append(f"{well_id}: {str(e)}")

        elapsed_time = time.time() - start_time
        success_rate = (successful_wells / n_wells * 100) if n_wells > 0 else 0

    except Exception as e:
        elapsed_time = time.time() - start_time
        errors.append(str(e))
        successful_wells = 0
        success_rate = 0.0

    throughput_wells = n_wells / elapsed_time if elapsed_time > 0 else 0
    throughput_records = n_records / elapsed_time if elapsed_time > 0 else 0

    return BenchmarkResult(
        dataset_name=dataset_name,
        n_wells=n_wells,
        n_records=n_records,
        elapsed_time=elapsed_time,
        throughput_wells_per_sec=throughput_wells,
        throughput_records_per_sec=throughput_records,
        success_rate=success_rate,
        error_count=len(errors),
        errors=errors,
    )


def benchmark_single_well(
    production: pd.Series,
    model_name: str = "arps_hyperbolic",
    horizon: int = 12,
    n_iterations: int = 100,
) -> dict[str, float]:
    """
    Benchmark single well function with multiple iterations.

    Args:
        production: Production time series
        model_name: Model to use
        horizon: Forecast horizon
        n_iterations: Number of iterations to run

    Returns:
        Dictionary with performance metrics
    """
    times = []

    # Prepare data
    data = pd.DataFrame({"oil": production})

    for _ in range(n_iterations):
        start = time.time()
        try:
            _ = fit_forecast(data, model_name=model_name, horizon=horizon)
        except Exception:
            pass
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "median_time": np.median(times),
        "throughput_per_sec": 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
    }


def run_benchmark_suite(
    datasets: list[dict[str, Any]], output_path: str | None = None
) -> pd.DataFrame:
    """
    Run a suite of benchmarks on multiple datasets.

    Args:
        datasets: List of dataset dictionaries with 'name', 'df', and optional config
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with benchmark results

    Example:
        >>> datasets = [
        ...     {'name': 'Pennsylvania', 'df': pa_df},
        ...     {'name': 'New Mexico', 'df': nm_df},
        ... ]
        >>> results = run_benchmark_suite(datasets, 'benchmark_results.csv')
    """
    results = []

    for dataset in datasets:
        name = dataset["name"]
        df = dataset["df"]
        config = dataset.get("config", {})

        logger.info(f"Benchmarking dataset: {name}")

        # Benchmark fit_forecast
        batch_result = benchmark_fit_forecast(df, dataset_name=name, **config)
        results.append(
            {
                "dataset": name,
                "operation": "fit_forecast",
                "n_wells": batch_result.n_wells,
                "n_records": batch_result.n_records,
                "elapsed_time": batch_result.elapsed_time,
                "throughput_wells_per_sec": batch_result.throughput_wells_per_sec,
                "throughput_records_per_sec": batch_result.throughput_records_per_sec,
                "success_rate": batch_result.success_rate,
                "error_count": batch_result.error_count,
            }
        )

        # Benchmark EUR calculation
        eur_result = benchmark_eur_calculation(df, dataset_name=name, **config)
        results.append(
            {
                "dataset": name,
                "operation": "eur_calculation",
                "n_wells": eur_result.n_wells,
                "n_records": eur_result.n_records,
                "elapsed_time": eur_result.elapsed_time,
                "throughput_wells_per_sec": eur_result.throughput_wells_per_sec,
                "throughput_records_per_sec": eur_result.throughput_records_per_sec,
                "success_rate": eur_result.success_rate,
                "error_count": eur_result.error_count,
            }
        )

    results_df = pd.DataFrame(results)

    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Benchmark results saved to {output_path}")

    return results_df


def print_benchmark_summary(results_df: pd.DataFrame) -> None:
    """Print a formatted summary of benchmark results.

    Args:
        results_df: DataFrame with benchmark results
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for dataset in results_df["dataset"].unique():
        dataset_results = results_df[results_df["dataset"] == dataset]
        print(f"\nDataset: {dataset}")
        print("-" * 80)

        for _, row in dataset_results.iterrows():
            print(f"\n  Operation: {row['operation']}")
            print(f"    Wells: {row['n_wells']:,}")
            print(f"    Records: {row['n_records']:,}")
            print(f"    Time: {row['elapsed_time']:.2f}s")
            print(f"    Throughput: {row['throughput_wells_per_sec']:.1f} wells/sec")
            print(f"    Success Rate: {row['success_rate']:.1f}%")
            if row["error_count"] > 0:
                print(f"    Errors: {row['error_count']}")

    print("\n" + "=" * 80)
