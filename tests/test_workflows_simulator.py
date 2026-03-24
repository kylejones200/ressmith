"""Tests for workflows.simulator module."""

from pathlib import Path

import pandas as pd
import pytest

from ressmith.workflows.simulator import (
    compare_simulation_to_forecast,
    export_for_simulator,
    import_simulator_output,
    import_simulation_results,
)


def test_export_for_simulator_csv(tmp_path):
    """export_for_simulator writes CSV with production data."""
    df = pd.DataFrame(
        {"oil": [100, 95, 90]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    out_path = tmp_path / "well_data.csv"

    export_for_simulator(df, out_path, format="csv")

    assert out_path.exists()
    loaded = pd.read_csv(out_path, index_col=0, parse_dates=True)
    assert "oil" in loaded.columns
    assert len(loaded) == 3


def test_export_for_simulator_with_pressure(tmp_path):
    """export_for_simulator includes pressure data when provided."""
    prod = pd.DataFrame(
        {"oil": [100, 95, 90]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    pressure = pd.DataFrame(
        {"pressure": [5000, 4800, 4600]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    out_path = tmp_path / "with_pressure.csv"

    export_for_simulator(prod, out_path, pressure_data=pressure, format="csv")

    assert out_path.exists()
    loaded = pd.read_csv(out_path, index_col=0, parse_dates=True)
    assert "pressure_pressure" in loaded.columns or "pressure" in loaded.columns


def test_import_simulator_output_csv(tmp_path):
    """import_simulator_output reads CSV format."""
    df = pd.DataFrame(
        {"oil": [100, 95, 90]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    csv_path = tmp_path / "sim_output.csv"
    df.to_csv(csv_path)

    result = import_simulator_output(csv_path, format="csv")

    assert "production" in result
    assert len(result["production"]) == 3


def test_import_simulator_output_file_not_found():
    """import_simulator_output raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        import_simulator_output("/nonexistent/sim.rst")


def test_import_simulation_results_csv(tmp_path):
    """import_simulation_results reads CSV format."""
    df = pd.DataFrame(
        {"oil": [100, 95, 90]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    csv_path = tmp_path / "sim_results.csv"
    df.to_csv(csv_path)

    result = import_simulation_results(csv_path, format="csv")

    assert "production" in result
    assert len(result["production"]) == 3


def test_import_simulation_results_file_not_found():
    """import_simulation_results raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        import_simulation_results("/nonexistent/results.csv")


def test_compare_simulation_to_forecast():
    """compare_simulation_to_forecast returns metrics dict."""
    idx = pd.date_range("2020-01-01", periods=5, freq="ME")
    sim = pd.DataFrame({"oil": [100, 95, 90, 85, 80]}, index=idx)
    forecast = pd.DataFrame({"oil": [98, 94, 89, 84, 79]}, index=idx)

    metrics = compare_simulation_to_forecast(sim, forecast)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "mape" in metrics
    assert "r_squared" in metrics
    assert metrics["rmse"] >= 0


def test_compare_simulation_to_forecast_no_common_index():
    """compare_simulation_to_forecast raises when no common index."""
    sim = pd.DataFrame(
        {"oil": [100, 95]},
        index=pd.date_range("2020-01-01", periods=2, freq="ME"),
    )
    forecast = pd.DataFrame(
        {"oil": [90, 85]},
        index=pd.date_range("2025-01-01", periods=2, freq="ME"),
    )

    with pytest.raises(ValueError, match="common.*time"):
        compare_simulation_to_forecast(sim, forecast)
