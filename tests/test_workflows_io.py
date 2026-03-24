"""Tests for workflows.io module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ressmith.workflows.io import read_csv_production, write_csv_results


def test_read_csv_production_success(tmp_path):
    """read_csv_production reads valid CSV and returns DataFrame with datetime index."""
    df = pd.DataFrame(
        {"date": ["2020-01-01", "2020-02-01", "2020-03-01"], "oil": [100, 95, 90]}
    )
    csv_path = tmp_path / "production.csv"
    df.to_csv(csv_path, index=False)

    result = read_csv_production(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert isinstance(result.index, pd.DatetimeIndex)
    assert list(result.columns) == ["oil"]
    assert len(result) == 3


def test_read_csv_production_file_not_found():
    """read_csv_production raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        read_csv_production("/nonexistent/path/data.csv")


def test_read_csv_production_missing_time_column(tmp_path):
    """read_csv_production raises ValueError when time column missing."""
    df = pd.DataFrame({"oil": [100, 95, 90]})
    csv_path = tmp_path / "no_date.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Time column 'date' not found"):
        read_csv_production(csv_path)


def test_read_csv_production_accepts_path_object(tmp_path):
    """read_csv_production accepts Path objects."""
    df = pd.DataFrame({"date": ["2020-01-01"], "oil": [100]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    result = read_csv_production(Path(csv_path))
    assert len(result) == 1


def test_write_csv_results_dataframe(tmp_path):
    """write_csv_results writes DataFrame to CSV."""
    df = pd.DataFrame(
        {"oil": [100, 95, 90]},
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    out_path = tmp_path / "out.csv"

    write_csv_results(df, out_path)

    assert out_path.exists()
    loaded = pd.read_csv(out_path, index_col=0, parse_dates=True)
    assert len(loaded) == 3
    assert list(loaded.columns) == list(df.columns)


def test_write_csv_results_dict_with_yhat(tmp_path):
    """write_csv_results converts dict with yhat to DataFrame and writes."""
    series = pd.Series([100, 95, 90], index=pd.date_range("2020-01-01", periods=3))
    out_path = tmp_path / "out.csv"

    write_csv_results({"yhat": series}, out_path)

    assert out_path.exists()
    loaded = pd.read_csv(out_path, index_col=0, parse_dates=True)
    assert "yhat" in loaded.columns or len(loaded.columns) >= 1


def test_write_csv_results_dict_without_yhat(tmp_path):
    """write_csv_results converts simple dict to DataFrame and writes."""
    out_path = tmp_path / "out.csv"

    write_csv_results({"eur": 1000, "npv": 50000}, out_path)

    assert out_path.exists()


def test_write_csv_results_invalid_type():
    """write_csv_results raises ValueError for unsupported types."""
    with pytest.raises(ValueError, match="Cannot write type"):
        write_csv_results([1, 2, 3], "/tmp/out.csv")


def test_read_csv_production_path_traversal_rejected(tmp_path):
    """read_csv_production raises PermissionError when path escapes base_dir."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    df = pd.DataFrame({"date": ["2020-01-01"], "oil": [100]})
    csv_path = tmp_path / "data.csv"  # File is in tmp_path, outside allowed_dir
    df.to_csv(csv_path, index=False)

    # Path under base_dir should work when file is in allowed dir
    in_allowed = allowed_dir / "data.csv"
    df.to_csv(in_allowed, index=False)
    assert len(read_csv_production(in_allowed, base_dir=allowed_dir)) == 1

    # Path escaping base_dir (file in parent) should be rejected
    with pytest.raises(PermissionError, match="not under allowed base"):
        read_csv_production(csv_path, base_dir=allowed_dir)


def test_write_csv_results_path_traversal_rejected(tmp_path):
    """write_csv_results raises PermissionError when path escapes base_dir."""
    df = pd.DataFrame({"oil": [100]}, index=pd.date_range("2020-01-01", periods=1))
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()

    # Path under base_dir should work
    out_path = allowed_dir / "out.csv"
    write_csv_results(df, out_path, base_dir=allowed_dir)
    assert out_path.exists()

    # Path escaping base_dir (write to parent) should be rejected
    escape_path = allowed_dir / ".." / "out.csv"
    with pytest.raises(PermissionError, match="not under allowed base"):
        write_csv_results(df, escape_path, base_dir=allowed_dir)
