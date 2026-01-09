"""Integration tests for ressmith with timesmith.typing."""

import numpy as np
import pandas as pd
import pytest

from timesmith.typing import SeriesLike
from timesmith.typing.validators import assert_series_like

from ressmith import fit_forecast


def test_timesmith_typing_import():
    """Test that timesmith.typing can be imported."""
    from timesmith.typing import SeriesLike, PanelLike
    from timesmith.typing.validators import assert_series_like, assert_panel_like

    assert SeriesLike is not None
    assert PanelLike is not None
    assert assert_series_like is not None
    assert assert_panel_like is not None


def test_validate_series_like_with_ressmith():
    """Test that timesmith validators work with ressmith inputs."""
    # Create a valid pandas Series
    time_index = pd.date_range("2020-01-01", periods=12, freq="M")
    values = np.array([100.0 - i * 2.0 for i in range(12)])
    series = pd.Series(values, index=time_index, name="oil")

    # Validate using timesmith.typing
    assert_series_like(series, name="production_data")

    # Convert to DataFrame and use with ressmith
    df = series.to_frame(name="oil")
    forecast, params = fit_forecast(df, model_name="arps_hyperbolic", horizon=6)

    # Verify results
    assert forecast is not None
    assert len(forecast.yhat) == 6
    assert params is not None
    assert "qi" in params
    assert "di" in params


def test_integration_example_runs():
    """Test that the integration example can be imported and run."""
    # Import the example module
    import sys
    from pathlib import Path

    example_path = Path(__file__).parent.parent / "examples" / "integration_timesmith.py"
    if not example_path.exists():
        pytest.skip("Integration example not found")

    # Run the example in a subprocess to ensure it works end-to-end
    import subprocess

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Example failed: {result.stderr}"


def test_no_circular_imports():
    """Test that there are no circular imports between ressmith and timesmith."""
    # Import ressmith
    import ressmith

    # Import timesmith
    import timesmith

    # Verify they can coexist
    assert ressmith is not None
    assert timesmith is not None

    # Verify ressmith doesn't import timesmith internals incorrectly
    import ressmith.objects.domain
    import ressmith.workflows.core

    # Check that we're importing from timesmith.typing, not timesmith internals
    from timesmith.typing import SeriesLike
    assert SeriesLike is not None

