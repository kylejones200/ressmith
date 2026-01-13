"""Integration tests for ressmith with timesmith.typing."""

import numpy as np
import pandas as pd
import pytest

try:
    from timesmith.typing import SeriesLike
    from timesmith.typing.validators import assert_series_like
    HAS_TIMESMITH_TYPING = True
except ImportError:
    HAS_TIMESMITH_TYPING = False
    SeriesLike = None
    assert_series_like = None

from ressmith import fit_forecast


@pytest.mark.skipif(not HAS_TIMESMITH_TYPING, reason="timesmith.typing not available")
def test_timesmith_typing_import():
    """Test that timesmith.typing can be imported."""
    from timesmith.typing import SeriesLike, PanelLike
    from timesmith.typing.validators import assert_series_like, assert_panel_like

    assert SeriesLike is not None
    assert PanelLike is not None
    assert assert_series_like is not None
    assert assert_panel_like is not None


@pytest.mark.skipif(not HAS_TIMESMITH_TYPING, reason="timesmith.typing not available")
def test_validate_series_like_with_ressmith():
    """Test that timesmith validators work with ressmith inputs."""
    # Create a valid pandas Series
    time_index = pd.date_range("2020-01-01", periods=12, freq="M")
    values = np.array([100.0 - i * 2.0 for i in range(12)])
    series = pd.Series(values, index=time_index, name="oil")

    # Validate using timesmith.typing
    if assert_series_like:
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


@pytest.mark.skipif(not HAS_TIMESMITH, reason="timesmith not available or missing dependencies")
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

    # Allow example to run even if timesmith.typing not available (it has fallback)
    assert result.returncode == 0, f"Example failed: {result.stderr}"


@pytest.mark.skipif(not HAS_TIMESMITH, reason="timesmith not available or missing dependencies")
def test_no_circular_imports():
    """Test that there are no circular imports between ressmith and timesmith."""
    # Import ressmith
    import ressmith

    # Verify ressmith works
    assert ressmith is not None

    # Verify ressmith doesn't import timesmith internals incorrectly
    import ressmith.objects.domain
    import ressmith.workflows.core

    # Try to import timesmith (may not be available)
    try:
        import timesmith
        assert timesmith is not None
        
        # If timesmith available, check typing import
        try:
            from timesmith.typing import SeriesLike
            assert SeriesLike is not None
        except ImportError:
            # timesmith.typing not available yet, that's okay
            pass
    except ImportError:
        # timesmith not installed, that's okay for this test
        pass

