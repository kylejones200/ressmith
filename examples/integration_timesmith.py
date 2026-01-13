"""
Integration example demonstrating ressmith with timesmith.typing.

This example proves that ressmith works correctly with timesmith's shared typing.
It validates pandas Series inputs using timesmith.typing validators.
"""

import numpy as np
import pandas as pd

# Import from timesmith.typing (shared typing)
try:
    from timesmith.typing import SeriesLike
    from timesmith.typing.validators import assert_series_like
    HAS_TIMESMITH_TYPING = True
except (ImportError, AttributeError):
    # Fallback if timesmith.typing not available or has missing dependencies
    # AttributeError can occur when timesmith imports fail due to missing optional deps (e.g., networkx)
    HAS_TIMESMITH_TYPING = False
    print("Warning: timesmith.typing not available, using basic validation")

# Import ressmith workflows
from ressmith import fit_forecast


def validate_and_forecast(data: SeriesLike, model_name: str = "arps_hyperbolic") -> tuple:
    """
    Validate input using timesmith.typing, then fit and forecast.

    This demonstrates the integration between ressmith and timesmith.
    """
    # Validate input using timesmith validators (if available)
    if HAS_TIMESMITH_TYPING:
        assert_series_like(data, name="production_data")
    else:
        # Basic validation fallback
        if not isinstance(data, pd.Series):
            raise ValueError("data must be pandas Series")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data index must be DatetimeIndex")

    # Convert to DataFrame for ressmith (ressmith expects DataFrame with datetime index)
    if isinstance(data, pd.Series):
        df = data.to_frame(name="oil")
    else:
        # Already a DataFrame, ensure it has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

    # Fit and forecast using ressmith
    forecast, params = fit_forecast(df, model_name=model_name, horizon=12)

    return forecast, params


def main():
    """Run integration example."""
    print("ResSmith + TimeSmith Integration Example")
    print("=" * 50)

    # Generate synthetic production data
    time_index = pd.date_range("2020-01-01", periods=24, freq="M")
    t = np.arange(24)
    qi = 100.0
    di = 0.1
    b = 0.6
    q = qi / (1.0 + b * di * t) ** (1.0 / b)
    noise = np.random.normal(0, 2.0, len(q))
    q_noisy = np.maximum(q + noise, 0.1)

    # Create pandas Series (timesmith.typing.SeriesLike)
    production_series = pd.Series(q_noisy, index=time_index, name="oil_production")

    print(f"\nInput: pandas Series with {len(production_series)} points")
    print(f"Index type: {type(production_series.index).__name__}")
    print(f"First value: {production_series.iloc[0]:.2f}")
    print(f"Last value: {production_series.iloc[-1]:.2f}")

    # Validate and forecast
    print("\nValidating with timesmith.typing.validators.assert_series_like...")
    forecast, params = validate_and_forecast(production_series, model_name="arps_hyperbolic")

    print("\nForecast Results:")
    print(f"  Forecast periods: {len(forecast.yhat)}")
    print(f"  Forecast range: {forecast.yhat.min():.2f} - {forecast.yhat.max():.2f}")
    print(f"\nFitted Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")

    print("\nIntegration successful!")
    print("   - timesmith.typing.SeriesLike validated input")
    print("   - ressmith fit and forecast completed")
    print("   - No circular imports or type conflicts")


if __name__ == "__main__":
    main()

