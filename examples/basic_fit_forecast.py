"""
Basic fit and forecast example.

Generate synthetic hyperbolic decline series with noise.
Fit ArpsHyperbolicModel. Forecast 24 months.
Write results to examples/out/forecast.csv.
"""

import numpy as np
import pandas as pd

from ressmith.workflows import fit_forecast
from ressmith.workflows.io import write_csv_results


def generate_synthetic_hyperbolic(
    n_periods: int = 36, noise_level: float = 2.0
) -> pd.DataFrame:
    """Generate synthetic hyperbolic decline with noise."""
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    t = np.arange(n_periods)

    # Hyperbolic parameters
    qi = 100.0
    di = 0.1
    b = 0.6

    # Generate true decline
    q_true = qi / (1.0 + b * di * t) ** (1.0 / b)

    # Add noise
    noise = np.random.normal(0, noise_level, len(q_true))
    q_noisy = np.maximum(q_true + noise, 0.1)  # Ensure positive

    # Create DataFrame
    data = pd.DataFrame({"oil": q_noisy}, index=time_index)
    return data


def main():
    """Run basic fit and forecast example."""
    print("Generating synthetic hyperbolic decline data...")
    data = generate_synthetic_hyperbolic(n_periods=36, noise_level=2.0)
    print(f"Generated {len(data)} periods of data")

    print("\nFitting ArpsHyperbolicModel...")
    forecast, params = fit_forecast(
        data, model_name="arps_hyperbolic", horizon=24
    )

    print(f"\nFitted parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nForecast generated: {len(forecast.yhat)} periods")
    print(f"Forecast range: {forecast.yhat.min():.2f} - {forecast.yhat.max():.2f}")

    # Write results
    output_path = "examples/out/forecast.csv"
    write_csv_results(forecast.yhat, output_path)
    print(f"\nForecast written to {output_path}")


if __name__ == "__main__":
    main()

