"""
Segmented decline example.

Fit a segmented decline model with multiple ARPS segments.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith.workflows import fit_segmented_forecast
from ressmith.workflows.io import write_csv_results


def generate_synthetic_segmented(
    n_periods: int = 36,
) -> pd.DataFrame:
    """Generate synthetic data with two decline segments."""
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    t = np.arange(n_periods)

    # First segment: hyperbolic decline (0-12 months)
    q1 = 100.0 / (1.0 + 0.6 * 0.15 * t[:12]) ** (1.0 / 0.6)

    # Second segment: exponential decline (steeper, 12-36 months)
    q2_start = q1[-1]
    q2 = q2_start * np.exp(-0.2 * (t[12:] - 12))

    # Combine
    q = np.concatenate([q1, q2])
    # Add noise
    noise = np.random.normal(0, 2.0, len(q))
    q_noisy = np.maximum(q + noise, 0.1)

    # Create DataFrame
    data = pd.DataFrame({"oil": q_noisy}, index=time_index)
    return data


def main():
    """Run segmented decline example."""
    print("Generating synthetic segmented decline data...")
    data = generate_synthetic_segmented(n_periods=36)
    print(f"Generated {len(data)} periods of data")

    # Define segments
    segment_dates = [
        (datetime(2020, 1, 1), datetime(2021, 1, 1)),  # First 12 months
        (datetime(2021, 1, 1), datetime(2023, 1, 1)),  # Next 24 months
    ]
    kinds = ["hyperbolic", "exponential"]

    print("\nFitting SegmentedDeclineModel...")
    print(f"  Segment 1: {segment_dates[0]} - {kinds[0]}")
    print(f"  Segment 2: {segment_dates[1]} - {kinds[1]}")

    forecast, segments, errors = fit_segmented_forecast(
        data,
        segment_dates=segment_dates,
        kinds=kinds,
        horizon=24,
        enforce_continuity=False,  # Allow discontinuities for this example
    )

    print(f"\nFitted {len(segments)} segments")
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1} ({segment.kind}):")
        print(f"  qi: {segment.parameters['qi']:.2f}")
        print(f"  di: {segment.parameters['di']:.4f}")
        if segment.kind == "hyperbolic":
            print(f"  b: {segment.parameters['b']:.4f}")

    if errors:
        print(f"\nContinuity errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")

    print(f"\nForecast generated: {len(forecast.yhat)} periods")
    print(f"Forecast range: {forecast.yhat.min():.2f} - {forecast.yhat.max():.2f}")

    # Write results
    output_path = "examples/out/segmented_forecast.csv"
    write_csv_results(forecast.yhat, output_path)
    print(f"\nForecast written to {output_path}")


if __name__ == "__main__":
    main()

