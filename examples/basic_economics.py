"""
Basic economics example.

Reuse forecast output from basic_fit_forecast.py.
Run EconTask with simple EconSpec.
Write cashflows to examples/out/cashflows.csv.
"""

import pandas as pd

from ressmith.objects.domain import EconSpec
from ressmith.workflows import evaluate_economics
from ressmith.workflows.io import read_csv_production, write_csv_results


def main():
    """Run basic economics example."""
    # Try to read forecast from previous example
    try:
        forecast_data = read_csv_production("examples/out/forecast.csv")
        # Convert to ForecastResult format
        from ressmith.objects.domain import ForecastResult

        forecast = ForecastResult(yhat=forecast_data.iloc[:, 0])
    except FileNotFoundError:
        print("Forecast file not found. Generating synthetic forecast...")
        # Generate a simple forecast
        time_index = pd.date_range("2020-01-01", periods=24, freq="M")
        yhat = pd.Series(
            [100.0 - i * 2.0 for i in range(24)], index=time_index, name="forecast"
        )
        from ressmith.objects.domain import ForecastResult

        forecast = ForecastResult(yhat=yhat)

    print(f"Using forecast with {len(forecast.yhat)} periods")

    # Create economics specification
    spec = EconSpec(
        price_assumptions={"oil": 50.0},  # $50/bbl
        opex=10.0,  # $10/period operating expense
        capex=1000.0,  # $1000 capital expenditure
        discount_rate=0.1,  # 10% discount rate
    )

    print("\nEvaluating economics...")
    econ_result = evaluate_economics(forecast, spec)

    print(f"\nEconomics Results:")
    print(f"  NPV: ${econ_result.npv:.2f}")
    if econ_result.irr is not None:
        print(f"  IRR: {econ_result.irr*100:.2f}%")
    if econ_result.payout_time is not None:
        print(f"  Payout Time: {econ_result.payout_time:.1f} periods")

    # Write cashflows
    output_path = "examples/out/cashflows.csv"
    write_csv_results(econ_result.cashflows, output_path)
    print(f"\nCashflows written to {output_path}")


if __name__ == "__main__":
    main()

