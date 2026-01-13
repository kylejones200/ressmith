"""Command-line interface for reservoir engineering analysis.

This module provides a simple CLI for common reservoir engineering operations:
- ressmith fit: Fit decline curve model
- ressmith forecast: Generate production forecast
- ressmith batch: Batch process multiple wells
- ressmith report: Generate analysis reports
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from ressmith.primitives.diagnostics import compute_diagnostics
from ressmith.workflows.core import fit_forecast
from ressmith.workflows.io import read_csv_production, write_csv_results
from ressmith.workflows.portfolio import analyze_portfolio
from ressmith.workflows.reports import generate_well_report

logger = logging.getLogger(__name__)


def main():
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ResSmith Reservoir Engineering CLI",
        prog="ressmith",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit decline curve model")
    fit_parser.add_argument("input", help="Input data file (CSV)")
    fit_parser.add_argument("--output-dir", default="output", help="Output directory")
    fit_parser.add_argument("--model", default="arps_hyperbolic", help="Model type")
    fit_parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon")
    fit_parser.add_argument("--phase", default="oil", help="Phase name")

    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate forecast")
    forecast_parser.add_argument("input", help="Input data file (CSV)")
    forecast_parser.add_argument(
        "--output-dir", default="output", help="Output directory"
    )
    forecast_parser.add_argument(
        "--horizon", type=int, default=24, help="Forecast horizon"
    )
    forecast_parser.add_argument(
        "--model", default="arps_hyperbolic", help="Model type"
    )
    forecast_parser.add_argument("--phase", default="oil", help="Phase name")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch processing")
    batch_parser.add_argument("input", help="Input CSV file with well_id column")
    batch_parser.add_argument(
        "--output-dir", default="batch_output", help="Output directory"
    )
    batch_parser.add_argument("--model", default="arps_hyperbolic", help="Model type")
    batch_parser.add_argument(
        "--horizon", type=int, default=24, help="Forecast horizon"
    )
    batch_parser.add_argument("--well-id-col", default="well_id", help="Well ID column")
    batch_parser.add_argument("--date-col", default="date", help="Date column")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("input", help="Input data file (CSV)")
    report_parser.add_argument("--output", help="Output file path")
    report_parser.add_argument(
        "--format", choices=["html", "pdf"], default="html", help="Report format"
    )
    report_parser.add_argument("--model", default="arps_hyperbolic", help="Model type")
    report_parser.add_argument(
        "--horizon", type=int, default=24, help="Forecast horizon"
    )
    report_parser.add_argument("--well-id", help="Well identifier")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "fit":
        _cmd_fit(args)
    elif args.command == "forecast":
        _cmd_forecast(args)
    elif args.command == "batch":
        _cmd_batch(args)
    elif args.command == "report":
        _cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_fit(args):
    """Handle fit command."""
    logger.info(f"Fitting decline curve model: {args.input}")

    # Read input data
    data = read_csv_production(args.input)

    # Fit model
    forecast, params = fit_forecast(
        data, model_name=args.model, horizon=args.horizon, phase=args.phase
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save forecast
    forecast_path = output_dir / "forecast.csv"
    write_csv_results(forecast.yhat, forecast_path)
    logger.info(f"Forecast saved to {forecast_path}")

    # Save parameters
    params_path = output_dir / "parameters.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    logger.info(f"Parameters saved to {params_path}")

    logger.info("Fit complete")


def _cmd_forecast(args):
    """Handle forecast command."""
    logger.info(f"Generating forecast: {args.input}")

    # Read input data
    data = read_csv_production(args.input)

    # Generate forecast
    forecast, params = fit_forecast(
        data, model_name=args.model, horizon=args.horizon, phase=args.phase
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save forecast
    forecast_path = output_dir / "forecast.csv"
    write_csv_results(forecast.yhat, forecast_path)
    logger.info(f"Forecast saved to {forecast_path}")

    # Save parameters
    params_path = output_dir / "parameters.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    logger.info(f"Parameters saved to {params_path}")

    logger.info("Forecast complete")


def _cmd_batch(args):
    """Handle batch command."""
    logger.info(f"Running batch processing: {args.input}")

    # Read input data
    df = pd.read_csv(args.input)

    # Check required columns
    if args.well_id_col not in df.columns:
        raise ValueError(f"Well ID column '{args.well_id_col}' not found")
    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found")

    # Convert date column
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    # Group by well
    well_data = {}
    for well_id, well_df in df.groupby(args.well_id_col):
        well_df = well_df.set_index(args.date_col)
        # Find rate column (oil, gas, or first numeric column)
        rate_col = None
        for col in ["oil", "gas", "rate"]:
            if col in well_df.columns:
                rate_col = col
                break
        if rate_col is None:
            numeric_cols = well_df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                rate_col = numeric_cols[0]
            else:
                logger.warning(f"No rate column found for well {well_id}, skipping")
                continue

        well_df = well_df[[rate_col]].rename(columns={rate_col: "oil"})
        well_data[well_id] = well_df

    logger.info(f"Processing {len(well_data)} wells")

    # Analyze portfolio
    portfolio_results = analyze_portfolio(
        well_data, model_name=args.model, horizon=args.horizon
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_path = output_dir / "portfolio_results.csv"
    portfolio_results.to_csv(results_path, index=False)
    logger.info(f"Portfolio results saved to {results_path}")

    logger.info("Batch processing complete")


def _cmd_report(args):
    """Handle report command."""
    logger.info(f"Generating report: {args.input}")

    # Read input data
    data = read_csv_production(args.input)

    # Fit and forecast
    forecast, params = fit_forecast(data, model_name=args.model, horizon=args.horizon)

    # Compute diagnostics
    diagnostics = compute_diagnostics(data["oil"].values, forecast.yhat.values)

    # Generate report
    if args.output:
        output_path = args.output
    else:
        well_id = args.well_id or Path(args.input).stem
        ext = "html" if args.format == "html" else "pdf"
        output_path = f"{well_id}_report.{ext}"

    report_path = generate_well_report(
        forecast,
        diagnostics=diagnostics,
        params=params,
        output_path=output_path,
        format=args.format,
        well_id=args.well_id,
    )

    logger.info(f"Report saved to {report_path}")
    logger.info("Report complete")
