"""
Complete Single Well Analysis Demo

This demo showcases a comprehensive single-well workflow:
1. Load production data
2. Fit multiple decline models and compare
3. Generate probabilistic forecast (P10/P50/P90)
4. Evaluate economics
5. Run sensitivity analysis
6. Evaluate scenarios
7. Generate diagnostics and plots

Run: uv run python examples/demo_single_well_complete.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ressmith import (
    compare_models,
    estimate_eur,
    evaluate_economics,
    evaluate_scenarios,
    fit_forecast,
    probabilistic_forecast,
    scenario_summary,
    walk_forward_backtest,
)
from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.workflows.calendar import get_month_day_counts
from ressmith.workflows.diagnostics_plots import generate_diagnostic_plot_data
from ressmith.workflows.sensitivity import run_sensitivity


def monthly_volumes_from_rates(rates: pd.Series) -> pd.Series:
    """Integrate STB/day rates to STB/month using actual days per month."""
    days = get_month_day_counts(rates.index)
    return rates * days


def forecast_volumes_for_economics(rates: pd.Series) -> ForecastResult:
    """Economics expects monthly production volumes, not daily rates."""
    return ForecastResult(yhat=monthly_volumes_from_rates(rates))


def generate_synthetic_data(n_periods: int = 36, noise_level: float = 3.0) -> pd.DataFrame:
    """Generate synthetic production data with hyperbolic decline (monthly time base)."""
    rng = np.random.default_rng(42)
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="ME")
    t = np.arange(n_periods, dtype=float)  # months — matches fitter time index

    qi = 500.0  # STB/day
    di = 0.08  # monthly nominal decline
    b = 0.6

    q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
    noise = rng.normal(0, noise_level, len(q_true))
    q_noisy = np.maximum(q_true + noise, 10.0)

    data = pd.DataFrame(
        {
            "oil": q_noisy,
            "gas": q_noisy * 1000 + rng.normal(0, 100, len(q_noisy)),
            "water": np.maximum(q_noisy * 0.1 + rng.normal(0, 5, len(q_noisy)), 0.0),
        },
        index=time_index,
    )
    return data


def petroleum_percentiles(prob_result: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map statistical p10/p90 keys to petroleum P10 (high) / P50 / P90 (low)."""
    return prob_result["p90"], prob_result["p50"], prob_result["p10"]


def pick_primary_model(comparison: pd.DataFrame) -> str:
    """Prefer positive R²; if all fits are poor, fall back to hyperbolic truth model."""
    positive = comparison[comparison["r_squared"] > 0]
    if not positive.empty:
        return str(positive.loc[positive["r_squared"].idxmax(), "model_name"])
    # All in-sample R² non-positive — prefer Arps hyperbolic for a stable demo path
    names = set(comparison["model_name"].tolist())
    if "arps_hyperbolic" in names:
        return "arps_hyperbolic"
    finite_r2 = comparison["r_squared"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_r2.empty:
        return str(comparison.loc[comparison["r_squared"].idxmax(), "model_name"])
    return str(comparison.loc[comparison["rmse"].idxmin(), "model_name"])


def main() -> None:
    """Run complete single well analysis workflow."""
    print("=" * 80)
    print("COMPLETE SINGLE WELL ANALYSIS DEMO")
    print("=" * 80)

    print("\n1. Loading Production Data...")
    data = generate_synthetic_data(n_periods=36, noise_level=3.0)
    print(f"   ✓ Loaded {len(data)} months of production data")
    print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   ✓ Average oil rate: {data['oil'].mean():.1f} STB/day")
    print(f"   ✓ Peak oil rate: {data['oil'].max():.1f} STB/day")

    print("\n2. Comparing Multiple Decline Models...")
    model_names = ["arps_hyperbolic", "arps_exponential", "power_law", "duong"]
    comparison = compare_models(
        data,
        model_names=model_names,
        horizon=36,
        phase="oil",
    )
    print("\n   Model Comparison Results:")
    print(comparison[["model_name", "r_squared", "rmse", "mae", "mape"]].to_string(index=False))

    best_model = pick_primary_model(comparison)
    best_r2 = float(comparison.loc[comparison["model_name"] == best_model, "r_squared"].iloc[0])
    print(f"\n   ✓ Primary model: {best_model} (R² = {best_r2:.4f})")

    # Prefer Arps hyperbolic for EUR / Arps-parameter sensitivity when primary is non-Arps
    eur_model = best_model if best_model.startswith("arps_") else "arps_hyperbolic"

    print("\n3. Generating Probabilistic Forecast (P10/P50/P90)...")
    prob_result = probabilistic_forecast(
        data,
        model_name=best_model,
        horizon=36,
        n_samples=1000,
        seed=42,
    )

    p10_forecast, p50_forecast, p90_forecast = petroleum_percentiles(prob_result)
    p10_cum = float(monthly_volumes_from_rates(p10_forecast).sum())
    p50_cum = float(monthly_volumes_from_rates(p50_forecast).sum())
    p90_cum = float(monthly_volumes_from_rates(p90_forecast).sum())

    print(f"   ✓ Generated {len(p50_forecast)} month forecast")
    print(f"   ✓ P10 (high): {p10_cum:,.0f} STB")
    print(f"   ✓ P50 (median): {p50_cum:,.0f} STB")
    print(f"   ✓ P90 (low): {p90_cum:,.0f} STB")
    print(f"   ✓ P10–P90 half-width: {((p10_cum - p90_cum) / 2 / p50_cum * 100):.1f}% of P50")

    print("\n4. Estimating EUR...")
    forecast, params = fit_forecast(data, model_name=best_model, horizon=36)
    eur_result = estimate_eur(data, model_name=eur_model)
    eur = float(eur_result.get("eur") or 0.0)
    print(f"   ✓ Estimated EUR ({eur_model}): {eur:,.0f} STB")
    print(f"   ✓ Primary model parameters: {params}")
    _, arps_params = fit_forecast(data, model_name="arps_hyperbolic", horizon=36)

    print("\n5. Evaluating Economics...")
    econ_spec = EconSpec(
        price_assumptions={
            "oil": 75.0,
            "gas": 3.5,
        },
        opex=15.0,
        capex=2_500_000.0,
        discount_rate=0.10,
    )

    econ_result = evaluate_economics(
        forecast_volumes_for_economics(prob_result["forecast"].yhat),
        econ_spec,
    )
    print(f"   ✓ NPV: ${econ_result.npv:,.0f}")
    if econ_result.irr is not None:
        print(f"   ✓ IRR: {econ_result.irr * 100:.2f}%")
    if econ_result.payout_time is not None:
        print(f"   ✓ Payout Time: {econ_result.payout_time:.1f} months")
    print(f"   ✓ Total Revenue: ${econ_result.cashflows['revenue'].sum():,.0f}")
    print(f"   ✓ Total OPEX: ${abs(econ_result.cashflows['opex'].sum()):,.0f}")

    print("\n6. Running Scenario Analysis...")
    scenarios = {
        "base_case": {},
        "high_price": {"prices": {"oil": 90.0, "gas": 4.0}},
        "low_price": {"prices": {"oil": 60.0, "gas": 3.0}},
        "high_opex": {"opex": 20.0},
        "low_discount": {"discount_rate": 0.08},
    }

    scenario_results = evaluate_scenarios(
        forecast_volumes_for_economics(prob_result["forecast"].yhat),
        econ_spec,
        scenarios,
    )
    scenario_df = scenario_summary(scenario_results)
    print("\n   Scenario Results:")
    print(scenario_df[["scenario", "npv", "irr", "payout_time"]].to_string(index=False))

    print("\n7. Running Sensitivity Analysis (Arps params × price)...")
    qi = float(arps_params.get("qi", 500.0))
    di = float(arps_params.get("di", 0.1))
    b = float(arps_params.get("b", 0.5))
    param_grid = [
        (qi * 0.9, di, b),
        (qi, di, b),
        (qi * 1.1, di, b),
        (qi, di * 0.9, b),
        (qi, di * 1.1, b),
        (qi, di, max(0.01, b * 0.8)),
        (qi, di, min(1.5, b * 1.2)),
    ]
    sensitivity_result = run_sensitivity(
        param_grid=param_grid,
        prices=[60.0, 75.0, 90.0],
        opex=15.0,
        discount_rate=0.10,
        n_jobs=1,
    )
    print(f"   ✓ Ran sensitivity on {len(sensitivity_result)} parameter combinations")
    print(
        f"   ✓ NPV range: ${sensitivity_result['NPV'].min():,.0f} "
        f"to ${sensitivity_result['NPV'].max():,.0f}"
    )

    print("\n8. Validating Model with Walk-Forward Backtest...")
    backtest_results = walk_forward_backtest(
        data,
        model_name=best_model,
        forecast_horizons=[6, 12],
        min_train_size=12,
        phase="oil",
    )
    print(f"   ✓ Completed {len(backtest_results)} backtest evaluations")
    if not backtest_results.empty and "horizon" in backtest_results.columns:
        avg_rmse = backtest_results.groupby("horizon")["rmse"].mean()
        print("   ✓ Average RMSE by horizon:")
        for horizon, rmse in avg_rmse.items():
            print(f"     - {horizon} months: {rmse:.2f} STB/day")
        backtest_rmse_msg = f"{backtest_results['rmse'].mean():.2f} STB/day"
    else:
        print("   ⚠ No successful backtest folds for this model/horizon setup")
        backtest_rmse_msg = "N/A"

    print("\n9. Generating Diagnostic Plot Data...")
    t_days = np.array(
        [(data.index[i] - data.index[0]).days for i in range(len(data))],
        dtype=float,
    )
    diag_data = generate_diagnostic_plot_data(
        time=t_days,
        rate=data["oil"].values,
        plot_type="all",
    )
    print("   ✓ Generated diagnostic plot data")
    print(f"   ✓ Flow regime identified: {diag_data.get('flow_regime', 'unknown')}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Primary Model: {best_model}")
    print(f"  • EUR: {eur:,.0f} STB")
    print(f"  • P50 NPV: ${econ_result.npv:,.0f}")
    print(f"  • IRR: {econ_result.irr * 100:.2f}%" if econ_result.irr else "  • IRR: N/A")
    print(f"  • Model Validation: RMSE = {backtest_rmse_msg}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
