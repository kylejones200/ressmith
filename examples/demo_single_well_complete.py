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

Run: python examples/demo_single_well_complete.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith import (
    compare_models,
    estimate_eur,
    evaluate_economics,
    evaluate_scenarios,
    fit_forecast,
    probabilistic_forecast,
    run_sensitivity,
    scenario_summary,
    walk_forward_backtest,
)
from ressmith.objects.domain import EconSpec
from ressmith.workflows.diagnostics_plots import generate_diagnostic_plot_data


def generate_synthetic_data(n_periods=36, noise_level=3.0):
    """Generate synthetic production data with hyperbolic decline."""
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    t = np.arange(n_periods) / 12.0  # Convert to years
    
    # Hyperbolic decline parameters
    qi = 500.0  # STB/day initial rate
    di = 0.8  # 80% annual decline initially
    b = 0.6  # Hyperbolic exponent
    
    # Generate true decline curve
    q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level, len(q_true))
    q_noisy = np.maximum(q_true + noise, 10.0)  # Ensure positive, minimum 10 STB/day
    
    # Create DataFrame with multiple phases
    data = pd.DataFrame({
        'oil': q_noisy,
        'gas': q_noisy * 1000 + np.random.normal(0, 100, len(q_noisy)),  # 1000 SCF/bbl GOR
        'water': q_noisy * 0.1 + np.random.normal(0, 5, len(q_noisy)),  # 10% water cut
    }, index=time_index)
    
    return data


def main():
    """Run complete single well analysis workflow."""
    print("=" * 80)
    print("COMPLETE SINGLE WELL ANALYSIS DEMO")
    print("=" * 80)
    
    # Step 1: Generate/Load Production Data
    print("\n1. Loading Production Data...")
    data = generate_synthetic_data(n_periods=36, noise_level=3.0)
    print(f"   ✓ Loaded {len(data)} months of production data")
    print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   ✓ Average oil rate: {data['oil'].mean():.1f} STB/day")
    print(f"   ✓ Peak oil rate: {data['oil'].max():.1f} STB/day")
    
    # Step 2: Model Comparison
    print("\n2. Comparing Multiple Decline Models...")
    model_names = ['arps_hyperbolic', 'arps_exponential', 'power_law', 'duong']
    comparison = compare_models(
        data,
        model_names=model_names,
        horizon=36,
        phase='oil'
    )
    print("\n   Model Comparison Results:")
    print(comparison[['model_name', 'r_squared', 'rmse', 'mae', 'mape']].to_string(index=False))
    
    # Select best model based on R²
    best_model = comparison.loc[comparison['r_squared'].idxmax(), 'model_name']
    print(f"\n   ✓ Best model: {best_model} (R² = {comparison['r_squared'].max():.4f})")
    
    # Step 3: Probabilistic Forecasting (P10/P50/P90)
    print("\n3. Generating Probabilistic Forecast (P10/P50/P90)...")
    prob_result = probabilistic_forecast(
        data,
        model_name=best_model,
        horizon=36,
        n_samples=1000,
        seed=42
    )
    
    p50_forecast = prob_result['p50']
    p10_forecast = prob_result['p10']
    p90_forecast = prob_result['p90']
    
    print(f"   ✓ Generated {len(p50_forecast)} month forecast")
    print(f"   ✓ P50 EUR: {p50_forecast.sum():,.0f} STB")
    print(f"   ✓ P10 EUR: {p10_forecast.sum():,.0f} STB")
    print(f"   ✓ P90 EUR: {p90_forecast.sum():,.0f} STB")
    print(f"   ✓ Uncertainty range: {((p10_forecast.sum() - p90_forecast.sum()) / p50_forecast.sum() * 100):.1f}%")
    
    # Step 4: Estimate EUR
    print("\n4. Estimating EUR from Best Model...")
    forecast, params = fit_forecast(data, model_name=best_model, horizon=36)
    eur = estimate_eur(data, model_name=best_model)
    print(f"   ✓ Estimated EUR: {eur:,.0f} STB")
    print(f"   ✓ Model parameters: {params}")
    
    # Step 5: Economics Evaluation
    print("\n5. Evaluating Economics...")
    econ_spec = EconSpec(
        price_assumptions={
            'oil': 75.0,  # $75/bbl
            'gas': 3.5,   # $3.5/MCF
        },
        opex=15.0,  # $15/STB operating cost
        capex=2500000.0,  # $2.5M capital expenditure
        discount_rate=0.10,  # 10% discount rate
    )
    
    econ_result = evaluate_economics(prob_result['forecast'], econ_spec)
    print(f"   ✓ NPV: ${econ_result.npv:,.0f}")
    if econ_result.irr is not None:
        print(f"   ✓ IRR: {econ_result.irr*100:.2f}%")
    if econ_result.payout_time is not None:
        print(f"   ✓ Payout Time: {econ_result.payout_time:.1f} months")
    print(f"   ✓ Total Revenue: ${econ_result.cashflows['revenue'].sum():,.0f}")
    print(f"   ✓ Total OPEX: ${abs(econ_result.cashflows['opex'].sum()):,.0f}")
    
    # Step 6: Scenario Analysis
    print("\n6. Running Scenario Analysis...")
    scenarios = {
        'base_case': {},
        'high_price': {'prices': {'oil': 90.0, 'gas': 4.0}},
        'low_price': {'prices': {'oil': 60.0, 'gas': 3.0}},
        'high_opex': {'opex': 20.0},
        'low_discount': {'discount_rate': 0.08},
    }
    
    scenario_results = evaluate_scenarios(
        prob_result['forecast'],
        econ_spec,
        scenarios
    )
    
    scenario_df = scenario_summary(scenario_results)
    print("\n   Scenario Results:")
    print(scenario_df[['scenario', 'npv', 'irr', 'payout_time']].to_string(index=False))
    
    # Step 7: Sensitivity Analysis
    print("\n7. Running Sensitivity Analysis...")
    sensitivity_result = run_sensitivity(
        prob_result['forecast'],
        econ_spec,
        variables=['oil', 'opex', 'discount_rate'],
        ranges={
            'oil': (60.0, 90.0),
            'opex': (10.0, 20.0),
            'discount_rate': (0.08, 0.12),
        }
    )
    
    print(f"   ✓ Ran sensitivity on {len(sensitivity_result)} parameter combinations")
    print(f"   ✓ NPV range: ${sensitivity_result['npv'].min():,.0f} to ${sensitivity_result['npv'].max():,.0f}")
    
    # Step 8: Walk-Forward Backtest
    print("\n8. Validating Model with Walk-Forward Backtest...")
    backtest_results = walk_forward_backtest(
        data,
        model_name=best_model,
        forecast_horizons=[6, 12, 24],
        min_train_size=12,
        phase='oil'
    )
    
    print(f"   ✓ Completed {len(backtest_results)} backtest evaluations")
    avg_rmse = backtest_results.groupby('horizon')['rmse'].mean()
    print(f"   ✓ Average RMSE by horizon:")
    for horizon, rmse in avg_rmse.items():
        print(f"     - {horizon} months: {rmse:.2f} STB/day")
    
    # Step 9: Diagnostic Plots Data
    print("\n9. Generating Diagnostic Plot Data...")
    diag_data = generate_diagnostic_plot_data(
        time=data.index.values,
        rate=data['oil'].values,
        forecast=p50_forecast.values[:len(data)]
    )
    print(f"   ✓ Generated diagnostic plot data")
    print(f"   ✓ Flow regime identified: {diag_data.get('flow_regime', 'unknown')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Best Model: {best_model}")
    print(f"  • EUR: {eur:,.0f} STB")
    print(f"  • P50 NPV: ${econ_result.npv:,.0f}")
    print(f"  • IRR: {econ_result.irr*100:.2f}%" if econ_result.irr else "  • IRR: N/A")
    print(f"  • Model Validation: RMSE = {backtest_results['rmse'].mean():.2f} STB/day")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

