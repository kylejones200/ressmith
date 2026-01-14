"""
Ensemble & Probabilistic Forecasting Demo

This demo showcases uncertainty quantification workflows:
1. Ensemble forecasting (multiple models)
2. Probabilistic forecasting (P10/P50/P90)
3. Confidence intervals
4. Risk metrics
5. Scenario evaluation

Run: python examples/demo_ensemble_probabilistic.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith import (
    ensemble_forecast,
    probabilistic_forecast,
    evaluate_economics,
    evaluate_scenarios,
    scenario_summary,
)
from ressmith.objects.domain import EconSpec
from ressmith.workflows.forecast_statistical import calculate_confidence_intervals


def generate_synthetic_data(n_periods=36, noise_level=5.0):
    """Generate synthetic production data."""
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    t = np.arange(n_periods) / 12.0  # Years
    
    # Hyperbolic decline
    qi = 600.0
    di = 0.75
    b = 0.6
    q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(q_true))
    q_noisy = np.maximum(q_true + noise, 10.0)
    
    data = pd.DataFrame({
        'oil': q_noisy,
    }, index=time_index)
    
    return data


def main():
    """Run ensemble and probabilistic forecasting demo."""
    print("=" * 80)
    print("ENSEMBLE & PROBABILISTIC FORECASTING DEMO")
    print("=" * 80)
    
    # Step 1: Generate Data
    print("\n1. Loading Production Data...")
    data = generate_synthetic_data(n_periods=36, noise_level=5.0)
    print(f"   ✓ Loaded {len(data)} months of data")
    print(f"   ✓ Average rate: {data['oil'].mean():.1f} STB/day")
    
    # Step 2: Ensemble Forecasting
    print("\n2. Generating Ensemble Forecast...")
    model_names = ['arps_hyperbolic', 'arps_exponential', 'power_law', 'duong']
    
    ensemble_result = ensemble_forecast(
        data,
        model_names=model_names,
        method='weighted',  # Weighted average
        horizon=36,
        weights=[0.4, 0.3, 0.2, 0.1]  # Favor hyperbolic
    )
    
    print(f"   ✓ Ensemble forecast generated")
    print(f"   ✓ Forecast horizon: {len(ensemble_result.yhat)} months")
    print(f"   ✓ Total forecasted production: {ensemble_result.yhat.sum():,.0f} STB")
    print(f"   ✓ Peak rate: {ensemble_result.yhat.max():.1f} STB/day")
    
    # Step 3: Probabilistic Forecasting (P10/P50/P90)
    print("\n3. Generating Probabilistic Forecast (P10/P50/P90)...")
    prob_result = probabilistic_forecast(
        data,
        model_name='arps_hyperbolic',
        horizon=36,
        n_samples=2000,  # Monte Carlo samples
        seed=42
    )
    
    p10 = prob_result['p10']
    p50 = prob_result['p50']
    p90 = prob_result['p90']
    
    print(f"   ✓ Probabilistic forecast generated ({prob_result.get('n_samples', 0)} samples)")
    print(f"\n   EUR Statistics:")
    print(f"     P10 EUR: {p10.sum():,.0f} STB (90% probability of exceeding)")
    print(f"     P50 EUR: {p50.sum():,.0f} STB (median)")
    print(f"     P90 EUR: {p90.sum():,.0f} STB (10% probability of exceeding)")
    print(f"     Uncertainty: ±{((p10.sum() - p90.sum()) / 2 / p50.sum() * 100):.1f}%")
    
    # Step 4: Confidence Intervals
    print("\n4. Calculating Confidence Intervals...")
    forecast_series = p50
    
    ci_80 = calculate_confidence_intervals(
        forecast_series,
        z_score=1.28,  # 80% confidence
        method='naive'
    )
    
    ci_95 = calculate_confidence_intervals(
        forecast_series,
        z_score=1.96,  # 95% confidence
        method='naive'
    )
    
    if ci_80 and ci_95:
        print(f"   ✓ Confidence intervals calculated")
        print(f"   ✓ 80% CI width: {(ci_80[0] - ci_80[1]).mean():.1f} STB/day avg")
        print(f"   ✓ 95% CI width: {(ci_95[0] - ci_95[1]).mean():.1f} STB/day avg")
    
    # Step 5: Economics with Uncertainty
    print("\n5. Economics Evaluation with Uncertainty...")
    econ_spec = EconSpec(
        price_assumptions={'oil': 75.0},
        opex=15.0,
        capex=2000000.0,
        discount_rate=0.10,
    )
    
    # P50 economics
    econ_p50 = evaluate_economics(prob_result['forecast'], econ_spec)
    
    # P10 and P90 economics
    from ressmith.objects.domain import ForecastResult
    econ_p10 = evaluate_economics(
        ForecastResult(yhat=p10),
        econ_spec
    )
    econ_p90 = evaluate_economics(
        ForecastResult(yhat=p90),
        econ_spec
    )
    
    print(f"\n   NPV Statistics:")
    print(f"     P10 NPV: ${econ_p10.npv:,.0f} (optimistic)")
    print(f"     P50 NPV: ${econ_p50.npv:,.0f} (base case)")
    print(f"     P90 NPV: ${econ_p90.npv:,.0f} (conservative)")
    print(f"     Range: ${econ_p10.npv - econ_p90.npv:,.0f}")
    
    if econ_p50.irr and econ_p10.irr and econ_p90.irr:
        print(f"\n   IRR Statistics:")
        print(f"     P10 IRR: {econ_p10.irr*100:.2f}%")
        print(f"     P50 IRR: {econ_p50.irr*100:.2f}%")
        print(f"     P90 IRR: {econ_p90.irr*100:.2f}%")
    
    # Step 6: Scenario Analysis with Uncertainty
    print("\n6. Scenario Analysis with Uncertainty...")
    scenarios = {
        'base_case': {},
        'high_price': {'prices': {'oil': 90.0}},
        'low_price': {'prices': {'oil': 60.0}},
        'high_opex': {'opex': 20.0},
        'low_discount': {'discount_rate': 0.08},
    }
    
    # Evaluate scenarios for P50 case
    scenario_results = evaluate_scenarios(
        prob_result['forecast'],
        econ_spec,
        scenarios
    )
    
    scenario_df = scenario_summary(scenario_results)
    print(f"\n   Scenario Results (P50):")
    print(scenario_df[['scenario', 'npv', 'irr']].to_string(index=False))
    
    # Step 7: Risk Metrics
    print("\n7. Risk Metrics...")
    # Calculate probability of positive NPV
    samples = prob_result.get('samples', [])
    if len(samples) > 0:
        # For demonstration, use simplified calculation
        npv_samples = []
        for sample_forecast in samples[:100]:  # Use subset for speed
            sample_result = evaluate_economics(
                ForecastResult(yhat=pd.Series(sample_forecast)),
                econ_spec
            )
            npv_samples.append(sample_result.npv)
        
        npv_array = np.array(npv_samples)
        prob_positive = (npv_array > 0).mean() * 100
        var_90 = np.percentile(npv_array, 10)  # 90% VaR (10th percentile)
        
        print(f"   ✓ Probability of positive NPV: {prob_positive:.1f}%")
        print(f"   ✓ 90% VaR: ${var_90:,.0f} (90% chance of loss < ${abs(var_90):,.0f})")
        print(f"   ✓ Expected NPV: ${npv_array.mean():,.0f}")
        print(f"   ✓ NPV Std Dev: ${npv_array.std():,.0f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ENSEMBLE & PROBABILISTIC ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • P50 EUR: {p50.sum():,.0f} STB")
    print(f"  • P50 NPV: ${econ_p50.npv:,.0f}")
    print(f"  • Uncertainty range: ±{((p10.sum() - p90.sum()) / 2 / p50.sum() * 100):.1f}%")
    print(f"  • Probability of success: {prob_positive:.1f}%" if 'prob_positive' in locals() else "  • Probability: N/A")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

