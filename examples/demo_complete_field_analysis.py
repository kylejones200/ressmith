"""
Complete Field Analysis Demo

This comprehensive demo ties together all major ResSmith workflows:
1. Multi-well data loading and portfolio analysis
2. Ensemble and probabilistic forecasting
3. Multi-well interaction and spacing optimization
4. Advanced RTA and type curve matching
5. Multi-phase forecasting
6. Production operations optimization
7. Economic evaluation with scenarios
8. Risk analysis

Run: python examples/demo_complete_field_analysis.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith import (
    aggregate_portfolio_forecast,
    analyze_blasingame,
    analyze_multi_well_interaction,
    analyze_portfolio,
    analyze_waterflood,
    ensemble_forecast,
    evaluate_economics,
    evaluate_scenarios,
    forecast_with_yields,
    match_type_curve_workflow,
    optimize_field_spacing,
    probabilistic_forecast,
    rank_wells,
    scenario_summary,
)
from ressmith.objects.domain import EconSpec


def generate_field_data(n_wells=8, n_periods=30):
    """Generate synthetic field data with multiple wells and phases."""
    np.random.seed(42)
    
    well_data = {}
    well_locations = []
    base_dates = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    
    for i in range(n_wells):
        well_id = f"WELL_{i+1:03d}"
        
        # Vary parameters
        qi = 400 + np.random.uniform(-150, 250)
        di = 0.6 + np.random.uniform(-0.2, 0.3)
        b = 0.5 + np.random.uniform(-0.2, 0.3)
        
        start_offset = np.random.randint(0, 6)
        dates = base_dates[start_offset:]
        n_periods_well = len(dates)
        t = np.arange(n_periods_well) / 12.0
        
        q_oil = qi / (1.0 + b * di * t) ** (1.0 / b)
        noise = np.random.normal(0, 8, len(q_oil))
        q_oil = np.maximum(q_oil + noise, 10.0)
        
        # Multi-phase data
        q_gas = q_oil * (1000 + np.random.uniform(-200, 200))
        q_water = q_oil * (0.05 + np.random.uniform(0, 0.15))
        
        # Pressure data
        pi = 4500 + np.random.uniform(-500, 500)
        p = pi - (q_oil.cumsum() / 50000.0) * 800.0
        pressure = np.maximum(p, 2000)
        
        well_data[well_id] = pd.DataFrame({
            'oil': q_oil,
            'gas': q_gas,
            'water': q_water,
            'pressure': pressure,
        }, index=dates)
        
        # Well locations
        row = i // 3
        col = i % 3
        well_locations.append({
            'well_id': well_id,
            'latitude': 32.0 + row * 0.015,
            'longitude': -97.0 + col * 0.015,
        })
    
    return well_data, pd.DataFrame(well_locations)


def main():
    """Run complete field analysis workflow."""
    print("=" * 80)
    print("COMPLETE FIELD ANALYSIS DEMO")
    print("=" * 80)
    
    # Step 1: Load Field Data
    print("\n1. Loading Field Data...")
    well_data, well_locations = generate_field_data(n_wells=8, n_periods=30)
    print(f"   ✓ Loaded {len(well_data)} wells")
    print(f"   ✓ Average data length: {np.mean([len(df) for df in well_data.values()]):.1f} months")
    print(f"   ✓ Field coverage: {len(well_locations)} wells")
    
    # Step 2: Portfolio Analysis
    print("\n2. Portfolio Analysis...")
    econ_spec = EconSpec(
        price_assumptions={'oil': 75.0, 'gas': 3.5},
        opex=15.0,
        capex=2500000.0,
        discount_rate=0.10,
    )
    
    portfolio_results = analyze_portfolio(
        well_data,
        model_name='arps_hyperbolic',
        horizon=36,
        econ_spec=econ_spec
    )
    
    print(f"   ✓ Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"   ✓ Total NPV: ${portfolio_results['npv'].sum():,.0f}")
    
    # Step 3: Well Ranking
    print("\n3. Well Ranking & Selection...")
    ranked_wells = rank_wells(portfolio_results, sort_by='npv', ascending=False)
    print(f"   ✓ Top well: {ranked_wells.iloc[0]['well_id']} "
          f"(NPV: ${ranked_wells.iloc[0]['npv']:,.0f})")
    
    # Step 4: Ensemble Forecasting for Portfolio
    print("\n4. Ensemble Forecasting...")
    # Select top 5 wells for detailed analysis
    top_wells = {well_id: well_data[well_id] for well_id in 
                 ranked_wells.head(5)['well_id'].values}
    
    portfolio_forecast = aggregate_portfolio_forecast(
        top_wells,
        model_name='arps_hyperbolic',
        horizon=36
    )
    
    print(f"   ✓ Portfolio forecast: {portfolio_forecast.yhat.sum():,.0f} STB total")
    
    # Step 5: Probabilistic Forecasting
    print("\n5. Probabilistic Forecasting (P10/P50/P90)...")
    # Use top well for probabilistic analysis
    top_well_id = ranked_wells.iloc[0]['well_id']
    top_well_data = well_data[top_well_id]
    
    prob_result = probabilistic_forecast(
        top_well_data,
        model_name='arps_hyperbolic',
        horizon=36,
        n_samples=1000,
        seed=42
    )
    
    p50_eur = prob_result['p50'].sum()
    p10_eur = prob_result['p10'].sum()
    p90_eur = prob_result['p90'].sum()
    
    print(f"   ✓ P50 EUR: {p50_eur:,.0f} STB")
    print(f"   ✓ Uncertainty: ±{((p10_eur - p90_eur) / 2 / p50_eur * 100):.1f}%")
    
    # Step 6: Multi-Well Interaction Analysis
    print("\n6. Multi-Well Interaction Analysis...")
    interaction_results = analyze_multi_well_interaction(
        well_locations,
        portfolio_results,
        drainage_radius=600.0
    )
    
    overlap_count = (interaction_results['overlap_ratio'] > 0).sum()
    print(f"   ✓ Well pairs analyzed: {len(interaction_results)}")
    print(f"   ✓ Pairs with overlap: {overlap_count}")
    
    if overlap_count > 0:
        avg_overlap = interaction_results[interaction_results['overlap_ratio'] > 0]['overlap_ratio'].mean()
        print(f"   ✓ Average overlap: {avg_overlap*100:.1f}%")
    
    # Step 7: Field Spacing Optimization
    print("\n7. Field Spacing Optimization...")
    spacing_result = optimize_field_spacing(
        well_locations,
        portfolio_results,
        min_spacing=500.0,
        target_eur_reduction=0.05
    )
    
    print(f"   ✓ Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    
    # Step 8: Type Curve Matching (Sample Wells)
    print("\n8. Type Curve Matching...")
    sample_well_id = ranked_wells.iloc[1]['well_id']
    sample_data = well_data[sample_well_id]
    
    type_curve_result = match_type_curve_workflow(
        time=sample_data.index.values,
        rate=sample_data['oil'].values,
        model_type='arps'
    )
    
    if type_curve_result:
        print(f"   ✓ Type curve matched for {sample_well_id}")
    else:
        print(f"   ⚠ Type curve matching limited")
    
    # Step 9: Advanced RTA (Best Well)
    print("\n9. Advanced RTA Analysis...")
    best_well_data = well_data[top_well_id]
    
    if 'pressure' in best_well_data.columns:
        try:
            blasingame_result = analyze_blasingame(
                time=best_well_data.index.values,
                rate=best_well_data['oil'].values,
                pressure=best_well_data['pressure'].values,
                initial_pressure=best_well_data['pressure'].iloc[0]
            )
            print(f"   ✓ Blasingame analysis complete")
            print(f"     - Permeability: {blasingame_result.get('permeability', 0):.2f} md")
        except Exception as e:
            print(f"   ⚠ Blasingame analysis: {str(e)[:50]}...")
    
    # Step 10: Multi-Phase Forecasting
    print("\n10. Multi-Phase Forecasting...")
    try:
        multiphase_result = forecast_with_yields(
            top_well_data,
            primary_phase='oil',
            associated_phases=['gas', 'water'],
            model_name='arps_hyperbolic',
            yield_models={'gas': 'constant', 'water': 'hyperbolic'},
            horizon=24
        )
        
        total_oil = multiphase_result['oil'].yhat.sum()
        total_gas = multiphase_result['gas'].yhat.sum() if 'gas' in multiphase_result else 0
        print(f"   ✓ Multi-phase forecast generated")
        print(f"     - Oil: {total_oil:,.0f} STB")
        if total_gas > 0:
            print(f"     - Gas: {total_gas:,.0f} MCF")
    except Exception as e:
        print(f"   ⚠ Multi-phase forecast: {str(e)[:50]}...")
    
    # Step 11: EOR Analysis
    print("\n11. EOR Pattern Analysis...")
    try:
        waterflood_result = analyze_waterflood(
            pattern_type='five_spot',
            pattern_area=40.0,
            injector_count=4,
            producer_count=5,
            injection_rate=1500.0,
            mobility_ratio=0.6,
            pore_volume_injected=0.25
        )
        
        print(f"   ✓ Waterflood analysis complete")
        print(f"     - Sweep efficiency: {waterflood_result.get('sweep_efficiency', 0)*100:.1f}%")
        print(f"     - Estimated recovery: {waterflood_result.get('estimated_recovery', 0)*100:.1f}%")
    except Exception as e:
        print(f"   ⚠ Waterflood analysis: {str(e)[:50]}...")
    
    # Step 12: Portfolio Economics
    print("\n12. Portfolio Economics...")
    portfolio_econ = evaluate_economics(portfolio_forecast, econ_spec)
    
    print(f"   ✓ Portfolio NPV: ${portfolio_econ.npv:,.0f}")
    if portfolio_econ.irr:
        print(f"   ✓ Portfolio IRR: {portfolio_econ.irr*100:.2f}%")
    
    # Step 13: Scenario Analysis
    print("\n13. Scenario Analysis...")
    scenarios = {
        'base_case': {},
        'high_price': {'prices': {'oil': 90.0, 'gas': 4.0}},
        'low_price': {'prices': {'oil': 60.0, 'gas': 3.0}},
        'high_opex': {'opex': 20.0},
    }
    
    scenario_results = evaluate_scenarios(
        portfolio_forecast,
        econ_spec,
        scenarios
    )
    
    scenario_df = scenario_summary(scenario_results)
    print(f"   ✓ Evaluated {len(scenarios)} scenarios")
    print(f"\n   Scenario NPVs:")
    for _, row in scenario_df.iterrows():
        print(f"     {row['scenario']}: ${row['npv']:,.0f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE FIELD ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nField Overview:")
    print(f"  • Total Wells: {len(well_data)}")
    print(f"  • Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"  • Total NPV: ${portfolio_econ.npv:,.0f}")
    print(f"  • Portfolio IRR: {portfolio_econ.irr*100:.2f}%" if portfolio_econ.irr else "  • IRR: N/A")
    
    print(f"\nBest Performing Well:")
    print(f"  • Well ID: {ranked_wells.iloc[0]['well_id']}")
    print(f"  • EUR: {ranked_wells.iloc[0]['eur']:,.0f} STB")
    print(f"  • NPV: ${ranked_wells.iloc[0]['npv']:,.0f}")
    
    print(f"\nField Optimization:")
    print(f"  • Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    print(f"  • Well pairs with overlap: {overlap_count}")
    
    print(f"\nUncertainty:")
    print(f"  • P50 EUR (best well): {p50_eur:,.0f} STB")
    print(f"  • Uncertainty range: ±{((p10_eur - p90_eur) / 2 / p50_eur * 100):.1f}%")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Review outputs for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()

