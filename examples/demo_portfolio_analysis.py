"""
Complete Portfolio Analysis Demo

This demo showcases portfolio-level workflows:
1. Load multiple wells
2. Fit forecasts for all wells
3. Portfolio aggregation
4. Well ranking
5. Economic evaluation
6. Risk analysis
7. Multi-well interaction analysis

Run: python examples/demo_portfolio_analysis.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith import (
    aggregate_portfolio_forecast,
    analyze_multi_well_interaction,
    analyze_portfolio,
    evaluate_economics,
    optimize_field_spacing,
    rank_wells,
)
from ressmith.objects.domain import EconSpec


def generate_portfolio_data(n_wells=10, n_periods=24):
    """Generate synthetic production data for multiple wells."""
    np.random.seed(42)
    
    well_data = {}
    base_dates = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    
    for i in range(n_wells):
        well_id = f"WELL_{i+1:03d}"
        
        # Vary parameters across wells
        qi = 300 + np.random.uniform(-100, 200)  # 200-500 STB/day
        di = 0.5 + np.random.uniform(-0.2, 0.3)  # 30%-80% annual decline
        b = 0.4 + np.random.uniform(-0.2, 0.4)  # b-factor variation
        
        # Random start date (some wells start later)
        start_offset = np.random.randint(0, 6)
        dates = base_dates[start_offset:]
        n_periods_well = len(dates)
        t = np.arange(n_periods_well) / 12.0  # Years
        
        # Generate decline curve
        q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
        
        # Add noise
        noise = np.random.normal(0, 5, len(q_true))
        q_noisy = np.maximum(q_true + noise, 5.0)
        
        well_data[well_id] = pd.DataFrame({
            'oil': q_noisy,
        }, index=dates)
    
    return well_data


def generate_well_locations(n_wells=10):
    """Generate synthetic well locations."""
    np.random.seed(42)
    
    # Generate wells in a roughly grid pattern with some randomness
    locations = []
    for i in range(n_wells):
        well_id = f"WELL_{i+1:03d}"
        # Create a rough grid
        row = i // 3
        col = i % 3
        lat = 32.0 + row * 0.01 + np.random.uniform(-0.002, 0.002)
        lon = -97.0 + col * 0.01 + np.random.uniform(-0.002, 0.002)
        
        locations.append({
            'well_id': well_id,
            'latitude': lat,
            'longitude': lon,
        })
    
    return pd.DataFrame(locations)


def main():
    """Run complete portfolio analysis workflow."""
    print("=" * 80)
    print("COMPLETE PORTFOLIO ANALYSIS DEMO")
    print("=" * 80)
    
    # Step 1: Generate Portfolio Data
    print("\n1. Loading Portfolio Data...")
    well_data = generate_portfolio_data(n_wells=10, n_periods=24)
    well_locations = generate_well_locations(n_wells=10)
    
    print(f"   ✓ Loaded {len(well_data)} wells")
    print(f"   ✓ Average data length: {np.mean([len(df) for df in well_data.values()]):.1f} months")
    
    # Step 2: Portfolio Analysis
    print("\n2. Analyzing Portfolio...")
    econ_spec = EconSpec(
        price_assumptions={'oil': 75.0},
        opex=15.0,
        capex=2000000.0,
        discount_rate=0.10,
    )
    
    portfolio_results = analyze_portfolio(
        well_data,
        model_name='arps_hyperbolic',
        horizon=36,
        econ_spec=econ_spec
    )
    
    print(f"\n   Portfolio Summary:")
    print(f"   ✓ Total Wells: {len(portfolio_results)}")
    print(f"   ✓ Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"   ✓ Total NPV: ${portfolio_results['npv'].sum():,.0f}")
    print(f"   ✓ Average EUR per well: {portfolio_results['eur'].mean():,.0f} STB")
    print(f"   ✓ Average NPV per well: ${portfolio_results['npv'].mean():,.0f}")
    
    # Step 3: Well Ranking
    print("\n3. Ranking Wells...")
    ranked_wells = rank_wells(
        portfolio_results,
        sort_by='npv',
        ascending=False
    )
    
    print(f"\n   Top 5 Wells by NPV:")
    top_5 = ranked_wells.head(5)[['well_id', 'eur', 'npv', 'irr']]
    print(top_5.to_string(index=False))
    
    print(f"\n   Bottom 5 Wells by NPV:")
    bottom_5 = ranked_wells.tail(5)[['well_id', 'eur', 'npv', 'irr']]
    print(bottom_5.to_string(index=False))
    
    # Step 4: Portfolio Aggregation
    print("\n4. Aggregating Portfolio Forecast...")
    portfolio_forecast = aggregate_portfolio_forecast(
        well_data,
        model_name='arps_hyperbolic',
        horizon=36
    )
    
    print(f"   ✓ Generated {len(portfolio_forecast.yhat)} month aggregated forecast")
    print(f"   ✓ Total forecasted production: {portfolio_forecast.yhat.sum():,.0f} STB")
    print(f"   ✓ Peak monthly rate: {portfolio_forecast.yhat.max():.1f} STB/day")
    
    # Step 5: Portfolio Economics
    print("\n5. Evaluating Portfolio Economics...")
    portfolio_econ = evaluate_economics(portfolio_forecast, econ_spec)
    
    print(f"   ✓ Portfolio NPV: ${portfolio_econ.npv:,.0f}")
    if portfolio_econ.irr is not None:
        print(f"   ✓ Portfolio IRR: {portfolio_econ.irr*100:.2f}%")
    print(f"   ✓ Total Revenue: ${portfolio_econ.cashflows['revenue'].sum():,.0f}")
    
    # Step 6: Multi-Well Interaction Analysis
    print("\n6. Analyzing Multi-Well Interactions...")
    interaction_results = analyze_multi_well_interaction(
        well_locations,
        portfolio_results,
        drainage_radius=500.0  # 500 ft drainage radius
    )
    
    print(f"   ✓ Analyzed {len(interaction_results)} well pairs")
    print(f"   ✓ Average well spacing: {interaction_results['distance'].mean():.0f} ft")
    print(f"   ✓ Wells with overlap: {(interaction_results['overlap_ratio'] > 0).sum()}")
    
    if (interaction_results['overlap_ratio'] > 0).any():
        overlapping = interaction_results[interaction_results['overlap_ratio'] > 0]
        print(f"\n   Wells with Drainage Overlap:")
        for _, row in overlapping.head(5).iterrows():
            print(f"     {row['well_1']} <-> {row['well_2']}: "
                  f"{row['overlap_ratio']*100:.1f}% overlap, "
                  f"{row['distance']:.0f} ft apart")
    
    # Step 7: Field Spacing Optimization
    print("\n7. Optimizing Field Spacing...")
    spacing_result = optimize_field_spacing(
        well_locations,
        portfolio_results,
        min_spacing=400.0,  # Minimum 400 ft spacing
        target_eur_reduction=0.05  # Accept up to 5% EUR reduction
    )
    
    print(f"   ✓ Optimized spacing recommendations:")
    print(f"     - Current average spacing: {spacing_result.get('current_avg_spacing', 0):.0f} ft")
    print(f"     - Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    print(f"     - Expected EUR impact: {spacing_result.get('eur_impact', 0)*100:.1f}%")
    
    # Step 8: Portfolio Statistics
    print("\n8. Portfolio Statistics...")
    print(f"\n   EUR Distribution:")
    print(f"     - Mean: {portfolio_results['eur'].mean():,.0f} STB")
    print(f"     - Median: {portfolio_results['eur'].median():,.0f} STB")
    print(f"     - Std Dev: {portfolio_results['eur'].std():,.0f} STB")
    print(f"     - Min: {portfolio_results['eur'].min():,.0f} STB")
    print(f"     - Max: {portfolio_results['eur'].max():,.0f} STB")
    
    print(f"\n   NPV Distribution:")
    print(f"     - Mean: ${portfolio_results['npv'].mean():,.0f}")
    print(f"     - Median: ${portfolio_results['npv'].median():,.0f}")
    print(f"     - Std Dev: ${portfolio_results['npv'].std():,.0f}")
    print(f"     - Positive NPV wells: {(portfolio_results['npv'] > 0).sum()}/{len(portfolio_results)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PORTFOLIO ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Total Wells: {len(well_data)}")
    print(f"  • Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"  • Total NPV: ${portfolio_econ.npv:,.0f}")
    print(f"  • Best Well: {ranked_wells.iloc[0]['well_id']} (NPV: ${ranked_wells.iloc[0]['npv']:,.0f})")
    print(f"  • Average Well Spacing: {spacing_result.get('current_avg_spacing', 0):.0f} ft")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

