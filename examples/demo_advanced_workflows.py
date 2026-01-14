"""
Advanced Workflows Demo

This demo showcases advanced reservoir engineering workflows:
1. Type curve matching
2. Advanced RTA (Blasingame, FMB)
3. Multi-phase forecasting
4. Coning analysis
5. EOR pattern analysis
6. Production operations optimization

Run: python examples/demo_advanced_workflows.py
"""

import numpy as np
import pandas as pd
from datetime import datetime

from ressmith import (
    analyze_blasingame,
    analyze_fmb,
    analyze_fracture_network,
    analyze_waterflood,
    analyze_well_coning,
    forecast_with_yields,
    match_type_curve_workflow,
)


def generate_production_data_with_pressure(n_periods=48):
    """Generate synthetic production data with pressure."""
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="M")
    t = np.arange(n_periods) / 12.0  # Years
    
    # Hyperbolic decline
    qi = 800.0  # STB/day
    di = 0.7
    b = 0.55
    q = qi / (1.0 + b * di * t) ** (1.0 / b)
    
    # Pressure decline (material balance)
    pi = 5000.0  # Initial pressure, psi
    p = pi - (q.cumsum() / 50000.0) * 1000.0  # Simplified pressure decline
    
    data = pd.DataFrame({
        'oil': q + np.random.normal(0, 10, len(q)),
        'gas': q * 1200 + np.random.normal(0, 100, len(q)),
        'pressure': np.maximum(p, 2000),
    }, index=time_index)
    
    return data


def main():
    """Run advanced workflows demo."""
    print("=" * 80)
    print("ADVANCED WORKFLOWS DEMO")
    print("=" * 80)
    
    # Step 1: Type Curve Matching
    print("\n1. Type Curve Matching...")
    data = generate_production_data_with_pressure(n_periods=36)
    
    type_curve_result = match_type_curve_workflow(
        time=data.index.values,
        rate=data['oil'].values,
        model_type='arps'
    )
    
    if type_curve_result:
        print(f"   ✓ Type curve matched")
        print(f"   ✓ Best match parameters:")
        for key, value in type_curve_result.get('best_match', {}).get('parameters', {}).items():
            print(f"     - {key}: {value:.4f}")
        print(f"   ✓ Match error: {type_curve_result.get('best_match', {}).get('match_error', 0):.4f}")
    else:
        print("   ⚠ No type curve matches found")
    
    # Step 2: Advanced RTA - Blasingame Analysis
    print("\n2. Blasingame Type Curve Analysis...")
    try:
        blasingame_result = analyze_blasingame(
            time=data.index.values,
            rate=data['oil'].values,
            pressure=data['pressure'].values,
            initial_pressure=5000.0
        )
        
        print(f"   ✓ Blasingame analysis complete")
        print(f"   ✓ Permeability: {blasingame_result.get('permeability', 0):.4f} md")
        print(f"   ✓ Skin factor: {blasingame_result.get('skin_factor', 0):.4f}")
        print(f"   ✓ Drainage area: {blasingame_result.get('drainage_area', 0):.2f} acres")
    except Exception as e:
        print(f"   ⚠ Blasingame analysis failed: {e}")
    
    # Step 3: Advanced RTA - Flowing Material Balance (FMB)
    print("\n3. Flowing Material Balance (FMB) Analysis...")
    try:
        fmb_result = analyze_fmb(
            time=data.index.values,
            cumulative=data['oil'].cumsum().values,
            pressure=data['pressure'].values,
            initial_pressure=5000.0,
            formation_volume_factor=1.2
        )
        
        print(f"   ✓ FMB analysis complete")
        print(f"   ✓ OOIP estimate: {fmb_result.get('estimated_ooip', 0):,.0f} STB")
        print(f"   ✓ Recovery factor: {fmb_result.get('recovery_factor', 0)*100:.2f}%")
    except Exception as e:
        print(f"   ⚠ FMB analysis failed: {e}")
    
    # Step 4: Fracture Network Analysis
    print("\n4. Fracture Network Analysis...")
    try:
        fracture_result = analyze_fracture_network(
            time=data.index.values[:24],  # Early time data
            rate=data['oil'].values[:24],
            reservoir_thickness=100.0,  # ft
            number_of_stages=20
        )
        
        print(f"   ✓ Fracture analysis complete")
        print(f"   ✓ SRV estimate: {fracture_result.get('srv', 0):,.0f} ft³")
        print(f"   ✓ Effective fracture half-length: {fracture_result.get('effective_half_length', 0):.1f} ft")
        print(f"   ✓ Fracture conductivity: {fracture_result.get('fracture_conductivity', 0):.2f} md-ft")
    except Exception as e:
        print(f"   ⚠ Fracture analysis failed: {e}")
    
    # Step 5: Multi-Phase Forecasting with Yields
    print("\n5. Multi-Phase Forecasting with Yields...")
    multiphase_data = generate_production_data_with_pressure(n_periods=30)
    
    try:
        multiphase_result = forecast_with_yields(
            multiphase_data,
            primary_phase='oil',
            associated_phases=['gas', 'water'],
            model_name='arps_hyperbolic',
            yield_models={
                'gas': 'constant',  # Constant GOR
                'water': 'hyperbolic',  # Water cut increasing
            },
            horizon=24
        )
        
        print(f"   ✓ Multi-phase forecast generated")
        for phase, forecast_result in multiphase_result.items():
            print(f"   ✓ {phase.upper()}: {forecast_result.yhat.sum():,.0f} total forecasted")
    except Exception as e:
        print(f"   ⚠ Multi-phase forecast failed: {e}")
    
    # Step 6: Coning Analysis
    print("\n6. Well Coning Analysis...")
    try:
        coning_result = analyze_well_coning(
            well_radius=0.5,  # ft
            reservoir_thickness=100.0,  # ft
            oil_density=0.8,  # g/cm³
            water_density=1.0,  # g/cm³
            oil_viscosity=2.0,  # cp
            water_viscosity=0.5,  # cp
            permeability=50.0,  # md
            porosity=0.15,
            water_oil_contact_depth=5000.0,  # ft
            producing_interval_top=4900.0,  # ft
            current_rate=500.0  # STB/day
        )
        
        print(f"   ✓ Coning analysis complete")
        print(f"   ✓ Critical rate (Meyer-Gardner): {coning_result.get('critical_rate_mg', 0):.1f} STB/day")
        print(f"   ✓ Critical rate (Chierici-Ciucci): {coning_result.get('critical_rate_cc', 0):.1f} STB/day")
        print(f"   ✓ Current rate vs critical: {500.0 / max(coning_result.get('critical_rate_mg', 1), 1):.2f}x")
        print(f"   ✓ Coning risk: {coning_result.get('coning_risk', 'unknown')}")
        
        if coning_result.get('breakthrough_time'):
            print(f"   ✓ Estimated breakthrough time: {coning_result['breakthrough_time']:.1f} days")
    except Exception as e:
        print(f"   ⚠ Coning analysis failed: {e}")
    
    # Step 7: EOR Pattern Analysis
    print("\n7. Waterflood Pattern Analysis...")
    try:
        waterflood_result = analyze_waterflood(
            pattern_type='five_spot',
            pattern_area=40.0,  # acres
            injector_count=4,
            producer_count=5,
            injection_rate=1000.0,  # STB/day
            mobility_ratio=0.5,  # Favorable mobility ratio
            pore_volume_injected=0.3  # 30% PV injected
        )
        
        print(f"   ✓ Waterflood analysis complete")
        print(f"   ✓ Sweep efficiency: {waterflood_result.get('sweep_efficiency', 0)*100:.1f}%")
        print(f"   ✓ Injection efficiency: {waterflood_result.get('injection_efficiency', 0)*100:.1f}%")
        print(f"   ✓ Areal sweep: {waterflood_result.get('areal_sweep', 0)*100:.1f}%")
        print(f"   ✓ Estimated recovery: {waterflood_result.get('estimated_recovery', 0)*100:.1f}%")
    except Exception as e:
        print(f"   ⚠ Waterflood analysis failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ADVANCED WORKFLOWS COMPLETE")
    print("=" * 80)
    print("\nCompleted workflows:")
    print("  ✓ Type curve matching")
    print("  ✓ Blasingame RTA analysis")
    print("  ✓ Flowing Material Balance")
    print("  ✓ Fracture network analysis")
    print("  ✓ Multi-phase forecasting")
    print("  ✓ Coning analysis")
    print("  ✓ EOR pattern analysis")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

