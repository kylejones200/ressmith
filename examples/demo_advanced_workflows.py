"""
Advanced Workflows Demo

This demo showcases advanced reservoir engineering workflows:
1. Type curve matching
2. Advanced RTA (Blasingame, FMB)
3. Multi-phase forecasting
4. Coning analysis
5. EOR pattern analysis
6. Fracture network analysis

Run: uv run python examples/demo_advanced_workflows.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ressmith import (
    analyze_blasingame,
    analyze_fmb,
    analyze_fracture_network,
    analyze_waterflood,
    analyze_well_coning,
    forecast_with_yields,
    match_type_curve_workflow,
)


def generate_production_data_with_pressure(n_periods: int = 48) -> pd.DataFrame:
    """Generate synthetic production data with pressure."""
    rng = np.random.default_rng(42)
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq="ME")
    t = np.arange(n_periods) / 12.0

    qi = 800.0
    di = 0.7
    b = 0.55
    q = qi / (1.0 + b * di * t) ** (1.0 / b)

    pi = 5000.0
    p = pi - (q.cumsum() / 50000.0) * 1000.0

    oil = np.maximum(q + rng.normal(0, 10, len(q)), 1.0)
    return pd.DataFrame(
        {
            "oil": oil,
            "gas": oil * 1200 + rng.normal(0, 100, len(q)),
            "water": np.maximum(oil * 0.08 + rng.normal(0, 2, len(q)), 0.0),
            "pressure": np.maximum(p, 2000),
            "cumulative_oil": np.cumsum(oil),
        },
        index=time_index,
    )


def main() -> None:
    """Run advanced workflows demo."""
    print("=" * 80)
    print("ADVANCED WORKFLOWS DEMO")
    print("=" * 80)

    print("\n1. Type Curve Matching...")
    data = generate_production_data_with_pressure(n_periods=36)

    try:
        type_curve_result = match_type_curve_workflow(
            data,
            rate_col="oil",
        )
        print("   ✓ Type curve matched")
        best = type_curve_result.get("best_match", {})
        print("   ✓ Best match parameters:")
        for key, value in best.get("matched_params", {}).items():
            print(f"     - {key}: {value:.4f}")
        print(f"   ✓ Match error: {best.get('match_error', 0):.4f}")
        print(f"   ✓ Match type: {best.get('type', 'n/a')}")
    except Exception as e:
        print(f"   ⚠ Type curve matching failed: {e}")

    print("\n2. Blasingame Type Curve Analysis...")
    try:
        blasingame_result = analyze_blasingame(
            data,
            rate_col="oil",
            pressure_col="pressure",
            initial_pressure=5000.0,
        )
        print("   ✓ Blasingame analysis complete")
        print(f"   ✓ Flow regime: {blasingame_result.get('flow_regime', 'unknown')}")
        print(f"   ✓ Permeability: {blasingame_result.get('permeability', 0):.4f} md")
        print(f"   ✓ Drainage area: {blasingame_result.get('drainage_area', 0):.2f} acres")
    except Exception as e:
        print(f"   ⚠ Blasingame analysis failed: {e}")

    print("\n3. Flowing Material Balance (FMB) Analysis...")
    try:
        fmb_result = analyze_fmb(
            data,
            cumulative_col="cumulative_oil",
            pressure_col="pressure",
            initial_pressure=5000.0,
            formation_volume_factor=1.2,
        )
        print("   ✓ FMB analysis complete")
        print(f"   ✓ OOIP estimate: {fmb_result.get('estimated_ooip', 0):,.0f} STB")
        print(f"   ✓ Recovery factor: {fmb_result.get('recovery_factor', 0) * 100:.2f}%")
    except Exception as e:
        print(f"   ⚠ FMB analysis failed: {e}")

    print("\n4. Fracture Network Analysis...")
    try:
        early = data.iloc[:24]
        fracture_result = analyze_fracture_network(
            early,
            rate_col="oil",
            number_of_stages=20,
            stage_spacing=300.0,
        )
        print("   ✓ Fracture analysis complete")
        print(f"   ✓ SRV estimate: {fracture_result.get('estimated_srv', 0):,.0f} acre-ft")
        print(
            f"   ✓ Effective fracture half-length: "
            f"{fracture_result.get('effective_fracture_half_length', 0):.1f} ft"
        )
        if "estimated_decline_rate" in fracture_result:
            print(
                f"   ✓ Estimated decline rate: "
                f"{fracture_result['estimated_decline_rate']:.4f}"
            )
    except Exception as e:
        print(f"   ⚠ Fracture analysis failed: {e}")

    print("\n5. Multi-Phase Forecasting with Yields...")
    multiphase_data = generate_production_data_with_pressure(n_periods=30)
    try:
        multiphase_result = forecast_with_yields(
            multiphase_data,
            primary_phase="oil",
            associated_phases=["gas", "water"],
            model_name="arps_hyperbolic",
            yield_models={
                "gas": "constant",
                "water": "hyperbolic",
            },
            horizon=24,
        )
        print("   ✓ Multi-phase forecast generated")
        for phase, forecast_result in multiphase_result.items():
            print(f"   ✓ {phase.upper()}: {forecast_result.yhat.sum():,.0f} total forecasted")
    except Exception as e:
        print(f"   ⚠ Multi-phase forecast failed: {e}")

    print("\n6. Well Coning Analysis...")
    try:
        current_rate = 500.0
        coning_result = analyze_well_coning(
            production_rate=current_rate,
            oil_density=50.0,  # lb/ft³
            water_density=62.4,  # lb/ft³
            permeability=50.0,  # md
            reservoir_thickness=100.0,  # ft
            well_completion_interval=20.0,  # ft
            method="meyer_gardner",
            oil_viscosity=2.0,
            porosity=0.15,
        )
        critical = coning_result.get("critical_rate", 0)
        print("   ✓ Coning analysis complete")
        print(f"   ✓ Critical rate ({coning_result.get('method', 'n/a')}): {critical:.1f} STB/day")
        print(f"   ✓ Coning index: {coning_result.get('coning_index', 0):.3f}")
        print(f"   ✓ Current rate vs critical: {current_rate / max(critical, 1e-6):.2f}x")
        print(f"   ✓ Coning risk: {coning_result.get('coning_risk', 'unknown')}")
        if coning_result.get("breakthrough_time"):
            print(f"   ✓ Estimated breakthrough time: {coning_result['breakthrough_time']:.1f} days")
    except Exception as e:
        print(f"   ⚠ Coning analysis failed: {e}")

    print("\n7. Waterflood Pattern Analysis...")
    try:
        waterflood_result = analyze_waterflood(
            pattern_type="five_spot",
            injection_rate=1000.0,
            production_rate=800.0,
            mobility_ratio=0.5,
            oil_saturation_initial=0.70,
            oil_saturation_residual=0.25,
            pore_volumes_injected=0.3,
        )
        print("   ✓ Waterflood analysis complete")
        print(f"   ✓ Sweep efficiency: {waterflood_result.get('sweep_efficiency', 0) * 100:.1f}%")
        print(
            f"   ✓ Injection efficiency: "
            f"{waterflood_result.get('injection_efficiency', 0) * 100:.1f}%"
        )
        print(
            f"   ✓ Displacement efficiency: "
            f"{waterflood_result.get('displacement_efficiency', 0) * 100:.1f}%"
        )
        print(
            f"   ✓ Recovery efficiency: "
            f"{waterflood_result.get('recovery_efficiency', 0) * 100:.1f}%"
        )
    except Exception as e:
        print(f"   ⚠ Waterflood analysis failed: {e}")

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
