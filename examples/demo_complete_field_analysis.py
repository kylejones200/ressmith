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

Run: uv run python examples/demo_complete_field_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ressmith import (
    aggregate_portfolio_forecast,
    analyze_blasingame,
    analyze_multi_well_interaction,
    analyze_portfolio,
    analyze_waterflood,
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

DEFAULT_DRAINAGE_RADIUS_FT = 600.0


def generate_field_data(
    n_wells: int = 8, n_periods: int = 30
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate synthetic field data with multiple wells and phases."""
    rng = np.random.default_rng(42)

    well_data: dict[str, pd.DataFrame] = {}
    well_locations = []
    base_dates = pd.date_range("2020-01-01", periods=n_periods, freq="ME")

    for i in range(n_wells):
        well_id = f"WELL_{i + 1:03d}"

        qi = 400 + rng.uniform(-150, 250)
        di = 0.6 + rng.uniform(-0.2, 0.3)
        b = max(0.1, 0.5 + rng.uniform(-0.2, 0.3))

        start_offset = int(rng.integers(0, 6))
        dates = base_dates[start_offset:]
        n_periods_well = len(dates)
        t = np.arange(n_periods_well) / 12.0

        q_oil = qi / (1.0 + b * di * t) ** (1.0 / b)
        noise = rng.normal(0, 8, len(q_oil))
        q_oil = np.maximum(q_oil + noise, 10.0)

        q_gas = q_oil * (1000 + rng.uniform(-200, 200))
        q_water = np.maximum(q_oil * (0.05 + rng.uniform(0, 0.15)), 0.0)

        pi = 4500 + rng.uniform(-500, 500)
        p = pi - (q_oil.cumsum() / 50000.0) * 800.0
        pressure = np.maximum(p, 2000)

        well_data[well_id] = pd.DataFrame(
            {
                "oil": q_oil,
                "gas": q_gas,
                "water": q_water,
                "pressure": pressure,
            },
            index=dates,
        )

        row = i // 3
        col = i % 3
        well_locations.append(
            {
                "well_id": well_id,
                "latitude": 32.0 + row * 0.015,
                "longitude": -97.0 + col * 0.015,
            }
        )

    return well_data, pd.DataFrame(well_locations)


def petroleum_percentiles(prob_result: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map statistical p10/p90 keys to petroleum P10 (high) / P50 / P90 (low)."""
    return prob_result["p90"], prob_result["p50"], prob_result["p10"]


def main() -> None:
    """Run complete field analysis workflow."""
    print("=" * 80)
    print("COMPLETE FIELD ANALYSIS DEMO")
    print("=" * 80)

    print("\n1. Loading Field Data...")
    well_data, well_locations = generate_field_data(n_wells=8, n_periods=30)
    drainage_radii = {
        wid: DEFAULT_DRAINAGE_RADIUS_FT for wid in well_locations["well_id"]
    }
    print(f"   ✓ Loaded {len(well_data)} wells")
    print(f"   ✓ Average data length: {np.mean([len(df) for df in well_data.values()]):.1f} months")
    print(f"   ✓ Field coverage: {len(well_locations)} wells")

    print("\n2. Portfolio Analysis...")
    econ_spec = EconSpec(
        price_assumptions={"oil": 75.0, "gas": 3.5},
        opex=15.0,
        capex=2_500_000.0,
        discount_rate=0.10,
    )

    portfolio_results = analyze_portfolio(
        well_data,
        model_name="arps_hyperbolic",
        horizon=36,
        econ_spec=econ_spec,
    )

    print(f"   ✓ Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"   ✓ Total NPV: ${portfolio_results['npv'].sum():,.0f}")

    print("\n3. Well Ranking & Selection...")
    ranked_wells = rank_wells(portfolio_results, metric="npv", ascending=False)
    print(
        f"   ✓ Top well: {ranked_wells.iloc[0]['well_id']} "
        f"(NPV: ${ranked_wells.iloc[0]['npv']:,.0f})"
    )

    print("\n4. Ensemble Forecasting...")
    top_wells = {
        well_id: well_data[well_id]
        for well_id in ranked_wells.head(5)["well_id"].values
    }

    portfolio_forecast = aggregate_portfolio_forecast(
        top_wells,
        model_name="arps_hyperbolic",
        horizon=36,
    )
    print(f"   ✓ Portfolio forecast: {portfolio_forecast.yhat.sum():,.0f} STB total")

    print("\n5. Probabilistic Forecasting (P10/P50/P90)...")
    top_well_id = ranked_wells.iloc[0]["well_id"]
    top_well_data = well_data[top_well_id]

    prob_result = probabilistic_forecast(
        top_well_data,
        model_name="arps_hyperbolic",
        horizon=36,
        n_samples=1000,
        seed=42,
    )

    p10, p50, p90 = petroleum_percentiles(prob_result)
    p10_eur = float(p10.sum())
    p50_eur = float(p50.sum())
    p90_eur = float(p90.sum())

    print(f"   ✓ P50 cumulative: {p50_eur:,.0f} STB")
    print(f"   ✓ Uncertainty half-width: ±{((p10_eur - p90_eur) / 2 / p50_eur * 100):.1f}%")

    print("\n6. Multi-Well Interaction Analysis...")
    interaction_results = analyze_multi_well_interaction(
        well_locations,
        drainage_radii=drainage_radii,
        production_data=well_data,
    )

    well_pairs = interaction_results["interaction"]["well_pairs"]
    overlap_count = sum(1 for p in well_pairs if p.get("overlap_area", 0) > 0)
    print(f"   ✓ Well pairs analyzed: {len(well_pairs)}")
    print(f"   ✓ Pairs with overlap: {overlap_count}")

    if overlap_count > 0:
        overlaps = [p["overlap_area"] for p in well_pairs if p.get("overlap_area", 0) > 0]
        print(f"   ✓ Average overlap area: {np.mean(overlaps):.2f}")

    print("\n7. Field Spacing Optimization...")
    spacing_result = optimize_field_spacing(
        well_locations,
        drainage_radii,
        min_spacing=500.0,
        target_interference=5.0,
    )
    print(f"   ✓ Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")

    print("\n8. Type Curve Matching...")
    sample_well_id = ranked_wells.iloc[1]["well_id"]
    sample_data = well_data[sample_well_id]

    try:
        type_curve_result = match_type_curve_workflow(sample_data, rate_col="oil")
        if type_curve_result:
            print(f"   ✓ Type curve matched for {sample_well_id}")
            best = type_curve_result.get("best_match", {})
            print(f"     - Match type: {best.get('type', 'n/a')}")
            print(f"     - Match error: {best.get('match_error', 0):.4f}")
    except Exception as e:
        print(f"   ⚠ Type curve matching: {str(e)[:80]}")

    print("\n9. Advanced RTA Analysis...")
    best_well_data = well_data[top_well_id]
    if "pressure" in best_well_data.columns:
        try:
            blasingame_result = analyze_blasingame(
                best_well_data,
                rate_col="oil",
                pressure_col="pressure",
                initial_pressure=float(best_well_data["pressure"].iloc[0]),
            )
            print("   ✓ Blasingame analysis complete")
            print(f"     - Flow regime: {blasingame_result.get('flow_regime', 'unknown')}")
            print(f"     - Permeability: {blasingame_result.get('permeability', 0):.2f} md")
        except Exception as e:
            print(f"   ⚠ Blasingame analysis: {str(e)[:50]}...")

    print("\n10. Multi-Phase Forecasting...")
    try:
        multiphase_result = forecast_with_yields(
            top_well_data,
            primary_phase="oil",
            associated_phases=["gas", "water"],
            model_name="arps_hyperbolic",
            yield_models={"gas": "constant", "water": "hyperbolic"},
            horizon=24,
        )
        total_oil = multiphase_result["oil"].yhat.sum()
        total_gas = multiphase_result["gas"].yhat.sum() if "gas" in multiphase_result else 0
        print("   ✓ Multi-phase forecast generated")
        print(f"     - Oil: {total_oil:,.0f} STB")
        if total_gas > 0:
            print(f"     - Gas: {total_gas:,.0f}")
    except Exception as e:
        print(f"   ⚠ Multi-phase forecast: {str(e)[:50]}...")

    print("\n11. EOR Pattern Analysis...")
    try:
        waterflood_result = analyze_waterflood(
            pattern_type="five_spot",
            injection_rate=1500.0,
            production_rate=1200.0,
            mobility_ratio=0.6,
            oil_saturation_initial=0.70,
            oil_saturation_residual=0.25,
            pore_volumes_injected=0.25,
        )
        print("   ✓ Waterflood analysis complete")
        print(
            f"     - Sweep efficiency: "
            f"{waterflood_result.get('sweep_efficiency', 0) * 100:.1f}%"
        )
        print(
            f"     - Recovery efficiency: "
            f"{waterflood_result.get('recovery_efficiency', 0) * 100:.1f}%"
        )
    except Exception as e:
        print(f"   ⚠ Waterflood analysis: {str(e)[:50]}...")

    print("\n12. Portfolio Economics...")
    portfolio_econ = evaluate_economics(portfolio_forecast, econ_spec)
    print(f"   ✓ Portfolio NPV: ${portfolio_econ.npv:,.0f}")
    if portfolio_econ.irr:
        print(f"   ✓ Portfolio IRR: {portfolio_econ.irr * 100:.2f}%")

    print("\n13. Scenario Analysis...")
    scenarios = {
        "base_case": {},
        "high_price": {"prices": {"oil": 90.0, "gas": 4.0}},
        "low_price": {"prices": {"oil": 60.0, "gas": 3.0}},
        "high_opex": {"opex": 20.0},
    }
    scenario_results = evaluate_scenarios(portfolio_forecast, econ_spec, scenarios)
    scenario_df = scenario_summary(scenario_results)
    print(f"   ✓ Evaluated {len(scenarios)} scenarios")
    print("\n   Scenario NPVs:")
    for _, row in scenario_df.iterrows():
        print(f"     {row['scenario']}: ${row['npv']:,.0f}")

    print("\n" + "=" * 80)
    print("COMPLETE FIELD ANALYSIS SUMMARY")
    print("=" * 80)
    print("\nField Overview:")
    print(f"  • Total Wells: {len(well_data)}")
    print(f"  • Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"  • Total NPV: ${portfolio_econ.npv:,.0f}")
    print(
        f"  • Portfolio IRR: {portfolio_econ.irr * 100:.2f}%"
        if portfolio_econ.irr
        else "  • IRR: N/A"
    )

    print("\nBest Performing Well:")
    print(f"  • Well ID: {ranked_wells.iloc[0]['well_id']}")
    print(f"  • EUR: {ranked_wells.iloc[0]['eur']:,.0f} STB")
    print(f"  • NPV: ${ranked_wells.iloc[0]['npv']:,.0f}")

    print("\nField Optimization:")
    print(f"  • Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    print(f"  • Well pairs with overlap: {overlap_count}")

    print("\nUncertainty:")
    print(f"  • P50 cumulative (best well): {p50_eur:,.0f} STB")
    print(f"  • Uncertainty half-width: ±{((p10_eur - p90_eur) / 2 / p50_eur * 100):.1f}%")

    print("\n" + "=" * 80)
    print("Analysis complete! Review outputs for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
