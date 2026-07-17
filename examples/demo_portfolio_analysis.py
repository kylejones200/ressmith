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

Run: uv run python examples/demo_portfolio_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ressmith import (
    aggregate_portfolio_forecast,
    analyze_multi_well_interaction,
    analyze_portfolio,
    evaluate_economics,
    optimize_field_spacing,
    rank_wells,
)
from ressmith.objects.domain import EconSpec

DEFAULT_DRAINAGE_RADIUS_FT = 500.0


def generate_portfolio_data(n_wells: int = 10, n_periods: int = 24) -> dict[str, pd.DataFrame]:
    """Generate synthetic production data for multiple wells."""
    rng = np.random.default_rng(42)

    well_data: dict[str, pd.DataFrame] = {}
    base_dates = pd.date_range("2020-01-01", periods=n_periods, freq="ME")

    for i in range(n_wells):
        well_id = f"WELL_{i + 1:03d}"

        qi = 300 + rng.uniform(-100, 200)
        di = 0.5 + rng.uniform(-0.2, 0.3)
        b = max(0.1, 0.4 + rng.uniform(-0.2, 0.4))

        start_offset = int(rng.integers(0, 6))
        dates = base_dates[start_offset:]
        n_periods_well = len(dates)
        t = np.arange(n_periods_well) / 12.0

        q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
        noise = rng.normal(0, 5, len(q_true))
        q_noisy = np.maximum(q_true + noise, 5.0)

        well_data[well_id] = pd.DataFrame({"oil": q_noisy}, index=dates)

    return well_data


def generate_well_locations(n_wells: int = 10) -> pd.DataFrame:
    """Generate synthetic well locations."""
    rng = np.random.default_rng(42)

    locations = []
    for i in range(n_wells):
        well_id = f"WELL_{i + 1:03d}"
        row = i // 3
        col = i % 3
        lat = 32.0 + row * 0.01 + rng.uniform(-0.002, 0.002)
        lon = -97.0 + col * 0.01 + rng.uniform(-0.002, 0.002)
        locations.append(
            {
                "well_id": well_id,
                "latitude": lat,
                "longitude": lon,
            }
        )

    return pd.DataFrame(locations)


def drainage_radii_for(well_locations: pd.DataFrame, radius_ft: float = DEFAULT_DRAINAGE_RADIUS_FT) -> dict[str, float]:
    """Uniform drainage radius map keyed by well_id."""
    return {wid: radius_ft for wid in well_locations["well_id"]}


def main() -> None:
    """Run complete portfolio analysis workflow."""
    print("=" * 80)
    print("COMPLETE PORTFOLIO ANALYSIS DEMO")
    print("=" * 80)

    print("\n1. Loading Portfolio Data...")
    well_data = generate_portfolio_data(n_wells=10, n_periods=24)
    well_locations = generate_well_locations(n_wells=10)
    drainage_radii = drainage_radii_for(well_locations)

    print(f"   ✓ Loaded {len(well_data)} wells")
    print(f"   ✓ Average data length: {np.mean([len(df) for df in well_data.values()]):.1f} months")

    print("\n2. Analyzing Portfolio...")
    econ_spec = EconSpec(
        price_assumptions={"oil": 75.0},
        opex=15.0,
        capex=2_000_000.0,
        discount_rate=0.10,
    )

    portfolio_results = analyze_portfolio(
        well_data,
        model_name="arps_hyperbolic",
        horizon=36,
        econ_spec=econ_spec,
    )

    print("\n   Portfolio Summary:")
    print(f"   ✓ Total Wells: {len(portfolio_results)}")
    print(f"   ✓ Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"   ✓ Total NPV: ${portfolio_results['npv'].sum():,.0f}")
    print(f"   ✓ Average EUR per well: {portfolio_results['eur'].mean():,.0f} STB")
    print(f"   ✓ Average NPV per well: ${portfolio_results['npv'].mean():,.0f}")

    print("\n3. Ranking Wells...")
    ranked_wells = rank_wells(
        portfolio_results,
        metric="npv",
        ascending=False,
    )

    print("\n   Top 5 Wells by NPV:")
    top_5 = ranked_wells.head(5)[["well_id", "eur", "npv", "irr"]]
    print(top_5.to_string(index=False))

    print("\n   Bottom 5 Wells by NPV:")
    bottom_5 = ranked_wells.tail(5)[["well_id", "eur", "npv", "irr"]]
    print(bottom_5.to_string(index=False))

    print("\n4. Aggregating Portfolio Forecast...")
    portfolio_forecast = aggregate_portfolio_forecast(
        well_data,
        model_name="arps_hyperbolic",
        horizon=36,
    )

    print(f"   ✓ Generated {len(portfolio_forecast.yhat)} month aggregated forecast")
    print(f"   ✓ Total forecasted production: {portfolio_forecast.yhat.sum():,.0f} STB")
    print(f"   ✓ Peak monthly rate: {portfolio_forecast.yhat.max():.1f} STB/day")

    print("\n5. Evaluating Portfolio Economics...")
    portfolio_econ = evaluate_economics(portfolio_forecast, econ_spec)

    print(f"   ✓ Portfolio NPV: ${portfolio_econ.npv:,.0f}")
    if portfolio_econ.irr is not None:
        print(f"   ✓ Portfolio IRR: {portfolio_econ.irr * 100:.2f}%")
    print(f"   ✓ Total Revenue: ${portfolio_econ.cashflows['revenue'].sum():,.0f}")

    print("\n6. Analyzing Multi-Well Interactions...")
    interaction_results = analyze_multi_well_interaction(
        well_locations,
        drainage_radii=drainage_radii,
        production_data=well_data,
    )

    well_pairs = interaction_results["interaction"]["well_pairs"]
    total_overlap = interaction_results["interaction"]["total_overlap"]
    avg_interference = interaction_results["interaction"]["average_interference"]
    overlapping_pairs = [p for p in well_pairs if p.get("overlap_area", 0) > 0]

    print(f"   ✓ Analyzed {len(well_pairs)} well pairs")
    print(f"   ✓ Total drainage overlap: {total_overlap:.2f}")
    print(f"   ✓ Average interference: {avg_interference:.3f}")
    print(f"   ✓ Pairs with overlap: {len(overlapping_pairs)}")

    if overlapping_pairs:
        print("\n   Wells with Drainage Overlap:")
        for pair in overlapping_pairs[:5]:
            print(
                f"     {pair['well_id_1']} <-> {pair['well_id_2']}: "
                f"overlap_area={pair['overlap_area']:.2f}, "
                f"{pair['distance']:.0f} ft apart"
            )

    print("\n7. Optimizing Field Spacing...")
    spacing_result = optimize_field_spacing(
        well_locations,
        drainage_radii,
        min_spacing=400.0,
        target_interference=5.0,
    )

    print("   ✓ Optimized spacing recommendations:")
    print(f"     - Recommended spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    print(
        f"     - Current total overlap: "
        f"{spacing_result.get('current_total_overlap', 0):.2f}"
    )
    print(
        f"     - Average interference: "
        f"{spacing_result.get('average_interference', 0):.1f}%"
    )

    print("\n8. Portfolio Statistics...")
    print("\n   EUR Distribution:")
    print(f"     - Mean: {portfolio_results['eur'].mean():,.0f} STB")
    print(f"     - Median: {portfolio_results['eur'].median():,.0f} STB")
    print(f"     - Std Dev: {portfolio_results['eur'].std():,.0f} STB")
    print(f"     - Min: {portfolio_results['eur'].min():,.0f} STB")
    print(f"     - Max: {portfolio_results['eur'].max():,.0f} STB")

    print("\n   NPV Distribution:")
    print(f"     - Mean: ${portfolio_results['npv'].mean():,.0f}")
    print(f"     - Median: ${portfolio_results['npv'].median():,.0f}")
    print(f"     - Std Dev: ${portfolio_results['npv'].std():,.0f}")
    print(
        f"     - Positive NPV wells: "
        f"{(portfolio_results['npv'] > 0).sum()}/{len(portfolio_results)}"
    )

    print("\n" + "=" * 80)
    print("PORTFOLIO ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  • Total Wells: {len(well_data)}")
    print(f"  • Total EUR: {portfolio_results['eur'].sum():,.0f} STB")
    print(f"  • Total NPV: ${portfolio_econ.npv:,.0f}")
    print(
        f"  • Best Well: {ranked_wells.iloc[0]['well_id']} "
        f"(NPV: ${ranked_wells.iloc[0]['npv']:,.0f})"
    )
    print(f"  • Recommended Spacing: {spacing_result.get('recommended_spacing', 0):.0f} ft")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
