"""
Integrated study workflows: interference, coning, enhanced RTA.

Demonstrates the bundled entry points (single-call studies) that wrap
existing Layer-4 workflows:

- ``well_interference_study`` — distances, geometric interference, optional
  production/EUR-based interference, spacing hint
- ``coning_study`` — critical rate / risk, optional WOR–GOR vs time
- ``enhanced_rta_study`` — normalization, type-curve match, Blasingame, DN,
  diagnostic plot data (optional FMB if you add cumulative + pressure)

Run from repository root::

    uv run python examples/integrated_studies.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ressmith import (
    coning_study,
    enhanced_rta_study,
    well_interference_study,
)

# --- Shared demo RNG / calendar ---
DEMO_SEED = 42
MONTH_END_FREQ = "ME"

# --- Interference demo: well locations (degrees) ---
WELL_IDS = ("N-01", "N-02", "N-03")
WELL_LATITUDES = (32.10, 32.105, 32.11)
WELL_LONGITUDES = (-101.55, -101.545, -101.54)
INTERFERENCE_HISTORY_PERIODS = 36
INTERFERENCE_HISTORY_START = "2019-01-01"
PROD_WELL_RATE_SCALES = (1.0, 0.92, 0.88)
SYNTH_QI_BOPD = 180.0
SYNTH_DECLINE_B = 0.45
SYNTH_DECLINE_DI = 0.08
SYNTH_OIL_NOISE_STD = 4.0
MIN_OIL_RATE_BOPD = 1.0
DRAINAGE_RADII_FT = {"N-01": 550.0, "N-02": 520.0, "N-03": 500.0}
INTERFERENCE_FIT_MODEL = "arps_hyperbolic"
INTERFERENCE_PHASE = "oil"
TARGET_INTERFERENCE_PCT = 5.0

# --- Coning demo ---
CONING_PRODUCTION_RATE_STB_D = 480.0
CONING_OIL_DENSITY_LB_FT3 = 50.0
CONING_WATER_DENSITY_LB_FT3 = 62.4
CONING_PERMEABILITY_MD = 120.0
CONING_RESERVOIR_THICKNESS_FT = 48.0
CONING_COMPLETION_INTERVAL_FT = 18.0
CONING_METHOD = "meyer_gardner"
YIELD_FORECAST_DAY_STEP = 30.0
YIELD_FORECAST_DAY_MAX = 540.0

# --- Enhanced RTA demo ---
RTA_SEED = 7
RTA_HISTORY_PERIODS = 40
RTA_HISTORY_START = "2018-06-01"
RTA_QI = 220.0
RTA_DI = 0.12
RTA_B = 0.55
RTA_RATE_NOISE_STD = 3.0
RTA_PRESSURE_INITIAL_PSI = 4850.0
RTA_PRESSURE_DECLINE_PSI_PER_MONTH = 18.0
RTA_PRESSURE_NOISE_STD = 25.0
RTA_PRESSURE_FLOOR_PSI = 500.0
RTA_INITIAL_PRESSURE_FOR_STUDY_PSI = 5000.0
RTA_RATE_COL = "oil"
RTA_PRESSURE_COL = "pressure"
RTA_MIN_RATE_BOPD = 0.5


def _ensure_out_dir() -> Path:
    out = Path("examples/out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def demo_interference() -> None:
    print("\n=== Well interference study ===")
    locations = pd.DataFrame(
        {
            "well_id": list(WELL_IDS),
            "latitude": list(WELL_LATITUDES),
            "longitude": list(WELL_LONGITUDES),
        }
    )
    idx = pd.date_range(
        INTERFERENCE_HISTORY_START, periods=INTERFERENCE_HISTORY_PERIODS, freq=MONTH_END_FREQ
    )
    rng = np.random.default_rng(DEMO_SEED)
    prod = {}
    for wid, scale in zip(WELL_IDS, PROD_WELL_RATE_SCALES, strict=True):
        t = np.arange(len(idx), dtype=float)
        base = SYNTH_QI_BOPD / (1.0 + SYNTH_DECLINE_B * SYNTH_DECLINE_DI * t) ** (
            1.0 / SYNTH_DECLINE_B
        )
        noise = rng.normal(0, SYNTH_OIL_NOISE_STD, size=len(idx))
        prod[wid] = pd.DataFrame({"oil": np.maximum(base * scale + noise, MIN_OIL_RATE_BOPD)}, index=idx)

    result = well_interference_study(
        locations,
        drainage_radii=DRAINAGE_RADII_FT,
        production_data=prod,
        model_name=INTERFERENCE_FIT_MODEL,
        phase=INTERFERENCE_PHASE,
        target_interference=TARGET_INTERFERENCE_PCT,
    )

    print("Pair distances (head):")
    print(result["distances"].head())
    print("\nGeometric interference (head):")
    print(result["geometric_interference"].head())
    if result["production_interference"] is not None:
        print("\nProduction-based interference (head):")
        print(result["production_interference"].head())
    if result["spacing_recommendation"] is not None:
        sp = result["spacing_recommendation"]
        print("\nSpacing recommendation (keys):", list(sp.keys())[:8])


def demo_coning() -> None:
    print("\n=== Coning study ===")
    base = coning_study(
        production_rate=CONING_PRODUCTION_RATE_STB_D,
        oil_density=CONING_OIL_DENSITY_LB_FT3,
        water_density=CONING_WATER_DENSITY_LB_FT3,
        permeability=CONING_PERMEABILITY_MD,
        reservoir_thickness=CONING_RESERVOIR_THICKNESS_FT,
        well_completion_interval=CONING_COMPLETION_INTERVAL_FT,
        method=CONING_METHOD,
    )
    ca = base["coning_analysis"]
    print(f"Critical rate: {ca['critical_rate']:.1f} STB/day")
    print(f"Coning index: {ca['coning_index']:.3f}  Risk: {ca['coning_risk']}")

    days = np.arange(0.0, YIELD_FORECAST_DAY_MAX, YIELD_FORECAST_DAY_STEP)
    oil_rate = np.full_like(days, CONING_PRODUCTION_RATE_STB_D)
    with_yield = coning_study(
        production_rate=CONING_PRODUCTION_RATE_STB_D,
        oil_density=CONING_OIL_DENSITY_LB_FT3,
        water_density=CONING_WATER_DENSITY_LB_FT3,
        permeability=CONING_PERMEABILITY_MD,
        reservoir_thickness=CONING_RESERVOIR_THICKNESS_FT,
        well_completion_interval=CONING_COMPLETION_INTERVAL_FT,
        method=CONING_METHOD,
        include_yield_forecast=True,
        forecast_time=days,
        forecast_oil_rate=oil_rate,
    )
    yf = with_yield["yield_forecast"]
    assert yf is not None
    print("\nYield forecast (first 3 rows):")
    print(yf.head(3))


def demo_enhanced_rta() -> None:
    print("\n=== Enhanced RTA study ===")
    rng = np.random.default_rng(RTA_SEED)
    idx = pd.date_range(RTA_HISTORY_START, periods=RTA_HISTORY_PERIODS, freq=MONTH_END_FREQ)
    t = np.arange(len(idx), dtype=float)
    q = RTA_QI / (1.0 + RTA_B * RTA_DI * t) ** (1.0 / RTA_B) + rng.normal(0, RTA_RATE_NOISE_STD, size=len(t))
    p0 = RTA_PRESSURE_INITIAL_PSI - RTA_PRESSURE_DECLINE_PSI_PER_MONTH * t + rng.normal(
        0, RTA_PRESSURE_NOISE_STD, size=len(t)
    )
    df = pd.DataFrame(
        {
            RTA_RATE_COL: np.maximum(q, RTA_MIN_RATE_BOPD),
            RTA_PRESSURE_COL: np.maximum(p0, RTA_PRESSURE_FLOOR_PSI),
        },
        index=idx,
    )

    rta = enhanced_rta_study(
        df,
        rate_col=RTA_RATE_COL,
        pressure_col=RTA_PRESSURE_COL,
        initial_pressure=RTA_INITIAL_PRESSURE_FOR_STUDY_PSI,
        run_fmb=False,
        run_fracture_network=False,
    )
    meta = rta["metadata"]
    print("Steps run (flags):", meta["flags"])
    if rta.get("type_curve_match") and rta["type_curve_match"].get("best_match"):
        bm = rta["type_curve_match"]["best_match"]
        print("Type-curve best match keys:", list(bm.keys()))
    if rta.get("blasingame"):
        print("Blasingame flow_regime:", rta["blasingame"].get("flow_regime"))
    if rta.get("dn_type_curve"):
        print("DN flow_regime:", rta["dn_type_curve"].get("flow_regime"))
    if rta.get("diagnostic_plot_data") and "flow_regime" in rta["diagnostic_plot_data"]:
        print("Diagnostic flow_regime:", rta["diagnostic_plot_data"]["flow_regime"])

    out = _ensure_out_dir()
    path = out / "integrated_rta_oil_pressure.csv"
    df.to_csv(path)
    print(f"\nWrote sample input to {path}")


def main() -> None:
    np.random.seed(DEMO_SEED)
    demo_interference()
    demo_coning()
    demo_enhanced_rta()
    print("\nDone.")


if __name__ == "__main__":
    main()
