"""
Segmented decline, model competition, and walk-forward validation.

1. Two-segment **hyperbolic** fit on synthetic monthly history (hyperbolic early, exponential late)
2. ``compare_models`` — single-regime hyperbolic vs power law (in-sample merit)
3. ``walk_forward_backtest`` — rolling forecast quality for the single hyperbolic
4. Cumulative forecast oil: **segmented** vs **single** hyperbolic (same horizon)

Artifacts: ``segmented_forecast.csv``, ``single_model_forecast.csv``,
``segmented_model_comparison.csv``, ``segmented_walk_forward.csv``.

Run::

    uv run python examples/basic_segmented.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ressmith.workflows import (
    compare_models,
    fit_forecast,
    fit_segmented_forecast,
    walk_forward_backtest,
)
from ressmith.workflows.io import write_csv_results

# --- Synthetic two-regime history ---
SYNTHETIC_SEED = 7
SYNTHETIC_N_PERIODS = 48
SYNTHETIC_FREQ = "ME"
SYNTHETIC_START = "2018-01-01"
SEGMENT_BREAK_INDEX = 17  # inclusive end of first segment; second starts same month
SYNTHETIC_OIL_NOISE_STD = 5.0
MIN_OIL_RATE_BOPD = 0.5

# First segment: hyperbolic-style
SEG1_QI = 380.0
SEG1_B = 0.55
SEG1_DI = 0.12

# Second segment: exponential continuation from end of segment 1 (smoother demo history)
SEG2_EXPONENTIAL_DECLINE = 0.18

# --- Model comparison & walk-forward ---
COMPARE_MODEL_NAMES = ("arps_hyperbolic", "power_law")
COMPARE_HORIZON = 36
PHASE_OIL = "oil"

WALK_FORWARD_HORIZONS = (6, 12)
WALK_FORWARD_MIN_TRAIN = 18
WALK_FORWARD_STEP = 6
SINGLE_MODEL_NAME = "arps_hyperbolic"

# --- Segmented fit ---
SEGMENT_KINDS = ("hyperbolic", "hyperbolic")
SEGMENT_FORECAST_HORIZON = 36
ENFORCE_SEGMENT_CONTINUITY = False

# --- Output ---
OUT_SUBDIR = Path("examples/out")
RELATIVE_EPSILON = 1e-9


def _out_dir() -> Path:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    return OUT_SUBDIR


def generate_synthetic_segmented(
    n_periods: int = SYNTHETIC_N_PERIODS,
    seed: int = SYNTHETIC_SEED,
) -> pd.DataFrame:
    """Two-regime monthly rate: hyperbolic early, exponential late + noise (stable demo path)."""
    rng = np.random.default_rng(seed)
    time_index = pd.date_range(SYNTHETIC_START, periods=n_periods, freq=SYNTHETIC_FREQ)
    t = np.arange(n_periods, dtype=float)

    n1 = SEGMENT_BREAK_INDEX + 1
    t1 = t[:n1]
    q1 = SEG1_QI / (1.0 + SEG1_B * SEG1_DI * t1) ** (1.0 / SEG1_B)
    q2_start = float(q1[-1])
    t2 = t[n1:]
    t_rel = t2 - float(t2[0])
    q2 = q2_start * np.exp(-SEG2_EXPONENTIAL_DECLINE * t_rel)
    q = np.concatenate([q1, q2])
    noise = rng.normal(0, SYNTHETIC_OIL_NOISE_STD, len(q))
    return pd.DataFrame({"oil": np.maximum(q + noise, MIN_OIL_RATE_BOPD)}, index=time_index)


def main() -> None:
    out = _out_dir()
    print("Generating synthetic two-regime monthly decline...")
    data = generate_synthetic_segmented()
    print(f"  {len(data)} months")

    print("\n--- Single-regime model comparison (full history) ---")
    comp = compare_models(
        data,
        model_names=list(COMPARE_MODEL_NAMES),
        horizon=COMPARE_HORIZON,
        phase=PHASE_OIL,
    )
    comp.to_csv(out / "segmented_model_comparison.csv", index=False)
    print(comp[["model_name", "rmse", "r_squared", "eur"]].to_string(index=False))

    print("\n--- Walk-forward (single hyperbolic, coarse steps) ---")
    wf = walk_forward_backtest(
        data,
        model_name=SINGLE_MODEL_NAME,
        forecast_horizons=list(WALK_FORWARD_HORIZONS),
        min_train_size=WALK_FORWARD_MIN_TRAIN,
        step_size=WALK_FORWARD_STEP,
        phase=PHASE_OIL,
    )
    wf.to_csv(out / "segmented_walk_forward.csv", index=False)
    print(wf.groupby("horizon")["rmse"].mean().round(2))

    i0, i_mid, i_end = data.index[0], data.index[SEGMENT_BREAK_INDEX], data.index[-1]
    segment_dates = [
        (i0.to_pydatetime(), i_mid.to_pydatetime()),
        (i_mid.to_pydatetime(), i_end.to_pydatetime()),
    ]

    print("\n--- Segmented fit (two hyperbolic segments) ---")
    seg_forecast, segments, errors = fit_segmented_forecast(
        data,
        segment_dates=segment_dates,
        kinds=list(SEGMENT_KINDS),
        horizon=SEGMENT_FORECAST_HORIZON,
        enforce_continuity=ENFORCE_SEGMENT_CONTINUITY,
    )
    for i, seg in enumerate(segments):
        print(
            f"  [{i + 1}] {seg.kind}: qi={seg.parameters['qi']:.2f}, "
            f"di={seg.parameters['di']:.5f}, b={seg.parameters.get('b', float('nan')):.3f}"
        )
    if errors:
        print(f"Continuity notes: {len(errors)}")

    print("\n--- Single hyperbolic benchmark ---")
    single_forecast, single_params = fit_forecast(
        data, model_name=SINGLE_MODEL_NAME, horizon=SEGMENT_FORECAST_HORIZON
    )
    print(
        f"  qi={single_params.get('qi', float('nan')):.2f}, "
        f"di={single_params.get('di', float('nan')):.5f}, "
        f"b={single_params.get('b', float('nan')):.3f}"
    )

    cum_seg = float(seg_forecast.yhat.sum())
    cum_single = float(single_forecast.yhat.sum())
    denom = max(abs(cum_single), RELATIVE_EPSILON)
    print("\n--- Cumulative forecast oil (forecast window) ---")
    print(f"  Segmented: {cum_seg:,.0f}")
    print(f"  Single ARPS hyperbolic: {cum_single:,.0f}")
    print(f"  Δ vs single: {(cum_seg - cum_single) / denom * 100:+.1f}%")

    write_csv_results(seg_forecast.yhat.to_frame(name=PHASE_OIL), out / "segmented_forecast.csv")
    write_csv_results(
        single_forecast.yhat.to_frame(name=PHASE_OIL), out / "single_model_forecast.csv"
    )
    print(f"\nWrote forecasts and diagnostics under {out}")


if __name__ == "__main__":
    main()
