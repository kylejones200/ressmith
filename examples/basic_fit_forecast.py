"""
Fit, compare, validate, and forecast example (advanced basics).

Pipeline on synthetic **monthly** oil (optional **pressure**):

1. ``compare_models`` — in-sample metrics + EUR proxy per model
2. Primary model — prefers positive **R²**, else best R², else RMSE
3. ``walk_forward_backtest`` — out-of-sample error at rolling cutoffs (coarse ``step_size`` for speed)
4. ``ensemble_forecast`` — median blend of hyperbolic + power law
5. ``probabilistic_forecast`` — light Monte Carlo on parameters (P10/P50/P90 cumulative rate path)
6. ``normalize_production_with_pressure`` — rate vs pressure sanity check
7. ``generate_diagnostic_plot_data`` — log–log diagnostic table for RTA-style review

Artifacts under ``examples/out/`` (``forecast.csv`` stays compatible with ``basic_economics.py``).

Run::

    uv run python examples/basic_fit_forecast.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ressmith.workflows import (
    compare_models,
    ensemble_forecast,
    estimate_eur,
    fit_forecast,
    probabilistic_forecast,
    walk_forward_backtest,
)
from ressmith.workflows.diagnostics_plots import generate_diagnostic_plot_data
from ressmith.workflows.io import write_csv_results
from ressmith.workflows.pressure_normalization import normalize_production_with_pressure

# --- Synthetic decline (hyperbolic truth + noise) ---
SYNTHETIC_SEED = 42
SYNTHETIC_N_PERIODS = 48
SYNTHETIC_FREQ = "ME"
SYNTHETIC_START = "2019-01-01"
SYNTHETIC_OIL_NOISE_STD = 3.0
TRUE_QI_BOPD = 420.0
TRUE_DI = 0.08
TRUE_B = 0.52
MIN_OIL_RATE_BOPD = 1.0

# --- Synthetic pressure (linear drawdown + noise, psi) ---
PRESSURE_INITIAL_LINEAR_TERM_PSI = 5050.0
PRESSURE_DECLINE_PSI_PER_MONTH = 14.0
PRESSURE_NOISE_STD_PSI = 35.0
PRESSURE_CLIP_LOW_PSI = 1800.0
PRESSURE_CLIP_HIGH_PSI = 5200.0

# --- Model comparison & selection ---
MODEL_NAMES_COMPARE = ("arps_hyperbolic", "arps_exponential", "power_law")
PHASE_OIL = "oil"
R2_MIN_POSITIVE = 0.0

# --- Horizons & backtest (months / steps) ---
FORECAST_HORIZON_PERIODS = 24
WALK_FORWARD_HORIZONS = (6, 12)
WALK_FORWARD_MIN_TRAIN = 20
WALK_FORWARD_STEP = 6

ENSEMBLE_MODEL_NAMES = ("arps_hyperbolic", "power_law")
ENSEMBLE_METHOD = "median"

PROBABILISTIC_N_SAMPLES = 150
PROBABILISTIC_SEED = 42

# Parameter std devs for Monte Carlo (same units as fitted params)
UNC_QI_STDDEV = 25.0
UNC_DI_STDDEV = 0.02
UNC_B_STDDEV = 0.05
UNC_POWER_LAW_QI_STDDEV = 30.0
UNC_POWER_LAW_DI_STDDEV = 0.015
UNC_POWER_LAW_N_STDDEV = 0.03

# --- Pressure normalization ---
INITIAL_PRESSURE_MULTIPLIER_VS_FIRST_MEASURED = 1.02
PRESSURE_NORMALIZATION_METHOD = "pressure_ratio"

# --- EUR ---
EUR_FORECAST_HORIZON_MONTHS = 360.0

# --- Output ---
OUT_SUBDIR = Path("examples/out")


def _out_dir() -> Path:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    return OUT_SUBDIR


def generate_synthetic_hyperbolic(
    n_periods: int = SYNTHETIC_N_PERIODS,
    noise_std: float = SYNTHETIC_OIL_NOISE_STD,
    seed: int = SYNTHETIC_SEED,
) -> pd.DataFrame:
    """Monthly hyperbolic decline with noise and correlated **pressure** (psi)."""
    rng = np.random.default_rng(seed)
    time_index = pd.date_range(SYNTHETIC_START, periods=n_periods, freq=SYNTHETIC_FREQ)
    t = np.arange(n_periods, dtype=float)

    q_true = TRUE_QI_BOPD / (1.0 + TRUE_B * TRUE_DI * t) ** (1.0 / TRUE_B)
    noise = rng.normal(0, noise_std, size=len(q_true))
    oil = np.maximum(q_true + noise, MIN_OIL_RATE_BOPD)

    pressure = (
        PRESSURE_INITIAL_LINEAR_TERM_PSI
        - PRESSURE_DECLINE_PSI_PER_MONTH * t
        + rng.normal(0, PRESSURE_NOISE_STD_PSI, size=len(t))
    )
    pressure = np.clip(pressure, PRESSURE_CLIP_LOW_PSI, PRESSURE_CLIP_HIGH_PSI)

    return pd.DataFrame({"oil": oil, "pressure": pressure}, index=time_index)


def forecast_to_csv_frame(yhat: pd.Series) -> pd.DataFrame:
    frame = yhat.rename("oil").reset_index()
    frame.columns = ["date", "oil"]
    return frame


def pick_primary_model(comparison: pd.DataFrame) -> str:
    positive = comparison[comparison["r_squared"] > R2_MIN_POSITIVE]
    if not positive.empty:
        return str(positive.loc[positive["r_squared"].idxmax(), "model_name"])
    finite_r2 = comparison["r_squared"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_r2.empty:
        return str(comparison.loc[comparison["r_squared"].idxmax(), "model_name"])
    return str(comparison.loc[comparison["rmse"].idxmin(), "model_name"])


def param_uncertainty_for(model_name: str) -> dict[str, tuple[float, str]]:
    normal = "normal"
    if model_name == "arps_hyperbolic":
        return {
            "qi": (UNC_QI_STDDEV, normal),
            "di": (UNC_DI_STDDEV, normal),
            "b": (UNC_B_STDDEV, normal),
        }
    if model_name == "arps_exponential":
        return {"qi": (UNC_QI_STDDEV, normal), "di": (UNC_DI_STDDEV, normal)}
    if model_name == "power_law":
        return {
            "qi": (UNC_POWER_LAW_QI_STDDEV, normal),
            "di": (UNC_POWER_LAW_DI_STDDEV, normal),
            "n": (UNC_POWER_LAW_N_STDDEV, normal),
        }
    return {"qi": (UNC_QI_STDDEV, normal), "di": (UNC_DI_STDDEV, normal)}


def main() -> None:
    out = _out_dir()
    print("Generating synthetic monthly oil + pressure...")
    data = generate_synthetic_hyperbolic()
    print(f"  {len(data)} months | pressure {data['pressure'].iloc[0]:.0f} → {data['pressure'].iloc[-1]:.0f} psi")

    print("\n--- Model comparison ---")
    comparison = compare_models(
        data,
        model_names=list(MODEL_NAMES_COMPARE),
        horizon=FORECAST_HORIZON_PERIODS,
        phase=PHASE_OIL,
    )
    comparison.to_csv(out / "model_comparison.csv", index=False)
    print(comparison.sort_values("rmse")[["model_name", "rmse", "mae", "r_squared", "eur"]])

    best = pick_primary_model(comparison)
    print(f"\nPrimary model: {best}")

    print("\n--- Walk-forward backtest (primary model, coarse steps) ---")
    wf = walk_forward_backtest(
        data,
        model_name=best,
        forecast_horizons=list(WALK_FORWARD_HORIZONS),
        min_train_size=WALK_FORWARD_MIN_TRAIN,
        step_size=WALK_FORWARD_STEP,
        phase=PHASE_OIL,
    )
    wf.to_csv(out / "walk_forward_backtest.csv", index=False)
    print(wf.groupby("horizon")[["rmse", "r_squared"]].mean().round(4))

    print("\n--- Ensemble (median: hyperbolic + power_law) ---")
    ens = ensemble_forecast(
        data,
        model_names=list(ENSEMBLE_MODEL_NAMES),
        method=ENSEMBLE_METHOD,
        horizon=FORECAST_HORIZON_PERIODS,
    )
    write_csv_results(forecast_to_csv_frame(ens.yhat), out / "ensemble_median_forecast.csv")

    print(f"\n--- Probabilistic forecast (parameter noise, n={PROBABILISTIC_N_SAMPLES}) ---")
    prob = probabilistic_forecast(
        data,
        model_name=best,
        horizon=FORECAST_HORIZON_PERIODS,
        n_samples=PROBABILISTIC_N_SAMPLES,
        param_uncertainty=param_uncertainty_for(best),
        seed=PROBABILISTIC_SEED,
    )
    pband = pd.DataFrame(
        {
            "p10_cum": [float(prob["p10"].sum())],
            "p50_cum": [float(prob["p50"].sum())],
            "p90_cum": [float(prob["p90"].sum())],
        }
    )
    pband.to_csv(out / "probabilistic_cumulative_forecast_oil.csv", index=False)
    print(pband.to_string(index=False))

    print("\n--- Pressure-normalized rate (ratio method) ---")
    first_p = float(data["pressure"].iloc[0])
    norm = normalize_production_with_pressure(
        data,
        rate_col=PHASE_OIL,
        pressure_col="pressure",
        initial_pressure=first_p * INITIAL_PRESSURE_MULTIPLIER_VS_FIRST_MEASURED,
        method=PRESSURE_NORMALIZATION_METHOD,
    )
    norm[["oil", "pressure", "normalized_rate"]].to_csv(out / "normalized_rate_vs_pressure.csv")

    print("\n--- Diagnostic plot data (log–log) ---")
    t_days = np.array([(data.index[i] - data.index[0]).days for i in range(len(data))], dtype=float)
    diag = generate_diagnostic_plot_data(t_days, data[PHASE_OIL].values, plot_type="log_log")
    if "log_log" in diag and diag["log_log"] is not None:
        diag["log_log"].to_csv(out / "diagnostic_log_log.csv", index=False)
        print(f"  flow_regime (heuristic): {diag.get('flow_regime', 'n/a')}")

    print(f"\n--- Primary fit + EUR + point forecast ({best}) ---")
    forecast, params = fit_forecast(data, model_name=str(best), horizon=FORECAST_HORIZON_PERIODS)
    for key, value in params.items():
        print(f"  {key}: {value:.6g}")
    eur_block = estimate_eur(
        data, model_name=str(best), phase=PHASE_OIL, t_max=EUR_FORECAST_HORIZON_MONTHS
    )
    print(f"  EUR (30 yr, default limit): {eur_block.get('eur', float('nan')):.0f}")

    fc_path = out / "forecast.csv"
    write_csv_results(forecast_to_csv_frame(forecast.yhat), fc_path)
    print(f"\nPoint forecast for economics example: {fc_path}")


if __name__ == "__main__":
    main()
