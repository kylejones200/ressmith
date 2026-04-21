"""
Economics with scenarios, discount stress, and capital overrun.

Uses ``examples/out/forecast.csv`` from ``basic_fit_forecast.py`` (or fits a
fallback series). Evaluates:

* **Base** ``evaluate_economics``
* **Price / opex** bands (downside / base / upside)
* **Discount rate** stress (low / high discount)
* **Capex overrun** scenario

Writes ``economics_scenarios.csv`` (wide table), ``cashflows.csv``, and
``economics_scenarios_detail.csv`` (per-scenario NPV components).

Run::

    uv run python examples/basic_fit_forecast.py
    uv run python examples/basic_economics.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.workflows import (
    evaluate_economics,
    evaluate_scenarios,
    fit_forecast,
    scenario_summary,
)
from ressmith.workflows.io import read_csv_production, write_csv_results

# --- Base economics (units consistent with cashflow_from_forecast: $/period) ---
BASE_OIL_PRICE_USD_PER_BBL = 72.0
BASE_OPEX_USD_PER_PERIOD = 350.0
BASE_CAPEX_USD = 250_000.0
BASE_DISCOUNT_RATE_ANNUAL = 0.10
BASE_TAX_STATE_FRACTION = 0.05

# --- Scenario deltas (absolute prices / opex / capex where applicable) ---
DOWNSIDE_OIL_PRICE_USD_PER_BBL = 52.0
DOWNSIDE_OPEX_USD_PER_PERIOD = 420.0
UPSIDE_OIL_PRICE_USD_PER_BBL = 88.0
UPSIDE_OPEX_USD_PER_PERIOD = 300.0

LOW_DISCOUNT_RATE_ANNUAL = 0.07
HIGH_DISCOUNT_RATE_ANNUAL = 0.14

CAPEX_OVERRUN_USD = 420_000.0

# --- Fallback synthetic (if forecast.csv missing) ---
FALLBACK_SEED = 1
FALLBACK_SYNTH_START = "2021-01-01"
FALLBACK_SYNTH_N_PERIODS = 30
FALLBACK_SYNTH_FREQ = "ME"
FALLBACK_TRUE_QI = 300.0
FALLBACK_TRUE_DI = 0.1
FALLBACK_TRUE_B = 0.5
FALLBACK_OIL_NOISE_STD = 4.0
FALLBACK_MIN_OIL_RATE = 0.5
FALLBACK_FIT_MODEL = "arps_hyperbolic"
FALLBACK_FORECAST_HORIZON = 24

# --- I/O ---
FORECAST_CSV_PATH = Path("examples/out/forecast.csv")
TIME_COLUMN = "date"
OUT_SUBDIR = Path("examples/out")


def _out_dir() -> Path:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    return OUT_SUBDIR


def load_or_build_forecast() -> ForecastResult:
    if FORECAST_CSV_PATH.exists():
        df = read_csv_production(FORECAST_CSV_PATH, time_column=TIME_COLUMN)
        if "oil" not in df.columns:
            rate_col = df.columns[0]
            yhat = df[rate_col].astype(float)
        else:
            yhat = df["oil"].astype(float)
        yhat.index = df.index
        print(f"Loaded forecast from {FORECAST_CSV_PATH} ({len(yhat)} periods)")
        return ForecastResult(yhat=yhat)

    print(f"No {FORECAST_CSV_PATH} — fitting a small synthetic monthly series...")
    rng = np.random.default_rng(FALLBACK_SEED)
    idx = pd.date_range(
        FALLBACK_SYNTH_START, periods=FALLBACK_SYNTH_N_PERIODS, freq=FALLBACK_SYNTH_FREQ
    )
    t = np.arange(len(idx), dtype=float)
    oil_true = FALLBACK_TRUE_QI / (1.0 + FALLBACK_TRUE_B * FALLBACK_TRUE_DI * t) ** (
        1.0 / FALLBACK_TRUE_B
    )
    oil = oil_true + rng.normal(0, FALLBACK_OIL_NOISE_STD, size=len(t))
    data = pd.DataFrame({"oil": np.maximum(oil, FALLBACK_MIN_OIL_RATE)}, index=idx)
    forecast, _ = fit_forecast(
        data, model_name=FALLBACK_FIT_MODEL, horizon=FALLBACK_FORECAST_HORIZON
    )
    return forecast


def main() -> None:
    out = _out_dir()
    forecast = load_or_build_forecast()

    base_spec = EconSpec(
        price_assumptions={"oil": BASE_OIL_PRICE_USD_PER_BBL},
        opex=BASE_OPEX_USD_PER_PERIOD,
        capex=BASE_CAPEX_USD,
        discount_rate=BASE_DISCOUNT_RATE_ANNUAL,
        taxes={"state": BASE_TAX_STATE_FRACTION},
    )

    print("\n--- Base case economics ---")
    base_result = evaluate_economics(forecast, base_spec)
    print(f"NPV: ${base_result.npv:,.0f}")
    if base_result.irr is not None:
        print(f"IRR: {base_result.irr * 100:.2f}%")
    if base_result.payout_time is not None:
        print(f"Payout (period index): {base_result.payout_time:.1f}")

    scenarios: dict[str, dict[str, Any]] = {
        "downside_price_opex": {
            "prices": {"oil": DOWNSIDE_OIL_PRICE_USD_PER_BBL},
            "opex": DOWNSIDE_OPEX_USD_PER_PERIOD,
        },
        "base": {},
        "upside_price_opex": {
            "prices": {"oil": UPSIDE_OIL_PRICE_USD_PER_BBL},
            "opex": UPSIDE_OPEX_USD_PER_PERIOD,
        },
        "low_discount": {"discount_rate": LOW_DISCOUNT_RATE_ANNUAL},
        "high_discount": {"discount_rate": HIGH_DISCOUNT_RATE_ANNUAL},
        "capex_overrun": {"capex": CAPEX_OVERRUN_USD},
    }

    print("\n--- Scenario grid (evaluate_scenarios) ---")
    scenario_results = evaluate_scenarios(forecast, base_spec, scenarios)
    summary = scenario_summary(scenario_results)
    print(summary.to_string(index=False))

    summary.to_csv(out / "economics_scenarios.csv", index=False)

    detail_rows = []
    for name, res in scenario_results.items():
        detail_rows.append(
            {
                "scenario": name,
                "npv": res.npv,
                "irr": res.irr,
                "payout_time": res.payout_time,
                "revenue": res.cashflows["revenue"].sum(),
                "opex": res.cashflows["opex"].sum(),
                "capex": res.cashflows["capex"].sum(),
                "net_cashflow_sum": res.cashflows["net_cashflow"].sum(),
            }
        )
    pd.DataFrame(detail_rows).to_csv(out / "economics_scenarios_detail.csv", index=False)

    write_csv_results(base_result.cashflows, out / "cashflows.csv")
    print(f"\nSaved: economics_scenarios*.csv, cashflows.csv under {out}")


if __name__ == "__main__":
    main()
