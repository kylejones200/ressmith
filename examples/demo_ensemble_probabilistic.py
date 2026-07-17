"""
From Decline Curve to Capital Decision: Probabilistic Reservoir Forecasting with ResSmith

Demonstrates uncertainty-aware decline forecasting and economics:

1. Evidence-weighted ensemble (inverse validation RMSE, not opinion weights)
2. Parameter uncertainty via Monte Carlo (P10/P50/P90 with correct exceedance language)
3. Monthly rate integration for cumulative production and cashflow
4. NPV distributions across scenarios (not a single P50 path)
5. Holdout benchmark on public monthly production (OIL-016 sample well)

Run::

    uv run python examples/demo_ensemble_probabilistic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ressmith import (
    ensemble_forecast,
    evaluate_economics,
    evaluate_scenarios,
    fit_forecast,
    probabilistic_forecast,
    scenario_summary,
)
from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.primitives.economics import npv
from ressmith.workflows.calendar import get_month_day_counts

# --- Reproducibility & horizons ---
DEMO_SEED = 42
FORECAST_HORIZON_MONTHS = 36
MONTHLY_FREQ = "ME"
N_MC_SAMPLES = 2000

# --- Synthetic demo ---
SYNTHETIC_HISTORY_MONTHS = 36
SYNTHETIC_NOISE_STD = 5.0
VALIDATION_MONTHS = 6

# --- Benchmark (public OIL-016 sample: xpertsystems/oil016-sample on Hugging Face) ---
BENCHMARK_CSV = Path(__file__).resolve().parent / "data" / "well_0000318_monthly.csv"
BENCHMARK_HOLDOUT_MONTHS = 24
BENCHMARK_WELL_ID = "WELL_0000318 (OIL-016 sample)"

ENSEMBLE_MODELS = ["arps_hyperbolic", "arps_exponential", "power_law", "duong"]


def monthly_volumes_from_rates(rates: pd.Series) -> pd.Series:
    """Integrate STB/day rates to STB/month using actual days per month."""
    days = get_month_day_counts(rates.index)
    return rates * days


def cumulative_production_stb(rates: pd.Series) -> float:
    """Cumulative STB over the forecast window."""
    return float(monthly_volumes_from_rates(rates).sum())


def forecast_volumes_for_economics(rates: pd.Series) -> ForecastResult:
    """Economics expects monthly production volumes, not daily rates."""
    return ForecastResult(yhat=monthly_volumes_from_rates(rates))


def volumes_to_daily_rates(
    monthly_volumes: pd.Series,
) -> pd.Series:
    """Convert reported monthly STB to STB/day for decline fitting."""
    days = get_month_day_counts(monthly_volumes.index)
    return monthly_volumes / days


def validation_rmse_weights(
    data: pd.DataFrame,
    model_names: list[str],
    val_months: int,
    frequency: str = MONTHLY_FREQ,
    phase: str = "oil",
) -> list[float]:
    """Inverse-RMSE weights from a chronological validation window."""
    if len(data) <= val_months + 12:
        return [1.0 / len(model_names)] * len(model_names)

    train = data.iloc[:-val_months]
    actual = data[phase].iloc[-val_months:].values
    n_train = len(train)
    horizon = n_train + val_months

    inv_errors: list[float] = []
    for model_name in model_names:
        try:
            forecast, _ = fit_forecast(
                train,
                model_name=model_name,
                horizon=horizon,
                frequency=frequency,
                phase=phase,
            )
            pred = forecast.yhat.values[n_train : n_train + val_months]
            rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
            inv_errors.append(1.0 / max(rmse, 1e-6))
        except Exception:
            inv_errors.append(0.0)

    total = sum(inv_errors)
    if total <= 0:
        return [1.0 / len(model_names)] * len(model_names)
    return [w / total for w in inv_errors]


def npv_from_rate_path(rates: np.ndarray, index: pd.DatetimeIndex, spec: EconSpec) -> float:
    """NPV from a single monthly rate path (STB/day)."""
    days = get_month_day_counts(index).values
    volumes = rates * days
    price = spec.price_assumptions.get(
        "oil", next(iter(spec.price_assumptions.values()))
    )
    revenue = volumes * price
    net = revenue - spec.opex
    net = net.copy()
    net[0] -= spec.capex
    return float(npv(net, spec.discount_rate))


def npv_distribution_from_samples(
    samples: np.ndarray,
    index: pd.DatetimeIndex,
    spec: EconSpec,
) -> np.ndarray:
    """Evaluate NPV for every Monte Carlo rate sample."""
    return np.array(
        [npv_from_rate_path(sample, index, spec) for sample in samples],
        dtype=float,
    )


def pinball_loss(y: np.ndarray, yhat: np.ndarray, quantile: float) -> float:
    """Pinball loss at a given quantile."""
    err = y - yhat
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def interval_coverage(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of observations inside [lower, upper]."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def petroleum_percentiles(prob_result: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Map Monte Carlo output to petroleum P10 (high) / P50 / P90 (low).

    ``probabilistic_forecast`` stores the 10th/90th *statistical* percentiles on
    keys ``p10``/``p90``. Petroleum P10 is the high case (~10% exceedance); P90 is
    the low case (~90% exceedance).
    """
    return prob_result["p90"], prob_result["p50"], prob_result["p10"]


def generate_synthetic_data(
    n_periods: int = SYNTHETIC_HISTORY_MONTHS,
    noise_std: float = SYNTHETIC_NOISE_STD,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic monthly oil rates (STB/day)."""
    rng = rng or np.random.default_rng(DEMO_SEED)
    time_index = pd.date_range("2020-01-01", periods=n_periods, freq=MONTHLY_FREQ)
    t = np.arange(n_periods) / 12.0

    qi, di, b = 600.0, 0.75, 0.6
    q_true = qi / (1.0 + b * di * t) ** (1.0 / b)
    noise = rng.normal(0, noise_std, len(q_true))
    q_noisy = np.maximum(q_true + noise, 10.0)

    return pd.DataFrame({"oil": q_noisy}, index=time_index)


def load_benchmark_well(csv_path: Path = BENCHMARK_CSV) -> pd.DataFrame:
    """Load public monthly production and convert to STB/day rates."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {csv_path}. "
            "Expected OIL-016 sample well export under examples/data/."
        )
    raw = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    rates = volumes_to_daily_rates(raw["oil_bbl"].astype(float))
    return pd.DataFrame({"oil": rates}, index=rates.index)


def run_synthetic_workflow(rng: np.random.Generator) -> None:
    """Illustrate ensemble + probabilistic economics on synthetic monthly rates."""
    print("=" * 80)
    print("PART 1 — SYNTHETIC MONTHLY RATE WORKFLOW")
    print("=" * 80)

    data = generate_synthetic_data(rng=rng)
    print(f"\n1. History: {len(data)} months | avg rate {data['oil'].mean():.1f} STB/day")

    weights = validation_rmse_weights(
        data, ENSEMBLE_MODELS, val_months=VALIDATION_MONTHS, frequency=MONTHLY_FREQ
    )
    print("\n2. Ensemble (inverse validation-RMSE weights, not opinion weights):")
    for name, w in zip(ENSEMBLE_MODELS, weights):
        print(f"     {name}: {w:.3f}")

    ensemble_result = ensemble_forecast(
        data,
        model_names=ENSEMBLE_MODELS,
        method="weighted",
        horizon=FORECAST_HORIZON_MONTHS,
        frequency=MONTHLY_FREQ,
        weights=weights,
    )
    ens_cum = cumulative_production_stb(ensemble_result.yhat)
    print(
        f"\n   {FORECAST_HORIZON_MONTHS}-month cumulative production (ensemble): "
        f"{ens_cum:,.0f} STB"
    )
    print(f"   Peak rate: {ensemble_result.yhat.max():.1f} STB/day")

    print("\n3. Probabilistic forecast (parameter uncertainty, Monte Carlo):")
    prob_result = probabilistic_forecast(
        data,
        model_name="arps_hyperbolic",
        horizon=FORECAST_HORIZON_MONTHS,
        frequency=MONTHLY_FREQ,
        n_samples=N_MC_SAMPLES,
        seed=DEMO_SEED,
    )
    p10, p50, p90 = petroleum_percentiles(prob_result)
    idx = p50.index
    samples = prob_result["samples"]

    p10_cum = cumulative_production_stb(p10)
    p50_cum = cumulative_production_stb(p50)
    p90_cum = cumulative_production_stb(p90)
    half_width = (p10_cum - p90_cum) / 2.0

    print(f"   Samples: {prob_result['metadata']['n_samples']}")
    print(f"\n   {FORECAST_HORIZON_MONTHS}-month cumulative production (not EUR):")
    print(
        f"     P10: {p10_cum:,.0f} STB  (~10% chance of exceeding; high case)"
    )
    print(f"     P50: {p50_cum:,.0f} STB  (median)")
    print(
        f"     P90: {p90_cum:,.0f} STB  (~90% chance of exceeding; low case)"
    )
    print(
        f"     P10–P90 half-width: {half_width:,.0f} STB "
        f"({half_width / p50_cum * 100:.1f}% of P50 cumulative)"
    )

    print("\n4. Uncertainty layers (what each output represents):")
    print("     • Parameter uncertainty — Monte Carlo P10/P50/P90 rate paths above")
    print("     • Model uncertainty — spread across Arps / power-law / Duong in ensemble")
    print(
        "     • Outcome uncertainty — price, opex, discount rate in scenario NPV "
        "distributions below"
    )
    print(
        "   (Naive Gaussian CIs around P50 are omitted; they would double-count "
        "parameter bands without adding calibrated forecast skill.)"
    )

    econ_spec = EconSpec(
        price_assumptions={"oil": 75.0},
        opex=15.0,
        capex=2_000_000.0,
        discount_rate=0.10,
    )

    print("\n5. Economics from integrated monthly volumes:")
    econ_p10 = evaluate_economics(forecast_volumes_for_economics(p10), econ_spec)
    econ_p50 = evaluate_economics(forecast_volumes_for_economics(p50), econ_spec)
    econ_p90 = evaluate_economics(forecast_volumes_for_economics(p90), econ_spec)

    print(f"     P10 NPV: ${econ_p10.npv:,.0f}  (optimistic production case)")
    print(f"     P50 NPV: ${econ_p50.npv:,.0f}  (median production case)")
    print(f"     P90 NPV: ${econ_p90.npv:,.0f}  (conservative production case)")

    if econ_p50.irr is not None:
        irr_line = (
            f"     P50 IRR: {econ_p50.irr * 100:.2f}%"
            if econ_p10.irr is not None and econ_p90.irr is not None
            else f"     P50 IRR: {econ_p50.irr * 100:.2f}%"
        )
        print(irr_line)

    npv_samples = npv_distribution_from_samples(samples, idx, econ_spec)
    prob_positive = (npv_samples > 0).mean() * 100.0
    p90_downside_npv = float(np.percentile(npv_samples, 10))

    print("\n6. NPV distribution under base economics (all Monte Carlo samples):")
    print(f"     P(positive NPV): {prob_positive:.1f}%")
    print(f"     Expected NPV: ${npv_samples.mean():,.0f}")
    print(f"     NPV std dev: ${npv_samples.std():,.0f}")
    print(f"     P90 downside NPV: ${p90_downside_npv:,.0f}")

    scenarios = {
        "base_case": {},
        "high_price": {"prices": {"oil": 90.0}},
        "low_price": {"prices": {"oil": 60.0}},
        "high_opex": {"opex": 20.0},
        "low_discount": {"discount_rate": 0.08},
    }

    print("\n7. Scenario NPV distributions (parameter uncertainty × price/opex/rate):")
    header = (
        f"{'scenario':<14} {'E[NPV]':>12} {'P(NPV>0)':>10} "
        f"{'P90 down NPV':>14} {'P10 NPV':>12}"
    )
    print(header)
    print("-" * len(header))
    for scenario_name, overrides in scenarios.items():
        spec = EconSpec(
            price_assumptions={
                **econ_spec.price_assumptions,
                **overrides.get("prices", {}),
            },
            opex=overrides.get("opex", econ_spec.opex),
            capex=overrides.get("capex", econ_spec.capex),
            discount_rate=overrides.get("discount_rate", econ_spec.discount_rate),
            taxes=econ_spec.taxes,
            units=econ_spec.units,
        )
        scen_npvs = npv_distribution_from_samples(samples, idx, spec)
        print(
            f"{scenario_name:<14} "
            f"${scen_npvs.mean():>10,.0f} "
            f"{(scen_npvs > 0).mean() * 100:>9.1f}% "
            f"${np.percentile(scen_npvs, 10):>12,.0f} "
            f"${np.percentile(scen_npvs, 90):>10,.0f}"
        )

    # P50 path scenarios for comparison (legacy single-path view)
    scenario_results = evaluate_scenarios(
        forecast_volumes_for_economics(p50), econ_spec, scenarios
    )
    scenario_df = scenario_summary(scenario_results)
    print("\n   P50-only scenario table (single path — use distributions above for decisions):")
    print(scenario_df[["scenario", "npv", "irr"]].to_string(index=False))

    print("\n   Summary:")
    print(f"     {FORECAST_HORIZON_MONTHS}-month P50 cumulative: {p50_cum:,.0f} STB")
    print(f"     P50 NPV: ${econ_p50.npv:,.0f}")
    print(f"     P(positive NPV): {prob_positive:.1f}%")


def run_holdout_benchmark() -> None:
    """Holdout benchmark: fit on early history, forecast hidden tail, score error."""
    print("\n" + "=" * 80)
    print("PART 2 — PUBLIC WELL HOLDOUT BENCHMARK")
    print("=" * 80)

    full = load_benchmark_well()
    holdout = BENCHMARK_HOLDOUT_MONTHS
    train = full.iloc[:-holdout]
    actual = full["oil"].iloc[-holdout:].values
    n_train = len(train)
    horizon = n_train + holdout
    forecast_index = full.index[-holdout:]

    print(f"\nWell: {BENCHMARK_WELL_ID}")
    print(f"Source: {BENCHMARK_CSV.name} (OIL-016 sample, Hugging Face)")
    print(f"Train: {n_train} months | Holdout: {holdout} months (hidden from fit)")

    # Deterministic Arps
    arps_fc, _ = fit_forecast(
        train,
        model_name="arps_hyperbolic",
        horizon=horizon,
        frequency=MONTHLY_FREQ,
    )
    arps_pred = arps_fc.yhat.iloc[-holdout:].values

    # Evidence-weighted ensemble
    bench_weights = validation_rmse_weights(
        train, ENSEMBLE_MODELS, val_months=VALIDATION_MONTHS, frequency=MONTHLY_FREQ
    )
    ens_fc = ensemble_forecast(
        train,
        model_names=ENSEMBLE_MODELS,
        method="weighted",
        weights=bench_weights,
        horizon=horizon,
        frequency=MONTHLY_FREQ,
    )
    ens_pred = ens_fc.yhat.iloc[-holdout:].values

    # Probabilistic bands on train only
    prob = probabilistic_forecast(
        train,
        model_name="arps_hyperbolic",
        horizon=horizon,
        frequency=MONTHLY_FREQ,
        n_samples=1000,
        seed=DEMO_SEED,
    )
    p10_hold, p50_hold, p90_hold = petroleum_percentiles(prob)
    p10_hold = p10_hold.iloc[-holdout:].values
    p50_hold = p50_hold.iloc[-holdout:].values
    p90_hold = p90_hold.iloc[-holdout:].values

    def metrics(name: str, pred: np.ndarray) -> dict[str, float]:
        err = pred - actual
        cum_err = cumulative_production_stb(
            pd.Series(pred, index=forecast_index)
        ) - cumulative_production_stb(pd.Series(actual, index=forecast_index))
        return {
            "model": name,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "mape_pct": float(np.mean(np.abs(err / np.maximum(actual, 1.0))) * 100),
            "cum_prod_error_stb": float(cum_err),
        }

    rows = [metrics("arps_hyperbolic", arps_pred), metrics("ensemble_weighted", ens_pred)]
    metric_df = pd.DataFrame(rows)
    print("\nHoldout rate forecast error:")
    print(metric_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    coverage = interval_coverage(actual, p90_hold, p10_hold)
    pb_p10 = pinball_loss(actual, p10_hold, 0.10)
    pb_p50 = pinball_loss(actual, p50_hold, 0.50)
    pb_p90 = pinball_loss(actual, p90_hold, 0.90)

    print("\nProbabilistic calibration (Arps MC on train):")
    print(f"  P10–P90 interval coverage on holdout: {coverage * 100:.1f}%")
    print(f"  Pinball loss (P10/P50/P90): {pb_p10:.2f} / {pb_p50:.2f} / {pb_p90:.2f}")

    econ_spec = EconSpec(
        price_assumptions={"oil": 70.0},
        opex=12.0,
        capex=1_500_000.0,
        discount_rate=0.10,
    )
    actual_npv = npv_from_rate_path(actual, forecast_index, econ_spec)
    arps_npv = npv_from_rate_path(arps_pred, forecast_index, econ_spec)
    ens_npv = npv_from_rate_path(ens_pred, forecast_index, econ_spec)

    print("\nHoldout NPV error (vs realized rates):")
    print(f"  Realized NPV: ${actual_npv:,.0f}")
    print(f"  Arps NPV error: ${arps_npv - actual_npv:,.0f}")
    print(f"  Ensemble NPV error: ${ens_npv - actual_npv:,.0f}")
    regret_arps = max(0.0, actual_npv - arps_npv)
    regret_ens = max(0.0, actual_npv - ens_npv)
    print(f"  Economic regret (Arps): ${regret_arps:,.0f}")
    print(f"  Economic regret (ensemble): ${regret_ens:,.0f}")

    if ens_pred is not None and metric_df.loc[1, "rmse"] < metric_df.loc[0, "rmse"]:
        print(
            "\n  → Ensemble improves holdout RMSE vs single Arps on this public well slice."
        )


def main() -> None:
    rng = np.random.default_rng(DEMO_SEED)
    run_synthetic_workflow(rng)
    run_holdout_benchmark()
    print("\n" + "=" * 80)
    print("ENSEMBLE & PROBABILISTIC ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
