"""
Economics primitives: cashflow, NPV, IRR calculations.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd

try:
    from scipy import optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ressmith.objects.domain import EconSpec, ForecastResult


def cashflow_from_forecast(forecast: ForecastResult, spec: EconSpec) -> pd.DataFrame:
    """
    Build monthly cashflows from forecast rates, prices, and costs.

    Parameters
    ----------
    forecast : ForecastResult
        Forecast results with yhat (rates)
    spec : EconSpec
        Economics specification

    Returns
    -------
    pd.DataFrame
        Cashflow table with columns: period, revenue, opex, capex, net_cashflow
    """
    rates = forecast.yhat.values
    n_periods = len(rates)

    cashflows = pd.DataFrame(
        {
            "period": range(n_periods),
            "revenue": 0.0,
            "opex": -spec.opex,
            "capex": 0.0,
            "net_cashflow": 0.0,
        }
    )

    cashflows.loc[0, "capex"] = -spec.capex

    revenue = np.zeros(n_periods)
    if "phases" in forecast.metadata:
        phases = forecast.metadata["phases"]
        for phase, phase_rates in phases.items():
            if phase in spec.price_assumptions:
                revenue += phase_rates * spec.price_assumptions[phase]
    else:
        if "oil" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["oil"]
        elif "gas" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["gas"]
        elif "water" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["water"]
        else:
            price = list(spec.price_assumptions.values())[0]
            revenue = rates * price

    cashflows["revenue"] = revenue

    if spec.taxes:
        tax_rate = list(spec.taxes.values())[0]
        taxable_income = cashflows["revenue"] + cashflows["opex"]
        taxes = -taxable_income.clip(lower=0) * tax_rate
        cashflows["taxes"] = taxes
        cashflows["net_cashflow"] = (
            cashflows["revenue"] + cashflows["opex"] + cashflows["capex"] + taxes
        )
    else:
        cashflows["net_cashflow"] = (
            cashflows["revenue"] + cashflows["opex"] + cashflows["capex"]
        )

    return cashflows


def npv(cashflows: np.ndarray, discount_rate: float) -> float:
    """
    Compute Net Present Value of a monthly net-cashflow series.

    Delegates to the canonical ``decline-curve`` kernel so the discount
    convention lives in exactly one place. ``discount_rate`` is an **annual**
    rate, applied at the effective monthly rate ``(1 + r)**(1/12) - 1``
    (period 0 undiscounted). This is the correct effective-annual convention;
    earlier in-house code treated ``discount_rate`` as a raw per-period rate,
    which silently mis-discounted annual inputs by ~12x.

    Parameters
    ----------
    cashflows : np.ndarray
        Monthly net-cashflow values (can be negative; capex at index 0).
    discount_rate : float
        Annual discount rate (0.10 = 10%/yr).

    Returns
    -------
    float
        NPV.
    """
    from decline_curve.economics import npv_from_cashflow

    return float(npv_from_cashflow(np.asarray(cashflows, dtype=float), discount_rate))


def irr(cashflows: np.ndarray, use_scipy: bool | None = None) -> float | None:
    """
    Compute Internal Rate of Return (annual).

    The root search uses :func:`npv`, which now applies an effective-annual
    discount convention, so the returned IRR is an **annual** rate.

    Parameters
    ----------
    cashflows : np.ndarray
        Monthly net-cashflow values
    use_scipy : bool, optional
        Force use of scipy (default: auto-detect)

    Returns
    -------
    float or None
        Annual IRR if found, None otherwise
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    if use_scipy and HAS_SCIPY:

        def npv_func(rate: float) -> float:
            return npv(cashflows, rate)

        try:
            result = optimize.root_scalar(
                npv_func, bracket=[-0.99, 10.0], method="brentq"
            )
            if result.converged:
                return result.root
        except ValueError as e:
            logger.info(
                "IRR root_scalar failed, falling back to grid search: %s", e
            )

    rates = np.linspace(-0.9, 2.0, 1000)
    npvs = [npv(cashflows, r) for r in rates]
    npvs = np.array(npvs)
    sign_changes = np.where(np.diff(np.sign(npvs)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        if idx < len(rates) - 1:
            # Linear interpolation
            r1, r2 = rates[idx], rates[idx + 1]
            n1, n2 = npvs[idx], npvs[idx + 1]
            if n2 != n1:
                return r1 - n1 * (r2 - r1) / (n2 - n1)
            return r1

    return None


def scenario_apply(
    baseline_forecast: ForecastResult,
    spec: EconSpec,
    scenarios: dict[str, dict[str, float]],
) -> dict[str, pd.DataFrame]:
    """
    Apply price or cost scenarios to baseline forecast.

    Parameters
    ----------
    baseline_forecast : ForecastResult
        Baseline forecast
    spec : EconSpec
        Base economics specification
    scenarios : dict
        Dictionary of scenario names to parameter overrides

    Returns
    -------
    dict
        Dictionary of scenario names to cashflow DataFrames
    """
    results = {}
    for scenario_name, overrides in scenarios.items():
        modified_spec = EconSpec(
            price_assumptions={
                **spec.price_assumptions,
                **overrides.get("prices", {}),
            },
            opex=overrides.get("opex", spec.opex),
            capex=overrides.get("capex", spec.capex),
            discount_rate=overrides.get("discount_rate", spec.discount_rate),
            taxes=overrides.get("taxes", spec.taxes),
            units=spec.units,
        )
        cashflows = cashflow_from_forecast(baseline_forecast, modified_spec)
        results[scenario_name] = cashflows
    return results
