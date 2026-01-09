"""
Economics primitives: cashflow, NPV, IRR calculations.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import optimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ressmith.objects.domain import EconSpec, ForecastResult


def cashflow_from_forecast(
    forecast: ForecastResult, spec: EconSpec
) -> pd.DataFrame:
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

    # Initialize cashflows
    cashflows = pd.DataFrame(
        {
            "period": range(n_periods),
            "revenue": 0.0,
            "opex": -spec.opex,
            "capex": 0.0,
            "net_cashflow": 0.0,
        }
    )

    # Apply capex in first period
    cashflows.loc[0, "capex"] = -spec.capex

    # Compute revenue - multi-phase support
    # If forecast has multi-phase data in metadata, use it
    revenue = np.zeros(n_periods)
    if "phases" in forecast.metadata:
        # Multi-phase forecast
        phases = forecast.metadata["phases"]
        for phase, phase_rates in phases.items():
            if phase in spec.price_assumptions:
                revenue += phase_rates * spec.price_assumptions[phase]
    else:
        # Single-phase forecast - use price assumptions
        if "oil" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["oil"]
        elif "gas" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["gas"]
        elif "water" in spec.price_assumptions:
            revenue = rates * spec.price_assumptions["water"]
        else:
            # Use first price assumption
            price = list(spec.price_assumptions.values())[0]
            revenue = rates * price
    
    cashflows["revenue"] = revenue

    # Apply taxes if specified
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
    Compute Net Present Value.

    Parameters
    ----------
    cashflows : np.ndarray
        Cashflow values (can be negative)
    discount_rate : float
        Discount rate per period

    Returns
    -------
    float
        NPV
    """
    periods = np.arange(len(cashflows))
    discount_factors = (1.0 + discount_rate) ** periods
    return np.sum(cashflows / discount_factors)


def irr(
    cashflows: np.ndarray, use_scipy: Optional[bool] = None
) -> Optional[float]:
    """
    Compute Internal Rate of Return.

    Parameters
    ----------
    cashflows : np.ndarray
        Cashflow values
    use_scipy : bool, optional
        Force use of scipy (default: auto-detect)

    Returns
    -------
    float or None
        IRR if found, None otherwise
    """
    if use_scipy is None:
        use_scipy = HAS_SCIPY

    if use_scipy and HAS_SCIPY:
        # Use scipy root finding
        def npv_func(rate: float) -> float:
            return npv(cashflows, rate)

        try:
            result = optimize.root_scalar(
                npv_func, bracket=[-0.99, 10.0], method="brentq"
            )
            if result.converged:
                return result.root
        except ValueError:
            pass

    # Approximate IRR using grid search
    rates = np.linspace(-0.9, 2.0, 1000)
    npvs = [npv(cashflows, r) for r in rates]
    npvs = np.array(npvs)
    sign_changes = np.where(np.diff(np.sign(npvs)))[0]
    if len(sign_changes) > 0:
        # Find zero crossing
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
        # Create modified spec
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

