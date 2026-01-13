"""
Scenario analysis workflows for ResSmith.

Provides workflows for running multiple price/cost scenarios and comparing results.
"""

import logging
from typing import Any

import pandas as pd

from ressmith.objects.domain import EconResult, EconSpec, ForecastResult
from ressmith.primitives.economics import (
    cashflow_from_forecast,
    irr,
    npv,
)

logger = logging.getLogger(__name__)


def evaluate_scenarios(
    forecast: ForecastResult,
    base_spec: EconSpec,
    scenarios: dict[str, dict[str, Any]],
) -> dict[str, EconResult]:
    """
    Evaluate economics for multiple scenarios.

    Parameters
    ----------
    forecast : ForecastResult
        Base production forecast
    base_spec : EconSpec
        Base economics specification
    scenarios : dict
        Dictionary mapping scenario names to parameter overrides.
        Overrides can include:
        - 'prices': dict of price overrides (e.g., {'oil': 80.0})
        - 'opex': operating expense override
        - 'capex': capital expense override
        - 'discount_rate': discount rate override
        - 'taxes': tax rate override

    Returns
    -------
    dict
        Dictionary mapping scenario names to EconResult objects

    Examples
    --------
    >>> from ressmith import fit_forecast, evaluate_scenarios
    >>> from ressmith.objects import EconSpec
    >>>
    >>> forecast, _ = fit_forecast(data, model_name='arps_hyperbolic', horizon=24)
    >>> base_spec = EconSpec(
    ...     price_assumptions={'oil': 70.0},
    ...     opex=15.0,
    ...     capex=100000.0,
    ...     discount_rate=0.1
    ... )
    >>> scenarios = {
    ...     'low': {'prices': {'oil': 50.0}, 'opex': 18.0},
    ...     'base': {},  # Use base_spec
    ...     'high': {'prices': {'oil': 90.0}, 'opex': 12.0}
    ... }
    >>> results = evaluate_scenarios(forecast, base_spec, scenarios)
    >>> print(results['base'].npv)
    """
    logger.info(f"Evaluating {len(scenarios)} scenarios")

    results = {}
    for scenario_name, overrides in scenarios.items():
        modified_spec = EconSpec(
            price_assumptions={
                **base_spec.price_assumptions,
                **overrides.get("prices", {}),
            },
            opex=overrides.get("opex", base_spec.opex),
            capex=overrides.get("capex", base_spec.capex),
            discount_rate=overrides.get("discount_rate", base_spec.discount_rate),
            taxes=(
                overrides.get("taxes", base_spec.taxes)
                if overrides.get("taxes") is not None
                else base_spec.taxes
            ),
            units=base_spec.units,
        )

        cashflows_df = cashflow_from_forecast(forecast, modified_spec)

        net_cf = cashflows_df["net_cashflow"].values
        npv_value = npv(net_cf, modified_spec.discount_rate)
        irr_value = irr(net_cf)

        cumulative_cf = pd.Series(net_cf).cumsum()
        payout_idx = cumulative_cf[cumulative_cf >= 0].index
        payout_time = float(payout_idx[0]) if len(payout_idx) > 0 else None

        result = EconResult(
            cashflows=cashflows_df,
            npv=npv_value,
            irr=irr_value,
            payout_time=payout_time,
            metadata={"scenario": scenario_name},
        )
        results[scenario_name] = result

    return results


def scenario_summary(
    scenario_results: dict[str, EconResult],
) -> pd.DataFrame:
    """
    Create summary table of scenario results.

    Parameters
    ----------
    scenario_results : dict
        Dictionary mapping scenario names to EconResult objects

    Returns
    -------
    pd.DataFrame
        Summary table with columns: scenario, npv, irr, payout_time, total_revenue, total_opex, total_capex

    Examples
    --------
    >>> results = evaluate_scenarios(forecast, base_spec, scenarios)
    >>> summary = scenario_summary(results)
    >>> print(summary)
    """
    rows = []
    for scenario_name, result in scenario_results.items():
        row = {
            "scenario": scenario_name,
            "npv": result.npv,
            "irr": result.irr if result.irr is not None else None,
            "payout_time": (
                result.payout_time if result.payout_time is not None else None
            ),
            "total_revenue": result.cashflows["revenue"].sum(),
            "total_opex": abs(result.cashflows["opex"].sum()),
            "total_capex": abs(result.cashflows["capex"].sum()),
        }
        rows.append(row)

    return pd.DataFrame(rows)
