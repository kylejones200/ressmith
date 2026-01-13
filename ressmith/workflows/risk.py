"""Risk report helper for forecast distributions.

This module provides a simple risk report helper that reads forecast distributions
and price assumptions and returns key risk metrics. For example probability that
NPV exceeds a threshold at well level and portfolio level.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """
    Risk metrics for a single well or portfolio.

    Attributes:
        prob_positive_npv: Probability that NPV > 0
        prob_npv_above_threshold: Probability that NPV exceeds threshold
        expected_npv: Expected (mean) NPV
        value_at_risk_90: VaR at 90% confidence (P90 NPV)
        conditional_value_at_risk: CVaR (expected loss given VaR breach)
        eur_range: Range between P10 and P90 EUR
        npv_range: Range between P10 and P90 NPV
    """

    prob_positive_npv: float
    prob_npv_above_threshold: float
    expected_npv: float
    value_at_risk_90: float
    conditional_value_at_risk: float
    eur_range: float
    npv_range: float


def calculate_risk_metrics(
    probabilistic_result: dict[str, Any],
    npv_threshold: float = 0.0,
) -> RiskMetrics:
    """
    Calculate risk metrics from probabilistic forecast.

    Args:
        probabilistic_result: Dictionary from probabilistic_forecast() with keys:
            'p10', 'p50', 'p90', 'samples', 'metadata'
        npv_threshold: NPV threshold for probability calculation

    Returns:
        RiskMetrics object

    Example:
        >>> from ressmith import probabilistic_forecast
        >>> from ressmith.workflows.risk import calculate_risk_metrics
        >>> result = probabilistic_forecast(data, horizon=36)
        >>> metrics = calculate_risk_metrics(result, npv_threshold=1000000)
        >>> print(f"Probability of positive NPV: {metrics.prob_positive_npv:.1%}")
    """
    if "samples" not in probabilistic_result:
        raise ValueError("Probabilistic result must contain 'samples' key")

    samples = probabilistic_result["samples"]
    if samples is None or len(samples) == 0:
        raise ValueError("Forecast samples not available for risk calculation")

    # Calculate EUR for each sample
    eur_samples = np.sum(samples, axis=1)

    # Calculate NPV samples if economics provided in metadata
    npv_samples = None
    if "metadata" in probabilistic_result:
        metadata = probabilistic_result["metadata"]
        if "price" in metadata and "opex" in metadata:
            # Calculate NPV from samples
            price = metadata["price"]
            opex = metadata["opex"]
            discount_rate = metadata.get("discount_rate", 0.10)

            # Simple NPV calculation (can be enhanced)
            npv_samples = _calculate_npv_from_samples(
                samples, price, opex, discount_rate
            )

    if npv_samples is None:
        logger.warning(
            "NPV calculation requires price and opex in metadata. "
            "EUR-based metrics only."
        )
        # Use EUR as proxy for NPV if no economics provided
        npv_samples = eur_samples * 50.0  # Rough proxy: $50/bbl

    # Calculate risk metrics
    prob_positive_npv = float(np.mean(npv_samples > 0))
    prob_npv_above_threshold = float(np.mean(npv_samples > npv_threshold))
    expected_npv = float(np.mean(npv_samples))
    value_at_risk_90 = float(np.percentile(npv_samples, 10))  # P90 = 10th percentile
    conditional_value_at_risk = float(
        np.mean(npv_samples[npv_samples <= value_at_risk_90])
    )

    # Calculate EUR range from p10/p90
    eur_p10 = float(np.percentile(eur_samples, 90))
    eur_p90 = float(np.percentile(eur_samples, 10))
    eur_range = eur_p10 - eur_p90

    # Calculate NPV range
    npv_p10 = float(np.percentile(npv_samples, 90))
    npv_p90 = float(np.percentile(npv_samples, 10))
    npv_range = npv_p10 - npv_p90

    return RiskMetrics(
        prob_positive_npv=prob_positive_npv,
        prob_npv_above_threshold=prob_npv_above_threshold,
        expected_npv=expected_npv,
        value_at_risk_90=value_at_risk_90,
        conditional_value_at_risk=conditional_value_at_risk,
        eur_range=eur_range,
        npv_range=npv_range,
    )


def _calculate_npv_from_samples(
    samples: np.ndarray,
    price: float,
    opex: float,
    discount_rate: float = 0.10,
) -> np.ndarray:
    """Calculate NPV samples from forecast samples.

    Args:
        samples: Forecast samples array [n_samples, n_periods]
        price: Unit price
        opex: Operating cost per unit
        discount_rate: Discount rate

    Returns:
        NPV samples array [n_samples]
    """
    n_samples, n_periods = samples.shape

    # Calculate cashflow for each sample
    cashflows = (samples * price) - (samples * opex)

    # Discount to present value
    discount_factors = (1 + discount_rate) ** np.arange(1, n_periods + 1)
    npv_samples = np.sum(cashflows / discount_factors, axis=1)

    return npv_samples


def portfolio_risk_report(
    probabilistic_results: dict[str, dict[str, Any]],
    npv_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Generate portfolio-level risk report.

    Aggregates risk metrics across multiple wells to provide portfolio-level
    risk assessment.

    Args:
        probabilistic_results: Dictionary mapping well_id to probabilistic_forecast result
        npv_threshold: NPV threshold for probability calculation

    Returns:
        DataFrame with risk metrics for each well and portfolio summary

    Example:
        >>> from ressmith import probabilistic_forecast
        >>> from ressmith.workflows.risk import portfolio_risk_report
        >>> results = {
        ...     'well_001': probabilistic_forecast(data1),
        ...     'well_002': probabilistic_forecast(data2),
        ... }
        >>> report = portfolio_risk_report(results, npv_threshold=1000000)
        >>> print(report)
    """
    well_metrics = []

    for well_id, result in probabilistic_results.items():
        try:
            metrics = calculate_risk_metrics(result, npv_threshold)

            # Get EUR p50 from result
            eur_p50 = None
            if "p50" in result:
                eur_p50 = float(result["p50"].sum())

            # Get NPV p50 if available
            npv_p50 = None
            if "metadata" in result and "price" in result["metadata"]:
                # Calculate from p50 forecast
                price = result["metadata"]["price"]
                opex = result["metadata"].get("opex", 0.0)
                discount_rate = result["metadata"].get("discount_rate", 0.10)
                if "p50" in result:
                    cashflow = (result["p50"] * price) - (result["p50"] * opex)
                    discount_factors = (1 + discount_rate) ** np.arange(
                        1, len(result["p50"]) + 1
                    )
                    npv_p50 = float(np.sum(cashflow / discount_factors))

            well_metrics.append(
                {
                    "well_id": well_id,
                    "prob_positive_npv": metrics.prob_positive_npv,
                    "prob_npv_above_threshold": metrics.prob_npv_above_threshold,
                    "expected_npv": metrics.expected_npv,
                    "var_90": metrics.value_at_risk_90,
                    "cvar": metrics.conditional_value_at_risk,
                    "eur_p50": eur_p50,
                    "npv_p50": npv_p50,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to calculate risk metrics for {well_id}: {e}")

    df = pd.DataFrame(well_metrics)

    # Add portfolio summary row
    if len(df) > 0:
        portfolio_summary = {
            "well_id": "PORTFOLIO",
            "prob_positive_npv": df["prob_positive_npv"].mean(),
            "prob_npv_above_threshold": df["prob_npv_above_threshold"].mean(),
            "expected_npv": df["expected_npv"].sum(),
            "var_90": df["var_90"].sum(),  # Portfolio VaR (simplified)
            "cvar": df["cvar"].sum(),  # Portfolio CVaR (simplified)
            "eur_p50": df["eur_p50"].sum() if "eur_p50" in df else None,
            "npv_p50": df["npv_p50"].sum() if "npv_p50" in df else None,
        }
        df = pd.concat([df, pd.DataFrame([portfolio_summary])], ignore_index=True)

    return df
