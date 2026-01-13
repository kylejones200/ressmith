"""
Portfolio-level analysis workflows.

Provides workflows for analyzing multiple wells as a portfolio,
including aggregation, ranking, and portfolio-level economics.
"""

import logging
from typing import Any

import pandas as pd

from ressmith.objects.domain import EconSpec, ForecastResult
from ressmith.workflows.core import fit_forecast

logger = logging.getLogger(__name__)


def analyze_portfolio(
    well_data: dict[str, pd.DataFrame],
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    econ_spec: EconSpec | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Analyze multiple wells as a portfolio.

    Parameters
    ----------
    well_data : dict
        Dictionary mapping well_id to production DataFrame
    model_name : str
        Decline model name (default: 'arps_hyperbolic')
    horizon : int
        Forecast horizon in periods (default: 24)
    econ_spec : EconSpec, optional
        Economics specification for portfolio evaluation
    **kwargs
        Additional parameters for model fitting

    Returns
    -------
    pd.DataFrame
        Portfolio analysis results with columns:
        - well_id: Well identifier
        - eur: Estimated Ultimate Recovery
        - npv: Net Present Value (if econ_spec provided)
        - irr: Internal Rate of Return (if econ_spec provided)
        - params: Model parameters dict
        - fit_quality: Fit diagnostics (if available)

    Examples
    --------
    >>> from ressmith import analyze_portfolio
    >>> from ressmith.objects import EconSpec
    >>>
    >>> well_data = {
    ...     'well_1': df1,
    ...     'well_2': df2,
    ...     'well_3': df3
    ... }
    >>> econ = EconSpec(price_assumptions={'oil': 70.0}, opex=15.0, discount_rate=0.1)
    >>> portfolio = analyze_portfolio(
    ...     well_data,
    ...     model_name='arps_hyperbolic',
    ...     econ_spec=econ
    ... )
    >>> print(portfolio.sort_values('npv', ascending=False))
    """
    logger.info(f"Analyzing portfolio of {len(well_data)} wells")

    from ressmith.workflows.analysis import estimate_eur

    results = []

    for well_id, data in well_data.items():
        try:
            # Fit and forecast
            forecast, params = fit_forecast(
                data, model_name=model_name, horizon=horizon, **kwargs
            )

            # Estimate EUR
            eur_result = estimate_eur(data, model_name=model_name, **kwargs)

            # Evaluate economics if spec provided
            npv = None
            irr = None
            if econ_spec is not None:
                from ressmith.workflows.core import evaluate_economics

                econ_result = evaluate_economics(forecast, econ_spec)
                npv = econ_result.npv
                irr = econ_result.irr

            fit_quality = {}
            if hasattr(forecast, "metadata") and "diagnostics" in forecast.metadata:
                diag = forecast.metadata["diagnostics"]
                fit_quality = {
                    "rmse": diag.rmse if hasattr(diag, "rmse") else None,
                    "r_squared": diag.r_squared if hasattr(diag, "r_squared") else None,
                    "mae": diag.mae if hasattr(diag, "mae") else None,
                }

            results.append(
                {
                    "well_id": well_id,
                    "eur": eur_result.get("eur", 0.0),
                    "npv": npv,
                    "irr": irr,
                    "params": params,
                    "fit_quality": fit_quality,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to analyze well {well_id}: {e}")
            results.append(
                {
                    "well_id": well_id,
                    "eur": None,
                    "npv": None,
                    "irr": None,
                    "params": {},
                    "fit_quality": {},
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)


def aggregate_portfolio_forecast(
    well_data: dict[str, pd.DataFrame],
    model_name: str = "arps_hyperbolic",
    horizon: int = 24,
    aggregation_method: str = "sum",
    **kwargs: Any,
) -> ForecastResult:
    """
    Aggregate forecasts from multiple wells into portfolio-level forecast.

    Parameters
    ----------
    well_data : dict
        Dictionary mapping well_id to production DataFrame
    model_name : str
        Decline model name
    horizon : int
        Forecast horizon
    aggregation_method : str
        Aggregation method: 'sum', 'mean', 'median' (default: 'sum')
    **kwargs
        Additional parameters for model

    Returns
    -------
    ForecastResult
        Aggregated portfolio forecast

    Examples
    --------
    >>> from ressmith import aggregate_portfolio_forecast
    >>>
    >>> portfolio_forecast = aggregate_portfolio_forecast(
    ...     well_data,
    ...     model_name='arps_hyperbolic',
    ...     horizon=36,
    ...     aggregation_method='sum'
    ... )
    >>> print(portfolio_forecast.yhat.sum())  # Total portfolio production
    """
    logger.info(f"Aggregating forecasts from {len(well_data)} wells")

    forecasts = []
    for well_id, data in well_data.items():
        try:
            forecast, _ = fit_forecast(
                data, model_name=model_name, horizon=horizon, **kwargs
            )
            forecasts.append(forecast.yhat)
        except Exception as e:
            logger.warning(f"Failed to forecast well {well_id}: {e}")

    if len(forecasts) == 0:
        raise ValueError("No valid forecasts generated")

    # Align all forecasts to same index
    base_index = forecasts[0].index
    forecast_matrix = pd.DataFrame(
        {i: f.reindex(base_index, method="nearest") for i, f in enumerate(forecasts)}
    )

    # Aggregate
    if aggregation_method == "sum":
        aggregated = forecast_matrix.sum(axis=1)
    elif aggregation_method == "mean":
        aggregated = forecast_matrix.mean(axis=1)
    elif aggregation_method == "median":
        aggregated = forecast_matrix.median(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return ForecastResult(
        yhat=pd.Series(aggregated, index=base_index, name="portfolio_forecast"),
        metadata={
            "n_wells": len(forecasts),
            "aggregation_method": aggregation_method,
        },
    )


def rank_wells(
    portfolio_results: pd.DataFrame,
    metric: str = "npv",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank wells by specified metric.

    Parameters
    ----------
    portfolio_results : pd.DataFrame
        Results from analyze_portfolio
    metric : str
        Metric to rank by: 'npv', 'eur', 'irr' (default: 'npv')
    ascending : bool
        Sort ascending (default: False, highest first)

    Returns
    -------
    pd.DataFrame
        Ranked portfolio results with 'rank' column added

    Examples
    --------
    >>> from ressmith import rank_wells
    >>>
    >>> ranked = rank_wells(portfolio_results, metric='npv')
    >>> print(ranked[['well_id', 'rank', 'npv']].head(10))
    """
    if metric not in portfolio_results.columns:
        raise ValueError(f"Metric '{metric}' not found in portfolio results")

    ranked = portfolio_results.copy()
    ranked = ranked.sort_values(metric, ascending=ascending, na_position="last")
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return ranked
