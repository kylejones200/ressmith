"""
Integration with the decline-curve library for DCA (decline curve analysis).

Uses `decline-curve <https://pypi.org/project/decline-curve/>`_ for Arps
(exponential, hyperbolic, harmonic) and optional ML-based forecasting.
Install with: ``pip install petrosmith[dca]``
"""

from typing import Any, Dict, List, Literal, Optional

try:
    import pandas as pd
    from decline_curve import dca

    _DECLINE_CURVE_AVAILABLE = True
except ImportError:
    _DECLINE_CURVE_AVAILABLE = False
    pd = None
    dca = None


def decline_curve_available() -> bool:
    """Return True if the decline-curve library is installed."""
    return _DECLINE_CURVE_AVAILABLE


def forecast_with_dca(
    production_data: List[Dict[str, Any]],
    forecast_years: int = 5,
    model: Literal["arps", "arima", "timesfm", "chronos"] = "arps",
    kind: Literal["exponential", "harmonic", "hyperbolic"] = "hyperbolic",
    rate_key: str = "oil_rate",
    date_key: str = "date",
) -> Dict[str, Any]:
    """
    Run decline curve analysis using the decline-curve library.

    Converts PetroSmith production data (list of daily records) to a monthly
    series, fits the chosen model (e.g. Arps hyperbolic), and returns a
    forecast in the same shape as ProductionService.forecast_production.

    Args:
        production_data: List of dicts with keys date_key (YYYY-MM-DD) and
            rate_key (e.g. oil_rate in STB/day).
        forecast_years: Number of years to forecast.
        model: Forecasting model; 'arps' uses Arps decline (exponential,
            harmonic, or hyperbolic per kind).
        kind: Arps decline type (ignored if model != 'arps').
        rate_key: Key in each record for production rate (default 'oil_rate').
        date_key: Key in each record for date (default 'date').

    Returns:
        Dict with:
            - forecast: list of {year, rate, cumulative}
            - current_rate: last historical rate (STB/day)
            - decline_rate_annual: fitted decline rate (if available)
            - dca_params: fitted Arps params qi, di, b (if model='arps')
            - economic_limit: suggested economic limit (fraction of current rate)
            - model: model used
            - kind: Arps kind used

    Raises:
        ImportError: If decline-curve or pandas is not installed
            (install with: pip install petrosmith[dca])
        ValueError: If production_data is too short for fitting
    """
    if not _DECLINE_CURVE_AVAILABLE:
        raise ImportError(
            "Decline curve integration requires the decline-curve library. "
            "Install with: pip install petrosmith[dca]"
        )
    if not production_data or len(production_data) < 30:
        raise ValueError("Need at least 30 days of production data for DCA")

    # Build DataFrame and convert to monthly average rate
    df = pd.DataFrame(production_data)
    df[date_key] = pd.to_datetime(df[date_key])
    df = df.set_index(date_key).sort_index()
    # Resample to month-end and take mean rate (STB/day)
    monthly = df[rate_key].resample("ME").mean().dropna()
    if len(monthly) < 3:
        raise ValueError("Need at least 3 months of data after resampling for DCA")

    # Ensure regular monthly index for decline_curve (use month start for freq)
    monthly.index = monthly.index.to_period("M").to_timestamp()
    series = monthly.astype(float)

    # Forecast with decline-curve (horizon in months)
    horizon_months = forecast_years * 12
    try:
        result = dca.single_well(
            series,
            model=model,
            kind=kind,
            horizon=horizon_months,
            return_params=True,
        )
    except Exception as e:
        raise ValueError(f"Decline curve fitting failed: {e}") from e

    if isinstance(result, tuple):
        forecast_series, params_dict = result
    else:
        forecast_series = result
        params_dict = {}

    # Current rate = last historical rate (STB/day)
    current_rate = float(series.iloc[-1])

    # Keep only the future forecast (single_well returns history + forecast)
    if len(forecast_series) > horizon_months:
        forecast_series = forecast_series.iloc[-horizon_months:]

    # Convert monthly forecast to yearly for API compatibility
    forecast_df = forecast_series.to_frame("rate")
    forecast_df["year"] = forecast_df.index.year
    # Annual average rate (STB/day) and cumulative (approximate: sum(rate)*30.44 bbl/month)
    yearly = (
        forecast_df.groupby("year")
        .agg(
            rate=("rate", "mean"),
            # Cumulative production from forecast: sum of monthly rate * 30.44 days
            monthly_sum=("rate", "sum"),
        )
        .assign(
            cumulative=lambda x: (x["monthly_sum"] * 30.44).cumsum(),
        )
    )
    # Align year numbers: year 0 = first forecast year, etc.
    years = sorted(yearly.index.unique())
    year0 = years[0] if years else 0
    forecast_list = []
    cumulative_so_far = 0.0
    for i, year in enumerate(years):
        row = yearly.loc[year]
        rate = float(row["rate"])
        # Cumulative for this year from monthly sum * 30.44
        annual_production = float(row["monthly_sum"]) * 30.44
        cumulative_so_far += annual_production
        forecast_list.append({
            "year": i,
            "rate": rate,
            "cumulative": cumulative_so_far,
        })

    # Decline rate from Arps params if available (di is often per time unit in library)
    decline_rate_annual = 0.0
    if params_dict and "di" in params_dict:
        # decline_curve may report di in different units; assume annual
        decline_rate_annual = float(params_dict.get("di", 0))

    return {
        "forecast": forecast_list,
        "current_rate": current_rate,
        "decline_rate_annual": decline_rate_annual,
        "dca_params": params_dict,
        "economic_limit": current_rate * 0.1,
        "model": model,
        "kind": kind,
    }
