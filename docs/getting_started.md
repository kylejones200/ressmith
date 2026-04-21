This guide walks through a full ResSmith workflow using a single well. The example uses a synthetic decline series. The goal is clarity and structure.

## Overview

ResSmith provides a comprehensive toolkit for reservoir engineering analysis, from single-well decline curve analysis to portfolio-level forecasting and economics. This guide demonstrates the core workflow: production data → decline model fitting → forecast → economics.

## Production Data Preparation

We begin with a pandas DataFrame that represents monthly oil production.

```python
import numpy as np
import pandas as pd

idx = pd.date_range("2019-01-01", periods=48, freq="ME")
rate = 800 * (1 + 0.02 * idx.month) ** -1
rate = rate + np.random.normal(0, 20, size=len(rate))

# Default decline fitting expects an ``oil`` rate column (or pass ``phase`` / column mapping).
df = pd.DataFrame({"oil": rate}, index=idx)
```

ResSmith validates temporal inputs using shared typing from Timesmith (optional).

```python
from timesmith.typing.validators import assert_series_like

assert_series_like(df["oil"])
```

We fit a decline model. `fit_forecast` returns **(forecast, fitted_params)**.

```python
from ressmith.workflows import fit_forecast

forecast, params = fit_forecast(
    df,
    model_name="arps_hyperbolic",
    horizon=24,
)
```

The result contains a forecasted rate series and model metadata.

```python
print(forecast.yhat.tail())
print(params)
```

We now evaluate economics. `EconSpec` uses **per-period** operating expense in the current cashflow helper (not $/bbl); adjust to your forecast step.

```python
from ressmith.workflows import evaluate_economics
from ressmith.objects import EconSpec

econ = EconSpec(
    price_assumptions={"oil": 70.0},
    opex=400.0,
    capex=0.0,
    discount_rate=0.1,
)

econ_result = evaluate_economics(forecast, econ)
```

The output includes cashflows and value metrics.

```python
print(econ_result.npv)
print(econ_result.irr)
print(econ_result.cashflows.head())
```

## Model Selection

ResSmith supports multiple decline models. The choice depends on your reservoir:

- **ARPS Hyperbolic** (`arps_hyperbolic`): Most common for oil and gas wells. Use when decline rate decreases over time (b-factor > 0).
- **ARPS Exponential** (`arps_exponential`): Use for wells with constant decline rate (b = 0).
- **ARPS Harmonic** (`arps_harmonic`): Use when b = 1 (rare in practice).
- **Power Law** (`power_law`): Good for unconventional plays with complex decline behavior.
- **Duong** (`duong`): Designed for tight gas and shale wells.

When to use each model:
- **Early-time data (< 12 months)**: Use Power Law or Duong models
- **Mature wells (> 24 months)**: ARPS Hyperbolic is usually appropriate
- **Unconventional plays**: Try ensemble_forecast() with multiple models

## Common Workflows

### Comparing Multiple Models

```python
from ressmith.workflows import compare_models

comparison = compare_models(
    data=df,
    model_names=['arps_hyperbolic', 'power_law', 'duong'],
    horizon=36
)
print(comparison.sort_values('r_squared', ascending=False))
```

### Probabilistic Forecasting (P10/P50/P90)

```python
from ressmith.workflows import probabilistic_forecast

prob_result = probabilistic_forecast(
    data=df,
    model_name='arps_hyperbolic',
    horizon=36,
    n_samples=1000
)
print(f"P50 EUR: {prob_result['p50'].sum():.0f} bbl")
print(f"P10 EUR: {prob_result['p10'].sum():.0f} bbl")
print(f"P90 EUR: {prob_result['p90'].sum():.0f} bbl")
```

### Portfolio Analysis

Each well’s frame should match the fit workflow: **DatetimeIndex** and an **`oil`** column (unless you pass a column mapping via `fit_forecast` / portfolio kwargs).

```python
from ressmith.workflows import analyze_portfolio

well_data = {
    "well_1": df,
    "well_2": df.assign(oil=df["oil"] * 0.95),
    "well_3": df.assign(oil=df["oil"] * 1.05),
}
portfolio = analyze_portfolio(
    well_data,
    model_name="arps_hyperbolic",
    econ_spec=econ,
)
print(portfolio.sort_values("npv", ascending=False))
```

## Best Practices

1. **Data Quality**: Ensure production data is clean (no negative rates, valid dates)
2. **Model Selection**: Start with ARPS Hyperbolic, try ensemble for uncertainty
3. **Forecast Horizon**: Use 2-3x historical data length for forecasts
4. **Economics**: Always use probabilistic forecasting for reserves estimation
5. **Validation**: Use walk_forward_backtest() to validate model performance

## Next Steps

- See `examples/` directory for more complex workflows
- Check `docs/architecture.md` for architecture details
- Review API documentation for advanced features

This is the full ResSmith loop. Rates become forecasts. Forecasts become cashflow. Each step stays explicit and testable.

