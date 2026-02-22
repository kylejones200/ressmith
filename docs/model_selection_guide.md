# Model Selection Guide for Reservoir Engineers

## Overview

Choosing the right decline model is critical for accurate reserves estimation and forecasting. This guide provides engineering-focused guidance on when and why to use each model available in ResSmith.

## Understanding Decline Behavior

Before selecting a model, understand your reservoir's decline characteristics:

- **Exponential Decline**: Constant decline rate (b = 0). Rare in practice, but occurs in some high-permeability reservoirs with strong water drive.
- **Hyperbolic Decline**: Declining decline rate (0 < b < 1). Most common for oil and gas wells. The decline rate decreases over time.
- **Harmonic Decline**: Special case where b = 1. Rare, but can occur in some gas wells.
- **Power Law Decline**: Complex decline behavior with time-dependent decline rate. Common in unconventional plays.
- **Duong Decline**: Specifically designed for tight gas and shale reservoirs with fracture-dominated flow.

## Model Selection by Reservoir Type

### Conventional Oil Reservoirs

**Primary Choice: ARPS Hyperbolic**

- **When to Use**: Most conventional oil wells with 12+ months of production data
- **Physical Interpretation**: 
  - `qi`: Initial production rate (STB/day)
  - `di`: Initial decline rate (1/day or 1/month)
  - `b`: Hyperbolic exponent (typically 0.3-0.7 for oil wells)
- **Typical b-factors**: 0.3-0.7 for solution gas drive, 0.5-0.8 for water drive
- **Best Practices**:
  - Require at least 12 months of data for reliable fitting
  - Use `fit_segmented_forecast()` if decline behavior changes over time
  - Validate b-factor is between 0 and 1 (values > 1 may indicate data quality issues)

**Example**:
```python
from ressmith import fit_forecast

# Conventional oil well with 24 months of data
forecast, params = fit_forecast(
    data=production_data,
    model_name='arps_hyperbolic',
    horizon=60  # 5-year forecast
)

# Check b-factor is reasonable
assert 0 < params['b'] < 1, f"Unrealistic b-factor: {params['b']}"
```

### Unconventional (Shale/Tight) Reservoirs

**Primary Choice: Power Law or Duong**

- **When to Use**: 
  - Early-time data (< 12 months): Use Power Law or Duong
  - Mature wells (> 24 months): May transition to ARPS Hyperbolic
- **Physical Interpretation**:
  - **Power Law**: Captures complex decline with time-dependent behavior
    - `n`: Decline exponent (typically 0.1-0.5 for shale)
    - Higher n values indicate stronger early-time decline
  - **Duong**: Designed for fracture-dominated flow
    - `a`: Intercept parameter
    - `m`: Slope parameter (typically 0.5-1.0)
- **Best Practices**:
  - Use `ensemble_forecast()` to compare multiple models
  - Power Law often fits early-time data better than ARPS
  - Duong model specifically captures linear flow regime behavior
  - Consider using `probabilistic_forecast()` for uncertainty quantification

**Example**:
```python
from ressmith import ensemble_forecast, compare_models

# Compare models for shale well
comparison = compare_models(
    data=shale_data,
    model_names=['power_law', 'duong', 'arps_hyperbolic'],
    horizon=120
)

# Use ensemble for uncertainty
ensemble = ensemble_forecast(
    data=shale_data,
    model_names=['power_law', 'duong'],
    horizon=120
)
```

### Gas Reservoirs

**Primary Choice: ARPS Hyperbolic or Gas-Specific Models**

- **When to Use**:
  - Conventional gas: ARPS Hyperbolic (b typically 0.5-1.0)
  - Tight gas: Duong or Power Law
  - Gas with strong water drive: May show exponential-like behavior
- **Physical Interpretation**:
  - Gas wells often have higher b-factors than oil wells
  - Harmonic decline (b = 1) can occur but is rare
  - Use `gas_reservoir_pz_method()` for material balance analysis
- **Best Practices**:
  - Use pressure-normalized rates for gas wells: `normalize_production_with_pressure()`
  - Consider using `calculate_pseudopressure()` for high-pressure gas
  - Validate against material balance for long-term forecasts

### Waterflood Reservoirs

**Primary Choice: ARPS Hyperbolic with Segmentation**

- **When to Use**: Wells showing multiple decline segments
- **Physical Interpretation**:
  - Early segment: Primary depletion
  - Later segment: Waterflood response (often lower decline rate)
- **Best Practices**:
  - Use `fit_segmented_forecast()` to capture phase changes
  - Monitor WOR trends: `forecast_wor_gor_with_coning()`
  - Use EOR workflows: `analyze_waterflood()`, `predict_waterflood_performance()`

## Physical Interpretation of Parameters

### ARPS Hyperbolic Parameters

- **qi (Initial Rate)**: 
  - Represents the production rate at the start of the decline period
  - Should match observed early-time rates
  - Unrealistic values may indicate poor fit or data quality issues

- **di (Initial Decline Rate)**:
  - Decline rate at time zero
  - Units: 1/time (e.g., 1/month or 1/year)
  - Typical range: 0.001-0.1 per month for oil wells
  - Very high values (> 0.5/month) may indicate transient flow, not decline

- **b (Hyperbolic Exponent)**:
  - Controls how decline rate changes over time
  - b = 0: Exponential (constant decline)
  - 0 < b < 1: Hyperbolic (declining decline rate)
  - b = 1: Harmonic
  - b > 1: Physically unrealistic (check data quality)
  - Typical: 0.3-0.7 for oil, 0.5-1.0 for gas

### Power Law Parameters

- **qi**: Initial rate (same as ARPS)
- **di**: Initial decline rate
- **n**: Decline exponent
  - Lower n (0.1-0.3): Stronger early-time decline, common in shale
  - Higher n (0.3-0.5): More gradual decline
  - n > 0.5: May indicate transition to boundary-dominated flow

### Duong Parameters

- **qi**: Initial rate
- **a**: Intercept parameter (typically 0.1-10)
- **m**: Slope parameter
  - m ≈ 0.5: Linear flow (fracture-dominated)
  - m ≈ 1.0: Boundary-dominated flow
  - Typical range: 0.5-1.0

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Using Exponential Model for Hyperbolic Data

**Problem**: Exponential model (b=0) assumes constant decline rate, which is rarely realistic.

**Solution**: Always start with ARPS Hyperbolic. Check if b ≈ 0, then consider exponential.

```python
# Wrong approach
forecast_exp = fit_forecast(data, model_name='arps_exponential')

# Correct approach
forecast_hyper, params = fit_forecast(data, model_name='arps_hyperbolic')
if abs(params['b']) < 0.01:
    # Data truly shows exponential decline
    forecast_exp = fit_forecast(data, model_name='arps_exponential')
```

### Pitfall 2: Fitting Models to Transient Flow Data

**Problem**: Early-time data (< 6 months) may show transient flow, not true decline.

**Solution**: 
- Require minimum 12 months of data for decline analysis
- Use RTA workflows to identify flow regimes: `identify_flow_regime()`
- Consider using Power Law or Duong for early-time data

```python
# Check flow regime first
from ressmith import identify_flow_regime
regime = identify_flow_regime(time, rate)

if regime == 'transient':
    # Use RTA models, not decline models
    from ressmith import analyze_production_data
    rta_result = analyze_production_data(time, rate)
else:
    # Safe to use decline models
    forecast = fit_forecast(data, model_name='arps_hyperbolic')
```

### Pitfall 3: Unrealistic b-Factors

**Problem**: b > 1 or b < 0 indicates model fit issues or data quality problems.

**Solution**:
- Validate b-factor after fitting
- Check for outliers or data quality issues
- Consider segmented models if behavior changes

```python
forecast, params = fit_forecast(data, model_name='arps_hyperbolic')

# Validate b-factor
if not (0 <= params['b'] <= 1):
    logger.warning(f"Unrealistic b-factor: {params['b']}. Check data quality.")
    # Try segmented model
    forecast_seg = fit_segmented_forecast(data, n_segments=2)
```

### Pitfall 4: Ignoring Pressure Effects

**Problem**: Production rates should be normalized by pressure for accurate decline analysis.

**Solution**: Use pressure normalization workflows.

```python
# For gas wells or variable pressure
from ressmith import normalize_production_with_pressure

normalized_data = normalize_production_with_pressure(
    data,
    rate_col='gas',
    pressure_col='pressure',
    initial_pressure=5000.0
)

# Then fit decline model
forecast = fit_forecast(normalized_data, model_name='arps_hyperbolic')
```

### Pitfall 5: Over-Fitting with Too Many Segments

**Problem**: Using too many segments in segmented models can over-fit noise.

**Solution**:
- Start with single segment
- Add segments only when clear phase changes occur
- Use `walk_forward_backtest()` to validate

```python
# Start simple
forecast_simple = fit_forecast(data, model_name='arps_hyperbolic')

# Only add segments if needed
if has_clear_phase_change(data):
    forecast_seg = fit_segmented_forecast(data, n_segments=2)
    
    # Validate with backtesting
    from ressmith import walk_forward_backtest
    backtest = walk_forward_backtest(data, model_name='arps_hyperbolic')
```

## Industry Best Practices

### 1. Data Quality First

- Clean data before modeling: remove outliers, handle missing values
- Validate temporal alignment: ensure dates are correct
- Check for operational changes: shut-ins, workovers, etc.

```python
from ressmith import detect_outliers

# Clean data first
cleaned_data = detect_outliers(data, method='iqr')
```

### 2. Model Validation

- Always use `walk_forward_backtest()` for validation
- Compare multiple models: `compare_models()`
- Check fit diagnostics: R², RMSE, residuals

```python
from ressmith import compare_models, walk_forward_backtest

# Compare models
comparison = compare_models(
    data,
    model_names=['arps_hyperbolic', 'power_law', 'duong']
)

# Validate best model
backtest = walk_forward_backtest(
    data,
    model_name='arps_hyperbolic',  # Best from comparison
    forecast_horizons=[12, 24, 36]
)
```

### 3. Uncertainty Quantification

- Always use probabilistic forecasting for reserves
- Report P10/P50/P90 scenarios
- Use `probabilistic_forecast()` for uncertainty

```python
from ressmith import probabilistic_forecast

prob_result = probabilistic_forecast(
    data,
    model_name='arps_hyperbolic',
    horizon=120,
    n_samples=1000
)

# Report reserves
eur_p10 = prob_result['p10'].sum()
eur_p50 = prob_result['p50'].sum()
eur_p90 = prob_result['p90'].sum()
```

### 4. Forecast Horizon Guidelines

- Use 2-3x historical data length for forecasts
- Longer horizons require more uncertainty
- Consider economic limits in forecasts

```python
historical_length = len(data)
forecast_horizon = min(historical_length * 2, 120)  # Max 10 years

forecast = fit_forecast(
    data,
    model_name='arps_hyperbolic',
    horizon=forecast_horizon,
    econ_limit=10.0  # Stop at 10 STB/day
)
```

### 5. Integration with Other Analysis

- Combine decline analysis with RTA: `analyze_production_data()`
- Use material balance for validation: `solution_gas_drive_material_balance()`
- Consider well interference: `analyze_interference_with_production_history()`

## Worked Example: Shale Oil Well

```python
import pandas as pd
import numpy as np
from ressmith import (
    fit_forecast, compare_models, ensemble_forecast,
    probabilistic_forecast, estimate_eur
)

# Load production data (18 months of shale oil production)
data = pd.read_csv('shale_well_data.csv', index_col='date', parse_dates=True)

# Step 1: Compare models
comparison = compare_models(
    data,
    model_names=['arps_hyperbolic', 'power_law', 'duong'],
    horizon=120
)
print(comparison.sort_values('r_squared', ascending=False))

# Step 2: Use ensemble for uncertainty
ensemble = ensemble_forecast(
    data,
    model_names=['power_law', 'duong'],  # Best two models
    horizon=120
)

# Step 3: Probabilistic forecast
prob = probabilistic_forecast(
    data,
    model_name='power_law',  # Best single model
    horizon=120,
    n_samples=1000
)

# Step 4: Estimate EUR
eur_result = estimate_eur(
    data,
    model_name='power_law',
    t_max=360,  # 30 years
    econ_limit=5.0  # 5 STB/day economic limit
)

print(f"P50 EUR: {prob['p50'].sum():.0f} STB")
print(f"P10 EUR: {prob['p10'].sum():.0f} STB")
print(f"P90 EUR: {prob['p90'].sum():.0f} STB")
print(f"Deterministic EUR: {eur_result['eur']:.0f} STB")
```

## Summary

- **Conventional Oil**: Start with ARPS Hyperbolic (b typically 0.3-0.7)
- **Unconventional**: Use Power Law or Duong for early-time, consider ARPS for mature wells
- **Gas**: ARPS Hyperbolic with pressure normalization, higher b-factors (0.5-1.0)
- **Always**: Validate models, quantify uncertainty, check data quality
- **Never**: Fit models to transient flow data, ignore pressure effects, use unrealistic parameters

For more advanced workflows, see the [Advanced Workflows Guide](advanced_workflows.md).
