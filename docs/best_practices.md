# Best Practices for Reservoir Engineers

## Overview

This guide provides industry best practices for using ResSmith in production reservoir engineering workflows. These practices are based on industry standards and common pitfalls observed in real-world applications.

## Data Quality and Preparation

### 1. Validate Input Data

**Always validate your production data before analysis:**

```python
from ressmith import detect_outliers, validate_production_data

# Check for outliers
outliers = detect_outliers(data, method='iqr')
if len(outliers) > 0:
    logger.warning(f"Found {len(outliers)} outliers. Review before proceeding.")

# Validate temporal alignment
if not data.index.is_monotonic_increasing:
    data = data.sort_index()
```

**Key Checks:**
- No negative production rates
- Valid date ranges (no future dates, reasonable historical range)
- Consistent time intervals (monthly, daily, etc.)
- Handle missing values explicitly (don't let pandas fill automatically)

### 2. Handle Operational Changes

**Identify and account for operational events:**

```python
# Mark operational changes
data['is_operational'] = True
data.loc[shut_in_dates, 'is_operational'] = False
data.loc[workover_dates, 'is_operational'] = False

# Filter to operational periods only
operational_data = data[data['is_operational']]
```

**Common Operational Events:**
- Shut-ins (planned or unplanned)
- Workovers and stimulations
- Facility constraints
- Choke changes
- Artificial lift installation/optimization

### 3. Pressure Normalization

**For gas wells or variable pressure systems, always normalize rates:**

```python
from ressmith import normalize_production_with_pressure

# Gas well with pressure data
normalized_data = normalize_production_with_pressure(
    data,
    rate_col='gas',
    pressure_col='pressure',
    initial_pressure=5000.0
)

# Then fit decline model
forecast = fit_forecast(normalized_data, model_name='arps_hyperbolic')
```

**When to Normalize:**
- Gas wells (always)
- Oil wells with significant pressure decline
- Variable pressure systems
- Before comparing wells with different pressure histories

## Model Selection and Validation

### 1. Minimum Data Requirements

**Ensure sufficient data before fitting models:**

```python
# Check minimum data requirements
MIN_MONTHS = 12  # Minimum for decline analysis
if len(data) < MIN_MONTHS:
    raise ValueError(
        f"Insufficient data: {len(data)} months. "
        f"Minimum {MIN_MONTHS} months required for decline analysis."
    )
```

**Guidelines:**
- **Decline Analysis**: Minimum 12 months of production data
- **Early-Time Analysis**: 6-12 months (use RTA, not decline models)
- **Mature Wells**: 24+ months for reliable forecasts
- **Unconventional**: May need 18+ months due to complex decline behavior

### 2. Model Comparison

**Always compare multiple models:**

```python
from ressmith import compare_models

# Compare models
comparison = compare_models(
    data,
    model_names=['arps_hyperbolic', 'power_law', 'duong', 'arps_exponential'],
    horizon=60
)

# Select best model based on multiple criteria
best_model = comparison.loc[
    (comparison['r_squared'] > 0.9) & 
    (comparison['mape'] < 15)
].sort_values('r_squared', ascending=False).iloc[0]

print(f"Best model: {best_model['model_name']}")
print(f"R²: {best_model['r_squared']:.3f}")
print(f"MAPE: {best_model['mape']:.2f}%")
```

**Selection Criteria:**
- **R² > 0.9**: Good fit
- **MAPE < 15%**: Reasonable forecast accuracy
- **Realistic Parameters**: Check parameter ranges (see Model Selection Guide)
- **Physical Interpretation**: Parameters should make engineering sense

### 3. Model Validation

**Validate models with backtesting:**

```python
from ressmith import walk_forward_backtest

# Walk-forward validation
backtest = walk_forward_backtest(
    data,
    model_name='arps_hyperbolic',
    forecast_horizons=[12, 24, 36],
    min_train_size=12,
    step_size=3
)

# Check validation metrics
print(f"12-month forecast RMSE: {backtest[backtest['horizon']==12]['rmse'].mean():.2f}")
print(f"24-month forecast RMSE: {backtest[backtest['horizon']==24]['rmse'].mean():.2f}")
```

**Validation Guidelines:**
- Use walk-forward backtesting for time series
- Require RMSE < 20% of mean production rate
- Check forecast accuracy at multiple horizons
- Validate on out-of-sample data (not used for fitting)

## Forecasting Best Practices

### 1. Forecast Horizon

**Use appropriate forecast horizons:**

```python
# Calculate appropriate horizon
historical_length = len(data)
forecast_horizon = min(
    historical_length * 2,  # 2x historical data
    120  # Maximum 10 years
)

forecast = fit_forecast(
    data,
    model_name='arps_hyperbolic',
    horizon=forecast_horizon
)
```

**Guidelines:**
- **Short-term (1-2 years)**: Use 2x historical data length
- **Long-term (5-10 years)**: Maximum 10 years, use probabilistic methods
- **Reserves Estimation**: Use economic limits, not fixed horizons

### 2. Economic Limits

**Always apply economic limits:**

```python
from ressmith import estimate_eur

eur_result = estimate_eur(
    data,
    model_name='arps_hyperbolic',
    t_max=360,  # 30 years maximum
    econ_limit=10.0  # Stop at 10 STB/day
)

print(f"EUR: {eur_result['eur']:.0f} STB")
print(f"Economic limit reached at: {eur_result.get('t_econ_limit', 'N/A')} days")
```

**Economic Limit Guidelines:**
- **Oil Wells**: 5-15 STB/day (depends on operating costs)
- **Gas Wells**: 50-200 MCF/day (depends on processing costs)
- **Unconventional**: May be lower (5-10 STB/day) due to higher costs

### 3. Probabilistic Forecasting

**Always use probabilistic methods for reserves:**

```python
from ressmith import probabilistic_forecast

# Probabilistic forecast
prob = probabilistic_forecast(
    data,
    model_name='arps_hyperbolic',
    horizon=120,
    n_samples=1000,
    param_uncertainty={
        'qi': (0.1, 'relative'),  # ±10% uncertainty
        'di': (0.2, 'relative'),  # ±20% uncertainty
        'b': (0.1, 'absolute')    # ±0.1 absolute uncertainty
    }
)

# Report P10/P50/P90
print(f"P10 EUR: {prob['p10'].sum():.0f} STB")
print(f"P50 EUR: {prob['p50'].sum():.0f} STB")
print(f"P90 EUR: {prob['p90'].sum():.0f} STB")
```

**Uncertainty Guidelines:**
- **P10 (Optimistic)**: 10% probability of exceeding
- **P50 (Deterministic)**: 50% probability (most likely)
- **P90 (Conservative)**: 90% probability of exceeding
- **Use P90 for reserves booking** (conservative estimate)

## Reservoir Engineering Integration

### 1. Combine with RTA

**Use RTA to validate decline models:**

```python
from ressmith import analyze_production_data, identify_flow_regime

# Check flow regime
regime = identify_flow_regime(time, rate)

if regime == 'transient':
    # Use RTA, not decline models
    rta_result = analyze_production_data(time, rate)
    print(f"Permeability: {rta_result['permeability']:.2f} md")
    print(f"Flow regime: {rta_result['flow_regime']}")
else:
    # Safe to use decline models
    forecast = fit_forecast(data, model_name='arps_hyperbolic')
```

### 2. Material Balance Validation

**Validate forecasts with material balance:**

```python
from ressmith.primitives import solution_gas_drive_material_balance

# Material balance check
mb_result = solution_gas_drive_material_balance(
    pressure=current_pressure,
    cumulative_production=current_cumulative,
    params=mb_params
)

# Compare with forecast
forecast_cumulative = forecast.yhat.cumsum()
if abs(mb_result['Np_calculated'] - forecast_cumulative.iloc[-1]) > 0.1 * forecast_cumulative.iloc[-1]:
    logger.warning("Material balance and forecast don't match. Review assumptions.")
```

### 3. Well Interference

**Account for well interference in multi-well fields:**

```python
from ressmith import analyze_interference_with_production_history

# Analyze interference
interference = analyze_interference_with_production_history(
    well_data={
        'well_1': data1,
        'well_2': data2
    },
    well_locations={
        'well_1': (lat1, lon1),
        'well_2': (lat2, lon2)
    }
)

# Adjust forecasts for interference
if interference['interference_factor'] > 0.1:
    logger.warning(
        f"Significant interference detected: "
        f"{interference['interference_factor']*100:.1f}%"
    )
```

## Economics Best Practices

### 1. Scenario Analysis

**Always run multiple price scenarios:**

```python
from ressmith import evaluate_scenarios
from ressmith.objects import EconSpec

base_spec = EconSpec(
    price_assumptions={'oil': 70.0},
    opex=15.0,
    discount_rate=0.1
)

scenarios = {
    'low': {'prices': {'oil': 50.0}, 'opex': 18.0},
    'base': {},
    'high': {'prices': {'oil': 90.0}, 'opex': 12.0}
}

results = evaluate_scenarios(forecast, base_spec, scenarios)

for scenario, result in results.items():
    print(f"{scenario}: NPV = ${result.npv:,.0f}")
```

### 2. Discount Rate Selection

**Use appropriate discount rates:**

- **Corporate Hurdle**: Typically 10-15%
- **Risk-Adjusted**: Higher for riskier projects (15-20%)
- **Real Options**: Lower for flexible projects (8-12%)
- **Reserves Reporting**: Follow SEC/SPE guidelines

### 3. Operating Costs

**Include all relevant costs:**

```python
econ_spec = EconSpec(
    price_assumptions={'oil': 70.0, 'gas': 3.0},
    opex=15.0,  # $/STB operating costs
    capex=100000.0,  # Initial capital
    discount_rate=0.1,
    taxes=0.35  # Tax rate
)
```

**Cost Components:**
- **OPEX**: Lifting costs, processing, transportation
- **CAPEX**: Drilling, completion, facilities
- **Taxes**: Severance, ad valorem, income taxes
- **Abandonment**: End-of-life costs

## Portfolio Analysis

### 1. Consistent Methodology

**Use consistent methods across portfolio:**

```python
from ressmith import analyze_portfolio

# Analyze entire portfolio with same method
portfolio = analyze_portfolio(
    well_data,
    model_name='arps_hyperbolic',  # Consistent model
    horizon=60,  # Consistent horizon
    econ_spec=econ_spec  # Consistent economics
)

# Rank by value
ranked = portfolio.sort_values('npv', ascending=False)
```

### 2. Risk Reporting

**Report risk metrics:**

```python
# Portfolio risk metrics
portfolio_stats = {
    'total_eur_p50': portfolio['eur'].sum(),
    'total_npv': portfolio['npv'].sum(),
    'wells_count': len(portfolio),
    'avg_eur': portfolio['eur'].mean(),
    'std_eur': portfolio['eur'].std()
}

print(f"Portfolio P50 EUR: {portfolio_stats['total_eur_p50']:,.0f} STB")
print(f"Portfolio NPV: ${portfolio_stats['total_npv']:,.0f}")
```

## Documentation and Reproducibility

### 1. Document Assumptions

**Always document key assumptions:**

```python
# Document assumptions
assumptions = {
    'model': 'arps_hyperbolic',
    'horizon': 60,
    'econ_limit': 10.0,
    'oil_price': 70.0,
    'discount_rate': 0.1,
    'data_period': '2019-01 to 2023-12',
    'operational_events': 'Shut-in 2020-03 to 2020-05'
}

# Save with results
results['assumptions'] = assumptions
```

### 2. Version Control

**Track code and data versions:**

```python
import ressmith
import pandas as pd

# Document versions
metadata = {
    'ressmith_version': ressmith.__version__,
    'pandas_version': pd.__version__,
    'analysis_date': pd.Timestamp.now(),
    'analyst': 'Engineer Name'
}
```

## Summary Checklist

Before finalizing any analysis:

- [ ] Data validated (outliers, missing values, temporal alignment)
- [ ] Operational events accounted for
- [ ] Pressure normalized (if applicable)
- [ ] Multiple models compared
- [ ] Model validated with backtesting
- [ ] Forecast horizon appropriate
- [ ] Economic limits applied
- [ ] Probabilistic forecast generated (P10/P50/P90)
- [ ] RTA validation (if early-time data)
- [ ] Material balance check (if applicable)
- [ ] Well interference considered (if multi-well)
- [ ] Multiple price scenarios evaluated
- [ ] Assumptions documented
- [ ] Results reproducible

For more detailed guidance, see:
- [Model Selection Guide](model_selection_guide.md)
- [Advanced Workflows](advanced_workflows.md)
- [API Reference](api.rst)
