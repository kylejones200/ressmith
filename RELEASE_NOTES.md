# Release v0.2.3

## Summary

Patch release: integrated study orchestration, richer examples, documentation updates, and small workflow fixes.

## Highlights

- **Integrated studies** — `well_interference_study`, `coning_study`, and `enhanced_rta_study` in `ressmith.workflows.integrated_reservoir_workflows`, exported from `ressmith` and `ressmith.workflows`.
- **Examples** — `examples/integrated_studies.py`; expanded `basic_fit_forecast`, `basic_economics`, and `basic_segmented` with named constants for clarity.
- **Docs** — `docs/engineer_capability_map.md` and refreshed getting-started / index links.
- **Housekeeping** — Removed superseded top-level audit/migration markdown; minor fixes in leakage checks and spatial kriging.

## Installation

```bash
uv pip install ressmith==0.2.3
```

---

# Release v0.2.0

## Summary

This release includes major feature additions from the pydca migration, adding ensemble modeling, uncertainty quantification, portfolio analysis, and ecosystem integrations.

## New Features

### Advanced Decline Models
- **Fixed Terminal Decline Model** - Prevents unrealistic long-term forecasts by transitioning to fixed terminal decline rates (5-10% per year)
- Supports rate-based and time-based transition criteria

### Yield Models & Multi-Phase Forecasting
- **Yield Models** - Constant, declining, and hyperbolic yield models for GOR, CGR, and water cut
- **forecast_with_yields()** workflow - Forecast associated phases (gas, water) from primary phase (oil) using yield models
- Auto-fitting yield models from historical data

### Ensemble Modeling
- **ensemble_forecast()** workflow - Combine multiple decline models for improved forecast reliability
- Methods: weighted average, median, and confidence-weighted ensemble
- Support for custom model factories

### Uncertainty Quantification
- **probabilistic_forecast()** workflow - Monte Carlo simulation for probabilistic forecasting
- Parameter uncertainty sampling (normal, lognormal distributions)
- Generate P10/P50/P90 forecasts for risk analysis
- Configurable number of samples and parameter uncertainty

### Portfolio Analysis
- **analyze_portfolio()** - Analyze multiple wells with economics evaluation
- **aggregate_portfolio_forecast()** - Aggregate forecasts across portfolio (sum, mean, median)
- **rank_wells()** - Rank wells by metrics (NPV, EUR, IRR)
- Full integration with EUR and economics workflows

### Ecosystem Integrations
- **plot_forecast()** - Integration with plotsmith (graceful degradation if not installed)
- **detect_outliers()** - Integration with anomsmith for outlier detection
- **spatial_analysis()** - Integration with geosmith for spatial kriging and analysis
- All integrations degrade gracefully if optional packages aren't available

## Enhancements

- Enhanced EUR estimation with support for all decline models
- Multi-phase economics calculations with phase-specific price assumptions
- Comprehensive fit diagnostics (RMSE, MAE, MAPE, R²)
- Parameter constraints and bounds validation
- Ramp-aware initial guesses for better fitting

## API Changes

### New Workflows
- `forecast_with_yields()` - Multi-phase forecasting with yield models
- `ensemble_forecast()` - Ensemble forecasting from multiple models
- `probabilistic_forecast()` - Probabilistic forecasting with uncertainty
- `analyze_portfolio()` - Portfolio-level analysis
- `aggregate_portfolio_forecast()` - Portfolio forecast aggregation
- `rank_wells()` - Well ranking by metrics
- `plot_forecast()` - Plotting integration
- `detect_outliers()` - Outlier detection integration
- `spatial_analysis()` - Spatial analysis integration

### New Models
- `FixedTerminalDeclineModel` - Fixed terminal decline variant

## Migration from pydca

This release completes the migration of decline curve analysis features from pydca:
- All core ARPS models
- Advanced decline models (PowerLaw, Duong, StretchedExponential)
- Segmented decline models
- Yield models for multi-phase forecasting
- Ensemble modeling
- Uncertainty quantification
- Portfolio analysis
- Ecosystem integrations

## Testing

- 65+ tests passing
- Comprehensive test coverage for all new features
- Integration tests with timesmith.typing

## Breaking Changes

None - this is a feature addition release with backward compatibility.

## Dependencies

- Python >= 3.12
- numpy >= 1.24.0
- pandas >= 2.0.0
- timesmith >= 0.1.0
- Optional: scipy (for advanced fitting), matplotlib (for visualization)

## Installation

```bash
uv pip install ressmith==0.2.0
```

Or with optional dependencies:

```bash
uv pip install ressmith[fit]  # Include scipy for optimization
uv pip install ressmith[viz]  # Include matplotlib (or use plotsmith)
```

For development:

```bash
git clone https://github.com/kylejones200/ressmith.git
cd ressmith
uv sync --group dev
```

