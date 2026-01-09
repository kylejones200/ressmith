# Migration Plan: pydca → ResSmith

## Overview

This document outlines the migration of the decline curve analysis library from `/Users/kylejonespatricia/pydca` into ResSmith's strict 4-layer architecture, ensuring compatibility with plotsmith, anomsmith, and geosmith.

## Architecture Alignment

### ResSmith 4-Layer Structure
- **Layer 1 (Objects)**: Immutable dataclasses, no imports from other layers
- **Layer 2 (Primitives)**: Algorithms and base classes, imports only Objects
- **Layer 3 (Tasks)**: Task orchestration, imports Objects and Primitives
- **Layer 4 (Workflows)**: User-facing functions, imports everything + I/O/plotting

### Integration with Other Smith Libraries
- **plotsmith**: Use for all plotting (Layer 4 workflows can import plotsmith)
- **anomsmith**: Share typing from timesmith.typing
- **geosmith**: Share typing from timesmith.typing, potential spatial analysis integration

## Migration Priority

### Phase 1: Core Models (✅ Already Started)
- [x] ArpsHyperbolicModel
- [x] ArpsExponentialModel  
- [x] ArpsHarmonicModel
- [x] LinearDeclineModel (simple alternative)

### Phase 2: Segmented Models (✅ Completed)
- [x] SegmentedDeclineModel - Multi-segment ARPS with change points
- [x] HyperbolicToExponentialSwitchModel - Switch model

### Phase 3: Advanced Models (✅ Completed)
- [x] PowerLawDeclineModel
- [x] DuongModel
- [x] StretchedExponentialModel

### Phase 4: Fitting & Diagnostics (✅ Completed)
- [x] Enhanced fitting with ramp detection (ramp-aware initial guesses implemented)
- [x] Fit diagnostics and validation (FitDiagnostics module added)
- [x] Parameter constraints and bounds (constraints module with validation)

### Phase 5: Multi-phase Support (✅ Completed)
- [x] Full oil/gas/water handling (ProductionSeries supports all phases)
- [x] Multi-phase economics (enhanced cashflow_from_forecast)
- [x] Phase-specific decline models (all models support phase selection)

### Phase 6: Integration Features (✅ Completed)
- [x] Scenario analysis workflows (evaluate_scenarios, scenario_summary)
- [x] Backtesting workflows (walk_forward_backtest)
- [x] Model comparison workflows (compare_models)
- [x] EUR estimation workflows (estimate_eur)
- [x] Fixed Terminal Decline Model (FixedTerminalDeclineModel)
- [x] Yield models (GOR, CGR, water cut) - forecast_with_yields workflow
- [x] Ensemble modeling (ensemble_forecast workflow)
- [x] Uncertainty quantification (probabilistic_forecast workflow)
- [x] Portfolio analysis (analyze_portfolio, aggregate_portfolio_forecast, rank_wells)
- [x] Plotting integration with plotsmith (plot_forecast workflow - graceful degradation)
- [x] Anomaly detection integration with anomsmith (detect_outliers workflow - graceful degradation)
- [x] Spatial analysis integration with geosmith (spatial_analysis workflow - graceful degradation)

## Key Files to Migrate

### From pydca/decline_curve/models_arps.py
- ExponentialArps → Already in ressmith as ArpsExponentialModel
- HyperbolicArps → Already in ressmith as ArpsHyperbolicModel
- HarmonicArps → Already in ressmith as ArpsHarmonicModel
- HyperbolicToExponential → ✅ Already in ressmith as HyperbolicToExponentialSwitchModel

### From pydca/decline_curve/segmented_decline.py
- DeclineSegment → ✅ Already in ressmith as Layer 1 Object
- SegmentedDeclineResult → ✅ Already in ressmith as Layer 1 Object
- Segmented decline logic → ✅ Already in ressmith as Layer 2 Primitives

### From pydca/decline_curve/fitting.py
- Enhanced fitting logic with ramp detection → ✅ Completed (fitting_utils.py)
- Parameter validation → ✅ Completed (constraints.py)
- Initial guess heuristics → ✅ Completed

### From pydca/decline_curve/economics.py
- Enhanced economics calculations → ✅ Completed
- Multi-phase economics → ✅ Completed
- Scenario analysis → ✅ Completed (scenarios.py)

### From pydca/decline_curve/decline_variants.py
- fixed_terminal_decline → ✅ Completed (variants.py, FixedTerminalDeclineModel)

### From pydca/decline_curve/yield_models.py
- ConstantYield, DecliningYield, HyperbolicYield → ✅ Migrated (primitives/yields.py, forecast_with_yields workflow)

### From pydca/decline_curve/ensemble.py
- EnsembleForecaster → ✅ Migrated (primitives/ensemble.py, workflows/ensemble.py)

### From pydca/decline_curve/uncertainty.py or monte_carlo.py
- Monte Carlo simulation → ✅ Migrated (primitives/uncertainty.py, workflows/uncertainty.py)
- Probabilistic forecasts → ✅ Migrated (probabilistic_forecast workflow)

## Integration Points

### With plotsmith
- Use `plotsmith.workflows.plot_timeseries()` for forecast visualization
- Use `plotsmith.workflows.plot_residuals()` for fit diagnostics
- Use `plotsmith.workflows.plot_backtest()` for backtest results

### With anomsmith
- Share `timesmith.typing.SeriesLike` and `PanelLike` types
- Potential: Use anomsmith for outlier detection in production data

### With geosmith
- Share `timesmith.typing` types
- Potential: Spatial kriging for EUR estimation
- Potential: Geospatial well analysis

## Migration Strategy

1. **Preserve Functionality**: Ensure all migrated code maintains same behavior
2. **Respect Layer Boundaries**: Strictly enforce one-way imports
3. **Type Safety**: Use timesmith.typing for shared types
4. **Test Coverage**: Migrate tests alongside code
5. **Incremental**: Migrate one model/feature at a time

## Migration Status Summary

**✅ COMPLETED - Decline Curve Analysis Models:**
- All core ARPS models (Exponential, Hyperbolic, Harmonic)
- Advanced decline models (PowerLaw, Duong, StretchedExponential)
- Segmented decline models
- Fixed terminal decline variant
- Yield models for multi-phase forecasting
- Ensemble modeling for combining multiple decline models
- Uncertainty quantification with Monte Carlo
- Portfolio analysis workflows
- All Phase 1-6 features from original plan

**❌ NOT MIGRATED (Out of Scope for ResSmith):**
- Deep learning models (DeepAR, TFT, LSTM, TimeSFM, Chronos) - These belong in a separate ML forecasting library
- Statistical models (ARIMA, exponential smoothing, moving average) - Not decline curve specific
- Physics-based models (material balance, pressure decline) - Different domain, belong in reservoir engineering library
- RTA/IPR/VLP - Reservoir engineering tools, not decline curve analysis

**Note:** ResSmith focuses specifically on **decline curve analysis** models. Other forecasting and reservoir engineering capabilities in pydca are intentionally excluded to maintain clear boundaries and avoid scope creep.

