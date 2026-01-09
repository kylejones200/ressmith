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

### Phase 6: Integration Features
- [ ] Plotting integration with plotsmith
- [ ] Anomaly detection integration with anomsmith
- [ ] Spatial analysis integration with geosmith

## Key Files to Migrate

### From pydca/decline_curve/models_arps.py
- ExponentialArps → Already in ressmith as ArpsExponentialModel
- HyperbolicArps → Already in ressmith as ArpsHyperbolicModel
- HarmonicArps → Already in ressmith as ArpsHarmonicModel
- HyperbolicToExponential → Need to migrate

### From pydca/decline_curve/segmented_decline.py
- DeclineSegment → Port to Layer 1 Objects
- SegmentedDeclineResult → Port to Layer 1 Objects
- Segmented decline logic → Port to Layer 2 Primitives

### From pydca/decline_curve/fitting.py
- Enhanced fitting logic with ramp detection
- Parameter validation
- Initial guess heuristics

### From pydca/decline_curve/economics.py
- Enhanced economics calculations
- Multi-phase economics
- Scenario analysis

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

## Next Steps

1. Migrate SegmentedDeclineModel (highest priority)
2. Enhance fitting with ramp detection
3. Add plotsmith integration for visualization
4. Port additional decline models
5. Add multi-phase support

