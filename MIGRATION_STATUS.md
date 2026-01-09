# ResSmith Migration Status

## Current State

### ✅ Completed Architecture
- **4-Layer Structure**: Strictly enforced with one-way imports
- **Layer 1 (Objects)**: Domain dataclasses and validators
- **Layer 2 (Primitives)**: Base classes, ARPS algorithms, economics, preprocessing
- **Layer 3 (Tasks)**: FitDeclineTask, ForecastTask, EconTask, BatchTask
- **Layer 4 (Workflows)**: User-facing functions (fit_forecast, forecast_many, etc.)

### ✅ Migrated Models
1. **ArpsHyperbolicModel** - Fully functional end-to-end
2. **ArpsExponentialModel** - Fully functional end-to-end
3. **ArpsHarmonicModel** - Fully functional end-to-end
4. **LinearDeclineModel** - Simple alternative approach

### ✅ Infrastructure
- Tests for validators, primitives, economics, workflows
- Example scripts (basic_fit_forecast.py, basic_economics.py)
- Public API defined in `__init__.py`
- Dependencies configured (timesmith, numpy, pandas, optional scipy)

## Integration with Smith Ecosystem

### Timesmith (✅ Configured)
- Dependency: `timesmith>=0.2.0`
- Uses: `timesmith.typing.SeriesLike` for type hints
- Status: Ready for shared typing across smith libraries

### PlotSmith (📋 Planned)
- **Integration Point**: Layer 4 (Workflows) can import plotsmith
- **Use Cases**:
  - `plotsmith.workflows.plot_timeseries()` for forecast visualization
  - `plotsmith.workflows.plot_residuals()` for fit diagnostics
  - `plotsmith.workflows.plot_backtest()` for backtest results
- **Status**: Architecture ready, integration code needed

### AnomSmith (📋 Planned)
- **Integration Point**: Share `timesmith.typing` types
- **Use Cases**:
  - Outlier detection in production data (pre-processing)
  - Anomaly detection in forecast residuals
- **Status**: Architecture ready, integration code needed

### GeoSmith (📋 Planned)
- **Integration Point**: Share `timesmith.typing` types
- **Use Cases**:
  - Spatial kriging for EUR estimation
  - Geospatial well analysis
  - Basin-level aggregation
- **Status**: Architecture ready, integration code needed

## Migration from pydca

### High Priority Items

#### 1. Segmented Decline Model
**Source**: `pydca/decline_curve/segmented_decline.py`
**Target**: 
- Layer 1: `DeclineSegment`, `SegmentedDeclineResult` objects
- Layer 2: Segmented decline primitives and `SegmentedDeclineModel` class
- Layer 3: `SegmentedFitTask`
- Layer 4: `fit_segmented_forecast()` workflow

**Status**: 📋 Ready to migrate

#### 2. Enhanced Fitting Logic
**Source**: `pydca/decline_curve/models_arps.py` (initial_guess methods)
**Target**: Enhance `fit_arps_*` functions in `ressmith/primitives/decline.py`
**Improvements**:
- Ramp-aware initial guess (max rate in first 30% of data)
- Better edge case handling (b=0, b=1)
- Parameter validation and bounds checking

**Status**: 📋 Ready to enhance

#### 3. Hyperbolic-to-Exponential Switch Model
**Source**: `pydca/decline_curve/models_arps.py`
**Target**: New model class in `ressmith/primitives/models.py`
**Status**: 📋 Ready to migrate

### Medium Priority Items

#### 4. Multi-phase Support
- Full oil/gas/water handling in models
- Phase-specific decline models
- Multi-phase economics

#### 5. Advanced Models
- PowerLawDeclineModel
- DuongModel
- StretchedExponentialModel

#### 6. Plotting Integration
- Add plotsmith workflows to ressmith workflows
- Create visualization helpers

### Low Priority Items

#### 7. Deep Learning Models
- ForecastChronos, ForecastTFT, ForecastDeepAR
- These may belong in a separate ML layer or integration

#### 8. Spatial Analysis
- Kriging for EUR estimation
- Panel analysis

## Next Steps

### Immediate (This Session)
1. ✅ Architecture skeleton complete
2. ✅ Two models working end-to-end
3. ✅ Tests and examples in place

### Short Term (Next Session)
1. Migrate SegmentedDeclineModel
2. Enhance fitting with ramp detection
3. Add plotsmith integration example

### Medium Term
1. Migrate additional decline models
2. Add multi-phase support
3. Integrate with anomsmith for outlier detection

### Long Term
1. Full pydca feature parity
2. Deep learning model integration
3. Spatial analysis integration with geosmith

## File Structure

```
ressmith/
├── objects/              # Layer 1: Domain objects
│   ├── domain.py        # Core dataclasses
│   └── validate.py      # Validators
├── primitives/          # Layer 2: Algorithms
│   ├── base.py          # Base classes
│   ├── decline.py       # ARPS primitives
│   ├── models.py        # Model classes
│   ├── preprocessing.py # Data preprocessing
│   └── economics.py     # Economics primitives
├── tasks/               # Layer 3: Task orchestration
│   └── core.py          # Task classes
└── workflows/           # Layer 4: User-facing
    ├── core.py          # Main workflows
    └── io.py            # I/O helpers
```

## Dependencies

### Core
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `timesmith>=0.2.0` (for shared typing)

### Optional
- `scipy>=1.10.0` (extra: `fit`) - For optimization-based fitting
- `matplotlib>=3.7.0` (extra: `viz`) - For plotting (or use plotsmith)

### Future Integrations
- `plotsmith` - For all plotting
- `anomsmith` - For anomaly detection
- `geosmith` - For spatial analysis

## Testing Status

- ✅ Validator tests
- ✅ Preprocessing tests (cum/rate round-trip)
- ✅ Decline primitive tests
- ✅ Economics tests
- ✅ Workflow tests
- 📋 Segmented model tests (when migrated)
- 📋 Integration tests with plotsmith

## Examples Status

- ✅ `basic_fit_forecast.py` - Fit and forecast example
- ✅ `basic_economics.py` - Economics evaluation example
- 📋 `segmented_example.py` - Segmented decline example (when migrated)
- 📋 `plotsmith_integration.py` - Plotting integration example

## Acceptance Criteria

- ✅ pytest passes
- ✅ Examples run and write output files
- ✅ Four-layer architecture enforced
- ✅ Two models working end-to-end
- ✅ Public API defined
- 📋 Segmented model migrated
- 📋 Plotsmith integration working

