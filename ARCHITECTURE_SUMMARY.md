# ResSmith Architecture Summary

## Overview

ResSmith has been restructured with a strict 4-layer architecture enforcing one-way imports:

- **Layer 1 (Objects)**: Immutable dataclasses, no imports from other layers
- **Layer 2 (Primitives)**: Algorithms and base classes, imports only Objects
- **Layer 3 (Tasks)**: Task orchestration, imports Objects and Primitives
- **Layer 4 (Workflows)**: User-facing functions, imports everything plus I/O and plotting

## File Tree

```
ressmith/
├── __init__.py                    # Public API exports
├── objects/                       # Layer 1: Domain objects
│   ├── __init__.py
│   ├── domain.py                  # WellMeta, ProductionSeries, RateSeries, etc.
│   └── validate.py                # Validators: assert_monotonic_time, etc.
├── primitives/                    # Layer 2: Algorithms and interfaces
│   ├── __init__.py
│   ├── base.py                    # BaseObject, BaseEstimator, BaseDeclineModel, BaseEconModel
│   ├── decline.py                 # ARPS primitives and fitting functions
│   ├── models.py                  # Model classes: ArpsHyperbolicModel, LinearDeclineModel, etc.
│   ├── preprocessing.py           # Time series preprocessing functions
│   └── economics.py               # Cashflow, NPV, IRR functions
├── tasks/                         # Layer 3: Task orchestration
│   ├── __init__.py
│   └── core.py                    # FitDeclineTask, ForecastTask, EconTask, BatchTask
└── workflows/                     # Layer 4: User-facing functions
    ├── __init__.py
    ├── core.py                    # fit_forecast, forecast_many, evaluate_economics, full_run
    └── io.py                      # read_csv_production, write_csv_results

tests/
├── test_objects_validate.py      # Validator tests
├── test_primitives_preprocessing.py  # Preprocessing tests
├── test_primitives_decline.py     # Decline primitive tests
├── test_primitives_economics.py   # Economics tests
└── test_workflows.py              # Workflow tests

examples/
├── basic_fit_forecast.py          # Fit and forecast example
└── basic_economics.py             # Economics evaluation example
└── out/                           # Output directory for examples
```

## Public API

The public API is exposed in `ressmith/__init__.py`:

### Workflows (User Entry Points)
- `fit_forecast(data, model_name, horizon, **kwargs)` - Fit model and forecast
- `forecast_many(well_ids, loader, model_name, horizon, **kwargs)` - Batch forecasting
- `evaluate_economics(forecast, spec)` - Evaluate economics
- `full_run(data, model_name, horizon, econ_spec, **kwargs)` - Full workflow

### Base Types
- `BaseDeclineModel` - Base class for decline models
- `BaseEconModel` - Base class for economics models

### Core Objects
- `WellMeta` - Well metadata
- `ProductionSeries` - Multi-phase production data
- `RateSeries` - Single-phase rate data
- `CumSeries` - Cumulative production data
- `DeclineSpec` - Decline model specification
- `ForecastSpec` - Forecast specification
- `EconSpec` - Economics specification
- `ForecastResult` - Forecast results
- `EconResult` - Economics results

## Implemented Models

### 1. ArpsHyperbolicModel
- **Type**: Hyperbolic decline (0 < b < 1)
- **Status**: ✅ Fully implemented end-to-end
- **Features**: Fit and predict with ARPS hyperbolic equation
- **Usage**: `fit_forecast(data, model_name="arps_hyperbolic")`

### 2. LinearDeclineModel
- **Type**: Simple linear decline (empirical)
- **Status**: ✅ Fully implemented end-to-end
- **Features**: Linear regression fit, supports irregular time
- **Usage**: `fit_forecast(data, model_name="linear_decline")`

### Additional Models Available
- `ArpsExponentialModel` - Exponential decline (b=0)
- `ArpsHarmonicModel` - Harmonic decline (b=1)

## Dependencies

### Core Dependencies
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `timesmith>=0.2.0` - For typing protocols (SeriesLike, PanelLike)

### Optional Dependencies
- `scipy>=1.10.0` (extra: `fit`) - For optimization-based fitting
- `matplotlib>=3.7.0` (extra: `viz`) - For plotting (not yet used)

## Testing

All tests are in `tests/` directory:
- ✅ Validator tests (monotonic time, positive rates, alignment)
- ✅ Preprocessing tests (cum/rate round-trip)
- ✅ Decline primitive tests (ARPS functions, fitting)
- ✅ Economics tests (cashflow, NPV)
- ✅ Workflow tests (fit_forecast returns pandas outputs)

Run tests with: `pytest tests/`

## Examples

### basic_fit_forecast.py
Generates synthetic hyperbolic decline, fits ArpsHyperbolicModel, forecasts 24 months, writes to `examples/out/forecast.csv`

Run: `python examples/basic_fit_forecast.py`

### basic_economics.py
Reads forecast, evaluates economics with EconSpec, writes cashflows to `examples/out/cashflows.csv`

Run: `python examples/basic_economics.py`

## Next 10 Migrations (Priority Order)

1. **Segmented Decline Model** - Multi-segment ARPS with change points
2. **Power Law Decline Model** - Alternative empirical decline curve
3. **Duong Model** - Fracture-dominated decline model
4. **Stretched Exponential Model** - Weibull-type decline
5. **Multi-phase ProductionSeries Support** - Full oil/gas/water handling in models
6. **Prediction Intervals** - Uncertainty quantification for forecasts
7. **Censoring Support** - Handle zero/negative rates in fitting
8. **Irregular Time Support** - Better handling of missing/irregular data
9. **Unit Conversion System** - Comprehensive unit handling
10. **Plotting Workflows** - Visualization functions in workflows layer

## Architecture Compliance

### Import Rules
- ✅ Objects layer: Only stdlib, numpy, pandas, timesmith.typing
- ✅ Primitives layer: Only imports Objects
- ✅ Tasks layer: Only imports Objects and Primitives
- ✅ Workflows layer: Can import everything plus I/O/plotting

### Validation
- ✅ All user inputs converted to Layer 1 objects at task entry
- ✅ Validation via Layer 1 validators only
- ✅ No validation inside tight loops

### Model Implementation
- ✅ All models inherit from BaseDeclineModel
- ✅ Models implement fit() and predict() methods
- ✅ Models have tags describing capabilities
- ✅ Two models fully migrated end-to-end (ArpsHyperbolic, LinearDecline)

## Acceptance Criteria Status

- ✅ pytest passes
- ✅ `python examples/basic_fit_forecast.py` runs and writes forecast.csv
- ✅ `python examples/basic_economics.py` runs and writes cashflows.csv
- ✅ Repo tree matches four layers
- ✅ Two models run end-to-end through workflows
- ✅ Public API defined in `__init__.py`

