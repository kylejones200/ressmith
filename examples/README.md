# ResSmith Workflow Demos

This directory contains comprehensive workflow demonstrations for ResSmith.

## Available Demos

### 1. `demo_single_well_complete.py`
**Complete Single Well Analysis Workflow**

Demonstrates a full single-well analysis workflow:
- Loading production data
- Comparing multiple decline models
- Probabilistic forecasting (P10/P50/P90)
- Economics evaluation
- Scenario analysis
- Sensitivity analysis
- Walk-forward backtesting
- Diagnostic plots

**Run:**
```bash
python examples/demo_single_well_complete.py
```

### 2. `demo_portfolio_analysis.py`
**Complete Portfolio Analysis Workflow**

Demonstrates portfolio-level workflows:
- Multi-well data loading
- Portfolio forecasting
- Well ranking
- Portfolio aggregation
- Multi-well interaction analysis
- Field spacing optimization
- Portfolio statistics

**Run:**
```bash
python examples/demo_portfolio_analysis.py
```

### 3. `demo_advanced_workflows.py`
**Advanced Reservoir Engineering Workflows**

Demonstrates advanced workflows:
- Type curve matching
- Blasingame RTA analysis
- Flowing Material Balance (FMB)
- Fracture network analysis
- Multi-phase forecasting with yields
- Well coning analysis
- EOR waterflood pattern analysis

**Run:**
```bash
python examples/demo_advanced_workflows.py
```

### 4. `demo_ensemble_probabilistic.py`
**Ensemble & Probabilistic Forecasting**

Demonstrates uncertainty quantification:
- Ensemble forecasting (multiple models)
- Probabilistic forecasting (P10/P50/P90)
- Confidence intervals
- Risk metrics (VaR, probability of success)
- Scenario evaluation with uncertainty

**Run:**
```bash
python examples/demo_ensemble_probabilistic.py
```

### 5. `demo_complete_field_analysis.py`
**Complete Field Analysis Workflow**

Comprehensive demo tying together all major workflows:
- Multi-well portfolio analysis
- Ensemble and probabilistic forecasting
- Multi-well interaction and spacing optimization
- Advanced RTA and type curve matching
- Multi-phase forecasting
- EOR pattern analysis
- Production operations
- Economic evaluation with scenarios
- Risk analysis

**Run:**
```bash
python examples/demo_complete_field_analysis.py
```

## Basic Examples

### 5. `basic_fit_forecast.py`
Simple fit and forecast example.

### 6. `basic_economics.py`
Basic economics evaluation example.

### 7. `basic_segmented.py`
Segmented decline model example.

### 8. `integration_timesmith.py`
Integration with Timesmith library.

## Output

All demos generate output to `examples/out/` directory:
- Forecast CSV files
- Cashflow CSV files
- Diagnostic plots (if plotting enabled)

## Requirements

All demos require:
- `pandas`
- `numpy`
- `ressmith` package

Some advanced demos may require:
- `scipy` (for optimization)
- `matplotlib` or `plotsmith` (for visualization)

## Running All Demos

To run all demos sequentially:

```bash
cd examples
python demo_single_well_complete.py
python demo_portfolio_analysis.py
python demo_advanced_workflows.py
python demo_ensemble_probabilistic.py
```

## Customization

All demos use synthetic data generators. To use your own data:

1. Replace the `generate_synthetic_data()` functions with your data loading code
2. Ensure your DataFrame has a DatetimeIndex and appropriate column names ('oil', 'gas', 'water', etc.)
3. Adjust model parameters and economic assumptions as needed

## Notes

- All demos use random seeds for reproducibility
- Synthetic data is generated for demonstration purposes
- Some advanced workflows may require specific data formats (e.g., pressure data for FMB analysis)
- Error handling is included to demonstrate graceful degradation when data requirements aren't met

