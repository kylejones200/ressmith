# ResSmith

**ResSmith** is a comprehensive reservoir engineering library with a strict 4-layer architecture, designed to work seamlessly with the Smith ecosystem (plotsmith, anomsmith, geosmith, timesmith). ResSmith provides tools for production analysis, decline curve analysis, forecasting, economic evaluation, and fundamental reservoir engineering calculations.

## Architecture

ResSmith follows a strict 4-layer architecture with enforced one-way imports:

- **Layer 1 (Objects)**: Immutable dataclasses representing the core domain
- **Layer 2 (Primitives)**: Algorithms and base classes
- **Layer 3 (Tasks)**: Task orchestration
- **Layer 4 (Workflows)**: User-facing functions with I/O and plotting

See [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for details.

## Quick Start

### Decline Curve Analysis

```python
import pandas as pd
from ressmith import fit_forecast

# Load production data
data = pd.DataFrame({
    'oil': [100, 95, 90, 85, 80],
}, index=pd.date_range('2020-01-01', periods=5, freq='M'))

# Fit and forecast
forecast, params = fit_forecast(
    data,
    model_name='arps_hyperbolic',
    horizon=24
)

print(f"Forecast: {forecast.yhat.head()}")
print(f"Parameters: {params}")
```

### Reservoir Engineering

```python
from ressmith.primitives import (
    linear_ipr, vogel_ipr, perform_nodal_analysis,
    calculate_pvt_properties, solution_gas_drive_material_balance
)

# IPR calculation
rate = vogel_ipr(
    reservoir_pressure=5000,
    flowing_pressure=3000,
    productivity_index=1.0,
    bubble_point_pressure=3000
)

# PVT properties
pvt = calculate_pvt_properties(
    pressure=5000,
    temperature=200,
    api_gravity=35,
    gas_gravity=0.7
)

# Nodal analysis
result = perform_nodal_analysis(
    reservoir_pressure=5000,
    productivity_index=1.0,
    wellhead_pressure=500,
    tubing_depth=5000
)
```

## Installation

```bash
pip install ressmith
```

Or with optional dependencies:

```bash
pip install ressmith[fit]  # Include scipy for optimization
pip install ressmith[viz]  # Include matplotlib (or use plotsmith)
```

## Features

### Decline Curve Analysis
- **ArpsHyperbolicModel** - Hyperbolic decline (0 < b < 1)
- **ArpsExponentialModel** - Exponential decline (b=0)
- **ArpsHarmonicModel** - Harmonic decline (b=1)
- **LinearDeclineModel** - Simple linear decline
- Multiple advanced decline models (Power Law, Duong, Stretched Exponential, etc.)

### Reservoir Engineering
- **IPR (Inflow Performance Relationship)** - Linear, Vogel, Fetkovich, Composite, Joshi horizontal, Cinco-Ley fractured well models
- **VLP (Vertical Lift Performance)** - Tubing performance, nodal analysis, choke performance, artificial lift optimization
- **RTA (Rate Transient Analysis)** - Flow regime identification, permeability estimation, fracture analysis, SRV calculation
- **Material Balance** - Solution gas drive, water drive, gas cap drive, p/Z method for gas reservoirs
- **PVT Correlations** - Standing, Vasquez-Beggs, Lee-Gonzalez-Eakin, Beggs-Robinson correlations for oil, gas, and water properties

### Core Capabilities
- Fit decline models to production data
- Generate forecasts with configurable horizons
- Evaluate economics (NPV, IRR, cashflows)
- Batch processing for multiple wells
- Portfolio analysis and risk reporting
- Statistical forecasting methods
- Panel data analysis with fixed effects
- Report generation (HTML/PDF)
- Strict type safety with timesmith.typing

## Integration with Smith Ecosystem

### PlotSmith
Use PlotSmith for all visualization:

```python
from ressmith import fit_forecast
from plotsmith import plot_timeseries

forecast, _ = fit_forecast(data, model_name='arps_hyperbolic', horizon=24)
fig, ax = plot_timeseries(forecast.yhat, title='Production Forecast')
```

### AnomSmith
Share typing from timesmith.typing for consistent data structures across libraries.

### GeoSmith
Share typing and integrate spatial analysis for basin-level analysis.

## Reservoir Engineering Modules

ResSmith includes comprehensive reservoir engineering capabilities:

### IPR (Inflow Performance Relationship)
Calculate well deliverability using industry-standard IPR models:
- Linear IPR for single-phase flow
- Vogel IPR for solution gas drive reservoirs
- Fetkovich IPR for two-phase flow with decline
- Composite IPR for layered reservoirs
- Joshi model for horizontal wells
- Cinco-Ley model for fractured wells

### VLP (Vertical Lift Performance)
Analyze well performance and optimize production:
- Tubing performance curves
- Nodal analysis (IPR-VLP intersection)
- Choke performance
- Artificial lift optimization (ESP, gas lift, rod pump)

### RTA (Rate Transient Analysis)
Analyze production data to characterize reservoirs:
- Flow regime identification (linear, bilinear, boundary-dominated, transient)
- Permeability estimation from production data
- Fracture half-length estimation
- Stimulated Reservoir Volume (SRV) calculation

### Material Balance
Analyze reservoir drive mechanisms:
- Solution gas drive (depletion drive)
- Water drive with aquifer influx models (Fetkovich, Carter-Tracy)
- Gas cap drive
- Gas reservoir p/Z method
- Drive mechanism identification

### PVT (Pressure-Volume-Temperature)
Calculate fluid properties using industry correlations:
- Standing and Vasquez-Beggs correlations for oil FVF and solution GOR
- Gas Z-factor (Standing-Katz, Hall-Yarborough)
- Oil and gas viscosity (Beggs-Robinson, Lee-Gonzalez-Eakin)
- Water properties (FVF and viscosity)

## Examples

See `examples/` directory:

- `basic_fit_forecast.py` - Fit model and generate forecast
- `basic_economics.py` - Evaluate economics for a forecast

Run examples:

```bash
python examples/basic_fit_forecast.py
python examples/basic_economics.py
```

## Migration from pydca

This library is being migrated from the `pydca` (decline-curve) repository. See:
- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - Migration strategy
- [MIGRATION_STATUS.md](MIGRATION_STATUS.md) - Current status

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run examples
python examples/basic_fit_forecast.py
```

## Requirements

- Python 3.12+
- numpy >= 1.24.0
- pandas >= 2.0.0
- timesmith >= 0.2.0

## License

MIT License - see LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Projects

- [PlotSmith](https://github.com/kylejones200/plotsmith) - Layered plotting library
- [AnomSmith](https://github.com/kylejones200/anomsmith) - Anomaly detection
- [GeoSmith](https://github.com/kylejones200/geosmith) - Subsurface analysis
- [TimeSmith](https://github.com/kylejones200/timesmith) - Time series types
