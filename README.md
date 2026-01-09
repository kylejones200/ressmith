# ResSmith

**ResSmith** is a reservoir decline curve analysis and forecasting library with a strict 4-layer architecture, designed to work seamlessly with the Smith ecosystem (plotsmith, anomsmith, geosmith).

## Architecture

ResSmith follows a strict 4-layer architecture with enforced one-way imports:

- **Layer 1 (Objects)**: Immutable dataclasses representing the core domain
- **Layer 2 (Primitives)**: Algorithms and base classes
- **Layer 3 (Tasks)**: Task orchestration
- **Layer 4 (Workflows)**: User-facing functions with I/O and plotting

See [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for details.

## Quick Start

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

### Current Models
- **ArpsHyperbolicModel** - Hyperbolic decline (0 < b < 1)
- **ArpsExponentialModel** - Exponential decline (b=0)
- **ArpsHarmonicModel** - Harmonic decline (b=1)
- **LinearDeclineModel** - Simple linear decline

### Core Capabilities
- Fit decline models to production data
- Generate forecasts with configurable horizons
- Evaluate economics (NPV, IRR, cashflows)
- Batch processing for multiple wells
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
