# ResSmith

**ResSmith** is a comprehensive reservoir engineering library with a strict 4-layer architecture, designed to work seamlessly with the Smith ecosystem (plotsmith, anomsmith, geosmith, timesmith). ResSmith provides tools for production analysis, decline curve analysis, forecasting, economic evaluation, and fundamental reservoir engineering calculations.

## Architecture

ResSmith follows a strict 4-layer architecture with enforced one-way imports:

- **Layer 1 (Objects)**: Immutable dataclasses representing the core domain
- **Layer 2 (Primitives)**: Algorithms and base classes
- **Layer 3 (Tasks)**: Task orchestration
- **Layer 4 (Workflows)**: User-facing functions with I/O and plotting

See [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) for details.

## Documentation for reservoir engineers

- [Model selection](docs/model_selection_guide.md) — when to use ARPS vs Power Law vs Duong, typical pitfalls
- [Capability map](docs/engineer_capability_map.md) — **where to find** interference, coning, RTA/diagnostics, EOR, rel perm, allocation, well-test checks, simulator hooks
- [Advanced workflows](docs/advanced_workflows.md) — multi-well, history matching, optimization
- [Best practices](docs/best_practices.md) — workflow discipline

## Quick Start

### Decline Curve Analysis

```python
import pandas as pd
from ressmith import fit_forecast

# Load production data
data = pd.DataFrame({
    'oil': [100, 95, 90, 85, 80],
}, index=pd.date_range('2020-01-01', periods=5, freq='ME'))

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

For end users installing from PyPI:

```bash
uv pip install ressmith
```

Or with optional dependencies:

```bash
uv pip install ressmith[viz]  # Include matplotlib (or use plotsmith)
```

SciPy is a core dependency; optimization does not require an extra install group.

For development, clone the repository and use uv:

```bash
git clone https://github.com/kylejones200/ressmith.git
cd ressmith
uv sync --group dev
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

- `basic_fit_forecast.py` — Model compare, walk-forward, ensemble, probabilistic bands, pressure normalization, diagnostics, forecast
- `basic_economics.py` — Base economics plus discount / capex / price scenarios
- `basic_segmented.py` — Segmented decline, model compare, walk-forward vs single ARPS
- `integrated_studies.py` — Well interference, coning, and enhanced RTA bundled workflows (`examples/README.md` lists all demos)

Run examples:

```bash
uv run python examples/basic_fit_forecast.py
uv run python examples/basic_economics.py
uv run python examples/basic_segmented.py
uv run python examples/integrated_studies.py
```

## Migration from pydca

The decline-curve (`pydca`) codebase has been merged into ResSmith. See [MERGE_MIGRATION.md](MERGE_MIGRATION.md) for import changes, the optional `decline_curve_shim`, and when you can remove a local `pydca` clone.

## Development

**Public surface:** Prefer imports from the top-level `ressmith` package or documented subpackages (`ressmith.primitives`, `ressmith.objects`). Modules such as workflow runners, catalog, and config under `ressmith.workflows` are internal unless re-exported from `ressmith` or named in API docs.

**Errors:** Invalid inputs typically raise `ValueError` with a short message. Some workflow helpers return `None` on recoverable failure while logging the exception; check each function’s docstring for its contract.

**Tests:** Integration tests that require `timesmith.typing` validators are skipped when that stack is unavailable (`pytest` markers / `skipif`). Use `uv run pytest -m "not integration"` to exclude them.

```bash
# Install dependencies (including dev group)
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Format code
uv run black ressmith tests

# Run examples
uv run python examples/basic_fit_forecast.py
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
