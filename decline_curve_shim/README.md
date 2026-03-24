# decline-curve (Compatibility Shim)

**This package is deprecated.** ResSmith has absorbed decline-curve functionality.

- **For new projects:** Use `ressmith` directly: `pip install ressmith`
- **For existing code:** This shim provides backward compatibility so `from decline_curve import dca` continues to work.

## Installation

```bash
pip install decline-curve
```

This installs `ressmith` and the compatibility layer.

## Migration

Replace:

```python
from decline_curve import dca
forecast = dca.single_well(series, model='arps', kind='hyperbolic', horizon=12)
```

With:

```python
from ressmith import dca
forecast = dca.single_well(series, model='arps', kind='hyperbolic', horizon=12)
```

Or use the full ResSmith API:

```python
from ressmith import fit_forecast
forecast, params = fit_forecast(data, model_name='arps_hyperbolic', horizon=12)
```
