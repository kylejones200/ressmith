# pydca / decline-curve Merge into ResSmith

ResSmith is the primary repo. The decline-curve (pydca) functionality has been merged into ResSmith. You can remove pydca and use ResSmith instead.

## What Was Merged

- **DCA API** (`ressmith.dca`): `single_well` and `forecast` map to ResSmith workflows
- **Arps models**: exponential, harmonic, hyperbolic via `ressmith.workflows.core.fit_forecast`
- **PetroSmith** uses ResSmith for DCA instead of decline-curve

## Migration

### From decline-curve

```python
# Before (decline-curve)
from decline_curve import dca
forecast = dca.single_well(series, model='arps', kind='hyperbolic', horizon=12)
```

```python
# After (ResSmith)
from ressmith import dca
forecast = dca.single_well(series, model='arps', kind='hyperbolic', horizon=12)
```

API is compatible. Same arguments and return types.

### Backward-compat shim (optional)

A thin `decline-curve` shim lives in `decline_curve_shim/`. Install it so existing `from decline_curve import dca` keeps working:

```bash
cd decline_curve_shim && pip install -e .
```

The shim re-exports `from ressmith import dca` under the `decline_curve` package name.

### PetroSmith

PetroSmith now depends on `ressmith` instead of `decline-curve`. Update:

```bash
pip uninstall decline-curve
pip install ressmith
```

## Optional Extras (from pydca)

ResSmith has optional dependency groups for future DCA/ML features:

| Extra   | Purpose                     | Key deps            |
|---------|-----------------------------|---------------------|
| `stats` | ARIMA, statistical models   | statsmodels         |
| `spatial` | Kriging, spatial analysis | pykrige, scikit-learn |
| `llm`   | Chronos, TimesFM           | transformers, torch |
| `ml`    | DeepAR, TFT, ML models     | transformers, torch |

```bash
pip install ressmith[stats]   # ARIMA, etc.
pip install ressmith[spatial] # Kriging
pip install ressmith[llm]     # Chronos, TimesFM
pip install ressmith[ml]      # DeepAR, TFT
```

## Model Support

| Model   | Status                                      |
|---------|---------------------------------------------|
| arps    | ✅ Supported (exponential, harmonic, hyperbolic) |
| arima   | ⚠️ Falls back to arps (stats extra for future) |
| timesfm | ⚠️ Falls back to arps (llm extra for future)   |
| chronos | ⚠️ Falls back to arps (llm extra for future)   |

## pydca Modules Not Yet Migrated

The following pydca modules were not copied into ResSmith. The core DCA API is in ResSmith; these can be ported later if needed:

- `forecast_arima`, `forecast_chronos`, `forecast_deepar`, `forecast_timesfm`, `forecast_tft`
- `spatial_kriging`
- `data_contract`, `data_qa`
- Various ML forecasters

## Removing pydca

After migrating:

1. Update `pyproject.toml` / `requirements.txt`: replace `decline-curve` with `ressmith`
2. Change imports: `from decline_curve import dca` → `from ressmith import dca`
3. Uninstall: `pip uninstall decline-curve`
4. Archive or delete the pydca repo

## Key Paths

- ResSmith DCA: `ressmith/dca.py`
- Fit/forecast: `ressmith/workflows/core.py`
- Shim: `decline_curve_shim/`
- PetroSmith integration: `petrosmith/integrations/decline_curve.py`
