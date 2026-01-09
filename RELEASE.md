# Release v0.1.0

## Release Summary

This release includes major features from Phases 3, 4, and 5 of the migration plan:

### New Features

1. **Parameter Constraints and Bounds** (Phase 4)
   - New `ressmith.primitives.constraints` module
   - Parameter validation and bounds checking for all models
   - Automatic parameter clipping to enforce bounds

2. **Advanced Decline Models** (Phase 3)
   - `PowerLawDeclineModel`: Power law decline for unconventional reservoirs
   - `DuongModel`: Duong decline model for shale/tight formations
   - `StretchedExponentialModel`: Stretched exponential decline

3. **Multi-phase Support** (Phase 5)
   - Enhanced `cashflow_from_forecast()` for multi-phase revenue
   - Full support for oil/gas/water phases in economics

4. **Fit Diagnostics** (Phase 4)
   - Comprehensive fit quality assessment
   - RMSE, MAE, MAPE, R² metrics
   - Quality flags and warnings

### Changes

- All ARPS models now validate and clip parameters after fitting
- Enhanced economics module supports multi-phase price assumptions
- All models integrated into workflows

### Testing

- 7 new tests for advanced models
- All existing tests passing
- Comprehensive diagnostics tests

## Building and Publishing to PyPI

### Prerequisites

```bash
pip install build twine
```

### Build Distribution

```bash
python -m build
```

This creates:
- `dist/ressmith-0.1.0-py3-none-any.whl` (wheel)
- `dist/ressmith-0.1.0.tar.gz` (source distribution)

### Test Upload (TestPyPI)

```bash
twine upload --repository testpypi dist/*
```

### Production Upload (PyPI)

```bash
twine upload dist/*
```

You'll need PyPI credentials. The package is configured in `pyproject.toml`.

## Version History

- **v0.1.0** (2024-01-09): Advanced models, parameter constraints, multi-phase support
- **v0.0.1** (2024-01-08): Initial release with core ARPS models

