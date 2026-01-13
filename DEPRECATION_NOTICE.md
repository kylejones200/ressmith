# pydca Deprecation Notice

## ⚠️ pydca is Deprecated

**pydca** (decline curve analysis) has been **deprecated** and all functionality has been migrated to **resmith** (reservoir engineering smith).

## Migration to ressmith

All core functionality from `pydca` has been migrated to `ressmith`, which provides:

- ✅ **Comprehensive reservoir engineering** capabilities (not just decline curve analysis)
- ✅ **Strict 4-layer architecture** with clean boundaries
- ✅ **Integration with Smith ecosystem** (plotsmith, timesmith, geosmith, anomsmith)
- ✅ **Modern type hints** and improved code quality
- ✅ **All pydca features** plus additional reservoir engineering tools

## Quick Migration Guide

### Installation

```bash
# Old (deprecated)
pip install pydca

# New (recommended)
pip install ressmith
```

### Import Changes

```python
# Old (deprecated)
from decline_curve import fit_forecast, estimate_eur
from decline_curve.models import ArpsHyperbolicModel
from decline_curve.economics import economic_metrics

# New (recommended)
from ressmith import fit_forecast, estimate_eur
from ressmith.primitives.models import ArpsHyperbolicModel
from ressmith.primitives.economics import npv, irr
```

### Function Mapping

| pydca | ressmith | Notes |
|-------|----------|-------|
| `decline_curve.fit_forecast()` | `ressmith.fit_forecast()` | Same API |
| `decline_curve.estimate_eur()` | `ressmith.estimate_eur()` | Same API |
| `decline_curve.models.ArpsHyperbolicModel` | `ressmith.primitives.models.ArpsHyperbolicModel` | Same API |
| `decline_curve.economics.economic_metrics()` | `ressmith.workflows.core.evaluate_economics()` | Returns `EconResult` |
| `decline_curve.risk_report.calculate_risk_metrics()` | `ressmith.workflows.risk.calculate_risk_metrics()` | Same API |
| `decline_curve.panel_analysis.*` | `ressmith.workflows.panel_analysis.*` | Same functions |
| `decline_curve.reports.generate_well_report()` | `ressmith.workflows.reports.generate_well_report()` | Same API |
| `decline_curve.cli` | `ressmith.workflows.cli` | CLI: `ressmith` command |

### New Features in ressmith

ResSmith includes all pydca functionality **plus**:

- **Reservoir Engineering Modules:**
  - IPR (Inflow Performance Relationship)
  - VLP (Vertical Lift Performance)
  - RTA (Rate Transient Analysis)
  - Material Balance
  - PVT Correlations
  - Well Test Analysis

- **Enhanced Utilities:**
  - Unit conversion system
  - Calendar logic for monthly data
  - Downtime analysis
  - Sensitivity analysis
  - Batch processing with manifests

- **Better Integration:**
  - plotsmith for visualization
  - geosmith for spatial analysis
  - anomsmith for outlier detection
  - timesmith for time series validation

## Migration Checklist

- [ ] Install ressmith: `pip install ressmith`
- [ ] Update imports from `decline_curve` to `ressmith`
- [ ] Update economics calls to use `EconResult` instead of dict
- [ ] Test workflows with ressmith
- [ ] Update documentation and examples
- [ ] Remove pydca dependency

## Timeline

- **Deprecation Date:** 2024
- **End of Support:** TBD (pydca will remain available but not actively maintained)
- **Migration Deadline:** Recommended to migrate as soon as possible

## Questions?

- See [PDYCA_MIGRATION_ANALYSIS.md](PDYCA_MIGRATION_ANALYSIS.md) for detailed migration analysis
- See [README.md](README.md) for ressmith documentation
- Open an issue if you encounter migration problems

## Why ressmith?

1. **Broader Scope:** Reservoir engineering library, not just decline curve analysis
2. **Better Architecture:** Strict 4-layer architecture with enforced boundaries
3. **Ecosystem Integration:** Works seamlessly with other Smith libraries
4. **Modern Codebase:** Updated type hints, better error handling, improved documentation
5. **Active Development:** All new features will be in ressmith, not pydca

---

**Note:** This deprecation notice will be added to the pydca repository to guide users to ressmith.

