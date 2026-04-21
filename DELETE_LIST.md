# Delete List

Dead files, dead modules, unused code, duplicate logic.

## Safe to Delete (or Remove)

### None

No files identified as fully dead. All modules are imported or used somewhere.

## Documented for Removal (Plan)

### 1. Duplicate Logic — Extract then Delete

| Location | Description | Action |
|----------|-------------|--------|
| `primitives/decline.py` + `primitives/advanced_decline.py` | Shared scipy optimize + grid-search fallback | **Done** — shared helpers in `primitives/fitting_utils.py`; keep pruning only if new duplication appears |

### 2. Unused / Orphaned — Verify Before Deleting

| Item | Location | Status |
|------|----------|--------|
| `workflows/benchmarks.py` | Not in main `ressmith` exports | Used for benchmarking; keep but document as dev tool |
| `workflows/profiling.py` | Optional (line_profiler) | Keep; used for performance profiling |
| `workflows/runner.py` | Not in workflows `__init__` | Verify usage; if CLI or internal only, document |
| `workflows/config.py` | Not in workflows `__init__` | Verify usage; document if internal |
| `workflows/catalog.py` | Not in workflows `__init__` | Verify usage; document if internal |

### 3. Placeholder / Stub Code — Fix or Remove

| Location | Code | Action |
|----------|------|--------|
| `primitives/relative_permeability.py` | van Genuchten capillary branch | **Done** — implemented + tests (`test_relative_permeability_van_genuchten.py`) |
| `primitives/base.py` L112 | `BaseDeclineModel.predict`: `pass` | Expected abstract method; subclasses override; **no change** |

### 4. Unused Imports

Run `ruff check --select F401 ressmith` to find unused imports. Address as part of routine cleanup.

### 5. Dead Code Within Files

- No functions identified as completely unreachable.
- `workflows/leakage_check.py`: date-window branch now **logs at debug** instead of silent `pass`.

---

## Removal Plan

1. **Phase 1:** Van Genuchten — **complete** (see TECH_DEBT_REGISTER TD-011).
2. **Phase 2:** Shared fitting utilities — **complete** (fitting_utils).
3. **Phase 3:** Run ruff F401 periodically; remove unused imports when reported.
4. **Phase 4:** Document runner, config, catalog as internal or export if intended (README / capability map).
