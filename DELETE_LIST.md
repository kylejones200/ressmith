# Delete List

Dead files, dead modules, unused code, duplicate logic.

## Safe to Delete (or Remove)

### None

No files identified as fully dead. All modules are imported or used somewhere.

## Documented for Removal (Plan)

### 1. Duplicate Logic — Extract then Delete

| Location | Description | Action |
|----------|-------------|--------|
| `primitives/decline.py` + `primitives/advanced_decline.py` | Shared scipy optimize + grid-search fallback | Extract to `fitting_utils.py` or `_fitting_common.py`; refactor both to use it; remove duplicated blocks |

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
| `primitives/relative_permeability.py` L339 | `method == "van_genuchten"` branch: `pass` | Implement Van Genuchten or remove branch and raise `NotImplementedError` with clear message |
| `primitives/base.py` L112 | `BaseDeclineModel.predict`: `pass` | Expected abstract method; subclasses override; **no change** |

### 4. Unused Imports

Run `ruff check --select F401 ressmith` to find unused imports. Address as part of routine cleanup.

### 5. Dead Code Within Files

- No functions identified as completely unreachable.
- `workflows/leakage_check.py` L168: `pass` in exception handler — review; replace with logging or re-raise.

---

## Removal Plan

1. **Phase 1:** Fix Van Genuchten — implement or remove (see TD-011).
2. **Phase 2:** Extract fitting logic; remove duplicates (see TD-009).
3. **Phase 3:** Run ruff F401; remove unused imports.
4. **Phase 4:** Document runner, config, catalog as internal or export if intended.
