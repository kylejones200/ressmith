# Tech Debt Register

| ID | Title | Severity | Area | Impact | Recommended Fix | Effort | Status |
|----|-------|----------|------|--------|-----------------|--------|--------|
| TD-001 | Broken primitives exports | Critical | Exports | ImportError on `from ressmith.primitives import MaterialBalanceDecline` | Add imports from physics_informed, physics_reserves | 0.5h | Done |
| TD-002 | Silent exception handling in reserves | High | Error handling | Callers receive None without explanation; debugging difficult | Log exception, optionally re-raise; document return contract | 0.5h | Done |
| TD-003 | Silent exception handling in uncertainty, physics_informed, advanced_rta, simulator | High | Error handling | Failures invisible; wrong fallback values used silently | Log all exceptions; use specific exception types | 2h | Done (already logged) |
| TD-004 | benchmarks.py ignores fit_forecast exceptions | High | Error handling | Benchmark metrics skewed; failures hidden | Log failed iterations; exclude from timing or report failure count | 0.5h | Done |
| TD-005 | Path traversal in I/O functions | High | Security | User-supplied paths can read arbitrary files | Validate path is under allowed base; use Path.resolve() | 2h | Done |
| TD-006 | Missing input validation in fit_forecast | High | API | Invalid horizon/empty data causes confusing downstream errors | Validate horizon > 0, data non-empty; raise ValueError with clear message | 0.5h | Done (already validated) |
| TD-007 | Missing input validation in evaluate_economics | High | API | Empty forecast causes errors | Validate forecast.yhat non-empty | 0.5h | Done (already validated) |
| TD-008 | No tests for io, simulator | High | Tests | Regressions undetected | Add unit tests for read_csv_production, write_csv_results, simulator functions | 4h | Partial (io done) |
| TD-009 | Duplicated decline fitting logic | Medium | Core logic | Maintenance burden; inconsistent behavior | Extract shared fitting (bounds, fallback) to fitting_utils | 4h | Open |
| TD-010 | Docstrings reference decline_curve | Medium | Docs | Confusing for new users | Find-replace decline_curve → ressmith in docstrings | 1h | Open |
| TD-011 | Van Genuchten unimplemented | Medium | Features | Runtime error or wrong behavior if called | Implement or remove branch; document limitation | 2h | Open |
| TD-012 | Inconsistent error response contract | Medium | API | Callers cannot rely on consistent error handling | Document: None vs empty vs raise; standardize where possible | 2h | Open |
| TD-013 | Swallowed ImportError in tasks/core | Medium | Error handling | timesmith.typing validators skipped silently | Log when optional validators unavailable | 0.25h | Done |
| TD-014 | Swallowed ValueError in economics | Medium | Error handling | IRR fallback to grid search invisible | Log fallback | 0.25h | Done |
| TD-015 | Empty fit optional dependency | Low | Build | Redundant extra | Remove or add actual fit-only deps | 0.25h | Open |
| TD-016 | Workflow runner, catalog, config not exported | Low | API | Internal modules; unclear if public | Document as internal or export if intended public | 0.5h | Open |
| TD-017 | Weak exception handling in tests | Low | Tests | Unexpected exceptions may pass | Use pytest.raises or assert on exception type | 1h | Open |
| TD-018 | Conditional timesmith integration tests | Low | Tests | Tests skipped when timesmith.typing missing | Document in README; consider optional integration test group | 0.5h | Open |
