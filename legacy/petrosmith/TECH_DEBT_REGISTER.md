# Tech Debt Register

| ID | Title | Severity | Area | Why It Matters | Recommended Fix | Effort | Owner |
|----|-------|----------|------|----------------|-----------------|--------|-------|
| TD-001 | CLI bare except swallows errors | Critical | Error Handling | Silent failures hide bugs; user gets no feedback on validate --verbose load failure | Replace with `except Exception as e: logger.debug("Config summary failed: %s", e)` or let propagate | S | Engineering |
| TD-002 | Core uses assert for input validation | Critical | Core/Validation | assert disabled with `-O`; raises AssertionError not ValueError; tests fail | Replace assert with `if not cond: raise ValueError(msg)` or use InvalidMudWeightError/InvalidDepthError | M | Engineering |
| TD-003 | Test expects ValueError, core raises AssertionError | High | Tests | test_drilling validation tests fail | Align with TD-002: core raises proper exception; tests expect it | S | Engineering |
| TD-004 | GeostatsPipeline analyze/validate/visualize stubs | High | Workflows | Config workflow advertised but core steps do nothing | Implement or add explicit "not implemented" error when called | L | Engineering |
| TD-005 | DataConfig.validate_file_exists no-op | High | Config | Users expect file existence check; validator does nothing | Implement `Path(v).resolve().exists()` or remove if template-only | S | Engineering |
| TD-006 | Version mismatch (info shows 2.0.0, pyproject 0.3.0) | High | Ops | Confusing for users and support | Single source: `from petrosmith import __version__` in CLI | S | Engineering |
| TD-007 | Magic number 3.14159 in services | Medium | Services | Inconsistent with math.pi / PhysicalConstants.PI | Replace with `math.pi` or `PhysicalConstants.PI` | S | Engineering |
| TD-008 | Duplicate Darcy/IPR logic in WellService and ReservoirService | Medium | Services | Maintenance burden; risk of divergence | Extract shared function in core or service layer | M | Engineering |
| TD-009 | Extended core not in main export | Medium | Packaging | formation_pressure, rock_mechanics, etc. used by examples but not from petrosmith | Add to petrosmith.__init__ or document petrosmith.core.* submodules | S | Engineering |
| TD-010 | petrosmith.utils unused | Medium | Utils | Dead code; misleading docstrings | Remove or wire into logging setup; fix docstrings | S | Engineering |
| TD-011 | API state inconsistency (DrillingAPI stateless) | Medium | API | Different mental models for different APIs | Document or align: either all stateless or all stateful | S | Engineering |
| TD-012 | Duplicate depth/ID validation in models | Medium | Models | bottom_depth>top_depth, id<od repeated in Reservoir, Casing, Tubing, Perforation | Extract shared validators or mixin | M | Engineering |
| TD-013 | test_petrosmith.py at repo root | Low | Tests | Overlaps tests/; different entry point | Move to tests/ or document as manual verification script | S | Engineering |
| TD-014 | TODOs in docs (module2, module3) | Low | Docs | Incomplete examples | Implement or mark as exercises | M | Docs |
| TD-015 | utils docstrings reference petrosmith.utils.logging | Low | Docs | Wrong import path | Fix to `from petrosmith.utils import setup_logging` | S | Engineering |
| TD-016 | Integration tests: WellControlCalculations/DrillingService API mismatch | High | API/Tests | FIXED | Added missing methods; aligned params | — | — |
| TD-017 | test_bo_at_pressure_saturated assertion fails | High | Core/Tests | FIXED | Corrected assertion: bo_low < bo_high | — | — |

**Severity legend:** Critical = data/security/trust risk; High = bad behavior in common paths; Medium = fix before scale; Low = cleanup.

**Effort:** S = small (<1 day), M = medium (1–2 days), L = large (2+ days).
