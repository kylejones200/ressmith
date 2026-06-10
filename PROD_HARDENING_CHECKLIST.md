# Production Hardening Checklist

Concrete, verifiable items. Check off before release.

---

## Critical Paths

- [ ] **Drilling validation** — `pytest tests/test_drilling.py` passes (no assert/ValueError mismatch)
- [ ] **Well control** — `pytest tests/test_well_control.py` passes
- [ ] **Formation pressure** — `pytest tests/test_formation_pressure.py` passes
- [ ] **Reservoir** — `pytest tests/test_reservoir.py` passes
- [ ] **Integration** — `pytest tests/test_integration.py` passes
- [ ] **No bare except** — `grep -r "except:" petrosmith/ --include="*.py"` returns no matches
- [ ] **No assert for validation** — Core modules use explicit `raise` for invalid inputs

---

## Error Handling

- [ ] All `except` clauses specify exception type(s)
- [ ] No `except: pass` or equivalent silent swallow
- [ ] Custom exceptions used where appropriate (InvalidMudWeightError, etc.)
- [ ] CLI exits with non-zero on failure

---

## Validation

- [ ] API boundaries validate inputs before calling core
- [ ] Pydantic models used for config and data
- [ ] DataConfig.validate_file_exists either implemented or removed

---

## Security

- [ ] `yaml.safe_load` used (no `yaml.load`)
- [ ] No `eval()` or `exec()` of user input
- [ ] No hardcoded secrets or credentials
- [ ] `bandit -r petrosmith/` passes (or known issues documented)

---

## Consistency

- [ ] Single version source: `petrosmith.__version__` matches pyproject and CLI
- [ ] Constants: use `math.pi` or `PhysicalConstants.PI`, not literal 3.14159
- [ ] Logging: consistent format and levels

---

## Tests

- [ ] `pytest tests/ -v` passes
- [ ] `pytest tests/ --cov=petrosmith --cov-report=term-missing` shows coverage for core modules
- [ ] Critical paths: drilling, well control, formation pressure, reservoir have >80% coverage

---

## Documentation

- [ ] README matches actual usage (imports, commands)
- [ ] API docs build: `cd docs && make html` (or equivalent)
- [ ] No misleading "TODO: Implement" in user-facing docs without tracking

---

## Dependencies

- [ ] `uv sync` or `pip install -e .` succeeds
- [ ] `safety check` (or equivalent) shows no high/critical vulns
- [ ] Lockfile committed (uv.lock)

---

## CLI

- [ ] `petrosmith --version` works
- [ ] `petrosmith init --output /tmp/test.yaml` creates valid template
- [ ] `petrosmith validate /tmp/test.yaml` exits 0 or 1 with clear message
- [ ] `petrosmith info` shows correct version

---

## Config Workflow

- [ ] If GeostatsPipeline stubs used: either implemented or explicit "not implemented" error
- [ ] Example configs in examples/configs/ validate successfully
- [ ] Config schema matches templates

---

## Sign-Off

| Role | Name | Date |
|------|------|------|
| Engineer | | |
| Reviewer | | |
