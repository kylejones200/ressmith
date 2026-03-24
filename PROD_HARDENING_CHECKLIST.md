# Production Hardening Checklist

Concrete, verifiable items. Check off before deployment.

**Use `uv` for all package management** (not pip).

## Security

- [x] **SEC-1** File existence validated before read: `read_csv_production`, `load_production_csvs`, `load_price_csv`, `import_simulator_output`, `import_simulation_results` raise FileNotFoundError if path does not exist.
- [x] **SEC-2** No hardcoded secrets: `rg -i "password|api_key|secret|token" ressmith/` returns no matches in application code.
- [x] **SEC-3** CLI path inputs normalized with `Path.resolve()` and validated via `_resolve_input_path()`.

## Error Handling

- [x] **ERR-1** Broad `except Exception` handlers now log before returning/fallback.
- [x] **ERR-2** `reserves.py` `calculate_eur_from_params`: log exception before returning None.
- [x] **ERR-3** `uncertainty.py`, `physics_informed.py`, `advanced_rta.py`, `simulator.py`: log before fallback.
- [x] **ERR-4** `benchmarks.py` `benchmark_single_well`: log failures; return failed_iterations.
- [x] **ERR-5** `economics.py` IRR: log when falling back to grid search.

## Validation

- [x] **VAL-1** `fit_forecast`: `horizon` must be positive int; `data` must be non-empty.
- [x] **VAL-2** `evaluate_economics`: `forecast.yhat` non-empty.
- [x] **VAL-3** `walk_forward_backtest`: `forecast_horizons`, `min_train_size`, `step_size` validated.
- [x] **VAL-4** `read_csv_production`: file exists before read.

## Exports

- [x] **EXP-1** `from ressmith.primitives import MaterialBalanceDecline` succeeds.
- [x] **EXP-2** `from ressmith.primitives import ReservesClassification` succeeds.
- [x] **EXP-3** `from ressmith.primitives import classify_reserves_from_material_balance` succeeds.

## Tests

- [x] **TEST-1** `uv run pytest tests/` passes. (Run each command separately; do not paste `# comment` on same line.)
- [x] **TEST-2** Coverage report generated: `uv run pytest tests/ --cov=ressmith`.
- [x] **TEST-3** Unit tests for `read_csv_production`, `write_csv_results` in `test_workflows_io.py`.
- [x] **TEST-4** Unit tests for `import_simulator_output`, `export_for_simulator`, `import_simulation_results`, `compare_simulation_to_forecast` in `test_workflows_simulator.py`.

## Build & Dependencies (uv)

- [x] **BUILD-1** `uv sync` succeeds (or `uv sync --group dev` for development).
- [x] **BUILD-2** No broken imports: `uv run python -c "import ressmith; ressmith.fit_forecast"` succeeds.
- [x] **BUILD-3** Black passes. Ruff: `uv run ruff check ressmith --fix` (some pre-existing issues may remain).

## Documentation

- [x] **DOC-1** No docstring references to `decline_curve` (replaced with `ressmith`).
- [x] **DOC-2** README installation uses uv: `uv pip install ressmith`, `uv sync --group dev`.

## Observability

- [x] **OBS-1** Workflow entry points log at INFO (core.py, backtesting, io, etc.).
- [x] **OBS-2** Exceptions logged with stack trace where appropriate (simulator uses `exc_info=True`).
