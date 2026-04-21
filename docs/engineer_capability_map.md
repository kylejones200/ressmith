# Reservoir engineer capability map

This page ties **common engineering questions** to **ResSmith entry points**. It complements the [Model selection guide](model_selection_guide.md) (decline models) and [Advanced workflows](advanced_workflows.md) (multi-well and heavier flows).

If you are new to the package, start with [Getting started](getting_started.md).

## Bundled study workflows (single call)

| Goal | Function | Notes |
|------|----------|--------|
| Full interference pass (distances + matrix + optional EUR-based pairs + spacing hint) | `well_interference_study` | Wraps `calculate_well_distances`, `analyze_interference_matrix`, optional `analyze_interference_with_production_history`, `recommend_spacing` |
| Coning screen + optional WOR/GOR vs time | `coning_study` | Wraps `analyze_well_coning` and optionally `forecast_wor_gor_with_coning` |
| RTA stack (normalize, type curve, Blasingame, DN, optional FMB / fracture, log-log diagnostics) | `enhanced_rta_study` | Toggles per step; failures in one step do not stop the rest |

All three are exported from `import ressmith` and `ressmith.workflows`.

## By play type or problem

### Unconventional: spacing, interference, multi-well

| Question | Start here |
|----------|------------|
| One-call interference study | `well_interference_study` |
| How much do my wells interfere? | `analyze_interference_matrix`, `analyze_interference_with_production_history` — `ressmith/workflows/interference.py` |
| Distances and pairs | `calculate_well_distances` — same module |
| Spacing recommendation from EUR | `recommend_spacing_from_eur` |
| Field-level spacing target | `optimize_field_spacing` — `ressmith/workflows/multi_well.py` |
| Drainage overlap / multi-well interaction | `analyze_multi_well_interaction`, `calculate_drainage_overlap_matrix` — primitives `multi_well` + workflows |

### Conventional: coning, breakthrough, WOR/GOR

| Question | Start here |
|----------|------------|
| One-call coning + optional yield forecast | `coning_study` |
| Critical rate (vertical well) | `meyer_gardner_critical_rate`, `chierici_ciucci_critical_rate` — `ressmith/primitives/coning.py` |
| Full coning workflow | `analyze_well_coning` — `ressmith/workflows/coning.py` |
| Forecast with coning-style yields | `forecast_wor_gor_with_coning` |

### RTA, type curves, diagnostics

| Question | Start here |
|----------|------------|
| One-call enhanced RTA bundle | `enhanced_rta_study` |
| Pressure-normalized rates | `normalize_production_with_pressure`, `normalize_for_type_curve_matching` — `ressmith/workflows/pressure_normalization.py` |
| Type curve match | `match_type_curve_workflow` (workflow), `match_type_curve` (primitive) |
| Diagnostic plot **data** (for your own plots or PlotSmith) | `generate_diagnostic_plot_data`, `generate_all_diagnostic_plots` — `ressmith/workflows/diagnostic_plotting.py` |
| Blasingame / DN / FMB style analysis | `ressmith/primitives/advanced_rta.py` |

### EOR and waterflood

| Question | Start here |
|----------|------------|
| Pattern sweep and efficiency | `analyze_waterflood`, `predict_waterflood_performance` — workflows; `ressmith/primitives/eor.py` for correlations |
| Five-spot / line drive / peripheral | See `WaterfloodPatternResult` and pattern arguments in `workflows/eor.py` |

### Rel perm, Pc, hysteresis

| Question | Start here |
|----------|------------|
| Corey, LET, Brooks–Corey, van Genuchten | `ressmith/primitives/relative_permeability.py` |
| Three-phase curves | `generate_three_phase_relative_permeability` and related generators in workflows + primitives |
| Hysteresis adjustment | `apply_hysteresis_to_relative_permeability` |

### Production operations

| Question | Start here |
|----------|------------|
| Allocation factors on production tables | `apply_allocation_adjustment` — `ressmith/workflows/downtime.py` |
| Allocation and facility-style helpers | `ressmith/primitives/production_ops.py` |
| Choke / lift optimization | `optimize_choke_size`, `optimize_esp_system`, `optimize_gas_lift_system` — `ressmith/workflows/` |

### Well tests

| Question | Start here |
|----------|------------|
| “Does this test data make sense?” | `validate_well_test_data`, `validate_well_test_results`, `validate_and_analyze_well_test` — `ressmith/workflows/well_test_validation.py` (also exported from top-level `ressmith`) |
| Buildup / drawdown analysis | `analyze_buildup_test`, `analyze_drawdown_test` — `ressmith/primitives/well_test.py` |

### Simulator bridge and history matching

| Question | Start here |
|----------|------------|
| Export / import for external simulators | `export_for_simulator`, `import_simulator_output`, `compare_simulation_to_forecast` — `ressmith/workflows/simulator.py` |
| Material balance history match | `history_match_material_balance`, `calculate_history_match_objective` — `ressmith/workflows/history_matching.py` |

### Economics and portfolio

| Question | Start here |
|----------|------------|
| Single-well or batch economics | `evaluate_economics`, portfolio helpers under `ressmith/workflows` (e.g. `analyze_portfolio`) |
| Backtesting forecast quality | `walk_forward_backtest` |

### Plotting and ecosystem

| Question | Start here |
|----------|------------|
| Quick forecast plot via PlotSmith | `plot_forecast` — `ressmith/workflows/integrations.py` |
| Outliers | `detect_outliers` — same module |

## Pitfalls (short)

- **Optional packages:** Plotting and some validators need `plotsmith`, `timesmith.typing`, etc. Install extras or use workflows that degrade gracefully (see README Development section).
- **Heavy ML / spatial:** ARIMA, kriging, Chronos, etc. need `ressmith[stats]`, `[spatial]`, `[llm]`, or `[ml]` as documented in `pyproject.toml`.
- **Not a full simulator:** Use export/import and objectives to couple to your simulator; ResSmith stays explicit about what it computes vs. what you run externally.

## Related internal note

An older qualitative review (2024) motivated this map; an updated **implementation status** table lives alongside that note in the repository’s `.cursor/RESERVOIR_ENGINEER_REVIEW.md` for maintainers.
