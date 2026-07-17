# Reservoir engineer capability map

This page ties **common engineering questions** to **ResSmith entry points**. It complements the [Model selection guide](model_selection_guide.md) (decline models) and [Advanced workflows](advanced_workflows.md) (multi-well and heavier flows).

If you are new to the package, start with [Getting started](getting_started.md).

## Bundled study workflows (single call)

| Goal | Function | Notes |
|------|----------|--------|
| Full interference pass (distances + matrix + optional EUR-based pairs + spacing hint) | `well_interference_study` | Wraps `calculate_well_distances`, `analyze_interference_matrix`, optional `analyze_interference_with_production_history`, `recommend_spacing` |
| Coning screen + optional WOR/GOR vs time | `coning_study` | Wraps `analyze_well_coning` and optionally `forecast_wor_gor_with_coning` |
| RTA stack (normalize, type curve, Blasingame, DN, optional FMB / fracture, log-log diagnostics) | `enhanced_rta_study` | Toggles per step; failures in one step do not stop the rest |
| Mud system (rheology, gels, filtration) | `mud_system_study` | Wraps `analyze_mud_system` — `ressmith/workflows/drilling.py` |
| Kick detection + kill parameters | `kick_analysis_study` | Wraps `detect_and_analyze_kick` — same module |
| Elastic props, UCS, mud-weight window | `geomechanical_study` | Wraps `complete_geomechanical_analysis` — `ressmith/workflows/geomechanics.py` |
| Pore pressure / fracture gradient / drilling window | `formation_pressure_study` | Wraps `complete_pressure_analysis` — same module |

Study workflows are exported from `ressmith.workflows` (interference / coning / RTA also from top-level `import ressmith`).

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
| ESP head / HP, gas-lift efficiency | `esp_required_head`, `esp_horsepower`, `gas_lift_performance` — `ressmith/primitives/vlp.py` |
| Water cut / producing GOR | `water_cut`, `producing_gor` — same module |

### Well tests

| Question | Start here |
|----------|------------|
| “Does this test data make sense?” | `validate_well_test_data`, `validate_well_test_results`, `validate_and_analyze_well_test` — `ressmith/workflows/well_test_validation.py` (also exported from top-level `ressmith`) |
| Buildup / drawdown analysis | `analyze_buildup_test`, `analyze_drawdown_test` — `ressmith/primitives/well_test.py` |
| Flow-regime ID / wellbore storage / MDR | `identify_flow_regimes`, `calculate_wellbore_storage`, `analyze_mdr` — same module |
| Multirate / type-curve PTA | `analyze_multirate_superposition`, `match_well_test_type_curve`, `generate_dimensionless_pressure` |

### Volumetrics and reservoir basics

| Question | Start here |
|----------|------------|
| Volumetric OOIP / OGIP | `original_oil_in_place`, `original_gas_in_place` — `ressmith/primitives/volumetrics.py` |
| Recovery factor by drive | `recovery_factor` — same module |
| Darcy rate / PI-style helpers | `darcy_flow_rate` |
| Permeability / skin from buildup | `permeability_from_buildup`, `skin_factor_from_buildup`, `skin_factor_from_pressures` |

### Drilling and hydraulics

| Question | Start here |
|----------|------------|
| Hydrostatic / ECD / surge–swab | `hydrostatic_pressure`, `equivalent_circulating_density`, `surge_pressure`, `swab_pressure` — `ressmith/primitives/drilling.py` |
| Bit hydraulics / annular velocity | `bit_hydraulics`, `annular_velocity`, `circulating_pressure_loss` |
| Casing burst / collapse | `casing_burst`, `casing_collapse` |
| Hookload / torque / critical RPM | `hookload`, `torque`, `critical_rpm` |

### Drilling fluids

| Question | Start here |
|----------|------------|
| One-call mud system study | `mud_system_study` — `ressmith/workflows/drilling.py` |
| Mud weight / barite / dilution | `increase_mud_weight`, `dilute_mud_weight`, `ppg_to_psi_per_ft` — `ressmith/primitives/drilling_fluids.py` |
| Rheology (Bingham / Power Law / HB) | `bingham_plastic_model`, `power_law_model`, `herschel_bulkley_model`, `gel_strength_analysis` |
| Hydraulics optimization / ECD | `optimize_hydraulics`, `calculate_ecd`, `pressure_loss_pipe`, `pressure_loss_annulus` |
| Solids / LCM / treatment | `drilled_solids_fraction`, `lost_circulation_material`, `lime_treatment` |

### Well control

| Question | Start here |
|----------|------------|
| One-call kick analysis | `kick_analysis_study` — `ressmith/workflows/drilling.py` |
| Kick detection / intensity | `detect_kick`, `kick_intensity`, `formation_pressure_from_sidpp` — `ressmith/primitives/well_control.py` |
| Kill methods | `drillers_method`, `wait_and_weight_method`, `concurrent_method`, `calculate_kill_mud_weight` |
| Gas migration / choke / MAASP | `gas_migration_rate`, `choke_pressure`, `maximum_allowable_annular_pressure` |
| Kick simulation | `simulate_gas_kick`, `detect_and_analyze_kick` |

### Subsea / deepwater drilling

| Question | Start here |
|----------|------------|
| Riser tension / recoil / integrity | `riser_tension`, `riser_recoil`, `riser_pressure_integrity` — `ressmith/primitives/subsea_drilling.py` |
| Dual-gradient / MPD | `dual_gradient_drilling`, `surface_backpressure_required`, `mpd_pressure_profile` |
| Deepwater kick tolerance | `maximum_kick_tolerance`, `kick_margin_analysis`, `gas_rise_velocity` |
| End-to-end subsea design pass | `subsea_well_design_analysis` |

### Geomechanics and formation pressure

| Question | Start here |
|----------|------------|
| One-call geomechanical study | `geomechanical_study` — `ressmith/workflows/geomechanics.py` |
| Elastic moduli / UCS / friction | `calculate_youngs_modulus`, `ucs_from_sonic`, `estimate_friction_angle` — `ressmith/primitives/rock_mechanics.py` |
| Failure criteria | `mohr_coulomb_criterion`, `drucker_prager_criterion`, `hoek_brown_criterion` |
| Mud-weight window / sanding | `mud_weight_window`, `collapse_pressure`, `fracture_pressure`, `sand_production_index` |
| One-call pore-pressure study | `formation_pressure_study` — `ressmith/workflows/geomechanics.py` |
| Eaton / sonic / d-exponent / LOT | `eatons_method`, `sonic_method`, `d_exponent_method`, `leak_off_test_analysis` — `ressmith/primitives/formation_pressure.py` |
| Fracture gradient / drilling window | `matthews_kelly_method`, `hubert_willis_method`, `drilling_window_analysis` |

### Geostatistics

| Question | Start here |
|----------|------------|
| Variogram models / empirical fit | `variogram_model`, `compute_empirical_variogram`, `fit_variogram_wls` — `ressmith/primitives/geostats.py` |
| Ordinary kriging | `ordinary_kriging` — numpy fallback if SciPy absent; EUR mapping also via `spatial_analysis` in `workflows/integrations.py` |

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
- **SciPy:** Type-curve matching, choke optimization, and history matching need `ressmith[scipy]`. Core ARPS fitting and most drilling/geomech primitives run on NumPy alone; geostats has a NumPy distance fallback.
- **Heavy ML / spatial:** ARIMA, PyKrige EUR mapping, Chronos, etc. need `ressmith[stats]`, `[spatial]`, `[llm]`, or `[ml]` as documented in `pyproject.toml`. Prefer `primitives/geostats.py` when you want dependency-light variogram/kriging.
- **Not a full simulator:** Use export/import and objectives to couple to your simulator; ResSmith stays explicit about what it computes vs. what you run externally.

## Related internal note

An older qualitative review (2024) motivated this map; an updated **implementation status** table lives alongside that note in the repository’s `.cursor/RESERVOIR_ENGINEER_REVIEW.md` for maintainers.
