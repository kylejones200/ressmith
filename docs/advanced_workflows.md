# Advanced Workflows for Reservoir Engineers

## Overview

This guide covers advanced reservoir engineering workflows using ResSmith, including multi-well analysis, history matching, EOR evaluation, and production optimization.

## Multi-Well and Field Analysis

### Well Interference Analysis

**Analyze interference between wells:**

```python
from ressmith import (
    analyze_interference_with_production_history,
    recommend_spacing_from_eur,
    analyze_well_pattern
)

# Analyze interference
interference = analyze_interference_with_production_history(
    well_data={
        'well_1': data1,
        'well_2': data2,
        'well_3': data3
    },
    well_locations={
        'well_1': (lat1, lon1),
        'well_2': (lat2, lon2),
        'well_3': (lat3, lon3)
    }
)

# Get spacing recommendations
spacing = recommend_spacing_from_eur(
    well_data,
    well_locations,
    target_interference=0.1  # 10% maximum interference
)

print(f"Recommended spacing: {spacing['optimal_spacing']:.0f} ft")
print(f"Expected interference: {spacing['expected_interference']*100:.1f}%")
```

### Pattern Analysis

**Analyze well patterns (5-spot, 9-spot):**

```python
from ressmith import analyze_well_pattern

# 5-spot pattern analysis
pattern_result = analyze_well_pattern(
    well_data,
    pattern_type='five_spot',
    well_spacing=660.0  # 10-acre spacing
)

print(f"Pattern efficiency: {pattern_result['pattern_efficiency']:.2f}")
print(f"Drainage overlap: {pattern_result['drainage_overlap']*100:.1f}%")
```

## History Matching and Simulation Integration

### Material Balance History Matching

**Match historical production and pressure:**

```python
from ressmith import history_match_material_balance
import numpy as np

# Historical data
time = np.array([0, 30, 60, 90, 120, 150, 180])  # days
production = np.array([0, 5000, 12000, 20000, 28000, 35000, 42000])  # STB
pressure = np.array([5000, 4800, 4600, 4400, 4200, 4000, 3800])  # psi

# History match
result = history_match_material_balance(
    time=time,
    production=production,
    pressure=pressure,
    drive_mechanism='solution_gas'
)

print(f"Matched OOIP: {result.optimized_params['N']:,.0f} STB")
print(f"Initial pressure: {result.optimized_params['pi']:.0f} psi")
print(f"Match quality (RMSE): {result.production_match['rmse']:.0f} STB")
```

### Parameter Sensitivity Analysis

**Analyze sensitivity of history match parameters:**

```python
from ressmith import (
    run_parameter_sensitivity_analysis,
    analyze_parameter_sensitivity,
    identify_critical_parameters
)

# Sensitivity analysis
sensitivity = run_parameter_sensitivity_analysis(
    result,
    time=time,
    production=production,
    pressure=pressure,
    n_samples=50
)

# Identify critical parameters
critical = identify_critical_parameters(sensitivity, threshold=0.1)
print(f"Critical parameters: {critical}")
```

### Simulator Integration

**Export data for reservoir simulators:**

```python
from ressmith import export_simulator_input, import_simulator_output

# Export for Eclipse
export_simulator_input(
    production_data=production_df,
    pressure_data=pressure_df,
    output_path='eclipse_input.dat',
    format='eclipse',
    well_properties={
        'permeability': 50.0,  # md
        'porosity': 0.15,
        'thickness': 50.0  # ft
    }
)

# Import simulation results
sim_results = import_simulator_output(
    'simulation_output.rst',
    format='eclipse'
)

# Compare simulation to forecast
from ressmith import compare_simulation_to_forecast

comparison = compare_simulation_to_forecast(
    forecast=forecast,
    simulation_results=sim_results['production'],
    time_col='date'
)

print(f"Simulation vs Forecast RMSE: {comparison['rmse']:.2f}")
```

## Enhanced Oil Recovery (EOR)

### Waterflood Analysis

**Analyze waterflood performance:**

```python
from ressmith import (
    analyze_waterflood,
    predict_waterflood_performance,
    calculate_mobility_ratio_workflow
)

# Analyze existing waterflood
waterflood_result = analyze_waterflood(
    production_data=production_df,
    injection_data=injection_df,
    pattern_type='five_spot',
    well_spacing=660.0
)

print(f"Areal sweep efficiency: {waterflood_result['areal_sweep']:.2f}")
print(f"Displacement efficiency: {waterflood_result['displacement_efficiency']:.2f}")
print(f"Overall recovery factor: {waterflood_result['recovery_factor']:.2f}")

# Predict future performance
future_performance = predict_waterflood_performance(
    current_recovery=waterflood_result['recovery_factor'],
    target_recovery=0.35,  # 35% target
    injection_rate=1000.0,  # STB/day
    pattern_efficiency=waterflood_result['areal_sweep']
)

print(f"Time to target: {future_performance['time_to_target']:.0f} days")
```

### Mobility Ratio

**Calculate mobility ratio for waterflood:**

```python
mobility_ratio = calculate_mobility_ratio_workflow(
    oil_viscosity=1.0,  # cp
    water_viscosity=0.5,  # cp
    oil_relative_permeability=0.8,
    water_relative_permeability=0.3
)

print(f"Mobility ratio: {mobility_ratio:.2f}")
if mobility_ratio < 1.0:
    print("Favorable mobility ratio - good displacement efficiency expected")
else:
    print("Unfavorable mobility ratio - consider mobility control")
```

## Production Operations

### Well Test Validation

**Validate well test data and results:**

```python
from ressmith import validate_and_analyze_well_test
import numpy as np

# Well test data
time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])  # hours
pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])  # psi

# Validate and analyze
result = validate_and_analyze_well_test(
    time=time,
    pressure=pressure,
    test_type='buildup',
    production_rate=1000.0,  # STB/day
    production_time=720.0  # hours
)

if result['is_valid']:
    test_result = result['test_result']
    print(f"Permeability: {test_result.permeability:.2f} md")
    print(f"Skin factor: {test_result.skin:.2f}")
    print(f"Reservoir pressure: {test_result.reservoir_pressure:.0f} psi")
else:
    print("Test validation failed:")
    for error in result['data_validation'].errors:
        print(f"  - {error}")
```

### Production Allocation

**Optimize production allocation:**

```python
from ressmith import allocate_production, optimize_production
from ressmith.primitives import FacilityConstraints

# Well capacities
well_capacities = {
    'well_1': 1000.0,  # STB/day
    'well_2': 800.0,
    'well_3': 600.0
}

# Facility constraints
constraints = FacilityConstraints(
    max_total_rate=2000.0,  # Facility limit
    facility_capacity=2500.0,
    max_well_rate=1200.0
)

# Optimize allocation
allocation = optimize_production(
    well_capacities=well_capacities,
    total_target=2400.0,
    well_costs={'well_1': 10.0, 'well_2': 12.0, 'well_3': 15.0},
    constraints=constraints
)

print(f"Optimal allocation: {allocation['allocations']}")
print(f"Facility utilization: {allocation['facility_utilization']*100:.1f}%")
```

### Artificial Lift Optimization

**Optimize artificial lift systems:**

```python
from ressmith import (
    optimize_esp_system,
    optimize_gas_lift_system,
    compare_artificial_lift_methods
)

# ESP optimization
esp_result = optimize_esp_system(
    ipr_curve=(pressures, rates),
    vlp_curve=(pressures, rates),
    target_rate=500.0  # STB/day
)

print(f"Optimal ESP rate: {esp_result['optimal_rate']:.0f} STB/day")
print(f"Power required: {esp_result['power_required']:.1f} HP")

# Compare lift methods
comparison = compare_artificial_lift_methods(
    ipr_curve=(pressures, rates),
    vlp_curve=(pressures, rates),
    target_rate=500.0
)

print(f"Best method: {comparison['best_method']}")
print(f"Cost comparison: {comparison['cost_comparison']}")
```

## Rate Transient Analysis (RTA)

### Flow Regime Identification

**Identify flow regimes from production data:**

```python
from ressmith import (
    identify_flow_regime,
    analyze_production_data,
    generate_all_diagnostic_plots
)

# Identify flow regime
regime = identify_flow_regime(time, rate)
print(f"Flow regime: {regime}")

# Comprehensive RTA analysis
rta_result = analyze_production_data(
    time=time,
    rate=rate,
    pressure=pressure
)

print(f"Permeability: {rta_result['permeability']:.2f} md")
print(f"Fracture half-length: {rta_result['fracture_half_length']:.0f} ft")
print(f"SRV: {rta_result['srv']:.0f} acre-ft")

# Generate diagnostic plots
plots = generate_all_diagnostic_plots(
    time=time,
    rate=rate,
    save_path='rta_diagnostics'
)
```

### Type Curve Matching

**Match production to type curves:**

```python
from ressmith import (
    match_type_curve_workflow,
    analyze_blasingame,
    analyze_dn_type_curve
)

# Blasingame type curve
blasingame = analyze_blasingame(
    data=production_df,
    rate_col='oil',
    pressure_col='pressure'
)

# DN type curve (for tight gas/shale)
dn_result = analyze_dn_type_curve(
    data=production_df,
    rate_col='gas',
    initial_pressure=5000.0
)

print(f"DN flow regime: {dn_result['flow_regime']}")
print(f"Duong m parameter: {dn_result['duong_m']:.3f}")
```

## Coning and Breakthrough Analysis

### Water/Gas Coning

**Analyze coning and predict breakthrough:**

```python
from ressmith import (
    analyze_well_coning,
    forecast_wor_gor_with_coning
)

# Coning analysis
coning_result = analyze_well_coning(
    production_data=production_df,
    well_properties={
        'permeability': 50.0,  # md
        'thickness': 50.0,  # ft
        'oil_viscosity': 1.0,  # cp
        'water_viscosity': 0.5,  # cp
        'density_difference': 0.3  # g/cc
    }
)

print(f"Critical rate: {coning_result['critical_rate']:.0f} STB/day")
print(f"Coning risk: {coning_result['coning_risk']}")
print(f"Breakthrough time: {coning_result['breakthrough_time']:.0f} days")

# Forecast WOR/GOR with breakthrough
wor_gor_forecast = forecast_wor_gor_with_coning(
    production_data=production_df,
    coning_analysis=coning_result,
    horizon=60
)

print(f"Forecasted WOR at 60 months: {wor_gor_forecast['wor_forecast'].iloc[-1]:.2f}")
```

## Relative Permeability and Capillary Pressure

### Generate Relative Permeability Curves

**Create relative permeability curves:**

```python
from ressmith import (
    generate_relative_permeability_curves,
    generate_three_phase_relative_permeability,
    generate_capillary_pressure_curve
)

# Two-phase oil-water
kr_curves = generate_relative_permeability_curves(
    sw_values=np.linspace(0.2, 0.8, 100),
    swc=0.2,  # Connate water saturation
    sor=0.3   # Residual oil saturation
)

# Three-phase (oil/gas/water)
kr_3phase = generate_three_phase_relative_permeability(
    sw_values=np.linspace(0.2, 0.8, 100),
    sg_values=np.linspace(0.0, 0.3, 100),
    swc=0.2,
    sor=0.3,
    sgr=0.05  # Residual gas saturation
)

# Capillary pressure
pc_curve = generate_capillary_pressure_curve(
    sw_values=np.linspace(0.2, 0.8, 100),
    pore_throat_radius=10.0,  # microns
    interfacial_tension=30.0,  # dynes/cm
    contact_angle=0.0  # degrees
)
```

## Pressure Normalization

### Normalize Production Data

**Normalize rates and cumulative for RTA:**

```python
from ressmith import (
    normalize_production_with_pressure,
    normalize_for_rta_analysis,
    calculate_pseudopressure_workflow
)

# Normalize rates
normalized = normalize_production_with_pressure(
    data=production_df,
    rate_col='oil',
    pressure_col='pressure',
    initial_pressure=5000.0
)

# For RTA analysis
rta_normalized = normalize_for_rta_analysis(
    data=production_df,
    rate_col='gas',
    pressure_col='pressure',
    temperature=200.0,  # F
    gas_gravity=0.7
)

# Pseudopressure (for gas)
pseudopressure = calculate_pseudopressure_workflow(
    pressure=pressure_array,
    temperature=200.0,
    gas_gravity=0.7
)
```

## Summary

These advanced workflows enable comprehensive reservoir engineering analysis:

- **Multi-well analysis**: Interference, spacing, pattern efficiency
- **History matching**: Material balance, parameter sensitivity, simulator integration
- **EOR evaluation**: Waterflood analysis, mobility ratio, recovery prediction
- **Production operations**: Well test validation, allocation, artificial lift
- **RTA**: Flow regime identification, type curve matching, diagnostic plots
- **Coning analysis**: Breakthrough prediction, WOR/GOR forecasting
- **Fluid properties**: Relative permeability, capillary pressure

For more details, see:
- [Model Selection Guide](model_selection_guide.md)
- [Best Practices](best_practices.md)
- [API Reference](api.rst)
