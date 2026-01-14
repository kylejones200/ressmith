"""
Layer 2: Primitives

Algorithms and interfaces. No file I/O or plotting.
Can import numpy, pandas, and ressmith.objects only.
"""

# Base classes
from ressmith.primitives.base import (
    BaseDeclineModel,
    BaseEconModel,
    BaseEstimator,
    BaseObject,
)

# IPR (Inflow Performance Relationship)
from ressmith.primitives.ipr import (
    IPRResult,
    calculate_productivity_index,
    cinco_ley_fractured_ipr,
    composite_ipr,
    fetkovich_ipr,
    generate_ipr_curve,
    joshi_horizontal_ipr,
    linear_ipr,
    vogel_ipr,
)
# Material Balance
from ressmith.primitives.material_balance import (
    GasCapDriveParams,
    GasReservoirParams,
    SolutionGasDriveParams,
    WaterDriveParams,
    carter_tracy_water_influx,
    fetkovich_water_influx,
    gas_cap_drive_material_balance,
    gas_reservoir_pz_method,
    identify_drive_mechanism,
    solution_gas_drive_material_balance,
    undersaturated_oil_material_balance,
    water_drive_material_balance,
)
# PVT (Pressure-Volume-Temperature) Properties
from ressmith.primitives.pvt import (
    PVTProperties,
    beggs_robinson_oil_viscosity,
    calculate_pvt_properties,
    chew_connally_dead_oil_viscosity,
    gas_fvf,
    gas_z_factor,
    lee_gonzalez_gas_viscosity,
    standing_bo,
    standing_rs,
    vasquez_beggs_bo,
    vasquez_beggs_rs,
    water_fvf,
    water_viscosity,
)
# RTA (Rate Transient Analysis)
from ressmith.primitives.rta import (
    RTAResult,
    analyze_production_data,
    calculate_srv,
    estimate_fracture_half_length,
    estimate_permeability_from_production,
    identify_flow_regime,
)
# Well Interference
from ressmith.primitives.interference import (
    InterferenceResult,
    analyze_well_interference,
    calculate_interference_factor,
    calculate_well_distance,
    estimate_drainage_radius,
    estimate_production_interference,
    optimize_well_spacing,
)
# Coning Analysis
from ressmith.primitives.coning import (
    ConingResult,
    analyze_coning,
    calculate_coning_index,
    chierici_ciucci_critical_rate,
    estimate_breakthrough_time,
    meyer_gardner_critical_rate,
)
# Type Curves
from ressmith.primitives.type_curves import (
    TypeCurveMatch,
    calculate_type_curve_statistics,
    generate_arps_type_curve,
    match_multiple_type_curves,
    match_type_curve,
    normalize_production_data,
)
# EOR (Enhanced Oil Recovery)
from ressmith.primitives.eor import (
    WaterfloodPatternResult,
    analyze_waterflood_pattern,
    calculate_areal_sweep_efficiency,
    calculate_displacement_efficiency,
    calculate_injection_efficiency,
    calculate_mobility_ratio,
    calculate_vertical_sweep_efficiency,
)
# Multi-Well Interaction
from ressmith.primitives.multi_well import (
    DrainageVolume,
    MultiWellInteraction,
    analyze_drainage_volumes,
    calculate_drainage_overlap_matrix,
    calculate_drainage_volume,
    model_multi_well_interaction,
    optimize_multi_well_spacing,
)
# Production Operations
from ressmith.primitives.production_ops import (
    AllocationResult,
    FacilityConstraints,
    allocate_production_optimal,
    allocate_production_proportional,
    apply_facility_constraints,
    calculate_facility_utilization,
    optimize_production_allocation,
)
# Advanced RTA
from ressmith.primitives.advanced_rta import (
    BlasingameResult,
    FMBResult,
    analyze_complex_fracture_network,
    generate_blasingame_type_curve,
    generate_fmb_type_curve,
)
# Unit Conversion
from ressmith.primitives.units import (
    INTERNAL_RATE_UNIT,
    INTERNAL_TIME_UNIT,
    UnitConverter,
    UnitSystem,
    convert_decline_rate,
    validate_units,
)
# VLP (Vertical Lift Performance)
from ressmith.primitives.vlp import (
    NodalAnalysisResult,
    VLPResult,
    calculate_choke_performance,
    calculate_tubing_performance,
    generate_vlp_curve,
    optimize_artificial_lift,
    perform_nodal_analysis,
)
# Well Testing
from ressmith.primitives.well_test import (
    WellTestResult,
    analyze_buildup_test,
    analyze_drawdown_test,
    calculate_productivity_index_from_test,
    detect_boundaries,
)

__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseDeclineModel",
    "BaseEconModel",
    # IPR
    "IPRResult",
    "linear_ipr",
    "vogel_ipr",
    "fetkovich_ipr",
    "composite_ipr",
    "joshi_horizontal_ipr",
    "cinco_ley_fractured_ipr",
    "calculate_productivity_index",
    "generate_ipr_curve",
    # VLP
    "VLPResult",
    "NodalAnalysisResult",
    "calculate_tubing_performance",
    "generate_vlp_curve",
    "perform_nodal_analysis",
    "calculate_choke_performance",
    "optimize_artificial_lift",
    # RTA
    "RTAResult",
    "identify_flow_regime",
    "estimate_permeability_from_production",
    "estimate_fracture_half_length",
    "calculate_srv",
    "analyze_production_data",
    # Material Balance
    "SolutionGasDriveParams",
    "WaterDriveParams",
    "GasCapDriveParams",
    "GasReservoirParams",
    "solution_gas_drive_material_balance",
    "water_drive_material_balance",
    "gas_cap_drive_material_balance",
    "gas_reservoir_pz_method",
    "undersaturated_oil_material_balance",
    "fetkovich_water_influx",
    "carter_tracy_water_influx",
    "identify_drive_mechanism",
    # PVT
    "PVTProperties",
    "standing_rs",
    "standing_bo",
    "vasquez_beggs_rs",
    "vasquez_beggs_bo",
    "gas_z_factor",
    "gas_fvf",
    "beggs_robinson_oil_viscosity",
    "chew_connally_dead_oil_viscosity",
    "lee_gonzalez_gas_viscosity",
    "water_fvf",
    "water_viscosity",
    "calculate_pvt_properties",
    # Well Test
    "WellTestResult",
    "analyze_buildup_test",
    "analyze_drawdown_test",
    "detect_boundaries",
    "calculate_productivity_index_from_test",
    # Interference
    "InterferenceResult",
    "calculate_well_distance",
    "estimate_drainage_radius",
    "calculate_interference_factor",
    "estimate_production_interference",
    "analyze_well_interference",
    "optimize_well_spacing",
    # Coning
    "ConingResult",
    "meyer_gardner_critical_rate",
    "chierici_ciucci_critical_rate",
    "calculate_coning_index",
    "estimate_breakthrough_time",
    "analyze_coning",
    # Type Curves
    "TypeCurveMatch",
    "generate_arps_type_curve",
    "normalize_production_data",
    "match_type_curve",
    "match_multiple_type_curves",
    "calculate_type_curve_statistics",
    # EOR
    "WaterfloodPatternResult",
    "calculate_mobility_ratio",
    "calculate_areal_sweep_efficiency",
    "calculate_displacement_efficiency",
    "calculate_vertical_sweep_efficiency",
    "calculate_injection_efficiency",
    "analyze_waterflood_pattern",
    # Multi-Well
    "DrainageVolume",
    "MultiWellInteraction",
    "calculate_drainage_volume",
    "calculate_drainage_overlap_matrix",
    "analyze_drainage_volumes",
    "model_multi_well_interaction",
    "optimize_multi_well_spacing",
    # Production Operations
    "AllocationResult",
    "FacilityConstraints",
    "allocate_production_proportional",
    "allocate_production_optimal",
    "optimize_production_allocation",
    "calculate_facility_utilization",
    "apply_facility_constraints",
    # Advanced RTA
    "BlasingameResult",
    "FMBResult",
    "generate_blasingame_type_curve",
    "generate_fmb_type_curve",
    "analyze_complex_fracture_network",
    # Units
    "UnitSystem",
    "UnitConverter",
    "INTERNAL_TIME_UNIT",
    "INTERNAL_RATE_UNIT",
    "convert_decline_rate",
    "validate_units",
    # Physics-informed
    "MaterialBalanceDecline",
    "PressureDeclineModel",
    "MaterialBalanceParams",
    "PressureDeclineParams",
    # Physics reserves
    "ReservesClassification",
    "classify_reserves_from_material_balance",
]
