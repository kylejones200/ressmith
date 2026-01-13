"""
Layer 2: Primitives

Algorithms and interfaces. No file I/O or plotting.
Can import numpy, pandas, and ressmith.objects only.
"""

from ressmith.primitives.base import (
    BaseDeclineModel,
    BaseEconModel,
    BaseEstimator,
    BaseObject,
)

# Reservoir engineering primitives
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
from ressmith.primitives.rta import (
    RTAResult,
    analyze_production_data,
    calculate_srv,
    estimate_fracture_half_length,
    estimate_permeability_from_production,
    identify_flow_regime,
)
from ressmith.primitives.units import (
    INTERNAL_RATE_UNIT,
    INTERNAL_TIME_UNIT,
    UnitConverter,
    UnitSystem,
    convert_decline_rate,
    validate_units,
)
from ressmith.primitives.vlp import (
    NodalAnalysisResult,
    VLPResult,
    calculate_choke_performance,
    calculate_tubing_performance,
    generate_vlp_curve,
    optimize_artificial_lift,
    perform_nodal_analysis,
)
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
