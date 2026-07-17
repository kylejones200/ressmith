"""Physical constants and conversion factors for petroleum engineering.

Ported from petrosmith. Used by drilling, geomechanics, and reservoir primitives.
"""

from __future__ import annotations


class PhysicalConstants:
    """Physical constants in oilfield units unless noted."""

    # Pressure and Density
    HYDROSTATIC_GRADIENT = 0.052  # psi/ft per ppg
    ATMOSPHERIC_PRESSURE = 14.7  # psi
    STEEL_DENSITY = 65.5  # ppg
    WATER_DENSITY = 8.34  # ppg
    WATER_DENSITY_LB_GAL = 8.33  # lb/gal
    SEAWATER_DENSITY = 8.6  # ppg

    # Geometric
    PI = 3.14159265359
    CIRCLE_AREA_FACTOR = 0.7854  # pi/4

    # Conversion factors
    GPM_TO_FT_SEC = 0.408
    ANNULAR_VELOCITY_FACTOR = 24.5
    DARCY_FLOW_CONSTANT = 0.00708
    GALLONS_PER_BBL = 42.0
    BBL_TO_CUBIC_FT = 5.615
    MINUTES_PER_HOUR = 60.0
    HOURS_PER_DAY = 24.0

    # FVF
    STANDING_FVF_BASE = 0.9759
    STANDING_FVF_COEFF = 0.00012
    GAS_FVF_CONSTANT = 0.0283

    # Reservoir
    ACRE_TO_SQ_FT = 43560
    BBL_PER_ACRE_FT = 7758

    # Temperature
    RANKINE_OFFSET = 460
    KELVIN_OFFSET = 273.15
    STANDARD_TEMP_F = 60
    STANDARD_TEMP_R = 520

    # Well control
    MAX_SAFE_ECD_MARGIN = 0.5  # ppg
    KICK_TOLERANCE_FACTOR = 0.1

    # Drilling hydraulics
    HYDRAULIC_HP_CONVERSION = 1714
    IMPACT_FORCE_CONSTANT = 1930
    SURGE_SWAB_CONSTANT = 1500
    NOZZLE_DISCHARGE_COEFF = 0.95
    GRAVITY_CONSTANT = 32.2  # ft/s²

    # Rock mechanics
    UNIAXIAL_COMPRESSION_RATIO = 2.0
    HOEK_BROWN_MI_SANDSTONE = 17
    HOEK_BROWN_MI_LIMESTONE = 10
    HOEK_BROWN_MI_SHALE = 6

    # Well testing
    LOG_SLOPE_TO_PERM = 162.6
    HORNER_TIME_RATIO_BASE = 1.5
    SKIN_FACTOR_RATIO = 1.151

    # Production
    PI_CONVERSION_CONSTANT = 0.00708
    IPR_COEFFICIENT = 1.8
    GAS_LIFT_EFFICIENCY = 0.5
    ESP_EFFICIENCY_DEFAULT = 0.7

    # Fluid
    BUBBLE_POINT_EXPONENT = 1.175
    VISCOSITY_DEAD_OIL_EXP = 1.163
    Z_FACTOR_PSEUDOREDUCED = 0.27

    # Casing design
    BURST_SAFETY_FACTOR = 1.1
    COLLAPSE_SAFETY_FACTOR = 1.125
    TENSION_SAFETY_FACTOR = 1.6

    # Formation pressure
    OVERBURDEN_GRADIENT_TYPICAL = 1.0  # psi/ft
    NORMAL_PORE_PRESSURE_GRAD = 0.465  # psi/ft
    FRACTURE_GRADIENT_SHALLOW = 0.7
    FRACTURE_GRADIENT_DEEP = 1.0
    EATON_EXPONENT_SONIC = 3.0
    EATON_EXPONENT_RESISTIVITY = 1.2

    # Drilling fluids
    BARITE_SPECIFIC_GRAVITY = 4.2
    BARITE_SACK_WEIGHT_LBS = 100
    BARITE_SACK_VOLUME_GAL = 1.39
    PLASTIC_VISCOSITY_FACTOR = 300
    YIELD_POINT_FACTOR = 600
    POWER_LAW_EXPONENT_FACTOR = 3.32
    POWER_LAW_K_FACTOR = 5.11
    POWER_LAW_SHEAR_RATE = 1022

    # Subsea / deepwater
    RISER_BUOYANCY_FACTOR = 0.85
    SEAWATER_GRADIENT = 0.445  # psi/ft
    MUD_LINE_DEPTH_SHALLOW = 1000
    MUD_LINE_DEPTH_DEEP = 5000
    MUD_LINE_DEPTH_ULTRADEEP = 10000


class DefaultValues:
    """Default parameter values."""

    DEFAULT_WELLBORE_RADIUS = 0.328  # ft
    DEFAULT_FRICTION_COEFFICIENT = 0.25
    ECONOMIC_LIMIT_FACTOR = 0.10
    DEFAULT_PUMP_EFFICIENCY = 0.70
    DEFAULT_GAS_LIFT_EFFICIENCY = 0.50
    DEFAULT_DRAINAGE_RADIUS = 1000  # ft
    DEFAULT_SKIN_FACTOR = 0.0
    DEFAULT_SAFETY_MARGIN = 0.5  # ppg
    DEFAULT_OIL_VISCOSITY = 2.0  # cp
    DEFAULT_GAS_SPECIFIC_GRAVITY = 0.65
    DEFAULT_SALINITY = 0.0


class TypicalRanges:
    """Typical ranges for validation."""

    POROSITY_MIN = 0.05
    POROSITY_MAX = 0.40
    PERMEABILITY_MIN = 0.01
    PERMEABILITY_MAX = 10000
    MUD_WEIGHT_MIN = 8.0
    MUD_WEIGHT_MAX = 20.0
    BIT_SIZE_MIN = 3.0
    BIT_SIZE_MAX = 36.0
    WATER_CUT_MIN = 0.0
    WATER_CUT_MAX = 1.0
    GOR_MIN = 0.0
    GOR_MAX = 50000
    PRESSURE_MIN = 14.7
    PRESSURE_MAX = 25000


class UnitConversions:
    """Common unit conversion factors."""

    FT_TO_METERS = 0.3048
    METERS_TO_FT = 3.28084
    INCHES_TO_FT = 1 / 12
    FT_TO_INCHES = 12
    PSI_TO_KPA = 6.89476
    KPA_TO_PSI = 0.145038
    PSI_TO_BAR = 0.0689476
    BAR_TO_PSI = 14.5038
    BBL_TO_M3 = 0.158987
    M3_TO_BBL = 6.28981
    GAL_TO_BBL = 1 / 42
    BBL_TO_GAL = 42
    LBS_TO_KG = 0.453592
    KG_TO_LBS = 2.20462

    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        return (f - 32) * 5 / 9

    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        return c * 9 / 5 + 32

    @staticmethod
    def fahrenheit_to_rankine(f: float) -> float:
        return f + 460

    @staticmethod
    def celsius_to_kelvin(c: float) -> float:
        return c + 273.15


HYDROSTATIC_GRADIENT = 0.052
PI = PhysicalConstants.PI
ACRE_TO_SQ_FT = PhysicalConstants.ACRE_TO_SQ_FT
BBL_PER_ACRE_FT = PhysicalConstants.BBL_PER_ACRE_FT
WATER_DENSITY = 8.33
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
GALLONS_PER_BBL = 42

__all__ = [
    "PhysicalConstants",
    "DefaultValues",
    "TypicalRanges",
    "UnitConversions",
    "HYDROSTATIC_GRADIENT",
    "PI",
    "ACRE_TO_SQ_FT",
    "BBL_PER_ACRE_FT",
    "WATER_DENSITY",
    "MINUTES_PER_HOUR",
    "HOURS_PER_DAY",
    "GALLONS_PER_BBL",
]
