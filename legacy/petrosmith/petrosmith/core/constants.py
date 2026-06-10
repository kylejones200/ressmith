"""
Physical constants and conversion factors for petroleum engineering calculations.

All constants are documented with units and sources where applicable.
"""

class PhysicalConstants:
    """
    Physical constants used in petroleum engineering calculations.
    
    All values are in oilfield units unless otherwise specified.
    """
    
    # Pressure and Density
    HYDROSTATIC_GRADIENT = 0.052  # psi/ft per ppg - Pressure gradient constant
    ATMOSPHERIC_PRESSURE = 14.7   # psi - Standard atmospheric pressure
    STEEL_DENSITY = 65.5          # ppg - Density of steel
    WATER_DENSITY = 8.34          # ppg - Fresh water density at 60°F (same as 8.33 rounded)
    WATER_DENSITY_LB_GAL = 8.33   # lb/gal - Standard water density
    SEAWATER_DENSITY = 8.6        # ppg - Seawater density
    
    # Geometric Constants  
    PI = 3.14159265359
    CIRCLE_AREA_FACTOR = 0.7854   # pi/4 - For calculating circular areas from diameter
    
    # Conversion and Calculation Factors
    GPM_TO_FT_SEC = 0.408         # Convert gpm through pipe to ft/sec
    ANNULAR_VELOCITY_FACTOR = 24.5  # Factor for annular velocity from gpm
    DARCY_FLOW_CONSTANT = 0.00708 # Constant in Darcy radial flow equation
    GALLONS_PER_BBL = 42.0        # Gallons per barrel
    BBL_TO_CUBIC_FT = 5.615       # Cubic feet per barrel
    MINUTES_PER_HOUR = 60.0       # Time conversion
    HOURS_PER_DAY = 24.0          # Time conversion
    
    # Formation Volume Factor Constants
    STANDING_FVF_BASE = 0.9759    # Base value in Standing oil FVF correlation
    STANDING_FVF_COEFF = 0.00012  # Coefficient in Standing correlation
    GAS_FVF_CONSTANT = 0.0283     # Constant in gas FVF calculation
    
    # Reservoir Engineering
    ACRE_TO_SQ_FT = 43560         # Square feet per acre
    BBL_PER_ACRE_FT = 7758        # Barrels per acre-foot
    
    # Temperature
    RANKINE_OFFSET = 460          # Add to °F to get °R
    KELVIN_OFFSET = 273.15        # Add to °C to get K
    STANDARD_TEMP_F = 60          # °F - Standard temperature
    STANDARD_TEMP_R = 520         # °R - Standard temperature (60 + 460)
    
    # Well Control
    MAX_SAFE_ECD_MARGIN = 0.5     # ppg - Typical ECD safety margin below fracture
    KICK_TOLERANCE_FACTOR = 0.1   # 10% - Typical kick tolerance margin
    
    # Drilling Hydraulics
    HYDRAULIC_HP_CONVERSION = 1714  # Convert pressure*flow to hydraulic HP
    IMPACT_FORCE_CONSTANT = 1930    # Constant for bit impact force calculation
    SURGE_SWAB_CONSTANT = 1500      # Constant in surge/swab pressure calculations
    NOZZLE_DISCHARGE_COEFF = 0.95   # Typical discharge coefficient for bit nozzles
    GRAVITY_CONSTANT = 32.2         # ft/s² - Gravitational acceleration
    
    # Rock Mechanics
    UNIAXIAL_COMPRESSION_RATIO = 2.0  # Typical ratio of UCS to tensile strength
    HOEK_BROWN_MI_SANDSTONE = 17      # Hoek-Brown mi parameter for sandstone
    HOEK_BROWN_MI_LIMESTONE = 10      # Hoek-Brown mi parameter for limestone
    HOEK_BROWN_MI_SHALE = 6           # Hoek-Brown mi parameter for shale
    
    # Well Testing
    LOG_SLOPE_TO_PERM = 162.6     # Constant in semi-log analysis (162.6 * q * B * μ / k / h)
    HORNER_TIME_RATIO_BASE = 1.5  # Base ratio for Horner time calculation
    SKIN_FACTOR_RATIO = 1.151     # Constant in skin factor calculation (log(k/(phi*mu*ct*rw²)))
    
    # Production
    PI_CONVERSION_CONSTANT = 0.00708  # Productivity index constant
    IPR_COEFFICIENT = 1.8             # Vogel IPR coefficient
    GAS_LIFT_EFFICIENCY = 0.5         # 50% - Typical gas lift efficiency
    ESP_EFFICIENCY_DEFAULT = 0.7      # 70% - Default ESP efficiency
    
    # Fluid Properties
    BUBBLE_POINT_EXPONENT = 1.175     # Exponent in Standing bubble point correlation
    VISCOSITY_DEAD_OIL_EXP = 1.163    # Exponent in dead oil viscosity correlation
    Z_FACTOR_PSEUDOREDUCED = 0.27     # Coefficient in Z-factor correlation
    
    # Casing Design
    BURST_SAFETY_FACTOR = 1.1         # 110% - Minimum burst safety factor
    COLLAPSE_SAFETY_FACTOR = 1.125    # 112.5% - Minimum collapse safety factor
    TENSION_SAFETY_FACTOR = 1.6       # 160% - Minimum tension safety factor
    
    # Formation Pressure
    OVERBURDEN_GRADIENT_TYPICAL = 1.0    # psi/ft - Typical overburden gradient
    NORMAL_PORE_PRESSURE_GRAD = 0.465    # psi/ft - Normal hydrostatic gradient
    FRACTURE_GRADIENT_SHALLOW = 0.7      # psi/ft - Shallow formation fracture gradient
    FRACTURE_GRADIENT_DEEP = 1.0         # psi/ft - Deep formation fracture gradient
    EATON_EXPONENT_SONIC = 3.0           # Exponent for Eaton's method with sonic
    EATON_EXPONENT_RESISTIVITY = 1.2     # Exponent for Eaton's method with resistivity
    
    # Drilling Fluids
    BARITE_SPECIFIC_GRAVITY = 4.2        # Barite (BaSO4) specific gravity
    BARITE_SACK_WEIGHT_LBS = 100         # Standard barite sack weight
    BARITE_SACK_VOLUME_GAL = 1.39        # Volume per 100 lb sack of barite
    PLASTIC_VISCOSITY_FACTOR = 300       # Factor for Bingham plastic model
    YIELD_POINT_FACTOR = 600             # Factor for yield point calculation
    POWER_LAW_EXPONENT_FACTOR = 3.32     # Factor in power law n calculation
    POWER_LAW_K_FACTOR = 5.11            # Factor in power law K calculation
    POWER_LAW_SHEAR_RATE = 1022          # Shear rate at 600 RPM in power law
    
    # Subsea and Deepwater
    RISER_BUOYANCY_FACTOR = 0.85        # Typical riser buoyancy factor
    SEAWATER_GRADIENT = 0.445           # psi/ft - Seawater hydrostatic gradient
    MUD_LINE_DEPTH_SHALLOW = 1000       # ft - Shallow water depth threshold
    MUD_LINE_DEPTH_DEEP = 5000          # ft - Deep water depth threshold
    MUD_LINE_DEPTH_ULTRADEEP = 10000    # ft - Ultra-deep water threshold
    
class DefaultValues:
    """
    Default values and typical ranges for petroleum engineering parameters.
    """
    
    # Drilling
    DEFAULT_WELLBORE_RADIUS = 0.328  # ft - For 8.5" hole
    DEFAULT_FRICTION_COEFFICIENT = 0.25  # Dimensionless - Steel on steel
    
    # Production
    ECONOMIC_LIMIT_FACTOR = 0.10  # 10% of current rate
    DEFAULT_PUMP_EFFICIENCY = 0.70  # 70% - Typical ESP efficiency
    DEFAULT_GAS_LIFT_EFFICIENCY = 0.50  # 500 scf per barrel injected
    
    # Reservoir
    DEFAULT_DRAINAGE_RADIUS = 1000  # ft - Typical for 80-acre spacing
    DEFAULT_SKIN_FACTOR = 0.0  # Dimensionless - Undamaged well
    DEFAULT_SAFETY_MARGIN = 0.5  # ppg - Kill mud safety margin
    
    # Fluid Properties
    DEFAULT_OIL_VISCOSITY = 2.0  # cp
    DEFAULT_GAS_SPECIFIC_GRAVITY = 0.65  # Air = 1
    DEFAULT_SALINITY = 0.0  # ppm - Fresh water
    
class TypicalRanges:
    """
    Typical ranges for petroleum engineering parameters for validation.
    """
    
    # Reservoir Properties
    POROSITY_MIN = 0.05  # 5%
    POROSITY_MAX = 0.40  # 40%
    PERMEABILITY_MIN = 0.01  # md
    PERMEABILITY_MAX = 10000  # md
    
    # Drilling
    MUD_WEIGHT_MIN = 8.0  # ppg
    MUD_WEIGHT_MAX = 20.0  # ppg
    BIT_SIZE_MIN = 3.0  # inches
    BIT_SIZE_MAX = 36.0  # inches
    
    # Production
    WATER_CUT_MIN = 0.0  # 0%
    WATER_CUT_MAX = 1.0  # 100%
    GOR_MIN = 0.0  # scf/STB
    GOR_MAX = 50000  # scf/STB - High for gas condensate
    
    # Pressure
    PRESSURE_MIN = 14.7  # psi - Atmospheric
    PRESSURE_MAX = 25000  # psi - Ultra deep/HPHT
    
class UnitConversions:
    """
    Common unit conversion factors.
    """
    
    # Length
    FT_TO_METERS = 0.3048
    METERS_TO_FT = 3.28084
    INCHES_TO_FT = 1/12
    FT_TO_INCHES = 12
    
    # Pressure
    PSI_TO_KPA = 6.89476
    KPA_TO_PSI = 0.145038
    PSI_TO_BAR = 0.0689476
    BAR_TO_PSI = 14.5038
    
    # Volume
    BBL_TO_M3 = 0.158987
    M3_TO_BBL = 6.28981
    GAL_TO_BBL = 1/42
    BBL_TO_GAL = 42
    
    # Mass/Weight
    LBS_TO_KG = 0.453592
    KG_TO_LBS = 2.20462
    
    # Temperature
    @staticmethod
    def fahrenheit_to_celsius(f: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (f - 32) * 5/9
    
    @staticmethod
    def celsius_to_fahrenheit(c: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return c * 9/5 + 32
    
    @staticmethod
    def fahrenheit_to_rankine(f: float) -> float:
        """Convert Fahrenheit to Rankine."""
        return f + 460
    
    @staticmethod
    def celsius_to_kelvin(c: float) -> float:
        """Convert Celsius to Kelvin."""
        return c + 273.15

# Convenience imports for common usage
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
