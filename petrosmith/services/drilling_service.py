"""
Drilling service - orchestrates drilling operations.

This service layer coordinates drilling models and core calculations.
All methods are stateless - data is passed as parameters, not stored.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from petrosmith.models import DrillingParameters, DrillString, Casing, Mud
from petrosmith.core import DrillingCalculations, WellControlCalculations

logger = logging.getLogger(__name__)


class DrillingService:
    """
    Service for managing drilling operations and analysis.
    
    This is a STATELESS service - all methods take data as parameters.
    No data is stored in the service instance.
    """
    
    @staticmethod
    def analyze_hydraulics(
        drilling_params: DrillingParameters,
        drill_string: DrillString,
        mud: Mud,
        pump_rate: float,
        hole_diameter: float
    ) -> Dict[str, Any]:
        """
        Perform comprehensive hydraulics analysis.
        
        Args:
            drilling_params: Current drilling parameters
            drill_string: Drill string configuration
            mud: Current mud properties
            pump_rate: Pump rate in gpm
            hole_diameter: Hole diameter in inches
            
        Returns:
            Dictionary with hydraulics analysis
        """
        # Calculate hydrostatic pressure
        hydrostatic = DrillingCalculations.calculate_hydrostatic_pressure(
            mud_weight=mud.density,
            true_vertical_depth=drilling_params.true_vertical_depth
        )
        
        # Calculate annular velocity
        annular_velocity = DrillingCalculations.calculate_annular_velocity(
            flow_rate_gpm=pump_rate,
            hole_diameter=hole_diameter,
            pipe_od=drill_string.drill_pipe_od
        )
        
        # Calculate pressure loss in drill pipe
        pipe_pressure_loss = DrillingCalculations.calculate_circulating_pressure_loss(
            pump_rate=pump_rate,
            viscosity=mud.viscosity,
            pipe_length=drill_string.drill_pipe_length,
            pipe_id=drill_string.drill_pipe_id
        )
        
        # Calculate ECD
        # Simplified assumption: annular pressure loss is ~50% of pipe pressure loss
        ANNULAR_TO_PIPE_LOSS_RATIO = 0.5
        
        ecd = DrillingCalculations.calculate_equivalent_circulating_density(
            mud_weight=mud.density,
            annular_pressure_loss=pipe_pressure_loss * ANNULAR_TO_PIPE_LOSS_RATIO,
            true_vertical_depth=drilling_params.true_vertical_depth
        )
        
        # Calculate hookload
        from petrosmith.core.constants import DefaultValues
        # Approximate drill string weight (typical range 15-25 lb/ft)
        APPROX_STRING_WEIGHT_LB_FT = 20.0
        APPROX_STRING_DISPLACEMENT_BBL_FT = 0.01
        
        string_weight = drill_string.total_length * APPROX_STRING_WEIGHT_LB_FT
        hookload = DrillingCalculations.calculate_hookload(
            string_weight_air=string_weight / 1000,  # Convert to klbs
            mud_weight=mud.density,
            string_displacement=APPROX_STRING_DISPLACEMENT_BBL_FT
        )
        
        return {
            "hydrostatic_pressure": hydrostatic,
            "annular_velocity": annular_velocity,
            "pipe_pressure_loss": pipe_pressure_loss,
            "equivalent_circulating_density": ecd,
            "hookload": hookload,
            "mud_weight": mud.density,
            "total_depth": drilling_params.measured_depth,
            "warnings": DrillingService._check_hydraulics_warnings(annular_velocity, ecd)
        }
    
    @staticmethod
    def _check_hydraulics_warnings(
        annular_velocity: float,
        ecd: float
    ) -> List[str]:
        """Check for hydraulics warnings."""
        warnings = []
        
        # Industry standard annular velocity ranges (ft/min)
        MIN_ANNULAR_VELOCITY = 120  # ft/min - minimum for hole cleaning
        MAX_ANNULAR_VELOCITY = 400  # ft/min - maximum to avoid erosion
        MAX_SAFE_ECD = 18.0  # ppg - typical maximum before lost circulation risk
        
        if annular_velocity < MIN_ANNULAR_VELOCITY:
            warnings.append("Low annular velocity - risk of poor hole cleaning")
        elif annular_velocity > MAX_ANNULAR_VELOCITY:
            warnings.append("High annular velocity - risk of formation erosion")
        
        if ecd > MAX_SAFE_ECD:
            warnings.append("High ECD - risk of lost circulation")
        
        return warnings
    
    @staticmethod
    def analyze_well_control_situation(
        drilling_params: DrillingParameters,
        mud: Mud,
        pit_gain: float,
        drcp: float,
        dcpp: float
    ) -> Dict[str, Any]:
        """
        Analyze a well control situation (kick).
        
        Args:
            drilling_params: Current drilling parameters
            mud: Current mud properties
            pit_gain: Pit gain in barrels
            drcp: Drill pipe pressure reading in psi
            dcpp: Casing pressure reading in psi
            
        Returns:
            Dictionary with well control analysis
        """
        # Calculate formation pressure from SIDPP (drcp = shut-in drill pipe pressure)
        formation_pressure = WellControlCalculations.calculate_formation_pressure(
            sidpp=drcp,
            mud_weight=mud.density,
            true_vertical_depth=drilling_params.true_vertical_depth,
        )
        
        # Calculate kill mud weight
        kill_mud_weight = WellControlCalculations.calculate_kill_mud_weight(
            formation_pressure=formation_pressure,
            true_vertical_depth=drilling_params.true_vertical_depth,
            safety_margin=0.5
        )
        
        # Classify kick
        kick_volume, severity = WellControlCalculations.calculate_kick_volume(
            pit_gain=pit_gain
        )
        
        # Calculate kill pressures
        # Typical slow pump rate pressure (should be measured, this is example value)
        EXAMPLE_SPR_PRESSURE = 500.0  # psi
        spr_pressure = EXAMPLE_SPR_PRESSURE
        icp = WellControlCalculations.calculate_initial_circulating_pressure(
            slow_pump_rate_pressure=spr_pressure,
            shut_in_drillpipe_pressure=drcp,
        )
        
        fcp = WellControlCalculations.calculate_final_circulating_pressure(
            slow_pump_rate_pressure=spr_pressure,
            kill_mud_weight=kill_mud_weight,
            original_mud_weight=mud.density,
        )
        
        return {
            "kick_analysis": {
                "pit_gain": pit_gain,
                "kick_volume": kick_volume,
                "severity": severity,
                "formation_pressure": formation_pressure
            },
            "kill_parameters": {
                "current_mud_weight": mud.density,
                "kill_mud_weight": kill_mud_weight,
                "initial_circulating_pressure": icp,
                "final_circulating_pressure": fcp
            },
            "recommendations": DrillingService._get_well_control_recommendations(severity, kill_mud_weight)
        }
    
    @staticmethod
    def _get_well_control_recommendations(
        severity: int,
        kill_mud_weight: float
    ) -> List[str]:
        """Generate well control recommendations. Severity 0-9 from kick detection."""
        recommendations = []
        
        if severity >= 5:
            recommendations.append("IMMEDIATE ACTION: Shut in well and notify supervisor")
            recommendations.append("Consider evacuating non-essential personnel")
        elif severity >= 3:
            recommendations.append("Shut in well and prepare for well control operations")
        
        # High mud weight threshold (approaching operational limits)
        HIGH_MUD_WEIGHT_THRESHOLD = 19.0  # ppg
        
        if kill_mud_weight > HIGH_MUD_WEIGHT_THRESHOLD:
            recommendations.append("High kill mud weight - verify fracture gradient before circulating")
        
        recommendations.append("Monitor pit levels and pressures continuously")
        recommendations.append("Prepare kill mud and verify choke operation")
        
        return recommendations
    
    @staticmethod
    def optimize_drilling_parameters(
        drilling_params: DrillingParameters,
        drill_string: DrillString,
        target_rop: float = 50.0
    ) -> Dict[str, Any]:
        """
        Optimize drilling parameters for target ROP.
        
        Args:
            drilling_params: Current drilling parameters
            drill_string: Drill string configuration
            target_rop: Target rate of penetration in ft/hr
            
        Returns:
            Dictionary with optimized parameters
        """
        # Calculate optimal WOB (simplified empirical relationship)
        # Typical range: 3-5 klbs per inch of bit diameter
        WOB_PER_INCH_BIT = 4.0  # klbs per inch
        APPROX_DC_WEIGHT_LB_FT = 100.0  # Approximate drill collar weight
        CRITICAL_RPM_SAFETY_FACTOR = 0.7  # Stay at 70% of critical RPM
        MAX_RECOMMENDED_RPM = 120  # Maximum recommended RPM for most operations
        
        optimal_wob = drill_string.bit_diameter * WOB_PER_INCH_BIT
        
        # Calculate optimal RPM
        critical_rpm = DrillingCalculations.calculate_critical_rpm(
            drill_collar_length=drill_string.drill_collar_length,
            drill_collar_od=drill_string.drill_collar_od,
            drill_collar_weight=APPROX_DC_WEIGHT_LB_FT
        )
        
        optimal_rpm = min(MAX_RECOMMENDED_RPM, critical_rpm * CRITICAL_RPM_SAFETY_FACTOR)
        
        # Calculate required torque
        from petrosmith.core.constants import DefaultValues
        
        torque = DrillingCalculations.calculate_torque(
            weight_on_bit=optimal_wob,
            hole_diameter=drill_string.bit_diameter,
            friction_coefficient=DefaultValues.DEFAULT_FRICTION_COEFFICIENT
        )
        
        return {
            "current_parameters": {
                "wob": drilling_params.weight_on_bit,
                "rpm": drilling_params.rotary_speed,
                "rop": drilling_params.rate_of_penetration
            },
            "optimized_parameters": {
                "weight_on_bit": optimal_wob,
                "rotary_speed": optimal_rpm,
                "pump_rate": drilling_params.pump_rate,  # Keep current
                "expected_torque": torque
            },
            "limits": {
                "critical_rpm": critical_rpm,
                "max_wob": optimal_wob * 1.5,
                "min_wob": optimal_wob * 0.5
            }
        }
