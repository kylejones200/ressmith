"""
Drilling Calculations Module

This module provides calculations for drilling operations including:
- Hydrostatic pressure
- Circulating pressure loss
- Surge and swab pressures
- Casing design  
- Wellbore stability
"""

from __future__ import annotations

import logging
import math
from .constants import PhysicalConstants

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


class DrillingCalculations:
    """
    Drilling engineering calculations.
    
    This class provides methods for calculating various drilling parameters.
    """
    
    @staticmethod
    def calculate_hydrostatic_pressure(true_vertical_depth: float, mud_weight: float) -> float:
        """
        Calculate hydrostatic pressure.
        
        Formula: P = 0.052 * MW * TVD
        
        Note: Input validation should be done at API layer.
        Core layer assumes valid inputs for performance.
        """
        if mud_weight <= 0:
            raise ValueError("Mud weight must be positive")
        if true_vertical_depth < 0:
            raise ValueError("TVD cannot be negative")
        return 0.052 * mud_weight * true_vertical_depth
    
    @staticmethod
    def calculate_circulating_pressure_loss(
        pump_rate: float,
        viscosity: float,
        pipe_length: float,
        pipe_id: float
    ) -> float:
        """
        Calculate pressure loss in circulating system.
        
        This is a simplified calculation. For accurate results,
        use detailed hydraulics models.
        
        Note: Input validation should be done at API layer.
        """
        if not (pump_rate > 0 and viscosity > 0 and pipe_length > 0 and pipe_id > 0):
            raise ValueError("All parameters must be positive")
        
        # Simplified pressure loss (Fanning equation)
        # Convert pipe ID from inches to feet
        velocity = pump_rate / (math.pi * (pipe_id / 12) ** 2)
        reynolds = (928 * velocity * pipe_id) / viscosity
        
        if reynolds < 2100:
            friction_factor = 16 / reynolds
        else:
            friction_factor = 0.046 / (reynolds ** 0.2)
        
        pressure_loss = friction_factor * pipe_length * velocity ** 2 / (25.8 * pipe_id)
        return pressure_loss
    
    @staticmethod
    def calculate_annular_velocity(
        flow_rate_gpm: float,
        hole_diameter: float,
        pipe_od: float
    ) -> float:
        """
        Calculate annular velocity.
        
        Args:
            flow_rate_gpm: Flow rate in gallons per minute
            hole_diameter: Hole diameter in inches
            pipe_od: Pipe outer diameter in inches
            
        Returns:
            Annular velocity in ft/min
            
        Note: Input validation should be done at API layer.
        """
        if flow_rate_gpm <= 0:
            raise ValueError("Flow rate must be positive")
        if hole_diameter <= pipe_od:
            raise ValueError("Hole diameter must be greater than pipe OD")
        
        annular_area = (hole_diameter ** 2 - pipe_od ** 2) / 1029.4
        velocity = flow_rate_gpm / annular_area
        return velocity
    
    @staticmethod
    def calculate_surge_pressure(
        pipe_od: float,
        hole_id: float,
        pipe_speed: float,
        mud_weight: float,
        mud_viscosity: float
    ) -> float:
        """
        Calculate surge pressure when running pipe.
        
        Args:
            pipe_od: Pipe outer diameter (inches)
            hole_id: Hole inner diameter (inches)
            pipe_speed: Pipe running speed (ft/min)
            mud_weight: Mud weight (ppg)
            mud_viscosity: Mud viscosity (cp)
            
        Returns:
            Surge pressure (psi)
            
        Note: Input validation should be done at API layer.
        """
        if pipe_od >= hole_id:
            raise ValueError("Pipe OD must be less than hole ID")

        clearance = (hole_id - pipe_od) / 2
        # Simplified surge pressure calculation
        surge_pressure = (mud_viscosity * pipe_speed) / (PhysicalConstants.SURGE_SWAB_CONSTANT * clearance ** 2)
        return surge_pressure
    
    @staticmethod
    def calculate_swab_pressure(
        pipe_od: float,
        hole_id: float,
        pipe_speed: float,
        mud_weight: float,
        mud_viscosity: float
    ) -> float:
        """
        Calculate swab pressure when pulling pipe.
        
        Args:
            pipe_od: Pipe outer diameter (inches)
            hole_id: Hole inner diameter (inches)
            pipe_speed: Pipe pulling speed (ft/min)
            mud_weight: Mud weight (ppg)
            mud_viscosity: Mud viscosity (cp)
            
        Returns:
            Swab pressure (psi)
            
        Note: Input validation should be done at API layer.
        """
        if pipe_od >= hole_id:
            raise ValueError("Pipe OD must be less than hole ID")

        clearance = (hole_id - pipe_od) / 2
        # Simplified swab pressure calculation
        swab_pressure = (mud_viscosity * pipe_speed) / (PhysicalConstants.SURGE_SWAB_CONSTANT * clearance ** 2)
        return swab_pressure
    
    @staticmethod
    def calculate_equivalent_circulating_density(
        mud_weight: float,
        annular_pressure_loss: float,
        true_vertical_depth: float
    ) -> float:
        """
        Calculate Equivalent Circulating Density (ECD).
        
        Args:
            mud_weight: Static mud weight (ppg)
            annular_pressure_loss: Annular pressure loss (psi)
            true_vertical_depth: True vertical depth (ft)
            
        Returns:
            ECD (ppg)
            
        Note: Input validation should be done at API layer.
        """
        if true_vertical_depth <= 0:
            raise ValueError("TVD must be positive")

        ecd = mud_weight + (annular_pressure_loss / (0.052 * true_vertical_depth))
        return ecd
    
    @staticmethod
    def calculate_bit_hydraulics(
        pump_pressure: float,
        flow_rate: float,
        parasitic_loss: float
    ) -> dict:
        """
        Calculate bit hydraulic horsepower and impact force.
        
        Args:
            pump_pressure: Pump pressure (psi)
            flow_rate: Flow rate (gpm)
            parasitic_loss: Parasitic pressure loss (psi)
            
        Returns:
            dict with hydraulic HP and impact force
        """
        bit_pressure = pump_pressure - parasitic_loss
        
        # Hydraulic horsepower
        hhp = (bit_pressure * flow_rate) / PhysicalConstants.HYDRAULIC_HP_CONVERSION
        
        # Impact force
        impact_force = (flow_rate * math.sqrt(bit_pressure * PhysicalConstants.WATER_DENSITY_LB_GAL)) / PhysicalConstants.IMPACT_FORCE_CONSTANT
        
        return {
            'bit_pressure_psi': bit_pressure,
            'hydraulic_horsepower': hhp,
            'impact_force_lbs': impact_force
        }
    
    @staticmethod
    def calculate_casing_burst(
        internal_pressure: float,
        external_pressure: float,
        yield_strength: float,
        wall_thickness: float,
        od: float
    ) -> dict:
        """
        Calculate casing burst pressure.
        
        Args:
            internal_pressure: Internal pressure (psi)
            external_pressure: External pressure (psi)
            yield_strength: Yield strength (psi)
            wall_thickness: Wall thickness (inches)
            od: Outside diameter (inches)
            
        Returns:
            dict with burst analysis
        """
        # Barlow's formula for burst pressure
        burst_pressure = (2 * yield_strength * wall_thickness) / od
        
        differential = internal_pressure - external_pressure
        safety_factor = burst_pressure / differential if differential > 0 else float('inf')
        
        return {
            'burst_pressure_psi': burst_pressure,
            'internal_pressure_psi': internal_pressure,
            'external_pressure_psi': external_pressure,
            'safety_factor': safety_factor,
            'status': 'SAFE' if safety_factor > PhysicalConstants.BURST_SAFETY_FACTOR else 'UNSAFE'
        }
    
    @staticmethod
    def calculate_casing_collapse(
        external_pressure: float,
        internal_pressure: float,
        yield_strength: float,
        od: float,
        wall_thickness: float
    ) -> dict:
        """
        Calculate casing collapse pressure using full API 5C3 formulas.
        
        Implements all four collapse regimes:
        1. Yield strength collapse
        2. Plastic collapse
        3. Transition collapse
        4. Elastic collapse
        
        Args:
            external_pressure: External pressure (psi)
            internal_pressure: Internal pressure (psi)
            yield_strength: Yield strength (psi)
            od: Outside diameter (inches)
            wall_thickness: Wall thickness (inches)
            
        Returns:
            dict with collapse analysis including regime
            
        Reference: API Bulletin 5C3, "Bulletin on Formulas and Calculations 
                   for Casing, Tubing, Drillpipe, and Line Pipe Properties"
        """
        # Calculate D/t ratio
        d_over_t = od / wall_thickness
        
        # API 5C3 collapse equation parameters
        # A, B, C are yield-strength dependent constants
        A = 2.8762 + 0.10679e-5 * yield_strength + 0.21301e-10 * (yield_strength ** 2) - 0.53132e-16 * (yield_strength ** 3)
        B = 0.026233 + 0.50609e-6 * yield_strength
        C = -465.93 + 0.030867 * yield_strength - 0.10483e-7 * (yield_strength ** 2) + 0.36989e-13 * (yield_strength ** 3)
        
        # Young's modulus for steel (psi)
        E = 30.0e6
        
        # Determine collapse regime and calculate collapse pressure
        if d_over_t <= A:
            # Regime 1: Yield strength collapse
            # Pp = 2*Yp / (D/t - 1)
            collapse_pressure = 2 * yield_strength / (d_over_t - 1)
            regime = "Yield Strength"
            
        elif A < d_over_t <= ((A + B) / 2):
            # Regime 2: Plastic collapse  
            # Pp = Yp/[(D/t)(A-F)] - F
            # where F = (46.95×10^6) / (D/t)^3
            F = 46.95e6 / (d_over_t ** 3)
            collapse_pressure = yield_strength / (d_over_t * (A - F)) - F
            regime = "Plastic"
            
        elif ((A + B) / 2) < d_over_t <= C:
            # Regime 3: Transition collapse
            # Pp = F + G(B - g)
            # where F = 46.95×10^6/(D/t)^3
            #       G = Yp(A-2)/(A-3)  
            #       g = B(D/t - A)/(C - A)
            F = 46.95e6 / (d_over_t ** 3)
            G = yield_strength * (A - 2) / (A - 3)
            g = B * (d_over_t - A) / (C - A)
            collapse_pressure = F + G * (B - g)
            regime = "Transition"
            
        else:
            # Regime 4: Elastic collapse
            # Pp = 46.95×10^6 / (D/t)^3
            collapse_pressure = 46.95e6 / (d_over_t ** 3)
            regime = "Elastic"
        
        differential = external_pressure - internal_pressure
        safety_factor = collapse_pressure / differential if differential > 0 else float('inf')
        
        return {
            'collapse_pressure_psi': collapse_pressure,
            'external_pressure_psi': external_pressure,
            'internal_pressure_psi': internal_pressure,
            'safety_factor': safety_factor,
            'collapse_regime': regime,
            'd_over_t_ratio': d_over_t,
            'status': 'SAFE' if safety_factor > PhysicalConstants.COLLAPSE_SAFETY_FACTOR else 'UNSAFE'
        }

    @staticmethod
    def calculate_hookload(
        string_weight_air: float,
        mud_weight: float,
        string_displacement: float = 0.01
    ) -> float:
        """
        Calculate hookload accounting for buoyancy.

        Args:
            string_weight_air: String weight in air (klbs)
            mud_weight: Mud weight in ppg
            string_displacement: Pipe displacement in bbl/ft (default 0.01)

        Returns:
            Hookload in klbs
        """
        STEEL_DENSITY_PPG = 65.5
        buoyancy_factor = 1.0 - (mud_weight / STEEL_DENSITY_PPG)
        return string_weight_air * buoyancy_factor

    @staticmethod
    def calculate_critical_rpm(
        drill_collar_length: float,
        drill_collar_od: float,
        drill_collar_weight: float
    ) -> float:
        """
        Estimate critical RPM (first natural frequency) for drill string whirl.

        Uses simplified formula: Nc ~ 9500 / sqrt(L) for typical drill collars.

        Args:
            drill_collar_length: Length in ft
            drill_collar_od: Outer diameter in inches
            drill_collar_weight: Weight per foot in lb/ft

        Returns:
            Critical RPM (avoid operating near this to prevent resonance)
        """
        if drill_collar_length <= 0:
            raise ValueError("Drill collar length must be positive")
        return 9500.0 / math.sqrt(drill_collar_length)

    @staticmethod
    def calculate_torque(
        weight_on_bit: float,
        hole_diameter: float,
        friction_coefficient: float = 0.25
    ) -> float:
        """
        Estimate drilling torque from WOB and bit diameter.

        Simplified: T = WOB * (D/2) * friction / 12 for ft-lbs.

        Args:
            weight_on_bit: WOB in klbs
            hole_diameter: Hole/bit diameter in inches
            friction_coefficient: Effective friction (default 0.25)

        Returns:
            Torque in ft-lbs
        """
        if weight_on_bit <= 0 or hole_diameter <= 0:
            raise ValueError("WOB and hole diameter must be positive")
        wob_lbs = weight_on_bit * 1000
        radius_in = hole_diameter / 2
        torque_in_lbs = wob_lbs * (radius_in / 12) * friction_coefficient
        return torque_in_lbs

__all__ = ['DrillingCalculations']
