"""
Drilling API - external interface for drilling operations.

This is the top-level API layer providing user-friendly interfaces.
"""

from typing import List, Dict, Optional
from petrosmith.models import DrillingParameters, DrillString, Mud, Casing
from petrosmith.services import DrillingService


class DrillingAPI:
    """
    High-level API for drilling engineering and operations.
    
    This is the primary interface for users to interact with drilling data.
    All methods are stateless - data is passed as parameters.
    """
    
    def calculate_hydrostatic_pressure(
        self,
        mud_weight: float,
        tvd: float
    ) -> float:
        """
        Calculate hydrostatic pressure.
        
        Args:
            mud_weight: Mud weight in ppg (8.0 - 20.0)
            tvd: True vertical depth in feet (must be >= 0)
            
        Returns:
            Hydrostatic pressure in psi
            
        Raises:
            InvalidMudWeightError: If mud weight is outside valid range
            InvalidDepthError: If TVD is negative
            
        Example:
            >>> api = DrillingAPI()
            >>> pressure = api.calculate_hydrostatic_pressure(10.5, 8000)
            >>> print(f"Hydrostatic pressure: {pressure} psi")
        """
        from petrosmith.core import DrillingCalculations
        from petrosmith.exceptions import InvalidMudWeightError, InvalidDepthError
        
        # Validate inputs at API boundary
        if not (8.0 <= mud_weight <= 20.0):
            raise InvalidMudWeightError(mud_weight)
        if tvd < 0:
            raise InvalidDepthError(tvd, reason="must be non-negative")
        
        return DrillingCalculations.calculate_hydrostatic_pressure(
            mud_weight=mud_weight,
            true_vertical_depth=tvd
        )
    
    def calculate_ecd(
        self,
        mud_weight: float,
        annular_pressure_loss: float,
        tvd: float
    ) -> float:
        """
        Calculate Equivalent Circulating Density.
        
        Args:
            mud_weight: Static mud weight in ppg (8.0 - 20.0)
            annular_pressure_loss: Annular pressure loss in psi (>= 0)
            tvd: True vertical depth in feet (must be > 0)
            
        Returns:
            ECD in ppg
            
        Raises:
            InvalidMudWeightError: If mud weight is outside valid range
            InvalidPressureError: If pressure loss is negative
            InvalidDepthError: If TVD is zero or negative
            
        Example:
            >>> api = DrillingAPI()
            >>> ecd = api.calculate_ecd(10.5, 300, 8000)
        """
        from petrosmith.core import DrillingCalculations
        from petrosmith.exceptions import (
            InvalidMudWeightError, 
            InvalidPressureError, 
            InvalidDepthError
        )
        
        # Validate inputs at API boundary
        if not (8.0 <= mud_weight <= 20.0):
            raise InvalidMudWeightError(mud_weight)
        if annular_pressure_loss < 0:
            raise InvalidPressureError(annular_pressure_loss, min_val=0.0)
        if tvd <= 0:
            raise InvalidDepthError(tvd, reason="must be positive for ECD calculation")
        
        return DrillingCalculations.calculate_equivalent_circulating_density(
            mud_weight=mud_weight,
            annular_pressure_loss=annular_pressure_loss,
            true_vertical_depth=tvd
        )
    
    def analyze_hydraulics(
        self,
        drilling_params: DrillingParameters,
        drill_string: DrillString,
        mud: Mud,
        pump_rate: float,
        hole_diameter: float
    ) -> Dict:
        """
        Perform comprehensive hydraulics analysis.
        
        Args:
            drilling_params: Current drilling parameters
            drill_string: Drill string configuration
            mud: Current mud properties
            pump_rate: Pump rate in gpm
            hole_diameter: Hole diameter in inches
            
        Returns:
            Dictionary with hydraulics analysis results
            
        Example:
            >>> from petrosmith.api import DrillingAPI
            >>> from petrosmith.models import DrillingParameters, DrillString, Mud
            >>> api = DrillingAPI()
            >>> # Create your data models
            >>> drilling_params = DrillingParameters(...)
            >>> drill_string = DrillString(...)
            >>> mud = Mud(...)
            >>> results = api.analyze_hydraulics(drilling_params, drill_string, mud, 450, 8.5)
        """
        return DrillingService.analyze_hydraulics(
            drilling_params=drilling_params,
            drill_string=drill_string,
            mud=mud,
            pump_rate=pump_rate,
            hole_diameter=hole_diameter
        )
    
    # NOTE: Old stateful methods removed. Services are now stateless.
    # If you need to store well data, use a database or pass data as parameters.
    # See analyze_hydraulics() for the new stateless pattern.
    
    def analyze_kick(
        self,
        drilling_params: DrillingParameters,
        mud: Mud,
        pit_gain: float,
        drcp: float,
        dcpp: float
    ) -> Dict:
        """
        Analyze a well control situation (kick).
        
        Args:
            drilling_params: Current drilling parameters
            mud: Current mud properties
            pit_gain: Pit gain in barrels
            drcp: Drill pipe pressure in psi
            dcpp: Casing pressure in psi
            
        Returns:
            Dictionary with well control analysis and kill parameters
            
        Example:
            >>> from petrosmith.api import DrillingAPI
            >>> from petrosmith.models import DrillingParameters, Mud
            >>> api = DrillingAPI()
            >>> drilling_params = DrillingParameters(...)
            >>> mud = Mud(...)
            >>> kick_analysis = api.analyze_kick(drilling_params, mud, 15, 500, 600)
            >>> print(f"Kill mud weight: {kick_analysis['kill_parameters']['kill_mud_weight']} ppg")
        """
        return DrillingService.analyze_well_control_situation(
            drilling_params=drilling_params,
            mud=mud,
            pit_gain=pit_gain,
            drcp=drcp,
            dcpp=dcpp
        )
    
    def calculate_kill_mud_weight(
        self,
        formation_pressure: float,
        tvd: float,
        safety_margin: float = 0.5
    ) -> float:
        """
        Calculate required kill mud weight.
        
        Args:
            formation_pressure: Formation pressure in psi
            tvd: True vertical depth in feet
            safety_margin: Safety margin in ppg (default 0.5)
            
        Returns:
            Kill mud weight in ppg
            
        Example:
            >>> api = DrillingAPI()
            >>> kill_weight = api.calculate_kill_mud_weight(4500, 8000)
        """
        from petrosmith.core import WellControlCalculations
        
        return WellControlCalculations.calculate_kill_mud_weight(
            formation_pressure=formation_pressure,
            true_vertical_depth=tvd,
            safety_margin=safety_margin
        )
    
    def optimize_drilling_parameters(
        self,
        drilling_params: DrillingParameters,
        drill_string: DrillString,
        target_rop: float = 50.0
    ) -> Dict:
        """
        Optimize drilling parameters for target ROP.
        
        Args:
            drilling_params: Current drilling parameters
            drill_string: Drill string configuration
            target_rop: Target rate of penetration in ft/hr
            
        Returns:
            Dictionary with optimized drilling parameters
        """
        return DrillingService.optimize_drilling_parameters(
            drilling_params=drilling_params,
            drill_string=drill_string,
            target_rop=target_rop
        )
    
    def calculate_casing_design(
        self,
        outer_diameter: float,
        wall_thickness: float,
        yield_strength: float
    ) -> Dict:
        """
        Calculate casing burst and collapse ratings.
        
        Args:
            outer_diameter: Casing OD in inches
            wall_thickness: Wall thickness in inches
            yield_strength: Yield strength in psi
            
        Returns:
            Dictionary with burst and collapse pressures
            
        Example:
            >>> api = DrillingAPI()
            >>> design = api.calculate_casing_design(9.625, 0.545, 80000)
            >>> print(f"Burst: {design['burst_pressure']} psi")
            >>> print(f"Collapse: {design['collapse_pressure']} psi")
        """
        from petrosmith.core import DrillingCalculations
        
        # Core functions use different parameter names
        burst_result = DrillingCalculations.calculate_casing_burst(
            internal_pressure=0,
            external_pressure=0,
            yield_strength=yield_strength,
            wall_thickness=wall_thickness,
            od=outer_diameter
        )
        
        collapse_result = DrillingCalculations.calculate_casing_collapse(
            external_pressure=0,
            internal_pressure=0,
            yield_strength=yield_strength,
            od=outer_diameter,
            wall_thickness=wall_thickness
        )
        
        return {
            "outer_diameter": outer_diameter,
            "wall_thickness": wall_thickness,
            "yield_strength": yield_strength,
            "burst_pressure": burst_result['burst_pressure_psi'],
            "collapse_pressure": collapse_result['collapse_pressure_psi'],
            "inner_diameter": outer_diameter - (2 * wall_thickness)
        }
