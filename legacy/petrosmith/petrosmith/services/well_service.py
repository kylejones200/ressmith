"""
Well service - orchestrates well-related operations.

This service layer coordinates multiple domain models and core calculations.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Any
from petrosmith.models import Well, WellCompletion, Perforation, Tubing
from petrosmith.core import ReservoirCalculations, ProductionCalculations

logger = logging.getLogger(__name__)


class WellService:
    """
    Service for managing well operations and analysis.
    
    Orchestrates well data, completions, and production calculations.
    """
    
    def __init__(self):
        """Initialize well service with data storage."""
        self.wells: Dict[str, Well] = {}
        self.completions: Dict[str, WellCompletion] = {}
        self.well_to_completion_map: Dict[str, str] = {}  # Maps well_id -> completion_id for O(1) lookup
    
    def add_well(self, well: Well) -> None:
        """
        Add a well to the service.
        
        Args:
            well: Well domain model
        """
        self.wells[well.well_id] = well
    
    def get_well(self, well_id: str) -> Optional[Well]:
        """
        Retrieve a well by ID.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Well model or None
        """
        return self.wells.get(well_id)
    
    def add_completion(self, completion: WellCompletion) -> None:
        """
        Add a completion design to a well.
        
        Args:
            completion: Well completion model
        """
        if completion.well_id not in self.wells:
            raise ValueError(f"Well {completion.well_id} not found")
        
        self.completions[completion.completion_id] = completion
        self.well_to_completion_map[completion.well_id] = completion.completion_id
    
    def get_well_completion(self, well_id: str) -> Optional[WellCompletion]:
        """
        Get completion for a specific well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Completion model or None
        """
        # O(1) lookup using index
        completion_id = self.well_to_completion_map.get(well_id)
        if completion_id:
            return self.completions.get(completion_id)
        return None
    
    def analyze_well_productivity(
        self,
        well_id: str,
        reservoir_pressure: float,
        reservoir_permeability: float,
        reservoir_thickness: float,
        fluid_viscosity: float,
        fluid_fvf: float,
        drainage_radius: float = 1000.0,
        skin_factor: float = 0.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive well productivity analysis.
        
        Args:
            well_id: Well identifier
            reservoir_pressure: Reservoir pressure in psi
            reservoir_permeability: Permeability in md
            reservoir_thickness: Net pay thickness in feet
            fluid_viscosity: Fluid viscosity in cp
            fluid_fvf: Formation volume factor
            drainage_radius: Drainage radius in feet
            skin_factor: Skin factor
            
        Returns:
            Dictionary with productivity analysis results
        """
        well = self.get_well(well_id)
        if not well:
            raise ValueError(f"Well {well_id} not found")
        
        completion = self.get_well_completion(well_id)
        if completion:
            wellbore_radius = completion.tubing.inner_diameter / (2 * 12)  # Convert to feet
        else:
            wellbore_radius = 0.328  # Default 8.5" hole
        
        # Calculate productivity index
        pressure_points = []
        flow_rates = []
        
        for pwf_fraction in [0.9, 0.8, 0.7, 0.6, 0.5]:
            pwf = reservoir_pressure * pwf_fraction
            drawdown = reservoir_pressure - pwf
            
            # Calculate flow rate using Darcy
            flow_rate = ReservoirCalculations.calculate_darcy_flow_rate(
                permeability=reservoir_permeability,
                thickness=reservoir_thickness,
                pressure_drawdown=drawdown,
                viscosity=fluid_viscosity,
                formation_volume_factor=fluid_fvf,
                drainage_radius=drainage_radius,
                wellbore_radius=wellbore_radius,
                skin_factor=skin_factor
            )
            
            pressure_points.append(pwf)
            flow_rates.append(flow_rate)
        
        # Calculate productivity index at average conditions
        avg_drawdown = reservoir_pressure * 0.3  # 70% of reservoir pressure
        avg_pwf = reservoir_pressure - avg_drawdown
        avg_flow_rate = ReservoirCalculations.calculate_darcy_flow_rate(
            permeability=reservoir_permeability,
            thickness=reservoir_thickness,
            pressure_drawdown=avg_drawdown,
            viscosity=fluid_viscosity,
            formation_volume_factor=fluid_fvf,
            drainage_radius=drainage_radius,
            wellbore_radius=wellbore_radius,
            skin_factor=skin_factor
        )
        
        productivity_index = ReservoirCalculations.calculate_productivity_index(
            flow_rate=avg_flow_rate,
            reservoir_pressure=reservoir_pressure,
            bottomhole_pressure=avg_pwf
        )
        
        return {
            "well_id": well_id,
            "productivity_index": productivity_index,
            "estimated_max_rate": max(flow_rates),
            "ipr_curve": {
                "pressures": pressure_points,
                "flow_rates": flow_rates
            },
            "completion_interval": completion.total_perforated_interval if completion else 0,
            "skin_factor": skin_factor,
            "drainage_area": math.pi * (drainage_radius ** 2) / 43560  # acres
        }
    
    def compare_artificial_lift_options(
        self,
        well_id: str,
        depth: float,
        desired_rate: float,
        reservoir_pressure: float,
        wellhead_pressure: float = 100.0
    ) -> Dict[str, Any]:
        """
        Compare artificial lift options for a well.
        
        Args:
            well_id: Well identifier
            depth: Well depth in feet
            desired_rate: Desired production rate in STB/day
            reservoir_pressure: Reservoir pressure in psi
            wellhead_pressure: Required wellhead pressure in psi
            
        Returns:
            Dictionary comparing lift options
        """
        well = self.get_well(well_id)
        if not well:
            raise ValueError(f"Well {well_id} not found")
        
        completion = self.get_well_completion(well_id)
        tubing_diameter = completion.tubing.inner_diameter if completion else 2.441  # Default 2-7/8"
        
        # ESP analysis
        esp_head = ProductionCalculations.calculate_esp_required_head(
            depth=depth,
            wellhead_pressure=wellhead_pressure,
            flow_rate=desired_rate
        )
        
        esp_hp = ProductionCalculations.calculate_esp_horsepower(
            flow_rate=desired_rate,
            total_head=esp_head
        )
        
        # Gas lift analysis (simplified)
        gas_injection_rate = desired_rate * 0.5  # Rough estimate: 500 scf per barrel
        
        return {
            "well_id": well_id,
            "desired_rate": desired_rate,
            "esp": {
                "required_head": esp_head,
                "required_horsepower": esp_hp,
                "estimated_cost_per_day": esp_hp * 24 * 0.10,  # $0.10/kWh estimate
                "suitable": esp_hp < 200  # ESP economical below 200 HP
            },
            "gas_lift": {
                "estimated_injection_rate": gas_injection_rate,
                "estimated_cost_per_day": gas_injection_rate * 3.0,  # $3/Mscf estimate
                "suitable": depth < 12000  # Gas lift practical below 12000 ft
            },
            "rod_pump": {
                "suitable": depth < 8000 and desired_rate < 500  # Typical limits
            }
        }
    
    def list_wells(self) -> List[str]:
        """
        List all well IDs in the service.
        
        Returns:
            List of well IDs
        """
        return list(self.wells.keys())
    
    def get_well_summary(self, well_id: str) -> Dict[str, Any]:
        """
        Get summary information for a well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dictionary with well summary
        """
        well = self.get_well(well_id)
        if not well:
            raise ValueError(f"Well {well_id} not found")
        
        completion = self.get_well_completion(well_id)
        
        summary = {
            "well_id": well.well_id,
            "well_name": well.well_name,
            "location": {
                "latitude": well.location.latitude,
                "longitude": well.location.longitude
            },
            "measured_depth": well.total_depth,
            "status": well.status,
            "well_type": well.well_type
        }
        
        if completion:
            summary["completion"] = {
                "type": completion.completion_type,
                "tubing_size": completion.tubing.outer_diameter,
                "perforated_interval": completion.total_perforated_interval,
                "artificial_lift": completion.artificial_lift_type
            }
        
        return summary
