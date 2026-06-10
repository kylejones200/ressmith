"""
Reservoir service - orchestrates reservoir engineering operations.

This service layer coordinates reservoir models and core calculations.
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Any
from petrosmith.models import Reservoir, FluidProperties
from petrosmith.core import ReservoirCalculations

logger = logging.getLogger(__name__)


class ReservoirService:
    """
    Service for managing reservoir engineering and analysis.
    
    Orchestrates reservoir data, calculations, and workflows.
    """
    
    def __init__(self):
        """Initialize reservoir service with data storage."""
        self.reservoirs: Dict[str, Reservoir] = {}
        self.pressure_data: Dict[str, List[Dict]] = {}
        self.production_history: Dict[str, List[Dict]] = {}
    
    def add_reservoir(self, reservoir: Reservoir) -> None:
        """
        Add a reservoir to the service.
        
        Args:
            reservoir: Reservoir domain model
        """
        self.reservoirs[reservoir.reservoir_id] = reservoir
    
    def get_reservoir(self, reservoir_id: str) -> Optional[Reservoir]:
        """
        Retrieve a reservoir by ID.
        
        Args:
            reservoir_id: Reservoir identifier
            
        Returns:
            Reservoir model or None
        """
        return self.reservoirs.get(reservoir_id)
    
    def add_pressure_measurement(
        self,
        reservoir_id: str,
        date: str,
        pressure: float,
        measurement_type: str = "average"
    ) -> None:
        """
        Add pressure measurement for a reservoir.
        
        Args:
            reservoir_id: Reservoir identifier
            date: Measurement date
            pressure: Pressure in psi
            measurement_type: Type of measurement (average, buildup, etc.)
        """
        if reservoir_id not in self.pressure_data:
            self.pressure_data[reservoir_id] = []
        
        self.pressure_data[reservoir_id].append({
            "date": date,
            "pressure": pressure,
            "type": measurement_type
        })
    
    def calculate_reserves(
        self,
        reservoir_id: str,
        area: float,
        net_pay: float,
        water_saturation: float,
        formation_volume_factor: float
    ) -> Dict[str, Any]:
        """
        Calculate hydrocarbon reserves for a reservoir.
        
        Args:
            reservoir_id: Reservoir identifier
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            water_saturation: Water saturation fraction
            formation_volume_factor: Formation volume factor
            
        Returns:
            Dictionary with reserves calculation
        """
        reservoir = self.get_reservoir(reservoir_id)
        if not reservoir:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        # Calculate OOIP or OGIP based on reservoir type
        if reservoir.fluid_type.lower() in ["oil", "black_oil"]:
            oil_saturation = 1 - water_saturation
            
            ooip = ReservoirCalculations.calculate_original_oil_in_place(
                area=area,
                net_pay=net_pay,
                porosity=reservoir.porosity,
                oil_saturation=oil_saturation,
                formation_volume_factor=formation_volume_factor
            )
            
            # Estimate recovery factor
            recovery_factor = ReservoirCalculations.calculate_recovery_factor(
                initial_pressure=reservoir.initial_pressure,
                abandonment_pressure=reservoir.initial_pressure * 0.1,  # 10% of initial
                drive_mechanism=reservoir.drive_mechanism
            )
            
            recoverable_reserves = ooip * recovery_factor
            
            return {
                "reservoir_id": reservoir_id,
                "reservoir_type": "oil",
                "original_oil_in_place": ooip,
                "recovery_factor": recovery_factor,
                "recoverable_reserves": recoverable_reserves,
                "drive_mechanism": reservoir.drive_mechanism,
                "units": "STB"
            }
        
        else:  # Gas reservoir
            gas_saturation = 1 - water_saturation
            
            ogip = ReservoirCalculations.calculate_original_gas_in_place(
                area=area,
                net_pay=net_pay,
                porosity=reservoir.porosity,
                gas_saturation=gas_saturation,
                formation_volume_factor=formation_volume_factor
            )
            
            # Gas reservoirs typically have higher recovery factors
            recovery_factor = 0.80  # Typical for gas
            recoverable_reserves = ogip * recovery_factor
            
            return {
                "reservoir_id": reservoir_id,
                "reservoir_type": "gas",
                "original_gas_in_place": ogip,
                "recovery_factor": recovery_factor,
                "recoverable_reserves": recoverable_reserves,
                "units": "SCF"
            }
    
    def analyze_well_deliverability(
        self,
        reservoir_id: str,
        well_id: str,
        drainage_radius: float,
        wellbore_radius: float,
        fluid_viscosity: float,
        fluid_fvf: float,
        skin_factor: float = 0.0
    ) -> Dict[str, Any]:
        """
        Analyze well deliverability from reservoir.
        
        Args:
            reservoir_id: Reservoir identifier
            well_id: Well identifier
            drainage_radius: Drainage radius in feet
            wellbore_radius: Wellbore radius in feet
            fluid_viscosity: Fluid viscosity in cp
            fluid_fvf: Formation volume factor
            skin_factor: Skin factor
            
        Returns:
            Dictionary with deliverability analysis
        """
        reservoir = self.get_reservoir(reservoir_id)
        if not reservoir:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        # Get current reservoir pressure
        if reservoir_id in self.pressure_data and self.pressure_data[reservoir_id]:
            current_pressure = self.pressure_data[reservoir_id][-1]["pressure"]
        else:
            current_pressure = reservoir.initial_pressure
        
        # Calculate flow rates for different drawdowns
        ipr_data = []
        
        for drawdown_fraction in [0.1, 0.2, 0.3, 0.4, 0.5]:
            drawdown = current_pressure * drawdown_fraction
            bottomhole_pressure = current_pressure - drawdown
            
            flow_rate = ReservoirCalculations.calculate_darcy_flow_rate(
                permeability=reservoir.permeability,
                thickness=reservoir.net_pay,
                pressure_drawdown=drawdown,
                viscosity=fluid_viscosity,
                formation_volume_factor=fluid_fvf,
                drainage_radius=drainage_radius,
                wellbore_radius=wellbore_radius,
                skin_factor=skin_factor
            )
            
            ipr_data.append({
                "drawdown": drawdown,
                "bottomhole_pressure": bottomhole_pressure,
                "flow_rate": flow_rate
            })
        
        # Calculate productivity index at 30% drawdown
        reference_point = ipr_data[2]  # 30% drawdown
        productivity_index = ReservoirCalculations.calculate_productivity_index(
            flow_rate=reference_point["flow_rate"],
            reservoir_pressure=current_pressure,
            bottomhole_pressure=reference_point["bottomhole_pressure"]
        )
        
        return {
            "reservoir_id": reservoir_id,
            "well_id": well_id,
            "current_pressure": current_pressure,
            "productivity_index": productivity_index,
            "skin_factor": skin_factor,
            "ipr_curve": ipr_data,
            "max_theoretical_rate": ipr_data[-1]["flow_rate"],
            "permeability": reservoir.permeability,
            "drainage_area_acres": (math.pi * drainage_radius ** 2) / 43560
        }
    
    def perform_material_balance(
        self,
        reservoir_id: str,
        cumulative_production: float,
        original_in_place: float
    ) -> Dict[str, Any]:
        """
        Perform material balance calculation.
        
        Args:
            reservoir_id: Reservoir identifier
            cumulative_production: Cumulative production in STB or SCF
            original_in_place: Original hydrocarbons in place
            
        Returns:
            Dictionary with material balance results
        """
        reservoir = self.get_reservoir(reservoir_id)
        if not reservoir:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        # Estimate compressibility
        if reservoir.fluid_type.lower() in ["oil", "black_oil"]:
            compressibility = 1e-5  # Typical oil compressibility, 1/psi
        else:
            compressibility = 1e-4  # Typical gas compressibility
        
        # Calculate current pressure (simplified compressibility-based MB)
        current_pressure = ReservoirCalculations.calculate_material_balance_pressure_simple(
            initial_pressure=reservoir.initial_pressure,
            cumulative_production=cumulative_production,
            original_in_place=original_in_place,
            compressibility=compressibility
        )
        
        # Calculate recovery factor
        recovery_factor = cumulative_production / original_in_place if original_in_place > 0 else 0
        
        # Calculate remaining reserves
        remaining_reserves = original_in_place - cumulative_production
        
        return {
            "reservoir_id": reservoir_id,
            "initial_pressure": reservoir.initial_pressure,
            "current_pressure": current_pressure,
            "pressure_decline": reservoir.initial_pressure - current_pressure,
            "original_in_place": original_in_place,
            "cumulative_production": cumulative_production,
            "remaining_reserves": remaining_reserves,
            "recovery_factor": recovery_factor,
            "recovery_percent": recovery_factor * 100
        }
    
    def estimate_reservoir_performance(
        self,
        reservoir_id: str,
        number_of_wells: int,
        well_spacing_acres: float
    ) -> Dict[str, Any]:
        """
        Estimate overall reservoir performance.
        
        Args:
            reservoir_id: Reservoir identifier
            number_of_wells: Number of producing wells
            well_spacing_acres: Well spacing in acres
            
        Returns:
            Dictionary with performance estimates
        """
        reservoir = self.get_reservoir(reservoir_id)
        if not reservoir:
            raise ValueError(f"Reservoir {reservoir_id} not found")
        
        # Calculate drainage area per well
        drainage_radius = ((well_spacing_acres * 43560) / math.pi) ** 0.5
        
        # Estimate per-well production (simplified)
        typical_pi = reservoir.permeability * reservoir.net_pay / 1000  # Rough estimate
        drawdown = reservoir.initial_pressure * 0.3  # 30% drawdown
        estimated_rate_per_well = typical_pi * drawdown
        
        # Total reservoir production
        total_production_capacity = estimated_rate_per_well * number_of_wells
        
        # Estimate plateau period
        if reservoir.fluid_type.lower() in ["oil", "black_oil"]:
            plateau_years = max(2, min(10, number_of_wells / 10))
        else:
            plateau_years = max(3, min(15, number_of_wells / 5))
        
        return {
            "reservoir_id": reservoir_id,
            "development_plan": {
                "number_of_wells": number_of_wells,
                "well_spacing_acres": well_spacing_acres,
                "drainage_radius_ft": drainage_radius
            },
            "production_estimates": {
                "rate_per_well": estimated_rate_per_well,
                "total_initial_rate": total_production_capacity,
                "plateau_duration_years": plateau_years
            },
            "reservoir_properties": {
                "permeability": reservoir.permeability,
                "net_pay": reservoir.net_pay,
                "porosity": reservoir.porosity,
                "initial_pressure": reservoir.initial_pressure
            }
        }
