"""
Well API - external interface for well operations.

This is the top-level API layer providing user-friendly interfaces.
"""

from typing import List, Dict, Optional
from petrosmith.models import Well
from petrosmith.models.well import WellLocation
from petrosmith.services import WellService


class WellAPI:
    """
    High-level API for well management and analysis.
    
    This is the primary interface for users to interact with well data.
    """
    
    def __init__(self):
        """Initialize Well API with service layer."""
        self.service = WellService()
    
    def create_well(
        self,
        well_id: str,
        well_name: str,
        latitude: float,
        longitude: float,
        measured_depth: float,
        true_vertical_depth: float,
        operator: str = "Unknown",
        well_type: str = "vertical",
        status: str = "producing"
    ) -> Well:
        """
        Create a new well.
        
        Args:
            well_id: Unique well identifier
            well_name: Well name
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            measured_depth: Measured depth in feet
            true_vertical_depth: True vertical depth in feet
            operator: Operator name
            well_type: Type of well (vertical, horizontal, directional)
            status: Well status (producing, drilling, completed, etc.)
            
        Returns:
            Well object
            
        Example:
            >>> api = WellAPI()
            >>> well = api.create_well(
            ...     well_id="W-001",
            ...     well_name="Alpha-1",
            ...     latitude=29.7604,
            ...     longitude=-95.3698,
            ...     measured_depth=10000,
            ...     true_vertical_depth=9500
            ... )
        """
        well = Well(
            well_id=well_id,
            well_name=well_name,
            operator=operator,
            location=WellLocation(
                latitude=latitude,
                longitude=longitude,
                elevation=0.0  # Default elevation
            ),
            total_depth=measured_depth,
            target_formation="Unknown",
            well_type=well_type,
            status=status,
            spud_date=None
        )
        
        self.service.add_well(well)
        return well
    
    def get_well(self, well_id: str) -> Optional[Well]:
        """
        Retrieve a well by ID.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Well object or None if not found
        """
        return self.service.get_well(well_id)
    
    def list_wells(self) -> List[str]:
        """
        List all well IDs.
        
        Returns:
            List of well IDs
        """
        return self.service.list_wells()
    
    def get_well_summary(self, well_id: str) -> Dict:
        """
        Get comprehensive summary for a well.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dictionary with well summary
        """
        return self.service.get_well_summary(well_id)
    
    def analyze_productivity(
        self,
        well_id: str,
        reservoir_pressure: float,
        permeability: float,
        thickness: float,
        viscosity: float,
        fvf: float,
        **kwargs
    ) -> Dict:
        """
        Analyze well productivity.
        
        Args:
            well_id: Well identifier
            reservoir_pressure: Reservoir pressure in psi
            permeability: Permeability in md
            thickness: Net pay in feet
            viscosity: Fluid viscosity in cp
            fvf: Formation volume factor
            **kwargs: Additional parameters (drainage_radius, skin_factor)
            
        Returns:
            Dictionary with productivity analysis
            
        Example:
            >>> api = WellAPI()
            >>> analysis = api.analyze_productivity(
            ...     well_id="W-001",
            ...     reservoir_pressure=3000,
            ...     permeability=100,
            ...     thickness=50,
            ...     viscosity=2.0,
            ...     fvf=1.2,
            ...     skin_factor=5.0
            ... )
        """
        return self.service.analyze_well_productivity(
            well_id=well_id,
            reservoir_pressure=reservoir_pressure,
            reservoir_permeability=permeability,
            reservoir_thickness=thickness,
            fluid_viscosity=viscosity,
            fluid_fvf=fvf,
            drainage_radius=kwargs.get('drainage_radius', 1000.0),
            skin_factor=kwargs.get('skin_factor', 0.0)
        )
    
    def compare_lift_systems(
        self,
        well_id: str,
        depth: float,
        desired_rate: float,
        reservoir_pressure: float,
        wellhead_pressure: float = 100.0
    ) -> Dict:
        """
        Compare artificial lift system options.
        
        Args:
            well_id: Well identifier
            depth: Well depth in feet
            desired_rate: Desired production rate in STB/day
            reservoir_pressure: Reservoir pressure in psi
            wellhead_pressure: Required wellhead pressure in psi
            
        Returns:
            Dictionary comparing lift options
            
        Example:
            >>> api = WellAPI()
            >>> comparison = api.compare_lift_systems(
            ...     well_id="W-001",
            ...     depth=8000,
            ...     desired_rate=500,
            ...     reservoir_pressure=2500
            ... )
        """
        return self.service.compare_artificial_lift_options(
            well_id=well_id,
            depth=depth,
            desired_rate=desired_rate,
            reservoir_pressure=reservoir_pressure,
            wellhead_pressure=wellhead_pressure
        )
