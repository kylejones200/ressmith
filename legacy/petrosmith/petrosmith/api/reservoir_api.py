"""
Reservoir API - external interface for reservoir engineering.

This is the top-level API layer providing user-friendly interfaces.
"""

from typing import List, Dict, Optional
from petrosmith.models import Reservoir
from petrosmith.services import ReservoirService


class ReservoirAPI:
    """
    High-level API for reservoir engineering and analysis.
    
    This is the primary interface for users to interact with reservoir data.
    """
    
    def __init__(self):
        """Initialize Reservoir API with service layer."""
        self.service = ReservoirService()
    
    def create_reservoir(
        self,
        reservoir_id: str,
        reservoir_name: str,
        formation: str,
        top_depth: float,
        bottom_depth: float,
        net_pay: float,
        fluid_type: str,
        porosity: float,
        permeability: float,
        water_saturation: float,
        initial_pressure: float,
        temperature: float,
        drive_mechanism: str = "solution_gas"
    ) -> Reservoir:
        """
        Create a new reservoir.
        
        Args:
            reservoir_id: Unique reservoir identifier
            reservoir_name: Reservoir name
            formation: Geological formation name
            top_depth: Top of reservoir in feet TVD
            bottom_depth: Bottom of reservoir in feet TVD
            net_pay: Net pay thickness in feet
            fluid_type: Type of fluid (for service layer use)
            porosity: Porosity (fraction, 0-1)
            permeability: Permeability in millidarcies
            water_saturation: Water saturation (fraction, 0-1)
            initial_pressure: Initial reservoir pressure in psi
            temperature: Reservoir temperature in °F
            drive_mechanism: Drive mechanism type
            
        Returns:
            Reservoir object
            
        Example:
            >>> api = ReservoirAPI()
            >>> reservoir = api.create_reservoir(
            ...     reservoir_id="RES-001",
            ...     reservoir_name="Alpha Sand",
            ...     formation="Cretaceous Sandstone",
            ...     top_depth=8500,
            ...     bottom_depth=8600,
            ...     net_pay=80,
            ...     fluid_type="oil",
            ...     porosity=0.22,
            ...     permeability=150,
            ...     water_saturation=0.25,
            ...     initial_pressure=3500,
            ...     temperature=180
            ... )
        """
        reservoir = Reservoir(
            reservoir_id=reservoir_id,
            reservoir_name=reservoir_name,
            formation=formation,
            top_depth=top_depth,
            bottom_depth=bottom_depth,
            net_pay=net_pay,
            porosity=porosity,
            permeability=permeability,
            water_saturation=water_saturation,
            initial_pressure=initial_pressure,
            temperature=temperature,
            fluid_type=fluid_type,
            drive_mechanism=drive_mechanism
        )
        
        self.service.add_reservoir(reservoir)
        return reservoir
    
    def get_reservoir(self, reservoir_id: str) -> Optional[Reservoir]:
        """
        Retrieve a reservoir by ID.
        
        Args:
            reservoir_id: Reservoir identifier
            
        Returns:
            Reservoir object or None if not found
        """
        return self.service.get_reservoir(reservoir_id)
    
    def calculate_reserves(
        self,
        reservoir_id: str,
        area: float,
        net_pay: float,
        water_saturation: float,
        formation_volume_factor: float
    ) -> Dict:
        """
        Calculate hydrocarbon reserves.
        
        Args:
            reservoir_id: Reservoir identifier
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            water_saturation: Water saturation (fraction, 0-1)
            formation_volume_factor: Formation volume factor
            
        Returns:
            Dictionary with reserves calculation
            
        Example:
            >>> api = ReservoirAPI()
            >>> reserves = api.calculate_reserves(
            ...     "RES-001",
            ...     area=640,
            ...     net_pay=50,
            ...     water_saturation=0.25,
            ...     formation_volume_factor=1.2
            ... )
            >>> print(f"OOIP: {reserves['original_oil_in_place']:,.0f} STB")
            >>> print(f"Recoverable: {reserves['recoverable_reserves']:,.0f} STB")
        """
        return self.service.calculate_reserves(
            reservoir_id=reservoir_id,
            area=area,
            net_pay=net_pay,
            water_saturation=water_saturation,
            formation_volume_factor=formation_volume_factor
        )
    
    def calculate_ooip(
        self,
        area: float,
        net_pay: float,
        porosity: float,
        oil_saturation: float,
        formation_volume_factor: float
    ) -> float:
        """
        Calculate Original Oil In Place.
        
        Args:
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            porosity: Porosity (fraction)
            oil_saturation: Oil saturation (fraction)
            formation_volume_factor: Oil FVF (rb/STB)
            
        Returns:
            OOIP in stock tank barrels
            
        Example:
            >>> api = ReservoirAPI()
            >>> ooip = api.calculate_ooip(640, 50, 0.22, 0.75, 1.2)
            >>> print(f"OOIP: {ooip:,.0f} STB")
        """
        from petrosmith.core import ReservoirCalculations
        
        return ReservoirCalculations.calculate_original_oil_in_place(
            area=area,
            net_pay=net_pay,
            porosity=porosity,
            oil_saturation=oil_saturation,
            formation_volume_factor=formation_volume_factor
        )
    
    def calculate_ogip(
        self,
        area: float,
        net_pay: float,
        porosity: float,
        gas_saturation: float,
        formation_volume_factor: float
    ) -> float:
        """
        Calculate Original Gas In Place.
        
        Args:
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            porosity: Porosity (fraction)
            gas_saturation: Gas saturation (fraction)
            formation_volume_factor: Gas FVF (rcf/scf)
            
        Returns:
            OGIP in standard cubic feet
            
        Example:
            >>> api = ReservoirAPI()
            >>> ogip = api.calculate_ogip(640, 50, 0.20, 0.80, 0.005)
            >>> print(f"OGIP: {ogip:,.0f} SCF")
        """
        from petrosmith.core import ReservoirCalculations
        
        return ReservoirCalculations.calculate_original_gas_in_place(
            area=area,
            net_pay=net_pay,
            porosity=porosity,
            gas_saturation=gas_saturation,
            formation_volume_factor=formation_volume_factor
        )
    
    def analyze_well_deliverability(
        self,
        reservoir_id: str,
        well_id: str,
        drainage_radius: float,
        wellbore_radius: float,
        fluid_viscosity: float,
        fluid_fvf: float,
        skin_factor: float = 0.0
    ) -> Dict:
        """
        Analyze well deliverability from reservoir.
        
        Args:
            reservoir_id: Reservoir identifier
            well_id: Well identifier
            drainage_radius: Drainage radius in feet
            wellbore_radius: Wellbore radius in feet
            fluid_viscosity: Fluid viscosity in cp
            fluid_fvf: Formation volume factor
            skin_factor: Skin factor (dimensionless)
            
        Returns:
            Dictionary with deliverability analysis
            
        Example:
            >>> api = ReservoirAPI()
            >>> deliverability = api.analyze_well_deliverability(
            ...     "RES-001", "W-001", 1000, 0.328, 2.0, 1.2, skin_factor=5.0
            ... )
            >>> print(f"PI: {deliverability['productivity_index']:.2f} STB/day/psi")
        """
        return self.service.analyze_well_deliverability(
            reservoir_id=reservoir_id,
            well_id=well_id,
            drainage_radius=drainage_radius,
            wellbore_radius=wellbore_radius,
            fluid_viscosity=fluid_viscosity,
            fluid_fvf=fluid_fvf,
            skin_factor=skin_factor
        )
    
    def calculate_flow_rate(
        self,
        permeability: float,
        thickness: float,
        pressure_drawdown: float,
        viscosity: float,
        fvf: float,
        drainage_radius: float,
        wellbore_radius: float,
        skin_factor: float = 0.0
    ) -> float:
        """
        Calculate flow rate using Darcy's equation.
        
        Args:
            permeability: Permeability in md
            thickness: Net pay in feet
            pressure_drawdown: Pressure drawdown in psi
            viscosity: Fluid viscosity in cp
            fvf: Formation volume factor
            drainage_radius: Drainage radius in feet
            wellbore_radius: Wellbore radius in feet
            skin_factor: Skin factor
            
        Returns:
            Flow rate in STB/day or Mscf/day
            
        Example:
            >>> api = ReservoirAPI()
            >>> rate = api.calculate_flow_rate(
            ...     permeability=100,
            ...     thickness=50,
            ...     pressure_drawdown=500,
            ...     viscosity=2.0,
            ...     fvf=1.2,
            ...     drainage_radius=1000,
            ...     wellbore_radius=0.328
            ... )
            >>> print(f"Flow rate: {rate:.1f} STB/day")
        """
        from petrosmith.core import ReservoirCalculations
        
        return ReservoirCalculations.calculate_darcy_flow_rate(
            permeability=permeability,
            thickness=thickness,
            pressure_drawdown=pressure_drawdown,
            viscosity=viscosity,
            formation_volume_factor=fvf,
            drainage_radius=drainage_radius,
            wellbore_radius=wellbore_radius,
            skin_factor=skin_factor
        )
    
    def perform_material_balance(
        self,
        reservoir_id: str,
        cumulative_production: float,
        original_in_place: float
    ) -> Dict:
        """
        Perform material balance calculation.
        
        Args:
            reservoir_id: Reservoir identifier
            cumulative_production: Cumulative production
            original_in_place: Original hydrocarbons in place
            
        Returns:
            Dictionary with material balance results
            
        Example:
            >>> api = ReservoirAPI()
            >>> mb = api.perform_material_balance("RES-001", 1000000, 10000000)
            >>> print(f"Current pressure: {mb['current_pressure']:.0f} psi")
            >>> print(f"Recovery: {mb['recovery_percent']:.1f}%")
        """
        return self.service.perform_material_balance(
            reservoir_id=reservoir_id,
            cumulative_production=cumulative_production,
            original_in_place=original_in_place
        )

    def calculate_ogip_pz(
        self,
        initial_pressure: float,
        initial_z_factor: float,
        cumulative_gas_production: float,
        current_pressure: float,
        current_z_factor: float,
    ) -> float:
        """
        Calculate Original Gas In Place (OGIP) from p/Z material balance.

        For dry gas reservoirs: p/Z declines linearly with Gp; OGIP is
        G = Gp / (1 - (p/Z)/(pi/Zi)).

        Args:
            initial_pressure: Initial reservoir pressure in psi
            initial_z_factor: Gas Z-factor at initial pressure
            cumulative_gas_production: Cumulative gas produced in scf
            current_pressure: Current reservoir pressure in psi
            current_z_factor: Gas Z-factor at current pressure

        Returns:
            Original gas in place in scf

        Example:
            >>> api = ReservoirAPI()
            >>> ogip = api.calculate_ogip_pz(4000, 0.92, 2e9, 3200, 0.88)
        """
        from petrosmith.core import ReservoirCalculations
        return ReservoirCalculations.calculate_original_gas_in_place_pz(
            initial_pressure=initial_pressure,
            initial_z_factor=initial_z_factor,
            cumulative_gas_production=cumulative_gas_production,
            current_pressure=current_pressure,
            current_z_factor=current_z_factor,
        )

    def calculate_pressure_from_pz_gas_material_balance(
        self,
        initial_pressure: float,
        initial_z_factor: float,
        cumulative_gas_production: float,
        original_gas_in_place: float,
        current_z_factor: float,
    ) -> float:
        """
        Calculate current reservoir pressure from p/Z gas material balance.

        Uses p/Z = (pi/Zi)*(1 - Gp/G). For accuracy, provide current_z_factor
        from a Z(p) correlation at the estimated pressure (iterate if needed).

        Args:
            initial_pressure: Initial reservoir pressure in psi
            initial_z_factor: Gas Z-factor at initial pressure
            cumulative_gas_production: Cumulative gas produced in scf
            original_gas_in_place: Original gas in place in scf
            current_z_factor: Gas Z-factor at current pressure

        Returns:
            Current reservoir pressure in psi

        Example:
            >>> api = ReservoirAPI()
            >>> p = api.calculate_pressure_from_pz_gas_material_balance(
            ...     4000, 0.92, 2e9, 10e9, 0.88
            ... )
        """
        from petrosmith.core import ReservoirCalculations
        return ReservoirCalculations.calculate_pressure_from_pz_material_balance(
            initial_pressure=initial_pressure,
            initial_z_factor=initial_z_factor,
            cumulative_gas_production=cumulative_gas_production,
            original_gas_in_place=original_gas_in_place,
            current_z_factor=current_z_factor,
        )

    def estimate_reservoir_performance(
        self,
        reservoir_id: str,
        number_of_wells: int,
        well_spacing_acres: float
    ) -> Dict:
        """
        Estimate overall reservoir performance.
        
        Args:
            reservoir_id: Reservoir identifier
            number_of_wells: Number of producing wells
            well_spacing_acres: Well spacing in acres
            
        Returns:
            Dictionary with performance estimates
            
        Example:
            >>> api = ReservoirAPI()
            >>> performance = api.estimate_reservoir_performance(
            ...     "RES-001", number_of_wells=10, well_spacing_acres=80
            ... )
            >>> print(f"Total rate: {performance['production_estimates']['total_initial_rate']:.0f} STB/day")
        """
        return self.service.estimate_reservoir_performance(
            reservoir_id=reservoir_id,
            number_of_wells=number_of_wells,
            well_spacing_acres=well_spacing_acres
        )
    
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
            date: Measurement date (YYYY-MM-DD)
            pressure: Pressure in psi
            measurement_type: Type of measurement
            
        Example:
            >>> api = ReservoirAPI()
            >>> api.add_pressure_measurement("RES-001", "2026-01-20", 3200)
        """
        self.service.add_pressure_measurement(
            reservoir_id=reservoir_id,
            date=date,
            pressure=pressure,
            measurement_type=measurement_type
        )
