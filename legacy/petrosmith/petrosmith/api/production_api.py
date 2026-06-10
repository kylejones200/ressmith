"""
Production API - external interface for production operations.

This is the top-level API layer providing user-friendly interfaces.
"""

from typing import List, Dict, Optional
from petrosmith.models import FluidProperties
from petrosmith.services import ProductionService


class ProductionAPI:
    """
    High-level API for production engineering and optimization.
    
    This is the primary interface for users to interact with production data.
    """
    
    def __init__(self):
        """Initialize Production API with service layer."""
        self.service = ProductionService()
    
    def add_production_data(
        self,
        well_id: str,
        date: str,
        oil_rate: float,
        gas_rate: float,
        water_rate: float
    ) -> None:
        """
        Add daily production data for a well.
        
        Args:
            well_id: Well identifier
            date: Production date (YYYY-MM-DD)
            oil_rate: Oil rate in STB/day
            gas_rate: Gas rate in Mscf/day
            water_rate: Water rate in STB/day
            
        Example:
            >>> api = ProductionAPI()
            >>> api.add_production_data("W-001", "2026-01-20", 500, 250, 100)
        """
        self.service.add_production_data(
            well_id=well_id,
            date=date,
            oil_rate=oil_rate,
            gas_rate=gas_rate,
            water_rate=water_rate
        )
    
    def analyze_well_performance(self, well_id: str) -> Dict:
        """
        Analyze well production performance.
        
        Args:
            well_id: Well identifier
            
        Returns:
            Dictionary with performance metrics
            
        Example:
            >>> api = ProductionAPI()
            >>> performance = api.analyze_well_performance("W-001")
            >>> print(f"Water cut: {performance['performance_indicators']['water_cut']:.1f}%")
        """
        return self.service.analyze_well_performance(well_id)
    
    def forecast_production(
        self,
        well_id: str,
        forecast_years: int = 5,
        use_dca: bool = True,
        model: str = "arps",
        kind: str = "hyperbolic",
    ) -> Dict:
        """
        Forecast future production using decline curve analysis.

        Uses the `decline-curve <https://pypi.org/project/decline-curve/>`_
        library when installed (``pip install petrosmith[dca]``) for Arps
        exponential/hyperbolic/harmonic or other models. Otherwise uses
        built-in exponential decline.

        Args:
            well_id: Well identifier
            forecast_years: Number of years to forecast (default 5)
            use_dca: If True (default), use decline-curve when available
            model: DCA model: 'arps', 'arima', 'timesfm', 'chronos'
            kind: Arps decline type: 'exponential', 'harmonic', 'hyperbolic'

        Returns:
            Dictionary with production forecast (forecast, current_rate,
            decline_rate_annual, economic_limit; plus dca_params when using DCA)

        Example:
            >>> api = ProductionAPI()
            >>> forecast = api.forecast_production("W-001", forecast_years=10)
            >>> for year_data in forecast['forecast']:
            ...     print(f"Year {year_data['year']}: {year_data['rate']:.1f} STB/day")
            >>> # Use hyperbolic Arps via decline-curve (install petrosmith[dca])
            >>> forecast = api.forecast_production("W-001", kind="hyperbolic")
        """
        return self.service.forecast_production(
            well_id=well_id,
            forecast_years=forecast_years,
            use_dca=use_dca,
            model=model,
            kind=kind,
        )
    
    def calculate_water_cut(
        self,
        water_rate: float,
        oil_rate: float
    ) -> float:
        """
        Calculate water cut percentage.
        
        Args:
            water_rate: Water production rate in STB/day
            oil_rate: Oil production rate in STB/day
            
        Returns:
            Water cut as percentage (0-100)
            
        Example:
            >>> api = ProductionAPI()
            >>> wc = api.calculate_water_cut(100, 400)
            >>> print(f"Water cut: {wc:.1f}%")
        """
        from petrosmith.core import ProductionCalculations
        
        return ProductionCalculations.calculate_water_cut(
            water_production=water_rate,
            oil_production=oil_rate
        )
    
    def calculate_gor(
        self,
        gas_rate: float,
        oil_rate: float
    ) -> float:
        """
        Calculate gas-oil ratio.
        
        Args:
            gas_rate: Gas production rate in Mscf/day
            oil_rate: Oil production rate in STB/day
            
        Returns:
            GOR in scf/STB
            
        Example:
            >>> api = ProductionAPI()
            >>> gor = api.calculate_gor(250, 500)
            >>> print(f"GOR: {gor:.0f} scf/STB")
        """
        from petrosmith.core import ProductionCalculations
        
        return ProductionCalculations.calculate_gor(
            gas_production=gas_rate,
            oil_production=oil_rate
        )
    
    def optimize_artificial_lift(
        self,
        well_id: str,
        reservoir_pressure: float,
        depth: float,
        desired_rate: float,
        fluid_density: float
    ) -> Dict:
        """
        Optimize artificial lift system selection.
        
        Args:
            well_id: Well identifier
            reservoir_pressure: Reservoir pressure in psi
            depth: Well depth in feet
            desired_rate: Desired production rate in STB/day
            fluid_density: Fluid density in ppg
            
        Returns:
            Dictionary with lift system recommendations
            
        Example:
            >>> api = ProductionAPI()
            >>> optimization = api.optimize_artificial_lift(
            ...     "W-001", 2500, 8000, 500, 7.2
            ... )
            >>> print(f"Recommended: {optimization['recommended_system']}")
        """
        # Create fluid properties
        fluid = FluidProperties(
            fluid_id="temp",
            fluid_type="oil",
            density=fluid_density,
            viscosity=2.0,  # Default
            temperature=150.0  # Default
        )
        
        return self.service.optimize_artificial_lift(
            well_id=well_id,
            reservoir_pressure=reservoir_pressure,
            depth=depth,
            desired_rate=desired_rate,
            fluid_properties=fluid
        )
    
    def calculate_esp_requirements(
        self,
        depth: float,
        flow_rate: float,
        wellhead_pressure: float = 100.0,
        fluid_specific_gravity: float = 0.85
    ) -> Dict:
        """
        Calculate ESP pump requirements.
        
        Args:
            depth: Pump setting depth in feet
            flow_rate: Desired flow rate in STB/day
            wellhead_pressure: Required wellhead pressure in psi
            fluid_specific_gravity: Fluid specific gravity (default 0.85)
            
        Returns:
            Dictionary with ESP requirements
            
        Example:
            >>> api = ProductionAPI()
            >>> esp = api.calculate_esp_requirements(8000, 500)
            >>> print(f"Required head: {esp['head']:.0f} ft")
            >>> print(f"Required HP: {esp['horsepower']:.1f}")
        """
        from petrosmith.core import ProductionCalculations
        
        head = ProductionCalculations.calculate_esp_required_head(
            depth=depth,
            wellhead_pressure=wellhead_pressure,
            flow_rate=flow_rate,
            fluid_specific_gravity=fluid_specific_gravity
        )
        
        horsepower = ProductionCalculations.calculate_esp_horsepower(
            flow_rate=flow_rate,
            total_head=head,
            efficiency=0.70
        )
        
        return {
            "depth": depth,
            "flow_rate": flow_rate,
            "head": head,
            "horsepower": horsepower,
            "estimated_power_cost_per_day": horsepower * 24 * 0.10  # $0.10/kWh
        }
    
    def analyze_nodal(
        self,
        reservoir_pressure: float,
        productivity_index: float,
        wellhead_pressure: float,
        depth: float,
        tubing_diameter: float
    ) -> Dict:
        """
        Perform nodal analysis to find optimum production rate.
        
        Args:
            reservoir_pressure: Reservoir pressure in psi
            productivity_index: Productivity index in STB/day/psi
            wellhead_pressure: Wellhead pressure in psi
            depth: Well depth in feet
            tubing_diameter: Tubing ID in inches
            
        Returns:
            Dictionary with nodal analysis results
            
        Example:
            >>> api = ProductionAPI()
            >>> nodal = api.analyze_nodal(3000, 2.5, 100, 8000, 2.441)
            >>> print(f"Optimum rate: {nodal['optimum_rate']:.0f} STB/day")
        """
        from petrosmith.core import ProductionCalculations
        
        # Generate IPR curve
        max_rate = productivity_index * reservoir_pressure
        ipr_curve = ProductionCalculations.calculate_inflow_performance_relationship(
            reservoir_pressure=reservoir_pressure,
            productivity_index=productivity_index,
            max_flow_rate=max_rate,
            exponent=1.0  # Linear IPR
        )
        
        # Find optimum rate
        optimum_rate, operating_pressure = ProductionCalculations.calculate_nodal_analysis_optimum_rate(
            ipr_curve=ipr_curve,
            wellhead_pressure=wellhead_pressure,
            depth=depth,
            tubing_diameter=tubing_diameter
        )
        
        return {
            "reservoir_pressure": reservoir_pressure,
            "productivity_index": productivity_index,
            "optimum_rate": optimum_rate,
            "operating_pressure": operating_pressure,
            "ipr_curve": ipr_curve
        }
