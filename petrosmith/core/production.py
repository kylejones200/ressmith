"""
Production calculations - core business logic for production engineering.

Pure functions with no side effects.
Depends only on Layer 1 models.
"""

import math
from .constants import PhysicalConstants

from petrosmith.models import FluidProperties


class ProductionCalculations:
    """
    Core production engineering calculations.
    
    All methods are static/class methods with no state.
    """
    
    @staticmethod
    def calculate_inflow_performance_relationship(
        reservoir_pressure: float,
        productivity_index: float,
        max_flow_rate: float,
        exponent: float = 1.0
    ) -> dict[float, float]:
        """
        Calculate Inflow Performance Relationship (IPR) curve.
        
        Args:
            reservoir_pressure: Average reservoir pressure in psi
            productivity_index: Productivity index in STB/day/psi
            max_flow_rate: Maximum flow rate in STB/day
            exponent: Flow exponent (1.0 for linear, <1 for Vogel)
            
        Returns:
            Dictionary mapping bottomhole pressure to flow rate
        """
        if any(x <= 0 for x in [reservoir_pressure, productivity_index, max_flow_rate]):
            raise ValueError("All inputs must be positive")
        
        ipr_curve = {}
        
        # Generate pressure points
        for pwf_fraction in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            pwf = reservoir_pressure * pwf_fraction
            
            if exponent == 1.0:
                # Linear (straight line) IPR
                q = productivity_index * (reservoir_pressure - pwf)
            else:
                # Vogel IPR for solution gas drive
                q = max_flow_rate * (1 - 0.2 * (pwf / reservoir_pressure) - 
                                     0.8 * ((pwf / reservoir_pressure) ** 2))
            
            ipr_curve[pwf] = max(q, 0.0)
        
        return ipr_curve
    
    @staticmethod
    def calculate_tubing_performance_relationship(
        wellhead_pressure: float,
        depth: float,
        tubing_diameter: float,
        flow_rate: float,
        fluid_gradient: float = 0.433
    ) -> float:
        """
        Calculate bottomhole flowing pressure from wellhead conditions.
        
        Args:
            wellhead_pressure: Wellhead pressure in psi
            depth: Well depth in feet
            tubing_diameter: Tubing ID in inches
            flow_rate: Flow rate in STB/day
            fluid_gradient: Fluid gradient in psi/ft (default 0.433 for water)
            
        Returns:
            Bottomhole flowing pressure in psi
        """
        if any(x < 0 for x in [wellhead_pressure, depth, tubing_diameter, flow_rate]):
            raise ValueError("All inputs must be non-negative")
        if tubing_diameter <= 0:
            raise ValueError("Tubing diameter must be positive")
        
        # Hydrostatic pressure
        hydrostatic = fluid_gradient * depth
        
        # Friction loss (simplified)
        velocity = (flow_rate * 0.0119) / (tubing_diameter ** 2)  # ft/sec
        friction_factor = 0.02  # Approximate
        friction_loss = (friction_factor * velocity ** 2 * depth) / (2 * 32.2 * tubing_diameter / 12)
        
        # Total bottomhole pressure
        pwf = wellhead_pressure + hydrostatic + friction_loss
        
        return pwf
    
    @staticmethod
    def calculate_nodal_analysis_optimum_rate(
        ipr_curve: dict[float, float],
        wellhead_pressure: float,
        depth: float,
        tubing_diameter: float
    ) -> tuple[float, float]:
        """
        Find optimum production rate using nodal analysis.
        
        Args:
            ipr_curve: IPR curve as dict {pressure: flow_rate}
            wellhead_pressure: Wellhead pressure in psi
            depth: Well depth in feet
            tubing_diameter: Tubing ID in inches
            
        Returns:
            tuple of (optimum flow rate, operating pressure)
        """
        if not ipr_curve:
            raise ValueError("IPR curve cannot be empty")
        
        best_rate = 0.0
        best_pressure = 0.0
        
        # Find intersection of IPR and TPR curves
        for pwf, q_ipr in ipr_curve.items():
            # Calculate TPR for this flow rate
            pwf_tpr = ProductionCalculations.calculate_tubing_performance_relationship(
                wellhead_pressure, depth, tubing_diameter, q_ipr
            )
            
            # Check if curves intersect (within tolerance)
            if abs(pwf - pwf_tpr) < 50:  # 50 psi tolerance
                if q_ipr > best_rate:
                    best_rate = q_ipr
                    best_pressure = pwf
        
        return best_rate, best_pressure
    
    @staticmethod
    def calculate_gas_lift_performance(
        injection_rate: float,
        injection_pressure: float,
        operating_pressure: float,
        liquid_rate: float
    ) -> float:
        """
        Calculate gas lift efficiency.
        
        Args:
            injection_rate: Gas injection rate in Mscf/day
            injection_pressure: Injection pressure in psi
            operating_pressure: Operating valve pressure in psi
            liquid_rate: Liquid production rate in STB/day
            
        Returns:
            Gas lift efficiency (STB per Mscf)
        """
        if any(x < 0 for x in [injection_rate, injection_pressure, operating_pressure, liquid_rate]):
            raise ValueError("All inputs must be non-negative")
        
        if injection_rate == 0:
            return 0.0
        
        # Gas lift efficiency
        efficiency = liquid_rate / injection_rate
        
        # Pressure efficiency factor
        pressure_factor = min(operating_pressure / injection_pressure, 1.0)
        
        adjusted_efficiency = efficiency * pressure_factor
        
        return adjusted_efficiency
    
    @staticmethod
    def calculate_esp_required_head(
        depth: float,
        wellhead_pressure: float,
        flow_rate: float,
        fluid_specific_gravity: float = 0.85
    ) -> float:
        """
        Calculate required head for Electric Submersible Pump.
        
        Args:
            depth: Pump setting depth in feet
            wellhead_pressure: Required wellhead pressure in psi
            flow_rate: Flow rate in STB/day
            fluid_specific_gravity: Fluid SG (default 0.85)
            
        Returns:
            Required head in feet
        """
        if depth < 0 or wellhead_pressure < 0 or flow_rate < 0:
            raise ValueError("All inputs must be non-negative")
        if fluid_specific_gravity <= 0:
            raise ValueError("Fluid specific gravity must be positive")
        
        # Convert wellhead pressure to feet of fluid
        wellhead_head = (wellhead_pressure * 2.31) / fluid_specific_gravity
        
        # Friction losses in tubing (simplified)
        friction_head = flow_rate * 0.0001  # Rough estimate
        
        # Total dynamic head
        total_head = depth + wellhead_head + friction_head
        
        return total_head
    
    @staticmethod
    def calculate_esp_horsepower(
        flow_rate: float,
        total_head: float,
        efficiency: float = 0.70
    ) -> float:
        """
        Calculate ESP hydraulic horsepower.
        
        Args:
            flow_rate: Flow rate in STB/day
            total_head: Total dynamic head in feet
            efficiency: Pump efficiency (default 0.70)
            
        Returns:
            Required horsepower
        """
        if flow_rate < 0 or total_head < 0:
            raise ValueError("Flow rate and head must be non-negative")
        if not (0 < efficiency <= 1):
            raise ValueError("Efficiency must be between 0 and 1")
        
        # Hydraulic horsepower
        hydraulic_hp = (flow_rate * total_head * 0.85) / (3960 * efficiency)
        
        return hydraulic_hp
    
    @staticmethod
    def calculate_water_cut(
        water_production: float,
        oil_production: float
    ) -> float:
        """
        Calculate water cut percentage.
        
        Args:
            water_production: Water production rate in STB/day
            oil_production: Oil production rate in STB/day
            
        Returns:
            Water cut as percentage (0-100)
        """
        if water_production < 0 or oil_production < 0:
            raise ValueError("Production rates cannot be negative")
        
        total_liquid = water_production + oil_production
        
        if total_liquid == 0:
            return 0.0
        
        water_cut = (water_production / total_liquid) * 100
        
        return water_cut
    
    @staticmethod
    def calculate_gor(
        gas_production: float,
        oil_production: float
    ) -> float:
        """
        Calculate producing gas-oil ratio.
        
        Args:
            gas_production: Gas production rate in Mscf/day
            oil_production: Oil production rate in STB/day
            
        Returns:
            GOR in scf/STB
        """
        if gas_production < 0 or oil_production < 0:
            raise ValueError("Production rates cannot be negative")
        
        if oil_production == 0:
            return 0.0
        
        gor = (gas_production * 1000) / oil_production
        
        return gor
    
    @staticmethod
    def calculate_oil_production_index(
        current_rate: float,
        reservoir_pressure: float,
        bottomhole_pressure: float
    ) -> float:
        """
        Calculate current oil production index.
        
        Args:
            current_rate: Current production rate in STB/day
            reservoir_pressure: Average reservoir pressure in psi
            bottomhole_pressure: Bottomhole flowing pressure in psi
            
        Returns:
            Production index in STB/day/psi
        """
        if current_rate < 0:
            raise ValueError("Production rate cannot be negative")
        if reservoir_pressure <= bottomhole_pressure:
            raise ValueError("Reservoir pressure must be greater than bottomhole pressure")
        
        drawdown = reservoir_pressure - bottomhole_pressure
        
        production_index = current_rate / drawdown
        
        return production_index
    
    @staticmethod
    def calculate_decline_curve_exponential(
        initial_rate: float,
        decline_rate: float,
        time: float
    ) -> float:
        """
        Calculate production rate using exponential decline.
        
        Args:
            initial_rate: Initial production rate in STB/day
            decline_rate: Decline rate per year (decimal)
            time: Time in years
            
        Returns:
            Production rate at given time in STB/day
        """
        if initial_rate < 0:
            raise ValueError("Initial rate cannot be negative")
        if decline_rate < 0:
            raise ValueError("Decline rate cannot be negative")
        if time < 0:
            raise ValueError("Time cannot be negative")
        
        # Exponential decline: q(t) = qi * e^(-D*t)
        rate = initial_rate * math.exp(-decline_rate * time)
        
        return rate
    
    @staticmethod
    def calculate_cumulative_production_exponential(
        initial_rate: float,
        decline_rate: float,
        time: float
    ) -> float:
        """
        Calculate cumulative production using exponential decline.
        
        Args:
            initial_rate: Initial production rate in STB/day
            decline_rate: Decline rate per year (decimal)
            time: Time in years
            
        Returns:
            Cumulative production in STB
        """
        if initial_rate < 0:
            raise ValueError("Initial rate cannot be negative")
        if decline_rate <= 0:
            raise ValueError("Decline rate must be positive")
        if time < 0:
            raise ValueError("Time cannot be negative")
        
        # Cumulative: Np = (qi / D) * (1 - e^(-D*t))
        # Convert to days
        cumulative = (initial_rate * 365 / decline_rate) * (1 - math.exp(-decline_rate * time))
        
        return cumulative
