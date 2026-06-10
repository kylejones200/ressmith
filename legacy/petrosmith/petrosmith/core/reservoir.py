"""
Reservoir calculations - core business logic for reservoir engineering.

Pure functions with no side effects.
Depends only on Layer 1 models.
"""

import math
from petrosmith.models import Reservoir, FluidProperties
from .constants import PhysicalConstants


class ReservoirCalculations:
    """
    Core reservoir engineering calculations.
    
    All methods are static/class methods with no state.
    """
    
    @staticmethod
    def calculate_original_oil_in_place(
        area: float,
        net_pay: float,
        porosity: float,
        oil_saturation: float,
        formation_volume_factor: float
    ) -> float:
        """
        Calculate Original Oil In Place (OOIP) volumetrically.
        
        Args:
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            porosity: Porosity (fraction)
            oil_saturation: Oil saturation (fraction)
            formation_volume_factor: Oil FVF (rb/stb)
            
        Returns:
            OOIP in stock tank barrels
            
        Formula: OOIP = 7758 * A * h * φ * So / Bo
        """
        if any(x <= 0 for x in [area, net_pay, formation_volume_factor]):
            raise ValueError("Area, net pay, and FVF must be positive")
        if not (0 < porosity <= 1):
            raise ValueError("Porosity must be between 0 and 1")
        if not (0 <= oil_saturation <= 1):
            raise ValueError("Oil saturation must be between 0 and 1")
        
        ooip = (7758 * area * net_pay * porosity * oil_saturation) / formation_volume_factor
        
        return ooip
    
    @staticmethod
    def calculate_original_gas_in_place(
        area: float,
        net_pay: float,
        porosity: float,
        gas_saturation: float,
        formation_volume_factor: float
    ) -> float:
        """
        Calculate Original Gas In Place (OGIP).
        
        Args:
            area: Reservoir area in acres
            net_pay: Net pay thickness in feet
            porosity: Porosity (fraction)
            gas_saturation: Gas saturation (fraction)
            formation_volume_factor: Gas FVF (rcf/scf)
            
        Returns:
            OGIP in standard cubic feet
        """
        if any(x <= 0 for x in [area, net_pay, formation_volume_factor]):
            raise ValueError("Area, net pay, and FVF must be positive")
        if not (0 < porosity <= 1):
            raise ValueError("Porosity must be between 0 and 1")
        if not (0 <= gas_saturation <= 1):
            raise ValueError("Gas saturation must be between 0 and 1")
        
        ogip = (43560 * area * net_pay * porosity * gas_saturation) / formation_volume_factor
        
        return ogip
    
    @staticmethod
    def calculate_recovery_factor(
        initial_pressure: float,
        abandonment_pressure: float,
        drive_mechanism: str
    ) -> float:
        """
        Estimate recovery factor based on drive mechanism.
        
        Uses base recovery factors by drive type, adjusted by pressure depletion
        ratio. **Limitation:** This formula is most appropriate for depletion-
        dominated reservoirs. For strong water drive or gas cap, pressure may
        not drop much, so the pressure-ratio adjustment can underestimate
        recovery; use drive-mechanism-specific methods or analog data in
        those cases.
        
        Args:
            initial_pressure: Initial reservoir pressure in psi
            abandonment_pressure: Abandonment pressure in psi
            drive_mechanism: Type of drive mechanism
            
        Returns:
            Recovery factor (fraction)
        """
        if initial_pressure <= abandonment_pressure:
            raise ValueError("Initial pressure must be greater than abandonment pressure")
        
        # Typical recovery factors by drive mechanism
        recovery_factors = {
            "solution_gas": 0.20,
            "gas_cap": 0.40,
            "water_drive": 0.50,
            "gravity_drainage": 0.60,
            "combination": 0.45
        }
        
        base_rf = recovery_factors.get(drive_mechanism.lower(), 0.30)
        
        # Adjust for pressure depletion
        pressure_ratio = (initial_pressure - abandonment_pressure) / initial_pressure
        adjusted_rf = base_rf * pressure_ratio
        
        return min(adjusted_rf, 0.70)  # Cap at 70%
    
    @staticmethod
    def calculate_darcy_flow_rate(
        permeability: float,
        thickness: float,
        pressure_drawdown: float,
        viscosity: float,
        formation_volume_factor: float,
        drainage_radius: float,
        wellbore_radius: float,
        skin_factor: float = 0.0
    ) -> float:
        """
        Calculate flow rate using Darcy's equation for radial flow.
        
        Args:
            permeability: Permeability in millidarcies
            thickness: Net pay thickness in feet
            pressure_drawdown: Pressure drawdown (Pe - Pwf) in psi
            viscosity: Fluid viscosity in cp
            formation_volume_factor: FVF (rb/stb)
            drainage_radius: Drainage radius in feet
            wellbore_radius: Wellbore radius in feet
            skin_factor: Skin factor (dimensionless)
            
        Returns:
            Flow rate in STB/day (oil) or Mscf/day (gas)
        """
        if any(x <= 0 for x in [permeability, thickness, viscosity, formation_volume_factor]):
            raise ValueError("Permeability, thickness, viscosity, and FVF must be positive")
        if pressure_drawdown < 0:
            raise ValueError("Pressure drawdown cannot be negative")
        if drainage_radius <= wellbore_radius:
            raise ValueError("Drainage radius must be greater than wellbore radius")
        
        # Darcy's equation for radial flow
        numerator = 0.00708 * permeability * thickness * pressure_drawdown
        denominator = viscosity * formation_volume_factor * (math.log(drainage_radius / wellbore_radius) + skin_factor)
        
        flow_rate = numerator / denominator
        
        return flow_rate
    
    @staticmethod
    def calculate_productivity_index(
        flow_rate: float,
        reservoir_pressure: float,
        bottomhole_pressure: float
    ) -> float:
        """
        Calculate productivity index.
        
        Args:
            flow_rate: Flow rate in STB/day
            reservoir_pressure: Average reservoir pressure in psi
            bottomhole_pressure: Bottomhole flowing pressure in psi
            
        Returns:
            Productivity index in STB/day/psi
        """
        if flow_rate < 0:
            raise ValueError("Flow rate cannot be negative")
        if reservoir_pressure <= bottomhole_pressure:
            raise ValueError("Reservoir pressure must be greater than bottomhole pressure")
        
        drawdown = reservoir_pressure - bottomhole_pressure
        
        if drawdown == 0:
            return 0.0
        
        productivity_index = flow_rate / drawdown
        
        return productivity_index
    
    @staticmethod
    def calculate_material_balance_pressure(
        initial_pressure: float,
        cumulative_production: float,
        original_in_place: float,
        initial_fvf: float,
        current_fvf: float,
        initial_solution_gor: float = 0.0,
        cumulative_gas_production: float = 0.0,
        initial_gas_cap: float = 0.0,
        gas_expansion_factor: float = 1.0,
        water_influx: float = 0.0,
        connate_water_expansion: float = 0.0,
        rock_expansion: float = 0.0
    ) -> float:
        """
        Calculate reservoir pressure using general material balance equation.
        
        **Note:** This is a conceptual/simplified implementation. It does not
        iterate on pressure with pressure-dependent PVT (Bo(P), Bg(P), Rs(P)).
        For screening or when you have precomputed FVF terms, it is usable; for
        history matching or accurate pressure prediction use an iterative
        solution with PVT tables, or use calculate_material_balance_pressure_simple
        for compressibility-based screening.
        
        Implements the complete material balance equation accounting for:
        - Oil and gas production
        - Solution gas drive
        - Gas cap expansion
        - Water influx
        - Rock and water compressibility
        
        Args:
            initial_pressure: Initial reservoir pressure in psi
            cumulative_production: Cumulative oil production in STB
            original_in_place: Original oil in place in STB
            initial_fvf: Initial oil formation volume factor rb/STB
            current_fvf: Current oil formation volume factor rb/STB
            initial_solution_gor: Initial solution GOR scf/STB
            cumulative_gas_production: Cumulative gas production in scf
            initial_gas_cap: Initial gas cap size (ratio to oil volume)
            gas_expansion_factor: Gas expansion factor Bg/Bgi
            water_influx: Cumulative water influx in rb
            connate_water_expansion: Connate water expansion term
            rock_expansion: Formation compaction term
            
        Returns:
            Current reservoir pressure in psi
            
        Reference: Craft, B.C. and Hawkins, M.F., "Applied Petroleum Reservoir 
                   Engineering," Prentice-Hall, 1991.
        """
        if initial_pressure <= 0:
            raise ValueError("Initial pressure must be positive")
        if cumulative_production < 0:
            raise ValueError("Cumulative production cannot be negative")
        if original_in_place <= 0:
            raise ValueError("Original in place must be positive")
        if any(x <= 0 for x in [initial_fvf, current_fvf]):
            raise ValueError("Formation volume factors must be positive")
        
        # Material balance equation:
        # N[(Bo - Boi) + (Rsi - Rs)Bg + mBoi(Bg/Bgi - 1)] = NpBo + (Gp - GOR*Np)Bg - We + BwDeltaWp + cfVfDeltaP
        
        # Underground withdrawal
        # Oil expansion
        oil_expansion = cumulative_production * current_fvf
        
        # Solution gas released
        solution_gas_released = cumulative_production * (initial_solution_gor * current_fvf - cumulative_gas_production / cumulative_production if cumulative_production > 0 else 0)
        
        # Total underground withdrawal
        underground_withdrawal = oil_expansion + solution_gas_released
        
        # Expansion terms (drive mechanisms)
        # Oil expansion due to pressure drop
        oil_volume_change = original_in_place * (current_fvf - initial_fvf)
        
        # Solution gas expansion
        solution_gas_expansion = original_in_place * initial_solution_gor * (1.0 - 1.0)  # Simplified
        
        # Gas cap expansion (if present)
        gas_cap_expansion = 0.0
        if initial_gas_cap > 0:
            gas_cap_expansion = initial_gas_cap * original_in_place * initial_fvf * (gas_expansion_factor - 1.0)
        
        # Water influx and formation compaction
        total_expansion = oil_volume_change + solution_gas_expansion + gas_cap_expansion + water_influx + connate_water_expansion + rock_expansion
        
        # Iterative pressure calculation
        # This is simplified - full solution requires iteration
        if total_expansion == 0:
            # Depletion drive only
            recovery_fraction = cumulative_production / original_in_place
            pressure_drop = initial_pressure * recovery_fraction
        else:
            # Use ratio method
            pressure_ratio = 1.0 - (underground_withdrawal / (original_in_place * initial_fvf + total_expansion))
            pressure_drop = initial_pressure * (1.0 - pressure_ratio)
        
        current_pressure = initial_pressure - pressure_drop
        
        return max(current_pressure, 14.7)  # Minimum atmospheric pressure
    
    @staticmethod
    def calculate_material_balance_pressure_simple(
        initial_pressure: float,
        cumulative_production: float,
        original_in_place: float,
        compressibility: float
    ) -> float:
        """
        Calculate reservoir pressure using simplified material balance.
        
        Simplified version for solution gas drive reservoirs.
        Use full material balance method for accurate results.
        
        Args:
            initial_pressure: Initial reservoir pressure in psi
            cumulative_production: Cumulative production in STB
            original_in_place: Original oil in place in STB
            compressibility: Total compressibility in 1/psi
            
        Returns:
            Current reservoir pressure in psi
        """
        if initial_pressure <= 0:
            raise ValueError("Initial pressure must be positive")
        if cumulative_production < 0:
            raise ValueError("Cumulative production cannot be negative")
        if original_in_place <= 0:
            raise ValueError("Original in place must be positive")
        
        recovery_fraction = cumulative_production / original_in_place
        
        # Simplified material balance for compressible fluid
        pressure_drop = initial_pressure * recovery_fraction / (1 + compressibility * initial_pressure)
        
        current_pressure = initial_pressure - pressure_drop
        
        return max(current_pressure, 14.7)  # Minimum atmospheric pressure

    @staticmethod
    def calculate_original_gas_in_place_pz(
        initial_pressure: float,
        initial_z_factor: float,
        cumulative_gas_production: float,
        current_pressure: float,
        current_z_factor: float,
    ) -> float:
        """
        Calculate Original Gas In Place (OGIP) from p/Z material balance.

        For dry gas reservoirs: p/Z = (pi/Zi) * (1 - Gp/G). Solving for G:
        G = Gp / (1 - (p/Z) / (pi/Zi)).

        Args:
            initial_pressure: Initial reservoir pressure in psi
            initial_z_factor: Gas deviation factor Z at initial pressure
            cumulative_gas_production: Cumulative gas produced in scf
            current_pressure: Current reservoir pressure in psi
            current_z_factor: Gas deviation factor Z at current pressure

        Returns:
            Original gas in place in scf

        Reference: Craft, B.C. and Hawkins, M.F., "Applied Petroleum Reservoir
                   Engineering," Prentice-Hall, 1991. Dry gas material balance.
        """
        if initial_pressure <= 0 or initial_z_factor <= 0:
            raise ValueError("Initial pressure and Z factor must be positive")
        if current_pressure < 0 or current_z_factor <= 0:
            raise ValueError("Current pressure must be non-negative, Z positive")
        if cumulative_gas_production < 0:
            raise ValueError("Cumulative gas production cannot be negative")

        p_over_z_initial = initial_pressure / initial_z_factor
        p_over_z_current = current_pressure / current_z_factor

        if p_over_z_current >= p_over_z_initial:
            raise ValueError(
                "p/Z must decrease with production; check pressures and Z factors"
            )

        # G = Gp / (1 - (p/Z)/(pi/Zi))
        ratio = p_over_z_current / p_over_z_initial
        if ratio >= 1.0:
            raise ValueError("p/Z ratio must be < 1 for producing reservoir")
        ogip = cumulative_gas_production / (1.0 - ratio)
        return ogip

    @staticmethod
    def calculate_pressure_from_pz_material_balance(
        initial_pressure: float,
        initial_z_factor: float,
        cumulative_gas_production: float,
        original_gas_in_place: float,
        current_z_factor: float,
    ) -> float:
        """
        Calculate current reservoir pressure from p/Z material balance.

        For dry gas: p/Z = (pi/Zi) * (1 - Gp/G), so
        p = Z * (pi/Zi) * (1 - Gp/G).

        Z is pressure-dependent; this method uses the provided current_z_factor.
        For best accuracy, use a Z(p) correlation and iterate: guess p, get Z(p),
        compute p_new from the formula, repeat until converged.

        Args:
            initial_pressure: Initial reservoir pressure in psi
            initial_z_factor: Gas deviation factor at initial pressure
            cumulative_gas_production: Cumulative gas produced in scf
            original_gas_in_place: Original gas in place in scf
            current_z_factor: Gas deviation factor at current pressure (e.g. from
                correlation at estimated pressure, or from previous iteration)

        Returns:
            Current reservoir pressure in psi
        """
        if initial_pressure <= 0 or initial_z_factor <= 0:
            raise ValueError("Initial pressure and Z factor must be positive")
        if original_gas_in_place <= 0:
            raise ValueError("Original gas in place must be positive")
        if cumulative_gas_production < 0:
            raise ValueError("Cumulative gas production cannot be negative")
        if current_z_factor <= 0:
            raise ValueError("Current Z factor must be positive")
        if cumulative_gas_production >= original_gas_in_place:
            raise ValueError("Cumulative production cannot exceed original in place")

        p_over_z_initial = initial_pressure / initial_z_factor
        depletion_factor = 1.0 - (cumulative_gas_production / original_gas_in_place)
        p_over_z_current = p_over_z_initial * depletion_factor
        current_pressure = current_z_factor * p_over_z_current
        return max(current_pressure, PhysicalConstants.ATMOSPHERIC_PRESSURE)

    @staticmethod
    def calculate_permeability_from_pressure_buildup(
        thickness: float,
        viscosity: float,
        formation_volume_factor: float,
        flow_rate: float,
        pressure_buildup_slope: float
    ) -> float:
        """
        Calculate permeability from pressure buildup test.
        
        Args:
            thickness: Net pay thickness in feet
            viscosity: Fluid viscosity in cp
            formation_volume_factor: FVF
            flow_rate: Flow rate before shut-in in STB/day
            pressure_buildup_slope: Slope of Horner plot in psi/cycle
            
        Returns:
            Permeability in millidarcies
        """
        if any(x <= 0 for x in [thickness, viscosity, formation_volume_factor, flow_rate, pressure_buildup_slope]):
            raise ValueError("All inputs must be positive")
        
        # From Horner analysis
        permeability = (162.6 * flow_rate * viscosity * formation_volume_factor) / (thickness * pressure_buildup_slope)
        
        return permeability
    
    @staticmethod
    def calculate_skin_factor(
        shut_in_pressure: float,
        flowing_pressure_1hr: float,
        horner_slope: float,
        permeability: float,
        porosity: float,
        viscosity: float,
        total_compressibility: float,
        wellbore_radius: float,
        flow_time: float
    ) -> float:
        """
        Calculate skin factor from pressure buildup test data.
        
        Complete skin factor equation including logarithmic term.
        
        Args:
            shut_in_pressure: Extrapolated shut-in pressure in psi
            flowing_pressure_1hr: Flowing pressure at 1 hour in psi
            horner_slope: Horner plot slope in psi/cycle
            permeability: Permeability in md
            porosity: Porosity (fraction)
            viscosity: Viscosity in cp
            total_compressibility: Total compressibility in 1/psi
            wellbore_radius: Wellbore radius in feet
            flow_time: Flow time before shut-in in hours
            
        Returns:
            Skin factor (dimensionless)
            
        Reference: Matthews, C.S. and Russell, D.G., "Pressure Buildup and 
                   Flow Tests in Wells," SPE Monograph, 1967.
        """
        if any(x <= 0 for x in [horner_slope, permeability, porosity, viscosity, 
                                 total_compressibility, wellbore_radius, flow_time]):
            raise ValueError("All physical parameters must be positive")
        
        # Calculate Delta P at 1 hour
        delta_p_1hr = shut_in_pressure - flowing_pressure_1hr
        
        # Calculate skin using complete equation
        # S = 1.151 * [(P_1hr - P_wf)/m - log(k/(phi*mu*ct*rw^2)) + 3.23]
        
        # Logarithmic term
        log_term = math.log10(permeability / (porosity * viscosity * total_compressibility * (wellbore_radius ** 2)))
        
        # Complete skin factor equation
        skin = 1.151 * ((delta_p_1hr / horner_slope) - log_term + 3.23)
        
        return skin
    
    @staticmethod
    def calculate_skin_factor_from_pressures(
        actual_pressure: float,
        ideal_pressure: float,
        flow_rate: float,
        permeability: float,
        thickness: float,
        viscosity: float,
        formation_volume_factor: float
    ) -> float:
        """
        Calculate skin factor from steady-state pressure difference.
        
        Simplified method for quick estimates.
        
        Args:
            actual_pressure: Actual bottomhole pressure in psi
            ideal_pressure: Ideal bottomhole pressure (no skin) in psi
            flow_rate: Flow rate in STB/day
            permeability: Permeability in md
            thickness: Net pay thickness in feet
            viscosity: Viscosity in cp
            formation_volume_factor: FVF
            
        Returns:
            Skin factor (dimensionless)
        """
        if any(x <= 0 for x in [flow_rate, permeability, thickness, viscosity, formation_volume_factor]):
            raise ValueError("Flow rate, permeability, thickness, viscosity, and FVF must be positive")
        
        pressure_difference = actual_pressure - ideal_pressure
        
        # Skin factor from pressure difference
        skin = (0.00708 * permeability * thickness * pressure_difference) / (flow_rate * viscosity * formation_volume_factor)
        
        return skin
