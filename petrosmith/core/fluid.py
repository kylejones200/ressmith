"""
Fluid property calculations - core business logic for fluid properties.

Pure functions with no side effects.
Depends only on Layer 1 models.
"""

import math
from petrosmith.models import FluidProperties
from .constants import PhysicalConstants


class FluidCalculations:
    """
    Core fluid property calculations using standard correlations.
    
    All methods are static/class methods with no state.
    """
    
    @staticmethod
    def calculate_oil_fvf_standing(
        gas_oil_ratio: float,
        gas_specific_gravity: float,
        oil_api_gravity: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Calculate oil formation volume factor using Standing correlation.
        
        Args:
            gas_oil_ratio: Solution GOR in scf/STB
            gas_specific_gravity: Gas specific gravity (air=1)
            oil_api_gravity: Oil API gravity
            temperature: Temperature in °F
            pressure: Pressure in psi
            
        Returns:
            Oil FVF in rb/STB
        """
        if any(x <= 0 for x in [gas_oil_ratio, gas_specific_gravity, oil_api_gravity, temperature, pressure]):
            raise ValueError("All inputs must be positive")
        
        # Standing correlation
        F = (gas_oil_ratio * (gas_specific_gravity / oil_api_gravity) ** 0.5) + (1.25 * temperature)
        
        Bo = 0.9759 + 0.00012 * (F ** 1.2)
        
        return Bo
    
    @staticmethod
    def calculate_gas_fvf(
        pressure: float,
        temperature: float,
        gas_specific_gravity: float = 0.65
    ) -> float:
        """
        Calculate gas formation volume factor.
        
        Args:
            pressure: Pressure in psi
            temperature: Temperature in °F
            gas_specific_gravity: Gas specific gravity (default 0.65)
            
        Returns:
            Gas FVF in rcf/scf
        """
        if pressure <= 0 or temperature <= 0:
            raise ValueError("Pressure and temperature must be positive")
        if gas_specific_gravity <= 0:
            raise ValueError("Gas specific gravity must be positive")
        
        # Real gas law with Z-factor approximation
        z_factor = FluidCalculations.calculate_z_factor(pressure, temperature, gas_specific_gravity)
        
        Bg = 0.0283 * z_factor * (temperature + PhysicalConstants.RANKINE_OFFSET) / pressure
        
        return Bg
    
    @staticmethod
    def calculate_z_factor(
        pressure: float,
        temperature: float,
        gas_specific_gravity: float
    ) -> float:
        """
        Calculate gas compressibility factor (Z-factor) using Dranchuk-Abou-Kassem correlation.
        
        This is an explicit equation that closely approximates the Standing-Katz chart.
        
        Args:
            pressure: Pressure in psi
            temperature: Temperature in °F
            gas_specific_gravity: Gas specific gravity
            
        Returns:
            Z-factor (dimensionless)
            
        Reference: Dranchuk, P.M. and Abou-Kassem, J.H., "Calculation of Z-Factors 
                   for Natural Gases Using Equations of State," JCPT, July-Sept 1975.
        """
        if any(x <= 0 for x in [pressure, temperature, gas_specific_gravity]):
            raise ValueError("All inputs must be positive")
        
        # Calculate pseudo-critical properties using Sutton correlation
        Tpc = 168 + 325 * gas_specific_gravity - 12.5 * (gas_specific_gravity ** 2)
        Ppc = 677 + 15 * gas_specific_gravity - 37.5 * (gas_specific_gravity ** 2)
        
        # Calculate pseudo-reduced properties
        Tpr = (temperature + PhysicalConstants.RANKINE_OFFSET) / Tpc
        Ppr = pressure / Ppc
        
        # Dranchuk-Abou-Kassem correlation constants
        A1 = 0.3265
        A2 = -1.0700
        A3 = -0.5339
        A4 = 0.01569
        A5 = -0.05165
        A6 = 0.5475
        A7 = -0.7361
        A8 = 0.1844
        A9 = 0.1056
        A10 = 0.6134
        A11 = 0.7210
        
        # Iterative solution using Newton-Raphson
        # Initial guess
        z = 1.0
        
        for iteration in range(20):  # Maximum 20 iterations
            rho_r = 0.27 * Ppr / (z * Tpr)  # Reduced density
            
            # Calculate function value
            t1 = A1 + A2/Tpr + A3/(Tpr**3) + A4/(Tpr**4) + A5/(Tpr**5)
            t2 = A6 + A7/Tpr + A8/(Tpr**2)
            t3 = A9 * (A7/Tpr + A8/(Tpr**2))
            t4 = A10 * (1 + A11 * rho_r**2) * (rho_r**2 / Tpr**3) * math.exp(-A11 * rho_r**2)
            
            f = z - (1 + t1 * rho_r + t2 * rho_r**2 - t3 * rho_r**5 + t4)
            
            # Calculate derivative
            drho_dz = -rho_r / z
            df_dz = 1 - drho_dz * (t1 + 2*t2*rho_r - 5*t3*rho_r**4 + 
                                   2*A10*rho_r*(1 + A11*rho_r**2 - (A11*rho_r**2)**2)/(Tpr**3) * 
                                   math.exp(-A11*rho_r**2))
            
            # Newton-Raphson update
            z_new = z - f / df_dz
            
            # Check convergence
            if abs(z_new - z) < 1e-6:
                return max(0.2, min(z_new, 2.0))  # Bound Z-factor to physical range
            
            z = z_new
        
        # If not converged, return last estimate
        return max(0.2, min(z, 2.0))
    
    @staticmethod
    def calculate_oil_viscosity_beggs_robinson(
        oil_api_gravity: float,
        temperature: float,
        pressure: float = 0.0,
        solution_gor: float = 0.0
    ) -> float:
        """
        Calculate oil viscosity using Beggs-Robinson correlation.
        
        Args:
            oil_api_gravity: Oil API gravity
            temperature: Temperature in °F
            pressure: Pressure in psi (for saturated oil)
            solution_gor: Solution GOR in scf/STB
            
        Returns:
            Oil viscosity in cp
        """
        if oil_api_gravity <= 0 or temperature <= 0:
            raise ValueError("API gravity and temperature must be positive")
        
        # Dead oil viscosity
        X = (10 ** (0.43 + 8.33 / oil_api_gravity)) * (temperature ** -1.163)
        
        viscosity_dead = (10 ** X) - 1
        
        # If solution GOR provided, adjust for dissolved gas
        if solution_gor > 0:
            A = 10.715 * ((solution_gor + 100) ** -0.515)
            B = 5.44 * ((solution_gor + 150) ** -0.338)
            
            viscosity = A * (viscosity_dead ** B)
        else:
            viscosity = viscosity_dead
        
        return viscosity
    
    @staticmethod
    def calculate_gas_viscosity_lee(
        gas_specific_gravity: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Calculate gas viscosity using Lee correlation.
        
        Args:
            gas_specific_gravity: Gas specific gravity
            temperature: Temperature in °F
            pressure: Pressure in psi
            
        Returns:
            Gas viscosity in cp
        """
        if any(x <= 0 for x in [gas_specific_gravity, temperature, pressure]):
            raise ValueError("All inputs must be positive")
        
        # Gas density
        gas_density = pressure * gas_specific_gravity * 28.97 / (10.73 * (temperature + PhysicalConstants.RANKINE_OFFSET))
        
        # Molecular weight
        M = 28.97 * gas_specific_gravity
        
        # Lee correlation
        K = (9.4 + 0.02 * M) * ((temperature + PhysicalConstants.RANKINE_OFFSET) ** 1.5) / (209 + 19 * M + (temperature + PhysicalConstants.RANKINE_OFFSET))
        X = 3.5 + 986 / (temperature + PhysicalConstants.RANKINE_OFFSET) + 0.01 * M
        Y = 2.4 - 0.2 * X
        
        viscosity = K * 1e-4 * math.exp(X * (gas_density ** Y))
        
        return viscosity
    
    @staticmethod
    def calculate_bubble_point_pressure_standing(
        gas_oil_ratio: float,
        gas_specific_gravity: float,
        oil_api_gravity: float,
        temperature: float
    ) -> float:
        """
        Calculate bubble point pressure using Standing correlation.
        
        Args:
            gas_oil_ratio: Solution GOR in scf/STB
            gas_specific_gravity: Gas specific gravity
            oil_api_gravity: Oil API gravity
            temperature: Temperature in °F
            
        Returns:
            Bubble point pressure in psi
        """
        if any(x <= 0 for x in [gas_oil_ratio, gas_specific_gravity, oil_api_gravity, temperature]):
            raise ValueError("All inputs must be positive")
        
        # Standing correlation
        Pb = 18.2 * ((gas_oil_ratio / gas_specific_gravity) ** 0.83 * 
                     (10 ** (0.00091 * temperature - 0.0125 * oil_api_gravity)) - 1.4)
        
        return max(Pb, 14.7)  # Minimum atmospheric pressure
    
    @staticmethod
    def calculate_solution_gor_standing(
        pressure: float,
        temperature: float,
        gas_specific_gravity: float,
        oil_api_gravity: float
    ) -> float:
        """
        Calculate solution gas-oil ratio using Standing correlation.
        
        Args:
            pressure: Pressure in psi
            temperature: Temperature in °F
            gas_specific_gravity: Gas specific gravity
            oil_api_gravity: Oil API gravity
            
        Returns:
            Solution GOR in scf/STB
        """
        if any(x <= 0 for x in [pressure, temperature, gas_specific_gravity, oil_api_gravity]):
            raise ValueError("All inputs must be positive")
        
        # Standing correlation
        Rs = gas_specific_gravity * ((pressure / 18.2 + 1.4) * 
                                     (10 ** (0.0125 * oil_api_gravity - 0.00091 * temperature))) ** 1.2048
        
        return Rs

    @staticmethod
    def calculate_oil_fvf_at_pressure(
        pressure: float,
        temperature: float,
        gas_specific_gravity: float,
        oil_api_gravity: float,
        bubble_point_pressure: float | None = None,
        oil_compressibility_above_pb: float = 1e-5,
    ) -> float:
        """
        Oil formation volume factor as a function of pressure for MB iteration.

        Below bubble point: Bo(P) = Bo(Rs(P), T, ...) via Standing.
        Above bubble point: Bo(P) = Bo(Pb) * (1 + co * (P - Pb)).

        Args:
            pressure: Pressure in psi
            temperature: Temperature in °F
            gas_specific_gravity: Gas specific gravity (air=1)
            oil_api_gravity: Oil API gravity
            bubble_point_pressure: Bubble point pressure in psi; if None, entire
                range is treated as saturated (Bo from Rs(P))
            oil_compressibility_above_pb: Oil compressibility in 1/psi above Pb
                (default 1e-5)

        Returns:
            Oil FVF Bo in rb/STB
        """
        if any(x <= 0 for x in [pressure, temperature, gas_specific_gravity, oil_api_gravity]):
            raise ValueError("Pressure, temperature, gas gravity, and API gravity must be positive")

        if bubble_point_pressure is None:
            # Saturated: Bo = f(Rs(P), ...)
            rs = FluidCalculations.calculate_solution_gor_standing(
                pressure, temperature, gas_specific_gravity, oil_api_gravity
            )
            return FluidCalculations.calculate_oil_fvf_standing(
                rs, gas_specific_gravity, oil_api_gravity, temperature, pressure
            )

        if pressure <= bubble_point_pressure:
            rs = FluidCalculations.calculate_solution_gor_standing(
                pressure, temperature, gas_specific_gravity, oil_api_gravity
            )
            return FluidCalculations.calculate_oil_fvf_standing(
                rs, gas_specific_gravity, oil_api_gravity, temperature, pressure
            )

        # Above bubble point: undersaturated oil
        rsb = FluidCalculations.calculate_solution_gor_standing(
            bubble_point_pressure, temperature, gas_specific_gravity, oil_api_gravity
        )
        bo_at_pb = FluidCalculations.calculate_oil_fvf_standing(
            rsb, gas_specific_gravity, oil_api_gravity, temperature, bubble_point_pressure
        )
        bo = bo_at_pb * (1.0 + oil_compressibility_above_pb * (pressure - bubble_point_pressure))
        return bo

    @staticmethod
    def get_pvt_at_pressure(
        pressure: float,
        temperature: float,
        gas_specific_gravity: float,
        oil_api_gravity: float | None = None,
        bubble_point_pressure: float | None = None,
        fluid: str = "oil",
    ) -> dict:
        """
        Return Bo(P), Bg(P), Rs(P) (and Z for gas) for material balance iteration.

        Single call to get all pressure-dependent PVT properties at a given
        pressure. Use in iterative MB or when building PVT tables.

        Args:
            pressure: Pressure in psi
            temperature: Temperature in °F
            gas_specific_gravity: Gas specific gravity (air=1)
            oil_api_gravity: Oil API gravity (required for oil/condensate)
            bubble_point_pressure: Bubble point in psi (optional for oil)
            fluid: 'oil' or 'gas'

        Returns:
            For oil: {"Bo": float, "Bg": float, "Rs": float} (Bo rb/STB, Bg rcf/scf, Rs scf/STB).
            For gas: {"Bg": float, "Z": float}.
        """
        if pressure <= 0 or temperature <= 0 or gas_specific_gravity <= 0:
            raise ValueError("Pressure, temperature, and gas specific gravity must be positive")

        if fluid.lower() == "gas":
            z = FluidCalculations.calculate_z_factor(
                pressure, temperature, gas_specific_gravity
            )
            bg = FluidCalculations.calculate_gas_fvf(
                pressure, temperature, gas_specific_gravity
            )
            return {"Bg": bg, "Z": z}

        # Oil
        if oil_api_gravity is None or oil_api_gravity <= 0:
            raise ValueError("oil_api_gravity required for oil PVT")
        rs = FluidCalculations.calculate_solution_gor_standing(
            pressure, temperature, gas_specific_gravity, oil_api_gravity
        )
        if bubble_point_pressure is not None and pressure > bubble_point_pressure:
            rs = FluidCalculations.calculate_solution_gor_standing(
                bubble_point_pressure, temperature, gas_specific_gravity, oil_api_gravity
            )
        bo = FluidCalculations.calculate_oil_fvf_at_pressure(
            pressure,
            temperature,
            gas_specific_gravity,
            oil_api_gravity,
            bubble_point_pressure=bubble_point_pressure,
        )
        bg = FluidCalculations.calculate_gas_fvf(
            pressure, temperature, gas_specific_gravity
        )
        return {"Bo": bo, "Bg": bg, "Rs": rs}

    @staticmethod
    def calculate_water_fvf_mccain(
        temperature: float,
        pressure: float,
        salinity: float = 0.0
    ) -> float:
        """
        Calculate water formation volume factor using McCain correlation.
        
        Args:
            temperature: Temperature in °F
            pressure: Pressure in psi
            salinity: Salinity in ppm (default 0)
            
        Returns:
            Water FVF in rb/STB
        """
        if temperature <= 0 or pressure <= 0:
            raise ValueError("Temperature and pressure must be positive")
        
        # McCain correlation
        dVwp = -1.0001e-2 + 1.33391e-4 * temperature + 5.50654e-7 * (temperature ** 2)
        dVwt = -1.95301e-9 * pressure * temperature - 1.72834e-13 * (pressure ** 2) * temperature
        
        Bw = 1 + dVwp + dVwt
        
        # Salinity correction (simplified)
        if salinity > 0:
            salinity_correction = -0.00001 * salinity / 1000000  # Rough estimate
            Bw = Bw + salinity_correction
        
        return Bw
    
    @staticmethod
    def calculate_interfacial_tension(
        oil_api_gravity: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Calculate oil-water interfacial tension.
        
        Args:
            oil_api_gravity: Oil API gravity
            temperature: Temperature in °F
            pressure: Pressure in psi
            
        Returns:
            Interfacial tension in dynes/cm
            
        Reference: Empirical correlation based on API gravity and temperature
        """
        if any(x <= 0 for x in [oil_api_gravity, temperature, pressure]):
            raise ValueError("All inputs must be positive")
        
        # Empirical correlation
        # Base tension at 60°F, 14.7 psi
        base_tension = 35 - 0.1 * (oil_api_gravity - 30)
        
        # Temperature effect (IFT decreases with temperature)
        # Reference temperature is 60°F
        temp_effect = -0.05 * (temperature - 60.0)
        
        # Pressure effect (minor influence on IFT)
        pressure_effect = 0.001 * (pressure - PhysicalConstants.ATMOSPHERIC_PRESSURE)
        
        ift = base_tension + temp_effect + pressure_effect
        
        return max(ift, 1.0)  # Minimum realistic IFT
