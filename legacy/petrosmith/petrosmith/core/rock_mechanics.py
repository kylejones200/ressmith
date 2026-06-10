"""
Rock Mechanics Module

This module implements rock mechanics calculations and analysis including:
- Rock strength properties (UCS, tensile strength, shear strength)
- Elastic properties (Young's modulus, Poisson's ratio, bulk modulus)
- In-situ stress calculations
- Wellbore stability analysis
- Rock failure criteria (Mohr-Coulomb, Drucker-Prager, Hoek-Brown)
- Fracture mechanics
- Geomechanical property correlations from logs
- Sand production prediction
- Wellbore breakout and fracture analysis

Based on petroleum geomechanics and rock mechanics principles.
"""

from __future__ import annotations
from .constants import PhysicalConstants

import logging
import math
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class RockStrengthProperties:
    """Rock strength properties"""
    ucs: float  # Unconfined compressive strength (psi)
    tensile_strength: float  # Tensile strength (psi)
    cohesion: float  # Cohesion (psi)
    friction_angle: float  # Internal friction angle (degrees)
    poissons_ratio: float  # Poisson's ratio
    youngs_modulus: float  # Young's modulus (psi)


@dataclass
class StressState:
    """In-situ stress state"""
    vertical_stress: float  # Vertical stress (psi)
    max_horizontal_stress: float  # Maximum horizontal stress (psi)
    min_horizontal_stress: float  # Minimum horizontal stress (psi)
    pore_pressure: float  # Pore pressure (psi)
    depth: float  # Depth (ft)


@dataclass
class WellboreGeometry:
    """Wellbore geometry for stability analysis"""
    wellbore_radius: float  # Wellbore radius (inches)
    mud_weight: float  # Mud weight (ppg)
    wellbore_azimuth: float  # Wellbore azimuth (degrees from North)
    wellbore_inclination: float  # Inclination from vertical (degrees)


class ElasticProperties:
    """
    Elastic properties calculations
    
    Calculates rock elastic properties from logs and correlations.
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_youngs_modulus(vp: float, vs: float, density: float) -> float:
        """
        Calculate dynamic Young's modulus from sonic logs
        
        Args:
            vp: P-wave velocity (ft/s)
            vs: S-wave velocity (ft/s)
            density: Bulk density (g/cc)
            
        Returns:
            Young's modulus (psi)
        """
        # Convert density to lb/ft³
        rho = density * 62.4
        
        # Dynamic Young's modulus (psi)
        # E = ρ * Vs² * (3Vp² - 4Vs²) / (Vp² - Vs²)
        
        if vp ** 2 <= vs ** 2:
            return 0.0
        
        E = (rho * vs ** 2 * (3 * vp ** 2 - 4 * vs ** 2)) / (vp ** 2 - vs ** 2)
        
        # Convert from lbf/ft² to psi
        E_psi = E / 144
        
        return E_psi
    
    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_poissons_ratio(vp: float, vs: float) -> float:
        """
        Calculate Poisson's ratio from velocities
        
        Args:
            vp: P-wave velocity (ft/s)
            vs: S-wave velocity (ft/s)
            
        Returns:
            Poisson's ratio (dimensionless)
        """
        # ν = (Vp² - 2Vs²) / (2(Vp² - Vs²))
        
        if vp ** 2 <= vs ** 2:
            return 0.5
        
        nu = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
        
        # Limit to physical range
        return max(0.0, min(nu, 0.5))
    
    @staticmethod
    def calculate_bulk_modulus(vp: float, vs: float, density: float) -> float:
        """
        Calculate bulk modulus
        
        Args:
            vp: P-wave velocity (ft/s)
            vs: S-wave velocity (ft/s)
            density: Bulk density (g/cc)
            
        Returns:
            Bulk modulus (psi)
        """
        # Convert density to lb/ft³
        rho = density * 62.4
        
        # K = ρ(Vp² - 4/3 * Vs²)
        K = rho * (vp ** 2 - (4/3) * vs ** 2)
        
        # Convert to psi
        K_psi = K / 144
        
        return K_psi
    
    @staticmethod
    def calculate_shear_modulus(vs: float, density: float) -> float:
        """
        Calculate shear modulus
        
        Args:
            vs: S-wave velocity (ft/s)
            density: Bulk density (g/cc)
            
        Returns:
            Shear modulus (psi)
        """
        # Convert density
        rho = density * 62.4
        
        # G = ρ * Vs²
        G = rho * vs ** 2
        
        # Convert to psi
        G_psi = G / 144
        
        return G_psi
    
    @staticmethod
    def sonic_to_velocity(dt: float) -> float:
        """
        Convert sonic transit time to velocity
        
        Args:
            dt: Sonic transit time (μs/ft)
            
        Returns:
            Velocity (ft/s)
        """
        if dt <= 0:
            return 0.0
        
        # V = 1,000,000 / Δt
        velocity = 1_000_000 / dt
        
        return velocity
    
    @staticmethod
    def static_to_dynamic_correction(dynamic_E: float,
                                     dynamic_nu: float,
                                     correction_factor: float = 0.7) -> dict:
        """
        Convert dynamic to static elastic properties
        
        Args:
            dynamic_E: Dynamic Young's modulus (psi)
            dynamic_nu: Dynamic Poisson's ratio
            correction_factor: Empirical correction (typically 0.6-0.8)
            
        Returns:
            dict with static properties
        """
        # Static properties are typically lower than dynamic
        static_E = dynamic_E * correction_factor
        static_nu = dynamic_nu * correction_factor
        
        return {
            'static_youngs_modulus_psi': round(static_E, 0),
            'static_poissons_ratio': round(static_nu, 3),
            'correction_factor': correction_factor
        }


class RockStrength:
    """
    Rock strength calculations
    
    Estimates rock strength from logs and correlations.
    """
    
    @staticmethod
    def ucs_from_sonic(dt_compressional: float,
                      correlation: str = 'default') -> float:
        """
        Estimate UCS from sonic log using empirical correlations.
        
        Args:
            dt_compressional: Compressional sonic (μs/ft)
            correlation: Correlation type ('default', 'sandstone', 'shale')
            
        Returns:
            UCS (psi)
        """
        if dt_compressional <= 0:
            logger.warning(f"Invalid dt_compressional: {dt_compressional}")
            return 0.0
        
        # Correlation coefficients: {type: (intercept, slope)}
        correlations = {
            'sandstone': (5.88, -0.92),
            'shale': (6.21, -1.11),
            'default': (6.0, -1.0)
        }
        
        intercept, slope = correlations.get(correlation, correlations['default'])
        ucs = 10 ** (intercept + slope * math.log10(dt_compressional))
        
        logger.debug(f"UCS from sonic: dt={dt_compressional:.1f}, correlation={correlation}, UCS={ucs:.0f}")
        return ucs
    
    @staticmethod
    def ucs_from_porosity(porosity: float,
                         lithology: str = 'sandstone') -> float:
        """
        Estimate UCS from porosity
        
        Args:
            porosity: Porosity (fraction)
            lithology: Lithology type
            
        Returns:
            UCS (psi)
        """
        if lithology == 'sandstone':
            # Empirical correlation for sandstone
            # UCS decreases exponentially with porosity
            ucs = 40000 * math.exp(-7 * porosity)
        
        elif lithology == 'carbonate':
            # Carbonate correlation
            ucs = 50000 * math.exp(-8 * porosity)
        
        else:
            # Generic
            ucs = 35000 * math.exp(-6 * porosity)
        
        return max(ucs, 100)  # Minimum UCS
    
    @staticmethod
    def tensile_strength_from_ucs(ucs: float,
                                  ratio: float = 0.1) -> float:
        """
        Estimate tensile strength from UCS
        
        Args:
            ucs: Unconfined compressive strength (psi)
            ratio: Tensile to compressive strength ratio (typically 0.05-0.15)
            
        Returns:
            Tensile strength (psi)
        """
        return ucs * ratio
    
    @staticmethod
    def cohesion_from_ucs(ucs: float, friction_angle: float) -> float:
        """
        Calculate cohesion from UCS and friction angle
        
        Args:
            ucs: UCS (psi)
            friction_angle: Internal friction angle (degrees)
            
        Returns:
            Cohesion (psi)
        """
        # From Mohr-Coulomb: UCS = 2c * cos(φ) / (1 - sin(φ))
        phi_rad = math.radians(friction_angle)
        
        denominator = 2 * math.cos(phi_rad) / (1 - math.sin(phi_rad))
        if denominator == 0:
            return 0.0
        
        cohesion = ucs / denominator
        
        return cohesion
    
    @staticmethod
    def estimate_friction_angle(lithology: str = 'sandstone') -> float:
        """
        Estimate internal friction angle from lithology
        
        Args:
            lithology: Rock type
            
        Returns:
            Friction angle (degrees)
        """
        friction_angles = {
            'sandstone': 35,
            'shale': 25,
            'limestone': 40,
            'dolomite': 45,
            'granite': 50,
            'basalt': 48
        }
        
        return friction_angles.get(lithology.lower(), 30)


class InSituStress:
    """
    In-situ stress calculations
    
    Determines the stress state in the subsurface.
    """
    
    @staticmethod
    def calculate_vertical_stress(depths: list[float],
                                  densities: list[float]) -> dict:
        """
        Calculate vertical stress from density log
        
        Args:
            depths: Depth points (ft)
            densities: Bulk density at each depth (g/cc)
            
        Returns:
            dict with vertical stress profile
        """
        depths = np.array(depths)
        densities = np.array(densities)
        
        # Integrate density
        vertical_stress = np.zeros_like(depths)
        
        for i in range(len(depths)):
            if i == 0:
                vertical_stress[i] = 0.433 * densities[i] * depths[i]
            else:
                depth_interval = depths[i] - depths[i-1]
                avg_density = (densities[i] + densities[i-1]) / 2.0
                vertical_stress[i] = vertical_stress[i-1] + 0.433 * avg_density * depth_interval
        
        gradients = vertical_stress / depths
        
        return {
            'depths_ft': depths.tolist(),
            'vertical_stress_psi': vertical_stress.tolist(),
            'vertical_stress_gradient_psi_ft': gradients.tolist()
        }
    
    @staticmethod
    def estimate_horizontal_stress(vertical_stress: float,
                                   pore_pressure: float,
                                   poissons_ratio: float,
                                   tectonic_strain: float = 0.0,
                                   stress_regime: str = 'normal') -> dict:
        """
        Estimate horizontal stresses
        
        Args:
            vertical_stress: Vertical stress (psi)
            pore_pressure: Pore pressure (psi)
            poissons_ratio: Poisson's ratio
            tectonic_strain: Tectonic strain component
            stress_regime: 'normal', 'strike-slip', or 'reverse'
            
        Returns:
            dict with horizontal stresses
        """
        # Effective vertical stress
        sigma_v_eff = vertical_stress - pore_pressure
        
        # Base horizontal stress (no tectonic)
        # σh = (ν/(1-ν)) * σv_eff + α * Pp
        K0 = poissons_ratio / (1 - poissons_ratio)
        
        # Biot coefficient (assume 1 for simplicity)
        alpha = 1.0
        
        base_horizontal = K0 * sigma_v_eff + alpha * pore_pressure
        
        # Add tectonic component
        horizontal_stress = base_horizontal + tectonic_strain
        
        # Stress regime adjustments
        if stress_regime == 'normal':
            # σv > σH > σh
            sigma_H = horizontal_stress * 1.1
            sigma_h = horizontal_stress * 0.9
        
        elif stress_regime == 'strike-slip':
            # σH > σv > σh
            sigma_H = vertical_stress * 1.2
            sigma_h = horizontal_stress * 0.8
        
        elif stress_regime == 'reverse':
            # σH > σh > σv
            sigma_H = vertical_stress * 1.5
            sigma_h = vertical_stress * 1.2
        
        else:
            sigma_H = horizontal_stress
            sigma_h = horizontal_stress
        
        return {
            'vertical_stress_psi': round(vertical_stress, 0),
            'max_horizontal_stress_psi': round(sigma_H, 0),
            'min_horizontal_stress_psi': round(sigma_h, 0),
            'pore_pressure_psi': round(pore_pressure, 0),
            'stress_regime': stress_regime,
            'K0_coefficient': round(K0, 3)
        }


class FailureCriteria:
    """
    Rock failure criteria
    
    Implements various failure criteria for rock mechanics.
    """
    
    @staticmethod
    def mohr_coulomb_criterion(sigma1: float,
                               sigma3: float,
                               cohesion: float,
                               friction_angle: float) -> dict:
        """
        Mohr-Coulomb failure criterion
        
        Args:
            sigma1: Maximum principal stress (psi)
            sigma3: Minimum principal stress (psi)
            cohesion: Cohesion (psi)
            friction_angle: Friction angle (degrees)
            
        Returns:
            dict with failure analysis
        """
        phi_rad = math.radians(friction_angle)
        
        # Shear strength
        # τ = c + σn * tan(φ)
        
        # Normal stress on failure plane
        sigma_n = (sigma1 + sigma3) / 2 - ((sigma1 - sigma3) / 2) * math.sin(phi_rad)
        
        # Shear stress
        tau = (sigma1 - sigma3) / 2 * math.cos(phi_rad)
        
        # Shear strength
        tau_f = cohesion + sigma_n * math.tan(phi_rad)
        
        # Factor of safety
        if tau > 0:
            fos = tau_f / tau
        else:
            fos = 999
        
        # Failure status using vectorized comparison
        status_thresholds = [(1.0, 'FAILURE'), (1.5, 'CRITICAL'), (np.inf, 'STABLE')]
        status = next((s for threshold, s in status_thresholds if fos < threshold), 'STABLE')
        
        return {
            'criterion': 'Mohr-Coulomb',
            'shear_stress_psi': round(tau, 0),
            'shear_strength_psi': round(tau_f, 0),
            'factor_of_safety': round(fos, 2),
            'status': status
        }
    
    @staticmethod
    def drucker_prager_criterion(sigma1: float,
                                 sigma2: float,
                                 sigma3: float,
                                 cohesion: float,
                                 friction_angle: float) -> dict:
        """
        Drucker-Prager failure criterion
        
        Args:
            sigma1: Maximum principal stress (psi)
            sigma2: Intermediate principal stress (psi)
            sigma3: Minimum principal stress (psi)
            cohesion: Cohesion (psi)
            friction_angle: Friction angle (degrees)
            
        Returns:
            dict with failure analysis
        """
        phi_rad = math.radians(friction_angle)
        
        # Material parameters
        # α = 2sin(φ) / (√3(3-sin(φ)))
        # k = 6c*cos(φ) / (√3(3-sin(φ)))
        
        sin_phi = math.sin(phi_rad)
        cos_phi = math.cos(phi_rad)
        
        alpha = (2 * sin_phi) / (math.sqrt(3) * (3 - sin_phi))
        k = (6 * cohesion * cos_phi) / (math.sqrt(3) * (3 - sin_phi))
        
        # Mean stress
        I1 = sigma1 + sigma2 + sigma3
        
        # Second invariant of deviatoric stress
        J2 = ((sigma1 - sigma2)**2 + (sigma2 - sigma3)**2 + (sigma3 - sigma1)**2) / 6
        
        # Drucker-Prager function
        # F = √J2 + α*I1 - k
        F = math.sqrt(J2) + alpha * I1 - k
        
        # Status determination using threshold mapping
        status_thresholds = [(0, 'FAILURE'), (-500, 'CRITICAL'), (-np.inf, 'STABLE')]
        status = next((s for threshold, s in status_thresholds if F > threshold), 'STABLE')
        
        return {
            'criterion': 'Drucker-Prager',
            'failure_function': round(F, 0),
            'status': status,
            'mean_stress_psi': round(I1/3, 0)
        }
    
    @staticmethod
    def hoek_brown_criterion(sigma1: float,
                            sigma3: float,
                            ucs: float,
                            mi: float = 10,
                            gsi: float = 50) -> dict:
        """
        Hoek-Brown failure criterion
        
        Args:
            sigma1: Maximum principal stress (psi)
            sigma3: Minimum principal stress (psi)
            ucs: Unconfined compressive strength (psi)
            mi: Material constant (typically 7-35)
            gsi: Geological Strength Index (0-100)
            
        Returns:
            dict with failure analysis
        """
        # Hoek-Brown parameters
        # mb = mi * exp((GSI-100)/28)
        # s = exp((GSI-100)/9)
        # a = 0.5
        
        mb = mi * math.exp((gsi - 100) / 28)
        s = math.exp((gsi - 100) / 9)
        a = 0.5
        
        # Hoek-Brown criterion
        # σ1 = σ3 + σci * (mb*(σ3/σci) + s)^a
        
        if ucs <= 0:
            return {'error': 'Invalid UCS value'}
        
        sigma3_normalized = sigma3 / ucs
        
        predicted_sigma1 = sigma3 + ucs * ((mb * sigma3_normalized + s) ** a)
        
        # Factor of safety
        if predicted_sigma1 > 0:
            fos = predicted_sigma1 / max(sigma1, 1)
        else:
            fos = 0
        
        if fos < 1.0:
            status = 'FAILURE'
        elif fos < 1.5:
            status = 'CRITICAL'
        else:
            status = 'STABLE'
        
        return {
            'criterion': 'Hoek-Brown',
            'mb_parameter': round(mb, 3),
            's_parameter': round(s, 4),
            'predicted_sigma1_psi': round(predicted_sigma1, 0),
            'actual_sigma1_psi': round(sigma1, 0),
            'factor_of_safety': round(fos, 2),
            'status': status
        }


class WellboreStability:
    """
    Wellbore stability analysis
    
    Analyzes wellbore stability and calculates mud weight window.
    """
    
    @staticmethod
    def calculate_wellbore_stresses(stress_state: StressState,
                                   wellbore_pressure: float,
                                   wellbore_geo: WellboreGeometry) -> dict:
        """
        Calculate stresses around wellbore
        
        Args:
            stress_state: In-situ stress state
            wellbore_pressure: Wellbore pressure (psi)
            wellbore_geo: Wellbore geometry
            
        Returns:
            dict with wellbore stresses
        """
        # Kirsch equations for stresses around circular hole
        # At wellbore wall (r = a)
        
        # Radial stress
        sigma_r = wellbore_pressure
        
        # Tangential stress (hoop stress)
        # σθ = σH + σh - 2(σH - σh)cos(2θ) - Pw
        
        # For vertical well (simplified)
        sigma_theta_max = (3 * stress_state.max_horizontal_stress - 
                          stress_state.min_horizontal_stress - 
                          wellbore_pressure)
        
        sigma_theta_min = (3 * stress_state.min_horizontal_stress - 
                          stress_state.max_horizontal_stress - 
                          wellbore_pressure)
        
        # Axial stress
        sigma_z = stress_state.vertical_stress - 2 * 0.25 * (
            stress_state.max_horizontal_stress - stress_state.min_horizontal_stress)
        
        return {
            'radial_stress_psi': round(sigma_r, 0),
            'tangential_stress_max_psi': round(sigma_theta_max, 0),
            'tangential_stress_min_psi': round(sigma_theta_min, 0),
            'axial_stress_psi': round(sigma_z, 0),
            'wellbore_pressure_psi': round(wellbore_pressure, 0)
        }
    
    @staticmethod
    def collapse_pressure(stress_state: StressState,
                         rock_props: RockStrengthProperties,
                         safety_factor: float = 1.2) -> dict:
        """
        Calculate collapse pressure (minimum mud weight)
        
        Args:
            stress_state: In-situ stress state
            rock_props: Rock strength properties
            safety_factor: Safety factor
            
        Returns:
            dict with collapse analysis
        """
        # Simplified collapse pressure using Mohr-Coulomb
        
        phi_rad = math.radians(rock_props.friction_angle)
        sin_phi = math.sin(phi_rad)
        
        # Collapse pressure
        Nφ = (1 + sin_phi) / (1 - sin_phi)
        
        Pc = ((3 * stress_state.max_horizontal_stress - stress_state.min_horizontal_stress) - 
              Nφ * (stress_state.max_horizontal_stress - stress_state.pore_pressure + 
                    2 * rock_props.cohesion / math.sqrt(Nφ))) / (2 * (Nφ - 1))
        
        # Apply safety factor
        Pc_safe = Pc * safety_factor
        
        # Convert to EMW
        emw_collapse = Pc_safe / (PhysicalConstants.HYDROSTATIC_GRADIENT * stress_state.depth)
        
        return {
            'collapse_pressure_psi': round(Pc, 0),
            'collapse_pressure_safe_psi': round(Pc_safe, 0),
            'minimum_mud_weight_ppg': round(emw_collapse, 2),
            'safety_factor': safety_factor
        }
    
    @staticmethod
    def fracture_pressure(stress_state: StressState,
                         rock_props: RockStrengthProperties) -> dict:
        """
        Calculate fracture pressure (maximum mud weight)
        
        Args:
            stress_state: In-situ stress state
            rock_props: Rock strength properties
            
        Returns:
            dict with fracture analysis
        """
        # Tensile failure criterion
        # Pf = 3*σh - σH - Pp + T
        
        Pf = (3 * stress_state.min_horizontal_stress - 
              stress_state.max_horizontal_stress - 
              stress_state.pore_pressure + 
              rock_props.tensile_strength)
        
        # Convert to EMW
        emw_fracture = Pf / (PhysicalConstants.HYDROSTATIC_GRADIENT * stress_state.depth)
        
        return {
            'fracture_pressure_psi': round(Pf, 0),
            'maximum_mud_weight_ppg': round(emw_fracture, 2),
            'tensile_strength_psi': round(rock_props.tensile_strength, 0)
        }
    
    @staticmethod
    def mud_weight_window(stress_state: StressState,
                         rock_props: RockStrengthProperties,
                         current_mud_weight: float) -> dict:
        """
        Calculate complete mud weight window
        
        Args:
            stress_state: In-situ stress state
            rock_props: Rock strength properties
            current_mud_weight: Current mud weight (ppg)
            
        Returns:
            dict with complete stability analysis
        """
        # Collapse analysis
        collapse = WellboreStability.collapse_pressure(stress_state, rock_props)
        
        # Fracture analysis
        fracture = WellboreStability.fracture_pressure(stress_state, rock_props)
        
        min_mw = collapse['minimum_mud_weight_ppg']
        max_mw = fracture['maximum_mud_weight_ppg']
        
        # Window width
        window = max_mw - min_mw
        
        # Status
        if current_mud_weight < min_mw:
            status = 'UNDERBALANCED - Risk of collapse'
            risk = 'COLLAPSE RISK'
        elif current_mud_weight > max_mw:
            status = 'OVERBALANCED - Risk of fracture'
            risk = 'FRACTURE RISK'
        else:
            status = 'Within stability window'
            risk = 'STABLE'
        
        # Window quality
        if window < 1.0:
            quality = 'Narrow - Difficult operations'
        elif window < 2.0:
            quality = 'Moderate - Careful control needed'
        else:
            quality = 'Wide - Good margin'
        
        return {
            'depth_ft': stress_state.depth,
            'minimum_mud_weight_ppg': round(min_mw, 2),
            'maximum_mud_weight_ppg': round(max_mw, 2),
            'mud_weight_window_ppg': round(window, 2),
            'current_mud_weight_ppg': current_mud_weight,
            'status': status,
            'risk': risk,
            'window_quality': quality,
            'collapse_analysis': collapse,
            'fracture_analysis': fracture
        }


class SandProduction:
    """
    Sand production prediction
    
    Predicts likelihood of sand production in weak formations.
    """
    
    @staticmethod
    def sand_production_index(ucs: float,
                             drawdown_pressure: float,
                             stress_ratio: float) -> dict:
        """
        Calculate sand production index
        
        Args:
            ucs: Rock UCS (psi)
            drawdown_pressure: Production drawdown (psi)
            stress_ratio: Stress concentration ratio
            
        Returns:
            dict with sand production assessment
        """
        # Simplified sand production index
        # Higher index = higher risk
        
        if ucs <= 0:
            return {'error': 'Invalid UCS'}
        
        spi = (drawdown_pressure * stress_ratio) / ucs
        
        # Risk classification
        if spi > 2.0:
            risk = 'SEVERE - Sand control required'
            recommendation = 'Install gravel pack or screens'
        elif spi > 1.0:
            risk = 'HIGH - Monitor production'
            recommendation = 'Consider sand control completion'
        elif spi > 0.5:
            risk = 'MODERATE - Possible sand'
            recommendation = 'Monitor and have sand handling equipment'
        else:
            risk = 'LOW - Minimal risk'
            recommendation = 'Standard completion acceptable'
        
        return {
            'sand_production_index': round(spi, 2),
            'ucs_psi': round(ucs, 0),
            'drawdown_psi': round(drawdown_pressure, 0),
            'risk_level': risk,
            'recommendation': recommendation
        }


# Convenience functions
def complete_geomechanical_analysis(depth: float,
                                    sonic_dt_compressional: float,
                                    sonic_dt_shear: float,
                                    bulk_density: float,
                                    porosity: float,
                                    pore_pressure: float,
                                    mud_weight: float) -> dict:
    """
    Complete geomechanical analysis
    
    Args:
        depth: Depth (ft)
        sonic_dt_compressional: P-wave sonic (μs/ft)
        sonic_dt_shear: S-wave sonic (μs/ft)
        bulk_density: Bulk density (g/cc)
        porosity: Porosity (fraction)
        pore_pressure: Pore pressure (psi)
        mud_weight: Current mud weight (ppg)
        
    Returns:
        dict with complete analysis
    """
    # Calculate velocities
    vp = ElasticProperties.sonic_to_velocity(sonic_dt_compressional)
    vs = ElasticProperties.sonic_to_velocity(sonic_dt_shear)
    
    # Elastic properties
    E = ElasticProperties.calculate_youngs_modulus(vp, vs, bulk_density)
    nu = ElasticProperties.calculate_poissons_ratio(vp, vs)
    G = ElasticProperties.calculate_shear_modulus(vs, bulk_density)
    
    # Rock strength
    ucs = RockStrength.ucs_from_sonic(sonic_dt_compressional)
    tensile = RockStrength.tensile_strength_from_ucs(ucs)
    friction_angle = RockStrength.estimate_friction_angle('sandstone')
    cohesion = RockStrength.cohesion_from_ucs(ucs, friction_angle)
    
    # Stresses
    vertical_stress = 0.433 * bulk_density * depth
    horizontal_stress = InSituStress.estimate_horizontal_stress(
        vertical_stress, pore_pressure, nu)
    
    # Create objects
    rock_props = RockStrengthProperties(
        ucs=ucs,
        tensile_strength=tensile,
        cohesion=cohesion,
        friction_angle=friction_angle,
        poissons_ratio=nu,
        youngs_modulus=E
    )
    
    stress_state = StressState(
        vertical_stress=vertical_stress,
        max_horizontal_stress=horizontal_stress['max_horizontal_stress_psi'],
        min_horizontal_stress=horizontal_stress['min_horizontal_stress_psi'],
        pore_pressure=pore_pressure,
        depth=depth
    )
    
    # Stability analysis
    stability = WellboreStability.mud_weight_window(stress_state, rock_props, mud_weight)
    
    return {
        'depth_ft': depth,
        'elastic_properties': {
            'youngs_modulus_psi': round(E, 0),
            'poissons_ratio': round(nu, 3),
            'shear_modulus_psi': round(G, 0)
        },
        'strength_properties': {
            'ucs_psi': round(ucs, 0),
            'tensile_strength_psi': round(tensile, 0),
            'cohesion_psi': round(cohesion, 0),
            'friction_angle_deg': friction_angle
        },
        'stress_state': horizontal_stress,
        'stability_analysis': stability
    }
