"""
Drilling Fluids Module

This module implements drilling fluids calculations and models including:
- Mud weight and density calculations
- Rheology models (Bingham Plastic, Power Law, Herschel-Bulkley)
- Hydraulics calculations
- ECD (Equivalent Circulating Density) calculations
- Gel strength analysis
- Mud treatment and conditioning
- Solids control calculations
- Filtration properties

Based on API and industry-standard drilling fluids engineering principles.
"""

import numpy as np
import logging
from .constants import PhysicalConstants

# Configure module logger
logger = logging.getLogger(__name__)
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class MudComposition:
    """Drilling mud composition"""
    water_content: float  # % by volume
    oil_content: float  # % by volume
    solids_content: float  # % by volume
    barite_content: float  # lb/bbl
    bentonite_content: float  # lb/bbl
    salt_content: float  # lb/bbl
    polymer_content: float  # lb/bbl


@dataclass
class RheologyData:
    """Rheological properties from viscometer"""
    reading_600: float  # Dial reading at 600 RPM
    reading_300: float  # Dial reading at 300 RPM
    reading_200: float  # Dial reading at 200 RPM
    reading_100: float  # Dial reading at 100 RPM
    reading_6: float  # Dial reading at 6 RPM
    reading_3: float  # Dial reading at 3 RPM
    temperature: float  # Temperature in °F


@dataclass
class FiltrationProperties:
    """Filtration test results"""
    api_filtrate: float  # API filtrate volume in ml/30 min
    hthp_filtrate: float  # HTHP filtrate volume in ml/30 min
    filter_cake_thickness: float  # Filter cake thickness in 1/32 inch
    spurt_loss: float  # Spurt loss in ml


class MudWeightCalculations:
    """
    Mud weight and density calculations
    
    Handles conversions and calculations related to mud weight.
    """
    
    @staticmethod
    def ppg_to_sg(ppg: float) -> float:
        """Convert pounds per gallon to specific gravity"""
        return ppg / 8.33
    
    @staticmethod
    def sg_to_ppg(sg: float) -> float:
        """Convert specific gravity to pounds 8.33gallon"""
        return sg * 8.33
    
    @staticmethod
    def ppg_to_psi_per_ft(ppg: float) -> float:
        """Convert ppg to pressure gradient in psi/ft"""
        return 0.052 * ppg
    
    @staticmethod
    def calculate_hydrostatic_pressure(mud_weight: float, depth: float) -> float:
        """
        Calculate hydrostatic pressure
        
        Args:
            mud_weight: Mud weight in ppg
            depth: True vertical depth in ft
            
        Returns:
            Hydrostatic pressure i0.052
        """
        return 0.052 * mud_weight * depth
    
    @staticmethod
    def increase_mud_weight(current_weight: float,
                          current_volume: float,
                          target_weight: float,
                          material_sg: float = 4.2) -> dict:
        """
        Calculate material required to increase mud weight
        
        Args:
            current_weight: Current mud weight in ppg
            current_volume: Current volume in bbls
            target_weight: Target mud weight in ppg
            material_sg: Specific gravity of weighting material (barite=4.2)
            
        Returns:
            dict with sacks required and final volume
        """
        if target_weight <= current_weight:
            return {
                'sacks_required': 0,
                'material_weight_lbs': 0,
                'final_volume_bbls': current_volume,
                'message': 'Target weight equal to or less than current weight'
            }
        
        # Calculate specific gravities
        sg1 = current_weight / PhysicalConstants.WATER_DENSITY_LB_GAL
        sg2 = target_weight / PhysicalConstants.WATER_DENSITY_LB_GAL
        
        # Weight of material per barrel
        # Formula: Sacks/bbl = 1470 * (MW2 - MW1) / (94 * SGm - MW2)
        sacks_per_bbl = 1470 * (target_weight - current_weight) / (94 * material_sg - target_weight)
        
        # Total sacks needed
        total_sacks = sacks_per_bbl * current_volume
        
        # Material weight in lbs
        material_weight = total_sacks * 100  # 100 lb sacks
        
        # Final volume (volume increases with solids addition)
        volume_increase = total_sacks * PhysicalConstants.BARITE_SACK_VOLUME_GAL / PhysicalConstants.GALLONS_PER_BBL  # 1.39 gallons per 100 lb sack
        final_volume = current_volume + volume_increase
        
        return {
            'sacks_required': round(total_sacks, 1),
            'sacks_per_bbl': round(sacks_per_bbl, 2),
            'material_weight_lbs': round(material_weight, 0),
            'final_volume_bbls': round(final_volume, 1),
            'volume_increase_bbls': round(volume_increase, 1)
        }
    
    @staticmethod
    def dilute_mud_weight(current_weight: float,
                         current_volume: float,
                         target_weight: float,
                         diluent_weight: float = PhysicalConstants.WATER_DENSITY_LB_GAL) -> dict:
        """
        Calculate volume needed to dilute mud weight
        
        Args:
            current_weight: Current mud weight 8.33pg
            current_volume: Current volume in bbls
            target_weight: Target mud weight in ppg
            diluent_weight: Weight of diluent in ppg (water=8.33)
            
        Returns:
            dict with volume of diluent required
        """
        if target_weight >= current_weight:
            return {
                'diluent_volume_bbls': 0,
                'final_volume_bbls': current_volume,
                'message': 'Target weight equal to or greater than current weight'
            }
        
        # Formula: Vd = V1 * (MW1 - MW2) / (MW2 - MWd)
        diluent_volume = (current_volume * (current_weight - target_weight) / 
                         (target_weight - diluent_weight))
        
        final_volume = current_volume + diluent_volume
        
        return {
            'diluent_volume_bbls': round(diluent_volume, 1),
            'final_volume_bbls': round(final_volume, 1),
            'diluent_weight_ppg': diluent_weight
        }


class RheologyModels:
    """
    Rheological models for drilling fluids
    
    Implements various rheological models to characterize fluid behavior.
    """
    
    def __init__(self, rheology_data: RheologyData):
        """
        Initialize with rheology data
        
        Args:
            rheology_data: Viscometer readings
        """
        self.data = rheology_data
        
    def bingham_plastic_model(self) -> dict:
        """
        Calculate Bingham Plastic parameters
        
        Returns:
            dict with PV and YP
        """
        # Plastic Viscosity (PV) = 600 RPM reading - 300 RPM reading
        pv = self.data.reading_600 - self.data.reading_300
        
        # Yield Point (YP) = 300 RPM reading - PV
        yp = self.data.reading_300 - pv
        
        # Apparent Viscosity at 600 RPM
        av = self.data.reading_600 / 2
        
        return {
            'plastic_viscosity_cp': pv,
            'yield_point_lbf_100ft2': yp,
            'apparent_viscosity_cp': av,
            'model': 'Bingham Plastic'
        }
    
    def power_law_model(self) -> dict:
        """
        Calculate Power Law parameters
        
        Returns:
            dict with n and K values
        """
        # Flow behavior index (n)
        # n = 3.32 * log(R600 / R300)
        if self.data.reading_300 == 0:
            return {'error': 'Invalid reading at 300 RPM'}
        
        n = PhysicalConstants.POWER_LAW_EXPONENT_FACTOR * math.log10(self.data.reading_600 / self.data.reading_300)
        
        # Consistency index (K)
        # K = 5.11 * R600 / (1022^n)
        k = PhysicalConstants.POWER_LAW_K_FACTOR * self.data.reading_600 / (PhysicalConstants.POWER_LAW_SHEAR_RATE ** n)
        
        return {
            'flow_behavior_index_n': round(n, 3),
            'consistency_index_k': round(k, 3),
            'model': 'Power Law',
            'fluid_type': 'Pseudoplastic' if n < 1 else 'Dilatant' if n > 1 else 'Newtonian'
        }
    
    def herschel_bulkley_model(self) -> dict:
        """
        Calculate Herschel-Bulkley parameters
        
        Three-parameter model: τ = τ0 + K * γ^n
        
        Returns:
            dict with τ0, K, and n
        """
        # This requires iterative solution or multi-point regression
        # Using simplified approach with multiple readings
        
        # Shear rates at different RPMs (1/sec)
        shear_rates = {
            600: 1022,
            300: 511,
            200: 340,
            100: 170,
            6: 10.2,
            3: 5.1
        }
        
        # Shear stresses (lbf/100ft²)
        shear_stresses = {
            600: self.data.reading_600,
            300: self.data.reading_300,
            200: self.data.reading_200,
            100: self.data.reading_100,
            6: self.data.reading_6,
            3: self.data.reading_3
        }
        
        # Simplified estimation using 3 and 6 RPM readings
        # Yield stress approximately equals lowest reading
        tau_0 = self.data.reading_3
        
        # Estimate n using mid-range readings
        if self.data.reading_300 - tau_0 > 0 and self.data.reading_600 - tau_0 > 0:
            n = math.log10((self.data.rPhysicalConstants.POWER_LAW_SHEAR_RATEng_600 - tau_0) / (self.data.reading_300 - tau_0)) / math.log10(1022 / 511)
        else:
            n = 1.0
        
        # Estimate K from 600 RPM reading
        if n > 0:
            k = (self.data.reading_600 - tau_0) / (1022 ** n)
        else:
            k = 0
        
        return {
            'yield_stress_tau0': round(tau_0, 2),
            'consistency_index_k': round(k, 3),
            'flow_behavior_index_n': round(n, 3),
            'model': 'Herschel-Bulkley'
        }
    
    def gel_strength_analysis(self, gel_10sec: float, gel_10min: float) -> dict:
        """
        Analyze gel strength development
        
        Args:
            gel_10sec: 10-second gel strength in lbf/100ft²
            gel_10min: 10-minute gel strength in lbf/100ft²
            
        Returns:
            dict with gel strength analysis
        """
        # Gel strength progression
        ratio = gel_10min / max(gel_10sec, 1.0)
        
        # Classification
        if ratio < 2.0:
            classification = 'Flat gel - Good'
            recommendation = 'Excellent suspension properties'
        elif ratio < 3.0:
            classification = 'Progressive gel - Acceptable'
            recommendation = 'Monitor for excessive gel development'
        else:
            classification = 'High progressive gel - Poor'
            recommendation = 'Treatment required to reduce gels'
        
        return {
            'gel_10sec_lbf_100ft2': gel_10sec,
            'gel_10min_lbf_100ft2': gel_10min,
            'ratio_10min_10sec': round(ratio, 2),
            'classification': classification,
            'recommendation': recommendation
        }


class HydraulicsCalculations:
    """
    Drilling hydraulics calculations
    
    Calculates pressure losses, ECD, and hydraulic optimization.
    """
    
    @staticmethod
    def calculate_pressure_loss_pipe(flow_rate: float,
                                    pv: float,
                                    yp: float,
                                    pipe_id: float,
                                    pipe_length: float,
                                    mud_weight: float) -> float:
        """
        Calculate pressure loss in pipe (drillpipe/drillcollar)
        
        Args:
            flow_rate: Flow rate in gpm
            pv: Plastic viscosity in cp
            yp: Yield point in lbf/100ft²
            pipe_id: Pipe ID in inches
            pipe_length: Pipe length in ft
            mud_weight: Mud weight in ppg
            
        Returns:
            Pressure loss in psi
        """
        # Velocity (ft/sec)
        velocity = 0.408 * flow_rate / (pipe_id ** 2)
        
        # Reynolds number for Bingham plastic
        n_re = 928 * mud_weight * velocity * pipe_id / pv
        
        # Friction factor
        if n_re < 2100:
            # Laminar flow
            f = 16 / n_re
        else:
            # Turbulent flow (simplified)
            f = 0.046 / (n_re ** 0.2)
        
        # Pressure loss (psi)
        # ΔP = (f * ρ * v² * L) / (25.8 * D)
        dp = (f * mud_weight * velocity ** 2 * pipe_length) / (25.8 * pipe_id)
        
        # Add yield point contribution for laminar flow
        if n_re < 2100:
            dp_yp = (yp * velocity * pipe_length) / (225 * pipe_id)
            dp += dp_yp
        
        return dp
    
    @staticmethod
    def calculate_pressure_loss_annulus(flow_rate: float,
                                       pv: float,
                                       yp: float,
                                       hole_id: float,
                                       pipe_od: float,
                                       length: float,
                                       mud_weight: float) -> float:
        """
        Calculate pressure loss in annulus
        
        Args:
            flow_rate: Flow rate in gpm
            pv: Plastic viscosity in cp
            yp: Yield point in lbf/100ft²
            hole_id: Hole/casing ID in inches
            pipe_od: Pipe OD in inches
            length: Annulus length in ft
            mud_weight: Mud weight in ppg
            
        Returns:
            Pressure loss in psi
        """
        # Annular velocity (ft/sec)
        velocity = 0.408 * flow_rate / (hole_id ** 2 - pipe_od ** 2)
        
        # Hydraulic diameter
        dh = hole_id - pipe_od
        
        # Reynolds number for annulus
        n_re = 757 * mud_weight * velocity * dh / pv
        
        # Friction factor
        if n_re < 2100:
            # Laminar flow
            f = 24 / n_re
        else:
            # Turbulent flow
            f = 0.046 / (n_re ** 0.2)
        
        # Pressure loss
        dp = (f * mud_weight * velocity ** 2 * length) / (25.8 * dh)
        
        # Add yield point contribution for laminar flow
        if n_re < 2100:
            dp_yp = (yp * velocity * length) / (200 * dh)
            dp += dp_yp
        
        return dp
    
    @staticmethod
    def calculate_bit_pressure_loss(flow_rate: float,
                                   mud_weight: float,
                                   nozzle_sizes: list[float]) -> float:
        """
        Calculate bit pressure loss
        
        Args:
            flow_rate: Flow rate in gpm
            mud_weight: Mud weight in ppg
            nozzle_sizes: list of nozzle sizes in 32nds of an inch
            
        Returns:
            Bit pressure loss in psi
        """
        # Total flow area (in²)
        total_area = sum([(size / 32) ** 2 * math.pi / 4 for size in nozzle_sizes])
        
        if total_area == 0:
            return 0
        
        # Bit pressure loss
        # ΔP = (MW * Q²) / (10858 * A²)
        dp_bit = (mud_weight * flow_rate ** 2) / (10858 * total_area ** 2)
        
        return dp_bit
    
    @staticmethod
    def calculate_surface_pressure(flow_rate: float,
                                  pv: float,
                                  yp: float,
                                  dp_id: float,
                                  dp_length: float,
                                  dc_id: float,
                                  dc_length: float,
                                  hole_id: float,
                                  dp_od: float,
                                  dc_od: float,
                                  annulus_length: float,
                                  nozzle_sizes: list[float],
                                  mud_weight: float) -> dict:
        """
        Calculate total system pressure
        
        Returns:
            dict with pressure components
        """
        # Drillpipe pressure loss
        dp_drillpipe = HydraulicsCalculations.calculate_pressure_loss_pipe(
            flow_rate, pv, yp, dp_id, dp_length, mud_weight)
        
        # Drillcollar pressure loss
        dp_drillcollar = HydraulicsCalculations.calculate_pressure_loss_pipe(
            flow_rate, pv, yp, dc_id, dc_length, mud_weight)
        
        # Bit pressure loss
        dp_bit = HydraulicsCalculations.calculate_bit_pressure_loss(
            flow_rate, mud_weight, nozzle_sizes)
        
        # Annulus pressure loss (DC annulus)
        dp_annulus_dc = HydraulicsCalculations.calculate_pressure_loss_annulus(
            flow_rate, pv, yp, hole_id, dc_od, dc_length, mud_weight)
        
        # Annulus pressure loss (DP annulus)
        dp_annulus_dp = HydraulicsCalculations.calculate_pressure_loss_annulus(
            flow_rate, pv, yp, hole_id, dp_od, annulus_length, mud_weight)
        
        # Surface equipment (estimated)
        dp_surface = 50  # Typical value
        
        # Total
        total_pressure = (dp_drillpipe + dp_drillcollar + dp_bit + 
                         dp_annulus_dc + dp_annulus_dp + dp_surface)
        
        return {
            'drillpipe_psi': round(dp_drillpipe, 1),
            'drillcollar_psi': round(dp_drillcollar, 1),
            'bit_psi': round(dp_bit, 1),
            'annulus_dc_psi': round(dp_annulus_dc, 1),
            'annulus_dp_psi': round(dp_annulus_dp, 1),
            'surface_equipment_psi': dp_surface,
            'total_pressure_psi': round(total_pressure, 1)
        }
    
    @staticmethod
    def calculate_ecd(static_mud_weight: float,
                    annular_pressure_loss: float,
                    tvd: float) -> float:
        """
        Calculate Equivalent Circulating Density (ECD)
        
        Args:
            static_mud_weight: Static mud weight in ppg
            annular_pressure_loss: Annular pressure loss in psi
            tvd: True vertical depth in ft
            
        Returns:
            ECD in ppg
        """
        if tvd == 0:
            return static_mud_weight
        
        # ECD = MW + (APL / (0.052 * TVD))
        ecd = static_mud_weight + (annular_pressure_loss / (PhysicalConstants.HYDROSTATIC_GRADIENT * tvd))
        
        return ecd
    
    @staticmethod
    def optimize_hydraulics(flow_rate: float,
                          mud_weight: float,
                          max_pressure: float,
                          optimization_method: str = 'max_bit_hp') -> dict:
        """
        Optimize hydraulics for maximum efficiency
        
        Args:
            flow_rate: Flow rate in gpm
            mud_weight: Mud weight in ppg
            max_pressure: Maximum surface pressure in psi
            optimization_method: 'max_bit_hp' or 'max_impact_force'
            
        Returns:
            dict with optimization results
        """
        if optimization_method == 'max_bit_hp':
            # Maximum bit horsepower: 65% of pressure at bit
            optimal_bit_pressure = 0.65 * max_pressure
            
            # Calculate required TFA
            # TFA = (MW * Q²) / (10858 * ΔPbit)
            tfa = math.sqrt((mud_weight * flow_rate ** 2) / (10858 * optimal_bit_pressure))
            
            # Bit horsepower
            bit_hp = (optimal_bit_pressure * flow_rate) / 1714
            
            return {
                'method': 'Maximum Bit Horsepower',
                'optimal_bit_pressure_psi': round(optimal_bit_pressure, 1),
                'optimal_tfa_in2': round(tfa, 3),
                'bit_horsepower': round(bit_hp, 1),
                'pressure_allocation': '65% at bit, 35% in system'
            }
        
        elif optimization_method == 'max_impact_force':
            # Maximum impact force: 48% of pressure at bit
            optimal_bit_pressure = 0.48 * max_pressure
            
            # TFA calculation
            tfa = math.sqrt((mud_weight * flow_rate ** 2) / (10858 * optimal_bit_pressure))
            
            # Impact force (lbs)
            impact_force = (flow_rate * math.sqrt(mud_weight * optimal_bit_pressure)) / 1930
            
            return {
                'method': 'Maximum Impact Force',
                'optimal_bit_pressure_psi': round(optimal_bit_pressure, 1),
                'optimal_tfa_in2': round(tfa, 3),
                'impact_force_lbs': round(impact_force, 1),
                'pressure_allocation': '48% at bit, 52% in system'
            }
        
        else:
            return {'error': 'Invalid optimization method'}


class SolidsControl:
    """
    Solids control calculations
    
    Manages solids content and removal calculations.
    """
    
    @staticmethod
    def calculate_drilled_solids_fraction(mud_weight: float,
                                         base_fluid_weight: float,
                                         barite_sg: float = 4.2,
                                         drilled_solids_sg: float = 2.6) -> float:
        """
        Calculate fraction of drilled solids in mud
        
        Args:
            mud_weight: Current mud weight in ppg
            base_fluid_weight: Base fluid weight in ppg
            barite_sg: Barite specific gravity
            d8.33ed_so8.33_sg: Drilled solids specific gravity
            
        Returns:
            Drilled solids fraction (0-1)
        """
        # Convert to specific gravities
        mud_sg = mud_weight / 8.33
        base_sg = base_fluid_weight / 8.33
        
        # Simplified calculation assuming no LGS
        if mud_sg <= base_sg:
            return 0.0
        
        drilled_solids = (mud_sg - base_sg) / (drilled_solids_sg - base_sg)
        
        return max(0, min(drilled_solids, 1.0))
    
    @staticmethod
    def dilution_rate_required(current_solids: float,
                              target_solids: float,
                              system_volume: float) -> dict:
        """
        Calculate dilution rate to reduce solids
        
        Args:
            current_solids: Current solids fraction
            target_solids: Target solids fraction
            system_volume: System volume in bbls
            
        Returns:
            dict with dilution requirements
        """
        if current_solids <= target_solids:
            return {
                'dilution_required': False,
                'message': 'Solids within acceptable range'
            }
        
        # Volume of clean fluid needed
        dilution_volume = system_volume * ((current_solids - target_solids) / (1 - target_solids))
        
        # Discard volume (assuming perfect mixing)
        discard_volume = dilution_volume
        
        return {
            'dilution_required': True,
            'dilution_volume_bbls': round(dilution_volume, 1),
            'discard_volume_bbls': round(discard_volume, 1),
            'cycles_required': round(dilution_volume / system_volume, 2)
        }


class MudTreatment:
    """
    Mud treatment and conditioning calculations
    """
    
    @staticmethod
    def calculate_polymer_concentration(product_weight: float,
                                       volume: float) -> float:
        """
        Calculate polymer concentration
        
        Args:
            product_weight: Weight of polymer product in lbs
            volume: Volume of mud in bbls
            
        Returns:
            Concentration in lb/bbl
        """
        return product_weight / volume
    
    @staticmethod
    def lime_treatment(volume: float,
                      target_lime_concentration: float = 1.0) -> dict:
        """
        Calculate lime treatment requirements
        
        Args:
            volume: Volume of mud in bbls
            target_lime_concentration: Target concentration in lb/bbl
            
        Returns:
            dict with treatment requirements
        """
        lime_required = volume * target_lime_concentration
        
        # Sacks (assuming 50 lb sacks)
        sacks = lime_required / 50
        
        return {
            'lime_required_lbs': round(lime_required, 1),
            'sacks_required': round(sacks, 1),
            'target_concentration_lb_bbl': target_lime_concentration
        }
    
    @staticmethod
    def calculate_lost_circulation_material(loss_rate: float,
                                          lcm_type: str = 'medium') -> dict:
        """
        Calculate LCM requirements for lost circulation
        
        Args:
            loss_rate: Loss rate in bbl/hr
            lcm_type: Type of LCM ('fine', 'medium', 'coarse')
            
        Returns:
            dict with LCM treatment recommendations
        """
        # Concentration recommendations
        concentrations = {
            'fine': (5, 15),  # lb/bbl range
            'medium': (10, 30),
            'coarse': (20, 50)
        }
        
        min_conc, max_conc = concentrations.get(lcm_type, (10, 30))
        
        # Severity classification
        if loss_rate < 5:
            severity = 'Minor seepage'
            recommended_conc = min_conc
        elif loss_rate < 20:
            severity = 'Moderate losses'
            recommended_conc = (min_conc + max_conc) / 2
        else:
            severity = 'Severe losses'
            recommended_conc = max_conc
        
        return {
            'loss_rate_bbl_hr': loss_rate,
            'severity': severity,
            'lcm_type': lcm_type,
            'recommended_concentration_lb_bbl': recommended_conc,
            'recommendation': f'Add {recommended_conc} lb/bbl {lcm_type} LCM'
        }


# Convenience functions
def analyze_mud_system(mud_weight: float,
                      rheology: RheologyData,
                      gel_10sec: float,
                      gel_10min: float,
                      filtration: FiltrationProperties) -> dict:
    """
    Complete mud system analysis
    
    Args:
        mud_weight: Mud weight in ppg
        rheology: Rheology data
        gel_10sec: 10-second gel strength
        gel_10min: 10-minute gel strength
        filtration: Filtration properties
        
    Returns:
        dict with complete analysis
    """
    # Rheology analysis
    rheo = RheologyModels(rheology)
    bp_model = rheo.bingham_plastic_model()
    pl_model = rheo.power_law_model()
    gel_analysis = rheo.gel_strength_analysis(gel_10sec, gel_10min)
    
    # Mud weight analysis
    pressure_gradient = MudWeightCalculations.ppg_to_psi_per_ft(mud_weight)
    
    return {
        'mud_weight_ppg': mud_weight,
        'pressure_gradient_psi_ft': round(pressure_gradient, 4),
        'bingham_plastic': bp_model,
        'power_law': pl_model,
        'gel_strength': gel_analysis,
        'filtration': {
            'api_filtrate_ml': filtration.api_filtrate,
            'hthp_filtrate_ml': filtration.hthp_filtrate,
            'filter_cake_32nds': filtration.filter_cake_thickness,
            'quality': 'Good' if filtration.api_filtrate < 10 else 'Poor'
        }
    }
