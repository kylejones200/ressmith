"""
Well Control Module (WELCON Model)

This module implements well control calculations and kick detection/analysis
including:
- Kick detection and analysis
- Kill procedures (Driller's Method, Wait & Weight, etc.)
- Choke pressure calculations
- Gas migration modeling
- Wellbore pressure profiles during kicks
- BOP response and control procedures

Based on industry-standard well control principles.
"""

from __future__ import annotations
from .constants import PhysicalConstants

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class KickData:
    """Container for kick data"""
    pit_gain: float  # Pit gain in bbls
    shut_in_drillpipe_pressure: float  # SIDPP in psi
    shut_in_casing_pressure: float  # SICP in psi
    formation_pressure: float  # Formation pressure in psi
    kick_intensity: float  # Kick intensity in ppg
    kick_height: float  # Height of kick in wellbore (ft)
    kick_type: str  # 'gas', 'oil', or 'water'


@dataclass
class WellGeometry:
    """Wellbore geometry data"""
    measured_depth: float  # MD in ft
    true_vertical_depth: float  # TVD in ft
    hole_diameter: float  # inches
    drillpipe_od: float  # Drillpipe OD in inches
    drillpipe_id: float  # Drillpipe ID in inches
    drillcollar_od: float  # Drill collar OD in inches
    drillcollar_id: float  # Drill collar ID in inches
    drillcollar_length: float  # Drill collar length in ft
    casing_id: float  # Casing ID in inches


@dataclass
class MudProperties:
    """Drilling mud properties"""
    weight: float  # Mud weight in ppg
    plastic_viscosity: float  # PV in cp
    yield_point: float  # YP in lb/100ft²
    funnel_viscosity: float  # Funnel viscosity in sec/qt
    gel_strength_10sec: float  # 10-second gel strength in lb/100ft²
    gel_strength_10min: float  # 10-minute gel strength in lb/100ft²


class KickDetection:
    """
    Kick detection and early warning system
    
    Monitors drilling parameters for kick indicators.
    """
    
    @staticmethod
    def detect_kick(pit_volume_gain: float = 0.0,
                   flow_rate_increase: float = 0.0,
                   pump_rate_decrease: bool = False,
                   drilling_break: bool = False,
                   connection_flow: bool = False) -> dict:
        """
        Detect potential kick based on indicators
        
        Args:
            pit_volume_gain: Pit gain in bbls
            flow_rate_increase: Flow rate increase percentage
            pump_rate_decrease: Pump rate decreased but flow increased
            drilling_break: Sudden increase in ROP
            connection_flow: Flow observed during connection
            
        Returns:
            dict with kick detection results
        """
        indicators = []
        severity = 0
        
        # Primary indicator: pit gain
        if pit_volume_gain > 5.0:
            indicators.append('significant_pit_gain')
            severity += 3
        elif pit_volume_gain > 2.0:
            indicators.append('moderate_pit_gain')
            severity += 2
        elif pit_volume_gain > 0.5:
            indicators.append('minor_pit_gain')
            severity += 1
        
        # Secondary indicators
        if flow_rate_increase > 10.0:
            indicators.append('flow_rate_increase')
            severity += 2
        
        if pump_rate_decrease:
            indicators.append('pump_rate_flow_anomaly')
            severity += 2
        
        if drilling_break:
            indicators.append('drilling_break')
            severity += 1
        
        if connection_flow:
            indicators.append('connection_flow')
            severity += 3
        
        # Determine kick likelihood
        if severity >= 5:
            kick_status = 'KICK DETECTED - IMMEDIATE ACTION REQUIRED'
            action = 'SHUT IN WELL'
        elif severity >= 3:
            kick_status = 'POSSIBLE KICK - MONITOR CLOSELY'
            action = 'INCREASE MONITORING, PREPARE TO SHUT IN'
        elif severity >= 1:
            kick_status = 'ANOMALY DETECTED'
            action = 'VERIFY INDICATORS, INCREASE MONITORING'
        else:
            kick_status = 'NORMAL'
            action = 'CONTINUE OPERATIONS'
        
        return {
            'status': kick_status,
            'severity': severity,
            'indicators': indicators,
            'recommended_action': action
        }
    
    @staticmethod
    def calculate_formation_pressure(shut_in_drillpipe_pressure: float,
                                    mud_weight: float,
                                    tvd: float) -> float:
        """
        Calculate formation pressure from SIDPP
        
        Args:
            shut_in_drillpipe_pressure: SIDPP in psi
            mud_weight: Mud weight in ppg
            tvd: True vertical depth in ft
            
        Returns:
            Formation pressure in psi
        """
        # Formation pressure = Hydrostatic pressure + SIDPP
        hydrostatic_pressure = 0.052 * mud_weight * tvd
        formation_pressure = hydrostatic_pressure + shut_in_drillpipe_pressure
        
        return formation_pressure
    
    @staticmethod
    def calculate_kick_intensity(formation_pressure: float,
                                 tvd: float) -> float:
        """
        Calculate kick intensity (equivalent mud weight)
        
        Args:
            formation_pressure: Formation pressure in psi
            tvd: True vertical depth in ft
            
        Returns:
            Kick intensity in ppg
        """
        if tvd <= 0:
            return 0.0
        
        kick_intensity = formation_pressure / (0.052 * tvd)
        return kick_intensity


class WellControlProcedures:
    """
    Well control kill procedures
    
    Implements various well kill methods including Driller's Method,
    Wait & Weight, and Concurrent methods.
    """
    
    def __init__(self, well_geometry: WellGeometry, mud_props: MudProperties,
                 kick_data: KickData):
        """
        Initialize well control procedure
        
        Args:
            well_geometry: Wellbore geometry
            mud_props: Current mud properties
            kill_data: Kick data
        """
        self.geometry = well_geometry
        self.mud = mud_props
        self.kick = kick_data
        
    def calculate_kill_mud_weight(self, safety_margin: float = 0.5) -> float:
        """
        Calculate required kill mud weight
        
        Args:
            safety_margin: Safety margin in ppg
            
        Returns:
            Kill mud weight in ppg
        """
        kill_mud_weight = self.kick.kick_intensity + safety_margin
        return kill_mud_weight
    
    def drillers_method(self, pump_rate: float, pump_pressure: float) -> dict:
        """
        Driller's Method kill procedure
        
        This is a two-circulation method:
        1. First circulation: circulate kick out with original mud
        2. Second circulation: circulate kill mud through system
        
        Args:
            pump_rate: Pump rate in gpm
            pump_pressure: Initial circulating pressure in psi
            
        Returns:
            dict with kill procedure parameters
        """
        # Calculate kill mud weight
        kmw = self.calculate_kill_mud_weight()
        
        # Initial circulating pressure (ICP) = SIDPP + pump pressure at kill rate
        icp = self.kick.shut_in_drillpipe_pressure + pump_pressure
        
        # Final circulating pressure (FCP) for first circulation
        # FCP should decline to pump pressure as kick reaches surface
        fcp_first = pump_pressure
        
        # Calculate annular pressure loss with original mud
        apl_original = self._calculate_annular_pressure_loss(
            self.mud.weight, pump_rate)
        
        # Calculate annular pressure loss with kill mud
        apl_kill = self._calculate_annular_pressure_loss(kmw, pump_rate)
        
        # ICP for second circulation
        icp_second = pump_pressure + apl_kill - apl_original
        
        # FCP for second circulation (pump pressure only)
        fcp_second = pump_pressure
        
        # Calculate volumes and time
        annular_volume = self._calculate_annular_volume()
        strokes_to_bit = self._calculate_drillpipe_capacity() / pump_rate
        time_to_bit = strokes_to_bit / pump_rate * PhysicalConstants.MINUTES_PER_HOUR  # minutes
        
        return {
            'method': 'Drillers Method',
            'kill_mud_weight': kmw,
            'first_circulation': {
                'icp': icp,
                'fcp': fcp_first,
                'mud_weight': self.mud.weight,
                'time_minutes': time_to_bit + annular_volume / pump_rate
            },
            'second_circulation': {
                'icp': icp_second,
                'fcp': fcp_second,
                'mud_weight': kmw,
                'time_minutes': time_to_bit + annular_volume / pump_rate
            },
            'total_time_minutes': 2 * (time_to_bit + annular_volume / pump_rate),
            'choke_pressure_schedule': self._generate_pressure_schedule(
                icp, fcp_first, strokes_to_bit)
        }
    
    def wait_and_weight_method(self, pump_rate: float, pump_pressure: float) -> dict:
        """
        Wait and Weight Method kill procedure
        
        This is a single-circulation method where kill mud is pumped
        immediately after mixing.
        
        Args:
            pump_rate: Pump rate in gpm
            pump_pressure: Initial circulating pressure in psi
            
        Returns:
            dict with kill procedure parameters
        """
        # Calculate kill mud weight
        kmw = self.calculate_kill_mud_weight()
        
        # Calculate new ICP with kill mud
        apl_original = self._calculate_annular_pressure_loss(
            self.mud.weight, pump_rate)
        apl_kill = self._calculate_annular_pressure_loss(kmw, pump_rate)
        
        # ICP = SIDPP + pump pressure + (kill mud weight - original weight) * 0.052 * TVD
        pressure_increase = (kmw - self.mud.weight) * PhysicalConstants.HYDROSTATIC_GRADIENT * self.geometry.true_vertical_depth
        icp = self.kick.shut_in_drillpipe_pressure + pump_pressure + pressure_increase
        
        # FCP with kill mud
        fcp = pump_pressure + apl_kill - apl_original
        
        # Calculate volumes and time
        drillpipe_capacity = self._calculate_drillpipe_capacity()
        annular_volume = self._calculate_annular_volume()
        strokes_to_bit = drillpipe_capacity / pump_rate
        
        return {
            'method': 'Wait and Weight Method',
            'kill_mud_weight': kmw,
            'icp': icp,
            'fcp': fcp,
            'time_minutes': (drillpipe_capacity + annular_volume) / pump_rate,
            'strokes_to_bit': strokes_to_bit,
            'choke_pressure_schedule': self._generate_pressure_schedule(
                icp, fcp, strokes_to_bit)
        }
    
    def concurrent_method(self, pump_rate: float, pump_pressure: float) -> dict:
        """
        Concurrent Method (Bullheading)
        
        Used when unable to circulate normally.
        
        Args:
            pump_rate: Pump rate in gpm
            pump_pressure: Initial circulating pressure in psi
            
        Returns:
            dict with kill procedure parameters
        """
        # Calculate kill mud weight
        kmw = self.calculate_kill_mud_weight()
        
        # Calculate pressure required to force kick back into formation
        formation_fracture_pressure = self._estimate_fracture_pressure()
        
        # Maximum allowable pressure
        max_pressure = min(
            formation_fracture_pressure * 0.9,  # 90% of fracture pressure
            formation_fracture_pressure - 200  # Or FP - 200 psi
        )
        
        # Required bullheading pressure
        required_pressure = self.kick.formation_pressure + pump_pressure
        
        return {
            'method': 'Concurrent Method (Bullheading)',
            'kill_mud_weight': kmw,
            'required_pressure': required_pressure,
            'max_allowable_pressure': max_pressure,
            'feasible': required_pressure < max_pressure,
            'warning': 'Use only when normal circulation is not possible'
        }
    
    def _calculate_annular_volume(self) -> float:
        """Calculate annular volume in bbls"""
        # Annular capacity = (D_hole^2 - D_pipe^2) / 1029.4
        annular_cap = (self.geometry.hole_diameter**2 - 
                      self.geometry.drillpipe_od**2) / 1029.4
        volume = annular_cap * self.geometry.measured_depth
        return volume
    
    def _calculate_drillpipe_capacity(self) -> float:
        """Calculate drillpipe capacity in bbls"""
        # Capacity = D_id^2 / 1029.4
        dp_cap = self.geometry.drillpipe_id**2 / 1029.4
        volume = dp_cap * self.geometry.measured_depth
        return volume
    
    def _calculate_annular_pressure_loss(self, mud_weight: float, 
                                        flow_rate: float) -> float:
        """Calculate annular pressure loss in psi"""
        # Simplified calculation using Fanning friction factor
        # APL ≈ (PV * V * L) / (300 * Dh)
        # where V is velocity, L is length, Dh is hydraulic diameter
        
        # Hydraulic diameter
        dh = self.geometry.hole_diameter - self.geometry.drillpipe_od
        
        # Velocity in annulus (ft/min)
        annular_area = (self.geometry.hole_diameter**2 - 
                       self.geometry.drillpipe_od**2) / 183.3
        velocity = flow_rate / annular_area
        
        # Pressure loss
        apl = (self.mud.plastic_viscosity * velocity * 
              self.geometry.measured_depth) / (300 * dh)
        
        return apl
    
    def _generate_pressure_schedule(self, icp: float, fcp: float, 
                                   strokes: float) -> list[tuple[int, float]]:
        """Generate choke pressure schedule"""
        # Generate pressure schedule at 10-stroke intervals
        schedule = []
        num_points = int(strokes / 10) + 1
        
        for i in range(num_points):
            stroke_number = i * 10
            if stroke_number > strokes:
                stroke_number = strokes
            
            # Linear pressure decline
            pressure = icp - (icp - fcp) * (stroke_number / strokes)
            schedule.append((int(stroke_number), round(pressure, 1)))
        
        return schedule
    
    def _estimate_fracture_pressure(self) -> float:
        """Estimate formation fracture pressure"""
        # Simplified fracture gradient estimation
        # Fracture gradient typically 0.8-1.0 psi/ft depending on depth
        
        if self.geometry.true_vertical_depth < 5000:
            fracture_gradient = 0.8
        elif self.geometry.true_vertical_depth < 10000:
            fracture_gradient = 0.9
        else:
            fracture_gradient = 1.0
        
        fracture_pressure = fracture_gradient * self.geometry.true_vertical_depth
        return fracture_pressure


class GasMigration:
    """
    Gas migration modeling
    
    Models gas kick migration in wellbore after shut-in.
    """
    
    @staticmethod
    def calculate_migration_rate(gas_gradient: float = 0.1,
                                 mud_weight: float = 10.0,
                                 wellbore_diameter: float = 8.5) -> float:
        """
        Calculate gas migration rate
        
        Args:
            gas_gradient: Gas gradient in psi/ft
            mud_weight: Mud weight in ppg
            wellbore_diameter: Wellbore diameter in inches
            
        Returns:
            Migration rate in ft/hr
        """
        # Simplified migration rate calculation
        # Migration rate increases with pressure differential
        
        mud_gradient = 0.052 * mud_weight
        pressure_differential = mud_gradient - gas_gradient
        
        # Migration rate typically 100-1000 ft/hr
        # Higher differential = higher rate
        migration_rate = 100 + pressure_differential * 100
        
        return max(0, min(migration_rate, 1000))
    
    @staticmethod
    def calculate_pressure_increase(initial_gas_depth: float,
                                   migration_distance: float,
                                   gas_gradient: float,
                                   mud_weight: float) -> float:
        """
        Calculate pressure increase due to gas migration
        
        Args:
            initial_gas_depth: Initial depth of gas top (ft)
            migration_distance: Distance gas migrated (ft)
            gas_gradient: Gas gradient in psi/ft
            mud_weight: Mud weight in ppg
            
        Returns:
            Pressure increase in psi
        """
        # As gas migrates up, it displaces heavier mud
        # Pressure increase = (mud gradient - gas gradient) * migration distance
        
        mud_gradient = 0.052 * mud_weight
        pressure_increase = (mud_gradient - gas_gradient) * migration_distance
        
        return pressure_increase
    
    @staticmethod
    def gas_expansion(initial_pressure: float, initial_volume: float,
                     final_pressure: float, temperature: float = 150.0,
                     gas_gravity: float = 0.6) -> float:
        """
        Calculate gas expansion using real gas law
        
        Args:
            initial_pressure: Initial pressure in psia
            initial_volume: Initial volume in bbls
            final_pressure: Final pressure in psia
            temperature: Temperature in °F
            gas_gravity: Gas specific gravity (air=1.0)
            
        Returns:
            Final volume in bbls
        """
        # Using real gas law: P1*V1/Z1 = P2*V2/Z2
        # Simplified: assuming Z1 ≈ Z2 for moderate pressures
        
        # Convert temperature to Rankine
        T_rankine = temperature + 460
        
        # Simplified compressibility factor (Z) estimation
        # For low to moderate pressures, Z ≈ 1
        Z1 = 1.0
        Z2 = 1.0
        
        # Calculate final volume
        final_volume = initial_volume * (initial_pressure / final_pressure) * (Z2 / Z1)
        
        return final_volume


class ChokeManagement:
    """
    Choke pressure management during well control
    
    Calculates required choke pressures and manages choke operations.
    """
    
    @staticmethod
    def calculate_choke_pressure(drillpipe_pressure: float,
                                target_pressure: float,
                                current_strokes: int,
                                total_strokes: int,
                                icp: float,
                                fcp: float) -> float:
        """
        Calculate required choke pressure
        
        Args:
            drillpipe_pressure: Current drillpipe pressure in psi
            target_pressure: Target drillpipe pressure from schedule
            current_strokes: Current pump strokes
            total_strokes: Total strokes to complete circulation
            icp: Initial circulating pressure in psi
            fcp: Final circulating pressure in psi
            
        Returns:
            Required choke adjustment in psi
        """
        # Calculate expected pressure from schedule
        expected_pressure = icp - (icp - fcp) * (current_strokes / total_strokes)
        
        # Required choke adjustment
        adjustment = drillpipe_pressure - expected_pressure
        
        return adjustment
    
    @staticmethod
    def maximum_allowable_annular_pressure(casing_pressure_test: float,
                                          mud_weight: float,
                                          tvd: float,
                                          safety_factor: float = 0.9) -> float:
        """
        Calculate maximum allowable annular surface pressure (MAASP)
        
        Args:
            casing_pressure_test: Casing pressure test value in psi
            mud_weight: Current mud weight in ppg
            tvd: True vertical depth in ft
            safety_factor: Safety factor (typically 0.8-0.9)
            
        Returns:
            MAASP in psi
        """
        # MAASP = (Casing test pressure * safety factor) - hydrostatic pressure
        hydrostatic = 0.052 * mud_weight * tvd
        maasp = (casing_pressure_test * safety_factor) - hydrostatic
        
        return max(0, maasp)


class KickSimulation:
    """
    Kick simulation and analysis
    
    Simulates kick behavior and wellbore response.
    """
    
    def __init__(self, well_geometry: WellGeometry, mud_props: MudProperties):
        """Initialize kick simulation"""
        self.geometry = well_geometry
        self.mud = mud_props
        
    def simulate_gas_kick(self, formation_pressure: float,
                         permeability: float,
                         time_step: float = 1.0,
                         duration: float = 10.0) -> dict:
        """
        Simulate gas kick influx
        
        Args:
            formation_pressure: Formation pressure in psi
            permeability: Formation permeability in md
            time_step: Time step for simulation in minutes
            duration: Total simulation duration in minutes
            
        Returns:
            dict with simulation results
0.052   """
        # Calculate bottomhole pressure
        bhp = PhysicalConstants.HYDROSTATIC_GRADIENT * self.mud.weight * self.geometry.true_vertical_depth
        
        # Pressure differential
        pressure_diff = formation_pressure - bhp
        
        if pressure_diff <= 0:
            return {
                'kick_occurred': False,
                'message': 'No kick - mud weight sufficient'
            }
        
        # Simulate kick influx over time
        time_points = np.arange(0, duration, time_step)
        influx_rate = self._calculate_influx_rate(pressure_diff, permeability)
        
        cumulative_influx = []
        pit_gain = []
        
        for t in time_points:
            # Influx rate decreases as kick enters wellbore
            current_influx = influx_rate * np.exp(-0.1 * t)
            cum_influx = current_influx * t
            cumulative_influx.append(cum_influx)
            pit_gain.append(cum_influx * 5.615)  # Convert to bbls
        
        return {
            'kick_occurred': True,
            'time_minutes': time_points.tolist(),
            'influx_rate_cuft_min': influx_rate,
            'cumulative_influx_cuft': cumulative_influx,
            'pit_gain_bbls': pit_gain,
            'formation_pressure': formation_pressure,
            'bottomhole_pressure': bhp,
            'pressure_differential': pressure_diff
        }
    
    def _calculate_influx_rate(self, pressure_diff: float, 
                              permeability: float) -> float:
        """Calculate gas influx rate using Darcy's law"""
        # Simplified Darcy's law for radial flow
        # q = (2 * pi * k * h * dp) / (mu * ln(re/rw))
        
        h = 10  # Assume 10 ft pay zone
        mu = 0.02  # Gas viscosity (cp)
        re = 1000  # Drainage radius (ft)
        rw = self.geometry.hole_diameter / (2 * 12)  # Convert to ft
        
        if re <= rw:
            return 0.0
        
        q = (2 * math.pi * permeability * h * pressure_diff) / (mu * math.log(re / rw))
        
        # Convert from cu-ft/day to cu-ft/min
        q_cuft_min = q / (1440)
        
        return max(0, q_cuft_min)


# Legacy class for backward compatibility
class WellControlCalculations:
    """
    Legacy well control calculations class for backward compatibility
    
    Wraps the new well control functionality with the old interface.
    """
    
    @staticmethod
    def calculate_kill_mud_weight(
        formation_pressure: float,
        tvd: float = None,
        *,
        true_vertical_depth: float = None,
        safety_margin: float = 0.5,
    ) -> float:
        """Calculate kill mud weight from formation pressure and TVD."""
        depth = true_vertical_depth if true_vertical_depth is not None else tvd
        if depth is None:
            raise ValueError("Must provide tvd or true_vertical_depth")
        kick_intensity = KickDetection.calculate_kick_intensity(formation_pressure, depth)
        return kick_intensity + safety_margin

    @staticmethod
    def calculate_formation_pressure(
        sidpp: float = None,
        mud_weight: float = None,
        tvd: float = None,
        *,
        true_vertical_depth: float = None,
    ) -> float:
        """Calculate formation pressure from SIDPP, mud weight, and TVD."""
        depth = true_vertical_depth if true_vertical_depth is not None else tvd
        if sidpp is None or mud_weight is None or depth is None:
            raise ValueError("Must provide sidpp, mud_weight, and tvd or true_vertical_depth")
        return KickDetection.calculate_formation_pressure(sidpp, mud_weight, depth)

    @staticmethod
    def calculate_kick_volume(pit_gain: float) -> tuple[float, int]:
        """
        Return kick volume and severity from pit gain.

        Returns:
            (kick_volume_bbls, severity_score)
        """
        detection = KickDetection.detect_kick(pit_volume_gain=pit_gain)
        return pit_gain, detection["severity"]

    @staticmethod
    def calculate_initial_circulating_pressure(
        slow_pump_rate_pressure: float,
        shut_in_drillpipe_pressure: float = None,
        *,
        kill_mud_weight: float = None,
        original_mud_weight: float = None,
    ) -> float:
        """
        Initial circulating pressure when opening choke to circulate kick out.

        ICP = SPP + SIDPP, or ICP = SPP * (KMW/OMW) if SIDPP not available.
        """
        if shut_in_drillpipe_pressure is not None:
            return slow_pump_rate_pressure + shut_in_drillpipe_pressure
        if kill_mud_weight is not None and original_mud_weight is not None and original_mud_weight > 0:
            return slow_pump_rate_pressure * (kill_mud_weight / original_mud_weight)
        return slow_pump_rate_pressure

    @staticmethod
    def calculate_final_circulating_pressure(
        slow_pump_rate_pressure: float,
        kill_mud_weight: float = None,
        original_mud_weight: float = None,
    ) -> float:
        """
        Final circulating pressure when kill mud has filled the wellbore.

        FCP = SPP * (KMW/OMW)
        """
        if kill_mud_weight is not None and original_mud_weight is not None and original_mud_weight > 0:
            return slow_pump_rate_pressure * (kill_mud_weight / original_mud_weight)
        return slow_pump_rate_pressure


# Convenience functions
def detect_and_analyze_kick(pit_gain: float,
                           sidpp: float,
                           sicp: float,
                           mud_weight: float,
                           tvd: float) -> dict:
    """
    Complete kick detection and analysis
    
    Args:
        pit_gain: Pit volume gain in bbls
        sidpp: Shut-in drillpipe pressure in psi
        sicp: Shut-in casing pressure in psi
        mud_weight: Mud weight in ppg
        tvd: True vertical depth in ft
        
    Returns:
        dict with kick analysis results
    """
    # Detect kick
    detection = KickDetection.detect_kick(pit_volume_gain=pit_gain)
    
    # Calculate formation pressure
    fp = KickDetection.calculate_formation_pressure(sidpp, mud_weight, tvd)
    
    # Calculate kick intensity
    kick_intensity = KickDetection.calculate_kick_intensity(fp, tvd)
    
    # Estimate kick type based on pressures
    pressure_ratio = sicp / max(sidpp, 1.0)
    if pressure_ratio > 2.0:
        kick_type = 'gas'
    elif pressure_ratio > 1.3:
        kick_type = 'oil'
    else:
        kick_type = 'water'
    
    return {
        'detection': detection,
        'formation_pressure_psi': fp,
        'kick_intensity_ppg': kick_intensity,
        'kick_type': kick_type,
        'pressure_ratio': pressure_ratio,
        'recommended_kmw': kick_intensity + 0.5
    }
