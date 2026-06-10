"""
Well Testing Module

This module implements comprehensive well testing analysis including:
- Drawdown tests
- Buildup tests
- Multi-rate tests
- Type curve analysis
- Boundary detection
- Well test interpretation

Based on standard petroleum engineering well testing principles.
"""

import numpy as np
import logging
from .constants import PhysicalConstants

# Configure module logger
logger = logging.getLogger(__name__)
from typing import Optional, Union
from dataclasses import dataclass
import math


@dataclass
class WellTestData:
    """Container for well test data"""
    time: np.ndarray  # Time in hours
    pressure: np.ndarray  # Pressure in psi
    rate: np.ndarray  # Flow rate in STB/day
    test_type: str  # 'drawdown', 'buildup', 'multirate'
    
    def __post_init__(self):
        """Validate data arrays"""
        if len(self.time) != len(self.pressure):
            raise ValueError("Time and pressure arrays must have same length")
        if len(self.time) != len(self.rate):
            raise ValueError("Time and rate arrays must have same length")


@dataclass
class ReservoirProperties:
    """Reservoir properties derived from well tests"""
    permeability: float  # md
    skin_factor: float  # dimensionless
    wellbore_storage: float  # bbl/psi
    initial_pressure: float  # psi
    formation_thickness: float  # ft
    porosity: float  # fraction
    total_compressibility: float  # 1/psi
    reservoir_radius: float  # ft (if bounded)
    drainage_area: float  # acres


class DrawdownAnalysis:
    """
    Drawdown test analysis
    
    Analyzes constant-rate drawdown tests to determine reservoir
    properties including permeability and skin factor.
    """
    
    def __init__(self, test_data: WellTestData, fluid_properties: dict):
        """
        Initialize drawdown analysis
        
        Args:
            test_data: Well test data
            fluid_properties: dict with 'viscosity' (cp), 'FVF' (rb/stb),
                            'compressibility' (1/psi)
        """
        self.data = test_data
        self.mu = fluid_properties['viscosity']
        self.B = fluid_properties['FVF']
        self.ct = fluid_properties['compressibility']
        
    def log_log_analysis(self) -> dict:
        """
        Log-log analysis for flow regime identification
        
        Returns:
            dict with:
                - flow_regimes: list of identified flow regimes
                - derivative: Pressure derivative
                - log_time: Log of time
                - log_pressure_change: Log of pressure change
        """
        # Calculate pressure change
        dp = self.data.pressure[0] - self.data.pressure
        
        # Calculate derivative
        dt = np.diff(self.data.time)
        ddp = np.diff(dp)
        derivative = np.abs(ddp / dt)
        
        # Remove zeros for log calculations
        valid_idx = (dp[1:] > 0) & (derivative > 0)
        
        log_time = np.log10(self.data.time[1:][valid_idx])
        log_dp = np.log10(dp[1:][valid_idx])
        log_derivative = np.log10(derivative[valid_idx])
        
        # Identify flow regimes based on derivative behavior
        regimes = self._identify_flow_regimes(log_time, log_derivative)
        
        return {
            'flow_regimes': regimes,
            'derivative': derivative[valid_idx],
            'log_time': log_time,
            'log_pressure_change': log_dp,
            'log_derivative': log_derivative
        }
    
    def _identify_flow_regimes(self, log_time: np.ndarray, 
                               log_derivative: np.ndarray) -> list[dict]:
        """Identify flow regimes from derivative behavior"""
        regimes = []
        
        # Wellbore storage: slope = 1
        # Radial flow: flat derivative (slope = 0)
        # Boundary effect: upward trending derivative
        
        if len(log_time) < 3:
            return regimes
        
        # Simple regime identification
        slopes = np.diff(log_derivative) / np.diff(log_time)
        
        for i in range(len(slopes)):
            if slopes[i] > 0.8:
                regimes.append({
                    'type': 'wellbore_storage',
                    'start_time': 10**log_time[i],
                    'end_time': 10**log_time[i+1]
                })
            elif abs(slopes[i]) < 0.1:
                regimes.append({
                    'type': 'radial_flow',
                    'start_time': 10**log_time[i],
                    'end_time': 10**log_time[i+1]
                })
            elif slopes[i] > 0.3:
                regimes.append({
                    'type': 'boundary_effect',
                    'start_time': 10**log_time[i],
                    'end_time': 10**log_time[i+1]
                })
        
        return regimes
    
    def semilog_analysis(self, h: float, rw: float, phi: float) -> ReservoirProperties:
        """
        Semi-log analysis for permeability and skin calculation
        
        Args:
            h: Formation thickness (ft)
            rw: Wellbore radius (ft)
            phi: Porosity (fraction)
            
        Returns:
            ReservoirProperties object
        """
        # Calculate log time
        log_time = np.log10(self.data.time)
        
        # Find straight line portion (radial flow)
        # This is simplified - in practice would use more sophisticated detection
        mid_start = len(self.data.time) // 3
        mid_end = 2 * len(self.data.time) // 3
        
        # Linear regression on middle portion
        x = log_time[mid_start:mid_end]
        y = self.data.pressure[mid_start:mid_end]
        
        # Calculate slope (m) and intercept
        A = np.vstack([x, np.ones(len(x))]).T
        m, p1hr = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate permeability from slope
        # k = 162.6 * q * B * mu / (m * h)
        q = np.mean(self.data.rate)
        k = 162.6 * q * self.B * self.mu / (abs(m) * h)
        
        # Calculate skin factor
        # s = 1.151 * [(p1hr - pi) / m - log(k / (phi * mu * ct * rw^2)) + 3.23]
        pi = self.data.pressure[0]
        
        # Estimate p1hr from regression
        p1hr_est = m * 0 + p1hr  # log10(1) = 0
        
        skin = PhysicalConstants.SKIN_FACTOR_RATIO * ((p1hr_est - pi) / m - 
                       math.log10(k / (phi * self.mu * self.ct * rw**2)) + 3.23)
        
        # Estimate wellbore storage from early-time data
        C = self._calculate_wellbore_storage()
        
        # Estimate drainage radius (if bounded)
        drainage_radius = self._estimate_drainage_radius(k, h, phi)
        
        return ReservoirProperties(
            permeability=k,
            skin_factor=skin,
            wellbore_storage=C,
            initial_pressure=pi,
            formation_thickness=h,
            porosity=phi,
            total_compressibility=self.ct,
            reservoir_radius=drainage_radius,
            drainage_area=math.pi * drainage_radius**2 / 43560  # Convert to acres
        )
    
    def _calculate_wellbore_storage(self) -> float:
        """Calculate wellbore storage coefficient"""
        # From early time data, C = q * t / dp (during unit slope)
        if len(self.data.time) < 5:
            return 0.0
        
        # Use first few data points
        early_time = self.data.time[:5]
        early_dp = self.data.pressure[0] - self.data.pressure[:5]
        early_rate = self.data.rate[:5]
        
        # Avoid division by zero
        valid = early_dp > 1.0
        if not np.any(valid):
            return 0.0
        
        C_values = early_rate[valid] * early_time[valid] / (24 * early_dp[valid])
        return np.mean(C_values)
    
    def _estimate_drainage_radius(self, k: float, h: float, phi: float) -> float:
        """Estimate drainage radius from late-time data"""
        # Check for boundary effects in late-time data
        # This is simplified - actual implementation would be more sophisticated
        
        # If pressure is still declining linearly, assume no boundary yet
        late_pressures = self.data.pressure[-10:]
        if np.std(np.diff(late_pressures)) < 1.0:
            return 1000.0  # Default large radius
        
        # Otherwise estimate from deviation
        # Using simplified correlation
        tpss = self.data.time[-1]  # Pseudo-steady state time
        re = math.sqrt(0.000264 * k * tpss / (phi * self.mu * self.ct))
        
        return min(re, 2000.0)  # Cap at reasonable value


class BuildupAnalysis:
    """
    Pressure buildup test analysis
    
    Analyzes pressure buildup after shut-in to determine reservoir
    properties.
    """
    
    def __init__(self, test_data: WellTestData, fluid_properties: dict,
                 production_time: float):
        """
        Initialize buildup analysis
        
        Args:
            test_data: Well test data (shut-in period)
            fluid_properties: dict with viscosity, FVF, compressibility
            production_time: Production time before shut-in (hours)
        """
        self.data = test_data
        self.mu = fluid_properties['viscosity']
        self.B = fluid_properties['FVF']
        self.ct = fluid_properties['compressibility']
        self.tp = production_time
        
    def horner_analysis(self, h: float, rw: float, phi: float) -> ReservoirProperties:
        """
        Horner analysis for pressure buildup
        
        Args:
            h: Formation thickness (ft)
            rw: Wellbore radius (ft)
            phi: Porosity (fraction)
            
        Returns:
            ReservoirProperties object
        """
        # Calculate Horner time
        dt = self.data.time  # Shut-in time
        horner_time = (self.tp + dt) / dt
        log_horner_time = np.log10(horner_time)
        
        # Find straight line portion
        mid_start = len(self.data.pressure) // 4
        mid_end = 3 * len(self.data.pressure) // 4
        
        # Linear regression
        x = log_horner_time[mid_start:mid_end]
        y = self.data.pressure[mid_start:mid_end]
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, pstar = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate permeability
        mid_start = len(self.data.time) // 3
        mid_point = len(self.data.time) * 2 // 3
        mid_end = min(mid_point + 50, len(self.data.time))
        
        q = np.mean(np.abs(self.data.rate[mid_point:mid_end]))
        k = PhysicalConstants.LOG_SLOPE_TO_PERM * q * self.B * self.mu / (abs(m) * h)
        
        # Calculate skin factor
        pws = self.data.pressure[0]  # Shut-in pressure
        p1hr = m * math.log10((self.tp + 1) / 1) + PhysicalConstants.SKIN_FACTOR_RATIO
        
        skin = PhysicalConstants.SKIN_FACTOR_RATIO * ((p1hr - pws) / m - 
                       math.log10(k / (phi * self.mu * self.ct * rw**2)) + 3.23)
        
        # Extrapolate to infinite shut-in time for average pressure
        # When horner_time = 1 (infinite time), we get p*
        avg_pressure = pstar
        
        return ReservoirProperties(
            permeability=k,
            skin_factor=skin,
            wellbore_storage=0.0,  # Not typically calculated from buildup
            initial_pressure=avg_pressure,
            formation_thickness=h,
            porosity=phi,
            total_compressibility=self.ct,
            reservoir_radius=1000.0,  # Default value
            drainage_area=math.pi * 1000**2 / 43560
        )
    
    def mdr_analysis(self) -> dict:
        """
        MDR (Miller-Dyes-Hutchinson) analysis
        
        Returns:
            dict with analysis results
        """
        # Calculate pressure change
        dp = self.data.pressure - self.data.pressure[0]
        
        # MDR plot: pws vs log(dt)
        log_dt = np.log10(self.data.time)
        
        # Find straight line
        mid_start = len(self.data.pressure) // 4
        mid_end = 3 * len(self.data.pressure) // 4
        
        x = log_dt[mid_start:mid_end]
        y = self.data.pressure[mid_start:mid_end]
        
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return {
            'slope': m,
            'intercept': c,
            'log_time': log_dt,
            'pressure': self.data.pressure
        }


class MultirateAnalysis:
    """
    Multi-rate test analysis
    
    Analyzes tests with multiple flow rate periods to determine
    reservoir properties.
    """
    
    def __init__(self, test_data: WellTestData, fluid_properties: dict):
        """
        Initialize multirate analysis
        
        Args:
            test_data: Well test data with varying rates
            fluid_properties: dict with viscosity, FVF, compressibility
        """
        self.data = test_data
        self.mu = fluid_properties['viscosity']
        self.B = fluid_properties['FVF']
        self.ct = fluid_properties['compressibility']
        
    def superposition_analysis(self, rate_changes: list[tuple[float, float]],
                               h: float, rw: float, phi: float) -> ReservoirProperties:
        """
        Superposition analysis for multi-rate tests
        
        Args:
            rate_changes: list of (time, new_rate) tuples
            h: Formation thickness (ft)
            rw: Wellbore radius (ft)
            phi: Porosity (fraction)
            
        Returns:
            ReservoirProperties object
        """
        # Calculate rate-normalized pressure
        # This uses superposition principle
        
        # Build superposition time function
        superposition_time = np.zeros_like(self.data.time)
        
        for i, t in enumerate(self.data.time):
            # Find active rate changes up to time t
            active_rates = [(0, self.data.rate[0])]  # Initial rate
            for tc, qc in rate_changes:
                if tc < t:
                    active_rates.append((tc, qc))
            
            # Calculate superposition time
            st = 0.0
            for j in range(len(active_rates)):
                tc, qc = active_rates[j]
                if j < len(active_rates) - 1:
                    q_delta = active_rates[j+1][1] - qc
                else:
                    q_delta = self.data.rate[i] - qc
                
                if abs(q_delta) > 0.01:
                    st += q_delta * math.log10(t - tc + 1e-6)
            
            superposition_time[i] = st
        
        # Normalize pressure by rate
        current_rate = self.data.rate
        p_normalized = self.data.pressure / (current_rate + 1e-6)
        
        # Semilog analysis on normalized pressure
        mid_start = len(superposition_time) // 3
        mid_end = 2 * len(superposition_time) // 3
        
        x = superposition_time[mid_start:mid_end]
        y = p_normalized[mid_start:mid_end]
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate permeability
        q_avg = np.mean(self.data.rate)
        k = 162.6 * q_avg * self.B * self.mu / (abs(m) * h)
        
        # Calculate skin
        pi = self.data.pressure[0]
        skin = PhysicalConstants.SKIN_FACTOR_RATIO * ((c - pi / q_avg) / m - 
                       math.log10(k / (phi * self.mu * self.ct * rw**2)) + 3.23)
        
        return ReservoirProperties(
            permeability=k,
            skin_factor=skin,
            wellbore_storage=0.0,
            initial_pressure=pi,
            formation_thickness=h,
            porosity=phi,
            total_compressibility=self.ct,
            reservoir_radius=1000.0,
            drainage_area=math.pi * 1000**2 / 43560
        )


class TypeCurveAnalysis:
    """
    Type curve matching for well test analysis
    
    Provides dimensionless type curve matching for various
    reservoir models.
    """
    
    @staticmethod
    def generate_dimensionless_pressure(tD: np.ndarray, CD: float = 100,
                                       skin: float = 0) -> np.ndarray:
        """
        Generate dimensionless pressure response
        
        Args:
            tD: Dimensionless time array
            CD: Dimensionless wellbore storage
            skin: Skin factor
            
        Returns:
            Dimensionless pressure array
        """
        pD = np.zeros_like(tD)
        
        for i, t in enumerate(tD):
            if t < 1e-3:
                # Early time: wellbore storage dominated
                pD[i] = t / CD
            elif t < 100:
                # Transition and radial flow
                pD[i] = 0.5 * (np.log(t) + 0.80907 + 2 * skin)
            else:
                # Late time: boundary effects
                pD[i] = 0.5 * (np.log(t) + 0.80907 + 2 * skin) + t / 1000
        
        return pD
    
    @staticmethod
    def match_type_curve(pressure_data: np.ndarray, time_data: np.ndarray,
                        rate: float, h: float, phi: float, mu: float,
                        ct: float, rw: float) -> dict:
        """
        Match field data to type curves
        
        Args:
            pressure_data: Measured pressure data (psi)
            time_data: Time data (hours)
            rate: Flow rate (STB/day)
            h: Formation thickness (ft)
            phi: Porosity (fraction)
            mu: Viscosity (cp)
            ct: Total compressibility (1/psi)
            rw: Wellbore radius (ft)
            
        Returns:
            dict with match parameters
        """
        # This is a simplified matching algorithm
        # Real implementation would use numerical optimization
        
        # Try different CD and skin values
        best_match = None
        best_error = float('inf')
        
        for CD_exp in range(-2, 8):
            CD = 10 ** CD_exp
            for skin in np.linspace(-5, 10, 16):
                # Generate type curve
                tD = time_data * 0.000264 * 100 / (phi * mu * ct * rw**2)  # Assuming k=100
                pD_curve = TypeCurveAnalysis.generate_dimensionless_pressure(tD, CD, skin)
                
                # Scale to match data
                dp = pressure_data[0] - pressure_data
                dp_max = np.max(dp)
                pD_max = np.max(pD_curve)
                
                if pD_max > 0:
                    scaled_pD = pD_curve * (dp_max / pD_max)
                    error = np.mean((dp - scaled_pD)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_match = {
                            'CD': CD,
                            'skin': skin,
                            'k_estimate': 100 * (dp_max / pD_max),
                            'error': error
                        }
        
        return best_match


class BoundaryDetection:
    """
    Boundary detection from well test data
    
    Identifies reservoir boundaries and their characteristics.
    """
    
    @staticmethod
    def detect_boundaries(test_data: WellTestData, properties: ReservoirProperties) -> dict:
        """
        Detect boundaries from well test data
        
        Args:
            test_data: Well test data
            properties: Reservoir properties
            
        Returns:
            dict with boundary information
        """
        # Calculate pressure derivative
        dt = np.diff(test_data.time)
        dp = np.diff(test_data.pressure)
        derivative = dp / dt
        
        # Look for characteristic boundary signatures
        boundaries = {
            'detected': False,
            'type': None,
            'distance': None,
            'time_to_boundary': None
        }
        
        # Check for boundary effects in late-time data
        if len(derivative) < 10:
            return boundaries
        
        late_derivative = derivative[-10:]
        
        # No-flow boundary: pressure derivative increases
        if np.mean(late_derivative[-5:]) > 2 * np.mean(late_derivative[:5]):
            boundaries['detected'] = True
            boundaries['type'] = 'no_flow'
            
            # Estimate distance to boundary
            # Using simplified correlation
            t_boundary = test_data.time[len(test_data.time) - 10]
            k = properties.permeability
            distance = math.sqrt(0.000264 * k * t_boundary / 
                               (properties.porosity * properties.total_compressibility))
            boundaries['distance'] = distance
            boundaries['time_to_boundary'] = t_boundary
        
        # Constant pressure boundary: pressure derivative decreases
        elif np.mean(late_derivative[-5:]) < 0.5 * np.mean(late_derivative[:5]):
            boundaries['detected'] = True
            boundaries['type'] = 'constant_pressure'
            t_boundary = test_data.time[len(test_data.time) - 10]
            boundaries['time_to_boundary'] = t_boundary
        
        return boundaries


# Convenience function
def analyze_well_test(test_data: WellTestData, 
                     fluid_properties: dict,
                     formation_properties: dict,
                     test_type: str = 'auto') -> dict:
    """
    Comprehensive well test analysis
    
    Args:
        test_data: Well test data
        fluid_properties: Fluid properties (viscosity, FVF, compressibility)
        formation_properties: Formation properties (h, rw, phi)
        test_type: Test type or 'auto' for automatic detection
        
    Returns:
        dict with comprehensive analysis results
    """
    h = formation_properties['thickness']
    rw = formation_properties['wellbore_radius']
    phi = formation_properties['porosity']
    
    results = {
        'test_type': test_type,
        'reservoir_properties': None,
        'flow_regimes': None,
        'boundaries': None
    }
    
    # Detect test type if auto
    if test_type == 'auto':
        if np.all(test_data.rate > 0):
            test_type = 'drawdown'
        elif np.all(test_data.rate == 0):
            test_type = 'buildup'
        else:
            test_type = 'multirate'
        results['test_type'] = test_type
    
    # Perform appropriate analysis
    if test_type == 'drawdown':
        analyzer = DrawdownAnalysis(test_data, fluid_properties)
        results['flow_regimes'] = analyzer.log_log_analysis()
        results['reservoir_properties'] = analyzer.semilog_analysis(h, rw, phi)
        
    elif test_type == 'buildup':
        # Need production time for buildup
        tp = formation_properties.get('production_time', 100.0)
        analyzer = BuildupAnalysis(test_data, fluid_properties, tp)
        results['reservoir_properties'] = analyzer.horner_analysis(h, rw, phi)
        
    elif test_type == 'multirate':
        analyzer = MultirateAnalysis(test_data, fluid_properties)
        # Detect rate changes
        rate_changes = []
        for i in range(1, len(test_data.rate)):
            if abs(test_data.rate[i] - test_data.rate[i-1]) > 1.0:
                rate_changes.append((test_data.time[i], test_data.rate[i]))
        results['reservoir_properties'] = analyzer.superposition_analysis(
            rate_changes, h, rw, phi)
    
    # Boundary detection
    if results['reservoir_properties']:
        results['boundaries'] = BoundaryDetection.detect_boundaries(
            test_data, results['reservoir_properties'])
    
    return results
