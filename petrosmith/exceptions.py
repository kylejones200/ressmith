"""
Custom exceptions for PetroSmith library.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the library.
"""

from __future__ import annotations


class PetroSmithError(Exception):
    """
    Base exception for all PetroSmith library errors.
    
    All custom exceptions should inherit from this class.
    """
    pass


# ============================================================================
# Input Validation Errors
# ============================================================================

class ValidationError(PetroSmithError):
    """Base class for validation errors."""
    pass


class InvalidMudWeightError(ValidationError):
    """Raised when mud weight is outside acceptable range."""
    
    def __init__(self, mud_weight: float, min_val: float = 8.0, max_val: float = 20.0):
        self.mud_weight = mud_weight
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            f"Mud weight {mud_weight:.2f} ppg is outside acceptable range "
            f"[{min_val:.1f}, {max_val:.1f}] ppg"
        )


class InvalidDepthError(ValidationError):
    """Raised when depth value is invalid."""
    
    def __init__(self, depth: float, reason: str = "must be positive"):
        self.depth = depth
        self.reason = reason
        super().__init__(f"Invalid depth {depth:.1f} ft: {reason}")


class InvalidPressureError(ValidationError):
    """Raised when pressure value is invalid."""
    
    def __init__(self, pressure: float, min_val: float = 0.0, max_val: float = 25000.0):
        self.pressure = pressure
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            f"Pressure {pressure:.1f} psi is outside acceptable range "
            f"[{min_val:.1f}, {max_val:.1f}] psi"
        )


class InvalidPorosityError(ValidationError):
    """Raised when porosity is outside valid range."""
    
    def __init__(self, porosity: float):
        self.porosity = porosity
        super().__init__(
            f"Porosity {porosity:.3f} is outside valid range [0.0, 1.0] "
            "(must be a fraction, not percentage)"
        )


class InvalidPermeabilityError(ValidationError):
    """Raised when permeability is invalid."""
    
    def __init__(self, permeability: float):
        self.permeability = permeability
        super().__init__(
            f"Permeability {permeability:.3f} md must be positive"
        )


# ============================================================================
# Calculation Errors
# ============================================================================

class CalculationError(PetroSmithError):
    """Base class for calculation errors."""
    pass


class ConvergenceError(CalculationError):
    """Raised when iterative calculation fails to converge."""
    
    def __init__(self, calculation: str, iterations: int, tolerance: float):
        self.calculation = calculation
        self.iterations = iterations
        self.tolerance = tolerance
        super().__init__(
            f"{calculation} failed to converge after {iterations} iterations "
            f"(tolerance: {tolerance:.2e})"
        )


class NumericalInstabilityError(CalculationError):
    """Raised when numerical instability is detected."""
    
    def __init__(self, calculation: str, reason: str):
        self.calculation = calculation
        self.reason = reason
        super().__init__(
            f"Numerical instability in {calculation}: {reason}"
        )


# ============================================================================
# Well Control Errors
# ============================================================================

class WellControlError(PetroSmithError):
    """Base class for well control errors."""
    pass


class KickDetectedError(WellControlError):
    """Raised when a kick is detected."""
    
    def __init__(self, pit_gain: float, severity: str):
        self.pit_gain = pit_gain
        self.severity = severity
        super().__init__(
            f"KICK DETECTED: Pit gain {pit_gain:.1f} bbls, Severity: {severity}"
        )


class InvalidKillProcedureError(WellControlError):
    """Raised when kill procedure parameters are invalid."""
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid kill procedure: {reason}")


# ============================================================================
# Drilling Errors
# ============================================================================

class DrillingError(PetroSmithError):
    """Base class for drilling operation errors."""
    pass


class WellboreStabilityError(DrillingError):
    """Raised when wellbore stability criteria are violated."""
    
    def __init__(self, current_mw: float, min_mw: float, max_mw: float):
        self.current_mw = current_mw
        self.min_mw = min_mw
        self.max_mw = max_mw
        super().__init__(
            f"Wellbore instability: Current MW {current_mw:.2f} ppg is outside "
            f"safe window [{min_mw:.2f}, {max_mw:.2f}] ppg"
        )


class CasingFailureError(DrillingError):
    """Raised when casing design fails safety criteria."""
    
    def __init__(self, failure_mode: str, safety_factor: float, required: float):
        self.failure_mode = failure_mode
        self.safety_factor = safety_factor
        self.required = required
        super().__init__(
            f"Casing {failure_mode} failure: Safety factor {safety_factor:.2f} "
            f"is below required {required:.2f}"
        )


# ============================================================================
# Formation Errors
# ============================================================================

class FormationError(PetroSmithError):
    """Base class for formation-related errors."""
    pass


class AbnormalPressureError(FormationError):
    """Raised when abnormal formation pressure is detected."""
    
    def __init__(self, pressure_gradient: float, depth: float):
        self.pressure_gradient = pressure_gradient
        self.depth = depth
        super().__init__(
            f"Abnormal pressure detected: {pressure_gradient:.3f} psi/ft at {depth:.0f} ft "
            f"(Normal: ~0.465 psi/ft)"
        )


class FractureRiskError(FormationError):
    """Raised when formation fracture risk is detected."""
    
    def __init__(self, ecd: float, fracture_gradient: float, depth: float):
        self.ecd = ecd
        self.fracture_gradient = fracture_gradient
        self.depth = depth
        super().__init__(
            f"Fracture risk at {depth:.0f} ft: ECD {ecd:.2f} ppg exceeds "
            f"fracture gradient {fracture_gradient:.3f} psi/ft"
        )


# ============================================================================
# Data Errors
# ============================================================================

class DataError(PetroSmithError):
    """Base class for data-related errors."""
    pass


class MissingDataError(DataError):
    """Raised when required data is missing."""
    
    def __init__(self, data_name: str, context: str = ""):
        self.data_name = data_name
        self.context = context
        msg = f"Missing required data: {data_name}"
        if context:
            msg += f" (context: {context})"
        super().__init__(msg)


class InvalidDataFormatError(DataError):
    """Raised when data format is invalid."""
    
    def __init__(self, data_name: str, expected_format: str, actual: str):
        self.data_name = data_name
        self.expected_format = expected_format
        self.actual = actual
        super().__init__(
            f"Invalid format for {data_name}: expected {expected_format}, "
            f"got {actual}"
        )


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(PetroSmithError):
    """Base class for configuration errors."""
    pass


class UnsupportedOperationError(ConfigurationError):
    """Raised when an operation is not supported for the given configuration."""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Operation '{operation}' not supported: {reason}")


# ============================================================================
# Export all exceptions
# ============================================================================

__all__ = [
    # Base
    'PetroSmithError',
    
    # Validation
    'ValidationError',
    'InvalidMudWeightError',
    'InvalidDepthError',
    'InvalidPressureError',
    'InvalidPorosityError',
    'InvalidPermeabilityError',
    
    # Calculation
    'CalculationError',
    'ConvergenceError',
    'NumericalInstabilityError',
    
    # Well Control
    'WellControlError',
    'KickDetectedError',
    'InvalidKillProcedureError',
    
    # Drilling
    'DrillingError',
    'WellboreStabilityError',
    'CasingFailureError',
    
    # Formation
    'FormationError',
    'AbnormalPressureError',
    'FractureRiskError',
    
    # Data
    'DataError',
    'MissingDataError',
    'InvalidDataFormatError',
    
    # Configuration
    'ConfigurationError',
    'UnsupportedOperationError',
]
