"""
Unit tests for custom exceptions.
"""

import pytest
from petrosmith.exceptions import (
    PetroSmithError,
    InvalidMudWeightError,
    InvalidDepthError,
    InvalidPressureError,
    InvalidPorosityError,
    KickDetectedError,
    WellboreStabilityError,
    CasingFailureError,
    ConvergenceError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""
    
    def test_all_inherit_from_base(self):
        """Test that all custom exceptions inherit from PetroSmithError."""
        assert issubclass(InvalidMudWeightError, PetroSmithError)
        assert issubclass(InvalidDepthError, PetroSmithError)
        assert issubclass(KickDetectedError, PetroSmithError)
        assert issubclass(WellboreStabilityError, PetroSmithError)


class TestInvalidMudWeightError:
    """Tests for InvalidMudWeightError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = InvalidMudWeightError(25.0)
        assert "25.00 ppg" in str(error)
        assert "8.0" in str(error)
        assert "20.0" in str(error)
    
    def test_custom_range(self):
        """Test custom min/max range."""
        error = InvalidMudWeightError(7.0, min_val=8.5, max_val=18.0)
        assert "7.00 ppg" in str(error)
        assert "8.5" in str(error)
        assert "18.0" in str(error)
    
    def test_attributes(self):
        """Test error attributes."""
        error = InvalidMudWeightError(25.0)
        assert error.mud_weight == 25.0
        assert error.min_val == 8.0
        assert error.max_val == 20.0


class TestInvalidDepthError:
    """Tests for InvalidDepthError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = InvalidDepthError(-1000.0)
        assert "-1000.0 ft" in str(error)
        assert "must be positive" in str(error)
    
    def test_custom_reason(self):
        """Test custom reason message."""
        error = InvalidDepthError(50000.0, reason="exceeds maximum drilling depth")
        assert "50000.0 ft" in str(error)
        assert "exceeds maximum drilling depth" in str(error)


class TestInvalidPressureError:
    """Tests for InvalidPressureError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = InvalidPressureError(30000.0)
        assert "30000.0 psi" in str(error)
        assert "0.0" in str(error)
        assert "25000.0" in str(error)
    
    def test_attributes(self):
        """Test error attributes."""
        error = InvalidPressureError(-500.0)
        assert error.pressure == -500.0
        assert error.min_val == 0.0
        assert error.max_val == 25000.0


class TestInvalidPorosityError:
    """Tests for InvalidPorosityError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = InvalidPorosityError(1.5)
        assert "1.500" in str(error)
        assert "[0.0, 1.0]" in str(error)
        assert "fraction" in str(error)


class TestKickDetectedError:
    """Tests for KickDetectedError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = KickDetectedError(15.5, "HIGH")
        assert "KICK DETECTED" in str(error)
        assert "15.5 bbls" in str(error)
        assert "HIGH" in str(error)
    
    def test_attributes(self):
        """Test error attributes."""
        error = KickDetectedError(10.0, "MEDIUM")
        assert error.pit_gain == 10.0
        assert error.severity == "MEDIUM"


class TestWellboreStabilityError:
    """Tests for WellboreStabilityError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = WellboreStabilityError(
            current_mw=8.5,
            min_mw=10.0,
            max_mw=14.0
        )
        assert "8.50 ppg" in str(error)
        assert "10.00" in str(error)
        assert "14.00" in str(error)
        assert "instability" in str(error).lower()


class TestCasingFailureError:
    """Tests for CasingFailureError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = CasingFailureError(
            failure_mode="burst",
            safety_factor=0.95,
            required=1.1
        )
        assert "burst" in str(error)
        assert "0.95" in str(error)
        assert "1.1" in str(error)


class TestConvergenceError:
    """Tests for ConvergenceError."""
    
    def test_error_message(self):
        """Test error message formatting."""
        error = ConvergenceError(
            calculation="Newton-Raphson",
            iterations=100,
            tolerance=1e-6
        )
        assert "Newton-Raphson" in str(error)
        assert "100" in str(error)
        assert "converge" in str(error).lower()
    
    def test_attributes(self):
        """Test error attributes."""
        error = ConvergenceError("Test", 50, 0.001)
        assert error.calculation == "Test"
        assert error.iterations == 50
        assert error.tolerance == 0.001


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""
    
    def test_raise_and_catch_specific(self):
        """Test raising and catching specific exception."""
        with pytest.raises(InvalidMudWeightError) as exc_info:
            raise InvalidMudWeightError(25.0)
        
        assert exc_info.value.mud_weight == 25.0
    
    def test_catch_as_base_exception(self):
        """Test catching specific exception as base PetroSmithError."""
        with pytest.raises(PetroSmithError):
            raise InvalidMudWeightError(25.0)
    
    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise InvalidDepthError(-1000.0) from e
        except InvalidDepthError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
