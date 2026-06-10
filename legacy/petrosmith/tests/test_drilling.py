"""
Unit tests for drilling calculations.
"""

import pytest
from petrosmith.core.drilling import DrillingCalculations
from petrosmith.exceptions import InvalidMudWeightError, InvalidDepthError


class TestHydrostaticPressure:
    """Tests for hydrostatic pressure calculations."""
    
    def test_standard_calculation(self):
        """Test standard hydrostatic pressure calculation."""
        result = DrillingCalculations.calculate_hydrostatic_pressure(
            true_vertical_depth=10000,
            mud_weight=10.5
        )
        expected = 0.052 * 10.5 * 10000
        assert result == pytest.approx(expected, rel=1e-6)
        assert result == pytest.approx(5460.0, rel=1e-6)
    
    def test_zero_depth(self):
        """Test hydrostatic pressure at zero depth."""
        result = DrillingCalculations.calculate_hydrostatic_pressure(
            true_vertical_depth=0,
            mud_weight=10.5
        )
        assert result == 0.0
    
    def test_negative_mud_weight(self):
        """Test that negative mud weight raises error."""
        with pytest.raises(ValueError, match="Mud weight must be positive"):
            DrillingCalculations.calculate_hydrostatic_pressure(
                true_vertical_depth=10000,
                mud_weight=-1.0
            )
    
    def test_negative_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ValueError, match="TVD cannot be negative"):
            DrillingCalculations.calculate_hydrostatic_pressure(
                true_vertical_depth=-1000,
                mud_weight=10.5
            )
    
    def test_high_mud_weight(self):
        """Test calculation with high mud weight."""
        result = DrillingCalculations.calculate_hydrostatic_pressure(
            true_vertical_depth=10000,
            mud_weight=18.0
        )
        expected = 0.052 * 18.0 * 10000
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_shallow_depth(self):
        """Test calculation at shallow depth."""
        result = DrillingCalculations.calculate_hydrostatic_pressure(
            true_vertical_depth=1000,
            mud_weight=9.0
        )
        expected = 0.052 * 9.0 * 1000
        assert result == pytest.approx(expected, rel=1e-6)


class TestECD:
    """Tests for Equivalent Circulating Density calculations."""
    
    def test_standard_ecd(self):
        """Test standard ECD calculation."""
        result = DrillingCalculations.calculate_equivalent_circulating_density(
            mud_weight=10.5,
            annular_pressure_loss=300,
            true_vertical_depth=10000
        )
        expected = 10.5 + (300 / (0.052 * 10000))
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_zero_pressure_loss(self):
        """Test ECD with zero pressure loss equals static mud weight."""
        result = DrillingCalculations.calculate_equivalent_circulating_density(
            mud_weight=10.5,
            annular_pressure_loss=0,
            true_vertical_depth=10000
        )
        assert result == pytest.approx(10.5, rel=1e-6)
    
    def test_high_pressure_loss(self):
        """Test ECD with high annular pressure loss."""
        result = DrillingCalculations.calculate_equivalent_circulating_density(
            mud_weight=10.5,
            annular_pressure_loss=1000,
            true_vertical_depth=10000
        )
        assert result > 10.5  # ECD should be higher than static MW
        expected = 10.5 + (1000 / (0.052 * 10000))
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_zero_depth_raises_error(self):
        """Test that zero depth raises error."""
        with pytest.raises(ValueError, match="TVD must be positive"):
            DrillingCalculations.calculate_equivalent_circulating_density(
                mud_weight=10.5,
                annular_pressure_loss=300,
                true_vertical_depth=0
            )


class TestAnnularVelocity:
    """Tests for annular velocity calculations."""
    
    def test_standard_velocity(self):
        """Test standard annular velocity calculation."""
        result = DrillingCalculations.calculate_annular_velocity(
            flow_rate_gpm=400,
            hole_diameter=8.5,
            pipe_od=5.0
        )
        assert result > 0
        # Annular area = (8.5^2 - 5^2) / 1029.4
        annular_area = (8.5**2 - 5.0**2) / 1029.4
        expected = 400 / annular_area
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_zero_flow_rate(self):
        """Test that zero flow rate raises error."""
        with pytest.raises(ValueError, match="Flow rate must be positive"):
            DrillingCalculations.calculate_annular_velocity(
                flow_rate_gpm=0,
                hole_diameter=8.5,
                pipe_od=5.0
            )
    
    def test_pipe_larger_than_hole(self):
        """Test that pipe OD > hole diameter raises error."""
        with pytest.raises(ValueError, match="Hole diameter must be greater than pipe OD"):
            DrillingCalculations.calculate_annular_velocity(
                flow_rate_gpm=400,
                hole_diameter=5.0,
                pipe_od=8.5
            )


class TestCasingDesign:
    """Tests for casing design calculations."""
    
    def test_burst_pressure(self):
        """Test casing burst pressure calculation."""
        result = DrillingCalculations.calculate_casing_burst(
            internal_pressure=5000,
            external_pressure=1000,
            yield_strength=80000,
            wall_thickness=0.545,
            od=9.625
        )
        
        assert 'burst_pressure_psi' in result
        assert 'safety_factor' in result
        assert 'status' in result
        assert result['burst_pressure_psi'] > 0
        
        # Barlow's formula: P = 2 * Y * t / OD
        expected_burst = (2 * 80000 * 0.545) / 9.625
        assert result['burst_pressure_psi'] == pytest.approx(expected_burst, rel=1e-6)
    
    def test_collapse_pressure(self):
        """Test casing collapse pressure calculation."""
        result = DrillingCalculations.calculate_casing_collapse(
            external_pressure=5000,
            internal_pressure=1000,
            yield_strength=80000,
            od=9.625,
            wall_thickness=0.545
        )
        
        assert 'collapse_pressure_psi' in result
        assert 'safety_factor' in result
        assert 'status' in result
        assert result['collapse_pressure_psi'] > 0


class TestBitHydraulics:
    """Tests for bit hydraulics calculations."""
    
    def test_hydraulic_horsepower(self):
        """Test hydraulic horsepower calculation."""
        result = DrillingCalculations.calculate_bit_hydraulics(
            pump_pressure=3000,
            flow_rate=400,
            parasitic_loss=500
        )
        
        assert 'bit_pressure_psi' in result
        assert 'hydraulic_horsepower' in result
        assert 'impact_force_lbs' in result
        
        assert result['bit_pressure_psi'] == 2500  # 3000 - 500
        assert result['hydraulic_horsepower'] > 0
        assert result['impact_force_lbs'] > 0


@pytest.mark.parametrize("mud_weight,tvd,expected", [
    (8.5, 5000, 2210.0),
    (10.0, 8000, 4160.0),
    (12.0, 12000, 7488.0),
    (15.0, 15000, 11700.0),
])
def test_hydrostatic_pressure_parametrized(mud_weight, tvd, expected):
    """Parametrized test for various mud weights and depths."""
    result = DrillingCalculations.calculate_hydrostatic_pressure(
        true_vertical_depth=tvd,
        mud_weight=mud_weight
    )
    assert result == pytest.approx(expected, rel=1e-6)
