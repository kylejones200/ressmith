"""
Unit tests for reservoir engineering calculations.
"""

import pytest
from petrosmith.core.reservoir import ReservoirCalculations


class TestDarcyFlow:
    """Tests for Darcy flow calculations."""
    
    def test_darcy_flow_rate_positive(self):
        """Test that Darcy flow rate is positive."""
        flow_rate = ReservoirCalculations.calculate_darcy_flow_rate(
            permeability=100,  # md
            thickness=50,  # ft
            pressure_drawdown=500,  # psi
            viscosity=1.0,  # cp
            formation_volume_factor=1.2,
            drainage_radius=1000,  # ft
            wellbore_radius=0.328  # ft
        )
        
        assert flow_rate > 0
    
    def test_higher_permeability_more_flow(self):
        """Test that higher permeability gives more flow."""
        params = {
            'thickness': 50,
            'pressure_drawdown': 500,
            'viscosity': 1.0,
            'formation_volume_factor': 1.2,
            'drainage_radius': 1000,
            'wellbore_radius': 0.328
        }
        
        flow_low_perm = ReservoirCalculations.calculate_darcy_flow_rate(
            permeability=50, **params
        )
        
        flow_high_perm = ReservoirCalculations.calculate_darcy_flow_rate(
            permeability=200, **params
        )
        
        assert flow_high_perm > flow_low_perm


class TestProductivityIndex:
    """Tests for productivity index calculations."""
    
    def test_pi_calculation(self):
        """Test productivity index calculation."""
        pi = ReservoirCalculations.calculate_productivity_index(
            flow_rate=1000,  # STB/day
            reservoir_pressure=3000,  # psi
            bottomhole_pressure=2000  # psi
        )
        
        # PI = q / (Pr - Pwf)
        expected = 1000 / (3000 - 2000)
        assert pi == pytest.approx(expected, rel=1e-6)
    
    def test_pi_zero_drawdown(self):
        """Test PI with zero drawdown."""
        with pytest.raises(ValueError):
            ReservoirCalculations.calculate_productivity_index(
                flow_rate=1000,
                reservoir_pressure=2000,
                bottomhole_pressure=2000  # Same as reservoir
            )


class TestOOIP:
    """Tests for Original Oil In Place calculations."""
    
    def test_ooip_positive(self):
        """Test that OOIP is positive."""
        ooip = ReservoirCalculations.calculate_original_oil_in_place(
            area=640,  # acres
            net_pay=50,  # ft
            porosity=0.20,
            oil_saturation=0.70,
            formation_volume_factor=1.2
        )
        
        assert ooip > 0
    
    def test_ooip_increases_with_area(self):
        """Test that OOIP increases with area."""
        params = {
            'net_pay': 50,
            'porosity': 0.20,
            'oil_saturation': 0.70,
            'formation_volume_factor': 1.2
        }
        
        ooip_small = ReservoirCalculations.calculate_original_oil_in_place(
            area=320, **params
        )
        
        ooip_large = ReservoirCalculations.calculate_original_oil_in_place(
            area=640, **params
        )
        
        assert ooip_large > ooip_small
        # Should be roughly double
        assert ooip_large == pytest.approx(ooip_small * 2, rel=0.01)


class TestOGIP:
    """Tests for Original Gas In Place calculations."""
    
    def test_ogip_positive(self):
        """Test that OGIP is positive."""
        ogip = ReservoirCalculations.calculate_original_gas_in_place(
            area=640,
            net_pay=50,
            porosity=0.15,
            gas_saturation=0.75,
            formation_volume_factor=0.005
        )
        
        assert ogip > 0


class TestRecoveryFactor:
    """Tests for recovery factor calculations."""
    
    def test_recovery_factor_range(self):
        """Test that recovery factor is in valid range."""
        rf = ReservoirCalculations.calculate_recovery_factor(
            initial_pressure=4000,
            abandonment_pressure=400,
            drive_mechanism='solution_gas'
        )
        
        # Recovery factor should be between 0 and 1
        assert 0 < rf < 1
    
    def test_water_drive_higher_recovery(self):
        """Test that water drive has higher recovery than solution gas."""
        params = {
            'initial_pressure': 4000,
            'abandonment_pressure': 400
        }
        
        rf_solution_gas = ReservoirCalculations.calculate_recovery_factor(
            drive_mechanism='solution_gas', **params
        )
        
        rf_water_drive = ReservoirCalculations.calculate_recovery_factor(
            drive_mechanism='water_drive', **params
        )
        
        # Water drive typically has better recovery
        assert rf_water_drive > rf_solution_gas


class TestMaterialBalance:
    """Tests for material balance calculations."""
    
    def test_material_balance_pressure(self):
        """Test material balance pressure calculation (simple compressibility method)."""
        current_pressure = ReservoirCalculations.calculate_material_balance_pressure_simple(
            initial_pressure=4000,
            cumulative_production=1000000,  # STB
            original_in_place=10000000,  # STB
            compressibility=1e-5  # 1/psi
        )
        
        # Pressure should decrease with production
        assert current_pressure < 4000
        assert current_pressure > 0


class TestPZGasMaterialBalance:
    """Tests for p/Z dry gas material balance."""

    def test_ogip_pz_positive(self):
        """OGIP from p/Z should be positive and exceed Gp."""
        ogip = ReservoirCalculations.calculate_original_gas_in_place_pz(
            initial_pressure=4000,
            initial_z_factor=0.92,
            cumulative_gas_production=2e9,  # scf
            current_pressure=3200,
            current_z_factor=0.88,
        )
        assert ogip > 2e9
        assert ogip > 0

    def test_ogip_pz_formula(self):
        """OGIP = Gp / (1 - (p/Z)/(pi/Zi)); verify round-trip."""
        pi, zi = 4000.0, 0.92
        Gp = 2e9
        p, z = 3200.0, 0.88
        ogip = ReservoirCalculations.calculate_original_gas_in_place_pz(
            initial_pressure=pi,
            initial_z_factor=zi,
            cumulative_gas_production=Gp,
            current_pressure=p,
            current_z_factor=z,
        )
        # Recover pressure from MB: p_calc = z * (pi/zi) * (1 - Gp/OGIP)
        p_calc = ReservoirCalculations.calculate_pressure_from_pz_material_balance(
            initial_pressure=pi,
            initial_z_factor=zi,
            cumulative_gas_production=Gp,
            original_gas_in_place=ogip,
            current_z_factor=z,
        )
        assert p_calc == pytest.approx(p, rel=1e-6)

    def test_pressure_from_pz_decreases_with_Gp(self):
        """Current pressure should decrease as Gp increases."""
        p1 = ReservoirCalculations.calculate_pressure_from_pz_material_balance(
            initial_pressure=4000,
            initial_z_factor=0.92,
            cumulative_gas_production=1e9,
            original_gas_in_place=10e9,
            current_z_factor=0.89,
        )
        p2 = ReservoirCalculations.calculate_pressure_from_pz_material_balance(
            initial_pressure=4000,
            initial_z_factor=0.92,
            cumulative_gas_production=3e9,
            original_gas_in_place=10e9,
            current_z_factor=0.86,
        )
        assert p2 < p1

    def test_ogip_pz_invalid_raises(self):
        """p/Z must decrease with production."""
        with pytest.raises(ValueError):
            ReservoirCalculations.calculate_original_gas_in_place_pz(
                initial_pressure=3000,
                initial_z_factor=0.9,
                cumulative_gas_production=1e9,
                current_pressure=3500,  # pressure increased - invalid
                current_z_factor=0.9,
            )


@pytest.mark.parametrize("permeability,expected_range", [
    (10, (0, 500)),
    (100, (0, 5000)),
    (500, (0, 25000)),
])
def test_darcy_flow_ranges(permeability, expected_range):
    """Parametrized test for Darcy flow at different permeabilities."""
    flow_rate = ReservoirCalculations.calculate_darcy_flow_rate(
        permeability=permeability,
        thickness=50,
        pressure_drawdown=500,
        viscosity=1.0,
        formation_volume_factor=1.2,
        drainage_radius=1000,
        wellbore_radius=0.328
    )
    
    assert expected_range[0] <= flow_rate <= expected_range[1]
