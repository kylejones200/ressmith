"""
Integration tests for API -> Service -> Core flow.

These tests verify that the full stack works together correctly.
"""

import pytest
from petrosmith.api import DrillingAPI
from petrosmith.models import DrillingParameters, DrillString, Mud
from petrosmith.exceptions import InvalidMudWeightError, InvalidDepthError


class TestDrillingAPIIntegration:
    """Integration tests for Drilling API."""
    
    def test_hydrostatic_pressure_full_stack(self):
        """Test hydrostatic pressure calculation through full stack."""
        api = DrillingAPI()
        
        # Valid inputs
        pressure = api.calculate_hydrostatic_pressure(mud_weight=10.5, tvd=10000)
        assert pressure == pytest.approx(5460.0, rel=1e-6)
        
        # Invalid mud weight should raise custom exception
        with pytest.raises(InvalidMudWeightError) as exc_info:
            api.calculate_hydrostatic_pressure(mud_weight=25.0, tvd=10000)
        assert "25.00 ppg" in str(exc_info.value)
        assert "8.0, 20.0" in str(exc_info.value)
        
        # Invalid depth should raise custom exception
        with pytest.raises(InvalidDepthError) as exc_info:
            api.calculate_hydrostatic_pressure(mud_weight=10.5, tvd=-1000)
        assert "-1000" in str(exc_info.value)
    
    def test_ecd_calculation_full_stack(self):
        """Test ECD calculation through full stack."""
        api = DrillingAPI()
        
        # Valid inputs
        ecd = api.calculate_ecd(
            mud_weight=10.5,
            annular_pressure_loss=300,
            tvd=10000
        )
        expected = 10.5 + (300 / (0.052 * 10000))
        assert ecd == pytest.approx(expected, rel=1e-6)
        
        # Zero TVD should raise exception
        with pytest.raises(InvalidDepthError):
            api.calculate_ecd(mud_weight=10.5, annular_pressure_loss=300, tvd=0)
    
    def test_casing_design_full_stack(self):
        """Test casing design calculation through full stack."""
        api = DrillingAPI()
        
        result = api.calculate_casing_design(
            outer_diameter=9.625,
            wall_thickness=0.545,
            yield_strength=80000
        )
        
        assert 'burst_pressure' in result
        assert 'collapse_pressure' in result
        assert 'inner_diameter' in result
        assert result['burst_pressure'] > 0
        assert result['collapse_pressure'] > 0
        assert result['inner_diameter'] == pytest.approx(9.625 - 2 * 0.545, rel=1e-6)
    
    def test_kill_mud_weight_full_stack(self):
        """Test kill mud weight calculation through full stack."""
        api = DrillingAPI()
        
        kill_weight = api.calculate_kill_mud_weight(
            formation_pressure=4500,
            tvd=8000,
            safety_margin=0.5
        )
        
        assert kill_weight > 0
        # Kill weight should be higher than formation pressure equivalent
        formation_equiv = 4500 / (0.052 * 8000)
        assert kill_weight > formation_equiv


class TestStatelessServiceIntegration:
    """Integration tests for stateless service layer."""
    
    def test_hydraulics_analysis_stateless(self):
        """Test hydraulics analysis with stateless service."""
        api = DrillingAPI()
        
        # Create data models
        drilling_params = DrillingParameters(
            measured_depth=10000,
            true_vertical_depth=10000,
            weight_on_bit=30.0,
            rotary_speed=100,
            pump_rate=400,
            standpipe_pressure=3000,
            rate_of_penetration=50.0
        )
        
        drill_string = DrillString(
            bit_diameter=8.5,
            bit_type="PDC",
            drill_collar_od=6.75,
            drill_collar_id=2.75,
            drill_collar_length=600,
            drill_pipe_od=5.0,
            drill_pipe_id=4.276,
            drill_pipe_length=9400
        )
        
        mud = Mud(
            mud_id="MUD-001",
            mud_type="water-based",
            density=10.5,
            viscosity=25,
            yield_point=15,
            ph=9.5
        )
        
        # Call stateless method
        result = api.analyze_hydraulics(
            drilling_params=drilling_params,
            drill_string=drill_string,
            mud=mud,
            pump_rate=400,
            hole_diameter=8.5
        )
        
        # Verify results
        assert 'hydrostatic_pressure' in result
        assert 'annular_velocity' in result
        assert 'equivalent_circulating_density' in result
        assert 'warnings' in result
        
        assert result['hydrostatic_pressure'] > 0
        assert result['annular_velocity'] > 0
        assert result['equivalent_circulating_density'] >= mud.density
    
    def test_well_control_analysis_stateless(self):
        """Test well control analysis with stateless service."""
        api = DrillingAPI()
        
        drilling_params = DrillingParameters(
            measured_depth=10000,
            true_vertical_depth=10000,
            weight_on_bit=30.0,
            rotary_speed=100,
            pump_rate=400,
            standpipe_pressure=3000,
            rate_of_penetration=50.0
        )
        
        mud = Mud(
            mud_id="MUD-001",
            mud_type="water-based",
            density=10.5,
            viscosity=25,
            yield_point=15,
            ph=9.5
        )
        
        # Analyze kick
        result = api.analyze_kick(
            drilling_params=drilling_params,
            mud=mud,
            pit_gain=15.0,
            drcp=500,
            dcpp=600
        )
        
        # Verify results
        assert 'kick_analysis' in result
        assert 'kill_parameters' in result
        assert 'recommendations' in result
        
        assert result['kick_analysis']['pit_gain'] == 15.0
        assert result['kill_parameters']['kill_mud_weight'] > mud.density
        assert len(result['recommendations']) > 0
    
    def test_drilling_optimization_stateless(self):
        """Test drilling parameter optimization with stateless service."""
        api = DrillingAPI()
        
        drilling_params = DrillingParameters(
            measured_depth=10000,
            true_vertical_depth=10000,
            weight_on_bit=30.0,
            rotary_speed=100,
            pump_rate=400,
            standpipe_pressure=3000,
            rate_of_penetration=50.0
        )
        
        drill_string = DrillString(
            bit_diameter=8.5,
            bit_type="PDC",
            drill_collar_od=6.75,
            drill_collar_id=2.75,
            drill_collar_length=600,
            drill_pipe_od=5.0,
            drill_pipe_id=4.276,
            drill_pipe_length=9400
        )
        
        result = api.optimize_drilling_parameters(
            drilling_params=drilling_params,
            drill_string=drill_string,
            target_rop=60.0
        )
        
        # Verify results
        assert 'current_parameters' in result
        assert 'optimized_parameters' in result
        assert 'limits' in result
        
        assert result['optimized_parameters']['weight_on_bit'] > 0
        assert result['optimized_parameters']['rotary_speed'] > 0
        assert result['limits']['critical_rpm'] > result['optimized_parameters']['rotary_speed']


class TestErrorHandlingIntegration:
    """Integration tests for error handling across layers."""
    
    def test_api_validates_before_core(self):
        """Test that API layer validates inputs before passing to core."""
        api = DrillingAPI()
        
        # API should catch invalid mud weight
        with pytest.raises(InvalidMudWeightError):
            api.calculate_hydrostatic_pressure(mud_weight=30.0, tvd=10000)
        
        # API should catch negative depth
        with pytest.raises(InvalidDepthError):
            api.calculate_hydrostatic_pressure(mud_weight=10.5, tvd=-5000)
    
    def test_pydantic_model_validation(self):
        """Test that Pydantic models validate inputs."""
        from pydantic import ValidationError
        
        # Invalid mud density (negative)
        with pytest.raises(ValidationError):
            Mud(
                mud_id="MUD-001",
                mud_type="water-based",
                density=-10.5,  # Invalid
                viscosity=25,
                yield_point=15,
                ph=9.5
            )
        
        # Invalid drilling parameters
        with pytest.raises(ValidationError):
            DrillingParameters(
                measured_depth=-1000,  # Invalid
                true_vertical_depth=10000,
                weight_on_bit=30.0,
                rotary_speed=100,
                pump_rate=400,
                standpipe_pressure=3000,
                rate_of_penetration=50.0
            )


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_drilling_analysis_workflow(self):
        """Test a complete drilling analysis workflow."""
        api = DrillingAPI()
        
        # Step 1: Calculate hydrostatic pressure
        mud_weight = 10.5
        tvd = 10000
        hydrostatic = api.calculate_hydrostatic_pressure(mud_weight, tvd)
        assert hydrostatic > 0
        
        # Step 2: Calculate ECD
        annular_loss = 300
        ecd = api.calculate_ecd(mud_weight, annular_loss, tvd)
        assert ecd > mud_weight
        
        # Step 3: Design casing
        casing = api.calculate_casing_design(
            outer_diameter=9.625,
            wall_thickness=0.545,
            yield_strength=80000
        )
        assert casing['burst_pressure'] > hydrostatic
        
        # Step 4: Calculate kill mud weight (if needed)
        formation_pressure = hydrostatic + 500  # Overpressured
        kill_weight = api.calculate_kill_mud_weight(formation_pressure, tvd)
        assert kill_weight > mud_weight
        
        # All calculations completed successfully
        assert True
