"""
Unit tests for well control calculations.
"""

import pytest
from petrosmith.core.well_control import (
    KickDetection,
    WellControlProcedures,
    KickData,
    WellGeometry,
    MudProperties
)


class TestKickDetection:
    """Tests for kick detection functionality."""
    
    def test_no_kick_detected(self):
        """Test that small changes don't trigger kick detection."""
        result = KickDetection.detect_kick(
            pit_volume_gain=1.0,
            flow_rate_increase=2.0,
            connection_flow=False
        )
        # Severity is a number (0-10+), low severity < 3
        assert result['severity'] < 3
    
    def test_kick_detected_high_pit_gain(self):
        """Test kick detection with high pit gain."""
        result = KickDetection.detect_kick(
            pit_volume_gain=15.0,
            flow_rate_increase=5.0,
            connection_flow=False
        )
        # High pit gain (>5) + flow increase (>10) should trigger
        assert result['severity'] >= 3
        assert 'KICK' in result['status'] or 'POSSIBLE' in result['status']
    
    def test_kick_detected_connection_flow(self):
        """Test kick detection with flow at connection."""
        result = KickDetection.detect_kick(
            pit_volume_gain=5.0,
            flow_rate_increase=3.0,
            connection_flow=True
        )
        # Connection flow adds +3, pit gain >5 adds +3 = 6+ severity
        assert result['severity'] >= 5
        assert 'KICK DETECTED' in result['status']
    
    def test_formation_pressure_calculation(self):
        """Test formation pressure calculation."""
        fp = KickDetection.calculate_formation_pressure(
            shut_in_drillpipe_pressure=350,
            mud_weight=10.5,
            tvd=10000
        )
        
        # Formation pressure should be > hydrostatic
        hydrostatic = 0.052 * 10.5 * 10000
        assert fp > hydrostatic
        assert fp == pytest.approx(5810, rel=0.01)
    
    def test_kick_intensity_calculation(self):
        """Test kick intensity calculation."""
        formation_pressure = 5810
        tvd = 10000
        
        kick_intensity = KickDetection.calculate_kick_intensity(
            formation_pressure=formation_pressure,
            tvd=tvd
        )
        
        # Kick intensity = formation pressure / (0.052 * tvd)
        expected = formation_pressure / (0.052 * tvd)
        assert kick_intensity == pytest.approx(expected, rel=1e-6)
        assert kick_intensity > 10.5  # Should be higher than original mud weight


class TestWellControlProcedures:
    """Tests for well control procedures."""
    
    @pytest.fixture
    def sample_well_geometry(self):
        """Sample well geometry for testing."""
        return WellGeometry(
            measured_depth=10000,
            true_vertical_depth=9800,
            hole_diameter=8.5,
            drillpipe_od=5.0,
            drillpipe_id=4.276,
            drillcollar_od=6.5,
            drillcollar_id=2.75,
            drillcollar_length=500,
            casing_id=12.615
        )
    
    @pytest.fixture
    def sample_mud_properties(self):
        """Sample mud properties for testing."""
        return MudProperties(
            weight=10.5,
            plastic_viscosity=25,
            yield_point=15,
            funnel_viscosity=45,
            gel_strength_10sec=8,
            gel_strength_10min=12
        )
    
    @pytest.fixture
    def sample_kick_data(self):
        """Sample kick data for testing."""
        return KickData(
            pit_gain=15.0,
            shut_in_drillpipe_pressure=350,
            shut_in_casing_pressure=700,
            formation_pressure=5810,
            kick_intensity=11.17,
            kick_height=200,
            kick_type='gas'
        )
    
    def test_drillers_method_returns_valid_plan(
        self, 
        sample_well_geometry, 
        sample_mud_properties, 
        sample_kick_data
    ):
        """Test that Driller's Method returns a valid kill plan."""
        wcp = WellControlProcedures(
            sample_well_geometry,
            sample_mud_properties,
            sample_kick_data
        )
        
        plan = wcp.drillers_method(pump_rate=8.0, pump_pressure=2500)
        
        # Check required keys exist
        assert 'kill_mud_weight' in plan
        assert 'first_circulation' in plan
        assert 'second_circulation' in plan
        assert 'total_time_minutes' in plan
        
        # Check values are reasonable
        assert plan['kill_mud_weight'] > sample_mud_properties.weight
        assert plan['total_time_minutes'] > 0
        assert plan['first_circulation']['time_minutes'] > 0
        assert plan['second_circulation']['time_minutes'] > 0
    
    def test_kill_mud_weight_higher_than_current(
        self,
        sample_well_geometry,
        sample_mud_properties,
        sample_kick_data
    ):
        """Test that kill mud weight is higher than current mud weight."""
        wcp = WellControlProcedures(
            sample_well_geometry,
            sample_mud_properties,
            sample_kick_data
        )
        
        plan = wcp.drillers_method(pump_rate=8.0, pump_pressure=2500)
        
        assert plan['kill_mud_weight'] > sample_mud_properties.weight
        # Should be at least kick intensity
        assert plan['kill_mud_weight'] >= sample_kick_data.kick_intensity


class TestWellGeometry:
    """Tests for WellGeometry dataclass."""
    
    def test_well_geometry_creation(self):
        """Test creating a well geometry object."""
        geo = WellGeometry(
            measured_depth=10000,
            true_vertical_depth=9800,
            hole_diameter=8.5,
            drillpipe_od=5.0,
            drillpipe_id=4.276,
            drillcollar_od=6.5,
            drillcollar_id=2.75,
            drillcollar_length=500,
            casing_id=12.615
        )
        
        assert geo.measured_depth == 10000
        assert geo.true_vertical_depth == 9800
        assert geo.hole_diameter > geo.drillpipe_od


class TestMudProperties:
    """Tests for MudProperties dataclass."""
    
    def test_mud_properties_creation(self):
        """Test creating a mud properties object."""
        mud = MudProperties(
            weight=10.5,
            plastic_viscosity=25,
            yield_point=15,
            funnel_viscosity=45,
            gel_strength_10sec=8,
            gel_strength_10min=12
        )
        
        assert mud.weight == 10.5
        assert mud.plastic_viscosity == 25
        assert mud.gel_strength_10min >= mud.gel_strength_10sec


@pytest.mark.parametrize("pit_gain,min_severity", [
    (1.0, 1),
    (3.0, 2),
    (8.0, 3),
    (15.0, 3),
])
def test_kick_severity_levels(pit_gain, min_severity):
    """Parametrized test for kick severity levels."""
    result = KickDetection.detect_kick(
        pit_volume_gain=pit_gain,
        flow_rate_increase=1.0,
        connection_flow=False
    )
    
    # Severity should increase with pit gain
    assert result['severity'] >= min_severity
