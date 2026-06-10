"""
Simple unit tests for production calculations.
Tests only what actually exists in the module.
"""

import pytest
from petrosmith.core.production import ProductionCalculations


class TestIPR:
    """Tests for Inflow Performance Relationship."""
    
    def test_ipr_returns_dict(self):
        """Test that IPR calculation returns a dictionary."""
        result = ProductionCalculations.calculate_inflow_performance_relationship(
            reservoir_pressure=3000,
            productivity_index=2.0,
            max_flow_rate=2000
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_ipr_decreases_with_pwf(self):
        """Test that flow rate decreases as PWF increases."""
        result = ProductionCalculations.calculate_inflow_performance_relationship(
            reservoir_pressure=3000,
            productivity_index=2.0,
            max_flow_rate=2000
        )
        
        # Get two points from the IPR curve
        pressures = sorted(result.keys())
        low_press = pressures[0]
        high_press = pressures[-1]
        
        # At lower PWF (more drawdown), rate should be higher
        assert result[low_press] >= result[high_press]


class TestTPR:
    """Tests for Tubing Performance Relationship."""
    
    def test_tpr_positive_pressure(self):
        """Test that TPR returns positive pressure."""
        pwf = ProductionCalculations.calculate_tubing_performance_relationship(
            wellhead_pressure=100,
            depth=8000,
            tubing_diameter=2.441,
            flow_rate=1000
        )
        
        assert pwf > 0
        # Should be greater than wellhead due to hydrostatic
        assert pwf > 100


class TestWaterCut:
    """Tests for water cut calculations."""
    
    def test_water_cut_zero(self):
        """Test water cut with no water."""
        wc = ProductionCalculations.calculate_water_cut(
            water_production=0,
            oil_production=1000
        )
        
        assert wc == pytest.approx(0.0, abs=0.1)
    
    def test_water_cut_fifty_percent(self):
        """Test water cut at 50%."""
        wc = ProductionCalculations.calculate_water_cut(
            water_production=500,
            oil_production=500
        )
        
        assert wc == pytest.approx(50.0, abs=1.0)


class TestESP:
    """Tests for ESP calculations."""
    
    def test_esp_head_positive(self):
        """Test ESP required head is positive."""
        head = ProductionCalculations.calculate_esp_required_head(
            depth=8000,
            wellhead_pressure=100,
            flow_rate=3000
        )
        
        assert head > 0
        # Should be at least the depth
        assert head > 8000
    
    def test_esp_horsepower_positive(self):
        """Test ESP horsepower is positive."""
        hp = ProductionCalculations.calculate_esp_horsepower(
            flow_rate=3000,
            total_head=8500,
            efficiency=0.70
        )
        
        assert hp > 0


@pytest.mark.parametrize("water,oil,expected_range", [
    (0, 1000, (0, 5)),
    (250, 750, (20, 30)),
    (500, 500, (45, 55)),
    (900, 100, (85, 95)),
])
def test_water_cut_ranges(water, oil, expected_range):
    """Parametrized test for water cut ranges."""
    wc = ProductionCalculations.calculate_water_cut(water, oil)
    
    assert expected_range[0] <= wc <= expected_range[1]
