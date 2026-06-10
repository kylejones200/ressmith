"""
Simple unit tests for rock mechanics calculations.
Tests only what actually exists in the module.
"""

import pytest
from petrosmith.core.rock_mechanics import ElasticProperties


class TestElasticPropertiesSimple:
    """Basic tests for elastic properties."""
    
    def test_youngs_modulus_positive(self):
        """Test that Young's modulus is calculated."""
        vp = 12000  # ft/s
        vs = 7000   # ft/s
        density = 2.5  # g/cc
        
        youngs = ElasticProperties.calculate_youngs_modulus(vp, vs, density)
        
        # Just verify it returns a positive number
        assert youngs > 0
    
    def test_poissons_ratio_range(self):
        """Test that Poisson's ratio is in valid range."""
        vp = 12000
        vs = 7000
        
        poisson = ElasticProperties.calculate_poissons_ratio(vp, vs)
        
        # Valid range for rock
        assert 0.0 <= poisson <= 0.5
    
    def test_sonic_to_velocity(self):
        """Test sonic transit time conversion."""
        sonic_dt = 70  # us/ft
        
        velocity = ElasticProperties.sonic_to_velocity(sonic_dt)
        
        assert velocity > 0
    
    def test_bulk_modulus_positive(self):
        """Test bulk modulus calculation."""
        vp = 12000
        vs = 7000
        density = 2.5
        
        bulk = ElasticProperties.calculate_bulk_modulus(vp, vs, density)
        
        assert bulk > 0
    
    def test_different_velocities(self):
        """Test with different velocity combinations."""
        # Sandstone-like velocities
        poisson_sand = ElasticProperties.calculate_poissons_ratio(10000, 6000)
        
        # Limestone-like velocities
        poisson_lime = ElasticProperties.calculate_poissons_ratio(15000, 8500)
        
        # Both should be valid
        assert 0.0 <= poisson_sand <= 0.5
        assert 0.0 <= poisson_lime <= 0.5


class TestCachingWorks:
    """Verify LRU cache is functioning."""
    
    def test_cache_hit(self):
        """Test that caching reduces computation."""
        vp, vs, density = 12000, 7000, 2.5
        
        # First call
        result1 = ElasticProperties.calculate_youngs_modulus(vp, vs, density)
        
        # Second call with same params
        result2 = ElasticProperties.calculate_youngs_modulus(vp, vs, density)
        
        # Results should be identical
        assert result1 == result2
        
        # Cache should have registered a hit
        cache_info = ElasticProperties.calculate_youngs_modulus.cache_info()
        assert cache_info.hits > 0


@pytest.mark.parametrize("vp,vs", [
    (10000, 6000),
    (12000, 7000),
    (15000, 8500),
    (18000, 10000),
])
def test_poisson_various_velocities(vp, vs):
    """Parametrized test for various velocity pairs."""
    poisson = ElasticProperties.calculate_poissons_ratio(vp, vs)
    
    # Should always be in valid physical range
    assert 0.0 <= poisson <= 0.5
