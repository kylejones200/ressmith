"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_mud_weight():
    """Standard mud weight for testing."""
    return 10.5  # ppg


@pytest.fixture
def sample_depth():
    """Standard depth for testing."""
    return 10000.0  # ft


@pytest.fixture
def sample_well_geometry():
    """Sample well geometry for well control tests."""
    from petrosmith.core.well_control import WellGeometry
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
def sample_mud_properties():
    """Sample mud properties for testing."""
    from petrosmith.core.well_control import MudProperties
    return MudProperties(
        weight=10.5,
        plastic_viscosity=25,
        yield_point=15,
        funnel_viscosity=45,
        gel_strength_10sec=8,
        gel_strength_10min=12
    )
