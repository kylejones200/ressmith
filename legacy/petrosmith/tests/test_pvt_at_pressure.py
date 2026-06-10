"""
Tests for pressure-dependent PVT (Bo(P), Bg(P), Rs(P)) for MB iteration.
"""

import pytest
from petrosmith.core.fluid import FluidCalculations


class TestOilFvfAtPressure:
    """Tests for Bo(P) - oil FVF at pressure."""

    def test_bo_at_pressure_saturated(self):
        """Bo(P) below Pb: as P decreases, Rs decreases, so Bo decreases (less gas in solution)."""
        T, gamma_g, api = 180.0, 0.65, 35.0
        bo_high = FluidCalculations.calculate_oil_fvf_at_pressure(
            3000, T, gamma_g, api, bubble_point_pressure=2500
        )
        bo_low = FluidCalculations.calculate_oil_fvf_at_pressure(
            1500, T, gamma_g, api, bubble_point_pressure=2500
        )
        assert bo_low < bo_high

    def test_bo_at_pressure_above_pb(self):
        """Above Pb, Bo should be slightly larger than Bo at Pb (undersaturated)."""
        T, gamma_g, api, pb = 180.0, 0.65, 35.0, 2500.0
        bo_pb = FluidCalculations.calculate_oil_fvf_at_pressure(
            pb, T, gamma_g, api, bubble_point_pressure=pb
        )
        bo_above = FluidCalculations.calculate_oil_fvf_at_pressure(
            4000, T, gamma_g, api, bubble_point_pressure=pb
        )
        assert bo_above > bo_pb

    def test_bo_at_pressure_no_pb_treats_saturated(self):
        """When bubble_point_pressure is None, use Rs(P) and Standing Bo."""
        Bo = FluidCalculations.calculate_oil_fvf_at_pressure(
            2000, 180, 0.65, 35, bubble_point_pressure=None
        )
        assert Bo > 1.0
        assert Bo < 2.0


class TestGetPvtAtPressure:
    """Tests for get_pvt_at_pressure (Bo, Bg, Rs at P)."""

    def test_oil_pvt_returns_bo_bg_rs(self):
        """Oil PVT at pressure should return Bo, Bg, Rs."""
        pvt = FluidCalculations.get_pvt_at_pressure(
            pressure=2000,
            temperature=180,
            gas_specific_gravity=0.65,
            oil_api_gravity=35,
            fluid="oil",
        )
        assert "Bo" in pvt
        assert "Bg" in pvt
        assert "Rs" in pvt
        assert pvt["Bo"] > 1
        assert pvt["Bg"] > 0
        assert pvt["Rs"] >= 0

    def test_gas_pvt_returns_bg_z(self):
        """Gas PVT at pressure should return Bg and Z."""
        pvt = FluidCalculations.get_pvt_at_pressure(
            pressure=2000,
            temperature=180,
            gas_specific_gravity=0.65,
            fluid="gas",
        )
        assert "Bg" in pvt
        assert "Z" in pvt
        assert pvt["Z"] > 0.2
        assert pvt["Z"] < 2.0

    def test_oil_pvt_requires_api_gravity(self):
        """Oil PVT must be called with oil_api_gravity."""
        with pytest.raises(ValueError):
            FluidCalculations.get_pvt_at_pressure(
                2000, 180, 0.65, oil_api_gravity=None, fluid="oil"
            )
