import pytest

from ressmith.primitives.volumetrics import (
    darcy_flow_rate,
    original_gas_in_place,
    original_oil_in_place,
    recovery_factor,
)


def test_original_oil_in_place_formula():
    result = original_oil_in_place(640, 50, 0.20, 0.70, 1.2)
    assert result == pytest.approx(7758 * 640 * 50 * 0.20 * 0.70 / 1.2)


def test_in_place_volumes_scale_with_area():
    oil_small = original_oil_in_place(320, 50, 0.20, 0.70, 1.2)
    oil_large = original_oil_in_place(640, 50, 0.20, 0.70, 1.2)
    assert oil_large == pytest.approx(2 * oil_small)

    gas = original_gas_in_place(640, 50, 0.15, 0.75, 0.005)
    assert gas == pytest.approx(43_560 * 640 * 50 * 0.15 * 0.75 / 0.005)


def test_darcy_flow_is_positive_and_linear_in_permeability():
    params = dict(
        thickness=50,
        pressure_drawdown=500,
        viscosity=1,
        formation_volume_factor=1.2,
        drainage_radius=1000,
        wellbore_radius=0.328,
    )
    low = darcy_flow_rate(permeability=50, **params)
    high = darcy_flow_rate(permeability=200, **params)
    assert low > 0
    assert high == pytest.approx(4 * low)


def test_water_drive_recovery_exceeds_solution_gas_recovery():
    solution_gas = recovery_factor(4000, 400, "solution_gas")
    water_drive = recovery_factor(4000, 400, "water_drive")
    assert 0 < solution_gas < water_drive < 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(area=0, net_pay=50, porosity=0.2, oil_saturation=0.7, formation_volume_factor=1.2),
        dict(area=640, net_pay=50, porosity=1.2, oil_saturation=0.7, formation_volume_factor=1.2),
    ],
)
def test_ooip_rejects_nonphysical_inputs(kwargs):
    with pytest.raises(ValueError):
        original_oil_in_place(**kwargs)
