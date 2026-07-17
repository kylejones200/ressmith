import pytest

from ressmith.primitives.pvt import (
    bubble_point_pressure_standing,
    interfacial_tension,
)


def test_standing_bubble_point_matches_correlation():
    rs, gas_gravity, api, temperature = 600, 0.7, 35, 180
    expected = 18.2 * (
        (rs / gas_gravity) ** 0.83 * 10 ** (0.00091 * temperature - 0.0125 * api)
        - 1.4
    )
    assert bubble_point_pressure_standing(
        rs, gas_gravity, api, temperature
    ) == pytest.approx(expected)


def test_bubble_point_increases_with_solution_gas():
    low = bubble_point_pressure_standing(300, 0.7, 35, 180)
    high = bubble_point_pressure_standing(600, 0.7, 35, 180)
    assert high > low >= 14.7


def test_interfacial_tension_matches_empirical_terms():
    expected = 35 - 0.1 * (35 - 30) - 0.05 * (180 - 60) + 0.001 * (3000 - 14.7)
    assert interfacial_tension(35, 180, 3000) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (bubble_point_pressure_standing, (0, 0.7, 35, 180)),
        (interfacial_tension, (35, 0, 3000)),
    ],
)
def test_pvt_additions_reject_nonpositive_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)
