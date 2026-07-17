import pytest

from ressmith.primitives.drilling import (
    annular_velocity,
    bit_hydraulics,
    casing_burst,
    casing_collapse,
    equivalent_circulating_density,
    hydrostatic_pressure,
)


def test_hydrostatic_pressure_formula():
    assert hydrostatic_pressure(10_000, 10) == pytest.approx(5200)
    assert hydrostatic_pressure(0, 10) == 0


@pytest.mark.parametrize(
    ("depth", "mud_weight"),
    [(-1, 10), (10_000, -1)],
)
def test_hydrostatic_pressure_rejects_invalid_inputs(depth, mud_weight):
    with pytest.raises(ValueError):
        hydrostatic_pressure(depth, mud_weight)


def test_equivalent_circulating_density_formula():
    assert equivalent_circulating_density(10.5, 300, 10_000) == pytest.approx(
        10.5 + 300 / (0.052 * 10_000)
    )
    assert equivalent_circulating_density(10.5, 0, 10_000) == 10.5


def test_annular_velocity_formula_and_geometry_validation():
    expected = 400 / ((8.5**2 - 5.0**2) / 1029.4)
    assert annular_velocity(400, 8.5, 5.0) == pytest.approx(expected)
    with pytest.raises(ValueError, match="greater than pipe OD"):
        annular_velocity(400, 5.0, 5.0)


def test_bit_hydraulics_and_casing_design():
    hydraulics = bit_hydraulics(3000, 400, 500)
    assert hydraulics["bit_pressure_psi"] == 2500
    assert hydraulics["hydraulic_horsepower"] > 0
    assert hydraulics["impact_force_lbs"] > 0

    burst = casing_burst(5000, 1000, 80_000, 0.545, 9.625)
    assert burst["burst_pressure_psi"] == pytest.approx(
        2 * 80_000 * 0.545 / 9.625
    )
    assert {"safety_factor", "status"} <= burst.keys()

    collapse = casing_collapse(5000, 1000, 80_000, 9.625, 0.545)
    assert collapse["collapse_pressure_psi"] > 0
    assert {"safety_factor", "status", "collapse_regime"} <= collapse.keys()
