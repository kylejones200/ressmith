import pytest

from ressmith.primitives.rock_mechanics import (
    calculate_bulk_modulus,
    calculate_poissons_ratio,
    calculate_shear_modulus,
    calculate_youngs_modulus,
    sonic_to_velocity,
    tensile_strength_from_ucs,
    ucs_from_porosity,
)


@pytest.mark.parametrize(
    ("vp", "vs"),
    [(10_000, 6000), (12_000, 7000), (15_000, 8500), (18_000, 10_000)],
)
def test_poissons_ratio_is_physical(vp, vs):
    assert 0 <= calculate_poissons_ratio(vp, vs) <= 0.5


def test_elastic_moduli_are_positive():
    vp, vs, density = 12_000, 7000, 2.5
    assert calculate_youngs_modulus(vp, vs, density) > 0
    assert calculate_bulk_modulus(vp, vs, density) > 0
    assert calculate_shear_modulus(vs, density) > 0


def test_sonic_transit_time_conversion():
    assert sonic_to_velocity(70) == pytest.approx(1_000_000 / 70)
    assert sonic_to_velocity(0) == 0


def test_strength_correlations_preserve_expected_relationships():
    low_porosity = ucs_from_porosity(0.10, "sandstone")
    high_porosity = ucs_from_porosity(0.25, "sandstone")
    assert low_porosity > high_porosity > 0
    assert tensile_strength_from_ucs(low_porosity) == pytest.approx(
        0.1 * low_porosity
    )


def test_youngs_modulus_cache_is_used():
    calculate_youngs_modulus.cache_clear()
    first = calculate_youngs_modulus(12_000, 7000, 2.5)
    second = calculate_youngs_modulus(12_000, 7000, 2.5)
    assert first == second
    assert calculate_youngs_modulus.cache_info().hits == 1
