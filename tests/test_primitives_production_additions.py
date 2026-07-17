import pytest

from ressmith.primitives.vlp import (
    esp_horsepower,
    esp_required_head,
    gas_lift_performance,
    producing_gor,
    water_cut,
)


def test_esp_required_head_includes_depth_pressure_and_friction():
    expected = 8000 + 100 * 2.31 / 0.85 + 3000 * 0.0001
    assert esp_required_head(8000, 100, 3000) == pytest.approx(expected)


def test_esp_horsepower_formula_and_efficiency_effect():
    expected = 3000 * 8500 * 0.85 / (3960 * 0.70)
    assert esp_horsepower(3000, 8500, 0.70) == pytest.approx(expected)
    assert esp_horsepower(3000, 8500, 0.60) > esp_horsepower(3000, 8500, 0.80)


def test_gas_lift_performance_applies_pressure_factor():
    result = gas_lift_performance(
        injection_rate=1000,
        injection_pressure=2000,
        operating_pressure=1500,
        liquid_rate=500,
    )
    assert result == pytest.approx((500 / 1000) * (1500 / 2000))
    assert gas_lift_performance(0, 2000, 1500, 500) == 0


@pytest.mark.parametrize(
    ("water", "oil", "expected"),
    [(0, 1000, 0), (250, 750, 25), (500, 500, 50), (900, 100, 90)],
)
def test_water_cut_percentages(water, oil, expected):
    assert water_cut(water, oil) == pytest.approx(expected)


def test_producing_gor_converts_mscf_to_scf():
    assert producing_gor(gas_production=2.5, oil_production=100) == pytest.approx(25)
    assert producing_gor(gas_production=2.5, oil_production=0) == 0


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (esp_required_head, (-1, 100, 3000)),
        (esp_horsepower, (3000, 8500, 0)),
        (gas_lift_performance, (-1, 2000, 1500, 500)),
        (water_cut, (-1, 100)),
        (producing_gor, (1, -1)),
    ],
)
def test_production_helpers_reject_invalid_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)
