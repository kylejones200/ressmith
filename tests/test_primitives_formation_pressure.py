import pytest

from ressmith.primitives.formation_pressure import (
    drilling_window_analysis,
    eatons_method,
    estimate_overburden_from_depth,
    matthews_kelly_method,
)


def test_onshore_overburden_uses_density_gradient():
    result = estimate_overburden_from_depth(
        depth=10_000, water_depth=0, average_density=2.5
    )
    assert result["overburden_pressure_psi"] == pytest.approx(0.433 * 2.5 * 10_000)
    assert result["gradient_psi_ft"] == pytest.approx(1.082, abs=0.001)
    assert result["equivalent_mud_weight_ppg"] > 0


def test_offshore_overburden_accounts_for_water_column():
    offshore = estimate_overburden_from_depth(10_000, 5000, 2.5)
    onshore = estimate_overburden_from_depth(10_000, 0, 2.5)
    assert 0 < offshore["overburden_pressure_psi"] < onshore["overburden_pressure_psi"]


def test_eatons_method_returns_normal_gradient_when_parameters_match():
    result = eatons_method(70, 70, 1.04, 0.465, 3.0)
    assert result["pore_pressure_gradient_psi_ft"] == pytest.approx(0.465)
    assert result["equivalent_mud_weight_ppg"] == pytest.approx(0.465 / 0.052, rel=0.01)


def test_matthews_kelly_gradient_lies_between_pore_and_overburden():
    result = matthews_kelly_method(10_000, 1.04, 0.65)
    assert 0.65 < result["fracture_gradient_psi_ft"] < 1.04
    assert result["fracture_pressure_psi"] == pytest.approx(
        result["fracture_gradient_psi_ft"] * 10_000
    )


def test_drilling_window_classifies_low_and_high_mud_weights():
    low = drilling_window_analysis(10_000, 0.65, 0.85, 8.0)
    high = drilling_window_analysis(10_000, 0.465, 0.75, 18.0)
    assert low["drilling_window_ppg"] > 0
    assert "UNDERBALANCED" in low["status"]
    assert "OVERBALANCED" in high["status"]
