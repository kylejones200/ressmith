import pytest

from ressmith.primitives.well_control import (
    calculate_final_circulating_pressure,
    calculate_initial_circulating_pressure,
    calculate_kill_mud_weight,
    detect_and_analyze_kick,
    detect_kick,
    formation_pressure_from_sidpp,
    kick_intensity,
)


def test_kick_detection_severity_tracks_indicators():
    assert detect_kick(pit_volume_gain=1, flow_rate_increase=2)["severity"] < 3

    possible = detect_kick(pit_volume_gain=15, flow_rate_increase=12)
    assert possible["severity"] >= 3
    assert "KICK" in possible["status"]

    detected = detect_kick(pit_volume_gain=5, connection_flow=True)
    assert detected["severity"] >= 5
    assert "KICK DETECTED" in detected["status"]


def test_formation_pressure_and_kick_intensity_formulas():
    pressure = formation_pressure_from_sidpp(350, 10.5, 10_000)
    assert pressure == pytest.approx(5810)
    assert kick_intensity(pressure, 10_000) == pytest.approx(
        pressure / (0.052 * 10_000)
    )
    assert calculate_kill_mud_weight(pressure, 10_000) == pytest.approx(
        kick_intensity(pressure, 10_000) + 0.5
    )


def test_circulating_pressure_helpers():
    assert calculate_initial_circulating_pressure(1000, 350) == 1350
    assert calculate_final_circulating_pressure(1000, 12, 10) == 1200


def test_complete_kick_analysis_has_engineering_outputs():
    result = detect_and_analyze_kick(
        pit_gain=15,
        sidpp=350,
        sicp=700,
        mud_weight=10.5,
        tvd=10_000,
    )
    assert {
        "detection",
        "formation_pressure_psi",
        "kick_intensity_ppg",
        "kick_type",
        "recommended_kmw",
    } <= result.keys()
    assert result["formation_pressure_psi"] == pytest.approx(5810)
    assert result["recommended_kmw"] > result["kick_intensity_ppg"]
