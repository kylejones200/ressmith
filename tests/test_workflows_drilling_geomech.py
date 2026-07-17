from ressmith.workflows.drilling import kick_analysis_study, mud_system_study
from ressmith.workflows.geomechanics import (
    formation_pressure_study,
    geomechanical_study,
)


def test_mud_system_study_smoke():
    result = mud_system_study(
        mud_weight=10.5,
        reading_600=60,
        reading_300=40,
        gel_10sec=8,
        gel_10min=12,
    )
    assert isinstance(result, dict)
    assert {
        "mud_weight_ppg",
        "pressure_gradient_psi_ft",
        "bingham_plastic",
        "power_law",
        "gel_strength",
        "filtration",
    } <= result.keys()


def test_kick_analysis_study_smoke():
    result = kick_analysis_study(15, 350, 700, 10.5, 10_000)
    assert isinstance(result, dict)
    assert {
        "detection",
        "formation_pressure_psi",
        "kick_intensity_ppg",
        "kick_type",
        "recommended_kmw",
    } <= result.keys()


def test_geomechanical_study_smoke():
    result = geomechanical_study(
        depth=10_000,
        sonic_dt_compressional=70,
        sonic_dt_shear=130,
        bulk_density=2.5,
        porosity=0.18,
        pore_pressure=5000,
        mud_weight=10.5,
    )
    assert isinstance(result, dict)
    assert {
        "elastic_properties",
        "strength_properties",
        "stress_state",
        "stability_analysis",
    } <= result.keys()


def test_formation_pressure_study_smoke():
    result = formation_pressure_study(
        depth=10_000,
        mud_weight=10.5,
        overburden_gradient=1.04,
        sonic_dt=80,
        normal_sonic=70,
    )
    assert isinstance(result, dict)
    assert {
        "overburden",
        "pore_pressure",
        "fracture_gradient",
        "pressure_classification",
        "drilling_window",
    } <= result.keys()
