"""Smoke tests for integrated interference, coning, and RTA study workflows."""

import numpy as np
import pandas as pd
import pytest

from ressmith.workflows.integrated_reservoir_workflows import (
    coning_study,
    enhanced_rta_study,
    well_interference_study,
)


def test_well_interference_study_geometric_only():
    loc = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "latitude": [32.0, 32.002],
            "longitude": [-97.0, -97.002],
        }
    )
    out = well_interference_study(loc, default_drainage_radius=500.0)
    assert "distances" in out and len(out["distances"]) == 1
    assert "geometric_interference" in out and len(out["geometric_interference"]) == 1
    assert out["production_interference"] is None
    assert out["spacing_recommendation"] is not None
    assert out["metadata"]["n_wells"] == 2


def test_well_interference_study_with_production():
    loc = pd.DataFrame(
        {
            "well_id": ["w1", "w2"],
            "latitude": [32.0, 32.001],
            "longitude": [-97.0, -97.001],
        }
    )
    idx = pd.date_range("2020-01-01", periods=24, freq="ME")
    w1 = pd.DataFrame({"oil": np.linspace(120, 80, len(idx))}, index=idx)
    w2 = pd.DataFrame({"oil": np.linspace(100, 70, len(idx))}, index=idx)
    prod = {"w1": w1, "w2": w2}
    out = well_interference_study(loc, production_data=prod)
    assert out["production_interference"] is not None
    assert len(out["production_interference"]) >= 1


def test_coning_study_basic():
    out = coning_study(
        production_rate=400.0,
        oil_density=50.0,
        water_density=62.4,
        permeability=150.0,
        reservoir_thickness=40.0,
        well_completion_interval=15.0,
    )
    assert "coning_analysis" in out
    ca = out["coning_analysis"]
    assert "critical_rate" in ca and "coning_risk" in ca
    assert out["yield_forecast"] is None


def test_coning_study_with_yield_forecast():
    t = np.arange(0.0, 360.0, 30.0)
    q = np.full_like(t, 450.0)
    out = coning_study(
        production_rate=450.0,
        oil_density=50.0,
        water_density=62.4,
        permeability=120.0,
        reservoir_thickness=45.0,
        well_completion_interval=18.0,
        include_yield_forecast=True,
        forecast_time=t,
        forecast_oil_rate=q,
    )
    assert out["yield_forecast"] is not None
    assert len(out["yield_forecast"]) == len(t)


def test_coning_study_yield_requires_arrays():
    with pytest.raises(ValueError, match="forecast_time"):
        coning_study(
            production_rate=100.0,
            oil_density=50.0,
            water_density=62.4,
            permeability=100.0,
            reservoir_thickness=50.0,
            well_completion_interval=20.0,
            include_yield_forecast=True,
        )


def test_enhanced_rta_study_smoke():
    idx = pd.date_range("2018-01-01", periods=30, freq="ME")
    rate = 500 * (1 + np.arange(len(idx), dtype=float) * 0.01) ** -0.4
    df = pd.DataFrame({"oil": rate}, index=idx)
    out = enhanced_rta_study(
        df,
        run_fmb=False,
        run_fracture_network=False,
        run_blasingame=True,
        run_dn=True,
    )
    assert "metadata" in out
    assert out["metadata"]["rows"] == 30
    assert "type_curve_match" in out or out.get("type_curve_match") is not None
