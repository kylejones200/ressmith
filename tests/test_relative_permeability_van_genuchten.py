"""Van Genuchten capillary pressure and effective saturation."""

import numpy as np
import pytest

from ressmith.primitives.relative_permeability import (
    calculate_capillary_pressure,
    van_genuchten_capillary_pressure,
    van_genuchten_effective_saturation,
    van_genuchten_m_from_n,
)


def test_van_genuchten_m_from_n() -> None:
    assert van_genuchten_m_from_n(2.0) == pytest.approx(0.5)
    assert van_genuchten_m_from_n(3.0) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="n must be > 1"):
        van_genuchten_m_from_n(1.0)


def test_van_genuchten_roundtrip_se() -> None:
    """Pc(Sw) then Se(Pc) should recover Se."""
    alpha = 0.2
    n = 3.0
    m = van_genuchten_m_from_n(n)
    swr = 0.15
    sw = np.linspace(0.2, 0.95, 30)
    pc = van_genuchten_capillary_pressure(sw, swr, alpha, n, m=m, pc_max=1e6)
    se_expected = (sw - swr) / (1.0 - swr)
    se_from_pc = van_genuchten_effective_saturation(pc, alpha, n, m=m)
    np.testing.assert_allclose(se_from_pc, se_expected, rtol=1e-5, atol=1e-5)


def test_van_genuchten_via_calculate_capillary_pressure() -> None:
    sw = np.array([0.25, 0.45, 0.7])
    pc = calculate_capillary_pressure(
        sw,
        entry_pressure=10.0,
        method="van_genuchten",
        saturation_irreducible=0.2,
        vg_n=2.5,
    )
    assert np.all(np.diff(pc) < 0), "Pc should decrease as Sw increases"
    assert pc[-1] < pc[0]
    assert np.all(pc >= 0)


def test_calculate_capillary_pressure_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown capillary pressure method"):
        calculate_capillary_pressure(
            np.array([0.5]), method="not_a_model", saturation_irreducible=0.2
        )


def test_brooks_corey_unchanged() -> None:
    sw = np.array([0.3, 0.5])
    pc = calculate_capillary_pressure(
        sw,
        entry_pressure=5.0,
        lambda_parameter=2.0,
        saturation_irreducible=0.2,
        method="brooks_corey",
    )
    s_norm = (sw - 0.2) / 0.8
    expected = 5.0 * (np.clip(s_norm, 0.01, 1.0) ** (-0.5))
    np.testing.assert_allclose(pc, np.clip(expected, 0.0, 1000.0), rtol=1e-9)
