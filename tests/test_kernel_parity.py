"""Parity guards: ressmith primitives vs the decline-curve kernel.

Consolidation policy (see fire_vault 04-maps/Petro-Econ Surface Inventory):
- ECONOMICS: ressmith delegates to the kernel at runtime (one convention).
- DECLINE RATE / EUR / VARIANTS: ressmith keeps its own implementations
  (identical textbook closed-forms, and a *more accurate* analytic EUR than the
  kernel's trapezoid), but these tests LOCK them to the kernel so any future
  divergence is caught immediately. This is the S1 "single source of truth"
  guarantee enforced by test rather than by indirection.
"""
import numpy as np
import pytest

from decline_curve.models import ArpsParams, predict_arps
from decline_curve.economics import npv_from_cashflow
from ressmith.primitives import decline as rdecl
from ressmith.primitives import economics as recon
from ressmith.primitives import reserves as rres


T = np.linspace(0.0, 240.0, 241)


@pytest.mark.parametrize("b", [0.3, 0.5, 0.7, 0.9, 1.0])
def test_arps_rate_matches_kernel(b):
    """Hyperbolic/harmonic rate is bit-identical to decline-curve.predict_arps."""
    qi, di = 1000.0, 0.08
    q_rs = rdecl.arps_hyperbolic(T, qi, di, b)
    q_dc = predict_arps(T, ArpsParams(qi=qi, di=di, b=b))
    assert np.max(np.abs(q_rs - q_dc)) < 1e-9


def test_exponential_is_pure_exp():
    """ressmith keeps an exact exponential (kernel's b->0 limit is only ~1e-5 close)."""
    qi, di = 1000.0, 0.08
    q = rdecl.arps_exponential(T, qi, di)
    assert np.allclose(q, qi * np.exp(-di * T), atol=1e-9)


def test_eur_close_to_kernel():
    """Analytic EUR (ressmith) agrees with kernel's numeric EUR within 0.1%.

    They are intentionally NOT bit-identical: ressmith integrates the closed form,
    the kernel uses trapezoid. ressmith's is the more accurate of the two.
    """
    from decline_curve.reserves import forecast_and_reserves

    qi, di, b = 1000.0, 0.08, 0.9
    eur_rs = rres.calculate_eur_hyperbolic(qi, di, b, t_max=360.0, econ_limit=10.0)
    eur_dc = forecast_and_reserves(
        ArpsParams(qi=qi, di=di, b=b), t_max=360.0, dt=1.0, econ_limit=10.0
    )["eur"]
    assert abs(eur_rs - eur_dc) / eur_rs < 1e-3


def test_npv_delegates_exactly():
    """ressmith.npv is bit-identical to the kernel helper (true runtime delegation)."""
    cf = np.array([-1500.0, 300.0, 280.0, 260.0, 240.0, 220.0, 200.0])
    for r in (0.05, 0.10, 0.15):
        assert recon.npv(cf, r) == pytest.approx(npv_from_cashflow(cf, r), rel=1e-12)
