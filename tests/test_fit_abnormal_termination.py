"""Regression: L-BFGS-B ABNORMAL exits must not discard a correct optimum.

On uneven calendar-day time grids (real monthly production dates), L-BFGS-B
frequently terminates with ``success=False`` (``ABNORMAL`` line search) while
its ``x`` is already at the optimum. The fitters previously threw that answer
away and fell back to a coarse grid search, returning bound-clamped garbage
(e.g. qi = mean(q), di = 1e-4, b = 0.1) through the whole ``fit_forecast``
workflow. Now the candidate is kept and compared to the grid by objective.
"""

import numpy as np
import pandas as pd
import pytest

from ressmith.primitives.decline import (
    fit_arps_exponential,
    fit_arps_harmonic,
    fit_arps_hyperbolic,
)

# Calendar month-start offsets in days — the uneven grid that triggers ABNORMAL.
CALENDAR_T = np.array(
    [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,
     366, 397, 425, 456, 486, 517, 547, 578, 609, 639, 670, 700],
    dtype=float,
)

QI = 1000.0
DI_DAY = 0.10 / 30.4375  # 10%/month nominal


def test_hyperbolic_fit_on_calendar_grid_recovers_params():
    b = 0.8
    q = QI / (1.0 + b * DI_DAY * CALENDAR_T) ** (1.0 / b)
    params = fit_arps_hyperbolic(CALENDAR_T, q)
    assert params["qi"] == pytest.approx(QI, rel=0.01)
    assert params["di"] == pytest.approx(DI_DAY, rel=0.05)
    assert params["b"] == pytest.approx(b, abs=0.1)


def test_exponential_fit_on_calendar_grid_recovers_params():
    q = QI * np.exp(-DI_DAY * CALENDAR_T)
    params = fit_arps_exponential(CALENDAR_T, q)
    assert params["qi"] == pytest.approx(QI, rel=0.01)
    assert params["di"] == pytest.approx(DI_DAY, rel=0.05)


def test_harmonic_fit_on_calendar_grid_recovers_params():
    q = QI / (1.0 + DI_DAY * CALENDAR_T)
    params = fit_arps_harmonic(CALENDAR_T, q)
    assert params["qi"] == pytest.approx(QI, rel=0.01)
    assert params["di"] == pytest.approx(DI_DAY, rel=0.05)


def test_fit_forecast_workflow_end_to_end_on_monthly_dates():
    """The full workflow (the reservesos/lmk-devsim consumer path) fits sanely."""
    from ressmith.workflows import fit_forecast

    b = 0.8
    idx = pd.date_range("2024-01-01", periods=24, freq="MS")
    t = (idx - idx[0]).days.values.astype(float)
    q = QI / (1.0 + b * DI_DAY * t) ** (1.0 / b)
    frame = pd.DataFrame({"oil": q}, index=idx)

    forecast, params = fit_forecast(frame, model_name="arps_hyperbolic", horizon=6)
    assert params["qi"] == pytest.approx(QI, rel=0.01)
    assert params["b"] == pytest.approx(b, abs=0.1)
    # yhat is IN-SAMPLE (starts at the fit start date): first value ~ qi, and
    # it declines — not the flat mean line the discarded-candidate bug produced.
    yhat = forecast.yhat.values.astype(float)
    assert yhat[0] == pytest.approx(QI, rel=0.01)
    assert np.all(np.diff(yhat) < 0)
